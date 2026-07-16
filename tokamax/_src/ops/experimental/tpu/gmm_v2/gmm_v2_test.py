# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.tpu.gmm_v2 import gmm_v2
from tokamax._src.ops.experimental.tpu.gmm_v2 import tgmm_v2
from tokamax._src.ops.experimental.utils.test_utils import poison_tpu_memory

jax.config.parse_flags_with_absl()


def swigluoai(gate: jax.Array, up: jax.Array):
  gate = jnp.clip(gate, a_max=7.0)
  up = jnp.clip(up, a_min=-7.0, a_max=7.0)
  glu = gate * jax.nn.sigmoid(1.702 * gate)
  return (up + 1.0) * glu


def apply_act_fn(acc1: jax.Array, acc3, act_fn):
  match act_fn:
    case "silu":
      return jax.nn.silu(acc1) * acc3
    case "gelu":
      return jax.nn.gelu(acc1) * acc3
    case "swigluoai":
      return swigluoai(acc1, acc3)
    case _:
      raise NotImplementedError(f"Unsupported activation function: {act_fn}")


def get_group_sizes(batch_size: int, num_groups: int) -> jax.Array:
  distribution = jax.random.uniform(
      jax.random.key(0), (num_groups - 1,), dtype=jnp.float32
  )
  distribution = distribution / jnp.sum(distribution)
  group_sizes = jnp.floor(distribution * batch_size).astype(jnp.int32)
  return jnp.append(group_sizes, batch_size - jnp.sum(group_sizes))


def quantize_tensor(
    x: jax.Array, dtype: jnp.dtype, axis: int = -1, block_size: int = 256
):
  if jnp.issubdtype(dtype, jnp.integer):
    dtype_info = jnp.iinfo(dtype)
    max_val = int(dtype_info.max)
    min_val = int(dtype_info.min)
  else:
    dtype_info = jnp.finfo(dtype)
    max_val = float(dtype_info.max)
    min_val = float(dtype_info.min)

  orig_shape = x.shape
  blocked_shape = orig_shape[:axis] + (-1, block_size) + orig_shape[axis + 1 :]
  x_blocked = x.reshape(blocked_shape)

  x_blocked_abs_max = jnp.max(jnp.abs(x_blocked), axis=axis + 1, keepdims=True)
  scale = x_blocked_abs_max / max_val
  x_blocked_q = jnp.clip(x_blocked / scale, min_val, max_val).astype(dtype)

  x_q = x_blocked_q.reshape(orig_shape)
  scale = scale.squeeze(axis=axis + 1).astype(jnp.float32)
  return x_q, scale


def reference_gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,
):
  num_tokens = lhs.shape[0]
  num_groups, in_size, out_size = rhs.shape
  assert lhs.shape[1] == in_size

  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    group_offset = jnp.atleast_1d(group_offset)

  if rhs_scale is not None:
    num_blocks = rhs_scale.shape[1]
  else:
    num_blocks = 1
  block_size = in_size // num_blocks

  start = 0
  gmm_out = []
  for global_group in range(group_sizes.size):
    group_size = group_sizes[global_group]

    group = global_group - group_offset[0]
    end = min(start + group_size, num_tokens)
    group_size = end - start
    if 0 <= group and group < num_groups:
      lhs_slice = lhs[start:end]
      rhs_slice = rhs[group]

      out = jnp.zeros((group_size, out_size), dtype=jnp.float32)
      for block in range(num_blocks):
        block_start = block * block_size
        block_end = block_start + block_size
        lhs_block = lhs_slice[:, block_start:block_end].astype(jnp.float32)
        rhs_block = rhs_slice[block_start:block_end, :].astype(jnp.float32)

        acc = jnp.einsum("bd,dh->bh", lhs_block, rhs_block)
        if rhs_scale is not None:
          acc *= rhs_scale[group][block]
        out += acc
      if rhs_bias is not None:
        out = out + rhs_bias[group]
    else:
      out = jnp.zeros((group_size, out_size), dtype=lhs.dtype)

    gmm_out.append(out.astype(lhs.dtype))
    start = end

  return jnp.concat(gmm_out, axis=0)


def reference_tgmm(
    lhs,  # [k, m]
    rhs,  # [m, n]
    group_sizes,  # [num_groups]
    # num_actual_groups comes from weights.shape[0]
    num_actual_groups,  # int32
    # group_offset is obtained from
    # jnp.arange(0, num_experts, num_experts_per_shard)
    group_offset=None,
):  # [num_groups, k, n]
  # Compute lhs[:, sizes[i-1]:sizes[i]] @ rhs[sizes[i-1]:sizes[i], :]
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    group_offset = jnp.atleast_1d(group_offset)

  start = 0
  out = []
  for global_group in range(group_sizes.size):
    group_size = group_sizes[global_group]
    group = global_group - group_offset[0]
    end = start + group_size
    if 0 <= group and group < num_actual_groups:
      out.append(lhs[:, start:end] @ rhs[start:end, :])
    start = end
  return jnp.stack(out)


class GmmV2Test(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    try:
      if not pltpu.get_tpu_info().generation >= 7:
        self.skipTest("Pallas TPU kernel requires TPU v7 or newer.")
    except Exception:
      self.skipTest("Failed to get TPU info.")

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[16],
      has_bias=[True],
      group_offset=[0],
  )
  def test_gmm_basic(
      self, batch_size, in_size, out_size, num_groups, has_bias, group_offset
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
      )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs, rhs, group_sizes, rhs_bias=rhs_bias, group_offset=group_offset
    )

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
        fuse_act="silu",
    )

    gate = expected[..., : out_size // 2]
    up = expected[..., out_size // 2 :]
    expected = apply_act_fn(
        gate.astype(jnp.float32), up.astype(jnp.float32), "silu"
    ).astype(actual.dtype)

    chex.assert_trees_all_close(actual, expected, rtol=2e-2, atol=2e-2)

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[5],
      group_offset=[2],
  )
  def test_tgmm_basic(
      self, batch_size, in_size, out_size, num_groups, group_offset
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(
        key1, (batch_size, in_size), dtype=jnp.bfloat16
    )  # [m, k]
    grad = jax.random.normal(
        key2, (batch_size, out_size), dtype=jnp.bfloat16
    )  # [m, n]
    group_sizes = get_group_sizes(batch_size, num_groups)
    # if batch_size=128, num_groups=3, an example group_size is
    # group_sizes=Array([14, 14, ..., 7]).
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)  # [k, m]
    expected = reference_tgmm(
        lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset
    )
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    chex.assert_trees_all_close(
        actual.shape, (num_local_groups, in_size, out_size)
    )
    # diff = jnp.abs(expected - actual)
    # max_diff_idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)
    # print(f"Output max diff: {jnp.max(diff)} at index {max_diff_idx}")
    # print(f"Output mean diff: {jnp.mean(jnp.abs(expected - actual))}")
    chex.assert_trees_all_close(actual, expected)

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[16],
      has_bias=[True],
      quant_config=[
          {"weight_dtype": jnp.int8, "block_size": 128},
          {"weight_dtype": jnp.float4_e2m1fn, "block_size": 16},
          {"weight_dtype": jnp.float4_e2m1fn, "block_size": 32},
          {"weight_dtype": jnp.int4, "block_size": 32},
      ],
      group_offset=[0],
      fuse_act=["silu"],
  )
  def test_gmm_weight_quantized(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      has_bias,
      quant_config,
      group_offset,
      fuse_act,
  ):
    weight_dtype = quant_config["weight_dtype"]
    block_size = quant_config["block_size"]
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_q, rhs_scale = quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
      )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        rhs_bias=rhs_bias,
        maybe_quantize_lhs=False,
        fuse_act=fuse_act,
    ).astype(lhs.dtype)

    if fuse_act is not None:
      gate = expected[..., : out_size // 2]
      up = expected[..., out_size // 2 :]
      expected = apply_act_fn(
          gate.astype(jnp.float32), up.astype(jnp.float32), fuse_act
      ).astype(actual.dtype)

    # TODO: Investigate high atol
    chex.assert_trees_all_close(actual, expected, atol=3e1, rtol=3e-1)

  def test_gmm_security_isolation(self):
    """Verifies that sequences (experts) are isolated from each other.

    This test checks that NaNs or extreme values in one expert group do not
    pollute the output of other expert groups, even if they share the same
    sublane tile.
    """
    batch_size = 128
    in_size = 512
    out_size = 512
    num_groups = 4
    key = jax.random.key(42)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_groups, in_size, out_size), dtype=jnp.bfloat16
    )

    # We use very small group sizes to force expert groups to share tiles.
    # sublane_size is typically 8 or 16.
    group_sizes = jnp.array([4, 4, 4, batch_size - 12], dtype=jnp.int32)

    # 1. Run baseline
    actual_clean = gmm_v2.gmm_v2(lhs, rhs, group_sizes)

    # 2. Inject NaNs into all experts except the first one.
    # If isolation fails, the NaNs will leak into the first expert's output.
    rhs_malicious = rhs.at[1:].set(jnp.nan)
    actual_malicious = gmm_v2.gmm_v2(lhs, rhs_malicious, group_sizes)

    # Verify that the first expert's output is identical and NaN-free.
    first_expert_size = group_sizes[0]
    chex.assert_trees_all_close(
        actual_malicious[:first_expert_size],
        actual_clean[:first_expert_size],
        atol=0,
        rtol=0,
    )
    self.assertFalse(jnp.any(jnp.isnan(actual_malicious[:first_expert_size])))

  def test_gmm_uninitialized_memory_robustness(self):
    """Verifies that the kernel is robust against uninitialized scratchpads.

    This test intentionally poisons TPU VMEM/SMEM with NaNs before running the
    GMM kernel. This ensures that no stale data from previous sessions can leak
    into the output.
    """
    # 1. Poison TPU memory with NaNs
    poison_tpu_memory()

    # 2. Run GMM kernel
    batch_size = 128
    in_size = 512
    out_size = 512
    num_groups = 4
    key = jax.random.key(0)
    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    group_sizes = jnp.array([batch_size // 4] * 4, dtype=jnp.int32)

    actual = gmm_v2.gmm_v2(lhs, rhs, group_sizes)

    # 3. Verify that the output is NaN-free
    self.assertFalse(jnp.any(jnp.isnan(actual)))

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[16],
      block_size=[512],
      weight_dtype=[jnp.int8],
      group_offset=[0],
  )
  def test_gmm_activation_weight_quantized(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      block_size,
      weight_dtype,
      group_offset,
  ):
    block_size = in_size
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_q, rhs_scale = quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)
    lhs_q_dtype = (
        jnp.float8_e4m3fn if weight_dtype == jnp.float8_e4m3fn else jnp.int8
    )
    lhs_block_size = min(512, in_size)
    lhs_q, lhs_scale_factor = quantize_tensor(
        lhs, lhs_q_dtype, axis=1, block_size=lhs_block_size
    )

    lhs_q_blocked = lhs_q.reshape(batch_size, -1, lhs_block_size).astype(
        jnp.float32
    )
    lhs_scale_expanded = jnp.expand_dims(lhs_scale_factor, axis=2)
    lhs_simulated = (
        (lhs_q_blocked * lhs_scale_expanded)
        .reshape(lhs.shape)
        .astype(lhs.dtype)
    )

    expected = reference_gmm(
        lhs_simulated,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        maybe_quantize_lhs=True,
        fuse_act="silu",
    )
    gate = expected[..., : out_size // 2]
    up = expected[..., out_size // 2 :]
    expected = apply_act_fn(
        gate.astype(jnp.float32), up.astype(jnp.float32), "silu"
    ).astype(actual.dtype)

    chex.assert_trees_all_close(actual, expected, atol=4, rtol=2)

  @parameterized.product(
      batch_size=[128],
      in_size=[256],
      out_size=[256],
      num_groups=[16],
      has_bias=[True],
      group_offset=[0],
  )
  def test_gmm_implicit_padding(
      self, batch_size, in_size, out_size, num_groups, has_bias, group_offset
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
      )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
        fuse_act="silu",
    )
    gate = expected[..., : out_size // 2]
    up = expected[..., out_size // 2 :]
    expected = apply_act_fn(
        gate.astype(jnp.float32), up.astype(jnp.float32), "silu"
    ).astype(actual.dtype)
    self.assertEqual(actual.shape, (batch_size, out_size // 2))

    chex.assert_trees_all_close(actual, expected, rtol=2e-2, atol=2e-2)

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[500],
      num_groups=[16],
      has_bias=[True],
      weight_dtype=[jnp.int8],
      block_size=[512],
      group_offset=[0],
  )
  def test_gmm_weight_quantized_padding(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      has_bias,
      weight_dtype,
      block_size,
      group_offset,
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    rhs_q, rhs_scale = quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
      )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        rhs_bias=rhs_bias,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    self.assertEqual(actual.shape, (batch_size, out_size))
    chex.assert_trees_all_close(actual, expected, atol=3e-1, rtol=3e-1)


if __name__ == "__main__":
  absltest.main()
