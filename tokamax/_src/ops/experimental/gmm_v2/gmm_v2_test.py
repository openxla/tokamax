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
import collections

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src import mosaic_tpu
from tokamax._src import test_utils
from tokamax._src.ops.experimental.gmm_v2 import gmm_v2
from tokamax._src.ops.experimental.gmm_v2 import tgmm_v2

import pytest

jax.config.parse_flags_with_absl()


_GroupConfig = collections.namedtuple(
    "_GroupConfig", ["num_groups", "group_offset", "num_local_groups"]
)


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
  x_q = jnp.nan_to_num(x_q)
  scale = scale.squeeze(axis=axis + 1).astype(jnp.float32)
  return x_q, scale


def reference_gmm(
    lhs: jax.Array,  # [m, k]
    rhs: jax.Array,  # [num_groups, k, n]
    group_sizes: jax.Array,  # [num_groups]
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,  # int32[1]
):
  num_tokens = lhs.shape[0]
  num_groups, in_size, out_size = rhs.shape
  assert num_groups > 0, f"rhs must have at least 1 group, got {num_groups}"
  assert lhs.shape[1] == in_size

  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  elif jnp.isscalar(group_offset):
    assert group_offset.size == 1
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]

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

      out = jnp.array(0.0, dtype=jnp.float32)
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


# For TGMM quantized case, we want the reference to use the most accurate
# computation of what the kernel is asked to compute
# (f32 and highest precision) as opposed to the hardware-mirroring way
# (reference also does native fp8 dot_general) which we measure noise against
# noise.
def reference_tgmm(
    lhs,  # [k, m]
    rhs,  # [m, n]
    group_sizes,  # [num_groups]
    # num_actual_groups comes from weights.shape[0]
    num_actual_groups,  # int32
    rhs_scale: jax.Array | None = None,
    # group_offset is obtained from
    # jnp.arange(0, num_experts, num_experts_per_shard)
    group_offset=None,
    out_dtype: jnp.dtype | None = None,
):  # [num_groups, k, n]
  # Compute lhs[:, sizes[i-1]:sizes[i]] @ rhs[sizes[i-1]:sizes[i], :]
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  elif jnp.isscalar(group_offset):
    assert group_offset.size == 1
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]

  assert group_sizes.size >= int(group_offset[0]) + num_actual_groups, (
      f"group_sizes.size ({group_sizes.size}) must be >= "
      f"group_offset ({int(group_offset[0])}) + num_actual_groups "
      f"({num_actual_groups})"
  )

  start = 0
  out = []
  for global_group in range(group_sizes.size):
    group_size = group_sizes[global_group]
    group = global_group - group_offset[0]
    end = start + group_size
    if 0 <= group and group < num_actual_groups:
      if rhs_scale is None:
        out.append(lhs[:, start:end] @ rhs[start:end, :])
      else:
        # rhs_scale.shape==(1, 1, N).
        partial = jax.lax.dot_general(
            lhs[:, start:end].astype(jnp.float32),
            rhs[start:end, :].astype(jnp.float32),
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        partial *= rhs_scale[0]  # rhs_scale[0]: shape [1, N]
        output_dtype = out_dtype if out_dtype is not None else lhs.dtype
        out.append(partial.astype(output_dtype))
    start = end
  return jnp.stack(out)


# Default per-dtype tolerances, mirroring
# jax._src.public_test_util._default_tolerance. Extend this map if a new output
# dtype is introduced into a default-tolerance assertion.
_DTYPE_TOL = {
    jnp.dtype(jnp.bfloat16): 1e-1,
}


def _lookup_tol(dtype):
  key = jnp.dtype(dtype)
  if key not in _DTYPE_TOL:
    raise KeyError(
        f"No default tolerance for dtype {key!r}. "
        f"Add it to _DTYPE_TOL or pass explicit atol/rtol."
    )
  return _DTYPE_TOL[key]


def assert_arrays_all_close(actual, desired, *, atol=None, rtol=None):
  if atol is None:
    atol = max(_lookup_tol(actual.dtype), _lookup_tol(desired.dtype))
  if rtol is None:
    rtol = max(_lookup_tol(actual.dtype), _lookup_tol(desired.dtype))
  chex.assert_trees_all_close(actual, desired, atol=atol, rtol=rtol)


class GmmTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  @pytest.mark.long
  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[16, 32],
      has_bias=[True],
      group_offset=[0],
  )
  def test_gmm_basic(
      self, batch_size, in_size, out_size, num_groups, has_bias, group_offset
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    k0, k1, k2 = jax.random.split(key, 3)

    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        k1, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          k2, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
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
    )

    assert_arrays_all_close(actual, expected)

  @pytest.mark.long
  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[5, 16],
      group_offset=[0],
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
    tgmm_v2.validate_tgmm_inputs(
        group_sizes, num_local_groups, group_offset)
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    # diff = jnp.abs(expected - actual)
    # max_diff_idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)
    # print(f"Output max diff: {jnp.max(diff)} at index {max_diff_idx}")
    # print(f"Output mean diff: {jnp.mean(jnp.abs(expected - actual))}")
    assert_arrays_all_close(actual, expected)

  @pytest.mark.long
  @parameterized.product(
      batch_size=[128, 256],
      in_size=[255],
      out_size=[255],
      num_groups=[16],
      group_offset=[0],
  )
  def test_tgmm_implicit_padding(
      self, batch_size, in_size, out_size, num_groups, group_offset
  ):
    # Test the case where there is implicit padding in the dim size_k (in_size)
    # and size_n (out_size).
    # Notice that tile_n and tile_k are aligned to the num_lanes in
    # calculate_tgmm_tiling.
    # The output shape is [num_groups, size_k, aligned_n] but there is implicit
    # padding on the k-dim to a multiple of sublanes. So the kernel is able to
    # write the full [i, aligned_tile_k, aligned_tile_n] to hbm with no problem
    # at the last k block.
    # Within the kernel, because k is not the contracting dim, so the padded k
    # is also not a problem.
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(
        lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset
    )
    tgmm_v2.validate_tgmm_inputs(
        group_sizes, num_local_groups, group_offset)
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.long
  @parameterized.product(
      batch_size=[256],
      in_size=[1024],
      out_size=[1024],
      num_groups=[16],
      group_offset=[0],
      tile_k=[256, 512],
      tile_n=[256, 512],
  )
  def test_tgmm_with_tile_info(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      group_offset,
      tile_k,
      tile_n,
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(
        lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset
    )

    tile_m = 256
    tile_info = gmm_v2.TileSizes(
        tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, bucket_base=tile_m
    )
    tgmm_v2.validate_tgmm_inputs(
        group_sizes, num_local_groups, group_offset)
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
        tile_info=tile_info,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.long
  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[4],
      group_offset=[0],
      empty_group_index=[0, 1],
  )
  def test_tgmm_empty_group(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      group_offset,
      empty_group_index,
  ):
    """Test that TGMM correctly zeros output for empty groups."""
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    # Redistribute the empty group's tokens to the last group.
    group_sizes = group_sizes.at[-1].add(group_sizes[empty_group_index])
    group_sizes = group_sizes.at[empty_group_index].set(0)

    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(
        lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset
    )
    tgmm_v2.validate_tgmm_inputs(
        group_sizes, num_local_groups, group_offset)
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  def test_tgmm_explicitly_exercises_all_branches(self):
    # Group 0 (size 4*tile_m, 4 gm tiles): matmul_new_group, matmul, matmul,
    # matmul_group_changing.
    # Group 1 (size 64, 1 gm tile): matmul_new_group_and_changing.

    tile_m = tile_k = tile_n = 256
    in_size = out_size = 256
    num_local_groups = 2
    g0, g1 = 4 * tile_m, 64
    batch_size = g0 + g1

    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = jnp.array([g0, g1], dtype=jnp.int32)
    group_offset = jnp.array(0, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(
        lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset
    )
    tile_info = gmm_v2.TileSizes(
        tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, bucket_base=tile_m
    )
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
        tile_info=tile_info,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.long
  @parameterized.product(
      batch_size=[128],
      in_size=[256],
      out_size=[256],
      num_groups=[16],
      group_offset=[0],
      dtype_pair=[
          (jnp.float8_e4m3fn, jnp.float8_e5m2),       # production
          (jnp.float8_e4m3fn, jnp.float8_e4m3fn),     # symmetric fp8
      ],
  )
  def test_tgmm_with_rhs_scale(
      self, batch_size, in_size, out_size, num_groups, group_offset, dtype_pair
  ):
    lhs_dtype, rhs_quant_dtype = dtype_pair
    num_local_groups = num_groups - group_offset

    key1, key2 = jax.random.split(jax.random.key(0), 2)
    lhs = jax.random.normal(
        key1, (batch_size, in_size), dtype=jnp.bfloat16
    ).astype(lhs_dtype)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.float32)

    grad_q, grad_scale = quantize_tensor(
        grad, rhs_quant_dtype, axis=0, block_size=batch_size,
    )
    grad_scale = jnp.expand_dims(grad_scale, axis=1)  # [1, 1, N]
    assert grad_scale.shape == (1, 1, out_size)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset_arr = jnp.array([group_offset], dtype=jnp.int32)

    expected = reference_tgmm(
        lhs.swapaxes(0, 1), grad_q, group_sizes, num_local_groups,
        rhs_scale=grad_scale,
        group_offset=group_offset_arr,
        out_dtype=jnp.bfloat16,
    )
    tgmm_v2.validate_tgmm_inputs(
        group_sizes, num_local_groups, group_offset_arr)
    actual = tgmm_v2.tgmm_v2(
        lhs, grad_q, group_sizes, num_local_groups,
        rhs_scale=grad_scale,
        group_offset=group_offset_arr,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    # Tolerance rationale: see test_tgmm_dynamic_quant_basic.
    chex.assert_trees_all_close(actual, expected, rtol=1e-2, atol=4e-1)

  @pytest.mark.long
  def test_tgmm_with_rhs_scale_n_padding(self):
    # Test the case where there is implicit padding in the dim size_n (out_size)
    # Pins tile_n=128 with out_size=300 so the kernel runs 3 n-tiles over an
    # aligned width of 384; the last tile (n_id=2) reads scale[..., 256:384]
    # where columns 300..383 are pad.

    batch_size, in_size, out_size = 128, 256, 300
    num_groups = 4
    rhs_quant_dtype = jnp.float8_e5m2

    key1, key2 = jax.random.split(jax.random.key(0), 2)
    lhs = jax.random.normal(
        key1, (batch_size, in_size), dtype=jnp.bfloat16
    ).astype(jnp.float8_e4m3fn)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.float32)

    grad_q, grad_scale = quantize_tensor(
        grad, rhs_quant_dtype, axis=0, block_size=batch_size,
    )
    grad_scale = jnp.expand_dims(grad_scale, axis=1)  # [1, 1, N]
    assert grad_scale.shape == (1, 1, out_size)

    group_sizes = get_group_sizes(batch_size, num_groups)
    tile_m = 128
    tile_info = gmm_v2.TileSizes(
        tile_m=tile_m, tile_k=256, tile_n=128, bucket_base=tile_m
    )

    expected = reference_tgmm(
        lhs.swapaxes(0, 1), grad_q, group_sizes, num_groups,
        rhs_scale=grad_scale,
        out_dtype=jnp.bfloat16,
    )
    tgmm_v2.validate_tgmm_inputs(group_sizes, num_groups)
    actual = tgmm_v2.tgmm_v2(
        lhs, grad_q, group_sizes, num_groups,
        rhs_scale=grad_scale,
        tile_info=tile_info,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_groups, in_size, out_size))
    chex.assert_trees_all_close(actual, expected, rtol=1e-2, atol=4e-1)

  @pytest.mark.long
  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[16],
      has_bias=[True],
      weight_dtype=[jnp.int8, jnp.float8_e4m3fn, jnp.float4_e2m1fn],
      block_size=[64],
      group_offset=[0],
  )
  def test_gmm_weight_quantized(
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
    if weight_dtype == jnp.float4_e2m1fn and test_utils.get_tpu_version() < 7:
      self.skipTest("Expect TPUv7+")
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
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=3e-1, rtol=3e-1)

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
        atol=0.0,
        rtol=0.0,
    )
    self.assertFalse(jnp.any(jnp.isnan(actual_malicious[:first_expert_size])))

  def test_gmm_uninitialized_memory_robustness(self):
    """Verifies that the kernel is robust against uninitialized scratchpads.

    This test intentionally poisons TPU VMEM/SMEM with NaNs before running the
    GMM kernel. This ensures that  no stale data from previous sessions can leak
    into the output.
    """
    # 1. Poison TPU memory with NaNs
    mosaic_tpu.poison_tpu_memory()

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

  @pytest.mark.long
  @parameterized.product(
      batch_size=[128],
      in_size=[1024],
      out_size=[512],
      num_groups=[16],
      weight_dtype=[jnp.int8],
      block_size=[1024],
      tile_k=[128, 256],
      group_offset=[0],
  )
  def test_gmm_weight_quantized_block_larger_than_tile_k(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      weight_dtype,
      block_size,
      tile_k,
      group_offset,
  ):
    """Test that quant_block_size > tile_k is handled correctly."""
    if weight_dtype == jnp.float4_e2m1fn and test_utils.get_tpu_version() < 7:
      self.skipTest("Expect TPUv7+")
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

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    tile_info = gmm_v2.TileSizes(
        tile_m=128, tile_k=tile_k, tile_n=out_size, bucket_base=128
    )
    actual = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        tile_info=tile_info,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=3e-1, rtol=3e-1)

  @pytest.mark.long
  @parameterized.product(
      batch_size=[128],
      in_size=[1024],
      out_size=[512],
      num_groups=[16],
      weight_dtype=[jnp.int4, jnp.int8],
      block_size=[1024],
      tile_k=[128, 256],
      group_offset=[0],
  )
  def test_gmm_activation_weight_quantized_block_larger_than_tile_k(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      weight_dtype,
      block_size,
      tile_k,
      group_offset,
  ):
    """Test activation+weight quantized path with quant_block_size > tile_k."""
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

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    tile_info = gmm_v2.TileSizes(
        tile_m=128, tile_k=tile_k, tile_n=out_size, bucket_base=128
    )
    actual = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        tile_info=tile_info,
        maybe_quantize_lhs=True,
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=1.2, rtol=1.2)

  @pytest.mark.long
  @parameterized.product(
      batch_size=[128],
      in_size=[1024],
      out_size=[1024],
      num_groups=[16, 32],
      weight_dtype=[jnp.int4, jnp.int8],
      block_size=[1024],
      group_offset=[0],
  )
  def test_gmm_activation_weight_quantized(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      weight_dtype,
      block_size,
      group_offset,
  ):
    if weight_dtype == jnp.float4_e2m1fn and test_utils.get_tpu_version() < 7:
      self.skipTest("Expect TPUv7+")
    if block_size > in_size:
      self.skipTest("block_size must be <= in_size")
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

    expected = reference_gmm(
        lhs,
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
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=1.1, rtol=1.1)

  @pytest.mark.long
  @parameterized.product(
      batch_size=[128],
      in_size=[1024],
      out_size=[1024],
      num_groups=[16, 32],
      block_size=[1024],
      group_offset=[0],
  )
  def test_gmm_quantize_lhs_with_lhs_scale(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      block_size,
      group_offset,
  ):
    """LHS quantized with a user provided lhs_scale."""
    if block_size > in_size:
      self.skipTest("block_size must be <= in_size")
    # Per-tensor fp8 quant scale (bound / finfo(fp8).max = 224 / 448),
    # matching qwix's "fixed,-224,224" act calibration.
    lhs_scale = jnp.full((1, 1), 224.0 / 448.0, dtype=jnp.float32)

    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    # Pin fp8 weights so rhs is dequantized after the matmul, which is the only
    # path that enables lhs quantization.
    rhs_q, rhs_scale = quantize_tensor(
        rhs, jnp.float8_e4m3fn, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    # The kernel quantizes LHS to fp8 internally (constant fixed scale,
    # clip-before-cast), which perturbs every LHS value before the matmul. We
    # apply the identical quantize->dequantize round-trip to the reference's LHS
    # so both sides carry the same quantization error. Otherwise the comparison
    # would measure fp8 quantization noise itself (large for fp8_e4m3fn) rather
    # than whether the kernel computes the correct grouped matmul. Because the
    # fixed scale is data-independent (not per-block absmax), the quantized
    # values are identical regardless of how the kernel slices K, so this
    # whole-tensor simulation is faithful; only accumulation order /
    # intermediate-dtype differences remain, within the tolerance below.
    scale = lhs_scale.item()
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    lhs_q = jnp.clip(
        lhs.astype(jnp.float32) / scale, -fp8_max, fp8_max
    ).astype(jnp.float8_e4m3fn)
    lhs_simulated = (lhs_q.astype(jnp.float32) * scale).astype(lhs.dtype)

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
        lhs_scale=lhs_scale,
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=0.75, rtol=3e-2)

  @pytest.mark.long
  @parameterized.product(
      batch_size=[128, 256],
      in_size=[255],
      out_size=[255],
      num_groups=[16],
      has_bias=[True, False],
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
    )

    self.assertEqual(actual.shape, (batch_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.long
  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[500],
      num_groups=[16],
      has_bias=[True, False],
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

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      # group_config: (num_groups, group_offset, num_local_groups)
      group_config=[
          # groups 0-1: group<0, groups 2-5: local and active,
          # groups 6-15: group>=num_local_groups
          _GroupConfig(num_groups=16, group_offset=2, num_local_groups=4),
          # no negative groups, groups 0-7: local and active,
          # groups 8-15: group>=num_local_groups
          _GroupConfig(num_groups=16, group_offset=0, num_local_groups=8),
          # groups 0-3: group<0, groups 4-7: local and active,
          # groups 8-31: group>=num_local_groups
          _GroupConfig(num_groups=32, group_offset=4, num_local_groups=4),
      ],
  )
  def test_gmm_nonlocal_groups_produce_zeros(
      self, batch_size, in_size, out_size, group_config
  ):
    num_groups, group_offset, num_local_groups = group_config
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
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
    )

    self.assertEqual(actual.shape, (batch_size, out_size))
    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[16],
      has_bias=[True, False],
      use_weight_scale=[True, False],
      maybe_quantize_lhs=[True, False],
      fuse_act=["silu", "swigluoai", "gelu"],
      group_offset=[0, 2],
      block_size=[256, 512],
  )
  def test_gmm_fused_activation(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      has_bias,
      use_weight_scale,
      maybe_quantize_lhs,
      fuse_act,
      group_offset,
      block_size,
  ):
    if maybe_quantize_lhs and not use_weight_scale:
      self.skipTest(
          "LHS quantization requires RHS quantization/scale in this config."
      )
    if block_size > in_size:
      self.skipTest("block_size must be <= in_size")
    key = jax.random.key(0)
    final_out_size = out_size // 2
    num_local_groups = num_groups - group_offset

    # 1. Generate Inputs
    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )

    rhs_q = rhs
    rhs_scale = None
    if use_weight_scale:
      rhs_q, rhs_scale = quantize_tensor(
          rhs, jnp.int8, axis=1, block_size=block_size
      )
      rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
      )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array([group_offset], dtype=jnp.int32)

    # 2. Simulate LHS Quantization Noise
    lhs_simulated = lhs
    # because the kernel quantizes LHS in blocks, while reference does it at the
    # whole tensor level, and output is casted down we need to simulate that
    # quantization noise in the reference as well for a fair comparison
    if maybe_quantize_lhs:
      lhs_block_size = min(512, in_size)
      lhs_q, lhs_scale_factor = quantize_tensor(
          lhs, jnp.int8, axis=1, block_size=lhs_block_size
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

    # 3. Compute Reference Output
    raw_expected = reference_gmm(
        lhs_simulated,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    # Slice the reference and apply the activation function
    if fuse_act is not None:
      raw_gate, raw_up = jnp.split(raw_expected, 2, axis=-1)
      raw_expected = gmm_v2.interleave_lane(raw_gate, raw_up)
    expected = gmm_v2.apply_act_fn(
        raw_expected.astype(jnp.float32), fuse_act).astype(lhs.dtype)

    # 4. Compute Actual Kernel Output
    actual = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
        maybe_quantize_lhs=maybe_quantize_lhs,
        fuse_act=fuse_act,
    ).astype(lhs.dtype)

    # 5. Compare Results
    self.assertEqual(actual.shape, (batch_size, final_out_size))

    # tolerances based quantization noise difference between reference and
    # gmm_v2
    if maybe_quantize_lhs:
      atol, rtol = 4.0, 2.0  # Act + Weight Quantization
    elif use_weight_scale:
      atol, rtol = 3e-1, 3e-1  # Weight Quantization Only
    else:
      atol, rtol = 5e-2, 5e-2  # Unquantized Path (bfloat16 precision diffs)

    chex.assert_trees_all_close(actual, expected, atol=atol, rtol=rtol)

  @pytest.mark.long
  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[5, 16],
      dtype_pair=[
          (jnp.float8_e4m3fn, jnp.float8_e4m3fn),
          (jnp.float8_e4m3fn, jnp.float8_e5m2),
      ],
      pin_tiles=[True, False],
  )
  def test_tgmm_dynamic_quant_basic(
      self, batch_size, in_size, out_size, num_groups, dtype_pair, pin_tiles
  ):
    """Kernel quantizes bf16 lhs/dout per m-tile; compare against f32 tgmm."""
    if test_utils.get_tpu_version() < 7:
      self.skipTest("float8_e4m3fn matmul requires TPUv7+")
    lhs_quant_dtype, rhs_quant_dtype = dtype_pair

    key1, key2 = jax.random.split(jax.random.key(0), 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)

    # The reference is the most accurate computation of what the kernel is asked
    # to compute -- f32, unquantized -- rather than a mirror of the kernel's own
    # per-tile quantization. The tolerance below absorbs the fp8 error.
    expected = reference_tgmm(
        lhs.astype(jnp.float32).swapaxes(0, 1),
        grad.astype(jnp.float32),
        group_sizes,
        num_groups,
    )

    # Pinned: tile_m=128 with 8-25 rows per group keeps each group inside one
    # m-tile, while tile_k/tile_n=256 force 2 k-tiles and 2 n-tiles. Unpinned,
    # calculate_tgmm_tiling returns (128, 512, 512) at this shape -- a single
    # k/n tile -- so only the pinned arm covers scales being associated with the
    # right k/n slice, and the unpinned arm is the smoke test that the tiler
    # runs at all on the quant path.
    tile_info = (
        gmm_v2.TileSizes(tile_m=128, tile_k=256, tile_n=256, bucket_base=128)
        if pin_tiles
        else tgmm_v2.calculate_tgmm_tiling
    )
    tgmm_v2.validate_tgmm_inputs(group_sizes, num_groups)
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_groups,
        tile_info=tile_info,
        preferred_element_type=jnp.bfloat16,
        lhs_quant_dtype=lhs_quant_dtype,
        rhs_quant_dtype=rhs_quant_dtype,
    )
    self.assertEqual(actual.shape, (num_groups, in_size, out_size))
    # N.B. The reason why atol need to multiple by RMS is that
    # for num_groups=5 (groups of 44, 45, ...), num_groups=16 (groups of ~8)
    # 16 groups means each output is smaler than the 5 groups.
    # So multiplying RMS makes the atol dimensionless.
    # The 5e-1 coefficient is empirical: the per-element error is heavy-tailed
    # (sigma/RMS 0.024, max/RMS 0.27), so this is a max over ~1e6 samples rather
    # than a multiple of sigma. Worst measured need is 0.39, for a
    # float8_e5m2 dout.
    chex.assert_trees_all_close(
        actual.astype(jnp.float32),
        expected,
        rtol=1e-1,
        atol=5e-1 * float(jnp.sqrt(jnp.mean(jnp.square(expected)))),
    )

  @pytest.mark.long
  def test_tgmm_dynamic_quant_all_branches(self):
    """Per-tile quant across all four gm branches, including a multi-tile group.

    Group 0 (size 4*tile_m, 4 gm tiles): matmul_new_group, matmul, matmul,
    matmul_group_changing. Each tile carries its own scale, so this is the case
    that detects a dequant applied after accumulation instead of before.
    Group 1 (size 64, 1 gm tile): matmul_new_group_and_changing.
    """
    if test_utils.get_tpu_version() < 7:
      self.skipTest("float8_e4m3fn matmul requires TPUv7+")

    tile_m = tile_k = tile_n = 256
    in_size = out_size = 256
    num_local_groups = 2
    g0, g1 = 4 * tile_m, 64
    batch_size = g0 + g1

    key1, key2 = jax.random.split(jax.random.key(0), 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = jnp.array([g0, g1], dtype=jnp.int32)

    expected = reference_tgmm(
        lhs.astype(jnp.float32).swapaxes(0, 1),
        grad.astype(jnp.float32),
        group_sizes,
        num_local_groups,
    )

    tile_info = gmm_v2.TileSizes(
        tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, bucket_base=tile_m
    )
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        tile_info=tile_info,
        preferred_element_type=jnp.bfloat16,
        lhs_quant_dtype=jnp.float8_e4m3fn,
        rhs_quant_dtype=jnp.float8_e4m3fn,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    # Tolerance rationale: see test_tgmm_dynamic_quant_basic.
    chex.assert_trees_all_close(
        actual.astype(jnp.float32),
        expected,
        rtol=1e-1,
        atol=5e-1 * float(jnp.sqrt(jnp.mean(jnp.square(expected)))),
    )

  @pytest.mark.long
  def test_tgmm_dynamic_quant_ragged_groups(self):
    """Empty and tiny groups, a non-zero group_offset, and all-zero columns.

    The tiny groups start mid-tile, so local_offset != 0 and most tiles are
    truncated on both ends. The zeroed lhs column / grad column are what make
    the where(scale == 0, 0, 1/scale) guard load-bearing: that column's absmax
    is 0, so an unguarded 1/scale gives inf and then 0*inf = NaN. Note an empty
    *group* does not reach the guard -- the grid skips it and the output is
    zeroed by DMA -- so verified by mutation: dropping the guard fails this
    test and no other.
    """
    if test_utils.get_tpu_version() < 7:
      self.skipTest("float8_e4m3fn matmul requires TPUv7+")

    batch_size = in_size = out_size = 256
    num_groups, offset = 8, 2
    num_local_groups = num_groups - offset

    key1, key2 = jax.random.split(jax.random.key(0), 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    # Sums to batch_size. Globals 3, 6, 7 are empty; at group_offset=2 they are
    # local indices 1, 4, 5.
    group_sizes = jnp.array([1, 7, 15, 0, 200, 33, 0, 0], dtype=jnp.int32)
    empty_local = (1, 4, 5)
    # Force a zero absmax on one k column and one n column.
    lhs = lhs.at[:, 3].set(0)
    grad = grad.at[:, 5].set(0)
    group_offset = jnp.array(offset, dtype=jnp.int32)

    expected = reference_tgmm(
        lhs.astype(jnp.float32).swapaxes(0, 1),
        grad.astype(jnp.float32),
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
    )

    tile_info = gmm_v2.TileSizes(
        tile_m=128, tile_k=256, tile_n=256, bucket_base=128
    )
    tgmm_v2.validate_tgmm_inputs(group_sizes, num_local_groups, group_offset)
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        tile_info=tile_info,
        preferred_element_type=jnp.bfloat16,
        lhs_quant_dtype=jnp.float8_e4m3fn,
        rhs_quant_dtype=jnp.float8_e4m3fn,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    for i in empty_local:
      self.assertTrue(jnp.all(actual[i] == 0), f"group {i} is not zeroed")
    self.assertFalse(jnp.any(jnp.isnan(actual)), "zero-scale guard let NaN out")
    # Tolerance rationale: see test_tgmm_dynamic_quant_basic.
    chex.assert_trees_all_close(
        actual.astype(jnp.float32),
        expected,
        rtol=1e-1,
        atol=5e-1 * float(jnp.sqrt(jnp.mean(jnp.square(expected)))),
    )

  @pytest.mark.long
  def test_tgmm_dynamic_quant_padding(self):
    """Unaligned size_k and size_n together.

    in_size=255 leaves a pad column in the k-tile, so lhs_scale is computed over
    it; out_size=300 aligns to 384 across 3 n-tiles, so rhs_scale covers columns
    300..383 of the last one. k and n are independent output axes, so testing
    both at once cannot mask a bug in either.
    """
    if test_utils.get_tpu_version() < 7:
      self.skipTest("float8_e4m3fn matmul requires TPUv7+")

    batch_size, in_size, out_size, num_groups = 128, 255, 300, 5
    key1, key2 = jax.random.split(jax.random.key(0), 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)

    expected = reference_tgmm(
        lhs.astype(jnp.float32).swapaxes(0, 1),
        grad.astype(jnp.float32),
        group_sizes,
        num_groups,
    )

    tile_info = gmm_v2.TileSizes(
        tile_m=128, tile_k=256, tile_n=128, bucket_base=128
    )
    tgmm_v2.validate_tgmm_inputs(group_sizes, num_groups)
    actual = tgmm_v2.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_groups,
        tile_info=tile_info,
        preferred_element_type=jnp.bfloat16,
        lhs_quant_dtype=jnp.float8_e4m3fn,
        rhs_quant_dtype=jnp.float8_e4m3fn,
    )
    self.assertEqual(actual.shape, (num_groups, in_size, out_size))
    # Tolerance rationale: see test_tgmm_dynamic_quant_basic.
    chex.assert_trees_all_close(
        actual.astype(jnp.float32),
        expected,
        rtol=1e-1,
        atol=5e-1 * float(jnp.sqrt(jnp.mean(jnp.square(expected)))),
    )

  def test_tgmm_dynamic_quant_input_validation(self):
    """The API contract from validate_dynamic_quant."""
    lhs = jnp.zeros((128, 256), dtype=jnp.bfloat16)
    rhs = jnp.zeros((128, 256), dtype=jnp.bfloat16)
    rhs_scale = jnp.ones((1, 1, 256), dtype=jnp.float32)
    e4 = jnp.float8_e4m3fn

    with self.assertRaisesRegex(ValueError, "must be set together"):
      tgmm_v2.validate_dynamic_quant(lhs, rhs, None, e4, None)
    with self.assertRaisesRegex(ValueError, "must be set together"):
      tgmm_v2.validate_dynamic_quant(lhs, rhs, None, None, e4)
    with self.assertRaisesRegex(ValueError, "must differ from the input dtype"):
      tgmm_v2.validate_dynamic_quant(lhs, rhs, None, lhs.dtype, e4)
    with self.assertRaisesRegex(ValueError, "must differ from the input dtype"):
      tgmm_v2.validate_dynamic_quant(lhs, rhs, None, e4, rhs.dtype)
    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
      tgmm_v2.validate_dynamic_quant(lhs, rhs, rhs_scale, e4, e4)
    with self.assertRaisesRegex(ValueError, ">=16-bit"):
      tgmm_v2.validate_dynamic_quant(
          lhs.astype(jnp.float8_e5m2), rhs, None, e4, e4
      )
    with self.assertRaisesRegex(NotImplementedError, "integer input"):
      tgmm_v2.validate_dynamic_quant(
          lhs.astype(jnp.int32), rhs, None, e4, e4
      )

    # Wired into the public API, not merely reachable as a helper.
    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
      tgmm_v2.tgmm_v2(
          lhs,
          rhs,
          jnp.array([128], dtype=jnp.int32),
          1,
          rhs_scale,
          preferred_element_type=jnp.bfloat16,
          lhs_quant_dtype=e4,
          rhs_quant_dtype=e4,
      )


if __name__ == "__main__":
  absltest.main()
