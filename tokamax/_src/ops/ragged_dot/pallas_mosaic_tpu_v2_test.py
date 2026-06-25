# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for Pallas Mosaic TPU v2 Ragged Dot (GMM v2)."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2_gmm_kernel as gmm_backend
from tokamax._src.ops.ragged_dot.gmm_v2_kernel_tests import pallas_mosaic_tpu_v2_kernel_test as kernel_test
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_backend


class PallasMosaicTpuV2OpParameterPipingTest(parameterized.TestCase):
  """Verifies PallasMosaicTpuV2RaggedDot pipes kwargs correctly to the kernel.

  Each test below mirrors the kernel test cases in
  `pallas_mosaic_tpu_v2_kernel_test.py`. Here we instead route the *same* inputs
  through `PallasMosaicTpuV2RaggedDot` Op and compare with the direct kernel
  call. Because both paths end up in the same kernel with the same default
  tiling, any disagreement means the API failed to thread a kwarg through to
  the kernel unchanged.
  """

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().generation < 6:
      self.skipTest("Only supported on TPU gen 6+.")
    super().setUp()

  def _assert_gmm_api_matches_kernel(
      self,
      lhs,
      rhs,
      group_sizes,
      *,
      op_kwargs,
      kernel_kwargs,
      atol=2e-2,
      rtol=2e-2
  ):
    op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot()
    via_op = op(lhs, rhs, group_sizes=group_sizes, **op_kwargs)
    via_kernel = gmm_backend.gmm_v2(
        lhs, rhs, group_sizes=group_sizes, **kernel_kwargs)
    chex.assert_trees_all_close(via_op, via_kernel, atol=atol, rtol=rtol)

  def test_gmm_basic_pipes(self):
    # Mirrors test_gmm_basic: exercises `rhs_bias` + `group_offset`.
    batch_size, in_size, out_size = 256, 256, 256
    num_groups, group_offset = 4, 1
    num_local_groups = num_groups - group_offset
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)

    lhs = jax.random.normal(k0, (batch_size, in_size), jnp.bfloat16)
    rhs = jax.random.normal(
        k1, (num_local_groups, in_size, out_size), jnp.bfloat16
    )
    rhs_bias = jax.random.normal(
        k2, (num_local_groups, 1, out_size), jnp.bfloat16
    )
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    kwargs = dict(
        rhs_bias=rhs_bias,
        group_offset=jnp.array([group_offset], jnp.int32),
    )
    self._assert_gmm_api_matches_kernel(
        lhs,
        rhs,
        group_sizes,
        op_kwargs=kwargs,
        kernel_kwargs=kwargs,
    )

  def test_tgmm_drhs_pipes(self):
    """Exercises the drhs (tgmm) path: the op must match a direct `tgmm_v2`.

    This one mirrors
    `test_tgmm_basic` in the kernel test but routes through the op. drhs is
    op-only: it needs the local group count (`num_actual_groups`, normally
    injected by the custom vjp), so we set it on a dedicated op instance, and
    `api.ragged_dot_general` rejects a direct drhs call.
    """
    m, k, n, num_groups = 256, 256, 256, 4
    k0, k1 = jax.random.split(jax.random.key(0), 2)

    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)  # [m, k]
    grad = jax.random.normal(k1, (m, n), jnp.bfloat16)  # [m, n]
    group_sizes = kernel_test.get_group_sizes(m, num_groups)

    drhs_op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot(
        num_actual_groups=num_groups
    )
    via_op = drhs_op(
        lhs,
        grad,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=pallas_mosaic_tpu_v2.DRHS_RAGGED_DOT_DIM_NUMS,
        preferred_element_type=jnp.bfloat16,
    )
    via_kernel = tgmm_backend.tgmm_v2(
        lhs,
        grad,
        group_sizes=group_sizes,
        num_actual_groups=num_groups,
        preferred_element_type=jnp.bfloat16,
    )
    chex.assert_trees_all_close(via_op, via_kernel, atol=2e-2, rtol=2e-2)

  def test_tgmm_drhs_with_tile_info_pipes(self):
    """Mirrors `test_tgmm_with_tile_info`.

    The op selects tiling from its `Config`, so we set `config` to the same
    tiles passed as `tile_info` to the direct `tgmm_v2` call; the two must then
    produce identical results across tile sizes.
    """
    tile_m, tile_k, tile_n = 256, 256, 256
    m, k, n, num_groups = 256, 1024, 1024, 16
    k0, k1 = jax.random.split(jax.random.key(0), 2)

    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)  # [m, k]
    grad = jax.random.normal(k1, (m, n), jnp.bfloat16)  # [m, n]
    group_sizes = kernel_test.get_group_sizes(m, num_groups)

    config = pallas_mosaic_tpu_v2.Config(
        tile_m=tile_m, tile_k=tile_k, tile_n=tile_n
    )
    drhs_op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot(
        num_actual_groups=num_groups, config=config
    )
    via_op = drhs_op(
        lhs,
        grad,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=pallas_mosaic_tpu_v2.DRHS_RAGGED_DOT_DIM_NUMS,
        preferred_element_type=jnp.bfloat16,
    )
    via_kernel = tgmm_backend.tgmm_v2(
        lhs,
        grad,
        group_sizes=group_sizes,
        num_actual_groups=num_groups,
        preferred_element_type=jnp.bfloat16,
        tile_info=gmm_backend.TileSizes(
            tile_m=tile_m, tile_k=tile_k, tile_n=tile_n
        ),
    )
    chex.assert_trees_all_close(via_op, via_kernel, atol=2e-2, rtol=2e-2)

  def test_tgmm_drhs_with_rhs_scale_pipes(self):
    """Mirrors `test_tgmm_with_tile_info`.

    The op selects tiling from its `Config`, so we set `config` to the same
    tiles passed as `tile_info` to the direct `tgmm_v2` call; the two must then
    produce identical results across tile sizes.
    """
    lhs_dtype, rhs_quant_dtype = jnp.float8_e4m3fn, jnp.float8_e5m2
    m, k, n, num_groups = 256, 1024, 1024, 16
    k0, k1 = jax.random.split(jax.random.key(0), 2)

    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16).astype(
        lhs_dtype
    )  # [m, k]
    grad = jax.random.normal(k1, (m, n), jnp.float32)  # [m, n]
    grad_q, grad_scale = kernel_test.quantize_tensor(
        grad,
        rhs_quant_dtype,
        axis=0,
        block_size=m,
    )
    grad_scale = jnp.expand_dims(grad_scale, axis=1)  # [1, 1, N]
    assert grad_scale.shape == (1, 1, n)
    group_sizes = kernel_test.get_group_sizes(m, num_groups)

    drhs_op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot(
        num_actual_groups=num_groups
    )
    via_op = drhs_op(
        lhs,
        grad_q,
        group_sizes=group_sizes,
        rhs_scale=grad_scale,
        ragged_dot_dimension_numbers=pallas_mosaic_tpu_v2.DRHS_RAGGED_DOT_DIM_NUMS,
        preferred_element_type=jnp.bfloat16,
    )
    via_kernel = tgmm_backend.tgmm_v2(
        lhs,
        grad_q,
        group_sizes=group_sizes,
        rhs_scale=grad_scale,
        num_actual_groups=num_groups,
        preferred_element_type=jnp.bfloat16,
    )
    chex.assert_trees_all_close(via_op, via_kernel, atol=2e-2, rtol=2e-2)

  def test_gmm_weight_quantized_pipes(self):
    # Mirrors test_gmm_weight_quantized: jnp.float8_e4m3fn
    # `rhs` + `rhs_scale` + `rhs_bias` +
    # `group_offset`, with `maybe_quantize_lhs=False`.
    batch_size, in_size, out_size = 128, 512, 512
    num_groups, group_offset, block_size = 4, 1, 256
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_q, rhs_scale = kernel_test.quantize_tensor(
        rhs, jnp.float8_e4m3fn, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    rhs_bias = jax.random.normal(
        key, (num_local_groups, 1, out_size), jnp.bfloat16
    )
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    kwargs = dict(
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=jnp.array([group_offset], jnp.int32),
        maybe_quantize_lhs=False,
    )
    self._assert_gmm_api_matches_kernel(
        lhs,
        rhs_q,
        group_sizes,
        op_kwargs=kwargs,
        kernel_kwargs=kwargs,
    )

  def test_gmm_activation_weight_quantized_pipes(self):
    # Mirrors test_gmm_activation_weight_quantized:
    # jnp.float8_e4m3fn `rhs` + `rhs_scale` with
    # `maybe_quantize_lhs=True` (the lhs-quantization path).
    batch_size, in_size, out_size = 128, 512, 512
    num_groups, block_size = 4, 512
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_q, rhs_scale = kernel_test.quantize_tensor(
        rhs, jnp.float8_e4m3fn, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    kwargs = dict(rhs_scale=rhs_scale, maybe_quantize_lhs=True)
    self._assert_gmm_api_matches_kernel(
        lhs,
        rhs_q,
        group_sizes,
        op_kwargs=kwargs,
        kernel_kwargs=kwargs,
    )

  def test_gmm_implicit_padding_pipes(self):
    # Mirrors test_gmm_implicit_padding: non-tile-aligned `in_size`/`out_size`
    # with `rhs_bias`.
    batch_size, in_size, out_size = 128, 255, 255
    num_groups = 4
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)

    lhs = jax.random.normal(k0, (batch_size, in_size), jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_groups, in_size, out_size), jnp.bfloat16)
    rhs_bias = jax.random.normal(k2, (num_groups, 1, out_size), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    kwargs = dict(rhs_bias=rhs_bias)
    self._assert_gmm_api_matches_kernel(
        lhs, rhs, group_sizes, op_kwargs=kwargs, kernel_kwargs=kwargs
    )

  def test_gmm_weight_quantized_padding_pipes(self):
    # Mirrors test_gmm_weight_quantized_padding:
    # jnp.float8_e4m3fn `rhs` + `rhs_scale` +
    # `rhs_bias` with a non-tile-aligned `out_size`.
    batch_size, in_size, out_size = 128, 512, 500
    num_groups, block_size = 4, 512
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), jnp.bfloat16)
    rhs = jax.random.normal(key, (num_groups, in_size, out_size), jnp.bfloat16)
    rhs_q, rhs_scale = kernel_test.quantize_tensor(
        rhs, jnp.float8_e4m3fn, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    rhs_bias = jax.random.normal(key, (num_groups, 1, out_size), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    kwargs = dict(
        rhs_scale=rhs_scale, rhs_bias=rhs_bias, maybe_quantize_lhs=False
    )
    self._assert_gmm_api_matches_kernel(
        lhs,
        rhs_q,
        group_sizes,
        op_kwargs=kwargs,
        kernel_kwargs=kwargs,
    )

  def test_gmm_nonlocal_groups_produce_zeros_pipes(self):
    # Mirrors test_gmm_nonlocal_groups_produce_zeros:
    # a `group_offset` that makes
    # some global groups non-local (group < 0 or group >= num_local_groups), so
    # the kernel zero-fills their rows. `group_sizes` has `num_groups` entries
    # while `rhs` carries only `num_local_groups` weights.
    batch_size, in_size, out_size = 128, 256, 256
    num_groups, group_offset, num_local_groups = 16, 2, 4
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16
    )
    rhs_bias = jax.random.normal(
        key, (num_local_groups, 1, out_size), jnp.bfloat16
    )
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    kwargs = dict(
        rhs_bias=rhs_bias,
        group_offset=jnp.array([group_offset], jnp.int32),
    )
    self._assert_gmm_api_matches_kernel(
        lhs,
        rhs,
        group_sizes,
        op_kwargs=kwargs,
        kernel_kwargs=kwargs,
    )

  def test_gmm_fused_activation_pipes(self):
    # Mirrors test_gmm_fused_activation: `fuse_act` (GLU-style, halves `n`) plus
    # `rhs_bias`. `out_size` is a multiple of 2 * num_lanes (256) so the gate/up
    # split validates.
    batch_size, in_size, out_size = 128, 512, 512
    num_groups = 4
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)

    lhs = jax.random.uniform(k0, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        k1, (num_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_bias = jax.random.normal(k2, (num_groups, 1, out_size), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    op_kwargs = dict(rhs_bias=rhs_bias, fuse_gateup_activation="silu")
    kernel_kwargs = dict(rhs_bias=rhs_bias, fuse_act="silu")
    self._assert_gmm_api_matches_kernel(
        lhs,
        rhs,
        group_sizes,
        op_kwargs=op_kwargs,
        kernel_kwargs=kernel_kwargs,
    )

  def test_gmm_preferred_element_type_pipes(self):
    """`preferred_element_type` must reach the kernel and set the output dtype.

    With a non-default output dtype (f32), a wrapper that dropped the kwarg
    would return bf16 (`lhs.dtype`); the explicit dtype checks below catch that
    even though the values themselves would still be numerically close.
    """
    batch_size, in_size, out_size = 256, 256, 256
    num_groups = 4
    k0, k1 = jax.random.split(jax.random.key(0), 2)

    lhs = jax.random.normal(k0, (batch_size, in_size), jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_groups, in_size, out_size), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot()
    via_op = op(
        lhs, rhs, group_sizes=group_sizes, preferred_element_type=jnp.float32
    )
    via_kernel = gmm_backend.gmm_v2(
        lhs, rhs, group_sizes=group_sizes, preferred_element_type=jnp.float32
    )
    self.assertEqual(via_op.dtype, jnp.float32)
    self.assertEqual(via_kernel.dtype, jnp.float32)
    chex.assert_trees_all_close(via_op, via_kernel, atol=2e-2, rtol=2e-2)

  def test_gmm_precision_pipes(self):
    """`precision` must thread through the API to the kernel without error.

    The v2 kernel ignores `precision` (it exists only for API compatibility and
    is `del`-eted inside `gmm_v2`), so this is an accepts-and-forwards check
    rather than a numeric one: the API must accept a non-default precision and
    still produce the same result as the direct kernel call.
    """
    batch_size, in_size, out_size = 256, 256, 256
    num_groups = 4
    k0, k1 = jax.random.split(jax.random.key(0), 2)

    lhs = jax.random.normal(k0, (batch_size, in_size), jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_groups, in_size, out_size), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    kwargs = dict(precision=jax.lax.Precision.HIGHEST)
    self._assert_gmm_api_matches_kernel(
        lhs,
        rhs,
        group_sizes,
        op_kwargs=kwargs,
        kernel_kwargs=kwargs,
    )

  def test_gmm_tpu_inference(self):
    """Mirrors how tpu-inference's fused MoE invokes gmm_v2.
    """
    tokens, hidden, intermediate = 256, 256, 256
    n = 2 * intermediate  # fused gate/up; `fuse_act` halves it back down.
    num_experts = 4
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)

    x = jax.random.uniform(k0, (tokens, hidden), jnp.bfloat16, -1, 1)
    w1 = jax.random.uniform(k1, (num_experts, hidden, n), jnp.bfloat16, -1, 1)
    w1_bias = jax.random.normal(k2, (num_experts, 1, n), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(tokens, num_experts)
    # tpu-inference passes `group_offset[0]` (a 0-d scalar) from
    # `jnp.array([0])`.
    group_offset = jnp.array([0], jnp.int32)[0]

    op_kwargs = dict(
        rhs_bias=w1_bias,
        group_offset=group_offset,
        fuse_gateup_activation="silu",
        zero_initialize=False,
        preferred_element_type=x.dtype,
    )
    kernel_kwargs = dict(
        rhs_bias=w1_bias,
        group_offset=group_offset,
        fuse_act="silu",
        zero_initialize=False,
        preferred_element_type=x.dtype,
    )
    self._assert_gmm_api_matches_kernel(
        x,
        w1,
        group_sizes,
        op_kwargs=op_kwargs,
        kernel_kwargs=kernel_kwargs,
    )

  def test_gmm_maxtext(self):
    m, k, n, num_groups = 256, 256, 256, 4
    block_size = 256
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)

    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_groups, k, n), jnp.bfloat16)
    grad = jax.random.normal(k2, (m, n), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(m, num_groups)

    # MaxText quantizes the expert weights and passes `rhs_scale` to the forward
    # gmm; mirror that with a quantized rhs here.
    rhs_q, rhs_scale = kernel_test.quantize_tensor(
        rhs, jnp.float8_e4m3fn, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot()
    # `drhs` (tgmm) needs the local group count, which the custom vjp normally
    # injects; for a *direct* call we construct an op with it set.
    drhs_op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot(
        num_actual_groups=num_groups
    )

    # Forward: gmm over the quantized weights (DEFAULT dims).
    out = op(
        lhs,
        rhs_q,
        group_sizes=group_sizes,
        rhs_scale=rhs_scale,
        preferred_element_type=jnp.bfloat16,
    )
    # Backward dlhs: gmm over a transposed rhs (DLHS dims; the op transposes the
    # rhs internally).
    dlhs = op(
        grad,
        rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=pallas_mosaic_tpu_v2.DLHS_RAGGED_DOT_DIM_NUMS,
        preferred_element_type=lhs.dtype,
    )
    # Backward drhs: tgmm (DRHS dims).
    drhs = drhs_op(
        lhs,
        grad,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=pallas_mosaic_tpu_v2.DRHS_RAGGED_DOT_DIM_NUMS,
        preferred_element_type=rhs.dtype,
    )

    # References: the exact kernel calls the op makes internally for each path.
    # `maybe_quantize_lhs=False` matches the op default (gmm_v2's own default is
    # True, so it must be set explicitly here).
    out_ref = gmm_backend.gmm_v2(
        lhs,
        rhs_q,
        group_sizes=group_sizes,
        rhs_scale=rhs_scale,
        maybe_quantize_lhs=False,
        preferred_element_type=jnp.bfloat16,
    )
    dlhs_ref = gmm_backend.gmm_v2(
        grad,
        rhs.swapaxes(1, 2),
        group_sizes=group_sizes,
        preferred_element_type=lhs.dtype,
    )
    drhs_ref = tgmm_backend.tgmm_v2(
        lhs,
        grad,
        group_sizes=group_sizes,
        num_actual_groups=num_groups,
        preferred_element_type=rhs.dtype,
    )

    chex.assert_trees_all_close(out, out_ref, atol=2e-2, rtol=2e-2)
    chex.assert_trees_all_close(dlhs, dlhs_ref, atol=2e-2, rtol=2e-2)
    chex.assert_trees_all_close(drhs, drhs_ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
  absltest.main()
