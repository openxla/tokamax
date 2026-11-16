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
"""Tests for Pallas Mosaic GPU Gated Linear Unit."""

import re
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import batching
from tokamax._src import gpu_utils
from tokamax._src import hlo_utils
from tokamax._src import hlo_utils_common
from tokamax._src.ops.gated_linear_unit import base
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu as pl_glu
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu_common as common
from tokamax._src.ops.gated_linear_unit import test_base

_DISALLOWED_STABLEHLO_OPS_RE = re.compile(
    r"\bstablehlo\.(?:slice|dynamic_slice|concatenate|pad)\b"
)
_DISALLOWED_COMPILED_HLO_OPS_RE = re.compile(
    r"\b(?:slice|dynamic-slice|concatenate|pad)\(%"
)


class PallasMosaicGpuGatedLinearUnitTest(test_base.GatedLinearUnitTestBase):

  def __init__(self, *args):
    super().__init__(*args, glu_fn=pl_glu.PallasMosaicGpuGatedLinearUnit())

  def setUp(self):
    if not gpu_utils.has_mosaic_gpu_support():
      self.skipTest("Not supported on TPUs.")
    super().setUp()

  @parameterized.product(
      use_tuple_weights=(False, True),
      use_vmap_shape=(False, True),
  )
  def test_autotuning_configs(self, use_tuple_weights, use_vmap_shape):
    # Test that heuristics and autotuning configs are valid for fused & unfused.
    if use_vmap_shape:
      x = batching.BatchedShapeDtype(
          shape=(64, 128), dtype=jnp.bfloat16, vmap_axes=((0, 2),)
      )
      w_vmap_axes = (None,)
    else:
      x = jax.ShapeDtypeStruct(shape=(2, 256, 128), dtype=jnp.bfloat16)
      w_vmap_axes = ()

    if use_tuple_weights:
      w_leaf = batching.BatchedShapeDtype(
          shape=(128, 512), dtype=jnp.bfloat16, vmap_axes=w_vmap_axes
      )
      w = (w_leaf, w_leaf)
    else:
      w = batching.BatchedShapeDtype(
          shape=(128, 2, 512), dtype=jnp.bfloat16, vmap_axes=w_vmap_axes
      )

    ba = self._glu_fn.bind(x, w)
    self.assertIsInstance(ba.heuristics_config, common.Config)
    for cfg in self._glu_fn._get_autotuning_configs(ba):
      self.assertIsInstance(cfg, common.Config)

  @parameterized.parameters(
      dict(x_dtype=jnp.float64, w_dtypes=jnp.float64),
      dict(x_dtype=jnp.float64, w_dtypes=(jnp.float64, jnp.float64)),
      dict(x_dtype=jnp.bfloat16, w_dtypes=jnp.float16),
      dict(x_dtype=jnp.bfloat16, w_dtypes=(jnp.bfloat16, jnp.float16)),
      dict(x_dtype=jnp.float32, w_dtypes=jnp.bfloat16),
  )
  def test_unsupported_dtype_raises_not_implemented(self, x_dtype, w_dtypes):
    with jax.enable_x64(True):
      x = jnp.zeros((128, 64), dtype=x_dtype)
      if isinstance(w_dtypes, tuple):
        w = (
            jnp.zeros((64, 128), dtype=w_dtypes[0]),
            jnp.zeros((64, 128), dtype=w_dtypes[1]),
        )
      else:
        w = jnp.zeros((64, 2, 128), dtype=w_dtypes)
      with self.assertRaises(NotImplementedError):
        self._glu_fn(x, w)

  @parameterized.product(
      dtype=(jnp.bfloat16, jnp.float16),
      use_tuple_weights=(False, True),
  )
  def test_fused_and_unfused_weights_no_slices_or_joins(
      self, dtype, use_tuple_weights
  ):
    xla_glu = base.GatedLinearUnit()
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(11), 3)
    x = jax.random.normal(rng0, (256, 128), dtype=dtype)
    if use_tuple_weights:
      w = (
          jax.random.normal(rng1, (128, 256), dtype=dtype),
          jax.random.normal(rng2, (128, 256), dtype=dtype),
      )
    else:
      w = jax.random.normal(rng1, (128, 2, 256), dtype=dtype)

    f = jax.jit(lambda x, w: self._glu_fn(x, w, activation=jax.nn.silu))
    expected = xla_glu(x, w, activation=jax.nn.silu)
    actual = f(x, w)
    chex.assert_trees_all_close(actual, expected, atol=0.1, rtol=0.02)

    lowered = f.lower(x, w)
    stablehlo_text = lowered.as_text("stablehlo")
    self.assertIsNone(
        _DISALLOWED_STABLEHLO_OPS_RE.search(stablehlo_text),
        msg=f"Unexpected slice/join op in StableHLO:\n{stablehlo_text}",
    )
    compiled_hlo_text = lowered.compile().as_text()
    assert compiled_hlo_text is not None
    self.assertIsNone(
        _DISALLOWED_COMPILED_HLO_OPS_RE.search(compiled_hlo_text),
        msg=f"Unexpected slice/join op in HLO:\n{compiled_hlo_text}",
    )
    kernels = hlo_utils.get_kernel_info(lowered, include_xla_kernels=True)
    self.assertLen(kernels, 1)
    self.assertIsInstance(kernels[0], hlo_utils_common.MosaicGpuKernelInfo)
    self.assertLen(kernels[0].inputs, 3 if use_tuple_weights else 2)

  @parameterized.product(
      use_tuple_weights=(False, True),
      wg_dimension=(common.MatmulDimension.M, common.MatmulDimension.N),
      cluster_mn=((1, 1), (2, 1), (1, 2)),
  )
  def test_sm90_configs_fused_and_unfused(
      self, use_tuple_weights, wg_dimension, cluster_mn
  ):
    if not gpu_utils.is_sm90():
      self.skipTest("SM90-specific config test.")

    cluster_size_m, cluster_size_n = cluster_mn
    cfg = common.Config(
        tile_m=128,
        tile_n=64,
        tile_k=64,
        num_stages=2,
        epi_tile_m=64,
        epi_tile_n=32,
        wg_dimension=wg_dimension,
        cluster_size_m=cluster_size_m,
        cluster_size_n=cluster_size_n,
    )
    op = pl_glu.PallasMosaicGpuGatedLinearUnit(config=cfg)
    xla_glu = base.GatedLinearUnit()

    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(23), 3)
    x = jax.random.normal(rng0, (512, 128), dtype=jnp.bfloat16)
    if use_tuple_weights:
      w = (
          jax.random.normal(rng1, (128, 256), dtype=jnp.bfloat16),
          jax.random.normal(rng2, (128, 256), dtype=jnp.bfloat16),
      )
    else:
      w = jax.random.normal(rng1, (128, 2, 256), dtype=jnp.bfloat16)

    f = jax.jit(lambda x, w: op(x, w, activation=jax.nn.sigmoid))
    chex.assert_trees_all_close(
        f(x, w),
        xla_glu(x, w, activation=jax.nn.sigmoid),
        atol=0.1,
        rtol=0.02,
    )
    lowered = f.lower(x, w)
    kernels = hlo_utils.get_kernel_info(lowered, include_xla_kernels=True)
    self.assertLen(kernels, 1)
    self.assertLen(kernels[0].inputs, 3 if use_tuple_weights else 2)

  @parameterized.product(
      use_tuple_weights=(False, True),
      cluster_size_m=(1, 2),
      return_residuals=(False, True),
  )
  def test_sm100_configs_fused_and_unfused(
      self, use_tuple_weights, cluster_size_m, return_residuals
  ):
    if not gpu_utils.is_sm100():
      self.skipTest("SM100-specific config test.")

    cfg = common.Config(
        tile_m=128,
        tile_n=128,
        tile_k=64,
        num_stages=2,
        epi_tile_n=64,
        cluster_size_m=cluster_size_m,
        cluster_size_n=1,
    )
    op = pl_glu.PallasMosaicGpuGatedLinearUnit(config=cfg)
    xla_glu = base.GatedLinearUnit()

    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(37), 3)
    x = jax.random.normal(rng0, (267, 128), dtype=jnp.bfloat16)
    scale = jnp.bfloat16(1.0 / (128**0.5))
    if use_tuple_weights:
      w = (
          jax.random.normal(rng1, (128, 256), dtype=jnp.bfloat16) * scale,
          jax.random.normal(rng2, (128, 256), dtype=jnp.bfloat16) * scale,
      )
    else:
      w = jax.random.normal(rng1, (128, 2, 256), dtype=jnp.bfloat16) * scale

    f = jax.jit(
        lambda a, b: op(
            a, b, activation=jax.nn.silu, return_residuals=return_residuals
        )
    )
    chex.assert_trees_all_close(
        f(x, w),
        xla_glu(
            x, w, activation=jax.nn.silu, return_residuals=return_residuals
        ),
        atol=0.1,
        rtol=0.02,
    )
    lowered = f.lower(x, w)
    kernels = hlo_utils.get_kernel_info(lowered, include_xla_kernels=False)
    self.assertLen(kernels, 1)
    self.assertIsInstance(kernels[0], hlo_utils_common.MosaicGpuKernelInfo)
    self.assertLen(kernels[0].inputs, 3 if use_tuple_weights else 2)

  @parameterized.parameters(False, True)
  def test_vmap_x_fused_and_unfused(self, use_tuple_weights):
    xla_glu = base.GatedLinearUnit()
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(19), 3)
    x = jax.random.normal(rng0, (2, 128, 64), dtype=jnp.bfloat16)
    if use_tuple_weights:
      w = (
          jax.random.normal(rng1, (64, 128), dtype=jnp.bfloat16),
          jax.random.normal(rng2, (64, 128), dtype=jnp.bfloat16),
      )
    else:
      w = jax.random.normal(rng1, (64, 2, 128), dtype=jnp.bfloat16)

    f = jax.jit(
        jax.vmap(
            lambda xi, wi: self._glu_fn(xi, wi, activation=jax.nn.silu),
            in_axes=(0, None),
        )
    )
    f_ref = jax.jit(
        jax.vmap(
            lambda xi, wi: xla_glu(xi, wi, activation=jax.nn.silu),
            in_axes=(0, None),
        )
    )
    chex.assert_trees_all_close(f(x, w), f_ref(x, w), atol=0.1, rtol=0.02)

    lowered = f.lower(x, w)
    stablehlo_text = lowered.as_text("stablehlo")
    self.assertIsNone(
        _DISALLOWED_STABLEHLO_OPS_RE.search(stablehlo_text),
        msg=f"Unexpected slice/join op in StableHLO:\n{stablehlo_text}",
    )
    compiled_hlo_text = lowered.compile().as_text()
    assert compiled_hlo_text is not None
    self.assertIsNone(
        _DISALLOWED_COMPILED_HLO_OPS_RE.search(compiled_hlo_text),
        msg=f"Unexpected slice/join op in HLO:\n{compiled_hlo_text}",
    )
    kernels = hlo_utils.get_kernel_info(lowered, include_xla_kernels=True)
    self.assertLen(kernels, 1)
    self.assertIsInstance(kernels[0], hlo_utils_common.MosaicGpuKernelInfo)
    self.assertLen(kernels[0].inputs, 3 if use_tuple_weights else 2)
    self.assertEqual(kernels[0].inputs[0].shape, (256, 64))

  @parameterized.parameters(False, True)
  def test_vmap_weights_fused_and_unfused(self, use_tuple_weights):
    xla_glu = base.GatedLinearUnit()
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(31), 3)
    x = jax.random.normal(rng0, (2, 128, 64), dtype=jnp.bfloat16)
    if use_tuple_weights:
      w = (
          jax.random.normal(rng1, (2, 64, 128), dtype=jnp.bfloat16),
          jax.random.normal(rng2, (2, 64, 128), dtype=jnp.bfloat16),
      )
    else:
      w = jax.random.normal(rng1, (2, 64, 2, 128), dtype=jnp.bfloat16)

    f = jax.jit(
        jax.vmap(
            lambda xi, wi: self._glu_fn(xi, wi, activation=jax.nn.silu),
            in_axes=(0, 0),
        )
    )
    f_ref = jax.jit(
        jax.vmap(
            lambda xi, wi: xla_glu(xi, wi, activation=jax.nn.silu),
            in_axes=(0, 0),
        )
    )
    chex.assert_trees_all_close(f(x, w), f_ref(x, w), atol=0.1, rtol=0.02)

    lowered = f.lower(x, w)
    stablehlo_text = lowered.as_text("stablehlo")
    self.assertIsNone(
        _DISALLOWED_STABLEHLO_OPS_RE.search(stablehlo_text),
        msg=f"Unexpected slice/join op in StableHLO:\n{stablehlo_text}",
    )
    compiled_hlo_text = lowered.compile().as_text()
    assert compiled_hlo_text is not None
    self.assertIsNone(
        _DISALLOWED_COMPILED_HLO_OPS_RE.search(compiled_hlo_text),
        msg=f"Unexpected slice/join op in HLO:\n{compiled_hlo_text}",
    )
    kernels = hlo_utils.get_kernel_info(lowered, include_xla_kernels=True)
    self.assertLen(kernels, 1)
    self.assertIsInstance(kernels[0], hlo_utils_common.MosaicGpuKernelInfo)
    self.assertLen(kernels[0].inputs, 3 if use_tuple_weights else 2)

  @parameterized.product(
      shape=((0, 64, 128), (128, 0, 128), (128, 64, 0), (128, 33, 128)),
      use_tuple_weights=(False, True),
  )
  def test_empty_and_odd_shapes_raise_not_implemented(
      self, shape, use_tuple_weights
  ):
    m, k, n = shape
    x = jnp.zeros((m, k), dtype=jnp.bfloat16)
    if use_tuple_weights:
      w = (
          jnp.zeros((k, n), dtype=jnp.bfloat16),
          jnp.zeros((k, n), dtype=jnp.bfloat16),
      )
    else:
      w = jnp.zeros((k, 2, n), dtype=jnp.bfloat16)
    with self.assertRaises(NotImplementedError):
      self._glu_fn(x, w)

  @parameterized.product(
      dtype_and_precision=(
          (jnp.float32, jax.lax.Precision.DEFAULT),
          (jnp.float32, jax.lax.Precision.HIGHEST),
          (jnp.float32, jax.lax.DotAlgorithmPreset.BF16_BF16_F32),
          (jnp.float32, jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3),
          (jnp.float32, jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6),
          (jnp.float32, jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X9),
          (jnp.float32, jax.lax.DotAlgorithmPreset.TF32_TF32_F32_X3),
          (jnp.bfloat16, jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3),
          (jnp.bfloat16, jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6),
      ),
      use_tuple_weights=(False, True),
      return_residuals=(False, True),
  )
  def test_float32_and_dot_algorithm_presets(
      self, dtype_and_precision, use_tuple_weights, return_residuals
  ):
    dtype, precision = dtype_and_precision
    m, k, n = 67, 64, 128
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(59), 3)
    x = jax.random.normal(rng0, (m, k), dtype=dtype)
    scale = dtype(1.0 / (k**0.5))
    if use_tuple_weights:
      w = (
          jax.random.normal(rng1, (k, n), dtype=dtype) * scale,
          jax.random.normal(rng2, (k, n), dtype=dtype) * scale,
      )
    else:
      w = jax.random.normal(rng1, (k, 2, n), dtype=dtype) * scale

    f = jax.jit(
        lambda x_in, w_in: self._glu_fn(
            x_in,
            w_in,
            activation=jax.nn.silu,
            precision=precision,
            return_residuals=return_residuals,
        )
    )
    lowered = f.lower(x, w)
    self.assertIn("mosaic_gpu_v2", lowered.as_text("hlo"))
    kernels = hlo_utils.get_kernel_info(lowered, include_xla_kernels=False)
    self.assertLen(kernels, 1)
    self.assertIsInstance(kernels[0], hlo_utils_common.MosaicGpuKernelInfo)
    self.assertLen(kernels[0].inputs, 3 if use_tuple_weights else 2)

    actual = f(x, w)
    with jax.enable_x64(True):
      x64 = x.astype(jnp.float64)
      w64 = (
          jnp.stack(w, axis=-2).astype(jnp.float64)
          if isinstance(w, tuple)
          else w.astype(jnp.float64)
      )
      pre64 = jnp.einsum(
          "mk,kjn->mjn", x64, w64, precision=jax.lax.Precision.HIGHEST
      )
      ref_out = (jax.nn.silu(pre64[:, 0, :]) * pre64[:, 1, :]).astype(dtype)
      ref_res = pre64.astype(dtype)

    if dtype == jnp.bfloat16 or precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
      atol, rtol = 2e-2, 2e-2
    elif precision in (
        jax.lax.Precision.DEFAULT,
        jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3,
    ):
      atol, rtol = 2e-3, 2e-3
    else:
      atol, rtol = 1e-4, 1e-4

    if return_residuals:
      out, res = actual
      chex.assert_trees_all_close(out, ref_out, atol=atol, rtol=rtol)
      chex.assert_trees_all_close(res, ref_res, atol=atol, rtol=rtol)
    else:
      chex.assert_trees_all_close(actual, ref_out, atol=atol, rtol=rtol)

  @parameterized.product(
      m=(1, 3, 13, 64, 169),
      use_tuple_weights=(False, True),
  )
  def test_unaligned_m_vmap_and_vjp(self, m, use_tuple_weights):
    xla_glu = base.GatedLinearUnit()
    rng0, rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(71 + m), 4)
    x = jax.random.normal(rng0, (4, m, 64), dtype=jnp.bfloat16)
    scale = jnp.bfloat16(0.125)
    if use_tuple_weights:
      w = (
          jax.random.normal(rng1, (64, 128), dtype=jnp.bfloat16) * scale,
          jax.random.normal(rng2, (64, 128), dtype=jnp.bfloat16) * scale,
      )
    else:
      w = jax.random.normal(rng1, (64, 2, 128), dtype=jnp.bfloat16) * scale
    dout = jax.random.normal(rng3, (4, m, 128), dtype=jnp.bfloat16)

    def forward(x_in, w_in):
      return jax.vmap(
          lambda xi, wi: self._glu_fn(xi, wi, activation=jax.nn.silu),
          in_axes=(0, None),
      )(x_in, w_in)

    @jax.jit
    def run_vjp(x_in, w_in, dout_in):
      y, f_vjp = jax.vjp(forward, x_in, w_in)
      return y, *f_vjp(dout_in)

    lowered = run_vjp.lower(x, w, dout)
    self.assertIn("mosaic_gpu_v2", lowered.as_text("hlo"))
    y, dx, dw = run_vjp(x, w, dout)

    x_ref = x.astype(jnp.float32)
    w_ref = jax.tree.map(lambda t: t.astype(jnp.float32), w)
    y_ref, ref_vjp = jax.vjp(
        lambda a, b: xla_glu(a, b, activation=jax.nn.silu), x_ref, w_ref
    )
    dx_ref, dw_ref = ref_vjp(dout.astype(jnp.float32))
    chex.assert_trees_all_close(y, y_ref, atol=0.1, rtol=0.02)
    chex.assert_trees_all_close(dx, dx_ref, atol=0.5, rtol=0.05)
    chex.assert_trees_all_close(dw, dw_ref, atol=1.0, rtol=0.05)

  def test_symbolic_shapes_raise_not_implemented(self):
    (b_sym,) = jax.export.symbolic_shape("b")
    x = jax.ShapeDtypeStruct((b_sym, 128, 64), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((64, 2, 128), jnp.bfloat16)
    with self.assertRaises(NotImplementedError):
      self._glu_fn(x, w)

  def test_all_autotuning_configs_execute(self):
    xla_glu = base.GatedLinearUnit()
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(89), 2)
    x = jax.random.normal(rng0, (67, 64), dtype=jnp.bfloat16)
    w = (
        jax.random.normal(rng1, (64, 2, 256), dtype=jnp.bfloat16)
        * jnp.bfloat16(0.125)
    )
    expected = xla_glu(x, w, activation=jax.nn.silu)
    ba = self._glu_fn.bind(x, w, activation=jax.nn.silu)
    self.assertNotEmpty(ba.autotuning_configs)
    for cfg in sorted(ba.autotuning_configs, key=repr)[:12]:
      actual = jax.jit(
          lambda a, b, c=cfg: self._glu_fn.replace(config=c)(
              a, b, activation=jax.nn.silu
          )
      )(x, w)
      chex.assert_trees_all_close(actual, expected, atol=0.1, rtol=0.02)


if __name__ == "__main__":
  absltest.main()
