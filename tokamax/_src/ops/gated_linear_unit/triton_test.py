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
import re
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src import hlo_utils
from tokamax._src import hlo_utils_common
from tokamax._src.ops.gated_linear_unit import base
from tokamax._src.ops.gated_linear_unit import test_base

try:
  import jax_triton  # pylint: disable=g-import-not-at-top,unused-import
  import triton  # pylint: disable=g-import-not-at-top
except ImportError:
  jttl = None
  triton = None
  triton_glu = None
  _JAX_TRITON_AVAILABLE = False
else:
  from jax_triton import triton_lib as jttl  # pylint: disable=g-import-not-at-top
  from tokamax._src.ops.gated_linear_unit import triton as triton_glu  # pylint: disable=g-import-not-at-top

  _JAX_TRITON_AVAILABLE = True


def _triton_version() -> tuple[int, ...]:
  assert triton is not None
  return tuple(int(x) for x in triton.__version__.split(".")[:2])


_DISALLOWED_STABLEHLO_OPS_RE = re.compile(
    r"\bstablehlo\.(?:slice|dynamic_slice|concatenate|pad)\b"
)
_DISALLOWED_COMPILED_HLO_OPS_RE = re.compile(
    r"\b(?:slice|dynamic-slice|concatenate|pad)\(%"
)


@absltest.skipIf(not _JAX_TRITON_AVAILABLE, "Requires jax_triton.")
class TritonGatedLinearUnitTest(test_base.GatedLinearUnitTestBase):

  def __init__(self, *args):
    glu_fn = (
        triton_glu.TritonGatedLinearUnit() if triton_glu is not None else None
    )
    super().__init__(*args, glu_fn=glu_fn)

  def setUp(self):
    if not _JAX_TRITON_AVAILABLE:
      self.skipTest("Requires jax_triton.")
    if not gpu_utils.has_triton_support():
      self.skipTest("Triton not supported on this platform.")
    super().setUp()

  @parameterized.parameters(False, True)
  def test_autotuning_search_space(self, use_tuple_weights):
    assert triton_glu is not None
    m = 256
    n = 256
    k = 64
    glu = triton_glu.TritonGatedLinearUnit()
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(0), 2)
    x = jax.random.normal(rng0, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(rng1, (k, 2, n), dtype=jnp.bfloat16)
    if use_tuple_weights:
      w = (w[:, 0], w[:, 1])

    glu_bound_args = glu.bind(x, w)
    autotuning_configs = glu_bound_args.autotuning_configs
    self.assertNotEmpty(autotuning_configs)

  @parameterized.named_parameters(
      dict(testcase_name="bfloat16", dtype=jnp.bfloat16, precision=None),
      dict(testcase_name="float16", dtype=jnp.float16, precision=None),
      dict(testcase_name="float32_tf32", dtype=jnp.float32, precision=None),
      dict(
          testcase_name="float32_ieee",
          dtype=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      ),
  )
  def test_unaligned_shapes_and_dtypes(self, dtype, precision):
    self._run_test(
        m=65,
        n=70,
        k=35,
        dtype=dtype,
        activation=jax.nn.silu,
        precision=precision,
    )

  @parameterized.named_parameters(
      dict(testcase_name="single_row_m1", m=1, k=35, n=19, activation="silu"),
      dict(
          testcase_name="unaligned_m_only", m=16, k=32, n=128, activation="silu"
      ),
      dict(
          testcase_name="unaligned_n_only", m=64, k=32, n=8, activation="silu"
      ),
      dict(
          testcase_name="single_tile_all_unaligned",
          m=17,
          k=35,
          n=19,
          activation="silu",
      ),
      dict(
          testcase_name="batched",
          batch=(2, 3),
          m=11,
          k=35,
          n=19,
          activation="silu",
      ),
      dict(
          testcase_name="tuple_weights",
          m=17,
          k=35,
          n=19,
          activation="silu",
          use_tuple_weights=True,
      ),
      dict(
          testcase_name="no_activation",
          m=65,
          k=35,
          n=70,
          activation=None,
      ),
      dict(
          testcase_name="large_tile_config",
          m=130,
          k=65,
          n=260,
          activation="silu",
          config_kwargs=dict(
              block_m=128, block_n=256, block_k=32, num_warps=8, num_stages=4
          ),
      ),
      dict(
          testcase_name="single_stage_config",
          m=35,
          k=35,
          n=37,
          activation="silu",
          config_kwargs=dict(
              block_m=32, block_n=32, block_k=32, num_warps=4, num_stages=1
          ),
      ),
  )
  def test_unaligned_variants(self, config_kwargs=None, **kwargs):
    self.assertIsNotNone(triton_glu)
    assert triton_glu is not None
    glu_fn = None
    if config_kwargs is not None:
      glu_fn = triton_glu.TritonGatedLinearUnit(
          config=triton_glu.Config(**config_kwargs)
      )
    self._run_test(dtype=jnp.float16, glu_fn=glu_fn, **kwargs)

  @parameterized.named_parameters(
      dict(
          testcase_name="all_aligned",
          shape=(64, 32, 64),
          expected_masked_loads=0,
          expected_masked_stores=0,
      ),
      dict(
          testcase_name="all_aligned_pipelined",
          shape=(64, 64, 64),
          expected_masked_loads=0,
          expected_masked_stores=0,
      ),
      dict(
          testcase_name="all_aligned_residuals",
          shape=(64, 32, 64),
          return_residuals=True,
          expected_masked_loads=0,
          expected_masked_stores=0,
      ),
      dict(
          testcase_name="unaligned_m_only",
          shape=(19, 32, 64),
          expected_masked_loads=1,
          expected_masked_stores=1,
      ),
      dict(
          testcase_name="unaligned_n_only",
          shape=(64, 32, 23),
          expected_masked_loads=2,
          expected_masked_stores=1,
      ),
      dict(
          testcase_name="unaligned_k_only",
          shape=(64, 35, 64),
          expected_masked_loads=3,
          expected_masked_stores=0,
      ),
      dict(
          testcase_name="unaligned_m_and_k",
          shape=(19, 35, 64),
          expected_masked_loads=3,
          expected_masked_stores=1,
          expected_andi=1,
      ),
      dict(
          testcase_name="unaligned_k_and_n",
          shape=(64, 35, 23),
          expected_masked_loads=3,
          expected_masked_stores=1,
          expected_andi=1,
      ),
      dict(
          testcase_name="unaligned_m_and_n",
          shape=(19, 32, 23),
          expected_masked_loads=3,
          expected_masked_stores=1,
          expected_andi=1,
      ),
      dict(
          testcase_name="all_unaligned",
          shape=(19, 35, 23),
          expected_masked_loads=3,
          expected_masked_stores=1,
          expected_andi=3,
      ),
      dict(
          testcase_name="unaligned_m_pipelined_residuals",
          shape=(19, 64, 64),
          return_residuals=True,
          expected_masked_loads=1,
          expected_masked_stores=3,
      ),
  )
  def test_load_mask_codegen(
      self,
      shape,
      expected_masked_loads,
      expected_masked_stores,
      expected_andi=0,
      return_residuals=False,
  ):
    self.assertIsNotNone(jttl)
    self.assertIsNotNone(triton)
    self.assertIsNotNone(triton_glu)
    assert jttl is not None and triton is not None and triton_glu is not None
    m, k, n = shape
    config = triton_glu.Config(
        block_m=32, block_n=32, block_k=32, num_warps=4, num_stages=4
    )
    glu = triton_glu.TritonGatedLinearUnit(config=config)
    x = jax.ShapeDtypeStruct((m, k), jnp.float16)
    w = jax.ShapeDtypeStruct((k, 2, n), jnp.float16)

    compiled = []
    orig_compile = jttl.compile_ttir_inplace

    def capture_compile(ttir, backend, options, gpu_target):
      ttir_str = str(ttir)
      result = orig_compile(ttir, backend, options, gpu_target)
      compiled.append((ttir_str, result.binary, gpu_target.backend))
      return result

    fresh_kernel = triton.jit(
        triton_glu._gated_linear_unit_kernel.fn  # pylint: disable=protected-access
    )
    with (
        mock.patch.object(
            triton_glu, "_gated_linear_unit_kernel", new=fresh_kernel
        ),
        mock.patch.object(
            jttl, "compile_ttir_inplace", new=capture_compile
        ),
    ):
      jax.jit(
          lambda x, w: glu(
              x, w, activation=jax.nn.silu, return_residuals=return_residuals
          )
      ).lower(x, w)

    self.assertLen(compiled, 1)
    ttir, ptx, target_backend = compiled[0]

    loads = re.findall(r"tt\.load\s+([^{:]+)", ttir)
    stores = re.findall(r"tt\.store\s+([^{:]+)", ttir)
    self.assertLen(loads, 3)
    self.assertLen(stores, 3 if return_residuals else 1)
    unmasked_loads = sum(operands.count(",") == 0 for operands in loads)
    masked_loads = sum(operands.count(",") == 2 for operands in loads)
    masked_stores = sum(operands.count(",") >= 2 for operands in stores)
    self.assertEqual(unmasked_loads, 3 - expected_masked_loads)
    self.assertEqual(masked_loads, expected_masked_loads)
    self.assertEqual(masked_stores, expected_masked_stores)
    and_mask_calls = re.findall(
        r"tt\.call\s+@\S*_and_masks\S*\([^)\n]*,", ttir
    )
    tensor_andi = re.findall(r"arith\.andi\b[^\n]*tensor<32x32xi1>", ttir)
    self.assertLen(and_mask_calls, expected_andi)
    self.assertLen(tensor_andi, 1 if expected_andi > 0 else 0)

    if target_backend == "cuda":
      self.assertIsInstance(ptx, str)
      self.assertIn(".version", ptx)
      self.assertEmpty(re.findall(r"setp\.\w+\.[su]64", ptx))
      masked_global_loads = re.findall(
          r"@%p\d+\s+ld\.global|cp\.async[^;]+,\s*0x\w+,\s*%p\d+", ptx
      )
      # Multi-tile K loops predicate cp.async prefetches per pipeline stage.
      if k <= config.block_k:
        unmasked_global_loads = re.findall(
            r"(?:^|\n)\s*ld\.global|cp\.async[^;]+,\s*0x\w+,\s*0x\w+", ptx
        )
        if expected_masked_loads > 0:
          self.assertNotEmpty(masked_global_loads)
        else:
          self.assertEmpty(masked_global_loads)
        if expected_masked_loads < 3:
          self.assertNotEmpty(unmasked_global_loads)
      elif _triton_version() >= (3, 9):
        # Triton < 3.9 doesn't predicate pipelined cp.async prefetches this way.
        self.assertNotEmpty(masked_global_loads)
      masked_global_stores = re.findall(r"@%p\d+\s+st\.global", ptx)
      if expected_masked_stores > 0:
        self.assertNotEmpty(masked_global_stores)
      else:
        self.assertEmpty(masked_global_stores)

  @parameterized.product(
      dtype=(jnp.bfloat16, jnp.float16, jnp.float32),
      use_tuple_weights=(False, True),
      return_residuals=(False, True),
  )
  def test_fused_and_unfused_weights_no_slices_or_joins(
      self, dtype, use_tuple_weights, return_residuals
  ):
    assert triton_glu is not None
    glu = triton_glu.TritonGatedLinearUnit()
    xla_glu = base.GatedLinearUnit()

    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(42), 3)
    x = jax.random.normal(rng0, (128, 96), dtype=dtype)
    if use_tuple_weights:
      w = (
          jax.random.normal(rng1, (96, 160), dtype=dtype),
          jax.random.normal(rng2, (96, 160), dtype=dtype),
      )
    else:
      w = jax.random.normal(rng1, (96, 2, 160), dtype=dtype)

    @jax.jit
    def f(x, weights):
      return glu(
          x,
          weights,
          activation=jax.nn.swish,
          precision=jax.lax.Precision.HIGHEST,
          return_residuals=return_residuals,
      )

    expected = xla_glu(
        x,
        w,
        activation=jax.nn.swish,
        precision=jax.lax.Precision.HIGHEST,
        return_residuals=return_residuals,
    )
    actual = f(x, w)
    tol = 1e-4 if dtype == jnp.float32 else 0.1
    chex.assert_trees_all_close(actual, expected, atol=tol, rtol=0.02)

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
    self.assertIsInstance(kernels[0], hlo_utils_common.TritonKernelInfo)
    self.assertLen(kernels[0].inputs, 3 if use_tuple_weights else 2)

  @parameterized.parameters(False, True)
  def test_vmap_x_fused_and_unfused(self, use_tuple_weights):
    assert triton_glu is not None
    glu = triton_glu.TritonGatedLinearUnit()
    xla_glu = base.GatedLinearUnit()

    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(7), 3)
    x = jax.random.normal(rng0, (4, 64, 64), dtype=jnp.bfloat16)
    if use_tuple_weights:
      w = (
          jax.random.normal(rng1, (64, 128), dtype=jnp.bfloat16),
          jax.random.normal(rng2, (64, 128), dtype=jnp.bfloat16),
      )
    else:
      w = jax.random.normal(rng1, (64, 2, 128), dtype=jnp.bfloat16)

    f = jax.jit(
        jax.vmap(
            lambda xi, wi: glu(xi, wi, activation=jax.nn.silu),
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
    self.assertLen(kernels[0].inputs, 3 if use_tuple_weights else 2)
    self.assertEqual(kernels[0].inputs[0].shape, (256, 64))

  @parameterized.product(
      use_tuple_weights=(False, True),
      return_residuals=(False, True),
  )
  def test_vmap_weights_fused_and_unfused(
      self, use_tuple_weights, return_residuals
  ):
    assert triton_glu is not None
    glu = triton_glu.TritonGatedLinearUnit()
    xla_glu = base.GatedLinearUnit()

    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(13), 3)
    x = jax.random.normal(rng0, (3, 64, 64), dtype=jnp.bfloat16)
    if use_tuple_weights:
      w = (
          jax.random.normal(rng1, (3, 64, 96), dtype=jnp.bfloat16),
          jax.random.normal(rng2, (3, 64, 96), dtype=jnp.bfloat16),
      )
    else:
      w = jax.random.normal(rng1, (3, 64, 2, 96), dtype=jnp.bfloat16)

    f = jax.jit(
        jax.vmap(
            lambda xi, wi: glu(
                xi,
                wi,
                activation=jax.nn.silu,
                return_residuals=return_residuals,
            ),
            in_axes=(0, 0),
        )
    )
    f_ref = jax.jit(
        jax.vmap(
            lambda xi, wi: xla_glu(
                xi,
                wi,
                activation=jax.nn.silu,
                return_residuals=return_residuals,
            ),
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
    self.assertLen(kernels[0].inputs, 3 if use_tuple_weights else 2)

  @parameterized.product(
      shape=((1, 4096, 33), (0, 64, 96), (64, 0, 96)),
      use_tuple_weights=(False, True),
  )
  def test_single_token_and_empty_shapes(self, shape, use_tuple_weights):
    assert triton_glu is not None
    glu = triton_glu.TritonGatedLinearUnit()
    xla_glu = base.GatedLinearUnit()

    m, k, n = shape
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(29), 3)
    x = jax.random.normal(rng0, (m, k), dtype=jnp.bfloat16)
    if k > 0:
      x = (x / (k**0.5)).astype(x.dtype)
    if use_tuple_weights:
      w = (
          jax.random.normal(rng1, (k, n), dtype=jnp.bfloat16),
          jax.random.normal(rng2, (k, n), dtype=jnp.bfloat16),
      )
    else:
      w = jax.random.normal(rng1, (k, 2, n), dtype=jnp.bfloat16)

    f = jax.jit(lambda x, w: glu(x, w, activation=jax.nn.silu))
    chex.assert_trees_all_close(
        f(x, w), xla_glu(x, w, activation=jax.nn.silu), atol=0.1, rtol=0.02
    )

  @parameterized.product(
      precision=(
          jax.lax.Precision.DEFAULT,
          jax.lax.Precision.HIGHEST,
          jax.lax.DotAlgorithmPreset.TF32_TF32_F32,
          jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
          jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3,
          jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6,
          jax.lax.DotAlgorithmPreset.F32_F32_F32,
      ),
      use_tuple_weights=(False, True),
      return_residuals=(False, True),
  )
  def test_float32_precisions_and_presets(
      self, precision, use_tuple_weights, return_residuals
  ):
    assert triton_glu is not None
    glu = triton_glu.TritonGatedLinearUnit()
    m, k, n = 67, 128, 128
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(101), 3)
    x = jax.random.normal(rng0, (m, k), dtype=jnp.float32)
    scale = jnp.float32(1.0 / (k**0.5))
    if use_tuple_weights:
      w = (
          jax.random.normal(rng1, (k, n), dtype=jnp.float32) * scale,
          jax.random.normal(rng2, (k, n), dtype=jnp.float32) * scale,
      )
    else:
      w = jax.random.normal(rng1, (k, 2, n), dtype=jnp.float32) * scale

    f = jax.jit(
        lambda x_in, w_in: glu(
            x_in,
            w_in,
            activation=jax.nn.silu,
            precision=precision,
            return_residuals=return_residuals,
        )
    )
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
      ref_out = (jax.nn.silu(pre64[:, 0, :]) * pre64[:, 1, :]).astype(
          jnp.float32
      )
      ref_res = pre64.astype(jnp.float32)

    if precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
      atol, rtol = 2e-2, 2e-2
    elif precision in (
        jax.lax.Precision.DEFAULT,
        jax.lax.DotAlgorithmPreset.TF32_TF32_F32,
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

  def test_mixed_dtype_residuals_and_vjp(self):
    assert triton_glu is not None
    glu = triton_glu.TritonGatedLinearUnit()
    xla_glu = base.GatedLinearUnit()
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(107), 3)
    x = jax.random.normal(rng0, (64, 64), dtype=jnp.bfloat16)
    w = jax.random.normal(rng1, (64, 2, 128), dtype=jnp.float32) * 0.125
    dout = jax.random.normal(rng2, (64, 128), dtype=jnp.bfloat16)

    out, res = jax.jit(
        lambda a, b: glu(a, b, activation=jax.nn.silu, return_residuals=True)
    )(x, w)
    self.assertEqual(out.dtype, jnp.bfloat16)
    self.assertEqual(res.dtype, jnp.float32)

    y, f_vjp = jax.vjp(lambda a, b: glu(a, b, activation=jax.nn.silu), x, w)
    dx, dw = jax.jit(f_vjp)(dout)
    y_ref, ref_vjp = jax.vjp(
        lambda a, b: xla_glu(a, b, activation=jax.nn.silu), x, w
    )
    dx_ref, dw_ref = jax.jit(ref_vjp)(dout)
    chex.assert_trees_all_close(y, y_ref, atol=0.1, rtol=0.02)
    chex.assert_trees_all_close(dx, dx_ref, atol=0.2, rtol=0.05)
    chex.assert_trees_all_close(dw, dw_ref, atol=0.5, rtol=0.05)

  @parameterized.parameters(False, True)
  def test_vmap_vjp_bfloat16(self, use_tuple_weights):
    assert triton_glu is not None
    glu = triton_glu.TritonGatedLinearUnit()
    rng0, rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(113), 4)
    x = jax.random.normal(rng0, (2, 3, 21, 64), dtype=jnp.bfloat16)
    scale = jnp.bfloat16(0.125)
    if use_tuple_weights:
      w = (
          jax.random.normal(rng1, (64, 128), dtype=jnp.bfloat16) * scale,
          jax.random.normal(rng2, (64, 128), dtype=jnp.bfloat16) * scale,
      )
    else:
      w = jax.random.normal(rng1, (64, 2, 128), dtype=jnp.bfloat16) * scale
    dout = jax.random.normal(rng3, (2, 3, 21, 128), dtype=jnp.bfloat16)

    def forward(x_in, w_in):
      fn = lambda xi, wi: glu(xi, wi, activation=jax.nn.silu)
      fn = jax.vmap(fn, in_axes=(0, None))
      fn = jax.vmap(fn, in_axes=(0, None))
      return fn(x_in, w_in)

    @jax.jit
    def run_vjp(x_in, w_in, dout_in):
      y, f_vjp = jax.vjp(forward, x_in, w_in)
      return y, *f_vjp(dout_in)

    y, dx, dw = run_vjp(x, w, dout)
    x_ref = x.astype(jnp.float32)
    w_ref = jax.tree.map(lambda t: t.astype(jnp.float32), w)
    y_ref, ref_vjp = jax.vjp(
        lambda a, b: base.GatedLinearUnit()(a, b, activation=jax.nn.silu),
        x_ref,
        w_ref,
    )
    dx_ref, dw_ref = ref_vjp(dout.astype(jnp.float32))
    chex.assert_trees_all_close(y, y_ref, atol=0.1, rtol=0.02)
    chex.assert_trees_all_close(dx, dx_ref, atol=0.5, rtol=0.05)
    chex.assert_trees_all_close(dw, dw_ref, atol=1.0, rtol=0.05)

  @parameterized.parameters(jnp.bfloat16, jnp.float32)
  def test_all_autotuning_configs_execute(self, dtype):
    assert triton_glu is not None
    glu = triton_glu.TritonGatedLinearUnit()
    xla_glu = base.GatedLinearUnit()
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(127), 2)
    x = jax.random.normal(rng0, (67, 64), dtype=dtype)
    w = jax.random.normal(rng1, (64, 2, 128), dtype=dtype) * dtype(0.125)
    expected = xla_glu(x, w, activation=jax.nn.gelu)
    ba = glu.bind(x, w, activation=jax.nn.gelu)
    tol = 5e-3 if dtype == jnp.float32 else 0.1
    for cfg in ba.autotuning_configs:
      actual = jax.jit(
          lambda a, b, c=cfg: glu.replace(config=c)(
              a, b, activation=jax.nn.gelu
          )
      )(x, w)
      chex.assert_trees_all_close(actual, expected, atol=tol, rtol=0.02)


if __name__ == "__main__":
  absltest.main()
