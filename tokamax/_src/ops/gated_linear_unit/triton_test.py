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
import jax
import jax.numpy as jnp
from tokamax._src import gpu_utils
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

  def test_autotuning_search_space(self):
    assert triton_glu is not None
    m = 256
    n = 256
    k = 64
    glu = triton_glu.TritonGatedLinearUnit()
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(0), 2)
    x = jax.random.normal(rng0, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(rng1, (k, 2, n), dtype=jnp.bfloat16)

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


if __name__ == "__main__":
  absltest.main()
