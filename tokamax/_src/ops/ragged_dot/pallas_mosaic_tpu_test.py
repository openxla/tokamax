# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tokamax Megablox TPU tests for core functionality."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from tokamax._src import quantization
from tokamax._src.ops import op as op_base
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu
from tokamax._src.ops.ragged_dot import test_base


QuantizedArray = quantization.QuantizedArray


def _is_scale_tiling_along_reduction(
    x: QuantizedArray, axis: int, min_tiling: int
) -> bool:
  all_equal = all(
      s1 == s2 for i, (s1, s2) in enumerate(zip(x.values.shape, x.scales.shape))
      if i != (axis % x.ndim)
  )
  scale_tiling = x.values.shape[axis] // x.scales.shape[axis]
  return all_equal and scale_tiling >= min_tiling


def _is_config_supported(
    lhs: jax.Array | QuantizedArray,
    rhs: jax.Array | QuantizedArray,
    config: pallas_mosaic_tpu.Config,
) -> bool:
  (m, k), (_, _, n) = lhs.shape, rhs.shape
  if (m < config.gmm_tiling[0]
      or k < config.gmm_tiling[1]
      or n < config.gmm_tiling[2]):
    return False
  if (isinstance(lhs, QuantizedArray)
      and not _is_scale_tiling_along_reduction(
          lhs, 1, config.gmm_tiling[1])):
    return False
  if (isinstance(rhs, QuantizedArray)
      and not _is_scale_tiling_along_reduction(
          rhs, 1, config.gmm_tiling[1])):
    return False
  return True


# TODO : Add QWIX tests for ragged dot once QWIX is in Ragged Dot.
# TODO: Merge QWIX quantization tests into ragged dot API tests.
# also add shapes which tile sizes do not cleanly divide to test masking.
class PallasMosaicTpuRaggedDotTest(test_base.RaggedDotTestBase):
  """Pallas Mosaic TPU Ragged Dot tests."""

  def __init__(self, *args):

    def fn(lhs, rhs, *, config=None, **kwargs):
      config = config or pallas_mosaic_tpu.Config()
      op = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(config=config)

      # skip unsupported tiling and quantization
      if _is_config_supported(lhs, rhs, config):
        return op(lhs, rhs, **kwargs)

      with self.assertRaises(NotImplementedError) as e:
        _ = op(lhs, rhs, **kwargs)
      self.skipTest(f"Test not supported: {e.msg}")

    super().__init__(*args, dot_fn=fn)
    self.tol = dict(atol=1e-2, rtol=0)

    def assert_close(a, b, **tol):

      def l2_rel(a, b):
        l2_diff = jnp.linalg.norm(a - b, axis=-1)
        l2_norm = jnp.maximum(jnp.linalg.norm(b, axis=-1), 1e-6)
        return l2_diff / l2_norm

      l2_rel = jax.tree.map(l2_rel, a, b)
      expected = jax.tree.map(jnp.zeros_like, l2_rel)
      chex.assert_trees_all_close(l2_rel, expected, **dict(self.tol, **tol))

    self.assert_close = assert_close

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  def test_bench_memory_bound(self):
    self.skipTest("GPU Only test.")

  def test_maxtext_config(self):
    # Test to ensure that we can get the correct config for a specific model.
    # For this test we are using jax.ShapeDtypeStruct instead of jax.Array
    # because jax.Array would trigger OOM for our tests.
    tpu_ragged_dot = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot()
    maxtext_config = tpu_ragged_dot._get_heuristics_config(
        op_base.BoundArguments(
            op=tpu_ragged_dot,
            arguments={
                "lhs": jax.ShapeDtypeStruct((262144, 7168), dtype=jnp.bfloat16),
                "rhs": jax.ShapeDtypeStruct(
                    (256, 7168, 2048), dtype=jnp.bfloat16
                ),
            },
        )
    )
    self.assertEqual(maxtext_config.gmm_tiling, (256, 7168, 512))


if __name__ == "__main__":
  absltest.main()