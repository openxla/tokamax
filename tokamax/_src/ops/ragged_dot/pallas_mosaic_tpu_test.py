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
import jax
import jax.numpy as jnp
from tokamax._src import mosaic_tpu as common
from tokamax._src import quantization
from tokamax._src.ops import op as op_lib
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu
from tokamax._src.ops.ragged_dot import test_base
from typing_extensions import override


QuantizedArray = quantization.QuantizedArray


def _is_scale_tiling_supported(x: QuantizedArray, axis: int) -> bool:
  min_addressable_sizes = (
      [1] * x.ndim + [common._sublane_size(), common.LANES]
  )[-x.ndim:]
  cdiv = lambda x, y: (x + y - 1) // y
  eps_list = [cdiv(x, y) for x, y in zip(x.values.shape, x.scales.shape)]
  for ax, (mas, eps) in enumerate(zip(min_addressable_sizes, eps_list)):
    if eps != 1 and eps % mas != 0:
      return False
    if ax != axis and not (eps == 1 or eps == x.values.shape[ax]):
      return False
  return True


def _is_config_supported(
    lhs: jax.Array | QuantizedArray,
    rhs: jax.Array | QuantizedArray,
    config: pallas_mosaic_tpu.Config,
) -> bool:
  (m, k), (_, _, n) = lhs.shape, rhs.shape
  if (
      m < config.gmm_tiling[0]
      or k < config.gmm_tiling[1]
      or n < config.gmm_tiling[2]
  ):
    return False
  if isinstance(lhs, QuantizedArray) and not _is_scale_tiling_supported(lhs, 1):
    return False
  if isinstance(rhs, QuantizedArray) and not _is_scale_tiling_supported(rhs, 1):
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

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  def test_vjp0(self):
    with test_base.override_chex_args(atol=0.2, rtol=0.01):
      super().test_vjp0()  # pytype: disable=attribute-error

  @override
  def _test_quantized(self, dtype, a_tile_shape, b_tile_shape):
    with test_base.override_chex_args(atol=0.4, rtol=0.1):
      super()._test_quantized(dtype, a_tile_shape, b_tile_shape)

  @override
  def _test_bench(self, spec):
    if "i8xi8" in self._testMethodName:
      kwargs = dict(atol=1.5, rtol=0.5)  # This is really bad!
    elif "i4" in self._testMethodName:
      kwargs = dict(atol=0.6, rtol=0.1)
    else:
      kwargs = {}
    with test_base.override_chex_args(**kwargs):
      super()._test_bench(spec)

  def test_maxtext_config(self):
    # Test to ensure that we can get the correct config for a specific model.
    # For this test we are using jax.ShapeDtypeStruct instead of jax.Array
    # because jax.Array would trigger OOM for our tests.
    tpu_ragged_dot = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot()
    maxtext_config = tpu_ragged_dot._get_heuristics_config(
        op_lib.BoundArguments(
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

  def test_autotuning_configs(self):
    tpu_ragged_dot = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot()
    ba = op_lib.BoundArguments(
        op=tpu_ragged_dot,
        arguments={
            "lhs": jax.ShapeDtypeStruct((262144, 7168), dtype=jnp.bfloat16),
            "rhs": jax.ShapeDtypeStruct((256, 7168, 2048), dtype=jnp.bfloat16),
        },
    )
    autotuning_configs = ba.autotuning_configs
    self.assertGreaterEqual(len(autotuning_configs), 3 * 3 * 3)


if __name__ == "__main__":
  absltest.main()
