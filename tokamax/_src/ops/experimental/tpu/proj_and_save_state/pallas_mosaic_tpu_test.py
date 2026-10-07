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
"""Tests for the Pallas/Mosaic compressor projection operator and kernel."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
from tokamax._src.ops.experimental.tpu.proj_and_save_state import pallas_mosaic_tpu
from tokamax._src.ops.experimental.tpu.proj_and_save_state import reference
from tokamax._src.ops.experimental.tpu.proj_and_save_state import test_base

jax.config.parse_flags_with_absl()

Config = pallas_mosaic_tpu.Config


def _tpu_older_than_v6e() -> bool:
  """Whether the default device is not a TPU v6e or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 6
  )


class PallasTpuProjAndSaveStateTest(test_base.ProjAndSaveStateTestBase):

  def __init__(self, *args):
    super().__init__(
        *args, proj_fn=pallas_mosaic_tpu.PallasTpuProjAndSaveState()
    )

  def setUp(self):
    super().setUp()
    if _tpu_older_than_v6e():
      self.skipTest("Only tested on TPU v6e and newer.")

  @parameterized.parameters(
      # (geometry, num_tokens, hidden_size, tile_k, tile_n)
      ("indexer", 8, 4096, 2048, 8),
      ("indexer", 13, 4096, 2048, 16),
      ("indexer", 200, 7168, 3584, 128),
      ("indexer", 1000, 2048, 2048, 128),
      ("indexer", 8, 4608, 2304, 8),
      ("hca", 200, 7168, 3584, 128),
      ("csa", 200, 7168, 3584, 128),
  )
  def test_heuristics_config(
      self, geometry, num_tokens, hidden_size, tile_k, tile_n
  ):
    geometry = test_base.GEOMETRIES[geometry]
    inputs = test_base.make_inputs(
        geometry, num_tokens, hidden_size=hidden_size
    )
    op = pallas_mosaic_tpu.PallasTpuProjAndSaveState()
    ba = op.bind(*inputs, compress_ratio=geometry.compress_ratio)
    self.assertEqual(ba.heuristics_config, Config(tile_k=tile_k, tile_n=tile_n))

  @parameterized.parameters(
      ("csa", 4096, 8),
      ("csa", 4096, 200),
      ("csa", 4096, 1000),
      ("csa", 7168, 600),
      ("hca", 7168, 600),
      ("indexer", 7168, 600),
  )
  def test_autotuning_configs(self, geometry, hidden_size, num_tokens):
    """Checks every autotuning config runs and matches the reference."""
    geometry = test_base.GEOMETRIES[geometry]
    compress_ratio = geometry.compress_ratio
    inputs = test_base.make_inputs(
        geometry, num_tokens, hidden_size=hidden_size, num_padding=7
    )
    op = pallas_mosaic_tpu.PallasTpuProjAndSaveState()
    configs = op.bind(*inputs, compress_ratio=compress_ratio).autotuning_configs
    self.assertGreater(len(configs), 1)

    expected = reference.proj_and_save_state(
        *inputs, compress_ratio=compress_ratio
    )
    for config in configs:
      with self.subTest(str(config)):
        actual = op.replace(config=config)(
            *inputs, compress_ratio=compress_ratio
        )
        test_base.assert_caches_match(
            actual, expected, inputs[4], geometry.rows_per_token
        )

  def test_autotuning_configs_fit_vmem(self):
    """Checks extra candidates over the VMEM budget are skipped."""
    # A 3584 `tile_k` is over the budget here (see `pallas_mosaic_tpu`); only
    # upstream's heuristic config keeps it.
    inputs = test_base.make_inputs(test_base.CSA, 600, hidden_size=7168)
    op = pallas_mosaic_tpu.PallasTpuProjAndSaveState()
    ba = op.bind(*inputs, compress_ratio=4)
    heuristic = pallas_mosaic_tpu.Config(tile_k=3584, tile_n=128)
    self.assertEqual(ba.heuristics_config, heuristic)
    configs = ba.autotuning_configs - {heuristic}
    self.assertEqual({c.tile_k for c in configs}, {1024, 1792})
    self.assertEqual({c.tile_n for c in configs}, {128, 256, 512})


if __name__ == "__main__":
  absltest.main()
