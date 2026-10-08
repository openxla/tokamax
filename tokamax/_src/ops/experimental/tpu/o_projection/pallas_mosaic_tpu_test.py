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
"""Tests for the Pallas/Mosaic DeepSeek-V4 `wo_a` projection on TPU."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
from tokamax._src.ops.experimental.tpu.o_projection import pallas_mosaic_tpu
from tokamax._src.ops.experimental.tpu.o_projection import test_base

jax.config.parse_flags_with_absl()


def _tpu_older_than_7x() -> bool:
  """Whether the default device is not a TPU7x or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 7
  )


class PallasTpuOProjectionTest(test_base.OProjectionTestBase):

  def __init__(self, *args):
    super().__init__(
        *args, o_projection_fn=pallas_mosaic_tpu.PallasTpuOProjection()
    )

  def setUp(self):
    super().setUp()
    if _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")

  @parameterized.parameters(
      dict(num_tokens=256, num_groups=8, lora_rank=1024),
      dict(num_tokens=2048, num_groups=2, lora_rank=1024),
      dict(num_tokens=96, num_groups=2, lora_rank=512),
  )
  def test_autotuning_configs(self, num_tokens, num_groups, lora_rank):
    """Checks every autotuning config against the reference."""
    args = test_base.make_inputs(
        num_tokens=num_tokens, num_groups=num_groups, lora_rank=lora_rank
    )
    op = pallas_mosaic_tpu.PallasTpuOProjection()
    ba = op.bind(*args)
    configs = ba.autotuning_configs
    self.assertIn(op._get_heuristics_config(ba), configs)  # pylint: disable=protected-access
    self.assertGreater(len(configs), 1)

    for quantize_activations in (False, True):
      for config in configs:
        with self.subTest(f"{config}, {quantize_activations=}"):
          out = op.replace(config=config)(
              *args, quantize_activations=quantize_activations
          )
          test_base.check_output(
              out, args, quantize_activations=quantize_activations
          )

  @parameterized.parameters(
      (256, 1024, 256, 1024, 128),
      (4096, 512, 1024, 512, 128),
      (96, 256, 96, 256, 96),
  )
  def test_heuristics_config(
      self, num_tokens, lora_rank, tile_t, tile_r, sub_t
  ):
    """Checks the heuristic reproduces upstream's tile sizes."""
    args = test_base.make_inputs(
        num_tokens=num_tokens, num_groups=1, lora_rank=lora_rank, head_dim=128
    )
    op = pallas_mosaic_tpu.PallasTpuOProjection()
    ba = op.bind(*args)
    self.assertEqual(
        op._get_heuristics_config(ba),  # pylint: disable=protected-access
        pallas_mosaic_tpu.Config(tile_t=tile_t, tile_r=tile_r, sub_t=sub_t),
    )

  def test_head_dim_128_not_implemented(self):
    args = test_base.make_inputs(
        num_tokens=64,
        num_groups=2,
        lora_rank=256,
        head_dim=128,
        rotary_dim=128,
    )
    with self.assertRaisesRegex(NotImplementedError, "head_dim"):
      pallas_mosaic_tpu.PallasTpuOProjection()(*args)

  @parameterized.parameters(
      # `gather_cos_sin`'s tile_n would be 100.
      dict(num_tokens=200, config=None, match="tile_n"),
      # `gather_cos_sin`'s tile_n would be 103.
      dict(num_tokens=1030, config=None, match="tile_n"),
      # Upstream's default tile_t would be 524.
      dict(num_tokens=1048, config=None, match="tile_t"),
      dict(
          num_tokens=96,
          config=pallas_mosaic_tpu.Config(tile_t=12, tile_r=256, sub_t=12),
          match="tile_t",
      ),
  )
  def test_unaligned_token_tiles_not_implemented(
      self, num_tokens, config, match
  ):
    args = test_base.make_inputs(
        num_tokens=num_tokens, num_groups=1, lora_rank=256
    )
    op = pallas_mosaic_tpu.PallasTpuOProjection()
    if config is not None:
      op = op.replace(config=config)
    with self.assertRaisesRegex(NotImplementedError, match):
      op(*args)

  def test_unusual_aligned_num_tokens(self):
    """1040 tokens: `gather_cos_sin`'s tile_n is 104 and `tile_t` 520."""
    self._check(
        quantize_activations=False, num_tokens=1040, num_groups=1, lora_rank=256
    )


if __name__ == "__main__":
  absltest.main()
