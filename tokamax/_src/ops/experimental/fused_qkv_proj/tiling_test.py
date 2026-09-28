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
"""Tests for the tile choice (tiling.TileConfig); runs on CPU via TpuInfo presets."""

import contextlib
from typing import NamedTuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa.kernel import configs as rpa_configs
from tokamax._src.ops.experimental.fused_qkv_proj import models
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import tiling
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gqa import config as gqa_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.out_proj import config as out_proj_config

CHIPS = (pltpu.ChipVersion.TPU_7X, pltpu.ChipVersion.TPU_8T)
KV_LAYOUTS = (
    rpa_configs.KVLayout.SEQ_ALONG_LANE,
    rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
)


def _on_chip(
    chip: pltpu.ChipVersion,
) -> tuple[pltpu.TpuInfo, contextlib.AbstractContextManager[object]]:
  tpu_info = pltpu.get_tpu_info_for_chip(chip, 1)
  return tpu_info, mock.patch(
      "jax.experimental.pallas.tpu.get_tpu_info", return_value=tpu_info
  )


def gqa_cfg(
    model: models.Model,
    num_tokens: int,
    chip: pltpu.ChipVersion = pltpu.ChipVersion.TPU_7X,
    kv_layout: rpa_configs.KVLayout = rpa_configs.KVLayout.SEQ_ALONG_LANE,
    tile_size: int | None = None,
) -> tuple[pltpu.TpuInfo, gqa_config.FusedQkvConfig]:
  """fused_qkv's config on chip, with abstract prepared weights."""
  tpu_info, on_chip = _on_chip(chip)
  with on_chip:
    s = model.config
    weights = jax.eval_shape(
        lambda: models.prepare_gqa_weights(s, models.make_inputs(model, 128)[2])
    )
    x = jax.ShapeDtypeStruct((num_tokens, weights.w.shape[0]), jnp.bfloat16)
    residual = x if s.residual else None
    cfg = gqa_config.FusedQkvConfig.create(
        s,
        x,
        residual,
        weights,
        kv_layout=kv_layout,
        page_size=128,
        tile_size=tile_size,
    )
  return tpu_info, cfg


class TileConfigTest(parameterized.TestCase):

  def assert_fits(
      self,
      tpu_info: pltpu.TpuInfo,
      row_bytes: NamedTuple,
      t: tiling.TileConfig,
  ) -> None:
    """Asserts t's tiles fit in VMEM and divide evenly."""
    need = tiling.total_bytes(row_bytes) * t.tile_size
    self.assertLessEqual(
        need + t.policy.vmem_estimate_margin_bytes, t.vmem_limit_bytes
    )
    self.assertEqual(t.num_tokens % t.tile_size, 0)
    self.assertEqual(t.tile_size % t.matmul_rows, 0)
    self.assertEqual(t.matmul_rows % t.prep_rows, 0)
    del tpu_info

  @parameterized.product(
      name=[m.name for m in models.MODELS],
      num_tokens=(128, 1024, 16384),
      kv_layout=KV_LAYOUTS,
      chip=CHIPS,
  )
  def test_gqa_valid(self, name, num_tokens, kv_layout, chip):
    model = models.BY_NAME[name]
    tpu_info, cfg = gqa_cfg(model, num_tokens, chip, kv_layout)
    t = cfg.tile
    self.assertEqual(t.tile_size % tpu_info.num_lanes, 0)
    self.assertEqual(t.norm_cols % tpu_info.num_lanes, 0)
    with _on_chip(chip)[1]:
      bytes_per_row = gqa_config.bytes_per_row(
          model.config, kv_layout, cfg.hidden_size
      )
    self.assert_fits(tpu_info, bytes_per_row, t)

  @parameterized.parameters(
      (pltpu.ChipVersion.TPU_7X, 32, 512),
      (pltpu.ChipVersion.TPU_8T, 64, 512),
  )
  def test_hardware_tiles(self, chip, prep_rows, norm_cols):
    _, cfg = gqa_cfg(models.QWEN3_5_TP8, 1024, chip)
    self.assertEqual(cfg.tile.prep_rows, prep_rows)
    self.assertEqual(cfg.tile.norm_cols, norm_cols)

  # The "picked" column of the sweep table in kernel/common/tiling.py.
  @parameterized.parameters(
      ("qwen3_coder_tp8", 1024, 256),
      ("qwen3_coder_tp8", 16384, 256),
      ("qwen3_5_tp8", 1024, 256),
      ("qwen3_5_tp8", 16384, 512),
      ("qwen3_5_tp2", 1024, 128),
      ("qwen3_5_tp2", 16384, 128),
      ("gemma4_sliding_tp4", 1024, 256),
      ("gemma4_sliding_tp4", 16384, 512),
      ("gemma4_full_tp4", 1024, 256),
      ("gemma4_full_tp4", 16384, 512),
      ("gpt_oss_tp4", 1024, 256),
      ("gpt_oss_tp4", 16384, 512),
  )
  def test_matches_sweep(self, name, num_tokens, picked):
    _, cfg = gqa_cfg(models.BY_NAME[name], num_tokens)
    self.assertEqual(cfg.tile.tile_size, picked)

  def test_tile_size_override(self):
    _, cfg = gqa_cfg(models.GPT_OSS_TP4, 16384, tile_size=128)
    self.assertEqual(cfg.tile.tile_size, 128)
    self.assertEqual(cfg.tile.matmul_rows, 128)

  @parameterized.product(
      name=[m.name for m in models.MODELS],
      num_tokens=(64, 128, 1024, 16384),
      chip=CHIPS,
  )
  def test_out_proj_valid(self, name, num_tokens, chip):
    model = models.BY_NAME[name]
    tpu_info, on_chip = _on_chip(chip)
    with on_chip:
      s = model.config
      o, weights, gate = jax.eval_shape(
          lambda: models.make_out_proj_inputs(
              model, num_tokens, t_rpa=max(num_tokens, 128)
          )
      )
      cfg = out_proj_config.FusedOutProjConfig.create(
          s, o, gate, weights, num_tokens=num_tokens
      )
      bytes_per_row = out_proj_config.bytes_per_row(s, cfg.hidden_size)
    self.assert_fits(tpu_info, bytes_per_row, cfg.tile)


if __name__ == "__main__":
  absltest.main()
