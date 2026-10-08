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
"""Tests for Pallas Mosaic TPU Lightning Indexer operator."""

from absl.testing import absltest
import numpy as np
from tokamax._src.ops.experimental.lightning_indexer import pallas_mosaic_tpu
from tokamax._src.ops.experimental.lightning_indexer import test_base


class PallasTpuLightningIndexerTest(test_base.LightningIndexerTestBase):

  def __init__(self, *args):
    super().__init__(
        *args,
        topk_fn=pallas_mosaic_tpu.PallasTpuLightningIndexer(),
    )

  def setUp(self):
    super().setUp()
    test_base.skip_if_unsupported(self)

  def test_custom_config(self):
    cfg = pallas_mosaic_tpu.Config(
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 8, 8),
        decode_req_batch_size=1,
    )
    op = pallas_mosaic_tpu.PallasTpuLightningIndexer(config=cfg)
    inputs = test_base.make_test_inputs([1, 4], [256, 256], seed=13)
    actual = np.sort(np.asarray(op(**inputs, k=16)), axis=-1)
    expected = np.sort(np.asarray(self._ref_fn(**inputs, k=16)), axis=-1)
    np.testing.assert_array_equal(actual, expected)

  def test_autotuning_configs_match_reference(self):
    op = pallas_mosaic_tpu.PallasTpuLightningIndexer()
    inputs = test_base.make_test_inputs(
        [1, 4], [256, 256], pages_per_seq=4, seed=21
    )
    bound = op.bind(**inputs, k=16, compression_ratio=1)
    configs = bound.autotuning_configs
    self.assertIn(bound.default_config, configs)

    expected = np.sort(
        np.asarray(self._ref_fn(**inputs, k=16, compression_ratio=1)),
        axis=-1,
    )
    for cfg in configs:
      with self.subTest(f"{cfg=}"):
        actual = np.sort(
            np.asarray(
                op.replace(config=cfg)(**inputs, k=16, compression_ratio=1)
            ),
            axis=-1,
        )
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
  absltest.main()
