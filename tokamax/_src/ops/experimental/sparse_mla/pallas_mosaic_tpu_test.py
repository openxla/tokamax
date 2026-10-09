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
"""Tests for the Pallas/Mosaic SparseMLA operator on TPU."""

from absl.testing import absltest
from jax.extend import backend
from tokamax._src.ops.experimental.sparse_mla import pallas_mosaic_tpu
from tokamax._src.ops.experimental.sparse_mla import test_base


class PallasTpuSparseMlaTest(test_base.SparseMlaTestBase):

  def __init__(self, *args):
    super().__init__(
        *args, sparse_mla_fn=pallas_mosaic_tpu.PallasTpuSparseMla()
    )

  def setUp(self):
    super().setUp()
    if backend.get_default_device().device_kind != "TPU7x":
      self.skipTest("Only tested on TPU7x.")

  def test_autotuning_configs(self):
    """Checks every autotuning config runs and matches the reference."""
    inputs, actual_tokens = test_base.make_inputs(
        batch_size=4, topk=512, pad_tokens=True
    )
    op = pallas_mosaic_tpu.PallasTpuSparseMla()
    ba = op.bind(*inputs, sm_scale=1.0)
    configs = ba.autotuning_configs
    self.assertIn(ba.default_config, configs)
    self.assertGreater(len(configs), 1)

    for config in configs:
      with self.subTest(str(config)):
        actual = op.replace(config=config)(*inputs, sm_scale=1.0)
        test_base.assert_matches_reference(actual, inputs, actual_tokens)


if __name__ == "__main__":
  absltest.main()
