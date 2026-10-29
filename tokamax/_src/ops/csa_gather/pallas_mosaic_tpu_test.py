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
"""Tests for the Pallas/Mosaic CSA Gather operator on TPU."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import numpy as np
from tokamax._src.ops.csa_gather import pallas_mosaic_tpu
from tokamax._src.ops.csa_gather import reference
from tokamax._src.ops.csa_gather import test_base

jax.config.parse_flags_with_absl()


def _tpu_older_than_7x() -> bool:
  """Whether the default device is not a TPU7x or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 7
  )


class PallasTpuCsaGatherTest(test_base.CsaGatherTestBase):

  def __init__(self, *args):
    super().__init__(*args, gather_fn=pallas_mosaic_tpu.PallasTpuCsaGather())

  def setUp(self):
    super().setUp()
    if _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")

  @parameterized.parameters(128, 256, 1024, 2048)
  def test_autotuning_configs(self, top_k):
    """Checks every autotuning config is bit-exact against the reference."""
    num_indices = 8 * top_k
    nope_cache, rope_cache, indices = test_base.make_inputs(num_indices)
    op = pallas_mosaic_tpu.PallasTpuCsaGather()
    ba = op.bind(nope_cache, rope_cache, indices, top_k=top_k)
    configs = ba.autotuning_configs
    self.assertIn(pallas_mosaic_tpu.Config(), configs)
    self.assertGreater(len(configs), 1)

    nope_ref, rope_ref = reference.csa_gather(
        nope_cache, rope_cache, indices, top_k=top_k
    )
    for config in configs:
      with self.subTest(str(config)):
        nope_out, rope_out = op.replace(config=config)(
            nope_cache, rope_cache, indices, top_k=top_k
        )
        np.testing.assert_array_equal(nope_out, nope_ref)
        np.testing.assert_array_equal(rope_out, rope_ref)


if __name__ == "__main__":
  absltest.main()
