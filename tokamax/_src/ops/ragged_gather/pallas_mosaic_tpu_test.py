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
"""Tests for Pallas/Mosaic Ragged Gather operator on TPU."""

from absl.testing import absltest
import jax
from tokamax._src.ops.ragged_gather import pallas_mosaic_tpu
from tokamax._src.ops.ragged_gather import test_base

jax.config.parse_flags_with_absl()


class PallasTpuRaggedGatherTest(test_base.RaggedGatherTestBase):

  def __init__(self, *args):
    super().__init__(*args, gather_fn=pallas_mosaic_tpu.PallasTpuRaggedGather())


if __name__ == "__main__":
  absltest.main()
