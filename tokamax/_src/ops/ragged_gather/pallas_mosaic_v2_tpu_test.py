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
"""Tests for Pallas/Mosaic Ragged Gather V2 operator on TPU."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.ragged_gather import pallas_mosaic_v2_tpu
from tokamax._src.ops.ragged_gather import test_base

jax.config.parse_flags_with_absl()


class PallasTpuRaggedGatherV2Test(test_base.RaggedGatherTestBase):

  def __init__(self, *args):
    super().__init__(
        *args, gather_fn=pallas_mosaic_v2_tpu.PallasV2TpuRaggedGather()
    )

  @parameterized.product(
      in_out_size=[(512, 32), (512, 400), (512, 1024)],
      start_end=[(3, 28), (3, 338), (10, 422)],
      hidden_size=[128, 512, 8192],
      dtype=[jnp.int4],
  )
  def test_sc_gather_int4(self, in_out_size, hidden_size, start_end, dtype):
    self.check_sc_gather(in_out_size, hidden_size, start_end, dtype)


if __name__ == "__main__":
  absltest.main()
