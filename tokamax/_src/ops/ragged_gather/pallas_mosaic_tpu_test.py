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
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from tokamax._src.ops.ragged_gather import pallas_mosaic_tpu
from tokamax._src.ops.ragged_gather import pallas_mosaic_tpu_kernel
from tokamax._src.ops.ragged_gather import test_base

jax.config.parse_flags_with_absl()


class PallasTpuRaggedGatherTest(test_base.RaggedGatherTestBase):

  def __init__(self, *args):
    super().__init__(*args, gather_fn=pallas_mosaic_tpu.PallasTpuRaggedGather())

  @parameterized.parameters("TPU7x", "TPU8i")
  def test_sparse_core_vmem_capacity_bytes(self, device_kind):
    device = jax.sharding.AbstractDevice(
        device_kind=device_kind, num_cores=1, platform="tpu"
    )
    with jax.sharding.use_abstract_mesh(
        jax.sharding.AbstractMesh((), (), abstract_device=device)
    ):
      sc_info = pltpu.get_tpu_info().sparse_core
      self.assertIsNotNone(sc_info)
      self.assertEqual(sc_info.vmem_capacity_bytes, 512 * 1024)

  @parameterized.product(
      device_kind=("TPU7x", "TPU8i"),
      case=(
          (128, 1, 128),
          (512, 2, 512),
          (7168, 2, 1792),
          (8192, 1, 1024),
          (8192, 2, 2048),
          (8192, 4, 2048),
          (8192, 8, 2048),
      ),
  )
  def test_calculate_col_size(self, device_kind, case):
    hidden_size, packing, expected = case
    device = jax.sharding.AbstractDevice(
        device_kind=device_kind, num_cores=1, platform="tpu"
    )
    with jax.sharding.use_abstract_mesh(
        jax.sharding.AbstractMesh((), (), abstract_device=device)
    ):
      actual = pallas_mosaic_tpu_kernel.calculate_col_size(hidden_size, packing)
    self.assertEqual(actual, expected)

  def check_sc_gather(
      self, in_out_size, hidden_size, start_end, dtype, **kwargs
  ):
    for max_row_subchunks in [1, 4]:
      for trim_rows in [True, False]:
        with self.subTest(
            max_row_subchunks=max_row_subchunks, trim_rows=trim_rows
        ):
          super().check_sc_gather(
              in_out_size,
              hidden_size,
              start_end,
              dtype,
              max_row_subchunks=max_row_subchunks,
              trim_rows=trim_rows,
              **kwargs,
          )


if __name__ == "__main__":
  absltest.main()
