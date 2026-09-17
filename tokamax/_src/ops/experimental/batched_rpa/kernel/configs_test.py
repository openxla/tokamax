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
"""Tests for batched RPA config and block size calculations across TPU generations."""

import dataclasses
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.batched_rpa.kernel import configs
from tokamax._src.ops.experimental.batched_rpa.kernel import wrapper


@dataclasses.dataclass
class _MockTpuInfo:
  generation: int
  chip_version: str = "v8i"
  num_lanes: int = 128
  num_sublanes: int = 16
  mxu_column_size: int = 128
  vmem_capacity_bytes: int = 64 * 1024 * 1024
  smem_capacity_bytes: int = 16 * 1024 * 1024
  fp8_ops_per_second: int = 2
  bf16_ops_per_second: int = 1


def _create_mock_tpu_info(
    generation: int,
    chip_version: str = "v8i",
    num_lanes: int = 128,
    num_sublanes: int = 16,
    mxu_column_size: int = 128,
    vmem_capacity_bytes: int = 64 * 1024 * 1024,
    smem_capacity_bytes: int = 16 * 1024 * 1024,
    fp8_ops_per_second: int = 2,
    bf16_ops_per_second: int = 1,
) -> _MockTpuInfo:
  return _MockTpuInfo(
      generation=generation,
      chip_version=chip_version,
      num_lanes=num_lanes,
      num_sublanes=num_sublanes,
      mxu_column_size=mxu_column_size,
      vmem_capacity_bytes=vmem_capacity_bytes,
      smem_capacity_bytes=smem_capacity_bytes,
      fp8_ops_per_second=fp8_ops_per_second,
      bf16_ops_per_second=bf16_ops_per_second,
  )


class ConfigsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("v8i_fp8", 8, "v8i", True, 1),
      ("v8i_bf16", 8, "v8i", False, 1),
      ("v8t_fp8", 8, "v8t", True, 1),
      ("v8t_bf16", 8, "v8t", False, 1),
      ("v7x_fp8", 7, "v7x", True, 1),
      ("v7x_bf16", 7, "v7x", False, 1),
      ("v5p_bf16", 5, "v5p", False, 1),
  )
  def test_calculate_block_sizes_generation(
      self, generation, chip_version, is_8bit, expected_bq_c_min
  ):
    dtype = jnp.float8_e4m3fn if is_8bit else jnp.bfloat16
    model_cfg = configs.ModelConfigs(
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=8,
        page_size=16,
        total_q_tokens=8,
        num_page_indices=128,
        dtype_q=dtype,
        dtype_kv=dtype,
        dtype_out=jnp.bfloat16,
    )
    vmem_limit = 64 * 1024 * 1024
    with mock.patch.object(
        pltpu,
        "get_tpu_info",
        autospec=True,
        return_value=_create_mock_tpu_info(
            generation, chip_version=chip_version
        ),
    ):
      decode_blocks, prefill_blocks = wrapper.calculate_block_sizes(
          model_cfg, serve_cfg, vmem_limit
      )
      self.assertGreaterEqual(decode_blocks.bq_sz, decode_blocks.bq_c_sz)
      self.assertGreaterEqual(decode_blocks.bq_c_sz, expected_bq_c_min)
      self.assertGreaterEqual(prefill_blocks.bq_sz, prefill_blocks.bq_c_sz)
      self.assertGreaterEqual(prefill_blocks.bq_c_sz, expected_bq_c_min)


if __name__ == "__main__":
  absltest.main()
