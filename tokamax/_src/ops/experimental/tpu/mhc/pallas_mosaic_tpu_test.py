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
"""Tests for the Pallas/Mosaic mHC operators on TPU."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import pydantic
from tokamax._src.ops.experimental.tpu.mhc import pallas_mosaic_tpu
from tokamax._src.ops.experimental.tpu.mhc import reference
from tokamax._src.ops.experimental.tpu.mhc import test_base

jax.config.parse_flags_with_absl()


def _tpu_older_than_v5e() -> bool:
  """Whether the default device is not a TPU v5e or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 5
  )


_DSV4_HIDDEN_SIZE = 7168


class PallasTpuMhcTest(test_base.MhcTestBase):

  def __init__(self, *args):
    super().__init__(
        *args,
        pre_fn=pallas_mosaic_tpu.PallasTpuMhcPre(),
        post_fn=pallas_mosaic_tpu.PallasTpuMhcPost(),
        fused_fn=pallas_mosaic_tpu.PallasTpuMhcFusedPostPre(),
    )

  def setUp(self):
    super().setUp()
    if _tpu_older_than_v5e():
      self.skipTest("Only tested on TPU v5e and newer.")

  @parameterized.parameters(16, 2048)
  def test_pre_autotuning_configs(self, num_tokens):
    """Checks every autotuning config at the DeepSeek-V4 shape."""
    args = test_base.make_pre_inputs(6, num_tokens, _DSV4_HIDDEN_SIZE)
    consts = test_base.gate_constants()
    op = pallas_mosaic_tpu.PallasTpuMhcPre()
    configs = op.bind(*args, *consts).autotuning_configs
    self.assertNotEmpty(configs)
    want = reference.mhc_pre(*args, *consts)
    for config in configs:
      with self.subTest(str(config)):
        got = op.replace(config=config)(*args, *consts)
        test_base.assert_pre_close(self, got, want)

  @parameterized.parameters(16, 2048)
  def test_post_autotuning_configs(self, num_tokens):
    """Checks every autotuning config at the DeepSeek-V4 shape."""
    args = test_base.make_post_inputs(7, num_tokens, _DSV4_HIDDEN_SIZE)
    op = pallas_mosaic_tpu.PallasTpuMhcPost()
    configs = op.bind(*args).autotuning_configs
    self.assertNotEmpty(configs)
    want = reference.mhc_post(*args)
    for config in configs:
      with self.subTest(str(config)):
        got = op.replace(config=config)(*args)
        test_base.assert_allclose(
            got, want, rtol=test_base.BF16_RTOL, atol=test_base.BF16_ATOL
        )

  @parameterized.parameters(16, 2048)
  def test_fused_autotuning_configs(self, num_tokens):
    """Checks every autotuning config at the DeepSeek-V4 shape."""
    args = test_base.make_fused_inputs(8, num_tokens, _DSV4_HIDDEN_SIZE)
    consts = test_base.gate_constants()
    op = pallas_mosaic_tpu.PallasTpuMhcFusedPostPre()
    configs = op.bind(*args, *consts).autotuning_configs
    self.assertNotEmpty(configs)
    want = reference.mhc_fused_post_pre(*args, *consts)
    for config in configs:
      with self.subTest(str(config)):
        got = op.replace(config=config)(*args, *consts)
        test_base.assert_allclose(
            got[0], want[0], rtol=test_base.BF16_RTOL, atol=test_base.BF16_ATOL
        )
        test_base.assert_pre_close(
            self, got[1:], want[1:], f32_rtol=test_base.FUSED_F32_RTOL
        )

  def test_config_rejects_unaligned_token_block_size(self):
    with self.assertRaisesRegex(pydantic.ValidationError, "multiple of 16"):
      pallas_mosaic_tpu.Config(token_block_size=24)


if __name__ == "__main__":
  absltest.main()
