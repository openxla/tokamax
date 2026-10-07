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
"""Hypothesis tests for the mHC Pallas kernels on TPU.

These sweep ragged token counts, token block sizes and stream counts through
the flat kernel entry points the Pallas ops call.
"""

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
import hypothesis.strategies as hps
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
from tokamax._src.ops.experimental.tpu.mhc import pallas_mosaic_tpu
from tokamax._src.ops.experimental.tpu.mhc import pre_kernel
from tokamax._src.ops.experimental.tpu.mhc import reference
from tokamax._src.ops.experimental.tpu.mhc import test_base

jax.config.parse_flags_with_absl()


def _tpu_older_than_v5e() -> bool:
  """Whether the default device is not a TPU v5e or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 5
  )


hp.settings.register_profile(
    name="deterministic",
    database=None,
    derandomize=True,
    deadline=None,
    max_examples=10,
    print_blob=True,
    verbosity=hp.Verbosity.verbose,
)
hp.settings.load_profile(name="deterministic")

_DSV4_HIDDEN_SIZE = 7168


def _draw_shape(data) -> tuple[int, int, int, int]:
  """Draws `(num_tokens, hidden_size, hc_mult, token_block_size)`."""
  num_tokens = data.draw(hps.integers(1, 300), label="num_tokens")
  hidden_size = data.draw(hps.sampled_from([128, 256, 512]), label="hidden")
  hc_mult = data.draw(hps.sampled_from([2, 4]), label="hc_mult")
  token_block_size = data.draw(hps.sampled_from([16, 32, 64]), label="tb")
  return num_tokens, hidden_size, hc_mult, token_block_size


def _config(token_block_size: int) -> pallas_mosaic_tpu.Config:
  return pallas_mosaic_tpu.Config(token_block_size=token_block_size)


class PallasMosaicTpuKernelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if _tpu_older_than_v5e():
      self.skipTest("Only tested on TPU v5e and newer.")

  @hp.given(hps.data())
  def test_pre_mixes_kernel(self, data):
    """Checks the raw mix-GEMM kernel against the reference."""
    num_tokens, hidden_size, hc_mult, tb = _draw_shape(data)
    residual, fn, _, _ = test_base.make_pre_inputs(
        0, num_tokens, hidden_size, hc_mult
    )
    got = pre_kernel.mhc_pre_mixes(residual, fn, token_block_size=tb)
    want = reference.mhc_pre_mixes(residual, fn)
    for g, w in zip(got, want):
      test_base.assert_allclose(
          g, w, rtol=test_base.F32_RTOL, atol=test_base.F32_ATOL
      )

  @hp.given(hps.data())
  def test_pre_kernel(self, data):
    """Checks the mixes + collapse kernel and gates against the reference."""
    num_tokens, hidden_size, hc_mult, tb = _draw_shape(data)
    sinkhorn_repeat = data.draw(hps.sampled_from([1, 20]), label="repeat")
    args = test_base.make_pre_inputs(1, num_tokens, hidden_size, hc_mult)
    consts = test_base.gate_constants(sinkhorn_repeat)
    op = pallas_mosaic_tpu.PallasTpuMhcPre(config=_config(tb))
    got = op(*args, *consts)
    want = reference.mhc_pre(*args, *consts)
    test_base.assert_pre_close(self, got, want)

  @hp.given(hps.data())
  def test_post_kernel(self, data):
    """Checks the post kernel against the reference."""
    num_tokens, hidden_size, hc_mult, tb = _draw_shape(data)
    args = test_base.make_post_inputs(2, num_tokens, hidden_size, hc_mult)
    got = pallas_mosaic_tpu.PallasTpuMhcPost(config=_config(tb))(*args)
    want = reference.mhc_post(*args)
    self.assertEqual(got.dtype, jnp.bfloat16)
    test_base.assert_allclose(
        got, want, rtol=test_base.BF16_RTOL, atol=test_base.BF16_ATOL
    )

  @hp.given(hps.data())
  def test_fused_post_pre_kernel(self, data):
    """Checks the fused kernel and gates against the sequential reference."""
    num_tokens, hidden_size, hc_mult, tb = _draw_shape(data)
    args = test_base.make_fused_inputs(3, num_tokens, hidden_size, hc_mult)
    consts = test_base.gate_constants()
    op = pallas_mosaic_tpu.PallasTpuMhcFusedPostPre(config=_config(tb))
    got = op(*args, *consts)
    want = reference.mhc_fused_post_pre(*args, *consts)
    test_base.assert_allclose(
        got[0], want[0], rtol=test_base.BF16_RTOL, atol=test_base.BF16_ATOL
    )
    test_base.assert_pre_close(
        self, got[1:], want[1:], f32_rtol=test_base.FUSED_SMALL_H_RTOL
    )

  def test_fused_matches_unfused_kernels(self):
    """Checks the fused kernel against the separate post and pre kernels."""
    args = test_base.make_fused_inputs(4, 64, _DSV4_HIDDEN_SIZE)
    consts = test_base.gate_constants()
    fused = pallas_mosaic_tpu.PallasTpuMhcFusedPostPre()(*args, *consts)
    residual_cur = pallas_mosaic_tpu.PallasTpuMhcPost()(*args[:4])
    unfused = (
        residual_cur,
        *pallas_mosaic_tpu.PallasTpuMhcPre()(residual_cur, *args[4:], *consts),
    )
    test_base.assert_allclose(fused[0], unfused[0], rtol=0, atol=0)
    test_base.assert_pre_close(self, fused[1:], unfused[1:])


if __name__ == "__main__":
  absltest.main()
