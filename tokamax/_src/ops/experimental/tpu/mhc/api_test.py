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
"""Tests for the mHC API."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
from tokamax._src import hlo_utils
from tokamax._src.ops.experimental.tpu.mhc import api
from tokamax._src.ops.experimental.tpu.mhc import reference
from tokamax._src.ops.experimental.tpu.mhc import test_base

jax.config.parse_flags_with_absl()


def _tpu_older_than_v5e() -> bool:
  """Whether the default device is not a TPU v5e or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 5
  )


_IMPLS = ("xla", "mosaic_tpu")


class ApiTest(parameterized.TestCase):

  def _skip_if_unsupported(self, impl):
    if impl == "mosaic_tpu" and _tpu_older_than_v5e():
      self.skipTest("The Pallas TPU kernels need a TPU.")

  def _check_implementation(self, lowered, impl, implementations):
    """Checks the lowered HLO for the kernel that was really used."""
    opspecs = hlo_utils.get_opspecs(lowered, include_xla_kernels=False)
    if impl == "xla":
      self.assertEmpty(opspecs)
    else:
      self.assertNotEmpty(opspecs)
      self.assertIsInstance(opspecs[0].op, type(implementations["mosaic_tpu"]))

  @parameterized.parameters(*_IMPLS)
  def test_mhc_pre(self, impl):
    self._skip_if_unsupported(impl)
    args = test_base.make_pre_inputs(0, 33, 512)
    consts = test_base.gate_constants()

    @jax.jit
    def f(residual, fn, hc_scale, hc_base):
      return api.mhc_pre(
          residual, fn, hc_scale, hc_base, *consts, implementation=impl
      )

    got = f(*args)
    test_base.assert_pre_close(self, got, reference.mhc_pre(*args, *consts))
    self._check_implementation(f.lower(*args), impl, api.PRE_IMPLEMENTATIONS)

  @parameterized.parameters(*_IMPLS)
  def test_mhc_post(self, impl):
    self._skip_if_unsupported(impl)
    args = test_base.make_post_inputs(1, 33, 512)

    @jax.jit
    def f(x, residual, post_layer_mix, comb_res_mix):
      return api.mhc_post(
          x, residual, post_layer_mix, comb_res_mix, implementation=impl
      )

    got = f(*args)
    test_base.assert_allclose(
        got,
        reference.mhc_post(*args),
        rtol=test_base.BF16_RTOL,
        atol=test_base.BF16_ATOL,
    )
    self._check_implementation(f.lower(*args), impl, api.POST_IMPLEMENTATIONS)

  @parameterized.parameters(*_IMPLS)
  def test_mhc_fused_post_pre(self, impl):
    self._skip_if_unsupported(impl)
    args = test_base.make_fused_inputs(2, 33, 512)
    consts = test_base.gate_constants()

    @jax.jit
    def f(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base):
      return api.mhc_fused_post_pre(
          x,
          residual,
          post_layer_mix,
          comb_res_mix,
          fn,
          hc_scale,
          hc_base,
          *consts,
          implementation=impl,
      )

    got = f(*args)
    want = reference.mhc_fused_post_pre(*args, *consts)
    test_base.assert_allclose(
        got[0], want[0], rtol=test_base.BF16_RTOL, atol=test_base.BF16_ATOL
    )
    test_base.assert_pre_close(
        self, got[1:], want[1:], f32_rtol=test_base.FUSED_SMALL_H_RTOL
    )
    self._check_implementation(
        f.lower(*args), impl, api.FUSED_POST_PRE_IMPLEMENTATIONS
    )

  def test_default_implementation(self):
    args = test_base.make_post_inputs(3, 8, 256)
    got = api.mhc_post(*args)
    test_base.assert_allclose(
        got,
        reference.mhc_post(*args),
        rtol=test_base.BF16_RTOL,
        atol=test_base.BF16_ATOL,
    )

  def test_unknown_implementation(self):
    args = test_base.make_post_inputs(3, 8, 256)
    bad_implementation: Any = "triton"
    with self.assertRaisesRegex(ValueError, "Unknown implementation"):
      api.mhc_post(*args, implementation=bad_implementation)


if __name__ == "__main__":
  absltest.main()
