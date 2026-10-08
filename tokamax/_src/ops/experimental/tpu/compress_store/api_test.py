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
"""Tests for the compress-and-store API."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
from tokamax._src import hlo_utils
from tokamax._src.ops.experimental.tpu.compress_store import api
from tokamax._src.ops.experimental.tpu.compress_store import test_base

jax.config.parse_flags_with_absl()


def _tpu_older_than_7x() -> bool:
  """Whether the default device is not a TPU7x or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 7
  )


class ApiTest(parameterized.TestCase):

  @parameterized.product(
      case_name=[
          "csa_decode_batch_mixed",
          "hca_decode_batch_seq",
          "csa_indexer_decode",
      ],
      impl=["xla", "mosaic_tpu"],
  )
  def test_basic_api(self, case_name, impl):
    if "mosaic" in impl and _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")

    inputs = test_base.make_case_inputs(case_name)
    kwargs = inputs.kwargs
    arrays = {
        k: kwargs.pop(k) for k in ("cos_sin_cache", "state_cache", "rope_cache")
    }

    @jax.jit
    def f(args, arrays):
      return api.compress_store(*args, **arrays, **kwargs, implementation=impl)

    out = f(inputs.args, arrays)

    with self.subTest("value"):
      test_base.assert_matches_reference(inputs, out)

    with self.subTest("correct_implementation_used"):
      # Check the lowered HLO for the kernel that was really used.
      opspecs = hlo_utils.get_opspecs(
          f.lower(inputs.args, arrays), include_xla_kernels=False
      )
      if impl == "xla":
        self.assertEmpty(opspecs)
      else:
        self.assertNotEmpty(opspecs)
        self.assertIsInstance(
            opspecs[0].op, type(api.IMPLEMENTATIONS["mosaic_tpu"])
        )


if __name__ == "__main__":
  absltest.main()
