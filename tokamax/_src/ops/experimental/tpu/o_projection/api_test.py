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
"""Tests for the DeepSeek-V4 `wo_a` projection API."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
from tokamax._src import hlo_utils
from tokamax._src.ops.experimental.tpu.o_projection import api
from tokamax._src.ops.experimental.tpu.o_projection import test_base

jax.config.parse_flags_with_absl()


def _tpu_older_than_7x() -> bool:
  """Whether the default device is not a TPU7x or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 7
  )


class ApiTest(parameterized.TestCase):

  @parameterized.product(
      inverse=[False, True],
      quantize_activations=[False, True],
      impl=["xla", "mosaic_tpu"],
  )
  def test_basic_api(self, inverse, quantize_activations, impl):
    if "mosaic" in impl and _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")

    args = test_base.make_inputs(num_tokens=128, num_groups=2, lora_rank=512)

    @jax.jit
    def f(*args):
      return api.o_projection(
          *args,
          inverse=inverse,
          quantize_activations=quantize_activations,
          implementation=impl,
      )

    out = f(*args)

    with self.subTest("value"):
      test_base.check_output(
          out,
          args,
          inverse=inverse,
          quantize_activations=quantize_activations,
      )

    with self.subTest("correct_implementation_used"):
      # Check the lowered HLO for the kernel that was really used.
      opspecs = hlo_utils.get_opspecs(f.lower(*args), include_xla_kernels=False)
      if impl == "xla":
        self.assertEmpty(opspecs)
      else:
        self.assertNotEmpty(opspecs)
        self.assertIsInstance(
            opspecs[0].op, type(api.IMPLEMENTATIONS["mosaic_tpu"])
        )

  def test_falls_back_to_xla_for_unaligned_num_tokens(self):
    if _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")
    # The Pallas `gather_cos_sin` tile_n would be 100.
    args = test_base.make_inputs(num_tokens=200, num_groups=1, lora_rank=256)
    out = api.o_projection(*args, quantize_activations=False)
    test_base.check_output(out, args, quantize_activations=False)


if __name__ == "__main__":
  absltest.main()
