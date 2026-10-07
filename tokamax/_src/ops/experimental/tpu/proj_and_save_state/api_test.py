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
"""Tests for the compressor projection (proj_and_save_state) API."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
from tokamax._src import hlo_utils
from tokamax._src.ops.experimental.tpu.proj_and_save_state import api
from tokamax._src.ops.experimental.tpu.proj_and_save_state import reference
from tokamax._src.ops.experimental.tpu.proj_and_save_state import test_base

jax.config.parse_flags_with_absl()


def _tpu_older_than_v6e() -> bool:
  """Whether the default device is not a TPU v6e or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 6
  )


class ApiTest(parameterized.TestCase):

  @parameterized.product(
      geometry=("csa", "hca", "indexer"),
      num_tokens=(8, 200),
      impl=("xla", "mosaic_tpu"),
  )
  def test_basic_api(self, geometry, num_tokens, impl):
    if "mosaic" in impl and _tpu_older_than_v6e():
      self.skipTest("Only tested on TPU v6e and newer.")

    geometry = test_base.GEOMETRIES[geometry]
    compress_ratio = geometry.compress_ratio
    inputs = test_base.make_inputs(geometry, num_tokens, num_padding=3)

    @jax.jit
    def f(*inputs):
      return api.proj_and_save_state(
          *inputs, compress_ratio=compress_ratio, implementation=impl
      )

    out = f(*inputs)
    expected = reference.proj_and_save_state(
        *inputs, compress_ratio=compress_ratio
    )

    with self.subTest("value"):
      test_base.assert_caches_match(
          out, expected, inputs[4], geometry.rows_per_token
      )

    with self.subTest("correct_implementation_used"):
      # Check the lowered HLO for the kernel that was really used.
      opspecs = hlo_utils.get_opspecs(
          f.lower(*inputs), include_xla_kernels=False
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
