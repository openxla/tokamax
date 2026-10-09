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
"""Tests for the SparseMLA API."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import backend
from tokamax._src import hlo_utils
from tokamax._src.ops.experimental.sparse_mla import api
from tokamax._src.ops.experimental.sparse_mla import test_base


class ApiTest(parameterized.TestCase):

  @parameterized.product(
      pad_tokens=[False, True],
      impl=["xla", "mosaic_tpu"],
  )
  def test_basic_api(self, pad_tokens: bool, impl: api.Implementation):
    if (
        impl == "mosaic_tpu"
        and backend.get_default_device().device_kind != "TPU7x"
    ):
      self.skipTest("Only tested on TPU7x.")

    inputs, actual_tokens = test_base.make_inputs(
        batch_size=4, topk=512, pad_tokens=pad_tokens
    )

    @jax.jit
    def f(*args):
      return api.sparse_mla(*args, sm_scale=1.0, implementation=impl)

    actual = f(*inputs)

    with self.subTest("value"):
      test_base.assert_matches_reference(actual, inputs, actual_tokens)

    with self.subTest("correct_implementation_used"):
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
