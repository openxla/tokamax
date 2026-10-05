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
"""Tests for Ragged Gather API."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src import hlo_utils
from tokamax._src.ops.ragged_gather import api

jax.config.parse_flags_with_absl()


class ApiTest(parameterized.TestCase):

  @parameterized.product(
      in_out_size=[(512, 400), (512, 1024), (512, 32)],
      start_end=[(3, 338), (10, 422), (3, 28)],
      hidden_size=[128, 512],
      dtype=[jnp.bfloat16, jnp.float32],
      impl=["xla", "mosaic", "mosaic_tpu", "mosaic_tpu_v2"],
  )
  def test_basic_api(self, in_out_size, hidden_size, start_end, dtype, impl):
    if "mosaic" in impl and backend.get_default_device().device_kind != "TPU7x":
      self.skipTest("Only tested on TPU7x.")

    in_size, out_size = in_out_size
    start, end = start_end
    start = min(start, out_size)
    end = min(end, out_size)
    key = jax.random.key(0)
    x = jax.random.normal(key, (in_size, hidden_size), jnp.float32).astype(
        dtype
    )
    indices = jax.random.randint(key, (out_size,), 0, in_size, jnp.int32)

    start_arr = jnp.array([start], jnp.int32)
    end_arr = jnp.array([end], jnp.int32)

    @jax.jit
    def f(x, indices, start, end):
      return api.ragged_gather(x, indices, start, end, implementation=impl)

    actual = f(x, indices, start_arr, end_arr)
    desired = x[indices]

    with self.subTest("value"):
      np.testing.assert_allclose(
          actual[start:end], desired[start:end], rtol=1e-2, atol=1e-2
      )

    with self.subTest("correct_implementation_used"):
      # Check the lowered HLO for the kernel that was really used.
      opspecs = hlo_utils.get_opspecs(
          f.lower(x, indices, start_arr, end_arr), include_xla_kernels=False
      )
      if impl == "xla":
        self.assertEmpty(opspecs)
      else:
        expected = impl
        if expected == "mosaic":
          expected = (
              "mosaic_tpu_v2"
              if "mosaic_tpu_v2" in api.IMPLEMENTATIONS
              else "mosaic_tpu"
          )
        self.assertNotEmpty(opspecs)
        self.assertIsInstance(
            opspecs[0].op, type(api.IMPLEMENTATIONS[expected])
        )


if __name__ == "__main__":
  absltest.main()
