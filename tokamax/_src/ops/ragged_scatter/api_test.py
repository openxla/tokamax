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
"""Tests for Ragged Scatter API."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src import hlo_utils
from tokamax._src.ops.ragged_scatter import api
from tokamax._src.ops.ragged_scatter import base

jax.config.parse_flags_with_absl()


class ApiTest(parameterized.TestCase):

  @parameterized.product(
      dtype=[jnp.bfloat16, jnp.float32],
      impl=["xla", "mosaic", "mosaic_tpu"],
  )
  def test_basic_api(self, dtype, impl):
    if "mosaic" in impl:
      mosaic_op = api.IMPLEMENTATIONS.get("mosaic_tpu")
      supported_on = getattr(mosaic_op, "supported_on", None)
      if supported_on is None or not supported_on(backend.get_default_device()):
        self.skipTest("mosaic_tpu implementation not supported on this device.")

    out_size = 1024
    hidden_size = 128
    start, end = 10, 422
    key = jax.random.key(0)
    x = jax.random.normal(key, (out_size, hidden_size), jnp.float32).astype(
        dtype
    )
    indices = jax.random.permutation(key, out_size).astype(jnp.int32)
    start_arr = jnp.array([start], jnp.int32)
    end_arr = jnp.array([end], jnp.int32)

    @jax.jit
    def f(x, indices, start_arr, end_arr):
      return api.ragged_scatter(
          x, indices, start_arr, end_arr, implementation=impl
      )

    actual = f(x, indices, start_arr, end_arr)
    desired = base.ragged_scatter(x, indices, start_arr, end_arr)

    with self.subTest("value"):
      # The Pallas kernel is undefined outside `[start, end)`, so mask both
      # sides before comparing, as the kernel test does.
      mask = (indices >= start) & (indices < end)
      np.testing.assert_allclose(
          jnp.where(mask[:, None], actual, 0).astype(jnp.float32),
          jnp.where(mask[:, None], desired, 0).astype(jnp.float32),
          rtol=1e-2,
          atol=1e-2,
      )

    with self.subTest("correct_implementation_used"):
      # The Mosaic kernel is registered inside a `try: ... except ImportError`,
      # so what `impl` asks for and what the op actually dispatches to can
      # differ. The values above cannot tell the difference -- the reference is
      # the XLA implementation -- so check the lowered HLO for the kernel that
      # was really used.
      lowered = f.lower(x, indices, start_arr, end_arr)
      opspecs = hlo_utils.get_opspecs(lowered, include_xla_kernels=False)
      if impl == "xla":
        self.assertEmpty(opspecs)
      else:
        # "mosaic" is an alias the API resolves to the TPU kernel.
        self.assertNotEmpty(opspecs)
        self.assertIsInstance(
            opspecs[0].op, type(api.IMPLEMENTATIONS["mosaic_tpu"])
        )

  def test_unknown_implementation(self):
    x = jnp.zeros((8, 128), jnp.float32)
    indices = jnp.arange(8, dtype=jnp.int32)
    start = jnp.array([0], jnp.int32)
    end = jnp.array([8], jnp.int32)

    with self.assertRaisesRegex(ValueError, "Unknown implementation"):
      api.ragged_scatter(
          x,
          indices,
          start,
          end,
          implementation="unsupported",  # pyrefly: ignore[bad-argument-type]
      )


if __name__ == "__main__":
  absltest.main()
