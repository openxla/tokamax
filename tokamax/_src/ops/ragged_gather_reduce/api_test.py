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
"""Tests for Ragged Gather Reduce API."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src import hlo_utils
from tokamax._src.ops.ragged_gather_reduce import api
from tokamax._src.ops.ragged_gather_reduce import base

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

    input_size = 1024
    hidden_size = 128
    reduce_group_size = 4
    key = jax.random.key(0)
    x = jax.random.normal(key, (input_size, hidden_size), jnp.float32).astype(
        dtype
    )
    indices = jax.random.randint(key, (input_size,), 0, input_size, jnp.int32)
    topk_weights = jax.random.normal(key, (input_size,), jnp.float32).astype(
        dtype
    )
    valid_rows_mask = jnp.ones((input_size,), jnp.bool_)

    @jax.jit
    def f(x, indices, topk_weights, valid_rows_mask):
      return api.ragged_gather_reduce(
          x,
          indices,
          topk_weights,
          valid_rows_mask,
          reduce_group_size,
          implementation=impl,
      )

    actual = f(x, indices, topk_weights, valid_rows_mask)
    desired = base.ragged_gather_reduce(
        x, indices, topk_weights, valid_rows_mask, reduce_group_size
    )

    with self.subTest("value"):
      np.testing.assert_allclose(
          actual.astype(jnp.float32),
          desired.astype(jnp.float32),
          rtol=1e-2,
          atol=1e-2,
      )

    with self.subTest("correct_implementation_used"):
      lowered = f.lower(x, indices, topk_weights, valid_rows_mask)
      opspecs = hlo_utils.get_opspecs(lowered, include_xla_kernels=False)
      if impl == "xla":
        self.assertEmpty(opspecs)
      elif opspecs:
        # "mosaic" is an alias the API resolves to the TPU kernel.
        self.assertIsInstance(
            opspecs[0].op, type(api.IMPLEMENTATIONS["mosaic_tpu"])
        )

  def test_unknown_implementation(self):
    input_size = 8
    x = jnp.zeros((input_size, 128), jnp.float32)
    indices = jnp.zeros((input_size,), jnp.int32)
    topk_weights = jnp.ones((input_size,), jnp.float32)
    valid_rows_mask = jnp.ones((input_size,), jnp.bool_)

    with self.assertRaisesRegex(ValueError, "Unknown implementation"):
      api.ragged_gather_reduce(
          x,
          indices,
          topk_weights,
          valid_rows_mask,
          reduce_group_size=4,
          implementation="unsupported",  # pyrefly: ignore[bad-argument-type]
      )


if __name__ == "__main__":
  absltest.main()
