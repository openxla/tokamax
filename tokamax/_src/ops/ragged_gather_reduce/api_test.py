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
"""Tests for the Ragged Gather Reduce API."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src import hlo_utils
from tokamax._src.ops.ragged_gather_reduce import api
from tokamax._src.ops.ragged_gather_reduce import reference
from tokamax._src.ops.ragged_gather_reduce import test_base

jax.config.parse_flags_with_absl()


def _tpu_older_than_7x() -> bool:
  """Whether the default device is not a TPU7x or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 7
  )


class ApiTest(parameterized.TestCase):

  @parameterized.product(
      shape=[(2048, 8, 2048), (1024, 8, 7168)],
      impl=["xla", "mosaic_tpu"],
  )
  def test_basic_api(self, shape, impl):
    if "mosaic" in impl and _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")

    num_tokens, reduce_group_size, hidden_size = shape
    x, indices, topk_weights, valid_rows_mask = test_base.make_inputs(
        num_tokens, reduce_group_size, hidden_size
    )

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

    out = f(x, indices, topk_weights, valid_rows_mask)
    expected = reference.ragged_gather_reduce(
        x, indices, topk_weights, valid_rows_mask, reduce_group_size
    )

    with self.subTest("value"):
      np.testing.assert_allclose(
          np.asarray(out, np.float32),
          np.asarray(expected, np.float32),
          atol=test_base.ATOL,
          rtol=test_base.RTOL,
      )

    with self.subTest("correct_implementation_used"):
      # Check the lowered HLO for the kernel that was really used.
      opspecs = hlo_utils.get_opspecs(
          f.lower(x, indices, topk_weights, valid_rows_mask),
          include_xla_kernels=False,
      )
      if impl == "xla":
        self.assertEmpty(opspecs)
      else:
        self.assertNotEmpty(opspecs)
        self.assertIsInstance(
            opspecs[0].op, type(api.IMPLEMENTATIONS["mosaic_tpu"])
        )

  def test_unsupported_dtype_falls_back_to_xla(self):
    if _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")
    num_tokens, reduce_group_size, hidden_size = 64, 4, 128
    x, indices, topk_weights, valid_rows_mask = test_base.make_inputs(
        num_tokens, reduce_group_size, hidden_size
    )
    x = x.astype(jnp.float32)

    with self.assertRaisesRegex(NotImplementedError, "bfloat16"):
      api.ragged_gather_reduce(
          x,
          indices,
          topk_weights,
          valid_rows_mask,
          reduce_group_size,
          implementation="mosaic_tpu",
      )

    # The default implementations fall back to XLA.
    out = api.ragged_gather_reduce(
        x, indices, topk_weights, valid_rows_mask, reduce_group_size
    )
    expected = reference.ragged_gather_reduce(
        x, indices, topk_weights, valid_rows_mask, reduce_group_size
    )
    np.testing.assert_allclose(out, expected, atol=1e-5, rtol=1e-5)

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
