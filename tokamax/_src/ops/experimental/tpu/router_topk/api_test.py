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
"""Tests for the sort-free MoE router top-k API."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src import hlo_utils
from tokamax._src.ops.experimental.tpu.router_topk import api
from tokamax._src.ops.experimental.tpu.router_topk import test_base

jax.config.parse_flags_with_absl()


def _tpu_older_than_v6e() -> bool:
  """Whether the default device is not a TPU v6e or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 6
  )


class ApiTest(parameterized.TestCase):

  @parameterized.product(
      rows_experts_k_dtype=[
          (64, 512, 10, jnp.float32),
          (1000, 512, 10, jnp.float32),
          (300, 256, 8, jnp.bfloat16),
      ],
      impl=["xla", "mosaic_tpu", None],
  )
  def test_basic_api(self, rows_experts_k_dtype, impl):
    if impl != "xla" and _tpu_older_than_v6e():
      self.skipTest("Only tested on TPU v6e and newer.")

    rows, experts, k, dtype = rows_experts_k_dtype
    scores = jnp.asarray(test_base.make_scores(rows, experts), dtype)

    @jax.jit
    def f(scores):
      return api.router_topk(scores, k, implementation=impl)

    weights, indices = f(scores)

    with self.subTest("value"):
      rw, ri = test_base.reference_topk(scores.astype(jnp.float32), k)
      np.testing.assert_array_equal(weights, rw)
      np.testing.assert_array_equal(indices, ri)

    with self.subTest("correct_implementation_used"):
      # Check the lowered HLO for the kernel that was really used.
      opspecs = hlo_utils.get_opspecs(
          f.lower(scores), include_xla_kernels=False
      )
      if impl == "xla":
        self.assertEmpty(opspecs)
      else:
        self.assertNotEmpty(opspecs)
        self.assertIsInstance(
            opspecs[0].op, type(api.IMPLEMENTATIONS["mosaic_tpu"])
        )

  def test_falls_back_to_xla_for_many_experts(self):
    if _tpu_older_than_v6e():
      self.skipTest("Only tested on TPU v6e and newer.")
    # Upstream's 512-row block of 8192 experts exceeds the VMEM budget.
    scores = jnp.asarray(test_base.make_scores(600, 8192))
    weights, indices = api.router_topk(scores, 8)
    rw, ri = test_base.reference_topk(scores, 8)
    np.testing.assert_array_equal(weights, rw)
    np.testing.assert_array_equal(indices, ri)

  def test_rejects_unknown_implementation(self):
    with self.assertRaisesRegex(ValueError, "Unknown implementation"):
      api.router_topk(jnp.zeros((8, 16)), 2, implementation="triton")  # pyrefly: ignore[bad-argument-type]


if __name__ == "__main__":
  absltest.main()
