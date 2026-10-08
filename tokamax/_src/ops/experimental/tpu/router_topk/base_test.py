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
"""Tests for the baseline JAX implementation of the MoE router top-k."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.router_topk import base
from tokamax._src.ops.experimental.tpu.router_topk import reference
from tokamax._src.ops.experimental.tpu.router_topk import test_base

jax.config.parse_flags_with_absl()


class BaseRouterTopKTest(test_base.RouterTopKTestBase):

  def __init__(self, *args):
    super().__init__(*args, topk_fn=base.RouterTopK())

  @parameterized.parameters((16,), (2, 8, 16))
  def test_rejects_bad_rank(self, *shape):
    with self.assertRaisesRegex(ValueError, "rank 2"):
      base.RouterTopK()(jnp.zeros(shape, jnp.float32), 2)

  @parameterized.parameters(0, -1, 17)
  def test_rejects_bad_k(self, k):
    with self.assertRaisesRegex(ValueError, r"k must be in \[1, num_experts"):
      base.RouterTopK()(jnp.zeros((8, 16), jnp.float32), k)

  @parameterized.parameters(np.int32, np.float64, jnp.float8_e4m3fn)
  def test_rejects_bad_dtype(self, dtype):
    with self.assertRaisesRegex(ValueError, "scores must be one of"):
      base.RouterTopK().bind(np.zeros((8, 16), dtype), 2)

  def test_binds_abstract_scores(self):
    ba = base.RouterTopK().bind(jax.ShapeDtypeStruct((8, 16), jnp.bfloat16), 2)
    self.assertEqual(ba.arguments["k"], 2)

  def test_reference_rejects_k_above_num_experts(self):
    with self.assertRaisesRegex(ValueError, "exceeds the expert count"):
      reference.rowmax_topk(jnp.zeros((8, 16), jnp.float32), 17)

  def test_reference_keeps_dtype(self):
    # The reference works in the dtype of `scores`; the op casts to float32.
    w, i = reference.rowmax_topk(jnp.zeros((8, 16), jnp.bfloat16), 2)
    self.assertEqual(w.dtype, jnp.bfloat16)
    self.assertEqual(i.dtype, jnp.int32)


if __name__ == "__main__":
  absltest.main()
