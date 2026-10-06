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
"""Tests for base TopK operator."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.topk import base
from tokamax._src.ops.experimental.tpu.topk import test_base


class BaseTopKTest(test_base.TopKTestBase):

  def __init__(self, *args):
    super().__init__(*args, topk_fn=base.TopK())

  def test_known_values(self):
    op = base.TopK()
    scores = jnp.array(
        [[3.0, 1.0, 4.0, 2.0], [5.0, -jnp.inf, 2.0, 6.0]], dtype=jnp.float32
    )
    row_lengths = jnp.array([4, 3], dtype=jnp.int32)
    indices, scores_bits = op(
        scores, 3, row_lengths=row_lengths, return_scores=True
    )
    np.testing.assert_array_equal(
        indices, np.array([[2, 0, 3], [0, 2, -1]], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        jax.lax.bitcast_convert_type(scores_bits, jnp.float32),
        np.array([[4.0, 3.0, 2.0], [5.0, 2.0, -np.inf]], dtype=np.float32),
    )

  def test_bind_validation(self):
    op = base.TopK()
    scores = jnp.zeros((2, 4), dtype=jnp.float32)
    with self.assertRaises(ValueError):
      op.bind(scores, 0)
    with self.assertRaises(ValueError):
      op.bind(scores, 5)


if __name__ == "__main__":
  absltest.main()
