# Copyright 2026 Google LLC. All Rights Reserved.
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

"""Unit tests for MoBA base operator and API."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.moba import api
from tokamax._src.ops.experimental.moba import base


class MixtureOfBlockAttentionBaseTest(absltest.TestCase):

  def test_base_forward(self):
    key = jax.random.PRNGKey(42)
    batch_size, num_heads, seq_len, head_dim = 2, 4, 512, 16
    chunk_size = 256
    topk = 2

    k0, k1, k2 = jax.random.split(key, 3)
    q = jax.random.normal(
        k0, (batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32
    )
    k = jax.random.normal(
        k1, (batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32
    )
    v = jax.random.normal(
        k2, (batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32
    )

    op_instance = base.MixtureOfBlockAttention()
    actual_out, _ = op_instance._fwd(
        q, k, v, topk=topk, chunk_size=chunk_size
    )
    expected_out, _ = base.moba_reference(
        q, k, v, topk=topk, chunk_size=chunk_size
    )

    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-5, atol=1e-5)
    self.assertEqual(
        actual_out.shape, (batch_size, num_heads, seq_len, head_dim)
    )

  def test_api_forward(self):
    key = jax.random.PRNGKey(101)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 512, 16
    chunk_size = 256
    topk = 2

    k0, k1, k2 = jax.random.split(key, 3)
    q = jax.random.normal(
        k0, (batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32
    )
    k = jax.random.normal(
        k1, (batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32
    )
    v = jax.random.normal(
        k2, (batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32
    )

    api_out = api.mixture_of_block_attention(
        q, k, v, topk=topk, chunk_size=chunk_size, implementation="xla"
    )
    ref_out, _ = base.moba_reference(q, k, v, topk=topk, chunk_size=chunk_size)
    np.testing.assert_allclose(api_out, ref_out, rtol=1e-5, atol=1e-5)

  def test_invalid_seq_len_raises(self):
    q = jnp.zeros((1, 2, 300, 16))
    k = jnp.zeros((1, 2, 300, 16))
    v = jnp.zeros((1, 2, 300, 16))
    op_instance = base.MixtureOfBlockAttention()
    with self.assertRaisesRegex(ValueError, "must be divisible by chunk_size"):
      op_instance._fwd(q, k, v, chunk_size=256)

  def test_shape_mismatch_raises(self):
    q = jnp.zeros((1, 2, 256, 16))
    k = jnp.zeros((1, 2, 256, 32))
    v = jnp.zeros((1, 2, 256, 16))
    op_instance = base.MixtureOfBlockAttention()
    with self.assertRaises((ValueError, TypeError)):
      op_instance._fwd(q, k, v, chunk_size=256)


if __name__ == "__main__":
  absltest.main()
