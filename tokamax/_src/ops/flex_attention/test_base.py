# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for FlexAttention."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from tokamax._src import shape as shape_lib


# pylint: disable=missing-function-docstring
class FlexAttentionTestBase(absltest.TestCase):
  """Base class for FlexAttention tests.

  NOTE: There are relatively few tests here, as most of the functionality is
  tested via the wrapped dot-product attention tests (see `wrapper_test_base`).
  """

  def __init__(self, *args, flex_attn):
    super().__init__(*args)
    self._flex_attn = flex_attn

  def test_bias_upscale(self):
    if jax.__version_info__ < (0, 7, 2):
      self.skipTest("Requires JAX 0.7.2 or later.")

    rng0, rng1, rng2, rng3 = jax.random.split(jax.random.PRNGKey(0), 4)
    q = jax.random.normal(rng0, (2, 512, 4, 64))
    k = jax.random.normal(rng1, (2, 512, 4, 64))
    v = jax.random.normal(rng2, (2, 512, 4, 64))
    bias = jax.random.normal(rng3, (2, 1, 256, 128))

    def upscale_bias(x):
      with shape_lib.upcast_broadcast():
        return jnp.repeat(jnp.repeat(x, 2, axis=-2), 4, axis=-1)

    with jax.default_matmul_precision("highest"):
      expected = jax.nn.dot_product_attention(q, k, v, bias=upscale_bias(bias))

    def score_mod(scores):
      return scores / jnp.sqrt(q.shape[-1]) + upscale_bias(bias)

    actual = self._flex_attn(q, k, v, score_mod=score_mod, precision="highest")
    chex.assert_trees_all_close(actual, expected, atol=2e-6)

  def test_bias_relative_positional(self):
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(rng0, (2, 512, 4, 64))
    k = jax.random.normal(rng1, (2, 512, 4, 64))
    v = jax.random.normal(rng2, (2, 512, 4, 64))

    with jax.default_matmul_precision("highest"):
      bias = jnp.arange(q.shape[-3])[:, None] - jnp.arange(k.shape[-3])
      expected = jax.nn.dot_product_attention(q, k, v, bias=bias)

    def score_mod(scores):
      q_idx = jnp.arange(q.shape[-3])
      k_idx = jnp.arange(k.shape[-3])
      return scores / jnp.sqrt(q.shape[-1]) + (q_idx[:, None] - k_idx)

    actual = self._flex_attn(q, k, v, score_mod=score_mod, precision="highest")
    chex.assert_trees_all_close(actual, expected, atol=2e-6)


# pylint: enable=missing-function-docstring
