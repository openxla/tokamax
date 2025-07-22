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

from absl.testing import absltest
import chex
import jax
from jax import export
import jax.numpy as jnp
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import test_base


class MaskTest(absltest.TestCase):

  def test_as_array(self):
    mask = jax.random.bernoulli(jax.random.PRNGKey(0), shape=(1, 2, 3))
    chex.assert_trees_all_equal(base.Mask(mask).as_array(2, 3), mask)

  def test_as_array_symbolic_args_specs(self):
    input_specs = export.symbolic_args_specs(
        {"mask": jax.random.bernoulli(jax.random.PRNGKey(0), shape=(1, 2, 3))},
        {"mask": "(b, q_len, k_len)"},
    )

    def f(mask):
      return base.Mask(mask).as_array(mask.shape[-2], mask.shape[-1])

    _ = jax.eval_shape(f, **input_specs)

  def test_take(self):
    mask_array = jax.random.bernoulli(jax.random.PRNGKey(0), shape=(32, 32))
    mask, is_causal = base.Mask(mask_array).take("is_causal")
    chex.assert_trees_all_equal(mask.as_array(32, 32), mask_array[None, ...])
    self.assertFalse(is_causal)

    causal_mask = base.Mask(mask_array, is_causal=True)
    causal_mask, is_causal = causal_mask.take("is_causal")
    actual = causal_mask.as_array(32, 32)
    chex.assert_trees_all_equal(actual, mask_array[None, ...])
    self.assertTrue(is_causal)

  def test_k_range(self):
    k_start = jnp.array([0, 0, 1, 2])
    k_end = jnp.array([1, 2, 3, 4])
    actual = base.Mask(k_start=k_start, k_end=k_end).as_array(4, 3)
    expected = jnp.array(
        [[[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]]], dtype=bool
    )
    chex.assert_trees_all_equal(actual, expected)

  def test_q_range(self):
    q_end = jnp.array([1, 3, 0])
    actual = base.Mask(q_end=q_end).as_array(4, 3)
    expected = jnp.array(
        [[[1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]]], dtype=bool
    )
    chex.assert_trees_all_equal(actual, expected)

  def test_is_causal(self):
    actual = base.Mask(is_causal=True).as_array(4, 4)
    expected = jnp.array(
        [[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]], dtype=bool
    )
    chex.assert_trees_all_equal(actual, expected)

  def test_intersection(self):
    mask0 = jax.random.bernoulli(jax.random.PRNGKey(0), shape=(32, 32))
    mask1 = jax.random.bernoulli(jax.random.PRNGKey(0), shape=(32, 32))
    actual = (base.Mask(mask0) & base.Mask(mask1)).as_array(32, 32)
    chex.assert_trees_all_equal(actual, (mask0 & mask1)[None, ...])

    q_start0 = jnp.array([0, 1, 2, 3])
    q_start1 = jnp.array([2, 1, 0, 4])
    mask = base.Mask(q_start=q_start0) & base.Mask(q_start=q_start1)
    chex.assert_trees_all_equal(mask.q_start, jnp.array([2, 1, 2, 4]))

    q_end0 = jnp.array([0, 1, 2, 3])
    q_end1 = jnp.array([2, 1, 0, 4])
    mask = base.Mask(q_end=q_end0) & base.Mask(q_end=q_end1)
    chex.assert_trees_all_equal(mask.q_end, jnp.array([0, 1, 0, 3]))

    k_start0 = jnp.array([0, 1, 2, 3])
    k_start1 = jnp.array([2, 1, 0, 4])
    mask = base.Mask(k_start=k_start0) & base.Mask(k_start=k_start1)
    chex.assert_trees_all_equal(mask.k_start, jnp.array([2, 1, 2, 4]))

    k_end0 = jnp.array([0, 1, 2, 3])
    k_end1 = jnp.array([2, 1, 0, 4])
    mask = base.Mask(k_end=k_end0) & base.Mask(k_end=k_end1)
    chex.assert_trees_all_equal(mask.k_end, jnp.array([0, 1, 0, 3]))

    for a in [True, False]:
      for b in [True, False]:
        mask = base.Mask(is_causal=a) & base.Mask(is_causal=b)
        self.assertEqual(mask.is_causal, a or b)

    two_sequence_mask = jnp.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=bool
    )
    actual = (base.Mask(is_causal=True) & two_sequence_mask).as_array(4, 4)
    expected = jnp.array(
        [[[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]], dtype=bool
    )
    chex.assert_trees_all_equal(actual, expected)


class DotProductAttentionTest(test_base.AttentionTestBase):

  def __init__(self, *args, vjp=None):
    super().__init__(*args, attention_fn=base.DotProductAttention(vjp=vjp))

  def _run_test(self, q_shape, *args, dtype=jnp.float32, **kwargs):
    if q_shape[1] >= 16384:
      self.skipTest("XLA seems to fail for so long sequences (b/384038935)")

    impl_kwargs = kwargs.pop("impl_kwargs", {})
    if dtype == jnp.float32:
      impl_kwargs["precision"] = jax.lax.Precision.HIGHEST
    super()._run_test(
        q_shape, *args, dtype=dtype, impl_kwargs=impl_kwargs, **kwargs
    )


class DotProductAttentionWithExplicitVjpTest(DotProductAttentionTest):

  def __init__(self, *args):
    super().__init__(*args, vjp=base.DotProductAttentionVjp())


if __name__ == "__main__":
  absltest.main()
