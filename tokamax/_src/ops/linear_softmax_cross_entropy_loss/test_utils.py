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

"""Test utils for flash loss package."""

import jax
from jaxtyping import Array, Integer, Key, Real


def generate_random_data(
    key: Key, b_dim: int, h_dim: int, v_dim: int
) -> tuple[Real[Array, "B H"], Integer[Array, "B"], Real[Array, "H V"]]:
  """Generates random data for flash loss tests.

  Args:
      key: JAX PRNG key.
      b_dim: The batch*sequence size.
      h_dim: The hidden size.
      v_dim: The vocabulary size.

  Returns:
      A tuple of JAX arrays (x, labels, w) representing the input data.
      x: A JAX array of shape (b_dim, h_dim) containing random normal data.
      labels: A JAX array of shape (b_dim,) containing labels index
      w: A JAX array of shape (h_dim, v_dim) containing random normal data.
  """
  key1, key2, key3 = jax.random.split(key, 3)
  x = jax.random.normal(key1, (b_dim, h_dim))
  labels = jax.random.randint(key2, shape=(b_dim,), minval=0, maxval=v_dim)
  w = jax.random.normal(key3, (h_dim, v_dim))
  return x, labels, w
