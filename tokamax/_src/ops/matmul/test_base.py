# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
import itertools

from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from tokamax._src.ops.matmul import base


def ref(a, b):
  """Reference implementation of matmul."""

  return jax.lax.dot(
      a,
      b,
      dimension_numbers=base.DIMENSION_NUMBERS,
      precision=jax.lax.Precision.HIGHEST,
  )


class MatmulTestBase(parameterized.TestCase):

  def __init__(self, *args, matmul_fn):
    super().__init__(*args)
    self._matmul_fn = matmul_fn

  @parameterized.named_parameters(
    (f"{m=}-{n=}-{k=}", m, n, k)
    for m, n, k in itertools.product(
        (1024, 2048),
        (1024, 2048),
        (1024, 2048),
    )
  )
  def test_simple(self, m, n, k):
    dtype = jnp.float16
    key0, key1 = jr.split(jr.key(0), 2)
    a = jr.normal(key0, shape=(m, k), dtype=dtype)
    b = jr.normal(key1, shape=(n, k), dtype=dtype)
    actual = self._matmul_fn(a, b)
    chex.assert_trees_all_close(actual, ref(a, b))
