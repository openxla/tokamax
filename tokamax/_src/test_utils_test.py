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

import functools
import random
from absl.testing import absltest
import jax
from tokamax._src import test_utils


class TestUtilsTest(absltest.TestCase):

  def test_assert_trees_all_close(self):
    epsilon = 1e-6
    strict_atol = 1e-5
    relaxed_atol = 1e-4
    relaxed_fraction = 0.01

    assert_trees_all_close = functools.partial(
        test_utils.assert_trees_all_close,
        rtol=0.0,
        atol={(1.0 - relaxed_fraction): strict_atol, 1.0: relaxed_atol},
    )

    x = jax.random.normal(jax.random.PRNGKey(42), shape=(256, 256))
    y = x + strict_atol - epsilon
    z = x - strict_atol + epsilon

    # Generate 1% of random indices. Use a set to avoid duplicate indices.
    idx = set()
    rand_idx = lambda: (random.randint(0, 255), random.randint(0, 255))
    while len(idx) < int(x.size * relaxed_fraction):
      idx.add(rand_idx())

    for i in idx:
      y = y.at[*i].set(y[*i] - strict_atol + relaxed_atol)
      z = z.at[*i].set(z[*i] + strict_atol - relaxed_atol)

    assert_trees_all_close(x, y, z)

    # Add another value within relaxed tolerance to make it >1% and fail.
    while (i := rand_idx()) in idx:
      pass

    y = y.at[*i].set(y[*i] - strict_atol + relaxed_atol)
    z = z.at[*i].set(z[*i] + strict_atol - relaxed_atol)

    with self.assertRaises(AssertionError):
      assert_trees_all_close(x, y, z)


if __name__ == "__main__":
  absltest.main()
