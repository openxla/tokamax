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
from absl.testing import parameterized
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Array, Float  # pylint: disable=g-importing-member,g-multiple-import
from tokamax._src import jaxtyping


@jaxtyping.jaxtyped
def f(x: Float[Array, "A B"]) -> Float[Array, "A B"]:
  return x**2


class JaxtypingTest(parameterized.TestCase):

  def test_disable_jaxtyping(self):
    x = jnp.ones((2, 2, 2))

    with self.assertRaises(jt.TypeCheckError):
      f(x).block_until_ready()

    with self.assertRaises(jt.TypeCheckError):
      with jaxtyping.disable_jaxtyping(disable=False):
        f(x).block_until_ready()

    with jaxtyping.disable_jaxtyping():
      f(x).block_until_ready()

    with self.assertRaises(jt.TypeCheckError):
      f(x).block_until_ready()


if __name__ == "__main__":
  absltest.main()
