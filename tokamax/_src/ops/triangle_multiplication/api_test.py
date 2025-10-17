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
import typing
from typing import Final

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from tokamax._src import jaxtyping
from tokamax._src.ops.triangle_multiplication import api

_IMPLEMENTATIONS: Final[tuple[str | None, ...]] = typing.get_args(
    api.Implementation
) + (None,)


def _get_params(n, c, h, d, dtype):
  keys = jax.random.split(jax.random.PRNGKey(0), 10)
  x = jax.random.normal(keys[0], (n, n, c), dtype=dtype)
  mask = jax.random.bernoulli(keys[1], shape=(n, n))
  gate_projection_weights = jax.random.normal(
      keys[2], (c, 2, h, 2), dtype=dtype
  )
  projection_out_weights = jax.random.normal(keys[3], (h, d), dtype=dtype)
  gate_out_weights = jax.random.normal(keys[4], (c, d), dtype=dtype)
  layernorm_in_scale = jax.random.normal(keys[5], (c,), dtype=dtype)
  layernorm_in_offset = jax.random.normal(keys[6], (c,), dtype=dtype)
  layernorm_out_scale = jax.random.normal(keys[7], (h,), dtype=dtype)
  layernorm_out_offset = jax.random.normal(keys[8], (h,), dtype=dtype)
  return (
      x,
      mask,
      gate_projection_weights,
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
  )


class TriangleMultiplicationTest(parameterized.TestCase):

  @parameterized.product(
      triangle_type=["incoming", "outgoing"],
      dtype=[jnp.bfloat16, jnp.float32],
      implementation=_IMPLEMENTATIONS,
  )
  def test_triangle_multiplication(self, triangle_type, dtype, implementation):
    n, c, h, d = 8, 16, 32, 64
    x, *params = _get_params(n, c, h, d, dtype)

    @jax.jit
    def f(x, *args):
      return api.triangle_multiplication(
          x, *args, triangle_type=triangle_type, implementation=implementation
      )

    out = f(x, *params)
    self.assertEqual(out.shape, (n, n, d))
    self.assertEqual(out.dtype, dtype)

  def test_unsupported_implementation(self):
    n, c, h, d = 8, 16, 32, 64
    x, *params = _get_params(n, c, h, d, jnp.float32)
    with self.assertRaisesRegex(NotImplementedError, "Only XLA"):
      with jaxtyping.disable_jaxtyping():
        api.triangle_multiplication(
            x,
            *params,
            triangle_type="incoming",
            implementation="unsupported",
        )


if __name__ == "__main__":
  absltest.main()
