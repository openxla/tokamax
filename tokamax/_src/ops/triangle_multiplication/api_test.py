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
from tokamax._src import numerics
from tokamax._src.ops.triangle_multiplication import api

_IMPLEMENTATIONS: Final[tuple[str | None, ...]] = typing.get_args(
    api.Implementation
) + (None,)


def _get_params(n, c, h, d, dtype):
  params = dict(
      x=jax.ShapeDtypeStruct((n, n, c), dtype=dtype),
      mask=jax.ShapeDtypeStruct((n, n), dtype=jnp.bool),
      projection_in_weights=jax.ShapeDtypeStruct((c, 2, h), dtype=dtype),
      gate_in_weights=jax.ShapeDtypeStruct((c, 2, h), dtype=dtype),
      projection_out_weights=jax.ShapeDtypeStruct((h, d), dtype=dtype),
      gate_out_weights=jax.ShapeDtypeStruct((c, d), dtype=dtype),
      layernorm_in_scale=jax.ShapeDtypeStruct((c,), dtype=dtype),
      layernorm_in_offset=jax.ShapeDtypeStruct((c,), dtype=dtype),
      layernorm_out_scale=jax.ShapeDtypeStruct((h,), dtype=dtype),
      layernorm_out_offset=jax.ShapeDtypeStruct((h,), dtype=dtype),
  )
  return numerics.random_initialize(params)


class TriangleMultiplicationTest(parameterized.TestCase):

  @parameterized.product(
      triangle_type=["incoming", "outgoing"],
      dtype=[jnp.bfloat16, jnp.float32],
      implementation=_IMPLEMENTATIONS,
  )
  def test_triangle_multiplication(self, triangle_type, dtype, implementation):
    n = 8
    d = 64
    params = _get_params(n=n, c=16, h=32, d=d, dtype=dtype)

    @jax.jit
    def f(params):
      return api.triangle_multiplication(
          **params, triangle_type=triangle_type, implementation=implementation
      )

    out = f(params)
    self.assertEqual(out.shape, (n, n, d))
    self.assertEqual(out.dtype, dtype)

  def test_unsupported_implementation(self):
    params = _get_params(n=8, c=16, h=32, d=64, dtype=jnp.float32)
    with self.assertRaisesRegex(NotImplementedError, "Only XLA"):
      with jaxtyping.disable_jaxtyping():
        api.triangle_multiplication(
            **params,
            triangle_type="incoming",
            implementation="unsupported",
        )


if __name__ == "__main__":
  absltest.main()
