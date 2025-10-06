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
import typing
from typing import Final
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax import autotuning
from tokamax._src import mosaic_gpu
from tokamax._src import triton
from tokamax._src.ops.gated_linear_unit import api
from tokamax._src.ops.gated_linear_unit import test_base

_IMPLEMENTATIONS: Final[tuple[str | None, ...]] = typing.get_args(
    api.Implementation
) + (None,)


def _get_input_data(m, k, n, dtype=jnp.bfloat16):
  rng0, rng1 = jax.random.split(jax.random.PRNGKey(0))
  lhs = jax.random.normal(rng0, (m, k), dtype=dtype)
  rhs = jax.random.normal(rng1, (k, 2, n), dtype=dtype)
  return (lhs, rhs)


class GatedLinearUnitTest(parameterized.TestCase):

  @parameterized.parameters(*_IMPLEMENTATIONS)
  def test_basic_api(self, implementation):
    if implementation == "triton" and not triton.has_triton_support():
      self.skipTest("Triton not supported on this platform.")

    if not mosaic_gpu.has_mosaic_gpu_support() and implementation is not None:
      if "mosaic" in implementation:
        self.skipTest("Mosaic not supported on this platform.")

    lhs, rhs = _get_input_data(m=128, k=64, n=128)

    @jax.jit
    def f(x, weights):
      out = api.gated_linear_unit(
          x, weights, activation=jax.nn.sigmoid, implementation=implementation
      )
      return jnp.sum(out)

    @jax.jit
    def f_xla(x, weights):
      out = api.gated_linear_unit(
          x, weights, activation=jax.nn.sigmoid, implementation="xla"
      )
      return jnp.sum(out)

    out = f(lhs, rhs)
    out_golden = f_xla(lhs, rhs)

    with self.subTest("value"):
      chex.assert_trees_all_close(out, out_golden)

    args = autotuning.get_bound_args(f.lower(lhs, rhs))
    self.assertLen(args, 1)

    self.assertEqual(lhs.dtype, jnp.bfloat16)
    self.assertEqual(
        args[0].arguments["precision"],
        (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
    )

    with self.subTest("correct_implementation_used"):
      op = args[0].op
      if implementation is None:
        if jax.default_backend() == "gpu":
          # Ensure either a Triton or Mosaic kernel is used.
          self.assertTrue(
              isinstance(op, api.IMPLEMENTATIONS["triton"].__class__)
          )
      else:
        self.assertIsInstance(op, api.IMPLEMENTATIONS[implementation].__class__)


class GatedLinearUnitTritonTest(test_base.GatedLinearUnitTestBase):

  def __init__(self, *args):
    fn = functools.partial(api.gated_linear_unit, implementation="triton")
    super().__init__(*args, glu_fn=fn)


class GatedLinearUnitXlaTest(test_base.GatedLinearUnitTestBase):

  def __init__(self, *args):
    fn = functools.partial(api.gated_linear_unit, implementation="xla")
    super().__init__(*args, glu_fn=fn)


if __name__ == "__main__":
  absltest.main()
