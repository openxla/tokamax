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
"""Tests for CuTeDSL SM100 Gated Linear Unit."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import tokamax
from tokamax._src import gpu_utils
from tokamax._src.ops.gated_linear_unit import base
from tokamax._src.ops.gated_linear_unit import cutedsl


class CuteDslGatedLinearUnitTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.op = cutedsl.CuteDslGatedLinearUnit()
    self.xla_op = base.GatedLinearUnit()

  @parameterized.product(
      dtype=[jnp.bfloat16, jnp.float16],
      use_tuple_weights=[False, True],
      batch_shape=[(), (2,), (2, 3)],
      problem_size=[(128, 64, 128), (256, 128, 256)],
  )
  def test_forward(self, dtype, use_tuple_weights, batch_shape, problem_size):
    if not gpu_utils.is_sm100():
      self.skipTest("Requires SM100+ GPU.")

    m, k, n = problem_size
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(rng0, (*batch_shape, m, k), dtype=dtype)
    w = jax.random.normal(rng1, (k, 2, n), dtype=dtype)
    if use_tuple_weights:
      w = (w[:, 0, :], w[:, 1, :])

    f = lambda x, w: self.op(x, w, activation=jax.nn.swish)
    out = f(x, w)
    out_ref = self.xla_op(x, w, activation=jax.nn.swish)
    chex.assert_trees_all_close(out, out_ref, atol=0.05, rtol=0.05)

    bound_args = tokamax.autotuning.get_bound_args(f, x, w)
    self.assertIn(
        cutedsl.CuteDslGatedLinearUnit, [type(ba.op) for ba in bound_args]
    )

    autotune_res = tokamax.autotune(f, x, w, max_workers=1, progress_bar=False)
    self.assertIsInstance(autotune_res, tokamax.AutotuningResult)
    with autotune_res:
      out_autotuned = f(x, w)
    chex.assert_trees_all_close(out, out_autotuned, atol=0.05, rtol=0.05)


if __name__ == "__main__":
  absltest.main()
