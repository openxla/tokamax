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
"""Tests for jax-triton normalization op."""

import functools

from absl.testing import absltest
import chex
import jax
from tokamax._src.ops.normalization import jax_triton
from tokamax._src.ops.normalization import test_base


class JaxTritonNormalizationTest(test_base.NormalizationTestBase):
  """Runs the standard normalization test suite against the jax-triton impl."""

  def __init__(self, *args):
    super().__init__(*args, norm_fn=jax_triton.JaxTritonNormalization())

  def setUp(self):
    if jax.default_backend() == 'tpu':
      self.skipTest('Not supported on TPUs.')
    super().setUp()

  # `jax_triton.triton_call` has no `jax.vmap` batching rule (it raises by
  # design; the rule is inherently per-kernel). Skipping the shared helper skips
  # all parameterized `test_layer_norm_vmap*` cases until a
  # `jax.custom_batching.custom_vmap` rule is added to `JaxTritonNormalization`
  # (see `triton_call_vmap_plan.md`). TODO: re-enable (with the heuristics-mock
  # override and `test_remat_with_vmap` from `pallas_triton_test.py`) then.
  def _test_layer_norm_vmap(self, axis, vmap_in_axes):
    self.skipTest('jax_triton normalization does not support `vmap` yet.')

  def test_layer_norm_with_pre_scale(self):
    rngs = list(jax.random.split(jax.random.PRNGKey(0), 4))

    shape = (128, 32)
    x = jax.random.normal(rngs.pop(), shape)
    scale = jax.random.uniform(rngs.pop(), (shape[-1],))
    offset = jax.random.uniform(rngs.pop(), (shape[-1],))
    pre_scale = jax.random.uniform(rngs.pop(), (shape[-1],))
    epsilon = 1e-6

    y_expected = jax.nn.standardize(x * pre_scale, epsilon=epsilon) * scale
    y_expected += offset
    y_actual = self._norm_fn(
        lambda: x * pre_scale, scale, offset, epsilon=epsilon
    )
    chex.assert_trees_all_close(y_actual, y_expected, atol=1e-6)

  def test_remat(self):
    rngs = list(jax.random.split(jax.random.PRNGKey(0), 4))

    shape = (128, 32)
    x = jax.random.normal(rngs.pop(), shape)
    scale = jax.random.uniform(rngs.pop(), (shape[-1],))
    offset = jax.random.uniform(rngs.pop(), (shape[-1],))
    epsilon = 1e-6

    f = functools.partial(self._norm_fn, epsilon=epsilon)
    g_ref = jax.value_and_grad(lambda *args: f(*args).sum())
    g_remat = jax.value_and_grad(lambda *args: jax.remat(f)(*args).sum())
    g_remat_lowered = jax.jit(g_remat).lower(x, scale, offset)

    # jax-triton lowers every kernel to the same `triton_kernel_call` target
    # (no per-kernel names like pallas), so we count the total instead of
    # per-name: remat gives fwd + recomputed fwd-res + vjp = 3.
    hlo = str(g_remat_lowered.compiler_ir('stablehlo'))
    self.assertEqual(hlo.count('triton_kernel_call'), 3, msg=hlo)

    g_out = g_remat_lowered.compile()(x, scale, offset)
    chex.assert_trees_all_equal(g_out, g_ref(x, scale, offset))


if __name__ == '__main__':
  absltest.main()
