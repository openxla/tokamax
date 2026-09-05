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
"""Tests for the `with_memory_space_constraint` batching rule. Runs anywhere; needs no TPU."""

from absl.testing import absltest
import jax
from jax._src.pallas import core as pallas_core
import jax.numpy as jnp
from tokamax._src import pallas_batching_compat


_PRIM = pallas_core.with_memory_space_constraint_p


class PallasBatchingCompatTest(absltest.TestCase):

  def test_rule_is_registered(self):
    pallas_batching_compat.register()
    # Probe via the module's own helper: the registries are proxies, and the primitive cannot be
    # bound eagerly (its `def_impl` raises by design), so a lookup is the only safe check.
    self.assertTrue(pallas_batching_compat._has_rule())  # pylint: disable=protected-access

  def test_register_is_idempotent_and_yields_to_jax(self):
    """A second call must not install over an existing rule, whoever owns it."""
    pallas_batching_compat.register()
    self.assertFalse(pallas_batching_compat.register())

  def test_vmap_traces_through_the_constraint(self):
    """Without the rule this raises NotImplementedError while tracing."""
    pallas_batching_compat.register()

    def f(x):
      # `pl_core` rather than `pltpu` so the test does not require the TPU memory-space enum.
      return _PRIM.bind(x, memory_space=pallas_core.MemorySpace.ANY)

    # jax.eval_shape only traces, which is the part the batching rule governs; the primitive's
    # def_impl deliberately raises, so it must not be evaluated eagerly.
    out = jax.eval_shape(jax.vmap(f), jax.ShapeDtypeStruct((4, 8), jnp.float32))
    self.assertEqual(out.shape, (4, 8))

  def test_batch_dim_is_preserved_off_axis_zero(self):
    """The rule must return the incoming batch dim, not assume 0."""
    pallas_batching_compat.register()

    def f(x):
      return _PRIM.bind(x, memory_space=pallas_core.MemorySpace.ANY)

    x = jax.ShapeDtypeStruct((3, 5, 7), jnp.float32)
    out = jax.eval_shape(jax.vmap(f, in_axes=1, out_axes=1), x)
    self.assertEqual(out.shape, (3, 5, 7))

  def test_dtype_and_shape_are_preserved(self):
    """It is an annotation, so batching must not change shape or dtype.

    Checked under a trace: the primitive's `def_impl` raises, so the rule cannot be called with
    concrete arrays.
    """
    pallas_batching_compat.register()

    def f(x):
      return _PRIM.bind(x, memory_space=pallas_core.MemorySpace.ANY)

    out = jax.eval_shape(jax.vmap(f), jax.ShapeDtypeStruct((2, 6), jnp.int32))
    self.assertEqual(out.shape, (2, 6))
    self.assertEqual(out.dtype, jnp.int32)


if __name__ == "__main__":
  absltest.main()
