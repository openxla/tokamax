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
"""Gated Linear Unit test base."""

import functools
import math

from absl import logging
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import mosaic_gpu
from tokamax._src import triton

# pylint: disable=missing-function-docstring
class GatedLinearUnitTestBase(parameterized.TestCase):
  """Base class for Gated Linear Unit op tests."""

  def __init__(self, *args, glu_fn):
    super().__init__(*args)
    self._glu_fn = glu_fn

  @parameterized.parameters(
      dict(m=256, n=256, k=64, activation="sigmoid"),
      dict(m=256, n=256, k=64, activation="tanh"),
      dict(batch=(2, 3, 4), m=256, n=256, k=64),
  )
  def test_gated_linear_unit(self, *, batch=(), m, n, k, activation=None):

    if isinstance(self._glu_fn, functools.partial):
      keywords = self._glu_fn.keywords

      if keywords["implementation"] == "triton":
        if not triton.has_triton_support():
          self.skipTest("Triton not supported on this platform.")

      # Currently the MosaicGpuGatedLinearUnit only supports sigmoid and swish
      # activations and no batch dimensions.
      if keywords["implementation"] == "plain_mosaic":
        if not mosaic_gpu.has_mosaic_gpu_support():
          self.skipTest("Mosaic not supported on this platform.")
        if activation == "tanh":
          self.skipTest("Tanh not supported for MosaicGpuGatedLinearUnit")
        elif activation is None:
          activation = "sigmoid"

      if batch:
        self.skipTest("Batch not supported for MosaicGpuGatedLinearUnit")

    if activation is not None:
      activation = getattr(jax.nn, activation)

    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(0), 3)
    x = jax.random.normal(rng0, (*batch, m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(rng1, (k, 2, n), dtype=jnp.bfloat16)

    def f_ref(x, weights):
      w, v = jnp.unstack(weights, axis=1)
      gates = jnp.matmul(x, w, precision=jax.lax.Precision.HIGHEST)
      projection = jnp.matmul(x, v, precision=jax.lax.Precision.HIGHEST)
      return projection * (gates if activation is None else activation(gates))

    f = functools.partial(self._glu_fn, activation=activation)

    atol = 5e-2 * math.sqrt(k)
    assert_all_close = functools.partial(
        chex.assert_trees_all_close, atol=atol, rtol=5e-3
    )
    assert_all_close(f(x, w), f_ref(x, w))

    dout = jax.random.normal(rng2, (*batch, m, n), dtype=jnp.bfloat16)
    expected, f_ref_vjp = jax.vjp(f_ref, x, w)
    actual, f_vjp = jax.vjp(f, x, w)
    assert_all_close(actual, expected)
    atol_grads = atol * 5  # Need larger tolerance for grads.
    assert_all_close(f_vjp(dout), f_ref_vjp(dout), atol=atol_grads)


# pylint: enable=missing-function-docstring
