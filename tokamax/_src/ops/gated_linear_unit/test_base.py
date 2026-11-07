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

from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import gpu_utils


def _default_tolerances(dtype, precision):
  dtype = jnp.dtype(dtype)
  if dtype == jnp.bfloat16:
    return (0.1, 0.02), (2.0, 0.2)
  if dtype == jnp.float16:
    return (2e-2, 5e-3), (0.2, 2e-2)
  if dtype == jnp.float64 or precision == jax.lax.Precision.HIGHEST:
    return (1e-4, 1e-4), (5e-4, 5e-4)
  # Default FP32 on GPU uses TF32 (10-bit mantissa truncation on inputs).
  return (8e-2, 1e-2), (0.2, 2e-2)


# pylint: disable=missing-function-docstring
class GatedLinearUnitTestBase(parameterized.TestCase):
  """Base class for Gated Linear Unit op tests."""

  def __init__(self, *args, glu_fn):
    super().__init__(*args)
    self._glu_fn = glu_fn

  def _run_test(
      self,
      *,
      batch=(),
      m,
      n,
      k,
      dtype=jnp.bfloat16,
      activation=None,
      precision=None,
      use_tuple_weights=False,
      glu_fn=None,
      atol=None,
      rtol=None,
      atol_grads=None,
      rtol_grads=None,
  ):
    if isinstance(activation, str):
      activation = getattr(jax.nn, activation)

    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(0), 3)
    x = jax.random.normal(rng0, (*batch, m, k), dtype=dtype)
    w = jax.random.normal(rng1, (k, 2, n), dtype=dtype)
    dout = jax.random.normal(rng2, (*batch, m, n), dtype=dtype)
    ref_dtype = jnp.promote_types(dtype, jnp.float32)
    x_ref = x.astype(ref_dtype)
    w_ref = w.astype(ref_dtype)
    dout_ref = dout.astype(ref_dtype)

    if use_tuple_weights:
      w = jnp.unstack(w, axis=1)
      w_ref = jnp.unstack(w_ref, axis=1)

    def f_ref(x, weights):
      if isinstance(weights, tuple):
        w_gate, w_proj = weights
      else:
        w_gate, w_proj = jnp.unstack(weights, axis=1)
      gates = jnp.matmul(x, w_gate, precision=jax.lax.Precision.HIGHEST)
      projection = jnp.matmul(x, w_proj, precision=jax.lax.Precision.HIGHEST)
      return projection * (gates if activation is None else activation(gates))

    fn = self._glu_fn if glu_fn is None else glu_fn
    kwargs = {"activation": activation}
    if precision is not None:
      kwargs["precision"] = precision
    f = functools.partial(fn, **kwargs)

    (default_atol, default_rtol), (default_atol_grads, default_rtol_grads) = (
        _default_tolerances(dtype, precision)
    )
    atol = default_atol if atol is None else atol
    rtol = default_rtol if rtol is None else rtol
    atol_grads = default_atol_grads if atol_grads is None else atol_grads
    rtol_grads = default_rtol_grads if rtol_grads is None else rtol_grads

    expected, f_ref_vjp = jax.vjp(f_ref, x_ref, w_ref)
    out = f(x, w)
    chex.assert_trees_all_equal_shapes_and_dtypes(out, dout)
    chex.assert_trees_all_close(out, expected, atol=atol, rtol=rtol)

    actual, f_vjp = jax.vjp(f, x, w)
    grads = f_vjp(dout)
    chex.assert_trees_all_equal_shapes_and_dtypes(actual, dout)
    chex.assert_trees_all_equal_shapes_and_dtypes(grads, (x, w))
    chex.assert_trees_all_close(actual, expected, atol=atol, rtol=rtol)
    chex.assert_trees_all_close(
        grads, f_ref_vjp(dout_ref), atol=atol_grads, rtol=rtol_grads
    )

  @parameterized.parameters(
      dict(m=256, n=256, k=64, activation="sigmoid"),
      dict(m=256, n=256, k=64, activation="tanh"),
      dict(batch=(2, 3), m=256, n=256, k=64),
      dict(m=256, n=256, k=64, use_tuple_weights=True),
  )
  def test_gated_linear_unit(
      self, *, batch=(), m, n, k, activation=None, use_tuple_weights=False
  ):

    if isinstance(self._glu_fn, functools.partial):
      keywords = self._glu_fn.keywords

      if keywords["implementation"] == "triton":
        if not gpu_utils.has_triton_support():
          self.skipTest("Triton not supported on this platform.")

      if keywords["implementation"] == "mosaic":
        if not gpu_utils.has_mosaic_gpu_support():
          self.skipTest("Mosaic not supported on this platform.")

      # Currently the MosaicGpuGatedLinearUnit only supports sigmoid and swish
      # activations and no batch dimensions.
      if keywords["implementation"] == "plain_mosaic":
        if not gpu_utils.has_mosaic_gpu_support():
          self.skipTest("Mosaic not supported on this platform.")
        if activation == "tanh":
          self.skipTest("Tanh not supported for MosaicGpuGatedLinearUnit")
        elif activation is None:
          activation = "sigmoid"

      if batch:
        self.skipTest("Batch not supported for MosaicGpuGatedLinearUnit")

    self._run_test(
        batch=batch,
        m=m,
        n=n,
        k=k,
        activation=activation,
        use_tuple_weights=use_tuple_weights,
    )


# pylint: enable=missing-function-docstring
