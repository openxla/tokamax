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
"""Auto-differentiation utilities."""

from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.extend as jex


R = TypeVar("R")
T = TypeVar("T")


def get_vjp_taking_residuals(
    fn: Callable[..., tuple[T, R]], *primals: ...
) -> Callable[[R, T], Any] | None:
  """Attempts to create a VJP function with respect to the requested residuals.

  When using `jax.vjp`, we get a VJP function that closes over the necessary
  residuals. `get_vjp_taking_residuals` can be used to create a function that
  explicitly takes user-defined residuals as parameters.

  `get_vjp_taking_residuals` may be used, for example, to create a VJP function
  with respect to the outputs of a function. One function that allows this is
  the sigmoid function.

  Example:
    ```
    def sigmoid_with_residual(x):
      y = jax.nn.sigmoid(x)
      return y, y  # `y` is both the output and residual.

    # This is equivalent to `sigmoid_vjp = lambda y, dy: (y - y * y) * dy`.
    sigmoid_vjp = get_vjp_taking_residuals(sigmoid_with_residual, x)
    ```

  Args:
    fn: A function mapping `(primals) -> (primals_out, residuals)`. The choice
      of `residuals` is up the caller, but only some combinations can produce a
      VJP function (depending on the calculation of `primals_out`). The
      residuals may come from `primals`, `primals_out`, or any intermediate
      value in the calculation.
    *primals: The inputs to the function.

  Returns:
    A function mapping `(residuals, cotangents) -> tangents`, if possible,
    otherwise `None`.
  """

  def residuals_and_tangents(primals, cotangents):
    _, fn_vjp, residuals = jax.vjp(fn, *primals, has_aux=True)
    return residuals, fn_vjp(cotangents)

  primals_out, residuals = jax.eval_shape(fn, *primals)
  closed_jaxpr = jax.make_jaxpr(residuals_and_tangents)(primals, primals_out)
  jaxpr = closed_jaxpr.jaxpr

  primals_tree = jax.tree.structure(primals)
  num_primals = primals_tree.num_leaves
  num_residuals = len(jax.tree.leaves(residuals))

  split = lambda x, n: (x[:n], x[n:])
  primal_vars, cotangent_vars = split(jaxpr.invars, num_primals)
  residual_vars, tangent_vars = split(jaxpr.outvars, num_residuals)

  # Calculate which vars depend upon primal values not in the residuals.
  # The VJP function will not have these inputs, so no operation may depend upon
  # them (including transitively).
  banned_vars = set(primal_vars) - set(residual_vars)
  eqns = []
  for eqn in jaxpr.eqns:
    invars = {v for v in eqn.invars if not isinstance(v, jex.core.Literal)}
    if banned_vars.isdisjoint(invars):
      # The residual values become inputs to the VJP function. We replace the
      # outvars that originally produced the residual values to prevent the new
      # inputs from being shadowed.
      for i, var in enumerate(eqn.outvars):
        if var in residual_vars:
          eqn.outvars[i] = jax.core.DropVar(var.aval)
      eqns.append(eqn)  # `eqn` is still present in VJP function.
    else:
      banned_vars |= set(eqn.outvars) - set(residual_vars)

  if not banned_vars.isdisjoint(tangent_vars):
    return None

  # TODO: Remove unneeded eqns?
  # Create jaxpr for the VJP function, with the new inputs, outputs, and eqns.
  invars = residual_vars + cotangent_vars
  jaxpr = jaxpr.replace(
      invars=invars, outvars=tangent_vars, eqns=eqns, debug_info=None
  )
  closed_jaxpr = closed_jaxpr.replace(jaxpr=jaxpr)

  def vjp_taking_residuals(residuals, cotangents):
    args_flat = jax.tree.leaves((residuals, cotangents))
    tangents_flat = jex.core.jaxpr_as_fun(closed_jaxpr)(*args_flat)
    return primals_tree.unflatten(tangents_flat)

  return vjp_taking_residuals
