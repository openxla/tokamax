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
"""Kimi Delta Attention base implementation."""

from typing import Any, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from typing_extensions import override


_Config = TypeVar("_Config")
_Key = TypeVar("_Key")
Output: TypeAlias = tuple[jax.Array, jax.Array | None]
Residuals: TypeAlias = None


def _accumulator_dtype(dtype: jax.typing.DTypeLike) -> jnp.dtype:
  dtype = jnp.dtype(dtype)
  return jnp.float64 if dtype == jnp.float64 else jnp.float32


def _check_array_rank(x: jax.Array, rank: int, name: str):
  if x.ndim != rank:
    raise ValueError(f"`{name}` must be rank {rank}, got shape {x.shape}.")


class KimiDeltaAttention(op.Op[Any, Output, Residuals, _Config, _Key]):
  """Kimi Delta Attention.

  This XLA implementation evaluates the recurrent KDA update:

    S' = S * exp(g_t)
    residual = v_t - k_t^T @ S'
    S = S' + beta_t * k_t outer residual
    o_t = scale * q_t^T @ S

  It is intended as the portable reference implementation for accelerated
  chunk-wise backends.
  """

  supports_symbolic_shapes = False

  @override
  def bind(
      self,
      q: Float[Array, "B T H K"],
      k: Float[Array, "B T H K"],
      v: Float[Array, "B T H V"],
      g: Float[Array, "B T H K"],
      beta: Float[Array, "B T H"],
      *,
      scale: float | None = None,
      initial_state: Float[Array, "B H K V"] | None = None,
      output_final_state: bool = False,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds KDA arguments and validates the dense recurrent contract."""
    _check_array_rank(q, 4, "q")
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]

    if k.shape != q.shape:
      raise ValueError(f"`k` shape {k.shape} must match `q` shape {q.shape}.")
    if g.shape != q.shape:
      raise ValueError(f"`g` shape {g.shape} must match `q` shape {q.shape}.")
    if v.shape != (batch, seq_len, heads, value_dim):
      raise ValueError(
          f"`v` shape {v.shape} must be {(batch, seq_len, heads, value_dim)}."
      )
    if beta.shape != (batch, seq_len, heads):
      raise ValueError(
          f"`beta` shape {beta.shape} must be {(batch, seq_len, heads)}."
      )
    if initial_state is not None:
      expected_state_shape = (batch, heads, key_dim, value_dim)
      if initial_state.shape != expected_state_shape:
        raise ValueError(
            "`initial_state` shape "
            f"{initial_state.shape} must be {expected_state_shape}."
        )

    if scale is None:
      scale = key_dim**-0.5

    return super().bind(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        return_residuals=return_residuals,
    )

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array, "B T H K"],
      k: Float[Array, "B T H K"],
      v: Float[Array, "B T H V"],
      g: Float[Array, "B T H K"],
      beta: Float[Array, "B T H"],
      *,
      scale: float,
      initial_state: Float[Array, "B H K V"] | None,
      output_final_state: bool,
      return_residuals: bool,
      config: _Config,
  ) -> tuple[Output, Residuals]:
    """Computes KDA with a time-axis scan."""
    del config, return_residuals  # Unused.

    batch, _, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    acc_dtype = _accumulator_dtype(q.dtype)
    output_dtype = q.dtype

    q_h = jnp.swapaxes(q, 1, 2).astype(acc_dtype) * scale
    k_h = jnp.swapaxes(k, 1, 2).astype(acc_dtype)
    v_h = jnp.swapaxes(v, 1, 2).astype(acc_dtype)
    g_h = jnp.swapaxes(g, 1, 2).astype(acc_dtype)
    beta_h = jnp.swapaxes(beta, 1, 2).astype(acc_dtype)

    state = jnp.zeros((batch, heads, key_dim, value_dim), dtype=acc_dtype)
    if initial_state is not None:
      state = state + initial_state.astype(acc_dtype)

    def step(state, inputs):
      q_t, k_t, v_t, g_t, beta_t = inputs
      state = state * jnp.exp(g_t)[..., None]
      prediction = jnp.einsum("bhk,bhkv->bhv", k_t, state)
      residual = v_t - prediction
      state = state + jnp.einsum(
          "bhk,bhv->bhkv", beta_t[..., None] * k_t, residual
      )
      out = jnp.einsum("bhk,bhkv->bhv", q_t, state)
      return state, out

    final_state, output_h = jax.lax.scan(
        step,
        state,
        (
            jnp.moveaxis(q_h, 2, 0),
            jnp.moveaxis(k_h, 2, 0),
            jnp.moveaxis(v_h, 2, 0),
            jnp.moveaxis(g_h, 2, 0),
            jnp.moveaxis(beta_h, 2, 0),
        ),
    )
    output = jnp.moveaxis(output_h, 0, 1).astype(output_dtype)

    return (output, final_state if output_final_state else None), None

