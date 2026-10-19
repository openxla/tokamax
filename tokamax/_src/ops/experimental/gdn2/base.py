# Copyright 2026 Google LLC. All Rights Reserved.
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

"""Base operator class for Gated Delta Net 2 (GDN-2)."""

import dataclasses
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from tokamax._src.ops import op
from typing_extensions import override

Output: TypeAlias = tuple[jax.Array, jax.Array | None]
Residuals: TypeAlias = Any


def _step_token(
    state: jax.Array,
    inputs: tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
    ],
) -> tuple[jax.Array, jax.Array]:
  """Single token recurrence step for GDN-2."""
  q_t, k_t, v_t, g_t, b_t, w_t = inputs
  state_decay = state * jnp.exp(g_t)[:, None]
  beta = b_t * k_t
  v_pred = jnp.matmul(
      beta,
      state_decay,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  v_new = (w_t * v_t) - v_pred
  state_new = state_decay + k_t[:, None] * v_new[None, :]
  out_t = jnp.matmul(
      q_t,
      state_new,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return state_new, out_t


def _scan_head(
    q_i: jax.Array,
    k_i: jax.Array,
    v_i: jax.Array,
    g_i: jax.Array,
    b_i: jax.Array,
    w_i: jax.Array,
    s0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Scans GDN-2 token-by-token recurrence across one head."""
  s_final, out_seq = jax.lax.scan(
      _step_token, s0, (q_i, k_i, v_i, g_i, b_i, w_i)
  )
  return out_seq, s_final


def gdn2_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    *,
    chunk_size: int = 64,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
  """Token-by-token recurrence reference implementation of GDN-2."""
  (out, final_state), _ = GatedDeltaNet2()._fwd(
      q,
      k,
      v,
      g,
      b,
      w,
      chunk_size=chunk_size,
      scale=scale,
      initial_state=initial_state,
      output_final_state=output_final_state,
  )
  return out, final_state


@dataclasses.dataclass(frozen=True, kw_only=True)
class GatedDeltaNet2[C](op.Op[Any, Output, Residuals, C, None]):
  """Tokamax base operator for Gated Delta Net 2 (GDN-2).

  Evaluates the linear recurrent delta-rule state update with input-dependent
  erase and write gating using an explicit token-by-token recurrence.
  """

  supports_symbolic_shapes = False

  @override
  def _fwd(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g: jax.Array,
      b: jax.Array,
      w: jax.Array,
      *,
      chunk_size: int = 64,
      scale: float | None = None,
      initial_state: jax.Array | None = None,
      output_final_state: bool = False,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[Output, None]:
    del return_residuals, config
    batch_size, seq_len, num_heads, key_dim = q.shape
    if chunk_size <= 0 or seq_len % chunk_size != 0:
      raise ValueError(
          f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})."
      )
    val_dim = v.shape[-1]
    if scale is None:
      scale = key_dim**-0.5

    q_bh = jnp.transpose(q, (0, 2, 1, 3)) * scale
    k_bh = jnp.transpose(k, (0, 2, 1, 3))
    v_bh = jnp.transpose(v, (0, 2, 1, 3))
    g_bh = jnp.transpose(g, (0, 2, 1, 3))
    b_bh = jnp.transpose(b, (0, 2, 1, 3))
    w_bh = jnp.transpose(w, (0, 2, 1, 3))

    if initial_state is None:
      s0 = jnp.zeros(
          (batch_size, num_heads, key_dim, val_dim), dtype=jnp.float32
      )
    else:
      s0 = initial_state.astype(jnp.float32)

    vmap_scan = jax.vmap(jax.vmap(_scan_head))
    out_bh, s_final = vmap_scan(q_bh, k_bh, v_bh, g_bh, b_bh, w_bh, s0)
    out = jnp.transpose(out_bh, (0, 2, 1, 3)).astype(q.dtype)
    final_state = s_final.astype(q.dtype) if output_final_state else None
    return (out, final_state), None
