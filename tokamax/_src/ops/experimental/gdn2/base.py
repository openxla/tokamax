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

# Dimensions.
#   B   batch size
#   T   sequence length
#   H   number of attention heads
#   K   key head dimension
#   V   value head dimension

import dataclasses
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import (  # pylint: disable=g-multiple-import,g-importing-member
    Array,
    Float,
)
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from typing_extensions import override

Output: TypeAlias = tuple[
    Float[Array, "B T H V"], Float[Array, "B H K V"] | None
]
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
  # Pin HIGHEST precision so TPU does not lower f32 reference matmuls to bf16.
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
    q_h: jax.Array,
    k_h: jax.Array,
    v_h: jax.Array,
    g_h: jax.Array,
    b_h: jax.Array,
    w_h: jax.Array,
    s0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Scans GDN-2 token-by-token recurrence across one head."""
  s_final, out_seq = jax.lax.scan(
      _step_token, s0, (q_h, k_h, v_h, g_h, b_h, w_h)
  )
  return out_seq, s_final


@jaxtyping.jaxtyped
def gdn2_reference(
    q: Float[Array, "B T H K"],
    k: Float[Array, "B T H K"],
    v: Float[Array, "B T H V"],
    g: Float[Array, "B T H K"],
    b: Float[Array, "B T H K"],
    w: Float[Array, "B T H V"],
    *,
    chunk_size: int = 64,
    scale: float | None = None,
    initial_state: Float[Array, "B H K V"] | None = None,
    output_final_state: bool = False,
) -> Output:
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

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array, "B T H K"],
      k: Float[Array, "B T H K"],
      v: Float[Array, "B T H V"],
      g: Float[Array, "B T H K"],
      b: Float[Array, "B T H K"],
      w: Float[Array, "B T H V"],
      *,
      chunk_size: int = 64,
      scale: float | None = None,
      initial_state: Float[Array, "B H K V"] | None = None,
      output_final_state: bool = False,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[Output, None]:
    del return_residuals, config
    B, T, H, K = q.shape
    _, _, _, V = v.shape
    if chunk_size <= 0 or T % chunk_size != 0:
      raise ValueError(
          f"seq_len ({T}) must be divisible by chunk_size ({chunk_size})."
      )
    if scale is None:
      scale = K**-0.5

    q_bh = jnp.transpose(q, (0, 2, 1, 3)) * scale
    k_bh = jnp.transpose(k, (0, 2, 1, 3))
    v_bh = jnp.transpose(v, (0, 2, 1, 3))
    g_bh = jnp.transpose(g, (0, 2, 1, 3))
    b_bh = jnp.transpose(b, (0, 2, 1, 3))
    w_bh = jnp.transpose(w, (0, 2, 1, 3))

    if initial_state is None:
      s0 = jnp.zeros((B, H, K, V), dtype=jnp.float32)
    else:
      s0 = initial_state.astype(jnp.float32)

    vmap_scan = jax.vmap(jax.vmap(_scan_head))
    out_bh, s_final = vmap_scan(q_bh, k_bh, v_bh, g_bh, b_bh, w_bh, s0)
    out = jnp.transpose(out_bh, (0, 2, 1, 3)).astype(q.dtype)
    final_state = s_final.astype(q.dtype) if output_final_state else None
    return (out, final_state), None
