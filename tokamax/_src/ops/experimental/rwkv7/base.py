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
"""Base operator class for RWKV-7 (DPLR Delta Rule)."""

# Dimensions.
#   B      batch size
#   T      sequence length (nt * tt)
#   H      number of attention heads
#   K      key head dimension
#   V      value head dimension
#   tt     sequence chunk tile extent (chunk_size)
#   nt     grid extent over T: nt = T // tt

import dataclasses
import math
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from typing_extensions import override

# pylint: disable=invalid-name

Output: TypeAlias = tuple[
    Float[Array, "B T H V"], Float[Array, "B H K V"] | None
]
Residuals: TypeAlias = Any

_CHUNK_SIZE = 64


def _chunkify(x: jax.Array, B: int, nt: int, tt: int) -> jax.Array:
  # [B, T, H, ...] -> [B, H, nt, tt, ...]
  _, _, H, *rest = x.shape
  x = x.reshape((B, nt, tt, H, *rest))
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _move_nt_front(x: jax.Array) -> jax.Array:
  return jnp.moveaxis(x, 2, 0)


def _mm(x: jax.Array, y: jax.Array) -> jax.Array:
  # Pin HIGHEST precision on f32 contractions so TPU MXU does not downcast
  # reference matmuls to bf16.
  return jnp.matmul(
      x,
      y,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )


def _scan_body(
    s: jax.Array, chunk_inputs: tuple[Any, ...]
) -> tuple[jax.Array, jax.Array]:
  q_i, k_i, v_i, beta_i, gk_cs_i, aqk_i, aqb_i, u_i, w_i = chunk_inputs
  v2_i = u_i + _mm(w_i, s)
  o_1 = _mm(aqk_i, v_i)
  o_2 = _mm(aqb_i, v2_i)
  o_3 = _mm(q_i * jnp.exp(jnp.minimum(gk_cs_i, 0.0)), s)
  o_i = o_1 + o_2 + o_3
  g_last = gk_cs_i[-1]
  decay_last = jnp.exp(jnp.minimum(g_last[None, :] - gk_cs_i, 0.0))
  s_new = (
      s * jnp.exp(jnp.minimum(g_last, 0.0))[:, None]
      + _mm((k_i * decay_last).T, v_i)
      + _mm((beta_i * decay_last).T, v2_i)
  )
  return s_new.astype(s.dtype), o_i.astype(u_i.dtype)


def _intra_chunk_one_plain(
    q_i: jax.Array,
    k_i: jax.Array,
    v_i: jax.Array,
    alpha_i: jax.Array,
    beta_i: jax.Array,
    gk_i: jax.Array,
    gk_cs_i: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  tt, _ = q_i.shape
  _, V = v_i.shape
  row_idx = jnp.arange(tt)[:, None]
  col_idx = jnp.arange(tt)[None, :]
  incl_causal = (col_idx <= row_idx).astype(jnp.float32)
  strict_causal = (col_idx < row_idx).astype(jnp.float32)

  decay_incl = jnp.exp(
      jnp.minimum(gk_cs_i[:, None, :] - gk_cs_i[None, :, :], 0.0)
  )
  aqk = (
      jnp.sum(q_i[:, None, :] * k_i[None, :, :] * decay_incl, axis=-1)
      * incl_causal
  )
  aqb = (
      jnp.sum(q_i[:, None, :] * beta_i[None, :, :] * decay_incl, axis=-1)
      * incl_causal
  )

  gk_excl = gk_cs_i - gk_i
  decay_excl = jnp.exp(
      jnp.minimum(gk_excl[:, None, :] - gk_cs_i[None, :, :], 0.0)
  )
  a_ab = (
      jnp.sum(alpha_i[:, None, :] * beta_i[None, :, :] * decay_excl, axis=-1)
      * strict_causal
  )
  a_ak = (
      jnp.sum(alpha_i[:, None, :] * k_i[None, :, :] * decay_excl, axis=-1)
      * strict_causal
  )

  # Invert (I - A_ab) via Neumann series doubling since A_ab is strictly lower
  # triangular (nilpotent with A_ab^tt = 0).
  eye = jnp.eye(tt, dtype=jnp.float32)
  a_ab_f32 = a_ab.astype(jnp.float32)
  p = eye + a_ab_f32
  curr = a_ab_f32
  num_doubling_steps = max(0, math.ceil(math.log2(tt)) - 1)
  for _ in range(num_doubling_steps):
    curr = _mm(curr, curr)
    p = _mm(p, eye + curr)
  a_ab_inv = p

  ak_v = _mm(a_ak, v_i)
  decayed_alpha = jnp.exp(jnp.minimum(gk_excl, 0.0)) * alpha_i
  merged_rhs = jnp.concat([ak_v, decayed_alpha], axis=-1)
  merged_uw = _mm(a_ab_inv, merged_rhs)
  u_i = merged_uw[:, :V]
  w_i = merged_uw[:, V:]

  return (
      aqk.astype(q_i.dtype),
      aqb.astype(q_i.dtype),
      u_i.astype(q_i.dtype),
      w_i.astype(q_i.dtype),
  )


@jaxtyping.jaxtyped
def rwkv7_reference(
    q: Float[Array, "B T H K"],
    k: Float[Array, "B T H K"],
    v: Float[Array, "B T H V"],
    alpha: Float[Array, "B T H K"],
    beta: Float[Array, "B T H K"],
    gk: Float[Array, "B T H K"],
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: Float[Array, "B H K V"] | None = None,
    output_final_state: bool = False,
) -> tuple[Float[Array, "B T H V"], Float[Array, "B H K V"] | None]:
  """Reference implementation of RWKV-7 DPLR delta rule recurrence."""
  (out, final_state), _ = RWKV7()._fwd(
      q,
      k,
      v,
      alpha,
      beta,
      gk,
      chunk_size=chunk_size,
      scale=scale,
      initial_state=initial_state,
      output_final_state=output_final_state,
  )
  return out, final_state


@dataclasses.dataclass(frozen=True, kw_only=True)
class RWKV7[C](op.Op[Any, Output, Residuals, C, None]):
  """Tokamax base operator for RWKV-7 (DPLR Delta Rule).

  Evaluates Diagonal Plus Low-Rank (DPLR) recurrent state transitions.
  """

  supports_symbolic_shapes = False

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array, "B T H K"],
      k: Float[Array, "B T H K"],
      v: Float[Array, "B T H V"],
      alpha: Float[Array, "B T H K"],
      beta: Float[Array, "B T H K"],
      gk: Float[Array, "B T H K"],
      *,
      chunk_size: int = _CHUNK_SIZE,
      scale: float | None = None,
      initial_state: Float[Array, "B H K V"] | None = None,
      output_final_state: bool = False,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[Output, None]:
    del return_residuals, config
    B, T, H, K = q.shape
    _, _, _, V = v.shape
    tt = chunk_size
    if tt <= 0 or T % tt != 0:
      raise ValueError(
          f"seq_len ({T}) must be divisible by chunk_size ({tt})."
      )
    nt = T // tt
    if scale is None:
      scale = K**-0.5

    q_c = _chunkify(q, B, nt, tt) * scale
    k_c = _chunkify(k, B, nt, tt)
    v_c = _chunkify(v, B, nt, tt)
    alpha_c = _chunkify(alpha, B, nt, tt)
    beta_c = _chunkify(beta, B, nt, tt)
    gk_c = _chunkify(gk, B, nt, tt)
    gk_cs_c = jnp.cumsum(gk_c, axis=-2)

    vmap_intra = jax.vmap(jax.vmap(jax.vmap(_intra_chunk_one_plain)))
    aqk, aqb, u, w = vmap_intra(q_c, k_c, v_c, alpha_c, beta_c, gk_c, gk_cs_c)

    q_s, k_s, v_s, beta_s, gk_cs_s, aqk_s, aqb_s, u_s, w_s = map(
        _move_nt_front, (q_c, k_c, v_c, beta_c, gk_cs_c, aqk, aqb, u, w)
    )

    scan_body_batched = jax.vmap(jax.vmap(_scan_body))

    if initial_state is None:
      s0 = jnp.zeros((B, H, K, V), dtype=jnp.float32)
    else:
      s0 = initial_state.astype(jnp.float32)

    s_final, o_stacked = jax.lax.scan(
        scan_body_batched,
        s0,
        (q_s, k_s, v_s, beta_s, gk_cs_s, aqk_s, aqb_s, u_s, w_s),
    )
    o = jnp.moveaxis(o_stacked, 0, 2)
    o = jnp.transpose(o, (0, 2, 3, 1, 4)).reshape(B, T, H, V)
    final_state = s_final if output_final_state else None
    return (o, final_state), None
