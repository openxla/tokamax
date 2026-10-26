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

import dataclasses
import math
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from tokamax._src.ops import op
from typing_extensions import override

Output: TypeAlias = tuple[jax.Array, jax.Array | None]
Residuals: TypeAlias = Any

_CHUNK_SIZE = 64


def _chunkify(
    x: jax.Array, batch_size: int, num_chunks: int, chunk_size: int
) -> jax.Array:
  # (batch_size, seq_len, num_heads, ...) -> (batch_size, num_heads, num_chunks, chunk_size, ...)
  rest = x.shape[3:]
  x = x.reshape((batch_size, num_chunks, chunk_size, x.shape[2]) + rest)
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _move_nt_front(x: jax.Array) -> jax.Array:
  return jnp.moveaxis(x, 2, 0)


def _mm(x: jax.Array, y: jax.Array) -> jax.Array:
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
  bt = q_i.shape[0]
  row_idx = jnp.arange(bt)[:, None]
  col_idx = jnp.arange(bt)[None, :]
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

  eye = jnp.eye(bt, dtype=jnp.float32)
  a_ab_f32 = a_ab.astype(jnp.float32)
  p = eye + a_ab_f32
  curr = a_ab_f32
  num_doubling_steps = max(0, math.ceil(math.log2(bt)) - 1)
  for _ in range(num_doubling_steps):
    curr = _mm(curr, curr)
    p = _mm(p, eye + curr)
  a_ab_inv = p

  v_dim = v_i.shape[-1]
  ak_v = _mm(a_ak, v_i)
  decayed_alpha = jnp.exp(jnp.minimum(gk_excl, 0.0)) * alpha_i
  merged_rhs = jnp.concat([ak_v, decayed_alpha], axis=-1)
  merged_uw = _mm(a_ab_inv, merged_rhs)
  u_i = merged_uw[:, :v_dim]
  w_i = merged_uw[:, v_dim:]

  return (
      aqk.astype(q_i.dtype),
      aqb.astype(q_i.dtype),
      u_i.astype(q_i.dtype),
      w_i.astype(q_i.dtype),
  )


def rwkv7_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    gk: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
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

  @override
  def _fwd(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      alpha: jax.Array,
      beta: jax.Array,
      gk: jax.Array,
      *,
      chunk_size: int = _CHUNK_SIZE,
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
    num_chunks = seq_len // chunk_size
    if scale is None:
      scale = key_dim**-0.5

    q_c = _chunkify(q, batch_size, num_chunks, chunk_size) * scale
    k_c = _chunkify(k, batch_size, num_chunks, chunk_size)
    v_c = _chunkify(v, batch_size, num_chunks, chunk_size)
    alpha_c = _chunkify(alpha, batch_size, num_chunks, chunk_size)
    beta_c = _chunkify(beta, batch_size, num_chunks, chunk_size)
    gk_c = _chunkify(gk, batch_size, num_chunks, chunk_size)
    gk_cs_c = jnp.cumsum(gk_c, axis=-2)

    vmap_intra = jax.vmap(jax.vmap(jax.vmap(_intra_chunk_one_plain)))
    aqk, aqb, u, w = vmap_intra(q_c, k_c, v_c, alpha_c, beta_c, gk_c, gk_cs_c)

    q_s, k_s, v_s, beta_s, gk_cs_s, aqk_s, aqb_s, u_s, w_s = map(
        _move_nt_front, (q_c, k_c, v_c, beta_c, gk_cs_c, aqk, aqb, u, w)
    )

    scan_body_batched = jax.vmap(jax.vmap(_scan_body))

    if initial_state is None:
      s0 = jnp.zeros(
          (batch_size, num_heads, key_dim, val_dim), dtype=jnp.float32
      )
    else:
      s0 = initial_state.astype(jnp.float32)

    s_final, o_stacked = jax.lax.scan(
        scan_body_batched,
        s0,
        (q_s, k_s, v_s, beta_s, gk_cs_s, aqk_s, aqb_s, u_s, w_s),
    )
    o = jnp.moveaxis(o_stacked, 0, 2)
    o = jnp.transpose(o, (0, 2, 3, 1, 4)).reshape(
        batch_size, seq_len, num_heads, val_dim
    )
    final_state = s_final if output_final_state else None
    return (o, final_state), None
