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

"""Pure JAX reference implementation of Gated Delta Net 2 (GDN-2)."""

from typing import Any
import jax
import jax.numpy as jnp

_CHUNK_SIZE = 64


def _chunkify(x: jax.Array, b: int, nt: int, bt: int) -> jax.Array:
  # (B, T, H, ...) -> (B, H, NT, BT, ...)
  rest = x.shape[3:]
  x = x.reshape((b, nt, bt, x.shape[2]) + rest)
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _intra_chunk_one_plain(
    q_i: jax.Array,
    k_i: jax.Array,
    v_i: jax.Array,
    g_i: jax.Array,
    b_i: jax.Array,
    w_i: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  bt = k_i.shape[0]
  bk = b_i * k_i
  decay = jnp.exp(jnp.minimum(g_i[:, None, :] - g_i[None, :, :], 0.0))
  t_lower = jnp.sum(bk[:, None, :] * k_i[None, :, :] * decay, axis=-1)
  mask_incl_diag = jnp.triu(jnp.ones((bt, bt), dtype=bool), k=0)
  t_lower = -jnp.where(mask_incl_diag, 0.0, t_lower)

  eye = jnp.eye(bt, dtype=jnp.float32)
  t_f32 = t_lower.astype(jnp.float32)
  p = eye + t_f32
  curr = t_f32
  for _ in range(1, 6):
    curr = jnp.matmul(curr, curr, preferred_element_type=jnp.float32)
    p = jnp.matmul(p, eye + curr, preferred_element_type=jnp.float32)
  a_mat = p

  k_g_b = jnp.exp(g_i) * k_i * b_i
  wv = w_i * v_i
  merged_rhs = jnp.concat([k_g_b, wv], axis=-1)
  merged_out = jnp.matmul(a_mat, merged_rhs, preferred_element_type=jnp.float32)
  k_dim = k_i.shape[-1]
  wwy_i = merged_out[:, :k_dim]
  u_i = merged_out[:, k_dim:]

  qg = q_i[:, None, :] * jnp.exp(
      jnp.minimum(g_i[:, None, :] - g_i[None, :, :], 0.0)
  )
  mask_upper_strict = jnp.triu(jnp.ones((bt, bt), dtype=bool), k=1)
  aqk = jnp.sum(qg * k_i[None, :, :], axis=-1)
  aqk = jnp.where(mask_upper_strict, 0.0, aqk)

  return wwy_i.astype(q_i.dtype), u_i.astype(q_i.dtype), aqk.astype(q_i.dtype)


def gdn2_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
  """Reference implementation of GDN-2 chunk recurrence."""
  b_dim, t, h, k_dim = q.shape
  v_dim = v.shape[-1]
  bt = chunk_size
  nt = t // bt
  if scale is None:
    scale = k_dim**-0.5

  q_c = _chunkify(q, b_dim, nt, bt) * scale
  k_c = _chunkify(k, b_dim, nt, bt)
  v_c = _chunkify(v, b_dim, nt, bt)
  g_c = jnp.cumsum(_chunkify(g, b_dim, nt, bt), axis=-2)
  b_c = _chunkify(b, b_dim, nt, bt)
  w_c = _chunkify(w, b_dim, nt, bt)

  vmap_intra = jax.vmap(jax.vmap(jax.vmap(_intra_chunk_one_plain)))
  wwy, u, aqk = vmap_intra(q_c, k_c, v_c, g_c, b_c, w_c)

  def move_nt_front(x: jax.Array) -> jax.Array:
    return jnp.moveaxis(x, 2, 0)

  q_s, k_s, g_s, wwy_s, u_s, aqk_s = map(
      move_nt_front, (q_c, k_c, g_c, wwy, u, aqk)
  )

  def scan_body(s, chunk_inputs):
    q_i, k_i, g_i, wwy_i, u_i, aqk_i = chunk_inputs
    v_i = u_i - jnp.matmul(wwy_i, s, preferred_element_type=jnp.float32)
    o_i = jnp.matmul(
        q_i * jnp.exp(g_i), s, preferred_element_type=jnp.float32
    ) + jnp.matmul(aqk_i, v_i, preferred_element_type=jnp.float32)
    g_last = g_i[-1]
    decayed_k = jnp.exp(g_last[None, :] - g_i) * k_i
    s_update = jnp.matmul(decayed_k.T, v_i, preferred_element_type=jnp.float32)
    s_new = s * jnp.exp(g_last)[:, None] + s_update
    return s_new.astype(s.dtype), o_i.astype(u_i.dtype)

  scan_body_batched = jax.vmap(jax.vmap(scan_body))

  if initial_state is None:
    s0 = jnp.zeros((b_dim, h, k_dim, v_dim), dtype=jnp.float32)
  else:
    s0 = initial_state.astype(jnp.float32)

  s_final, o_stacked = jax.lax.scan(
      scan_body_batched, s0, (q_s, k_s, g_s, wwy_s, u_s, aqk_s)
  )
  o = jnp.moveaxis(o_stacked, 0, 2)
  o = jnp.transpose(o, (0, 2, 3, 1, 4)).reshape(b_dim, t, h, v_dim)
  return o, (s_final if output_final_state else None)
