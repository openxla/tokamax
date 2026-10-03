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

"""Pallas TPU compute kernel for Gated Delta Net 2 (GDN-2)."""

from typing import Any
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

_CHUNK_SIZE = 64


def _chunkify(x: jax.Array, b: int, nt: int, bt: int) -> jax.Array:
  # (B, T, H, ...) -> (B, H, NT, BT, ...)
  rest = x.shape[3:]
  x = x.reshape((b, nt, bt, x.shape[2]) + rest)
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _make_gdn2_intra_chunk_kernel(bt: int) -> Any:
  def _kernel(
      q_ref: Any,
      k_ref: Any,
      v_ref: Any,
      g_ref: Any,
      b_ref: Any,
      w_ref: Any,
      wwy_ref: Any,
      u_ref: Any,
      aqk_ref: Any,
  ) -> None:
    q_i = q_ref[0, 0, 0]  # (BT, K)
    k_i = k_ref[0, 0, 0]  # (BT, K)
    v_i = v_ref[0, 0, 0]  # (BT, V)
    g_i = g_ref[0, 0, 0]  # (BT, K)
    b_i = b_ref[0, 0, 0]  # (BT, K) -- erase gate
    w_i = w_ref[0, 0, 0]  # (BT, V) -- write gate

    row_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
    col_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
    strictly_lower = (col_idx < row_idx).astype(jnp.float32)
    causal_keep = (col_idx <= row_idx).astype(jnp.float32)

    bk = b_i * k_i
    decay = jnp.exp(jnp.minimum(g_i[:, None, :] - g_i[None, :, :], 0.0))
    t_lower = -jnp.sum(bk[:, None, :] * k_i[None, :, :] * decay, axis=-1) * strictly_lower

    # Schulz polynomial doubling
    eye = jnp.eye(bt, dtype=t_lower.dtype)
    p_mat = eye + t_lower
    curr = t_lower
    for _ in range(1, 6):
      curr = jnp.matmul(curr, curr, preferred_element_type=jnp.float32)
      p_mat = jnp.matmul(p_mat, eye + curr, preferred_element_type=jnp.float32)
    a_mat = p_mat

    # Concatenated projection fusion (weight-stationary on TPU MXU)
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
    aqk = jnp.sum(qg * k_i[None, :, :], axis=-1) * causal_keep

    wwy_ref[0, 0, 0] = wwy_i.astype(wwy_ref.dtype)
    u_ref[0, 0, 0] = u_i.astype(u_ref.dtype)
    aqk_ref[0, 0, 0] = aqk.astype(aqk_ref.dtype)

  return _kernel


def _intra_chunk_gdn2_pallas(
    q_c: jax.Array,
    k_c: jax.Array,
    v_c: jax.Array,
    g_c: jax.Array,
    b_c: jax.Array,
    w_c: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  b, h, nt, bt, k = q_c.shape
  v_dim = v_c.shape[-1]
  kernel = _make_gdn2_intra_chunk_kernel(bt)
  grid = (b, h, nt)
  bspec_k = pl.BlockSpec((1, 1, 1, bt, k), lambda i, j, n: (i, j, n, 0, 0))
  bspec_v = pl.BlockSpec((1, 1, 1, bt, v_dim), lambda i, j, n: (i, j, n, 0, 0))
  bspec_qq = pl.BlockSpec((1, 1, 1, bt, bt), lambda i, j, n: (i, j, n, 0, 0))

  interpret = jax.default_backend() == "cpu"
  return pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[bspec_k, bspec_k, bspec_v, bspec_k, bspec_k, bspec_v],
      out_specs=[bspec_k, bspec_v, bspec_qq],
      out_shape=[
          jax.ShapeDtypeStruct((b, h, nt, bt, k), q_c.dtype),
          jax.ShapeDtypeStruct((b, h, nt, bt, v_dim), v_c.dtype),
          jax.ShapeDtypeStruct((b, h, nt, bt, bt), q_c.dtype),
      ],
      interpret=interpret,
  )(q_c, k_c, v_c, g_c, b_c, w_c)


def gdn2_pallas_fwd(
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
  """Computes GDN-2 forward pass using Pallas TPU intra-chunk kernel."""
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

  wwy, u, aqk = _intra_chunk_gdn2_pallas(q_c, k_c, v_c, g_c, b_c, w_c)

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
