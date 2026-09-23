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

"""Pallas TPU compute kernel for RWKV-7 (DPLR Delta Rule)."""

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


def _make_rwkv7_intra_chunk_kernel(bt: int) -> Any:
  def _kernel(
      q_ref: Any,
      k_ref: Any,
      v_ref: Any,
      alpha_ref: Any,
      beta_ref: Any,
      gk_ref: Any,
      gk_cs_ref: Any,
      aqk_ref: Any,
      aqb_ref: Any,
      u_ref: Any,
      w_ref: Any,
  ) -> None:
    q_i = q_ref[0, 0, 0]  # (BT, K)
    k_i = k_ref[0, 0, 0]  # (BT, K)
    v_i = v_ref[0, 0, 0]  # (BT, V)
    alpha_i = alpha_ref[0, 0, 0]  # (BT, K)
    beta_i = beta_ref[0, 0, 0]  # (BT, K)
    gk_i = gk_ref[0, 0, 0]  # (BT, K)
    gk_cs = gk_cs_ref[0, 0, 0]  # (BT, K)

    row_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
    col_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
    incl_causal = (col_idx <= row_idx).astype(jnp.float32)
    strict_causal = (col_idx < row_idx).astype(jnp.float32)

    decay_incl = jnp.exp(
        jnp.minimum(gk_cs[:, None, :] - gk_cs[None, :, :], 0.0)
    )
    aqk = jnp.sum(q_i[:, None, :] * k_i[None, :, :] * decay_incl, axis=-1) * incl_causal
    aqb = jnp.sum(q_i[:, None, :] * beta_i[None, :, :] * decay_incl, axis=-1) * incl_causal

    gk_excl = gk_cs - gk_i
    decay_excl = jnp.exp(
        jnp.minimum(gk_excl[:, None, :] - gk_cs[None, :, :], 0.0)
    )
    a_ab = jnp.sum(alpha_i[:, None, :] * beta_i[None, :, :] * decay_excl, axis=-1) * strict_causal
    a_ak = jnp.sum(alpha_i[:, None, :] * k_i[None, :, :] * decay_excl, axis=-1) * strict_causal

    # Schulz polynomial doubling
    eye = jnp.eye(bt, dtype=jnp.float32)
    a_ab_f32 = a_ab.astype(jnp.float32)
    p_mat = eye + a_ab_f32
    curr = a_ab_f32
    for _ in range(1, 6):
      curr = jnp.matmul(curr, curr, preferred_element_type=jnp.float32)
      p_mat = jnp.matmul(p_mat, eye + curr, preferred_element_type=jnp.float32)
    a_ab_inv = p_mat

    v_dim = v_i.shape[-1]
    ak_v = jnp.matmul(a_ak, v_i, preferred_element_type=jnp.float32)
    decayed_alpha = jnp.exp(jnp.minimum(gk_excl, 0.0)) * alpha_i
    merged_rhs = jnp.concat([ak_v, decayed_alpha], axis=-1)
    merged_uw = jnp.matmul(a_ab_inv, merged_rhs, preferred_element_type=jnp.float32)
    u_i = merged_uw[:, :v_dim]
    w_i = merged_uw[:, v_dim:]

    aqk_ref[0, 0, 0] = aqk.astype(aqk_ref.dtype)
    aqb_ref[0, 0, 0] = aqb.astype(aqb_ref.dtype)
    u_ref[0, 0, 0] = u_i.astype(u_ref.dtype)
    w_ref[0, 0, 0] = w_i.astype(w_ref.dtype)

  return _kernel


def _intra_chunk_rwkv7_pallas(
    q_c: jax.Array,
    k_c: jax.Array,
    v_c: jax.Array,
    alpha_c: jax.Array,
    beta_c: jax.Array,
    gk_c: jax.Array,
    gk_cs_c: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  b, h, nt, bt, k = q_c.shape
  v_dim = v_c.shape[-1]
  kernel = _make_rwkv7_intra_chunk_kernel(bt)
  grid = (b, h, nt)
  bspec_k = pl.BlockSpec((1, 1, 1, bt, k), lambda i, j, n: (i, j, n, 0, 0))
  bspec_v = pl.BlockSpec((1, 1, 1, bt, v_dim), lambda i, j, n: (i, j, n, 0, 0))
  bspec_qq = pl.BlockSpec((1, 1, 1, bt, bt), lambda i, j, n: (i, j, n, 0, 0))

  interpret = jax.default_backend() == "cpu"
  return pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[bspec_k, bspec_k, bspec_v, bspec_k, bspec_k, bspec_k, bspec_k],
      out_specs=[bspec_qq, bspec_qq, bspec_v, bspec_k],
      out_shape=[
          jax.ShapeDtypeStruct((b, h, nt, bt, bt), q_c.dtype),
          jax.ShapeDtypeStruct((b, h, nt, bt, bt), q_c.dtype),
          jax.ShapeDtypeStruct((b, h, nt, bt, v_dim), v_c.dtype),
          jax.ShapeDtypeStruct((b, h, nt, bt, k), q_c.dtype),
      ],
      interpret=interpret,
  )(q_c, k_c, v_c, alpha_c, beta_c, gk_c, gk_cs_c)


def rwkv7_pallas_fwd(
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
  """Computes RWKV-7 forward pass using Pallas TPU intra-chunk kernel."""
  b_dim, t, h, k_dim = q.shape
  v_dim = v.shape[-1]
  bt = chunk_size
  nt = t // bt
  if scale is None:
    scale = k_dim**-0.5

  q_c = _chunkify(q, b_dim, nt, bt) * scale
  k_c = _chunkify(k, b_dim, nt, bt)
  v_c = _chunkify(v, b_dim, nt, bt)
  alpha_c = _chunkify(alpha, b_dim, nt, bt)
  beta_c = _chunkify(beta, b_dim, nt, bt)
  gk_c = _chunkify(gk, b_dim, nt, bt)
  gk_cs_c = jnp.cumsum(gk_c, axis=-2)

  aqk, aqb, u, w = _intra_chunk_rwkv7_pallas(
      q_c, k_c, v_c, alpha_c, beta_c, gk_c, gk_cs_c
  )

  def move_nt_front(x: jax.Array) -> jax.Array:
    return jnp.moveaxis(x, 2, 0)

  q_s, k_s, v_s, beta_s, gk_cs_s, aqk_s, aqb_s, u_s, w_s = map(
      move_nt_front, (q_c, k_c, v_c, beta_c, gk_cs_c, aqk, aqb, u, w)
  )

  def scan_body(s, chunk_inputs):
    q_i, k_i, v_i, beta_i, gk_cs_i, aqk_i, aqb_i, u_i, w_i = chunk_inputs
    v2_i = u_i + jnp.matmul(w_i, s, preferred_element_type=jnp.float32)
    o_1 = jnp.matmul(aqk_i, v_i, preferred_element_type=jnp.float32)
    o_2 = jnp.matmul(aqb_i, v2_i, preferred_element_type=jnp.float32)
    o_3 = jnp.matmul(
        q_i * jnp.exp(jnp.minimum(gk_cs_i, 0.0)), s, preferred_element_type=jnp.float32
    )
    o_i = o_1 + o_2 + o_3
    g_last = gk_cs_i[-1]
    decay_last = jnp.exp(jnp.minimum(g_last[None, :] - gk_cs_i, 0.0))
    s_new = (
        s * jnp.exp(jnp.minimum(g_last, 0.0))[:, None]
        + jnp.matmul((k_i * decay_last).T, v_i, preferred_element_type=jnp.float32)
        + jnp.matmul((beta_i * decay_last).T, v2_i, preferred_element_type=jnp.float32)
    )
    return s_new.astype(s.dtype), o_i.astype(u_i.dtype)

  scan_body_batched = jax.vmap(jax.vmap(scan_body))

  if initial_state is None:
    s0 = jnp.zeros((b_dim, h, k_dim, v_dim), dtype=jnp.float32)
  else:
    s0 = initial_state.astype(jnp.float32)

  s_final, o_stacked = jax.lax.scan(
      scan_body_batched, s0, (q_s, k_s, v_s, beta_s, gk_cs_s, aqk_s, aqb_s, u_s, w_s)
  )
  o = jnp.moveaxis(o_stacked, 0, 2)
  o = jnp.transpose(o, (0, 2, 3, 1, 4)).reshape(b_dim, t, h, v_dim)
  return o, (s_final if output_final_state else None)
