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

"""Pallas TPU compute kernel for Raven Gated Slot Attention (GSA)."""

from typing import Any
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

_CHUNK_SIZE = 64
_NUM_SLOTS = 8


def _chunkify(x: jax.Array, b: int, nt: int, bt: int) -> jax.Array:
  # (B, T, H, ...) -> (B, H, NT, BT, ...)
  rest = x.shape[3:]
  x = x.reshape((b, nt, bt, x.shape[2]) + rest)
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _make_raven_stage1_kernel(bt: int) -> Any:
  def _kernel(
      q_ref: Any,
      k_ref: Any,
      s_ref: Any,
      g_ref: Any,
      ydiag_ref: Any,
      sc_ref: Any,
      glast_ref: Any,
  ) -> None:
    q_i = q_ref[0, 0, 0]  # (BT, K)
    k_i = k_ref[0, 0, 0]  # (BT, K)
    s_i = s_ref[0, 0, 0]  # (BT, M)
    g_i = g_ref[0, 0, 0]  # (BT, M)

    row_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
    col_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
    causal = (col_idx <= row_idx).astype(jnp.float32)

    qk = jnp.matmul(q_i, k_i.T, preferred_element_type=jnp.float32) * causal
    diff = jnp.minimum(g_i[:, None, :] - g_i[None, :, :], 0.0)
    decay = jnp.exp(diff)
    y_diag = jnp.sum(qk[:, :, None] * decay * s_i[None, :, :], axis=1)

    g_last = g_i[-1]
    decay_tail = jnp.exp(jnp.minimum(g_last[None, :] - g_i, 0.0))
    state_contrib = jnp.matmul(
        k_i.T, s_i * decay_tail, preferred_element_type=jnp.float32
    )

    ydiag_ref[0, 0, 0] = y_diag.astype(ydiag_ref.dtype)
    sc_ref[0, 0, 0] = state_contrib.astype(sc_ref.dtype)
    glast_ref[0, 0, 0] = g_last.astype(glast_ref.dtype)[None, :]

  return _kernel


def _stage1_intra_raven_pallas(
    q_c: jax.Array, k_c: jax.Array, s_c: jax.Array, g_c: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
  b, h, nt, bt, k = q_c.shape
  m = s_c.shape[-1]
  kernel = _make_raven_stage1_kernel(bt)
  grid = (b, h, nt)
  bspec_k = pl.BlockSpec((1, 1, 1, bt, k), lambda i, j, n: (i, j, n, 0, 0))
  bspec_m = pl.BlockSpec((1, 1, 1, bt, m), lambda i, j, n: (i, j, n, 0, 0))
  bspec_sc = pl.BlockSpec((1, 1, 1, k, m), lambda i, j, n: (i, j, n, 0, 0))
  bspec_glast = pl.BlockSpec((1, 1, 1, 1, m), lambda i, j, n: (i, j, n, 0, 0))

  interpret = jax.default_backend() == "cpu"
  return pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[bspec_k, bspec_k, bspec_m, bspec_m],
      out_specs=[bspec_m, bspec_sc, bspec_glast],
      out_shape=[
          jax.ShapeDtypeStruct((b, h, nt, bt, m), q_c.dtype),
          jax.ShapeDtypeStruct((b, h, nt, k, m), q_c.dtype),
          jax.ShapeDtypeStruct((b, h, nt, 1, m), g_c.dtype),
      ],
      interpret=interpret,
  )(q_c, k_c, s_c, g_c)


def raven_pallas_stage1_fwd(
    q: jax.Array,
    k: jax.Array,
    s: jax.Array,
    g: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
  """Computes Raven Stage 1 forward pass using Pallas TPU intra-chunk kernel."""
  b_dim, t, h, k_dim = q.shape
  m_dim = s.shape[-1]
  bt = chunk_size
  nt = t // bt

  q_c = _chunkify(q, b_dim, nt, bt)
  k_c = _chunkify(k, b_dim, nt, bt)
  s_c = _chunkify(s, b_dim, nt, bt)
  g_c = jnp.cumsum(_chunkify(g, b_dim, nt, bt), axis=-2)

  y_diag, state_contrib, g_last_block = _stage1_intra_raven_pallas(
      q_c, k_c, s_c, g_c
  )
  g_last = g_last_block[..., 0, :]

  def move_nt_front(x: jax.Array) -> jax.Array:
    return jnp.moveaxis(x, 2, 0)

  q_s, g_s, y_diag_s, sc_s, glast_s = map(
      move_nt_front, (q_c, g_c, y_diag, state_contrib, g_last)
  )

  def scan_body(s_state, inputs):
    q_i, g_i, y_diag_i, sc_i, glast_i = inputs
    y_off = jnp.matmul(
        q_i, s_state, preferred_element_type=jnp.float32
    ) * jnp.exp(g_i)
    y = y_diag_i + y_off
    s_new = s_state * jnp.exp(glast_i)[None, :] + sc_i
    return s_new.astype(s_state.dtype), y.astype(y_diag_i.dtype)

  scan_body_batched = jax.vmap(jax.vmap(scan_body))

  if initial_state is None:
    s0 = jnp.zeros((b_dim, h, k_dim, m_dim), dtype=jnp.float32)
  else:
    s0 = initial_state.astype(jnp.float32)

  s_final, o_stacked = jax.lax.scan(
      scan_body_batched, s0, (q_s, g_s, y_diag_s, sc_s, glast_s)
  )
  o = jnp.moveaxis(o_stacked, 0, 2)
  o = jnp.transpose(o, (0, 2, 3, 1, 4)).reshape(b_dim, t, h, m_dim)
  return o, (s_final if output_final_state else None)
