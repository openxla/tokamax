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

"""Pure JAX reference implementation of Raven Gated Slot Attention (GSA)."""

from typing import Any
import jax
import jax.numpy as jnp

_CHUNK_SIZE = 64


def _chunkify(
    x: jax.Array, batch_size: int, num_chunks: int, chunk_size: int
) -> jax.Array:
  # (batch_size, seq_len, num_heads, ...) -> (batch_size, num_heads, num_chunks, chunk_size, ...)
  rest = x.shape[3:]
  x = x.reshape((batch_size, num_chunks, chunk_size, x.shape[2]) + rest)
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _stage1_intra_one_plain(
    q_i: jax.Array, k_i: jax.Array, s_i: jax.Array, g_i: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
  chunk_size = q_i.shape[0]
  mask_upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=1)
  qk = jnp.where(
      mask_upper,
      0.0,
      jnp.matmul(q_i, k_i.T, precision=jax.lax.Precision.HIGHEST),
  )
  diff = jnp.minimum(g_i[:, None, :] - g_i[None, :, :], 0.0)
  decay = jnp.exp(diff)
  y_diag = jnp.sum(qk[:, :, None] * decay * s_i[None, :, :], axis=1)
  g_last = g_i[-1]
  decay_tail = jnp.exp(jnp.minimum(g_last[None, :] - g_i, 0.0))
  state_contrib = jnp.matmul(
      k_i.T, s_i * decay_tail, precision=jax.lax.Precision.HIGHEST
  )
  return y_diag, state_contrib, g_last[None, :]


def raven_stage1_reference(
    q: jax.Array,
    k: jax.Array,
    s: jax.Array,
    g: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
  """Stage 1 reference: decay on slot output (M) axis."""
  batch_size, seq_len, num_heads, key_dim = q.shape
  num_slots = s.shape[-1]
  num_chunks = seq_len // chunk_size

  q_c = _chunkify(q, batch_size, num_chunks, chunk_size)
  k_c = _chunkify(k, batch_size, num_chunks, chunk_size)
  s_c = _chunkify(s, batch_size, num_chunks, chunk_size)
  g_c = jnp.cumsum(_chunkify(g, batch_size, num_chunks, chunk_size), axis=-2)

  vmap_intra = jax.vmap(jax.vmap(jax.vmap(_stage1_intra_one_plain)))
  y_diag, state_contrib, g_last_block = vmap_intra(q_c, k_c, s_c, g_c)
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
    s0 = jnp.zeros((batch_size, num_heads, key_dim, num_slots), dtype=jnp.float32)
  else:
    s0 = initial_state.astype(jnp.float32)

  s_final, o_stacked = jax.lax.scan(
      scan_body_batched, s0, (q_s, g_s, y_diag_s, sc_s, glast_s)
  )
  o = jnp.moveaxis(o_stacked, 0, 2)
  o = jnp.transpose(o, (0, 2, 3, 1, 4)).reshape(batch_size, seq_len, num_heads, num_slots)
  return o, (s_final if output_final_state else None)
