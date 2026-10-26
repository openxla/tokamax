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

"""Base operator and JAX reference for Raven Gated Slot Attention (GSA)."""

import dataclasses
from typing import Any, TypeAlias, TypeVar
import jax
import jax.numpy as jnp
from tokamax._src.ops import op
from typing_extensions import override

_CHUNK_SIZE = 64
_Config = TypeVar("_Config")
State: TypeAlias = jax.Array | tuple[jax.Array | None, jax.Array | None] | None
Output: TypeAlias = tuple[jax.Array, State]
Residuals: TypeAlias = Any


def _validate_chunk_size(seq_len: int, chunk_size: int) -> None:
  """Validates that ``chunk_size`` is positive and divides ``seq_len``."""
  if chunk_size <= 0:
    raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
  if seq_len % chunk_size != 0:
    raise ValueError(
        f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})."
    )


def _chunkify(
    x: jax.Array, batch_size: int, num_chunks: int, chunk_size: int
) -> jax.Array:
  # (batch_size, seq_len, num_heads, ...) ->
  # (batch_size, num_heads, num_chunks, chunk_size, ...)
  rest = x.shape[3:]
  x = x.reshape((batch_size, num_chunks, chunk_size, x.shape[2]) + rest)
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _move_nt_front(x: jax.Array) -> jax.Array:
  return jnp.moveaxis(x, 2, 0)


def _stage1_scan_step(
    s_state: jax.Array,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
  q_i, g_i, y_diag_i, state_contrib_i, g_last_i = inputs
  y_off = jnp.matmul(
      q_i,
      s_state,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  ) * jnp.exp(g_i)
  y = y_diag_i + y_off
  s_new = s_state * jnp.exp(g_last_i)[None, :] + state_contrib_i
  return s_new.astype(s_state.dtype), y.astype(y_diag_i.dtype)


def _stage1_intra_one_plain(
    q_i: jax.Array, k_i: jax.Array, s_i: jax.Array, g_i: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
  chunk_size = q_i.shape[0]
  mask_upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=1)
  qk = jnp.where(
      mask_upper,
      0.0,
      jnp.matmul(
          q_i,
          k_i.T,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      ),
  )
  diff = jnp.minimum(g_i[:, None, :] - g_i[None, :, :], 0.0)
  decay = jnp.exp(diff)
  y_diag = jnp.sum(qk[:, :, None] * decay * s_i[None, :, :], axis=1)
  g_last = g_i[-1]
  decay_tail = jnp.exp(jnp.minimum(g_last[None, :] - g_i, 0.0))
  state_contrib = jnp.matmul(
      k_i.T,
      s_i * decay_tail,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return y_diag, state_contrib, g_last[None, :]


def raven_stage1_reference(
    q: jax.Array,
    k: jax.Array,
    s: jax.Array,
    g: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
  """Stage 1 reference: decay on slot output (M) axis."""
  batch_size, seq_len, num_heads, key_dim = q.shape
  _validate_chunk_size(seq_len, chunk_size)
  num_slots = s.shape[-1]
  num_chunks = seq_len // chunk_size

  if scale is not None:
    q_scaled = (q.astype(jnp.float32) * scale).astype(q.dtype)
  else:
    q_scaled = q

  q_c = _chunkify(
      q_scaled.astype(jnp.float32), batch_size, num_chunks, chunk_size
  )
  k_c = _chunkify(k.astype(jnp.float32), batch_size, num_chunks, chunk_size)
  s_c = _chunkify(s.astype(jnp.float32), batch_size, num_chunks, chunk_size)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), batch_size, num_chunks, chunk_size),
      axis=-2,
  )

  vmap_intra = jax.vmap(jax.vmap(jax.vmap(_stage1_intra_one_plain)))
  y_diag, state_contrib, g_last_block = vmap_intra(q_c, k_c, s_c, g_c)
  g_last = g_last_block[..., 0, :]

  q_s, g_s, y_diag_s, state_contrib_s, g_last_s = map(
      _move_nt_front, (q_c, g_c, y_diag, state_contrib, g_last)
  )

  scan_body_batched = jax.vmap(jax.vmap(_stage1_scan_step))

  if initial_state is None:
    initial_recurrent_state = jnp.zeros(
        (batch_size, num_heads, key_dim, num_slots), dtype=jnp.float32
    )
  else:
    initial_recurrent_state = initial_state.astype(jnp.float32)

  s_final, o_stacked = jax.lax.scan(
      scan_body_batched,
      initial_recurrent_state,
      (q_s, g_s, y_diag_s, state_contrib_s, g_last_s),
  )
  o = jnp.moveaxis(o_stacked, 0, 2)
  o = (
      jnp.transpose(o, (0, 2, 3, 1, 4))
      .reshape(batch_size, seq_len, num_heads, num_slots)
      .astype(s.dtype)
  )
  final_state = s_final.astype(q.dtype) if output_final_state else None
  return o, final_state


def _stage2_scan_step(
    s_state: jax.Array,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
  qv_i, g_i, y_diag_i, state_contrib_i, g_last_i = inputs
  y_off = jnp.matmul(
      qv_i * jnp.exp(g_i),
      s_state,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  y = y_diag_i + y_off
  s_new = s_state * jnp.exp(g_last_i)[:, None] + state_contrib_i
  return s_new.astype(s_state.dtype), y.astype(y_diag_i.dtype)


def _stage2_intra_one_plain(
    qv_i: jax.Array, s_i: jax.Array, v_i: jax.Array, g_i: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
  chunk_size = qv_i.shape[0]
  mask_upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=1)
  diff = jnp.minimum(g_i[:, None, :] - g_i[None, :, :], 0.0)
  decay = jnp.exp(diff)
  aqk_raw = jnp.einsum(
      "tm,nm,tnm->tn",
      qv_i,
      s_i,
      decay,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  aqk = jnp.where(mask_upper, 0.0, aqk_raw)
  y_diag = jnp.matmul(
      aqk,
      v_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  g_last = g_i[-1]
  decay_tail = jnp.exp(jnp.minimum(g_last[None, :] - g_i, 0.0))
  state_contrib = jnp.matmul(
      (s_i * decay_tail).T,
      v_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return y_diag, state_contrib, g_last[None, :]


def raven_stage2_reference(
    qv: jax.Array,
    s: jax.Array,
    v: jax.Array,
    g: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
  """Stage 2 reference: decay on slot input (M) axis, output on value (V) axis."""
  batch_size, seq_len, num_heads, num_slots = qv.shape
  _validate_chunk_size(seq_len, chunk_size)
  val_dim = v.shape[-1]
  num_chunks = seq_len // chunk_size

  qv_c = _chunkify(qv.astype(jnp.float32), batch_size, num_chunks, chunk_size)
  s_c = _chunkify(s.astype(jnp.float32), batch_size, num_chunks, chunk_size)
  v_c = _chunkify(v.astype(jnp.float32), batch_size, num_chunks, chunk_size)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), batch_size, num_chunks, chunk_size),
      axis=-2,
  )

  vmap_intra = jax.vmap(jax.vmap(jax.vmap(_stage2_intra_one_plain)))
  y_diag, state_contrib, g_last_block = vmap_intra(qv_c, s_c, v_c, g_c)
  g_last = g_last_block[..., 0, :]

  qv_s, g_s, y_diag_s, state_contrib_s, g_last_s = map(
      _move_nt_front, (qv_c, g_c, y_diag, state_contrib, g_last)
  )

  scan_body_batched = jax.vmap(jax.vmap(_stage2_scan_step))

  if initial_state is None:
    initial_recurrent_state = jnp.zeros(
        (batch_size, num_heads, num_slots, val_dim), dtype=jnp.float32
    )
  else:
    initial_recurrent_state = initial_state.astype(jnp.float32)

  s_final, o_stacked = jax.lax.scan(
      scan_body_batched,
      initial_recurrent_state,
      (qv_s, g_s, y_diag_s, state_contrib_s, g_last_s),
  )
  o = jnp.moveaxis(o_stacked, 0, 2)
  o = (
      jnp.transpose(o, (0, 2, 3, 1, 4))
      .reshape(batch_size, seq_len, num_heads, val_dim)
      .astype(v.dtype)
  )
  final_state = s_final.astype(v.dtype) if output_final_state else None
  return o, final_state


def raven_gsa_reference(
    q: jax.Array,
    k: jax.Array,
    s: jax.Array,
    g: jax.Array,
    v: jax.Array | None = None,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: State = None,
    output_final_state: bool = False,
) -> Output:
  """Pure JAX reference for Raven Gated Slot Attention (Stage 1 or full 2-stage)."""
  if v is None:
    if isinstance(initial_state, tuple):
      s1_init = initial_state[0]
    else:
      s1_init = initial_state
    return raven_stage1_reference(
        q,
        k,
        s,
        g,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=s1_init,
        output_final_state=output_final_state,
    )

  if isinstance(initial_state, tuple):
    s1_init, s2_init = initial_state
  else:
    s1_init, s2_init = initial_state, None

  ok, s1_final = raven_stage1_reference(
      q,
      k,
      s,
      g,
      chunk_size=chunk_size,
      scale=scale,
      initial_state=s1_init,
      output_final_state=output_final_state,
  )
  qv = jax.nn.softmax(ok.astype(jnp.float32), axis=-1).astype(v.dtype)
  out, s2_final = raven_stage2_reference(
      qv,
      s,
      v,
      g,
      chunk_size=chunk_size,
      initial_state=s2_init,
      output_final_state=output_final_state,
  )
  final_state = (s1_final, s2_final) if output_final_state else None
  return out, final_state


@dataclasses.dataclass(frozen=True, kw_only=True)
class RavenGSA[C](op.Op[Any, Output, Residuals, C, None]):
  """Tokamax base operator for Raven Gated Slot Attention (GSA).

  Supports both standalone Stage 1 slot-logit computation (when ``v is None``)
  and full two-stage Gated Slot Attention (Stage 1 -> slot softmax -> Stage 2
  when ``v`` is provided).
  """

  supports_symbolic_shapes = False

  @override
  def _fwd(
      self,
      q: jax.Array,
      k: jax.Array,
      s: jax.Array,
      g: jax.Array,
      v: jax.Array | None = None,
      *,
      chunk_size: int = _CHUNK_SIZE,
      scale: float | None = None,
      initial_state: State = None,
      output_final_state: bool = False,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[Output, None]:
    del config, return_residuals
    out = raven_gsa_reference(
        q,
        k,
        s,
        g,
        v,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
    return out, None
