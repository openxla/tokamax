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

import functools
import os
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Dimensions:
#   B: batch size
#   T: sequence length (seq_len)
#   H: number of attention heads (num_heads)
#   BH: flattened batch * heads (B * H)
#   MB: stream-batching factor per Pallas program instance
#   NT: number of chunks along sequence length (T // BT)
#   BT: chunk size (chunk_size)
#   K: key/query head dimension (key_dim)
#   M: number of memory slots (num_slots)
#   V: value head dimension (val_dim)

_CHUNK_SIZE = 64


def _resolve_interpret(interpret: bool | None = None) -> bool:
  """Resolves whether ``pallas_call`` should run in CPU interpret mode."""
  if interpret is not None:
    return interpret
  env = os.environ.get("PALLAS_INTERPRET", "").strip().lower()
  if env in ("0", "false"):
    return False
  if env in ("1", "true"):
    return True
  return jax.default_backend() != "tpu"


def _validate_chunk_size(seq_len: int, chunk_size: int) -> None:
  """Validates that ``chunk_size`` is positive, divides ``seq_len``, and fits VMEM."""
  if chunk_size <= 0:
    raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
  if seq_len % chunk_size != 0:
    raise ValueError(
        f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})."
    )
  if chunk_size > 64:
    raise NotImplementedError(
        f"chunk_size must be <= 64 for TPU VMEM layout, got {chunk_size}."
    )


def _select_mb(bh: int) -> int:
  """Selects the largest stream-batching factor ``MB <= 16`` dividing ``bh``."""
  for d in (16, 8, 4, 2):
    if bh % d == 0:
      return d
  return 1


def _chunkify(
    x: jax.Array, batch_size: int, num_chunks: int, chunk_size: int
) -> jax.Array:
  # (batch_size, seq_len, num_heads, ...) ->
  # (batch_size, num_heads, num_chunks, chunk_size, ...)
  rest = x.shape[3:]
  x = x.reshape((batch_size, num_chunks, chunk_size, x.shape[2]) + rest)
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _unchunk(
    x_c: jax.Array,
    rest_dim: int,
    batch_size: int,
    num_chunks: int,
    chunk_size: int,
    num_heads: int,
) -> jax.Array:
  x_ = jnp.transpose(x_c, (0, 2, 3, 1, 4))
  return x_.reshape(batch_size, num_chunks * chunk_size, num_heads, rest_dim)


def _decayed_gram(
    a: jax.Array, b: jax.Array, g_a: jax.Array, g_b: jax.Array
) -> jax.Array:
  """Computes decayed Gram matrix ``(a @ b^T) * exp(g_a[:, None] - g_b[None, :])``."""
  g0 = g_b[..., :1, :]
  lhs = (a * jnp.exp(jnp.minimum(g_a - g0, 80.0))).astype(jnp.float32)
  rhs = (b * jnp.exp(jnp.minimum(g0 - g_b, 80.0))).astype(jnp.float32)
  out = jnp.matmul(
      lhs,
      jnp.swapaxes(rhs, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return out.astype(a.dtype)


def _decayed_apply(qk: jax.Array, s: jax.Array, g: jax.Array) -> jax.Array:
  """Computes ``out[r, m] = sum_c qk[r, c] * exp(g[r, m] - g[c, m]) * s[c, m]``."""
  g0 = g[..., :1, :]
  rhs = (s * jnp.exp(jnp.minimum(g0 - g, 80.0))).astype(jnp.float32)
  out = jnp.matmul(
      qk.astype(jnp.float32),
      rhs,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return (out * jnp.exp(jnp.minimum(g - g0, 80.0))).astype(s.dtype)


def _stage1_fwd_kernel(
    # Inputs.
    q_vmem_ref,  # [MB, 1, BT, K]
    k_vmem_ref,  # [MB, 1, BT, K]
    s_vmem_ref,  # [MB, 1, BT, M]
    g_vmem_ref,  # [MB, 1, BT, M]
    s0_vmem_ref,  # [MB, K, M] f32
    # Outputs.
    ok_vmem_ref,  # [MB, 1, BT, M]
    s1_final_vmem_ref,  # [MB, K, M] f32
    s1_in_vmem_ref,  # [MB, 1, K, M] f32
    # Scratch.
    scratch_vmem_ref,  # [MB, K, M] f32
    *,
    num_chunks: int,
) -> None:
  """Fused single-pass MB-batched compute kernel for Raven Stage 1."""
  # Dimensions.
  chunk_size = q_vmem_ref.shape[2]
  chunk_idx = pl.program_id(1)

  @pl.when(chunk_idx == 0)
  def _() -> None:
    scratch_vmem_ref[...] = s0_vmem_ref[...].astype(jnp.float32)

  s1 = scratch_vmem_ref[...]  # [MB, K, M]
  s1_in_vmem_ref[:, 0] = s1.astype(s1_in_vmem_ref.dtype)

  q_i = q_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, K]
  k_i = k_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, K]
  s_i = s_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, M]
  g_i = g_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, M]

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 1)
  causal = (col_idx <= row_idx).astype(jnp.float32)

  qk = (
      jnp.matmul(
          q_i,
          jnp.swapaxes(k_i, -1, -2),
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * causal
  )
  y_diag = _decayed_apply(qk, s_i, g_i)
  y_off = jnp.matmul(
      q_i,
      s1,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  ) * jnp.exp(g_i)
  ok_i = y_diag + y_off

  g_last = g_i[:, -1, :]  # [MB, M]
  decay_tail = jnp.exp(jnp.minimum(g_last[:, None, :] - g_i, 0.0))
  state_contrib = jnp.matmul(
      jnp.swapaxes(k_i, -1, -2),
      s_i * decay_tail,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  s1_new = s1 * jnp.exp(g_last)[:, None, :] + state_contrib

  scratch_vmem_ref[...] = s1_new
  ok_vmem_ref[:, 0] = ok_i.astype(ok_vmem_ref.dtype)

  @pl.when(chunk_idx == num_chunks - 1)
  def _() -> None:
    s1_final_vmem_ref[...] = s1_new.astype(s1_final_vmem_ref.dtype)


def raven_pallas_stage1_fwd(
    q: jax.Array,
    k: jax.Array,
    s: jax.Array,
    g: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    return_residuals: bool = False,
) -> (
    tuple[jax.Array, jax.Array | None]
    | tuple[tuple[jax.Array, jax.Array | None], jax.Array]
):
  """Computes Raven Stage 1 forward pass using a single-pass MB-batched Pallas kernel."""
  batch_size, seq_len, num_heads, key_dim = q.shape
  _validate_chunk_size(seq_len, chunk_size)
  num_slots = s.shape[-1]
  num_chunks = seq_len // chunk_size
  bh = batch_size * num_heads
  mb = _select_mb(bh)

  if scale is not None:
    q_scaled = (q.astype(jnp.float32) * scale).astype(q.dtype)
  else:
    q_scaled = q

  q_c = _chunkify(q_scaled, batch_size, num_chunks, chunk_size).reshape(
      bh, num_chunks, chunk_size, key_dim
  )
  k_c = _chunkify(k, batch_size, num_chunks, chunk_size).reshape(
      bh, num_chunks, chunk_size, key_dim
  )
  s_c = _chunkify(s, batch_size, num_chunks, chunk_size).reshape(
      bh, num_chunks, chunk_size, num_slots
  )
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), batch_size, num_chunks, chunk_size),
      axis=-2,
  ).reshape(bh, num_chunks, chunk_size, num_slots)

  if initial_state is None:
    s0 = jnp.zeros((bh, key_dim, num_slots), dtype=jnp.float32)
  else:
    s0 = initial_state.astype(jnp.float32).reshape(bh, key_dim, num_slots)

  grid = (bh // mb, num_chunks)
  bspec_k = pl.BlockSpec(
      (mb, 1, chunk_size, key_dim), lambda i, n: (i, n, 0, 0)
  )
  bspec_m = pl.BlockSpec(
      (mb, 1, chunk_size, num_slots), lambda i, n: (i, n, 0, 0)
  )
  bspec_s = pl.BlockSpec((mb, key_dim, num_slots), lambda i, n: (i, 0, 0))
  bspec_s_in = pl.BlockSpec(
      (mb, 1, key_dim, num_slots), lambda i, n: (i, n, 0, 0)
  )

  interpret = _resolve_interpret()
  ok_c, s1_final, s1_in_c = pl.pallas_call(
      functools.partial(_stage1_fwd_kernel, num_chunks=num_chunks),
      grid=grid,
      in_specs=[bspec_k, bspec_k, bspec_m, bspec_m, bspec_s],
      out_specs=[bspec_m, bspec_s, bspec_s_in],
      scratch_shapes=[pltpu.VMEM((mb, key_dim, num_slots), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((bh, num_chunks, chunk_size, num_slots), s.dtype),
          jax.ShapeDtypeStruct((bh, key_dim, num_slots), jnp.float32),
          jax.ShapeDtypeStruct((bh, num_chunks, key_dim, num_slots), jnp.float32),
      ],
      interpret=interpret,
  )(q_c, k_c, s_c, g_c, s0)

  ok = _unchunk(
      ok_c.reshape(batch_size, num_heads, num_chunks, chunk_size, num_slots),
      num_slots,
      batch_size,
      num_chunks,
      chunk_size,
      num_heads,
  )
  s1_final_4d = s1_final.reshape(batch_size, num_heads, key_dim, num_slots)
  s1_in_5d = s1_in_c.reshape(
      batch_size, num_heads, num_chunks, key_dim, num_slots
  )
  final_state = s1_final_4d.astype(q.dtype) if output_final_state else None
  if return_residuals:
    return (ok, final_state), s1_in_5d
  return ok, final_state


def _stage2_fwd_kernel(
    # Inputs.
    qv_vmem_ref,  # [MB, 1, BT, M]
    s_vmem_ref,  # [MB, 1, BT, M]
    v_vmem_ref,  # [MB, 1, BT, V]
    g_vmem_ref,  # [MB, 1, BT, M]
    s0_vmem_ref,  # [MB, M, V] f32
    # Outputs.
    out_vmem_ref,  # [MB, 1, BT, V]
    s2_final_vmem_ref,  # [MB, M, V] f32
    s2_in_vmem_ref,  # [MB, 1, M, V] f32
    # Scratch.
    scratch_vmem_ref,  # [MB, M, V] f32
    *,
    num_chunks: int,
) -> None:
  """Fused single-pass MB-batched compute kernel for Raven Stage 2."""
  # Dimensions.
  chunk_size = qv_vmem_ref.shape[2]
  chunk_idx = pl.program_id(1)

  @pl.when(chunk_idx == 0)
  def _() -> None:
    scratch_vmem_ref[...] = s0_vmem_ref[...].astype(jnp.float32)

  s2 = scratch_vmem_ref[...]  # [MB, M, V]
  s2_in_vmem_ref[:, 0] = s2.astype(s2_in_vmem_ref.dtype)

  qv_i = qv_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, M]
  s_i = s_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, M]
  v_i = v_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, V]
  g_i = g_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, M]

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 1)
  causal = (col_idx <= row_idx).astype(jnp.float32)

  aqk = _decayed_gram(qv_i, s_i, g_i, g_i) * causal
  y_diag = jnp.matmul(
      aqk,
      v_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  y_off = jnp.matmul(
      qv_i * jnp.exp(g_i),
      s2,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  out_i = y_diag + y_off

  g_last = g_i[:, -1, :]  # [MB, M]
  decay_tail = jnp.exp(jnp.minimum(g_last[:, None, :] - g_i, 0.0))
  state_contrib = jnp.matmul(
      jnp.swapaxes(s_i * decay_tail, -1, -2),
      v_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  s2_new = s2 * jnp.exp(g_last)[:, :, None] + state_contrib

  scratch_vmem_ref[...] = s2_new
  out_vmem_ref[:, 0] = out_i.astype(out_vmem_ref.dtype)

  @pl.when(chunk_idx == num_chunks - 1)
  def _() -> None:
    s2_final_vmem_ref[...] = s2_new.astype(s2_final_vmem_ref.dtype)


def raven_pallas_stage2_fwd(
    qv: jax.Array,
    s: jax.Array,
    v: jax.Array,
    g: jax.Array,
    *,
    chunk_size: int = _CHUNK_SIZE,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    return_residuals: bool = False,
) -> (
    tuple[jax.Array, jax.Array | None]
    | tuple[tuple[jax.Array, jax.Array | None], jax.Array]
):
  """Computes Raven Stage 2 forward pass using a single-pass MB-batched Pallas kernel."""
  batch_size, seq_len, num_heads, num_slots = qv.shape
  _validate_chunk_size(seq_len, chunk_size)
  val_dim = v.shape[-1]
  num_chunks = seq_len // chunk_size
  bh = batch_size * num_heads
  mb = _select_mb(bh)

  qv_c = _chunkify(qv, batch_size, num_chunks, chunk_size).reshape(
      bh, num_chunks, chunk_size, num_slots
  )
  s_c = _chunkify(s, batch_size, num_chunks, chunk_size).reshape(
      bh, num_chunks, chunk_size, num_slots
  )
  v_c = _chunkify(v, batch_size, num_chunks, chunk_size).reshape(
      bh, num_chunks, chunk_size, val_dim
  )
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), batch_size, num_chunks, chunk_size),
      axis=-2,
  ).reshape(bh, num_chunks, chunk_size, num_slots)

  if initial_state is None:
    s0 = jnp.zeros((bh, num_slots, val_dim), dtype=jnp.float32)
  else:
    s0 = initial_state.astype(jnp.float32).reshape(bh, num_slots, val_dim)

  grid = (bh // mb, num_chunks)
  bspec_m = pl.BlockSpec(
      (mb, 1, chunk_size, num_slots), lambda i, n: (i, n, 0, 0)
  )
  bspec_v = pl.BlockSpec(
      (mb, 1, chunk_size, val_dim), lambda i, n: (i, n, 0, 0)
  )
  bspec_s = pl.BlockSpec((mb, num_slots, val_dim), lambda i, n: (i, 0, 0))
  bspec_s_in = pl.BlockSpec(
      (mb, 1, num_slots, val_dim), lambda i, n: (i, n, 0, 0)
  )

  interpret = _resolve_interpret()
  out_c, s2_final, s2_in_c = pl.pallas_call(
      functools.partial(_stage2_fwd_kernel, num_chunks=num_chunks),
      grid=grid,
      in_specs=[bspec_m, bspec_m, bspec_v, bspec_m, bspec_s],
      out_specs=[bspec_v, bspec_s, bspec_s_in],
      scratch_shapes=[pltpu.VMEM((mb, num_slots, val_dim), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((bh, num_chunks, chunk_size, val_dim), v.dtype),
          jax.ShapeDtypeStruct((bh, num_slots, val_dim), jnp.float32),
          jax.ShapeDtypeStruct((bh, num_chunks, num_slots, val_dim), jnp.float32),
      ],
      interpret=interpret,
  )(qv_c, s_c, v_c, g_c, s0)

  out = _unchunk(
      out_c.reshape(batch_size, num_heads, num_chunks, chunk_size, val_dim),
      val_dim,
      batch_size,
      num_chunks,
      chunk_size,
      num_heads,
  )
  s2_final_4d = s2_final.reshape(batch_size, num_heads, num_slots, val_dim)
  s2_in_5d = s2_in_c.reshape(
      batch_size, num_heads, num_chunks, num_slots, val_dim
  )
  final_state = s2_final_4d.astype(v.dtype) if output_final_state else None
  if return_residuals:
    return (out, final_state), s2_in_5d
  return out, final_state


def raven_pallas_fwd(
    q: jax.Array,
    k: jax.Array,
    s: jax.Array,
    g: jax.Array,
    v: jax.Array | None = None,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: (
        jax.Array | tuple[jax.Array | None, jax.Array | None] | None
    ) = None,
    output_final_state: bool = False,
    return_residuals: bool = False,
) -> Any:
  """Computes Raven GSA forward pass (Stage 1 when ``v is None``, or full 2-stage)."""
  if v is None:
    if isinstance(initial_state, tuple):
      s1_init = initial_state[0]
    else:
      s1_init = initial_state
    if return_residuals:
      (ok, s1_final), s1_in_c = raven_pallas_stage1_fwd(
          q,
          k,
          s,
          g,
          chunk_size=chunk_size,
          scale=scale,
          initial_state=s1_init,
          output_final_state=output_final_state,
          return_residuals=True,
      )
      return (ok, s1_final), (s1_in_c, None, None)
    return raven_pallas_stage1_fwd(
        q,
        k,
        s,
        g,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=s1_init,
        output_final_state=output_final_state,
        return_residuals=False,
    )

  if isinstance(initial_state, tuple):
    s1_init, s2_init = initial_state
  else:
    s1_init, s2_init = initial_state, None

  (ok, s1_final), s1_in_c = raven_pallas_stage1_fwd(
      q,
      k,
      s,
      g,
      chunk_size=chunk_size,
      scale=scale,
      initial_state=s1_init,
      output_final_state=output_final_state,
      return_residuals=True,
  )
  qv = jax.nn.softmax(ok.astype(jnp.float32), axis=-1).astype(v.dtype)
  (out, s2_final), s2_in_c = raven_pallas_stage2_fwd(
      qv,
      s,
      v,
      g,
      chunk_size=chunk_size,
      initial_state=s2_init,
      output_final_state=output_final_state,
      return_residuals=True,
  )
  final_state = (s1_final, s2_final) if output_final_state else None
  if return_residuals:
    return (out, final_state), (s1_in_c, qv, s2_in_c)
  return out, final_state
