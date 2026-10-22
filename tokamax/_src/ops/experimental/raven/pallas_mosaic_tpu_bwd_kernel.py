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

"""Analytical backward pass kernels for Raven Gated Slot Attention (GSA)."""

import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from tokamax._src.ops.experimental.raven import pallas_mosaic_tpu_kernel

_CHUNK_SIZE = 64
_select_mb = pallas_mosaic_tpu_kernel._select_mb
_chunkify = pallas_mosaic_tpu_kernel._chunkify
_unchunk = pallas_mosaic_tpu_kernel._unchunk
_decayed_gram = pallas_mosaic_tpu_kernel._decayed_gram


def _decayed_gram_bwd(
    d_out: jax.Array,
    a: jax.Array,
    b: jax.Array,
    g_a: jax.Array,
    g_b: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Analytical backward of ``_decayed_gram`` using factored MXU matmuls."""
  g0 = g_b[..., :1, :]
  exp_pos = jnp.exp(jnp.minimum(g_a - g0, 80.0)).astype(jnp.float32)
  exp_neg = jnp.exp(jnp.minimum(g0 - g_b, 80.0)).astype(jnp.float32)
  lhs = a.astype(jnp.float32) * exp_pos
  rhs = b.astype(jnp.float32) * exp_neg
  d_lhs = jnp.matmul(
      d_out.astype(jnp.float32),
      rhs,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  d_rhs = jnp.matmul(
      jnp.swapaxes(d_out.astype(jnp.float32), -1, -2),
      lhs,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  da = d_lhs * exp_pos
  db = d_rhs * exp_neg
  dg_decay = d_lhs * lhs - d_rhs * rhs
  return da, db, dg_decay


def _decayed_apply_bwd(
    dy_diag: jax.Array,
    a: jax.Array,
    s: jax.Array,
    g: jax.Array,
    causal: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Analytical backward of ``_decayed_apply`` using factored MXU matmuls."""
  g0 = g[..., :1, :]
  exp_pos = jnp.exp(jnp.minimum(g - g0, 80.0)).astype(jnp.float32)
  exp_neg = jnp.exp(jnp.minimum(g0 - g, 80.0)).astype(jnp.float32)
  s_scaled = s.astype(jnp.float32) * exp_neg
  u_mat = jnp.matmul(
      a.astype(jnp.float32),
      s_scaled,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  y_diag = u_mat * exp_pos
  d_u = dy_diag.astype(jnp.float32) * exp_pos
  da = (
      jnp.matmul(
          d_u,
          jnp.swapaxes(s_scaled, -1, -2),
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * causal
  )
  d_s_scaled = jnp.matmul(
      jnp.swapaxes(a.astype(jnp.float32), -1, -2),
      d_u,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  ds_diag = d_s_scaled * exp_neg
  dg_decay = dy_diag.astype(jnp.float32) * y_diag - d_s_scaled * s_scaled
  return da, ds_diag, dg_decay


def _stage1_bwd_kernel(
    q_ref: Any,
    k_ref: Any,
    s_ref: Any,
    g_ref: Any,
    do_ref: Any,
    s1_in_ref: Any,
    d_s1_final_ref: Any,
    dq_ref: Any,
    dk_ref: Any,
    ds_ref: Any,
    dg_ref: Any,
    ds0_ref: Any,
    scratch_ref: Any,
    *,
    num_chunks: int,
) -> None:
  """Fused single-pass MB-batched reverse recurrence backward kernel for Raven Stage 1."""
  rev_c = pl.program_id(1)
  bt = q_ref.shape[2]

  @pl.when(rev_c == 0)
  def _() -> None:
    scratch_ref[...] = d_s1_final_ref[...].astype(jnp.float32)

  ds1_out = scratch_ref[...]  # (MB, K, M)
  s1_in = s1_in_ref[:, 0].astype(jnp.float32)  # (MB, K, M)

  q_i = q_ref[:, 0].astype(jnp.float32)  # (MB, BT, K)
  k_i = k_ref[:, 0].astype(jnp.float32)  # (MB, BT, K)
  s_i = s_ref[:, 0].astype(jnp.float32)  # (MB, BT, M)
  g_i = g_ref[:, 0].astype(jnp.float32)  # (MB, BT, M)
  do_i = do_ref[:, 0].astype(jnp.float32)  # (MB, BT, M)

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
  causal = (col_idx <= row_idx).astype(jnp.float32)
  upper_tri = (row_idx <= col_idx).astype(jnp.float32)

  exp_g = jnp.exp(g_i)
  d_qS = do_i * exp_g
  dq_off = jnp.matmul(
      d_qS,
      jnp.swapaxes(s1_in, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  qS = jnp.matmul(
      q_i,
      s1_in,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  dg_off = qS * d_qS
  ds1_from_off = jnp.matmul(
      jnp.swapaxes(q_i, -1, -2),
      d_qS,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  g_last = g_i[:, -1, :]  # (MB, M)
  exp_glast = jnp.exp(g_last)
  ds_next_last = jnp.sum(ds1_out * s1_in * exp_glast[:, None, :], axis=1)
  ds1_prev = ds1_out * exp_glast[:, None, :] + ds1_from_off
  scratch_ref[...] = ds1_prev

  @pl.when(rev_c == num_chunks - 1)
  def _() -> None:
    ds0_ref[...] = ds1_prev.astype(ds0_ref.dtype)

  qk = (
      jnp.matmul(
          q_i,
          jnp.swapaxes(k_i, -1, -2),
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * causal
  )
  da, ds_diag, dg_decay = _decayed_apply_bwd(do_i, qk, s_i, g_i, causal)
  dq_diag = jnp.matmul(
      da,
      k_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  dk_diag = jnp.matmul(
      jnp.swapaxes(da, -1, -2),
      q_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  decay_tail = jnp.exp(jnp.minimum(g_last[:, None, :] - g_i, 0.0))
  u = s_i * decay_tail
  dk_contrib = jnp.matmul(
      u,
      jnp.swapaxes(ds1_out, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  du = jnp.matmul(
      k_i,
      ds1_out,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  ds_contrib = du * decay_tail
  d_tail_exp = du * u

  mask_strictly_prior = (
      jax.lax.broadcasted_iota(jnp.int32, (1, bt, 1), 1) < (bt - 1)
  ).astype(jnp.float32)
  dg_tail = -d_tail_exp * mask_strictly_prior
  dg_last_tail = jnp.sum(d_tail_exp * mask_strictly_prior, axis=1)
  mask_last = (
      jax.lax.broadcasted_iota(jnp.int32, (1, bt, 1), 1) == (bt - 1)
  ).astype(jnp.float32)

  dq_i = dq_diag + dq_off
  dk_i = dk_diag + dk_contrib
  ds_i = ds_diag + ds_contrib
  dg_i = (
      dg_decay
      + dg_tail
      + dg_off
      + mask_last * (dg_last_tail + ds_next_last)[:, None, :]
  )
  dg_raw_i = jnp.matmul(
      upper_tri,
      dg_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  dq_ref[:, 0] = dq_i.astype(dq_ref.dtype)
  dk_ref[:, 0] = dk_i.astype(dk_ref.dtype)
  ds_ref[:, 0] = ds_i.astype(ds_ref.dtype)
  dg_ref[:, 0] = dg_raw_i.astype(dg_ref.dtype)


def raven_pallas_stage1_bwd(
    q: jax.Array,
    k: jax.Array,
    s: jax.Array,
    g: jax.Array,
    grad_out: jax.Array,
    grad_final_state: jax.Array | None = None,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    s1_in_c: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
  """Analytical closed-form backward pass for Raven Stage 1 using Pallas TPU."""
  if chunk_size > 64:
    raise NotImplementedError(
        f"chunk_size must be <= 64 for TPU VMEM layout, got {chunk_size}."
    )

  batch_size, seq_len, num_heads, key_dim = q.shape
  num_slots = s.shape[-1]
  num_chunks = seq_len // chunk_size
  bh = batch_size * num_heads
  mb = _select_mb(bh)

  if s1_in_c is None:
    _, s1_in_c = pallas_mosaic_tpu_kernel.raven_pallas_stage1_fwd(
        q,
        k,
        s,
        g,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=initial_state,
        output_final_state=False,
        return_residuals=True,
    )

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
  do_c = _chunkify(grad_out, batch_size, num_chunks, chunk_size).reshape(
      bh, num_chunks, chunk_size, num_slots
  )
  s1_in_bh = s1_in_c.reshape(bh, num_chunks, key_dim, num_slots)

  if grad_final_state is None:
    ds1_final = jnp.zeros((bh, key_dim, num_slots), dtype=jnp.float32)
  else:
    ds1_final = grad_final_state.astype(jnp.float32).reshape(
        bh, key_dim, num_slots
    )

  grid = (bh // mb, num_chunks)
  rev_idx = lambda i, n: (i, num_chunks - 1 - n, 0, 0)  # noqa: E731
  bspec_k = pl.BlockSpec((mb, 1, chunk_size, key_dim), rev_idx)
  bspec_m = pl.BlockSpec((mb, 1, chunk_size, num_slots), rev_idx)
  bspec_s_in = pl.BlockSpec((mb, 1, key_dim, num_slots), rev_idx)
  bspec_s = pl.BlockSpec((mb, key_dim, num_slots), lambda i, n: (i, 0, 0))

  interpret = jax.default_backend() == "cpu"
  dq_c, dk_c, ds_c, dg_raw_c, ds0 = pl.pallas_call(
      functools.partial(_stage1_bwd_kernel, num_chunks=num_chunks),
      grid=grid,
      in_specs=[
          bspec_k,
          bspec_k,
          bspec_m,
          bspec_m,
          bspec_m,
          bspec_s_in,
          bspec_s,
      ],
      out_specs=[bspec_k, bspec_k, bspec_m, bspec_m, bspec_s],
      scratch_shapes=[pltpu.VMEM((mb, key_dim, num_slots), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct(
              (bh, num_chunks, chunk_size, key_dim), jnp.float32
          ),
          jax.ShapeDtypeStruct((bh, num_chunks, chunk_size, key_dim), k.dtype),
          jax.ShapeDtypeStruct(
              (bh, num_chunks, chunk_size, num_slots), s.dtype
          ),
          jax.ShapeDtypeStruct(
              (bh, num_chunks, chunk_size, num_slots), g.dtype
          ),
          jax.ShapeDtypeStruct((bh, key_dim, num_slots), jnp.float32),
      ],
      interpret=interpret,
  )(q_c, k_c, s_c, g_c, do_c, s1_in_bh, ds1_final)

  dq = _unchunk(
      dq_c.reshape(batch_size, num_heads, num_chunks, chunk_size, key_dim),
      key_dim,
      batch_size,
      num_chunks,
      chunk_size,
      num_heads,
  )
  if scale is not None:
    dq = dq * scale
  dk = _unchunk(
      dk_c.reshape(batch_size, num_heads, num_chunks, chunk_size, key_dim),
      key_dim,
      batch_size,
      num_chunks,
      chunk_size,
      num_heads,
  )
  ds = _unchunk(
      ds_c.reshape(batch_size, num_heads, num_chunks, chunk_size, num_slots),
      num_slots,
      batch_size,
      num_chunks,
      chunk_size,
      num_heads,
  )
  dg = _unchunk(
      dg_raw_c.reshape(
          batch_size, num_heads, num_chunks, chunk_size, num_slots
      ),
      num_slots,
      batch_size,
      num_chunks,
      chunk_size,
      num_heads,
  )
  ds0_4d = ds0.reshape(batch_size, num_heads, key_dim, num_slots)
  ds0_out = (
      ds0_4d.astype(initial_state.dtype) if initial_state is not None else None
  )
  return (
      dq.astype(q.dtype),
      dk.astype(k.dtype),
      ds.astype(s.dtype),
      dg.astype(g.dtype),
      ds0_out,
  )


def _stage2_bwd_kernel(
    qv_ref: Any,
    s_ref: Any,
    v_ref: Any,
    g_ref: Any,
    do_ref: Any,
    s2_in_ref: Any,
    d_s2_final_ref: Any,
    dqv_ref: Any,
    ds_ref: Any,
    dv_ref: Any,
    dg_ref: Any,
    ds0_ref: Any,
    scratch_ref: Any,
    *,
    num_chunks: int,
) -> None:
  """Fused single-pass MB-batched reverse recurrence backward kernel for Raven Stage 2."""
  rev_c = pl.program_id(1)
  bt = qv_ref.shape[2]

  @pl.when(rev_c == 0)
  def _() -> None:
    scratch_ref[...] = d_s2_final_ref[...].astype(jnp.float32)

  ds2_out = scratch_ref[...]  # (MB, M, V)
  s2_in = s2_in_ref[:, 0].astype(jnp.float32)  # (MB, M, V)

  qv_i = qv_ref[:, 0].astype(jnp.float32)
  s_i = s_ref[:, 0].astype(jnp.float32)
  v_i = v_ref[:, 0].astype(jnp.float32)
  g_i = g_ref[:, 0].astype(jnp.float32)
  do_i = do_ref[:, 0].astype(jnp.float32)

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
  causal = (col_idx <= row_idx).astype(jnp.float32)
  upper_tri = (row_idx <= col_idx).astype(jnp.float32)

  exp_g = jnp.exp(g_i)
  a_off = qv_i * exp_g
  dA = jnp.matmul(
      do_i,
      jnp.swapaxes(s2_in, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  dqv_off = dA * exp_g
  dg_off = dA * a_off
  ds2_from_off = jnp.matmul(
      jnp.swapaxes(a_off, -1, -2),
      do_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  g_last = g_i[:, -1, :]  # (MB, M)
  exp_glast = jnp.exp(g_last)
  ds_next_last = jnp.sum(ds2_out * s2_in * exp_glast[:, :, None], axis=2)
  ds2_prev = ds2_out * exp_glast[:, :, None] + ds2_from_off
  scratch_ref[...] = ds2_prev

  @pl.when(rev_c == num_chunks - 1)
  def _() -> None:
    ds0_ref[...] = ds2_prev.astype(ds0_ref.dtype)

  aqk = _decayed_gram(qv_i, s_i, g_i, g_i) * causal
  dv_diag = jnp.matmul(
      jnp.swapaxes(aqk, -1, -2),
      do_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  d_aqk = (
      jnp.matmul(
          do_i,
          jnp.swapaxes(v_i, -1, -2),
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * causal
  )
  dqv_diag, ds_diag, dg_decay = _decayed_gram_bwd(d_aqk, qv_i, s_i, g_i, g_i)

  decay_tail = jnp.exp(jnp.minimum(g_last[:, None, :] - g_i, 0.0))
  u = s_i * decay_tail
  dv_contrib = jnp.matmul(
      u,
      ds2_out,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  du = jnp.matmul(
      v_i,
      jnp.swapaxes(ds2_out, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  ds_contrib = du * decay_tail
  d_tail_exp = du * u

  mask_strictly_prior = (
      jax.lax.broadcasted_iota(jnp.int32, (1, bt, 1), 1) < (bt - 1)
  ).astype(jnp.float32)
  dg_tail = -d_tail_exp * mask_strictly_prior
  dg_last_tail = jnp.sum(d_tail_exp * mask_strictly_prior, axis=1)
  mask_last = (
      jax.lax.broadcasted_iota(jnp.int32, (1, bt, 1), 1) == (bt - 1)
  ).astype(jnp.float32)

  dqv_i = dqv_diag + dqv_off
  ds_i = ds_diag + ds_contrib
  dv_i = dv_diag + dv_contrib
  dg_i = (
      dg_decay
      + dg_tail
      + dg_off
      + mask_last * (dg_last_tail + ds_next_last)[:, None, :]
  )
  dg_raw_i = jnp.matmul(
      upper_tri,
      dg_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  dqv_ref[:, 0] = dqv_i.astype(dqv_ref.dtype)
  ds_ref[:, 0] = ds_i.astype(ds_ref.dtype)
  dv_ref[:, 0] = dv_i.astype(dv_ref.dtype)
  dg_ref[:, 0] = dg_raw_i.astype(dg_ref.dtype)


def raven_pallas_stage2_bwd(
    qv: jax.Array,
    s: jax.Array,
    v: jax.Array,
    g: jax.Array,
    grad_out: jax.Array,
    grad_final_state: jax.Array | None = None,
    *,
    chunk_size: int = _CHUNK_SIZE,
    initial_state: jax.Array | None = None,
    s2_in_c: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
  """Analytical closed-form backward pass for Raven Stage 2 using Pallas TPU."""
  if chunk_size > 64:
    raise NotImplementedError(
        f"chunk_size must be <= 64 for TPU VMEM layout, got {chunk_size}."
    )

  batch_size, seq_len, num_heads, num_slots = qv.shape
  val_dim = v.shape[-1]
  num_chunks = seq_len // chunk_size
  bh = batch_size * num_heads
  mb = _select_mb(bh)

  if s2_in_c is None:
    _, s2_in_c = pallas_mosaic_tpu_kernel.raven_pallas_stage2_fwd(
        qv,
        s,
        v,
        g,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=False,
        return_residuals=True,
    )

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
  do_c = _chunkify(grad_out, batch_size, num_chunks, chunk_size).reshape(
      bh, num_chunks, chunk_size, val_dim
  )
  s2_in_bh = s2_in_c.reshape(bh, num_chunks, num_slots, val_dim)

  if grad_final_state is None:
    ds2_final = jnp.zeros((bh, num_slots, val_dim), dtype=jnp.float32)
  else:
    ds2_final = grad_final_state.astype(jnp.float32).reshape(
        bh, num_slots, val_dim
    )

  grid = (bh // mb, num_chunks)
  rev_idx = lambda i, n: (i, num_chunks - 1 - n, 0, 0)  # noqa: E731
  bspec_m = pl.BlockSpec((mb, 1, chunk_size, num_slots), rev_idx)
  bspec_v = pl.BlockSpec((mb, 1, chunk_size, val_dim), rev_idx)
  bspec_s_in = pl.BlockSpec((mb, 1, num_slots, val_dim), rev_idx)
  bspec_s = pl.BlockSpec((mb, num_slots, val_dim), lambda i, n: (i, 0, 0))

  interpret = jax.default_backend() == "cpu"
  dqv_c, ds_c, dv_c, dg_raw_c, ds0 = pl.pallas_call(
      functools.partial(_stage2_bwd_kernel, num_chunks=num_chunks),
      grid=grid,
      in_specs=[
          bspec_m,
          bspec_m,
          bspec_v,
          bspec_m,
          bspec_v,
          bspec_s_in,
          bspec_s,
      ],
      out_specs=[bspec_m, bspec_m, bspec_v, bspec_m, bspec_s],
      scratch_shapes=[pltpu.VMEM((mb, num_slots, val_dim), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct(
              (bh, num_chunks, chunk_size, num_slots), qv.dtype
          ),
          jax.ShapeDtypeStruct(
              (bh, num_chunks, chunk_size, num_slots), s.dtype
          ),
          jax.ShapeDtypeStruct((bh, num_chunks, chunk_size, val_dim), v.dtype),
          jax.ShapeDtypeStruct(
              (bh, num_chunks, chunk_size, num_slots), g.dtype
          ),
          jax.ShapeDtypeStruct((bh, num_slots, val_dim), jnp.float32),
      ],
      interpret=interpret,
  )(qv_c, s_c, v_c, g_c, do_c, s2_in_bh, ds2_final)

  dqv = _unchunk(
      dqv_c.reshape(batch_size, num_heads, num_chunks, chunk_size, num_slots),
      num_slots,
      batch_size,
      num_chunks,
      chunk_size,
      num_heads,
  )
  ds = _unchunk(
      ds_c.reshape(batch_size, num_heads, num_chunks, chunk_size, num_slots),
      num_slots,
      batch_size,
      num_chunks,
      chunk_size,
      num_heads,
  )
  dv = _unchunk(
      dv_c.reshape(batch_size, num_heads, num_chunks, chunk_size, val_dim),
      val_dim,
      batch_size,
      num_chunks,
      chunk_size,
      num_heads,
  )
  dg = _unchunk(
      dg_raw_c.reshape(
          batch_size, num_heads, num_chunks, chunk_size, num_slots
      ),
      num_slots,
      batch_size,
      num_chunks,
      chunk_size,
      num_heads,
  )
  ds0_4d = ds0.reshape(batch_size, num_heads, num_slots, val_dim)
  ds0_out = (
      ds0_4d.astype(initial_state.dtype) if initial_state is not None else None
  )
  return (
      dqv.astype(qv.dtype),
      ds.astype(s.dtype),
      dv.astype(v.dtype),
      dg.astype(g.dtype),
      ds0_out,
  )
