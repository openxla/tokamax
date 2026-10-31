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

# Dimensions.
#   B      batch size
#   T      sequence length
#   H      number of attention heads
#   BH     flattened batch * heads (B * H)
#   K      key/query head dimension
#   M      number of memory slots
#   V      value head dimension
#   tb     tile extent of BH (stream-batching factor per Pallas program)
#   tt     tile extent of T (chunk_size)
#   nb     grid extent along BH: nb = BH // tb
#   nt     grid extent along T: nt = T // tt

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from tokamax._src.ops.experimental.raven import pallas_mosaic_tpu_kernel

_CHUNK_SIZE = 64
_validate_chunk_size = pallas_mosaic_tpu_kernel._validate_chunk_size
_resolve_interpret = pallas_mosaic_tpu_kernel._resolve_interpret
_select_tb = pallas_mosaic_tpu_kernel._select_tb
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
    # Inputs.
    q_ref,  # [tb, 1, tt, K] bf16/f32  reverse chunk order via index_map
    k_ref,  # [tb, 1, tt, K] bf16/f32  reverse chunk order via index_map
    s_ref,  # [tb, 1, tt, M] bf16/f32  reverse chunk order via index_map
    g_ref,  # [tb, 1, tt, M] f32       reverse chunk order via index_map
    do_ref,  # [tb, 1, tt, M] bf16/f32  reverse chunk order via index_map
    s1_in_ref,  # [tb, 1, K, M] f32        reverse chunk order via index_map
    ds1_final_ref,  # [tb, K, M] f32
    # Outputs.
    dq_ref,  # [tb, 1, tt, K] f32       reverse chunk order via index_map
    dk_ref,  # [tb, 1, tt, K] bf16/f32  reverse chunk order via index_map
    ds_ref,  # [tb, 1, tt, M] bf16/f32  reverse chunk order via index_map
    dg_ref,  # [tb, 1, tt, M] f32       reverse chunk order via index_map
    ds0_ref,  # [tb, K, M] f32
    # Scratch.
    ds1_scratch_ref,  # [tb, K, M] f32   persists across the reverse nt grid axis
    *,
    nt: int,
) -> None:
  """Fused single-pass tb-batched reverse recurrence backward kernel for Raven Stage 1."""
  rev_t_i = pl.program_id(1)
  _, _, tt, _ = q_ref.shape

  @pl.when(rev_t_i == 0)
  def _():
    ds1_scratch_ref[...] = ds1_final_ref[...].astype(jnp.float32)

  ds1_out = ds1_scratch_ref[...]  # [tb, K, M]
  s1_in = s1_in_ref[:, 0].astype(jnp.float32)  # [tb, K, M]

  q_i = q_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  k_i = k_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  s_i = s_ref[:, 0].astype(jnp.float32)  # [tb, tt, M]
  g_i = g_ref[:, 0].astype(jnp.float32)  # [tb, tt, M]
  do_i = do_ref[:, 0].astype(jnp.float32)  # [tb, tt, M]

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 1)
  causal = (col_idx <= row_idx).astype(jnp.float32)
  upper_tri = (row_idx <= col_idx).astype(jnp.float32)

  exp_g = jnp.exp(g_i)
  d_qs = do_i * exp_g
  dq_off = jnp.matmul(
      d_qs,
      jnp.swapaxes(s1_in, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  qs = jnp.matmul(
      q_i,
      s1_in,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  dg_off = qs * d_qs
  ds1_from_off = jnp.matmul(
      jnp.swapaxes(q_i, -1, -2),
      d_qs,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  g_last = g_i[:, -1, :]  # [tb, M]
  exp_glast = jnp.exp(g_last)
  ds_next_last = jnp.sum(ds1_out * s1_in * exp_glast[:, None, :], axis=1)
  ds1_prev = ds1_out * exp_glast[:, None, :] + ds1_from_off

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
      jax.lax.broadcasted_iota(jnp.int32, (1, tt, 1), 1) < (tt - 1)
  ).astype(jnp.float32)
  dg_tail = -d_tail_exp * mask_strictly_prior
  dg_last_tail = jnp.sum(d_tail_exp * mask_strictly_prior, axis=1)
  mask_last = (
      jax.lax.broadcasted_iota(jnp.int32, (1, tt, 1), 1) == (tt - 1)
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
  # Reverse cumulative sum within the chunk via upper-triangular MXU matmul.
  dg_raw_i = jnp.matmul(
      upper_tri,
      dg_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  ds1_scratch_ref[...] = ds1_prev
  dq_ref[:, 0] = dq_i.astype(dq_ref.dtype)
  dk_ref[:, 0] = dk_i.astype(dk_ref.dtype)
  ds_ref[:, 0] = ds_i.astype(ds_ref.dtype)
  dg_ref[:, 0] = dg_raw_i.astype(dg_ref.dtype)

  @pl.when(rev_t_i == nt - 1)
  def _():
    ds0_ref[...] = ds1_prev.astype(ds0_ref.dtype)


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
  B, T, H, K = q.shape
  _, _, _, M = s.shape
  tt = chunk_size
  _validate_chunk_size(T, tt)
  nt = T // tt
  BH = B * H
  tb = _select_tb(BH)
  nb = BH // tb

  if scale is None:
    scale = float(K**-0.5)

  if s1_in_c is None:
    _, s1_in_c = pallas_mosaic_tpu_kernel.raven_pallas_stage1_fwd(
        q,
        k,
        s,
        g,
        chunk_size=tt,
        scale=scale,
        initial_state=initial_state,
        output_final_state=False,
        return_residuals=True,
    )
  assert s1_in_c is not None

  q_scaled = (q.astype(jnp.float32) * scale).astype(q.dtype)

  q_c = _chunkify(q_scaled, B, nt, tt).reshape(BH, nt, tt, K)
  k_c = _chunkify(k, B, nt, tt).reshape(BH, nt, tt, K)
  s_c = _chunkify(s, B, nt, tt).reshape(BH, nt, tt, M)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), B, nt, tt),
      axis=-2,
  ).reshape(BH, nt, tt, M)
  do_c = _chunkify(grad_out, B, nt, tt).reshape(BH, nt, tt, M)
  s1_in_bh = s1_in_c.reshape(BH, nt, K, M)

  if grad_final_state is None:
    ds1_final = jnp.zeros((BH, K, M), dtype=jnp.float32)
  else:
    ds1_final = grad_final_state.astype(jnp.float32).reshape(BH, K, M)

  grid = (nb, nt)
  rev_idx = lambda b_i, rev_t_i: (b_i, nt - 1 - rev_t_i, 0, 0)  # noqa: E731
  bspec_k = pl.BlockSpec((tb, 1, tt, K), rev_idx)
  bspec_m = pl.BlockSpec((tb, 1, tt, M), rev_idx)
  bspec_s_in = pl.BlockSpec((tb, 1, K, M), rev_idx)
  bspec_s = pl.BlockSpec((tb, K, M), lambda b_i, rev_t_i: (b_i, 0, 0))

  interpret = _resolve_interpret()
  dq_c, dk_c, ds_c, dg_raw_c, ds0 = pl.pallas_call(
      functools.partial(_stage1_bwd_kernel, nt=nt),
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
      scratch_shapes=[pltpu.VMEM((tb, K, M), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((BH, nt, tt, K), jnp.float32),
          jax.ShapeDtypeStruct((BH, nt, tt, K), k.dtype),
          jax.ShapeDtypeStruct((BH, nt, tt, M), s.dtype),
          jax.ShapeDtypeStruct((BH, nt, tt, M), g.dtype),
          jax.ShapeDtypeStruct((BH, K, M), jnp.float32),
      ],
      interpret=interpret,
  )(q_c, k_c, s_c, g_c, do_c, s1_in_bh, ds1_final)

  dq = _unchunk(dq_c.reshape(B, H, nt, tt, K), K, B, nt, tt, H)
  if scale is not None:
    dq = dq * scale
  dk = _unchunk(dk_c.reshape(B, H, nt, tt, K), K, B, nt, tt, H)
  ds = _unchunk(ds_c.reshape(B, H, nt, tt, M), M, B, nt, tt, H)
  dg = _unchunk(dg_raw_c.reshape(B, H, nt, tt, M), M, B, nt, tt, H)
  ds0_4d = ds0.reshape(B, H, K, M)
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
    # Inputs.
    qv_ref,  # [tb, 1, tt, M] bf16/f32  reverse chunk order via index_map
    s_ref,  # [tb, 1, tt, M] bf16/f32  reverse chunk order via index_map
    v_ref,  # [tb, 1, tt, V] bf16/f32  reverse chunk order via index_map
    g_ref,  # [tb, 1, tt, M] f32       reverse chunk order via index_map
    do_ref,  # [tb, 1, tt, V] bf16/f32  reverse chunk order via index_map
    s2_in_ref,  # [tb, 1, M, V] f32        reverse chunk order via index_map
    ds2_final_ref,  # [tb, M, V] f32
    # Outputs.
    dqv_ref,  # [tb, 1, tt, M] bf16/f32  reverse chunk order via index_map
    ds_ref,  # [tb, 1, tt, M] bf16/f32  reverse chunk order via index_map
    dv_ref,  # [tb, 1, tt, V] bf16/f32  reverse chunk order via index_map
    dg_ref,  # [tb, 1, tt, M] f32       reverse chunk order via index_map
    ds0_ref,  # [tb, M, V] f32
    # Scratch.
    ds2_scratch_ref,  # [tb, M, V] f32   persists across the reverse nt grid axis
    *,
    nt: int,
) -> None:
  """Fused single-pass tb-batched reverse recurrence backward kernel for Raven Stage 2."""
  rev_t_i = pl.program_id(1)
  _, _, tt, _ = qv_ref.shape

  @pl.when(rev_t_i == 0)
  def _():
    ds2_scratch_ref[...] = ds2_final_ref[...].astype(jnp.float32)

  ds2_out = ds2_scratch_ref[...]  # [tb, M, V]
  s2_in = s2_in_ref[:, 0].astype(jnp.float32)  # [tb, M, V]

  qv_i = qv_ref[:, 0].astype(jnp.float32)  # [tb, tt, M]
  s_i = s_ref[:, 0].astype(jnp.float32)  # [tb, tt, M]
  v_i = v_ref[:, 0].astype(jnp.float32)  # [tb, tt, V]
  g_i = g_ref[:, 0].astype(jnp.float32)  # [tb, tt, M]
  do_i = do_ref[:, 0].astype(jnp.float32)  # [tb, tt, V]

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 1)
  causal = (col_idx <= row_idx).astype(jnp.float32)
  upper_tri = (row_idx <= col_idx).astype(jnp.float32)

  exp_g = jnp.exp(g_i)
  a_off = qv_i * exp_g
  da_off = jnp.matmul(
      do_i,
      jnp.swapaxes(s2_in, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  dqv_off = da_off * exp_g
  dg_off = da_off * a_off
  ds2_from_off = jnp.matmul(
      jnp.swapaxes(a_off, -1, -2),
      do_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  g_last = g_i[:, -1, :]  # [tb, M]
  exp_glast = jnp.exp(g_last)
  ds_next_last = jnp.sum(ds2_out * s2_in * exp_glast[:, :, None], axis=2)
  ds2_prev = ds2_out * exp_glast[:, :, None] + ds2_from_off

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
      jax.lax.broadcasted_iota(jnp.int32, (1, tt, 1), 1) < (tt - 1)
  ).astype(jnp.float32)
  dg_tail = -d_tail_exp * mask_strictly_prior
  dg_last_tail = jnp.sum(d_tail_exp * mask_strictly_prior, axis=1)
  mask_last = (
      jax.lax.broadcasted_iota(jnp.int32, (1, tt, 1), 1) == (tt - 1)
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

  ds2_scratch_ref[...] = ds2_prev
  dqv_ref[:, 0] = dqv_i.astype(dqv_ref.dtype)
  ds_ref[:, 0] = ds_i.astype(ds_ref.dtype)
  dv_ref[:, 0] = dv_i.astype(dv_ref.dtype)
  dg_ref[:, 0] = dg_raw_i.astype(dg_ref.dtype)

  @pl.when(rev_t_i == nt - 1)
  def _():
    ds0_ref[...] = ds2_prev.astype(ds0_ref.dtype)


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
  B, T, H, M = qv.shape
  _, _, _, V = v.shape
  tt = chunk_size
  _validate_chunk_size(T, tt)
  nt = T // tt
  BH = B * H
  tb = _select_tb(BH)
  nb = BH // tb

  if s2_in_c is None:
    _, s2_in_c = pallas_mosaic_tpu_kernel.raven_pallas_stage2_fwd(
        qv,
        s,
        v,
        g,
        chunk_size=tt,
        initial_state=initial_state,
        output_final_state=False,
        return_residuals=True,
    )
  assert s2_in_c is not None

  qv_c = _chunkify(qv, B, nt, tt).reshape(BH, nt, tt, M)
  s_c = _chunkify(s, B, nt, tt).reshape(BH, nt, tt, M)
  v_c = _chunkify(v, B, nt, tt).reshape(BH, nt, tt, V)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), B, nt, tt),
      axis=-2,
  ).reshape(BH, nt, tt, M)
  do_c = _chunkify(grad_out, B, nt, tt).reshape(BH, nt, tt, V)
  s2_in_bh = s2_in_c.reshape(BH, nt, M, V)

  if grad_final_state is None:
    ds2_final = jnp.zeros((BH, M, V), dtype=jnp.float32)
  else:
    ds2_final = grad_final_state.astype(jnp.float32).reshape(BH, M, V)

  grid = (nb, nt)
  rev_idx = lambda b_i, rev_t_i: (b_i, nt - 1 - rev_t_i, 0, 0)  # noqa: E731
  bspec_m = pl.BlockSpec((tb, 1, tt, M), rev_idx)
  bspec_v = pl.BlockSpec((tb, 1, tt, V), rev_idx)
  bspec_s_in = pl.BlockSpec((tb, 1, M, V), rev_idx)
  bspec_s = pl.BlockSpec((tb, M, V), lambda b_i, rev_t_i: (b_i, 0, 0))

  interpret = _resolve_interpret()
  dqv_c, ds_c, dv_c, dg_raw_c, ds0 = pl.pallas_call(
      functools.partial(_stage2_bwd_kernel, nt=nt),
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
      scratch_shapes=[pltpu.VMEM((tb, M, V), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((BH, nt, tt, M), qv.dtype),
          jax.ShapeDtypeStruct((BH, nt, tt, M), s.dtype),
          jax.ShapeDtypeStruct((BH, nt, tt, V), v.dtype),
          jax.ShapeDtypeStruct((BH, nt, tt, M), g.dtype),
          jax.ShapeDtypeStruct((BH, M, V), jnp.float32),
      ],
      interpret=interpret,
  )(qv_c, s_c, v_c, g_c, do_c, s2_in_bh, ds2_final)

  dqv = _unchunk(dqv_c.reshape(B, H, nt, tt, M), M, B, nt, tt, H)
  ds = _unchunk(ds_c.reshape(B, H, nt, tt, M), M, B, nt, tt, H)
  dv = _unchunk(dv_c.reshape(B, H, nt, tt, V), V, B, nt, tt, H)
  dg = _unchunk(dg_raw_c.reshape(B, H, nt, tt, M), M, B, nt, tt, H)
  ds0_4d = ds0.reshape(B, H, M, V)
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


def raven_pallas_bwd(
    q: jax.Array,
    k: jax.Array,
    s: jax.Array,
    g: jax.Array,
    v: jax.Array | None,
    grad_out: jax.Array,
    grad_final_state: (
        jax.Array | tuple[jax.Array | None, jax.Array | None] | None
    ) = None,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: (
        jax.Array | tuple[jax.Array | None, jax.Array | None] | None
    ) = None,
    s1_in_c: jax.Array | None = None,
    qv: jax.Array | None = None,
    s2_in_c: jax.Array | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
    jax.Array | tuple[jax.Array | None, jax.Array | None] | None,
]:
  """Analytical backward pass for Raven GSA (Stage 1 or full 2-stage)."""
  if v is None:
    s1_init = initial_state[0] if isinstance(initial_state, tuple) else initial_state
    ds1_final = (
        grad_final_state[0]
        if isinstance(grad_final_state, tuple)
        else grad_final_state
    )
    dq, dk, ds, dg, ds1_0 = raven_pallas_stage1_bwd(
        q,
        k,
        s,
        g,
        grad_out,
        grad_final_state=ds1_final,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=s1_init,
        s1_in_c=s1_in_c,
    )
    return dq, dk, ds, dg, None, ds1_0

  if isinstance(initial_state, tuple):
    s1_init, s2_init = initial_state
  else:
    s1_init, s2_init = initial_state, None

  if isinstance(grad_final_state, tuple):
    ds1_final, ds2_final = grad_final_state
  else:
    ds1_final, ds2_final = None, grad_final_state

  if qv is None or s1_in_c is None or s2_in_c is None:
    (_, _), (s1_in_c, qv, s2_in_c) = pallas_mosaic_tpu_kernel.raven_pallas_fwd(
        q,
        k,
        s,
        g,
        v,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=initial_state,
        output_final_state=False,
        return_residuals=True,
    )

  dqv, ds2, dv, dg2, ds2_0 = raven_pallas_stage2_bwd(
      qv,
      s,
      v,
      g,
      grad_out,
      grad_final_state=ds2_final,
      chunk_size=chunk_size,
      initial_state=s2_init,
      s2_in_c=s2_in_c,
  )
  qv_f32 = qv.astype(jnp.float32)
  dqv_f32 = dqv.astype(jnp.float32)
  dok = (
      qv_f32 * (dqv_f32 - jnp.sum(dqv_f32 * qv_f32, axis=-1, keepdims=True))
  ).astype(s.dtype)
  dq, dk, ds1, dg1, ds1_0 = raven_pallas_stage1_bwd(
      q,
      k,
      s,
      g,
      dok,
      grad_final_state=ds1_final,
      chunk_size=chunk_size,
      scale=scale,
      initial_state=s1_init,
      s1_in_c=s1_in_c,
  )
  ds = (ds1.astype(jnp.float32) + ds2.astype(jnp.float32)).astype(s.dtype)
  dg = (dg1.astype(jnp.float32) + dg2.astype(jnp.float32)).astype(g.dtype)
  if isinstance(initial_state, tuple):
    ds0_out = (ds1_0, ds2_0)
  else:
    ds0_out = ds1_0
  return dq, dk, ds, dg, dv, ds0_out
