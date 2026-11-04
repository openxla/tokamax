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

"""Analytical backward pass kernel for Gated Delta Net 2 (GDN-2)."""

# Dimensions.
#   B   batch size
#   T   sequence length (nt * tt)
#   H   number of attention heads
#   K   key head dimension
#   V   value head dimension
#   BH  flattened batch-head count (B * H)
#   tb  tile extent of BH (micro-batch size dividing BH)
#   tt  tile extent of T (chunk_size)
#   nb  grid extent along BH (BH // tb)
#   nt  grid extent along T (T // tt, number of chunks)

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from tokamax._src.ops.experimental.gdn2 import pallas_mosaic_tpu_kernel

_CHUNK_SIZE = 64
_select_tb = pallas_mosaic_tpu_kernel._select_tb
_chunkify = pallas_mosaic_tpu_kernel._chunkify
_unchunk = pallas_mosaic_tpu_kernel._unchunk
_decayed_gram = pallas_mosaic_tpu_kernel._decayed_gram


def _decayed_gram_bwd(
    d_out: jax.Array,
    lhs_in: jax.Array,
    rhs_in: jax.Array,
    g_lhs: jax.Array,
    g_rhs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Analytical backward of `_decayed_gram` using factored batched matmuls."""
  g0 = g_rhs[..., :1, :]
  exp_lhs = jnp.exp(jnp.minimum(g_lhs - g0, 80.0)).astype(jnp.float32)
  exp_rhs = jnp.exp(jnp.minimum(g0 - g_rhs, 80.0)).astype(jnp.float32)
  lhs = lhs_in.astype(jnp.float32) * exp_lhs
  rhs = rhs_in.astype(jnp.float32) * exp_rhs

  d_out_f32 = d_out.astype(jnp.float32)
  d_lhs = jnp.matmul(
      d_out_f32,
      rhs,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  d_rhs = jnp.matmul(
      jnp.swapaxes(d_out_f32, -1, -2),
      lhs,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  d_lhs_in = d_lhs * exp_lhs
  d_rhs_in = d_rhs * exp_rhs
  dg_lhs = d_lhs * lhs
  dg_rhs = -d_rhs * rhs
  return d_lhs_in, d_rhs_in, dg_lhs, dg_rhs


def _gdn2_bwd_kernel(
    # Inputs.
    q_ref,  # [tb, 1, tt, K]
    k_ref,  # [tb, 1, tt, K]
    v_ref,  # [tb, 1, tt, V]
    g_ref,  # [tb, 1, tt, K] f32
    b_ref,  # [tb, 1, tt, K]
    w_ref,  # [tb, 1, tt, V]
    a_mat_ref,  # [tb, 1, tt, tt]
    merged_out_ref,  # [tb, 1, tt, K + V] f32
    s_in_ref,  # [tb, 1, K, V] f32
    do_ref,  # [tb, 1, tt, V]
    d_final_state_ref,  # [tb, K, V] f32
    # Outputs.
    dq_ref,  # [tb, 1, tt, K] f32   transposed in-chunk index_map (nt - 1 - rev_t_i)
    dk_ref,  # [tb, 1, tt, K]
    dv_ref,  # [tb, 1, tt, V]
    dg_ref,  # [tb, 1, tt, K]
    db_ref,  # [tb, 1, tt, K]
    dw_ref,  # [tb, 1, tt, V]
    ds0_ref,  # [tb, K, V] f32
    # Scratch.
    ds_scratch_ref,  # [tb, K, V] f32   persists across the reverse nt grid axis
    *,
    nt: int,
) -> None:
  """Single-pass `tb`-batched reverse state + intra-chunk backward kernel."""
  rev_t_i = pl.program_id(1)
  _, _, tt, K = q_ref.shape

  # Seed the reverse-time state adjoint `ds_scratch_ref` on the first reverse
  # step (`rev_t_i == 0`, corresponding to the last forward chunk `nt - 1`).
  @pl.when(rev_t_i == 0)
  def _():
    ds_scratch_ref[...] = d_final_state_ref[...].astype(jnp.float32)

  ds_out = ds_scratch_ref[...]

  q_tile = q_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  k_tile = k_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  v_tile = v_ref[:, 0].astype(jnp.float32)  # [tb, tt, V]
  g_tile = g_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  b_tile = b_ref[:, 0].astype(jnp.float32)  # [tb, tt, K]
  w_tile = w_ref[:, 0].astype(jnp.float32)  # [tb, tt, V]
  a_mat = a_mat_ref[:, 0].astype(jnp.float32)  # [tb, tt, tt]
  merged_out = merged_out_ref[:, 0].astype(jnp.float32)  # [tb, tt, K + V]
  s_prev = s_in_ref[:, 0].astype(jnp.float32)  # [tb, K, V]
  do_tile = do_ref[:, 0].astype(jnp.float32)  # [tb, tt, V]

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 1)
  strictly_lower = (col_idx < row_idx).astype(jnp.float32)
  causal_keep = (col_idx <= row_idx).astype(jnp.float32)
  upper_tri = (row_idx <= col_idx).astype(jnp.float32)

  aqk_tile = _decayed_gram(q_tile, k_tile, g_tile, g_tile) * causal_keep

  bk = b_tile * k_tile
  exp_g = jnp.exp(g_tile)
  k_g_b = exp_g * bk

  wwy_tile = merged_out[:, :, :K]
  u_tile = merged_out[:, :, K:]
  v_new = u_tile - jnp.matmul(
      wwy_tile,
      s_prev,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  g_last = g_tile[:, -1, :]  # [tb, K]
  exp_tail = jnp.exp(g_last[:, None, :] - g_tile)
  decayed_k = exp_tail * k_tile

  q_exp_g = q_tile * exp_g
  d_q_exp_g = jnp.matmul(
      do_tile,
      jnp.swapaxes(s_prev, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  dq_off = d_q_exp_g * exp_g
  dg_off = d_q_exp_g * q_exp_g

  d_s_from_o = jnp.matmul(
      jnp.swapaxes(q_exp_g, -1, -2),
      do_tile,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  dv_from_o = jnp.matmul(
      jnp.swapaxes(aqk_tile, -1, -2),
      do_tile,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  dv_new = dv_from_o + jnp.matmul(
      decayed_k,
      ds_out,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  ds_prev = (
      ds_out * jnp.exp(g_last)[:, :, None]
      + d_s_from_o
      - jnp.matmul(
          jnp.swapaxes(wwy_tile, -1, -2),
          dv_new,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
  )
  ds_scratch_ref[...] = ds_prev

  @pl.when(rev_t_i == nt - 1)
  def _():
    ds0_ref[...] = ds_prev.astype(ds0_ref.dtype)

  d_aqk = (
      jnp.matmul(
          do_tile,
          jnp.swapaxes(v_new, -1, -2),
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * causal_keep
  )

  d_decayed_k = jnp.matmul(
      v_new,
      jnp.swapaxes(ds_out, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  dk_tail = d_decayed_k * exp_tail
  d_tail_exp = d_decayed_k * decayed_k
  # For `t == tt - 1`, `g_last - g_tile[:, -1, :] == 0` identically, so its
  # derivative with respect to `g_tile` cancels out; mask out the last row.
  mask_strictly_prior = (
      jax.lax.broadcasted_iota(jnp.int32, (tt, 1), 0) < (tt - 1)
  ).astype(jnp.float32)
  dg_tail = -d_tail_exp * mask_strictly_prior
  dg_last_tail = jnp.sum(d_tail_exp * mask_strictly_prior, axis=1)  # [tb, K]
  dg_last_state = jnp.sum(
      ds_out * s_prev * jnp.exp(g_last)[:, :, None], axis=2
  )  # [tb, K]
  dg_last = dg_last_tail + dg_last_state

  du_tile = dv_new
  dwwy_tile = -jnp.matmul(
      dv_new,
      jnp.swapaxes(s_prev, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  d_merged_out = jnp.concatenate([dwwy_tile, du_tile], axis=-1)
  d_merged_rhs = jnp.matmul(
      jnp.swapaxes(a_mat, -1, -2),
      d_merged_out,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  dk_g_b = d_merged_rhs[:, :, :K]
  dwv = d_merged_rhs[:, :, K:]

  # Implicit differentiation of `a_mat = (I - t_lower)^{-1}` gives
  # `d_tlower = strictly_lower(a_mat^T @ d_a_mat @ a_mat^T) = strictly_lower(d_merged_rhs @ merged_out^T)`.
  d_tlower = (
      jnp.matmul(
          d_merged_rhs,
          jnp.swapaxes(merged_out, -1, -2),
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * strictly_lower
  )

  dw = dwv * v_tile
  dv = dwv * w_tile

  db_kgb = dk_g_b * exp_g * k_tile
  dk_kgb = dk_g_b * exp_g * b_tile
  dg_kgb = dk_g_b * k_g_b

  dq_intra, dk_intra, dg_aqk_a, dg_aqk_b = _decayed_gram_bwd(
      d_aqk, q_tile, k_tile, g_tile, g_tile
  )
  dg_decay_aqk = dg_aqk_a + dg_aqk_b

  dbk, dk_tlower, dg_tl_a, dg_tl_b = _decayed_gram_bwd(
      -d_tlower, bk, k_tile, g_tile, g_tile
  )
  db_tlower = dbk * k_tile
  dk_tlower2 = dbk * b_tile
  dg_decay_tlower = dg_tl_a + dg_tl_b

  mask_last = (
      jax.lax.broadcasted_iota(jnp.int32, (tt, 1), 0) == (tt - 1)
  ).astype(jnp.float32)

  dq = dq_off + dq_intra
  dk = dk_kgb + dk_tail + dk_intra + dk_tlower + dk_tlower2
  db = db_kgb + db_tlower
  dg = (
      dg_off
      + dg_kgb
      + dg_tail
      + dg_decay_aqk
      + dg_decay_tlower
      + mask_last * dg_last[:, None, :]
  )

  # Convert gradient w.r.t. cumulative chunk decay `g_c` into gradient w.r.t.
  # token decay `g` via reverse cumulative sum (`upper_tri @ dg`).
  dg_raw = jnp.matmul(
      upper_tri,
      dg,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  dq_ref[:, 0] = dq.astype(dq_ref.dtype)
  dk_ref[:, 0] = dk.astype(dk_ref.dtype)
  dv_ref[:, 0] = dv.astype(dv_ref.dtype)
  dg_ref[:, 0] = dg_raw.astype(dg_ref.dtype)
  db_ref[:, 0] = db.astype(db_ref.dtype)
  dw_ref[:, 0] = dw.astype(dw_ref.dtype)


def _gdn2_bwd_pallas(
    q_c: jax.Array,
    k_c: jax.Array,
    v_c: jax.Array,
    g_c: jax.Array,
    b_c: jax.Array,
    w_c: jax.Array,
    a_mat_c: jax.Array,
    merged_out_c: jax.Array,
    s_in_c: jax.Array,
    do_c: jax.Array,
    ds_final: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
  """Runs fused single-pass `tb`-batched backward pass across all chunks."""
  BH, nt, tt, K = q_c.shape
  _, _, _, V = v_c.shape
  tb = _select_tb(BH)
  nb = BH // tb

  grid = (nb, nt)
  rev_idx = lambda b_i, rev_t_i: (b_i, nt - 1 - rev_t_i, 0, 0)  # noqa: E731
  bspec_k = pl.BlockSpec((tb, 1, tt, K), rev_idx)
  bspec_v = pl.BlockSpec((tb, 1, tt, V), rev_idx)
  bspec_tt = pl.BlockSpec((tb, 1, tt, tt), rev_idx)
  bspec_merged = pl.BlockSpec((tb, 1, tt, K + V), rev_idx)
  bspec_s_in = pl.BlockSpec((tb, 1, K, V), rev_idx)
  bspec_s = pl.BlockSpec((tb, K, V), lambda b_i, rev_t_i: (b_i, 0, 0))

  interpret = jax.default_backend() != "tpu"
  return pl.pallas_call(
      functools.partial(_gdn2_bwd_kernel, nt=nt),
      grid=grid,
      in_specs=[
          bspec_k,
          bspec_k,
          bspec_v,
          bspec_k,
          bspec_k,
          bspec_v,
          bspec_tt,
          bspec_merged,
          bspec_s_in,
          bspec_v,
          bspec_s,
      ],
      out_specs=[
          bspec_k,
          bspec_k,
          bspec_v,
          bspec_k,
          bspec_k,
          bspec_v,
          bspec_s,
      ],
      scratch_shapes=[pltpu.VMEM((tb, K, V), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct((BH, nt, tt, K), jnp.float32),
          jax.ShapeDtypeStruct((BH, nt, tt, K), k_c.dtype),
          jax.ShapeDtypeStruct((BH, nt, tt, V), v_c.dtype),
          jax.ShapeDtypeStruct((BH, nt, tt, K), g_c.dtype),
          jax.ShapeDtypeStruct((BH, nt, tt, K), b_c.dtype),
          jax.ShapeDtypeStruct((BH, nt, tt, V), w_c.dtype),
          jax.ShapeDtypeStruct((BH, K, V), jnp.float32),
      ],
      interpret=interpret,
  )(
      q_c,
      k_c,
      v_c,
      g_c,
      b_c,
      w_c,
      a_mat_c,
      merged_out_c,
      s_in_c,
      do_c,
      ds_final,
  )


def gdn2_pallas_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    grad_out: jax.Array,
    grad_final_state: jax.Array | None = None,
    *,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    a_mat_c: jax.Array | None = None,
    merged_out_c: jax.Array | None = None,
    s_in_c: jax.Array | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
]:
  """Analytical closed-form backward pass for GDN-2 operator using Pallas TPU."""
  B, T, H, K = q.shape
  _, _, _, V = v.shape
  if chunk_size > 64:
    raise NotImplementedError(
        f"chunk_size must be <= 64 for TPU VMEM layout, got {chunk_size}."
    )

  BH = B * H
  tt = chunk_size
  nt = T // tt
  if scale is None:
    scale = float(K**-0.5)

  if a_mat_c is None or merged_out_c is None or s_in_c is None:
    _, (a_mat_c, merged_out_c, s_in_c) = (
        pallas_mosaic_tpu_kernel.gdn2_pallas_fwd(
            q,
            k,
            v,
            g,
            b,
            w,
            chunk_size=chunk_size,
            scale=scale,
            initial_state=initial_state,
            output_final_state=False,
            return_residuals=True,
        )
    )
  assert a_mat_c is not None
  assert merged_out_c is not None
  assert s_in_c is not None

  q_c = _chunkify(q.astype(jnp.float32) * scale, B, nt, tt).astype(q.dtype)
  k_c = _chunkify(k, B, nt, tt)
  v_c = _chunkify(v, B, nt, tt)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), B, nt, tt),
      axis=-2,
  )
  b_c = _chunkify(b, B, nt, tt)
  w_c = _chunkify(w, B, nt, tt)
  do_c = _chunkify(grad_out, B, nt, tt)

  if grad_final_state is None:
    ds_final = jnp.zeros((BH, K, V), dtype=jnp.float32)
  else:
    ds_final = grad_final_state.reshape((BH, K, V)).astype(jnp.float32)

  dq_c, dk_c, dv_c, dg_raw_c, db_c, dw_c, ds0 = _gdn2_bwd_pallas(
      q_c,
      k_c,
      v_c,
      g_c,
      b_c,
      w_c,
      a_mat_c,
      merged_out_c,
      s_in_c,
      do_c,
      ds_final,
  )

  dq = _unchunk(dq_c, K, B, nt, tt, H) * scale
  dk = _unchunk(dk_c, K, B, nt, tt, H)
  dv = _unchunk(dv_c, V, B, nt, tt, H)
  dg = _unchunk(dg_raw_c, K, B, nt, tt, H)
  db = _unchunk(db_c, K, B, nt, tt, H)
  dw = _unchunk(dw_c, V, B, nt, tt, H)

  ds0_out = (
      ds0.reshape((B, H, K, V)).astype(initial_state.dtype)
      if initial_state is not None
      else None
  )
  return (
      dq.astype(q.dtype),
      dk.astype(k.dtype),
      dv.astype(v.dtype),
      dg.astype(g.dtype),
      db.astype(b.dtype),
      dw.astype(w.dtype),
      ds0_out,
  )
