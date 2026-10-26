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

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from tokamax._src.ops.experimental.gdn2 import pallas_mosaic_tpu_kernel

# Dimensions:
#   B: batch size
#   T: sequence length (NT * BT)
#   H: number of attention heads
#   BH: flattened batch-head count (B * H)
#   MB: stream micro-batch size dividing BH
#   NT: number of chunks (T // BT)
#   BT: chunk size (chunk_size)
#   K: key head dimension (key_dim)
#   V: value head dimension (val_dim)

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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Analytical backward of `_decayed_gram` using factored batched matmuls."""
  g0 = g_b[..., :1, :]
  exp_a = jnp.exp(jnp.minimum(g_a - g0, 80.0)).astype(jnp.float32)
  exp_b = jnp.exp(jnp.minimum(g0 - g_b, 80.0)).astype(jnp.float32)
  lhs = a.astype(jnp.float32) * exp_a
  rhs = b.astype(jnp.float32) * exp_b

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

  da = d_lhs * exp_a
  db = d_rhs * exp_b
  dg_a = d_lhs * lhs
  dg_b = -d_rhs * rhs
  return da, db, dg_a, dg_b


def _gdn2_bwd_kernel(
    # Inputs.
    q_vmem_ref,  # [MB, 1, BT, K] dtype
    k_vmem_ref,  # [MB, 1, BT, K] dtype
    v_vmem_ref,  # [MB, 1, BT, V] dtype
    g_vmem_ref,  # [MB, 1, BT, K] float32
    b_vmem_ref,  # [MB, 1, BT, K] dtype
    w_vmem_ref,  # [MB, 1, BT, V] dtype
    a_mat_vmem_ref,  # [MB, 1, BT, BT] dtype
    merged_out_vmem_ref,  # [MB, 1, BT, K + V] float32
    s_in_vmem_ref,  # [MB, 1, K, V] float32
    do_vmem_ref,  # [MB, 1, BT, V] dtype
    d_final_state_vmem_ref,  # [MB, K, V] float32
    # Outputs.
    dq_vmem_ref,  # [MB, 1, BT, K] float32
    dk_vmem_ref,  # [MB, 1, BT, K] dtype
    dv_vmem_ref,  # [MB, 1, BT, V] dtype
    dg_vmem_ref,  # [MB, 1, BT, K] dtype
    db_vmem_ref,  # [MB, 1, BT, K] dtype
    dw_vmem_ref,  # [MB, 1, BT, V] dtype
    ds0_vmem_ref,  # [MB, K, V] float32
    # Scratch.
    scratch_vmem_ref,  # [MB, K, V] float32
    *,
    num_chunks: int,
) -> None:
  """Single-pass MB-batched reverse state + intra-chunk backward kernel."""
  rev_c = pl.program_id(1)
  bt = q_vmem_ref.shape[2]
  k_dim = q_vmem_ref.shape[3]

  @pl.when(rev_c == 0)
  def _():
    scratch_vmem_ref[...] = d_final_state_vmem_ref[...].astype(jnp.float32)

  ds_out = scratch_vmem_ref[...]

  q_i = q_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, K]
  k_i = k_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, K]
  v_i = v_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, V]
  g_i = g_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, K]
  b_i = b_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, K]
  w_i = w_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, V]
  a_mat = a_mat_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, BT]
  merged_out = merged_out_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, K + V]
  s_prev = s_in_vmem_ref[:, 0].astype(jnp.float32)  # [MB, K, V]
  do_i = do_vmem_ref[:, 0].astype(jnp.float32)  # [MB, BT, V]

  row_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
  col_idx = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
  strictly_lower = (col_idx < row_idx).astype(jnp.float32)
  causal_keep = (col_idx <= row_idx).astype(jnp.float32)
  upper_tri = (row_idx <= col_idx).astype(jnp.float32)

  aqk_i = _decayed_gram(q_i, k_i, g_i, g_i) * causal_keep

  bk = b_i * k_i
  exp_g = jnp.exp(g_i)
  k_g_b = exp_g * bk

  wwy_i = merged_out[:, :, :k_dim]
  u_i = merged_out[:, :, k_dim:]
  v_new = u_i - jnp.matmul(
      wwy_i,
      s_prev,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  g_last = g_i[:, -1, :]  # [MB, K]
  exp_tail = jnp.exp(g_last[:, None, :] - g_i)
  decayed_k = exp_tail * k_i

  q_exp_g = q_i * exp_g
  d_q_exp_g = jnp.matmul(
      do_i,
      jnp.swapaxes(s_prev, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  dq_off = d_q_exp_g * exp_g
  dg_off = d_q_exp_g * q_exp_g

  d_s_from_o = jnp.matmul(
      jnp.swapaxes(q_exp_g, -1, -2),
      do_i,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  dv_from_o = jnp.matmul(
      jnp.swapaxes(aqk_i, -1, -2),
      do_i,
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
          jnp.swapaxes(wwy_i, -1, -2),
          dv_new,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
  )
  scratch_vmem_ref[...] = ds_prev

  @pl.when(rev_c == num_chunks - 1)
  def _():
    ds0_vmem_ref[...] = ds_prev.astype(ds0_vmem_ref.dtype)

  d_aqk = (
      jnp.matmul(
          do_i,
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
  mask_strictly_prior = (
      jax.lax.broadcasted_iota(jnp.int32, (bt, 1), 0) < (bt - 1)
  ).astype(jnp.float32)
  dg_tail = -d_tail_exp * mask_strictly_prior
  dg_last_tail = jnp.sum(d_tail_exp * mask_strictly_prior, axis=1)  # [MB, K]
  dg_last_state = jnp.sum(
      ds_out * s_prev * jnp.exp(g_last)[:, :, None], axis=2
  )  # [MB, K]
  dg_last = dg_last_tail + dg_last_state

  du_i = dv_new
  dwwy_i = -jnp.matmul(
      dv_new,
      jnp.swapaxes(s_prev, -1, -2),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  d_merged_out = jnp.concatenate([dwwy_i, du_i], axis=-1)
  d_merged_rhs = jnp.matmul(
      jnp.swapaxes(a_mat, -1, -2),
      d_merged_out,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  dk_g_b = d_merged_rhs[:, :, :k_dim]
  dwv = d_merged_rhs[:, :, k_dim:]

  d_tlower = (
      jnp.matmul(
          d_merged_rhs,
          jnp.swapaxes(merged_out, -1, -2),
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * strictly_lower
  )

  dw = dwv * v_i
  dv = dwv * w_i

  db_kgb = dk_g_b * exp_g * k_i
  dk_kgb = dk_g_b * exp_g * b_i
  dg_kgb = dk_g_b * k_g_b

  dq_intra, dk_intra, dg_aqk_a, dg_aqk_b = _decayed_gram_bwd(
      d_aqk, q_i, k_i, g_i, g_i
  )
  dg_decay_aqk = dg_aqk_a + dg_aqk_b

  dbk, dk_tlower, dg_tl_a, dg_tl_b = _decayed_gram_bwd(
      -d_tlower, bk, k_i, g_i, g_i
  )
  db_tlower = dbk * k_i
  dk_tlower2 = dbk * b_i
  dg_decay_tlower = dg_tl_a + dg_tl_b

  mask_last = (
      jax.lax.broadcasted_iota(jnp.int32, (bt, 1), 0) == (bt - 1)
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

  dg_raw = jnp.matmul(
      upper_tri,
      dg,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )

  dq_vmem_ref[:, 0] = dq.astype(dq_vmem_ref.dtype)
  dk_vmem_ref[:, 0] = dk.astype(dk_vmem_ref.dtype)
  dv_vmem_ref[:, 0] = dv.astype(dv_vmem_ref.dtype)
  dg_vmem_ref[:, 0] = dg_raw.astype(dg_vmem_ref.dtype)
  db_vmem_ref[:, 0] = db.astype(db_vmem_ref.dtype)
  dw_vmem_ref[:, 0] = dw.astype(dw_vmem_ref.dtype)


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
  """Runs fused single-pass MB-batched backward pass across all chunks."""
  bh, num_chunks, chunk_size, key_dim = q_c.shape
  val_dim = v_c.shape[-1]
  mb = _select_mb(bh)

  grid = (bh // mb, num_chunks)
  rev_idx = lambda i, n: (i, num_chunks - 1 - n, 0, 0)  # noqa: E731
  bspec_k = pl.BlockSpec((mb, 1, chunk_size, key_dim), rev_idx)
  bspec_v = pl.BlockSpec((mb, 1, chunk_size, val_dim), rev_idx)
  bspec_bt = pl.BlockSpec((mb, 1, chunk_size, chunk_size), rev_idx)
  bspec_merged = pl.BlockSpec((mb, 1, chunk_size, key_dim + val_dim), rev_idx)
  bspec_s_in = pl.BlockSpec((mb, 1, key_dim, val_dim), rev_idx)
  bspec_s = pl.BlockSpec((mb, key_dim, val_dim), lambda i, n: (i, 0, 0))

  interpret = jax.default_backend() == "cpu"
  return pl.pallas_call(
      functools.partial(_gdn2_bwd_kernel, num_chunks=num_chunks),
      grid=grid,
      in_specs=[
          bspec_k,
          bspec_k,
          bspec_v,
          bspec_k,
          bspec_k,
          bspec_v,
          bspec_bt,
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
      scratch_shapes=[pltpu.VMEM((mb, key_dim, val_dim), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      out_shape=[
          jax.ShapeDtypeStruct(
              (bh, num_chunks, chunk_size, key_dim), jnp.float32
          ),
          jax.ShapeDtypeStruct(
              (bh, num_chunks, chunk_size, key_dim), k_c.dtype
          ),
          jax.ShapeDtypeStruct(
              (bh, num_chunks, chunk_size, val_dim), v_c.dtype
          ),
          jax.ShapeDtypeStruct(
              (bh, num_chunks, chunk_size, key_dim), g_c.dtype
          ),
          jax.ShapeDtypeStruct(
              (bh, num_chunks, chunk_size, key_dim), b_c.dtype
          ),
          jax.ShapeDtypeStruct(
              (bh, num_chunks, chunk_size, val_dim), w_c.dtype
          ),
          jax.ShapeDtypeStruct((bh, key_dim, val_dim), jnp.float32),
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
  batch_size, seq_len, num_heads, key_dim = q.shape
  if chunk_size <= 0 or seq_len % chunk_size != 0:
    raise ValueError(
        f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})."
    )
  if chunk_size > 64:
    raise NotImplementedError(
        f"chunk_size must be <= 64 for TPU VMEM layout, got {chunk_size}."
    )

  val_dim = v.shape[-1]
  bh = batch_size * num_heads
  num_chunks = seq_len // chunk_size
  if scale is None:
    scale = key_dim**-0.5

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

  q_c = _chunkify(
      q.astype(jnp.float32) * scale, batch_size, num_chunks, chunk_size
  ).astype(q.dtype)
  k_c = _chunkify(k, batch_size, num_chunks, chunk_size)
  v_c = _chunkify(v, batch_size, num_chunks, chunk_size)
  g_c = jnp.cumsum(
      _chunkify(g.astype(jnp.float32), batch_size, num_chunks, chunk_size),
      axis=-2,
  )
  b_c = _chunkify(b, batch_size, num_chunks, chunk_size)
  w_c = _chunkify(w, batch_size, num_chunks, chunk_size)
  do_c = _chunkify(grad_out, batch_size, num_chunks, chunk_size)

  if grad_final_state is None:
    ds_final = jnp.zeros((bh, key_dim, val_dim), dtype=jnp.float32)
  else:
    ds_final = grad_final_state.reshape((bh, key_dim, val_dim)).astype(
        jnp.float32
    )

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

  dq = (
      _unchunk(dq_c, key_dim, batch_size, num_chunks, chunk_size, num_heads)
      * scale
  )
  dk = _unchunk(dk_c, key_dim, batch_size, num_chunks, chunk_size, num_heads)
  dv = _unchunk(dv_c, val_dim, batch_size, num_chunks, chunk_size, num_heads)
  dg = _unchunk(
      dg_raw_c, key_dim, batch_size, num_chunks, chunk_size, num_heads
  )
  db = _unchunk(db_c, key_dim, batch_size, num_chunks, chunk_size, num_heads)
  dw = _unchunk(dw_c, val_dim, batch_size, num_chunks, chunk_size, num_heads)

  ds0_out = (
      ds0.reshape((batch_size, num_heads, key_dim, val_dim)).astype(
          initial_state.dtype
      )
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
