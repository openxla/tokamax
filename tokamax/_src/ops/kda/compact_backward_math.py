# Copyright 2026 Ant Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Ported from primatrix/pallas-kernel dca6a09feec2659f2777a6db02f2ee55560f991e.

from __future__ import annotations
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from tokamax._src.ops.kda.utils import get_interpret
from tokamax._src.ops.kda.compact_utils import align_up, assert_shape, assert_shape_or_none
from tokamax._src.ops.kda.compact_hardware import get_tpu_config


def _chunk_segment_metadata(chunk_seg_ids, batch_idx, chunk_id, NT):
  """Return segment metadata for one chunk from compressed per-chunk IDs.

  chunk_seg_ids: [B, NT] int32 — ``segment_ids.reshape(B, NT, BT)[:, :, 0]``.
  batch_idx: which batch element (0..B-1).
  chunk_id: chunk index within that batch (0..NT-1).
  NT: per-batch chunk count (T // BT).
  Returns: (seg_cur, seq_idx, is_valid, is_first_chunk, is_last_chunk).
  """
  chunk_id = jnp.asarray(chunk_id, dtype=jnp.int32)
  batch_idx = jnp.asarray(batch_idx, dtype=jnp.int32)

  seg_cur = chunk_seg_ids[batch_idx, chunk_id]
  is_valid = seg_cur != 0

  prev_seg = jnp.where(
      chunk_id == 0,
      jnp.int32(0),
      chunk_seg_ids[batch_idx, jnp.maximum(chunk_id - 1, 0)],
  )
  next_seg = jnp.where(
      chunk_id + 1 >= NT,
      jnp.int32(0),
      chunk_seg_ids[batch_idx, jnp.minimum(chunk_id + 1, NT - 1)],
  )

  is_first_chunk = is_valid & (prev_seg != seg_cur)
  is_last_chunk = is_valid & (next_seg != seg_cur)
  seq_idx = jnp.maximum(seg_cur - 1, 0).astype(jnp.int32)
  return seg_cur, seq_idx, is_valid, is_first_chunk, is_last_chunk


def _matmul_in_dtype(lhs, rhs, compute_dtype, precision=None):
  # TPU MXU requires an FP32 accumulator even for BF16 operands. Narrow pure
  # low-precision results immediately; state-dependent contractions stay FP32.
  return jnp.matmul(
      lhs.astype(compute_dtype),
      rhs.astype(compute_dtype),
      precision=precision,
      preferred_element_type=jnp.float32,
  ).astype(compute_dtype)


@partial(jax.jit, static_argnames=["dtype", "compute_dtype"])
def compute_m1_recompute(
    bq,
    bk,
    bv,
    bb,
    bA,
    bg,
    bh,
    dtype,
    compute_dtype=jnp.float32,
    segment_ids=None,
):
  """(u, w, v_new, qg, kg) from saved h."""
  bq = bq.astype(compute_dtype)
  bk = bk.astype(compute_dtype)
  bv = bv.astype(compute_dtype)
  bb = bb.astype(compute_dtype)
  bA = bA.astype(compute_dtype)
  bg = bg.astype(jnp.float32)
  bh = bh.astype(jnp.float32)
  g_exp = jnp.exp2(bg)
  v_beta = bv * bb[:, :, None]
  u = _matmul_in_dtype(bA, v_beta, compute_dtype)
  w = jnp.matmul(
      bA.astype(jnp.float32),
      (bk * bb[:, :, None] * g_exp).astype(jnp.float32),
      preferred_element_type=jnp.float32,
  )
  u_mat = u.astype(dtype)
  w_mat = w.astype(jnp.float32)
  if segment_ids is not None:
    from tokamax._src.ops.kda.compact_segments import segmented_suffix_last

    valid = segment_ids != 0
    first_fragment = valid & (segment_ids == segment_ids[..., :1])
    w_mat = w_mat * first_fragment.astype(w_mat.dtype)[..., None]
    terminal_g = segmented_suffix_last(bg, segment_ids)
  else:
    terminal_g = bg[:, -1:, :]
  v_new = u_mat.astype(jnp.float32) - jnp.matmul(
      w_mat.astype(jnp.float32),
      bh,
      preferred_element_type=jnp.float32,
  )
  qg = bq * g_exp
  kg_exp = terminal_g - bg
  if segment_ids is not None:
    # Padding has terminal_g=0 but retains the negative scan coordinate.
    # Mask before exp2; masking an overflowing kg afterwards gives inf * 0.
    # Expand integer IDs before comparing: Mosaic cannot reshape i1 tiles
    # from [MB, BT] to [MB, BT, 1].
    kg_exp = jnp.where(segment_ids[..., None] != 0, kg_exp, 0.0)
  kg = bk * jnp.exp2(kg_exp)
  return u, w_mat, v_new, qg, kg


@partial(
    jax.jit,
    static_argnames=["scale", "precision", "return_b_dgk"],
)
def compute_wy_backward(
    bdo,
    bdv,
    bvn,
    bv,
    bh,
    dh,
    bq,
    bk,
    bg,
    bb,
    bA,
    scale,
    precision,
    mask_last=None,
    mask_lower=None,
    return_b_dgk=False,
    segment_ids=None,
):
  BT = bdo.shape[1]
  dq_acc = (
      jnp.matmul(
          bdo,
          bh.transpose(0, 2, 1),
          precision=precision,
          preferred_element_type=jnp.float32,
      )
      * scale
  )
  b_dw = -jnp.matmul(
      bdv,
      bh.transpose(0, 2, 1),
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  dk_acc = jnp.matmul(
      bvn,
      dh.transpose(0, 2, 1),
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  if segment_ids is not None:
    valid = segment_ids != 0
    first_fragment = valid & (segment_ids == segment_ids[..., :1])
    last_fragment = valid & (segment_ids == segment_ids[..., -1:])
    dq_acc = dq_acc * first_fragment.astype(dq_acc.dtype)[..., None]
    b_dw = b_dw * first_fragment.astype(b_dw.dtype)[..., None]
    dk_acc = dk_acc * last_fragment.astype(dk_acc.dtype)[..., None]
  b_dvb = jnp.matmul(
      bA.transpose(0, 2, 1),
      bdv,
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  db_acc = (b_dvb * bv).sum(axis=2)

  g_exp = jnp.exp2(bg)
  g_exp_last = g_exp[:, BT - 1, :]
  state_b_dgk = (bh * dh).sum(axis=2) * g_exp_last
  if segment_ids is not None:
    # h enters at the first fragment while dh leaves from the final fragment.
    # They belong to the same recurrence only when both chunk edges have the
    # same non-padding segment ID. Keep the local k*dk term below unchanged.
    same_edge_segment = (segment_ids[:, 0] != 0) & (
        segment_ids[:, 0] == segment_ids[:, BT - 1]
    )
    state_b_dgk = jnp.where(same_edge_segment[:, None], state_b_dgk, 0.0)
  b_dgk = state_b_dgk
  dq_acc = dq_acc * g_exp
  dk_acc = dk_acc * jnp.exp2(bg[:, BT - 1 : BT, :] - bg)

  gb = g_exp * bb[:, :, None]
  kg_local = bk * g_exp
  dAkk_local = jnp.matmul(
      jnp.concatenate([bdv, b_dw], axis=2),
      jnp.concatenate([bv, kg_local], axis=2).transpose(0, 2, 1),
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  dkgb = jnp.matmul(
      bA.transpose(0, 2, 1),
      b_dw,
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  db_acc = db_acc + (dkgb * kg_local).sum(axis=2)

  kdk = bk * dk_acc
  b_dgk = b_dgk + kdk.sum(axis=1)
  dg_acc = bq * dq_acc - kdk + kg_local * dkgb * bb[:, :, None]
  if not return_b_dgk:
    if mask_last is None:
      idx = jnp.arange(BT, dtype=jnp.int32)
      mask_last = (idx == BT - 1).astype(jnp.float32)
    dg_acc = dg_acc + mask_last[None, :, None] * b_dgk[:, None, :]
  dk_acc = dk_acc + dkgb * gb

  if mask_lower is None:
    idx = jnp.arange(BT, dtype=jnp.int32)
    mask_lower = (idx[:, None] > idx[None, :]).astype(jnp.float32)
    if segment_ids is not None:
      mask_lower = mask_lower * (
          (segment_ids[..., :, None] == segment_ids[..., None, :])
          & (segment_ids[..., :, None] != 0)
      ).astype(mask_lower.dtype)
  mask_lower_batched = (
      mask_lower[None, :, :] if mask_lower.ndim == 2 else mask_lower
  )
  dAkk_local = jnp.where(mask_lower_batched, dAkk_local * bb[:, None, :], 0.0)
  dAkk_local = jnp.matmul(
      dAkk_local,
      bA.transpose(0, 2, 1),
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  dAkk_local = jnp.matmul(
      bA.transpose(0, 2, 1),
      dAkk_local,
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  dAkk_local = jnp.where(mask_lower_batched, -dAkk_local, 0.0)
  outputs = (dq_acc, dk_acc, b_dvb, db_acc, dg_acc, dAkk_local)
  if return_b_dgk:
    return (*outputs, b_dgk)
  return outputs


def _chunk_window_has_valid(
    chunk_seg_ids,
    batch_idx,
    start_chunk,
    window_size,
):
  """Return whether a contiguous chunk window contains a real segment."""
  assert window_size > 0
  has_valid = chunk_seg_ids[batch_idx, start_chunk] != 0
  for offset in range(1, window_size):
    has_valid = jnp.logical_or(
        has_valid,
        chunk_seg_ids[batch_idx, start_chunk + offset] != 0,
    )
  return has_valid


def compute_intra_backward_sequential_to_chunk_refs(
    bq,
    bk,
    bg,
    bb,
    dAqk,
    dAkk,
    dq_acc,
    dk_acc,
    db_acc,
    dg_acc,
    dq_out_ref,
    dk_out_ref,
    db_out_ref,
    dg_out_ref,
    operand_dtype=None,
    b_dgk=None,
    return_dg=False,
):
  """Compute one intra block into already-sliced per-chunk output refs.

  ``operand_dtype`` lets a caller with fp32 output scratch (such as M5) retain
  the contraction mode selected by its public input dtype. BF16 contractions
  preserve RHS low bits for gate-parameter gradients.
  With ``return_dg``, return the FP32 gate cotangent instead of using the
  public output ref as intermediate scratch. Reverse cumsum must precede
  any narrowing to the public gradient dtype.
  """
  BT = dq_acc.shape[1]
  # Keep the positive exp2 argument bounded under the gate lower bound -5.
  # With a midpoint reference, BC=32 spans at most 16 * 5 = 80 in either
  # factor; a boundary reference would span 31 * 5 = 155 and overflow f32.
  BC = min(32, BT)
  assert (
      BT % BC == 0
  ), f"M4 intra contraction requires BT divisible by BC={BC}, got BT={BT}"
  NC = BT // BC
  _b1 = (((2,), (1,)), ((0,), (0,)))
  _b1t = (((1,), (1,)), ((0,), (0,)))

  def dot_rhs_compensated(lhs, rhs, dimension_numbers):
    """Use the contraction mode required by the public output dtypes."""
    contraction_dtype = (
        dq_out_ref.dtype if operand_dtype is None else operand_dtype
    )
    if contraction_dtype == jnp.float32:
      return jax.lax.dot_general(
          lhs,
          rhs,
          dimension_numbers,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )

    lhs_bf16 = lhs.astype(jnp.bfloat16)
    rhs_hi = rhs.astype(jnp.bfloat16)
    result = jax.lax.dot_general(
        lhs_bf16,
        rhs_hi,
        dimension_numbers,
        preferred_element_type=jnp.float32,
    )

    # Preserve decay-created RHS low bits on every BF16 path without using
    # HIGHEST.
    rhs_lo = (rhs - rhs_hi.astype(jnp.float32)).astype(jnp.bfloat16)
    return result + jax.lax.dot_general(
        lhs_bf16,
        rhs_lo,
        dimension_numbers,
        preferred_element_type=jnp.float32,
    )

  dg_blocks = []
  for ii in range(NC):
    s_ii = slice(ii * BC, (ii + 1) * BC)
    q_ii = bq[:, s_ii, :]
    k_ii = bk[:, s_ii, :]
    g_ii = bg[:, s_ii, :]
    beta_ii = bb[:, s_ii, None]

    # dAqk and dAkk are lower triangular.  Factor the row decay around the
    # diagonal block midpoint and contract the complete causal prefix at once.
    # This replaces one diagonal contraction plus ``ii`` pair contractions.
    g_ref = g_ii[:, BC // 2 : BC // 2 + 1, :]
    row_decay = jnp.exp2(g_ii - g_ref)
    row_stop = (ii + 1) * BC
    s_prefix = slice(0, row_stop)
    k_prefix_decayed = bk[:, s_prefix, :] * jnp.exp2(g_ref - bg[:, s_prefix, :])
    row_grads = dot_rhs_compensated(
        jnp.concatenate(
            [dAkk[:, s_ii, s_prefix], dAqk[:, s_ii, s_prefix]],
            axis=1,
        ),
        k_prefix_decayed,
        _b1,
    )
    dk_row_pre_ii = row_decay * row_grads[:, :BC, :]
    dq_ii = row_decay * row_grads[:, BC:, :]

    # Reuse the midpoint for the column factorization.  Future-block decays
    # stay <= 1, while either within-block positive exponent remains <= 80.
    s_suffix = slice(ii * BC, BT)
    suffix_decay = jnp.exp2(bg[:, s_suffix, :] - g_ref)
    q_suffix_decayed = bq[:, s_suffix, :] * suffix_decay
    kb_suffix_decayed = (
        bk[:, s_suffix, :] * bb[:, s_suffix, None] * suffix_decay
    )
    col_decay = jnp.exp2(g_ref - g_ii)
    dk_col_ii = col_decay * dot_rhs_compensated(
        jnp.concatenate(
            [dAqk[:, s_suffix, s_ii], dAkk[:, s_suffix, s_ii]],
            axis=1,
        ),
        jnp.concatenate([q_suffix_decayed, kb_suffix_decayed], axis=1),
        _b1t,
    )

    db_intra_ii = jnp.sum(k_ii * dk_row_pre_ii, axis=-1)
    dk_row_ii = beta_ii * dk_row_pre_ii
    dk_intra_ii = dk_row_ii + dk_col_ii
    dg_intra_ii = q_ii * dq_ii + k_ii * (dk_row_ii - dk_col_ii)

    dq_out_ref[:, s_ii, :] = (dq_acc[:, s_ii, :] + dq_ii).astype(
        dq_out_ref.dtype
    )
    dk_out_ref[:, s_ii, :] = (dk_acc[:, s_ii, :] + dk_intra_ii).astype(
        dk_out_ref.dtype
    )
    db_out_ref[pl.ds(0, db_acc.shape[0]), s_ii] = (
        db_acc[:, s_ii] + db_intra_ii
    ).astype(db_out_ref.dtype)
    dg_ii = dg_acc[:, s_ii, :] + dg_intra_ii
    if b_dgk is not None and ii == NC - 1:
      dg_ii = jnp.concatenate(
          [
              dg_ii[:, : BC - 1, :],
              dg_ii[:, BC - 1 : BC, :] + b_dgk[:, None, :],
          ],
          axis=1,
      )
    if return_dg:
      dg_blocks.append(dg_ii)
    else:
      dg_out_ref[:, s_ii, :] = dg_ii.astype(dg_out_ref.dtype)
  if return_dg:
    return jnp.concatenate(dg_blocks, axis=1).astype(jnp.float32)


@jax.jit
def compute_reverse_cumsum_dg_scan(dg_total):
  """Compute reverse cumsum with a vectorized Hillis-Steele scan."""
  BT = dg_total.shape[1]
  out = dg_total
  for d in range((BT - 1).bit_length()):
    stride = 1 << d
    out = jnp.concatenate(
        [
            out[:, : BT - stride, :] + out[:, stride:, :],
            out[:, BT - stride :, :],
        ],
        axis=1,
    )
  return out
