# Copyright 2026 Ant Group. All Rights Reserved.
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
"""CP KDA megakernel from pallas-kernel c82625a2bdadd5881b12f1ab37ec3fa2004c9780."""

from functools import partial
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int
from tokamax._src.ops.kda.common import get_tpu_limits, estimate_mini_batch
from tokamax._src.ops.kda.utils import get_interpret, align_up
from tokamax._src.ops.kda.cp_utils import ContextParallelMetadata as CPContext


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
  v_new = u_mat.astype(jnp.float32) - jnp.matmul(
      w_mat.astype(jnp.float32),
      bh,
      preferred_element_type=jnp.float32,
  )
  qg = bq * g_exp
  kg = bk * jnp.exp2(bg[:, -1:, :] - bg)
  return u, w, v_new, qg, kg


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
  b_dvb = jnp.matmul(
      bA.transpose(0, 2, 1),
      bdv,
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  db_acc = (b_dvb * bv).sum(axis=2)

  g_exp = jnp.exp2(bg)
  g_exp_last = g_exp[:, BT - 1, :]
  b_dgk = (bh * dh).sum(axis=2) * g_exp_last
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
  dAkk_local = jnp.where(
      mask_lower[None, :, :], dAkk_local * bb[:, None, :], 0.0
  )
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
  dAkk_local = jnp.where(mask_lower[None, :, :], -dAkk_local, 0.0)
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
):
  """Compute one intra block into already-sliced per-chunk output refs.

  ``operand_dtype`` lets a caller with fp32 output scratch (such as M5) retain
  the contraction mode selected by its public input dtype. BF16 contractions
  preserve RHS low bits for gate-parameter gradients.
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
    dg_out_ref[:, s_ii, :] = dg_ii.astype(dg_out_ref.dtype)


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


# Mega-kernel tuning knobs. Each benchmark process reads MB/CB before JAX
# tracing, so those variants can be swept without source edits. The backward
# pipeline uses independent two-way input and output ping-pong buffers; output
# storage never aliases an input tile.
KDA_DMA_CHUNK_BATCH = 6
KDA_DMA_INPUT_BUFFER_COUNT = 2
KDA_DMA_OUTPUT_BUFFER_COUNT = 2
KDA_DMA_MINI_BATCH = None

# Keep the backward mega-kernel within a portable 64 MiB ceiling.  The
# hardware-specific compiler budget may be smaller (TPU v7 reserves 64 KiB),
# so every Pallas call below must also intersect this ceiling with the detected
# compiler limit.
KDA_BWD_SCOPED_VMEM_LIMIT_BYTES = 64 * 1024 * 1024
# Keep custom collective IDs stable and separate from JAX's automatic range.
KDA_BWD_CP_COLLECTIVE_ID = 101


@partial(jax.jit, static_argnames=["scale"])
def compute_dav(bdo, bvn, bAqk, scale):
  # Forward stores Aqk with its strict upper triangle zeroed, so masking it
  # again before the dV contraction only adds a full-tile VPU select.
  BT = bdo.shape[1]
  m_causal = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
  bdAqk = jnp.matmul(
      bdo.astype(jnp.float32),
      bvn.transpose(0, 2, 1),
      preferred_element_type=jnp.float32,
  )
  bdAqk = jnp.where(m_causal[None, :, :], bdAqk * scale, 0.0)
  bdv0 = jnp.matmul(
      bAqk.transpose(0, 2, 1),
      bdo.astype(jnp.float32),
      preferred_element_type=jnp.float32,
  )
  return bdAqk, bdv0


@jax.jit
def compute_dv0(bdo, bAqk):
  """Compute the Aqk contribution to dV without materializing dAqk."""
  return jnp.matmul(
      bAqk.transpose(0, 2, 1),
      bdo.astype(jnp.float32),
      preferred_element_type=jnp.float32,
  )


def _bwd_mega_kernel(
    chunk_seg_ids_ref,
    orig_cu_seqlens_ref,
    chunk_metadata_ref,
    cp_post_num_ref,
    cp_is_last_ref,
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    Aqk_ref,
    Akk_ref,
    g_ref,
    h_ref,
    do_ref,
    dht_ref,
    initial_state_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    db_ref,
    dg_ref,
    dh0_ref,
    h_all_ref,
    *,
    H,
    B,
    BT,
    BT_PAD,
    K,
    V,
    NT,
    T_INPUT,
    SCALE,
    MB,
    MB_PAD,
    CHUNK_BATCH,
    DMA_INPUT_BUFFER_COUNT,
    DMA_OUTPUT_BUFFER_COUNT,
    DISABLE_RECOMPUTE,
    CP_ACTIVE,
    CP_SIZE,
    CP_AXIS_NAME,
    HAS_H0,
    PACKED_INPUTS,
    MULTI_SEGMENT_PACKED,
    BATCH_FIRST_INPUTS,
    COMPACT_CHUNK_METADATA,
):
  """Single-program M5 kernel with manual HBM<->VMEM DMA.

  Cross-phase state is allocated inside each batch/head-group iteration with
  pl.run_scoped. The enclosing pallas_call has no externally declared
  temporary VMEM.
  """
  compute_dtype = q_ref.dtype
  precision = (
      None if compute_dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST
  )

  def _matmul_low(lhs, rhs, precision=precision):
    return _matmul_in_dtype(lhs, rhs, compute_dtype, precision)

  @jax.jit
  def _dma_copy(src, dst, sem):
    cp = pltpu.make_async_copy(src, dst, sem)
    cp.start()
    cp.wait()

  @jax.jit
  def _dma_in(hbm_ref, vmem_ref, sem, mb_offset, b, c):
    src = hbm_ref.at[
        (
            pl.ds(mb_offset * MB, MB),
            pl.ds(b, 1),
            pl.ds(c, 1),
            pl.ds(None),
            pl.ds(None),
        )
    ]
    _dma_copy(src, vmem_ref, sem)

  @jax.jit
  def _dma_out(vmem_ref, hbm_ref, sem, mb_offset, b, c):
    dst = hbm_ref.at[
        (
            pl.ds(mb_offset * MB, MB),
            pl.ds(b, 1),
            pl.ds(c, 1),
            pl.ds(None),
            pl.ds(None),
        )
    ]
    _dma_copy(vmem_ref, dst, sem)

  def _async_copy(src, dst, sem, wait=False):
    cp = pltpu.make_async_copy(src, dst, sem)
    if wait:
      cp.wait()
    else:
      cp.start()

  def _dma_beta_in(beta_buf, sem, mb_offset, b, c, wait=False):
    """Copy one compact [head-group, chunk] beta tile into VMEM."""
    beta_offset = ((mb_offset * B + b) * NT + c) * MB_PAD
    src = beta_ref.at[(pl.ds(beta_offset, MB_PAD), pl.ds(None))]
    _async_copy(src, beta_buf, sem, wait=wait)

  @jax.jit
  def _dma_state_in(state_ref, vmem_ref, sem, mb_offset, b, seq):
    src = state_ref.at[
        (pl.ds(mb_offset * MB, MB), b, seq, pl.ds(None), pl.ds(None))
    ]
    _dma_copy(src, vmem_ref, sem)

  @jax.jit
  def _dma_state_out(vmem_ref, state_ref, sem, mb_offset, b, seq):
    dst = state_ref.at[
        (pl.ds(mb_offset * MB, MB), b, seq, pl.ds(None), pl.ds(None))
    ]
    _dma_copy(vmem_ref, dst, sem)

  packed_load_rows = BT + 8
  # A packed token DMA loads BT + 8 rows and may shift the window right by
  # at most BT rows at the end of the input. No branch reads beyond this.
  packed_buffer_rows = 2 * BT + 8
  packed_dma_rows = min(T_INPUT, packed_load_rows)

  def _packed_input_window(batch, chunk):
    seq_idx = chunk_metadata_ref[batch, chunk, 0]
    block_idx = chunk_metadata_ref[batch, chunk, 1]
    bos = orig_cu_seqlens_ref[batch, seq_idx]
    eos = orig_cu_seqlens_ref[batch, seq_idx + 1]
    valid_chunk = block_idx * BT < eos - bos
    packed_start = jnp.where(valid_chunk, bos + block_idx * BT, jnp.int32(0))
    window_start = packed_start - jnp.mod(packed_start, jnp.int32(8))
    source_start = jnp.minimum(
        window_start, jnp.int32(max(T_INPUT - packed_dma_rows, 0))
    )
    source_start = source_start - jnp.mod(source_start, jnp.int32(8))
    source_start = pl.multiple_of(source_start, 8)
    source_shift = (window_start - source_start) // jnp.int32(8)
    beta_group = window_start // jnp.int32(BT)
    beta_source_start = beta_group - jnp.mod(beta_group, jnp.int32(8))
    beta_source_start = pl.multiple_of(beta_source_start, 8)
    beta_source_shift = beta_group - beta_source_start
    packed_offset = jnp.mod(packed_start, jnp.int32(8))
    beta_group_offset = jnp.mod(packed_start // jnp.int32(8), jnp.int32(8))
    remaining = eos - packed_start
    return (
        source_start,
        source_shift,
        beta_source_start,
        beta_source_shift,
        packed_offset,
        beta_group_offset,
        remaining,
    )

  def _packed_token_dma_descs(
      buf,
      mb_offset,
      batch,
      chunk,
      q_buf,
      k_buf,
      v_buf,
      do_buf,
      beta_buf,
      sems,
  ):
    source_start, _, beta_source_start, _, _, _, _ = _packed_input_window(
        batch, chunk
    )
    # Batch-first HBM tiles are 8-head tiled. MB is an 8-aligned packed-path
    # specialization, but the dynamic group index needs an explicit Mosaic
    # alignment annotation before it can be used as a DMA slice offset.
    head_start = pl.multiple_of(mb_offset * MB, 8)

    def _token_desc(src_ref, dst_ref, sem):
      src = src_ref.at[
          pl.ds(head_start, MB),
          pl.ds(batch, 1),
          pl.ds(source_start, packed_dma_rows),
          pl.ds(None),
      ]
      dst = dst_ref.at[buf, :, :, pl.ds(0, packed_dma_rows), :]
      return pltpu.make_async_copy(
          src,
          dst,
          sem,
      )

    return (
        _token_desc(q_ref, q_buf, sems.at[0, buf]),
        _token_desc(k_ref, k_buf, sems.at[1, buf]),
        _token_desc(v_ref, v_buf, sems.at[2, buf]),
        _token_desc(do_ref, do_buf, sems.at[3, buf]),
        pltpu.make_async_copy(
            beta_ref.at[
                pl.ds(head_start, MB),
                pl.ds(batch, 1),
                pl.ds(beta_source_start, 16),
                pl.ds(None),
            ],
            beta_buf.at[buf],
            sems.at[4, buf],
        ),
    )

  def _align_packed_tail_window(buf, source_shift, *token_bufs):
    for shift in range(1, BT // 8 + 1):
      row_start = shift * 8

      @pl.when(source_shift == shift)
      def _shift(row_start=row_start):
        for token_buf in token_bufs:
          token_buf[buf, :, :, :packed_load_rows, :] = token_buf[
              buf, :, :, row_start : row_start + packed_load_rows, :
          ]

  def _packed_token_view(token_buf, buf, packed_offset):
    pass
    branches = tuple(
        lambda operand, offset=offset: operand[
            buf, :, 0, offset : offset + BT, :
        ]
        for offset in range(8)
    )
    return jax.lax.switch(packed_offset, branches, token_buf)

  def _packed_beta_view(
      beta_buf, buf, beta_source_shift, beta_group_offset, packed_offset
  ):
    beta_group_branches = tuple(
        lambda operand, shift=shift: operand[buf, :, 0, shift : shift + 2, :]
        for shift in range(8)
    )
    beta_pair = jax.lax.switch(beta_source_shift, beta_group_branches, beta_buf)

    def _beta_aligned_window(x):
      branches = []
      for group_offset in range(8):
        start = group_offset * 8
        head = start + 8
        branches.append(
            lambda operand, start=start, head=head: jnp.concatenate(
                [operand[:, 0, start:BT], operand[:, 1, :head]], axis=1
            )
        )
      return jax.lax.switch(beta_group_offset, tuple(branches), x)

    beta_window = _beta_aligned_window(beta_pair)
    branches = tuple(
        lambda operand, offset=offset: operand[:, offset : offset + BT]
        for offset in range(8)
    )
    return jax.lax.switch(packed_offset, branches, beta_window)

  def main_loop_body(
      batch_idx,
      mb_offset,
      dh_tmp_ref,
      dS_acc_ref=None,
      M_acc_ref=None,
      ag_dS_ref=None,
      ag_buf_ref=None,
  ):
    # Only CP consumes the first/last local segment IDs. Keeping this static
    # NT-wide scan out of the non-CP program materially shortens the Pallas
    # jaxpr embedded in large layer scans.
    if CP_ACTIVE:
      first_seg_id = chunk_seg_ids_ref[batch_idx, 0]
      last_seg_id = first_seg_id
      for seg_i in range(1, NT):
        seg_i_id = chunk_seg_ids_ref[batch_idx, seg_i]
        first_seg_id = jnp.where(
            (first_seg_id == 0) & (seg_i_id != 0), seg_i_id, first_seg_id
        )
        last_seg_id = jnp.where(seg_i_id != 0, seg_i_id, last_seg_id)

    # =================================================================
    # Phase 1: Forward recompute_h  (only when !disable_recompute)
    # =================================================================
    if not DISABLE_RECOMPUTE:

      @functools.partial(
          pl.run_scoped,
          k_fwd_buf=pltpu.VMEM(
              (
                  (1, 1, packed_buffer_rows, MB, K)
                  if BATCH_FIRST_INPUTS
                  else (1, MB, 1, packed_buffer_rows, K)
              )
              if PACKED_INPUTS
              else (MB, 1, 1, BT, K),
              k_ref.dtype,
          ),
          v_fwd_buf=pltpu.VMEM(
              (
                  (1, 1, packed_buffer_rows, MB, V)
                  if BATCH_FIRST_INPUTS
                  else (1, MB, 1, packed_buffer_rows, V)
              )
              if PACKED_INPUTS
              else (MB, 1, 1, BT, V),
              v_ref.dtype,
          ),
          beta_fwd_buf=pltpu.VMEM(
              (1, MB, 1, 16, 128) if PACKED_INPUTS else (MB_PAD, 128),
              beta_ref.dtype,
          ),
          Akk_fwd_buf=pltpu.VMEM((MB, 1, 1, BT, BT_PAD), Akk_ref.dtype),
          g_fwd_buf=pltpu.VMEM((MB, 1, 1, BT, K), g_ref.dtype),
          h_fwd_out_buf=pltpu.VMEM((MB, 1, 1, K, V), jnp.float32),
          g_fwd_scratch=pltpu.VMEM((MB, BT, K), jnp.float32),
          in_sems_fwd=pltpu.SemaphoreType.DMA((5,)),
          out_sem_fwd=pltpu.SemaphoreType.DMA,
          state_sem_fwd=pltpu.SemaphoreType.DMA,
      )
      def _forward_recompute_scoped(
          k_fwd_buf,
          v_fwd_buf,
          beta_fwd_buf,
          Akk_fwd_buf,
          g_fwd_buf,
          h_fwd_out_buf,
          g_fwd_scratch,
          in_sems_fwd,
          out_sem_fwd,
          state_sem_fwd,
      ):
        @pl.loop(0, NT, unroll=False)
        def forward_loop(chunk_id):
          # chunk_id: 0 -> NT-1  (forward scan)
          seg_cur, seq_idx, is_valid, is_first_chunk, is_last_chunk = (
              _chunk_segment_metadata(
                  chunk_seg_ids_ref, batch_idx, chunk_id, NT
              )
          )

          # Match PR #463's all-padding DMA skip. Invalid chunks are masked
          # below, so their VMEM buffers do not need to be refreshed.
          @pl.when(is_valid)
          def _load_forward_chunk():
            _dma_in(
                k_ref,
                k_fwd_buf,
                in_sems_fwd.at[0],
                mb_offset,
                batch_idx,
                chunk_id,
            )
            _dma_in(
                v_ref,
                v_fwd_buf,
                in_sems_fwd.at[1],
                mb_offset,
                batch_idx,
                chunk_id,
            )
            _dma_beta_in(
                beta_fwd_buf,
                in_sems_fwd.at[2],
                mb_offset,
                batch_idx,
                chunk_id,
            )
            _dma_beta_in(
                beta_fwd_buf,
                in_sems_fwd.at[2],
                mb_offset,
                batch_idx,
                chunk_id,
                wait=True,
            )
            _dma_in(
                Akk_ref,
                Akk_fwd_buf,
                in_sems_fwd.at[3],
                mb_offset,
                batch_idx,
                chunk_id,
            )
            _dma_in(
                g_ref,
                g_fwd_buf,
                in_sems_fwd.at[4],
                mb_offset,
                batch_idx,
                chunk_id,
            )

          @pl.when(~is_valid)
          def _zero_forward_chunk():
            k_fwd_buf[:] = jnp.zeros_like(k_fwd_buf[:])
            v_fwd_buf[:] = jnp.zeros_like(v_fwd_buf[:])
            beta_fwd_buf[:] = jnp.zeros_like(beta_fwd_buf[:])
            Akk_fwd_buf[:] = jnp.zeros_like(Akk_fwd_buf[:])
            g_fwd_buf[:] = jnp.zeros_like(g_fwd_buf[:])

          g_fwd_scratch[:] = g_fwd_buf[:, 0, 0]
          g_fwd_scratch[:] = jnp.where(
              is_valid, g_fwd_scratch[:], jnp.zeros_like(g_fwd_scratch[:])
          )

          @pl.when(is_first_chunk)
          def _load_initial():
            if HAS_H0:
              _dma_state_in(
                  initial_state_ref,
                  dh_tmp_ref,
                  state_sem_fwd,
                  mb_offset,
                  batch_idx,
                  0 if CP_ACTIVE else seq_idx,
              )
              if CP_ACTIVE:
                # Tokamax stores the incoming CP state in local slot zero,
                # even when the first segment retains a larger global ID.
                dh_tmp_ref[:] = jnp.where(
                    seg_cur == first_seg_id,
                    dh_tmp_ref[:],
                    jnp.zeros_like(dh_tmp_ref[:]),
                )
            else:
              dh_tmp_ref[:] = jnp.zeros_like(dh_tmp_ref[:])

          @pl.when(is_valid)
          def _recompute_h():
            h_cur = dh_tmp_ref[:].astype(jnp.float32)
            fk = k_fwd_buf[:, 0, 0].astype(compute_dtype)
            fv = v_fwd_buf[:, 0, 0].astype(compute_dtype)
            fb = beta_fwd_buf[:MB, :BT].astype(compute_dtype)
            fA = Akk_fwd_buf[:, 0, 0, :, :BT].astype(compute_dtype)
            fg = g_fwd_scratch[:].astype(jnp.float32)
            fg_exp = jnp.exp2(fg)
            fg_last = fg[:, BT - 1, :]
            fv_beta = fv * fb[:, :, None]
            fu = _matmul_low(fA, fv_beta)
            fw = jnp.matmul(
                fA.astype(jnp.float32),
                (fk * fb[:, :, None] * fg_exp).astype(jnp.float32),
                preferred_element_type=jnp.float32,
            )
            fvn = fu.astype(jnp.float32) - jnp.matmul(
                fw.astype(jnp.float32),
                h_cur,
                precision=precision,
                preferred_element_type=jnp.float32,
            )
            h_fwd_out_buf[:, 0, 0] = h_cur
            _dma_out(
                h_fwd_out_buf,
                h_all_ref,
                out_sem_fwd,
                mb_offset,
                batch_idx,
                chunk_id,
            )
            fkg = fk * jnp.exp2(fg[:, BT - 1 : BT, :] - fg)
            dh_tmp_ref[:] = dh_tmp_ref[:] * jnp.exp2(fg_last)[:, :, None]
            dh_tmp_ref[:] = dh_tmp_ref[:] + jnp.matmul(
                fkg.astype(jnp.float32).transpose(0, 2, 1),
                fvn,
                precision=precision,
                preferred_element_type=jnp.float32,
            )

    # =================================================================
    # Phase 2: Preprocess (dAv + CP accumulate)  (only when cp_active)
    # =================================================================
    if CP_ACTIVE:

      @functools.partial(
          pl.run_scoped,
          q_cp_buf=pltpu.VMEM(
              (2, 1, packed_buffer_rows, MB, K)
              if PACKED_INPUTS
              else (2, MB, 1, 1, BT, K),
              q_ref.dtype,
          ),
          k_cp_buf=pltpu.VMEM(
              (2, 1, packed_buffer_rows, MB, K)
              if PACKED_INPUTS
              else (2, MB, 1, 1, BT, K),
              k_ref.dtype,
          ),
          do_cp_buf=pltpu.VMEM(
              (2, 1, packed_buffer_rows, MB, V)
              if PACKED_INPUTS
              else (2, MB, 1, 1, BT, V),
              do_ref.dtype,
          ),
          beta_cp_buf=pltpu.VMEM(
              (2, MB, 1, 16, 128) if PACKED_INPUTS else (2, MB_PAD, 128),
              beta_ref.dtype,
          ),
          Aqk_cp_buf=pltpu.VMEM((2, MB, 1, 1, BT, BT_PAD), Aqk_ref.dtype),
          Akk_cp_buf=pltpu.VMEM((2, MB, 1, 1, BT, BT_PAD), Akk_ref.dtype),
          g_cp_buf=pltpu.VMEM((2, MB, 1, 1, BT, K), g_ref.dtype),
          g_cp_scratch=pltpu.VMEM((MB, BT, K), jnp.float32),
          in_sems_cp=pltpu.SemaphoreType.DMA((7, 2)),
      )
      def _cp_preprocess_scoped(
          q_cp_buf,
          k_cp_buf,
          do_cp_buf,
          beta_cp_buf,
          Aqk_cp_buf,
          Akk_cp_buf,
          g_cp_buf,
          g_cp_scratch,
          in_sems_cp,
      ):
        def _dma_in_cp(hbm_ref, vmem_ref, sem, buf, c, wait=False):
          src = hbm_ref.at[
              (
                  pl.ds(mb_offset * MB, MB),
                  pl.ds(batch_idx, 1),
                  pl.ds(c, 1),
                  pl.ds(None),
                  pl.ds(None),
              )
          ]
          _async_copy(src, vmem_ref.at[buf], sem, wait=wait)

        def _packed_cp_input_descs(buf, chunk_id):
          source_start, _, beta_source_start, _, _, _, _ = _packed_input_window(
              batch_idx, chunk_id
          )
          head_start = pl.multiple_of(mb_offset * MB, 8)

          def _token_desc(src_ref, dst_ref, sem):
            return pltpu.make_async_copy(
                src_ref.at[
                    pl.ds(batch_idx, 1),
                    pl.ds(source_start, packed_dma_rows),
                    pl.ds(head_start, MB),
                    pl.ds(None),
                ],
                dst_ref.at[buf, :, pl.ds(0, packed_dma_rows), :, :],
                sem,
            )

          return (
              _token_desc(q_ref, q_cp_buf, in_sems_cp.at[0, buf]),
              _token_desc(k_ref, k_cp_buf, in_sems_cp.at[1, buf]),
              _token_desc(do_ref, do_cp_buf, in_sems_cp.at[5, buf]),
              pltpu.make_async_copy(
                  beta_ref.at[
                      pl.ds(head_start, MB),
                      pl.ds(batch_idx, 1),
                      pl.ds(beta_source_start, 16),
                      pl.ds(None),
                  ],
                  beta_cp_buf.at[buf],
                  in_sems_cp.at[2, buf],
              ),
          )

        def _start_cp_input_dma(buf, chunk_id, is_valid):
          @pl.when(is_valid)
          def _start_valid_cp_input_dma():
            _dma_in_cp(q_ref, q_cp_buf, in_sems_cp.at[0, buf], buf, chunk_id)
            _dma_in_cp(k_ref, k_cp_buf, in_sems_cp.at[1, buf], buf, chunk_id)
            _dma_beta_in(
                beta_cp_buf.at[buf],
                in_sems_cp.at[2, buf],
                mb_offset,
                batch_idx,
                chunk_id,
            )
            _dma_in_cp(
                Aqk_ref, Aqk_cp_buf, in_sems_cp.at[3, buf], buf, chunk_id
            )
            _dma_in_cp(
                Akk_ref, Akk_cp_buf, in_sems_cp.at[4, buf], buf, chunk_id
            )
            _dma_in_cp(do_ref, do_cp_buf, in_sems_cp.at[5, buf], buf, chunk_id)
            _dma_in_cp(g_ref, g_cp_buf, in_sems_cp.at[6, buf], buf, chunk_id)

          @pl.when(~is_valid)
          def _zero_invalid_cp_inputs():
            q_cp_buf[buf] = jnp.zeros_like(q_cp_buf[buf])
            k_cp_buf[buf] = jnp.zeros_like(k_cp_buf[buf])
            beta_cp_buf[buf] = jnp.zeros_like(beta_cp_buf[buf])
            Aqk_cp_buf[buf] = jnp.zeros_like(Aqk_cp_buf[buf])
            Akk_cp_buf[buf] = jnp.zeros_like(Akk_cp_buf[buf])
            do_cp_buf[buf] = jnp.zeros_like(do_cp_buf[buf])
            g_cp_buf[buf] = jnp.zeros_like(g_cp_buf[buf])

        def _wait_cp_input_dma(buf, chunk_id, is_valid):
          @pl.when(is_valid)
          def _wait_valid_cp_input_dma():
            _dma_in_cp(
                q_ref, q_cp_buf, in_sems_cp.at[0, buf], buf, chunk_id, wait=True
            )
            _dma_in_cp(
                k_ref, k_cp_buf, in_sems_cp.at[1, buf], buf, chunk_id, wait=True
            )
            _dma_beta_in(
                beta_cp_buf.at[buf],
                in_sems_cp.at[2, buf],
                mb_offset,
                batch_idx,
                chunk_id,
                wait=True,
            )
            _dma_in_cp(
                Aqk_ref,
                Aqk_cp_buf,
                in_sems_cp.at[3, buf],
                buf,
                chunk_id,
                wait=True,
            )
            _dma_in_cp(
                Akk_ref,
                Akk_cp_buf,
                in_sems_cp.at[4, buf],
                buf,
                chunk_id,
                wait=True,
            )
            _dma_in_cp(
                do_ref,
                do_cp_buf,
                in_sems_cp.at[5, buf],
                buf,
                chunk_id,
                wait=True,
            )
            _dma_in_cp(
                g_ref, g_cp_buf, in_sems_cp.at[6, buf], buf, chunk_id, wait=True
            )

        first_segment_chunks = NT
        cp_preprocess_nt = NT

        first_chunk_id = jnp.maximum(first_segment_chunks - 1, 0)
        _, _, first_is_valid, _, _ = _chunk_segment_metadata(
            chunk_seg_ids_ref, batch_idx, first_chunk_id, NT
        )
        _start_cp_input_dma(0, first_chunk_id, first_is_valid)

        @pl.loop(0, cp_preprocess_nt, unroll=False)
        def preprocess_loop(rev_idx):
          chunk_id = jnp.maximum(first_segment_chunks - 1 - rev_idx, 0)
          buf = rev_idx % 2
          next_buf = 1 - buf

          seg_cur, seq_idx, is_valid, is_first_chunk, is_last_chunk = (
              _chunk_segment_metadata(
                  chunk_seg_ids_ref, batch_idx, chunk_id, NT
              )
          )
          is_first_local_segment = is_valid & (seg_cur == first_seg_id)

          _wait_cp_input_dma(buf, chunk_id, is_valid)

          @pl.when(rev_idx + 1 < cp_preprocess_nt)
          def _prefetch_next_cp_chunk():
            next_chunk_id = jnp.maximum(first_segment_chunks - 2 - rev_idx, 0)
            _, _, next_is_valid, _, _ = _chunk_segment_metadata(
                chunk_seg_ids_ref, batch_idx, next_chunk_id, NT
            )
            _start_cp_input_dma(next_buf, next_chunk_id, next_is_valid)

          g_cp_scratch[:] = g_cp_buf[buf, :, 0, 0]
          g_cp_scratch[:] = jnp.where(
              is_valid, g_cp_scratch[:], jnp.zeros_like(g_cp_scratch[:])
          )

          # Init CP accumulators at first chunk (rev_idx == 0 -> chunk_id == NT-1)
          @pl.when(rev_idx == 0)
          def _init_cp_acc():
            dS_acc_ref[:] = jnp.zeros((MB, K, V), dtype=jnp.float32)
            M_acc_ref[:] = jnp.broadcast_to(
                jnp.eye(K, dtype=jnp.float32), (MB, K, K)
            )

          bq = q_cp_buf[buf, :, 0, 0].astype(compute_dtype)
          bk = k_cp_buf[buf, :, 0, 0].astype(compute_dtype)
          bdo = do_cp_buf[buf, :, 0, 0].astype(compute_dtype)
          bb = beta_cp_buf[buf, :MB, :BT].astype(compute_dtype)
          bg = g_cp_scratch[:].astype(jnp.float32)
          bA = Akk_cp_buf[buf, :, 0, 0, :, :BT].astype(compute_dtype)
          bAqk = Aqk_cp_buf[buf, :, 0, 0, :, :BT].astype(compute_dtype)

          bq = jnp.where(is_valid, bq, jnp.zeros_like(bq))
          bk = jnp.where(is_valid, bk, jnp.zeros_like(bk))
          bb = jnp.where(is_valid, bb, jnp.zeros_like(bb))
          bA = jnp.where(is_valid, bA, jnp.zeros_like(bA))
          bAqk = jnp.where(is_valid, bAqk, jnp.zeros_like(bAqk))
          bdo = jnp.where(is_valid, bdo, jnp.zeros_like(bdo))

          # CP preprocessing only consumes dV. Avoid loading v/h and skip
          # their M1 contractions; the full backward stage recomputes them.
          bdv0 = compute_dv0(bdo, bAqk)
          bkg = bk * jnp.exp2(bg[:, BT - 1 : BT, :] - bg)
          bw = jnp.matmul(
              bA.astype(jnp.float32),
              (bk * bb[:, :, None] * jnp.exp2(bg)).astype(jnp.float32),
              preferred_element_type=jnp.float32,
          )
          bqg = bq * jnp.exp2(bg)

          @pl.when(is_first_local_segment)
          def _cp_preprocess():
            cp_decay = jnp.exp2(
                jnp.maximum(
                    bg[:, BT - 1, :], jnp.asarray(-126.0, dtype=jnp.float32)
                )
            )
            dv_cur = (
                jnp.matmul(
                    bkg, dS_acc_ref[:], preferred_element_type=jnp.float32
                )
                + bdv0
            )
            dS_acc_ref[:] = (
                dS_acc_ref[:] * cp_decay[:, :, None]
                + jnp.matmul(
                    bqg.transpose(0, 2, 1),
                    bdo,
                    preferred_element_type=jnp.float32,
                )
                * SCALE
                - jnp.matmul(
                    bw.transpose(0, 2, 1),
                    dv_cur,
                    preferred_element_type=jnp.float32,
                )
            )
            kM = jnp.matmul(
                bkg,
                M_acc_ref[:],
                precision=precision,
                preferred_element_type=jnp.float32,
            )
            WkM = jnp.matmul(
                bw.transpose(0, 2, 1),
                kM,
                precision=precision,
                preferred_element_type=jnp.float32,
            )
            M_acc_ref[:] = M_acc_ref[:] * cp_decay[:, :, None] - WkM

      # --- CP transition: ring all-gather + merge ---
      my_id = jax.lax.axis_index(CP_AXIS_NAME)

      ag_dS_ref[my_id] = dS_acc_ref[:]
      ag_buf_ref[my_id, :, :, :V] = dS_acc_ref[:]
      ag_buf_ref[my_id, :, :, V : V + K] = M_acc_ref[:]

      post_meta = cp_post_num_ref[batch_idx]
      my_post_num = post_meta & 0xFF
      max_post_num = post_meta >> 8

      @functools.partial(
          pl.run_scoped,
          dS_send=pltpu.SemaphoreType.DMA,
          dS_recv=pltpu.SemaphoreType.DMA((CP_SIZE - 1,)),
          full_send=pltpu.SemaphoreType.DMA,
          full_recv=pltpu.SemaphoreType.DMA((CP_SIZE - 1,)),
      )
      def _cp_ring(dS_send, dS_recv, full_send, full_recv):
        right = (my_id + 1) % CP_SIZE
        left = (my_id + CP_SIZE - 1) % CP_SIZE

        def _sync_ring_start():
          # Remote DMA writes into each peer's run_scoped VMEM. On multi-host
          # meshes, wait until every rank has entered the scope and initialized
          # its local ring slots before any peer starts transferring into them.
          barrier_sem = pltpu.get_barrier_semaphore()
          for peer in range(CP_SIZE):
            pl.semaphore_signal(
                barrier_sem,
                device_id={CP_AXIS_NAME: peer},
                device_id_type=pl.DeviceIdType.MESH,
            )
          pl.semaphore_wait(barrier_sem, CP_SIZE)

        _sync_ring_start()

        @pl.when((max_post_num > 0) & (max_post_num <= 1))
        def _fetch_adjacent_dS():
          # A chain spanning one CP boundary needs only the next rank's dS.
          # Send each local slot left so every rank receives that value in the
          # exact slot consumed by _merge_adjacent, avoiding a full ring.
          op = pltpu.make_async_remote_copy(
              src_ref=ag_dS_ref.at[my_id],
              dst_ref=ag_dS_ref.at[my_id],
              send_sem=dS_send,
              recv_sem=dS_recv.at[0],
              device_id={CP_AXIS_NAME: left},
              device_id_type=pl.DeviceIdType.MESH,
          )
          op.start()
          op.wait()

        @pl.when(max_post_num > 1)
        def _gather_full_state():
          for step in range(CP_SIZE - 1):
            slot = (my_id - step) % CP_SIZE
            op = pltpu.make_async_remote_copy(
                src_ref=ag_buf_ref.at[slot],
                dst_ref=ag_buf_ref.at[slot],
                send_sem=full_send,
                recv_sem=full_recv.at[step],
                device_id={CP_AXIS_NAME: right},
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            op.wait()

      my_is_last = cp_is_last_ref[batch_idx]

      @pl.when(max_post_num <= 1)
      def _merge_adjacent():
        idx = jnp.minimum(my_id + 1, CP_SIZE - 1)
        dh_tmp_ref[:] = jnp.where(
            (my_post_num == 1) & (my_is_last == 0),
            ag_dS_ref[idx],
            jnp.zeros((MB, K, V), dtype=jnp.float32),
        )

      @pl.when(max_post_num > 1)
      def _merge_general():
        merged = jnp.zeros((MB, K, V), dtype=jnp.float32)
        for i in range(CP_SIZE):
          idx = jnp.clip(my_id + my_post_num - i, 0, CP_SIZE - 1)
          slot = ag_buf_ref[idx]
          dS_i = slot[:, :, :V]
          dM_i = slot[:, :, V : V + K]
          new_merged = (
              jnp.matmul(dM_i, merged, preferred_element_type=jnp.float32)
              + dS_i
          )
          active = (i < my_post_num) & (my_is_last == 0)
          merged = jnp.where(active, new_merged, merged)
        dh_tmp_ref[:] = merged

    # =================================================================
    # Phase 3: Backward (dhu + WY + intra + cumsum + write)  (always)
    # =================================================================
    @functools.partial(
        pl.run_scoped,
        q_buf=pltpu.VMEM(
            (1, 1, 1, 1, 1, 1)
            if PACKED_INPUTS
            else (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, K),
            q_ref.dtype,
        ),
        k_buf=pltpu.VMEM(
            (1, 1, 1, 1, 1, 1)
            if PACKED_INPUTS
            else (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, K),
            k_ref.dtype,
        ),
        v_buf=pltpu.VMEM(
            (1, 1, 1, 1, 1, 1)
            if PACKED_INPUTS
            else (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, V),
            v_ref.dtype,
        ),
        do_buf=pltpu.VMEM(
            (1, 1, 1, 1, 1, 1)
            if PACKED_INPUTS
            else (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, V),
            do_ref.dtype,
        ),
        beta_buf=pltpu.VMEM(
            (1, 1)
            if PACKED_INPUTS
            else (DMA_INPUT_BUFFER_COUNT, CHUNK_BATCH * MB_PAD, 128),
            beta_ref.dtype,
        ),
        packed_q_buf=pltpu.VMEM(
            (
                (2, 1, packed_buffer_rows, MB, K)
                if BATCH_FIRST_INPUTS
                else (2, MB, 1, packed_buffer_rows, K)
            )
            if PACKED_INPUTS
            else (1, 1, 1, 1, 1),
            q_ref.dtype,
        ),
        packed_k_buf=pltpu.VMEM(
            (
                (2, 1, packed_buffer_rows, MB, K)
                if BATCH_FIRST_INPUTS
                else (2, MB, 1, packed_buffer_rows, K)
            )
            if PACKED_INPUTS
            else (1, 1, 1, 1, 1),
            k_ref.dtype,
        ),
        packed_v_buf=pltpu.VMEM(
            (
                (2, 1, packed_buffer_rows, MB, V)
                if BATCH_FIRST_INPUTS
                else (2, MB, 1, packed_buffer_rows, V)
            )
            if PACKED_INPUTS
            else (1, 1, 1, 1, 1),
            v_ref.dtype,
        ),
        packed_do_buf=pltpu.VMEM(
            (
                (2, 1, packed_buffer_rows, MB, V)
                if BATCH_FIRST_INPUTS
                else (2, MB, 1, packed_buffer_rows, V)
            )
            if PACKED_INPUTS
            else (1, 1, 1, 1, 1),
            do_ref.dtype,
        ),
        packed_beta_buf=pltpu.VMEM(
            (2, MB, 1, 16, 128) if PACKED_INPUTS else (1, 1, 1, 1, 1),
            beta_ref.dtype,
        ),
        Aqk_buf=pltpu.VMEM(
            (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, BT_PAD),
            Aqk_ref.dtype,
        ),
        Akk_buf=pltpu.VMEM(
            (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, BT_PAD),
            Akk_ref.dtype,
        ),
        g_buf=pltpu.VMEM(
            (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, K), g_ref.dtype
        ),
        h_buf=pltpu.VMEM(
            (DMA_INPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, K, V), h_ref.dtype
        ),
        # Keep output staging independent from the input ping-pong buffers so
        # the next input prefetch never depends on the previous output DMA.
        dq_buf=pltpu.VMEM(
            (DMA_OUTPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, K), dq_ref.dtype
        ),
        dk_buf=pltpu.VMEM(
            (DMA_OUTPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, K), dk_ref.dtype
        ),
        dv_buf=pltpu.VMEM(
            (DMA_OUTPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, V), dv_ref.dtype
        ),
        db_buf=pltpu.VMEM(
            (DMA_OUTPUT_BUFFER_COUNT, CHUNK_BATCH * MB_PAD, 128), db_ref.dtype
        ),
        dg_buf=pltpu.VMEM(
            (DMA_OUTPUT_BUFFER_COUNT, MB, 1, CHUNK_BATCH, BT, K), dg_ref.dtype
        ),
        in_sems_bwd=pltpu.SemaphoreType.DMA((9, DMA_INPUT_BUFFER_COUNT)),
        packed_in_sems_bwd=pltpu.SemaphoreType.DMA((5, 2)),
        out_sems_bwd=pltpu.SemaphoreType.DMA((5, DMA_OUTPUT_BUFFER_COUNT)),
        state_sems_bwd=pltpu.SemaphoreType.DMA((3,)),
    )
    def _backward_scoped(
        q_buf,
        k_buf,
        v_buf,
        do_buf,
        beta_buf,
        packed_q_buf,
        packed_k_buf,
        packed_v_buf,
        packed_do_buf,
        packed_beta_buf,
        Aqk_buf,
        Akk_buf,
        g_buf,
        h_buf,
        dq_buf,
        dk_buf,
        dv_buf,
        db_buf,
        dg_buf,
        in_sems_bwd,
        packed_in_sems_bwd,
        out_sems_bwd,
        state_sems_bwd,
    ):
      # Every positive CHUNK_BATCH uses the contiguous-window path, including CB1.
      # Slots are consumed in reverse order so the cross-chunk dh recurrence
      # remains sequential.
      LOOP_NUM = pl.cdiv(NT, CHUNK_BATCH)
      REM = NT % CHUNK_BATCH
      DB_NT = LOOP_NUM * CHUNK_BATCH
      DB_TAIL_PAD = 0 if REM == 0 else CHUNK_BATCH - REM

      def start_dmas(descs):
        for desc in descs:
          desc.start()

      def wait_dmas(descs):
        for desc in descs:
          desc.wait()

      def dma_in_batch_desc(hbm_ref, vmem_ref, sem, buf, start_c):
        src = hbm_ref.at[
            (
                pl.ds(mb_offset * MB, MB),
                pl.ds(batch_idx, 1),
                pl.ds(start_c, CHUNK_BATCH),
                pl.ds(None),
                pl.ds(None),
            )
        ]
        return pltpu.make_async_copy(src, vmem_ref.at[buf], sem)

      def dma_in_tail_desc(hbm_ref, vmem_ref, sem, buf):
        src = hbm_ref.at[
            (
                pl.ds(mb_offset * MB, MB),
                pl.ds(batch_idx, 1),
                pl.ds(0, REM),
                pl.ds(None),
                pl.ds(None),
            )
        ]
        return pltpu.make_async_copy(
            src, vmem_ref.at[buf, :, :, :REM, :, :], sem
        )

      def dma_in_batch_beta_desc(buf, start_c):
        beta_offset = ((mb_offset * B + batch_idx) * NT + start_c) * MB_PAD
        packed_heads = CHUNK_BATCH * MB_PAD
        src = beta_ref.at[(pl.ds(beta_offset, packed_heads), pl.ds(None))]
        return pltpu.make_async_copy(
            src, beta_buf.at[buf], in_sems_bwd.at[3, buf]
        )

      def dma_out_batch_desc(vmem_ref, hbm_ref, sem, buf, start_c):
        dst = hbm_ref.at[
            (
                pl.ds(mb_offset * MB, MB),
                pl.ds(batch_idx, 1),
                pl.ds(start_c, CHUNK_BATCH),
                pl.ds(None),
                pl.ds(None),
            )
        ]
        return pltpu.make_async_copy(vmem_ref.at[buf], dst, sem)

      def dma_out_tail_desc(vmem_ref, hbm_ref, sem, buf):
        dst = hbm_ref.at[
            (
                pl.ds(mb_offset * MB, MB),
                pl.ds(batch_idx, 1),
                pl.ds(0, REM),
                pl.ds(None),
                pl.ds(None),
            )
        ]
        return pltpu.make_async_copy(
            vmem_ref.at[buf, :, :, :REM, :, :], dst, sem
        )

      def dma_out_batch_db_desc(buf, start_c):
        db_offset = (
            (mb_offset * B + batch_idx) * DB_NT + start_c + DB_TAIL_PAD
        ) * MB_PAD
        packed_heads = CHUNK_BATCH * MB_PAD
        dst = db_ref.at[(pl.ds(db_offset, packed_heads), pl.ds(None))]
        return pltpu.make_async_copy(
            db_buf.at[buf], dst, out_sems_bwd.at[3, buf]
        )

      def dma_out_tail_db_desc(buf):
        # A complete beta/db tile avoids Jellyfish's failing masked BF16
        # StridedMemcopy path. Guard chunks are removed by the wrapper.
        db_offset = ((mb_offset * B + batch_idx) * DB_NT) * MB_PAD
        packed_heads = CHUNK_BATCH * MB_PAD
        dst = db_ref.at[(pl.ds(db_offset, packed_heads), pl.ds(None))]
        return pltpu.make_async_copy(
            db_buf.at[buf], dst, out_sems_bwd.at[3, buf]
        )

      def input_batch_descs(buf, start_c):
        token_descs = (
            []
            if PACKED_INPUTS
            else [
                dma_in_batch_desc(
                    q_ref, q_buf, in_sems_bwd.at[0, buf], buf, start_c
                ),
                dma_in_batch_desc(
                    k_ref, k_buf, in_sems_bwd.at[1, buf], buf, start_c
                ),
                dma_in_batch_desc(
                    v_ref, v_buf, in_sems_bwd.at[2, buf], buf, start_c
                ),
                dma_in_batch_beta_desc(buf, start_c),
                dma_in_batch_desc(
                    do_ref, do_buf, in_sems_bwd.at[6, buf], buf, start_c
                ),
            ]
        )
        descs = token_descs + [
            dma_in_batch_desc(
                Aqk_ref, Aqk_buf, in_sems_bwd.at[4, buf], buf, start_c
            ),
            dma_in_batch_desc(
                Akk_ref, Akk_buf, in_sems_bwd.at[5, buf], buf, start_c
            ),
            dma_in_batch_desc(
                g_ref, g_buf, in_sems_bwd.at[8, buf], buf, start_c
            ),
            dma_in_batch_desc(
                h_ref if DISABLE_RECOMPUTE else h_all_ref,
                h_buf,
                in_sems_bwd.at[7, buf],
                buf,
                start_c,
            ),
        ]
        return descs

      def input_tail_descs(buf):
        # The regular operands use one REM-wide DMA. beta deliberately
        # transfers a complete CHUNK_BATCH tile into the complete buffer;
        # this is still one tail DMA and avoids the BF16 prefix-DMA crash.
        token_descs = (
            []
            if PACKED_INPUTS
            else [
                dma_in_tail_desc(q_ref, q_buf, in_sems_bwd.at[0, buf], buf),
                dma_in_tail_desc(k_ref, k_buf, in_sems_bwd.at[1, buf], buf),
                dma_in_tail_desc(v_ref, v_buf, in_sems_bwd.at[2, buf], buf),
                dma_in_batch_beta_desc(buf, 0),
                dma_in_tail_desc(do_ref, do_buf, in_sems_bwd.at[6, buf], buf),
            ]
        )
        descs = token_descs + [
            dma_in_tail_desc(Aqk_ref, Aqk_buf, in_sems_bwd.at[4, buf], buf),
            dma_in_tail_desc(Akk_ref, Akk_buf, in_sems_bwd.at[5, buf], buf),
            dma_in_tail_desc(g_ref, g_buf, in_sems_bwd.at[8, buf], buf),
            dma_in_tail_desc(
                h_ref if DISABLE_RECOMPUTE else h_all_ref,
                h_buf,
                in_sems_bwd.at[7, buf],
                buf,
            ),
        ]
        return descs

      def output_batch_descs(buf, start_c):
        return (
            dma_out_batch_desc(
                dq_buf, dq_ref, out_sems_bwd.at[0, buf], buf, start_c
            ),
            dma_out_batch_desc(
                dk_buf, dk_ref, out_sems_bwd.at[1, buf], buf, start_c
            ),
            dma_out_batch_desc(
                dv_buf, dv_ref, out_sems_bwd.at[2, buf], buf, start_c
            ),
            dma_out_batch_db_desc(buf, start_c),
            dma_out_batch_desc(
                dg_buf, dg_ref, out_sems_bwd.at[4, buf], buf, start_c
            ),
        )

      def output_tail_descs(buf):
        return (
            dma_out_tail_desc(dq_buf, dq_ref, out_sems_bwd.at[0, buf], buf),
            dma_out_tail_desc(dk_buf, dk_ref, out_sems_bwd.at[1, buf], buf),
            dma_out_tail_desc(dv_buf, dv_ref, out_sems_bwd.at[2, buf], buf),
            dma_out_tail_db_desc(buf),
            dma_out_tail_desc(dg_buf, dg_ref, out_sems_bwd.at[4, buf], buf),
        )

      def input_window_has_valid(full_start_c, is_tail):
        full_has_valid = _chunk_window_has_valid(
            chunk_seg_ids_ref,
            batch_idx,
            full_start_c,
            CHUNK_BATCH,
        )
        if REM > 0:
          tail_has_valid = _chunk_window_has_valid(
              chunk_seg_ids_ref, batch_idx, 0, REM
          )
          return jnp.where(is_tail, tail_has_valid, full_has_valid)
        return full_has_valid

      def start_input_window(buf, full_start_c, is_tail, has_valid):
        if REM > 0:

          @pl.when(jnp.logical_and(is_tail, has_valid))
          def _():
            start_dmas(input_tail_descs(buf))

          @pl.when(jnp.logical_and(jnp.logical_not(is_tail), has_valid))
          def _():
            start_dmas(input_batch_descs(buf, full_start_c))

        else:

          @pl.when(has_valid)
          def _():
            start_dmas(input_batch_descs(buf, full_start_c))

      def wait_input_window(buf, full_start_c, is_tail, has_valid):
        if REM > 0:

          @pl.when(jnp.logical_and(is_tail, has_valid))
          def _():
            wait_dmas(input_tail_descs(buf))

          @pl.when(jnp.logical_and(jnp.logical_not(is_tail), has_valid))
          def _():
            wait_dmas(input_batch_descs(buf, full_start_c))

        else:

          @pl.when(has_valid)
          def _():
            wait_dmas(input_batch_descs(buf, full_start_c))

      def start_packed_chunk(buf, chunk_id):
        pass

      def wait_packed_chunk(buf, chunk_id):
        pass

      def packed_chunk_operands(buf, chunk_id):
        (
            _,
            source_shift,
            _,
            beta_source_shift,
            packed_offset,
            beta_group_offset,
            remaining,
        ) = _packed_input_window(batch_idx, chunk_id)
        _align_packed_tail_window(
            buf,
            source_shift,
            packed_q_buf,
            packed_k_buf,
            packed_v_buf,
            packed_do_buf,
        )
        pq = _packed_token_view(packed_q_buf, buf, packed_offset)
        pk = _packed_token_view(packed_k_buf, buf, packed_offset)
        pv = _packed_token_view(packed_v_buf, buf, packed_offset)
        pdo = _packed_token_view(packed_do_buf, buf, packed_offset)
        pbeta = _packed_beta_view(
            packed_beta_buf,
            buf,
            beta_source_shift,
            beta_group_offset,
            packed_offset,
        )

        pq_valid = jax.lax.broadcasted_iota(jnp.int32, pq.shape, 1) < remaining
        pk_valid = jax.lax.broadcasted_iota(jnp.int32, pk.shape, 1) < remaining
        pv_valid = jax.lax.broadcasted_iota(jnp.int32, pv.shape, 1) < remaining
        pdo_valid = (
            jax.lax.broadcasted_iota(jnp.int32, pdo.shape, 1) < remaining
        )
        pbeta_valid = (
            jax.lax.broadcasted_iota(jnp.int32, pbeta.shape, 1) < remaining
        )
        pq = jnp.where(pq_valid, pq, jnp.zeros_like(pq))
        pk = jnp.where(pk_valid, pk, jnp.zeros_like(pk))
        pv = jnp.where(pv_valid, pv, jnp.zeros_like(pv))
        pdo = jnp.where(pdo_valid, pdo, jnp.zeros_like(pdo))
        pbeta = jnp.where(pbeta_valid, pbeta, jnp.zeros_like(pbeta))
        return pq, pk, pv, pbeta, pdo

      def compute_chunk(
          q_chunk_ref,
          k_chunk_ref,
          v_chunk_ref,
          beta_chunk_ref,
          Aqk_chunk_ref,
          Akk_chunk_ref,
          g_chunk_ref,
          h_chunk_ref,
          do_chunk_ref,
          dq_chunk_ref,
          dk_chunk_ref,
          dv_chunk_ref,
          db_chunk_ref,
          dg_chunk_ref,
          chunk_id,
      ):
        with jax.named_scope("_chunk_segment_metadata"):
          seg_cur, seq_idx, is_valid, is_first_chunk, is_last_chunk = (
              _chunk_segment_metadata(
                  chunk_seg_ids_ref, batch_idx, chunk_id, NT
              )
          )

        if CP_ACTIVE:
          is_last_local_segment = is_valid & (seg_cur == last_seg_id)

          @pl.when(is_last_chunk & (~is_last_local_segment))
          def _load_dht_cp():
            if dht_ref is None:
              dh_tmp_ref[:] = jnp.zeros(dh_tmp_ref.shape, dh_tmp_ref.dtype)
            else:
              _dma_state_in(
                  dht_ref, dh_tmp_ref, state_sems_bwd.at[1],
                  mb_offset, batch_idx, seq_idx,
              )

        else:

          @pl.when(is_last_chunk)
          def load_dht():
            with jax.named_scope("_dma_state_in"):
              if dht_ref is None:
                dh_tmp_ref[:] = jnp.zeros(dh_tmp_ref.shape, dh_tmp_ref.dtype)
              else:
                _dma_state_in(
                    dht_ref, dh_tmp_ref, state_sems_bwd.at[1],
                    mb_offset, batch_idx, seq_idx,
                )

        @pl.when(jnp.logical_not(is_valid))
        def _zero_invalid_slot():
          dq_chunk_ref[:] = jnp.zeros_like(dq_chunk_ref[:])
          dk_chunk_ref[:] = jnp.zeros_like(dk_chunk_ref[:])
          dv_chunk_ref[:] = jnp.zeros_like(dv_chunk_ref[:])
          db_chunk_ref[:] = jnp.zeros_like(db_chunk_ref[:])
          dg_chunk_ref[:] = jnp.zeros_like(dg_chunk_ref[:])

        @pl.when(is_valid)
        def compute_valid_chunk():
          with jax.named_scope("mega_compute_prepare"):
            bq = q_chunk_ref[:].astype(compute_dtype)
            bk = k_chunk_ref[:].astype(compute_dtype)
            bv = v_chunk_ref[:].astype(compute_dtype)
            bg = g_chunk_ref[:].astype(jnp.float32)
            bb = (
                beta_chunk_ref[:].astype(compute_dtype)
                if PACKED_INPUTS
                else beta_chunk_ref[pl.ds(0, MB), :BT].astype(compute_dtype)
            )
            bA = Akk_chunk_ref[:, :, :BT].astype(compute_dtype)
            bdo = do_chunk_ref[:].astype(compute_dtype)
            bh = h_chunk_ref[:].astype(jnp.float32)
            g_exp_last = jnp.exp2(bg[:, BT - 1, :])

          with jax.named_scope("mega_compute_m1"):
            bAqk = Aqk_chunk_ref[:, :, :BT].astype(compute_dtype)
            _, w, bvn, bqg, bkg = compute_m1_recompute(
                bq,
                bk,
                bv,
                bb,
                bA,
                bg,
                bh,
                dtype=compute_dtype,
                compute_dtype=compute_dtype,
            )
            bw = w.astype(jnp.float32)
          with jax.named_scope("mega_compute_dav"):
            bdAqk, bdv0 = compute_dav(bdo, bvn, bAqk, SCALE)
          del bAqk, w

          # Finish the recurrence before WY/intra so bw/bqg do not
          # remain live across the largest contractions.
          with jax.named_scope("mega_compute_state_dv"):
            dh = dh_tmp_ref[:].astype(jnp.float32)
            bdv = (
                jnp.matmul(
                    bkg.astype(jnp.float32),
                    dh,
                    preferred_element_type=jnp.float32,
                )
                + bdv0
            )
          with jax.named_scope("mega_compute_dh"):
            dh_next = dh * g_exp_last[:, :, None]
            dh_next = dh_next + jnp.matmul(
                jnp.concatenate([bqg * SCALE, -bw], axis=1)
                .astype(jnp.float32)
                .transpose(0, 2, 1),
                jnp.concatenate([bdo.astype(jnp.float32), bdv], axis=1),
                preferred_element_type=jnp.float32,
            )
            dh_tmp_ref[:] = dh_next
          del bw, bqg, bkg, bdv0, g_exp_last, dh_next

          with jax.named_scope("mega_compute_wy"):
            (
                dq_acc,
                dk_acc,
                b_dvb,
                db_acc,
                dg_acc,
                dAkk_local,
                b_dgk,
            ) = compute_wy_backward(
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
                SCALE,
                precision,
                return_b_dgk=True,
            )
          del bdo, bdv, bvn, bv, bh, dh, bA

          # Consume dV immediately instead of carrying b_dvb through
          # the intra contraction and reverse cumsum.
          with jax.named_scope("mega_compute_output"):
            dv_chunk_ref[:] = (b_dvb * bb[:, :, None]).astype(
                dv_chunk_ref.dtype
            )
          del b_dvb

          with jax.named_scope("mega_compute_intra"):
            db_chunk_ref[:] = jnp.zeros_like(db_chunk_ref[:])
            compute_intra_backward_sequential_to_chunk_refs(
                bq,
                bk,
                bg,
                bb,
                bdAqk,
                dAkk_local,
                dq_acc,
                dk_acc,
                db_acc,
                dg_acc,
                dq_chunk_ref,
                dk_chunk_ref,
                db_chunk_ref,
                dg_chunk_ref,
                operand_dtype=q_ref.dtype,
                b_dgk=b_dgk,
            )
          del dq_acc, dk_acc, db_acc, dg_acc, dAkk_local
          del bq, bk, bg, bb, bdAqk, b_dgk

          with jax.named_scope("mega_compute_cumsum"):
            dg_total = dg_chunk_ref[:].astype(jnp.float32)
            dg_reverse_cumsum = compute_reverse_cumsum_dg_scan(dg_total)
          with jax.named_scope("mega_compute_output"):
            dg_chunk_ref[:] = dg_reverse_cumsum.astype(dg_chunk_ref.dtype)

        @pl.when(is_first_chunk)
        def write_dh0():
          if dh0_ref is not None:
            with jax.named_scope("_dma_state_out"):
              _dma_state_out(
                  dh_tmp_ref, dh0_ref, state_sems_bwd.at[2],
                  mb_offset, batch_idx, seq_idx,
              )

      assert DMA_INPUT_BUFFER_COUNT == 2 and DMA_OUTPUT_BUFFER_COUNT == 2

      pass

      initial_is_tail = REM > 0 and LOOP_NUM == 1
      initial_start_c = 0 if initial_is_tail else NT - CHUNK_BATCH
      initial_has_valid = input_window_has_valid(
          initial_start_c, initial_is_tail
      )
      with jax.named_scope("mega_dma_input_start"):
        start_input_window(
            0, initial_start_c, initial_is_tail, initial_has_valid
        )

      # input ping-pong and output ping
      @pl.loop(0, LOOP_NUM, unroll=False)
      def async_loop(loop_idx):
        input_buf = loop_idx % 2
        next_input_buf = 1 - input_buf
        output_buf = loop_idx % 2
        full_start_c = NT - (loop_idx + 1) * CHUNK_BATCH
        is_tail = loop_idx == LOOP_NUM - 1
        if REM > 0:
          safe_full_start_c = jnp.where(is_tail, 0, full_start_c)
        else:
          is_tail = False
          safe_full_start_c = full_start_c

        has_valid = input_window_has_valid(safe_full_start_c, is_tail)
        with jax.named_scope("mega_dma_input_wait"):
          wait_input_window(input_buf, safe_full_start_c, is_tail, has_valid)

        @pl.when(loop_idx + 1 < LOOP_NUM)
        def prefetch_next_window():
          next_full_start_c = NT - (loop_idx + 2) * CHUNK_BATCH
          if REM > 0:
            next_is_tail = loop_idx + 1 == LOOP_NUM - 1
            next_safe_start_c = jnp.where(next_is_tail, 0, next_full_start_c)
          else:
            next_is_tail = False
            next_safe_start_c = next_full_start_c
          next_has_valid = input_window_has_valid(
              next_safe_start_c, next_is_tail
          )
          with jax.named_scope("mega_dma_input_prefetch"):
            start_input_window(
                next_input_buf,
                next_safe_start_c,
                next_is_tail,
                next_has_valid,
            )

        # Output(i - 2) owns the same ping-pong slot as output(i). Wait
        # only when that slot is about to be reused; output(i - 1) remains
        # in flight in the other slot while this window computes.
        @pl.when(loop_idx >= DMA_OUTPUT_BUFFER_COUNT)
        def _wait_reused_output():
          previous_window = loop_idx - DMA_OUTPUT_BUFFER_COUNT
          previous_start_c = NT - (previous_window + 1) * CHUNK_BATCH
          with jax.named_scope("mega_dma_output_wait"):
            wait_dmas(output_batch_descs(output_buf, previous_start_c))

        pass

        # =================== compute start ====================================
        @pl.when(jnp.logical_not(is_tail))
        def _process_full_window():
          @pl.loop(0, CHUNK_BATCH, unroll=False)
          def _process_full_slot(rev_slot):
            slot = CHUNK_BATCH - 1 - rev_slot
            chunk_id = safe_full_start_c + slot
            q_chunk = q_buf.at[input_buf, :, 0, slot, :, :]
            k_chunk = k_buf.at[input_buf, :, 0, slot, :, :]
            v_chunk = v_buf.at[input_buf, :, 0, slot, :, :]
            beta_chunk = beta_buf.at[input_buf, pl.ds(slot * MB_PAD, MB_PAD), :]
            do_chunk = do_buf.at[input_buf, :, 0, slot, :, :]
            compute_chunk(
                q_chunk,
                k_chunk,
                v_chunk,
                beta_chunk,
                Aqk_buf.at[input_buf, :, 0, slot, :, :],
                Akk_buf.at[input_buf, :, 0, slot, :, :],
                g_buf.at[input_buf, :, 0, slot, :, :],
                h_buf.at[input_buf, :, 0, slot, :, :],
                do_chunk,
                dq_buf.at[output_buf, :, 0, slot, :, :],
                dk_buf.at[output_buf, :, 0, slot, :, :],
                dv_buf.at[output_buf, :, 0, slot, :, :],
                db_buf.at[output_buf, pl.ds(slot * MB_PAD, MB_PAD), :],
                dg_buf.at[output_buf, :, 0, slot, :, :],
                chunk_id,
            )

        if REM > 0:

          @pl.when(is_tail)
          def _process_tail_window():
            @pl.loop(0, REM, unroll=False)
            def _process_tail_slot(rev_slot):
              slot = REM - 1 - rev_slot
              q_chunk = q_buf.at[input_buf, :, 0, slot, :, :]
              k_chunk = k_buf.at[input_buf, :, 0, slot, :, :]
              v_chunk = v_buf.at[input_buf, :, 0, slot, :, :]
              beta_chunk = beta_buf.at[
                  input_buf, pl.ds(slot * MB_PAD, MB_PAD), :
              ]
              do_chunk = do_buf.at[input_buf, :, 0, slot, :, :]
              compute_chunk(
                  q_chunk,
                  k_chunk,
                  v_chunk,
                  beta_chunk,
                  Aqk_buf.at[input_buf, :, 0, slot, :, :],
                  Akk_buf.at[input_buf, :, 0, slot, :, :],
                  g_buf.at[input_buf, :, 0, slot, :, :],
                  h_buf.at[input_buf, :, 0, slot, :, :],
                  do_chunk,
                  dq_buf.at[output_buf, :, 0, slot, :, :],
                  dk_buf.at[output_buf, :, 0, slot, :, :],
                  dv_buf.at[output_buf, :, 0, slot, :, :],
                  db_buf.at[output_buf, pl.ds(slot * MB_PAD, MB_PAD), :],
                  dg_buf.at[output_buf, :, 0, slot, :, :],
                  slot,
              )

        # =================== compute end ====================================

        @pl.when(is_tail)
        def _launch_tail_output():
          with jax.named_scope("mega_dma_output_start"):
            start_dmas(output_tail_descs(output_buf))

        @pl.when(jnp.logical_not(is_tail))
        def _launch_full_output():
          with jax.named_scope("mega_dma_output_start"):
            start_dmas(output_batch_descs(output_buf, safe_full_start_c))

      # Both output slots may still own an in-flight DMA after the loop.
      # Drain the older (penultimate) window first, then the final window.
      last_window = LOOP_NUM - 1
      with jax.named_scope("mega_dma_output_drain"):
        if LOOP_NUM > 1:
          previous_window = LOOP_NUM - 2
          previous_buf = previous_window % DMA_OUTPUT_BUFFER_COUNT
          previous_start_c = NT - (previous_window + 1) * CHUNK_BATCH
          wait_dmas(output_batch_descs(previous_buf, previous_start_c))
        last_buf = last_window % DMA_OUTPUT_BUFFER_COUNT
        if REM > 0:
          wait_dmas(output_tail_descs(last_buf))
        else:
          last_start_c = NT - (last_window + 1) * CHUNK_BATCH
          wait_dmas(output_batch_descs(last_buf, last_start_c))
      return

  # Execute the body in batch-major, head-group-minor order. Scratch refs
  # live for exactly one (batch, head-group) iteration.
  @pl.loop(0, B, unroll=False)
  def batch_loop(batch_idx):
    @pl.loop(0, H // MB, unroll=False)
    def head_group_loop(mb_offset):
      if CP_ACTIVE:

        @functools.partial(
            pl.run_scoped,
            dh_tmp_ref=pltpu.VMEM((MB, K, V), jnp.float32),
            dS_acc_ref=pltpu.VMEM((MB, K, V), jnp.float32),
            M_acc_ref=pltpu.VMEM((MB, K, K), jnp.float32),
            ag_dS_ref=pltpu.VMEM((CP_SIZE, MB, K, V), jnp.float32),
            ag_buf_ref=pltpu.VMEM((CP_SIZE, MB, K, V + K), jnp.float32),
        )
        def _cp_head_group_scoped(
            dh_tmp_ref,
            dS_acc_ref,
            M_acc_ref,
            ag_dS_ref,
            ag_buf_ref,
        ):
          main_loop_body(
              batch_idx,
              mb_offset,
              dh_tmp_ref,
              dS_acc_ref,
              M_acc_ref,
              ag_dS_ref,
              ag_buf_ref,
          )

      else:

        @functools.partial(
            pl.run_scoped,
            dh_tmp_ref=pltpu.VMEM((MB, K, V), jnp.float32),
        )
        def _non_cp_head_group_scoped(dh_tmp_ref):
          main_loop_body(batch_idx, mb_offset, dh_tmp_ref)


@partial(
    jax.jit,
    static_argnames=[
        "chunk_size",
        "use_exp2",
        "scale",
        "mini_batch",
        "chunk_batch",
        "return_dh0",
        "N_MAX",
        "disable_recompute",
        "cp_active",
        "has_initial_state",
        "cp_size",
        "cp_axis_name",
        "cp_context",
        "batch_first_inputs",
        "cast_dg_to_q_dtype",
    ],
)
def chunk_kda_bwd_fusion(
    q: Float[Array, "H B T K"],
    k: Float[Array, "H B T K"],
    v: Float[Array, "H B T V"],
    beta: Float[Array, "H B T"],
    Aqk: Float[Array, "H B T BT_ALIGN"],
    Akk: Float[Array, "H B T BT_ALIGN"],
    g: Float[Array, "H B T K"],
    h: Float[Array, "H B NT K V"] | None,
    do: Float[Array, "H B T V"],
    dht: Float[Array, "B N_MAX H K V"] | None,
    initial_state: Float[Array, "B N_MAX H K V"] | None,
    scale: float,
    *,
    segment_ids: Int[Array, "B T"] | None = None,
    cu_seqlens: Int[Array, "B N_PLUS_1"] | None = None,
    chunk_indices: Int[Array, "B NT 2"] | None = None,
    chunk_size: int = 64,
    use_exp2: bool = True,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    mini_batch: int | None = None,
    chunk_batch: int = KDA_DMA_CHUNK_BATCH,
    return_dh0: bool = True,
    disable_recompute: bool = True,
    cp_active: bool = False,
    has_initial_state: bool = False,
    cp_context: CPContext | None = None,
    cp_size: int | None = None,
    cp_axis_name: str | None = None,
    N_MAX: int | None = None,
    batch_first_inputs: bool = False,
    cast_dg_to_q_dtype: bool = False,
) -> tuple[
    Float[Array, "H B T K"],  # q
    Float[Array, "H B T K"],  # k
    Float[Array, "H B T V"],  # v
    Float[Array, "H B T"],  # beta
    Float[Array, "H B T K"],  # g
    Float[Array, "B N_MAX H K V"] | None,  # dh0
]:
  """M5 mega kernel: fuses M1+dAv+M4 (all phases) into a single Pallas call.

  All branch decisions (``disable_recompute`` and ``cp_active``) are resolved
  at trace time via ``static_argnames`` 鈥?the kernel body is a single source
  of truth for every supported path. Intra contractions always preserve the
  BF16 RHS low bits.

  Same return signature as ``_fused_dhu_wy_intra_cumsum_pallas_jit``.
  Eliminates HBM round-trips for w, qg, kg, v_new, dAqk, dv0.
  ``K`` and ``V`` must be multiples of 128. ``Aqk``/``Akk`` may expose
  either the logical ``BT`` width or its 128-aligned physical width; only
  legacy logical-width inputs are padded before the Pallas call.
  """
  H, B, T_INPUT, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  T = Aqk.shape[2]
  NT = T // BT
  BT_pad = pl.cdiv(BT, 128) * 128
  BETA_pad = 128
  # Packed custom-VJP callers can request the final public gate dtype here.
  # Casting in the kernel halves dg write/compaction traffic for BF16 inputs.
  dg_out_dtype = q.dtype if cast_dg_to_q_dtype else g.dtype

  assert chunk_size == 64, f"only chunk_size=64 is supported, got {chunk_size}"
  packed_inputs = False
  if batch_first_inputs or cu_seqlens is not None:
    raise ValueError("CP megakernel uses aligned Tokamax inputs")
  assert (
      not batch_first_inputs or packed_inputs
  ), "batch-first token inputs are supported only by native packed backward"
  assert T % BT == 0, f"residual T={T} must be divisible by chunk_size={BT}"
  assert (
      not packed_inputs
  ) or q.dtype == jnp.bfloat16, (
      "native packed backward currently supports BF16 token inputs"
  )
  assert (
      not packed_inputs
  ) or T_INPUT % 8 == 0, (
      f"native packed backward output T={T_INPUT} must be divisible by 8"
  )
  assert use_exp2 is True, "only use_exp2=True is supported"
  assert K % 128 == 0, f"K={K} must be a multiple of 128"
  assert V % 128 == 0, f"V={V} must be a multiple of 128"
  assert Aqk.shape[-1] in (
      BT,
      BT_pad,
  ), f"Aqk last dimension must be {BT} or {BT_pad}, got {Aqk.shape[-1]}"
  assert Akk.shape[-1] in (
      BT,
      BT_pad,
  ), f"Akk last dimension must be {BT} or {BT_pad}, got {Akk.shape[-1]}"
  assert (disable_recompute is False) or (
      h is not None
  ), "h is required when disable_recompute=True"
  assert (cp_active is False) or (
      cp_context is not None
      and cp_size is not None
      and cp_size > 0
      and cp_axis_name is not None
  )
  assert (segment_ids is None) or (
      N_MAX is not None
  ), "N_MAX must be provided for varlen input"
  assert (
      dht is None or dht.ndim == 5
  ), f"dht must be 5D [B, N_MAX, H, K, V], got shape {dht.shape}"
  assert (
      initial_state is None or initial_state.ndim == 5
  ), f"initial_state must be 5D [B, N_MAX, H, K, V], got shape {initial_state.shape}"

  cp_mb16_supported = (
      cp_active
      and packed_inputs
      and q.dtype == jnp.bfloat16
      and K == 128
      and V == 128
  )

  chunk_batch = min(chunk_batch, NT)
  pass
  assert chunk_batch > 0, f"chunk_batch must be positive, got {chunk_batch}"
  if mini_batch is None:
    if KDA_DMA_MINI_BATCH is not None:
      MB = KDA_DMA_MINI_BATCH
    else:
      # Packed BF16 CP with 128-wide states fits 16 heads when each DMA
      # window contains two chunks. Other CP/FP32 shapes retain the
      # conservative eight-head bound.
      needs_fp32_budget = q.dtype == jnp.float32 or cp_active
      max_mini_batch = (
          16 if cp_mb16_supported else 8 if needs_fp32_budget else 16
      )
      per_head = (
          12 * chunk_batch * BT * max(K, V)
          + 8 * BT * BT_pad
          + (12 + 2 * (cp_size or 1)) * K * max(K, V)
      ) * 4
      MB = estimate_mini_batch(
          per_head,
          H,
          max_mb=max_mini_batch,
          vmem_budget=min(
              get_tpu_limits().vmem_limit_bytes, KDA_BWD_SCOPED_VMEM_LIMIT_BYTES
          ),
      )
      while H % MB != 0 and MB > 1:
        MB -= 1
  else:
    MB = mini_batch
  assert MB > 0, f"mini_batch must be positive, got {MB}"
  assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"
  if cp_active and packed_inputs and MB > 8:
    # Smaller staging windows preserve input/output ping-pong while keeping
    # the 16-head program below TPU v7x's 64 MiB scoped VMEM limit.
    chunk_batch = min(chunk_batch, 2)

  # Dynamic chunk-slot indexing into the packed BF16 beta/dbeta tile must
  # advance by a 16-row-aligned stride on TPU. Eight-row padding works only
  # when Mosaic can constant-fold the slot index.
  MB_PAD = align_up(MB, 16)
  N_HG = H // MB
  DB_NT = ((NT + chunk_batch - 1) // chunk_batch) * chunk_batch
  DB_TAIL_PAD = DB_NT - NT
  N_MAX = (
      cu_seqlens.shape[-1] - 1
      if packed_inputs
      else 1
      if (segment_ids is None)
      else N_MAX
  )

  is_varlen = segment_ids is not None
  segment_ids = (
      jnp.ones((B, T), dtype=jnp.int32) if not is_varlen else segment_ids
  )
  chunk_seg_ids = segment_ids.reshape(B, NT, BT)[:, :, 0]
  cu_seqlens = jnp.zeros((B, 1), dtype=jnp.int32)
  # Aligned execution never reads packed chunk metadata.
  chunk_indices = jnp.zeros((1,), dtype=jnp.int32)
  compact_chunk_metadata = False
  chunk_metadata = chunk_indices

  dht_arr = dht
  initial_state_arr = (
      initial_state
      if initial_state is not None
      else jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.float32)
  )

  # Match the compact PR #463 M4 layout: one 128-wide row per head/chunk,
  # rather than expanding every beta scalar to a 128-wide row.
  beta_r = beta.reshape(N_HG, MB, B, NT, BT).transpose(0, 2, 3, 1, 4)
  beta_r = jnp.pad(
      beta_r,
      ((0, 0), (0, 0), (0, 0), (0, MB_PAD - MB), (0, BETA_pad - BT)),
  ).reshape(N_HG * B * NT * MB_PAD, BETA_pad)

  h_r = h if h is not None else jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.float32)
  initial_state_arr = initial_state_arr.astype(jnp.float32).transpose(
      2, 0, 1, 3, 4
  )

  # 鈹€鈹€ CP metadata arrays (per-batch) 鈹€鈹€
  if cp_active and cp_context is not None:
    cp_post_num = jnp.broadcast_to(
        jnp.asarray(cp_context.post_num_ranks, dtype=jnp.int32).reshape(B), (B,)
    )
    cp_max_post = jax.lax.pmax(cp_post_num, cp_axis_name)
    cp_post_num = (cp_max_post << 8) | cp_post_num
    cp_is_last = jnp.broadcast_to(
        jnp.asarray(cp_context.is_last_rank, dtype=jnp.int32).reshape(B), (B,)
    )
  else:
    cp_post_num = jnp.zeros((B,), dtype=jnp.int32)
    cp_is_last = jnp.zeros((B,), dtype=jnp.int32)

  kernel = partial(
      _bwd_mega_kernel,
      H=H,
      B=B,
      SCALE=scale,
      BT=BT,
      BT_PAD=BT_pad,
      K=K,
      V=V,
      NT=NT,
      T_INPUT=T_INPUT,
      MB=MB,
      MB_PAD=MB_PAD,
      CHUNK_BATCH=chunk_batch,
      DMA_INPUT_BUFFER_COUNT=KDA_DMA_INPUT_BUFFER_COUNT,
      DMA_OUTPUT_BUFFER_COUNT=KDA_DMA_OUTPUT_BUFFER_COUNT,
      DISABLE_RECOMPUTE=disable_recompute,
      CP_ACTIVE=cp_active,
      CP_SIZE=cp_size,
      CP_AXIS_NAME=cp_axis_name,
      HAS_H0=has_initial_state,
      PACKED_INPUTS=packed_inputs,
      MULTI_SEGMENT_PACKED=packed_inputs and N_MAX > 1,
      BATCH_FIRST_INPUTS=batch_first_inputs,
      COMPACT_CHUNK_METADATA=compact_chunk_metadata,
  )
  out_shape = [
      jax.ShapeDtypeStruct((H, B, NT, BT, K), q.dtype),  # dq
      jax.ShapeDtypeStruct((H, B, NT, BT, K), k.dtype),  # dk
      jax.ShapeDtypeStruct((H, B, NT, BT, V), v.dtype),  # dv
      jax.ShapeDtypeStruct(
          (N_HG * B * DB_NT * MB_PAD, BETA_pad), beta.dtype
      ),  # db
      jax.ShapeDtypeStruct((H, B, NT, BT, K), dg_out_dtype),  # dg
      jax.ShapeDtypeStruct((H, B, N_MAX, K, V), jnp.float32) if return_dh0 else None,
      jax.ShapeDtypeStruct(
          (H, B, NT, K, V) if not disable_recompute else (1, 1, 1, 1, 1),
          jnp.float32,
      ),
  ]

  any_spec = pl.BlockSpec(memory_space=pl.ANY)

  q_r = q if packed_inputs else q.reshape(H, B, NT, BT, K)
  k_r = k if packed_inputs else k.reshape(H, B, NT, BT, K)
  v_r = v if packed_inputs else v.reshape(H, B, NT, BT, V)
  g_r = g.reshape(H, B, NT, BT, K)

  def _pad_bt_4d(x):
    if x.shape[-1] < BT_pad:
      return jnp.pad(
          x,
          ((0, 0), (0, 0), (0, 0), (0, BT_pad - x.shape[-1])),
      )
    return x

  Aqk_r = _pad_bt_4d(Aqk).reshape(H, B, NT, BT, BT_pad)
  Akk_r = _pad_bt_4d(Akk).reshape(H, B, NT, BT, BT_pad)
  do_r = do if packed_inputs else do.reshape(H, B, NT, BT, V)
  if dht_arr is not None:
    dht_arr = dht_arr.astype(jnp.float32).transpose(2, 0, 1, 3, 4)
  dq_r, dk_r, dv_r, db_r, dg_r, dh0_r, _ = pl.pallas_call(
      kernel,
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=5,
          grid=(),
          in_specs=[any_spec] * 9 + [any_spec if dht_arr is not None else None, any_spec],
          out_specs=[any_spec] * 5 + [any_spec if return_dh0 else None, any_spec],
      ),
      compiler_params=pltpu.CompilerParams(
          disable_bounds_checks=True,
          collective_id=KDA_BWD_CP_COLLECTIVE_ID if cp_active else None,
          vmem_limit_bytes=min(
              get_tpu_limits().vmem_limit_bytes,
              KDA_BWD_SCOPED_VMEM_LIMIT_BYTES,
          ),
      ),
      interpret=pltpu.InterpretParams(detect_races=True)
      if get_interpret()
      else False,
  )(
      chunk_seg_ids,
      cu_seqlens.astype(jnp.int32),
      chunk_metadata.astype(jnp.int32),
      cp_post_num,
      cp_is_last,
      q_r,
      k_r,
      v_r,
      beta_r,
      Aqk_r,
      Akk_r,
      g_r,
      h_r,
      do_r,
      dht_arr,
      initial_state_arr,
  )

  db_r = db_r.reshape(N_HG, B, DB_NT, MB_PAD, BETA_pad)[..., :MB, :BT]
  if DB_TAIL_PAD:
    REM = NT % chunk_batch
    db_r = jnp.concatenate(
        [db_r[:, :, :REM], db_r[:, :, chunk_batch:]],
        axis=2,
    )
  db_r = db_r.transpose(0, 3, 1, 2, 4).reshape(H, B, T)

  pass

  dh0_out = dh0_r.transpose(1, 2, 0, 3, 4) if return_dh0 else None
  if dh0_out is not None and initial_state is not None:
    dh0_out = dh0_out.astype(initial_state.dtype)
  return (
      dq_r.reshape(H, B, T, K),
      dk_r.reshape(H, B, T, K),
      dv_r.reshape(H, B, T, V),
      db_r,
      dg_r.reshape(H, B, T, K),
      dh0_out if return_dh0 else None,
  )
