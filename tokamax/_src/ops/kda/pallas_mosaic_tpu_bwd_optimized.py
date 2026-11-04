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
"""Opt-in per-channel KDA backward schedules with packed VMEM bridges.

These schedules retain the migrated rounding and operand layouts. The default,
scalar-gate, convolution and explicitly tuned mini-batch paths stay in the
upstream backward module; unchanged arithmetic helpers are shared with it.
"""

from __future__ import annotations

import dataclasses
import functools
from functools import partial
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member

from tokamax._src import jaxtyping
from tokamax._src.ops.kda.common import (
    chunk_gated_delta_rule_fwd_h,
    estimate_mini_batch,
    get_tpu_limits,
    kda_gate_chunk_cumsum,
)
from tokamax._src.ops.kda.cp_utils import (
    ContextParallelMetadata,
    _merge_dht,
    all_gather_into_tensor,
)
from tokamax._src.ops.kda.pallas_mosaic_tpu_output import compact_output
from tokamax._src.ops.kda.pallas_mosaic_tpu_types import KdaResiduals
from tokamax._src.ops.kda.utils import (
    _align_seqs,
    _unalign_output,
    exp2,
    get_interpret,
    l2norm_bwd,
    segment_ids_to_seqlens,
)


from tokamax._src.ops.kda.pallas_mosaic_tpu_bwd_kernel import (
    compute_wy_backward,
    compute_intra_backward,
    _chunk_segment_metadata,
    chunk_gated_delta_rule_bwd_dhu_pre_process,
    compute_dhu_recurrence,
    compute_m1_recompute,
    compute_reverse_cumsum_dg,
    fused_recompute_w_u_vnew_from_h_pallas,
    kda_gate_bwd,
)

RCP_LN2 = 1.0 / math.log(2)


def _storage_boundary(value, dtype):
  """Keep staged materialization boundaries in the fused interpreter."""
  value = value.astype(dtype)
  return jax.lax.optimization_barrier(value) if get_interpret() else value


def _recompute_w_u_fwd(q, k, v, beta, A, gk, chunk_size):
  """Recompute WY intermediates for the low-memory backward path."""
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT

  A_chunks = A.reshape(H * B * NT, BT, BT).astype(jnp.float32)
  q_chunks = q.reshape(H * B * NT, BT, K)
  k_chunks = k.reshape(H * B * NT, BT, K)
  v_chunks = v.reshape(H * B * NT, BT, V)
  beta_chunks = beta.reshape(H * B * NT, BT, 1)
  g_chunks = gk.reshape(H * B * NT, BT, K)
  g_exp = jnp.exp2(g_chunks)
  precision = None if q.dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST

  u = jnp.matmul(
      A_chunks,
      (v_chunks * beta_chunks).astype(jnp.float32),
      precision=precision,
      preferred_element_type=jnp.float32,
  ).astype(v.dtype)
  w = jnp.matmul(
      A_chunks,
      (k_chunks * beta_chunks * g_exp).astype(jnp.float32),
      precision=precision,
      preferred_element_type=jnp.float32,
  ).astype(k.dtype)
  qg = (q_chunks * g_exp).astype(q.dtype)
  kg = (k_chunks * jnp.exp2(g_chunks[:, -1:, :] - g_chunks)).astype(k.dtype)
  if get_interpret():
    u, w, qg, kg = jax.lax.optimization_barrier((u, w, qg, kg))

  return (
      w.reshape(H, B, T, K),
      u.reshape(H, B, T, V),
      qg.reshape(H, B, T, K),
      kg.reshape(H, B, T, K),
  )


def _rematerialize_states_kernel(
    segments_ref,
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    a_ref,
    g_ref,
    initial_ref,
    h_ref,
    v_new_ref,
    state_ref,
    *,
    chunk_size,
    chunks,
):
  """Rebuild chunk states with WY intermediates local to each program."""
  batch, chunk = pl.program_id(1), pl.program_id(2)
  _, _, valid, first, _ = _chunk_segment_metadata(
      segments_ref, batch, chunk, chunks
  )

  @pl.when(first | ~valid)
  def initialize():
    state_ref[...] = jnp.zeros_like(state_ref[...])
    if initial_ref is not None:
      state_ref[...] = jnp.where(
          valid, initial_ref[:, 0, 0].astype(jnp.float32), 0
      )

  state = state_ref[...]
  h_ref[:, 0, 0] = _storage_boundary(state, h_ref.dtype)
  w, u, _, kg = _recompute_w_u_fwd(
      q_ref[...],
      k_ref[...],
      v_ref[...],
      beta_ref[..., 0],
      a_ref[...],
      g_ref[...],
      chunk_size,
  )
  w, u, kg = w[:, 0], u[:, 0], kg[:, 0]
  v_new = u.astype(jnp.float32) - jnp.matmul(
      w.astype(jnp.float32),
      state,
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
  )
  v_new_ref[:, 0] = _storage_boundary(
      jnp.where(valid, v_new, 0), v_new_ref.dtype
  )
  gate_last = g_ref[:, 0, -1].astype(jnp.float32)
  state = state * exp2(gate_last)[:, :, None] + jnp.matmul(
      kg.astype(jnp.float32).transpose(0, 2, 1),
      v_new,
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
  )
  state_ref[...] = jnp.where(valid, state, 0)


@partial(jax.jit, static_argnames=["chunk_size"])
def _rematerialize_states_pallas(
    q,
    k,
    v,
    beta,
    a,
    g,
    initial_state,
    segment_ids,
    *,
    chunk_size=64,
):
  """Return the existing low-memory path's h and FP32 v_new layouts."""
  heads, batch, tokens, key_dim = q.shape
  value_dim = v.shape[-1]
  chunks = tokens // chunk_size
  if (
      chunk_size != 64
      or tokens % chunk_size
      or key_dim % 128
      or value_dim % 128
  ):
    raise ValueError(
        "Fused rematerialization requires BT=64 and 128-aligned K/V."
    )
  if heads == 1:
    # The fused MB=1 recursive state update is not numerically equivalent to
    # the established staged kernel on current Mosaic.  Preserve correctness
    # for this degenerate edge case without changing the multi-head fast path.
    w, u, _, kg = _recompute_w_u_fwd(q, k, v, beta, a, g, chunk_size)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=False,
        chunk_size=chunk_size,
    )
    return h, v_new
  if segment_ids is None:
    segment_ids = jnp.ones((batch, tokens), jnp.int32)
    if initial_state is not None and initial_state.ndim == 4:
      initial_state = initial_state[:, None]
  segments = segment_ids.reshape(batch, chunks, chunk_size)[:, :, 0]
  per_head = 4 * (
      3 * key_dim * value_dim
      + 12 * chunk_size * (key_dim + value_dim)
      + 4 * chunk_size**2
  )
  mini_batch = estimate_mini_batch(per_head, heads, max_mb=16)

  def token_spec(width):
    return pl.BlockSpec(
        (mini_batch, 1, chunk_size, width), lambda h, b, c, _: (h, b, c, 0)
    )

  def state_index(h, b, c, segments):
    _, seq, _, _, _ = _chunk_segment_metadata(segments, b, c, chunks)
    return h, b, seq, 0, 0

  initial_spec = pl.BlockSpec(
      (mini_batch, 1, 1, key_dim, value_dim), state_index
  )
  state_spec = pl.BlockSpec(
      (mini_batch, 1, 1, key_dim, value_dim), lambda h, b, c, _: (h, b, c, 0, 0)
  )
  return pl.pallas_call(
      partial(
          _rematerialize_states_kernel, chunk_size=chunk_size, chunks=chunks
      ),
      out_shape=(
          jax.ShapeDtypeStruct(
              (heads, batch, chunks, key_dim, value_dim), k.dtype
          ),
          jax.ShapeDtypeStruct(v.shape, jnp.float32),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(heads // mini_batch, batch, chunks),
          in_specs=[
              token_spec(key_dim),
              token_spec(key_dim),
              token_spec(value_dim),
              token_spec(1),
              token_spec(chunk_size),
              token_spec(key_dim),
              initial_spec if initial_state is not None else None,
          ],
          out_specs=(state_spec, token_spec(value_dim)),
          scratch_shapes=[
              pltpu.VMEM((mini_batch, key_dim, value_dim), jnp.float32)
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")
      ),
      interpret=get_interpret(),
  )(
      segments,
      q,
      k,
      v,
      beta[..., None],
      a,
      g,
      initial_state.transpose(2, 0, 1, 3, 4)
      if initial_state is not None
      else None,
  )


def _fused_recompute_w_u_vnew_from_h_kernel(
    k_ref,
    v_ref,
    beta_ref,
    A_ref,
    q_ref,
    g_ref,
    h_ref,
    w_ref,
    qg_ref,
    kg_ref,
    v_new_ref,
    *,
    BT,
    K,
    V,
    MB,
):
  """Compute w, qg, kg, and v_new for MB chunk tiles.

  Per chunk:
      u     = A @ (v * beta)
      w     = A @ (k * beta * exp2(g))
      qg    = q * exp2(g)
      kg    = k * exp2(g_last - g)
      v_new = u - w @ h
  """
  q = q_ref[:]  # [MB, BT, K]
  k = k_ref[:]  # [MB, BT, K]
  v = v_ref[:]  # [MB, BT, V]
  beta = beta_ref[:]  # [MB, BT]
  A = A_ref[:]  # [MB, BT, BT]
  g = g_ref[:]  # [MB, BT, K]
  h = h_ref[:]  # [MB, K, V]

  u, w, v_new, qg, kg = compute_m1_recompute(
      q, k, v, beta, A, g, h, k_ref.dtype
  )

  w_ref[:] = _storage_boundary(w, w_ref.dtype)
  qg_ref[:] = _storage_boundary(qg, qg_ref.dtype)
  kg_ref[:] = _storage_boundary(kg, kg_ref.dtype)
  v_new_ref[:] = _storage_boundary(v_new, v_new_ref.dtype)


def _fused_dhu_wy_intra_cumsum_kernel(
    chunk_seg_ids_ref,
    q_ref,
    k_ref,
    v_ref,
    v_new_ref,
    qg_ref,
    kg_ref,
    w_ref,
    g_ref,
    beta_ref,
    A_ref,
    h_ref,
    do_ref,
    dv0_ref,
    dAqk_ref,
    dht_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    db_ref,
    dg_ref,
    dh0_ref,
    dh_tmp_ref,
    *,
    BT,
    K,
    V,
    NT,
    scale,
    MB,
):
  """Fuse Dhu, WY, intra backward, and reverse cumsum for one chunk tile."""
  head_group = pl.program_id(0)
  batch_idx = pl.program_id(1)
  rev_c = pl.program_id(2)
  chunk_id = NT - 1 - rev_c
  _, seq_idx, _, is_first_chunk, is_last_chunk = _chunk_segment_metadata(
      chunk_seg_ids_ref,
      batch_idx,
      chunk_id,
      NT,
  )
  precision = None if q_ref.dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST

  @pl.when(is_last_chunk)
  def _():
    if dht_ref is None:
      dh_tmp_ref[:] = jnp.zeros(dh_tmp_ref.shape, dh_tmp_ref.dtype)
    else:
      dh_tmp_ref[:] = dht_ref[:, 0, 0, :].astype(dh_tmp_ref.dtype)

  dh = dh_tmp_ref[:].astype(jnp.float32)
  bq = q_ref[:, 0, 0].astype(jnp.float32)
  bk = k_ref[:, 0, 0].astype(jnp.float32)
  bv = v_ref[:, 0, 0].astype(jnp.float32)
  bvn = v_new_ref[:, 0, 0].astype(jnp.float32)
  bqg = qg_ref[:, 0, 0].astype(jnp.float32)
  bkg = kg_ref[:, 0, 0]
  bw = w_ref[:, 0, 0].astype(jnp.float32)
  bg = g_ref[:, 0, 0].astype(jnp.float32)
  g_exp_last = jnp.exp2(bg[:, BT - 1, :])
  # Keep the trailing tiled dimension while loading from VMEM.  Squeezing it
  # on the ref produces an unsupported tpu.memref_squeeze with nested tiles.
  bb = beta_ref[:, 0, 0].astype(jnp.float32).reshape(MB, BT)
  bA = A_ref[:, 0, 0].astype(jnp.float32)
  bh = h_ref[:, 0, 0].astype(jnp.float32)
  bdo = do_ref[:, 0, 0].astype(jnp.float32)
  bdv0 = dv0_ref[:, 0, 0].astype(jnp.float32)
  bdAqk = dAqk_ref[:, 0, 0].astype(jnp.float32)

  # --- dhu reverse recurrence ---
  bdv, dh_new = compute_dhu_recurrence(
      bkg,
      dh,
      bdv0,
      dh_tmp_ref[:],
      g_exp_last,
      bqg,
      bw,
      bdo,
      scale,
      precision,
  )
  dh_tmp_ref[:] = dh_new

  # --- WY backward ---
  dq_acc, dk_acc, b_dvb, db_acc, dg_acc, dAkk_local = compute_wy_backward(
      bdo, bdv, bvn, bv, bh, dh, bq, bk, bg, bb, bA, scale, precision
  )

  # --- Intra backward + reverse cumsum ---
  dq_total, dk_total, db_total, dg_total = compute_intra_backward(
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
      precision=precision,
  )
  dg_reverse_cumsum = compute_reverse_cumsum_dg(dg_total)

  dq_ref[:, 0, 0] = dq_total.astype(dq_ref.dtype)
  dk_ref[:, 0, 0] = dk_total.astype(dk_ref.dtype)
  dv_ref[:, 0, 0] = (b_dvb * bb[:, :, None]).astype(dv_ref.dtype)
  db_ref[:, 0, 0] = db_total.astype(db_ref.dtype)[..., None]
  dg_ref[:, 0, 0] = dg_reverse_cumsum.astype(dg_ref.dtype)

  @pl.when(is_first_chunk)
  def _():
    if dh0_ref is not None:
      dh0_ref[:, 0, 0, :] = dh_tmp_ref[:].astype(dh0_ref.dtype)


def _saved_state_backward_kernel(
    chunk_seg_ids_ref,
    q_ref,
    k_ref,
    v_ref,
    g_ref,
    beta_ref,
    akk_ref,
    aqk_ref,
    h_ref,
    do_ref,
    dht_ref,
    rematerialized_v_new_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    db_ref,
    dg_ref,
    dh0_ref,
    dh_ref,
    w_ref,
    qg_ref,
    kg_ref,
    v_new_ref,
    dAqk_ref,
    dv0_ref,
    *,
    BT,
    K,
    V,
    NT,
    scale,
    MB,
):
  """Fuses saved-state recomputation and all local backward stages.

  The bridge refs retain the staged kernels' dtypes, including FP32 dAqk
  and dv0, so fusion does not change the rounding at their former HBM
  boundaries. dh_ref persists across the reverse-ordered chunk grid.
  """
  if rematerialized_v_new_ref is not None:
    # Preserve supplied v_new: FP32 running-state values during rematerialization
    # or the pre-collective saved-path values during CP backward.
    # Only WY weights and gated Q/K are needed in this reverse pass.
    q = q_ref[:, 0, 0].astype(jnp.float32)
    k = k_ref[:, 0, 0].astype(jnp.float32)
    beta = beta_ref[:, 0, 0].astype(jnp.float32)
    g = g_ref[:, 0, 0]
    gate_exp = jnp.exp2(g)
    precision = (
        None if q_ref.dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST
    )
    w = jnp.matmul(
        akk_ref[:, 0, 0].astype(jnp.float32),
        (k * beta * gate_exp).astype(jnp.float32),
        precision=precision,
        preferred_element_type=jnp.float32,
    )
    w_ref[:, 0, 0] = _storage_boundary(w, w_ref.dtype)
    qg_ref[:, 0, 0] = _storage_boundary(q * gate_exp, qg_ref.dtype)
    kg_ref[:, 0, 0] = _storage_boundary(
        k * jnp.exp2(g[:, -1:] - g), kg_ref.dtype
    )
    v_new_ref[...] = _storage_boundary(
        rematerialized_v_new_ref[...], v_new_ref.dtype
    )
  else:
    # Load beta before dropping its trailing singleton dimension. Passing a
    # squeezed Ref view is illegal for the nested tiled VMEM layout.
    beta_tile = beta_ref[:, 0, 0].reshape(MB, BT)
    _fused_recompute_w_u_vnew_from_h_kernel(
        k_ref.at[:, 0, 0],
        v_ref.at[:, 0, 0],
        beta_tile,
        akk_ref.at[:, 0, 0],
        q_ref.at[:, 0, 0],
        g_ref.at[:, 0, 0],
        h_ref.at[:, 0, 0],
        w_ref.at[:, 0, 0],
        qg_ref.at[:, 0, 0],
        kg_ref.at[:, 0, 0],
        v_new_ref.at[:, 0, 0],
        BT=BT,
        K=K,
        V=V,
        MB=MB,
    )
  _chunk_kda_bwd_dAv_kernel(
      v_new_ref.at[:, 0, 0],
      aqk_ref.at[:, 0, 0],
      do_ref.at[:, 0, 0],
      dAqk_ref.at[:, 0, 0],
      dv0_ref.at[:, 0, 0],
      scale=scale,
      BT=BT,
      BV=V,
      NV=1,
      V=V,
      MB=MB,
  )
  _fused_dhu_wy_intra_cumsum_kernel(
      chunk_seg_ids_ref,
      q_ref,
      k_ref,
      v_ref,
      v_new_ref,
      qg_ref,
      kg_ref,
      w_ref,
      g_ref,
      beta_ref,
      akk_ref,
      h_ref,
      do_ref,
      dv0_ref,
      dAqk_ref,
      dht_ref,
      dq_ref,
      dk_ref,
      dv_ref,
      db_ref,
      dg_ref,
      dh0_ref,
      dh_ref,
      BT=BT,
      K=K,
      V=V,
      NT=NT,
      scale=scale,
      MB=MB,
  )


def _packed_saved_state_backward_kernel(
    segments_ref, windows_ref, *refs, **kwargs
):
  """Stage original packed Q/K/V/beta for the reverse chunk traversal."""
  batch = pl.program_id(1)
  chunk = kwargs["NT"] - 1 - pl.program_id(2)
  offset_index = 2 * (batch * kwargs["NT"] + chunk)
  start, end = windows_ref[offset_index], windows_ref[offset_index + 1]
  offset = start % 8
  branches = tuple(
      (lambda x, i=i: x[:, :, i : i + kwargs["BT"], :]) for i in range(8)
  )
  args = list(refs[:-4])
  for index, tile in zip((0, 1, 2, 4), refs[-4:], strict=True):
    values = jax.lax.switch(offset, branches, refs[index][...])
    token = jax.lax.broadcasted_iota(jnp.int32, values.shape, dimension=2)
    values = jnp.where(start + token < end, values, 0)
    tile[...] = values.reshape(tile.shape)
    args[index] = tile
  _saved_state_backward_kernel(segments_ref, *args, **kwargs)


@partial(
    jax.jit,
    static_argnames=[
        "chunk_size",
        "use_exp2",
        "scale",
        "mini_batch",
        "return_dh0",
        "max_num_segments",
        "fuse_recompute",
    ],
)
@jaxtyping.jaxtyped
def _fused_dhu_wy_intra_cumsum_pallas_jit(
    q: Float[Array, "H B T K"],
    k: Float[Array, "H B T K"],
    v: Float[Array, "H B T V"],
    v_new: Float[Array, "H B T V"] | None,
    qg: Float[Array, "H B T K"] | None,
    kg: Float[Array, "H B T K"] | None,
    w: Float[Array, "H B T K"] | None,
    g: Float[Array, "H B T K"],
    beta: Float[Array, "H B T"],
    A: Float[Array, "H B T BT"],
    h: Float[Array, "H B NT K V"],
    do: Float[Array, "H B T V"],
    dv0: Float[Array, "H B T V"] | None,
    dAqk: Float[Array, "H B T BT"] | None,
    dht: Float[Array, "B N H K V"] | None,
    scale: float,
    *,
    segment_ids: Int[Array, "B T"] | None = None,
    chunk_size: int = 64,
    use_exp2: bool = True,
    mini_batch: int | None = None,
    return_dh0: bool = True,
    max_num_segments: int | None = None,
    fuse_recompute: bool = False,
    forward_aqk: Float[Array, "H B T BT"] | None = None,
    packed_inputs: tuple[jax.Array, ...] | None = None,
    original_cu_seqlens: jax.Array | None = None,
    aligned_cu_seqlens: jax.Array | None = None,
) -> tuple[
    Float[Array, "H B T K"],
    Float[Array, "H B T K"],
    Float[Array, "H B T V"],
    Float[Array, "H B T"],
    Float[Array, "H B T K"],
    Float[Array, "B N_OUT H K V"] | None,
]:
  """Fuses Dhu recurrence, WY backward, intra backward, and gate cumsum.

  `segment_ids` applies per-batch varlen boundaries, and `return_dh0`
  controls whether the initial-state gradient is materialized. With
  `fuse_recompute`, w/qg/kg/v_new/dAqk/dv0 are produced in VMEM from h
  and forward_aqk rather than read from HBM. An optional v_new preserves
  FP32 value corrections from fused state rematerialization. Output shapes
  are unchanged.
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT
  HB = H * B

  is_varlen = segment_ids is not None
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  assert (
      A.shape[-1] == BT
  ), f"A.shape[-1]={A.shape[-1]} must equal chunk_size={BT}"
  assert h.shape[2] == NT, f"h has NT={h.shape[2]}, expected {NT}"
  assert (
      use_exp2 is True
  ), "fused_dhu_wy_intra_cumsum_pallas currently expects log2 gates and use_exp2=True"

  segment_ids = (
      jnp.ones((B, T), dtype=jnp.int32) if not is_varlen else segment_ids
  )
  chunk_seg_ids = segment_ids.reshape(B, NT, BT)[:, :, 0]  # [B, NT]
  if dht is not None:
    N = dht.shape[1]  # dht is [B, N, H, K, V], N is dim 1
  elif is_varlen:
    N = max_num_segments
  else:
    N = 1  # uniform: one sequence per batch element
  if dht is not None:
    assert dht.shape[1] == N, f"dht has N={dht.shape[1]}, expected {N}"
  dht_arr = dht

  if mini_batch is None:
    elem_size = 2 if q.dtype == jnp.bfloat16 else 4
    io_per_head = (
        8 * BT * K + 4 * BT * V + BT + 3 * BT * BT + 2 * K * V
    ) * elem_size + (K * V + 5 * BT * K + BT * V + BT + 2 * BT * BT + K * V) * 4
    per_head = io_per_head + io_per_head * 3 // 2
    if packed_inputs is not None:
      per_head += 4 * (2 * K + V + 1) * (3 * BT + 16)
    hw = get_tpu_limits()
    vmem_budget = hw.vmem_limit_bytes
    MB = max(1, vmem_budget // per_head)
    MB = min(MB, H, 16)
    while H % MB != 0 and MB > 1:
      MB -= 1
  else:
    MB = mini_batch
    assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

  if fuse_recompute:
    if forward_aqk is None or forward_aqk.shape != A.shape:
      raise ValueError("Fused backward requires forward Aqk matching Akk.")
    if K % 128 or V % 128 or BT != 64:
      raise ValueError("Fused backward requires BT=64 and 128-aligned K/V.")

  # Keep [H, B, ...] layout; reshape T → (NT, BT) only. No transpose.
  # B is an independent dimension handled by a separate grid axis.
  q_r = q.reshape(H, B, NT, BT, K)
  k_r = k.reshape(H, B, NT, BT, K)
  v_r = v.reshape(H, B, NT, BT, V)
  vn_r = v_new.reshape(H, B, NT, BT, V) if v_new is not None else None
  qg_r = qg.reshape(H, B, NT, BT, K) if qg is not None else None
  kg_r = kg.reshape(H, B, NT, BT, K) if kg is not None else None
  w_r = w.reshape(H, B, NT, BT, K) if w is not None else None
  g_r = g.reshape(H, B, NT, BT, K)
  beta_r = beta.reshape(H, B, NT, BT, 1)
  A_r = A.reshape(H, B, NT, BT, BT)
  h_r = h  # already [H, B, NT, K, V]
  do_r = do.reshape(H, B, NT, BT, V)
  dv0_r = dv0.reshape(H, B, NT, BT, V) if dv0 is not None else None
  dAqk_r = dAqk.reshape(H, B, NT, BT, BT) if dAqk is not None else None

  def idx_chunk(head_group, batch, chunk, chunk_seg_ids_ref, *_):
    return (head_group, batch, NT - 1 - chunk, 0, 0)

  def idx_state(head_group, batch, chunk, chunk_seg_ids_ref, *_):
    chunk_id = NT - 1 - chunk
    _, seq_idx, _, _, _ = _chunk_segment_metadata(
        chunk_seg_ids_ref, batch, chunk_id, NT
    )
    return (head_group, batch, seq_idx, 0, 0)

  # dht_arr [B, N, H, K, V] → [H, B, N, K, V]
  if dht_arr is not None:
    dht_arr = dht_arr.transpose(2, 0, 1, 3, 4)

  qk_spec = pl.BlockSpec((MB, 1, 1, BT, K), index_map=idx_chunk)
  v_spec = pl.BlockSpec((MB, 1, 1, BT, V), index_map=idx_chunk)
  b_spec = pl.BlockSpec((MB, 1, 1, BT, 1), index_map=idx_chunk)
  A_spec = pl.BlockSpec((MB, 1, 1, BT, BT), index_map=idx_chunk)
  h_spec = pl.BlockSpec((MB, 1, 1, K, V), index_map=idx_chunk)
  state_spec = pl.BlockSpec((MB, 1, 1, K, V), index_map=idx_state)

  kernel = partial(
      _fused_dhu_wy_intra_cumsum_kernel,
      scale=scale,
      BT=BT,
      K=K,
      V=V,
      NT=NT,
      MB=MB,
  )
  dh_tmp = pltpu.VMEM((MB, K, V), jnp.float32)
  out_shape = [
      jax.ShapeDtypeStruct((H, B, NT, BT, K), jnp.float32),
      jax.ShapeDtypeStruct((H, B, NT, BT, K), jnp.float32),
      jax.ShapeDtypeStruct((H, B, NT, BT, V), jnp.float32),
      jax.ShapeDtypeStruct((H, B, NT, BT, 1), jnp.float32),
      jax.ShapeDtypeStruct((H, B, NT, BT, K), jnp.float32),
      jax.ShapeDtypeStruct((H, B, N, K, V), jnp.float32)
      if return_dh0
      else None,
  ]

  in_specs = [
      qk_spec,
      qk_spec,
      v_spec,
      v_spec,
      qk_spec,
      qk_spec,
      qk_spec,
      qk_spec,
      b_spec,
      A_spec,
      h_spec,
      v_spec,
      v_spec,
      A_spec,
      state_spec if dht_arr is not None else None,
  ]
  kernel_inputs = [
      q_r,
      k_r,
      v_r,
      vn_r,
      qg_r,
      kg_r,
      w_r,
      g_r,
      beta_r,
      A_r,
      h_r,
      do_r,
      dv0_r,
      dAqk_r,
      dht_arr,
  ]
  scratch_shapes = [dh_tmp]
  if fuse_recompute:
    kernel = partial(
        _saved_state_backward_kernel, BT=BT, K=K, V=V, NT=NT, scale=scale, MB=MB
    )
    in_specs = [
        qk_spec,
        qk_spec,
        v_spec,
        qk_spec,
        b_spec,
        A_spec,
        A_spec,
        h_spec,
        v_spec,
        state_spec if dht_arr is not None else None,
        v_spec if vn_r is not None else None,
    ]
    kernel_inputs = [
        q_r,
        k_r,
        v_r,
        g_r,
        beta_r,
        A_r,
        forward_aqk.reshape(H, B, NT, BT, BT),
        h_r,
        do_r,
        dht_arr,
        vn_r,
    ]
    scratch_shapes += [
        *[pltpu.VMEM((MB, 1, 1, BT, K), q.dtype) for _ in range(3)],
        pltpu.VMEM(
            (MB, 1, 1, BT, V), v_new.dtype if v_new is not None else v.dtype
        ),
        pltpu.VMEM((MB, 1, 1, BT, BT), jnp.float32),
        pltpu.VMEM((MB, 1, 1, BT, V), jnp.float32),
    ]

  scalar_inputs = [chunk_seg_ids]
  if packed_inputs is not None:
    if (
        not fuse_recompute
        or original_cu_seqlens is None
        or aligned_cu_seqlens is None
    ):
      raise ValueError(
          "Packed backward inputs require local fusion and packed boundaries."
      )
    seq = jnp.maximum(chunk_seg_ids - 1, 0)
    original_start = jnp.take_along_axis(
        original_cu_seqlens[:, :-1], seq, axis=1
    )
    aligned_start = jnp.take_along_axis(aligned_cu_seqlens[:, :-1], seq, axis=1)
    end = jnp.take_along_axis(original_cu_seqlens[:, 1:], seq, axis=1)
    start = original_start + jnp.arange(NT)[None, :] * BT - aligned_start
    active = (chunk_seg_ids != 0) & (start >= original_start) & (start < end)
    windows = jnp.stack(
        [jnp.where(active, start, 0), jnp.where(active, end, 0)], axis=-1
    )
    # A narrow [..., 2] SMEM table is physically tiled/padded. Flatten it so
    # large aligned workspaces do not turn a small logical map into >1 MiB.
    scalar_inputs.append(windows.reshape(-1).astype(jnp.int32))

    def packed_index(h, b, c, _, windows):
      start = windows[2 * (b * NT + NT - 1 - c)]
      return h * MB, b, pl.multiple_of(start - start % 8, 8), 0

    pq, pk, pv, pb = packed_inputs
    for index, x in zip((0, 1, 2, 4), (pq, pk, pv, pb[..., None]), strict=True):
      kernel_inputs[index] = jnp.pad(
          x, ((0, 0), (0, 0), (0, (-x.shape[2]) % 8 + BT + 8), (0, 0))
      )
      in_specs[index] = pl.BlockSpec(
          tuple(pl.Element(d) for d in (MB, 1, BT + 8, x.shape[-1])),
          packed_index,
      )
      scratch_shapes.append(pltpu.VMEM((MB, 1, 1, BT, x.shape[-1]), x.dtype))
    kernel = partial(
        _packed_saved_state_backward_kernel,
        BT=BT,
        K=K,
        V=V,
        NT=NT,
        scale=scale,
        MB=MB,
    )

  dq_r, dk_r, dv_r, db_r, dg_r, dh0_r = pl.pallas_call(
      kernel,
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_inputs),
          grid=(H // MB, B, NT),
          in_specs=in_specs,
          out_specs=[
              qk_spec,
              qk_spec,
              v_spec,
              b_spec,
              qk_spec,
              state_spec if return_dh0 else None,
          ],
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          disable_bounds_checks=True,
          vmem_limit_bytes=get_tpu_limits().vmem_limit_bytes,
      ),
      interpret=get_interpret(),
  )(*scalar_inputs, *kernel_inputs)

  dh0_out = dh0_r.transpose(1, 2, 0, 3, 4) if return_dh0 else None
  return (
      dq_r.reshape(H, B, T, K),
      dk_r.reshape(H, B, T, K),
      dv_r.reshape(H, B, T, V),
      db_r.reshape(H, B, T),
      dg_r.reshape(H, B, T, K),
      dh0_out,
  )


def _chunk_kda_bwd_dAv_kernel(
    v_ref,
    A_ref,
    do_ref,
    dA_ref,
    dv_ref,
    *,
    scale,
    BT,
    BV,
    NV,
    V,
    MB,
):
  bv = v_ref[:]  # [MB, BT, V]
  bA = A_ref[:]  # [MB, BT, BT]
  bdo = do_ref[:].astype(jnp.float32)  # [MB, BT, V]
  precision = None if A_ref.dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST

  m_causal = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
  bA_masked = jnp.where(m_causal[None, :, :], bA, 0.0)  # [MB, BT, BT]

  b_dA = jnp.zeros((MB, BT, BT), jnp.float32)
  dv_blocks = []

  for i_v in range(NV):
    vs = i_v * BV
    ve = vs + BV

    b_v_blk = bv[:, :, vs:ve]  # [MB, BT, BV]
    b_do_blk = bdo[:, :, vs:ve]  # [MB, BT, BV]

    # dA += do @ v^T — contract BV (dim 2), batch MB (dim 0)
    if dA_ref is not None:
      b_dA += jax.lax.dot_general(
          b_do_blk,
          b_v_blk,
          (((2,), (2,)), ((0,), (0,))),
          precision=precision,
          preferred_element_type=jnp.float32,
      )

    # dv = A^T @ do — contract BT_row (dim 1), batch MB (dim 0)
    b_dv_blk = jax.lax.dot_general(
        bA_masked,
        b_do_blk.astype(bA_masked.dtype),
        (((1,), (1,)), ((0,), (0,))),
        precision=precision,
        preferred_element_type=jnp.float32,
    )
    dv_blocks.append(b_dv_blk)

  b_dv = jnp.concatenate(dv_blocks, axis=2) if NV > 1 else dv_blocks[0]

  # Apply causal mask and scale
  b_dA = jnp.where(m_causal[None, :, :], b_dA * scale, 0.0)

  if dA_ref is not None:
    dA_ref[:] = _storage_boundary(b_dA, dA_ref.dtype)
  dv_ref[:] = _storage_boundary(b_dv, dv_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=[
        "chunk_size",
        "scale",
        "block_V",
        "mini_batch",
        "store_dAqk",
    ],
)
@jaxtyping.jaxtyped
def chunk_kda_bwd_dAv_kernel(
    q: Float[Array, "H B T K"],
    k: Float[Array, "H B T K"],
    v: Float[Array, "H B T V"],
    do: Float[Array, "H B T V"],
    A: Float[Array, "H B T BT"],
    scale: float,
    chunk_size: int = 64,
    block_V: int | None = None,
    mini_batch: int | None = None,
    store_dAqk: bool = True,
) -> tuple[
    Float[Array, "H B T BT"] | None,
    Float[Array, "H B T V"],
]:
  """Computes attention and value gradients with a tiled Pallas kernel.

  `block_V` controls value-dimension tiling; `mini_batch` controls the
  flattened chunks processed by each program. With `store_dAqk=False`,
  only value gradients are computed; the first returned value is None.
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT

  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  assert (
      A.shape[-1] == BT
  ), f"A.shape[-1]={A.shape[-1]} must equal chunk_size={BT}"

  BV = block_V if block_V is not None else V
  assert V % BV == 0, f"V={V} must be divisible by block_V={BV}"
  NV = V // BV
  BH = B * H

  v_r = v.reshape(BH * NT, BT, V)
  A_r = A.reshape(BH * NT, BT, BT)
  do_r = do.reshape(BH * NT, BT, V)

  total = BH * NT

  # ---- auto-compute mini-batch (MB) to maximise VMEM utilisation ----
  if mini_batch is None:
    elem_size = 2 if v.dtype == jnp.bfloat16 else 4
    in_bytes = (2 * BT * V + BT * BT) * elem_size
    out_bytes = BT * BT * 4 + BT * V * 2
    per_chunk = in_bytes + out_bytes
    MB = estimate_mini_batch(per_chunk, total, max_mb=32)
  else:
    MB = mini_batch
    assert (
        total % MB == 0
    ), f"total={total} must be divisible by mini_batch={MB}"

  def _spec3(d1, d2):
    return pl.BlockSpec(
        block_shape=(MB, d1, d2), index_map=lambda idx: (idx, 0, 0)
    )

  in_specs = [
      _spec3(BT, V),  # v
      _spec3(BT, BT),  # A
      _spec3(BT, V),  # do
  ]

  out_specs = [
      _spec3(BT, BT) if store_dAqk else None,  # dA
      _spec3(BT, V),  # dv
  ]

  out_shape = [
      jax.ShapeDtypeStruct((total, BT, BT), jnp.float32)
      if store_dAqk
      else None,
      jax.ShapeDtypeStruct((total, BT, V), jnp.float32),
  ]

  kernel = partial(
      _chunk_kda_bwd_dAv_kernel,
      scale=scale,
      BT=BT,
      BV=BV,
      NV=NV,
      V=V,
      MB=MB,
  )

  interpret = get_interpret()

  dA_r, dv_r = pl.pallas_call(
      kernel,
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(total // MB,),
          in_specs=in_specs,
          out_specs=out_specs,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          disable_bounds_checks=True,
          vmem_limit_bytes=get_tpu_limits().vmem_limit_bytes,
      ),
      interpret=interpret,
  )(v_r, A_r, do_r)

  # [HB*NT, BT, X] -> [H, B, T, X]
  def _ir(x, d):
    return x.reshape(H, B, T, d)

  return _ir(dA_r, BT) if dA_r is not None else None, _ir(dv_r, V)


@functools.partial(
    jax.jit,
    static_argnames=[
        "scale",
        "use_qk_l2norm",
        "use_gate_in_kernel",
        "lower_bound",
        "disable_recompute",
        "context_parallel_metadata",
        "chunk_size",
        "max_num_segments",
        "has_initial_state",
        "fuse_backward",
        "fuse_rematerialization",
        "packed_gradients",
        "fuse_cp_backward",
        "cp_megakernel",
    ],
)
@jaxtyping.jaxtyped
def chunk_kda_bwd_custom(
    scale: float | None,
    use_qk_l2norm: bool,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
    disable_recompute: bool,
    context_parallel_metadata: ContextParallelMetadata | None,
    chunk_size: int,
    max_num_segments: int | None,
    has_initial_state: bool,
    residuals: KdaResiduals,
    grad_outputs: tuple[
        Float[Array, "H B T_ORIG V"],
        Float[Array, "B H K V"] | Float[Array, "B N H K V"] | None,
    ],
    fuse_backward: bool = True,
    fuse_rematerialization: bool = False,
    packed_gradients: bool = False,
    fuse_cp_backward: bool = False,
    cp_megakernel: bool = False,
    packed_inputs: tuple[jax.Array, ...] | None = None,
) -> tuple[
    Float[Array, "H B T_ORIG K"],
    Float[Array, "H B T_ORIG K"],
    Float[Array, "H B T_ORIG V"],
    Float[Array, "H B T_ORIG K"],
    Float[Array, "H B T_ORIG"],
    Float[Array, "H"] | None,
    Float[Array, "H*K"] | None,
    Float[Array, "B N H K V"] | None,
    None,
]:
  """Runs the full KDA backward pipeline from forward residuals."""
  do, dht = grad_outputs
  # JAX cotangents match the bf16 output dtype; backward accumulates in fp32.
  do = do.astype(jnp.float32)
  q = residuals.q
  k = residuals.k
  v = residuals.v
  beta = residuals.beta
  g_cumsum = residuals.g_cumsum
  Aqk = residuals.aqk
  Akk = residuals.akk
  initial_state = residuals.initial_state
  g_org = residuals.g_org
  a_log = residuals.a_log
  delta_time_bias = residuals.delta_time_bias
  h = residuals.h
  g_dtype_marker = residuals.g_dtype_marker
  rstd_q = residuals.q_rstd
  rstd_k = residuals.k_rstd
  cu_seqlens = residuals.cu_seqlens
  aligned_cu = residuals.aligned_cu_seqlens
  chunk_indices_bwd = residuals.chunk_indices
  segment_ids_aligned = residuals.aligned_segment_ids
  segment_ids = residuals.segment_ids
  cp_metadata = residuals.cp_metadata

  cu_seqlens_bwd = None
  T_orig = None
  if cu_seqlens is not None:
    T_orig = do.shape[2]
    [do], [], _, _ = _align_seqs(
        [do],
        [],
        cu_seqlens,
        align=chunk_size,
        aligned_cu_seqlens=aligned_cu,
    )
    cu_seqlens_bwd = aligned_cu

  if (
      context_parallel_metadata is not None
      and context_parallel_metadata.is_cp_enabled
  ):
    if segment_ids is None:
      raise ValueError("backward CP requires rank-local `segment_ids`.")
    if cp_metadata is None:
      raise ValueError("backward CP requires metadata retained by forward.")
    is_first_rank, is_last_rank, pre_num_ranks, post_num_ranks = cp_metadata
    context_parallel_metadata = dataclasses.replace(
        context_parallel_metadata,
        is_first_rank=is_first_rank,
        is_last_rank=is_last_rank,
        pre_num_ranks=pre_num_ranks,
        post_num_ranks=post_num_ranks,
    )

  effective_max_num_segments = max_num_segments
  if aligned_cu is not None:
    effective_max_num_segments = aligned_cu.shape[-1] - 1

  original_cu_seqlens = cu_seqlens
  g = g_cumsum
  cu_seqlens = cu_seqlens_bwd
  chunk_indices = chunk_indices_bwd
  segment_ids = (
      segment_ids_aligned if segment_ids_aligned is not None else segment_ids
  )
  max_num_segments = effective_max_num_segments

  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT
  scale = K**-0.5 if scale is None else scale
  if (cu_seqlens is None) and (segment_ids is not None):
    # per-batch cu_seqlens [B, N+1]
    caller_max_num_segments = max_num_segments
    if caller_max_num_segments is not None:
      max_num_segments = caller_max_num_segments
    else:
      if initial_state is None:
        raise ValueError(
            "`max_num_segments` is required when `segment_ids` is provided "
            "without `initial_state`."
        )
      # Varlen: (B, N, H, K, V)
      max_num_segments = initial_state.shape[1]
    cu_seqlens = segment_ids_to_seqlens(segment_ids, max_segs=max_num_segments)

  # max_num_segments must be provided when segment_ids are used, unless
  # cu_seqlens was already derived upstream (e.g. from forward residuals).
  if (
      segment_ids is not None
      and cu_seqlens is None
      and max_num_segments is None
  ):
    raise ValueError(
        "`max_num_segments` is required when segment metadata is unavailable."
    )
  if do.shape[2] != T:
    raise ValueError(f"aligned do has T={do.shape[2]}, expected {T}")
  if Aqk.shape[-1] != BT or Akk.shape[-1] != BT:
    raise ValueError(f"Aqk/Akk block size must equal chunk_size={BT}")

  # When use_gate_in_kernel=True and disable_recompute=False, g_cumsum is
  # None from forward (freed to save memory) and will be recomputed in
  # Stage 0 below from g_org.  Only assert non-None when we won't recompute.
  if not (use_gate_in_kernel and not disable_recompute):
    if g is None:
      raise ValueError("g (post-cumsum, log2 space) must be provided")
  if T % BT != 0:
    raise ValueError(f"T={T} must be divisible by chunk_size={BT}")
  _cp_active = (
      context_parallel_metadata is not None
      and context_parallel_metadata.is_cp_enabled
  )

  # ============= assert input shapes and static properties =============
  # initial_state/dht: [B, H, K, V] (non-varlen) or [N, H, K, V] (varlen)

  if (
      cp_megakernel
      and _cp_active
      and q.dtype == jnp.bfloat16
      and K % 128 == 0
      and V % 128 == 0
  ):
    from tokamax._src.ops.kda import pallas_mosaic_tpu_cp_megakernel as mega

    if use_gate_in_kernel:
      g = kda_gate_chunk_cumsum(
          g=g_org,
          a_log=a_log,
          chunk_size=BT,
          scale=RCP_LN2,
          delta_time_bias=delta_time_bias,
          lower_bound=lower_bound,
      )
    dq, dk, dv, db, dg, dh0 = mega.chunk_kda_bwd_fusion(
        q,
        k,
        v,
        beta,
        Aqk,
        Akk,
        g,
        h,
        do,
        dht,
        initial_state,
        scale,
        segment_ids=segment_ids,
        chunk_size=BT,
        disable_recompute=disable_recompute,
        cp_active=True,
        cp_context=context_parallel_metadata,
        cp_size=context_parallel_metadata.cp_size,
        cp_axis_name=context_parallel_metadata.axis_name,
        # `return_dh0` and `has_initial_state` answer different questions:
        # the former follows the caller's intent (propagate the gradient of a
        # user-supplied state through the VJP), while the latter tells the
        # kernel whether a recurrence seed actually exists. In CP the forward
        # replaces a missing user state with the CP-prepared state merged from
        # the previous rank, so `initial_state is not None` is always true
        # here and that state must seed the recurrence regardless of user
        # intent; passing the caller flag instead would drop the seed and
        # corrupt the gradients.
        N_MAX=max_num_segments,
        return_dh0=has_initial_state,
        has_initial_state=initial_state is not None,
    )
    initial_state = None
    # The shared empty-sequence fixup below reads dht_m4; bind it here so the
    # staged-only assignment does not leave it unbound on the mega path.
    dht_m4 = dht
  else:
    # CP still needs w/qg/kg and dv before its state-gradient collective.
    # Rematerialization retains CP state reconstruction before that barrier.
    fuse_cp_local = (
        fuse_cp_backward
        and fuse_backward
        and (disable_recompute or fuse_rematerialization)
        and _cp_active
        and K % 128 == 0
        and V % 128 == 0
    )
    fuse_remat = (
        fuse_rematerialization
        and fuse_backward
        and not disable_recompute
        and not _cp_active
        and H > 1
        and K % 128 == 0
        and V % 128 == 0
    )
    fuse_local_backward = (
        fuse_backward
        and (disable_recompute or fuse_remat)
        and not _cp_active
        and K % 128 == 0
        and V % 128 == 0
    )

    if disable_recompute:
      # Path A: save-h fast path.
      if use_gate_in_kernel:
        if a_log is None:
          raise ValueError("a_log is required when use_gate_in_kernel=True")
        assert g_org is not None
        g_cumsum = kda_gate_chunk_cumsum(
            g=g_org,
            a_log=a_log,
            chunk_size=BT,
            scale=RCP_LN2,
            delta_time_bias=delta_time_bias,
            lower_bound=lower_bound,
        )
        g = g_cumsum  # already [H,B,T,K]

      if h is None:
        raise ValueError("saved h is required when recompute is disabled")
      if h.shape[2] != NT:
        raise ValueError(f"saved h has NT={h.shape[2]}, expected {NT}")

      # M1 fusion: recompute w/qg/kg + v_new in one kernel (no u HBM round-trip).
      if fuse_local_backward:
        w = qg = kg = None
        v_new = residuals.v_new
      else:
        w, qg, kg, v_new = fused_recompute_w_u_vnew_from_h_pallas(
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=Akk,
            g=g,
            h=h,
            chunk_size=BT,
        )
    else:
      # Path B: full recompute fallback.
      if use_gate_in_kernel:
        if a_log is None:
          raise ValueError("a_log is required when use_gate_in_kernel=True")
        assert g_org is not None
        g_cumsum = kda_gate_chunk_cumsum(
            g=g_org,
            a_log=a_log,
            chunk_size=BT,
            scale=RCP_LN2,
            delta_time_bias=delta_time_bias,
            lower_bound=lower_bound,
        )
        g = g_cumsum  # already [H,B,T,K]

      if fuse_remat:
        h, v_new = _rematerialize_states_pallas(
            q, k, v, beta, Akk, g, initial_state, segment_ids, chunk_size=BT
        )
        w = u = qg = kg = None
      else:
        # recompute_w_u_fwd is natively [H,B,T,X]
        w, u, qg, kg = _recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=Akk,
            q=q,
            gk=g,
            chunk_size=BT,
        )
        if kg is None:
          raise RuntimeError("KDA recompute did not produce gated keys.")

        # chunk_gated_delta_rule_fwd_h expects [B,T,H,X]
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u,
            gk=g,
            initial_state=initial_state,
            output_final_state=False,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
        )
        # Varlen: pad h from NT_total to NT chunks (padding chunks get zero state)
        if cu_seqlens is not None and h.shape[2] < NT:
          h = jnp.pad(h, ((0, 0), (0, 0), (0, NT - h.shape[2]), (0, 0), (0, 0)))

    # ---- Stage 1: dAqk and initial dv ----
    if fuse_local_backward:
      dAqk = dv = None
    else:
      dAqk, dv = chunk_kda_bwd_dAv_kernel(
          q=q,
          k=k,
          v=v_new,
          do=do,
          A=Aqk,
          scale=scale,
          chunk_size=chunk_size,
          store_dAqk=not fuse_cp_local,
      )

    if _cp_active:
      assert context_parallel_metadata is not None
      if segment_ids is None:
        raise ValueError("backward CP requires rank-local segment_ids")
      if context_parallel_metadata.post_num_ranks is None:
        raise ValueError("backward CP requires post_num_ranks")
      if context_parallel_metadata.is_last_rank is None:
        raise ValueError("backward CP requires is_last_rank")
      # pre_process requires segment_ids length == aligned T (q.shape[2]).
      # Caller's segment_ids is un-aligned (length T_orig); pad with 0
      # (= padding seg id, OOB chunks naturally inactive).
      if segment_ids.ndim == 1:
        segment_ids = segment_ids[None,]
      T_seg = segment_ids.shape[-1]
      if T_seg < T:
        pad_width = ((0, 0), (0, T - T_seg))
        segment_ids = jnp.pad(segment_ids, pad_width)
      elif T_seg > T:
        segment_ids = segment_ids[..., :T]
      # Inputs already in [H, B, T, X]; pre_process consumes this layout
      # directly (no transpose round-trip).
      dS_ext, dM = chunk_gated_delta_rule_bwd_dhu_pre_process(
          q=qg,
          k=kg,
          w=w,
          do=do,
          dv=dv,
          gk=g,
          scale=scale,
          segment_ids=segment_ids,
          chunk_size=BT,
          use_exp2=True,
      )
      # Pack dS_ext [B,H,K,V] and dM [B,H,K,K] into one tensor along the last
      # axis so a single all_gather covers both, halving the CP collective cost.
      packed = jnp.concatenate([dS_ext, dM], axis=-1)  # [B, H, K, V+K]
      packed_all, _ = all_gather_into_tensor(
          packed, context_parallel_metadata.axis_name
      )
      dS_ext_all = packed_all[..., :V]  # [cp, B, H, K, V]
      dM_all = packed_all[..., V : V + K]  # [cp, B, H, K, K]
      rank = jax.lax.axis_index(context_parallel_metadata.axis_name)
      post_num = jnp.asarray(context_parallel_metadata.post_num_ranks)
      is_last = jnp.asarray(context_parallel_metadata.is_last_rank)
      ds_list = []
      for b in range(B):
        ds_b = _merge_dht(
            dS_ext_all[:, b : b + 1],  # [cp, 1, H, K, V]
            dM_all[:, b : b + 1],  # [cp, 1, H, K, K]
            rank=rank,
            post_num_ranks=post_num[b],
            is_last_rank=is_last[b],
        )
        ds_list.append(ds_b)  # [1, H, K, V]
      dS_in = jnp.concatenate(ds_list, axis=0)  # [B, H, K, V]

      # dS_in: [B, H, K, V] — merged downstream gradient for each batch element.
      # M4 expects dht as [B, N, H, K, V] with per-batch segment indexing.
      # dS_in[b] goes to slot (last_seg_id_b - 1) in its own N dimension.
      assert max_num_segments is not None
      N = max_num_segments
      dht = jnp.zeros((B, N, H, K, V), dtype=jnp.float32)
      max_per_batch = jnp.max(segment_ids, axis=1)  # [B]
      for b in range(B):
        last_seg_id_b = max_per_batch[b]
        has_real_b = last_seg_id_b > 0
        dht_slot = jnp.maximum(last_seg_id_b - 1, 0)
        dht_slot = jnp.minimum(dht_slot, N - 1)
        dht_value_b = jnp.where(has_real_b, dS_in[b], jnp.zeros_like(dS_in[b]))
        dht = dht.at[b, dht_slot, :, :, :].set(dht_value_b)
      initial_state = None

    # M4 requires 2D segment_ids [B, T]
    if segment_ids is not None and segment_ids.ndim == 1:
      segment_ids = segment_ids[None,]

    # ---- Stage 2+3+4+5: M4 mega fusion (dhu + WY + intra + cumsum) ----
    # M4 expects dht as [B, N, H, K, V].
    # Normalize 4D dht to 5D (uniform: insert N=1 at dim1).
    if dht is not None and dht.ndim == 4 and segment_ids is None:
      dht = dht[:, None, :, :, :]
    dht_m4 = dht
    fuse_reverse = fuse_local_backward or fuse_cp_local
    dq, dk, dv, db, dg, dh0 = _fused_dhu_wy_intra_cumsum_pallas_jit(
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        qg=qg,
        kg=kg,
        w=w,
        g=g,
        beta=beta,
        A=Akk,
        h=h,
        do=do,
        dv0=dv,
        dAqk=dAqk,
        dht=dht_m4,
        scale=scale,
        segment_ids=segment_ids,
        chunk_size=chunk_size,
        use_exp2=True,
        return_dh0=initial_state is not None,
        max_num_segments=max_num_segments,
        fuse_recompute=fuse_reverse,
        forward_aqk=Aqk if fuse_reverse else None,
        packed_inputs=packed_inputs if fuse_local_backward else None,
        original_cu_seqlens=original_cu_seqlens,
        aligned_cu_seqlens=aligned_cu,
    )

  # Padding chunks need not write their backward output buffers. Clear those
  # lanes before gate-parameter reductions, rather than reducing uninitialized
  # values or multiplying a raw-gate -inf sentinel by a zero cotangent.
  if segment_ids is not None:
    valid = segment_ids[None, :, :] != 0
    dq = jnp.where(valid[..., None], dq, 0)
    dk = jnp.where(valid[..., None], dk, 0)
    dv = jnp.where(valid[..., None], dv, 0)
    dg = jnp.where(valid[..., None], dg, 0)
    db = jnp.where(valid, db, 0)
    if g_org is not None:
      g_org = jnp.where(valid[..., None], g_org, 0)

  # Empty sequences preserve their initial state, so their initial-state
  # gradient is exactly the final-state cotangent (zero when not requested).
  if dh0 is not None and cu_seqlens is not None:
    empty = (jnp.diff(cu_seqlens, axis=-1) == 0)[..., None, None, None]
    dh0 = jnp.where(empty, 0 if dht_m4 is None else dht_m4, dh0)

  dA, dbias = None, None
  if use_gate_in_kernel:
    assert g_org is not None and a_log is not None
    dg, dA, dbias = kda_gate_bwd(
        g=g_org,
        a_log=a_log,
        delta_time_bias=delta_time_bias,
        dyg=dg,
        lower_bound=lower_bound,
    )

  # Non-varlen: squeeze dh0 5D [B,1,H,K,V] → 4D [B,H,K,V]
  # (varlen callers pass 5D, so dh0 stays 5D)
  _is_varlen = cu_seqlens is not None or (segment_ids is not None)
  if dh0 is not None and dh0.ndim == 5 and not _is_varlen:
    dh0 = dh0[:, 0]

  cu_seqlens = original_cu_seqlens
  if use_qk_l2norm:
    assert rstd_q is not None and rstd_k is not None
    dq = l2norm_bwd(q, rstd_q, dq)
    dk = l2norm_bwd(k, rstd_k, dk)

  if cu_seqlens is not None:
    if packed_gradients and not _cp_active:
      # Gate reductions and normalization are complete. Casting commutes with
      # token selection and reduces the compactor's input/output traffic.
      dq = compact_output(dq.astype(q.dtype), cu_seqlens, aligned_cu, T_orig)
      dk = compact_output(dk.astype(k.dtype), cu_seqlens, aligned_cu, T_orig)
      dv = compact_output(dv.astype(v.dtype), cu_seqlens, aligned_cu, T_orig)
      dg = compact_output(
          dg.astype(g_dtype_marker.dtype), cu_seqlens, aligned_cu, T_orig
      )
    else:
      dq = _unalign_output(dq, cu_seqlens, aligned_cu, T_orig)
      dk = _unalign_output(dk, cu_seqlens, aligned_cu, T_orig)
      dv = _unalign_output(dv, cu_seqlens, aligned_cu, T_orig)
      dg = _unalign_output(dg, cu_seqlens, aligned_cu, T_orig)
    # Scalar beta uses the existing gather, matching the source migration.
    db = _unalign_output(db, cu_seqlens, aligned_cu, T_orig)

  if dh0 is not None and dh0.ndim == 4 and has_initial_state:
    dh0 = dh0[:, None]

  return (
      dq.astype(q.dtype),
      dk.astype(k.dtype),
      dv.astype(v.dtype),
      dg.astype(g_dtype_marker.dtype),
      db.astype(beta.dtype),
      dA,
      dbias,
      dh0,
      None,
  )
