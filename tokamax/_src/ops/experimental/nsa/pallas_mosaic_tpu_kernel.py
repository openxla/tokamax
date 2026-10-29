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

"""Pallas TPU compute kernels for Native Sparse Attention (NSA)."""

# Dimensions.
#   B   batch size
#   H   number of attention heads
#   T   sequence length
#   D   head dimension
#   W   local causal sliding window size
#   tt  query/KV block tile extent along T (chunk_size)
#   nt  number of query/KV blocks along T (nt = T // tt)
#   tc  compression block tile extent along T (cmp_block_size)
#   nc  number of compressed blocks along T (nc = T // tc)
#   tk  number of selected historical blocks (min(topk, nt))
#   nk  selection grid extent per query block (nk = tk + 1)
#   tw  number of preceding blocks spanned by sliding window
#   nw  sliding-window grid extent per query block (nw = tw + 1)

import functools
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_CHUNK_SIZE = 256
_TOPK = 4
_WINDOW = 128
_CMP_BLOCK_SIZE = 64


def _compute_cmp_branch(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    cmp_block_size: int = _CMP_BLOCK_SIZE,
) -> jax.Array:
  """Computes coarse block-compression attention branch."""
  B, H, T, D = q.shape
  tc = cmp_block_size
  nc = T // tc

  k_cmp = jnp.mean(k.reshape(B, H, nc, tc, D), axis=3)
  v_cmp = jnp.mean(v.reshape(B, H, nc, tc, D), axis=3)

  logits = (
      jnp.einsum(
          "bhtd,bhcd->bhtc",
          q,
          k_cmp,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * scale
  )

  q_pos = jnp.arange(T)[:, None]
  c_pos = (jnp.arange(nc) * tc)[None, :]
  causal_mask = (c_pos + tc - 1) <= q_pos

  logits = jnp.where(causal_mask[None, None], logits, -1e9)
  weights = jax.nn.softmax(logits, axis=-1)
  weights = jnp.where((q_pos >= tc - 1)[None, None], weights, 0.0)
  return jnp.einsum(
      "bhtc,bhcd->bhtd",
      weights,
      v_cmp,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )


def _flash_slc_kernel(
    # Inputs.
    topk_smem_ref,  # [B, H, nt, tk] i32         full array in SMEM
    q_vmem_ref,     # [1, 1, 1, tt, D] f32       one query block in VMEM
    k_vmem_ref,     # [1, 1, 1, tt, D] f32       routed KV block in VMEM
    v_vmem_ref,     # [1, 1, 1, tt, D] f32       routed KV block in VMEM
    # Outputs.
    o_vmem_ref,     # [1, 1, 1, tt, D] f32       one output block in VMEM
    # Scratch.
    m_scratch_ref,  # [tt, D] f32                running row-max across nk
    l_scratch_ref,  # [tt, D] f32                running row-sum across nk
    o_scratch_ref,  # [tt, D] f32                unnormalized output across nk
    *,
    scale: float,
    nk: int,
) -> None:
  del topk_smem_ref
  q_i = pl.program_id(2)
  s_i = pl.program_id(3)
  _, _, _, tt, D = q_vmem_ref.shape

  # Step s_i == 0 processes the causal diagonal block (kv_block == q_i) to seed
  # the online softmax accumulators without needing an extra zero-init pass.
  @pl.when(s_i == 0)
  def _():
    q_tile = q_vmem_ref[0, 0, 0].astype(jnp.float32)
    k_tile = k_vmem_ref[0, 0, 0].astype(jnp.float32)
    v_tile = v_vmem_ref[0, 0, 0].astype(jnp.float32)

    # Accumulate in f32 with HIGHEST precision so MXU bf16 rounding does not
    # drift against the f32 reference across long block reductions.
    qk = (
        jnp.matmul(
            q_tile,
            k_tile.T,
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        * scale
    )
    row_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 0)
    col_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 1)
    qk = jnp.where(col_idx <= row_idx, qk, -1e9)

    m_curr = jnp.max(qk, axis=-1, keepdims=True)
    p = jnp.exp(qk - m_curr)
    l_curr = jnp.sum(p, axis=-1, keepdims=True)
    o_curr = jnp.matmul(
        p,
        v_tile,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )

    # Broadcast 1D row statistics across D so VMEM scratch buffers maintain
    # 2D [sublane, lane] alignment without sublane relayout overhead.
    m_scratch_ref[...] = jnp.broadcast_to(m_curr, (tt, D))
    l_scratch_ref[...] = jnp.broadcast_to(l_curr, (tt, D))
    o_scratch_ref[...] = o_curr

  # Steps s_i > 0 stream the tk routed historical blocks; only blocks strictly
  # preceding q_i (s_i - 1 < q_i) are causally valid.
  @pl.when((s_i > 0) & ((s_i - 1) < q_i))
  def _():
    q_tile = q_vmem_ref[0, 0, 0].astype(jnp.float32)
    k_tile = k_vmem_ref[0, 0, 0].astype(jnp.float32)
    v_tile = v_vmem_ref[0, 0, 0].astype(jnp.float32)

    qk = (
        jnp.matmul(
            q_tile,
            k_tile.T,
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        * scale
    )
    m_prev = m_scratch_ref[...]
    l_prev = l_scratch_ref[...]
    o_prev = o_scratch_ref[...]

    m_curr = jnp.broadcast_to(jnp.max(qk, axis=-1, keepdims=True), (tt, D))
    m_new = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(qk - m_new[:, :1])
    l_curr = jnp.broadcast_to(jnp.sum(p, axis=-1, keepdims=True), (tt, D))
    l_new = alpha * l_prev + l_curr
    o_new = alpha * o_prev + jnp.matmul(
        p,
        v_tile,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )

    m_scratch_ref[...] = m_new
    l_scratch_ref[...] = l_new
    o_scratch_ref[...] = o_new

  @pl.when(s_i == nk - 1)
  def _():
    o_final = o_scratch_ref[...] / l_scratch_ref[...]
    o_vmem_ref[0, 0, 0] = o_final.astype(o_vmem_ref.dtype)


def _compute_slc_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    topk: int = _TOPK,
    chunk_size: int = _CHUNK_SIZE,
    interpret: bool | None = None,
) -> jax.Array:
  """Computes fine-grained block selection branch via Pallas TPU kernel."""
  B, H, T, D = q.shape
  tt = chunk_size
  nt = T // tt
  tk = min(topk, nt)
  nk = tk + 1

  q_blocks = q.reshape(B, H, nt, tt, D)
  k_blocks = k.reshape(B, H, nt, tt, D)
  v_blocks = v.reshape(B, H, nt, tt, D)

  q_mean = jnp.mean(q_blocks, axis=3)
  k_mean = jnp.mean(k_blocks, axis=3)
  routing_scores = (
      jnp.einsum(
          "bhqd,bhnd->bhqn",
          q_mean,
          k_mean,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * scale
  )
  q_idx = jnp.arange(nt)[:, None]
  k_idx = jnp.arange(nt)[None, :]
  historical_mask = k_idx < q_idx
  masked_routing = jnp.where(
      historical_mask[None, None], routing_scores, -jnp.inf
  )
  _, topk_idx = jax.lax.top_k(masked_routing, tk)
  topk_idx = jax.lax.stop_gradient(topk_idx).astype(jnp.int32)

  def _q_index_map(b_i, h_i, q_i, s_i, topk_smem_ref):
    del s_i, topk_smem_ref
    return (b_i, h_i, q_i, 0, 0)

  def _kv_index_map(b_i, h_i, q_i, s_i, topk_smem_ref):
    routed_idx = topk_smem_ref[b_i, h_i, q_i, jnp.maximum(s_i - 1, 0)]
    kv_block_idx = jnp.where(s_i == 0, q_i, routed_idx)
    return (b_i, h_i, kv_block_idx, 0, 0)

  q_spec = pl.BlockSpec((1, 1, 1, tt, D), _q_index_map)
  kv_spec = pl.BlockSpec((1, 1, 1, tt, D), _kv_index_map)
  scratch_shape = pltpu.VMEM((tt, D), jnp.float32)

  if interpret is None:
    interpret = jax.default_backend() != "tpu"
  out_blocks = pl.pallas_call(
      functools.partial(_flash_slc_kernel, scale=scale, nk=nk),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(B, H, nt, nk),
          in_specs=[q_spec, kv_spec, kv_spec],
          out_specs=q_spec,
          scratch_shapes=[scratch_shape, scratch_shape, scratch_shape],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
      ),
      out_shape=jax.ShapeDtypeStruct((B, H, nt, tt, D), dtype=q.dtype),
      interpret=interpret,
  )(topk_idx, q_blocks, k_blocks, v_blocks)
  return out_blocks.reshape(B, H, T, D)


def _flash_swa_kernel(
    # Inputs.
    q_ref,          # [1, 1, 1, tt, D] f32       one query block
    k_ref,          # [1, 1, 1, tt, D] f32       window KV block at q_i - w_i
    v_ref,          # [1, 1, 1, tt, D] f32       window KV block at q_i - w_i
    # Outputs.
    o_ref,          # [1, 1, 1, tt, D] f32       one output block
    # Scratch.
    m_scratch_ref,  # [tt, D] f32                running row-max across nw
    l_scratch_ref,  # [tt, D] f32                running row-sum across nw
    o_scratch_ref,  # [tt, D] f32                unnormalized output across nw
    *,
    scale: float,
    window: int,
    nw: int,
) -> None:
  q_i = pl.program_id(2)
  w_i = pl.program_id(3)
  _, _, _, tt, D = q_ref.shape

  # Step w_i == 0 processes the intra-block causal sliding window on the
  # diagonal block and initializes the online softmax accumulators.
  @pl.when(w_i == 0)
  def _():
    q_tile = q_ref[0, 0, 0].astype(jnp.float32)
    k_tile = k_ref[0, 0, 0].astype(jnp.float32)
    v_tile = v_ref[0, 0, 0].astype(jnp.float32)

    qk = (
        jnp.matmul(
            q_tile,
            k_tile.T,
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        * scale
    )
    row_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 0)
    col_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 1)
    win_mask = (col_idx <= row_idx) & (col_idx >= row_idx - window)
    qk = jnp.where(win_mask, qk, -1e9)

    m_curr = jnp.max(qk, axis=-1, keepdims=True)
    p = jnp.where(win_mask, jnp.exp(qk - m_curr), 0.0)
    l_curr = jnp.sum(p, axis=-1, keepdims=True)
    o_curr = jnp.matmul(
        p,
        v_tile,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )

    m_scratch_ref[...] = jnp.broadcast_to(m_curr, (tt, D))
    l_scratch_ref[...] = jnp.broadcast_to(l_curr, (tt, D))
    o_scratch_ref[...] = o_curr

  # Steps w_i > 0 accumulate the preceding blocks (q_i - w_i) overlapped by the
  # causal sliding window of width W.
  @pl.when((w_i > 0) & (q_i >= w_i))
  def _():
    q_tile = q_ref[0, 0, 0].astype(jnp.float32)
    k_tile = k_ref[0, 0, 0].astype(jnp.float32)
    v_tile = v_ref[0, 0, 0].astype(jnp.float32)

    qk = (
        jnp.matmul(
            q_tile,
            k_tile.T,
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        * scale
    )
    row_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 0)
    col_idx = jax.lax.broadcasted_iota(jnp.int32, (tt, tt), 1)
    dist = w_i * tt + row_idx - col_idx
    win_mask = dist <= window
    qk = jnp.where(win_mask, qk, -1e9)

    m_prev = m_scratch_ref[...]
    l_prev = l_scratch_ref[...]
    o_prev = o_scratch_ref[...]

    m_curr = jnp.broadcast_to(jnp.max(qk, axis=-1, keepdims=True), (tt, D))
    m_new = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.where(win_mask, jnp.exp(qk - m_new[:, :1]), 0.0)
    l_curr = jnp.broadcast_to(jnp.sum(p, axis=-1, keepdims=True), (tt, D))
    l_new = alpha * l_prev + l_curr
    o_new = alpha * o_prev + jnp.matmul(
        p,
        v_tile,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )

    m_scratch_ref[...] = m_new
    l_scratch_ref[...] = l_new
    o_scratch_ref[...] = o_new

  @pl.when(w_i == nw - 1)
  def _():
    o_final = o_scratch_ref[...] / l_scratch_ref[...]
    o_ref[0, 0, 0] = o_final.astype(o_ref.dtype)


def _compute_swa_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    window: int = _WINDOW,
    chunk_size: int = _CHUNK_SIZE,
    interpret: bool | None = None,
) -> jax.Array:
  """Computes local sliding-window attention branch via Pallas TPU kernel."""
  B, H, T, D = q.shape
  tt = chunk_size
  nt = T // tt
  tw = min(math.ceil(window / tt), nt - 1)
  nw = tw + 1

  q_blocks = q.reshape(B, H, nt, tt, D)
  k_blocks = k.reshape(B, H, nt, tt, D)
  v_blocks = v.reshape(B, H, nt, tt, D)

  def _q_index_map(b_i, h_i, q_i, w_i):
    del w_i
    return (b_i, h_i, q_i, 0, 0)

  def _kv_index_map(b_i, h_i, q_i, w_i):
    kv_idx = jnp.maximum(q_i - w_i, 0)
    return (b_i, h_i, kv_idx, 0, 0)

  q_spec = pl.BlockSpec((1, 1, 1, tt, D), _q_index_map)
  kv_spec = pl.BlockSpec((1, 1, 1, tt, D), _kv_index_map)
  scratch_shape = pltpu.VMEM((tt, D), jnp.float32)

  if interpret is None:
    interpret = jax.default_backend() != "tpu"
  out_blocks = pl.pallas_call(
      functools.partial(
          _flash_swa_kernel,
          scale=scale,
          window=window,
          nw=nw,
      ),
      grid=(B, H, nt, nw),
      in_specs=[q_spec, kv_spec, kv_spec],
      out_specs=q_spec,
      scratch_shapes=[scratch_shape, scratch_shape, scratch_shape],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
      ),
      out_shape=jax.ShapeDtypeStruct((B, H, nt, tt, D), dtype=q.dtype),
      interpret=interpret,
  )(q_blocks, k_blocks, v_blocks)
  return out_blocks.reshape(B, H, T, D)


def nsa_pallas_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_cmp: jax.Array | None = None,
    g_slc: jax.Array | None = None,
    g_swa: jax.Array | None = None,
    *,
    chunk_size: int = _CHUNK_SIZE,
    topk: int = _TOPK,
    window: int = _WINDOW,
    cmp_block_size: int = _CMP_BLOCK_SIZE,
    scale: float | None = None,
    interpret: bool | None = None,
) -> tuple[jax.Array, None]:
  """Executes the 3-branch Native Sparse Attention forward pass."""
  if q.shape != k.shape or q.shape != v.shape:
    raise ValueError(
        f"Shapes of q, k, v must match, got q={q.shape}, k={k.shape},"
        f" v={v.shape}"
    )
  _, _, T, D = q.shape
  if T % chunk_size != 0:
    raise ValueError(
        f"Sequence length ({T}) must be divisible by chunk_size ({chunk_size})."
    )
  if T % cmp_block_size != 0:
    raise ValueError(
        f"Sequence length ({T}) must be divisible by cmp_block_size"
        f" ({cmp_block_size})."
    )

  orig_dtype = q.dtype
  if scale is None:
    scale = D ** -0.5

  q_f32 = q.astype(jnp.float32)
  k_f32 = k.astype(jnp.float32)
  v_f32 = v.astype(jnp.float32)

  o_cmp = _compute_cmp_branch(
      q_f32, k_f32, v_f32, scale, cmp_block_size=cmp_block_size
  )
  o_slc = _compute_slc_pallas(
      q_f32,
      k_f32,
      v_f32,
      scale,
      topk=topk,
      chunk_size=chunk_size,
      interpret=interpret,
  )
  o_swa = _compute_swa_pallas(
      q_f32,
      k_f32,
      v_f32,
      scale,
      window=window,
      chunk_size=chunk_size,
      interpret=interpret,
  )

  if g_cmp is None:
    g_cmp = jnp.full_like(q_f32, 1.0 / 3.0)
  else:
    g_cmp = g_cmp.astype(jnp.float32)
  if g_slc is None:
    g_slc = jnp.full_like(q_f32, 1.0 / 3.0)
  else:
    g_slc = g_slc.astype(jnp.float32)
  if g_swa is None:
    g_swa = jnp.full_like(q_f32, 1.0 / 3.0)
  else:
    g_swa = g_swa.astype(jnp.float32)

  out = g_cmp * o_cmp + g_slc * o_slc + g_swa * o_swa
  return out.astype(orig_dtype), None
