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

"""Pallas TPU compute kernel for Native Sparse Attention (NSA)."""

# Dimensions:
# B: batch_size
# H: num_heads
# T: seq_len
# D: head_dim
# NB: num_q_blocks (T // block_q)
# BQ: block_q (chunk_size)
# K_eff: effective_topk (min(topk, NB))
# W: sliding window size
# BC: cmp_block_size

import functools
import math
from typing import Any

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
    bs_cmp: int = _CMP_BLOCK_SIZE,
) -> jax.Array:
  """Computes coarse block-compression attention branch."""
  batch_size, num_heads, seq_len, head_dim = q.shape
  num_cmp_blocks = seq_len // bs_cmp

  k_cmp = jnp.mean(
      k.reshape(batch_size, num_heads, num_cmp_blocks, bs_cmp, head_dim),
      axis=3,
  )
  v_cmp = jnp.mean(
      v.reshape(batch_size, num_heads, num_cmp_blocks, bs_cmp, head_dim),
      axis=3,
  )

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

  q_pos = jnp.arange(seq_len)[:, None]
  c_pos = (jnp.arange(num_cmp_blocks) * bs_cmp)[None, :]
  causal_mask = (c_pos + bs_cmp - 1) <= q_pos

  logits = jnp.where(causal_mask[None, None], logits, -1e9)
  weights = jax.nn.softmax(logits, axis=-1)
  weights = jnp.where((q_pos >= bs_cmp - 1)[None, None], weights, 0.0)
  return jnp.einsum(
      "bhtc,bhcd->bhtd",
      weights,
      v_cmp,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )


def _flash_slc_kernel(
    # Scalar prefetch (SMEM).
    topk_smem_ref,  # [B, H, NB, K_eff] int32
    # Inputs (VMEM).
    q_vmem_ref,  # [1, 1, 1, BQ, D] dtype
    k_vmem_ref,  # [1, 1, 1, BQ, D] dtype
    v_vmem_ref,  # [1, 1, 1, BQ, D] dtype
    # Outputs (VMEM).
    o_vmem_ref,  # [1, 1, 1, BQ, D] dtype
    # Scratch (VMEM).
    m_scratch_ref,  # [BQ, D] float32
    l_scratch_ref,  # [BQ, D] float32
    o_scratch_ref,  # [BQ, D] float32
    *,
    scale: float,
) -> None:
  # Dimensions.
  q_idx = pl.program_id(2)
  s_idx = pl.program_id(3)
  block_q = q_vmem_ref.shape[3]
  head_dim = q_vmem_ref.shape[4]
  effective_topk = topk_smem_ref.shape[3]

  @pl.when(s_idx == 0)
  def _init_diag():
    q_i = q_vmem_ref[0, 0, 0].astype(jnp.float32)
    k_j = k_vmem_ref[0, 0, 0].astype(jnp.float32)
    v_j = v_vmem_ref[0, 0, 0].astype(jnp.float32)

    qk = (
        jnp.matmul(
            q_i,
            k_j.T,
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        * scale
    )
    row_idx = jax.lax.broadcasted_iota(jnp.int32, (block_q, block_q), 0)
    col_idx = jax.lax.broadcasted_iota(jnp.int32, (block_q, block_q), 1)
    qk = jnp.where(col_idx <= row_idx, qk, -1e9)

    m_curr = jnp.max(qk, axis=-1, keepdims=True)
    p = jnp.exp(qk - m_curr)
    l_curr = jnp.sum(p, axis=-1, keepdims=True)
    o_curr = jnp.matmul(
        p,
        v_j,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )

    m_scratch_ref[...] = jnp.broadcast_to(m_curr, (block_q, head_dim))
    l_scratch_ref[...] = jnp.broadcast_to(l_curr, (block_q, head_dim))
    o_scratch_ref[...] = o_curr

  @pl.when((s_idx > 0) & ((s_idx - 1) < q_idx))
  def _acc_hist():
    q_i = q_vmem_ref[0, 0, 0].astype(jnp.float32)
    k_j = k_vmem_ref[0, 0, 0].astype(jnp.float32)
    v_j = v_vmem_ref[0, 0, 0].astype(jnp.float32)

    qk = (
        jnp.matmul(
            q_i,
            k_j.T,
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        * scale
    )
    m_prev = m_scratch_ref[...]
    l_prev = l_scratch_ref[...]
    o_prev = o_scratch_ref[...]

    m_curr = jnp.broadcast_to(
        jnp.max(qk, axis=-1, keepdims=True), (block_q, head_dim)
    )
    m_new = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.exp(qk - m_new[:, :1])
    l_curr = jnp.broadcast_to(
        jnp.sum(p, axis=-1, keepdims=True), (block_q, head_dim)
    )
    l_new = alpha * l_prev + l_curr
    o_new = alpha * o_prev + jnp.matmul(
        p,
        v_j,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )

    m_scratch_ref[...] = m_new
    l_scratch_ref[...] = l_new
    o_scratch_ref[...] = o_new

  @pl.when(s_idx == effective_topk)
  def _finalize():
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
  batch_size, num_heads, seq_len, head_dim = q.shape
  block_q = chunk_size
  num_q_blocks = seq_len // block_q
  effective_topk = min(topk, num_q_blocks)

  q_blocks = q.reshape(batch_size, num_heads, num_q_blocks, block_q, head_dim)
  k_blocks = k.reshape(batch_size, num_heads, num_q_blocks, block_q, head_dim)
  v_blocks = v.reshape(batch_size, num_heads, num_q_blocks, block_q, head_dim)

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
  q_idx = jnp.arange(num_q_blocks)[:, None]
  k_idx = jnp.arange(num_q_blocks)[None, :]
  historical_mask = k_idx < q_idx
  masked_routing = jnp.where(
      historical_mask[None, None], routing_scores, -jnp.inf
  )
  _, topk_idx = jax.lax.top_k(masked_routing, effective_topk)
  topk_idx = jax.lax.stop_gradient(topk_idx).astype(jnp.int32)

  def _q_index_map(b_i, h_i, q_i, s_i, topk_ref):
    del s_i, topk_ref
    return (b_i, h_i, q_i, 0, 0)

  def _kv_index_map(b_i, h_i, q_i, s_i, topk_ref):
    routed_idx = topk_ref[b_i, h_i, q_i, jnp.maximum(s_i - 1, 0)]
    kv_block_idx = jnp.where(s_i == 0, q_i, routed_idx)
    return (b_i, h_i, kv_block_idx, 0, 0)

  q_spec = pl.BlockSpec((1, 1, 1, block_q, head_dim), _q_index_map)
  kv_spec = pl.BlockSpec((1, 1, 1, block_q, head_dim), _kv_index_map)
  scratch_shape = pltpu.VMEM((block_q, head_dim), jnp.float32)

  if interpret is None:
    interpret = jax.default_backend() == "cpu"
  out_blocks = pl.pallas_call(
      functools.partial(_flash_slc_kernel, scale=scale),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(batch_size, num_heads, num_q_blocks, effective_topk + 1),
          in_specs=[q_spec, kv_spec, kv_spec],
          out_specs=q_spec,
          scratch_shapes=[scratch_shape, scratch_shape, scratch_shape],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
      ),
      out_shape=jax.ShapeDtypeStruct(
          (batch_size, num_heads, num_q_blocks, block_q, head_dim),
          dtype=q.dtype,
      ),
      interpret=interpret,
  )(topk_idx, q_blocks, k_blocks, v_blocks)
  return out_blocks.reshape(batch_size, num_heads, seq_len, head_dim)


def _flash_swa_kernel(
    # Inputs (VMEM).
    q_vmem_ref,  # [1, 1, 1, BQ, D] dtype
    k_vmem_ref,  # [1, 1, 1, BQ, D] dtype
    v_vmem_ref,  # [1, 1, 1, BQ, D] dtype
    # Outputs (VMEM).
    o_vmem_ref,  # [1, 1, 1, BQ, D] dtype
    # Scratch (VMEM).
    m_scratch_ref,  # [BQ, D] float32
    l_scratch_ref,  # [BQ, D] float32
    o_scratch_ref,  # [BQ, D] float32
    *,
    scale: float,
    window: int,
    num_win_blocks: int,
) -> None:
  # Dimensions.
  q_idx = pl.program_id(2)
  w_step = pl.program_id(3)
  block_q = q_vmem_ref.shape[3]
  head_dim = q_vmem_ref.shape[4]

  @pl.when(w_step == 0)
  def _init_diag():
    q_i = q_vmem_ref[0, 0, 0].astype(jnp.float32)
    k_j = k_vmem_ref[0, 0, 0].astype(jnp.float32)
    v_j = v_vmem_ref[0, 0, 0].astype(jnp.float32)

    qk = (
        jnp.matmul(
            q_i,
            k_j.T,
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        * scale
    )
    row_idx = jax.lax.broadcasted_iota(jnp.int32, (block_q, block_q), 0)
    col_idx = jax.lax.broadcasted_iota(jnp.int32, (block_q, block_q), 1)
    win_mask = (col_idx <= row_idx) & (col_idx >= row_idx - window)
    qk = jnp.where(win_mask, qk, -1e9)

    m_curr = jnp.max(qk, axis=-1, keepdims=True)
    p = jnp.where(win_mask, jnp.exp(qk - m_curr), 0.0)
    l_curr = jnp.sum(p, axis=-1, keepdims=True)
    o_curr = jnp.matmul(
        p,
        v_j,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )

    m_scratch_ref[...] = jnp.broadcast_to(m_curr, (block_q, head_dim))
    l_scratch_ref[...] = jnp.broadcast_to(l_curr, (block_q, head_dim))
    o_scratch_ref[...] = o_curr

  @pl.when((w_step > 0) & (q_idx >= w_step))
  def _acc_prev():
    q_i = q_vmem_ref[0, 0, 0].astype(jnp.float32)
    k_j = k_vmem_ref[0, 0, 0].astype(jnp.float32)
    v_j = v_vmem_ref[0, 0, 0].astype(jnp.float32)

    qk = (
        jnp.matmul(
            q_i,
            k_j.T,
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        * scale
    )
    row_idx = jax.lax.broadcasted_iota(jnp.int32, (block_q, block_q), 0)
    col_idx = jax.lax.broadcasted_iota(jnp.int32, (block_q, block_q), 1)
    dist = w_step * block_q + row_idx - col_idx
    win_mask = dist <= window
    qk = jnp.where(win_mask, qk, -1e9)

    m_prev = m_scratch_ref[...]
    l_prev = l_scratch_ref[...]
    o_prev = o_scratch_ref[...]

    m_curr = jnp.broadcast_to(
        jnp.max(qk, axis=-1, keepdims=True), (block_q, head_dim)
    )
    m_new = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.where(win_mask, jnp.exp(qk - m_new[:, :1]), 0.0)
    l_curr = jnp.broadcast_to(
        jnp.sum(p, axis=-1, keepdims=True), (block_q, head_dim)
    )
    l_new = alpha * l_prev + l_curr
    o_new = alpha * o_prev + jnp.matmul(
        p,
        v_j,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )

    m_scratch_ref[...] = m_new
    l_scratch_ref[...] = l_new
    o_scratch_ref[...] = o_new

  @pl.when(w_step == num_win_blocks)
  def _finalize():
    o_final = o_scratch_ref[...] / l_scratch_ref[...]
    o_vmem_ref[0, 0, 0] = o_final.astype(o_vmem_ref.dtype)


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
  batch_size, num_heads, seq_len, head_dim = q.shape
  block_q = chunk_size
  num_q_blocks = seq_len // block_q
  num_win_blocks = min(math.ceil(window / block_q), num_q_blocks - 1)

  q_blocks = q.reshape(batch_size, num_heads, num_q_blocks, block_q, head_dim)
  k_blocks = k.reshape(batch_size, num_heads, num_q_blocks, block_q, head_dim)
  v_blocks = v.reshape(batch_size, num_heads, num_q_blocks, block_q, head_dim)

  def _q_index_map(b_i, h_i, q_i, w_step):
    del w_step
    return (b_i, h_i, q_i, 0, 0)

  def _kv_index_map(b_i, h_i, q_i, w_step):
    kv_idx = jnp.maximum(q_i - w_step, 0)
    return (b_i, h_i, kv_idx, 0, 0)

  q_spec = pl.BlockSpec((1, 1, 1, block_q, head_dim), _q_index_map)
  kv_spec = pl.BlockSpec((1, 1, 1, block_q, head_dim), _kv_index_map)
  scratch_shape = pltpu.VMEM((block_q, head_dim), jnp.float32)

  if interpret is None:
    interpret = jax.default_backend() == "cpu"
  out_blocks = pl.pallas_call(
      functools.partial(
          _flash_swa_kernel,
          scale=scale,
          window=window,
          num_win_blocks=num_win_blocks,
      ),
      grid=(batch_size, num_heads, num_q_blocks, num_win_blocks + 1),
      in_specs=[q_spec, kv_spec, kv_spec],
      out_specs=q_spec,
      scratch_shapes=[scratch_shape, scratch_shape, scratch_shape],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
      ),
      out_shape=jax.ShapeDtypeStruct(
          (batch_size, num_heads, num_q_blocks, block_q, head_dim),
          dtype=q.dtype,
      ),
      interpret=interpret,
  )(q_blocks, k_blocks, v_blocks)
  return out_blocks.reshape(batch_size, num_heads, seq_len, head_dim)


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
  seq_len = q.shape[2]
  if seq_len % chunk_size != 0:
    raise ValueError(
        f"Sequence length ({seq_len}) must be divisible by chunk_size"
        f" ({chunk_size})."
    )
  if seq_len % cmp_block_size != 0:
    raise ValueError(
        f"Sequence length ({seq_len}) must be divisible by cmp_block_size"
        f" ({cmp_block_size})."
    )

  orig_dtype = q.dtype
  if scale is None:
    head_dim = q.shape[-1]
    scale = head_dim ** -0.5

  q_f32 = q.astype(jnp.float32)
  k_f32 = k.astype(jnp.float32)
  v_f32 = v.astype(jnp.float32)

  o_cmp = _compute_cmp_branch(q_f32, k_f32, v_f32, scale, bs_cmp=cmp_block_size)
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
