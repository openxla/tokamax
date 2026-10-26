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

"""Pallas TPU compute kernel for Mixture of Block Attention (MoBA)."""

# Dimensions:
# B: batch_size
# H: num_heads
# T: seq_len
# D: head_dim
# NB: num_blocks (T // block_q)
# BQ: block_q (chunk_size)
# K_eff: effective_topk

import functools
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_CHUNK_SIZE = 256
_TOPK = 4


def _compute_block_selection(
    q: jax.Array,
    k: jax.Array,
    scale: float,
    topk: int,
    chunk_size: int = _CHUNK_SIZE,
) -> jax.Array:
  """Hardware-aligned Option 1 Query-Block routing."""
  batch_size, num_heads, seq_len, head_dim = q.shape
  block_q = chunk_size
  num_q_blocks = seq_len // block_q
  effective_topk = min(topk, num_q_blocks)

  q_blocks = q.reshape(batch_size, num_heads, num_q_blocks, block_q, head_dim)
  k_blocks = k.reshape(batch_size, num_heads, num_q_blocks, block_q, head_dim)

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
  return jax.lax.stop_gradient(topk_idx).astype(jnp.int32)


def _flash_moba_kernel(
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


def _flash_moba_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    topk_idx: jax.Array,
    scale: float,
    chunk_size: int = _CHUNK_SIZE,
    interpret: bool | None = None,
) -> jax.Array:
  """Executes MoBA attention via Pallas TPU kernel with SMEM block routing."""
  batch_size, num_heads, seq_len, head_dim = q.shape
  block_q = chunk_size
  num_blocks = seq_len // block_q
  effective_topk = topk_idx.shape[-1]

  q_blocks = q.reshape(batch_size, num_heads, num_blocks, block_q, head_dim)
  k_blocks = k.reshape(batch_size, num_heads, num_blocks, block_q, head_dim)
  v_blocks = v.reshape(batch_size, num_heads, num_blocks, block_q, head_dim)
  topk_idx = topk_idx.astype(jnp.int32)

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
    interpret = jax.default_backend() != "tpu"
  out_blocks = pl.pallas_call(
      functools.partial(_flash_moba_kernel, scale=scale),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(batch_size, num_heads, num_blocks, effective_topk + 1),
          in_specs=[q_spec, kv_spec, kv_spec],
          out_specs=q_spec,
          scratch_shapes=[scratch_shape, scratch_shape, scratch_shape],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
      ),
      out_shape=jax.ShapeDtypeStruct(
          (batch_size, num_heads, num_blocks, block_q, head_dim), dtype=q.dtype
      ),
      interpret=interpret,
  )(topk_idx, q_blocks, k_blocks, v_blocks)
  return out_blocks.reshape(batch_size, num_heads, seq_len, head_dim)


def moba_pallas_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    topk: int = _TOPK,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
    interpret: bool | None = None,
) -> tuple[jax.Array, None]:
  """Executes the Mixture of Block Attention forward pass."""
  if scale is None:
    head_dim = q.shape[-1]
    scale = head_dim ** -0.5

  topk_idx = _compute_block_selection(q, k, scale, topk, chunk_size)
  out = _flash_moba_pallas(
      q, k, v, topk_idx, scale, chunk_size, interpret=interpret
  )
  return out, None
