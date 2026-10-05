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

from typing import Any
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

_CHUNK_SIZE = 256
_TOPK = 4


def _compute_block_selection(
    q: jax.Array, k: jax.Array, scale: float, topk: int, chunk_size: int = _CHUNK_SIZE
) -> jax.Array:
  """Hardware-aligned Option 1 Query-Block routing."""
  batch_size, num_heads, seq_len, head_dim = q.shape
  block_chunk = chunk_size
  num_blocks = seq_len // block_chunk

  q_blocks = q.reshape(batch_size, num_heads, num_blocks, block_chunk, head_dim)
  k_blocks = k.reshape(batch_size, num_heads, num_blocks, block_chunk, head_dim)
  q_mean = jnp.mean(q_blocks, axis=3)
  k_mean = jnp.mean(k_blocks, axis=3)

  routing_scores = jnp.einsum("bhqd,bhnd->bhqn", q_mean, k_mean) * scale
  q_idx = jnp.arange(num_blocks)[:, None]
  k_idx = jnp.arange(num_blocks)[None, :]
  historical_mask = k_idx < q_idx

  masked_routing = jnp.where(
      historical_mask[None, None], routing_scores, -jnp.inf
  )
  effective_topk = min(topk, num_blocks)
  _, topk_idx = jax.lax.top_k(masked_routing, effective_topk)
  return topk_idx


def _make_flash_moba_kernel(block_q: int, effective_topk: int, scale: float) -> Any:
  del effective_topk
  def _kernel(
      q_ref: Any,
      k_sub_ref: Any,
      v_sub_ref: Any,
      mask_ref: Any,
      o_ref: Any,
  ) -> None:
    q_i = q_ref[0, 0, 0]  # (block_q, head_dim)
    k_sub = k_sub_ref[0, 0, 0]  # (block_k_sub, head_dim)
    v_sub = v_sub_ref[0, 0, 0]  # (block_k_sub, head_dim)
    mask_sub = mask_ref[0, 0, 0]  # (block_q, block_k_sub)

    qk = jnp.matmul(q_i, k_sub.T, preferred_element_type=jnp.float32) * scale
    qk = jnp.where(mask_sub, qk, -1e9)

    max_scores = jnp.max(qk, axis=-1, keepdims=True)
    exp_scores = jnp.exp(qk - max_scores)
    denominator = jnp.sum(exp_scores, axis=-1, keepdims=True)
    weights = exp_scores / denominator

    o_i = jnp.matmul(weights, v_sub, preferred_element_type=jnp.float32)
    o_ref[0, 0, 0] = o_i.astype(o_ref.dtype)

  return _kernel


def _flash_moba_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    topk_idx: jax.Array,
    scale: float,
    chunk_size: int = _CHUNK_SIZE,
) -> jax.Array:
  batch_size, num_heads, seq_len, head_dim = q.shape
  block_q = chunk_size
  num_q_blocks = seq_len // block_q
  effective_topk = topk_idx.shape[-1]
  block_k_sub = (effective_topk + 1) * block_q

  q_blocks = q.reshape(batch_size, num_heads, num_q_blocks, block_q, head_dim)
  k_blocks = k.reshape(batch_size, num_heads, num_q_blocks, block_q, head_dim)
  v_blocks = v.reshape(batch_size, num_heads, num_q_blocks, block_q, head_dim)

  def gather_sub(
      k_bh: jax.Array, v_bh: jax.Array, topk_bh: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    def for_i(block_idx: int | Any) -> tuple[jax.Array, jax.Array]:
      k_diag = k_bh[block_idx][None, :, :]
      v_diag = v_bh[block_idx][None, :, :]
      k_hist = k_bh[topk_bh[block_idx]]
      v_hist = v_bh[topk_bh[block_idx]]
      k_s = jnp.concatenate([k_diag, k_hist], axis=0).reshape(block_k_sub, head_dim)
      v_s = jnp.concatenate([v_diag, v_hist], axis=0).reshape(block_k_sub, head_dim)
      return k_s, v_s

    return jax.vmap(for_i)(jnp.arange(num_q_blocks))

  k_sub, v_sub = jax.vmap(jax.vmap(gather_sub))(k_blocks, v_blocks, topk_idx)

  causal_diag = jnp.tril(jnp.ones((block_q, block_q), dtype=bool))

  def make_mask_i(block_idx: int | Any) -> jax.Array:
    ranks = jnp.arange(effective_topk)
    valid_hist = jnp.broadcast_to(
        jnp.repeat(ranks < block_idx, block_q)[None, :],
        (block_q, effective_topk * block_q),
    )
    return jnp.concatenate([causal_diag, valid_hist], axis=1)

  mask_sub = jax.vmap(make_mask_i)(jnp.arange(num_q_blocks))
  mask_sub_b = jnp.broadcast_to(
      mask_sub[None, None],
      (batch_size, num_heads, num_q_blocks, block_q, block_k_sub),
  )

  kernel = _make_flash_moba_kernel(block_q, effective_topk, scale)

  grid = (batch_size, num_heads, num_q_blocks)
  q_spec = pl.BlockSpec(
      (1, 1, 1, block_q, head_dim), lambda i, j, n: (i, j, n, 0, 0)
  )
  k_sub_spec = pl.BlockSpec(
      (1, 1, 1, block_k_sub, head_dim), lambda i, j, n: (i, j, n, 0, 0)
  )
  v_sub_spec = pl.BlockSpec(
      (1, 1, 1, block_k_sub, head_dim), lambda i, j, n: (i, j, n, 0, 0)
  )
  mask_spec = pl.BlockSpec(
      (1, 1, 1, block_q, block_k_sub), lambda i, j, n: (i, j, n, 0, 0)
  )
  o_spec = pl.BlockSpec(
      (1, 1, 1, block_q, head_dim), lambda i, j, n: (i, j, n, 0, 0)
  )

  interpret = jax.default_backend() == "cpu"
  out = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[q_spec, k_sub_spec, v_sub_spec, mask_spec],
      out_specs=o_spec,
      out_shape=jax.ShapeDtypeStruct(
          (batch_size, num_heads, num_q_blocks, block_q, head_dim), q.dtype
      ),
      interpret=interpret,
  )(q_blocks, k_sub, v_sub, mask_sub_b)
  return out.reshape(batch_size, num_heads, seq_len, head_dim)


def moba_pallas_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    topk: int = _TOPK,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
) -> tuple[jax.Array, None]:
  """Computes MoBA forward pass using Pallas TPU kernel."""
  if scale is None:
    head_dim = q.shape[-1]
    scale = head_dim ** -0.5
  topk_idx = jax.lax.stop_gradient(
      _compute_block_selection(q, k, scale, topk, chunk_size)
  )
  out = _flash_moba_pallas(q, k, v, topk_idx, scale, chunk_size)
  return out, None
