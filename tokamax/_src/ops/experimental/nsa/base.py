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

"""Base operator class and reference for Native Sparse Attention (NSA)."""

import dataclasses
from typing import Any, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
from tokamax._src.ops import op
from typing_extensions import override

_Config = TypeVar("_Config")
Output: TypeAlias = jax.Array
Residuals: TypeAlias = Any


def _compute_cmp_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    bs_cmp: int = 64,
) -> jax.Array:
  """Coarse block-compression attention branch."""
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


def _compute_slc_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    topk: int = 4,
    chunk_size: int = 256,
) -> jax.Array:
  """Fine-grained top-k block selection attention branch."""
  batch_size, num_heads, seq_len, head_dim = q.shape
  block_q = chunk_size
  num_q_blocks = seq_len // block_q
  effective_topk = min(topk, num_q_blocks)
  block_k_sub = (effective_topk + 1) * block_q

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
  topk_idx = jax.lax.stop_gradient(topk_idx)

  def gather_head(x_h: jax.Array, idx_h: jax.Array) -> jax.Array:
    diag = x_h[:, None, :, :]
    hist = x_h[idx_h]
    return jnp.concatenate([diag, hist], axis=1).reshape(
        num_q_blocks, block_k_sub, head_dim
    )

  gather_fn = jax.vmap(jax.vmap(gather_head))
  k_sub = gather_fn(k_blocks, topk_idx)
  v_sub = gather_fn(v_blocks, topk_idx)

  causal_diag = jnp.tril(jnp.ones((block_q, block_q), dtype=bool))
  ranks = jnp.arange(effective_topk)

  def make_mask_i(block_idx: int | Any) -> jax.Array:
    valid_hist = jnp.broadcast_to(
        jnp.repeat(ranks < block_idx, block_q)[None, :],
        (block_q, effective_topk * block_q),
    )
    return jnp.concatenate([causal_diag, valid_hist], axis=1)

  mask_sub = jax.vmap(make_mask_i)(jnp.arange(num_q_blocks))

  logits = (
      jnp.einsum(
          "bhqtd,bhqkd->bhqtk",
          q_blocks,
          k_sub,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * scale
  )
  logits = jnp.where(mask_sub[None, None], logits, -1e9)
  weights = jax.nn.softmax(logits, axis=-1)
  out_blocks = jnp.einsum(
      "bhqtk,bhqkd->bhqtd",
      weights,
      v_sub,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return out_blocks.reshape(batch_size, num_heads, seq_len, head_dim)


def _compute_swa_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    window: int = 128,
) -> jax.Array:
  """Local sliding-window attention branch."""
  seq_len = q.shape[2]
  logits = (
      jnp.einsum(
          "bhtd,bhsd->bhts",
          q,
          k,
          preferred_element_type=jnp.float32,
          precision=jax.lax.Precision.HIGHEST,
      )
      * scale
  )

  q_pos = jnp.arange(seq_len)[:, None]
  k_pos = jnp.arange(seq_len)[None, :]
  win_mask = (k_pos <= q_pos) & (k_pos >= q_pos - window)

  logits = jnp.where(win_mask[None, None], logits, -1e9)
  weights = jax.nn.softmax(logits, axis=-1)
  return jnp.einsum(
      "bhts,bhsd->bhtd",
      weights,
      v,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )


def nsa_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_cmp: jax.Array | None = None,
    g_slc: jax.Array | None = None,
    g_swa: jax.Array | None = None,
    *,
    chunk_size: int = 256,
    topk: int = 4,
    window: int = 128,
    cmp_block_size: int = 64,
    scale: float | None = None,
) -> tuple[jax.Array, None]:
  """Reference implementation of Native Sparse Attention (NSA)."""
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

  o_cmp = _compute_cmp_reference(
      q_f32, k_f32, v_f32, scale, bs_cmp=cmp_block_size
  )
  o_slc = _compute_slc_reference(
      q_f32, k_f32, v_f32, scale, topk=topk, chunk_size=chunk_size
  )
  o_swa = _compute_swa_reference(q_f32, k_f32, v_f32, scale, window=window)

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


@dataclasses.dataclass(frozen=True, kw_only=True)
class NativeSparseAttention[C](op.Op[Any, Output, Residuals, C, None]):
  """Tokamax base operator for Native Sparse Attention (NSA).

  Fuses coarse compression, fine-grained top-k block selection, and sliding
  window attention branches via a learned 3-way gate.
  """

  supports_symbolic_shapes = False

  @override
  def _fwd(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g_cmp: jax.Array | None = None,
      g_slc: jax.Array | None = None,
      g_swa: jax.Array | None = None,
      *,
      chunk_size: int = 256,
      topk: int = 4,
      window: int = 128,
      cmp_block_size: int = 64,
      scale: float | None = None,
      config: C | None = None,
      return_residuals: bool = False,
  ) -> tuple[Output, None]:
    del config, return_residuals
    out, _ = nsa_reference(
        q,
        k,
        v,
        g_cmp=g_cmp,
        g_slc=g_slc,
        g_swa=g_swa,
        chunk_size=chunk_size,
        topk=topk,
        window=window,
        cmp_block_size=cmp_block_size,
        scale=scale,
    )
    return out, None
