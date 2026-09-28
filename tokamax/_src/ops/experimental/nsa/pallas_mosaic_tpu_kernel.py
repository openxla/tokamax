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

from typing import Any
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

_CHUNK_SIZE = 256
_TOPK = 4
_WINDOW = 128
_CMP_BLOCK_SIZE = 64


def _compute_block_selection(
    q: jax.Array, k: jax.Array, scale: float, topk: int, chunk_size: int = _CHUNK_SIZE
) -> jax.Array:
  """Hardware-aligned Option 1 Query-Block routing."""
  b, h, t, d = q.shape
  bc = chunk_size
  nb = t // bc

  q_blocks = q.reshape(b, h, nb, bc, d)
  k_blocks = k.reshape(b, h, nb, bc, d)
  q_mean = jnp.mean(q_blocks, axis=3)
  k_mean = jnp.mean(k_blocks, axis=3)

  routing_scores = jnp.einsum("bhqd,bhnd->bhqn", q_mean, k_mean) * scale
  q_idx = jnp.arange(nb)[:, None]
  k_idx = jnp.arange(nb)[None, :]
  historical_mask = k_idx < q_idx

  masked_routing = jnp.where(
      historical_mask[None, None], routing_scores, -jnp.inf
  )
  k_eff = min(topk, nb)
  _, topk_idx = jax.lax.top_k(masked_routing, k_eff)
  return topk_idx


def _compute_cmp_branch(
    q: jax.Array, k: jax.Array, v: jax.Array, scale: float, bs_cmp: int = _CMP_BLOCK_SIZE
) -> jax.Array:
  """Coarse Compression branch over mean-pooled key and value tokens."""
  b, h, t, d = q.shape
  nc = t // bs_cmp
  k_blocks = k.reshape(b, h, nc, bs_cmp, d)
  v_blocks = v.reshape(b, h, nc, bs_cmp, d)
  k_cmp = jnp.mean(k_blocks, axis=3)  # (B, H, NC, D)
  v_cmp = jnp.mean(v_blocks, axis=3)  # (B, H, NC, D)

  logits = jnp.einsum("bhtd,bhcd->bhtc", q, k_cmp) * scale
  q_pos = jnp.arange(t)[:, None]
  c_pos = (jnp.arange(nc) * bs_cmp)[None, :]
  causal_mask = c_pos < q_pos
  logits = jnp.where(causal_mask[None, None], logits, -1e9)
  weights = jax.nn.softmax(logits, axis=-1)
  return jnp.einsum("bhtc,bhcd->bhtd", weights, v_cmp)


def _make_flash_slc_kernel(bq: int, k_eff: int, scale: float) -> Any:
  def _kernel(
      q_ref: Any,
      k_sub_ref: Any,
      v_sub_ref: Any,
      mask_ref: Any,
      o_ref: Any,
  ) -> None:
    q_i = q_ref[0, 0, 0]  # (BQ, D)
    k_sub = k_sub_ref[0, 0, 0]  # (BK_SUB, D)
    v_sub = v_sub_ref[0, 0, 0]  # (BK_SUB, D)
    mask_sub = mask_ref[0, 0, 0]  # (BQ, BK_SUB)

    qk = jnp.matmul(q_i, k_sub.T, preferred_element_type=jnp.float32) * scale
    qk = jnp.where(mask_sub, qk, -1e9)

    m = jnp.max(qk, axis=-1, keepdims=True)
    p = jnp.exp(qk - m)
    den = jnp.sum(p, axis=-1, keepdims=True)
    w = p / den

    o_i = jnp.matmul(w, v_sub, preferred_element_type=jnp.float32)
    o_ref[0, 0, 0] = o_i.astype(o_ref.dtype)

  return _kernel


def _compute_slc_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    topk_idx: jax.Array,
    scale: float,
    chunk_size: int = _CHUNK_SIZE,
) -> jax.Array:
  """Fine-Grained Selection branch evaluated via Pallas on TPU VMEM."""
  b, h, t, d = q.shape
  bq = chunk_size
  nq_blocks = t // bq
  k_eff = topk_idx.shape[-1]
  bk_sub = (k_eff + 1) * bq

  q_blocks = q.reshape(b, h, nq_blocks, bq, d)
  k_blocks = k.reshape(b, h, nq_blocks, bq, d)
  v_blocks = v.reshape(b, h, nq_blocks, bq, d)

  def gather_sub(
      k_bh: jax.Array, v_bh: jax.Array, topk_bh: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    def for_i(i: int | Any) -> tuple[jax.Array, jax.Array]:
      k_diag = k_bh[i][None, :, :]
      v_diag = v_bh[i][None, :, :]
      k_hist = k_bh[topk_bh[i]]
      v_hist = v_bh[topk_bh[i]]
      k_s = jnp.concatenate([k_diag, k_hist], axis=0).reshape(bk_sub, d)
      v_s = jnp.concatenate([v_diag, v_hist], axis=0).reshape(bk_sub, d)
      return k_s, v_s

    return jax.vmap(for_i)(jnp.arange(nq_blocks))

  k_sub, v_sub = jax.vmap(jax.vmap(gather_sub))(k_blocks, v_blocks, topk_idx)

  causal_diag = jnp.tril(jnp.ones((bq, bq), dtype=bool))

  def make_mask_i(i: int | Any) -> jax.Array:
    ranks = jnp.arange(k_eff)
    valid_hist = jnp.broadcast_to(
        jnp.repeat(ranks < i, bq)[None, :], (bq, k_eff * bq)
    )
    return jnp.concatenate([causal_diag, valid_hist], axis=1)

  mask_sub = jax.vmap(make_mask_i)(jnp.arange(nq_blocks))
  mask_sub_b = jnp.broadcast_to(
      mask_sub[None, None], (b, h, nq_blocks, bq, bk_sub)
  )

  kernel = _make_flash_slc_kernel(bq, k_eff, scale)

  grid = (b, h, nq_blocks)
  q_spec = pl.BlockSpec((1, 1, 1, bq, d), lambda i, j, n: (i, j, n, 0, 0))
  k_sub_spec = pl.BlockSpec((1, 1, 1, bk_sub, d), lambda i, j, n: (i, j, n, 0, 0))
  v_sub_spec = pl.BlockSpec((1, 1, 1, bk_sub, d), lambda i, j, n: (i, j, n, 0, 0))
  mask_spec = pl.BlockSpec((1, 1, 1, bq, bk_sub), lambda i, j, n: (i, j, n, 0, 0))
  o_spec = pl.BlockSpec((1, 1, 1, bq, d), lambda i, j, n: (i, j, n, 0, 0))

  interpret = jax.default_backend() == "cpu"
  out = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[q_spec, k_sub_spec, v_sub_spec, mask_spec],
      out_specs=o_spec,
      out_shape=jax.ShapeDtypeStruct((b, h, nq_blocks, bq, d), q.dtype),
      interpret=interpret,
  )(q_blocks, k_sub, v_sub, mask_sub_b)
  return out.reshape(b, h, t, d)


def _compute_swa_branch(
    q: jax.Array, k: jax.Array, v: jax.Array, scale: float, window: int = _WINDOW
) -> jax.Array:
  """Sliding Window Attention branch for local high-resolution tokens."""
  b, h, t, d = q.shape
  scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale
  t_idx = jnp.arange(t)[:, None]
  s_idx = jnp.arange(t)[None, :]
  mask = (s_idx <= t_idx) & (s_idx >= t_idx - window)
  scores = jnp.where(mask[None, None], scores, -1e9)
  weights = jax.nn.softmax(scores, axis=-1)
  return jnp.einsum("bhts,bhsd->bhtd", weights, v)


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
) -> tuple[jax.Array, None]:
  """Computes NSA forward pass using Pallas TPU kernel for selection branch."""
  if scale is None:
    scale = q.shape[-1] ** -0.5
  if g_cmp is None:
    g_cmp = jnp.full_like(q, 1.0 / 3.0)
  if g_slc is None:
    g_slc = jnp.full_like(q, 1.0 / 3.0)
  if g_swa is None:
    g_swa = jnp.full_like(q, 1.0 / 3.0)

  topk_idx = jax.lax.stop_gradient(
      _compute_block_selection(q, k, scale, topk, chunk_size)
  )
  o_cmp = _compute_cmp_branch(q, k, v, scale, cmp_block_size)
  o_slc = _compute_slc_pallas(q, k, v, topk_idx, scale, chunk_size)
  o_swa = _compute_swa_branch(q, k, v, scale, window)
  out = g_cmp * o_cmp + g_slc * o_slc + g_swa * o_swa
  return out, None
