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

"""Pure JAX reference implementation of Mixture of Block Attention (MoBA)."""

from typing import Any
import jax
import jax.numpy as jnp

_CHUNK_SIZE = 256
_TOPK = 4


def _single_seq_moba(
    q_s: jax.Array,
    k_s: jax.Array,
    v_s: jax.Array,
    scale: float,
    topk: int,
    chunk_size: int = _CHUNK_SIZE,
) -> jax.Array:
  orig_dtype = q_s.dtype
  q_s = q_s.astype(jnp.float32)
  k_s = k_s.astype(jnp.float32)
  v_s = v_s.astype(jnp.float32)

  h, t, d = q_s.shape
  bq = chunk_size
  nq_blocks = t // bq
  k_eff = min(topk, nq_blocks)
  bk_sub = (k_eff + 1) * bq

  q_blocks = q_s.reshape(h, nq_blocks, bq, d)
  k_blocks = k_s.reshape(h, nq_blocks, bq, d)
  v_blocks = v_s.reshape(h, nq_blocks, bq, d)

  q_mean = jnp.mean(q_blocks, axis=2)
  k_mean = jnp.mean(k_blocks, axis=2)

  routing_scores = jnp.einsum("hqd,hnd->hqn", q_mean, k_mean) * scale
  q_idx = jnp.arange(nq_blocks)[:, None]
  k_idx = jnp.arange(nq_blocks)[None, :]
  historical_mask = k_idx < q_idx
  masked_routing = jnp.where(historical_mask[None], routing_scores, -jnp.inf)
  _, topk_idx = jax.lax.top_k(masked_routing, k_eff)
  topk_idx = jax.lax.stop_gradient(topk_idx)

  causal_diag = jnp.tril(jnp.ones((bq, bq), dtype=bool))
  ranks = jnp.arange(k_eff)

  def head_fn(
      q_h: jax.Array, k_h: jax.Array, v_h: jax.Array, topk_h: jax.Array
  ) -> jax.Array:
    def block_fn(i: int | Any) -> jax.Array:
      q_i = q_h[i]
      k_diag = k_h[i][None, :, :]
      v_diag = v_h[i][None, :, :]
      k_hist = k_h[topk_h[i]]
      v_hist = v_h[topk_h[i]]
      k_sub = jnp.concatenate([k_diag, k_hist], axis=0).reshape(bk_sub, d)
      v_sub = jnp.concatenate([v_diag, v_hist], axis=0).reshape(bk_sub, d)

      valid_hist = jnp.broadcast_to(
          jnp.repeat(ranks < i, bq)[None, :], (bq, k_eff * bq)
      )
      mask_sub = jnp.concatenate([causal_diag, valid_hist], axis=1)

      logits = jnp.matmul(q_i, k_sub.T) * scale
      logits = jnp.where(mask_sub, logits, -1e9)
      w = jax.nn.softmax(logits, axis=-1)
      return jnp.matmul(w, v_sub)

    return jax.vmap(block_fn)(jnp.arange(nq_blocks)).reshape(t, d)

  out = jax.vmap(head_fn)(q_blocks, k_blocks, v_blocks, topk_idx)
  return out.astype(orig_dtype)


def moba_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    topk: int = _TOPK,
    chunk_size: int = _CHUNK_SIZE,
    scale: float | None = None,
) -> tuple[jax.Array, None]:
  """Reference implementation of Mixture of Block Attention (MoBA)."""
  if scale is None:
    scale = q.shape[-1] ** -0.5

  def batch_fn(q_b: jax.Array, k_b: jax.Array, v_b: jax.Array) -> jax.Array:
    return _single_seq_moba(q_b, k_b, v_b, scale, topk, chunk_size)

  out = jax.vmap(batch_fn)(q, k, v)
  return out, None
