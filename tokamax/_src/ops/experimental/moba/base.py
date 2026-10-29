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

"""Base operator class for Mixture of Block Attention (MoBA)."""

# Dimensions.
#   B      batch
#   H      number of attention heads
#   T      sequence length
#   D      head dimension
#   tt     query/key block chunk size (chunk_size)
#   nt     grid extent over T: nt = T // tt
#   K      effective number of routed historical blocks per query block

import dataclasses
from typing import Any, TypeAlias, TypeVar
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from typing_extensions import override

_Config = TypeVar("_Config")
Output: TypeAlias = jax.Array
Residuals: TypeAlias = Any


def _single_seq_moba(
    q_s: jax.Array,
    k_s: jax.Array,
    v_s: jax.Array,
    scale: float,
    topk: int,
    chunk_size: int = 256,
) -> jax.Array:
  """Computes reference MoBA attention for a single sequence."""
  orig_dtype = q_s.dtype
  q_s = q_s.astype(jnp.float32)
  k_s = k_s.astype(jnp.float32)
  v_s = v_s.astype(jnp.float32)

  H, T, D = q_s.shape
  tt = chunk_size
  nt = T // tt
  K = min(topk, nt)
  tk_sub = (K + 1) * tt

  q_blocks = q_s.reshape(H, nt, tt, D)
  k_blocks = k_s.reshape(H, nt, tt, D)
  v_blocks = v_s.reshape(H, nt, tt, D)

  q_mean = jnp.mean(q_blocks, axis=2)
  k_mean = jnp.mean(k_blocks, axis=2)

  routing_scores = (
      jnp.einsum(
          "hqd,hnd->hqn",
          q_mean,
          k_mean,
          precision=jax.lax.Precision.HIGHEST,
      )
      * scale
  )
  q_idx = jnp.arange(nt)[:, None]
  k_idx = jnp.arange(nt)[None, :]
  historical_mask = k_idx < q_idx
  masked_routing = jnp.where(historical_mask[None], routing_scores, -jnp.inf)
  _, topk_idx = jax.lax.top_k(masked_routing, K)
  topk_idx = jax.lax.stop_gradient(topk_idx)

  causal_diag = jnp.tril(jnp.ones((tt, tt), dtype=bool))
  ranks = jnp.arange(K)

  def head_fn(
      q_h: jax.Array, k_h: jax.Array, v_h: jax.Array, topk_h: jax.Array
  ) -> jax.Array:
    def block_fn(block_idx: int | Any) -> jax.Array:
      q_i = q_h[block_idx]
      k_diag = k_h[block_idx][None, :, :]
      v_diag = v_h[block_idx][None, :, :]
      k_hist = k_h[topk_h[block_idx]]
      v_hist = v_h[topk_h[block_idx]]
      k_sub = jnp.concatenate([k_diag, k_hist], axis=0).reshape(tk_sub, D)
      v_sub = jnp.concatenate([v_diag, v_hist], axis=0).reshape(tk_sub, D)

      valid_hist = jnp.broadcast_to(
          jnp.repeat(ranks < block_idx, tt)[None, :],
          (tt, K * tt),
      )
      mask_sub = jnp.concatenate([causal_diag, valid_hist], axis=1)

      logits = (
          jnp.matmul(q_i, k_sub.T, precision=jax.lax.Precision.HIGHEST) * scale
      )
      logits = jnp.where(mask_sub, logits, -1e9)
      weights = jax.nn.softmax(logits, axis=-1)
      return jnp.matmul(weights, v_sub, precision=jax.lax.Precision.HIGHEST)

    return jax.vmap(block_fn)(jnp.arange(nt)).reshape(T, D)

  out = jax.vmap(head_fn)(q_blocks, k_blocks, v_blocks, topk_idx)
  return out.astype(orig_dtype)


def moba_reference(
    q: Float[Array, "B H T D"],
    k: Float[Array, "B H T D"],
    v: Float[Array, "B H T D"],
    *,
    topk: int = 4,
    chunk_size: int = 256,
    scale: float | None = None,
) -> tuple[Float[Array, "B H T D"], None]:
  """Reference implementation of Mixture of Block Attention (MoBA)."""
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
  if scale is None:
    scale = D**-0.5

  def batch_fn(q_b: jax.Array, k_b: jax.Array, v_b: jax.Array) -> jax.Array:
    return _single_seq_moba(q_b, k_b, v_b, scale, topk, chunk_size)

  out = jax.vmap(batch_fn)(q, k, v)
  return out, None


@dataclasses.dataclass(frozen=True, kw_only=True)
class MixtureOfBlockAttention[C](op.Op[Any, Output, Residuals, C, None]):
  """Tokamax base operator for Mixture of Block Attention (MoBA).

  Routes each query block to the top-k most relevant historical key/value
  blocks via block-mean routing logits.
  """

  supports_symbolic_shapes = False

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array, "B H T D"],
      k: Float[Array, "B H T D"],
      v: Float[Array, "B H T D"],
      *,
      topk: int = 4,
      chunk_size: int = 256,
      scale: float | None = None,
      config: C | None = None,
      return_residuals: bool = False,
  ) -> tuple[Float[Array, "B H T D"], None]:
    del config, return_residuals
    out, _ = moba_reference(
        q,
        k,
        v,
        topk=topk,
        chunk_size=chunk_size,
        scale=scale,
    )
    return out, None
