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
#   nk  selection block extent per query block (nk = tk + 1)

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


def _compute_cmp_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    cmp_block_size: int = 64,
) -> jax.Array:
  """Coarse block-compression attention branch."""
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


def _compute_slc_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    topk: int = 4,
    chunk_size: int = 256,
) -> jax.Array:
  """Fine-grained top-k block selection attention branch."""
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
  topk_idx = jax.lax.stop_gradient(topk_idx)

  def gather_head(x_h: jax.Array, idx_h: jax.Array) -> jax.Array:
    diag = x_h[:, None, :, :]
    hist = x_h[idx_h]
    return jnp.concatenate([diag, hist], axis=1).reshape(nt, nk * tt, D)

  gather_fn = jax.vmap(jax.vmap(gather_head))
  k_sub = gather_fn(k_blocks, topk_idx)
  v_sub = gather_fn(v_blocks, topk_idx)

  causal_diag = jnp.tril(jnp.ones((tt, tt), dtype=bool))
  ranks = jnp.arange(tk)

  def make_mask_i(q_i: int | Any) -> jax.Array:
    valid_hist = jnp.broadcast_to(
        jnp.repeat(ranks < q_i, tt)[None, :],
        (tt, tk * tt),
    )
    return jnp.concatenate([causal_diag, valid_hist], axis=1)

  mask_sub = jax.vmap(make_mask_i)(jnp.arange(nt))

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
  return out_blocks.reshape(B, H, T, D)


def _compute_swa_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    window: int = 128,
) -> jax.Array:
  """Local sliding-window attention branch."""
  _, _, T, _ = q.shape
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

  q_pos = jnp.arange(T)[:, None]
  k_pos = jnp.arange(T)[None, :]
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
  if chunk_size <= 0:
    raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
  if topk <= 0:
    raise ValueError(f"topk must be positive, got {topk}.")
  if window <= 0:
    raise ValueError(f"window must be positive, got {window}.")
  if cmp_block_size <= 0:
    raise ValueError(f"cmp_block_size must be positive, got {cmp_block_size}.")

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

  o_cmp = _compute_cmp_reference(
      q_f32, k_f32, v_f32, scale, cmp_block_size=cmp_block_size
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

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array, "B H T D"],
      k: Float[Array, "B H T D"],
      v: Float[Array, "B H T D"],
      g_cmp: Float[Array, "B H T D"] | None = None,
      g_slc: Float[Array, "B H T D"] | None = None,
      g_swa: Float[Array, "B H T D"] | None = None,
      *,
      chunk_size: int = 256,
      topk: int = 4,
      window: int = 128,
      cmp_block_size: int = 64,
      scale: float | None = None,
      config: C | None = None,
      return_residuals: bool = False,
  ) -> tuple[Float[Array, "B H T D"], None]:
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
