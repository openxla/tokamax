# Copyright 2026 Google LLC
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
"""Base class for Multi-Head Latent Sliding Window Attention (MLA-SWA)."""

from typing import Any, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, UInt8  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.dsv4_compressor import csa_cache_layout
from tokamax._src.ops.experimental.mla_swa import reference


class MlaSwa[C](
    op.Op[
        Any,
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        None,
        C,
        Any,
    ]
):
  """Tokamax operator for Multi-Head Latent Sliding Window Attention."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      q: Float[Array, "T H D"],
      new_kv: Float[Array, "T D"],
      cache_kv: UInt8[Array, "P S 4 128"] | Int[Array, "P S 128"],
      kv_lens: Int[Array, "M"],
      page_indices: Int[Array, "N"],
      cu_q_lens: Int[Array, "B"],
      distribution: Int[Array, "3"],
      attention_sinks: Float[Array, "H"],
      *,
      sliding_window: int,
      logical_page_size: int,
      sm_scale: float = 1.0,
      chunk_prefill_size: int | None = None,
      unnormalized_output: bool = False,
      non_causal_block: bool = False,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    if new_kv.dtype != jnp.bfloat16:
      raise ValueError(f"new_kv must be bfloat16, got {new_kv.dtype}.")
    if cache_kv.dtype not in (jnp.uint8, jnp.int32):
      raise ValueError(
          f"cache_kv must be uint8 or int32, got {cache_kv.dtype}."
      )
    if attention_sinks.dtype != jnp.float32:
      raise ValueError(
          f"attention_sinks must be float32, got {attention_sinks.dtype}."
      )
    if sliding_window <= 0:
      raise ValueError(
          f"sliding_window must be positive, got {sliding_window}."
      )
    head_dim = q.shape[-1]
    if head_dim % 128 != 0:
      raise ValueError(f"head_dim must be a multiple of 128, got {head_dim}.")

    _, physical_page_size_per_kv_packing, kv_packing, lkv_dim = (
        csa_cache_layout.u8_slab_shape(cache_kv.shape, cache_kv.dtype)
    )
    slot_bytes = kv_packing * lkv_dim
    token_bytes = head_dim * 2
    if token_bytes % slot_bytes != 0:
      raise ValueError(
          f"token_bytes ({token_bytes}) must be divisible by slot_bytes"
          f" ({slot_bytes})."
      )
    slots_per_token = token_bytes // slot_bytes
    phys_tokens_per_page = physical_page_size_per_kv_packing // slots_per_token
    if logical_page_size <= 0 or logical_page_size > phys_tokens_per_page:
      raise ValueError(
          f"logical_page_size ({logical_page_size}) must be in"
          f" (0, {phys_tokens_per_page}]."
      )

    max_num_seqs = cu_q_lens.shape[0] - 1
    if max_num_seqs <= 0 or kv_lens.shape[0] != max_num_seqs:
      raise ValueError(
          f"Expected kv_lens length ({kv_lens.shape[0]}) to equal"
          f" cu_q_lens.shape[0] - 1 ({max_num_seqs})."
      )
    num_page_indices = page_indices.shape[0]
    if num_page_indices % max_num_seqs != 0:
      raise ValueError(
          f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
      )
    return super().bind(
        q=q,
        new_kv=new_kv,
        cache_kv=cache_kv,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        attention_sinks=attention_sinks,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        sm_scale=sm_scale,
        chunk_prefill_size=chunk_prefill_size,
        unnormalized_output=unnormalized_output,
        non_causal_block=non_causal_block,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      q: Float[Array, "T H D"],
      new_kv: Float[Array, "T D"],
      cache_kv: UInt8[Array, "P S 4 128"] | Int[Array, "P S 128"],
      kv_lens: Int[Array, "M"],
      page_indices: Int[Array, "N"],
      cu_q_lens: Int[Array, "B"],
      distribution: Int[Array, "3"],
      attention_sinks: Float[Array, "H"],
      *,
      sliding_window: int,
      logical_page_size: int,
      sm_scale: float = 1.0,
      chunk_prefill_size: int | None = None,
      unnormalized_output: bool = False,
      non_causal_block: bool = False,
      return_residuals: bool = False,
      config: C,
  ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
    del return_residuals, config
    return (
        reference.mla_sliding_window_ragged_paged_attention(
            q,
            new_kv,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            attention_sinks,
            sliding_window=sliding_window,
            logical_page_size=logical_page_size,
            sm_scale=sm_scale,
            chunk_prefill_size=chunk_prefill_size,
            unnormalized_output=unnormalized_output,
            non_causal_block=non_causal_block,
        ),
        None,
    )
