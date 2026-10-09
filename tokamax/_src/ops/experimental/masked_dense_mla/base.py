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
"""Base class for Masked-Dense Multi-Head Latent Attention (MaskedDenseMLA)."""

from typing import Any, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, UInt8  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.masked_dense_mla import reference


class MaskedDenseMla[C](op.Op[Any, jax.Array, None, C, Any]):
  """Tokamax operator for Masked-Dense Multi-Head Latent Attention."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      q: Float[Array, "T H D"],
      cache_kv_nope: UInt8[Array, "P S 4 128"],
      cache_kv_rope: UInt8[Array, "P R 4 128"],
      kv_lens: Int[Array, "M"],
      topk_indices: Int[Array, "T K"],
      page_indices: Int[Array, "N"],
      cu_q_lens: Int[Array, "B"],
      distribution: Int[Array, "3"],
      *,
      sm_scale: float = 1.0,
      k_scale: float = 1.0,
      mask_value: float | None = None,
      max_kv_len: int | None = None,
      chunk_prefill_size: int | None = None,
      sequence_start: Int[Array, ""] | None = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    if cache_kv_nope.dtype != jnp.uint8 or cache_kv_rope.dtype != jnp.uint8:
      raise ValueError(
          "Caches must be uint8, got"
          f" {cache_kv_nope.dtype} and {cache_kv_rope.dtype}."
      )
    page_size = cache_kv_nope.shape[1]
    rope_page_rows = cache_kv_rope.shape[1]
    if page_size != rope_page_rows * 4:
      raise ValueError(
          f"Expected cache_kv_nope page_size ({page_size}) to equal"
          f" 4 * cache_kv_rope.shape[1] ({rope_page_rows * 4})."
      )
    if topk_indices.shape[1] <= 0:
      raise ValueError(
          f"topk must be positive, got {topk_indices.shape[1]}."
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
    pages_per_seq = num_page_indices // max_num_seqs
    page_table_span = pages_per_seq * page_size
    if max_kv_len is not None and not (0 < max_kv_len <= page_table_span):
      raise ValueError(
          f"max_kv_len={max_kv_len} must be in (0, {page_table_span}]."
      )
    return super().bind(
        q=q,
        cache_kv_nope=cache_kv_nope,
        cache_kv_rope=cache_kv_rope,
        kv_lens=kv_lens,
        topk_indices=topk_indices,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=sm_scale,
        k_scale=k_scale,
        mask_value=mask_value,
        max_kv_len=max_kv_len,
        chunk_prefill_size=chunk_prefill_size,
        sequence_start=sequence_start,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      q: Float[Array, "T H D"],
      cache_kv_nope: UInt8[Array, "P S 4 128"],
      cache_kv_rope: UInt8[Array, "P R 4 128"],
      kv_lens: Int[Array, "M"],
      topk_indices: Int[Array, "T K"],
      page_indices: Int[Array, "N"],
      cu_q_lens: Int[Array, "B"],
      distribution: Int[Array, "3"],
      *,
      sm_scale: float = 1.0,
      k_scale: float = 1.0,
      mask_value: float | None = None,
      max_kv_len: int | None = None,
      chunk_prefill_size: int | None = None,
      sequence_start: Int[Array, ""] | None = None,
      return_residuals: bool = False,
      config: C,
  ) -> tuple[jax.Array, None]:
    return (
        reference.masked_dense_ragged_paged_attention(
            q,
            cache_kv_nope,
            cache_kv_rope,
            kv_lens,
            topk_indices,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            k_scale=k_scale,
            mask_value=mask_value,
            max_kv_len=max_kv_len,
            chunk_prefill_size=chunk_prefill_size,
            sequence_start=sequence_start,
        ),
        None,
    )
