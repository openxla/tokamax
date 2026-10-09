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
"""Base class for Sparse Multi-Head Latent Attention (SparseMLA)."""

from typing import Any, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.sparse_mla import reference


class SparseMla[C](op.Op[Any, jax.Array, None, C, Any]):
  """Tokamax operator for Sparse Multi-Head Latent Attention."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      q: Float[Array, "T H D"],
      cache_kv_nope: Int[Array, "P S 128"],
      cache_kv_rope: Int[Array, "P R 128"],
      topk_indices: Int[Array, "T K"],
      page_indices: Int[Array, "N"],
      cu_q_lens: Int[Array, "B"],
      distribution: Int[Array, "3"],
      attention_sinks: Float[Array, "H"],
      swa_accumution: Float[Array, "T H D"],
      swa_l: Float[Array, "T H"],
      swa_m: Float[Array, "T H"],
      *,
      sm_scale: float = 1.0,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    if cache_kv_nope.dtype != jnp.int32 or cache_kv_rope.dtype != jnp.int32:
      raise ValueError(
          "Caches must be int32, got"
          f" {cache_kv_nope.dtype} and {cache_kv_rope.dtype}."
      )
    page_size = cache_kv_nope.shape[1]
    rope_page_rows = cache_kv_rope.shape[1]
    if page_size != rope_page_rows * 4:
      raise ValueError(
          f"Expected cache_kv_nope page_size ({page_size}) to equal"
          f" 4 * cache_kv_rope.shape[1] ({rope_page_rows * 4})."
      )
    if q.shape[-1] != reference.HEAD_DIM:
      raise ValueError(
          f"Expected head_dim={reference.HEAD_DIM}, got {q.shape[-1]}."
      )
    if swa_accumution.dtype != q.dtype:
      raise ValueError(
          f"Expected swa_accumution.dtype ({swa_accumution.dtype}) to match"
          f" q.dtype ({q.dtype})."
      )
    topk = topk_indices.shape[1]
    if topk <= 0 or topk % 128 != 0:
      raise ValueError(f"topk must be a positive multiple of 128, got {topk}.")
    max_num_seqs = cu_q_lens.shape[0] - 1
    num_page_indices = page_indices.shape[0]
    if max_num_seqs <= 0 or num_page_indices % max_num_seqs != 0:
      raise ValueError(
          f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
      )
    return super().bind(
        q=q,
        cache_kv_nope=cache_kv_nope,
        cache_kv_rope=cache_kv_rope,
        topk_indices=topk_indices,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        attention_sinks=attention_sinks,
        swa_accumution=swa_accumution,
        swa_l=swa_l,
        swa_m=swa_m,
        sm_scale=sm_scale,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      q: Float[Array, "T H D"],
      cache_kv_nope: Int[Array, "P S 128"],
      cache_kv_rope: Int[Array, "P R 128"],
      topk_indices: Int[Array, "T K"],
      page_indices: Int[Array, "N"],
      cu_q_lens: Int[Array, "B"],
      distribution: Int[Array, "3"],
      attention_sinks: Float[Array, "H"],
      swa_accumution: Float[Array, "T H D"],
      swa_l: Float[Array, "T H"],
      swa_m: Float[Array, "T H"],
      *,
      sm_scale: float = 1.0,
      return_residuals: bool = False,
      config: C,
  ) -> tuple[jax.Array, None]:
    return (
        reference.sparse_ragged_paged_attention(
            q,
            cache_kv_nope,
            cache_kv_rope,
            topk_indices,
            page_indices,
            cu_q_lens,
            distribution,
            attention_sinks,
            swa_accumution,
            swa_l,
            swa_m,
            sm_scale=sm_scale,
        ),
        None,
    )
