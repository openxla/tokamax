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
"""MultiHeadLatentAttention operator definition."""

import dataclasses
import enum
from typing import Annotated, Any, ClassVar, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.mla import reference
from tokamax._src.ops.experimental.mla import utils

AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray | np.ndarray


class MlaLayout(enum.StrEnum):
  """Memory layout of `ql_nope`, the output, and the paged KV cache.

  The cache is allocated by the caller, so the layout is part of the op's
  contract rather than an implementation detail: an implementation supports
  some layouts and rejects the rest. `q_pe`, `new_kv_c` and `new_k_pe` are
  token-major in every layout.

  - TOKEN_MAJOR: `ql_nope` and the output are `[tokens, heads, lkv_dim]`; the
    cache is `[pages, page_size // packing, packing, kv_dim]` (tokens packed
    along sublanes).
  - HEAD_MAJOR_SEQ_ALONG_LANE: `ql_nope` and the output are
    `[heads, tokens, lkv_dim]`, which is what the absorbed `W_UK` / `W_UV`
    einsums around MLA produce and consume; the cache is
    `[pages, kv_dim, page_size]` (tokens along lanes).

  In both, `kv_dim` is `align_to(lkv_dim, 128) + align_to(r_dim, 128)`.
  """

  TOKEN_MAJOR = enum.auto()
  HEAD_MAJOR_SEQ_ALONG_LANE = enum.auto()


# Precomputed, implementation-specific metadata (e.g. MLA v3's per-pass DMA
# schedules). It is derived from the other arguments and the config, so it is
# neither serialized into the HLO metadata nor part of the autotuning key.
OpaqueMetadata = Annotated[Any, pydantic.PlainSerializer(lambda _: None)]


class MultiHeadLatentAttention[C](op.Op[Any, Any, None, C, Any]):
  """Tokamax operator for Multi-Head Latent Attention."""

  # Inference only and never vmapped, so skip the `custom_vjp` / custom batching
  # wrappers `Op.__call__` would otherwise put around every call.
  supports_batched_args_capture: ClassVar[bool] = False

  @jaxtyping.jaxtyped
  def bind(
      self,
      # [tokens, heads, lkv_dim] or [heads, tokens, lkv_dim]; see `layout`.
      ql_nope: Float[Array | AbstractArray, "ql_dim0 ql_dim1 actual_lkv_dim"],
      q_pe: Float[
          Array | AbstractArray,
          "max_num_tokens actual_num_q_heads actual_r_dim",
      ],
      new_kv_c: Float[Array | AbstractArray, "max_num_tokens actual_lkv_dim"],
      new_k_pe: Float[Array | AbstractArray, "max_num_tokens actual_r_dim"],
      # Rank depends on `layout`.
      cache_kv: Float[Array | AbstractArray, "..."],
      kv_lens: Int[Array | AbstractArray, "max_num_seqs"],
      page_indices: Int[Array | AbstractArray, "num_page_indices"],
      cu_q_lens: Int[Array | AbstractArray, "max_num_seqs_plus_1"],
      distribution: Int[Array | AbstractArray, "3"],
      *,
      sm_scale: float = 1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      mask_value: float | None = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      s_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      layout: MlaLayout = MlaLayout.TOKEN_MAJOR,
      schedules: OpaqueMetadata = None,
      debug_mode: bool = False,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds the arguments.

    Args:
      ql_nope: Query latent without positional encoding; layout per `layout`.
      q_pe: Query positional encoding, `[tokens, heads, r_dim]`.
      new_kv_c: New latent KV, `[tokens, lkv_dim]`.
      new_k_pe: New key positional encoding, `[tokens, r_dim]`.
      cache_kv: Paged KV cache; layout per `layout`.
      kv_lens: Per-sequence KV length, including the new tokens.
      page_indices: Flattened `[num_seqs * pages_per_seq]` page table.
      cu_q_lens: Cumulative query lengths, `[num_seqs + 1]`.
      distribution: `[d, p, n]`, splitting the sequences into decode `[0, d)`,
        prefill `[d, p)` and mixed `[p, n)` bands.
      sm_scale: Softmax scale.
      sliding_window: Optional sliding window, in tokens.
      soft_cap: Optional logit soft cap.
      mask_value: Optional value for masked logits.
      q_scale: Optional query dequantization scale.
      k_scale: Optional key dequantization scale.
      v_scale: Optional value dequantization scale.
      s_dtype: Logits dtype, for implementations that honor it.
      layout: Layout of `ql_nope`, the output and `cache_kv`.
      schedules: Optional precomputed metadata for implementations that support
        it (see e.g. `PallasTpuMultiHeadLatentAttentionV3.build_schedules`),
        letting a model build it once and share it across layers. It must have
        been built for the same arguments and config. Ignored by implementations
        that do not use it.
      debug_mode: Whether to log debugging information.
      return_residuals: Must be `False`; MLA is forward-only.

    Returns:
      The bound arguments.
    """
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    if num_page_indices % max_num_seqs != 0:
      raise ValueError(
          f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
      )
    if cu_q_lens.shape != (max_num_seqs + 1,):
      raise ValueError(
          f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)."
      )

    layout = MlaLayout(layout)
    num_tokens, num_heads = q_pe.shape[:2]
    if layout == MlaLayout.TOKEN_MAJOR:
      expected_ql_nope_dims = (num_tokens, num_heads)
      expected_cache_rank = 4
    else:
      expected_ql_nope_dims = (num_heads, num_tokens)
      expected_cache_rank = 3
    if tuple(ql_nope.shape[:2]) != expected_ql_nope_dims:
      raise ValueError(
          f"Expected {ql_nope.shape[:2]=} to be {expected_ql_nope_dims} for"
          f" {layout=}, given {q_pe.shape=}."
      )
    if len(cache_kv.shape) != expected_cache_rank:
      raise ValueError(
          f"Expected a rank-{expected_cache_rank} cache_kv for {layout=}, got"
          f" {cache_kv.shape=}."
      )

    return super().bind(
        ql_nope=ql_nope,
        q_pe=q_pe,
        new_kv_c=new_kv_c,
        new_k_pe=new_k_pe,
        cache_kv=cache_kv,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        s_dtype=s_dtype,
        layout=layout,
        schedules=schedules,
        debug_mode=debug_mode,
        return_residuals=return_residuals,
    )

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Any:
    # `schedules` is derived from the other arguments and the config, so it
    # must not affect which config is chosen (and is not hashable anyway).
    if ba.arguments.get("schedules") is not None:
      ba = dataclasses.replace(
          ba, arguments=dict(ba.arguments) | dict(schedules=None)
      )
    return super()._get_autotuning_cache_key(ba)

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      ql_nope: Float[Array, "ql_dim0 ql_dim1 actual_lkv_dim"],
      q_pe: Float[Array, "max_num_tokens actual_num_q_heads actual_r_dim"],
      new_kv_c: Float[Array, "max_num_tokens actual_lkv_dim"],
      new_k_pe: Float[Array, "max_num_tokens actual_r_dim"],
      cache_kv: Float[Array, "..."],
      kv_lens: Int[Array, "max_num_seqs"],
      page_indices: Int[Array, "num_page_indices"],
      cu_q_lens: Int[Array, "max_num_seqs_plus_1"],
      distribution: Int[Array, "3"],
      *,
      sm_scale: float = 1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      mask_value: float | None = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      s_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      layout: MlaLayout = MlaLayout.TOKEN_MAJOR,
      schedules: OpaqueMetadata = None,
      debug_mode: bool = False,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array], None]:
    # `schedules` is for other implementations.
    del schedules, debug_mode, return_residuals, config
    head_major = layout == MlaLayout.HEAD_MAJOR_SEQ_ALONG_LANE
    if head_major:
      ql_nope = jnp.swapaxes(ql_nope, 0, 1)
      cache_kv = utils.transpose_kv_cache_from_seq_along_lane(cache_kv)

    out, updated_cache_kv = reference.mla_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        s_dtype=s_dtype,
    )

    if head_major:
      out = jnp.swapaxes(out, 0, 1)
      updated_cache_kv = utils.transpose_kv_cache_to_seq_along_lane(
          updated_cache_kv
      )
    return (out, updated_cache_kv), None
