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
"""Base class for the DeepSeek-V4 compressor's compress-and-store.

See `reference` for the modes and cache layouts.
"""

from typing import Any, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.compress_store import reference

AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray | np.ndarray

# The RoPE dimension the kernel supports: its RoPE rotates the last 64 lanes of
# the last 128-lane head tile.
ROPE_HEAD_DIM = 64
# The lane period of CSA's FP8 scales: one scale per lane of a 64-lane cycle.
CSA_QUANT_BLOCK = 64
_LANES = 128
_INDEXER_LANES = 256


def _check_index_array(name: str, x: Any, num_tokens: int | None = None):
  if x.dtype != jnp.int32:
    raise ValueError(f"{name} must be int32, got {x.dtype}.")
  if num_tokens is not None and x.shape != (num_tokens,):
    raise ValueError(f"{name} must have shape ({num_tokens},), got {x.shape}.")


class CompressStore[C](
    op.Op[Any, tuple[jax.Array, jax.Array | None], None, C, Any]
):
  """Tokamax operator for the DeepSeek-V4 compress-and-store.

  The kernel updates the caches in place (`input_output_aliases`). As upstream,
  the Pallas implementation donates `cache` and `rope_cache`: the caller must
  not reuse them after the call (donation is ignored inside an enclosing
  `jax.jit`, which donates its own arguments instead).
  """

  @jaxtyping.jaxtyped
  def bind(
      self,
      cache: Shaped[Array | AbstractArray, "num_pages page_size *lanes"],
      positions: Int[Array | AbstractArray, "N"],
      block_table: Int[Array | AbstractArray, "B"],
      token_to_req_indices: Int[Array | AbstractArray, "N"],
      kv_slot_mapping: Int[Array | AbstractArray, "N"],
      rms_weight: Float[Array | AbstractArray, "head_dim"],
      *,
      cos_sin_cache: Float[Array | AbstractArray, "max_pos rope_head_dim"],
      block_table_stride: int,
      state_block_size: int,
      compress_ratio: int,
      overlap: bool,
      state_cache: (
          Shaped[Array | AbstractArray, "state_pages state_page_size 4 128"]
          | None
      ) = None,
      rope_cache: (
          Int[Array | AbstractArray, "num_pages rope_rows 128"] | None
      ) = None,
      quant_block: int = CSA_QUANT_BLOCK,
      rms_eps: float = 1e-6,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    num_tokens = positions.shape[0]
    _check_index_array("positions", positions)
    _check_index_array("token_to_req_indices", token_to_req_indices, num_tokens)
    _check_index_array("kv_slot_mapping", kv_slot_mapping, num_tokens)
    _check_index_array("block_table", block_table)
    if block_table_stride <= 0 or block_table.shape[0] % block_table_stride:
      raise ValueError(
          f"block_table size ({block_table.shape[0]}) must be a multiple of"
          f" block_table_stride ({block_table_stride})."
      )
    if state_block_size <= 0 or compress_ratio <= 0:
      raise ValueError(
          "state_block_size and compress_ratio must be positive, got"
          f" {state_block_size} and {compress_ratio}."
      )
    if cos_sin_cache.dtype != jnp.float32:
      raise ValueError(f"cos_sin_cache must be f32, got {cos_sin_cache.dtype}.")
    if cos_sin_cache.shape[1] != ROPE_HEAD_DIM:
      raise ValueError(
          f"rope_head_dim must be {ROPE_HEAD_DIM}, got"
          f" {cos_sin_cache.shape[1]}."
      )

    head_dim = rms_weight.shape[0]
    if head_dim % _LANES:
      raise ValueError(f"head_dim must be a multiple of 128, got {head_dim}.")

    if head_dim == reference.INDEXER_HEAD_DIM:
      # CSA indexer: 256-byte records packed 4 per uint8 `(4, 256)` row.
      if not overlap:
        raise ValueError("The CSA indexer (head_dim=128) requires overlap.")
      if cache.dtype != jnp.uint8 or cache.shape[2:] != (4, _INDEXER_LANES):
        raise ValueError(
            "The CSA indexer cache must be uint8 (num_pages, rows, 4, 256),"
            f" got {cache.dtype} {cache.shape}."
        )
      if quant_block <= 0 or _LANES % quant_block:
        raise ValueError(
            f"quant_block must divide 128 for the indexer, got {quant_block}."
        )
    elif overlap:
      # CSA: int32 NoPE + RoPE caches (see `csa_cache_layout`).
      if cache.dtype != jnp.int32 or cache.shape[2:] != (_LANES,):
        raise ValueError(
            "The CSA cache must be int32 (num_pages, rows, 128), got"
            f" {cache.dtype} {cache.shape}."
        )
      if rope_cache is None:
        raise ValueError("CSA (overlap=True, head_dim!=128) needs rope_cache.")
      if rope_cache.dtype != jnp.int32 or rope_cache.shape != (
          cache.shape[0],
          cache.shape[1] // 4,
          _LANES,
      ):
        raise ValueError(
            "rope_cache must be int32 (num_pages, rows // 4, 128), got"
            f" {rope_cache.dtype} {rope_cache.shape}."
        )
      if quant_block != CSA_QUANT_BLOCK:
        raise ValueError(
            f"quant_block must be {CSA_QUANT_BLOCK} for CSA, got {quant_block}."
        )
    else:
      # HCA: bf16 records in a uint8 `(4, 128)` slab cache.
      if cache.dtype != jnp.uint8 or cache.shape[2:] != (4, _LANES):
        raise ValueError(
            "The HCA cache must be uint8 (num_pages, rows, 4, 128), got"
            f" {cache.dtype} {cache.shape}."
        )
    if state_cache is not None and state_cache.dtype != jnp.uint8:
      raise ValueError(f"state_cache must be uint8, got {state_cache.dtype}.")

    return super().bind(
        cache=cache,
        positions=positions,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        block_table_stride=block_table_stride,
        state_block_size=state_block_size,
        compress_ratio=compress_ratio,
        overlap=overlap,
        state_cache=state_cache,
        rope_cache=rope_cache,
        quant_block=quant_block,
        rms_eps=rms_eps,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      cache: Shaped[Array, "num_pages page_size *lanes"],
      positions: Int[Array, "N"],
      block_table: Int[Array, "B"],
      token_to_req_indices: Int[Array, "N"],
      kv_slot_mapping: Int[Array, "N"],
      rms_weight: Float[Array, "head_dim"],
      *,
      cos_sin_cache: Float[Array, "max_pos rope_head_dim"],
      block_table_stride: int,
      state_block_size: int,
      compress_ratio: int,
      overlap: bool,
      state_cache: (
          Shaped[Array, "state_pages state_page_size 4 128"] | None
      ) = None,
      rope_cache: Int[Array, "num_pages rope_rows 128"] | None = None,
      quant_block: int = CSA_QUANT_BLOCK,
      rms_eps: float = 1e-6,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array | None], None]:
    """Compresses each boundary token's window and stores it into the cache.

    Runs `reference.compress_norm_rope_store`. See `reference` for the modes
    and cache layouts.

    Args:
      cache: The compressed KV cache: uint8 `(num_pages, rows, 4, 128)` (HCA),
        int32 `(num_pages, rows, 128)` (CSA) or uint8 `(num_pages, rows, 4,
        256)` (CSA indexer). For CSA and the indexer it also hosts the f32
        state.
      positions: `(N,)` int32 token positions.
      block_table: `(num_reqs * block_table_stride,)` int32 state pages of each
        request.
      token_to_req_indices: `(N,)` int32 request of each token.
      kv_slot_mapping: `(N,)` int32 compressed-KV slot of each token, or -1 to
        skip it. Only boundary tokens (`(position + 1) % compress_ratio == 0`)
        may have a slot.
      rms_weight: `(head_dim,)` f32 RMSNorm weight.
      cos_sin_cache: `(max_pos, 64)` f32 RoPE `[cos | sin]` table, indexed by
        the compressed position `position // compress_ratio * compress_ratio`.
      block_table_stride: Row stride of `block_table`.
      state_block_size: Token states per state page.
      compress_ratio: Tokens compressed into one record.
      overlap: Whether windows overlap; selects CSA (or the indexer when
        `head_dim == 128`) over HCA.
      state_cache: The separate uint8 f32-state array (HCA), or `None` when
        `cache` hosts the state.
      rope_cache: `(num_pages, rows // 4, 128)` int32 RoPE cache, required for
        CSA and ignored otherwise.
      quant_block: FP8 quantization block: the scale lane period (64) for CSA,
        the scale block for the indexer. Unused for HCA.
      rms_eps: RMSNorm epsilon.
      return_residuals: Unused; the op has no residuals.
      config: Unused.

    Returns:
      `((cache, rope_cache), None)` with the updated caches; `rope_cache` is
      `None` outside CSA.
    """
    del config  # Unused.
    return (
        reference.compress_norm_rope_store(
            cache,
            positions,
            block_table,
            token_to_req_indices,
            kv_slot_mapping,
            rms_weight,
            cos_sin_cache=cos_sin_cache,
            block_table_stride=block_table_stride,
            state_block_size=state_block_size,
            compress_ratio=compress_ratio,
            overlap=overlap,
            state_cache=state_cache,
            rope_cache=rope_cache,
            quant_block=quant_block,
            rms_eps=rms_eps,
        ),
        None,
    )
