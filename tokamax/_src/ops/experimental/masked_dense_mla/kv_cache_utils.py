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
"""Utilities for paged KV cache indexing and byte packing.

NOTE: Currently designed for 8-bit / FP8 KV caches (1 byte per element). We use
32-bit physical storage for SparseCore layout but the logical cache is still fp8
"""

import dataclasses
import enum

import jax
import jax.numpy as jnp

# Both layouts store a token as 32-bit words: the native SparseCore cache holds
# them as uint32, the TensorCore-tiled cache as WORD_BYTES uint8 rows that
# `ref.bitcast(jnp.int32)` folds into one word row inside the kernel.
WORD_BYTES = 4
TILE_LANE_BYTES = 128  # lane width of a TensorCore tile, in bytes

# Sentinels for tokens that must not be written. JAX drops out-of-bounds
# scatters; the Pallas writers predicate on the sign instead.
OOB_PAGE = 2**30
SKIP_ROW = -1


class KVCacheLayout(enum.Enum):
  """Physical HBM layout of a paged KV cache.

  TENSORCORE: uint8, 2D-tiled. 4 sublanes fold into 32-bit words via hardware
  TC tiling.
  SPARSECORE: uint32, 1D-flat. The writer explicitly bit-packs `WORD_BYTES`
  byte bands into 32-bit words (pack_tokens) for native SC DMA.
  """

  SPARSECORE = "sparsecore"
  TENSORCORE = "tensorcore"


class KVCacheType(enum.Enum):
  """The type of a paged KV cache."""

  NOPE = "nope"
  ROPE = "rope"


_LAYOUT_DTYPES = {
    KVCacheLayout.TENSORCORE: jnp.uint8,
    KVCacheLayout.SPARSECORE: jnp.uint32,
}


def align_to(x: int, a: int) -> int:
  return (x + a - 1) // a * a


def get_dtype_bitwidth(dtype: jax.typing.DTypeLike) -> int:
  return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype: jax.typing.DTypeLike) -> int:
  bits = get_dtype_bitwidth(dtype)
  return 32 // bits


def get_kv_cache_shape(
    total_num_pages: int,
    page_size: int,
    kv_dim: int,
    kv_dtype: jax.typing.DTypeLike | None = None,
    kv_packing: int | None = None,
) -> tuple[int, int, int, int]:
  """Computes padded 4D shape `(total_num_pages, page_size // kv_packing, kv_packing, kv_dim)`."""
  if kv_packing is None:
    assert kv_dtype is not None, "must pass kv_dtype if kv_packing is None"
    kv_packing = get_dtype_packing(kv_dtype)
  return (
      total_num_pages,
      align_to(page_size, kv_packing) // kv_packing,
      kv_packing,
      align_to(kv_dim, TILE_LANE_BYTES),
  )


@dataclasses.dataclass(frozen=True)
class SparseMLAKVCacheSpec:
  """Layout descriptor for a paged sparse-MLA KV cache."""

  cache_type: KVCacheType
  layout: KVCacheLayout
  num_pages: int
  page_size: int
  head_dim: int
  kv_packing: int

  @classmethod
  def create(
      cls,
      cache_type: KVCacheType,
      layout: KVCacheLayout,
      num_pages: int,
      page_size: int,
      head_dim: int,
      kv_packing: int = 4,
  ) -> "SparseMLAKVCacheSpec":
    """Builds a spec with the kernel's page/head-dim padding applied."""
    num_pages, packed_page_size, kv_packing, head_dim = get_kv_cache_shape(
        num_pages, page_size, head_dim, None, kv_packing
    )
    page_size = packed_page_size * kv_packing

    if layout is KVCacheLayout.SPARSECORE:
      assert (
          head_dim % WORD_BYTES == 0
      ), f"head_dim {head_dim} must be a multiple of {WORD_BYTES}"
      if cache_type is KVCacheType.ROPE:
        assert (
            page_size % WORD_BYTES == 0
        ), f"page_size {page_size} must be a multiple of {WORD_BYTES}"
    elif layout is KVCacheLayout.TENSORCORE:
      if cache_type is KVCacheType.NOPE:
        assert head_dim == WORD_BYTES * TILE_LANE_BYTES, (
            f"NOPE TC head_dim {head_dim} must equal"
            f" {WORD_BYTES * TILE_LANE_BYTES}"
            f" ({WORD_BYTES * TILE_LANE_BYTES}B per token)"
        )

    return cls(
        cache_type=cache_type,
        layout=layout,
        num_pages=num_pages,
        page_size=page_size,
        head_dim=head_dim,
        kv_packing=kv_packing,
    )

  @property
  def jax_dtype(self) -> jax.typing.DTypeLike:
    return _LAYOUT_DTYPES[self.layout]

  @property
  def token_bytes(self) -> int:
    """Bytes reserved per token slot (assumes 1 byte/element for FP8)."""
    return self.head_dim

  @property
  def shape(self) -> tuple[int, ...]:
    if self.cache_type is KVCacheType.NOPE:
      if self.layout is KVCacheLayout.TENSORCORE:
        return (
            self.num_pages,
            self.page_size,
            self.head_dim // TILE_LANE_BYTES,
            TILE_LANE_BYTES,
        )
      return (self.num_pages, self.page_size, self.head_dim // WORD_BYTES)
    if self.cache_type is KVCacheType.ROPE:
      if self.layout is KVCacheLayout.TENSORCORE:
        return (
            self.num_pages,
            self.page_size // self.kv_packing,
            self.kv_packing,
            self.head_dim,
        )
      return (self.num_pages, self.page_size // WORD_BYTES, self.head_dim)
    raise ValueError(f"unsupported cache type {self.cache_type}")


def as_token_bytes(values: jax.Array, token_bytes: int) -> jax.Array:
  """`[num_tokens, n]` -> `[num_tokens, token_bytes]` uint8, zero-padded."""
  u8 = jax.lax.bitcast_convert_type(values, jnp.uint8).reshape(
      values.shape[0], -1
  )
  assert (
      u8.shape[-1] <= token_bytes
  ), f"token is {u8.shape[-1]}B, more than the {token_bytes}B reserved for it"
  pad = token_bytes - u8.shape[-1]
  return jnp.pad(u8, ((0, 0), (0, pad))) if pad else u8


def pack_tokens(values: jax.Array, token_bytes: int) -> jax.Array:
  """`[num_tokens, n]` -> `[num_tokens, token_bytes // WORD_BYTES]` uint32.

  Each word takes one byte from each of WORD_BYTES equal bands of the token,
  the same interleave `ref.bitcast(jnp.int32)` produces from uint8 rows.
  """
  assert (
      token_bytes % WORD_BYTES == 0
  ), f"token_bytes ({token_bytes}) must be a multiple of {WORD_BYTES}"
  u8 = as_token_bytes(values, token_bytes)
  bands = u8.reshape(u8.shape[0], WORD_BYTES, token_bytes // WORD_BYTES).astype(
      jnp.uint32
  )
  return (
      bands[:, 0]
      | (bands[:, 1] << 8)
      | (bands[:, 2] << 16)
      | (bands[:, 3] << 24)
  )


def _row_positions(
    num_tokens: int,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Per-row `(token, sequence, global KV position, valid)`."""
  num_seqs = seq_lens.shape[0]
  tok = jnp.arange(num_tokens, dtype=jnp.int32)
  seq_id = jnp.searchsorted(query_start_loc[1:], tok, side="right").astype(
      jnp.int32
  )
  valid = tok < query_start_loc[-1]

  seq_id = jnp.minimum(seq_id, num_seqs - 1)
  q_len = query_start_loc[seq_id + 1] - query_start_loc[seq_id]
  pos = seq_lens[seq_id] - q_len + (tok - query_start_loc[seq_id])
  return tok, seq_id, pos, valid


def get_page_and_slot(
    num_tokens: int,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    page_size: int,
) -> tuple[jax.Array, jax.Array]:
  """Computes target (page, slot) in the paged KV cache for each token."""
  _, seq_id, pos, valid = _row_positions(num_tokens, seq_lens, query_start_loc)

  block_tables_2d = block_tables.reshape(seq_lens.shape[0], -1)
  max_pages_per_seq = block_tables_2d.shape[1]
  safe_page_idx = jnp.clip(pos // page_size, 0, max_pages_per_seq - 1)
  page = jnp.where(
      valid, block_tables_2d[seq_id, safe_page_idx], jnp.int32(OOB_PAGE)
  )
  slot = jnp.where(valid, pos % page_size, 0)
  return page, slot


def get_dst_rows(
    num_tokens: int,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    page_size: int,
) -> jax.Array:
  """Computes flattened row indices for SparseCore scatter kernels."""
  page, slot = get_page_and_slot(
      num_tokens, seq_lens, block_tables, query_start_loc, page_size
  )
  return _dst_rows(page, slot, page_size)


def _dst_rows(page: jax.Array, slot: jax.Array, page_size: int) -> jax.Array:
  """(page, slot) -> flattened row, `SKIP_ROW` wherever `page` is `OOB_PAGE`."""
  return jnp.where(page < OOB_PAGE, page * page_size + slot, SKIP_ROW)


def _scatter_rows(
    cache: jax.Array,
    spec: SparseMLAKVCacheSpec,
    values: jax.Array,
    page: jax.Array,
    slot: jax.Array,
) -> jax.Array:
  """Writes each token's `values` into the (`page`, `slot`) it lands in."""
  if spec.layout is KVCacheLayout.TENSORCORE:
    src = as_token_bytes(values, spec.token_bytes)
    if spec.cache_type is KVCacheType.NOPE:
      return cache.at[page, slot].set(
          src.reshape(src.shape[0], *cache.shape[2:])
      )
    return cache.at[page, slot // spec.kv_packing, slot % spec.kv_packing].set(
        src
    )
  if spec.layout is KVCacheLayout.SPARSECORE:
    words = pack_tokens(values, spec.token_bytes)
    rows = cache.reshape(spec.num_pages, spec.page_size, -1)
    return rows.at[page, slot].set(words).reshape(cache.shape)

  raise ValueError(f"unsupported layout {spec.layout}")


def _check_inputs(
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    kv_c_normed: jax.Array,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
) -> None:
  """Asserts the caches were allocated from the specs they are written with."""
  assert kv_c_normed.dtype == jnp.float8_e4m3fn, (
      f"sparse MLA kernel requires --kv-cache-dtype fp8 (got {kv_c_normed.dtype})"
  )
  assert kv_cache_nope.dtype == nope_spec.jax_dtype, (
      f"nope cache {kv_cache_nope.dtype} does not match its spec "
      f"{nope_spec.jax_dtype}"
  )
  assert kv_cache_rope.dtype == rope_spec.jax_dtype, (
      f"rope cache {kv_cache_rope.dtype} does not match its spec "
      f"{rope_spec.jax_dtype}"
  )
  assert kv_cache_nope.shape == nope_spec.shape, (
      f"nope cache {kv_cache_nope.shape} does not match its spec {nope_spec.shape}"
  )
  assert kv_cache_rope.shape == rope_spec.shape, (
      f"rope cache {kv_cache_rope.shape} does not match its spec {rope_spec.shape}"
  )
  assert nope_spec.page_size == rope_spec.page_size, (
      f"nope page size {nope_spec.page_size} != rope page size {rope_spec.page_size}"
  )


def update_sparse_mla_kv_cache(
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    *,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
) -> tuple[jax.Array, jax.Array]:
  """Scatters this step's new MLA latents into the split sparse-MLA cache."""
  _check_inputs(kv_cache_nope, kv_cache_rope, kv_c_normed, nope_spec, rope_spec)
  page, slot = get_page_and_slot(
      kv_c_normed.shape[0],
      seq_lens,
      block_tables,
      query_start_loc,
      nope_spec.page_size,
  )
  return _scatter_caches(
      kv_cache_nope,
      kv_cache_rope,
      kv_c_normed,
      k_pe,
      page,
      slot,
      nope_spec,
      rope_spec,
  )


def _scatter_caches(
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    page: jax.Array,
    slot: jax.Array,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
) -> tuple[jax.Array, jax.Array]:
  """Writes both caches at (`page`, `slot`), with Pallas wherever it can."""
  # Imported here, not at module scope: `scatter` imports this module back.
  from tokamax._src.ops.experimental.masked_dense_mla import scatter  # pylint: disable=g-import-not-at-top

  dst_rows = _dst_rows(page, slot, nope_spec.page_size)
  kv_cache_nope = scatter.scatter(
      kv_cache_nope, kv_c_normed, nope_spec, dst_rows=dst_rows
  )
  if rope_spec.layout is KVCacheLayout.SPARSECORE:
    kv_cache_rope = scatter.scatter(
        kv_cache_rope, k_pe, rope_spec, dst_rows=dst_rows
    )
  else:
    kv_cache_rope = _scatter_rows(kv_cache_rope, rope_spec, k_pe, page, slot)
  return kv_cache_nope, kv_cache_rope


def update_sparse_mla_kv_cache_jax(
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    *,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
) -> tuple[jax.Array, jax.Array]:
  """Pure-XLA `update_sparse_mla_kv_cache`, for every layout."""
  _check_inputs(kv_cache_nope, kv_cache_rope, kv_c_normed, nope_spec, rope_spec)
  page, slot = get_page_and_slot(
      kv_c_normed.shape[0],
      seq_lens,
      block_tables,
      query_start_loc,
      nope_spec.page_size,
  )
  return (
      _scatter_rows(kv_cache_nope, nope_spec, kv_c_normed, page, slot),
      _scatter_rows(kv_cache_rope, rope_spec, k_pe, page, slot),
  )


def update_sparse_mla_kv_cache_dcp(
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    dcp_rank: jax.Array,
    *,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
    dcp_size: int,
    interleave_size: int,
) -> tuple[jax.Array, jax.Array]:
  """Scatters this step's MLA latents into a DCP position-sharded cache."""
  _check_inputs(kv_cache_nope, kv_cache_rope, kv_c_normed, nope_spec, rope_spec)
  page, slot = _get_dcp_page_and_slot(
      kv_c_normed.shape[0],
      seq_lens,
      block_tables,
      query_start_loc,
      dcp_rank,
      nope_spec=nope_spec,
      dcp_size=dcp_size,
      interleave_size=interleave_size,
  )
  return _scatter_caches(
      kv_cache_nope,
      kv_cache_rope,
      kv_c_normed,
      k_pe,
      page,
      slot,
      nope_spec,
      rope_spec,
  )


def update_sparse_mla_kv_cache_dcp_jax(
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    dcp_rank: jax.Array,
    *,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
    dcp_size: int,
    interleave_size: int,
) -> tuple[jax.Array, jax.Array]:
  """Pure-XLA `update_sparse_mla_kv_cache_dcp`, for every layout."""
  _check_inputs(kv_cache_nope, kv_cache_rope, kv_c_normed, nope_spec, rope_spec)
  page, slot = _get_dcp_page_and_slot(
      kv_c_normed.shape[0],
      seq_lens,
      block_tables,
      query_start_loc,
      dcp_rank,
      nope_spec=nope_spec,
      dcp_size=dcp_size,
      interleave_size=interleave_size,
  )
  return (
      _scatter_rows(kv_cache_nope, nope_spec, kv_c_normed, page, slot),
      _scatter_rows(kv_cache_rope, rope_spec, k_pe, page, slot),
  )


def _get_dcp_page_and_slot(
    num_tokens: int,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    dcp_rank: jax.Array,
    *,
    nope_spec: SparseMLAKVCacheSpec,
    dcp_size: int,
    interleave_size: int,
) -> tuple[jax.Array, jax.Array]:
  """This rank's (page, slot) per token; `OOB_PAGE` for rows it does not own."""
  page_size = nope_spec.page_size
  if interleave_size % WORD_BYTES:
    raise ValueError(
        f"interleave_size={interleave_size} must be a multiple of "
        f"WORD_BYTES={WORD_BYTES}, or a rope tile -- which packs that many "
        "consecutive tokens into one word -- would straddle two ranks."
    )
  if page_size % interleave_size:
    raise ValueError(
        f"interleave_size={interleave_size} must divide "
        f"page_size={page_size}. Otherwise an interleave chunk straddles "
        "a page boundary and the local index a reader derives from the "
        "global position stops agreeing with the (page, slot) written "
        "here."
    )

  _, seq_id, pos, in_batch = _row_positions(
      num_tokens, seq_lens, query_start_loc
  )

  virtual_page_size = jnp.int32(page_size * dcp_size)
  cycle = jnp.int32(dcp_size * interleave_size)
  interleave = jnp.int32(interleave_size)
  offset = pos % virtual_page_size
  owner = (offset % cycle) // interleave
  slot = (offset // cycle) * interleave + (offset % interleave)

  local_block_tables = jnp.mod(
      block_tables, jnp.int32(nope_spec.num_pages)
  ).reshape(seq_lens.shape[0], -1)
  max_pages_per_seq = local_block_tables.shape[1]
  virtual_page = jnp.clip(pos // virtual_page_size, 0, max_pages_per_seq - 1)

  valid = jnp.logical_and(in_batch, owner == dcp_rank)
  page = jnp.where(
      valid, local_block_tables[seq_id, virtual_page], jnp.int32(OOB_PAGE)
  )
  return page, slot
