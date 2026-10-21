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
"""Static geometry of the compress-and-store Pallas kernel.

Modes:
  HCA:
    - State Bytes: 1024 (dim) x 4 (bytes per fp32) = 4096 bytes
    - Head Dim:    512 (bf16, no quantization)
    - Cache:       [num_pages, _, 4, 128] uint8 (where _ = state_block_size * 8
    or kv_block_size * 2)

  CSA:
    - State Bytes: 2048 (dim) x 4 (bytes per fp32) = 8192 bytes
    - Head Dim:    448 fp8 + 64 lane-periodic scales = 512 bytes
    - Cache:       [num_pages, _, 4, 128] uint8 (where _ = state_block_size * 16
    or kv_block_size), allocated as [num_pages, _, 128] int32
    - RoPE Cache:  64 bf16 = 128 bytes, [num_pages, kv_block_size // 4, 4, 128]
    uint8, allocated as [num_pages, kv_block_size // 4, 128] int32 (see
    `csa_cache_layout`)

  CSA-Indexer:
    - State Bytes: 512 (dim) x 4 (bytes per fp32) = 2048 bytes
    - Head Dim:    128 fp8 + 1 scale = 129 bytes -> 256 bytes
    - Cache:       [num_pages, _, 4, 256] uint8 (where _ = state_block_size * 4
    or kv_block_size // 4).
"""

import dataclasses
import enum
from typing import Any

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

# --- physical layout constants ------------------------------------------------
LANE = 128  # bytes per sub-slot / TPU lane width
SLOT_PACK = 4  # sub-slots packed into one physical HBM slot row
N_FIELDS = 2  # values stored per token: kv + score
FP32_BYTES = 4
CSA_COMPRESS_RATIO = 4  # the compress ratio that selects Mode.CSA


class Mode(enum.Enum):
  HCA = "hca"
  CSA = "csa"
  CSA_INDEXER = "csa_indexer"


_MODE_DEFAULTS: dict[Mode, dict[str, Any]] = {
    Mode.HCA: dict(
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=128,
        quant_block=0,
        overlap=False,
        has_rope_cache=False,
    ),
    Mode.CSA: dict(
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=4,
        quant_block=64,
        overlap=True,
        has_rope_cache=True,
    ),
    Mode.CSA_INDEXER: dict(
        head_dim=128,
        rope_head_dim=64,
        compress_ratio=4,
        quant_block=128,
        overlap=True,
        has_rope_cache=False,
    ),
}


def select_mode(head_dim: int, overlap: bool) -> Mode:
  """Picks the storage mode from the compressor's head_dim / overlap flag."""
  if head_dim == LANE:
    return Mode.CSA_INDEXER
  return Mode.CSA if overlap else Mode.HCA


def physical_page_size(
    mode: Mode, kv_cache_block_size: int, compress_ratio: int
) -> int:
  """Rows per page of the uint8 cache array allocated for `mode`.

  This must mirror the shapes `KVCacheManager._create_dsv4_kv_caches`
  allocates, because `compress_norm_rope_store` re-derives the whole page
  geometry from `cache.shape[1]`. Anything that computes slot indices for
  that array (the compressor wrapper's state-cache `block_size`, and
  `derive_metadata`) has to agree with it or writes land on the wrong page.

  Args:
    mode: The storage mode.
    kv_cache_block_size: Uncompressed tokens per KV cache block.
    compress_ratio: Tokens compressed into one record.

  Returns:
    The number of rows per page.
  """
  # Compressed KV tokens per page (vLLM's `spec.storage_block_size`).
  storage_block_size = kv_cache_block_size // compress_ratio
  if mode is Mode.CSA_INDEXER:
    # (N, T // 4, 4, 256) u8
    return storage_block_size // SLOT_PACK
  if mode is Mode.CSA:
    # (N, T, 4, 128) u8
    return storage_block_size
  # HCA (N, T * 2, 4, 128) u8
  return storage_block_size * 2


def state_host_page_size(
    mode: Mode, kv_cache_block_size: int, compress_ratio: int
) -> int:
  """Rows per page of the array that *hosts* `mode`'s f32 compressor state."""
  if mode is Mode.HCA:
    return physical_page_size(Mode.CSA, kv_cache_block_size, CSA_COMPRESS_RATIO)
  return physical_page_size(mode, kv_cache_block_size, compress_ratio)


def last_dim_size(mode: Mode) -> int:
  """Size of the last HBM cache dimension in bytes."""
  return 2 * LANE if mode is Mode.CSA_INDEXER else LANE


def row_size_bytes(mode: Mode) -> int:
  """Size of one physical HBM row in bytes."""
  return SLOT_PACK * last_dim_size(mode)


def state_width(head_dim: int, overlap: bool) -> int:
  """Width of one stored field: head_dim, doubled when windows overlap."""
  # Overlapping windows keep two copies per token: prev + curr.
  return (1 + int(overlap)) * head_dim


def state_rows_per_token(mode: Mode, head_dim: int, overlap: bool) -> int:
  """Physical HBM rows one token's full state (kv + score) occupies."""
  state_bytes = N_FIELDS * state_width(head_dim, overlap) * FP32_BYTES
  return state_bytes // row_size_bytes(mode)


def _state_page_capacity(
    mode: Mode,
    kv_cache_block_size: int,
    compress_ratio: int | None = None,
    head_dim: int | None = None,
) -> int:
  """Token states one page of `mode`'s host array physically holds."""
  defaults = _MODE_DEFAULTS[mode]
  compress_ratio = (
      defaults["compress_ratio"] if compress_ratio is None else compress_ratio
  )
  head_dim = defaults["head_dim"] if head_dim is None else head_dim
  page_rows = state_host_page_size(mode, kv_cache_block_size, compress_ratio)
  return page_rows // state_rows_per_token(mode, head_dim, defaults["overlap"])


def state_block_size(
    mode: Mode,
    kv_cache_block_size: int,
    compress_ratio: int | None = None,
    head_dim: int | None = None,
) -> int:
  """Token states vLLM should page `mode`'s state cache at.

  vLLM buckets DeepSeek-V4 cache groups by `(block_size, sliding_window)`,
  we let CSA attention and the indexer's compressor state to have same block
  size to land in the kv cache group.

  Args:
    mode: The storage mode.
    kv_cache_block_size: Uncompressed tokens per KV cache block.
    compress_ratio: Tokens compressed into one record; the mode's default if
      `None`.
    head_dim: Head dimension; the mode's default if `None`.

  Returns:
    The number of token states per state page.
  """
  own = _state_page_capacity(
      mode, kv_cache_block_size, compress_ratio, head_dim
  )
  # CSA and the indexer are each other's peer; HCA has none and pages alone.
  peer_mode = {
      Mode.CSA: Mode.CSA_INDEXER,
      Mode.CSA_INDEXER: Mode.CSA,
  }.get(mode)
  if peer_mode is None:
    return own
  return min(own, _state_page_capacity(peer_mode, kv_cache_block_size))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TileSizes:
  """Tile sizes for the kernel."""

  tile_n: int


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Dimensions:
  """True inputs only. Anything derivable is a property below."""

  mode: Mode = dataclasses.field(metadata=dict(static=True))
  size_n: int  # num_tokens
  head_dim: int
  rope_head_dim: int
  compress_ratio: int
  physical_page_size: int
  state_physical_page_size: int
  state_block_size: int = dataclasses.field(metadata=dict(static=True))
  quant_block: int
  overlap: bool
  has_rope_cache: bool
  rms_eps: float = 1e-6

  cos_sin_dtype: jax.typing.DTypeLike = jnp.float32
  rope_width: int = 128

  @property
  def is_quantized(self) -> bool:
    return self.quant_block > 0

  @property
  def has_rope(self) -> bool:
    return self.rope_head_dim > 0

  @property
  def nope_dtype(self) -> jax.typing.DTypeLike:
    return jnp.uint8 if self.is_quantized else jnp.bfloat16

  @property
  def rope_dtype(self) -> jax.typing.DTypeLike:
    return self.nope_dtype


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Configs:
  """Configuration for the kernel."""

  tile_sizes: TileSizes
  dims: Dimensions

  # --- factory ---------------------------------------------------------------
  @classmethod
  def make(
      cls,
      mode: Mode,
      *,
      size_n,
      physical_page_size,
      state_block_size,
      state_physical_page_size=None,
      rms_eps=1e-6,
      tile_n=4,
      **overrides,
  ) -> "Configs":
    """Build a config for `mode`; `overrides` replace per-mode defaults.

    `state_physical_page_size` defaults to `physical_page_size`, i.e. the
    state and the compressed KV share one buffer.

    Args:
      mode: The storage mode.
      size_n: The number of tokens.
      physical_page_size: Rows per page of the compressed-KV array.
      state_block_size: Token states per state page.
      state_physical_page_size: Rows per page of the state array.
      rms_eps: RMSNorm epsilon.
      tile_n: Tokens processed per grid step.
      **overrides: Per-mode `Dimensions` defaults to replace.

    Returns:
      The kernel configuration.
    """
    actual_overrides = {**_MODE_DEFAULTS[mode], **overrides}
    if mode == Mode.HCA:
      actual_overrides["quant_block"] = 0

    dims = Dimensions(
        mode=mode,
        size_n=size_n,
        physical_page_size=physical_page_size,
        state_block_size=state_block_size,
        state_physical_page_size=(
            physical_page_size
            if state_physical_page_size is None
            else state_physical_page_size
        ),
        rms_eps=rms_eps,
        **actual_overrides,
    )
    return cls(tile_sizes=TileSizes(tile_n=tile_n), dims=dims)

  # --- compute tiling (logical, in LANE-wide tiles) --------------------------
  @property
  def overlap_factor(self) -> int:
    """State copies kept per step: 1, or 2 when windows overlap (prev+curr)."""
    return 1 + int(self.dims.overlap)

  @property
  def state_width(self) -> int:
    """Width of one stored field (head_dim, doubled when overlapping)."""
    return state_width(self.dims.head_dim, self.dims.overlap)

  @property
  def head_tiles(self) -> int:
    """head_dim split into LANE-wide compute tiles (old: slots_per_part)."""
    return self.dims.head_dim // LANE

  @property
  def window(self) -> int:
    """Timesteps compressed together (x overlap_factor when overlapping)."""
    return self.dims.compress_ratio * self.overlap_factor

  # --- rope sub-layout -------------------------------------------------------
  @property
  def nope_dim(self) -> int:
    """Width of the non-rope (nope) part of a head."""
    return self.dims.head_dim - self.dims.rope_head_dim

  @property
  def nope_store_dim(self) -> int:
    """Dimension of the nope storage (contains rope if no separate rope cache)."""
    return (
        self.dims.head_dim - self.dims.rope_head_dim
        if self.dims.has_rope_cache
        else self.dims.head_dim
    )

  @property
  def half_rope(self) -> int:
    """cos/sin split point (half the rope dim)."""
    return self.dims.rope_head_dim // 2

  @property
  def rope_slot(self) -> int:
    """Head-tile holding the rope channels (the last one)."""
    return self.head_tiles - 1

  # --- output record size ----------------------------------------------------
  @property
  def record_bytes(self) -> int:
    """Bytes in one packed output record (old: total_bytes_out)."""
    if not self.dims.is_quantized:
      return self.dims.head_dim * 2  # bf16
    # fp8 payload + scale + padding, capped to a fixed record width.
    return 256 if self.dims.head_dim == LANE else 512

  @property
  def record_rows(self) -> int:
    """Packed physical rows per output record."""
    return pl.cdiv(self.record_bytes, self.row_size_bytes)

  # --- HBM storage packing ---------------------------------------------------
  @property
  def row_size_bytes(self) -> int:
    """Size of one physical HBM row in bytes."""
    return row_size_bytes(self.dims.mode)

  @property
  def hbm_pack(self) -> int:
    """Sub-slots physically packed per slot row, <= SLOT_PACK."""
    return SLOT_PACK

  @property
  def last_dim_size(self) -> int:
    """Size of the last HBM cache dimension in bytes."""
    return last_dim_size(self.dims.mode)

  @property
  def cache_last_dims(self) -> tuple[int, ...]:
    return (self.hbm_pack, self.last_dim_size)

  # --- slot translation helpers -----------------------------------------
  @property
  def tokens_in_second_minor(self) -> int:
    """Divisor to map wrapper row offset to new physical row offset."""
    tokens_per_row = self.row_size_bytes // self.record_bytes
    return max(1, tokens_per_row)

  @property
  def kv_stride(self) -> int:
    """Token stride in logical slot units."""
    if self.record_bytes <= self.row_size_bytes:
      return 1
    return self.record_rows

  # --- block sizes and physical page size ------------------------------------
  @property
  def physical_page_size(self) -> int:
    """Number of physical HBM rows per page of the compressed-KV array."""
    return self.dims.physical_page_size

  @property
  def state_physical_page_size(self) -> int:
    """Number of physical HBM rows per page of the *state* array."""
    return self.dims.state_physical_page_size

  @property
  def state_rows_per_token(self) -> int:
    """Number of physical HBM rows occupied by one token's full state.

    The state is kv + score.
    """
    return state_rows_per_token(
        self.dims.mode, self.dims.head_dim, self.dims.overlap
    )

  @property
  def field_rows(self) -> int:
    """Number of physical HBM rows spanned by one field during gather."""
    field_bytes = self.dims.head_dim * FP32_BYTES
    return pl.cdiv(field_bytes, self.row_size_bytes)

  @property
  def state_page_capacity(self) -> int:
    """Number of token states one page of the state array physically holds."""
    return self.state_physical_page_size // self.state_rows_per_token

  @property
  def state_block_size(self) -> int:
    """Number of state tokens per page of the state array."""
    return self.dims.state_block_size

  @property
  def kv_block_size(self) -> int:
    """Number of compressed KV tokens per page."""
    page_bytes = self.physical_page_size * self.row_size_bytes
    return page_bytes // self.record_bytes

  @property
  def rope_page_size(self) -> int:
    """Number of physical HBM rows per page in RoPE cache."""
    return self.physical_page_size // self.hbm_pack

  @property
  def pages_to_buffer_per_token(self) -> int:
    """Pages that must be resident to cover one token's window (+1 guard)."""
    return pl.cdiv(self.window, self.state_block_size) + 1

  # --- shapes (single source of truth for every reshape / BlockSpec) ---------
  @property
  def _tile_n(self) -> int:
    return self.tile_sizes.tile_n

  def window_shape(self) -> tuple[int, ...]:
    """f32 window scratch: (fields, tile, window, head_tiles, lane)."""
    return (N_FIELDS, self._tile_n, self.window, self.head_tiles, LANE)

  def window_bytes_shape(self) -> tuple[int, ...]:
    """uint8 view of the window scratch; each f32 lane -> FP32_BYTES rows.

    Note: that trailing FP32_BYTES (4) is bytes-per-f32, NOT the HBM SLOT_PACK
    (also 4) -- they're numerically equal but mean different things.

    Returns:
      The shape of the window bytes scratch.
    """
    return (
        N_FIELDS,
        self._tile_n,
        self.window,
        self.head_tiles,
        FP32_BYTES,
        LANE,
    )

  def output_shape(self) -> tuple[int, ...]:
    """Packed output tile / VMEM block: (tile, record_rows) + cache_last_dims."""
    return (self._tile_n, self.record_rows) + self.cache_last_dims

  def page_buffer_shape(self) -> tuple[int, ...]:
    """Page-buffer VMEM block: (tile, pages, state page rows) + cache_last_dims.

    Pages are DMA'd whole out of the *state* array, so this is sized by
    `state_physical_page_size`, not by the compressed-KV page.

    Returns:
      The shape of the page buffer.
    """
    return (
        self._tile_n,
        self.pages_to_buffer_per_token,
        self.state_physical_page_size,
    ) + self.cache_last_dims

  def rope_output_shape(self) -> tuple[int, ...]:
    """RoPE output VMEM block: (tile, 4, lane)."""
    assert self.dims.has_rope_cache
    return (self._tile_n, 4, LANE)

  def cos_sin_shape(self) -> tuple[int, ...]:
    """cos/sin VMEM block: (tile, rope_head_dim)."""
    return (self._tile_n, self.dims.rope_head_dim)

  def cache_shape(self, num_pages: int) -> tuple[int, ...]:
    """Shape of the global HBM KV cache, as uint8 slabs.

    CSA's arrays are allocated as `(num_pages, rows, 128)` int32
    (`csa_cache_layout`), the same bytes as this shape.

    Args:
      num_pages: The number of pages.

    Returns:
      The cache shape.
    """
    return (num_pages, self.physical_page_size) + self.cache_last_dims

  def state_cache_shape(self, num_pages: int) -> tuple[int, ...]:
    """Shape of the global HBM array hosting the f32 compressor state."""
    return (num_pages, self.state_physical_page_size) + self.cache_last_dims

  def rope_cache_shape(self, num_pages: int) -> tuple[int, ...]:
    """Shape of the global HBM RoPE cache, as uint8 slabs."""
    assert self.dims.has_rope_cache
    return (num_pages, self.rope_page_size, 4, 128)
