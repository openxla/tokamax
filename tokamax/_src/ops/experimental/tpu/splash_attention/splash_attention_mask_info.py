# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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

"""Mini-mask creation library."""

import collections
from collections.abc import Callable
import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib

# mypy: ignore-errors

lax = jax.lax
MaskCallable = Any


def find_bounds(
    arr: jax.Array | np.ndarray,
) -> tuple[jax.Array | np.ndarray | None, jax.Array | np.ndarray | None]:
  # Find the first and last block of a row to determine when to initialize/store
  # the output.

  if arr is None:
    return None, None

  bounds_start = (arr != jnp.roll(arr, shift=1, axis=-1)).astype(jnp.int32)
  bounds_end = (arr != jnp.roll(arr, shift=-1, axis=-1)).astype(jnp.int32)
  bounds_start = bounds_start.at[0].set(1)
  bounds_end = bounds_end.at[-1].set(1)

  return bounds_start, bounds_end


# Logic for processing NumPy masks for kernels
class MaskInfo(NamedTuple):
  """Contains runtime masking information for the Splash attention kernel.

  The arrays, mask_next and block_mask are placed in TPU
  scalar-memory. This is a scarse resource so the mask creation logic attempts
  to shrink the data-type of these arrays to the smallest possible one.
  This can be: np.int32, np.int16 or np.int8.

  Attributes:
    mask_next: An integer[num_active_blocks] NumPy array where each entry
      contains the next mask block index in `partial_mask_blocks` to prefetch.
    active_rows: An integer[num_active_blocks] NumPy array where each entry
      contains the row index of the corresponding active block in the original
      mask.
    active_cols: An integer[num_active_blocks] NumPy array where each entry
      contains the column index of the corresponding active block in the
      original mask.
    block_mask: An integer[num_active_blocks] NumPy array where each entry is 0,
      1 or 2. 1 means the corresponding block is partially masked (and so the
      mask in `partial_mask_blocks` must be applied), and 2 means the
      corresponding block is full (no mask needs to be applied). 0 means the
      block is entirely masked out; it only appears in a dense (uncompacted)
      `block_mask`, or as a block that is scheduled purely so that its row's
      output gets written.
    num_active_blocks: An integer[1] NumPy array containing the number of
      leading entries of the compacted arrays above that are valid. The
      remaining entries are padding.
    partial_mask_blocks: An int8[num_partial_blocks, block_q, block_kv] NumPy
      array that contains the blocks of the original mask that contained both
      zeros and ones. The entries in `mask_next` point to indices in the first
      axis of this array.
    q_sequence: A i32[q_sequence_length] NumPy array. When using causal masking,
      this contains the list of indices that correspond to q tokens. For plain
      causal this is just np.arange(q_sequence_length).
  """

  mask_next: np.ndarray | jax.Array | None
  active_rows: np.ndarray | jax.Array | None
  active_cols: np.ndarray | jax.Array | None
  block_mask: np.ndarray | jax.Array | None
  num_active_blocks: np.ndarray | jax.Array | None
  partial_mask_blocks: np.ndarray | jax.Array | None
  q_sequence: np.ndarray | None


def _downcast_to_small_type(array: np.ndarray) -> np.ndarray:
  """Downcast numpy array.

  If possible, downcast the data-type of the input array to the smallest numpy
  type (among np.int16 and np.int8) that fits the content of the array.

  Args:
    array: the array to downcast

  Returns:
    The downcasted array.

  Raises:
    ValueError: if the input array is not np.int32 or if its elements are not
    all positive.
  """
  if array.dtype != np.int32:
    raise ValueError(f'Expected int32 input, but got {array.dtype}.')

  if not np.all(array >= -1):
    # Allow -1 for padding.
    raise ValueError('Expected non-negative array.')

  if array.size == 0:
    return array

  max_value = np.max(array)

  if max_value <= np.iinfo(np.int8).max:
    return array.astype(np.int8)
  elif max_value <= np.iinfo(np.int16).max:
    return array.astype(np.int16)
  else:
    return array.astype(np.int32)


def _check_mask(mask: mask_lib.Mask) -> None:
  """Check that the given mask is valid.

  A row of all zeros along the kv dimension would result in a division by zero
  when computing the softmax. This function is meant to protect against that
  case.

  Args:
    mask: the mask to check.

  Raises:
    ValueError: the mask is invalid.
  """

  assert len(mask.shape) == 2

  exception_message = (
      'Some rows of the mask (along the kv dimension) are all zeros.\nThis is'
      ' would result in a division by zero when computing the attention'
      ' softmax.'
  )

  is_row_non_zero = np.zeros(mask.shape[0], dtype=np.bool_)
  for col in range(mask.shape[1]):
    # Mask only supports slice indices.
    is_row_non_zero = np.logical_or(
        is_row_non_zero,
        mask[(slice(0, mask.shape[0]), slice(col, col + 1))][:, 0],
    )
  if not is_row_non_zero.all():
    raise ValueError(exception_message)


class _HashableNDArray:
  """Helper to make a numpy array hashable: can be added associative containers.

  Attributes:
    array: The underlying numpy array.
  """

  __slots__ = ('array', '_hash')
  array: np.ndarray

  def __init__(self, array: np.ndarray):
    self.array = array
    self._hash = hash(array.tobytes())

  def __hash__(self):
    return self._hash

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, _HashableNDArray):
      return NotImplemented
    return np.array_equal(self.array, other.array, equal_nan=True)


def _generate_shard_metadata(
    block_mask: np.ndarray,
    partial_blocks: np.ndarray,
    is_dkv: bool,
    return_dynamic_grid: bool,
):
  if is_dkv:
    block_mask = block_mask.mT
    partial_blocks = partial_blocks.mT

  if return_dynamic_grid:
    active_mask = block_mask > 0
    if is_dkv:
      # If an entire row is masked then that kv output tile won't be visited.
      # We extend the grid to visit these tiles to initialize them.
      active_mask[:, 0] |= ~active_mask.any(axis=1)
    active_indices = np.argwhere(active_mask)
    active_rows = active_indices[:, 0].astype(np.int32)
    active_cols = active_indices[:, 1].astype(np.int32)
    block_mask = block_mask[active_mask > 0]
    grid_size = active_rows.size
  else:
    active_indices = np.ndindex(block_mask.shape)
    active_rows = active_cols = grid_size = None

  partial_coords = np.argwhere(partial_blocks != -1)
  if partial_coords.size > 0:
    mask_next = []
    mask_coords_iter = iter([tuple(c) for c in partial_coords])
    first_m = coord_m = next(mask_coords_iter)

    for idx in active_indices:
      is_next_mask = tuple(idx) > tuple(coord_m)
      if is_next_mask:
        try:
          coord_m = next(mask_coords_iter)  # type: ignore
        except StopIteration:
          coord_m = first_m
      mask_next.append(partial_blocks[coord_m])
  else:
    mask_next = np.full(block_mask.size, -1, dtype=np.int32)

  mask_next = np.array(mask_next, dtype=np.int32)
  flat_block_mask = block_mask.flatten()

  return active_rows, active_cols, mask_next, flat_block_mask, grid_size


def _process_dynamic_mask(
    mask: jax.Array,
    block_shape: tuple[int, int],
    is_dkv: bool,
    *,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
) -> MaskInfo:
  """Process a dynamic mask to compute it's local sparsity data.

  Note that this operates on a single shard of the mask.

  Args:
    mask: [q_seq_len, kv_seq_len] jax.Array representing a dense mask to
      process.
    block_shape: A Tuple[int, int] representing the shape of the Pallas grid
      block.
    is_dkv: True if we are processing the dKV mask
    downcast_smem_data: If True, downcast the scalar-memory data of MaskInfo to
      a data type smaller than np.int32 (if possible).

  Returns:
    `MaskInfo`, a sparse representation of the dense mask.

  Raises:
    ValueError: if the input mask is invalid or the block sizes are not
    compatible with the mask sizes.
  """
  if len(mask.shape) != 2:
    raise ValueError(f'Expected a 2-dim mask, instead got: {mask.shape}.')

  q_seq_len, kv_seq_len = mask.shape
  q_block_size, kv_block_size = block_shape
  q_blocks_count, q_mod = divmod(q_seq_len, q_block_size)
  kv_blocks_count, kv_mod = divmod(kv_seq_len, kv_block_size)

  if q_mod != 0:
    raise ValueError(f'{q_block_size=} should divide {q_seq_len=}.')
  if kv_mod != 0:
    raise ValueError(f'{kv_block_size=} should divide {kv_seq_len=}.')

  # Tile the last 2 dimensions of the mask into 2D tiles of size `block_shape`.
  mask_blocks = (
      mask.reshape(
          q_blocks_count,
          q_block_size,
          kv_blocks_count,
          kv_block_size,
      )
      .swapaxes(-2, -3)
      .astype(partial_mask_blocks_dtype)
  )

  any_mask = jnp.any(mask_blocks, axis=(-1, -2)).astype(np.int32)
  all_mask = jnp.all(mask_blocks, axis=(-1, -2)).astype(np.int32)
  block_mask = any_mask + all_mask

  block_ids = jnp.arange(block_mask.size, dtype=np.int32).reshape(
      block_mask.shape
  )
  if is_dkv:
    block_mask = block_mask.swapaxes(-1, -2)
    block_ids = block_ids.swapaxes(-1, -2)
    mask_blocks = mask_blocks.swapaxes(-1, -2)

  active_mask = block_mask > 0
  if is_dkv:
    # If an entire row is masked then that kv output tile won't be visited.
    # We extend the grid to visit these tiles to initialize them.
    empty_rows = jnp.all(block_mask == 0, axis=-1)
    first_col = jnp.arange(block_mask.shape[1]) == 0
    active_mask |= (empty_rows[:, None] & first_col)

  num_active_blocks = active_mask.flatten().sum(keepdims=True)
  active_indices = jnp.argwhere(
      active_mask, size=active_mask.size, fill_value=-1
  )
  active_rows = active_indices[:, 0].astype(np.int32)
  active_cols = active_indices[:, 1].astype(np.int32)

  block_mask = block_mask[active_rows, active_cols]
  mask_next = block_ids.at[active_rows, active_cols].get(
      wrap_negative_indices=False
  )
  mask_next = jnp.where(block_mask == 1, mask_next, 0)

  # Mask out the blocks that aren't active.
  mask = (jnp.arange(block_mask.size) < num_active_blocks).astype(np.int32)
  block_mask = block_mask * mask

  # Collapsing because the block ids are linearized.
  mask_blocks = lax.collapse(mask_blocks, 0, 2)

  def _downcast(array: jax.Array, max_value: int) -> jax.Array:
    if array.size == 0:
      return array

    if array.dtype != np.int32:
      raise ValueError(f'Expected int32 input, but got {array.dtype}.')

    if max_value <= np.iinfo(np.int8).max:
      return array.astype(np.int8)
    elif max_value <= np.iinfo(np.int16).max:
      return array.astype(np.int16)
    else:
      return array.astype(np.int32)

  if downcast_smem_data:
    block_mask = block_mask.astype(np.int8)  # values are in the range [0, 1, 2]
    mask_next = _downcast(mask_next, q_blocks_count * kv_blocks_count)

  return MaskInfo(
      mask_next=mask_next,
      active_rows=active_rows,
      active_cols=active_cols,
      block_mask=block_mask,
      num_active_blocks=num_active_blocks,
      partial_mask_blocks=mask_blocks,
      q_sequence=None,
  )


def _downcast_jax(array: jax.Array, max_value: int) -> jax.Array:
  """Downcast a int32 jax.Array to the smallest signed integer type that fits max_value."""
  if array.size == 0:
    return array
  if max_value <= np.iinfo(np.int8).max:
    return array.astype(jnp.int8)
  elif max_value <= np.iinfo(np.int16).max:
    return array.astype(jnp.int16)
  else:
    return array.astype(jnp.int32)


def _onehot_lookup(table: jax.Array, idx: jax.Array) -> jax.Array:
  """Returns `table[idx]` for in-range `idx` without emitting an XLA gather."""
  onehot = idx[..., None] == jnp.arange(table.shape[-1], dtype=idx.dtype)
  return jnp.sum(jnp.where(onehot, table, jnp.zeros_like(table)), axis=-1)


def refine_mask_info_with_segments(
    mask_info: MaskInfo,
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    block_shape: tuple[int, int],
    bkv_compute: int,
    is_dkv: bool,
    *,
    unvmap_any: Callable[[jax.Array], jax.Array] = lambda x: x,
    unvmap_all: Callable[[jax.Array], jax.Array] = lambda x: x,
) -> MaskInfo:
  """Refines MaskInfo using runtime segment_ids to skip cross-segment tiles.

  Block mask encoding:
    - 0: inactive block (skipped).
    - Low 2 bits (bm & 3) when > 0:
      - 1: partial 2D mask (and potentially partial segment mask).
      - 2: full 2D mask and uniform single segment across the entire block.
      - 3: full 2D mask and partial segment mask.
    - Upper bits (bm >> 2) when bkv > bkv_compute and (bm & 3) in (1, 3):
      - Bit (t + 2) is 1 if KV compute sub-tile `t` overlaps with the Q block's
        segments, and 0 otherwise.

  Args:
    mask_info: The static or dynamic 2D MaskInfo for the current shard.
    q_segment_ids: 1D jax.Array of shape [q_seq_len].
    kv_segment_ids: 1D jax.Array of shape [kv_seq_len].
    block_shape: (block_q, block_kv) tuple.
    bkv_compute: Sub-tile compute size along KV.
    is_dkv: True if refining the backward dKV MaskInfo.
    unvmap_any: Optional callable to reduce boolean overlap masks across any
      active `vmap` batch dimensions with `any`.
    unvmap_all: Optional callable to reduce boolean uniformity masks across any
      active `vmap` batch dimensions with `all`.

  Returns:
    A refined MaskInfo with cross-segment blocks removed from active_rows and
    active_cols.
  """
  q_seq_len = q_segment_ids.shape[-1]
  kv_seq_len = kv_segment_ids.shape[-1]
  q_block_size, kv_block_size = block_shape
  q_steps = q_seq_len // q_block_size
  kv_steps = kv_seq_len // kv_block_size
  num_iters = kv_block_size // bkv_compute

  q_seg_blocks = q_segment_ids.reshape(
      *q_segment_ids.shape[:-1], q_steps, q_block_size
  )
  q_min = jnp.min(q_seg_blocks, axis=-1)  # [..., q_steps]
  q_max = jnp.max(q_seg_blocks, axis=-1)  # [..., q_steps]

  kv_seg_subblocks = kv_segment_ids.reshape(
      *kv_segment_ids.shape[:-1], kv_steps, num_iters, bkv_compute
  )
  kv_sub_min = jnp.min(kv_seg_subblocks, axis=-1)  # [..., kv_steps, num_iters]
  kv_sub_max = jnp.max(kv_seg_subblocks, axis=-1)  # [..., kv_steps, num_iters]
  kv_min = jnp.min(kv_sub_min, axis=-1)  # [..., kv_steps]
  kv_max = jnp.max(kv_sub_max, axis=-1)  # [..., kv_steps]

  num_rows, num_cols = (kv_steps, q_steps) if is_dkv else (q_steps, kv_steps)
  if mask_info.active_rows is None:
    m_size = num_rows * num_cols
    rows, cols = jnp.unravel_index(
        jnp.arange(m_size, dtype=jnp.int32), (num_rows, num_cols)
    )
    num_active = jnp.array([m_size], dtype=jnp.int32)
  else:
    rows = jnp.asarray(mask_info.active_rows, dtype=jnp.int32)
    cols = jnp.asarray(mask_info.active_cols, dtype=jnp.int32)
    m_size = rows.size
    if m_size == 0:
      return mask_info
    num_active = jnp.asarray(mask_info.num_active_blocks, dtype=jnp.int32)

  orig_bm = (
      jnp.full((m_size,), 2, dtype=jnp.int32)
      if mask_info.block_mask is None
      else jnp.asarray(mask_info.block_mask, dtype=jnp.int32).reshape(m_size)
  )
  idx = jnp.arange(m_size, dtype=jnp.int32)
  valid_static = (orig_bm > 0) & (idx < num_active[0])

  rows_c = jnp.clip(rows, 0, num_rows - 1)
  cols_c = jnp.clip(cols, 0, num_cols - 1)
  q_idx, kv_idx = (cols_c, rows_c) if is_dkv else (rows_c, cols_c)

  q_lo = _onehot_lookup(q_min, q_idx)
  q_hi = _onehot_lookup(q_max, q_idx)
  kv_lo = _onehot_lookup(kv_min, kv_idx)
  kv_hi = _onehot_lookup(kv_max, kv_idx)

  seg_uniform = unvmap_all(
      (q_lo == q_hi) & (kv_lo == kv_hi) & (q_lo == kv_lo)
  )
  if num_iters > 1:
    sub_overlap = jnp.stack(
        [
            unvmap_any(
                (q_hi >= _onehot_lookup(kv_sub_min[..., t], kv_idx))
                & (_onehot_lookup(kv_sub_max[..., t], kv_idx) >= q_lo)
            )
            for t in range(num_iters)
        ],
        axis=-1,
    )
    seg_any = jnp.any(sub_overlap, axis=-1)
  else:
    sub_overlap = None
    seg_any = unvmap_any((q_hi >= kv_lo) & (kv_hi >= q_lo))

  active = valid_static & seg_any
  bm_type = jnp.where(
      active,
      jnp.where((orig_bm == 2) & ~seg_uniform, jnp.int32(3), orig_bm),
      jnp.int32(0),
  )
  if num_iters > 1:
    assert sub_overlap is not None
    shifts = jnp.arange(2, num_iters + 2, dtype=jnp.int32)
    sub_bits = jnp.sum(
        sub_overlap.astype(jnp.int32) << shifts[None, :], axis=-1
    )
    new_bm = jnp.where(
        (bm_type == 1) | (bm_type == 3), bm_type | sub_bits, bm_type
    )
  else:
    new_bm = bm_type

  # Every output row needs >= 1 scheduled step (with block_mask=0 if inactive)
  # so `bounds_start`/`bounds_end` still zero-initialize and write that row.
  # Per-tile row lookups go through the `row_eq` one-hot rather than
  # `table[rows_c]`, and compaction below uses `lax.sort` rather than
  # `x[argsort(...)]`: this function emits no XLA gather (see `_onehot_lookup`).
  row_eq = (idx < num_active[0])[:, None] & (
      rows_c[:, None] == jnp.arange(num_rows, dtype=jnp.int32)[None, :]
  )
  row_has_active = jnp.any(row_eq & active[:, None], axis=0)
  first_in_row = jnp.argmax(row_eq, axis=0).astype(jnp.int32)
  keep = active | (
      (idx < num_active[0])
      & (is_dkv | jnp.any(active))
      & ~jnp.any(row_eq & row_has_active[None, :], axis=1)
      & (idx == jnp.sum(jnp.where(row_eq, first_in_row[None, :], 0), axis=1))
  )

  mask_next = (
      None
      if mask_info.mask_next is None
      else jnp.asarray(mask_info.mask_next).reshape(m_size)
  )

  # A stable sort on `~keep` moves kept tiles to the front in schedule order.
  new_num_active = jnp.sum(keep.astype(jnp.int32), keepdims=True)
  payload = (rows_c, cols_c, new_bm) + (
      () if mask_next is None else (mask_next,)
  )
  _, *payload = lax.sort(
      ((~keep).astype(jnp.int32), *payload), num_keys=1, is_stable=True
  )
  is_active_slot = idx < new_num_active[0]
  rows_s = jnp.where(is_active_slot, payload[0], -1)
  cols_s = jnp.where(is_active_slot, payload[1], -1)
  block_mask_s = jnp.where(is_active_slot, payload[2], 0)
  mask_next_s = (
      jnp.where(is_active_slot, payload[3], -1).astype(mask_next.dtype)
      if mask_next is not None
      else None
  )

  downcast = (
      mask_info.block_mask is None or mask_info.block_mask.dtype != jnp.int32
  )
  if downcast:
    active_rows = (
        _downcast_jax(rows_s, num_rows)
        if mask_info.active_rows is None
        else rows_s.astype(mask_info.active_rows.dtype)
    )
    active_cols = (
        _downcast_jax(cols_s, num_cols)
        if mask_info.active_cols is None
        else cols_s.astype(mask_info.active_cols.dtype)
    )
    max_bm_val = (1 << (num_iters + 2)) - 1 if num_iters > 1 else 3
    block_mask = _downcast_jax(block_mask_s, max_bm_val)
  else:
    active_rows = rows_s.astype(jnp.int32)
    active_cols = cols_s.astype(jnp.int32)
    block_mask = block_mask_s.astype(jnp.int32)

  return MaskInfo(
      mask_next=mask_next_s,
      active_rows=active_rows,
      active_cols=active_cols,
      block_mask=block_mask,
      num_active_blocks=new_num_active,
      partial_mask_blocks=mask_info.partial_mask_blocks,
      q_sequence=mask_info.q_sequence,
  )


# When used in a transformer network with multiple layers, the SplashAttention
# kernel is created several times with the same mask. Cache MaskInfo to avoid
# blowing up compile times. Ideally the size of the cache should be determined
# by the client.
@functools.lru_cache(maxsize=12)
def _process_mask(
    mask: mask_lib.Mask,  # [q_seq_len, kv_seq_len]
    block_shape: tuple[int, int],
    is_dkv: bool,
    *,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
    q_seq_shards: int = 1,
    kv_seq_shards: int = 1,
    return_dynamic_grid: bool = True,
) -> tuple[MaskInfo, MaskCallable | None]:
  """Transform a dense mask into a sparse representation.

  The number Q sequence shards are needed to create a MaskInfo
  object that is partitionable (with shard_map) along that dimension.
  Args:
    mask: Dense mask to process.
    block_shape: Shape of the Pallas grid block.
    is_dkv: True if we are processing the dKV mask
    downcast_smem_data: If True, downcast the SMEM data of MaskInfo to a data
      type smaller if possible.
    q_seq_shards: Number of Q sequence shards of the mesh in which the kernel is
      launched.

  Returns:
    `MaskInfo`, a sparse representation of the dense mask.
    `MaskCallable`: a callable that, given Q and KV indices, returns
      the value of the mask at those coordinates.

  Raises:
    ValueError: if the input mask is invalid or the block sizes are not
    compatible with the mask sizes.
  """

  if len(mask.shape) != 2:
    raise ValueError(f'Expected a 2-dim mask, instead got: {mask.shape=}')

  q_seq_len, kv_seq_len = mask.shape
  q_block_size, kv_block_size = block_shape
  q_blocks_count, q_mod = divmod(q_seq_len, q_block_size)
  kv_blocks_count, kv_mod = divmod(kv_seq_len, kv_block_size)

  if q_mod != 0:
    raise ValueError(f'{q_block_size=} should divide {q_seq_len=}.')
  if kv_mod != 0:
    raise ValueError(f'{kv_block_size=} should divide {kv_seq_len=}.')

  q_seq_len_per_shard, mod = divmod(q_seq_len, q_seq_shards)
  if mod != 0:
    raise ValueError(f'{q_seq_shards=} should divide {q_seq_len=}.')

  q_blocks_per_shard, mod = divmod(q_seq_len_per_shard, q_block_size)
  if mod != 0:
    raise ValueError(f'{q_block_size=} should divide {q_seq_len_per_shard=}.')

  kv_seq_len_per_shard, mod = divmod(kv_seq_len, kv_seq_shards)
  if mod != 0:
    raise ValueError(f'{kv_seq_shards=} should divide {kv_seq_len=}.')

  kv_blocks_per_shard, mod = divmod(kv_seq_len_per_shard, kv_block_size)
  if mod != 0:
    raise ValueError(f'{kv_block_size=} should divide {kv_seq_len_per_shard=}.')

  # TODO: checking the validity of the masks is slow for large masks.
  # Disable it for now, reevaluate in the future.

  # The mask object either define q_sequence and mask_function or none of
  # them.
  assert hasattr(mask, 'q_sequence') == hasattr(mask, 'mask_function')

  # If the mask object defines a q_sequence and a mask_function, then make use
  # of these in the kernel rather. This is preferable over loading the mask
  # from memory. When using a mask_function, then mask_next and
  # partial_mask_blocks are left undefined and not used in the kernel.
  if hasattr(mask, 'q_sequence') and hasattr(mask, 'mask_function'):
    q_sequence = mask.q_sequence
    mask_function = mask.mask_function
  else:
    q_sequence = mask_function = None

  # Identify the partial mask blocks and the value of the block mask for each
  # block.
  # Partial mask blocks are uniquified. When partitioning, all partial mask
  # blocks are replicated across shards.

  blocked_shape = (q_blocks_count, kv_blocks_count)
  state_grid = np.zeros(blocked_shape, dtype=np.int32)
  partial_id_grid = np.full(blocked_shape, -1, dtype=np.int32)

  partial_blocks_map = collections.defaultdict(lambda: len(partial_blocks_map))
  unique_chunks = []

  # Partition the dense mask into blocks and categorize them:
  # 0 = Empty, 1 = Partial (mixed 0s and 1s), 2 = Full (all 1s).
  # Partial blocks are deduplicated and stored in unique_chunks to save memory.
  for coords in np.ndindex((q_blocks_count, kv_blocks_count)):
    (q_idx, kv_idx) = coords
    chunk = mask[(
        slice(q_idx * q_block_size, (q_idx + 1) * q_block_size),
        slice(kv_idx * kv_block_size, (kv_idx + 1) * kv_block_size),
    )]
    if chunk.any():
      if chunk.all():
        state_grid[q_idx, kv_idx] = 2
      else:
        state_grid[q_idx, kv_idx] = 1
        chunk_id = partial_blocks_map[_HashableNDArray(chunk)]
        partial_id_grid[q_idx, kv_idx] = chunk_id

        if chunk_id == len(unique_chunks):
          unique_chunks.append(chunk)

  full_mask = (state_grid == 2).all()
  if full_mask:
    return MaskInfo(
        mask_next=None,
        active_rows=None,
        active_cols=None,
        block_mask=None,
        num_active_blocks=None,
        partial_mask_blocks=None,
        q_sequence=q_sequence,
    ), None

  if unique_chunks:
    partial_mask_blocks = np.stack(unique_chunks).astype(
        partial_mask_blocks_dtype
    )
    if is_dkv:
      partial_mask_blocks = partial_mask_blocks.mT
  else:
    partial_mask_blocks = None

  # Work on a fraction of the mask at the time to compute the mask. This is
  # needed to compute the correct data indices, which are relative to the
  # current slice of the mask.
  all_shards_metadata = []
  for q_shard_idx in range(q_seq_shards):
    for kv_shard_idx in range(kv_seq_shards):
      q_slice = slice(
          q_shard_idx * q_blocks_per_shard,
          (q_shard_idx + 1) * q_blocks_per_shard,
      )
      kv_slice = slice(
          kv_shard_idx * kv_blocks_per_shard,
          (kv_shard_idx + 1) * kv_blocks_per_shard,
      )
      metadata = _generate_shard_metadata(
          state_grid[q_slice, kv_slice],
          partial_id_grid[q_slice, kv_slice],
          is_dkv,
          return_dynamic_grid,
      )
      all_shards_metadata.append(metadata)

  (
      active_rows_slices,
      active_cols_slices,
      mask_next_slices,
      block_mask_slices,
      num_active_blocks,
  ) = zip(*all_shards_metadata)

  if return_dynamic_grid:
    # Pad each slice to the largest number of active blocks in any shard.
    max_size = max(num_active_blocks)
    pad_slice = lambda arr: np.pad(
        arr, (0, max_size - arr.shape[0]), mode='constant', constant_values=-1
    )
    active_rows_slices = list(map(pad_slice, active_rows_slices))
    active_cols_slices = list(map(pad_slice, active_cols_slices))
    mask_next_slices = list(map(pad_slice, mask_next_slices))
    block_mask_slices = list(map(pad_slice, block_mask_slices))

    # Concatenate the sequence shards.
    active_rows = np.concatenate(active_rows_slices, axis=0)
    active_cols = np.concatenate(active_cols_slices, axis=0)
    num_active_blocks = np.array(num_active_blocks, dtype=np.int32)

    if downcast_smem_data:
      active_rows = _downcast_to_small_type(active_rows)
      active_cols = _downcast_to_small_type(active_cols)
  else:
    active_rows = active_cols = num_active_blocks = None

  mask_next = np.concatenate(mask_next_slices, axis=0)
  block_mask = np.concatenate(block_mask_slices, axis=0)

  if downcast_smem_data:
    mask_next = _downcast_to_small_type(mask_next)
    block_mask = _downcast_to_small_type(block_mask)

  if partial_mask_blocks is None:
    mask_next = None

  assert (mask_function is not None) == (q_sequence is not None)
  # When the mask can be computed inside the kernel with a mask_function,
  # there is no need to load it from memory. So mask_next and
  # partial_mask_blocks are unused.
  return (
      MaskInfo(
          mask_next=mask_next if mask_function is None else None,
          active_rows=active_rows,
          active_cols=active_cols,
          block_mask=block_mask,
          num_active_blocks=num_active_blocks,
          partial_mask_blocks=partial_mask_blocks
          if mask_function is None
          else None,
          q_sequence=q_sequence,
      ),
      mask_function,
  )


process_mask = functools.partial(_process_mask, is_dkv=False)
process_mask_dkv = functools.partial(_process_mask, is_dkv=True)

process_dynamic_mask = functools.partial(_process_dynamic_mask, is_dkv=False)
process_dynamic_mask_dkv = functools.partial(_process_dynamic_mask, is_dkv=True)
