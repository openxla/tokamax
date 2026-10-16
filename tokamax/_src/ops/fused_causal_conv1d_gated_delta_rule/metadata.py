# Copyright 2026 Google LLC
# Copyright 2026 Rabdos AI
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
"""Schedule metadata and validation for fused GDN."""

from typing import Any

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config
from tokamax._src.ops.causal_conv1d_gated_delta_rule import memory_ref

# Shorten the first n8v24 prefill tile only at or above this request length.
_LONG_PREFILL_MIN_TOKENS = 4096


def metadata_storage(field: Any) -> Any:
  """Return field.data when present, otherwise field itself, without copying."""
  return getattr(field, "data", field)


def metadata_template(
    cfg: config.GDNConfig,
    requests: int,
    capacity: int,
) -> memory_ref.MetadataRef:
  """Create zero-valued packed metadata for a one-sequence-per-tile schedule.

  cfg.seq_tile_size must be one; requests and capacity nonnegative. records
  holds capacity [r_base, packed_word] int32 pairs. Request fields hold
  [requests] initial-state flags, write slots, read slots and offsets.

  Args:
    cfg: Static GDN configuration defining head layout, tile sizes, and dtypes.
    requests: Number of request entries, including inactive padding.
    capacity: Maximum number of packed prefill tile records.

  Returns:
    Zero-valued MetadataRef storage sized for the requests and tile capacity.
  """
  # get_record(p_id, 0) requires exactly one sequence record per tile.
  if cfg.seq_tile_size != 1:
    raise ValueError(
        f"fused path requires cfg.seq_tile_size == 1, got {cfg.seq_tile_size}"
    )
  # MetadataRef stores interleaved [r_base, packed_word] records.
  fields = (
      ("p_id_to_s_idx", capacity, jnp.int32),
      ("p_id_to_r_base", capacity, jnp.int32),
      ("p_id_to_r_size", capacity, jnp.int32),
      ("p_id_is_first_tile", capacity, jnp.bool_),
      ("p_id_is_last_tile", capacity, jnp.bool_),
      ("s_idx_has_initial_state", requests, jnp.bool_),
      ("s_idx_to_state_indices", requests, jnp.int32),
      # Transfers receive read addresses separately; these fields are placeholders.
      ("s_idx_to_read_offset", requests, jnp.int32),
      ("s_idx_to_read_indices", requests, jnp.int32),
  )
  metadata = memory_ref.MetadataRef.create(
      cfgs=cfg,
      num_tiles=jnp.zeros((), dtype=jnp.int32),
      **{
          name: jnp.zeros((extent,), dtype=dtype)
          for name, extent, dtype in fields
      },
  )
  records = metadata_storage(metadata.records)
  needed = capacity * memory_ref.PackedPIdRecord.STRUCT_SIZE
  if len(records.shape) != 1 or records.shape[0] < needed:
    raise ValueError(
        f"Unsupported packed metadata storage: {records.shape}, "
        f"needs at least {needed} scalars for {capacity} tiles"
    )
  for name in ("s_idx_has_initial_state", "s_idx_to_state_indices"):
    data = metadata_storage(getattr(metadata, name))
    if len(data.shape) != 1 or data.shape[0] < requests:
      raise ValueError(
          f"Unsupported scalar metadata storage for {name}: {data.shape}"
      )
  return metadata


def _zero_smem(ref: Any) -> None:
  """Zero an exclusively owned scalar or rank-one int32/bool SMEM ref.

  Args:
    ref: Exclusively owned scalar or rank-one SMEM ref to clear.
  """
  zero = jnp.asarray(0, dtype=ref.dtype)
  if not ref.shape:
    ref[...] = zero
    return
  if len(ref.shape) != 1:
    raise ValueError("Schedule SMEM storage must be scalar or rank one")

  def clear(index: jax.Array, unused: None) -> None:
    """Zero one element of the metadata vector."""
    ref[index] = zero
    return unused

  jax.lax.fori_loop(0, ref.shape[0], clear, None)


def _store_metadata_scalar(
    field: Any,
    index: jax.Array,
    value: jax.Array,
) -> None:
  """Store a scalar, casting to field dtype; index must be in bounds."""
  data = metadata_storage(field)
  data[index] = jnp.asarray(value, dtype=data.dtype)


def _store_record(
    metadata_ref: Any,
    p_id: jax.Array,
    s_idx: jax.Array,
    r_base: jax.Array,
    r_size: jax.Array,
    is_first_tile: jax.Array,
    is_last_tile: jax.Array,
) -> None:
  """Write [r_base, packed_word], matching MetadataRef.get_record's layout.

  Pack s_idx, r_size and tile flags with PackedPIdRecord.pack's shifts/masks.
  Clamp negative r_size to zero so its sign bits cannot overwrite s_idx.

  Args:
    metadata_ref: Packed tile/request metadata refs holding ownership, flags,
      and cache slots.
    p_id: Zero-based packed prefill tile index.
    s_idx: Request index to encode as the tile owner.
    r_base: First logical token row covered by the packed tile record.
    r_size: Number of valid token rows in the record; negative values clamp to
      zero.
    is_first_tile: Whether the record begins its request.
    is_last_tile: Whether the record finishes its request.
  """
  rec = memory_ref.PackedPIdRecord
  strides = pl.strides_from_shape(metadata_ref.shape)
  pos = (strides[0] * p_id + strides[1] * 0) * rec.STRUCT_SIZE
  word = jnp.int32(s_idx) << rec.S_IDX_SHIFT
  word |= jnp.maximum(jnp.int32(r_size), 0) << rec.R_SIZE_SHIFT
  word |= jnp.int32(is_last_tile) << rec.LAST_TILE_SHIFT
  word |= jnp.int32(is_first_tile) << rec.FIRST_TILE_SHIFT
  records = metadata_storage(metadata_ref.records)
  records[pos] = jnp.int32(r_base)
  records[pos + 1] = word


def active_request_count(distribution_ref: Any, requests: int) -> jax.Array:
  """Clamp mixed_end to [0, requests] before SMEM indexing.

  Upstream permits oversized endpoints, e.g. [0, 0, 3] for one prefill.
  Schedule validation rejects a zero active count.

  Args:
    distribution_ref: SMEM endpoints for decode, prefill, and active request
      prefixes.
    requests: Number of request entries, including inactive padding.

  Returns:
    Int32 scalar active endpoint clamped to [0, requests].
  """
  return jnp.clip(distribution_ref[2], jnp.int32(0), jnp.int32(requests))


def prefill_alignment_shift(
    base: jax.Array, length: jax.Array, n_kq: int, n_v: int
) -> jax.Array:
  """Shorten the first long n8v24 tile to align subsequent input windows.

  Args:
    base: First submitted token's offset in the packed input.
    length: Number of submitted tokens for the request.
    n_kq: Number of query/key heads.
    n_v: Number of value heads; supported layouts group them evenly over Q/K
      heads.

  Returns:
    Initial-tile shortening in [0, 8), or zero when the special alignment rule
    does not apply.
  """
  if (n_kq, n_v) != (8, 24):
    return jnp.int32(0)
  return jnp.where(length >= _LONG_PREFILL_MIN_TOKENS, base % 8, jnp.int32(0))


def prefill_schedule_capacity(
    requests: int, tokens: int, chunk_size: int, n_kq: int, n_v: int
) -> int:
  """Bound complete tiles plus request boundaries and alignment tails.

  Args:
    requests: Number of request entries, including inactive padding.
    tokens: Number of token rows in the bucket or request fixture.
    chunk_size: Number of token rows in one prefill tile.
    n_kq: Number of query/key heads.
    n_v: Number of value heads; supported layouts group them evenly over Q/K
      heads.

  Returns:
    Conservative number of tile records needed for the bucket and request
    boundaries.
  """
  boundaries = 2 if (n_kq, n_v) == (8, 24) else 1
  return boundaries * requests + tokens // chunk_size


def build_smem_schedule(
    query_start_ref: Any,
    state_indices_ref: Any,
    distribution_ref: Any,
    seq_lens_ref: Any,
    metadata_ref: memory_ref.MetadataRef,
    request_prefix_ref: Any,
    occupancy_ref: Any,
    *,
    cfg: config.GDNConfig,
    read_state_indices_ref: Any = None,
    read_offsets_ref: Any = None,
) -> jax.Array:
  """Validate all requests and materialize prefill tiles in SMEM.

  Decode is a contiguous one-token prefix; all active write slots must be
  distinct. Shapes must pass static eligibility; metadata uses one sequence
  per tile. request_prefix_ref holds [requests + 1] exclusive tile counts;
  occupancy_ref counts active writes per cache slot. Supply both read refs
  for explicit addresses, or neither when reads use validated write slots.
  Invalid schedules return False with zero prefill tiles.

  Args:
    query_start_ref: SMEM int32 request token boundaries, with requests + 1
      entries.
    state_indices_ref: SMEM per-request write-cache slots; active owners must be
      distinct.
    distribution_ref: SMEM endpoints for decode, prefill, and active request
      prefixes.
    seq_lens_ref: SMEM total sequence lengths, including previously consumed
      history.
    metadata_ref: Mutable schedule storage; records/flags are installed only for
      valid schedules.
    request_prefix_ref: Mutable SMEM array [requests + 1] of exclusive prefill
      tile counts.
    occupancy_ref: Mutable per-cache-slot counts used to reject duplicate active
      writes.
    cfg: Static GDN configuration defining head layout, tile sizes, and dtypes.
    read_state_indices_ref: Per-request initial-state cache slots, separate from
      write ownership.
    read_offsets_ref: Per-request int32 read-slot offsets; public dispatch zeros
      prefill offsets.

  Returns:
    Boolean schedule validity; invalid schedules leave num_tiles equal to zero.
  """
  requests, slots = state_indices_ref.shape[0], occupancy_ref.shape[0]
  capacity = (
      metadata_storage(metadata_ref.records).shape[0]
      // memory_ref.PackedPIdRecord.STRUCT_SIZE
  )
  tokens, chunk_size = cfg.batch_size, cfg.chunk_size

  # Only occupancy needs clearing: every consumed metadata field is written
  # before use, and unused tile capacity is never read.
  _zero_smem(occupancy_ref)
  request_prefix_ref[0] = jnp.int32(0)
  decode_count = distribution_ref[0]
  active_count = active_request_count(distribution_ref, requests)
  middle_count = jnp.minimum(distribution_ref[1], active_count)
  eligible = (
      (decode_count >= 0)
      & (decode_count <= middle_count)
      # Decode uses the raw endpoint in both pipelines; reject oversize values.
      & (decode_count <= requests)
      & (active_count > 0)
      & (query_start_ref[0] == 0)
      # The last active offset bounds tokens; later offsets may be padding.
      & (query_start_ref[active_count] <= tokens)
  )

  def validate_request(
      seq: jax.Array,
      carry: tuple[jax.Array, jax.Array],
  ) -> tuple[jax.Array, jax.Array]:
    """Extend (validity, tile count), visiting requests in order after clearing
    occupancy.

    Args:
      seq: Zero-based request index being validated.
      carry: Running (schedule validity, cumulative prefill tile count).

    Returns:
      Updated (validity, cumulative prefill tile count) loop carry.
    """
    valid_so_far, total = carry
    left = query_start_ref[seq]
    right = query_start_ref[seq + 1]
    length = right - left
    active, decode, slot = (
        seq < active_count,
        seq < decode_count,
        state_indices_ref[seq],
    )
    # Upstream padding may run backward; only active offsets must be ordered.
    bounds_ok = (left >= 0) & (right <= tokens) & (right >= left)
    offsets_valid = (~active) | bounds_ok
    # Slot zero is reserved and also serves as the inactive occupancy sentinel.
    slot_valid = (slot > 0) & (slot < slots)
    # Check explicit read addresses; default reads use the validated write slot.
    if read_state_indices_ref is None or read_offsets_ref is None:
      read_valid = jnp.bool_(True)
    else:
      read_slot = read_state_indices_ref[seq] + read_offsets_ref[seq]
      read_valid = (read_slot > 0) & (read_slot < slots)
    slot_valid = slot_valid & read_valid
    counted, safe_slot = active & slot_valid, jnp.where(
        active & slot_valid, slot, jnp.int32(0)
    )
    previous_count = occupancy_ref[safe_slot]
    distinct = (~counted) | (previous_count == 0)
    occupancy_ref[safe_slot] = previous_count + counted.astype(jnp.int32)
    # Inactive lengths are padding, not submitted tokens.
    request_valid = jnp.where(
        active,
        slot_valid
        & (seq_lens_ref[seq] >= length)
        & jnp.where(decode, length == 1, length > 0),
        jnp.bool_(True),
    )
    safe_length = jnp.where(active & bounds_ok, length, jnp.int32(0))
    shifted_length = safe_length + prefill_alignment_shift(
        left, safe_length, cfg.num_kq_heads, cfg.num_v_heads
    )
    prefill_tiles = shifted_length // chunk_size + (
        shifted_length % chunk_size != 0
    ).astype(jnp.int32)
    count = jnp.where(active & (~decode), prefill_tiles, jnp.int32(0))
    remaining = jnp.int32(capacity) - total
    count_fits = count <= remaining
    next_total = total + jnp.minimum(count, remaining)
    request_prefix_ref[seq + 1] = next_total
    return (
        valid_so_far & offsets_valid & request_valid & distinct & count_fits,
        next_total,
    )

  eligible, total = jax.lax.fori_loop(
      0, requests, validate_request, (eligible, jnp.int32(0))
  )
  num_tiles = jnp.where(eligible, total, jnp.int32(0))
  metadata_ref.num_tiles[...] = num_tiles

  def materialize_tile(p_id: jax.Array, previous_owner: jax.Array) -> jax.Array:
    """Install a validated tile and return its owner.

    Each prefill owns >= 1 tile; previous_owner is the prior owner, or the
    first prefill request for tile zero.

    Args:
      p_id: Zero-based packed prefill tile index.
      previous_owner: Owner of the preceding tile, or the first prefill owner
        for tile zero.

    Returns:
      Request index owning the newly written tile record.
    """
    owner = jnp.where(
        (p_id > 0) & (p_id >= request_prefix_ref[previous_owner + 1]),
        previous_owner + jnp.int32(1),
        previous_owner,
    )
    first_tile = request_prefix_ref[owner]
    next_request_tile = request_prefix_ref[owner + 1]
    tile_in_request = p_id - first_tile
    is_first = p_id == first_tile
    is_last = p_id == next_request_tile - 1
    start = query_start_ref[owner]
    length = query_start_ref[owner + 1] - start
    shift = prefill_alignment_shift(
        start, length, cfg.num_kq_heads, cfg.num_v_heads
    )
    base = start + tile_in_request * chunk_size - jnp.where(is_first, 0, shift)
    size = jnp.minimum(
        query_start_ref[owner + 1] - base,
        jnp.int32(chunk_size) - jnp.where(is_first, shift, 0),
    )
    _store_record(
        metadata_ref,
        p_id,
        s_idx=owner,
        r_base=base,
        r_size=size,
        is_first_tile=is_first,
        is_last_tile=is_last,
    )

    @pl.when(is_first)
    def install_request() -> None:
      """Install first-tile request metadata, including whether prior state must
      be loaded.
      """
      length = query_start_ref[owner + 1] - query_start_ref[owner]
      fields = (
          metadata_ref.s_idx_has_initial_state,
          metadata_ref.s_idx_to_state_indices,
      )
      for field, value in zip(
          fields, (seq_lens_ref[owner] > length, state_indices_ref[owner])
      ):
        _store_metadata_scalar(field, owner, value)

    return owner

  jax.lax.fori_loop(0, num_tiles, materialize_tile, decode_count)
  return eligible
