# Copyright 2026 Ant Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Ported from primatrix/pallas-kernel dca6a09feec2659f2777a6db02f2ee55560f991e.

"""Device-side primitives for compact packed KDA chunks.

The production packed mega kernel currently keeps token inputs compact but
executes a sequence-aligned chunk grid.  A compact grid may contain multiple
sequence fragments in one chunk, so its scan and pairwise operations must be
segmented.  The helpers in this module use fixed-shape JAX operations and are
safe to trace from a Pallas kernel; they do not introduce a host-side loop over
the dynamic number of fragments.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from tokamax._src.ops.kda.utils import get_interpret


class CompactChunkMetadata(NamedTuple):
  """Fixed-shape metadata shared by compact forward and backward chunks."""

  segment_ids: jax.Array
  previous_segment_ids: jax.Array
  next_segment_ids: jax.Array
  same_segment_causal: jax.Array
  reverse_continuation: jax.Array
  continues_from_previous: jax.Array
  continues_to_next: jax.Array


COMPACT_CHUNK_FULL = 0
COMPACT_CHUNK_TWO_FRAGMENT = 1
COMPACT_CHUNK_GENERAL = 2


def mask_reverse_chunk_carry(
    dh_from_next: jax.Array,
    last_segment_id: jax.Array,
    next_segment_id: jax.Array,
) -> jax.Array:
  """Keep dHU/WY reverse carry only across a continuing segment edge.

  ``last_segment_id`` and ``next_segment_id`` have the leading batch/chunk
  shape of ``dh_from_next``. Feature axes are appended automatically.
  """
  if last_segment_id.shape != next_segment_id.shape:
    raise ValueError(
        "last and next segment ID shapes differ: "
        f"{last_segment_id.shape} != {next_segment_id.shape}"
    )
  if dh_from_next.shape[: last_segment_id.ndim] != last_segment_id.shape:
    raise ValueError(
        "segment IDs must match the leading dH shape: "
        f"dH={dh_from_next.shape}, ids={last_segment_id.shape}"
    )
  continues = (
      (last_segment_id != 0)
      & (next_segment_id != 0)
      & (last_segment_id == next_segment_id)
  )
  return jnp.where(
      _expand_token_mask(continues, dh_from_next.ndim), dh_from_next, 0
  )


def compact_chunk_segment_ids(
    segment_ids: jax.Array, chunk_size: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """View original-order segment IDs as compact chunks plus neighbour IDs.

  This operation is shape-only: unlike the existing packed mega preparation it
  never rounds individual sequence lengths. For an input ``[..., T]`` it
  returns ``[..., T // chunk_size, chunk_size]`` and one previous/next ID per
  compact chunk.
  """
  if segment_ids.ndim < 1:
    raise ValueError("segment_ids must have a token axis")
  if chunk_size <= 0:
    raise ValueError(f"chunk_size must be positive, got {chunk_size}")
  token_count = segment_ids.shape[-1]
  if token_count % chunk_size:
    raise ValueError(
        f"token count {token_count} must be divisible by chunk_size={chunk_size}"
    )

  chunk_count = token_count // chunk_size
  chunks = segment_ids.reshape(
      segment_ids.shape[:-1] + (chunk_count, chunk_size)
  )
  zero = jnp.zeros(segment_ids.shape[:-1] + (1,), dtype=segment_ids.dtype)
  previous_by_token = jnp.concatenate([zero, segment_ids[..., :-1]], axis=-1)
  next_by_token = jnp.concatenate([segment_ids[..., 1:], zero], axis=-1)
  previous_ids = previous_by_token[..., ::chunk_size]
  next_ids = next_by_token[..., chunk_size - 1 :: chunk_size]
  return chunks, previous_ids, next_ids


def _expand_token_mask(mask: jax.Array, target_ndim: int) -> jax.Array:
  """Append feature axes so a token mask broadcasts over ``values``."""
  return mask.reshape(mask.shape + (1,) * (target_ndim - mask.ndim))


def fixed_width_prefix_sum(values: jax.Array) -> jax.Array:
  """Inclusive prefix sum using the fixed-width tree used by KDA kernels.

  Keeping a continuous coordinate across packed segment boundaries is safe
  because all semantic contractions are masked to equal segment IDs and gate
  differences within one segment are translation invariant.  It also avoids
  a large positive jump when a segmented prefix is reset inside a chunk,
  which can overflow the blockwise Q/K exponential rescaling before the
  cross-segment entries are masked.
  """
  width = values.shape[-2]
  result = values
  shift = 1
  while shift < width:
    shifted = jnp.concatenate(
        [jnp.zeros_like(result[..., :shift, :]), result[..., :-shift, :]],
        axis=-2,
    )
    result = result + shifted
    shift *= 2
  return result


def segmented_prefix_sum(
    values: jax.Array, segment_ids: jax.Array
) -> jax.Array:
  """Inclusive fixed-width prefix sum that restarts at segment boundaries.

  Args:
    values: Array whose penultimate logical axis is the token axis.  The token
      axis is selected by ``segment_ids.ndim - 1`` so both ``[T, D]`` and
      ``[H, T, D]`` layouts are supported.
    segment_ids: Integer array matching the leading/token axes of ``values``.
      IDs are only required to be equal within one fragment; ID 0 is padding.

  Returns:
    An array with the same shape and dtype as ``values``. Padding rows are zero.
  """
  if values.ndim < segment_ids.ndim:
    raise ValueError(
        f"values.ndim={values.ndim} must be >= segment_ids.ndim={segment_ids.ndim}"
    )
  if values.shape[: segment_ids.ndim] != segment_ids.shape:
    raise ValueError(
        "segment_ids must match the leading/token axes of values: "
        f"values={values.shape}, segment_ids={segment_ids.shape}"
    )

  width = segment_ids.shape[-1]
  row_ids = segment_ids[..., :, None]
  col_ids = segment_ids[..., None, :]
  same_segment = (row_ids == col_ids) & (row_ids > 0) & (col_ids > 0)
  causal = jnp.arange(width)[:, None] >= jnp.arange(width)[None, :]
  matrix = (same_segment & causal).astype(values.dtype)
  flat = values.reshape(segment_ids.shape + (-1,))
  return jnp.matmul(matrix, flat).reshape(values.shape)


def segmented_reverse_sum(
    values: jax.Array, segment_ids: jax.Array
) -> jax.Array:
  """Inclusive reverse sum that stops at every packed segment boundary."""
  width = segment_ids.shape[-1]
  row_ids = segment_ids[..., :, None]
  col_ids = segment_ids[..., None, :]
  same_segment = (row_ids == col_ids) & (row_ids > 0) & (col_ids > 0)
  reverse_causal = jnp.arange(width)[:, None] <= jnp.arange(width)[None, :]
  matrix = (same_segment & reverse_causal).astype(values.dtype)
  flat = values.reshape(segment_ids.shape + (-1,))
  return jnp.matmul(matrix, flat).reshape(values.shape)


def segmented_suffix_last(
    values: jax.Array, segment_ids: jax.Array
) -> jax.Array:
  """Broadcast each contiguous segment's final value to all of its tokens."""
  if values.ndim < segment_ids.ndim:
    raise ValueError(
        f"values.ndim={values.ndim} must be >= segment_ids.ndim={segment_ids.ndim}"
    )
  if values.shape[: segment_ids.ndim] != segment_ids.shape:
    raise ValueError(
        "segment_ids must match the leading/token axes of values: "
        f"values={values.shape}, segment_ids={segment_ids.shape}"
    )
  width = segment_ids.shape[-1]
  row_ids = segment_ids[..., :, None]
  col_ids = segment_ids[..., None, :]
  same_segment = (row_ids == col_ids) & (row_ids > 0) & (col_ids > 0)
  positions = jnp.arange(width, dtype=jnp.int32)
  last_positions = jnp.max(
      jnp.where(same_segment, positions, jnp.int32(-1)), axis=-1
  )
  selector = (
      same_segment
      & (positions == last_positions[..., None])
      & (last_positions[..., None] >= 0)
  ).astype(values.dtype)
  flat = values.reshape(segment_ids.shape + (-1,))
  return jnp.matmul(selector, flat).reshape(values.shape)


def same_segment_causal_mask(
    segment_ids: jax.Array, *, include_diagonal: bool = True
) -> jax.Array:
  """Return a per-chunk causal mask with cross-segment entries removed."""
  if segment_ids.ndim < 1:
    raise ValueError("segment_ids must have a token axis")
  width = segment_ids.shape[-1]
  row = jnp.arange(width, dtype=jnp.int32)[:, None]
  col = jnp.arange(width, dtype=jnp.int32)[None, :]
  causal = row >= col if include_diagonal else row > col
  # Expand int32 IDs before comparison. Mosaic cannot lower the equivalent
  # bool-vector shape cast from ``valid[..., :, None]`` on TPU.
  row_ids = segment_ids[..., :, None]
  col_ids = segment_ids[..., None, :]
  return (row_ids == col_ids) & (row_ids > 0) & (col_ids > 0) & causal


def compact_chunk_fragment_counts(segment_ids: jax.Array) -> jax.Array:
  """Count non-padding sequence fragments in each compact chunk.

  ``segment_ids`` is expected to already include a chunk axis, for example
  ``[B, NT, BT]``.  A fragment starts at a non-zero token whose ID differs
  from the preceding token *inside the same chunk*.  Consequently a segment
  continued from the previous chunk still counts as the first fragment of
  this chunk.
  """
  if segment_ids.ndim < 1:
    raise ValueError("segment_ids must have a token axis")
  valid = segment_ids != 0
  starts = jnp.concatenate(
      [
          valid[..., :1],
          valid[..., 1:] & (segment_ids[..., 1:] != segment_ids[..., :-1]),
      ],
      axis=-1,
  )
  return jnp.sum(starts, axis=-1, dtype=jnp.int32)


def compact_chunk_dispatch_kinds(segment_ids: jax.Array) -> jax.Array:
  """Classify compact chunks without changing the fast-kernel contract.

  Args:
    segment_ids: Integer IDs with a final fixed-width token axis, typically
      ``[B, NT, BT]``. ID 0 denotes padding.

  Returns:
    An int32 array with shape ``segment_ids.shape[:-1]``. Values are
    ``COMPACT_CHUNK_FULL`` for zero or one non-padding fragment,
    ``COMPACT_CHUNK_TWO_FRAGMENT`` for exactly two fragments, and
    ``COMPACT_CHUNK_GENERAL`` for three or more fragments. The final class is
    intended for a separate general segmented kernel so the common kernels do
    not inherit its control flow or metadata cost.
  """
  fragment_counts = compact_chunk_fragment_counts(segment_ids)
  return jnp.where(
      fragment_counts <= 1,
      jnp.int32(COMPACT_CHUNK_FULL),
      jnp.where(
          fragment_counts == 2,
          jnp.int32(COMPACT_CHUNK_TWO_FRAGMENT),
          jnp.int32(COMPACT_CHUNK_GENERAL),
      ),
  )


def compact_chunk_boundary_metadata(
    segment_ids: jax.Array,
    previous_segment_id: jax.Array,
    next_segment_id: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Build masks needed to route recurrent state through a compact chunk.

  Returns ``(first_fragment, last_fragment, continues_from_previous,
  continues_to_next)``. The first two masks have the same shape as
  ``segment_ids``; continuation flags have the leading shape with the token
  dimension removed.
  """
  if segment_ids.ndim < 1:
    raise ValueError("segment_ids must have a token axis")
  expected_boundary_shape = segment_ids.shape[:-1]
  if previous_segment_id.shape != expected_boundary_shape:
    raise ValueError(
        "previous_segment_id must match segment_ids leading shape: "
        f"expected={expected_boundary_shape}, got={previous_segment_id.shape}"
    )
  if next_segment_id.shape != expected_boundary_shape:
    raise ValueError(
        "next_segment_id must match segment_ids leading shape: "
        f"expected={expected_boundary_shape}, got={next_segment_id.shape}"
    )

  valid = segment_ids != 0
  positions = jnp.arange(segment_ids.shape[-1], dtype=jnp.int32)
  first_position = jnp.min(
      jnp.where(valid, positions, jnp.int32(segment_ids.shape[-1])), axis=-1
  )
  last_position = jnp.max(jnp.where(valid, positions, jnp.int32(-1)), axis=-1)
  has_valid = last_position >= 0
  first_id = jnp.take_along_axis(
      segment_ids,
      jnp.clip(first_position, 0, segment_ids.shape[-1] - 1)[..., None],
      axis=-1,
  )[..., 0]
  last_id = jnp.take_along_axis(
      segment_ids,
      jnp.clip(last_position, 0, segment_ids.shape[-1] - 1)[..., None],
      axis=-1,
  )[..., 0]
  first_id = jnp.where(has_valid, first_id, jnp.zeros_like(first_id))
  last_id = jnp.where(has_valid, last_id, jnp.zeros_like(last_id))
  first_fragment = valid & (segment_ids == first_id[..., None])
  last_fragment = valid & (segment_ids == last_id[..., None])
  continues_from_previous = (
      (first_id != 0)
      & (previous_segment_id != 0)
      & (first_id == previous_segment_id)
  )
  continues_to_next = (
      (last_id != 0) & (next_segment_id != 0) & (last_id == next_segment_id)
  )
  return (
      first_fragment,
      last_fragment,
      continues_from_previous,
      continues_to_next,
  )


def compact_chunk_metadata(
    segment_ids: jax.Array, chunk_size: int
) -> CompactChunkMetadata:
  """Build the fixed-shape ABI for compact segmented forward/backward.

  ``reverse_continuation[..., i]`` is true exactly when the reverse state
  tangent produced by token ``i + 1`` may flow into token ``i``.  The last
  position of each chunk compares against the first token in the next chunk,
  so this contract is symmetric across chunk boundaries.
  """
  chunks, previous_ids, next_ids = compact_chunk_segment_ids(
      segment_ids, chunk_size
  )
  _, _, continues_from_previous, continues_to_next = (
      compact_chunk_boundary_metadata(chunks, previous_ids, next_ids)
  )
  zero = jnp.zeros(segment_ids.shape[:-1] + (1,), dtype=segment_ids.dtype)
  next_by_token = jnp.concatenate([segment_ids[..., 1:], zero], axis=-1)
  reverse_continuation = (
      (segment_ids != 0) & (segment_ids == next_by_token)
  ).reshape(chunks.shape)
  return CompactChunkMetadata(
      segment_ids=chunks,
      previous_segment_ids=previous_ids,
      next_segment_ids=next_ids,
      same_segment_causal=same_segment_causal_mask(chunks),
      reverse_continuation=reverse_continuation,
      continues_from_previous=continues_from_previous,
      continues_to_next=continues_to_next,
  )


@jax.jit
def segmented_compute_dav(
    do: jax.Array,
    v_new: jax.Array,
    Aqk: jax.Array,
    segment_ids: jax.Array,
    scale: float,
) -> tuple[jax.Array, jax.Array]:
  """Compute compact chunk dAqk and dV with same-segment masking.

  Args:
    do: Output cotangent ``[H, BT, V]``.
    v_new: WY-transformed values ``[H, BT, V]``.
    Aqk: Forward compact attention matrix ``[H, BT, BT]``.
    segment_ids: Token segment IDs ``[BT]``.
    scale: Query/key scale applied to dAqk.

  Returns:
    ``dAqk [H,BT,BT]`` and ``dV [H,BT,V]``. Cross-segment and
    non-causal matrix entries are exactly zero.
  """
  if do.shape != v_new.shape:
    raise ValueError(f"do and v_new shapes differ: {do.shape} != {v_new.shape}")
  H, BT, _ = do.shape
  if Aqk.shape != (H, BT, BT):
    raise ValueError(f"Aqk must have shape {(H, BT, BT)}, got {Aqk.shape}")
  if segment_ids.shape != (BT,):
    raise ValueError(
        f"segment_ids must have shape {(BT,)}, got {segment_ids.shape}"
    )
  mask = same_segment_causal_mask(segment_ids)
  dAqk = jnp.matmul(
      do.astype(jnp.float32),
      v_new.astype(jnp.float32).transpose(0, 2, 1),
      preferred_element_type=jnp.float32,
  )
  dAqk = jnp.where(mask[None], dAqk * scale, 0)
  Aqk = jnp.where(mask[None], Aqk, 0)
  dV = jnp.matmul(
      Aqk.astype(jnp.float32).transpose(0, 2, 1),
      do.astype(jnp.float32),
      preferred_element_type=jnp.float32,
  )
  return dAqk, dV


def segmented_compute_dav_pallas(
    do: jax.Array,
    v_new: jax.Array,
    Aqk: jax.Array,
    segment_ids: jax.Array,
    scale: float,
) -> tuple[jax.Array, jax.Array]:
  """Standalone Pallas lowering of the compact segmented dAqk/dV stage.

  This checkpoint intentionally owns one complete small chunk per program.
  The M5 integration will replace the full-array refs with its existing
  ``[MB,BT,*]`` VMEM tiles after token-level segment-ID DMA is available.
  """
  if do.shape != v_new.shape:
    raise ValueError(f"do and v_new shapes differ: {do.shape} != {v_new.shape}")
  H, BT, V = do.shape
  if Aqk.shape != (H, BT, BT):
    raise ValueError(f"Aqk must have shape {(H, BT, BT)}, got {Aqk.shape}")
  if segment_ids.shape != (BT,):
    raise ValueError(
        f"segment_ids must have shape {(BT,)}, got {segment_ids.shape}"
    )

  def kernel(do_ref, v_ref, Aqk_ref, ids_ref, dA_ref, dV_ref):
    ids = ids_ref[:]
    width = ids.shape[0]
    row = jnp.arange(width, dtype=jnp.int32)[:, None]
    col = jnp.arange(width, dtype=jnp.int32)[None, :]
    mask = (
        (row >= col)
        & (ids[:, None] == ids[None, :])
        & (ids[:, None] != 0)
        & (ids[None, :] != 0)
    )
    bdo = do_ref[:].astype(jnp.float32)
    bv = v_ref[:].astype(jnp.float32)
    bA = jnp.where(mask[None], Aqk_ref[:], 0).astype(jnp.float32)
    dA_ref[:] = jnp.where(
        mask[None],
        jnp.matmul(
            bdo, bv.transpose(0, 2, 1), preferred_element_type=jnp.float32
        )
        * scale,
        0,
    )
    dV_ref[:] = jnp.matmul(
        bA.transpose(0, 2, 1), bdo, preferred_element_type=jnp.float32
    )

  return pl.pallas_call(
      kernel,
      out_shape=(
          jax.ShapeDtypeStruct((H, BT, BT), jnp.float32),
          jax.ShapeDtypeStruct((H, BT, V), jnp.float32),
      ),
      grid=(),
      interpret=get_interpret(),
      name="kda_segmented_dav_checkpoint",
  )(do, v_new, Aqk, segment_ids)
