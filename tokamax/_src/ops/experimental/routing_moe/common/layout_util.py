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

"""Functions relevant to layouts. See https://openxla.org/xla/tiled_layout."""

import jax
from jax.experimental import hijax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def get_uint_dtype(itemsize: int) -> jnp.dtype:
  match itemsize:
    case 4:
      return jnp.uint32.dtype
    case 2:
      return jnp.uint16.dtype
    case 1:
      return jnp.uint8.dtype
    case _:
      raise NotImplementedError()


def to_compact_shape(
    large_shape: tuple[int, int], dtype: jnp.dtype, lanes: int = 128
) -> tuple[int, int, int, int]:
  """Returns compact layout shape of a given large layout shape."""
  assert len(large_shape) == 2
  rows, cols = large_shape
  packing = 4 // jnp.dtype(dtype).itemsize
  return (rows, cols // (packing * lanes), packing, lanes)


def to_large_shape(compact_shape: tuple[int, int, int, int]) -> tuple[int, int]:
  """Returns large layout shape of a given compact layout shape."""
  assert len(compact_shape) == 4
  rows, chunks, packing, lanes = compact_shape

  return (rows, chunks * packing * lanes)


def to_bank_conflict_aware_stride(
    stride: int, bank_size: int = 32, num_steps: int = 8
) -> int:
  """Returns stride value that will not cause bank conflict."""

  assert num_steps <= bank_size

  def _is_power_of_two(n: int) -> bool:
    return n > 0 and n.bit_count() == 1

  assert _is_power_of_two(bank_size)
  assert _is_power_of_two(num_steps)
  # If the last log(threshold) bits of the stride contains the threshold value,
  # it is guaranteed that there will not be any conflict for num_steps.
  threshold = bank_size // num_steps
  # To check if last log(threshold) bits of the stride contains thereshold
  # value, we create bitmask (1 << threshold) - 1 and check against stride. This
  # is equivalent to performing stride % (2 * threshold). If this value is zero,
  # we simply add 1 to ensure the condition is satisfied.
  delta = 0 if stride % (2 * threshold) else 1
  return stride + delta


def load_compact_to_large(
    vmem: hijax.TransformedRef, real_cols: int
) -> jax.Array:
  """Strided load a compact layout VMEM into large layout VREG."""

  dtype = vmem.dtype
  itemsize = dtype.itemsize
  uint_dtype = get_uint_dtype(itemsize)
  stride, packing, lanes = vmem.shape[1:]
  words = vmem.bitcast(jnp.uint32).reshape(-1, lanes)

  chunks = []
  for c in range(real_cols // (packing * lanes)):
    packed = words[c::stride]
    if packing > 1:
      unpacked_list = []
      for p in range(packing):
        shift = p * (itemsize * 8)
        unpacked_shifted = jnp.bitwise_right_shift(packed, shift)
        # .astype groups "packing" number of elements from second last dim,
        # extracts rightmost "itemsize" bits per 32-bits, and compacts them into
        # the same 32-bits. Because we have already performed right shift, and
        # stride load, it is guaranteed that rightmost "itemsize" bits all came
        # from the same row.
        unpacked_list.append(unpacked_shifted.astype(uint_dtype))
      packed = jnp.concat(unpacked_list, axis=-1)
    chunks.append(pltpu.bitcast(packed, dtype))
  return jnp.concat(chunks, axis=-1)


def store_large_to_compact(vmem: hijax.TransformedRef, vreg: jax.Array):
  """Strided store a large layout VREG into compact layout VMEM."""

  dtype = vmem.dtype
  # TODO: Support use case where VREG is in different bitwidth.
  assert vmem.dtype == vreg.dtype
  itemsize = dtype.itemsize
  uint_dtype = get_uint_dtype(itemsize)
  stride, packing, lanes = vmem.shape[1:]
  words = vmem.bitcast(jnp.uint32).reshape(-1, lanes)
  blocks = vreg.shape[-1] // (lanes * packing)

  chunks = jnp.split(vreg, blocks, axis=-1)
  for i in range(blocks):
    packed_u32 = pltpu.bitcast(chunks[i], jnp.uint32)
    if packing > 1:
      for p in range(packing):
        shift = p * (itemsize * 8)
        shifted_u32 = jnp.bitwise_right_shift(packed_u32, shift)
        repacked = pltpu.pack_elementwise(
            jnp.split(shifted_u32, packing, axis=-1),
            packed_dtype=uint_dtype,
        )
        words[i + p * stride :: stride * packing] = pltpu.bitcast(
            repacked, jnp.uint32
        )
    else:
      words[i::stride] = packed_u32


def large_interleave_lane(
    left_vreg: jax.Array, right_vreg: jax.Array, chunk_size: int = 128
) -> jax.Array:
  """Interleaves two arrays along lane dim at zero-cost."""

  assert left_vreg.shape == right_vreg.shape
  num_chunks = left_vreg.shape[-1] // chunk_size
  # Because large layout splits a tensor into group of 8x128, both splitting
  # and concatenating them in groups of 128 column is free.
  lhs_chunks = jnp.split(left_vreg, num_chunks, axis=-1)
  rhs_chunks = jnp.split(right_vreg, num_chunks, axis=-1)
  interleaved = []
  for i in range(num_chunks):
    interleaved += [lhs_chunks[i], rhs_chunks[i]]
  return jnp.concat(interleaved, axis=-1)


def large_deinterleave_lane(
    vreg: jax.Array, lanesize: int = 128
) -> tuple[jax.Array, jax.Array]:
  """Deinterleaves an array along lane dim at zero-cost."""
  num_chunks = vreg.shape[-1] // lanesize

  # Because large layout splits a tensor into group of 8x128, both splitting
  # and concatenating them in groups of 128 column is free.
  chunks = jnp.split(vreg, num_chunks, axis=-1)
  lhs = jnp.concat(chunks[0::2], axis=-1)
  rhs = jnp.concat(chunks[1::2], axis=-1)
  return lhs, rhs
