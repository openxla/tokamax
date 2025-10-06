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
"""Grouped matrix multiplication kernels for TPU written in Pallas."""

# pylint: disable=too-many-positional-arguments, unnecessary-lambda-assignment

from collections.abc import Callable
import dataclasses
import functools
from functools import partial  # pylint: disable=g-importing-member
import json
from typing import Any, Optional

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src import mosaic_tpu as common
from tokamax._src import quantization

QuantizedArray = quantization.QuantizedArray


def _validate_args(
    *,
    lhs: jax.Array | QuantizedArray,
    rhs: jax.Array | QuantizedArray,
    group_sizes: jax.Array,
    expected_rhs_dims: int = 3,
) -> tuple[jax.Array | QuantizedArray,
           jax.Array | QuantizedArray,
           jnp.dtype]:
  """Validates the arguments for the gmm function."""
  # Validate 'lhs'.
  if lhs.ndim != 2:
    raise ValueError(f"Expected 2-tensor for 'lhs' but got {lhs.ndim}-tensor.")

  # Validate 'rhs'.
  if rhs.ndim != expected_rhs_dims:
    raise ValueError(
        f"Expected {expected_rhs_dims}-tensor for 'rhs' but got"
        f" {rhs.ndim}-tensor."
    )

  # Validate 'group_sizes'.
  if group_sizes.dtype == jnp.uint32:
    group_sizes = group_sizes.astype(jnp.int32)
  if group_sizes.dtype != jnp.int32:
    raise ValueError(
        f"Expected 32-bit integer 'group_sizes' but got {group_sizes.dtype}."
    )

  return lhs, group_sizes, common.select_input_dtype(lhs, rhs)


def _calculate_num_tiles(x: int, tx: int) -> int:
  tiles, rem = divmod(x, tx)
  if rem:
    raise NotImplementedError(f"{x} must be divisible by tile size ({tx}).")
  return tiles


def _calculate_irregular_num_tiles(x: int, tx: int) -> tuple[int, int]:
  tiles, rem = divmod(x, tx)
  if rem:
    tiles += 1
  return tiles, rem


def _get_quant_dot_operands_and_dtype(
    lhs: jax.Array | QuantizedArray,
    rhs: jax.Array | QuantizedArray,
) -> tuple[jax.Array, jax.Array, jnp.dtype]:
  """Discover the accumulation type: e.g., fp8 -> fp32 and int8 -> int32."""
  qdtype, lhs_, rhs_ = None, lhs, rhs
  if isinstance(lhs, QuantizedArray):
    lhs_, qdtype = lhs.values, lhs.values.dtype
  if isinstance(rhs, QuantizedArray):
    rhs_, qdtype = rhs.values, rhs.values.dtype
  assert qdtype is not None

  if (jnp.issubdtype(lhs_.dtype, jnp.integer)
      and jnp.issubdtype(rhs_.dtype, jnp.integer)
      and qdtype.name.startswith("int")):
    return lhs_, rhs_, jnp.int32
  else:
    return lhs_, rhs_, jnp.float32

# TODO: clean up the typing for GroupMetadata.
GroupMetadata = Any


def make_group_metadata(
    *,
    group_sizes: jax.Array,
    m: int,
    tm: int,
    start_group: jax.Array,
    num_nonzero_groups: int,
    visit_empty_groups: bool = True,
) -> GroupMetadata:
  """Create the metadata needed for grouped matmul computation.

  Args:
    group_sizes: A 1d, jax.Array with shape `[num_groups]` and `jnp.int32`
      dtype.
    m: The number of rows in lhs.
    tm: The m-dimension tile size being used.
    start_group: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    num_nonzero_groups: Number of groups in group sizes to compute on. Useful in
      combination with group_offset.
    visit_empty_groups: If True, do not squeeze tiles for empty groups out of
      the metadata. This is necessary for tgmm, where we at least need to zero
      the output for each group.

  Returns:
    tuple of:
      group_offsets: A 1d, jax.Array with shape [num_groups+1] and jnp.int32
        dtype. group_offsets[i] indicates the row at which group [i] starts in
        the lhs matrix and group_offsets[i-1] = m.
      group_ids: A 1d, jax.Array with shape [m_tiles + num_groups] and
        jnp.int32 dtype. group_ids[i] indicates which group grid index 'i' will
        work on.
      m_tile_ids: A 1d, jax.Array with shape [m_tiles + num_groups] and
        jnp.int32. m_tile_ids[i] indicates which m-dimension tile grid index 'i'
        will work on.
    num_tiles: The number of m-dimension tiles to execute.
  """
  num_groups = group_sizes.shape[0]
  end_group = start_group + num_nonzero_groups - 1

  # Calculate the offset of each group, starting at zero. This metadata is
  # similar to row offsets in a CSR matrix. The following properties hold:
  #
  # group_offsets.shape = [num_groups + 1]
  # group_offsets[0] = 0
  # group_offsets[num_groups] = m
  #
  # The row at which group 'i' starts is group_offsets[i].
  group_ends = jnp.cumsum(group_sizes)
  group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])

  # Assign a group id to each grid index.
  #
  # If a group starts somewhere other than the start of a tile or ends somewhere
  # other than the end of a tile we need to compute that full tile. Calculate
  # the number of tiles for each group by rounding their end up to the nearest
  # 'tm' and their start down to the nearest 'tm'.

  # (1) Round the group_ends up to the nearest multiple of 'tm'.
  #
  # NOTE: This does not change group_offsets[num_groups], which is m
  # (because we enforce m is divisible by tm).
  rounded_group_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)

  # (2) Round the group_starts down to the nearest multiple of 'tm'.
  group_starts = jnp.concatenate(
      [jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]]
  )
  rounded_group_starts = group_starts // tm * tm

  # (3) Calculate the number of rows in each group.
  #
  # NOTE: Handle zero-sized groups as a special case. If the start for a
  # zero-sized group is not divisible by 'tm' its start will be rounded down and
  # its end will be rounded up such that its size will become 1 tile here.
  rounded_group_sizes = rounded_group_ends - rounded_group_starts
  rounded_group_sizes = jnp.where(group_sizes == 0, 0, rounded_group_sizes)

  # (4) Convert the group sizes from units of rows to unit of 'tm' sized tiles.
  #
  # An m-dimension tile is 'owned' by group 'i' if the first row of the tile
  # belongs to group 'i'. In addition to owned tiles, each group can have 0 or 1
  # initial partial tiles if it's first row does not occur in the first row of a
  # tile. The '0-th' group never has a partial tile because it always starts at
  # the 0-th row.
  #
  # If no group has a partial tile, the total number of tiles is equal to
  # 'm // tm'. If every group has a partial except the 0-th group, the total
  # number of tiles is equal to 'm // tm + num_groups - 1'. Thus we know that
  #
  # tiles_m <= group_tiles.sum() <= tiles_m + num_groups - 1
  #
  # Where tiles_m = m // tm.
  #
  # NOTE: All group sizes are divisible by 'tm' because of the rounding in steps
  # (1) and (2) so this division is exact.
  group_tiles = rounded_group_sizes // tm

  if visit_empty_groups:
    # Insert one tile for empty groups.
    group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

  # Create the group ids for each grid index based on the tile counts for each
  # group.
  #
  # NOTE: This repeat(...) will pad group_ids with the final group id if
  # group_tiles.sum() < tiles_m + num_groups - 1. The kernel grid will be sized
  # such that we only execute the necessary number of tiles.
  tiles_m = _calculate_num_tiles(m, tm)
  group_ids = jnp.repeat(
      jnp.arange(num_groups, dtype=jnp.int32),
      group_tiles,
      total_repeat_length=tiles_m + num_groups - 1,
  )

  # Assign an m-dimension tile id to each grid index.
  #
  # NOTE: Output tiles can only be re-visited consecutively. The following
  # procedure guarantees that m-dimension tile indices respect this.

  # (1) Calculate how many times each m-dimension tile will be visited.
  #
  # Each tile is guaranteed to be visited once by the group that owns the tile.
  # The remaining possible visits occur when a group starts inside of a tile at
  # a position other than the first row. We can calculate which m-dimension tile
  # each group starts in by floor-dividing its offset with `tm` and then count
  # tile visits with a histogram.
  #
  # To avoid double counting tile visits from the group that owns the tile,
  # filter these out by assigning their tile id to `tile_m` (one beyond the max)
  # such that they're ignored by the subsequent histogram. Also filter out any
  # group which is empty.
  #
  # TODO: Invert the 'partial_tile_mask' predicates to be more clear.
  partial_tile_mask = jnp.logical_or(
      (group_offsets[:-1] % tm) == 0, group_sizes == 0
  )

  # Explicitly enable tiles for zero sized groups, if specified. This covers
  # zero sized groups that start on a tile-aligned row and those that do not.
  if visit_empty_groups:
    partial_tile_mask = jnp.where(group_sizes == 0, 0, partial_tile_mask)

  partial_tile_ids = jnp.where(
      partial_tile_mask, tiles_m, group_offsets[:-1] // tm
  )

  tile_visits = (
      jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0]
      + 1
  )

  # Create the m-dimension tile ids for each grid index based on the visit
  # counts for each tile.
  m_tile_ids = jnp.repeat(
      jnp.arange(tiles_m, dtype=jnp.int32),
      tile_visits.astype(jnp.int32),
      total_repeat_length=tiles_m + num_groups - 1,
  )

  # Account for sharding.
  #
  # Find the start of the groups owned by our shard and shift the group_ids and
  # m_tile_ids s.t. the metadata for our tiles are at the front of the arrays.
  #
  # TODO: Move this offset into the kernel to avoid these rolls.
  first_tile_in_shard = (group_ids < start_group).sum()
  group_ids = jnp.roll(group_ids, shift=-first_tile_in_shard, axis=0)
  m_tile_ids = jnp.roll(m_tile_ids, shift=-first_tile_in_shard, axis=0)

  # Calculate the number of tiles we need to compute for our shard.
  #
  # Remove tile visits that belong to a group not in our shard.
  iota = jnp.arange(num_groups, dtype=jnp.int32)
  active_group_mask = jnp.logical_and(iota <= end_group, iota >= start_group)
  group_tiles = jnp.where(active_group_mask, group_tiles, 0)
  num_tiles = group_tiles.sum()
  return (group_offsets, group_ids, m_tile_ids), num_tiles


def _get_group_size(
    *, grid_id: jax.Array, group_metadata: GroupMetadata
) -> jax.Array:
  """Calculate the number of rows in the current group."""
  group_offsets, group_ids = group_metadata[:2]
  group_id = group_ids[grid_id]
  group_start = group_offsets[group_id]
  group_end = group_offsets[group_id + 1]
  return group_end - group_start


def _get_store_mask(
    *,
    grid_id: jax.Array,
    group_metadata: GroupMetadata,
    tm: int,
    tn: int,
) -> jax.Array:
  """Mask for rows that belong to the current group in the current tile."""
  group_offsets, group_ids, m_tile_ids = group_metadata[:3]
  group_id = group_ids[grid_id]
  group_start = group_offsets[group_id]
  group_end = group_offsets[group_id + 1]
  m_id = m_tile_ids[grid_id] * tm
  iota = jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + m_id
  return jnp.logical_and(iota >= group_start, iota < group_end)


def _zero_uninitialized_memory(
    out: jnp.ndarray,
    *,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
    group_metadata: GroupMetadata,
) -> jnp.ndarray:
  """Zero out uninitialized memory from output."""
  group_offsets = group_metadata[0]
  group_start = group_offsets[start_group]
  group_end = group_offsets[start_group + num_nonzero_groups]
  valid_mask = jax.lax.broadcasted_iota(jnp.int32, (out.shape[0],), 0)
  valid_mask = (valid_mask >= group_start) & (valid_mask < group_end)
  return jnp.where(valid_mask[:, None], out, 0)



LutFn = Callable[[int, int, int], Optional[tuple[int, int, int]]]


@functools.partial(
    jax.jit,
    static_argnames=[
        "preferred_element_type",
        "tiling",
        "transpose_rhs",
        "interpret",
    ],
)
def gmm(
    lhs: jax.Array | QuantizedArray,
    rhs: jax.Array | QuantizedArray,
    group_sizes: jax.Array,
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jax.Array | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> jax.Array:
  """Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.

  Args:
    lhs: A 2d, jax.Array with shape [m, k].
    rhs: A 3d, jax.Array with shape [num_groups, k, n].
    group_sizes: A 1d, jax.Array with shape [num_groups] and jnp.int32 dtype.
    preferred_element_type: jnp.dtype, the element type for the output matrix.
    tiling: 3-tuple of ints. The m, k and n-dimension tile sizes.
    group_offset: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    transpose_rhs: True if the rhs needs to be transposed.
    interpret: Whether or not to run the kernel in interpret mode, helpful for
      testing and debugging.

  Returns:
    A 2d, jax.Array with shape [m, n].
  """
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    if group_offset.shape:
      raise ValueError(
          f"group_offset must be a ()-shaped array. Got: {group_offset.shape}."
      )
    group_offset = group_offset[None]
  num_current_groups = rhs.shape[0]
  num_total_groups = group_sizes.shape[0]

  lhs, group_sizes, input_dtype = _validate_args(
      lhs=lhs, rhs=rhs, group_sizes=group_sizes
  )

  # Gather shape information.
  m, k, n = (lhs.shape[0], lhs.shape[1], rhs.shape[2])
  if transpose_rhs:
    n = rhs.shape[1]

  # If tiling is callable, look up the problem dimensions in the LUT. If no
  # tuned tile dimensions are available throw an error.
  if callable(tiling):
    tiling = tiling(m, k, n)

  if tiling is None:
    raise ValueError(f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")

  tm, tk, tn = tiling
  tiles_k, k_rem = _calculate_irregular_num_tiles(k, tk)
  tiles_n, n_rem = _calculate_irregular_num_tiles(n, tn)
  del n_rem

  # Create the metadata we need for computation.
  group_metadata, num_active_tiles = make_group_metadata(  # pylint: disable=unbalanced-tuple-unpacking
      group_sizes=group_sizes,
      m=m,
      tm=tm,
      start_group=group_offset[0],
      num_nonzero_groups=rhs.shape[0],
      visit_empty_groups=False,
  )

  def kernel(
      group_metadata,
      group_offset,
      lhs_ref: jax.Array | QuantizedArray,
      rhs_ref: jax.Array | QuantizedArray,
      out,
      acc_scratch,
  ):
    if transpose_rhs:
      dimension_numbers = (((1,), (1,)), ((), ()))
    else:
      dimension_numbers = (((1,), (0,)), ((), ()))
    dot_general = lambda x, y, preferred_element_type: jax.lax.dot_general(
        x,
        y,
        dimension_numbers=dimension_numbers,
        preferred_element_type=preferred_element_type,
    )
    lhs_dim, rhs_dim = 1, (0 if not transpose_rhs else 1)

    group_offsets, group_ids, m_tile_ids = group_metadata
    del group_offsets, group_ids, group_offset, m_tile_ids

    grid_id = pl.program_id(1)
    k_i = pl.program_id(2)

    @pl.when(k_i == 0)
    def _zero_acc():
      acc_scratch[...] = jnp.zeros_like(acc_scratch)

    def mask_k_rem(
        x: jax.Array | QuantizedArray, *, dim: int
    ) -> jax.Array | QuantizedArray:
      if k_rem == 0:
        return x
      mask = lax.broadcasted_iota(jnp.int32, x.shape, dim) < k_rem
      if isinstance(x, QuantizedArray):
        return dataclasses.replace(x, values=jnp.where(mask, x.values, 0))
      return jnp.where(mask, x, 0)

    def _scale_output(out, scale, axis):
      repeats = out.shape[axis] // scale.shape[axis]
      return out * pltpu.repeat(scale, repeats, axis)

    def _store_accum():
      mask = _get_store_mask(
          grid_id=grid_id,
          group_metadata=group_metadata,
          tn=tn,
          tm=tm,
      )
      to_store = acc_scratch[...]
      out[...] = jax.lax.select(
          mask[...], to_store, out[...].astype(jnp.float32)
      ).astype(preferred_element_type)

    def _accum(is_last_k_tile):
      with jax.named_scope(f"accum-last_k_tile={is_last_k_tile}"):
        lhs = jax.tree.map(lambda x: x[...], lhs_ref)
        lhs = mask_k_rem(lhs, dim=lhs_dim) if is_last_k_tile else lhs
        rhs = jax.tree.map(lambda x: x[...], rhs_ref)
        rhs = mask_k_rem(rhs, dim=rhs_dim) if is_last_k_tile else rhs

        if isinstance(lhs, QuantizedArray) or isinstance(rhs, QuantizedArray):
          # discover the accumulation type, fp8 -> fp32 and int8 -> int32
          lhs_, rhs_, acc_qdtype = _get_quant_dot_operands_and_dtype(lhs, rhs)
          out = dot_general(lhs_, rhs_, acc_qdtype)

          if isinstance(lhs, QuantizedArray):
            out = _scale_output(out, lhs.scales, lhs_dim)
          if isinstance(rhs, QuantizedArray):
            rhs_scales = rhs.scales.T if transpose_rhs else rhs.scales
            out = _scale_output(out, rhs_scales, 0)  # always 0
        else:
          out = dot_general(lhs, rhs, jnp.float32)

        acc_scratch[...] += out.astype(acc_scratch.dtype)

        if is_last_k_tile:
          _store_accum()

    lax.cond(
        k_i == tiles_k - 1,
        partial(_accum, True),
        partial(_accum, False),
    )

  def lhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
    # lhs is (m, k). Load the [tm, tk] matrix for this m-tile.
    group_offsets, group_ids, m_tile_ids = group_metadata
    del n_i, group_offsets, group_ids, group_offset
    return m_tile_ids[grid_id], k_i

  def rhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
    # rhs is (num_groups, k, n). Load the [tk, tn] matrix based on the group id
    # for this m-tile.
    group_offsets, group_ids, m_tile_ids = group_metadata
    del group_offsets, m_tile_ids
    if transpose_rhs:
      k_i, n_i = n_i, k_i

    # NOTE: If we're working on only a shard of the rhs we need to adjust the
    # group index we load from to account for this. The group_ids are in the
    # "unsharded" domain.
    return group_ids[grid_id] - group_offset[0], k_i, n_i

  def out_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
    # out is (m, n). Load the [tm, tn] matrix for this m-tile.
    group_offsets, group_ids, m_tile_ids = group_metadata
    del k_i, group_offsets, group_ids, group_offset
    return m_tile_ids[grid_id], n_i

  lhs_block_spec = pl.BlockSpec((tm, tk), lhs_transform_indices)
  if transpose_rhs:
    rhs_block_spec = pl.BlockSpec((None, tn, tk), rhs_transform_indices)
  else:
    rhs_block_spec = pl.BlockSpec((None, tk, tn), rhs_transform_indices)
  out_block_spec = pl.BlockSpec((tm, tn), out_transform_indices)

  if isinstance(lhs, QuantizedArray):
    lhs, lhs_block_spec = common.quant_block_spec(lhs, lhs_block_spec, 1)
  if isinstance(rhs, QuantizedArray):
    rhs_axis = 2 if transpose_rhs else 1
    rhs, rhs_block_spec = common.quant_block_spec(rhs, rhs_block_spec, rhs_axis)

  if isinstance(lhs, QuantizedArray):
    lhs_bytes = (
        lhs.size * lhs.values.itemsize + lhs.scales.size * lhs.scales.itemsize
    )
  else:
    lhs_bytes = lhs.size * lhs.itemsize
  if isinstance(rhs, QuantizedArray):
    rhs_bytes = (k * n) * rhs.values.itemsize  # We don't read all of rhs
  else:
    rhs_bytes = k * n * rhs.itemsize

  out_bytes = (m * n) * jnp.dtype(preferred_element_type).itemsize
  max_active_tiles = group_metadata[1].size
  bytes_accessed = (
      (lhs_bytes * tiles_n) + (rhs_bytes * max_active_tiles) + out_bytes
  )
  flops = 2 * m * k * n
  cost_estimate = pl.CostEstimate(
      flops=flops, bytes_accessed=bytes_accessed, transcendentals=0
  )
  kernel_name = f"gmm_{tm}x{tk}x{tn}"
  if transpose_rhs:
    kernel_name += "_transpose_rhs"
  metadata = dict(
      prefer_element_type=jnp.dtype(preferred_element_type).name,
      tiling=dict(tile_m=tm, tile_k=tk, tile_n=tn),
      transpose_rhs=transpose_rhs,
  )
  pallas_call_fn = common.custom_buffered_pallas_call
  call_gmm = pallas_call_fn(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), preferred_element_type),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[lhs_block_spec, rhs_block_spec],
          out_specs=out_block_spec,
          grid=(tiles_n, num_active_tiles, tiles_k),
          scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary")
      ),
      interpret=interpret,
      cost_estimate=cost_estimate,
      name=kernel_name,
      metadata=dict(xprof_metadata=json.dumps(metadata)),
  )

  with jax.named_scope(kernel_name):
    out = call_gmm(group_metadata, group_offset, lhs, rhs)

  if num_current_groups < num_total_groups:
    out = _zero_uninitialized_memory(
        out,
        start_group=group_offset[0],
        num_nonzero_groups=rhs.shape[0],
        group_metadata=group_metadata,
    )
  return out


@functools.partial(
    jax.jit,
    static_argnames=[
        "preferred_element_type",
        "tiling",
        "num_actual_groups",
        "interpret",
    ],
)
def tgmm(
    lhs: jax.Array | QuantizedArray,
    rhs: jax.Array | QuantizedArray,
    group_sizes: jax.Array,
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jax.Array | None = None,
    num_actual_groups: int | None = None,
    interpret: bool = False,
) -> jax.Array:
  """Compute lhs[:, sizes[i-1]:sizes[i]] @ rhs[sizes[i-1]:sizes[i], :].

  Args:
    lhs: A 2d, jax.Array with shape [k, m].
    rhs: A 2d, jax.Array with shape [m, n].
    group_sizes: A 1d, jax.Array with shape [num_groups] and jnp.int32 dtype.
    preferred_element_type: jnp.dtype, the element type for the output matrix.
    tiling: 3-tuple of ints. The m, k and n-dimension tile sizes.
    group_offset: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    num_actual_groups: For when num_groups is sharded and we should only compute
      the groups that are local, starting from group_offset.
    interpret: Whether or not to run the kernel in interpret mode, helpful for
      testing and debugging.

  Returns:
    A  3d, jax.Array with shape [num_groups, k, n].
  """
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    group_offset = group_offset[None]
  lhs, group_sizes, input_dtype = _validate_args(
      lhs=lhs, rhs=rhs, group_sizes=group_sizes, expected_rhs_dims=2
  )

  # Gather shape information.
  k, m, n = (lhs.shape[0], lhs.shape[1], rhs.shape[1])
  # the general tgmm definition requires lhs @ rhs
  # but our memory pipeline loads (m, k), (m, n) and computes (m, k)^T @ (m, n)
  if isinstance(lhs, QuantizedArray):
    lhs = dataclasses.replace(lhs, values=lhs.values.mT, scales=lhs.scales.mT)
  else:
    lhs = lhs.mT

  num_groups = group_sizes.shape[0]
  num_actual_groups = (
      num_actual_groups if num_actual_groups is not None else num_groups
  )

  # If tiling is callable, look up the problem dimensions in the LUT. If no
  # tuned tile dimensions are available throw an error.
  if callable(tiling):
    tiling = tiling(m, k, n)

  if tiling is None:
    raise ValueError(f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")

  tm, tk, tn = tiling
  tiles_k, k_rem = _calculate_irregular_num_tiles(k, tk)
  del k_rem
  tiles_n, n_rem = _calculate_irregular_num_tiles(n, tn)
  del n_rem

  # Create the metadata we need for computation.
  group_metadata, num_active_tiles = make_group_metadata(
      group_sizes=group_sizes,
      m=m,
      tm=tm,
      start_group=group_offset[0],
      num_nonzero_groups=num_actual_groups,
      visit_empty_groups=True,
  )

  def _scale_output(out, scale, axis):
    repeats = out.shape[axis] // scale.shape[axis]
    return out * pltpu.repeat(scale, repeats, axis)

  def mask_fn(mask, x):
    if isinstance(x, QuantizedArray):
      return dataclasses.replace(x, values=jnp.where(mask, x.values, 0))
    else:
      return jnp.where(mask, x, 0)

  def kernel(
      group_metadata,
      group_offset,
      lhs_ref,
      rhs_ref,
      out_ref,
      acc_scratch,
  ):
    grid_id = pl.program_id(2)
    group_offsets, group_ids, m_tile_ids = group_metadata
    del group_offsets, group_offset, m_tile_ids

    group = group_ids[grid_id]
    prev_grid_id = jnp.where(grid_id > 0, grid_id - 1, 0)
    prev_group = group_ids[prev_grid_id]

    group_has_changed = jnp.logical_or(grid_id == 0, prev_group != group)

    @pl.when(group_has_changed)
    def _zero_acc():
      acc_scratch[...] = jnp.zeros_like(acc_scratch)

    # We'll only do computation if our group has a nonzero number of rows in it.
    dont_skip = (
        _get_group_size(grid_id=grid_id, group_metadata=group_metadata) > 0
    )

    @pl.when(dont_skip)
    def _do():
      opts = dict(grid_id=grid_id, group_metadata=group_metadata, tm=tm)
      dot = lambda x, y, preferred_element_type=preferred_element_type: lax.dot(
          x, y, preferred_element_type=preferred_element_type
      )

      lhs_mask = _get_store_mask(**opts, tn=tk)
      lhs = jax.tree.map(lambda x: x[...], lhs_ref)
      lhs = mask_fn(lhs_mask, lhs)

      rhs_mask = _get_store_mask(**opts, tn=tn)
      rhs = jax.tree.map(lambda x: x[...], rhs_ref)
      rhs = mask_fn(rhs_mask, rhs)

      if isinstance(lhs, QuantizedArray) or isinstance(rhs, QuantizedArray):
        lhs_, rhs_, acc_qdtype = _get_quant_dot_operands_and_dtype(lhs, rhs)
        out = dot(lhs_.T, rhs_, acc_qdtype)
        if isinstance(lhs, QuantizedArray):  # if lhs is not natively quant
          out = _scale_output(out, lhs.scales.T, 1)
        if isinstance(rhs, QuantizedArray):  # if rhs is not natively quant
          out = _scale_output(out, rhs.scales, 0)
      else:
        out = dot(lhs.T, rhs, jnp.float32)
      acc_scratch[...] += out.astype(acc_scratch.dtype)

    is_end_of_grid = grid_id == (pl.num_programs(2) - 1)
    next_grid_id = jnp.where(is_end_of_grid, grid_id, grid_id + 1)
    next_group = group_ids[next_grid_id]

    group_is_changing = jnp.logical_or(is_end_of_grid, group != next_group)

    @pl.when(group_is_changing)
    def _store_accum():
      out_ref[...] = acc_scratch[...].astype(preferred_element_type)

  def lhs_transform_indices(n_i, k_i, grid_id, group_metadata, group_offset):
    # lhs is (m, k). Load the [tm, tk] matrix for this m-tile.
    group_offsets, group_ids, m_tile_ids = group_metadata
    del n_i, group_offsets, group_ids, group_offset
    return m_tile_ids[grid_id], k_i

  def rhs_transform_indices(n_i, k_i, grid_id, group_metadata, group_offset):
    # rhs is (m, n). Load the [tm, tn] matrix for this m-tile.
    group_offsets, group_ids, m_tile_ids = group_metadata
    del k_i, group_offsets, group_ids, group_offset
    return m_tile_ids[grid_id], n_i

  def out_transform_indices(n_i, k_i, grid_id, group_metadata, group_offset):
    # out is (num_groups, k, n). Load the [tk, tn] matrix based on the group id
    # for this m-tile.
    group_offsets, group_ids, m_tile_ids = group_metadata
    del group_offsets, m_tile_ids

    # NOTE: If we're working on only a shard of the output we need to adjust the
    # group index we load from to account for this. The group_ids are in the
    # "unsharded" domain.
    return group_ids[grid_id] - group_offset[0], k_i, n_i

  lhs_block_spec = pl.BlockSpec((tm, tk), lhs_transform_indices)
  rhs_block_spec = pl.BlockSpec((tm, tn), rhs_transform_indices)
  out_block_spec = pl.BlockSpec((None, tk, tn), out_transform_indices)

  if isinstance(lhs, QuantizedArray):
    lhs, lhs_block_spec = common.quant_block_spec(lhs, lhs_block_spec, 0)
  if isinstance(rhs, QuantizedArray):
    rhs, rhs_block_spec = common.quant_block_spec(rhs, rhs_block_spec, 0)

  if isinstance(lhs, QuantizedArray):
    lhs_bytes = (
        lhs.size * lhs.values.itemsize + lhs.scales.size * lhs.scales.itemsize
    )
  else:
    lhs_bytes = lhs.size * lhs.itemsize
  if isinstance(rhs, QuantizedArray):
    rhs_bytes = (
        rhs.size * rhs.values.itemsize + rhs.scales.size * rhs.scales.itemsize
    )
  else:
    rhs_bytes = rhs.size * rhs.itemsize
  out_bytewidth = jnp.dtype(preferred_element_type).itemsize
  out_bytes = (num_actual_groups * k * n) * out_bytewidth
  bytes_accessed = (lhs_bytes * tiles_n) + (rhs_bytes * tiles_k) + out_bytes
  flops = 2 * m * k * n
  cost_estimate = pl.CostEstimate(
      flops=flops, bytes_accessed=bytes_accessed, transcendentals=0
  )

  kernel_name = f"tgmm_{tm}x{tk}x{tn}"
  pallas_call_fn = pl.pallas_call
  metadata = dict(
      tiling=dict(tile_m=tm, tile_k=tk, tile_n=tn),
      prefer_element_type=jnp.dtype(preferred_element_type).name,
      num_actual_groups=num_actual_groups
  )
  call_gmm = pallas_call_fn(
      kernel,
      out_shape=jax.ShapeDtypeStruct(
          (num_actual_groups, k, n), preferred_element_type
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[lhs_block_spec, rhs_block_spec],
          out_specs=out_block_spec,
          grid=(tiles_n, tiles_k, num_active_tiles),
          scratch_shapes=[pltpu.VMEM((tk, tn), jnp.float32)],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary")
      ),
      interpret=interpret,
      cost_estimate=cost_estimate,
      name=kernel_name,
      metadata=dict(xprof_metadata=json.dumps(metadata)),
  )

  with jax.named_scope(kernel_name):
    return call_gmm(group_metadata, group_offset, lhs, rhs)
