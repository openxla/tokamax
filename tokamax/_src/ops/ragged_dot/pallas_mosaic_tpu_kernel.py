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
import functools
import json

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import qwix
from tokamax._src import mosaic_tpu as common
from tokamax._src import precision as precision_lib
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import base


CanonicalPrecision = precision_lib.CanonicalPrecision
QArray = qwix.QArray


def _validate_args(
    lhs: jax.Array | QArray,
    rhs: jax.Array | QArray,
    group_sizes: jax.Array,
    *,
    expected_rhs_dims: int = 3,
) -> jax.Array:
  """Validates the arguments for the gmm function."""
  if lhs.ndim != 2:
    raise ValueError(f"Expected 2-tensor for 'lhs' but got {lhs.ndim}-tensor.")

  if rhs.ndim != expected_rhs_dims:
    raise ValueError(
        f"Expected {expected_rhs_dims}-tensor for 'rhs' but got"
        f" {rhs.ndim}-tensor."
    )

  if group_sizes.dtype not in (jnp.int32, jnp.uint32):
    raise ValueError(
        f"Expected 32-bit integer 'group_sizes' but got {group_sizes.dtype}."
    )
  return group_sizes.astype(jnp.int32)


GroupMetadata = tuple[jax.Array, jax.Array, jax.Array]


def make_group_metadata(
    *,
    group_sizes: jax.Array,
    m: int,
    tm: int,
    start_group: jax.Array,
    num_nonzero_groups: int,
    visit_empty_groups: bool,
) -> tuple[GroupMetadata, jax.Array]:
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
  if m % tm != 0:
    raise NotImplementedError(f"{m=} must be divisible by tile size ({tm}).")

  tiles_m = m // tm
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
  # TODO: Invert 'partial_tile_mask' predicates to be more clear.
  partial_tile_mask = ((group_offsets[:-1] % tm) == 0) | (group_sizes == 0)

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
  active_group_mask = (iota <= end_group) & (iota >= start_group)
  group_tiles = jnp.where(active_group_mask, group_tiles, 0)
  num_tiles = group_tiles.sum()
  return (group_offsets, group_ids, m_tile_ids), num_tiles


def _get_store_mask(
    *,
    grid_id: jax.Array,
    group_metadata: GroupMetadata,
    tm: int,
    tn: int,
) -> jax.Array:
  """Mask for rows that belong to the current group in the current tile."""
  group_offsets, group_ids, m_tile_ids = group_metadata
  group_id = group_ids[grid_id]
  group_start = group_offsets[group_id]
  group_end = group_offsets[group_id + 1]
  m_id = m_tile_ids[grid_id] * tm
  iota = jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + m_id
  return (iota >= group_start) & (iota < group_end)


def _quantize_as(x, qdtype: jnp.dtype, axis: int, scale: float | None):
  info_fn = jnp.iinfo if jnp.dtype(qdtype).name.startswith("int") else jnp.finfo
  max_val = min(info_fn(qdtype).max, -info_fn(qdtype).min)
  if scale is None:
    scales = jnp.max(jnp.abs(x), axis=axis, keepdims=True) / jnp.array(
        max_val
    ).astype(jnp.bfloat16)
    inv_scales = jnp.broadcast_to(1.0 / scales, x.shape)
  else:  # compile-time (static) quantization scale
    scales, inv_scales = scale, 1.0 / scale
  return QArray(jnp.round(x * inv_scales).astype(qdtype), scales)


def _scale_out_by_scale(
    out: jax.Array, scales: jax.Array, axis: int | None = None
):
  if isinstance(scales, float):
    out *= scales
  else:
    if axis is None:  # scale all axes
      if any(
          s1 % s2 != 0 for s1, s2 in zip(out.shape, scales.shape, strict=True)
      ):
        raise ValueError(
            f"{scales=} cannot be broadcast to {out=} because the"
            " scales shape cannot be broadcast."
        )
      for ax, (s1, s2) in enumerate(zip(out.shape, scales.shape, strict=True)):
        scales = pltpu.repeat(scales, s1 // s2, ax)
      out *= scales
    else:
      out *= pltpu.repeat(scales, out.shape[axis] // scales.shape[axis], axis)
  return out


_TilingFn = Callable[[int, int, int], tuple[int, int, int] | None]


@functools.partial(
    jax.jit,
    static_argnames=[
        "precision",
        "out_dtype",
        "tiling",
        "transpose_rhs",
        "interpret",
        "input_buffer_count",
        "lhs_qdtype",
        "lhs_static_scale",
        "rhs_qdtype",
        "rhs_static_scale",
        "activation",
    ],
)
def gmm(
    lhs: jax.Array | QArray,
    rhs: jax.Array | QArray,
    group_sizes: jax.Array,
    precision: CanonicalPrecision,
    out_dtype: jnp.dtype,
    tiling: tuple[int, int, int] | _TilingFn | None = (128, 128, 128),
    input_buffer_count: int = 2,
    group_offset: jax.Array | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    # dynamic in kernel quantization support
    lhs_qdtype: jnp.dtype | None = None,
    lhs_static_scale: float | None = None,
    rhs_qdtype: jnp.dtype | None = None,
    rhs_static_scale: float | None = None,
    activation: base.ActivationFunction | None = None,
) -> jax.Array:
  """Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.

  Args:
    lhs: A 2d, jax.Array with shape [m, k].
    rhs: A 3d, jax.Array with shape [num_groups, k, n].
    precision: The precision to use for the dot operation.
    group_sizes: A 1d, jax.Array with shape [num_groups] and jnp.int32 dtype.
    out_dtype: jnp.dtype, the element type for the output matrix.
    tiling: 3-tuple of ints. The m, k and n-dimension tile sizes.
    input_buffer_count: The buffering count to use for the lhs input buffer.
    group_offset: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    transpose_rhs: True if the rhs needs to be transposed.
    interpret: Whether or not to run the kernel in interpret mode, helpful for
      testing and debugging.
    lhs_qdtype: Quant dtype to quantize lhs to if lhs is not already quantized.
    lhs_static_scale: Compile-time scale when quantizing lhs instead of
      computing it from lhs values.
    rhs_qdtype: Quant dtype to quantize rhs to if rhs is not already quantized.
    rhs_static_scale: Compile-time scale when quantizing rhs instead of
      computing it from rhs values.
    activation: Activation function to apply to the output of the dot
      operation.

  Returns:
    A 2d, jax.Array with shape [m, n].
  """
  lhs, rhs = map(quantization.as_array_or_qarray_without_zero_point, (lhs, rhs))
  group_sizes = _validate_args(lhs, rhs, group_sizes)

  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    if group_offset.shape:
      raise ValueError(
          f"group_offset must be a ()-shaped array. Got: {group_offset.shape}."
      )
    group_offset = group_offset[None]

  m, k = lhs.shape
  n = rhs.shape[1 if transpose_rhs else 2]

  if callable(tiling):
    tiling = tiling(m, k, n)

  if tiling is None:
    raise ValueError(f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")

  tm, tk, tn = tiling
  tiles_k = pl.cdiv(k, tk)
  tiles_n = pl.cdiv(n, tn)

  group_metadata, num_active_tiles = make_group_metadata(
      group_sizes=group_sizes,
      m=m,
      tm=tm,
      start_group=group_offset[0],
      num_nonzero_groups=rhs.shape[0],
      visit_empty_groups=False,
  )
  group_offsets, group_ids, _ = group_metadata

  def dot(x, y, preferred_element_type, *, precision=precision):
    # TODO: Pallas-MTPU doesn't support `DotAlgorithmPreset`.
    if precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
      x = x.astype(jnp.bfloat16)
      y = y.astype(jnp.bfloat16)
      precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
      preferred_element_type = jnp.float32

    rhs_contracting_dim = 1 if transpose_rhs else 0
    return jax.lax.dot_general(
        x,
        y,
        dimension_numbers=(((1,), (rhs_contracting_dim,)), ((), ())),
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

  def _maybe_get_subslice(x, idx: int, subslice_count: int | None, axis: int):
    """Returns a subslice of the given array along the given axis."""
    if subslice_count == 1:  # we're not quantized or full-channel
      return jax.tree.map(lambda x: x[...], x)
    assert axis in (0, 1)

    def getter(x):
      size = x.shape[axis] // subslice_count
      s, e = idx * size, (idx + 1) * size
      return x[s:e, :] if axis == 0 else x[:, s:e]

    return jax.tree.map(getter, x)

  def kernel(
      group_metadata,
      _,
      lhs_ref,
      rhs_ref,
      out_ref,
      acc_scratch,
      *,
      subchannel_iters: int,
  ):
    grid_id, k_i = pl.program_id(1), pl.program_id(2)

    @pl.when(k_i == 0)
    def _zero_acc():
      acc_scratch[...] = jnp.zeros_like(acc_scratch)

    def accum(is_last_k_tile):
      with jax.named_scope(f"accum:{is_last_k_tile=}"):
        scales_per_k_tile = subchannel_iters
        sc_tile = tk // scales_per_k_tile

        # compute the actual number of iterations for the last tile
        iterations = scales_per_k_tile
        if is_last_k_tile:
          iterations = pl.cdiv(k - tk * (tiles_k - 1), sc_tile)

        for it in range(iterations):
          # read the value of the input buffers
          lhs = _maybe_get_subslice(lhs_ref, it, scales_per_k_tile, 1)
          rhs_axis = 1 if transpose_rhs else 0
          rhs = _maybe_get_subslice(rhs_ref, it, scales_per_k_tile, rhs_axis)

          # optional dynamic quantization within the kernel
          if lhs_qdtype is not None and not isinstance(lhs, QArray):
            lhs = _quantize_as(lhs, lhs_qdtype, 1, lhs_static_scale)
          if rhs_qdtype is not None and not isinstance(rhs, QArray):
            rhs = _quantize_as(rhs, rhs_qdtype, rhs_axis, rhs_static_scale)

          # unpack quantized arrays for dot operation
          scales = []
          if isinstance(lhs, QArray):
            scales.append(lhs.scale)
            lhs = lhs.qvalue
          if isinstance(rhs, QArray):
            scales.append(rhs.scale.T if transpose_rhs else rhs.scale)
            rhs = rhs.qvalue

          # mask values if the current tile extends beyond reduction axis size
          is_last_subtile = (
              is_last_k_tile and (tiles_k - 1) * tk + (it + 1) * sc_tile >= k
          )
          if is_last_subtile and (k_rem := ((k % tk) % sc_tile)) != 0:
            iota = lambda x, d: lax.broadcasted_iota(jnp.int32, x.shape, d)
            lhs = jnp.where(iota(lhs, 1) < k_rem, lhs, 0)
            rhs = jnp.where(iota(rhs, rhs_axis) < k_rem, rhs, 0)

          # perform the dot operation
          is_int = lambda x: jnp.issubdtype(x.dtype, jnp.integer)
          acc_dtype = jnp.int32 if is_int(lhs) and is_int(rhs) else jnp.float32
          out = dot(lhs, rhs, acc_dtype)

          # apply scales to the output if the inputs were quantized
          for scale in scales:
            out = _scale_out_by_scale(out, scale)

          # accumulate and possibly store the output
          acc_scratch[...] += out.astype(acc_scratch.dtype)
          if is_last_subtile:
            mask = _get_store_mask(
                grid_id=grid_id, group_metadata=group_metadata, tm=tm, tn=tn
            )
            acc = acc_scratch[...]
            if activation is not None:
              acc = activation(acc)
            acc = jax.lax.select(mask, acc, out_ref[...].astype(acc.dtype))
            out_ref[...] = acc.astype(out_dtype)

    lax.cond(k_i == tiles_k - 1, lambda: accum(True), lambda: accum(False))

  def lhs_index_map(n_i, grid_id, k_i, group_metadata, group_offset):
    del n_i, group_offset  # Unused.
    # lhs is (m, k). Load the [tm, tk] matrix for this m-tile.
    _, _, m_tile_ids = group_metadata
    return m_tile_ids[grid_id], k_i

  def rhs_index_map(n_i, grid_id, k_i, group_metadata, group_offset):
    # rhs is (num_groups, k, n). Load the [tk, tn] matrix based on the group id
    # for this m-tile.
    _, group_ids, _ = group_metadata
    if transpose_rhs:
      k_i, n_i = n_i, k_i

    # NOTE: If we're working on only a shard of the rhs we need to adjust the
    # group index we load from to account for this. The group_ids are in the
    # "unsharded" domain.
    return group_ids[grid_id] - group_offset[0], k_i, n_i

  def out_index_map(n_i, grid_id, k_i, group_metadata, group_offset):
    del k_i, group_offset  # Unused.
    # out is (m, n). Load the [tm, tn] matrix for this m-tile.
    _, _, m_tile_ids = group_metadata
    return m_tile_ids[grid_id], n_i

  lhs_block_spec = pl.BlockSpec((tm, tk), lhs_index_map)
  if transpose_rhs:
    rhs_block_spec = pl.BlockSpec((None, tn, tk), rhs_index_map)
  else:
    rhs_block_spec = pl.BlockSpec((None, tk, tn), rhs_index_map)
  out_block_spec = pl.BlockSpec((tm, tn), out_index_map)

  subchannel_iters = 1
  if isinstance(lhs, QArray):
    lhs_eps = pl.cdiv(lhs.qvalue.shape[1], lhs.scale.shape[1])
    subchannel_iters = max(subchannel_iters, tk // lhs_eps)
    lhs, lhs_block_spec = common.quant_block_spec(lhs, lhs_block_spec, 1)
  if isinstance(rhs, QArray):
    axis = 2 if transpose_rhs else 1
    rhs_eps = pl.cdiv(rhs.qvalue.shape[axis], rhs.scale.shape[axis])
    subchannel_iters = max(subchannel_iters, tk // rhs_eps)
    rhs, rhs_block_spec = common.quant_block_spec(rhs, rhs_block_spec, axis)

  lhs_bytes = jax.tree.reduce(lambda acc, x: acc + x.size * x.itemsize, lhs, 0)
  if isinstance(rhs, QArray):
    rhs_bytes = (k * n) * rhs.qvalue.itemsize  # We don't read all of rhs
  else:
    rhs_bytes = k * n * rhs.itemsize

  out_bytes = (m * n) * jnp.dtype(out_dtype).itemsize
  bytes_accessed = lhs_bytes * tiles_n + rhs_bytes * group_ids.size + out_bytes
  cost_estimate = pl.CostEstimate(
      flops=2 * m * k * n, bytes_accessed=bytes_accessed, transcendentals=0
  )
  kernel_name = "gmm_megablox"
  if transpose_rhs:
    kernel_name += "_transpose_rhs"
  metadata = dict(
      prefer_element_type=jnp.dtype(out_dtype).name,
      tiling=dict(tile_m=tm, tile_k=tk, tile_n=tn),
      transpose_rhs=transpose_rhs,
      input_buffer_count=input_buffer_count,
  )
  call_gmm = common.custom_buffered_pallas_call(
      functools.partial(kernel, subchannel_iters=subchannel_iters),
      out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
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
      input_buffer_count=(input_buffer_count, 2),  # rhs always default buffered
  )

  with jax.named_scope(kernel_name):
    out = call_gmm(group_metadata, group_offset, lhs, rhs)

  if rhs.shape[0] < group_sizes.shape[0]:  # Need to zero uninitialized memory.
    group_start = group_offsets[group_offset[0]]
    group_end = group_offsets[group_offset[0] + rhs.shape[0]]
    row_idxs = jnp.arange(out.shape[0], dtype=jnp.int32)
    valid_mask = (row_idxs >= group_start) & (row_idxs < group_end)
    out = jnp.where(valid_mask[:, None], out, 0)
  return out


@functools.partial(
    jax.jit,
    static_argnames=[
        "precision",
        "out_dtype",
        "tiling",
        "num_actual_groups",
        "interpret",
        "input_buffer_count",
        "lhs_qdtype",
        "lhs_static_scale",
        "rhs_qdtype",
        "rhs_static_scale",
        "activation",
    ],
)
def tgmm(
    lhs: jax.Array | QArray,
    rhs: jax.Array | QArray,
    group_sizes: jax.Array,
    precision: CanonicalPrecision,
    out_dtype: jnp.dtype,
    tiling: tuple[int, int, int] | _TilingFn | None = (128, 128, 128),
    input_buffer_count: int = 2,
    group_offset: jax.Array | None = None,
    num_actual_groups: int | None = None,
    interpret: bool = False,
    # dynamic in kernel quantization support
    lhs_qdtype: jnp.dtype | None = None,
    lhs_static_scale: float | None = None,
    rhs_qdtype: jnp.dtype | None = None,
    rhs_static_scale: float | None = None,
    activation: base.ActivationFunction | None = None,
) -> jax.Array:
  """Compute lhs[:, sizes[i-1]:sizes[i]] @ rhs[sizes[i-1]:sizes[i], :].

  Args:
    lhs: A 2d, jax.Array with shape [k, m].
    rhs: A 2d, jax.Array with shape [m, n].
    group_sizes: A 1d, jax.Array with shape [num_groups] and jnp.int32 dtype.
    precision: The precision to use for the dot operation.
    out_dtype: jnp.dtype, the element type for the output matrix.
    tiling: 3-tuple of ints. The m, k and n-dimension tile sizes.
    input_buffer_count: The buffering count to use for inputs lhs and rhs.
    group_offset: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    num_actual_groups: For when num_groups is sharded and we should only compute
      the groups that are local, starting from group_offset.
    interpret: Whether or not to run the kernel in interpret mode, helpful for
      testing and debugging.
    lhs_qdtype: Quant dtype to quantize lhs to if lhs is not already quantized.
    lhs_static_scale: Compile-time scale when quantizing lhs instead of
      computing it from lhs values.
    rhs_qdtype: Quant dtype to quantize rhs to if rhs is not already quantized.
    rhs_static_scale: Compile-time scale when quantizing rhs instead of
      computing it from rhs values.
    activation: Activation function to apply to the output of the dot
      operation.

  Returns:
    A  3d, jax.Array with shape [num_groups, k, n].
  """
  lhs, rhs = map(quantization.as_array_or_qarray_without_zero_point, (lhs, rhs))
  group_sizes = _validate_args(lhs, rhs, group_sizes, expected_rhs_dims=2)

  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    group_offset = group_offset[None]

  k, m = lhs.shape
  n = rhs.shape[1]
  # the general tgmm definition requires lhs @ rhs
  # but our memory pipeline loads (m, k), (m, n) and computes (m, k)^T @ (m, n)
  lhs = jax.tree.map(lambda x: x.mT, lhs)

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
  tiles_k = pl.cdiv(k, tk)
  tiles_n = pl.cdiv(n, tn)

  group_metadata, num_active_tiles = make_group_metadata(
      group_sizes=group_sizes,
      m=m,
      tm=tm,
      start_group=group_offset[0],
      num_nonzero_groups=num_actual_groups,
      visit_empty_groups=True,
  )

  def dot(x, y, preferred_element_type, *, precision=precision):
    # TODO: Pallas-MTPU doesn't support `DotAlgorithmPreset`.
    if precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
      x = x.astype(jnp.bfloat16)
      y = y.astype(jnp.bfloat16)
      precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
      preferred_element_type = jnp.float32

    return jax.lax.dot(
        x,
        y,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

  def kernel(
      group_metadata,
      _,
      lhs_ref,
      rhs_ref,
      out_ref,
      acc_scratch,
      *,
      subchannel_iters: int,
  ):
    if subchannel_iters != 1:
      raise NotImplementedError(
          "subchannel_iters != 1 not supported yet in tgmm."
      )
    grid_id = pl.program_id(2)
    group_offsets, group_ids, _ = group_metadata
    group = group_ids[grid_id]
    prev_group = group_ids[jnp.where(grid_id > 0, grid_id - 1, 0)]

    @pl.when((grid_id == 0) | (group != prev_group))  # Group has changed.
    def _zero_acc():
      acc_scratch[...] = jnp.zeros_like(acc_scratch)

    group_size = group_offsets[group + 1] - group_offsets[group]

    @pl.when(group_size > 0)
    def _do():
      # load lhs and rhs
      lhs = jax.tree.map(lambda x: x[...], lhs_ref)
      rhs = jax.tree.map(lambda x: x[...], rhs_ref)

      # optional dynamic quantization within the kernel
      if lhs_qdtype is not None and not isinstance(lhs, QArray):
        lhs = _quantize_as(lhs, lhs_qdtype, axis=1, scale=lhs_static_scale)
      if rhs_qdtype is not None and not isinstance(rhs, QArray):
        rhs = _quantize_as(rhs, rhs_qdtype, axis=1, scale=rhs_static_scale)

      # unpack quantized arrays for dot operation
      scales = []
      if isinstance(lhs, QArray):
        scales.append(lhs.scale.T)
        lhs = lhs.qvalue
      if isinstance(rhs, QArray):
        scales.append(rhs.scale)
        rhs = rhs.qvalue

      kwargs = dict(grid_id=grid_id, group_metadata=group_metadata, tm=tm)
      lhs = jnp.where(_get_store_mask(**kwargs, tn=tk), lhs, 0)
      rhs = jnp.where(_get_store_mask(**kwargs, tn=tn), rhs, 0)

      is_int = lambda x: jnp.issubdtype(x.dtype, jnp.integer)
      acc_dtype = jnp.int32 if is_int(lhs) and is_int(rhs) else jnp.float32
      out = dot(lhs.T, rhs, acc_dtype)

      # apply scales to the output if the inputs were quantized
      for scale in scales:
        out = _scale_out_by_scale(out, scale)

      acc_scratch[...] += out.astype(acc_scratch.dtype)

    is_end_of_grid = grid_id == (pl.num_programs(2) - 1)
    next_group = group_ids[jnp.where(is_end_of_grid, grid_id, grid_id + 1)]

    @pl.when(is_end_of_grid | (group != next_group))
    def _store_accum():
      acc = acc_scratch[...]
      if activation is not None:
        acc = activation(acc)
      out_ref[...] = acc.astype(out_dtype)

  def lhs_index_map(n_i, k_i, grid_id, group_metadata, group_offset):
    del n_i, group_offset  # Unused.
    # lhs is (m, k). Load the [tm, tk] matrix for this m-tile.
    _, _, m_tile_ids = group_metadata
    return m_tile_ids[grid_id], k_i

  def rhs_index_map(n_i, k_i, grid_id, group_metadata, group_offset):
    del k_i, group_offset  # Unused.
    # rhs is (m, n). Load the [tm, tn] matrix for this m-tile.
    _, _, m_tile_ids = group_metadata
    return m_tile_ids[grid_id], n_i

  def out_index_map(n_i, k_i, grid_id, group_metadata, group_offset):
    # out is (num_groups, k, n). Load the [tk, tn] matrix based on the group id
    # for this m-tile.
    _, group_ids, _ = group_metadata
    # NOTE: If we're working on only a shard of the output we need to adjust the
    # group index we load from to account for this. The group_ids are in the
    # "unsharded" domain.
    return group_ids[grid_id] - group_offset[0], k_i, n_i

  lhs_block_spec = pl.BlockSpec((tm, tk), lhs_index_map)
  rhs_block_spec = pl.BlockSpec((tm, tn), rhs_index_map)
  out_block_spec = pl.BlockSpec((None, tk, tn), out_index_map)

  subchannel_iters = 1
  if isinstance(lhs, QArray):
    lhs_eps = pl.cdiv(lhs.qvalue.shape[0], lhs.scale.shape[0])
    subchannel_iters = max(subchannel_iters, tk // lhs_eps)
    lhs, lhs_block_spec = common.quant_block_spec(lhs, lhs_block_spec, 0)
  if isinstance(rhs, QArray):
    rhs_eps = pl.cdiv(rhs.qvalue.shape[0], rhs.scale.shape[0])
    subchannel_iters = max(subchannel_iters, tk // rhs_eps)
    rhs, rhs_block_spec = common.quant_block_spec(rhs, rhs_block_spec, 0)

  lhs_bytes = jax.tree.reduce(lambda acc, x: acc + x.size * x.itemsize, lhs, 0)
  rhs_bytes = jax.tree.reduce(lambda acc, x: acc + x.size * x.itemsize, rhs, 0)
  out_bytes = (num_actual_groups * k * n) * jnp.dtype(out_dtype).itemsize
  bytes_accessed = (lhs_bytes * tiles_n) + (rhs_bytes * tiles_k) + out_bytes
  cost_estimate = pl.CostEstimate(
      flops=2 * m * k * n, bytes_accessed=bytes_accessed, transcendentals=0
  )

  kernel_name = "tgmm_megablox"
  metadata = dict(
      tiling=dict(tile_m=tm, tile_k=tk, tile_n=tn),
      prefer_element_type=jnp.dtype(out_dtype).name,
      num_actual_groups=num_actual_groups,
      input_buffer_count=input_buffer_count,
  )
  call_gmm = common.custom_buffered_pallas_call(
      functools.partial(kernel, subchannel_iters=subchannel_iters),
      out_shape=jax.ShapeDtypeStruct((num_actual_groups, k, n), out_dtype),
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
      input_buffer_count=(input_buffer_count, input_buffer_count),
  )

  with jax.named_scope(kernel_name):
    return call_gmm(group_metadata, group_offset, lhs, rhs)
