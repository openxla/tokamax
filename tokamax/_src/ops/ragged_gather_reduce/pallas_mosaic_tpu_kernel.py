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
"""Destination-major SparseCore Pallas/Mosaic kernel for Ragged Gather Reduce.

Ported from vllm-torchtpu's `ragged_gather_reduce_v3`.

Each SparseCore vector subcore owns a column slice of the hidden dimension and
a contiguous range of 64-token destination blocks. Per block, the wrapper
compacts the valid routes to the front (keeping the routes of each token
adjacent), and the kernel gathers every valid route's row slice exactly once,
scatter-adds the weighted rows into a float32 VMEM accumulator, and writes the
block back as full, tile-aligned bf16 tiles. Column slices too wide for the
subcore's VMEM are processed in narrower column chunks.
"""

import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp

# Destination tokens per vector op; must equal the SparseCore lane count.
TOKEN_SUBCHUNK = 16
NUM_TOKEN_SUBCHUNKS = 4
# Destination tokens accumulated per pipeline step.
TOKEN_BLOCK = TOKEN_SUBCHUNK * NUM_TOKEN_SUBCHUNKS
# A route's metadata word holds its source row in the low `ROUTE_INDEX_BITS`
# bits and its destination token within the block above them.
ROUTE_INDEX_BITS = 20
ROUTE_INDEX_MASK = (1 << ROUTE_INDEX_BITS) - 1
# Most rows of `x` the route metadata can address.
MAX_SOURCE_ROWS = 1 << ROUTE_INDEX_BITS


def default_num_column_partitions(
    hidden_size: int, num_cores: int, num_lanes: int
) -> int:
  """Upstream heuristic for the number of hidden-dimension partitions.

  Args:
    hidden_size: Width of `x`.
    num_cores: Total number of SparseCore vector subcores.
    num_lanes: TensorCore lane count; columns are split in multiples of it.

  Returns:
    The number of column partitions, a power of two dividing `num_cores`.
  """
  preferred_num_stages = 4
  num_column_partitions = 1
  while (
      num_cores % (num_column_partitions * 2) == 0
      and hidden_size % (num_lanes * num_column_partitions * 2) == 0
      and hidden_size // (num_column_partitions * 2 * num_lanes)
      >= preferred_num_stages
  ):
    num_column_partitions *= 2
  # Destination-major accumulation keeps a dense FP32 token block in VMEM.
  # For columns up to 1024, using one fewer column partition amortizes the
  # per-block metadata, clear, and output-DMA overhead while still leaving
  # enough room for the double-buffered indirect gather. Wider columns stay
  # with the conservative partitioning selected above.
  if num_column_partitions >= 2 and hidden_size <= 1024 * (
      num_column_partitions // 2
  ):
    num_column_partitions //= 2
  return num_column_partitions


def num_sparse_cores(tpu_info: Any) -> int:
  """Total number of SparseCore vector subcores on the chip."""
  sc_info = tpu_info.sparse_core
  return sc_info.num_cores * sc_info.num_subcores


def partition_counts(
    hidden_size: int,
    tpu_info: Any,
    num_column_partitions: int | None = None,
) -> tuple[int, int]:
  """Column and row partition counts of the SparseCore subcores for a width."""
  num_cores = num_sparse_cores(tpu_info)
  if num_column_partitions is None:
    num_column_partitions = default_num_column_partitions(
        hidden_size, num_cores, tpu_info.num_lanes
    )
  return num_column_partitions, num_cores // num_column_partitions


def token_block_alignment(
    hidden_size: int,
    tpu_info: Any,
    num_column_partitions: int | None = None,
) -> int:
  """Destination-token granularity: one 64-token block per row partition."""
  _, num_row_partitions = partition_counts(
      hidden_size, tpu_info, num_column_partitions
  )
  return num_row_partitions * TOKEN_BLOCK


def is_valid_num_column_partitions(
    num_column_partitions: int, hidden_size: int, tpu_info: Any
) -> bool:
  """Whether the kernel can split `hidden_size` into that many partitions."""
  return (
      num_column_partitions > 0
      and num_sparse_cores(tpu_info) % num_column_partitions == 0
      and hidden_size % (num_column_partitions * tpu_info.num_lanes) == 0
  )


def subcore_vmem_bytes(
    col_chunk_size: int, reduce_group_size: int, tpu_info: Any
) -> int:
  """Upper bound on the VMEM one vector subcore allocates for the kernel.

  The SparseCore subcores' VMEM is carved out of the SparseCore's shared
  memory, which every subcore allocates the same buffers from, so the kernel
  only fits if this is at most `tpu_info.sparse_core.vmem_capacity_bytes`.

  Args:
    col_chunk_size: Width of the column chunk the buffers hold.
    reduce_group_size: Routes per destination token.
    tpu_info: The TPU info.

  Returns:
    The bytes of the 32-bit scratch and pipeline buffers.
  """
  tile_words = tpu_info.num_sublanes * tpu_info.num_lanes
  round_up_to_tiles = lambda words: pl.cdiv(words, tile_words) * tile_words
  words = (
      # Float32 accumulator of one destination block.
      TOKEN_BLOCK * col_chunk_size
      # Two row-packed bf16 output tiles.
      + 2 * (TOKEN_SUBCHUNK // 2) * col_chunk_size
      # Double-buffered indirect gather of 16 int32 row pairs.
      + 2 * TOKEN_SUBCHUNK * col_chunk_size
      # Double-buffered route metadata, weights and counts of one block;
      # rounded up to whole tiles to bound their padding.
      + 2
      * (
          2 * round_up_to_tiles(TOKEN_BLOCK * reduce_group_size)
          + round_up_to_tiles(TOKEN_SUBCHUNK)
      )
  )
  return 4 * words


def select_col_chunk_size(
    col_size: int, reduce_group_size: int, tpu_info: Any
) -> int | None:
  """Widest column chunk of a `col_size` column slice that fits subcore VMEM.

  Args:
    col_size: Width of a subcore's column slice, a multiple of the TensorCore
      lane count.
    reduce_group_size: Routes per destination token.
    tpu_info: The TPU info.

  Returns:
    The widest divisor of `col_size` that is a multiple of the TensorCore lane
    count and whose buffers fit `tpu_info.sparse_core.vmem_capacity_bytes`, or
    `None` if there is none.
  """
  num_lanes = tpu_info.num_lanes
  vmem_capacity_bytes = tpu_info.sparse_core.vmem_capacity_bytes
  for num_col_chunks in range(1, col_size // num_lanes + 1):
    chunk_size, remainder = divmod(col_size, num_col_chunks)
    if (
        not remainder
        and chunk_size % num_lanes == 0
        and subcore_vmem_bytes(chunk_size, reduce_group_size, tpu_info)
        <= vmem_capacity_bytes
    ):
      return chunk_size
  return None


def get_unsupported_reason(
    num_rows: int,
    hidden_size: int,
    dtype: jax.typing.DTypeLike,
    reduce_group_size: int,
) -> str | None:
  """Returns why the kernel cannot run on `x`, or `None` if it can."""
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  if sc_info is None:
    return "SparseCore is not available."
  if sc_info.num_lanes != TOKEN_SUBCHUNK:
    return (
        f"Expected {TOKEN_SUBCHUNK} SparseCore lanes, got {sc_info.num_lanes}."
    )
  if dtype != jnp.bfloat16:
    return f"Only bfloat16 is supported, got {dtype}."
  if hidden_size % tpu_info.num_lanes:
    return (
        f"hidden_size ({hidden_size}) must be a multiple of"
        f" {tpu_info.num_lanes}."
    )
  # Rows are padded to an even count; see `ragged_gather_reduce`.
  if num_rows + num_rows % 2 > MAX_SOURCE_ROWS:
    return f"At most {MAX_SOURCE_ROWS} rows are supported, got {num_rows}."
  # The narrowest column chunk fits unless the per-block route metadata, which
  # scales with `reduce_group_size`, does not.
  if (
      select_col_chunk_size(tpu_info.num_lanes, reduce_group_size, tpu_info)
      is None
  ):
    return (
        f"reduce_group_size ({reduce_group_size}) is too large for the route"
        " metadata to fit SparseCore VMEM."
    )
  return None


def main_kernel(
    x_hbm_ref: Any,
    route_metadata_hbm_ref: Any,
    route_weights_hbm_ref: Any,
    route_counts_hbm_ref: Any,
    out_hbm_ref: Any,
    accum_vmem_ref: Any,
    out_tile_vmem_ref: Any,
    sem_ref: Any,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_column_partitions: int,
    num_col_chunks: int,
    blocks_per_row_partition: int,
    reduce_group_size: int,
):
  """Reduces `blocks_per_row_partition` destination blocks on one subcore.

  The subcore's column slice is reduced in `num_col_chunks` column chunks as
  wide as the accumulator.
  """
  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None
  assert sc_info.num_lanes == TOKEN_SUBCHUNK

  core_id = jax.lax.axis_index((core_axis_name, subcore_axis_name))
  row_partition_id = core_id // num_column_partitions
  col_partition_id = core_id % num_column_partitions
  col_chunk_size = accum_vmem_ref.shape[-1]
  col_start = col_partition_id * num_col_chunks * col_chunk_size
  metadata_block_size = reduce_group_size * TOKEN_BLOCK
  # bf16 rows 2r and 2r + 1 are the low and high halves of int32 row r.
  x_32b_hbm_ref = x_hbm_ref.bitcast(jnp.int32)
  out_32b_hbm_ref = out_hbm_ref.bitcast(jnp.uint32)

  def metadata_block_index(block_id):
    return (row_partition_id * blocks_per_row_partition + block_id,)

  @functools.partial(
      pltpu.emit_pipeline,
      grid=(blocks_per_row_partition,),
      in_specs=(
          pl.BlockSpec((metadata_block_size,), metadata_block_index),
          pl.BlockSpec((metadata_block_size,), metadata_block_index),
          pl.BlockSpec((TOKEN_SUBCHUNK,), metadata_block_index),
      ),
      out_specs=(),
  )
  def token_block_pipeline(
      route_metadata_ref, route_weights_ref, route_counts_ref, output_sem_ref
  ):
    block_id = pl.program_id(0)
    global_block_id = row_partition_id * blocks_per_row_partition + block_id
    valid_route_count = route_counts_ref[...][0]
    # The wrapper only reports exactly two routes per token when the routes of
    # each token are adjacent; see `_route_metadata`.
    fixed_two_routes = valid_route_count == 2 * TOKEN_BLOCK
    num_route_chunks = pl.cdiv(valid_route_count, TOKEN_SUBCHUNK)

    def reduce_col_chunk(col_chunk_id):
      chunk_start = col_start + col_chunk_id * col_chunk_size

      # Scatter-add below reads the previous accumulator value, so initialize
      # the complete dense destination block first. This also supplies zeros
      # for tokens with no local route.
      def zero_column(col_offset):
        col_slice = pl.ds(col_offset, TOKEN_SUBCHUNK)
        zero = jnp.zeros((TOKEN_SUBCHUNK,), jnp.float32)
        for token_row in range(TOKEN_BLOCK):
          accum_vmem_ref[token_row, col_slice] = zero

      plsc.parallel_loop(0, col_chunk_size, step=TOKEN_SUBCHUNK)(zero_column)

      def route_indices_slice(route_chunk):
        start = route_chunk * TOKEN_SUBCHUNK
        metadata = route_metadata_ref[pl.ds(start, TOKEN_SUBCHUNK)]
        return jnp.bitwise_and(metadata, ROUTE_INDEX_MASK)

      @functools.partial(
          pltpu.emit_pipeline,
          grid=(num_route_chunks,),
          in_specs=pl.BlockSpec(
              (pl.Indirect(TOKEN_SUBCHUNK), col_chunk_size),
              lambda route_chunk: (
                  jnp.bitwise_right_shift(route_indices_slice(route_chunk), 1),
                  col_partition_id * num_col_chunks + col_chunk_id,
              ),
          ),
          out_specs=(),
      )
      def route_pipeline(gather_ref):
        route_chunk = pl.program_id(0)
        metadata_start = route_chunk * TOKEN_SUBCHUNK
        metadata_slice = pl.ds(metadata_start, TOKEN_SUBCHUNK)
        route_metadata = route_metadata_ref[metadata_slice]
        source_indices = jnp.bitwise_and(route_metadata, ROUTE_INDEX_MASK)
        route_destinations = jnp.bitwise_right_shift(
            route_metadata, ROUTE_INDEX_BITS
        )
        route_weights = route_weights_ref[metadata_slice]

        def weighted_route(route_lane, col_slice):
          # Moves the route's bf16 half of the gathered int32 word into the
          # high bits, which makes it a float32 of the same value.
          value_i32 = gather_ref[route_lane, col_slice]
          shift = jnp.where(
              jnp.bitwise_and(source_indices[route_lane], 1) == 0, 16, 0
          )
          shifted = jnp.bitwise_and(
              jnp.left_shift(value_i32, shift),
              jnp.int32(-65536),  # 0xFFFF0000
          )
          value_f32 = plsc.bitcast(shifted, jnp.float32)
          return value_f32 * route_weights[route_lane]

        def fixed_two_route_column_loop(col_offset):
          col_slice = pl.ds(col_offset, TOKEN_SUBCHUNK)
          feature_indices = col_offset + jnp.arange(
              TOKEN_SUBCHUNK, dtype=jnp.int32
          )
          weighted_values = [
              weighted_route(route_lane, col_slice)
              for route_lane in range(TOKEN_SUBCHUNK)
          ]
          # Routes 2i and 2i + 1 share a destination: add them before the
          # scatter.
          for route_lane in range(0, TOKEN_SUBCHUNK, 2):
            destination_indices = jnp.full_like(
                feature_indices, route_destinations[route_lane]
            )
            plsc.addupdate_scatter(
                accum_vmem_ref,
                (destination_indices, feature_indices),
                weighted_values[route_lane] + weighted_values[route_lane + 1],
            )

        def generic_column_loop(col_offset):
          col_slice = pl.ds(col_offset, TOKEN_SUBCHUNK)
          feature_indices = col_offset + jnp.arange(
              TOKEN_SUBCHUNK, dtype=jnp.int32
          )
          route_batch_size = 4
          for route_batch_start in range(0, TOKEN_SUBCHUNK, route_batch_size):
            route_lanes = range(
                route_batch_start, route_batch_start + route_batch_size
            )
            weighted_values = [
                weighted_route(route_lane, col_slice)
                for route_lane in route_lanes
            ]
            for route_lane, weighted_value in zip(route_lanes, weighted_values):
              destination_indices = jnp.full_like(
                  feature_indices, route_destinations[route_lane]
              )
              route_valid = metadata_start + route_lane < valid_route_count
              scatter_mask = jnp.broadcast_to(route_valid, (TOKEN_SUBCHUNK,))
              plsc.addupdate_scatter(
                  accum_vmem_ref,
                  (destination_indices, feature_indices),
                  weighted_value,
                  mask=scatter_mask,
              )

        @pl.when(fixed_two_routes)
        def _run_fixed_two_route_path():
          plsc.parallel_loop(0, col_chunk_size, step=TOKEN_SUBCHUNK)(
              fixed_two_route_column_loop
          )

        @pl.when(jnp.logical_not(fixed_two_routes))
        def _run_generic_path():
          plsc.parallel_loop(0, col_chunk_size, step=TOKEN_SUBCHUNK)(
              generic_column_loop
          )

      route_pipeline(x_32b_hbm_ref)

      # Convert pairs of FP32 destination rows to row-packed BF16 and write
      # four complete, tile-aligned 16-row output tiles.
      for token_subchunk_pair in range(NUM_TOKEN_SUBCHUNKS // 2):
        copies = []
        for output_buffer in range(2):
          token_subchunk = token_subchunk_pair * 2 + output_buffer
          accum_row_start = token_subchunk * TOKEN_SUBCHUNK

          def cast_column(
              col_offset,
              output_buffer=output_buffer,
              accum_row_start=accum_row_start,
          ):
            col_slice = pl.ds(col_offset, TOKEN_SUBCHUNK)
            for token_lane in range(0, TOKEN_SUBCHUNK, 2):
              accum_row = accum_row_start + token_lane
              packed_bf16 = plsc.pack(
                  accum_vmem_ref[accum_row, col_slice],
                  accum_vmem_ref[accum_row + 1, col_slice],
                  format=plsc.PackFormat.INTERLEAVED,
              )
              out_tile_vmem_ref[output_buffer, token_lane // 2, col_slice] = (
                  plsc.bitcast(packed_bf16, jnp.uint32)
              )

          plsc.parallel_loop(0, col_chunk_size, step=TOKEN_SUBCHUNK)(
              cast_column
          )

          output_row = (
              global_block_id * TOKEN_BLOCK + token_subchunk * TOKEN_SUBCHUNK
          )
          output_row_packed = pl.multiple_of(
              output_row // 2, TOKEN_SUBCHUNK // 2
          )
          copy = pltpu.make_async_copy(
              out_tile_vmem_ref.at[output_buffer],
              out_32b_hbm_ref.at[
                  pl.ds(output_row_packed, TOKEN_SUBCHUNK // 2),
                  pl.ds(chunk_start, col_chunk_size),
              ],
              output_sem_ref.at[output_buffer],
          )
          copy.start()
          copies.append(copy)
        for copy in copies:
          copy.wait()

    if num_col_chunks == 1:
      reduce_col_chunk(0)
    else:
      # Column chunks reuse the accumulator and output tiles; the output DMAs
      # of a chunk complete before the next chunk clears the accumulator.
      jax.lax.fori_loop(
          0,
          num_col_chunks,
          lambda col_chunk_id, _: reduce_col_chunk(col_chunk_id),
          None,
      )

  token_block_pipeline(
      route_metadata_hbm_ref,
      route_weights_hbm_ref,
      route_counts_hbm_ref,
      scratches=(sem_ref,),
  )


def _route_metadata(
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    *,
    num_tokens: int,
    padded_num_tokens: int,
    reduce_group_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Compacts the valid routes of every 64-token destination block.

  Args:
    indices: `(num_tokens * reduce_group_size,)` source row of each route.
    topk_weights: `(num_tokens * reduce_group_size,)` weight of each route.
    valid_rows_mask: `(num_tokens * reduce_group_size,)` bool route validity.
    num_tokens: Number of destination tokens.
    padded_num_tokens: `num_tokens` rounded up to a whole number of blocks per
      row partition. Padding tokens have no valid route.
    reduce_group_size: Routes per destination token.

  Returns:
    `(route_metadata, route_weights, route_counts)`. `route_metadata` and
    `route_weights` hold `TOKEN_BLOCK * reduce_group_size` entries per block,
    valid routes first: int32 `source_row | (destination << ROUTE_INDEX_BITS)`
    and float32 weights, both zero for invalid routes. `route_counts` holds
    `TOKEN_SUBCHUNK` int32 copies of the number of routes the kernel processes
    in each block.
  """
  token_padding = padded_num_tokens - num_tokens
  indices_2d = indices.reshape(num_tokens, reduce_group_size)
  weights_2d = topk_weights.reshape(num_tokens, reduce_group_size)
  valid_2d = valid_rows_mask.reshape(num_tokens, reduce_group_size)
  if token_padding:
    indices_2d = jnp.pad(indices_2d, ((0, token_padding), (0, 0)))
    weights_2d = jnp.pad(weights_2d, ((0, token_padding), (0, 0)))
    valid_2d = jnp.pad(valid_2d, ((0, token_padding), (0, 0)))

  num_blocks = padded_num_tokens // TOKEN_BLOCK
  metadata_block_size = TOKEN_BLOCK * reduce_group_size
  indices_blocks = indices_2d.reshape(num_blocks, metadata_block_size)
  valid_blocks = valid_2d.reshape(num_blocks, metadata_block_size)
  weights_blocks = weights_2d.reshape(num_blocks, metadata_block_size)
  destination_template = jnp.repeat(
      jnp.arange(TOKEN_BLOCK, dtype=jnp.int32), reduce_group_size
  )
  destination_blocks = jnp.broadcast_to(
      destination_template, indices_blocks.shape
  )

  # A stable boolean partition keeps routes for each destination adjacent
  # while placing the exact valid prefix first in every destination block.
  route_order = jnp.argsort(~valid_blocks, axis=1, stable=True)
  compact_valid = jnp.take_along_axis(valid_blocks, route_order, axis=1)
  route_indices = jnp.take_along_axis(indices_blocks, route_order, axis=1)
  route_destinations = jnp.take_along_axis(
      destination_blocks, route_order, axis=1
  )
  route_weights = jnp.take_along_axis(weights_blocks, route_order, axis=1)
  route_indices = jnp.where(compact_valid, route_indices, 0).astype(jnp.int32)
  route_destinations = jnp.where(compact_valid, route_destinations, 0).astype(
      jnp.int32
  )
  route_metadata = jnp.bitwise_or(
      route_indices, jnp.left_shift(route_destinations, ROUTE_INDEX_BITS)
  ).reshape(-1)
  route_weights = jnp.where(compact_valid, route_weights, 0).reshape(-1)
  route_weights = route_weights.astype(jnp.float32)

  route_counts = jnp.sum(valid_blocks, axis=1, dtype=jnp.int32)
  full_two_route_blocks = route_counts == 2 * TOKEN_BLOCK
  if metadata_block_size >= 2 * TOKEN_BLOCK:
    paired_destinations = route_destinations[:, : 2 * TOKEN_BLOCK].reshape(
        num_blocks, TOKEN_BLOCK, 2
    )
    adjacent_pairs = jnp.all(
        paired_destinations[:, :, 0] == paired_destinations[:, :, 1], axis=1
    )
    # Count 128 selects the adjacent-pair fast path in the SC kernel. If a
    # full two-route block does not actually have adjacent equal
    # destinations, expose one already-zero compacted slot as a harmless
    # sentinel so the block takes the generic path instead.
    needs_zero_weight_sentinel = jnp.logical_and(
        full_two_route_blocks, jnp.logical_not(adjacent_pairs)
    )
    route_counts += needs_zero_weight_sentinel.astype(jnp.int32)
  route_counts = jnp.broadcast_to(
      route_counts[:, None], (num_blocks, TOKEN_SUBCHUNK)
  ).reshape(-1)
  return route_metadata, route_weights, route_counts


@functools.partial(
    jax.jit, static_argnames=("reduce_group_size", "num_column_partitions")
)
def ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
    *,
    num_column_partitions: int | None = None,
) -> jax.Array:
  """Destination-major SparseCore Ragged Gather Reduce.

  See `reference.ragged_gather_reduce` for the semantics.

  Args:
    x: `(num_rows, hidden_size)` bf16 expert outputs. `num_rows` is at most
      `MAX_SOURCE_ROWS` and `hidden_size` a multiple of the TensorCore lane
      count.
    indices: `(input_size,)` int32 row of `x` read by each route.
    topk_weights: `(input_size,)` weight of each route.
    valid_rows_mask: `(input_size,)` bool. Routes with a false entry are
      skipped.
    reduce_group_size: Routes summed into one output token. Must divide
      `input_size`.
    num_column_partitions: Number of hidden-dimension slices the SparseCore
      subcores split `x` into; the remaining factor of the subcore count splits
      the destination tokens. Must divide the subcore count, and `hidden_size`
      must be a multiple of it times the TensorCore lane count. Defaults to
      `default_num_column_partitions`.

  Returns:
    `(input_size // reduce_group_size, hidden_size)` bf16, accumulated in
    float32.

  Raises:
    ValueError: If the inputs or `num_column_partitions` are unsupported.
  """
  num_rows, hidden_size = x.shape
  if reason := get_unsupported_reason(
      num_rows, hidden_size, x.dtype, reduce_group_size
  ):
    raise ValueError(reason)
  input_size = indices.size
  if reduce_group_size <= 0 or input_size % reduce_group_size:
    raise ValueError(
        f"{input_size=} must be divisible by a positive {reduce_group_size=}."
    )
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  if num_column_partitions is not None and not is_valid_num_column_partitions(
      num_column_partitions, hidden_size, tpu_info
  ):
    raise ValueError(
        f"num_column_partitions ({num_column_partitions}) must divide the"
        f" SparseCore subcore count ({num_sparse_cores(tpu_info)}) and"
        f" hidden_size ({hidden_size}) must be a multiple of it times"
        f" {tpu_info.num_lanes}."
    )
  num_column_partitions, num_row_partitions = partition_counts(
      hidden_size, tpu_info, num_column_partitions
  )

  # SparseCore row-packing views BF16 as pairs of rows. Fewer than a tile of
  # row pairs are also padded to one: XLA tiles them more narrowly, and the
  # SparseCore compiler cannot gather a single 128-column block of those.
  padded_num_rows = max(num_rows + num_rows % 2, 2 * tpu_info.num_sublanes)
  if padded_num_rows != num_rows:
    x = jnp.pad(x, ((0, padded_num_rows - num_rows), (0, 0)))

  num_tokens = input_size // reduce_group_size
  token_alignment = num_row_partitions * TOKEN_BLOCK
  padded_num_tokens = pl.cdiv(num_tokens, token_alignment) * token_alignment
  route_metadata, route_weights, route_counts = _route_metadata(
      indices,
      topk_weights,
      valid_rows_mask,
      num_tokens=num_tokens,
      padded_num_tokens=padded_num_tokens,
      reduce_group_size=reduce_group_size,
  )

  col_size = hidden_size // num_column_partitions
  # Wide column slices would overflow the SparseCore memory the subcores'
  # VMEM is allocated from, so they are reduced in narrower chunks.
  col_chunk_size = select_col_chunk_size(col_size, reduce_group_size, tpu_info)
  assert col_chunk_size is not None  # Checked by `get_unsupported_reason`.
  blocks_per_row_partition = padded_num_tokens // token_alignment
  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  out = pl.kernel(
      functools.partial(
          main_kernel,
          core_axis_name=vector_mesh.core_axis_name,
          subcore_axis_name=vector_mesh.subcore_axis_name,
          num_column_partitions=num_column_partitions,
          num_col_chunks=col_size // col_chunk_size,
          blocks_per_row_partition=blocks_per_row_partition,
          reduce_group_size=reduce_group_size,
      ),
      out_type=jax.ShapeDtypeStruct(
          (padded_num_tokens, hidden_size), jnp.bfloat16
      ),
      scratch_types=(
          # FP32 accumulator of one destination block.
          pltpu.VMEM((TOKEN_BLOCK, col_chunk_size), jnp.float32),
          # Two row-packed bf16 output tiles, double-buffered for the DMA.
          pltpu.VMEM((2, TOKEN_SUBCHUNK // 2, col_chunk_size), jnp.uint32),
          pltpu.SemaphoreType.DMA((2,)),
      ),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          disable_bounds_checks=True,
          needs_layout_passes=False,
      ),
      mesh=vector_mesh,
      name="sc_ragged_gather_reduce",
  )(x, route_metadata, route_weights, route_counts)
  return out[:num_tokens]
