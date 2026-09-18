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
"""Runtime metadata for ragged PCP GDN communication stages.

The arrays returned here have compile-time capacities, but their active
prefixes are built with runtime-bounded loops.  In particular, ``num_stages``
and ``metadata.num_tiles`` depend only on the live request metadata.  A short
request compiled in a large token/request bucket therefore does not acquire
the bucket's padded communication or compute work.
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from tokamax._src.ops.causal_conv1d_gated_delta_rule import config
from tokamax._src.ops.causal_conv1d_gated_delta_rule import memory_ref


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PcpStageMetadata:
    """Replicated schedule consumed by the PCP Pallas kernel.

    Stage arrays use the active prefix ``[:num_stages]``.  Each stage is one
    request's intersection with one token-owner PCP interleave round.  The valid
    rows owned by rank ``r`` are contiguous in that rank's packed HBM shard and
    land in the aligned slot beginning at
    ``rank_recv_start[stage, r]`` in the receive window.  Tiles visit only the
    valid prefix of each slot, in logical token order.

    Tile arrays use the active prefix ``[:metadata.num_tiles]`` and map GDN
    compute tiles back to their communication stage.
    """

    num_stages: jax.Array
    rank_active_row_end: jax.Array
    request_id: jax.Array
    query_row_start: jax.Array
    num_tokens: jax.Array
    rank_row_start: jax.Array
    rank_valid_rows: jax.Array
    rank_recv_start: jax.Array
    first_tile: jax.Array
    num_tiles: jax.Array
    tile_stage: jax.Array
    tile_row_in_stage: jax.Array
    projection_catchup_start: jax.Array
    projection_catchup_count: jax.Array
    projection_work_offset: jax.Array
    projection_work_offset_end: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _BuilderState:
    request_index: jax.Array
    active_stages: jax.Array
    active_tiles: jax.Array
    rank_row_offsets: jax.Array
    rank_active_row_end: jax.Array
    request_id: jax.Array
    query_row_start: jax.Array
    num_tokens: jax.Array
    rank_row_start: jax.Array
    rank_valid_rows: jax.Array
    rank_recv_start: jax.Array
    first_tile: jax.Array
    stage_num_tiles: jax.Array
    tile_stage: jax.Array
    tile_row_in_stage: jax.Array
    tile_request_id: jax.Array
    tile_query_row_start: jax.Array
    tile_num_tokens: jax.Array
    tile_is_first: jax.Array
    tile_is_last: jax.Array


def _rank_token_count_before(
    position: jax.Array,
    ranks: jax.Array,
    *,
    pcp_size: int,
    comm_chunk_size: int,
) -> jax.Array:
    round_size = pcp_size * comm_chunk_size
    full_rounds = position // round_size
    round_offset = position - full_rounds * round_size
    rank_start = ranks * comm_chunk_size
    rows_in_round = jnp.clip(round_offset - rank_start,
                             min=0,
                             max=comm_chunk_size)
    return full_rounds * comm_chunk_size + rows_in_round


def _schedule_capacities(
    cfg: config.GDNConfig,
    *,
    max_num_seqs: int,
    pcp_size: int,
    comm_chunk_size: int,
) -> tuple[int, int]:
    """Return safe static storage capacities for stages and split tiles."""
    stage_size = pcp_size * comm_chunk_size
    # Splitting a request at token-owner PCP-round boundaries adds at most one
    # extra stage per request beyond the stages implied by its token count.
    max_stages = pl.cdiv(cfg.batch_size, stage_size) + max_num_seqs
    # GDN tiles are also split at rank-slot boundaries so every VMEM token
    # offset stays aligned for DMA.  The conservative rank factor keeps this
    # valid even in small test geometries where one communication chunk is
    # smaller than one GDN tile.
    max_tiles = (pl.cdiv(cfg.batch_size, cfg.tile_size) +
                 pcp_size * max_stages)
    return max_stages, max_tiles


def compute_pcp_stage_metadata(
    cfg: config.GDNConfig,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    start_seq: jax.Array,
    end_seq: jax.Array,
    *,
    pcp_size: int,
    comm_chunk_size: int,
    projection_token_block_size: int,
    num_qkv_out_blocks: int,
    num_projection_out_blocks: int,
) -> tuple[memory_ref.MetadataRef, PcpStageMetadata]:
    """Build PCP metadata from the current public request metadata.

    Query ownership follows the batch-flat request-major coordinate from
    ``query_start_loc[:-1]``. Request state and convolution semantics remain
    independent and use absolute starts reconstructed from ``seq_lens``.
    """
    if seq_lens.size != state_indices.size:
        raise ValueError("state_indices and seq_lens must have equal size.")
    if query_start_loc.size != seq_lens.size + 1:
        raise ValueError("query_start_loc must contain max_num_seqs + 1 "
                         "entries.")
    q_lens = query_start_loc[1:] - query_start_loc[:-1]
    request_absolute_starts = seq_lens.astype(jnp.int32) - q_lens.astype(
        jnp.int32)
    token_owner_starts = query_start_loc[:-1].astype(jnp.int32)
    return _compute_pcp_stage_metadata_from_coordinates(
        cfg,
        query_start_loc,
        state_indices,
        start_seq,
        end_seq,
        token_owner_starts=token_owner_starts,
        request_absolute_starts=request_absolute_starts,
        pcp_size=pcp_size,
        comm_chunk_size=comm_chunk_size,
        projection_token_block_size=projection_token_block_size,
        num_qkv_out_blocks=num_qkv_out_blocks,
        num_projection_out_blocks=num_projection_out_blocks,
    )


def _compute_pcp_stage_metadata_from_coordinates(
    cfg: config.GDNConfig,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    start_seq: jax.Array,
    end_seq: jax.Array,
    *,
    token_owner_starts: jax.Array,
    request_absolute_starts: jax.Array,
    pcp_size: int,
    comm_chunk_size: int,
    projection_token_block_size: int,
    num_qkv_out_blocks: int,
    num_projection_out_blocks: int,
) -> tuple[memory_ref.MetadataRef, PcpStageMetadata]:
    """Build live PCP stages and GDN tiles from independent coordinates.

    ``query_start_loc`` describes request-major query rows.
    ``token_owner_starts`` selects each row's PCP rank and determines the
    rank-major source/destination HBM offsets. ``request_absolute_starts`` is
    independent: it controls only request state/conv semantics such as whether
    the first current tile has an initial state.

    The output shapes depend only on the compile bucket.  All loops stop at
    ``end_seq`` and at each request's real final token; the active stage and
    tile counts therefore never include padded request slots or padded tokens.
    """
    if pcp_size <= 1:
        raise ValueError("PCP stage metadata requires pcp_size > 1.")
    if comm_chunk_size <= 0:
        raise ValueError("comm_chunk_size must be positive.")
    if projection_token_block_size <= 0:
        raise ValueError("projection_token_block_size must be positive.")
    if num_qkv_out_blocks <= 0:
        raise ValueError("num_qkv_out_blocks must be positive.")
    if num_projection_out_blocks < num_qkv_out_blocks:
        raise ValueError("num_projection_out_blocks must cover QKV blocks.")

    max_num_seqs = state_indices.size
    if query_start_loc.size != max_num_seqs + 1:
        raise ValueError("query_start_loc must contain max_num_seqs + 1 "
                         "entries.")
    if token_owner_starts.size != max_num_seqs:
        raise ValueError("token_owner_starts and state_indices must have "
                         "equal size.")
    if request_absolute_starts.size != max_num_seqs:
        raise ValueError("request_absolute_starts and state_indices must have "
                         "equal size.")

    max_stages, max_tiles = _schedule_capacities(
        cfg,
        max_num_seqs=max_num_seqs,
        pcp_size=pcp_size,
        comm_chunk_size=comm_chunk_size,
    )
    int_dtype = jnp.int32
    ranks = jnp.arange(pcp_size, dtype=int_dtype)
    query_start_loc = query_start_loc.astype(int_dtype)
    token_owner_starts = token_owner_starts.astype(int_dtype)
    request_absolute_starts = request_absolute_starts.astype(int_dtype)
    state_indices = state_indices.astype(int_dtype)
    start_seq = start_seq.astype(int_dtype)
    end_seq = end_seq.astype(int_dtype)
    round_size = pcp_size * comm_chunk_size

    initial = _BuilderState(
        request_index=jnp.asarray(0, dtype=int_dtype),
        active_stages=jnp.asarray(0, dtype=int_dtype),
        active_tiles=jnp.asarray(0, dtype=int_dtype),
        rank_row_offsets=jnp.zeros((pcp_size, ), dtype=int_dtype),
        rank_active_row_end=jnp.zeros((pcp_size, ), dtype=int_dtype),
        request_id=jnp.zeros((max_stages, ), dtype=int_dtype),
        query_row_start=jnp.zeros((max_stages, ), dtype=int_dtype),
        num_tokens=jnp.zeros((max_stages, ), dtype=int_dtype),
        rank_row_start=jnp.zeros((max_stages, pcp_size), dtype=int_dtype),
        rank_valid_rows=jnp.zeros((max_stages, pcp_size), dtype=int_dtype),
        rank_recv_start=jnp.zeros((max_stages, pcp_size), dtype=int_dtype),
        first_tile=jnp.zeros((max_stages, ), dtype=int_dtype),
        stage_num_tiles=jnp.zeros((max_stages, ), dtype=int_dtype),
        tile_stage=jnp.zeros((max_tiles, ), dtype=int_dtype),
        tile_row_in_stage=jnp.zeros((max_tiles, ), dtype=int_dtype),
        tile_request_id=jnp.zeros((max_tiles, ), dtype=int_dtype),
        tile_query_row_start=jnp.zeros((max_tiles, ), dtype=int_dtype),
        tile_num_tokens=jnp.zeros((max_tiles, ), dtype=int_dtype),
        tile_is_first=jnp.zeros((max_tiles, ), dtype=jnp.bool_),
        tile_is_last=jnp.zeros((max_tiles, ), dtype=jnp.bool_),
    )

    def _requests_pending(state: _BuilderState) -> jax.Array:
        return state.request_index < end_seq

    def _emit_request(state: _BuilderState) -> _BuilderState:
        req = state.request_index
        query_begin = query_start_loc[req]
        query_end = query_start_loc[req + 1]
        query_len = query_end - query_begin
        token_owner_begin = token_owner_starts[req]
        token_owner_end = token_owner_begin + query_len

        rank_counts = (_rank_token_count_before(
            token_owner_end,
            ranks,
            pcp_size=pcp_size,
            comm_chunk_size=comm_chunk_size,
        ) - _rank_token_count_before(
            token_owner_begin,
            ranks,
            pcp_size=pcp_size,
            comm_chunk_size=comm_chunk_size,
        ))
        request_rank_starts = state.rank_row_offsets
        next_rank_offsets = request_rank_starts + rank_counts

        should_emit = jnp.logical_and(req >= start_seq, query_len > 0)
        first_window = (token_owner_begin // round_size) * round_size
        initial_window = jnp.where(should_emit, first_window, token_owner_end)
        initial_consumed = jnp.zeros((pcp_size, ), dtype=int_dtype)

        def _windows_pending(carry) -> jax.Array:
            window, _, _ = carry
            return window < token_owner_end

        def _emit_window(carry):
            window, consumed, inner_state = carry
            rank_chunk_begin = window + ranks * comm_chunk_size
            rank_chunk_end = rank_chunk_begin + comm_chunk_size
            overlap_begin = jnp.maximum(rank_chunk_begin, token_owner_begin)
            overlap_end = jnp.minimum(rank_chunk_end, token_owner_end)
            valid_rows = jnp.maximum(overlap_end - overlap_begin, 0)
            dense_starts = (jnp.cumsum(valid_rows, dtype=int_dtype) -
                            valid_rows)
            recv_starts = ranks * comm_chunk_size
            stage_tokens = jnp.sum(valid_rows, dtype=int_dtype)
            stage_query_begin = (query_begin +
                                 jnp.maximum(window, token_owner_begin) -
                                 token_owner_begin)
            stage = inner_state.active_stages
            tile_begin = inner_state.active_tiles

            request_id = inner_state.request_id.at[stage].set(req)
            query_rows = inner_state.query_row_start.at[stage].set(
                stage_query_begin)
            num_tokens = inner_state.num_tokens.at[stage].set(stage_tokens)
            rank_rows = inner_state.rank_row_start.at[stage].set(
                request_rank_starts + consumed)
            rank_valid = inner_state.rank_valid_rows.at[stage].set(valid_rows)
            rank_recv = inner_state.rank_recv_start.at[stage].set(recv_starts)
            first_tile = inner_state.first_tile.at[stage].set(tile_begin)

            def _emit_rank_tiles(source_rank, rank_state):
                (tile_cursor, tile_stage, tile_row, tile_req, tile_query,
                 tile_size, tile_first, tile_last) = rank_state
                source_rows = valid_rows[source_rank]
                source_num_tiles = pl.cdiv(source_rows, cfg.tile_size)
                source_physical_row = source_rank * comm_chunk_size
                source_query_row = stage_query_begin + dense_starts[source_rank]

                def _emit_tile(tile_in_rank, tile_state):
                    (tile_stage, tile_row, tile_req, tile_query, tile_size,
                     tile_first, tile_last) = tile_state
                    tile = tile_cursor + tile_in_rank
                    source_offset = tile_in_rank * cfg.tile_size
                    real_size = jnp.minimum(source_rows - source_offset,
                                            cfg.tile_size)
                    query_row = source_query_row + source_offset
                    return (
                        tile_stage.at[tile].set(stage),
                        tile_row.at[tile].set(source_physical_row +
                                              source_offset),
                        tile_req.at[tile].set(req),
                        tile_query.at[tile].set(query_row),
                        tile_size.at[tile].set(real_size),
                        tile_first.at[tile].set(query_row == query_begin),
                        tile_last.at[tile].set(query_row +
                                               real_size == query_end),
                    )

                tile_arrays = jax.lax.fori_loop(
                    0,
                    source_num_tiles,
                    _emit_tile,
                    (
                        tile_stage,
                        tile_row,
                        tile_req,
                        tile_query,
                        tile_size,
                        tile_first,
                        tile_last,
                    ),
                )
                return (tile_cursor + source_num_tiles, *tile_arrays)

            (tile_end, tile_stage, tile_row, tile_req, tile_query, tile_size,
             tile_first, tile_last) = jax.lax.fori_loop(
                 0,
                 pcp_size,
                 _emit_rank_tiles,
                 (
                     tile_begin,
                     inner_state.tile_stage,
                     inner_state.tile_row_in_stage,
                     inner_state.tile_request_id,
                     inner_state.tile_query_row_start,
                     inner_state.tile_num_tokens,
                     inner_state.tile_is_first,
                     inner_state.tile_is_last,
                 ),
             )
            num_stage_tiles = tile_end - tile_begin
            stage_num_tiles = inner_state.stage_num_tiles.at[stage].set(
                num_stage_tiles)

            stage_row_ends = request_rank_starts + consumed + valid_rows
            next_state = dataclasses.replace(
                inner_state,
                active_stages=stage + 1,
                active_tiles=tile_end,
                rank_active_row_end=jnp.maximum(
                    inner_state.rank_active_row_end,
                    stage_row_ends,
                ),
                request_id=request_id,
                query_row_start=query_rows,
                num_tokens=num_tokens,
                rank_row_start=rank_rows,
                rank_valid_rows=rank_valid,
                rank_recv_start=rank_recv,
                first_tile=first_tile,
                stage_num_tiles=stage_num_tiles,
                tile_stage=tile_stage,
                tile_row_in_stage=tile_row,
                tile_request_id=tile_req,
                tile_query_row_start=tile_query,
                tile_num_tokens=tile_size,
                tile_is_first=tile_first,
                tile_is_last=tile_last,
            )
            return window + round_size, consumed + valid_rows, next_state

        _, _, state = jax.lax.while_loop(
            _windows_pending,
            _emit_window,
            (initial_window, initial_consumed, state),
        )
        return dataclasses.replace(
            state,
            request_index=req + 1,
            rank_row_offsets=next_rank_offsets,
        )

    result = jax.lax.while_loop(_requests_pending, _emit_request, initial)
    has_initial_state = request_absolute_starts > 0

    num_z_out_blocks = num_projection_out_blocks - num_qkv_out_blocks
    projection_catchup_count = jnp.zeros(
        (max_stages, pcp_size),
        dtype=int_dtype,
    )
    projection_catchup_start = jnp.zeros(
        (max_stages, pcp_size),
        dtype=int_dtype,
    )
    projection_work_offset = jnp.zeros(
        (max_stages, pcp_size),
        dtype=int_dtype,
    )

    def _plan_projection_catchup(stage, carry):
        (cumulative_catchup, catchup_starts, catchup_counts,
         cumulative_after_stage) = carry
        stage_row_end = (result.rank_row_start[stage] +
                         result.rank_valid_rows[stage])
        required_token_blocks = pl.cdiv(
            stage_row_end,
            projection_token_block_size,
        )
        # Token block zero's QKV is projected in the kernel prologue.  The
        # overlapped queue then contains Z for block zero followed by all QKVZ
        # output blocks for each later token block.  The stage may launch its
        # QKV communication only after that queue reaches the final QKV block
        # covering its live source rows.
        required_work = jnp.where(
            required_token_blocks <= 1,
            0,
            num_z_out_blocks +
            (required_token_blocks - 2) * num_projection_out_blocks +
            num_qkv_out_blocks,
        )
        # Stage zero launches before the pipeline.  Every later stage launches
        # from the preceding stage's first-tile prologue so its communication
        # can overlap that entire GDN stage.
        previous_stage = jnp.maximum(stage - 1, 0)
        launch_tile = jnp.where(
            stage == 0,
            0,
            result.first_tile[previous_stage],
        )
        work_before_stage = launch_tile + cumulative_catchup
        catchup = jnp.maximum(required_work - work_before_stage, 0)
        cumulative_catchup += catchup
        catchup_starts = catchup_starts.at[stage].set(work_before_stage)
        catchup_counts = catchup_counts.at[stage].set(catchup)
        cumulative_after_stage = cumulative_after_stage.at[stage].set(
            cumulative_catchup)
        return (cumulative_catchup, catchup_starts, catchup_counts,
                cumulative_after_stage)

    (projection_work_offset_end, projection_catchup_start,
     projection_catchup_count, cumulative_after_stage) = jax.lax.fori_loop(
         0,
         result.active_stages,
         _plan_projection_catchup,
         (
             jnp.zeros((pcp_size, ), dtype=int_dtype),
             projection_catchup_start,
             projection_catchup_count,
             projection_work_offset,
         ),
     )

    def _set_body_projection_offset(stage, work_offsets):
        # Before stage ``s`` computes its first tile, catch-up for stage
        # ``s + 1`` has already run in the body prologue.  Stage zero also
        # includes the pre-pipeline catch-up for itself.
        latest_launched_stage = jnp.minimum(
            stage + 1,
            result.active_stages - 1,
        )
        return work_offsets.at[stage].set(
            cumulative_after_stage[latest_launched_stage])

    projection_work_offset = jax.lax.fori_loop(
        0,
        result.active_stages,
        _set_body_projection_offset,
        projection_work_offset,
    )

    gdn_metadata = memory_ref.MetadataRef.create(
        cfgs=cfg,
        num_tiles=result.active_tiles,
        p_id_to_s_idx=result.tile_request_id,
        p_id_to_r_base=result.tile_query_row_start,
        p_id_to_r_size=result.tile_num_tokens,
        p_id_is_first_tile=result.tile_is_first,
        p_id_is_last_tile=result.tile_is_last,
        s_idx_has_initial_state=has_initial_state,
        s_idx_to_state_indices=state_indices,
        s_idx_to_read_offset=jnp.zeros_like(state_indices),
    )
    stage_metadata = PcpStageMetadata(
        num_stages=result.active_stages,
        rank_active_row_end=result.rank_active_row_end,
        request_id=result.request_id,
        query_row_start=result.query_row_start,
        num_tokens=result.num_tokens,
        rank_row_start=result.rank_row_start,
        rank_valid_rows=result.rank_valid_rows,
        rank_recv_start=result.rank_recv_start,
        first_tile=result.first_tile,
        num_tiles=result.stage_num_tiles,
        tile_stage=result.tile_stage,
        tile_row_in_stage=result.tile_row_in_stage,
        projection_catchup_start=projection_catchup_start,
        projection_catchup_count=projection_catchup_count,
        projection_work_offset=projection_work_offset,
        projection_work_offset_end=projection_work_offset_end,
    )
    return gdn_metadata, stage_metadata
