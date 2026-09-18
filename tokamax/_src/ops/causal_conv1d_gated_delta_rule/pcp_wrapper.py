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
"""PCP projection/communication fused with the GDN v3 compute pipeline.

The Qwen3.5 FP8 entry projects QKVZ in-kernel and interleaves all but the first
projection block with GDN tiles. BA exchange, gated normalization, and the
final output projection remain outside. GDN state streams directly to and from
the unified KV pool through a StateSourcePlan. QKV is written directly into a
compact destination-major HBM layout, so each source rank pushes one contiguous
rank-local token chunk per destination without head-width padding.

For PCP size ``P`` and local communication chunk ``C``, one logical compute
stage contains ``P * C`` request-major tokens and one PCP-local head shard.
While the existing GDN v3 inner kernel consumes stage ``i``, remote DMA fills
the other VMEM slot with stage ``i + 1``. Once GDN completes, each source rank
pushes the ``C`` output rows belonging to every destination into that rank's
packed token rows and the source rank's value-head columns.
"""

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tokamax._src.ops.causal_conv1d_gated_delta_rule import config
from tokamax._src.ops.causal_conv1d_gated_delta_rule import memory_ref
from tokamax._src.ops.causal_conv1d_gated_delta_rule import pcp_metadata
from tokamax._src.ops.causal_conv1d_gated_delta_rule import wrapper


@dataclasses.dataclass(frozen=True)
class _ProjectionConfig:
    """Static geometry for the in-kernel QKVZ projection schedule."""

    token_block_size: int
    out_block_size: int
    qkv_dim: int


_TPU_TILE_ROWS = 8
_TPU_TILE_COLUMNS = 128


def _qkv_receive_group_stride(
    comm_chunk_size: int,
    tile_size: int,
) -> int:
    """Return the tile-group capacity reserved for one source rank.

    A ragged source row may begin up to seven rows into its first physical
    tile. Both the communication copy and the final compute tile therefore
    need one possible leading partial group. Aligned rows use the smaller
    runtime copy/load shapes and do not pay for this worst-case capacity.
    """
    max_valid_groups = pl.cdiv(
        comm_chunk_size + _TPU_TILE_ROWS - 1,
        _TPU_TILE_ROWS,
    )
    max_compute_groups = pl.cdiv(
        tile_size + _TPU_TILE_ROWS - 1,
        _TPU_TILE_ROWS,
    )
    max_tile_start = ((comm_chunk_size - 1) // tile_size * tile_size)
    return max(
        max_valid_groups,
        max_tile_start // _TPU_TILE_ROWS + max_compute_groups,
    )


def _mesh_device_id(
    mesh_axis_names: tuple[str, ...],
    pcp_axis_name: str,
    pcp_rank: jax.Array | int,
) -> tuple[jax.Array | int, ...]:
    """Build the destination's full logical mesh coordinate.

    Keep remote-copy addressing identical to the PCP streaming attention
    kernels: only the PCP coordinate changes, while every other mesh coordinate
    remains the caller's runtime axis index.
    """
    return tuple(
        pcp_rank if axis_name == pcp_axis_name else lax.axis_index(axis_name)
        for axis_name in mesh_axis_names)


def _all_rank_barrier(
    barrier_sem: Any,
    rank: jax.Array,
    pcp_size: int,
    *,
    mesh_axis_names: tuple[str, ...],
    pcp_axis_name: str,
) -> None:
    """Barrier every PCP rank using a remotely addressable regular semaphore."""
    for peer_offset in range(1, pcp_size):
        peer = lax.rem(rank + peer_offset, pcp_size)
        pl.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=_mesh_device_id(
                mesh_axis_names,
                pcp_axis_name,
                peer,
            ),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier_sem, pcp_size - 1)


def _compute_gdn_tile(
    b_slot_ref: jax.Array,
    a_slot_ref: jax.Array,
    conv_state_slot_ref: jax.Array,
    recurrent_slot_ref: jax.Array,
    out_slot_ref: jax.Array,
    metadata_ref: memory_ref.MetadataRef,
    stage_metadata_ref: pcp_metadata.PcpStageMetadata,
    weights_ref: memory_ref.WeightRefs,
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    received_qkv_ref: jax.Array,
    compute_qkv_ref: jax.Array,
    *,
    cfg: config.GDNConfig,
    comm_chunk_size: int,
) -> None:
    """Convert one received BF16 tile and run the existing GDN inner kernel."""
    p_id = pl.program_id(0)
    stage = stage_metadata_ref.tile_stage[p_id]
    recv_slot = lax.rem(stage, 2)
    token_start = pl.multiple_of(
        stage_metadata_ref.tile_row_in_stage[p_id],
        16,
    )
    source_rank = token_start // comm_chunk_size
    source_offset = lax.rem(token_start, comm_chunk_size)
    source_row = stage_metadata_ref.rank_row_start[stage, source_rank]
    source_prefix = lax.rem(source_row, _TPU_TILE_ROWS)
    receive_group_stride = _qkv_receive_group_stride(
        comm_chunk_size,
        cfg.tile_size,
    )
    receive_group_start = (source_rank * receive_group_stride +
                           source_offset // _TPU_TILE_ROWS)
    tile_prefix = lax.rem(
        source_prefix + source_offset,
        _TPU_TILE_ROWS,
    )

    def _load_qkv(prefix: int) -> None:

        @pl.when(tile_prefix == prefix)
        def _load() -> None:
            compute_groups = pl.cdiv(prefix + cfg.tile_size, _TPU_TILE_ROWS)
            packed_qkv = received_qkv_ref[
                recv_slot,
                pl.ds(receive_group_start, compute_groups),
                :,
                :,
                :,
            ]
            dense_qkv = pltpu.einshape(
                "acbd->(ab)(cd)",
                packed_qkv,
                assert_is_tile_preserving=True,
            )
            compute_qkv_ref[
                0,
                :,
                0,
                :,
            ] = dense_qkv[prefix:prefix + cfg.tile_size, :].astype(jnp.float32)

    for prefix in range(_TPU_TILE_ROWS):
        _load_qkv(prefix)

    wrapper.inner_kernel(
        compute_qkv_ref,
        b_slot_ref,
        a_slot_ref,
        conv_state_slot_ref,
        recurrent_slot_ref,
        out_slot_ref,
        metadata_ref,
        weights_ref,
        carry_conv_scratch_ref,
        carry_recurrent_scratch_ref,
        cfg=cfg,
        p_id=p_id,
    )


def _pcp_qkvz_projection_gdn_outer_kernel(
    # Inputs.
    metadata_ref: memory_ref.MetadataRef,
    stage_metadata_ref: pcp_metadata.PcpStageMetadata,
    hidden_ref: jax.Array,
    projection_weight_ref: jax.Array,
    projection_weight_scale_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    state_source_ref: jax.Array,
    zero_gdn_ref: jax.Array,
    packed_out_hbm_ref: jax.Array,
    weights_ref: memory_ref.WeightRefs,
    # Outputs.
    gdn_out_ref: jax.Array,
    packed_out_ref: jax.Array,
    conv_state_updates_ref: jax.Array,
    recurrent_state_updates_ref: jax.Array,
    projected_qkv_ref: jax.Array,
    z_ref: jax.Array,
    active_rows_out_ref: jax.Array,
    # Scratches.
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    qkv_send_sems: jax.Array,
    qkv_recv_sems: jax.Array,
    output_send_sems: jax.Array,
    output_recv_sems: jax.Array,
    barrier_sems: jax.Array,
    received_qkv_ref: jax.Array,
    compute_qkv_ref: jax.Array,
    output_stage_ref: jax.Array,
    projection_x_sem: jax.Array,
    projection_weight_sem: jax.Array,
    projection_scale_sem: jax.Array,
    projection_out_sem: jax.Array,
    projection_x_ref: jax.Array,
    projection_x_q_ref: jax.Array,
    projection_x_scale_ref: jax.Array,
    projection_weight_vmem_ref: jax.Array,
    projection_weight_scale_vmem_ref: jax.Array,
    projection_out_ref: jax.Array,
    projection_z_out_ref: jax.Array,
    active_rows_smem_ref: jax.Array,
    active_rows_dma_sem: jax.Array,
    *,
    cfg: config.GDNConfig,
    pcp_size: int,
    comm_chunk_size: int,
    projection_cfg: _ProjectionConfig,
    mesh_axis_names: tuple[str, ...],
    pcp_axis_name: str,
) -> None:
    """Schedule QKVZ projection, QKV remote DMA, and per-sequence GDN."""
    del packed_out_hbm_ref

    rank = lax.axis_index(pcp_axis_name)
    active_rows_smem_ref[0] = stage_metadata_ref.rank_active_row_end[rank]
    active_rows_store = pltpu.make_async_copy(
        src_ref=active_rows_smem_ref.at[:],
        dst_ref=active_rows_out_ref.at[:],
        sem=active_rows_dma_sem,
    )
    active_rows_store.start()
    active_rows_store.wait()

    shard_key_dim = cfg.kq_dim_size
    shard_value_dim = cfg.v_dim_size

    def _start_output_stage(
        stage,
        output_stage_ref,
        packed_out_ref,
        output_send_sems,
        output_recv_sems,
    ) -> None:
        """Send one row-contiguous output segment to each destination."""
        slot = lax.rem(stage, 2)

        for destination in range(pcp_size):
            valid_rows = stage_metadata_ref.rank_valid_rows[stage, destination]
            is_local = rank == destination
            local_rows = lax.select(is_local, valid_rows, 0)
            remote_rows = lax.select(is_local, 0, valid_rows)
            src_row = stage_metadata_ref.rank_recv_start[stage, destination]
            dst_row = stage_metadata_ref.rank_row_start[stage, destination]

            pltpu.make_async_copy(
                src_ref=output_stage_ref.at[
                    slot,
                    pl.ds(src_row, local_rows),
                    :,
                    :,
                ],
                dst_ref=packed_out_ref.at[
                    pl.ds(dst_row, local_rows),
                    rank,
                    :,
                    :,
                ],
                sem=output_recv_sems.at[slot],
            ).start()
            pltpu.make_async_remote_copy(
                src_ref=output_stage_ref.at[
                    slot,
                    pl.ds(src_row, remote_rows),
                    :,
                    :,
                ],
                dst_ref=packed_out_ref.at[
                    pl.ds(dst_row, remote_rows),
                    rank,
                    :,
                    :,
                ],
                send_sem=output_send_sems.at[slot],
                recv_sem=output_recv_sems.at[slot],
                device_id=_mesh_device_id(
                    mesh_axis_names,
                    pcp_axis_name,
                    destination,
                ),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()

    def _wait_output_stage(
        stage,
        active_stages,
        output_stage_ref,
        packed_out_ref,
        output_send_sems,
        output_recv_sems,
    ) -> None:
        is_valid = jnp.logical_and(stage >= 0, stage < active_stages)
        safe_stage = jnp.maximum(stage, 0)
        rank_rows = stage_metadata_ref.rank_valid_rows[safe_stage, rank]
        receive_rows = lax.select(
            is_valid,
            rank_rows,
            0,
        )
        send_rows = lax.select(
            is_valid,
            stage_metadata_ref.num_tokens[safe_stage] - rank_rows,
            0,
        )
        slot = lax.rem(safe_stage, 2)

        recv_wait_ref = packed_out_ref.at[
            pl.ds(0, receive_rows),
            :,
            :,
            :,
        ]
        pltpu.make_async_copy(
            src_ref=recv_wait_ref,
            dst_ref=recv_wait_ref,
            sem=output_recv_sems.at[slot],
        ).wait()

        send_wait_ref = output_stage_ref.at[
            slot,
            pl.ds(0, send_rows),
            :,
            :,
        ]
        pltpu.make_async_copy(
            src_ref=send_wait_ref,
            dst_ref=send_wait_ref,
            sem=output_send_sems.at[slot],
        ).wait()

    projection_token_block_size = projection_cfg.token_block_size
    qkv_width = 2 * shard_key_dim + shard_value_dim
    receive_group_stride = _qkv_receive_group_stride(
        comm_chunk_size,
        cfg.tile_size,
    )
    num_qkv_out_blocks = pcp_size
    num_z_out_blocks = pcp_size
    num_projection_out_blocks = num_qkv_out_blocks + num_z_out_blocks
    full_key_dim = pcp_size * shard_key_dim

    def _projection_job(out_block):
        is_qkv = out_block < num_qkv_out_blocks
        destination = out_block
        z_group = out_block - num_qkv_out_blocks
        return is_qkv, destination, z_group

    def _start_packed_qkv_copy(
        stage,
        slot,
        destination: int,
    ) -> None:
        is_local = rank == destination
        valid_rows = stage_metadata_ref.rank_valid_rows[stage, rank]
        src_row = stage_metadata_ref.rank_row_start[stage, rank]
        src_prefix = lax.rem(src_row, _TPU_TILE_ROWS)
        valid_groups = lax.select(
            valid_rows > 0,
            pl.cdiv(src_prefix + valid_rows, _TPU_TILE_ROWS),
            0,
        )
        local_groups = lax.select(is_local, valid_groups, 0)
        remote_groups = lax.select(is_local, 0, valid_groups)
        src_group = src_row // _TPU_TILE_ROWS
        dst_group = rank * receive_group_stride

        pltpu.make_async_copy(
            src_ref=projected_qkv_ref.at[
                destination,
                pl.ds(src_group, local_groups),
                :,
                :,
                :,
            ],
            dst_ref=received_qkv_ref.at[
                slot,
                pl.ds(dst_group, local_groups),
                :,
                :,
                :,
            ],
            sem=qkv_recv_sems.at[slot],
        ).start()
        pltpu.make_async_remote_copy(
            src_ref=projected_qkv_ref.at[
                destination,
                pl.ds(src_group, remote_groups),
                :,
                :,
                :,
            ],
            dst_ref=received_qkv_ref.at[
                slot,
                pl.ds(dst_group, remote_groups),
                :,
                :,
                :,
            ],
            send_sem=qkv_send_sems.at[slot],
            recv_sem=qkv_recv_sems.at[slot],
            device_id=_mesh_device_id(
                mesh_axis_names,
                pcp_axis_name,
                destination,
            ),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()

    def _start_qkv_stage(stage) -> None:
        slot = lax.rem(stage, 2)
        for destination in range(pcp_size):
            _start_packed_qkv_copy(stage, slot, destination)

    def _wait_qkv_recv(stage) -> None:
        slot = lax.rem(stage, 2)
        receive_groups = jnp.asarray(0, dtype=jnp.int32)
        for source in range(pcp_size):
            valid_rows = stage_metadata_ref.rank_valid_rows[stage, source]
            src_row = stage_metadata_ref.rank_row_start[stage, source]
            src_prefix = lax.rem(src_row, _TPU_TILE_ROWS)
            receive_groups += lax.select(
                valid_rows > 0,
                pl.cdiv(src_prefix + valid_rows, _TPU_TILE_ROWS),
                0,
            )
        wait_ref = received_qkv_ref.at[
            slot,
            pl.ds(0, receive_groups),
            :,
            :,
            :,
        ]
        pltpu.make_async_copy(
            src_ref=wait_ref,
            dst_ref=wait_ref,
            sem=qkv_recv_sems.at[slot],
        ).wait()

    def _wait_qkv_send(stage, active_stages) -> None:
        is_valid = jnp.logical_and(stage >= 0, stage < active_stages)
        safe_stage = jnp.maximum(stage, 0)
        slot = lax.rem(safe_stage, 2)
        valid_rows = stage_metadata_ref.rank_valid_rows[safe_stage, rank]
        src_row = stage_metadata_ref.rank_row_start[safe_stage, rank]
        src_prefix = lax.rem(src_row, _TPU_TILE_ROWS)
        valid_groups = lax.select(
            valid_rows > 0,
            pl.cdiv(src_prefix + valid_rows, _TPU_TILE_ROWS),
            0,
        )
        remote_groups = lax.select(
            is_valid,
            (pcp_size - 1) * valid_groups,
            0,
        )
        wait_ref = received_qkv_ref.at[
            slot,
            pl.ds(0, remote_groups),
            :,
            :,
            :,
        ]
        pltpu.make_async_copy(
            src_ref=wait_ref,
            dst_ref=wait_ref,
            sem=qkv_send_sems.at[slot],
        ).wait()

    def _start_projection_tile(token_block, out_block) -> None:
        token_row = token_block * projection_token_block_size
        is_qkv, destination, z_group = _projection_job(out_block)

        @pl.when(out_block == 0)
        def _start_projection_x() -> None:
            pltpu.make_async_copy(
                src_ref=hidden_ref.at[
                    pl.ds(token_row, projection_token_block_size),
                    :,
                ],
                dst_ref=projection_x_ref,
                sem=projection_x_sem,
            ).start()

        def _start_weight_piece(
            source_row,
            destination_row: int,
            width: int,
        ) -> None:
            pltpu.make_async_copy(
                src_ref=projection_weight_ref.at[
                    pl.ds(source_row, width),
                    :,
                ],
                dst_ref=projection_weight_vmem_ref.at[
                    pl.ds(destination_row, width),
                    :,
                ],
                sem=projection_weight_sem,
            ).start()
            pltpu.make_async_copy(
                src_ref=projection_weight_scale_ref.at[pl.ds(
                    source_row, width)],
                dst_ref=projection_weight_scale_vmem_ref.at[pl.ds(
                    destination_row, width)],
                sem=projection_scale_sem,
            ).start()

        @pl.when(is_qkv)
        def _start_qkv() -> None:
            q_col = destination * shard_key_dim
            k_col = full_key_dim + destination * shard_key_dim
            v_col = 2 * full_key_dim + destination * shard_value_dim
            _start_weight_piece(q_col, 0, shard_key_dim)
            _start_weight_piece(k_col, shard_key_dim, shard_key_dim)
            _start_weight_piece(
                v_col,
                2 * shard_key_dim,
                shard_value_dim,
            )

        @pl.when(jnp.logical_not(is_qkv))
        def _start_z_group() -> None:
            z_col = projection_cfg.qkv_dim + z_group * shard_value_dim
            _start_weight_piece(z_col, 0, shard_value_dim)

    def _previous_projection_out_block(out_block):
        return lax.select(
            out_block > 0,
            out_block - 1,
            num_projection_out_blocks - 1,
        )

    def _wait_projection_output(out_block) -> None:
        is_qkv, _, _ = _projection_job(out_block)

        @pl.when(is_qkv)
        def _wait_qkv() -> None:
            pltpu.make_async_copy(
                src_ref=projection_out_ref,
                dst_ref=projection_out_ref,
                sem=projection_out_sem,
            ).wait()

        @pl.when(jnp.logical_not(is_qkv))
        def _wait_z() -> None:
            pltpu.make_async_copy(
                src_ref=projection_z_out_ref,
                dst_ref=projection_z_out_ref,
                sem=projection_out_sem,
            ).wait()

    def _finish_projection_tile(
        token_block,
        out_block,
        *,
        wait_previous,
        wait_final,
    ) -> None:

        @pl.when(wait_previous)
        def _wait_previous_projection_output() -> None:
            _wait_projection_output(
                _previous_projection_out_block(out_block), )

        @pl.when(out_block == 0)
        def _quantize_projection_x() -> None:
            pltpu.make_async_copy(
                src_ref=projection_x_ref,
                dst_ref=projection_x_ref,
                sem=projection_x_sem,
            ).wait()
            x = projection_x_ref[...]
            dtype_info = jnp.finfo(projection_weight_ref.dtype)
            x_abs_max = jnp.max(jnp.abs(x), axis=1)
            x_scale = (x_abs_max.astype(jnp.float32) / float(dtype_info.max))
            x_scale = jnp.where(x_scale == 0, 1.0, x_scale)
            projection_x_q_ref[...] = jnp.clip(
                x.astype(jnp.float32) / x_scale[:, None],
                float(dtype_info.min),
                float(dtype_info.max),
            ).astype(projection_weight_ref.dtype)
            projection_x_scale_ref[...] = x_scale

        is_qkv, destination, z_group = _projection_job(out_block)

        def _finish_matmul(width: int):
            weight_ref = projection_weight_vmem_ref.at[
                pl.ds(0, width),
                :,
            ]
            scale_ref = projection_weight_scale_vmem_ref.at[pl.ds(0, width)]
            pltpu.make_async_copy(
                src_ref=weight_ref,
                dst_ref=weight_ref,
                sem=projection_weight_sem,
            ).wait()
            pltpu.make_async_copy(
                src_ref=scale_ref,
                dst_ref=scale_ref,
                sem=projection_scale_sem,
            ).wait()

            acc = jax.lax.dot_general(
                projection_x_q_ref[...],
                projection_weight_vmem_ref[pl.ds(0, width), :],
                dimension_numbers=(((1, ), (1, )), ((), ())),
                preferred_element_type=jnp.float32,
            )
            acc *= projection_x_scale_ref[...][:, None]
            acc *= projection_weight_scale_vmem_ref[pl.ds(0, width)][None, :]
            return acc.astype(hidden_ref.dtype)

        @pl.when(is_qkv)
        def _finish_qkv() -> None:
            projection_out_ref[...] = pltpu.einshape(
                "(ab)(cd)->acbd",
                _finish_matmul(qkv_width),
                b=_TPU_TILE_ROWS,
                d=_TPU_TILE_COLUMNS,
                assert_is_tile_preserving=True,
            )

        @pl.when(jnp.logical_not(is_qkv))
        def _finish_z() -> None:
            lanes = shard_value_dim // _TPU_TILE_COLUMNS
            values = _finish_matmul(shard_value_dim).reshape(
                projection_token_block_size,
                lanes,
                _TPU_TILE_COLUMNS,
            )
            projection_z_out_ref[...] = jnp.pad(
                values,
                (
                    (0, 0),
                    (0, _TPU_TILE_ROWS - lanes),
                    (0, 0),
                ),
            )

        token_row = token_block * projection_token_block_size

        @pl.when(is_qkv)
        def _write_qkv_projection() -> None:
            group_start = token_row // _TPU_TILE_ROWS
            num_groups = (projection_token_block_size // _TPU_TILE_ROWS)
            pltpu.make_async_copy(
                src_ref=projection_out_ref,
                dst_ref=projected_qkv_ref.at[
                    destination,
                    pl.ds(group_start, num_groups),
                    :,
                    :,
                    :,
                ],
                sem=projection_out_sem,
            ).start()

        @pl.when(out_block >= num_qkv_out_blocks)
        def _write_z_projection() -> None:
            pltpu.make_async_copy(
                src_ref=projection_z_out_ref,
                dst_ref=z_ref.at[
                    pl.ds(token_row, projection_token_block_size),
                    z_group,
                    :,
                    :,
                ],
                sem=projection_out_sem,
            ).start()

        @pl.when(wait_final)
        def _wait_final_projection_output() -> None:
            _wait_projection_output(out_block)

    def _project_first_token_block() -> None:

        def _project_out_block(out_block, _) -> None:
            _start_projection_tile(0, out_block)
            _finish_projection_tile(
                0,
                out_block,
                wait_previous=out_block > 0,
                wait_final=out_block == num_qkv_out_blocks - 1,
            )

        lax.fori_loop(
            0,
            num_qkv_out_blocks,
            _project_out_block,
            None,
            unroll=False,
        )

    _, b_alloc, a_alloc, conv_alloc, recurrent_alloc, out_alloc = (
        memory_ref.create_allocs(
            metadata_ref=metadata_ref,
            qkv_ref=compute_qkv_ref,
            b_ref=b_ref,
            a_ref=a_ref,
            out_ref=gdn_out_ref,
            conv_state_ref=state_source_ref,
            recurrent_state_ref=state_source_ref,
            cfg=cfg,
            conv_state_output_ref=conv_state_updates_ref,
            recurrent_state_output_ref=recurrent_state_updates_ref,
        ))

    num_tiles = metadata_ref.num_tiles[...]
    active_stages = stage_metadata_ref.num_stages[...]
    # Static HBM shapes follow the compile bucket, while this live per-rank row
    # end follows only the real requests.  At most one projection token block
    # of tail padding is therefore computed for a short invocation.
    active_projection_token_blocks = pl.cdiv(
        stage_metadata_ref.rank_active_row_end[rank],
        projection_token_block_size,
    )
    num_z_out_blocks = num_projection_out_blocks - num_qkv_out_blocks
    # QKV for token block zero is the only unconditional prologue work.  The
    # remaining queue keeps the original block-major order so the quantized X
    # tile is reused: block-zero Z, then every later block's QKV followed by Z.
    num_projection_work_tiles = lax.select(
        active_projection_token_blocks > 0,
        num_z_out_blocks + jnp.maximum(
            active_projection_token_blocks - 1,
            0,
        ) * num_projection_out_blocks,
        0,
    )

    def _projection_tile_for_work_id(work_id):
        is_first_block_z = work_id < num_z_out_blocks
        later_projection_id = work_id - num_z_out_blocks
        token_block = lax.select(
            is_first_block_z,
            0,
            1 + later_projection_id // num_projection_out_blocks,
        )
        out_block = lax.select(
            is_first_block_z,
            num_qkv_out_blocks + work_id,
            lax.rem(later_projection_id, num_projection_out_blocks),
        )
        return token_block, out_block

    def _run_projection_catchup(stage) -> None:
        catchup_count = stage_metadata_ref.projection_catchup_count[stage,
                                                                    rank]
        work_start = stage_metadata_ref.projection_catchup_start[stage, rank]
        previous_stage = jnp.maximum(stage - 1, 0)
        launch_tile = lax.select(
            stage == 0,
            0,
            stage_metadata_ref.first_tile[previous_stage],
        )

        def _project_missing_tile(index, _) -> None:
            work_id = work_start + index
            token_block, out_block = _projection_tile_for_work_id(work_id)
            _start_projection_tile(token_block, out_block)
            _finish_projection_tile(
                token_block,
                out_block,
                # A preceding overlapped tile leaves its output DMA in flight.
                # Stage zero and stage one's catch-ups both precede pipeline
                # tile zero; when one follows the other, the earlier loop has
                # already drained its final DMA.
                wait_previous=jnp.logical_or(index > 0, launch_tile > 0),
                # QKV DMA may read the just-produced HBM block immediately
                # after this loop, so the final catch-up write must be visible.
                wait_final=index == catchup_count - 1,
            )

        lax.fori_loop(
            0,
            catchup_count,
            _project_missing_tile,
            None,
            unroll=False,
        )

    def _compute_gdn_tile_and_exchange_output(
        b_slot_ref,
        a_slot_ref,
        conv_state_slot_ref,
        recurrent_slot_ref,
        out_slot_ref,
        metadata_ref,
        stage_metadata_ref,
        weights_ref,
        carry_conv_scratch_ref,
        carry_recurrent_scratch_ref,
        received_qkv_ref,
        compute_qkv_ref,
        output_stage_ref,
        packed_out_ref,
        output_send_sems,
        output_recv_sems,
    ) -> None:
        p_id = pl.program_id(0)
        stage = stage_metadata_ref.tile_stage[p_id]
        stage_first_tile = stage_metadata_ref.first_tile[stage]
        is_stage_first_tile = p_id == stage_first_tile

        work_id = (p_id +
                   stage_metadata_ref.projection_work_offset[stage, rank])
        projection_is_active = work_id < num_projection_work_tiles
        projection_token_block, projection_out_block = (
            _projection_tile_for_work_id(work_id))
        safe_next_stage = jnp.minimum(stage + 1, active_stages - 1)
        next_stage_catchup = lax.select(
            stage + 1 < active_stages,
            stage_metadata_ref.projection_catchup_count[safe_next_stage, rank],
            0,
        )
        projection_dma_was_drained = jnp.logical_or(
            jnp.logical_and(
                stage == 0,
                stage_metadata_ref.projection_catchup_count[0, rank] > 0,
            ),
            next_stage_catchup > 0,
        )

        @pl.when(projection_is_active)
        def _prefetch_projection_tile() -> None:
            _start_projection_tile(
                projection_token_block,
                projection_out_block,
            )

        @pl.when(jnp.logical_and(is_stage_first_tile, stage >= 2))
        def _reuse_output_stage_slot() -> None:
            # A slot is reused only after every send sourced from it and every
            # receive targeting its paired semaphore have completed.
            _wait_output_stage(
                stage - 2,
                active_stages,
                output_stage_ref,
                packed_out_ref,
                output_send_sems,
                output_recv_sems,
            )

        _compute_gdn_tile(
            b_slot_ref,
            a_slot_ref,
            conv_state_slot_ref,
            recurrent_slot_ref,
            out_slot_ref,
            metadata_ref,
            stage_metadata_ref,
            weights_ref,
            carry_conv_scratch_ref,
            carry_recurrent_scratch_ref,
            received_qkv_ref,
            compute_qkv_ref,
            cfg=cfg,
            comm_chunk_size=comm_chunk_size,
        )

        @pl.when(projection_is_active)
        def _compute_projection_tile() -> None:
            _finish_projection_tile(
                projection_token_block,
                projection_out_block,
                wait_previous=jnp.logical_and(
                    work_id > 0,
                    jnp.logical_not(
                        jnp.logical_and(
                            is_stage_first_tile,
                            projection_dma_was_drained,
                        )),
                ),
                wait_final=(work_id == num_projection_work_tiles - 1),
            )

        output_slot = lax.rem(stage, 2)
        output_stage_ref[
            output_slot,
            pl.ds(stage_metadata_ref.tile_row_in_stage[p_id], cfg.tile_size),
            :,
            :,
        ] = out_slot_ref[0].reshape(
            cfg.tile_size,
            cfg.num_v_heads,
            cfg.v_head_dim,
        )

        is_stage_last_tile = (p_id == stage_first_tile +
                              stage_metadata_ref.num_tiles[stage] - 1)

        @pl.when(is_stage_last_tile)
        def _finish_output_stage() -> None:
            _start_output_stage(
                stage,
                output_stage_ref,
                packed_out_ref,
                output_send_sems,
                output_recv_sems,
            )

    pipeline_func = pltpu.emit_pipeline(
        body=_compute_gdn_tile_and_exchange_output,
        grid=(num_tiles, ),
        in_specs=(
            b_alloc.spec,
            a_alloc.spec,
            conv_alloc.spec,
            recurrent_alloc.spec,
        ),
        out_specs=(out_alloc.spec, ),
    )

    def _stage_prologue() -> None:
        p_id = pl.program_id(0)
        stage = stage_metadata_ref.tile_stage[p_id]
        is_stage_start = p_id == stage_metadata_ref.first_tile[stage]

        @pl.when(is_stage_start)
        def _start_stage() -> None:
            _wait_qkv_recv(stage)
            next_stage = stage + 1

            @pl.when(next_stage < active_stages)
            def _prefetch_next_stage() -> None:
                # Usually the preceding GDN tiles keep projection ahead.  A
                # highly imbalanced ragged batch can expose too few tiles; in
                # that case project only the live rank's exact missing prefix
                # before its QKV communication reads the HBM blocks.
                _run_projection_catchup(next_stage)

                @pl.when(stage > 0)
                def _reuse_slot() -> None:
                    _wait_qkv_send(stage - 1, active_stages)
                    barrier_slot = lax.rem(stage, 2)
                    _all_rank_barrier(
                        barrier_sems.at[barrier_slot],
                        rank,
                        pcp_size,
                        mesh_axis_names=mesh_axis_names,
                        pcp_axis_name=pcp_axis_name,
                    )

                _start_qkv_stage(next_stage)

    @pl.with_scoped(allocations=(
        b_alloc,
        a_alloc,
        conv_alloc,
        recurrent_alloc,
        out_alloc,
    ), )
    def _run(
        gdn_out_ref,
        packed_out_hbm_ref,
        output_send_sems,
        output_recv_sems,
        barrier_sems,
        allocations,
    ) -> None:

        # Only QKV from the first live projection block is unconditional
        # critical-path work. Subsequent QKVZ blocks are overlapped below and
        # dynamically caught up only when a ragged stage reaches them early.
        @pl.when(active_projection_token_blocks > 0)
        def _project_first_live_token_block() -> None:
            _project_first_token_block()

        @pl.when(active_stages > 0)
        def _start_first_qkv_stage() -> None:
            _run_projection_catchup(0)
            _start_qkv_stage(0)

        pipeline_func(
            b_ref,
            a_ref,
            state_source_ref,
            state_source_ref,
            gdn_out_ref,
            scratches=(
                metadata_ref,
                stage_metadata_ref,
                weights_ref,
                carry_conv_scratch_ref,
                carry_recurrent_scratch_ref,
                received_qkv_ref,
                compute_qkv_ref,
                output_stage_ref,
                packed_out_hbm_ref,
                output_send_sems,
                output_recv_sems,
            ),
            allocations=allocations,
            body_prologue=_stage_prologue,
        )

        # If the live batch did not expose enough GDN tiles to hide all of its
        # Z projection, finish only that real queue suffix here.  The loop bound
        # is runtime data and is independent of the compile bucket capacity.
        projection_work_end = (
            num_tiles + stage_metadata_ref.projection_work_offset_end[rank])
        remaining_projection_tiles = jnp.maximum(
            num_projection_work_tiles - projection_work_end,
            0,
        )

        def _finish_projection_tail(index, _) -> None:
            work_id = projection_work_end + index
            token_block, out_block = _projection_tile_for_work_id(work_id)
            _start_projection_tile(token_block, out_block)
            _finish_projection_tile(
                token_block,
                out_block,
                wait_previous=work_id > 0,
                wait_final=index == remaining_projection_tiles - 1,
            )

        lax.fori_loop(
            0,
            remaining_projection_tiles,
            _finish_projection_tail,
            None,
            unroll=False,
        )

        # All earlier send semaphores were drained before their slot was
        # reused. Only the final two stages can still be outstanding.
        _wait_qkv_send(active_stages - 2, active_stages)
        _wait_qkv_send(active_stages - 1, active_stages)
        _wait_output_stage(
            active_stages - 2,
            active_stages,
            output_stage_ref,
            packed_out_hbm_ref,
            output_send_sems,
            output_recv_sems,
        )
        _wait_output_stage(
            active_stages - 1,
            active_stages,
            output_stage_ref,
            packed_out_hbm_ref,
            output_send_sems,
            output_recv_sems,
        )

        # A two-phase final barrier prevents a fast rank from entering the next
        # invocation and reusing this remote-copy schedule while a peer is still
        # leaving the current one.
        _all_rank_barrier(
            barrier_sems.at[0],
            rank,
            pcp_size,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
        )
        _all_rank_barrier(
            barrier_sems.at[1],
            rank,
            pcp_size,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
        )

    _run(
        gdn_out_ref,
        packed_out_ref,
        output_send_sems,
        output_recv_sems,
        barrier_sems,
    )


def _scatter_compact_state_updates(
    state_source: jax.Array,
    updates: jax.Array,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    num_active_seqs: jax.Array,
    *,
    state_stride: int,
    region: config.StateRegion,
) -> jax.Array:
    """Apply raw per-sequence state tiles to a donated unified pool.

    Pallas emits only the state bytes touched by this invocation.  Expressing
    the final pool update with native dynamic slices keeps the mutation visible
    to JAX/XLA and allows donation to reuse the input allocation, without
    making the whole pool a Pallas output.
    """
    num_sequences = state_indices.shape[0]
    active_bound = jnp.minimum(num_active_seqs, num_sequences)

    def _apply_one(sequence_id, pool):
        state_index = state_indices[sequence_id]
        block_start = state_index * state_stride + region.kb0
        has_tokens = (query_start_loc[sequence_id + 1]
                      > query_start_loc[sequence_id])
        valid = jnp.logical_and(has_tokens, state_index >= 0)
        valid = jnp.logical_and(
            valid,
            block_start + region.nblocks <= state_source.shape[0],
        )

        def _write(pool):
            start_indices = (block_start, region.row0,
                             *(0, ) * (state_source.ndim - 2))
            return lax.dynamic_update_slice(pool, updates[sequence_id],
                                            start_indices)

        return lax.cond(valid, _write, lambda pool: pool, pool)

    # A runtime loop scales the update work with the live request count rather
    # than the compile bucket's maximum sequence capacity.
    return lax.fori_loop(0, active_bound, _apply_one, state_source)


@functools.partial(
    jax.jit,
    donate_argnames=("state_source", ),
    static_argnames=(
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "kernel_size",
        "pcp_size",
        "comm_chunk_size",
        "mixed_tile_size",
        "compute_precision",
        "mesh_axis_names",
        "pcp_axis_name",
        "state_plan",
    ),
)
def fused_qkvz_projection_pcp_gdn(
    hidden_states: jax.Array,
    qkvz_weight: jax.Array,
    qkvz_weight_scale: jax.Array,
    b: jax.Array,
    a: jax.Array,
    state_source: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    a_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pcp_size: int,
    comm_chunk_size: int,
    mesh_axis_names: tuple[str, ...],
    pcp_axis_name: str,
    state_plan: config.StateSourcePlan,
    mixed_tile_size: int = 64,
    compute_precision: jnp.dtype = jnp.float32.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run ragged PCP prefill with fused QKVZ projection and communication.

    hidden_states is the rank-local token shard. b and a have already been
    exchanged and restored to global request-major order for the PCP-local
    value-head shard. state_source is this rank's unified KV pool; state_plan
    describes its conv and recurrent byte regions. Runtime stage metadata
    handles mixed request lengths and arbitrary absolute query offsets without
    expanding active work to the compile bucket.
    """
    if pcp_size <= 1:
        raise ValueError("The PCP fused GDN kernel requires pcp_size > 1.")
    if pcp_axis_name not in mesh_axis_names:
        raise ValueError(
            f"PCP axis {pcp_axis_name!r} is absent from mesh axes "
            f"{mesh_axis_names!r}.")
    if mesh_axis_names.count(pcp_axis_name) != 1:
        raise ValueError(
            f"PCP axis {pcp_axis_name!r} must occur exactly once in mesh axes "
            f"{mesh_axis_names!r}.")
    if comm_chunk_size <= 0:
        raise ValueError("comm_chunk_size must be positive.")
    dma_row_alignment = 8 * (4 // hidden_states.dtype.itemsize)
    if comm_chunk_size % dma_row_alignment != 0:
        raise ValueError(
            "comm_chunk_size must align to the TPU VMEM row tile: "
            f"{comm_chunk_size=} {dma_row_alignment=}.")
    if n_kq % pcp_size != 0 or n_v % pcp_size != 0:
        raise ValueError("GDN head counts must be divisible by pcp_size.")
    if b.shape != a.shape:
        raise ValueError(f"b and a shapes must match, got {b.shape} and "
                         f"{a.shape}.")
    if state_plan is None:
        raise ValueError("The unified-pool PCP kernel requires a state plan.")

    global_batch_size = b.shape[0]
    local_num_tokens = hidden_states.shape[0]
    if global_batch_size != local_num_tokens * pcp_size:
        raise ValueError(
            "Global BA rows must equal PCP size times local hidden-state rows: "
            f"{global_batch_size=} local_rows={local_num_tokens} "
            f"{pcp_size=}.")
    if local_num_tokens % comm_chunk_size != 0:
        raise ValueError(
            "Rank-local hidden-state rows must be divisible by comm_chunk_size: "
            f"local_rows={local_num_tokens} {comm_chunk_size=}.")

    full_key_dim = n_kq * d_k
    full_value_dim = n_v * d_v
    local_n_kq = n_kq // pcp_size
    local_n_v = n_v // pcp_size
    shard_key_dim = local_n_kq * d_k
    shard_value_dim = local_n_v * d_v
    local_dim = 2 * shard_key_dim + shard_value_dim
    expected_local_qkv_dim = 2 * full_key_dim + full_value_dim
    # Two communication chunks keep short ragged batches from doing a large
    # rounded-up projection while still amortizing the projection loop.
    projection_token_block_size = 2 * comm_chunk_size
    projection_out_block_size = local_dim
    expected_qkvz_dim = expected_local_qkv_dim + full_value_dim
    if hidden_states.ndim != 2 or qkvz_weight.ndim != 2:
        raise ValueError("Projection fusion requires rank-2 hidden states "
                         "and QKVZ weight.")
    if qkvz_weight_scale.ndim != 1:
        raise ValueError("Projection fusion requires a per-channel rank-1 "
                         "weight scale.")
    if hidden_states.dtype != jnp.bfloat16:
        raise ValueError("Projection fusion requires BF16 hidden states, got "
                         f"{hidden_states.dtype}.")
    if qkvz_weight.dtype != jnp.float8_e4m3fn:
        raise ValueError("Projection fusion requires float8_e4m3fn weights, "
                         f"got {qkvz_weight.dtype}.")
    if qkvz_weight_scale.dtype != jnp.float32:
        raise ValueError("Projection fusion requires FP32 weight scales, got "
                         f"{qkvz_weight_scale.dtype}.")
    if hidden_states.shape[1] != qkvz_weight.shape[1]:
        raise ValueError("Projection input and weight K dimensions differ: "
                         f"{hidden_states.shape[1]} and "
                         f"{qkvz_weight.shape[1]}.")
    if qkvz_weight.shape[0] != expected_qkvz_dim:
        raise ValueError("Unexpected QKVZ projection width: "
                         f"expected={expected_qkvz_dim}, "
                         f"got={qkvz_weight.shape[0]}.")
    if qkvz_weight_scale.shape[0] != expected_qkvz_dim:
        raise ValueError(
            "QKVZ scale width differs from the projection weight: "
            f"{qkvz_weight_scale.shape[0]} and "
            f"{expected_qkvz_dim}.")
    if local_num_tokens % projection_token_block_size != 0:
        raise ValueError("Rank-local tokens must be divisible by two "
                         "communication chunks for projection fusion: "
                         f"{local_num_tokens=} "
                         f"{projection_token_block_size=}.")
    if comm_chunk_size % _TPU_TILE_ROWS != 0:
        raise ValueError(
            "The PCP communication chunk must be divisible by the TPU tile "
            f"height ({_TPU_TILE_ROWS}), got {comm_chunk_size=}.")
    if hidden_states.shape[1] % 256 != 0:
        raise ValueError("Projection K dimension must be MXU aligned to 256, "
                         f"got {hidden_states.shape[1]}.")
    if (shard_key_dim % _TPU_TILE_COLUMNS != 0
            or shard_value_dim % _TPU_TILE_COLUMNS != 0
            or shard_value_dim > _TPU_TILE_ROWS * _TPU_TILE_COLUMNS):
        raise ValueError(
            "QKVZ projection requires tile-aligned PCP-local shards and a Z "
            "shard that fits one physical output tile: "
            f"{shard_key_dim=} {shard_value_dim=} "
            f"{projection_out_block_size=}.")
    projection_cfg = _ProjectionConfig(
        token_block_size=projection_token_block_size,
        out_block_size=projection_out_block_size,
        qkv_dim=expected_local_qkv_dim,
    )

    if b.shape[1] != local_n_v:
        raise ValueError(
            f"Expected {local_n_v} PCP-local BA heads, got {b.shape[1]}.")
    if state_source.ndim < 3:
        raise ValueError("The unified state source must have at least three "
                         f"dimensions, got {state_source.shape}.")

    global_stage_tokens = pcp_size * comm_chunk_size
    tile_size = min(mixed_tile_size, global_stage_tokens, global_batch_size)
    if global_stage_tokens % tile_size != 0:
        raise ValueError(
            "The global PCP communication stage must be divisible by the GDN "
            f"compute tile: {global_stage_tokens=} {tile_size=}.")
    num_qkv_out_blocks = pcp_size
    num_z_out_blocks = pcp_size
    num_projection_out_blocks = num_qkv_out_blocks + num_z_out_blocks

    act_out_dtype = hidden_states.dtype

    b = b.astype(jnp.float32)
    a = a.astype(jnp.float32)
    conv_weight = conv_weight.swapaxes(0, 2).astype(jnp.float32)
    conv_bias = (None if conv_bias is None else conv_bias.astype(jnp.float32))

    num_lanes = pltpu.get_tpu_info().num_lanes
    aligned_num_v_heads = pl.cdiv(local_n_v, num_lanes) * num_lanes
    num_v_padding = aligned_num_v_heads - local_n_v
    b = jnp.pad(b,
                ((0, 0), (0, num_v_padding))).reshape(global_batch_size, 1,
                                                      aligned_num_v_heads)
    a = jnp.pad(a,
                ((0, 0), (0, num_v_padding))).reshape(global_batch_size, 1,
                                                      aligned_num_v_heads)

    cfg = config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=jnp.float32.dtype,
            act_out=act_out_dtype,
            compute=compute_precision,
            recurrent_state=jnp.float32.dtype,
            conv_state=jnp.float32.dtype,
        ),
        batch_size=global_batch_size,
        dim_size=local_dim,
        kernel_size=kernel_size,
        tile_size=tile_size,
        num_kq_heads=local_n_kq,
        num_v_heads=local_n_v,
        kq_head_dim=d_k,
        v_head_dim=d_v,
        state_plan=state_plan,
    )
    receive_group_stride = _qkv_receive_group_stride(comm_chunk_size,
                                                     tile_size)

    metadata_obj, stage_metadata_obj = (
        pcp_metadata.compute_pcp_stage_metadata(
            cfg=cfg,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            # The non-PCP V3 wrapper sends the leading decode segment to a
            # separate batched kernel.  This fused PCP op has no such sibling:
            # it must cover that segment with the correctness-first PER_SEQ
            # path as well.  Skipping ``distribution[0]`` requests makes a
            # decode-only batch produce zero active tiles.
            start_seq=jnp.zeros_like(distribution[0]),
            end_seq=distribution[-1],
            pcp_size=pcp_size,
            comm_chunk_size=comm_chunk_size,
            projection_token_block_size=projection_cfg.token_block_size,
            num_qkv_out_blocks=num_qkv_out_blocks,
            num_projection_out_blocks=num_projection_out_blocks,
        ))
    metadata_spec = jax.tree.map(
        lambda _: pl.BlockSpec(memory_space=pltpu.SMEM),
        metadata_obj,
    )
    stage_metadata_spec = jax.tree.map(
        lambda _: pl.BlockSpec(memory_space=pltpu.SMEM),
        stage_metadata_obj,
    )

    conv_weights = memory_ref.ConvWeightsRef(
        weight=conv_weight,
        bias=conv_bias,
    )
    gdn_weights = memory_ref.GDNWeightsRef(
        a_log=a_log,
        dt_bias=dt_bias,
    )
    weights = memory_ref.WeightRefs(
        conv=conv_weights,
        gdn=gdn_weights,
    )
    weights_spec = jax.tree.map(
        lambda _: pl.BlockSpec(memory_space=pltpu.VMEM),
        weights,
    )

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    gdn_out_shape = cfg.get_out_shape()
    zero_gdn_out = jnp.zeros(gdn_out_shape.shape, gdn_out_shape.dtype)
    packed_out_shape = (
        local_num_tokens,
        pcp_size,
        local_n_v,
        d_v,
    )
    zero_packed_out = jnp.zeros(packed_out_shape, act_out_dtype)
    projected_qkv_shape = jax.ShapeDtypeStruct(
        (
            pcp_size,
            local_num_tokens // _TPU_TILE_ROWS,
            local_dim // _TPU_TILE_COLUMNS,
            _TPU_TILE_ROWS,
            _TPU_TILE_COLUMNS,
        ),
        act_out_dtype,
    )
    z_shape = jax.ShapeDtypeStruct(
        (
            local_num_tokens,
            num_z_out_blocks,
            _TPU_TILE_ROWS,
            _TPU_TILE_COLUMNS,
        ),
        act_out_dtype,
    )
    state_update_tail = state_source.shape[2:]
    conv_state_updates_shape = jax.ShapeDtypeStruct(
        (
            state_indices.shape[0],
            state_plan.conv.nblocks,
            state_plan.conv.nrows,
            *state_update_tail,
        ),
        state_source.dtype,
    )
    recurrent_state_updates_shape = jax.ShapeDtypeStruct(
        (
            state_indices.shape[0],
            state_plan.recurrent.nblocks,
            state_plan.recurrent.nrows,
            *state_update_tail,
        ),
        state_source.dtype,
    )
    projection_scratch_shapes = (
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.VMEM(
            (projection_cfg.token_block_size, hidden_states.shape[1]),
            hidden_states.dtype,
        ),
        pltpu.VMEM(
            (projection_cfg.token_block_size, hidden_states.shape[1]),
            qkvz_weight.dtype,
        ),
        pltpu.VMEM(
            (projection_cfg.token_block_size, ),
            jnp.float32,
        ),
        pltpu.VMEM(
            (projection_cfg.out_block_size, hidden_states.shape[1]),
            qkvz_weight.dtype,
        ),
        pltpu.VMEM(
            (projection_cfg.out_block_size, ),
            qkvz_weight_scale.dtype,
        ),
        pltpu.VMEM(
            (
                projection_cfg.token_block_size // _TPU_TILE_ROWS,
                projection_cfg.out_block_size // _TPU_TILE_COLUMNS,
                _TPU_TILE_ROWS,
                _TPU_TILE_COLUMNS,
            ),
            act_out_dtype,
        ),
        pltpu.VMEM(
            (
                projection_cfg.token_block_size,
                _TPU_TILE_ROWS,
                _TPU_TILE_COLUMNS,
            ),
            act_out_dtype,
        ),
    )
    metadata_leaves = len(metadata_obj)
    stage_metadata_leaves = len(jax.tree_util.tree_leaves(stage_metadata_obj))
    state_input_offset = metadata_leaves + stage_metadata_leaves
    input_output_aliases = {
        state_input_offset + 6: 0,
        state_input_offset + 7: 1,
    }
    carry_shapes = cfg.get_scratch_shape_dict()
    kernel_name = ("fused_pcp_qkvz_projection_gdn_per_seq_compact_qkv"
                   f"_c{comm_chunk_size}_p{pcp_size}_pooled")
    # The compiler may retain more transient tiles as this limit rises. Use
    # the same 90% hardware budget as the production GMM kernels.
    vmem_limit_bytes = int(0.90 * pltpu.get_tpu_info().vmem_capacity_bytes)

    (gdn_out, packed_out, conv_state_updates, recurrent_state_updates,
     _projected_qkv, z, active_rows) = pl.pallas_call(
         functools.partial(
             _pcp_qkvz_projection_gdn_outer_kernel,
             cfg=cfg,
             pcp_size=pcp_size,
             comm_chunk_size=comm_chunk_size,
             projection_cfg=projection_cfg,
             mesh_axis_names=mesh_axis_names,
             pcp_axis_name=pcp_axis_name,
         ),
         out_shape=(
             zero_gdn_out,
             zero_packed_out,
             conv_state_updates_shape,
             recurrent_state_updates_shape,
             projected_qkv_shape,
             z_shape,
             jax.ShapeDtypeStruct((1, ), jnp.int32),
         ),
         in_specs=(
             metadata_spec,
             stage_metadata_spec,
             hbm_spec,
             hbm_spec,
             hbm_spec,
             hbm_spec,
             hbm_spec,
             hbm_spec,
             hbm_spec,
             hbm_spec,
             weights_spec,
         ),
         out_specs=(
             hbm_spec,
             hbm_spec,
             hbm_spec,
             hbm_spec,
             hbm_spec,
             hbm_spec,
             hbm_spec,
         ),
         scratch_shapes=(
             carry_shapes["carry_conv_scratch_ref"],
             carry_shapes["carry_recurrent_scratch_ref"],
             pltpu.SemaphoreType.DMA((2, )),
             pltpu.SemaphoreType.DMA((2, )),
             pltpu.SemaphoreType.DMA((2, )),
             pltpu.SemaphoreType.DMA((2, )),
             pltpu.SemaphoreType.REGULAR((2, )),
             pltpu.VMEM(
                 (
                     2,
                     pcp_size * receive_group_stride,
                     local_dim // _TPU_TILE_COLUMNS,
                     _TPU_TILE_ROWS,
                     _TPU_TILE_COLUMNS,
                 ),
                 act_out_dtype,
             ),
             pltpu.VMEM(
                 (1, tile_size, 1, local_dim),
                 jnp.float32,
             ),
             pltpu.VMEM(
                 (2, global_stage_tokens, local_n_v, d_v),
                 act_out_dtype,
             ),
             *projection_scratch_shapes,
             pltpu.SMEM((1, ), jnp.int32),
             pltpu.SemaphoreType.DMA,
         ),
         input_output_aliases=input_output_aliases,
         compiler_params=pltpu.CompilerParams(
             disable_bounds_checks=True,
             vmem_limit_bytes=vmem_limit_bytes,
         ),
         name=kernel_name,
         metadata=cfg.get_metadata(),  # pyrefly: ignore[bad-argument-type]
     )(
         metadata_obj,
         stage_metadata_obj,
         hidden_states,
         qkvz_weight,
         qkvz_weight_scale,
         b,
         a,
         state_source,
         zero_gdn_out,
         zero_packed_out,
         weights,
     )

    num_active_seqs = distribution[-1]
    state_source = _scatter_compact_state_updates(
        state_source,
        conv_state_updates,
        state_indices,
        query_start_loc,
        num_active_seqs,
        state_stride=state_plan.stride,
        region=state_plan.conv,
    )
    state_source = _scatter_compact_state_updates(
        state_source,
        recurrent_state_updates,
        state_indices,
        query_start_loc,
        num_active_seqs,
        state_stride=state_plan.stride,
        region=state_plan.recurrent,
    )

    packed_out = packed_out.reshape(local_num_tokens, n_v, d_v)
    value_tile_rows = shard_value_dim // _TPU_TILE_COLUMNS
    z = z[:, :, :value_tile_rows, :].reshape(local_num_tokens, n_v, d_v)
    # ``z`` is intentionally written only for live projection blocks so a
    # short invocation does not acquire the static compile bucket's matmul
    # work. Its unwritten HBM suffix is unspecified; do not expose that suffix
    # to normalization or later compiled layers.
    live_row = jnp.arange(local_num_tokens, dtype=jnp.int32) < active_rows[0]
    z = jnp.where(live_row[:, None, None], z, jnp.zeros((), dtype=z.dtype))
    return state_source, packed_out, z
