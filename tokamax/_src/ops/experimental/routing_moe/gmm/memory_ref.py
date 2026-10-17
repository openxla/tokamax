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

"""Module that encapsulates memory references, BlockSpecs, and DMA logic."""

import abc
import dataclasses
import functools
from typing import Any, override

import jax
from jax import lax
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.gmm import config
from tokamax._src.ops.experimental.routing_moe.gmm import metadata


GridIndices = tuple[jax.Array | int, jax.Array | int, jax.Array | int]


def map_memory_constraint(obj, memory_space):
  return jax.tree.map(
      lambda x: pltpu.with_memory_space_constraint(x, memory_space), obj
  )


class AbstractWeightRef(abc.ABC):
  """Abstract class that defines interfaces for rhs values."""

  @abc.abstractmethod
  def get_weight(self) -> jax.Array:
    ...

  @abc.abstractmethod
  def get_scale(self, replicate_size: int | None = None) -> jax.Array | None:
    """Returns scale array, optionally replicated across sublanes."""
    ...

  @abc.abstractmethod
  def get_bias(self) -> jax.Array:
    ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SingleWeightRef[T](AbstractWeightRef):
  """Dataclass for a single weights."""

  weight: T
  scale: T | None
  bias: T | None

  def get_weight(self) -> jax.Array:
    assert isinstance(self.weight, hijax.TransformedRef)
    return self.weight[...]

  def get_scale(self, replicate_size: int | None = None) -> jax.Array:
    assert self.scale is not None
    assert isinstance(self.scale, hijax.TransformedRef)
    if replicate_size is not None:
      # Perform zero-stride load for efficient broadcasting across sublanes.
      return self.scale[:, pl.ds(0, replicate_size, 0), :]
    return self.scale[...]

  def get_bias(self) -> jax.Array:
    assert self.bias is not None
    assert isinstance(self.bias, hijax.TransformedRef)
    return self.bias[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FusedWeightsRef[T](AbstractWeightRef):
  """Dataclass for gate and up weights used in fused activation."""

  gate: SingleWeightRef[T]
  up: SingleWeightRef[T]

  def get_weight(self) -> jax.Array:
    w_gate = self.gate.get_weight()
    w_up = self.up.get_weight()
    return layout_util.large_interleave_lane(w_gate, w_up)

  def get_scale(self, replicate_size: int | None = None) -> jax.Array:
    s_gate = self.gate.get_scale(replicate_size)
    s_up = self.up.get_scale(replicate_size)

    return layout_util.large_interleave_lane(s_gate, s_up)

  def get_bias(self) -> jax.Array:
    b_gate = self.gate.get_bias()
    b_up = self.up.get_bias()
    return layout_util.large_interleave_lane(b_gate, b_up)


class _BypassBufferedRef(pltpu.BufferedRef):

  # NOTE: This is extremely hacky and only here until JAX version update
  # arrives which WILL eliminate the need for this hack.
  # TODO: Remove this after JAX version update.
  def __post_init__(self):
    # pallas doesn't allow you to set buffer_count > 2 for output refs, so
    # we override to bypass this check.
    pass


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class RoutingBufferedRef(_BypassBufferedRef):
  """Shared common interface for both all routed BufferedRef instances."""

  metadata_ref: metadata.MetadataRef = dataclasses.field(
      metadata=dict(static=True)
  )
  # SMEM ref used for indexing src or dst ref.
  idx_ref: Any = dataclasses.field(default=None, metadata=dict(static=True))

  cfg: config.RoutingGmmConfig = dataclasses.field(metadata=dict(static=True))

  def bind_idx_ref(self, idx_bref):
    return dataclasses.replace(self, idx_ref=idx_bref.window_ref)

  @classmethod
  def create(
      cls,
      *args: Any,
      metadata_ref: metadata.MetadataRef,
      cfg: config.RoutingGmmConfig,
      **kwargs,
  ):
    standard_ref = _BypassBufferedRef.create(*args, **kwargs)
    return cls(
        metadata_ref=metadata_ref,
        cfg=cfg,
        **{
            f.name: getattr(standard_ref, f.name)
            for f in dataclasses.fields(pltpu.BufferedRef)
        },
    )

  @abc.abstractmethod
  def branchless_copy_in(
      self, src_ref: hijax.TransformedRef, grid_indices: GridIndices
  ) -> None:
    ...

  @abc.abstractmethod
  def branchless_wait_in(
      self, src_ref: hijax.TransformedRef, grid_indices: GridIndices
  ) -> None:
    ...

  @abc.abstractmethod
  def branchless_copy_out(
      self, dst_ref: hijax.TransformedRef, grid_indices: GridIndices
  ) -> None:
    ...

  @abc.abstractmethod
  def branchless_wait_out(
      self, dst_ref: hijax.TransformedRef, grid_indices: GridIndices
  ) -> None:
    ...

  def prologue(self) -> None:
    """Runs once before the pipeline."""

  def epilogue(self, dst_ref: hijax.TransformedRef) -> None:
    """Runs once after the last DMA wait."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BranchedBufferedRef(RoutingBufferedRef):
  """BufferedRef managed by emit_pipeline. Not overlaped with compute."""

  def _no_op(self, ref, grid_indices) -> None:
    del ref, grid_indices

  branchless_copy_in = branchless_wait_in = branchless_copy_out = (
      branchless_wait_out
  ) = _no_op


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class IndexBufferedRef(pltpu.BufferedRef):
  """Streams per-row routing indices from HBM into an SMEM ring buffer.

  This class is carefully designed to ensure downstream consumer object does
  not need to know about implementations details about how indices are buffered.
  To do so, we implement following core ideas.

  1: Beside semaphore management, there is no concrete concept of "slots".
    Each copy_in treats the buffer as a continuous memory space and performs
    write with respect to its global memory space without concerning which step
    that this copy_in is being invoked.

  2: Because there is no concrete concept of "slots", it was possible for a
    copy_in to overwrite a data region that is being used by compute step.
    To avoid this, we leverage the fact that even at worst case scenario,
    copy_in will write to 2*tile_m memory range. So we slightly overallocate
    to ensure there is a safe "gap" between copy_in range and compute step.

  3: A buffer is allocated in a way to contain "runway" where there is a valid
    memory space beyond "tile_m*num_buffers". This is to ensure that a
    single DMA can write its indices without worrying about the right end
    boundary edge case. This also means that we eliminate the need for
    (offset + i) % buffer_size check that is typically needed for ring buffer.
    This is especially important as Branchless execution is often SALU bound.

  Here is more concrete example:

  In below example, the maximum DMA transfer size is 2 elements and each
  number indicate indices needed for each step.
  | 0 1 | 1 2 | 2 3 | 3 4 | 4 5 |

  If we implement a naive ring buffer, at step 2, the data in SMEM will be
  stored like this where local_row_offset=3 and row_size=2:
  | 2 3 | 1 2 |

  This approach causes a problem because:
  1: When copy_in for the step 3 is being done in parallel, it will attempt
    to overwrite the data in "| 1 2 |" into "| 3 4 |" and cause numeric issues.
  2: To ensure local_row_offset + i does not trigger out-of-bound read, it
    need to introduce idx[(offset + i) % total_buffer_size]

  To combat this, we slightly overallocate the buffer than what is strictly
  necessary. For instance, at step 4, the data will look like this (note:
  acutal allocation introduces even more padding but it was ultimately choosen
  as it simplified many of the other core logics.)
  | 0 1 | 1 2 | 2 3 || 3 4 |

  "||" denotes that while write start cannot happen beyond this point, write
  end can happen beyond that point.

  This approach is able to fix above 2 issues mentioned:
  1: When copy_in for the step 4 is happening, it will overwrite "| 0 1 | 1 2 |"
    but the "| 2 3 || 3 4 |" that is being used for compute remains untouched.
  2: Because the valid data is allocated beyond total_buffer_size, downstream
    code can cleanly perform idx[offset + i] without worrying about OOB access.
  """

  metadata_ref: metadata.MetadataRef = dataclasses.field(
      metadata=dict(static=True)
  )
  cfg: config.RoutingGmmConfig = dataclasses.field(metadata=dict(static=True))
  fetch_offset: int = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(
      cls,
      *args: Any,
      cfg: config.RoutingGmmConfig,
      metadata_ref: metadata.MetadataRef,
      fetch_offset: int,
      **kwargs,
  ):
    standard_ref = pltpu.BufferedRef.create(*args, **kwargs)
    return cls(
        metadata_ref=metadata_ref,
        cfg=cfg,
        fetch_offset=fetch_offset,
        **{
            f.name: getattr(standard_ref, f.name)
            for f in dataclasses.fields(pltpu.BufferedRef)
        },
    )

  def _aligned_range(
      self,
      n_id: jax.Array | int,
      gm_id: jax.Array | int,
      k_id: jax.Array | int,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns (hbm start, local start, size) aligned to tile_m_idx."""
    block_size = jnp.uint32(self.cfg.plan.tile_m_idx)
    ring_size = jnp.uint32(self.cfg.plan.idx_ring_size)
    idx_buffer_size = jnp.uint32(self.cfg.plan.idx_buffer_size)
    buffer_count = jnp.uint32(self.buffer_count)
    n_id, gm_id, _ = self.metadata_ref.resolve(
        n_id, gm_id, k_id, dtype=jnp.uint32
    )
    # Calculate hbm address.
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id].astype(jnp.uint32)
    m_size = self.metadata_ref.gm_id_to_num_rows[gm_id].astype(jnp.uint32)
    m_end = m_start + m_size
    aligned_start = lax.div(m_start, block_size) * block_size
    aligned_end = pl.cdiv(m_end, block_size) * block_size

    # Calculate smem address. Each odd/even n value uses a different buffer
    # to ensure first step of n + 1 never overwrites the last step of n.
    buffer_offset = lax.rem(n_id, buffer_count) * idx_buffer_size
    local_start = lax.rem(m_start, ring_size)
    aligned_local_start = lax.div(local_start, block_size) * block_size
    aligned_buffer_start = buffer_offset + aligned_local_start
    return aligned_start, aligned_buffer_start, aligned_end - aligned_start

  @override
  def copy_in(self, idx_hbm_ref, grid_indices):
    assert self.is_input
    assert len(grid_indices) == 3
    sem_recvs = self.sem_recvs
    window_ref = self.window_ref
    assert sem_recvs is not None
    assert window_ref is not None
    n_id, gm_id, k_id = grid_indices
    slot = self.metadata_ref.slot(n_id, gm_id, k_id, self.buffer_count)

    aligned_start, local_start, aligned_size = self._aligned_range(
        n_id, gm_id + self.fetch_offset, k_id
    )
    pltpu.make_async_copy(
        src_ref=idx_hbm_ref.at[pl.ds(aligned_start, aligned_size)],
        dst_ref=window_ref.at[pl.ds(local_start, aligned_size)],
        sem=sem_recvs.at[slot],
    ).start()

  @override
  def wait_in(self, idx_hbm_ref, grid_indices):
    assert self.is_input
    assert len(grid_indices) == 3
    sem_recvs = self.sem_recvs
    window_ref = self.window_ref
    assert sem_recvs is not None
    assert window_ref is not None
    n_id, gm_id, k_id = grid_indices
    slot = self.metadata_ref.slot(n_id, gm_id, k_id, self.buffer_count)

    aligned_size = self._aligned_range(n_id, gm_id + self.fetch_offset, k_id)[
        -1
    ]
    copied = window_ref.at[pl.ds(0, aligned_size)]
    pltpu.make_async_copy(copied, copied, sem_recvs.at[slot]).wait()

    if self.fetch_offset < 0:
      # One of the unfortunate side-effect of piggy-backing on emit_pipeline
      # is that it fails to fetch a data for gm_id step because the last fetch
      # ends at (num_gm - 1) - fetch_offset. Therefore, we have to manually
      # perform fetch for the last gm_id.
      assert self.fetch_offset == -1
      step = self.metadata_ref.step(n_id, gm_id, k_id)
      is_last_step = step == self.metadata_ref.num_steps - 1

      # This can probably be done with branchless DMA but unlikely to be worth
      # it since this entire wait_in is already executed under emit_pipeline's
      # dynamic branch anyway.
      @pl.when(is_last_step)
      def _():
        next_slot = self.metadata_ref.slot(
            n_id, gm_id + 1, k_id, self.buffer_count
        )

        aligned_start, local_start, aligned_size = self._aligned_range(
            n_id, gm_id, k_id
        )
        # NOTE: Because this copy_in is manually triggered at the last step,
        # the caller MUST trigger corresponding wait_in at the epilogue.
        pltpu.make_async_copy(
            src_ref=idx_hbm_ref.at[pl.ds(aligned_start, aligned_size)],
            dst_ref=window_ref.at[pl.ds(local_start, aligned_size)],
            sem=sem_recvs.at[next_slot],
        ).start()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BranchlessBufferedRef(RoutingBufferedRef):
  """BufferedRef that performs branchless DMAs to allow matmul overlap."""

  def _idx_offset(
      self, grid_indices: GridIndices, rows_ds: pl.Slice
  ) -> jax.Array:
    """SMEM offset that accounts for N tile dim buffering."""
    idx_buffer_size = jnp.uint32(self.cfg.plan.idx_buffer_size)
    ring_size = jnp.uint32(self.cfg.plan.idx_ring_size)
    num_idx_buffers = jnp.uint32(self.cfg.plan.num_idx_buffers)
    n_id, _, _ = self.metadata_ref.resolve(*grid_indices, dtype=jnp.uint32)
    # Buffer is divided into two where only even/odd n_id can use to avoid
    # first step of n_id oveerwriting value of last step at n_id - 1.
    n_offset = lax.rem(n_id, num_idx_buffers) * idx_buffer_size
    rows_ds_start = jnp.uint32(rows_ds.start)
    return n_offset + lax.rem(rows_ds_start, ring_size)

  @override
  def branchless_copy_in(
      self, src_ref: hijax.TransformedRef, grid_indices: GridIndices
  ) -> None:
    assert self.sem_recvs is not None
    assert self.idx_ref is not None

    slot = self.metadata_ref.slot(*grid_indices, self.buffer_count)
    rows_ds, cols_ds, *_ = self.compute_index(*grid_indices)
    sem = self.sem_recvs.at[slot]
    vmem_dst = pltpu.annotate(self._window_ref_at(slot), no_hazard_no_deps=True)
    offset = self._idx_offset(grid_indices, rows_ds)

    for r in range(self.cfg.plan.tile_m):
      size = jnp.where(r < rows_ds.size, 1, 0)
      row = self.idx_ref[offset + r]
      pltpu.make_async_copy(
          src_ref=src_ref.at[pl.ds(row, size), cols_ds],
          dst_ref=vmem_dst.at[pl.ds(r, size), pl.ds(0, cols_ds.size)],
          sem=sem,
      ).start()

  @override
  def branchless_copy_out(
      self, dst_ref: hijax.TransformedRef, grid_indices: GridIndices
  ) -> None:
    assert self.sem_sends is not None
    assert self.idx_ref is not None

    slot = self.metadata_ref.slot(*grid_indices, self.buffer_count)
    rows_ds, cols_ds, *_ = self.compute_index(*grid_indices)
    sem = self.sem_sends.at[slot]
    vmem_src = pltpu.annotate(self._window_ref_at(slot), no_hazard_no_deps=True)
    offset = self._idx_offset(grid_indices, rows_ds)

    for r in range(self.cfg.plan.tile_m):
      live = r < rows_ds.size
      size = jnp.where(live, 1, 0)
      src = vmem_src.at[pl.ds(r, size), pl.ds(0, cols_ds.size)]
      row = self.idx_ref[offset + r]
      self._copy_out_row(src, dst_ref, row, size, cols_ds, live, sem)

  def _copy_out_row(
      self,
      src_ref: hijax.TransformedRef,
      dst_ref: hijax.TransformedRef,
      idx: jax.Array,
      size: jax.Array,
      cols_ds: pl.Slice,
      live: jax.Array,
      sem: Any,
  ) -> None:
    """Starts the DMA of one VMEM row `src` to output row `idx`."""
    del live
    pltpu.make_async_copy(
        src_ref=src_ref,
        dst_ref=dst_ref.at[pl.ds(idx, size), cols_ds],
        sem=sem,
    ).start(priority=1)

  @override
  def branchless_wait_in(
      self, src_ref: hijax.TransformedRef, grid_indices: GridIndices
  ) -> None:
    assert self.sem_recvs is not None

    slot = self.metadata_ref.slot(*grid_indices, self.buffer_count)
    rows_ds, cols_ds, *_ = self.compute_index(*grid_indices)
    # Only the byte count matters to the wait, so src_ref stands in for both.
    copied = src_ref.at[pl.ds(0, rows_ds.size), cols_ds]
    pltpu.make_async_copy(copied, copied, self.sem_recvs.at[slot]).wait()

  @override
  def branchless_wait_out(
      self, dst_ref: hijax.TransformedRef, grid_indices: GridIndices
  ) -> None:
    assert self.sem_sends is not None

    slot = self.metadata_ref.slot(*grid_indices, self.buffer_count)
    rows_ds, cols_ds, *_ = self.compute_index(*grid_indices)
    copied = dst_ref.at[pl.ds(0, rows_ds.size), cols_ds]
    pltpu.make_async_copy(copied, copied, self.sem_sends.at[slot]).wait()

  def _no_op(self, ref, grid_indices):
    del ref, grid_indices

  copy_in = copy_out = wait_in = wait_out = _no_op


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RemoteBranchlessBufferedRef(BranchlessBufferedRef):
  """Output ref that DMAs each row to the device that owns it.

  Attributes:
    recv_sem: DMA semaphore counting the rows this device receives from all
      devices.
  """

  recv_sem: hijax.TransformedRef | None = None

  @override
  def _copy_out_row(
      self,
      src_ref: hijax.TransformedRef,
      dst_ref: hijax.TransformedRef,
      idx: jax.Array,
      size: jax.Array,
      cols_ds: pl.Slice,
      live: jax.Array,
      sem: Any,
  ) -> None:
    assert self.cfg.scatter is not None
    axis = self.cfg.scatter.remote_axis
    assert axis is not None
    assert self.recv_sem is not None
    # Unsigned lax div/rem skip the sign fixups of int32 jnp.divmod.
    rows_per_dev = jnp.uint32(self.cfg.out.spec.row_size)
    uidx = idx.astype(jnp.uint32)
    dev = lax.div(uidx, rows_per_dev).astype(jnp.int32)
    row = lax.rem(uidx, rows_per_dev).astype(jnp.int32)
    # Remote DMAs must use priority 0. A dead row may read a stale index, so
    # its empty DMA is aimed at a real device.
    pltpu.make_async_remote_copy(
        src_ref=src_ref,
        dst_ref=dst_ref.at[pl.ds(row, size), cols_ds],
        send_sem=sem,
        recv_sem=self.recv_sem,
        device_id={axis: jnp.where(live, dev, 0)},
    ).start()

  @override
  def prologue(self) -> None:
    # Every peer's output buffer must exist before the first row lands.
    assert self.cfg.scatter is not None
    axis, ep = self.cfg.scatter.remote_axis, self.cfg.dim.size_ep
    assert axis is not None

    def signal_all(sem):
      for i in range(ep):
        pl.semaphore_signal(sem, 1, device_id={axis: jnp.int32(i)})
      pl.semaphore_wait(sem, ep)

    signal_all(pltpu.get_barrier_semaphore())
    pl.run_scoped(signal_all, pltpu.SemaphoreType.REGULAR)

  @override
  def epilogue(self, dst_ref: hijax.TransformedRef) -> None:
    # Wait for every row owned here before the output is read.
    recv = dst_ref.at[pl.ds(0, self.cfg.out.spec.row_size)]
    pltpu.make_async_copy(recv, recv, self.recv_sem).wait()


@dataclasses.dataclass(frozen=True)
class PipelineIndexMaps:
  """Provides grid-to-tile index mapping functions for pipeline BlockSpecs.

  Attributes:
    metadata_ref: SMEM metadata reference providing per-step group and row
      offsets.
    cfg: Static configuration for the routing GMM kernel.
  """

  metadata_ref: metadata.MetadataRef
  cfg: config.RoutingGmmConfig

  def _rows(self, gm_id: jax.Array) -> pl.Slice:
    m_offset = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_size = self.metadata_ref.gm_id_to_num_rows[gm_id]
    return pl.ds(m_offset, m_size)

  def x_index(
      self,
      n_id: jax.Array | int,
      gm_id: jax.Array | int,
      k_id: jax.Array | int,
      *,
      hbm_shape: tuple[int, int, int, int],
  ) -> tuple[pl.Slice, pl.Slice, int, int]:
    """Returns the block index tuple for the input activation."""
    chunks = hbm_shape[1] // self.cfg.num_k
    _, gm_id, k_id = self.metadata_ref.resolve(n_id, gm_id, k_id)
    return (self._rows(gm_id), pl.ds(k_id * chunks, chunks), 0, 0)

  def out_index(
      self,
      n_id: jax.Array | int,
      gm_id: jax.Array | int,
      k_id: jax.Array | int,
      *,
      hbm_shape: tuple[int, int, int, int],
  ) -> tuple[pl.Slice, pl.Slice, int, int]:
    """Returns the block index tuple for the output activation."""
    chunks = hbm_shape[1] // self.cfg.num_n
    n_id, gm_id, _ = self.metadata_ref.resolve(n_id, gm_id, k_id)
    return (self._rows(gm_id), pl.ds(n_id * chunks, chunks), 0, 0)

  def weight_index(
      self, n_id: jax.Array | int, gm_id: jax.Array | int, k_id: jax.Array | int
  ) -> tuple[jax.Array, int, jax.Array | int]:
    """Returns the block index tuple for an expert weight tensor."""
    del k_id
    return (self.metadata_ref.gm_id_to_group_id[gm_id], 0, n_id)

  def scale_index(
      self, n_id: jax.Array | int, gm_id: jax.Array | int, k_id: jax.Array | int
  ) -> tuple[jax.Array, int, int, jax.Array | int]:
    """Returns the block index tuple for a quantization scale tensor."""
    del k_id
    return (self.metadata_ref.gm_id_to_group_id[gm_id], 0, 0, n_id)

  def idx_index(
      self, n_id: jax.Array | int, gm_id: jax.Array | int, k_id: jax.Array | int
  ) -> jax.Array:
    """Returns an index that changes every step so copy_* always triggers."""
    return self.metadata_ref.step(n_id, gm_id, k_id)


@dataclasses.dataclass(frozen=True)
class PipelineBlockSpecs:
  """Factory for routing GMM pipeline BlockSpecs.

  Both DMA paths locate a tile through these index maps, while buffer sizing
  is decided here alone.

  Attributes:
    cfg: Static configuration for the routing GMM kernel.
    index_map_provider: Provider for grid-to-tile index mapping functions.
  """

  cfg: config.RoutingGmmConfig
  index_map_provider: PipelineIndexMaps

  def _act_spec(
      self, hbm_shape: tuple[int, ...], num_tiles: int, index_map
  ) -> pl.BlockSpec:
    tile_chunks = hbm_shape[1] // num_tiles
    num_tile_chunks = layout_util.to_bank_conflict_aware_stride(tile_chunks)
    block_shape = (
        pl.BoundedSlice(self.cfg.plan.tile_m),
        pl.BoundedSlice(num_tile_chunks),
        *hbm_shape[2:],
    )
    index_map = functools.partial(index_map, hbm_shape=hbm_shape)
    return pl.BlockSpec(
        block_shape=block_shape,
        index_map=index_map,
        memory_space=pltpu.VMEM,
        # TODO: This is redundant in later JAX versions, remove this
        # after JAX version update.
        pipeline_mode=pl.Buffered(buffer_count=self.cfg.plan.num_xo_buffers),
    )

  def get_x_spec(self) -> pl.BlockSpec:
    """Returns the BlockSpec for the input activation tensor."""
    return self._act_spec(
        self.cfg.x.spec.compact_shape,
        self.cfg.num_k,
        self.index_map_provider.x_index,
    )

  def get_out_spec(self) -> pl.BlockSpec:
    """Returns the BlockSpec for the output activation tensor."""
    return self._act_spec(
        self.cfg.out.spec.compact_shape,
        self.cfg.num_n,
        self.index_map_provider.out_index,
    )

  def get_weight_spec(self) -> pl.BlockSpec:
    """Returns the BlockSpec for the expert weight tensor."""
    block_shape = (None, self.cfg.plan.tile_k, self.cfg.plan.tile_n)
    return pl.BlockSpec(
        block_shape=block_shape,
        index_map=self.index_map_provider.weight_index,
        memory_space=pltpu.VMEM,
        # TODO: This is redundant in later JAX versions, remove this
        # after JAX version update.
        pipeline_mode=pl.Buffered(
            buffer_count=self.cfg.plan.num_w_buffers, use_lookahead=True
        ),
    )

  def get_scale_spec(self) -> pl.BlockSpec:
    """Returns the BlockSpec for the quantization scale tensor."""
    block_shape = (None, 1, 1, self.cfg.plan.tile_n)
    return pl.BlockSpec(
        block_shape=block_shape,
        index_map=self.index_map_provider.scale_index,
        memory_space=pltpu.VMEM,
        # TODO: This is redundant in later JAX versions, remove this
        # after JAX version update.
        pipeline_mode=pl.Buffered(
            buffer_count=self.cfg.plan.num_w_buffers, use_lookahead=True
        ),
    )

  def get_index_spec(self) -> pl.BlockSpec:
    """Returns the BlockSpec for the routing index."""
    # Single index buffer can houses not just one, but up-to entire
    # num_xo_buffers amount of data. Despite this, the consumer does not need
    # to know about this kind of additional padded allocation.
    block_shape = (self.cfg.plan.idx_buffer_size,)

    return pl.BlockSpec(
        block_shape=block_shape,
        index_map=self.index_map_provider.idx_index,
        memory_space=pltpu.SMEM,
        # TODO: This is redundant in later JAX versions, remove this
        # after JAX version update.
        pipeline_mode=pl.Buffered(buffer_count=self.cfg.plan.num_idx_buffers),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PipelineInputs[T]:
  """Pipelined input parameter bundle for the routing GMM kernel.

  Devoid of stateful data management so emit_pipeline and inner_pipeline can
  remains in the main orchestrators.

  Attributes:
    x: Input activation reference, BlockSpec, or BufferedRef.
    w: Expert weight reference, BlockSpec tree, or BufferedRef tree.
    gather_idx: Row indices used by x. Set to None when DIRECT mode is used.
    scatter_idx: Row indices used by out. Set to None when DIRECT mode is used.
  """

  x: T
  w: AbstractWeightRef
  gather_idx: T | None = None
  scatter_idx: T | None = None

  def transform_for_pipeline(
      self, cfg: config.RoutingGmmConfig
  ):
    """Returns inputs with weights sliced for fused gate/up activation."""
    if cfg.fuse_act is None:
      return self
    assert isinstance(self.w, SingleWeightRef)
    w_fused = FusedWeightsRef(
        gate=self.w,
        up=jax.tree.map(lambda ref: ref.at[..., cfg.dim.size_out_n :], self.w),
    )
    return dataclasses.replace(self, w=w_fused)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PipelineAllocs:
  """BufferedRef allocations for the routing GMM pipeline.

  Devoid of stateful data management so emit_pipeline and inner_pipeline can
  remains in the main orchestrators.

  Attributes:
    inputs: Pipelined input BufferedRef bundle.
    out: Output activation BufferedRef.
  """

  inputs: PipelineInputs[RoutingBufferedRef]
  out: RoutingBufferedRef

  @classmethod
  def create(
      cls,
      in_hbm_ref: PipelineInputs[hijax.TransformedRef],
      out_hbm_ref: hijax.TransformedRef,
      metadata_ref: metadata.MetadataRef,
      cfg: config.RoutingGmmConfig,
  ):
    """Creates BufferedRef allocations for all pipeline inputs and outputs.

    Args:
      in_hbm_ref: HBM references for the pipelined kernel inputs.
      out_hbm_ref: HBM reference for the kernel output.
      metadata_ref: SMEM metadata reference for routing and tile scheduling.
      cfg: Static configuration for the routing GMM kernel.

    Returns:
      A PipelineAllocs instance containing input and output BufferedRefs.
    """
    pipeline_index_maps = PipelineIndexMaps(metadata_ref, cfg)
    pipeline_block_specs = PipelineBlockSpecs(cfg, pipeline_index_maps)
    x_hbm_ref = in_hbm_ref.x
    w_hbm_ref = in_hbm_ref.w
    assert isinstance(w_hbm_ref, SingleWeightRef)

    scale_bref = None
    if cfg.w.has_scale:
      assert w_hbm_ref.scale is not None
      scale_bref = pltpu.BufferedRef.input(
          spec=pipeline_block_specs.get_scale_spec(),
          dtype_or_type=w_hbm_ref.scale.dtype,
          buffer_count=cfg.plan.num_w_buffers,
          use_lookahead=True,
          grid_rank=3,
      )

    weight_bref = pltpu.BufferedRef.input(
        spec=pipeline_block_specs.get_weight_spec(),
        dtype_or_type=w_hbm_ref.weight.dtype,
        buffer_count=cfg.plan.num_w_buffers,
        use_lookahead=True,
        grid_rank=3,
    )
    w_bref = SingleWeightRef(weight=weight_bref, scale=scale_bref, bias=None)
    if cfg.fuse_act is not None:
      w_bref = FusedWeightsRef(gate=w_bref, up=w_bref)

    def _get_buffered_ref(mode: config.RoutingMode) -> type[RoutingBufferedRef]:
      match mode:
        case config.RoutingMode.LOCAL:
          return BranchlessBufferedRef
        case config.RoutingMode.REMOTE:
          return RemoteBranchlessBufferedRef
        case config.RoutingMode.DIRECT:
          return BranchedBufferedRef
        case _:
          raise NotImplementedError(f"Unsupported routing mode: {mode}.")

    x_bref = _get_buffered_ref(cfg.x.mode).input(
        spec=pipeline_block_specs.get_x_spec(),
        dtype_or_type=x_hbm_ref.dtype,
        buffer_count=cfg.plan.num_xo_buffers,
        grid_rank=3,
        cfg=cfg,
        metadata_ref=metadata_ref,
    )
    out_bref = _get_buffered_ref(cfg.out.mode).output(
        spec=pipeline_block_specs.get_out_spec(),
        dtype_or_type=out_hbm_ref.dtype,
        buffer_count=cfg.plan.num_xo_buffers,
        grid_rank=3,
        cfg=cfg,
        metadata_ref=metadata_ref,
    )
    if cfg.out.mode == config.RoutingMode.REMOTE:
      out_bref = dataclasses.replace(
          out_bref, recv_sem=pltpu.SemaphoreType.DMA(())
      )

    # TODO: Support use case when indices are not stream from HBM and
    # instead entirely stored in SMEM.
    gather_bref = scatter_bref = None
    if in_hbm_ref.gather_idx is not None:
      # At step i the inner_pipeline gathers tile i + (num_xo_buffers - 1).
      fetch_offset = cfg.plan.num_xo_buffers - 1
      gather_bref = IndexBufferedRef.input(
          spec=pipeline_block_specs.get_index_spec(),
          dtype_or_type=in_hbm_ref.gather_idx.dtype,
          # IndexBufferedRef alternates two buffers by n. This ensures first
          # tile of n + 1 never overwrites the last tile's n.
          buffer_count=cfg.plan.num_idx_buffers,
          cfg=cfg,
          metadata_ref=metadata_ref,
          fetch_offset=fetch_offset,
      )
    if in_hbm_ref.scatter_idx is not None:
      # At step i the inner_pipeline scatters tile i - 1.
      fetch_offset = -1
      scatter_bref = IndexBufferedRef.input(
          spec=pipeline_block_specs.get_index_spec(),
          dtype_or_type=in_hbm_ref.scatter_idx.dtype,
          buffer_count=cfg.plan.num_idx_buffers,
          cfg=cfg,
          metadata_ref=metadata_ref,
          fetch_offset=fetch_offset,
      )

    return cls(
        inputs=PipelineInputs(
            x=x_bref, w=w_bref, gather_idx=gather_bref, scatter_idx=scatter_bref
        ),
        out=out_bref,
    )

  @property
  def all_tree(
      self,
  ) -> tuple[PipelineInputs[RoutingBufferedRef], RoutingBufferedRef]:
    """The tuple of (inputs, out) BufferedRefs for pipeline allocation."""
    return (self.inputs, self.out)

  @property
  def in_specs(self) -> PipelineInputs[pl.BlockSpec]:
    """The BlockSpec PyTree corresponding to the pipelined inputs."""
    return jax.tree.map(
        lambda bref: bref.spec,
        self.inputs,
        is_leaf=lambda node: isinstance(node, pltpu.BufferedRefBase),
    )

  @property
  def out_spec(self) -> pl.BlockSpec:
    """The BlockSpec for the pipelined output."""
    return self.out.spec
