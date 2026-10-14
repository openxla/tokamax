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
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.gmm import config
from tokamax._src.ops.experimental.routing_moe.gmm import metadata


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

  # TODO: Most of the objects are not used by BranchedBufferedRef.
  # Separate them out to ensure interface remains clean.
  metadata_ref: metadata.MetadataRef = dataclasses.field(
      metadata=dict(static=True)
  )
  src_ref: Any = dataclasses.field(metadata=dict(static=True))
  dst_ref: Any = dataclasses.field(metadata=dict(static=True))
  # SMEM ref used for indexing src or dst ref.
  idx_ref: Any = dataclasses.field(default=None, metadata=dict(static=True))

  cfg: config.RoutingGmmConfig = dataclasses.field(metadata=dict(static=True))

  def bind_idx_ref(self, idx_bref):
    return dataclasses.replace(self, idx_ref=idx_bref.window_ref)

  @classmethod
  def create(
      cls,
      *args: Any,
      cfg: config.RoutingGmmConfig,
      metadata_ref: metadata.MetadataRef,
      src_ref=None,
      dst_ref=None,
      **kwargs,
  ):
    standard_ref = _BypassBufferedRef.create(*args, **kwargs)
    return cls(
        metadata_ref=metadata_ref,
        src_ref=src_ref,
        dst_ref=dst_ref,
        cfg=cfg,
        **{
            f.name: getattr(standard_ref, f.name)
            for f in dataclasses.fields(pltpu.BufferedRef)
        },
    )

  @abc.abstractmethod
  def branchless_copy_in(self, gm_id: jax.Array | int) -> None:
    ...

  @abc.abstractmethod
  def branchless_wait_in(self, gm_id: jax.Array | int) -> None:
    ...

  @abc.abstractmethod
  def branchless_copy_out(self, gm_id: jax.Array | int) -> None:
    ...

  @abc.abstractmethod
  def branchless_wait_out(self, gm_id: jax.Array | int) -> None:
    ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BranchedBufferedRef(RoutingBufferedRef):
  """BufferedRef managed by emit_pipeline. Not overlaped with compute."""

  def _no_op(self, gm_id: jax.Array | int) -> None:
    del gm_id

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

  def _aligned_range(self, tile_id) -> tuple[jax.Array, jax.Array]:
    """Returns an tile range aligned to block size tile_m_idx."""
    block_size = self.cfg.plan.tile_m_idx
    m_start = self.metadata_ref.get_m_offset(tile_id)
    m_end = m_start + self.metadata_ref.get_num_rows(tile_id)
    aligned_start = (m_start // block_size) * block_size
    aligned_end = pl.cdiv(m_end, block_size) * block_size
    return aligned_start, aligned_end - aligned_start

  @override
  def copy_in(self, idx_hbm_ref, grid_indices):
    assert self.is_input
    assert len(grid_indices) == 3
    sem_recvs = self.sem_recvs
    window_ref = self.window_ref
    assert sem_recvs is not None
    assert window_ref is not None
    gm_id = grid_indices[1]

    aligned_start, aligned_size = self._aligned_range(gm_id + self.fetch_offset)
    local_start = aligned_start % self.cfg.plan.idx_buffer_size
    pltpu.make_async_copy(
        src_ref=idx_hbm_ref.at[pl.ds(aligned_start, aligned_size)],
        dst_ref=window_ref.at[pl.ds(local_start, aligned_size)],
        sem=sem_recvs.at[gm_id % self.buffer_count],
    ).start()

  @override
  def wait_in(self, idx_hbm_ref, grid_indices):
    assert self.is_input
    assert len(grid_indices) == 3
    sem_recvs = self.sem_recvs
    window_ref = self.window_ref
    assert sem_recvs is not None
    assert window_ref is not None
    gm_id = grid_indices[1]
    sem = sem_recvs.at[gm_id % self.buffer_count]

    aligned_size = self._aligned_range(gm_id + self.fetch_offset)[-1]
    copied = window_ref.at[pl.ds(0, aligned_size)]
    pltpu.make_async_copy(copied, copied, sem).wait()

    if self.fetch_offset < 0:
      # One of the unfortunate side-effect of piggy-backing on emit_pipeline
      # is that it fails to fetch a data for gm_id step because the last fetch
      # ends at (num_gm - 1) - fetch_offset. Therefore, we have to manually
      # perform fetch for the last gm_id.
      assert self.fetch_offset == -1
      num_gm = self.metadata_ref.num_gm[0]
      is_last_gm = (num_gm - 1) == gm_id

      # This can probably be done with branchless DMA but unlikely to be worth
      # it since this entire wait_in is already executed under emit_pipeline's
      # dynamic branch anyway.
      @pl.when(is_last_gm)
      def _():
        aligned_start, aligned_size = self._aligned_range(gm_id)
        local_start = aligned_start % self.cfg.plan.idx_buffer_size
        # NOTE: Because this copy_in is manually triggered at the last step,
        # the caller MUST trigger corresponding wait_in at the epilogue.
        pltpu.make_async_copy(
            src_ref=idx_hbm_ref.at[pl.ds(aligned_start, aligned_size)],
            dst_ref=window_ref.at[pl.ds(local_start, aligned_size)],
            sem=sem_recvs.at[(gm_id + 1) % self.buffer_count],
        ).start()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BranchlessBufferedRef(RoutingBufferedRef):
  """BufferedRef that performs branchless DMA to allow matmul overlap."""

  def _slot(self, gm_id: jax.Array | int) -> jax.Array:
    return jnp.mod(gm_id, self.cfg.plan.num_xo_buffers)

  @override
  def branchless_copy_in(self, gm_id: jax.Array | int) -> None:
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    assert self.src_ref is not None
    assert self.idx_ref is not None

    slot = self._slot(gm_id)
    sem = self.sem_recvs.at[slot]
    vmem_dst = self.window_ref.at[slot]
    vmem_dst = pltpu.annotate(vmem_dst, no_hazard_no_deps=True)
    hbm_src = self.src_ref
    blocks = hbm_src.shape[1]

    idx_buffer_size = self.cfg.plan.idx_buffer_size
    offset = self.metadata_ref.get_m_offset(gm_id) % idx_buffer_size
    rows = self.metadata_ref.get_num_rows(gm_id)

    for r in range(self.cfg.plan.tile_m):
      size = jnp.where(r < rows, 1, 0)
      row = self.idx_ref[offset + r]
      pltpu.make_async_copy(
          src_ref=hbm_src.at[pl.ds(row, size)],
          dst_ref=vmem_dst.at[pl.ds(r, size), :blocks],
          sem=sem,
      ).start()

  @override
  def branchless_copy_out(self, gm_id: jax.Array | int) -> None:
    assert self.sem_sends is not None
    assert self.window_ref is not None
    assert self.dst_ref is not None
    assert self.idx_ref is not None

    slot = self._slot(gm_id)
    sem = self.sem_sends.at[slot]
    vmem_src = self.window_ref.at[slot]
    vmem_src = pltpu.annotate(vmem_src, no_hazard_no_deps=True)
    hbm_dst = self.dst_ref
    blocks = hbm_dst.shape[1]

    idx_buffer_size = self.cfg.plan.idx_buffer_size
    offset = self.metadata_ref.get_m_offset(gm_id) % idx_buffer_size
    rows = self.metadata_ref.get_num_rows(gm_id)

    for r in range(self.cfg.plan.tile_m):
      size = jnp.where(r < rows, 1, 0)
      row = self.idx_ref[offset + r]
      pltpu.make_async_copy(
          src_ref=vmem_src.at[pl.ds(r, size), :blocks],
          dst_ref=hbm_dst.at[pl.ds(row, size)],
          sem=sem,
      ).start(priority=1)

  @override
  def branchless_wait_in(self, gm_id: jax.Array | int) -> None:
    assert self.sem_recvs is not None
    assert self.window_ref is not None

    slot = self._slot(gm_id)
    sem = self.sem_recvs.at[slot]
    hbm_src = self.src_ref

    rows = self.metadata_ref.get_num_rows(gm_id)
    copied = hbm_src.at[pl.ds(0, rows)]
    pltpu.make_async_copy(copied, copied, sem).wait()

  @override
  def branchless_wait_out(self, gm_id: jax.Array | int) -> None:
    assert self.sem_sends is not None
    assert self.window_ref is not None

    slot = self._slot(gm_id)
    sem = self.sem_sends.at[slot]
    hbm_dst = self.dst_ref
    rows = self.metadata_ref.get_num_rows(gm_id)
    copied = hbm_dst.at[pl.ds(0, rows)]
    pltpu.make_async_copy(copied, copied, sem).wait()

  def _no_op(self, ref, grid_indices):
    del ref, grid_indices

  copy_in = copy_out = wait_in = wait_out = _no_op


@dataclasses.dataclass(frozen=True)
class PipelineIndexMaps:
  """Provides grid-to-tile index mapping functions for pipeline BlockSpecs.

  Note that this is only for emit_pipeline managed indexing. Custom BufferedRef
  handles its own grid-to-tile indexing.

  Attributes:
    metadata_ref: SMEM metadata reference providing per-step group and row
      offsets.
    cfg: Static configuration for the routing GMM kernel.
  """

  metadata_ref: metadata.MetadataRef
  cfg: config.RoutingGmmConfig

  def act_index(self, n_id, gm_id, k_id, *, hbm_shape):
    """Returns the block index tuple for an activation tensor."""
    del n_id, k_id

    num_chunks = hbm_shape[1]
    m_offset = self.metadata_ref.get_m_offset(gm_id)
    m_size = self.metadata_ref.get_num_rows(gm_id)
    return (pl.ds(m_offset, m_size), pl.ds(0, num_chunks), 0, 0)

  def weight_index(self, n_id, gm_id, k_id):
    """Returns the block index tuple for an expert weight tensor."""
    del n_id, k_id
    return (self.metadata_ref.get_group_id(gm_id), 0, 0)

  def scale_index(self, n_id, gm_id, k_id):
    """Returns the block index tuple for a quantization scale tensor."""
    del n_id, k_id
    return (self.metadata_ref.get_group_id(gm_id), 0, 0, 0)

  def get_dummy_index(self, rank_size):
    """Dummy index that ensures custom copy_* and wait_* is always triggered."""

    def _dummy_index(n_id, gm_id, k_id):
      # TODO: Fix this logic when num_k > 1.
      return (n_id + gm_id + k_id,) * rank_size

    return _dummy_index


@dataclasses.dataclass(frozen=True)
class PipelineBlockSpecs:
  """Factory for routing GMM pipeline BlockSpecs.

  While index map might not be respected for custom BufferedRefs, when it comes
  to buffer memory allocation, everything is coordinated by this factory method.

  Attributes:
    cfg: Static configuration for the routing GMM kernel.
    index_map_provider: Provider for grid-to-tile index mapping functions.
  """

  cfg: config.RoutingGmmConfig
  index_map_provider: PipelineIndexMaps

  def _act_spec(self, hbm_shape: tuple[int, ...]) -> pl.BlockSpec:
    num_chunks = layout_util.to_bank_conflict_aware_stride(hbm_shape[1])
    block_shape = (
        pl.BoundedSlice(self.cfg.plan.tile_m),
        pl.BoundedSlice(num_chunks),
        *hbm_shape[2:],
    )
    index_map = functools.partial(
        self.index_map_provider.act_index, hbm_shape=hbm_shape
    )
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
    return self._act_spec(self.cfg.x.spec.compact_shape)

  def get_out_spec(self) -> pl.BlockSpec:
    """Returns the BlockSpec for the output activation tensor."""
    return self._act_spec(self.cfg.out.spec.compact_shape)

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
    # NOTE: Technically we can tighten this to use much less SMEM, but when
    # considering this approach does not take up too much SMEM, it is better
    # to just keep things simple.
    block_shape = (self.cfg.plan.idx_buffer_size,)

    return pl.BlockSpec(
        block_shape=block_shape,
        index_map=self.index_map_provider.get_dummy_index(1),
        memory_space=pltpu.SMEM,
        # TODO: This is redundant in later JAX versions, remove this
        # after JAX version update.
        pipeline_mode=pl.Buffered(buffer_count=2),
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
        case config.RoutingMode.DIRECT:
          return BranchedBufferedRef
        case _:
          raise NotImplementedError(
              "Only direct or local routing is supported."
          )

    x_bref = _get_buffered_ref(cfg.x.mode).input(
        spec=pipeline_block_specs.get_x_spec(),
        dtype_or_type=x_hbm_ref.dtype,
        buffer_count=cfg.plan.num_xo_buffers,
        grid_rank=3,
        metadata_ref=metadata_ref,
        src_ref=x_hbm_ref,
        cfg=cfg,
    )
    out_bref = _get_buffered_ref(cfg.out.mode).output(
        spec=pipeline_block_specs.get_out_spec(),
        dtype_or_type=out_hbm_ref.dtype,
        buffer_count=cfg.plan.num_xo_buffers,
        grid_rank=3,
        cfg=cfg,
        metadata_ref=metadata_ref,
        dst_ref=out_hbm_ref,
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
          # Use 2 buffers because index size is small enough that it doesn't
          # benefit much from larger buffers. Additionally, with small number
          # of buffers, less chance of worrying about overwrite issue.
          buffer_count=2,
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
          buffer_count=2,
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
