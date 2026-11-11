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
from collections.abc import Callable
import dataclasses
import functools
from typing import Any, Self, override

import jax
from jax import lax
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.common import quant_util
from tokamax._src.ops.experimental.routing_moe.common import spec as spec_lib
from tokamax._src.ops.experimental.routing_moe.gmm import config
from tokamax._src.ops.experimental.routing_moe.gmm import metadata

type GridIndices = tuple[jax.Array | int, jax.Array | int, jax.Array | int]


def map_memory_constraint[T](obj: T, memory_space: pltpu.MemorySpace) -> T:
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
  """Weight, optional scale, and optional bias references for one projection."""

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
  """Gate and up weight references for a fused gated activation."""

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
  def __post_init__(self) -> None:
    # pallas doesn't allow you to set buffer_count > 2 for output refs, so
    # we override to bypass this check.
    pass


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class RoutingBufferedRef(_BypassBufferedRef):
  """Shared interface for routed activation BufferedRef instances.

  Attributes:
    metadata_ref: SMEM metadata reference for routing and tile scheduling.
    idx_ref: SMEM row-index reference, or None when unindexed.
    cfg: Static configuration for the routing GMM kernel.
    act_cfg: Resolved spec and routing mode for this activation.
  """

  metadata_ref: metadata.AbstractMetadataRef = dataclasses.field(
      metadata=dict(static=True)
  )
  idx_ref: Any = dataclasses.field(default=None, metadata=dict(static=True))
  cfg: config.RoutingGmmConfig = dataclasses.field(metadata=dict(static=True))
  act_cfg: config.ActivationConfig = dataclasses.field(
      metadata=dict(static=True)
  )

  def bind_idx_ref(self, idx_bref: pltpu.BufferedRef) -> Self:
    return dataclasses.replace(self, idx_ref=idx_bref.window_ref)

  @classmethod
  @override
  def create(
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: pltpu.BufferType,
      buffer_count: int,
      grid_rank: int | None = None,
      *args: Any,
      metadata_ref: metadata.AbstractMetadataRef,
      cfg: config.RoutingGmmConfig,
      act_cfg: config.ActivationConfig,
      **kwargs: Any,
  ) -> Self:
    standard_ref = _BypassBufferedRef.create(
        spec, dtype_or_type, buffer_type, buffer_count, grid_rank, *args
    )
    return cls(
        metadata_ref=metadata_ref,
        cfg=cfg,
        act_cfg=act_cfg,
        **kwargs,
        **{
            f.name: getattr(standard_ref, f.name)
            for f in dataclasses.fields(pltpu.BufferedRef)
        },
    )

  def read_vmem(
      self,
      vmem_ref: hijax.TransformedRef,
      grid_indices: GridIndices,
  ) -> jax.Array:
    """Loads the input tile at `grid_indices` from VMEM into the large layout.

    Args:
      vmem_ref: VMEM slot reference for the tile's active rows.
      grid_indices: Grid coordinates `(n_id, gm_id, k_id)` of the tile.

    Returns:
      The loaded activation tile in `[rows, tile_k]` large layout.
    """
    del grid_indices
    assert self.is_input
    return layout_util.load_compact_to_large(vmem_ref, self.cfg.plan.tile_k)

  def write_vmem(
      self,
      vmem_ref: hijax.TransformedRef,
      value: jax.Array,
      grid_indices: GridIndices,
  ) -> None:
    """Stores the computed tile at `grid_indices` into VMEM for copy-out.

    The written tile is copied out to HBM at the next pipeline step.

    Args:
      vmem_ref: VMEM slot reference for the tile's `value.shape[0]` active rows.
      value: Computed tile result in `[rows, tile_n]` large layout.
      grid_indices: Grid coordinates `(n_id, gm_id, k_id)` of the tile.
    """
    del grid_indices
    assert self.is_output
    layout_util.store_large_to_compact(vmem_ref, value.astype(vmem_ref.dtype))

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

  @classmethod
  @override
  def output(
      cls, *args: Any, act_cfg: config.ActivationConfig, **kwargs: Any
  ) -> Self:
    assert act_cfg.spec.quant is None
    return super().output(*args, act_cfg=act_cfg, **kwargs)

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

  metadata_ref: metadata.AbstractMetadataRef = dataclasses.field(
      metadata=dict(static=True)
  )
  cfg: config.RoutingGmmConfig = dataclasses.field(metadata=dict(static=True))
  fetch_offset: int = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(
      cls,
      *args: Any,
      cfg: config.RoutingGmmConfig,
      metadata_ref: metadata.AbstractMetadataRef,
      fetch_offset: int,
      **kwargs: Any,
  ) -> Self:
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
  def copy_in(
      self, idx_hbm_ref: hijax.TransformedRef, grid_indices: GridIndices
  ) -> None:
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
  def wait_in(
      self, idx_hbm_ref: hijax.TransformedRef, grid_indices: GridIndices
  ) -> None:
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
class QuantStash:
  """VMEM staging buffers that quantize output tiles before copy-out DMA.

  `store` computes quantization scales as the result leaves the MXU and
  stashes the result (in `plan.stash_dtype`) and scales in VMEM. At the next
  step, `load_and_quantize_rows` loads and quantizes `num_sublanes` rows at a
  time into VREGs for the caller to pack into the output slot before DMA.

  Each output chunk is stashed as its own compact `plan.stash_dtype` tile, so
  loading `num_sublanes` rows of it and casting them packs them directly into
  `num_sublanes` compact output rows.

  Attributes:
    stash: VMEM buffer `[num_stash_buffers, num_chunks, tile_m, ratio,
      stash_packing, lanes]` of stashed results, one compact tile per chunk.
    scale: VMEM buffer `f32[num_stash_buffers, tile_m, lanes]` of per-lane
      quantization scales.
    inv_scale: VMEM buffer `u32[num_stash_buffers, ratio * tile_m, lanes]` of
      packed reciprocal scales.
    act_cfg: Resolved spec and routing mode for the quantized output.
    ratio: Number of u32 words of `stash_dtype` per u32 word of output dtype.
  """

  stash: hijax.TransformedRef
  scale: hijax.TransformedRef
  inv_scale: hijax.TransformedRef
  act_cfg: config.ActivationConfig = dataclasses.field(
      metadata=dict(static=True)
  )
  ratio: int = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(
      cls, cfg: config.RoutingGmmConfig, act_cfg: config.ActivationConfig
  ) -> Self:
    """Allocates VMEM staging buffers for quantized output tiles."""
    assert isinstance(act_cfg.spec.quant, spec_lib.LaneStridedQuantSpec)
    count = cfg.plan.num_stash_buffers
    stash_dtype = cfg.plan.stash_dtype
    assert count is not None and stash_dtype is not None
    lanes, tile_m = cfg.tpu_info.num_lanes, cfg.plan.tile_m
    out_packing = layout_util.get_packing(act_cfg.spec.dtype)
    ratio = out_packing // layout_util.get_packing(stash_dtype)
    chunk_shape = layout_util.to_compact_shape(
        (tile_m, out_packing * lanes), stash_dtype, lanes=lanes
    )
    num_chunks = act_cfg.spec.num_logical_chunks // cfg.num_n
    return cls(
        stash=pltpu.VMEM((count, num_chunks, *chunk_shape), stash_dtype),
        scale=pltpu.VMEM((count, tile_m, lanes), jnp.float32),
        inv_scale=pltpu.VMEM((count, ratio * tile_m, lanes), jnp.uint32),
        act_cfg=act_cfg,
        ratio=ratio,
    )

  def _stash_slot(self, gm_id: jax.Array | int) -> jax.Array:
    count = self.stash.shape[0]
    # Tiles cycle through the stash slots, and adding count keeps the dividend
    # non-negative when gm_id is -1.
    return lax.rem(gm_id + count, count)

  def store(self, value: jax.Array, gm_id: jax.Array | int) -> None:
    """Computes quantization scales and stashes `value` and scales in VMEM."""
    quant = self.act_cfg.spec.quant
    assert isinstance(quant, spec_lib.LaneStridedQuantSpec)
    value = value.astype(jnp.float32)
    scale, inv_scale = quant_util.compute_scale(
        value, self.act_cfg.spec.dtype, quant
    )
    slot = self._stash_slot(gm_id)
    rows = value.shape[0]
    stash_dtype = self.stash.dtype
    chunks = jnp.split(value.astype(stash_dtype), self.stash.shape[1], axis=-1)
    for chunk, chunk_value in enumerate(chunks):
      layout_util.store_large_to_compact(
          self.stash.at[slot, chunk, :rows], chunk_value
      )
    self.scale[slot, :rows] = scale
    packed_inv = pltpu.pack_elementwise(
        [inv_scale, inv_scale], packed_dtype=stash_dtype
    )
    for word in range(self.ratio):
      self.inv_scale[slot, pl.ds(word, rows, stride=self.ratio)] = packed_inv

  def load_and_quantize_rows(
      self, gm_id: jax.Array | int, first_row: int, num_rows: int
  ) -> tuple[list[jax.Array], jax.Array]:
    """Loads and quantizes `num_rows` stashed rows starting at `first_row`."""
    slot = self._stash_slot(gm_id)
    lanes = self.scale.shape[-1]
    stash_dtype = self.stash.dtype
    stash_rows = pl.ds(self.ratio * first_row, self.ratio * num_rows)

    inv_scale = pltpu.bitcast(self.inv_scale[slot, stash_rows], stash_dtype)
    q_chunks = []
    for chunk in range(self.stash.shape[1]):
      words = self.stash.at[slot, chunk].bitcast(jnp.uint32).reshape(-1, lanes)
      stashed = pltpu.bitcast(words[stash_rows], stash_dtype)
      q_chunks.append(
          quant_util.quantize(stashed, inv_scale, self.act_cfg.spec.dtype)
      )

    return q_chunks, self.scale[slot, pl.ds(first_row, num_rows)]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BranchlessBufferedRef(RoutingBufferedRef):
  """BufferedRef that performs branchless DMAs to allow matmul overlap.

  Attributes:
    quant_stash: VMEM stash that quantizes output tiles before copy-out DMAs, or
      None when unquantized or for inputs.
  """

  quant_stash: QuantStash | None

  @classmethod
  @override
  def input(
      cls,
      *args: Any,
      cfg: config.RoutingGmmConfig,
      act_cfg: config.ActivationConfig,
      **kwargs: Any,
  ) -> Self:
    assert act_cfg.spec.quant is None
    return super().input(
        *args, cfg=cfg, act_cfg=act_cfg, quant_stash=None, **kwargs
    )

  @classmethod
  @override
  def output(
      cls,
      *args: Any,
      cfg: config.RoutingGmmConfig,
      act_cfg: config.ActivationConfig,
      **kwargs: Any,
  ) -> Self:
    quant_stash = (
        QuantStash.create(cfg, act_cfg)
        if act_cfg.spec.quant is not None
        else None
    )
    return super().output(
        *args, cfg=cfg, act_cfg=act_cfg, quant_stash=quant_stash, **kwargs
    )

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
  def write_vmem(
      self,
      vmem_ref: hijax.TransformedRef,
      value: jax.Array,
      grid_indices: GridIndices,
  ) -> None:
    """Stashes `value` when `quant_stash` is set, or writes `vmem_ref`."""
    if self.quant_stash is None:
      super().write_vmem(vmem_ref, value, grid_indices)
    else:
      self.quant_stash.store(value, grid_indices[1])

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
    offset = self._idx_offset(grid_indices, rows_ds)

    vmem_src = self._window_ref_at(slot)
    if self.quant_stash is None:
      # Unquantized outputs were written in the previous step, whereas
      # quant_stash writes vmem_src right before the row DMAs below.
      vmem_src = pltpu.annotate(vmem_src, no_hazard_no_deps=True)

    sublanes = self.cfg.tpu_info.num_sublanes
    for r in range(self.cfg.plan.tile_m):
      if self.quant_stash is not None and r % sublanes == 0:
        # Quantizing sublanes rows at a time bounds the VREG live range and
        # lets row DMAs start before the full tile is quantized.
        q_chunks, scale = self.quant_stash.load_and_quantize_rows(
            grid_indices[1], r, sublanes
        )
        layout_util.store_compact_chunk(
            vmem_src, r, len(q_chunks), pltpu.bitcast(scale, jnp.uint32)
        )
        for chunk, q_chunk in enumerate(q_chunks):
          layout_util.store_compact_chunk(
              vmem_src, r, chunk, pltpu.bitcast(q_chunk, jnp.uint32)
          )
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
    """Starts the DMA of one VMEM row src to output row idx."""
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
    del src_ref
    assert self.sem_recvs is not None

    slot = self.metadata_ref.slot(*grid_indices, self.buffer_count)
    rows_ds, cols_ds, *_ = self.compute_index(*grid_indices)
    # Only the byte count matters to the wait.
    copied = self._window_ref_at(slot).at[
        pl.ds(0, rows_ds.size), pl.ds(0, cols_ds.size)
    ]
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

  def _no_op(
      self, ref: hijax.TransformedRef, grid_indices: GridIndices
  ) -> None:
    del ref, grid_indices

  copy_in = copy_out = wait_in = wait_out = _no_op


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class RemoteBranchlessBufferedRef(BranchlessBufferedRef):
  """Output ref that DMAs each row to the device that owns it.

  Attributes:
    recv_sem: DMA semaphore counting the rows this device receives from all
      devices.
  """

  recv_sem: hijax.TransformedRef

  @classmethod
  @override
  def input(cls, *args: Any, **kwargs: Any) -> Self:
    raise NotImplementedError("Remote gather is not supported.")

  @classmethod
  @override
  def output(cls, *args: Any, **kwargs: Any) -> Self:
    return super().output(*args, recv_sem=pltpu.SemaphoreType.DMA(()), **kwargs)

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
    assert self.cfg.spec.scatter is not None
    axis = self.cfg.spec.scatter.remote_axis
    assert axis is not None
    # Unsigned lax div/rem skip the sign fixups of int32 jnp.divmod.
    rows_per_dev = jnp.uint32(self.act_cfg.spec.row_size)
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
    assert self.cfg.spec.scatter is not None
    axis, ep = self.cfg.spec.scatter.remote_axis, self.cfg.spec.size_ep
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
    recv = dst_ref.at[pl.ds(0, self.act_cfg.spec.row_size)]
    pltpu.make_async_copy(recv, recv, self.recv_sem).wait()


def _get_buffered_ref(mode: config.RoutingMode) -> type[RoutingBufferedRef]:
  """Returns the RoutingBufferedRef subclass for `mode`."""
  match mode:
    case config.RoutingMode.LOCAL:
      return BranchlessBufferedRef
    case config.RoutingMode.REMOTE:
      return RemoteBranchlessBufferedRef
    case config.RoutingMode.DIRECT:
      return BranchedBufferedRef
    case _:
      raise NotImplementedError(f"Unsupported routing mode: {mode}.")


@dataclasses.dataclass(frozen=True)
class PipelineIndexMaps:
  """Provides grid-to-tile index mapping functions for pipeline BlockSpecs.

  Attributes:
    metadata_ref: SMEM metadata reference providing per-step group and row
      offsets.
    cfg: Static configuration for the routing GMM kernel.
  """

  metadata_ref: metadata.AbstractMetadataRef
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

  def bias_index(
      self, n_id: jax.Array | int, gm_id: jax.Array | int, k_id: jax.Array | int
  ) -> tuple[jax.Array, int, jax.Array | int]:
    """Returns the block index tuple for a bias tensor."""
    del k_id
    return (self.metadata_ref.gm_id_to_group_id[gm_id], 0, n_id)

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
      self,
      act_cfg: config.ActivationConfig,
      num_tiles: int,
      index_map: Callable[..., tuple[pl.Slice, pl.Slice, int, int]],
  ) -> pl.BlockSpec:
    """Builds the VMEM BlockSpec for a compact activation tensor."""
    hbm_shape = act_cfg.spec.physical_compact_shape
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
        self.cfg.x,
        self.cfg.num_k,
        self.index_map_provider.x_index,
    )

  def get_out_spec(self) -> pl.BlockSpec:
    """Returns the BlockSpec for the output activation tensor."""
    return self._act_spec(
        self.cfg.out,
        self.cfg.num_n,
        self.index_map_provider.out_index,
    )

  def get_weight_spec(self, tile_k: int, tile_n: int) -> pl.BlockSpec:
    """Returns the BlockSpec for an expert weight tensor."""
    block_shape = (None, tile_k, tile_n)
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

  def get_scale_spec(
      self, w_spec: spec_lib.WeightSpec, tile_k: int, tile_n: int
  ) -> pl.BlockSpec:
    """Returns the BlockSpec for a quantization scale tensor."""
    assert w_spec.quant is not None
    num_blocks = tile_k // w_spec.quant.block_size
    block_shape = (None, num_blocks, 1, tile_n)
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

  def get_bias_spec(
      self, w_spec: spec_lib.WeightSpec, _: int, tile_n: int
  ) -> pl.BlockSpec:
    """Returns the BlockSpec for a bias tensor."""
    assert w_spec.has_bias
    block_shape = (None, 1, tile_n)
    return pl.BlockSpec(
        block_shape=block_shape,
        index_map=self.index_map_provider.bias_index,
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
  remain in the main orchestrators.

  Attributes:
    x: Input activation reference, BlockSpec, or BufferedRef.
    w: Expert weight reference, BlockSpec tree, or BufferedRef tree.
    w2: Optional second expert weight reference, BlockSpec tree, or BufferedRef
      tree for fused gate/up and down execution.
    gather_idx: Row indices used by x. Set to None when DIRECT mode is used.
    scatter_idx: Row indices used by out. Set to None when DIRECT mode is used.
  """

  x: T
  w: AbstractWeightRef
  w2: AbstractWeightRef | None = None
  gather_idx: T | None = None
  scatter_idx: T | None = None

  def transform_for_pipeline(self, cfg: config.RoutingGmmConfig) -> Self:
    """Returns inputs with weights sliced for fused gate/up activation."""
    if not cfg.spec.is_fused_act:
      return self
    assert isinstance(self.w, SingleWeightRef)
    n = cfg.spec.w.out_size // 2
    w_fused = FusedWeightsRef(
        gate=self.w,
        up=jax.tree.map(lambda ref: ref.at[..., n:], self.w),
    )
    return dataclasses.replace(self, w=w_fused)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PipelineAllocs:
  """BufferedRef allocations for the routing GMM pipeline.

  Devoid of stateful data management so emit_pipeline and inner_pipeline can
  remain in the main orchestrators.

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
      metadata_ref: metadata.AbstractMetadataRef,
      cfg: config.RoutingGmmConfig,
  ) -> Self:
    """Creates BufferedRef allocations for all pipeline inputs and outputs."""
    pipeline_index_maps = PipelineIndexMaps(metadata_ref, cfg)
    pipeline_block_specs = PipelineBlockSpecs(cfg, pipeline_index_maps)

    def _make_weight_bref(
        w_hbm_ref: AbstractWeightRef,
        w_spec: spec_lib.WeightSpec,
        tile_k: int,
        tile_n: int,
        *,
        fuse_act: bool = False,
    ) -> AbstractWeightRef:
      assert isinstance(w_hbm_ref, SingleWeightRef)
      scale_bref = None
      if w_spec.has_scale:
        assert w_hbm_ref.scale is not None
        scale_bref = pltpu.BufferedRef.input(
            spec=pipeline_block_specs.get_scale_spec(w_spec, tile_k, tile_n),
            dtype_or_type=w_hbm_ref.scale.dtype,
            buffer_count=cfg.plan.num_w_buffers,
            use_lookahead=True,
            grid_rank=3,
        )
      weight_bref = pltpu.BufferedRef.input(
          spec=pipeline_block_specs.get_weight_spec(tile_k, tile_n),
          dtype_or_type=w_hbm_ref.weight.dtype,
          buffer_count=cfg.plan.num_w_buffers,
          use_lookahead=True,
          grid_rank=3,
      )
      bias_bref = None
      if w_spec.has_bias:
        assert w_hbm_ref.bias is not None
        bias_bref = pltpu.BufferedRef.input(
            spec=pipeline_block_specs.get_bias_spec(w_spec, tile_k, tile_n),
            dtype_or_type=w_hbm_ref.bias.dtype,
            buffer_count=cfg.plan.num_w_buffers,
            use_lookahead=True,
            grid_rank=3,
        )
      w_bref = SingleWeightRef(
          weight=weight_bref, scale=scale_bref, bias=bias_bref
      )
      if fuse_act:
        w_bref = FusedWeightsRef(gate=w_bref, up=w_bref)
      return w_bref

    w_tile_n = cfg.plan.tile_n if cfg.plan.tile_k2 is None else cfg.plan.tile_k2
    w_bref = _make_weight_bref(
        in_hbm_ref.w,
        cfg.spec.w,
        cfg.plan.tile_k,
        w_tile_n,
        fuse_act=cfg.spec.is_fused_act,
    )
    w2_bref = None
    if cfg.spec.w2 is not None:
      assert in_hbm_ref.w2 is not None and cfg.plan.tile_k2 is not None
      w2_bref = _make_weight_bref(
          in_hbm_ref.w2,
          cfg.spec.w2,
          cfg.plan.tile_k2,
          cfg.plan.tile_n,
      )

    x_bref = _get_buffered_ref(cfg.x.mode).input(
        spec=pipeline_block_specs.get_x_spec(),
        dtype_or_type=cfg.x.spec.dtype,
        buffer_count=cfg.plan.num_xo_buffers,
        grid_rank=3,
        cfg=cfg,
        act_cfg=cfg.x,
        metadata_ref=metadata_ref,
    )
    out_bref = _get_buffered_ref(cfg.out.mode).output(
        spec=pipeline_block_specs.get_out_spec(),
        dtype_or_type=cfg.out.spec.dtype,
        buffer_count=cfg.plan.num_xo_buffers,
        grid_rank=3,
        cfg=cfg,
        act_cfg=cfg.out,
        metadata_ref=metadata_ref,
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
            x=x_bref,
            w=w_bref,
            w2=w2_bref,
            gather_idx=gather_bref,
            scatter_idx=scatter_bref,
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
        is_leaf=lambda node: isinstance(node, pltpu.BufferedRef),
    )

  @property
  def out_spec(self) -> pl.BlockSpec:
    """The BlockSpec for the pipelined output."""
    return self.out.spec
