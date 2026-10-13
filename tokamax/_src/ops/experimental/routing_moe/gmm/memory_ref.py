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
class SingleWeightRef(AbstractWeightRef):
  """Dataclass for a single weights."""

  weight: Any
  scale: Any | None
  bias: Any | None

  def get_weight(self) -> jax.Array:
    return self.weight[...]

  def get_scale(self, replicate_size: int | None = None) -> jax.Array:
    assert self.scale is not None
    if replicate_size is not None:
      # Perform zero-stride load for efficient broadcasting across sublanes.
      return self.scale[:, pl.ds(0, replicate_size, 0), :]
    return self.scale[...]

  def get_bias(self) -> jax.Array:
    assert self.bias is not None
    return self.bias[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FusedWeightsRef(AbstractWeightRef):
  """Dataclass for gate and up weights used in fused activation."""

  gate: SingleWeightRef
  up: SingleWeightRef

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
class CustomBufferedRef(_BypassBufferedRef):
  """Shared interface for both fused and non-fused BufferedRef."""

  metadata_ref: metadata.MetadataRef = dataclasses.field(
      metadata=dict(static=True)
  )
  src_ref: Any = dataclasses.field(metadata=dict(static=True))
  dst_ref: Any = dataclasses.field(metadata=dict(static=True))

  cfg: config.RoutingGmmConfig = dataclasses.field(metadata=dict(static=True))

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
class BranchedBufferedRef(CustomBufferedRef):
  """BufferedRef managed by emit_pipeline. Gated by dynamic branching."""

  def _no_op(self, gm_id: jax.Array | int) -> None:
    del gm_id

  branchless_copy_in = branchless_wait_in = branchless_copy_out = (
      branchless_wait_out
  ) = _no_op


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BranchlessBufferedRef(CustomBufferedRef):
  """BufferedRef that performs branchless DMA to allow matmul overlap."""

  def _slot(self, gm_id: jax.Array | int) -> jax.Array:
    return jnp.mod(gm_id, self.cfg.tile.num_xo_buffers)

  @override
  def branchless_copy_in(self, gm_id: jax.Array | int) -> None:
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    assert self.src_ref is not None
    assert self.metadata_ref.gather_idx is not None

    slot = self._slot(gm_id)
    sem = self.sem_recvs.at[slot]
    vmem_dst = self.window_ref.at[slot]
    vmem_dst = pltpu.annotate(vmem_dst, no_hazard_no_deps=True)
    hbm_src = self.src_ref
    blocks = hbm_src.shape[1]
    row_idx = self.metadata_ref.gather_idx

    offset = self.metadata_ref.get_m_offset(gm_id)
    rows = self.metadata_ref.get_num_rows(gm_id)

    for r in range(self.cfg.tile.tile_m):
      size = jnp.where(r < rows, 1, 0)
      row = row_idx[offset + r]
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
    assert self.metadata_ref.scatter_idx is not None

    slot = self._slot(gm_id)
    sem = self.sem_sends.at[slot]
    vmem_src = self.window_ref.at[slot]
    vmem_src = pltpu.annotate(vmem_src, no_hazard_no_deps=True)
    hbm_dst = self.dst_ref
    blocks = hbm_dst.shape[1]
    row_idx = self.metadata_ref.scatter_idx

    offset = self.metadata_ref.get_m_offset(gm_id)
    rows = self.metadata_ref.get_num_rows(gm_id)

    for r in range(self.cfg.tile.tile_m):
      size = jnp.where(r < rows, 1, 0)
      row = row_idx[offset + r]
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
class IndexMapProvider:
  """Provides grid-to-tile index mapping functions for pipeline BlockSpecs.

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


@dataclasses.dataclass(frozen=True)
class BlockSpecFactory:
  """Factory for routing GMM pipeline BlockSpecs.

  Pairs static tile block shapes and buffering configurations with dynamic
  index maps from IndexMapProvider.

  Attributes:
    cfg: Static configuration for the routing GMM kernel.
    index_map_provider: Provider for grid-to-tile index mapping functions.
  """

  cfg: config.RoutingGmmConfig
  index_map_provider: IndexMapProvider

  def _act_block_shape(self, shape):
    num_chunks = layout_util.to_bank_conflict_aware_stride(shape[1])
    return (
        pl.BoundedSlice(self.cfg.tile.tile_m),
        pl.BoundedSlice(num_chunks),
        *shape[2:],
    )

  def _act_spec(self, hbm_shape):
    index_map = functools.partial(
        self.index_map_provider.act_index, hbm_shape=hbm_shape
    )
    return pl.BlockSpec(
        block_shape=self._act_block_shape(hbm_shape),
        index_map=index_map,
        pipeline_mode=pl.Buffered(
            buffer_count=self.cfg.tile.num_xo_buffers, use_lookahead=False
        ),
        memory_space=pltpu.VMEM,
    )

  def get_x_spec(self) -> pl.BlockSpec:
    """Returns the BlockSpec for the input activation tensor."""
    return self._act_spec(self.cfg.x.compact_shape)

  def get_out_spec(self) -> pl.BlockSpec:
    """Returns the BlockSpec for the output activation tensor."""
    return self._act_spec(self.cfg.out.compact_shape)

  def get_weight_spec(self) -> pl.BlockSpec:
    """Returns the BlockSpec for the expert weight tensor."""
    block_shape = (None, self.cfg.tile.tile_k, self.cfg.tile.tile_n)
    return pl.BlockSpec(
        block_shape=block_shape,
        index_map=self.index_map_provider.weight_index,
        pipeline_mode=pl.Buffered(
            buffer_count=self.cfg.tile.num_w_buffers, use_lookahead=True
        ),
        memory_space=pltpu.VMEM,
    )

  def get_scale_spec(self) -> pl.BlockSpec:
    """Returns the BlockSpec for the quantization scale tensor."""
    block_shape = (None, 1, 1, self.cfg.tile.tile_n)
    return pl.BlockSpec(
        block_shape=block_shape,
        index_map=self.index_map_provider.scale_index,
        pipeline_mode=pl.Buffered(
            buffer_count=self.cfg.tile.num_w_buffers, use_lookahead=True
        ),
        memory_space=pltpu.VMEM,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PipelineInputs:
  """Pipelined input parameter bundle for the routing GMM kernel.

  Devoid of stateful data movement so core orchestration remains in the main
  pipeline functions.

  Attributes:
    x: Input activation reference, BlockSpec, or BufferedRef.
    w: Expert weight reference, BlockSpec tree, or BufferedRef tree.
  """

  x: Any
  w: AbstractWeightRef

  def transform_for_pipeline(self, cfg: config.RoutingGmmConfig):
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

  Devoid of stateful data movement so core orchestration remains in the main
  pipeline functions.

  Attributes:
    inputs: Pipelined input BufferedRef bundle.
    out: Output activation BufferedRef.
  """

  inputs: PipelineInputs
  out: CustomBufferedRef

  @classmethod
  def create(
      cls,
      in_hbm_ref: PipelineInputs,
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
    index_map_provider = IndexMapProvider(metadata_ref, cfg)
    spec_factory = BlockSpecFactory(cfg, index_map_provider)
    x_hbm_ref = in_hbm_ref.x
    w_hbm_ref = in_hbm_ref.w
    assert isinstance(w_hbm_ref, SingleWeightRef)

    scale_bref = None
    if cfg.w.has_scale:
      assert w_hbm_ref.scale is not None
      scale_bref = pltpu.BufferedRef.input(
          spec=spec_factory.get_scale_spec(),
          dtype_or_type=w_hbm_ref.scale.dtype,
          buffer_count=cfg.tile.num_w_buffers,
          use_lookahead=True,
          grid_rank=3,
      )

    weight_bref = pltpu.BufferedRef.input(
        spec=spec_factory.get_weight_spec(),
        dtype_or_type=w_hbm_ref.weight.dtype,
        buffer_count=cfg.tile.num_w_buffers,
        use_lookahead=True,
        grid_rank=3,
    )
    w_bref = SingleWeightRef(weight=weight_bref, scale=scale_bref, bias=None)
    if cfg.fuse_act is not None:
      w_bref = FusedWeightsRef(gate=w_bref, up=w_bref)

    def _get_buffered_ref(mode: config.RoutingMode) -> type[CustomBufferedRef]:
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
        spec=spec_factory.get_x_spec(),
        dtype_or_type=x_hbm_ref.dtype,
        buffer_count=cfg.tile.num_xo_buffers,
        use_lookahead=False,
        grid_rank=3,
        metadata_ref=metadata_ref,
        src_ref=x_hbm_ref,
        cfg=cfg,
    )
    out_bref = _get_buffered_ref(cfg.out.mode).output(
        spec=spec_factory.get_out_spec(),
        dtype_or_type=out_hbm_ref.dtype,
        buffer_count=cfg.tile.num_xo_buffers,
        use_lookahead=False,
        grid_rank=3,
        cfg=cfg,
        metadata_ref=metadata_ref,
        dst_ref=out_hbm_ref,
    )

    return cls(
        inputs=PipelineInputs(x=x_bref, w=w_bref),
        out=out_bref,
    )

  @property
  def all_tree(self) -> tuple[PipelineInputs, CustomBufferedRef]:
    """The tuple of (inputs, out) BufferedRefs for pipeline allocation."""
    return (self.inputs, self.out)

  @property
  def in_specs(self) -> PipelineInputs:
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
