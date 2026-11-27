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

"""Input relayout helper from large to compact layout."""

import dataclasses
import functools
import math
from typing import Self

import jax
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.common import dataclass_util
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.common import out_format as out_format_lib
from tokamax._src.ops.experimental.routing_moe.common import quant_util
from tokamax._src.ops.experimental.routing_moe.common import spec as spec_lib


@dataclass_util.require_none_default
@dataclasses.dataclass(frozen=True, kw_only=True)
class InputPackPlan:
  """User-provided plan where None fields are determined by heuristic."""

  tile_rows: int | None = None
  num_buffers: int | None = None
  scoped_vmem_reserved_bytes: int | None = None
  # Heuristic bounds omitted once resolved.
  max_tile_rows: int | None = None
  max_scratch_vmem_ratio: float | None = None


@dataclass_util.resolves_from(
    InputPackPlan,
    dropped=("max_tile_rows", "max_scratch_vmem_ratio"),
)
@dataclasses.dataclass(frozen=True, kw_only=True)
class ResolvedInputPackPlan:
  """Plan with None fields from `InputPackPlan` resolved by heuristic."""

  tile_rows: int
  num_buffers: int
  scoped_vmem_reserved_bytes: int

  @staticmethod
  def select_tile_rows(
      batch_size: int,
      dim_size: int,
      vmem_chunks: int,
      in_dtype: jax.typing.DTypeLike,
      out_dtype: jax.typing.DTypeLike,
      tpu_info: pltpu.TpuInfo,
      requested_tile_rows: int | None,
      num_buffers: int,
      max_tile_rows: int,
      max_scratch_vmem_ratio: float,
  ) -> int:
    """Selects a sublane-aligned tile size within VMEM capacity."""
    x_dtype = jax.dtypes.canonicalize_dtype(in_dtype)
    out_dtype = jax.dtypes.canonicalize_dtype(out_dtype)
    # I think output dtype will always be the same or smaller than input dtype
    # but... just in case?
    sublane_size = max(
        tpu_info.get_sublane_tiling(x_dtype),
        tpu_info.get_sublane_tiling(out_dtype),
    )
    batch_sublanes = max(1, batch_size // sublane_size)
    if requested_tile_rows is None:
      # Cap at ceil(total_sublanes / 2) when possible so HBM DMAs overlap with
      # VPU relayout via double buffering without creating a tiny 3rd tile.
      total_sublanes = pl.cdiv(batch_size, sublane_size)
      max_tile_sublanes = max(
          1,
          min(max_tile_rows // sublane_size, pl.cdiv(total_sublanes, 2)),
      )
      # Prefer an exact divisor of `batch_sublanes` when it is at least half of
      # `max_tile_sublanes` (e.g. batch=768 -> tile=384) to avoid a partial last
      # tile; otherwise use `max_tile_sublanes` instead of degrading to
      # 1-sublane tiles on prime/irregular batches.
      gcd_sublanes = math.gcd(batch_sublanes, max_tile_sublanes)
      if batch_size % sublane_size == 0 and gcd_sublanes >= max(
          1, max_tile_sublanes // 2
      ):
        tile_sublanes = gcd_sublanes
      else:
        tile_sublanes = max_tile_sublanes
    else:
      if requested_tile_rows <= 0 or requested_tile_rows % sublane_size != 0:
        raise ValueError(
            f"tile_rows ({requested_tile_rows}) must be a positive multiple of"
            f" sublane_size ({sublane_size}) for {x_dtype} input and"
            f" {out_dtype} output."
        )
      tile_sublanes = min(batch_sublanes, requested_tile_rows // sublane_size)

    tile_rows = sublane_size * tile_sublanes

    # Each buffer slot holds one input tile `(tile_rows, dim_size)` and one
    # bank-conflict-padded output tile `(tile_rows, vmem_chunks, packing,
    # lanes)` in VMEM. When `dim_size` is large, the initial `tile_rows` may
    # exceed VMEM capacity; repeatedly halve the sublane count per tile
    # (preserving sublane alignment) until all scratch tensors fit within the
    # VMEM budget (`max_scratch_vmem_ratio` of capacity, leaving headroom for
    # compiler-managed VMEM allocations).
    out_packing = layout_util.get_packing(out_dtype)
    bytes_per_row = (
        dim_size * x_dtype.itemsize
        + vmem_chunks * out_packing * tpu_info.num_lanes * out_dtype.itemsize
    )

    max_scratch_bytes = int(
        tpu_info.vmem_capacity_bytes * max_scratch_vmem_ratio
    )

    def scratch_bytes(tile_rows: int) -> int:
      return num_buffers * tile_rows * bytes_per_row

    while (
        tile_rows > sublane_size
        and scratch_bytes(tile_rows) > max_scratch_bytes
    ):
      tile_rows = max(
          sublane_size, ((tile_rows // sublane_size) // 2) * sublane_size
      )
    return tile_rows

  @classmethod
  def create(
      cls,
      plan: InputPackPlan | None,
      *,
      batch_size: int,
      dim_size: int,
      vmem_chunks: int,
      in_dtype: jax.typing.DTypeLike,
      out_spec: spec_lib.ActivationSpec,
      tpu_info: pltpu.TpuInfo,
  ) -> Self:
    """Resolves unset plan fields into concrete execution parameters."""
    if plan is None:
      plan = InputPackPlan()

    out_dtype = out_spec.dtype
    max_num_buffers = 2 if plan.num_buffers is None else plan.num_buffers
    if max_num_buffers <= 0:
      raise ValueError(f"num_buffers ({max_num_buffers}) must be positive.")

    max_tile_rows = 512 if plan.max_tile_rows is None else plan.max_tile_rows
    if max_tile_rows <= 0:
      raise ValueError(f"max_tile_rows ({max_tile_rows}) must be positive.")

    scoped_vmem_reserved_bytes = plan.scoped_vmem_reserved_bytes
    if scoped_vmem_reserved_bytes is None:
      scoped_vmem_reserved_bytes = 64 * 1024
    if scoped_vmem_reserved_bytes < 0:
      raise ValueError(
          f"scoped_vmem_reserved_bytes ({scoped_vmem_reserved_bytes}) must be"
          " non-negative."
      )

    max_scratch_vmem_ratio = plan.max_scratch_vmem_ratio
    if max_scratch_vmem_ratio is None:
      max_scratch_vmem_ratio = 0.75

    if not 0 < max_scratch_vmem_ratio <= 1:
      raise ValueError(
          f"max_scratch_vmem_ratio ({max_scratch_vmem_ratio}) must be in"
          " (0, 1]."
      )

    tile_rows = ResolvedInputPackPlan.select_tile_rows(
        batch_size=batch_size,
        dim_size=dim_size,
        vmem_chunks=vmem_chunks,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        tpu_info=tpu_info,
        requested_tile_rows=plan.tile_rows,
        num_buffers=max_num_buffers,
        max_tile_rows=max_tile_rows,
        max_scratch_vmem_ratio=max_scratch_vmem_ratio,
    )
    num_tiles = pl.cdiv(batch_size, tile_rows)
    num_buffers = max(1, min(num_tiles, max_num_buffers))
    return cls(
        tile_rows=tile_rows,
        num_buffers=num_buffers,
        scoped_vmem_reserved_bytes=scoped_vmem_reserved_bytes,
    )


@dataclass_util.require_no_default
@dataclasses.dataclass(frozen=True, kw_only=True)
class _InputPackConfig:
  """Compile-time static configuration of the `input_pack` kernel."""

  batch_size: int
  dim_size: int
  in_dtype: jnp.dtype
  out_spec: spec_lib.ActivationSpec
  plan: ResolvedInputPackPlan
  tpu_info: pltpu.TpuInfo

  @classmethod
  def create(
      cls,
      *,
      x: jax.Array,
      out_format: out_format_lib.ActivationFormatLike | None,
      plan: InputPackPlan | None,
      tpu_info: pltpu.TpuInfo,
  ) -> Self:
    """Validates inputs, resolves `plan`, and builds the static config."""
    if x.ndim != 2:
      raise ValueError(
          f"Expected 2D input (batch_size, dim_size), got {x.shape}"
      )

    resolved_format = out_format_lib.ResolvedActivationFormat.create(
        out_format,
        default_dtype=x.dtype,
        default_quant=out_format_lib.ResolvedSubchannel(
            block_size=out_format_lib.tile_size
        ),
    )
    batch_size, dim_size = x.shape
    in_dtype, out_dtype = x.dtype, resolved_format.dtype
    quant_spec = out_format_lib.to_quant_spec(
        resolved_format.quant, tile_col_size=dim_size
    )

    match quant_spec:
      case None:
        if out_dtype != in_dtype:
          raise NotImplementedError(
              "input_pack does not support dtype conversion without"
              " quantization."
          )
      case spec_lib.SubchannelQuantSpec(block_size=block_size):
        if (
            in_dtype not in (jnp.bfloat16, jnp.float32)
            or out_dtype != jnp.float8_e4m3fn
        ):
          raise NotImplementedError(
              "input_pack only quantizes bfloat16 or float32 to float8_e4m3fn."
              f" Got {in_dtype} to {out_dtype}."
          )
        if block_size != dim_size:
          raise NotImplementedError(
              "input_pack only supports SubChannelQuant with"
              f" block_size=dim_size. Got {block_size=}"
          )
      case _:
        raise NotImplementedError(
            "input_pack only supports no output quantization or"
            f" SubChannelQuant. Got {quant_spec}."
        )

    packing = layout_util.get_packing(out_dtype)
    num_lanes = tpu_info.num_lanes
    if dim_size % (packing * num_lanes) != 0:
      raise ValueError(
          f"dim_size ({dim_size}) must be divisible by packing * num_lanes"
          f" ({packing * num_lanes}) for dtype {out_dtype}."
      )

    out_spec = spec_lib.ActivationSpec(
        row_size=batch_size,
        col_size=dim_size,
        dtype=out_dtype,
        quant=quant_spec,
    )

    vmem_chunks = layout_util.to_bank_conflict_aware_stride(
        out_spec.num_physical_chunks, num_steps=tpu_info.num_sublanes
    )
    resolved_plan = ResolvedInputPackPlan.create(
        plan,
        batch_size=batch_size,
        dim_size=dim_size,
        vmem_chunks=vmem_chunks,
        in_dtype=in_dtype,
        out_spec=out_spec,
        tpu_info=tpu_info,
    )
    return cls(
        batch_size=batch_size,
        dim_size=dim_size,
        in_dtype=in_dtype,
        out_spec=out_spec,
        plan=resolved_plan,
        tpu_info=tpu_info,
    )

  @property
  def num_lanes(self) -> int:
    return self.tpu_info.num_lanes

  @property
  def num_sublanes(self) -> int:
    return self.tpu_info.num_sublanes

  @property
  def packing(self) -> int:
    return layout_util.get_packing(self.out_spec.dtype)

  @property
  def num_logical_chunks(self) -> int:
    return self.out_spec.num_logical_chunks

  @property
  def num_physical_chunks(self) -> int:
    return self.out_spec.num_physical_chunks

  @property
  def vmem_chunks(self) -> int:
    # `store_large_to_compact` stores with stride `vmem_chunks * packing`.
    return layout_util.to_bank_conflict_aware_stride(
        self.num_physical_chunks, num_steps=self.num_sublanes
    )

  @property
  def tile_rows(self) -> int:
    return self.plan.tile_rows

  @property
  def num_buffers(self) -> int:
    return self.plan.num_buffers

  @property
  def num_tiles(self) -> int:
    return pl.cdiv(self.batch_size, self.tile_rows)

  @property
  def num_active_buffers(self) -> int:
    return min(self.num_buffers, self.num_tiles)

  @property
  def out_shape(self) -> tuple[int, ...]:
    return self.out_spec.physical_compact_shape

  @property
  def vmem_limit_bytes(self) -> int:
    return (
        self.tpu_info.vmem_capacity_bytes - self.plan.scoped_vmem_reserved_bytes
    )

  @property
  def kernel_name(self) -> str:
    name = f"input_pack_b{self.batch_size}_d{self.dim_size}_t{self.tile_rows}"
    if self.out_spec.quant is not None:
      name += f"_q{self.out_spec.dtype}"
    return name

  def scratch_types(self) -> dict[str, pl.MemoryRef]:
    """Scratch buffers of `_input_pack_kernel`, keyed by its parameter names."""
    return dict(
        in_scratch_ref=pltpu.VMEM(
            (self.num_buffers, self.tile_rows, self.dim_size), self.in_dtype
        ),
        out_scratch_ref=pltpu.VMEM(
            (
                self.num_buffers,
                self.tile_rows,
                self.vmem_chunks,
                self.packing,
                self.num_lanes,
            ),
            self.out_spec.dtype,
        ),
        send_sem_ref=pltpu.SemaphoreType.DMA((self.num_buffers,)),
        recv_sem_ref=pltpu.SemaphoreType.DMA((self.num_buffers,)),
    )


def _quantize_rows(
    x: jax.Array,  # [rows, dim_size]
    *,
    dtype: jnp.dtype,
    quant: spec_lib.SubchannelQuantSpec,
    num_lanes: int,
) -> tuple[jax.Array, jax.Array]:
  """Returns `x` quantized per row and its `f32[rows, num_lanes]` scales."""
  scale, inv_scale = quant_util.compute_scale(x, dtype, quant)
  # The scale chunk holds one f32 per lane, so the row scale is replicated.
  scale = jnp.broadcast_to(scale.astype(jnp.float32), (x.shape[0], num_lanes))
  return quant_util.quantize(x, inv_scale, dtype), scale


def _input_pack_kernel(
    in_hbm_ref: hijax.TransformedRef,
    out_hbm_ref: hijax.TransformedRef,
    *,
    in_scratch_ref: hijax.TransformedRef,
    out_scratch_ref: hijax.TransformedRef,
    send_sem_ref: hijax.TransformedRef,
    recv_sem_ref: hijax.TransformedRef,
    cfg: _InputPackConfig,
) -> None:
  """Double-buffered HBM-to-VMEM relayout from large to compact layout.

  When `batch_size` is not a multiple of `tile_rows`, the final tile reads the
  remaining rows rounded up to `num_sublanes` from the physically sublane-padded
  HBM input, relayouts the full VMEM tile, and writes only the remaining valid
  rows to the untiled output.

  Args:
    in_hbm_ref: 2D input HBM ref `(batch_size, dim_size)`.
    out_hbm_ref: 4D compact output HBM ref.
    in_scratch_ref: Multi-buffered VMEM input scratch ref.
    out_scratch_ref: Multi-buffered VMEM output scratch ref.
    send_sem_ref: DMA semaphores for output writes.
    recv_sem_ref: DMA semaphores for input reads.
    cfg: Static kernel configuration.
  """

  # -------- Define DMA helper functions --------

  def num_dma_rows(
      tile_idx: jax.Array | int, total_rows: int
  ) -> jax.Array | int:
    if total_rows % cfg.tile_rows == 0:
      return cfg.tile_rows
    return jnp.minimum(cfg.tile_rows, total_rows - tile_idx * cfg.tile_rows)

  def copy_in(tile_idx: jax.Array | int, slot: jax.Array | int):
    start_row = tile_idx * cfg.tile_rows
    num_rows = num_dma_rows(tile_idx, max(cfg.batch_size, cfg.tile_rows))
    num_rows = pl.cdiv(num_rows, cfg.num_sublanes) * cfg.num_sublanes
    return pltpu.make_async_copy(
        in_hbm_ref.at[pl.ds(start_row, num_rows)],
        in_scratch_ref.at[slot, pl.ds(0, num_rows)],
        recv_sem_ref.at[slot],
    )

  def copy_out(tile_idx: jax.Array | int, slot: jax.Array | int):
    start_row = tile_idx * cfg.tile_rows
    num_rows = num_dma_rows(tile_idx, cfg.batch_size)
    return pltpu.make_async_copy(
        out_scratch_ref.at[slot, pl.ds(0, num_rows), : cfg.num_physical_chunks],
        out_hbm_ref.at[pl.ds(start_row, num_rows)],
        send_sem_ref.at[slot],
    )

  # -------- Actual Kernel Part --------

  # Prologue.
  for slot in range(cfg.num_active_buffers):
    copy_in(slot, slot).start()

  # Body.
  @pl.loop(0, cfg.num_tiles)
  @jax.named_scope("step")
  def step(tile_idx: jax.Array) -> None:
    """Executes one double-buffered pipeline iteration for `tile_idx`."""
    slot = tile_idx % cfg.num_buffers

    # Step 1: Wait for previous output tile DMA on this buffer slot before
    # overwriting `out_scratch_ref.at[slot]`.
    @pl.when(tile_idx >= cfg.num_buffers)
    def _():
      copy_out(tile_idx - cfg.num_buffers, slot).wait()

    # Step 2: Wait for current input tile DMA and load into VREGs.
    copy_in(tile_idx, slot).wait()

    # Step 3: Fetch inputs from VMEM to VREG. Optionally quantize depending
    # on output requirement.
    val = in_scratch_ref.at[slot][...]
    if (quant := cfg.out_spec.quant) is not None:
      # Making linter happy despite config already having assertion.
      assert isinstance(quant, spec_lib.SubchannelQuantSpec)
      val, scale = _quantize_rows(
          val, dtype=cfg.out_spec.dtype, quant=quant, num_lanes=cfg.num_lanes
      )
      scale_u32 = pltpu.bitcast(scale, jnp.uint32)
      layout_util.store_compact_chunk(
          out_scratch_ref.at[slot], 0, cfg.num_logical_chunks, scale_u32
      )

    # Step 4: Store large VREG to compact VMEM using strided store.
    layout_util.store_large_to_compact(out_scratch_ref.at[slot], val)

    # Step 5: Write current compact tile to HBM.
    copy_out(tile_idx, slot).start(priority=1)

    # Step 6: Prefetch next input tile into this buffer slot now that `val`
    # has been loaded into VREGs.
    next_tile_idx = tile_idx + cfg.num_buffers

    @pl.when(next_tile_idx < cfg.num_tiles)
    def _():
      copy_in(next_tile_idx, slot).start()

  # Epilogue.
  for tile_idx in range(cfg.num_tiles - cfg.num_active_buffers, cfg.num_tiles):
    copy_out(tile_idx, tile_idx % cfg.num_buffers).wait()


@jax.jit(static_argnames=("out_format", "plan"))
def input_pack(
    x: jax.Array,
    out_format: out_format_lib.ActivationFormatLike | None = None,
    *,
    plan: InputPackPlan | None = None,
) -> arg.ActivationWrapper[jax.Array]:
  """Relayouts input `x` from large 2D layout to 4D compact layout."""
  cfg = _InputPackConfig.create(
      x=x,
      out_format=out_format,
      plan=plan,
      tpu_info=pltpu.get_tpu_info(),
  )
  # TODO: Remove this padding and use `pl.pad_ref` in
  # `_input_pack_kernel` once the minimum JAX version is >= 0.12.1.
  if x.shape[0] < cfg.tile_rows:
    x = jnp.pad(x, ((0, cfg.tile_rows - x.shape[0]), (0, 0)))
  # No layout constraint on `x`: the kernel is equally fast with any sublane
  # tiling, and leaving the choice to XLA avoids a relayout copy for inputs in
  # XLA's default layout and lets producers in the same jit match the kernel.
  out = pl.kernel(
      functools.partial(_input_pack_kernel, cfg=cfg),
      out_type=jax.ShapeDtypeStruct(cfg.out_shape, cfg.out_spec.dtype),
      mesh=pltpu.TensorCoreMesh(axis_name="core", num_cores=1),
      scratch_types=cfg.scratch_types(),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=cfg.vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      name=cfg.kernel_name,
  )(x)
  return arg.ActivationWrapper(data=out, scale=None, spec=cfg.out_spec)
