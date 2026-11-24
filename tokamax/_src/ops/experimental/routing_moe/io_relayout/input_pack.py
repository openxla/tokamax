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
      dtype: jax.typing.DTypeLike,
      tpu_info: pltpu.TpuInfo,
      requested_tile_rows: int | None,
      num_buffers: int,
      max_tile_rows: int,
      max_scratch_vmem_ratio: float,
  ) -> int:
    """Selects a sublane-aligned tile size within VMEM capacity."""
    x_dtype = jax.dtypes.canonicalize_dtype(dtype)
    sublane_size = tpu_info.get_sublane_tiling(x_dtype)
    # Floor so `tile_rows <= batch_size` whenever `batch_size >= sublane_size`;
    # the remaining `batch_size % tile_rows` rows are handled by the tail.
    batch_sublanes = max(1, batch_size // sublane_size)
    if requested_tile_rows is None:
      # Cap at ceil(batch_sublanes / 2) when possible so num_tiles >= 2 and HBM
      # DMAs overlap with VPU relayout via double buffering.
      max_tile_sublanes = max(
          1,
          min(max_tile_rows // sublane_size, (batch_sublanes + 1) // 2),
      )
      # Prefer an exact divisor of `batch_sublanes` when it is at least half of
      # `max_tile_sublanes` (e.g. batch=768 -> tile=384) to avoid a tail;
      # otherwise use `max_tile_sublanes` and let the tail handle the remaining
      # rows instead of degrading to 1-sublane tiles on prime/irregular batches.
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
            f" sublane_size ({sublane_size}) for dtype {x_dtype}."
        )
      tile_sublanes = min(batch_sublanes, requested_tile_rows // sublane_size)

    tile_rows = sublane_size * tile_sublanes

    # Each buffer slot holds one input tile `(tile_rows, dim_size)` and one
    # bank-conflict-padded output tile `(tile_rows, vmem_chunks, packing,
    # lanes)` in VMEM, and a ragged tail needs at most one more slot. When
    # `dim_size` is large, the initial `tile_rows` may exceed VMEM capacity;
    # repeatedly halve the sublane count per tile (preserving sublane
    # alignment) until all scratch tensors fit within the VMEM budget
    # (`max_scratch_vmem_ratio` of capacity, leaving headroom for
    # compiler-managed VMEM allocations).
    packing = layout_util.get_packing(x_dtype)
    bytes_per_row = (
        dim_size + vmem_chunks * packing * tpu_info.num_lanes
    ) * x_dtype.itemsize

    max_scratch_bytes = int(
        tpu_info.vmem_capacity_bytes * max_scratch_vmem_ratio
    )

    def scratch_bytes(tile_rows: int) -> int:
      num_slots = num_buffers + (batch_size % tile_rows != 0)
      return num_slots * tile_rows * bytes_per_row

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
      dtype: jax.typing.DTypeLike,
      tpu_info: pltpu.TpuInfo,
  ) -> Self:
    """Resolves unset plan fields into concrete execution parameters."""
    if plan is None:
      plan = InputPackPlan()

    max_num_buffers = 2 if plan.num_buffers is None else plan.num_buffers
    if max_num_buffers <= 0:
      raise ValueError(f"num_buffers ({max_num_buffers}) must be positive.")

    max_tile_rows = 512 if plan.max_tile_rows is None else plan.max_tile_rows
    if max_tile_rows <= 0:
      raise ValueError(f"max_tile_rows ({max_tile_rows}) must be positive.")

    scoped_vmem_reserved_bytes = (
        64 * 1024
        if plan.scoped_vmem_reserved_bytes is None
        else plan.scoped_vmem_reserved_bytes
    )
    if scoped_vmem_reserved_bytes < 0:
      raise ValueError(
          f"scoped_vmem_reserved_bytes ({scoped_vmem_reserved_bytes}) must be"
          " non-negative."
      )

    max_scratch_vmem_ratio = (
        0.75
        if plan.max_scratch_vmem_ratio is None
        else plan.max_scratch_vmem_ratio
    )
    if not 0 < max_scratch_vmem_ratio <= 1:
      raise ValueError(
          f"max_scratch_vmem_ratio ({max_scratch_vmem_ratio}) must be in"
          " (0, 1]."
      )

    tile_rows = ResolvedInputPackPlan.select_tile_rows(
        batch_size=batch_size,
        dim_size=dim_size,
        vmem_chunks=vmem_chunks,
        dtype=dtype,
        tpu_info=tpu_info,
        requested_tile_rows=plan.tile_rows,
        num_buffers=max_num_buffers,
        max_tile_rows=max_tile_rows,
        max_scratch_vmem_ratio=max_scratch_vmem_ratio,
    )
    num_tiles = batch_size // tile_rows
    # Allocate at least one slot so scratch shapes stay non-empty when
    # `batch_size < tile_rows` (i.e. `batch_size < sublane_size`) and every row
    # goes through the tail.
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
  dtype: jnp.dtype
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
    )
    if resolved_format.dtype != x.dtype:
      raise NotImplementedError(
          "input_pack does not support dtype conversion yet."
      )
    if resolved_format.quant is not None:
      raise NotImplementedError("input_pack does not support quantization yet.")

    batch_size, dim_size = x.shape
    dtype = x.dtype
    packing = layout_util.get_packing(dtype)
    num_lanes = tpu_info.num_lanes
    if dim_size % (packing * num_lanes) != 0:
      raise ValueError(
          f"dim_size ({dim_size}) must be divisible by packing * num_lanes"
          f" ({packing * num_lanes}) for dtype {dtype}."
      )

    num_packed_chunks = dim_size // (packing * num_lanes)
    vmem_chunks = layout_util.to_bank_conflict_aware_stride(
        num_packed_chunks, num_steps=tpu_info.num_sublanes
    )
    resolved_plan = ResolvedInputPackPlan.create(
        plan,
        batch_size=batch_size,
        dim_size=dim_size,
        vmem_chunks=vmem_chunks,
        dtype=dtype,
        tpu_info=tpu_info,
    )
    return cls(
        batch_size=batch_size,
        dim_size=dim_size,
        dtype=dtype,
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
    return layout_util.get_packing(self.dtype)

  @property
  def num_packed_chunks(self) -> int:
    return self.dim_size // (self.packing * self.num_lanes)

  @property
  def vmem_chunks(self) -> int:
    # `store_large_to_compact` stores with stride `vmem_chunks * packing`.
    return layout_util.to_bank_conflict_aware_stride(
        self.num_packed_chunks, num_steps=self.num_sublanes
    )

  @property
  def tile_rows(self) -> int:
    return self.plan.tile_rows

  @property
  def num_buffers(self) -> int:
    return self.plan.num_buffers

  @property
  def num_tiles(self) -> int:
    return self.batch_size // self.tile_rows

  @property
  def num_active_buffers(self) -> int:
    return min(self.num_buffers, self.num_tiles)

  @property
  def tail_row_start(self) -> int:
    return self.num_tiles * self.tile_rows

  @property
  def tail_rows(self) -> int:
    return self.batch_size - self.tail_row_start

  @property
  def padded_tail_rows(self) -> int:
    """Rounds `tail_rows` up to the input's HBM sublane tiling.

    The tail DMA must be aligned to that tiling. XLA tiles a `(batch_size, D)`
    kernel input by `num_sublanes` rows, or by a larger tile that already
    divides the tail, when `batch_size >= num_sublanes`; otherwise by the
    smallest power of two that covers `batch_size`, but at least `packing` rows.
    """
    if self.batch_size >= self.num_sublanes:
      return pl.cdiv(self.tail_rows, self.num_sublanes) * self.num_sublanes
    return max(self.packing, pl.next_power_of_2(self.batch_size))

  @property
  def num_sems(self) -> int:
    # One extra DMA semaphore per direction for the tail.
    return self.num_buffers + (self.tail_rows != 0)

  @property
  def out_shape(self) -> tuple[int, ...]:
    return layout_util.to_compact_shape(
        (self.batch_size, self.dim_size), self.dtype
    )

  @property
  def out_spec(self) -> spec_lib.ActivationSpec:
    return spec_lib.ActivationSpec(
        row_size=self.batch_size,
        col_size=self.dim_size,
        dtype=self.dtype,
        quant=None,
    )

  @property
  def vmem_limit_bytes(self) -> int:
    return (
        self.tpu_info.vmem_capacity_bytes - self.plan.scoped_vmem_reserved_bytes
    )

  @property
  def kernel_name(self) -> str:
    return f"input_pack_b{self.batch_size}_d{self.dim_size}_t{self.tile_rows}"

  def scratch_types(self) -> dict[str, pl.MemoryRef]:
    """Scratch buffers of `_input_pack_kernel`, keyed by its parameter names."""
    scratch: dict[str, pl.MemoryRef] = dict(
        in_scratch_ref=pltpu.VMEM(
            (self.num_buffers, self.tile_rows, self.dim_size), self.dtype
        ),
        out_scratch_ref=pltpu.VMEM(
            (
                self.num_buffers,
                self.tile_rows,
                self.vmem_chunks,
                self.packing,
                self.num_lanes,
            ),
            self.dtype,
        ),
        send_sem_ref=pltpu.SemaphoreType.DMA((self.num_sems,)),
        recv_sem_ref=pltpu.SemaphoreType.DMA((self.num_sems,)),
    )
    if self.tail_rows:
      scratch["tail_in_scratch_ref"] = pltpu.VMEM(
          (self.padded_tail_rows, self.dim_size), self.dtype
      )
      scratch["tail_out_scratch_ref"] = pltpu.VMEM(
          (
              self.padded_tail_rows,
              self.vmem_chunks,
              self.packing,
              self.num_lanes,
          ),
          self.dtype,
      )
    return scratch


def _input_pack_kernel(
    in_hbm_ref: hijax.TransformedRef,
    out_hbm_ref: hijax.TransformedRef,
    *,
    in_scratch_ref: hijax.TransformedRef,
    out_scratch_ref: hijax.TransformedRef,
    send_sem_ref: hijax.TransformedRef,
    recv_sem_ref: hijax.TransformedRef,
    cfg: _InputPackConfig,
    tail_in_scratch_ref: hijax.TransformedRef | None = None,
    tail_out_scratch_ref: hijax.TransformedRef | None = None,
) -> None:
  """Double-buffered HBM-to-VMEM relayout from large to compact layout.

  The first `num_tiles * tile_rows` rows go through the double-buffered tile
  loop. When `batch_size` is not a multiple of `tile_rows`, the remaining
  `tail_rows` rows are handled by a separate tail stage instead of padding the
  input and slicing the output in HBM:

  * Input: the HBM input is physically padded to its sublane tiling, so a
    `pl.multiple_of`-aligned dynamic slice reads the tail as one aligned
    `(padded_tail_rows, dim_size)` DMA without an HBM copy (only tail-only
    batches smaller than `padded_tail_rows` pad `x` in HBM). The extra rows are
    relaid out but never written back.
  * Output: the output's row dimension is untiled, so exactly `tail_rows` rows
    are written back.

  Args:
    in_hbm_ref: 2D input HBM ref `(batch_size, dim_size)`.
    out_hbm_ref: 4D compact output HBM ref.
    in_scratch_ref: Multi-buffered VMEM input scratch ref.
    out_scratch_ref: Multi-buffered VMEM output scratch ref.
    send_sem_ref: DMA semaphores for output writes.
    recv_sem_ref: DMA semaphores for input reads.
    cfg: Static kernel configuration.
    tail_in_scratch_ref: Optional VMEM input scratch ref for tail rows.
    tail_out_scratch_ref: Optional VMEM output scratch ref for tail rows.
  """

  def start_copy_in(tile_idx: jax.Array | int, slot: jax.Array | int):
    row_slice = pl.ds(tile_idx * cfg.tile_rows, cfg.tile_rows)
    pltpu.make_async_copy(
        in_hbm_ref.at[row_slice],
        in_scratch_ref.at[slot],
        recv_sem_ref.at[slot],
    ).start()

  def wait_copy_in(slot: jax.Array | int) -> None:
    dummy_in_ref = in_scratch_ref.at[0]
    pltpu.make_async_copy(
        dummy_in_ref,
        dummy_in_ref,
        recv_sem_ref.at[slot],
    ).wait()

  def start_copy_out(tile_idx: jax.Array | int, slot: jax.Array | int):
    row_slice = pl.ds(tile_idx * cfg.tile_rows, cfg.tile_rows)
    pltpu.make_async_copy(
        out_scratch_ref.at[slot, :, : cfg.num_packed_chunks],
        out_hbm_ref.at[row_slice],
        send_sem_ref.at[slot],
    ).start(priority=1)

  def wait_copy_out(slot: jax.Array | int) -> None:
    dummy_out_ref = out_scratch_ref.at[0, :, : cfg.num_packed_chunks]
    pltpu.make_async_copy(
        dummy_out_ref,
        dummy_out_ref,
        send_sem_ref.at[slot],
    ).wait()

  def tail_copy_in():
    assert tail_in_scratch_ref is not None
    tail_start = pl.multiple_of(
        jnp.int32(cfg.tail_row_start), cfg.num_sublanes
    )
    # The tail uses the semaphore after the `num_buffers` tile semaphores.
    return pltpu.make_async_copy(
        in_hbm_ref.at[pl.ds(tail_start, cfg.padded_tail_rows)],
        tail_in_scratch_ref,
        recv_sem_ref.at[cfg.num_buffers],
    )

  def tail_copy_out():
    assert tail_out_scratch_ref is not None
    return pltpu.make_async_copy(
        tail_out_scratch_ref.at[: cfg.tail_rows, : cfg.num_packed_chunks],
        out_hbm_ref.at[pl.ds(cfg.tail_row_start, cfg.tail_rows)],
        send_sem_ref.at[cfg.num_buffers],
    )

  @jax.named_scope("step")
  def step(tile_idx: jax.Array):
    """Executes one double-buffered pipeline iteration for `tile_idx`."""
    slot = tile_idx % cfg.num_buffers

    # Step 1: Wait for previous output tile DMA on this buffer slot before
    # overwriting `out_scratch_ref.at[slot]`.
    pl.when(tile_idx >= cfg.num_buffers)(functools.partial(wait_copy_out, slot))

    # Step 2: Wait for current input tile DMA and load into VREGs.
    wait_copy_in(slot)

    # Step 3: Relayout from large VREG layout to bank-conflict-free VMEM.
    val = in_scratch_ref.at[slot][...]
    layout_util.store_large_to_compact(out_scratch_ref.at[slot], val)

    # Step 4: Write current compact tile to HBM.
    start_copy_out(tile_idx, slot)

    # Step 5: Prefetch next input tile into this buffer slot now that `val`
    # has been loaded into VREGs.
    next_tile_idx = tile_idx + cfg.num_buffers
    pl.when(next_tile_idx < cfg.num_tiles)(
        functools.partial(start_copy_in, next_tile_idx, slot)
    )

  with jax.named_scope("prologue"):
    for slot in range(cfg.num_active_buffers):
      start_copy_in(slot, slot)
    # The tail's input is independent of the tile loop, so fetch it up front
    # to overlap with the pipelined tiles.
    if cfg.tail_rows:
      tail_copy_in().start()

  if cfg.num_tiles:
    with jax.named_scope("main_loop"):
      pl.loop(0, cfg.num_tiles)(step)

  # Tail: relayout the remainder rows and start their output DMA.
  if cfg.tail_rows:
    with jax.named_scope("tail"):
      assert tail_in_scratch_ref is not None
      assert tail_out_scratch_ref is not None
      tail_copy_in().wait()
      layout_util.store_large_to_compact(
          tail_out_scratch_ref, tail_in_scratch_ref[...]
      )
      tail_copy_out().start(priority=1)

  with jax.named_scope("epilogue"):
    for tile_idx in range(
        cfg.num_tiles - cfg.num_active_buffers, cfg.num_tiles
    ):
      wait_copy_out(tile_idx % cfg.num_buffers)
    if cfg.tail_rows:
      tail_copy_out().wait()


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
  if cfg.tail_rows and x.shape[0] < cfg.padded_tail_rows:
    x = jnp.pad(x, ((0, cfg.padded_tail_rows - x.shape[0]), (0, 0)))
  # No layout constraint on `x`: the kernel is equally fast with any sublane
  # tiling, and leaving the choice to XLA avoids a relayout copy for inputs in
  # XLA's default layout and lets producers in the same jit match the kernel.
  out = pl.kernel(
      functools.partial(_input_pack_kernel, cfg=cfg),
      out_type=jax.ShapeDtypeStruct(cfg.out_shape, cfg.dtype),
      mesh=pltpu.TensorCoreMesh(axis_name="core", num_cores=1),
      scratch_types=cfg.scratch_types(),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=cfg.vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      name=cfg.kernel_name,
  )(x)
  return arg.ActivationWrapper(data=out, scale=None, spec=cfg.out_spec)
