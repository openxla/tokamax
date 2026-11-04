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
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.common import out_format as out_format_lib
from tokamax._src.ops.experimental.routing_moe.common import spec as spec_lib

_DEFAULT_NUM_BUFFERS = 2
_DEFAULT_MAX_TILE_ROWS = 512
_SCOPED_VMEM_RESERVED_BYTES = 64 * 1024
_MAX_SCRATCH_VMEM_RATIO = 0.75


def _padded_tail_rows(
    batch_size: int, tail_rows: int, packing: int, num_sublanes: int
) -> int:
  """Rounds `tail_rows` up to the input's HBM sublane tiling.

  The tail DMA must be aligned to that tiling, and `pl.pad_ref` may only extend
  the input up to it. XLA tiles a `(batch_size, D)` kernel input by
  `num_sublanes` rows, or by a larger tile that already divides the tail, when
  `batch_size >= num_sublanes`; otherwise by the smallest power of two that
  covers `batch_size`, but at least `packing` rows.

  Args:
    batch_size: Number of input rows.
    tail_rows: Number of rows left after the full tiles.
    packing: Number of elements per 32-bit word.
    num_sublanes: Number of sublanes per vreg.

  Returns:
    `tail_rows` rounded up to the input's HBM sublane tiling.
  """
  if batch_size >= num_sublanes:
    return pl.cdiv(tail_rows, num_sublanes) * num_sublanes
  return max(packing, pl.next_power_of_2(batch_size))


def _select_tile_rows(
    batch_size: int,
    dim_size: int,
    vmem_chunks: int,
    dtype: jax.typing.DTypeLike,
    tpu_info: pltpu.TpuInfo,
    requested_tile_size: int | None,
    num_buffers: int = _DEFAULT_NUM_BUFFERS,
) -> int:
  """Selects a sublane-aligned tile size within VMEM capacity."""
  x_dtype = jax.dtypes.canonicalize_dtype(dtype)
  sublane_size = tpu_info.get_sublane_tiling(x_dtype)
  # Floor so `tile_rows <= batch_size` whenever `batch_size >= sublane_size`;
  # the remaining `batch_size % tile_rows` rows are handled by the tail.
  batch_sublanes = max(1, batch_size // sublane_size)
  if requested_tile_size is None:
    # Cap at ceil(batch_sublanes / 2) when possible so num_tiles >= 2 and HBM
    # DMAs overlap with VPU relayout via double buffering.
    max_tile_sublanes = max(
        1,
        min(_DEFAULT_MAX_TILE_ROWS // sublane_size, (batch_sublanes + 1) // 2),
    )
    # Prefer an exact divisor of `batch_sublanes` when it is at least half of
    # `max_tile_sublanes` (e.g. batch=768 -> tile=384) to avoid a tail;
    # otherwise use `max_tile_sublanes` and let the tail handle the remaining
    # rows rather than degrading to 1-sublane tiles on prime/irregular batches.
    gcd_sublanes = math.gcd(batch_sublanes, max_tile_sublanes)
    if batch_size % sublane_size == 0 and gcd_sublanes >= max(
        1, max_tile_sublanes // 2
    ):
      tile_sublanes = gcd_sublanes
    else:
      tile_sublanes = max_tile_sublanes
  else:
    if requested_tile_size <= 0 or requested_tile_size % sublane_size != 0:
      raise ValueError(
          f"tile_size ({requested_tile_size}) must be a positive multiple of"
          f" sublane_size ({sublane_size}) for dtype {x_dtype}."
      )
    tile_sublanes = min(batch_sublanes, requested_tile_size // sublane_size)

  tile_rows = sublane_size * tile_sublanes

  # Each buffer slot holds one input tile `(tile_rows, dim_size)` and one
  # bank-conflict-padded output tile `(tile_rows, vmem_chunks, packing, lanes)`
  # in VMEM, and a ragged tail needs at most one more slot. When `dim_size` is
  # large, the initial `tile_rows` may exceed VMEM capacity; repeatedly halve
  # the sublane count per tile (preserving sublane alignment) until all scratch
  # tensors fit within the VMEM budget (`_MAX_SCRATCH_VMEM_RATIO` of capacity,
  # leaving headroom for compiler-managed VMEM allocations).
  packing = 4 // x_dtype.itemsize
  bytes_per_row = (
      dim_size + vmem_chunks * packing * tpu_info.num_lanes
  ) * x_dtype.itemsize

  max_scratch_bytes = int(
      tpu_info.vmem_capacity_bytes * _MAX_SCRATCH_VMEM_RATIO
  )

  def scratch_bytes(tile_rows: int) -> int:
    num_slots = num_buffers + (batch_size % tile_rows != 0)
    return num_slots * tile_rows * bytes_per_row

  while (
      tile_rows > sublane_size and scratch_bytes(tile_rows) > max_scratch_bytes
  ):
    tile_rows = max(
        sublane_size, ((tile_rows // sublane_size) // 2) * sublane_size
    )
  return tile_rows


@dataclasses.dataclass(frozen=True)
class _InputPackPipeline:
  """Double-buffered HBM-to-VMEM relayout pipeline for `input_pack`.

  The first `num_tiles * tile_rows` rows go through the double-buffered tile
  loop. When `batch_size` is not a multiple of `tile_rows`, the remaining
  `tail_rows` rows are handled by a separate tail stage instead of padding the
  input and slicing the output in HBM:

  * Input: the HBM input is physically padded to its sublane tiling, so
    `pl.pad_ref` extends it to that tile-aligned bound and the tail is read as
    one aligned `(padded_tail_rows, dim_size)` DMA. The extra rows hold
    unspecified data; they are relaid out but never written back.
  * Output: the output's row dimension is untiled, so exactly `tail_rows` rows
    are written back.
  """

  in_hbm_ref: Any
  out_hbm_ref: Any
  in_scratch_ref: Any
  out_scratch_ref: Any
  # `None` when `tail_rows == 0`.
  tail_in_scratch_ref: Any
  tail_out_scratch_ref: Any
  send_sem_ref: Any
  recv_sem_ref: Any

  @property
  def batch_size(self) -> int:
    return self.in_hbm_ref.shape[0]

  @property
  def dim_size(self) -> int:
    return self.in_hbm_ref.shape[1]

  @property
  def num_packed_chunks(self) -> int:
    return self.out_hbm_ref.shape[1]

  @property
  def num_buffers(self) -> int:
    return self.in_scratch_ref.shape[0]

  @property
  def tile_rows(self) -> int:
    return self.in_scratch_ref.shape[1]

  @property
  def num_tiles(self) -> int:
    """Number of full `tile_rows` tiles processed by the pipelined loop."""
    return self.batch_size // self.tile_rows

  @property
  def tail_row_start(self) -> int:
    return self.num_tiles * self.tile_rows

  @property
  def tail_rows(self) -> int:
    return self.batch_size - self.tail_row_start

  def start_copy_in(self, tile_idx: jax.Array | int, slot: jax.Array | int):
    row_slice = pl.ds(tile_idx * self.tile_rows, self.tile_rows)
    pltpu.make_async_copy(
        self.in_hbm_ref.at[row_slice],
        self.in_scratch_ref.at[slot],
        self.recv_sem_ref.at[slot],
    ).start()

  # `tpu.wait_dma` only uses the ref shapes/tiling to compute the transfer
  # byte count (`llo::DMADoneOp`) and discards the base pointers. Passing a
  # constant-indexed VMEM ref (`at[0]`) for both `src` and `dst` avoids
  # lowering dynamic `MemRefSliceOp` scalar address arithmetic on `tile_idx`
  # or `slot` inside the pipeline loop.
  def wait_copy_in(self, slot: jax.Array | int):
    dummy_in_ref = self.in_scratch_ref.at[0]
    pltpu.make_async_copy(
        dummy_in_ref,
        dummy_in_ref,
        self.recv_sem_ref.at[slot],
    ).wait()

  def start_copy_out(self, tile_idx: jax.Array | int, slot: jax.Array | int):
    row_slice = pl.ds(tile_idx * self.tile_rows, self.tile_rows)
    pltpu.make_async_copy(
        self.out_scratch_ref.at[slot, :, : self.num_packed_chunks],
        self.out_hbm_ref.at[row_slice],
        self.send_sem_ref.at[slot],
    ).start(priority=1)

  def wait_copy_out(self, slot: jax.Array | int):
    dummy_out_ref = self.out_scratch_ref.at[0, :, : self.num_packed_chunks]
    pltpu.make_async_copy(
        dummy_out_ref,
        dummy_out_ref,
        self.send_sem_ref.at[slot],
    ).wait()

  def _tail_copy_in(self) -> Any:
    # `pl.pad_ref` extends the HBM input to its sublane-tile-aligned bound
    # (the buffer is physically padded to the `(sublane, lanes)` tiling), so
    # the tail DMA is a full aligned tile copy into the padded VMEM buffer.
    padded_tail_rows = self.tail_in_scratch_ref.shape[0]
    padded_in_hbm_ref = pl.pad_ref(
        self.in_hbm_ref,
        (self.tail_row_start + padded_tail_rows, self.dim_size),
    )
    # The tail uses the semaphore after the `num_buffers` tile semaphores.
    return pltpu.make_async_copy(
        padded_in_hbm_ref.at[pl.ds(self.tail_row_start, padded_tail_rows)],
        self.tail_in_scratch_ref,
        self.recv_sem_ref.at[self.num_buffers],
    )

  def _tail_copy_out(self) -> Any:
    tail_out_ref = self.tail_out_scratch_ref
    return pltpu.make_async_copy(
        tail_out_ref.at[: self.tail_rows, : self.num_packed_chunks],
        self.out_hbm_ref.at[pl.ds(self.tail_row_start, self.tail_rows)],
        self.send_sem_ref.at[self.num_buffers],
    )

  def run_tail(self):
    """Relayouts the `tail_rows` remainder rows and starts their output DMA."""
    self._tail_copy_in().wait()
    layout_util.store_large_to_compact(
        self.tail_out_scratch_ref, self.tail_in_scratch_ref[...]
    )
    self._tail_copy_out().start(priority=1)

  def step(self, tile_idx: jax.Array):
    """Executes one double-buffered pipeline iteration for `tile_idx`."""
    slot = tile_idx % self.num_buffers

    # Step 1: Wait for current input tile DMA and load into VREGs.
    self.wait_copy_in(slot)
    val = self.in_scratch_ref.at[slot][...]

    # Step 2: Prefetch next input tile into this buffer slot now that `val`
    # has been loaded into VREGs.
    next_tile_idx = tile_idx + self.num_buffers
    pl.when(next_tile_idx < self.num_tiles)(
        functools.partial(self.start_copy_in, next_tile_idx, slot)
    )

    # Step 3: Wait for previous output tile DMA on this buffer slot before
    # overwriting `out_scratch_ref.at[slot]`.
    pl.when(tile_idx >= self.num_buffers)(
        functools.partial(self.wait_copy_out, slot)
    )

    # Step 4: Relayout from large VREG layout to bank-conflict-free VMEM.
    layout_util.store_large_to_compact(self.out_scratch_ref.at[slot], val)

    # Step 5: Write current compact tile to HBM.
    self.start_copy_out(tile_idx, slot)

  def run(self):
    """Runs the prologue, main loop, tail, and epilogue DMAs."""
    num_active_buffers = min(self.num_buffers, self.num_tiles)
    for slot in range(num_active_buffers):
      self.start_copy_in(slot, slot)
    # The tail's input is independent of the tile loop, so fetch it up front
    # to overlap with the pipelined tiles.
    if self.tail_rows:
      self._tail_copy_in().start()

    if self.num_tiles:
      pl.loop(0, self.num_tiles)(self.step)

    if self.tail_rows:
      self.run_tail()

    for tile_idx in range(self.num_tiles - num_active_buffers, self.num_tiles):
      self.wait_copy_out(tile_idx % self.num_buffers)
    if self.tail_rows:
      self._tail_copy_out().wait()


def _input_pack_kernel(
    in_hbm_ref,
    out_hbm_ref,
    in_scratch_ref,
    out_scratch_ref,
    *scratch_refs,
    has_tail: bool,
):
  """Pallas kernel entry point for double-buffered HBM-to-VMEM relayout."""
  if has_tail:
    tail_in_scratch_ref, tail_out_scratch_ref, send_sem_ref, recv_sem_ref = (
        scratch_refs
    )
  else:
    tail_in_scratch_ref = tail_out_scratch_ref = None
    send_sem_ref, recv_sem_ref = scratch_refs
  _InputPackPipeline(
      in_hbm_ref=in_hbm_ref,
      out_hbm_ref=out_hbm_ref,
      in_scratch_ref=in_scratch_ref,
      out_scratch_ref=out_scratch_ref,
      tail_in_scratch_ref=tail_in_scratch_ref,
      tail_out_scratch_ref=tail_out_scratch_ref,
      send_sem_ref=send_sem_ref,
      recv_sem_ref=recv_sem_ref,
  ).run()


@jax.jit(static_argnames=("out_format", "tile_size"))
def input_pack(
    x: jax.Array,
    out_format: out_format_lib.ActivationFormatLike | None = None,
    *,
    tile_size: int | None = None,
) -> arg.ActivationWrapper[jax.Array]:
  """Relayouts input `x` from large 2D layout to 4D compact layout."""
  if x.ndim != 2:
    raise ValueError(f"Expected 2D input (batch_size, dim_size), got {x.shape}")

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

  tpu_info = pltpu.get_tpu_info()
  num_lanes = tpu_info.num_lanes
  x_dtype = x.dtype
  packing = 4 // x_dtype.itemsize
  batch_size, dim_size = x.shape

  if dim_size % (packing * num_lanes) != 0:
    raise ValueError(
        f"dim_size ({dim_size}) must be divisible by packing * num_lanes"
        f" ({packing * num_lanes}) for dtype {x_dtype}."
    )

  num_packed_chunks = dim_size // (packing * num_lanes)
  vmem_chunks = layout_util.to_bank_conflict_aware_stride(num_packed_chunks)

  tile_rows = _select_tile_rows(
      batch_size=batch_size,
      dim_size=dim_size,
      vmem_chunks=vmem_chunks,
      dtype=x_dtype,
      tpu_info=tpu_info,
      requested_tile_size=tile_size,
  )
  out_shape = layout_util.to_compact_shape((batch_size, dim_size), x_dtype)
  num_tiles = batch_size // tile_rows
  # Allocate at least one slot so scratch shapes stay non-empty when
  # `batch_size < tile_rows` (i.e. `batch_size < sublane_size`) and every row
  # goes through the tail.
  num_buffers = max(1, min(num_tiles, _DEFAULT_NUM_BUFFERS))
  tail_rows = batch_size % tile_rows
  padded_tail_rows = _padded_tail_rows(
      batch_size, tail_rows, packing, tpu_info.num_sublanes
  )
  # No layout constraint on `x`: the kernel is equally fast with any sublane
  # tiling, and leaving the choice to XLA avoids a relayout copy for inputs in
  # XLA's default layout and lets producers in the same jit match the kernel.

  tail_scratch_types = ()
  if tail_rows:
    tail_scratch_types = (
        pltpu.VMEM((padded_tail_rows, dim_size), x_dtype),
        pltpu.VMEM(
            (padded_tail_rows, vmem_chunks, packing, num_lanes), x_dtype
        ),
    )
  # One extra DMA semaphore per direction for the tail.
  num_sems = num_buffers + (tail_rows != 0)

  out = pl.kernel(
      functools.partial(_input_pack_kernel, has_tail=tail_rows != 0),
      out_type=jax.ShapeDtypeStruct(out_shape, x_dtype),
      mesh=pltpu.TensorCoreMesh(axis_name="core", num_cores=1),
      scratch_types=(
          pltpu.VMEM((num_buffers, tile_rows, dim_size), x_dtype),
          pltpu.VMEM(
              (num_buffers, tile_rows, vmem_chunks, packing, num_lanes),
              x_dtype,
          ),
          *tail_scratch_types,
          pltpu.SemaphoreType.DMA((num_sems,)),
          pltpu.SemaphoreType.DMA((num_sems,)),
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=(
              tpu_info.vmem_capacity_bytes - _SCOPED_VMEM_RESERVED_BYTES
          ),
          disable_bounds_checks=True,
          flags={"XLA_TPU_EVENLY_DISTRIBUTE_CRITICAL_NODES": True},
      ),
      name=f"input_pack_b{batch_size}_d{dim_size}_t{tile_rows}",
  )(x)
  spec = spec_lib.ActivationSpec(
      row_size=batch_size, col_size=dim_size, dtype=x_dtype, quant=None
  )
  return arg.ActivationWrapper(data=out, scale=None, spec=spec)
