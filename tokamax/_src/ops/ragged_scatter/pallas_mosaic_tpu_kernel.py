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
"""Pallas Sparse Core kernel for Ragged Scatter."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


@dataclasses.dataclass(frozen=True)
class Config:
  col_size: int
  num_cores: int
  num_simd_lanes: int
  block_size: int
  out_pad_size: int
  aligned_hidden_size: int
  packing: int


def main_kernel(
    # Inputs.
    total_num_rows_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    src_indices_hbm_ref: jax.Ref,
    dst_indices_hbm_ref: jax.Ref,
    # Outputs.
    out_hbm_ref: jax.Ref,
    # Scratch.
    total_num_rows_vmem_ref: jax.Ref,
    out_vmem_ref: jax.Ref,
    src_indices_vmem_ref: jax.Ref,
    dst_indices_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
) -> None:
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  assert sc_info is not None
  num_simd_lanes = sc_info.num_lanes
  num_lanes = tpu_info.num_lanes
  hidden_size = in_hbm_ref.shape[-1]
  col_size = out_vmem_ref.shape[-1]
  num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
  block_size = num_simd_lanes * num_cores

  recv_sem = sem_ref.at[0]
  send_sem = sem_ref.at[1]

  # Read total number of valid rows tensor values.
  dma = pltpu.make_async_copy(
      total_num_rows_ref, total_num_rows_vmem_ref.at[:1], recv_sem
  )
  dma.start()
  dma.wait()
  total_num_rows = total_num_rows_vmem_ref[...][0]

  # Calculate number of tiles to visit.
  num_blocks = jnp.where(
      total_num_rows == 0, 0, pl.cdiv(total_num_rows, block_size)
  )
  num_cols = pl.cdiv(hidden_size, col_size)

  def inner_kernel(
      block_id: int | jax.Array,
      core_id: int | jax.Array,
      col_id: int | jax.Array,
  ) -> None:
    row_tile_start = block_id * block_size + core_id * num_simd_lanes
    col_tile_start = col_id * col_size

    @pl.when(col_id == 0)
    def _():
      dma_list = []
      dma_list.append(
          pltpu.make_async_copy(
              src_indices_hbm_ref.at[pl.ds(row_tile_start, num_simd_lanes)],
              src_indices_vmem_ref,
              recv_sem,
          )
      )
      dma_list.append(
          pltpu.make_async_copy(
              dst_indices_hbm_ref.at[pl.ds(row_tile_start, num_simd_lanes)],
              dst_indices_vmem_ref,
              recv_sem,
          )
      )
      jax.tree.map(lambda x: x.start(), dma_list)
      jax.tree.map(lambda x: x.wait(), dma_list)

    # HBM to VMEM transfer.
    src_indices = src_indices_vmem_ref[...]
    dst_indices = dst_indices_vmem_ref[...]

    dtype = out_hbm_ref.dtype
    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = 32 // dtype_bits

    in_32b_hbm_ref = in_hbm_ref.bitcast(jnp.uint32)
    out_32b_hbm_ref = out_hbm_ref.bitcast(jnp.uint32)

    for col_vmem_start in range(0, col_size, num_lanes):
      col_hbm_start = col_tile_start + col_vmem_start
      for row_vmem in range(num_simd_lanes):
        row_hbm = src_indices[row_vmem] // packing
        pltpu.make_async_copy(
            in_32b_hbm_ref.at[row_hbm, pl.ds(col_hbm_start, num_lanes)],
            out_vmem_ref.at[row_vmem, pl.ds(col_vmem_start, num_lanes)],
            recv_sem,
        ).start()

    # VMEM to HBM transfer.
    @pl.loop(0, col_size, step=num_lanes)
    @jax.named_scope("dma_write_loop")
    def dma_write_loop(col_vmem_start: int | jax.Array) -> None:
      col_hbm_start = col_tile_start + col_vmem_start

      # Wait for data to be received.
      for _ in range(num_simd_lanes):
        pltpu.make_async_copy(
            in_32b_hbm_ref.at[0, :num_lanes],
            out_vmem_ref.at[0, :num_lanes],
            recv_sem,
        ).wait()

      # If multiple elements are packed in single 32-bits, extract the desired
      # elements and reorder them.
      if packing > 1:
        for col_compute_offset in range(0, num_lanes, num_simd_lanes):
          col_slice = pl.ds(
              col_vmem_start + col_compute_offset, num_simd_lanes
          )

          previous_data = None
          for row_src in range(num_simd_lanes):
            row_src_pack = src_indices[row_src] % packing
            row_dst_pack = dst_indices[row_src] % packing

            rightshift_bits = row_src_pack * dtype_bits
            leftshift_bits = row_dst_pack * dtype_bits

            # Load data from vmem.
            data = out_vmem_ref[row_src, col_slice]

            # Right shift to make first n bits stores target data.
            data = jnp.bitwise_right_shift(data, rightshift_bits)
            # Mask out unwanted bits.
            data = jnp.bitwise_and(data, 2**dtype_bits - 1)
            # Left shift data into the target bit location.
            data = jnp.bitwise_left_shift(data, leftshift_bits)
            row_hbm = dst_indices[row_src] // packing

            if row_src == 0:
              prev_row_hbm = -1
              previous_data = data
            else:
              prev_row_hbm = dst_indices[row_src - 1] // packing
              assert previous_data is not None

            data_to_write = jnp.where(
                row_hbm == prev_row_hbm,
                jnp.bitwise_or(previous_data, data),
                data,
            )
            out_vmem_ref[row_src, col_slice] = data_to_write
            previous_data = data_to_write

      # Start dma write.
      last_valid_row_vmem = -1
      last_valid_row_hbm = -1
      for row_vmem in range(num_simd_lanes):
        row_valid = (row_tile_start + row_vmem) < total_num_rows
        row_hbm = dst_indices[row_vmem] // packing
        if row_vmem < num_simd_lanes - 1:
          next_row_hbm = dst_indices[row_vmem + 1] // packing
          next_row_valid = (row_tile_start + row_vmem + 1) < total_num_rows
        else:
          next_row_hbm = -1
          next_row_valid = False

        merged_data_vmem_row = (row_vmem // packing) * packing + packing - 1
        src_row_vmem = jnp.where(
            jnp.logical_and(row_hbm == next_row_hbm, next_row_valid),
            merged_data_vmem_row,
            jnp.where(row_valid, row_vmem, last_valid_row_vmem),
        )
        dst_row_hbm = jnp.where(
            jnp.logical_and(row_hbm == next_row_hbm, next_row_valid),
            row_hbm,
            jnp.where(row_valid, row_hbm, last_valid_row_hbm),
        )
        pltpu.make_async_copy(
            out_vmem_ref.at[src_row_vmem, pl.ds(col_vmem_start, num_lanes)],
            out_32b_hbm_ref.at[dst_row_hbm, pl.ds(col_hbm_start, num_lanes)],
            send_sem,
        ).start()
        last_valid_row_vmem = jnp.where(
            row_valid, src_row_vmem, last_valid_row_vmem
        )
        last_valid_row_hbm = jnp.where(
            row_valid, dst_row_hbm, last_valid_row_hbm
        )

    # Wait for dma write to finish.
    for _ in range(0, col_size, num_lanes):
      for _ in range(num_simd_lanes):
        pltpu.make_async_copy(
            out_vmem_ref.at[0, :num_lanes],
            out_32b_hbm_ref.at[0, :num_lanes],
            send_sem,
        ).wait()

  @functools.partial(
      pltpu.emit_pipeline,
      grid=(num_blocks, num_cores, num_cols),
      core_axis_name=(core_axis_name, subcore_axis_name),
      dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL, pltpu.ARBITRARY),
  )
  def kernel_wrapper() -> None:
    block_id = pl.program_id(0)
    core_id = pl.program_id(1)
    col_id = pl.program_id(2)
    row_tile_start = block_id * block_size + core_id * num_simd_lanes

    @pl.when(row_tile_start < total_num_rows)
    def _():
      inner_kernel(block_id, core_id, col_id)

  kernel_wrapper()


_VMEM_USAGE_RATIO = 0.8
_VMEM_CAPACITY_GEN6_BYTES = 256 * 1024
_VMEM_CAPACITY_GEN7_BYTES = 512 * 1024
_VMEM_CAPACITY_DEFAULT_BYTES = 128 * 1024


def calculate_col_size(hidden_size: int) -> int:
  """Calculate col size for ragged gather kernel."""
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  assert sc_info is not None
  num_lanes = tpu_info.num_lanes
  num_simd_lanes = sc_info.num_lanes

  match tpu_info.generation:
    case 6:
      vmem_capacity = _VMEM_CAPACITY_GEN6_BYTES
    case 7:
      vmem_capacity = _VMEM_CAPACITY_GEN7_BYTES
    case _:
      vmem_capacity = _VMEM_CAPACITY_DEFAULT_BYTES

  target_bytes = vmem_capacity * _VMEM_USAGE_RATIO

  base_bytes = num_simd_lanes * hidden_size * (32 // 8)
  num_cols = 1

  while pl.cdiv(base_bytes, num_cols * num_lanes) * num_lanes > target_bytes:
    num_cols += 1
  return pl.cdiv(hidden_size, (num_cols * num_lanes)) * num_lanes


def _preprocess_indices(
    indices: jax.Array,
    start: jax.Array,
    end: jax.Array,
    out_pad_size: int,
    packing: int,
    row_tile_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Preprocesses indices for ragged scatter kernel."""
  assert indices.ndim == 1, "Ragged scatter only supports 1d indices."
  assert row_tile_size % packing == 0

  src_indices = jnp.where(
      jnp.logical_and(indices >= start, indices < end), indices, -1
  )
  src_indices = jnp.pad(src_indices, ((0, out_pad_size)), constant_values=-1)
  src_indices = src_indices.reshape(-1, packing)
  is_valid_src_row = src_indices != -1
  num_sublanes = src_indices.shape[0]
  num_valid_src_rows_per_dst_sublane = jnp.sum(
      is_valid_src_row, axis=-1, keepdims=False
  )
  num_valid_src_rows_per_dst_sublane = jnp.broadcast_to(
      num_valid_src_rows_per_dst_sublane[:, None], (num_sublanes, packing)
  )

  cnts = jnp.where(
      num_valid_src_rows_per_dst_sublane > 1,
      packing,
      jnp.where(is_valid_src_row, 1, 0),
  ).reshape(-1)
  sorted_by_cnts = jnp.argsort(cnts, descending=True, stable=True)
  src_indices = (
      jnp.pad(indices, ((0, out_pad_size)), constant_values=0)
  )[sorted_by_cnts]
  dst_indices = sorted_by_cnts

  total_num_valid_source_rows = jnp.sum(cnts > 0)[None]
  return src_indices, dst_indices, total_num_valid_source_rows


def create_config(
    x_shape: tuple[int, int],
    indices_size: int,
    dtype: jnp.dtype,
) -> Config:
  """Create Config dynamically based on array shapes."""
  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None

  dtype_bits = jax.dtypes.itemsize_bits(dtype)
  packing = 32 // dtype_bits
  hidden_size = x_shape[-1]

  num_simd_lanes = sc_info.num_lanes
  num_cores = sc_info.num_cores * sc_info.num_subcores
  block_size = num_simd_lanes * num_cores
  col_size = calculate_col_size(hidden_size)

  out_pad_size = (
      pl.cdiv(indices_size, block_size) * block_size - indices_size
  )
  aligned_hidden_size = pl.cdiv(hidden_size, col_size) * col_size

  return Config(
      col_size=col_size,
      num_cores=num_cores,
      num_simd_lanes=num_simd_lanes,
      block_size=block_size,
      out_pad_size=out_pad_size,
      aligned_hidden_size=aligned_hidden_size,
      packing=packing,
  )


def ragged_scatter_pallas(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array,
    end: jax.Array,
    *,
    config: Config | None = None,
) -> jax.Array:
  """Scatter function using Pallas kernel."""
  if config is None:
    config = create_config(x.shape, indices.size, x.dtype)

  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None

  if jnp.isscalar(start):
    start = start[None]
  if jnp.isscalar(end):
    end = end[None]

  hidden_size = x.shape[-1]
  out_size = indices.size

  src_indices, dst_indices, total_num_rows = _preprocess_indices(
      indices,
      start,
      end,
      config.out_pad_size,
      config.packing,
      row_tile_size=config.num_simd_lanes,
  )

  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )

  # Check that direct return is used here to avoid linter warnings
  return pl.kernel(
      functools.partial(
          main_kernel,
          core_axis_name=vector_mesh.core_axis_name,
          subcore_axis_name=vector_mesh.subcore_axis_name,
      ),
      out_type=jax.ShapeDtypeStruct(
          (out_size + config.out_pad_size, config.aligned_hidden_size), x.dtype
      ),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          disable_bounds_checks=True,
      ),
      scratch_types=dict(
          total_num_rows_vmem_ref=pltpu.VMEM(
              (config.num_simd_lanes,), jnp.int32
          ),
          out_vmem_ref=pltpu.VMEM(
              (config.num_simd_lanes, config.col_size), jnp.uint32
          ),
          src_indices_vmem_ref=pltpu.VMEM(
              (config.num_simd_lanes,), jnp.int32
          ),
          dst_indices_vmem_ref=pltpu.VMEM(
              (config.num_simd_lanes,), jnp.int32
          ),
          sem_ref=pltpu.SemaphoreType.DMA((2,)),
      ),
      mesh=vector_mesh,
      name="sc_ragged_scatter",
  )(total_num_rows, x, src_indices, dst_indices)[:out_size, :hidden_size]
