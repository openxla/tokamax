# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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

"""Common Pallas Mosaic GPU utilities."""

from collections.abc import Callable, Sequence
import dataclasses
import enum
import functools
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp
import pydantic

SMEM_CAPACITY_MAP = {
    "sm_120": (100 - 1) * 1024,
    "sm_103": (228 - 1) * 1024,
    "sm_100": (228 - 1) * 1024,
    "sm_90": (228 - 1) * 1024,
    "sm_80": (164 - 1) * 1024,
    "sm_86": (100 - 1) * 1024,
    "sm_89": (100 - 1) * 1024,
}


class MatmulDimension(enum.IntEnum):
  M = 0
  N = 1


@pydantic.dataclasses.dataclass(frozen=True, slots=True)
class Config:
  """Configuration for the ragged dot kernel."""

  block_m: pydantic.conint(multiple_of=8, gt=0)
  block_n: pydantic.PositiveInt
  block_k: pydantic.PositiveInt
  num_stages: pydantic.PositiveInt
  split_k: pydantic.PositiveInt
  split_m: pydantic.PositiveInt = 1
  grid_block_n: pydantic.PositiveInt = 1
  persistent: bool = True
  post_scale: bool = False
  # B200 collective MMA
  collective: bool = False
  # indicates the fastest changing dim.
  grid_minor_dim: MatmulDimension = MatmulDimension.N
  # The width of tiles along the fastest changing dim.
  grid_tile_width: int = 1


@dataclasses.dataclass(frozen=True, slots=True)
class GroupInfo:
  """Information regarding the group being processed in a block."""

  group_id: jax.Array
  block_start: jax.Array
  actual_start: jax.Array
  actual_end: jax.Array
  start_within_block: jax.Array
  actual_size: jax.Array

  @classmethod
  def create_aligned(
      cls,
      group_sizes: Sequence[jax.Array],
      tile: int,
      tid_size: int,
      align_tile: int = 8,
      noops_at_end: bool = True,
  ) -> "GroupInfo":
    """Creates a GroupInfo instance with block-aligned task assignments.

    This method calculates task assignments for processing a ragged tensor,
    ensuring that each block starts at an offset aligned with `align_tile`.

    Example:
     group_sizes=[17, 31, 24], block_m=32
     The `create` method will generate 5 blocks, starting at [ 0, 0, 32, 32, 64]
     `create_aligned` with align_tile=8 we will generate just three blocks at
     [0,  16,  48], thus avoiding wasting compute.


    Args:
      group_sizes: A sequence of jax.Array, where each element is the size of a
        group.
      tile: The size of the processing tile (e.g., block_m).
      tid_size: Max number of tasks available - usually calculated as
        `pl.cdiv(m, block_m) + len(group_sizes) - 1`.
      align_tile: The alignment boundary for the start of each block. Block
        starts will be multiples of this value. Defaults to 8.
      noops_at_end: If True, tasks that result in no actual work (actual_size ==
        0) are moved to the end of the task list. Defaults to True.

    Returns:
      A GroupInfo instance containing the calculated information for each task.
      Note, that block array is always None.
    """
    (
        group_idx,
        global_m_start,
        offset_in_block,
        actual_size,
        _,
    ) = calculate_group_info_tasks(
        group_sizes,
        max_tasks=tid_size,
        block_m=tile,
        align_block_size=align_tile,
        noops_at_end=noops_at_end,
    )
    actual_start = global_m_start + offset_in_block
    actual_end = global_m_start + offset_in_block + actual_size
    return cls(
        group_idx,
        global_m_start,
        actual_start,
        actual_end,
        offset_in_block,
        actual_size,
    )


def calculate_group_info_tasks(
    group_sizes: Sequence[jax.Array],
    max_tasks: int,
    block_m: int,
    align_block_size: int = 8,
    noops_at_end: bool = True,
):
  """Calculates task assignments for processing a ragged tensor with specified block alignment."""
  group_sizes = jnp.asarray(group_sizes).astype(jnp.int32)
  group_starts = jnp.pad(jnp.cumsum(group_sizes)[:-1], (1, 0))
  group_ends = group_starts + group_sizes
  aligned_starts = lax.div(group_starts, align_block_size) * align_block_size
  blocks_per_group = pl.cdiv(group_ends - aligned_starts, block_m)
  group_range = jnp.arange(group_sizes.shape[0])
  group_idx = jnp.repeat(
      group_range,
      axis=0,
      repeats=blocks_per_group,
      total_repeat_length=max_tasks,
  )

  block_starts_per_group = jnp.pad(jnp.cumsum(blocks_per_group)[:-1], (1, 0))
  cta_group_block_start = block_starts_per_group[group_idx]
  cta_id = jnp.arange(max_tasks)
  inside_block_idx = cta_id - cta_group_block_start
  global_m_start = aligned_starts[group_idx] + block_m * inside_block_idx
  actual_m_start = jnp.maximum(global_m_start, group_starts[group_idx])
  global_m_end = global_m_start + block_m
  actual_m_end = jnp.minimum(global_m_end, group_ends[group_idx])
  offset_in_block = actual_m_start - global_m_start
  actual_size = jnp.maximum(0, actual_m_end - actual_m_start)
  non_empty_mask = actual_size > 0
  if noops_at_end:
    noop_group_pos = 1024 * 1024
    idx = jnp.argsort(noop_group_pos * (1 - (actual_size > 0)) + group_idx)
    return (
        group_idx[idx],
        global_m_start[idx],
        offset_in_block[idx],
        actual_size[idx],
        non_empty_mask.sum(),
    )
  else:
    return (
        group_idx,
        global_m_start,
        offset_in_block,
        actual_size,
        non_empty_mask.sum(),
    )


def ragged_kernel(
    body, *, g, m, n, out_dtype, config, thread_axis=None, **kwargs
) -> Callable[..., jax.Array]:
  """Returns a Pallas kernel for ragged matmul.

  This kernel computes a ragged matmul, where the LHS is a dense
  matrix of shape (m, k) and the RHS is a ragged matrix of shape (g,
  k, n), where g is the number of groups. The output is a dense matrix
  of shape (m, n).

  The kernel uses a persistent kernel if config.persistent is True,
  otherwise it uses a non-persistent kernel.

  Args:
    body: The body of the kernel. This function will be called with the group
      info, the current m index, the current n index, and the arguments to the
      kernel.
    g: The number of groups.
    m: The m dimension of the LHS matrix.
    n: The n dimension of the RHS matrix.
    out_dtype: The dtype of the output matrix.
    config: The kernel config.
    thread_axis: The name of the thread axis to use for warp specialization. If
      None, warp specialization is not used.
    **kwargs: Additional keyword arguments to pass to the Pallas kernel.

  Returns:
    A Pallas kernel for ragged matmul.
  """

  num_compute_threads = 1 if thread_axis is None else 2
  inner_grid = (
      pl.cdiv(n, config.grid_block_n * config.block_n * num_compute_threads),
      pl.cdiv(m, config.block_m) + g - 1,
      config.grid_block_n,
  )

  def kernel_body(
      group_id_gmem,
      block_start_gmem,
      actual_start_gmem,
      actual_end_gmem,
      start_within_block_gmem,
      actual_size_gmem,
      *args,
  ):
    def loop_body(m_offset, loop_info: plgpu.NDLoopInfo):
      remainder_ni, mi, block_ni = loop_info.index
      mi += m_offset
      ni = (
          block_ni
          * pl.cdiv(
              n, config.block_n * config.grid_block_n * num_compute_threads
          )
          + remainder_ni
      )
      group_info = GroupInfo(
          group_id=group_id_gmem[mi],
          block_start=block_start_gmem[mi],
          actual_start=actual_start_gmem[mi],
          actual_end=actual_end_gmem[mi],
          start_within_block=start_within_block_gmem[mi],
          actual_size=actual_size_gmem[mi],
      )

      @pl.when(group_info.actual_size > 0)
      def _():
        body(group_info, mi, ni, *args)

    if config.persistent:
      # We stratify the grid: first emit a number of blocks that have
      # definitely work to do. Then schedule blocks that may be
      # noops. This way we lower the chances that noop bocks are
      # scheduled to the same SM.
      inner_grid_l = list(inner_grid)
      inner_grid_l[1] = pl.cdiv(m, config.block_m)
      plgpu.nd_loop(tuple(inner_grid_l), collective_axes="sm")(
          functools.partial(loop_body, 0)
      )
      inner_grid_l[1] = g - 1
      plgpu.nd_loop(tuple(inner_grid_l), collective_axes="sm")(
          functools.partial(loop_body, pl.cdiv(m, config.block_m))
      )
    else:
      loop_info = plgpu.NDLoopInfo(
          index=tuple(map(lax.axis_index, ("remainder_n", "m", "block_n"))),
          local_index=0,
          num_local_steps=1,
      )
      loop_body(0, loop_info)

  if config.persistent:
    grid = (backend.get_default_device().core_count,)
    grid_names = ("sm",)
  else:
    grid = inner_grid
    grid_names = ("remainder_n", "m", "block_n")

  return plgpu.kernel(
      kernel_body,
      out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
      grid=grid,
      grid_names=grid_names,
      thread_name=thread_axis,
      num_threads=thread_axis and (num_compute_threads + 1),
      **kwargs,
  )


def get_smem_capacity() -> int:
  """Returns the shared memory capacity of the device."""
  device = backend.get_default_device()

  if device.platform != "gpu":
    raise NotImplementedError(
        f"Unsupported device platform: {device}"
    )
  capacity = int(float(getattr(device, "compute_capability", 0)) * 10)
  sm_version = f"sm_{capacity}"
  if sm_version not in SMEM_CAPACITY_MAP:
    raise NotImplementedError(
        f"Unsupported device compute capability: {device} {sm_version}"
    )
  return SMEM_CAPACITY_MAP[sm_version]


def check_bf16xbf16_or_f16xf16(lhs: jax.Array, rhs: jax.Array):
  if lhs.dtype != rhs.dtype:
    raise NotImplementedError(
        f"lhs and rhs must have same dtype (got {lhs.dtype=}, {rhs.dtype=})."
    )

  if lhs.dtype not in (jnp.bfloat16, jnp.float16):
    raise NotImplementedError(
        f"Only bfloat16/float16 inputs are supported (got {lhs.dtype=})."
    )
