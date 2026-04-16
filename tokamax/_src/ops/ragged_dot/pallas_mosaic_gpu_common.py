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

from collections.abc import Sequence
import dataclasses
import enum

import jax
from jax import lax
from jax.experimental import pallas as pl
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

    Returns:
      A GroupInfo instance containing the calculated information for each task.
      Note, that block array is always None.
    """
    group_sizes = jnp.asarray(group_sizes)
    group_sizes = lax.max(jnp.zeros_like(group_sizes), group_sizes)
    group_sizes = group_sizes.astype(jnp.int32)
    group_starts = jnp.cumulative_sum(group_sizes, include_initial=True)
    group_starts_aligned = lax.div(group_starts[:-1], align_tile) * align_tile
    # We could use `group_starts[1:]`, but XLA fuses this better.
    group_ends = group_starts[:-1] + group_sizes
    group_num_blocks = jnp.where(
        group_sizes == 0, 0, pl.cdiv(group_ends - group_starts_aligned, tile)
    )
    group_start_tasks = jnp.cumulative_sum(
        group_num_blocks, include_initial=True
    )
    group_end_tasks = group_start_tasks[:-1] + group_num_blocks  # As above.
    task_idx = jnp.arange(tid_size)
    # `method="scan_unrolled"` should be better, particularly for larger numbers
    # of groups, but XLA doesn't fuse it into a single fusion.
    task_group_idx = jnp.searchsorted(
        group_end_tasks[:-1], task_idx, "right", method="compare_all"
    )
    task_block_idx = task_idx - group_start_tasks[task_group_idx]
    task_starts = group_starts_aligned[task_group_idx] + tile * task_block_idx
    task_valid_starts = lax.max(task_starts, group_starts[task_group_idx])
    task_valid_ends = lax.min(task_starts + tile, group_ends[task_group_idx])
    task_valid_size = lax.max(0, task_valid_ends - task_valid_starts)
    return cls(
        task_group_idx,
        task_starts,
        task_valid_starts,
        task_valid_ends,
        task_valid_starts - task_starts,
        task_valid_size,
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
