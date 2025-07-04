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
"""Pallas grid utilities."""

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from tokamax._src.jaxtyping import ScalarInt  # pylint: disable=g-importing-member


def _get_group_size_cost(
    group_size_m: int,
    grid_m: int,
    grid_n: int,
    block_m_cost: int,
    block_n_cost: int,
) -> int:
  """Returns the total cost for the given group size."""
  num_live_groups = pl.cdiv(jax.devices()[0].core_count, group_size_m)
  num_live_blocks_m = pl.cdiv(num_live_groups, grid_n) * group_size_m
  num_live_blocks_m = min(num_live_blocks_m, grid_m)
  num_live_blocks_n = min(num_live_groups, grid_n)
  return num_live_blocks_m * block_m_cost + num_live_blocks_n * block_n_cost


def get_grid_pids(
    pid: ScalarInt, grid_m: int, grid_n: int, group_size_m: int
) -> tuple[ScalarInt, ScalarInt]:
  """Returns the program IDs in each grid axis."""
  # For example, if `grid_m=5`, `grid_n=6` and `group_size_m=2`, then `pid` is
  # mapped as follows:
  #   pid_n     |  0 |  1 |  2 |  3 |  4 |  5 |
  #   -----------------------------------------
  #   pid_m   0 |  0 |  2 |  4 |  6 |  8 | 10 |
  #           1 |  1 |  3 |  5 |  7 |  9 | 11 |
  #           2 | 12 | 14 | 16 | 18 | 20 | 22 |
  #           3 | 13 | 15 | 17 | 19 | 21 | 23 |
  #           4 | 24 | 25 | 26 | 27 | 28 | 29 |

  # Use `floor_divide` and `remainder` (instead of lax.div and lax.rem)
  # to handle dtypes: pid (int32) vs. num_blocks_n (int64) when `jax_enable_x64`
  # is set.
  if group_size_m == 1:
    return jnp.floor_divide(pid, grid_n), jnp.remainder(pid, grid_n)

  num_progs_in_group = group_size_m * grid_n
  group_start_m = jnp.floor_divide(pid, num_progs_in_group) * group_size_m
  group_size_m = jnp.minimum(grid_m - group_start_m, group_size_m)
  pid_m = group_start_m + jnp.remainder(pid, group_size_m)
  pid_n = jnp.floor_divide(jnp.remainder(pid, num_progs_in_group), group_size_m)
  return pid_m, pid_n


def get_cheapest_grid_pids(
    pid: ScalarInt,
    *,
    grid_m: int,
    grid_n: int,
    block_m_cost: int,
    block_n_cost: int,
) -> tuple[ScalarInt, ScalarInt]:
  """Returns the grouped program IDs that minimize the total cost."""
  num_live_progs = jax.devices()[0].core_count

  def group_size_m_usage(group_size_m):
    return _get_group_size_cost(
        group_size_m, grid_m, grid_n, block_m_cost, block_n_cost
    )

  group_size_m = min(
      range(1, min(num_live_progs, grid_m) + 1), key=group_size_m_usage
  )

  def group_size_n_usage(group_size_n):
    return _get_group_size_cost(
        group_size_n, grid_n, grid_m, block_n_cost, block_m_cost
    )

  group_size_n = min(
      range(1, min(num_live_progs, grid_n) + 1), key=group_size_n_usage
  )

  if group_size_m_usage(group_size_m) <= group_size_n_usage(group_size_n):
    pid_m, pid_n = get_grid_pids(pid, grid_m, grid_n, group_size_m)
  else:
    pid_n, pid_m = get_grid_pids(pid, grid_n, grid_m, group_size_n)
  return pid_m, pid_n
