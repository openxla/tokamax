# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Common definitions for Pallas-Mosaic-GPU Gated Linear Unit."""

import enum
import pydantic


class MatmulDimension(enum.IntEnum):
  M = 0
  N = 1

  def __str__(self):
    return self.name

  def __repr__(self):
    return f"<MatmulDimension.{self.name}>"


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Configuration for the Pallas-Mosaic-GPU Gated Linear Unit.

  Attributes:
    tile_m: M dimension of the tile.
    tile_n: N dimension of the tile.
    tile_k: K dimension of the tile.
    num_stages: Maximum number of concurrent steps in the pipeline.
    epi_tile_n: Epilogue tile size in the N dimension.
    epi_tile_m: Epilogue tile size in the M dimension.
    grid_minor_dim: The minor dimension for the grid tiling.
    grid_tile_width: The width of the grid tile.
    wg_dimension: The dimension along which workgroups are distributed.
    cluster_size_m: The size of the cluster in the M dimension.
    cluster_size_n: The size of the cluster in the N dimension.
  """

  tile_m: pydantic.conint(ge=64, multiple_of=64) = 128
  tile_n: pydantic.conint(ge=64, multiple_of=64) = 128
  tile_k: pydantic.PositiveInt = 128
  num_stages: pydantic.PositiveInt = 4
  epi_tile_n: pydantic.PositiveInt | None = 64
  epi_tile_m: pydantic.PositiveInt | None = 64
  grid_minor_dim: MatmulDimension = MatmulDimension.N
  grid_tile_width: pydantic.PositiveInt = 1
  wg_dimension: MatmulDimension = MatmulDimension.N
  cluster_size_m: pydantic.conint(ge=1, le=2) = 1
  cluster_size_n: pydantic.conint(ge=1, le=2) = 1

  def __post_init__(self):
    if self.cluster_size_m > 1 and self.cluster_size_n > 1:
      raise ValueError(
          "At least one of cluster_size_m or cluster_size_n must be 1."
      )
    if self.cluster_size_m not in (1, 2) or self.cluster_size_n not in (1, 2):
      raise ValueError("Cluster sizes must be 1 or 2.")
    if self.epi_tile_n is not None and self.tile_n % self.epi_tile_n != 0:
      raise ValueError(
          f"{self.tile_n=} must be divisible by {self.epi_tile_n=}"
      )
    if self.epi_tile_m is not None and self.tile_m % self.epi_tile_m != 0:
      raise ValueError(
          f"{self.tile_m=} must be divisible by {self.epi_tile_m=}"
      )
