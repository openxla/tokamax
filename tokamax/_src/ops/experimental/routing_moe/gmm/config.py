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

"""Module for static config object."""

import dataclasses
import enum

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import layout_util


class RoutingMode(enum.StrEnum):
  """Instructs how DMA for input and output should be done."""

  DIRECT = enum.auto()  # DMA is performed on continuous memory region.
  LOCAL = enum.auto()  # Local DMA is performed in per-row basis.
  # TODO: Support remove DMA.
  REMOTE = enum.auto()  # Remote DMA is performed in per-row basis.


@dataclasses.dataclass(frozen=True)
class WeightConfig:
  quant_dtype: jnp.dtype | None
  quant_block_size: int | None
  dtype: jnp.dtype
  has_bias: bool = False
  has_scale: bool = False


@dataclasses.dataclass(frozen=True)
class ActivationConfig:
  """Contains how activation should be processed."""

  rows: int
  cols: int
  dtype: jnp.dtype
  quant_dtype: jnp.dtype | None
  quant_block_size: int | None
  mode: RoutingMode

  @property
  def should_quantize(self) -> bool:
    if self.quant_dtype is None:
      return False
    return self.quant_dtype != self.dtype

  @property
  def compact_shape(self) -> tuple[int, int, int, int]:
    return layout_util.to_compact_shape((self.rows, self.cols), self.dtype)


@dataclasses.dataclass(frozen=True)
class Dimension:
  size_m: int
  size_k: int
  size_n: int
  size_out_m: int
  size_out_n: int
  size_local_groups: int
  size_global_groups: int


@dataclasses.dataclass(frozen=True)
class TileConfig:
  """Contains how input and output should be tiled."""

  tile_m: int
  tile_k: int
  tile_n: int
  bucket_base: int
  num_xo_buffers: int = 3
  num_w_buffers: int = 2

  @property
  def bucket_rows(self):
    bucket_base = self.bucket_base
    # Explicitly include tile_m in case tile_m % bucket_base != 0.
    return [*range(bucket_base, self.tile_m, bucket_base), self.tile_m]

  @property
  def tile_m_idx(self):
    # TODO: Fetch this from tpu_info instead of using magic numbers.
    num_lanes = 128
    return pl.cdiv(self.tile_m, num_lanes) * num_lanes

  @property
  def idx_buffer_size(self):
    # idx buffer requires at least one gap between read range and write range
    # to avoid corruption.
    return self.tile_m_idx * max(self.num_xo_buffers, 3)

  @property
  def num_max_buffers(self):
    return max(self.num_xo_buffers, self.num_w_buffers)


@dataclasses.dataclass(frozen=True)
class RoutingGmmConfig:
  """Stores compile time static information of routing gmm."""

  dim: Dimension
  tile: TileConfig
  x: ActivationConfig
  w: WeightConfig
  out: ActivationConfig
  tpu_info: pltpu.TpuInfo
  fuse_act: str | None
  acc_dtype: jnp.dtype

  @classmethod
  def create(
      cls,
      x: jax.Array,
      w: jax.Array,
      w_scale: jax.Array | None,
      group_sizes: jax.Array,
      group_offset: jax.Array | None,
      gather_indices: jax.Array | None,
      scatter_indices: jax.Array | None,
      tile: TileConfig | None,
      fuse_act: str | None,
      acc_dtype: jnp.dtype,
  ):
    """Create routing gmm config object based on arguments."""

    del group_offset

    if tile is not None:
      raise NotImplementedError("Overriding tile config is not supported.")

    size_local_groups, size_k, size_n = w.shape
    size_global_groups = group_sizes.size

    size_m = size_out_m = x.shape[0]

    # Step 1: Initialize static config objects.
    x_mode = out_mode = RoutingMode.DIRECT
    if gather_indices is not None and scatter_indices is not None:
      raise NotImplementedError()
    elif gather_indices is not None:
      x_mode = RoutingMode.LOCAL
      size_out_m = gather_indices.shape[0]
    elif scatter_indices is not None:
      out_mode = RoutingMode.LOCAL
      size_out_m = scatter_indices.shape[0]

    w_quant_block_size = None
    w_quant_dtype = None
    x_quant_dtype = None
    has_scale = False
    if w_scale is not None:
      num_scales = w_scale.shape[1]
      w_quant_block_size = size_k
      w_quant_dtype = w.dtype
      has_scale = True

      # TODO: Support mixed precision.
      x_quant_dtype = w.dtype

    w_cfg = WeightConfig(
        quant_dtype=w_quant_dtype,
        dtype=w.dtype,
        quant_block_size=w_quant_block_size,
        has_scale=has_scale,
    )

    x_cfg = ActivationConfig(
        rows=size_m,
        cols=size_k,
        mode=x_mode,
        quant_dtype=x_quant_dtype,
        dtype=x.dtype,
        # TODO: Choose different block size based on HW.
        quant_block_size=512,
    )

    size_out_n = size_n
    if fuse_act is not None:
      size_out_n //= 2
    out_cfg = ActivationConfig(
        rows=size_out_m,
        cols=size_out_n,
        mode=out_mode,
        dtype=x.dtype,
        quant_dtype=None,
        quant_block_size=None,
    )

    dim = Dimension(
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        size_out_m=size_out_m,
        size_out_n=size_out_n,
        size_local_groups=size_local_groups,
        size_global_groups=size_global_groups,
    )

    tile = calculate_tile_size(x_cfg, w_cfg, out_cfg, dim)

    return cls(
        tpu_info=pltpu.get_tpu_info(),
        fuse_act=fuse_act,
        dim=dim,
        tile=tile,
        x=x_cfg,
        w=w_cfg,
        out=out_cfg,
        acc_dtype=acc_dtype,
    )

  @property
  def max_num_gm(self) -> int:
    """Returns maximum number of possible tiles including OOB safety padding."""
    max_num_single_group = pl.cdiv(self.dim.size_out_m, self.tile.tile_m)
    # Padd left and right by number of buffers to avoid OOB access.
    safety_padding = self.tile.num_max_buffers * 2
    return self.dim.size_local_groups + max_num_single_group + safety_padding

  @property
  def kernel_name(self):
    name = f"routing_gmm-{self.x.mode}-{self.out.mode}"
    name += f"-g{self.dim.size_local_groups}-m{self.dim.size_out_m}"
    name += f"-k{self.dim.size_k}-n{self.dim.size_out_n}"
    return name

  def validate(self, x, w, w_scale, group_sizes):
    """Validate if arguments are correct."""

    # TODO: Maybe perform this while config creation phase?

    if x.shape != self.x.compact_shape:
      raise ValueError(
          f"Input x must have shae of {self.x.compact_shape}. Got {x.shape}."
      )
    w_shape = (self.dim.size_local_groups, self.dim.size_k, self.dim.size_n)
    if w.shape != w_shape:
      raise ValueError(f"Weight shape must be {w_shape}. Got {w.shape}.")

    if group_sizes.size < self.dim.size_local_groups:
      raise ValueError(
          f"group_sizes must be {self.dim.size_local_groups}. Got"
          f" {group_sizes.shape}."
      )
    scale_shape = (self.dim.size_local_groups, 1, 1, self.dim.size_n)
    if w_scale is not None and w_scale.shape != scale_shape:
      raise ValueError(
          f"w_scale must be per channel with shape {scale_shape}. Got"
          f" {w_scale.shape}."
      )

    smem_threshold = int(0.45 * self.tpu_info.smem_capacity_bytes)
    # 3: _to_group_id, _to_m_offset, _to_num_rows. 4: int32.
    smem_allocation = (self.max_num_gm * 3) * 4
    if smem_allocation >= smem_threshold:
      raise ValueError(
          f"Total number of SMEM allocation must be less than {smem_threshold}"
          f" to avoid OOM. Got {smem_allocation}"
      )


def calculate_tile_size(
    x: ActivationConfig, w: WeightConfig, out: ActivationConfig, dim: Dimension
) -> TileConfig:
  """Utilize heuristic to calculate tile size for given inputs."""

  del x, w, out

  # TODO: Add more sophisticated tile selection logic.
  tile_m = min(256, dim.size_out_m)
  bucket_base = min(128, tile_m)

  # TODO: Support N/K dim tiling.
  tile_n = dim.size_out_n
  tile_k = dim.size_k

  return TileConfig(
      tile_m=tile_m,
      tile_k=tile_k,
      tile_n=tile_n,
      bucket_base=bucket_base,
  )
