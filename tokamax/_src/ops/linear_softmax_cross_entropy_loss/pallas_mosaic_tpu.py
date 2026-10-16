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

"""Linear Cross-Entropy kernel implementation."""

import dataclasses
import itertools
import math
from typing import ClassVar, Literal, override

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import jaxtyping as jt
from tokamax._src.ops import op
from tokamax._src.ops.linear_softmax_cross_entropy_loss import base
import tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_tpu_kernel as kernel

Config = kernel.Config
Array = jt.Array
Integer = jt.Integer
Real = jt.Real


def _lane_aligned_range(dim: int, num_lanes: int) -> set[int]:
  """Returns non-empty lane-aligned block-size candidates for `dim`."""
  divisors = [
      num_lanes + i * num_lanes
      for i in range(32)
      if dim % (num_lanes + i * num_lanes) == 0
  ]
  powers_of_two = [num_lanes * 2**i for i in range(6) if num_lanes * 2**i < dim]
  return set(divisors + powers_of_two) or {num_lanes}


def _get_autotuning_block_size_ranges(
    b_dim: int, h_dim: int, v_dim: int
) -> tuple[set[int], set[int], set[int]]:
  """Returns non-empty lane-aligned (b, h, v) block-size candidate ranges.

  All candidate block sizes are multiples of the TPU lane count. When a
  dimension is smaller than the minimum block size (so no lane-aligned divisor
  or smaller power-of-two multiple exists), its range falls back to the
  minimum valid block size (`min_b_block_size` for `b_dim`, `num_lanes` for
  `h_dim` and `v_dim`) so that autotuning still searches across the remaining
  dimensions.

  Args:
    b_dim: Batch dimension size.
    h_dim: Hidden dimension size.
    v_dim: Vocabulary dimension size.
  """
  tpu_info = pltpu.get_tpu_info()
  num_lanes = tpu_info.num_lanes
  min_b_block_size = tpu_info.num_sublanes * num_lanes

  b_block_size_range = set(
      [
          min_b_block_size + i * num_lanes
          for i in range(32)
          if b_dim % (min_b_block_size + i * num_lanes) == 0
      ]
      + [
          min_b_block_size + i * 4 * num_lanes
          for i in range(8)
          if min_b_block_size + i * 4 * num_lanes < b_dim
      ]
  ) or {min_b_block_size}
  h_block_size_range = _lane_aligned_range(h_dim, num_lanes)
  v_block_size_range = _lane_aligned_range(v_dim, num_lanes)

  return b_block_size_range, h_block_size_range, v_block_size_range


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicTpuLinearSoftmaxCrossEntropyLoss(
    base.LinearSoftmaxCrossEntropyLoss[Config]
):
  """Wrapper for the tokamax Op API for Pallas Mosaic TPU kernel."""

  config_cls: ClassVar[type[Config]] = Config

  def __post_init__(self):
    object.__setattr__(
        self,
        "vjp",
        PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp(config=self.config),
    )

  def _fwd(
      self,
      x: Real[Array, "B H"],
      labels: Integer[Array, "B"],
      w: Real[Array, "H V"],
      *,
      reduction: Literal["sum", "mean", "none"] = "sum",
      config: Config,
      return_residuals: bool,
  ) -> tuple[jax.Array, base.Residuals]:
    loss, lse = kernel.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
        x,
        labels,
        w,
        b_block_size=config.b_block_size,
        h_block_size=config.h_block_size,
        v_block_size=config.v_block_size,
        reduction=reduction,
    )
    return loss, (lse,)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    x = ba.arguments["x"]
    w = ba.arguments["w"]

    b_dim, h_dim = x.shape
    v_dim = w.shape[1]

    return kernel.get_heuristic_fwd_config(b_dim, h_dim, v_dim, dtype=x.dtype)

  @staticmethod
  def get_heuristic_fwd_config(
      b_dim: int,
      h_dim: int,
      v_dim: int,
      dtype: jnp.dtype = jnp.float32,  # pyrefly: ignore[bad-function-definition]
      vmem_limit_bytes: int | None = None,
  ) -> Config:
    return kernel.get_heuristic_fwd_config(
        b_dim, h_dim, v_dim, dtype=dtype, vmem_limit_bytes=vmem_limit_bytes
    )

  @staticmethod
  def get_heuristic_bwd_config(
      b_dim: int,
      h_dim: int,
      v_dim: int,
      dtype: jnp.dtype = jnp.float32,  # pyrefly: ignore[bad-function-definition]
      vmem_limit_bytes: int | None = None,
      buffer_count: tuple[int, int] | int = (2, 2),
  ) -> Config:
    return kernel.get_heuristic_bwd_config(
        b_dim,
        h_dim,
        v_dim,
        dtype=dtype,
        vmem_limit_bytes=vmem_limit_bytes,
        buffer_count=buffer_count,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    x = ba.arguments["x"]
    w = ba.arguments["w"]

    b_dim, h_dim = x.shape
    v_dim = w.shape[1]

    b_block_size_range, h_block_size_range, v_block_size_range = (
        _get_autotuning_block_size_ranges(b_dim, h_dim, v_dim)
    )

    vmem_limit_bytes = kernel._get_vmem_limit_bytes()
    max_cores = pltpu.get_tpu_info().num_cores

    configs = set()
    for b_block_size, h_block_size, v_block_size in itertools.product(
        b_block_size_range, h_block_size_range, v_block_size_range
    ):
      num_v_blocks = math.ceil(v_dim / v_block_size)
      if (
          max_cores > 1
          and num_v_blocks >= max_cores
          and num_v_blocks % max_cores != 0
      ):
        continue
      if (
          kernel._calculate_fwd_vmem_bytes(
              b_block_size, h_block_size, v_block_size, dtype=x.dtype
          )
          > vmem_limit_bytes
      ):
        continue
      configs.add(
          Config(
              b_block_size=b_block_size,
              h_block_size=h_block_size,
              v_block_size=v_block_size,
          )
      )

    if not configs:
      configs.add(
          kernel.get_heuristic_fwd_config(
              b_dim,
              h_dim,
              v_dim,
              dtype=x.dtype,
              vmem_limit_bytes=vmem_limit_bytes,
          )
      )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp(
    base.LinearSoftmaxCrossEntropyLossVjp[Config]
):
  """Pallas TPU implementation of Linear Softmax Cross-Entropy Loss VJP."""

  config_cls: ClassVar[type[Config]] = Config

  def _fwd(
      self,
      residuals: base.Residuals,
      out: Real[Array, ""],
      dout: Real[Array, ""],
      x: Real[Array, "B H"],
      labels: Integer[Array, "B"],
      w: Real[Array, "H V"],
      *,
      reduction: Literal["sum", "mean", "none"] = "sum",
      config: Config,
      return_residuals: bool,
  ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
    """Computes Linear Softmax Cross-Entropy Loss VJP `(dx, dlabels, dw)`."""
    (lse,) = residuals

    x_grad, w_grad = (
        kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
            dout,
            lse,
            x,
            labels,
            w,
            b_block_size=config.b_block_size,
            h_block_size=config.h_block_size,
            v_block_size=config.v_block_size,
            reduction=reduction,
            buffer_count=config.buffer_count,
        )
    )
    labels_grad = jnp.zeros_like(labels)
    return (x_grad, labels_grad, w_grad), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    x = ba.arguments["x"]
    w = ba.arguments["w"]

    b_dim, h_dim = x.shape
    v_dim = w.shape[1]

    return kernel.get_heuristic_bwd_config(b_dim, h_dim, v_dim, dtype=x.dtype)

  @staticmethod
  def get_heuristic_bwd_config(
      b_dim: int,
      h_dim: int,
      v_dim: int,
      dtype: jnp.dtype = jnp.float32,  # pyrefly: ignore[bad-function-definition]
      vmem_limit_bytes: int | None = None,
      buffer_count: tuple[int, int] | int = (2, 2),
  ) -> Config:
    return kernel.get_heuristic_bwd_config(
        b_dim,
        h_dim,
        v_dim,
        dtype=dtype,
        vmem_limit_bytes=vmem_limit_bytes,
        buffer_count=buffer_count,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    x = ba.arguments["x"]
    w = ba.arguments["w"]

    b_dim, h_dim = x.shape
    v_dim = w.shape[1]

    b_block_size_range, h_block_size_range, v_block_size_range = (
        _get_autotuning_block_size_ranges(b_dim, h_dim, v_dim)
    )

    buffer_count_range = [(2, 2), (2, 3), (3, 2), (3, 3)]

    vmem_limit_bytes = kernel._get_vmem_limit_bytes()
    max_cores = pltpu.get_tpu_info().num_cores

    configs = set()
    for (
        b_block_size,
        h_block_size,
        v_block_size,
        buffer_count,
    ) in itertools.product(
        b_block_size_range,
        h_block_size_range,
        v_block_size_range,
        buffer_count_range,
    ):
      num_v_blocks = math.ceil(v_dim / v_block_size)
      if (
          max_cores > 1
          and num_v_blocks >= max_cores
          and num_v_blocks % max_cores != 0
      ):
        continue
      if math.ceil(h_dim / h_block_size) == 2:
        continue
      if (
          kernel._calculate_bwd_vmem_bytes(
              b_block_size,
              h_block_size,
              v_block_size,
              dtype=x.dtype,
              buffer_count=buffer_count,
          )
          > vmem_limit_bytes
      ):
        continue
      configs.add(
          Config(
              b_block_size=b_block_size,
              h_block_size=h_block_size,
              v_block_size=v_block_size,
              buffer_count=buffer_count,
          )
      )

    if not configs:
      configs.add(
          kernel.get_heuristic_bwd_config(
              b_dim,
              h_dim,
              v_dim,
              dtype=x.dtype,
              vmem_limit_bytes=vmem_limit_bytes,
              buffer_count=(2, 2),
          )
      )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
