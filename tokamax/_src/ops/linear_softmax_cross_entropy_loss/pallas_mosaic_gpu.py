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
"""Pallas-Mosaic-GPU Op implementation of linear softmax cross-entropy loss.

Forward pass: SM90 WGMMA + TMA kernel (H100+).
Backward pass: SM90 WGMMA + TMA kernel (H100+) — purely Mosaic GPU, no Triton.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

import jax
import jax.numpy as jnp
from jax.extend import backend
from jaxtyping import Array, Integer, Real
from tokamax._src import gpu_utils
from tokamax._src.ops import op
from tokamax._src.ops.linear_softmax_cross_entropy_loss import base
from tokamax._src.ops.linear_softmax_cross_entropy_loss import (
    pallas_mosaic_gpu_common as common,
)
import tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_gpu_kernel_sm90 as kernel_sm90
from typing_extensions import override


Config = common.Config
Key = common.Key


def _mosaic_vjp(
    residuals: base.Residuals,
    out: jax.Array,
    dout: jax.Array,
    x: jax.Array,
    labels: jax.Array,
    w: jax.Array,
    *,
    reduction: str = "sum",
    return_residuals: bool = False,
):
  """Mosaic GPU backward kernel (purely SM90 WGMMA + TMA, no Triton)."""
  del out, return_residuals
  (lse,) = residuals
  config = common.get_heuristics_config(x, w)
  x_grad, w_grad = kernel_sm90.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_gpu_sm90(
      dout,
      lse,
      x,
      labels,
      w,
      tile_m=config.tile_m,
      tile_n=config.tile_n,
      tile_k=config.tile_k,
      num_stages=config.num_stages,
      reduction=reduction,
  )
  labels_grad = jnp.zeros_like(labels)
  return (x_grad, labels_grad, w_grad)


@dataclass(frozen=True, kw_only=True)
class PallasMosaicGpuLinearSoftmaxCrossEntropyLoss(
    base.LinearSoftmaxCrossEntropyLoss[Config]
):
  """Pallas/Mosaic-GPU SM90 forward + backward for linear softmax CE loss.

  Both forward and backward use WGMMA + TMA pipelining on H100 (SM90).
  No Triton dependency.
  """

  config_cls: ClassVar[type[Config]] = Config

  def __post_init__(self):
    object.__setattr__(self, "vjp", _mosaic_vjp)

  @override
  def _fwd(
      self,
      x: Real[Array, "B H"],
      labels: Integer[Array, "B"],
      w: Real[Array, "H V"],
      *,
      reduction: Literal["sum", "mean"] = "sum",
      config: Config,
      return_residuals: bool,
  ) -> tuple[jax.Array, base.Residuals]:
    device_kind = backend.get_default_device().device_kind.lower()
    if not (gpu_utils.is_sm90() or gpu_utils.is_sm100()):
      raise NotImplementedError(
          f"Mosaic GPU kernel requires SM90 or SM100; got {device_kind!r}."
      )

    loss, lse = kernel_sm90.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_gpu_sm90(
        x,
        labels,
        w,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        tile_k=config.tile_k,
        num_stages=config.num_stages,
        reduction=reduction,
    )
    return loss, (lse,)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return common.get_heuristics_config(ba.arguments["x"], ba.arguments["w"])

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    return common.get_autotuning_configs(ba.arguments["x"], ba.arguments["w"])

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Key:
    return common.get_key(**ba.arguments)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_mosaic_gpu_support(device)
