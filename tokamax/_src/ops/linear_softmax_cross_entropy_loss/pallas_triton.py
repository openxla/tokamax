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
"""Pallas-Triton Op implementation of linear softmax cross-entropy loss."""

from dataclasses import dataclass
from typing import ClassVar, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Integer, Real
from tokamax._src import gpu_utils
from tokamax._src.ops import op
from tokamax._src.ops.linear_softmax_cross_entropy_loss import base
from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_triton_config
import tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_triton_kernel as kernel
from typing_extensions import override


Config = pallas_triton_config.Config
Key = pallas_triton_config.Key


@dataclass(frozen=True, kw_only=True)
class PallasTritonLinearSoftmaxCrossEntropyLoss(
    base.LinearSoftmaxCrossEntropyLoss[Config]
):
  """Pallas/Triton GPU implementation of linear softmax cross-entropy loss."""

  config_cls: ClassVar[type[Config]] = Config

  def __post_init__(self):
    object.__setattr__(
        self,
        "vjp",
        PallasTritonLinearSoftmaxCrossEntropyLossVjp(config=self.config),
    )

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
    loss, lse = kernel.linear_softmax_cross_entropy_loss_fwd_pallas_triton(
        x,
        labels,
        w,
        b_block_size=config.b_block_size,
        h_block_size=config.h_block_size,
        v_block_size=config.v_block_size,
        reduction=reduction,
        num_warps=config.num_warps,
    )
    return loss, (lse,)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    x = ba.arguments["x"]
    w = ba.arguments["w"]
    return pallas_triton_config.get_heuristics_config(x, w)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    x = ba.arguments["x"]
    w = ba.arguments["w"]
    return pallas_triton_config.get_autotuning_configs(x, w)

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Key:
    return pallas_triton_config.get_key(**ba.arguments)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)


@dataclass(frozen=True, kw_only=True)
class PallasTritonLinearSoftmaxCrossEntropyLossVjp(
    base.LinearSoftmaxCrossEntropyLossVjp[Config]
):
  """Pallas/Triton GPU VJP for linear softmax cross-entropy loss."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  def _fwd(
      self,
      residuals: base.Residuals,
      out: Real[Array, ""],
      dout: Real[Array, ""],
      x: Real[Array, "B H"],
      labels: Integer[Array, "B"],
      w: Real[Array, "H V"],
      *,
      reduction: Literal["sum", "mean"] = "sum",
      config: Config,
      return_residuals: bool,
  ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
    del out
    (lse,) = residuals

    x_grad, w_grad = kernel.linear_softmax_cross_entropy_loss_bwd_pallas_triton(
        dout,
        lse,
        x,
        labels,
        w,
        b_block_size=config.b_block_size,
        h_block_size=config.h_block_size,
        v_block_size=config.v_block_size,
        reduction=reduction,
        num_warps=config.num_warps,
    )
    labels_grad = jnp.zeros_like(labels)
    return (x_grad, labels_grad, w_grad), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    x = ba.arguments["x"]
    w = ba.arguments["w"]
    return pallas_triton_config.get_heuristics_config(x, w)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    x = ba.arguments["x"]
    w = ba.arguments["w"]
    return pallas_triton_config.get_autotuning_configs(x, w)

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Key:
    x = ba.arguments["x"]
    labels = ba.arguments["labels"]
    w = ba.arguments["w"]
    reduction = ba.arguments["reduction"]
    return pallas_triton_config.get_key(x, labels, w, reduction=reduction)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)
