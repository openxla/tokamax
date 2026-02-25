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

from dataclasses import dataclass
import itertools
from typing import Annotated, ClassVar, Literal
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Integer, Real
import pydantic
from tokamax._src.ops import op
from tokamax._src.ops.linear_softmax_cross_entropy_loss import base
import tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_tpu_kernel as kernel
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """The configuration specific for the Pallas Mosaic TPU kernel.

  Attributes:
    b_block_size: The block size for the batch dimension.
    h_block_size: The block size for the hidden dimension.
    v_block_size: The block size for the vocabulary dimension.
  """

  b_block_size: Annotated[int, pydantic.Field(ge=1024, multiple_of=128)] = 1024
  h_block_size: Annotated[int, pydantic.Field(ge=128, multiple_of=128)] = 512
  v_block_size: Annotated[int, pydantic.Field(ge=128, multiple_of=128)] = 2048


def get_tpu_specific_default_config(b_dim: int, h_dim: int, v_dim: int) -> Config:
  """Returns the heuristic config for based on TPU version."""

  if pltpu.get_tpu_info().generation >= 7:
    return Config(b_block_size=1024, h_block_size=512, v_block_size=2048)
  elif pltpu.get_tpu_info().generation == 6:
    # If H dimens is not 512 aligned, it needs additional mask / padding causing
    # vmem spill over, which may OOM VMEM on tpu gen 6.
    h_block_size_512_aligned = h_dim % 512 == 0
    if h_block_size_512_aligned:
      return Config(b_block_size=1024, h_block_size=512, v_block_size=2048)
    else:
      return Config(b_block_size=1024, h_block_size=512, v_block_size=1024)
  else:
    return Config(b_block_size=1024, h_block_size=512, v_block_size=512)


@dataclass(frozen=True, kw_only=True)
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
      reduction: Literal["sum", "mean"] = "sum",
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

    return get_tpu_specific_default_config(b_dim, h_dim, v_dim)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    x = ba.arguments["x"]
    w = ba.arguments["w"]

    b_dim, h_dim = x.shape
    v_dim = w.shape[1]

    b_block_size_range = set(
        [1024 + i * 128 for i in range(32) if b_dim % (1024 + i * 128) == 0]
        + [1024 + i * 512 for i in range(8) if 1024 + i * 512 < b_dim]
    )
    h_block_size_range = set(
        [128 + i * 128 for i in range(32) if h_dim % (128 + i * 128) == 0]
        + [128 * 2**i for i in range(6) if 128 * 2**i < h_dim]
    )
    v_block_size_range = set(
        [128 + i * 128 for i in range(32) if v_dim % (128 + i * 128) == 0]
        + [128 * 2**i for i in range(6) if 128 * 2**i < v_dim]
    )

    return set(
        Config(
            b_block_size=b_block_size,
            h_block_size=h_block_size,
            v_block_size=v_block_size,
        )
        for b_block_size, h_block_size, v_block_size in itertools.product(
            b_block_size_range, h_block_size_range, v_block_size_range
        )
    )

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5


@dataclass(frozen=True, kw_only=True)
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
      reduction: Literal["sum", "mean"] = "sum",
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

    return get_tpu_specific_default_config(b_dim, h_dim, v_dim)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    x = ba.arguments["x"]
    w = ba.arguments["w"]

    b_dim, h_dim = x.shape
    v_dim = w.shape[1]

    b_block_size_range = set(
        [1024 + i * 128 for i in range(32) if b_dim % (1024 + i * 128) == 0]
        + [1024 + i * 512 for i in range(8) if 1024 + i * 512 < b_dim]
    )
    h_block_size_range = set(
        [128 + i * 128 for i in range(32) if h_dim % (128 + i * 128) == 0]
        + [128 * 2**i for i in range(6) if 128 * 2**i < h_dim]
    )
    v_block_size_range = set(
        [128 + i * 128 for i in range(32) if v_dim % (128 + i * 128) == 0]
        + [128 * 2**i for i in range(6) if 128 * 2**i < v_dim]
    )

    return set(
        Config(
            b_block_size=b_block_size,
            h_block_size=h_block_size,
            v_block_size=v_block_size,
        )
        for b_block_size, h_block_size, v_block_size in itertools.product(
            b_block_size_range, h_block_size_range, v_block_size_range
        )
    )

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
