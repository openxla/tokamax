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
"""Example proof of concept for using Torch TPU kernel with Pallas."""

from collections.abc import Sequence
import inspect
from typing import Any, Literal, cast, override

from absl import logging
import jax
import tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops.linear_softmax_cross_entropy import torch_base
import torch

Config = jax_pallas_mosaic_tpu.Config


class _PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp(
    torch_base._LinearSoftmaxCrossEntropyLossVjp[Config]  # pylint: disable=protected-access
):
  """This is the Pallas Mosaic TPU Tokamax LSCE VJP Op wrapped for PyTorch."""

  def __init__(
      self,
  ) -> None:
    super().__init__()
    self.op_impl_jax = (
        jax_pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp()
    )
    self.jax_op_name = "pallas_mosaic_tpu_linear_softmax_cross_entropy_loss_vjp"
    self.is_vjp = True

  @override
  def __call__(
      self,
      residuals: torch.Tensor,
      out: torch.Tensor,
      dout: torch.Tensor,
      x: torch.Tensor,
      labels: torch.Tensor,
      w: torch.Tensor,
      reduction: str,
      config: Config | tuple[int, ...] | list[int] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    assert (
        self._torch_tokamax_op is not None
    ), "Forward op not registered. This means that self.op_impl_jax is not set"
    " in the constructor."
    return self._torch_tokamax_op(
        residuals,
        out,
        dout,
        x,
        labels,
        w,
        reduction=reduction,
        return_residuals=False,
        config=self.deconstruct_config(config),
    )

  @override
  def op_impl_call(
      self,
      residuals: jax.Array,
      out: jax.Array,
      dout: jax.Array,
      x: jax.Array,
      labels: jax.Array,
      w: jax.Array,
      reduction: str,
      return_residuals: bool,
      config: tuple[int, int, int] | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    reduction = cast(Literal["mean", "none", "sum"], reduction)
    assert self.op_impl_jax is not None, "Forward class not set."
    assert config is not None, "Backward config not set."
    kernel_config = self.reconstruct_config(config)
    output, _ = self.op_impl_jax._fwd(
        (residuals,),
        out,
        dout,
        x,
        labels,
        w,
        reduction=reduction,
        config=kernel_config,
        return_residuals=return_residuals,
    )
    x_grad, _, w_grad = output
    return x_grad, w_grad


class _PallasMosaicTpuLinearSoftmaxCrossEntropyLoss(
    torch_base._LinearSoftmaxCrossEntropyLoss[Config]
):
  """This is the Pallas Mosaic TPU Tokamax LSCE Op wrapped for PyTorch."""

  def __init__(
      self,
  ) -> None:
    super().__init__()
    self.op_impl_jax = (
        jax_pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLoss()
    )
    self.jax_op_name = "pallas_mosaic_tpu_linear_softmax_cross_entropy_loss"
    self.backward_op_torch = _PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp()

  @override
  def __call__(
      self,
      x: torch.Tensor,
      labels: torch.Tensor,
      w: torch.Tensor,
      reduction: str,
      configs: tuple[Config | None, Config | None] | None = None,
  ):
    fwd_config, bwd_config = (None, None) if configs is None else configs
    assert (
        self._torch_tokamax_op is not None
    ), "Forward op not registered. Call register_ops first."
    assert self.backward_op_torch is not None, "Backward op not set."
    loss, lse = self._torch_tokamax_op(
        x,
        labels,
        w,
        reduction=reduction,
        return_residuals=not self.is_vjp,  # Only return residuals for forward pass.
        config=self.deconstruct_config(fwd_config),
        bwd_config=self.backward_op_torch.deconstruct_config(bwd_config),
    )
    return loss, lse

  @override
  def op_impl_call(
      self,
      x: jax.Array,
      labels: jax.Array,
      w: jax.Array,
      reduction: str,
      return_residuals: bool,
      config: tuple[int, int, int] | None = None,
      bwd_config: tuple[int, int, int] | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    del bwd_config
    reduction = cast(Literal["mean", "none", "sum"], reduction)
    assert self.op_impl_jax is not None, "Forward class not set."
    assert config is not None, "Forward config not set."
    kernel_config = self.reconstruct_config(config)
    loss, (lse,) = self.op_impl_jax._fwd(
        x,
        labels,
        w,
        reduction=reduction,
        return_residuals=return_residuals,
        config=kernel_config,
    )
    return loss, lse

  @override
  def setup_context(
      self,
      ctx: Any,
      inputs: Sequence[Any],
      output: tuple[torch.Tensor, torch.Tensor],
  ) -> None:
    """Callback for torch register_autograd."""
    (
        x,
        labels,
        w,
        reduction,
        _,
        _,
        bwd_config,
    ) = inputs
    loss, lse = output
    ctx.save_for_backward(x, labels, w, lse, loss)
    ctx.reduction = reduction
    ctx.bwd_config = bwd_config

  @override
  def backward(
      self,
      ctx: Any,
      d_loss: torch.Tensor,
      d_lse: torch.Tensor,
  ) -> tuple[torch.Tensor, None, torch.Tensor, None, None, None, None]:
    """Callback for torch register_autograd."""
    del d_lse  # Unused
    x, labels, w, lse, loss = ctx.saved_tensors
    assert self.backward_op_torch is not None, "Backward op not set."
    grad_x, grad_w = self.backward_op_torch(
        lse,
        loss,
        d_loss,
        x,
        labels,
        w,
        reduction=ctx.reduction,
        config=ctx.bwd_config,
    )
    return grad_x, None, grad_w, None, None, None, None


# Singleton instance of the Pallas Mosaic TPU Linear Softmax Cross-Entropy Loss.
PallasMosaicTpuLinearSoftmaxCrossEntropyLoss = (
    _PallasMosaicTpuLinearSoftmaxCrossEntropyLoss()
)
