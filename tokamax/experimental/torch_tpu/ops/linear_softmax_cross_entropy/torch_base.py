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
"""Linear Softmax Cross-Entropy Loss PyTorch Op API using reference impl."""

from collections.abc import Sequence
from typing import Any, Literal, cast
import jax
from jax import numpy as jnp
from tokamax._src.ops.linear_softmax_cross_entropy_loss import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops import torch_utils
import torch
from typing_extensions import override


class _LinearSoftmaxCrossEntropyLossVjp[Config](torch_op.TorchOp[None]):
  """Linear Softmax Cross-Entropy Loss PyTorch Op VJP API using reference impl."""

  def __init__(self):
    super().__init__()
    self.op_impl_jax = jax_base.LinearSoftmaxCrossEntropyLossVjp()
    self.jax_op_name = "base_linear_softmax_cross_entropy_loss_vjp"
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
      config: Config | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    self.configs = (None, config)
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
  ) -> tuple[jax.Array, jax.Array]:
    reduction = cast(Literal["mean", "none", "sum"], reduction)
    assert self.op_impl_jax is not None, "Forward class not set."
    (x_grad, _, w_grad), _ = self.op_impl_jax._fwd(
        (residuals,),
        out,
        dout,
        x,
        labels,
        w,
        reduction=reduction,
        return_residuals=return_residuals,
        config=self.configs[1],
    )
    return x_grad, w_grad

  @override
  def derive_backward_shapes(
      self, abstract_argument_dict: dict[str, Any]
  ) -> dict[str, Any]:
    """Derives the backward shapes for the backward pass.

    Args:
      abstract_argument_dict: The abstract argument dict of the forward pass. If
        abstract_argument_dict is empty, we return the keys and () for the
        shapes.

    Returns:
      A dict of the backward shapes for the backward pass.
    """
    if not abstract_argument_dict:
      return {
          "residuals": (jax.ShapeDtypeStruct(shape=(), dtype=jnp.bfloat16),),
          "out": jax.ShapeDtypeStruct(shape=(), dtype=jnp.bfloat16),
          "dout": jax.ShapeDtypeStruct(
              shape=(),
              dtype=jnp.bfloat16,
          ),
      }

    # Empty tensors for the backward pass
    residual_shape = (abstract_argument_dict["x"].shape[0],)
    x_type = abstract_argument_dict["x"].dtype
    return {
        "residuals": (
            jax.ShapeDtypeStruct(shape=residual_shape, dtype=x_type),
        ),
        "out": jax.ShapeDtypeStruct(shape=(), dtype=x_type),
        "dout": jax.ShapeDtypeStruct(
            shape=(),
            dtype=x_type,
        ),
    }


class _LinearSoftmaxCrossEntropyLoss[Config](torch_op.TorchOp[Config]):
  """Linear Softmax Cross-Entropy Loss PyTorch Op API using reference impl."""

  def __init__(self):
    super().__init__()
    self.op_impl_jax = jax_base.LinearSoftmaxCrossEntropyLoss()
    self.backward_op_torch = _LinearSoftmaxCrossEntropyLossVjp()
    self.jax_op_name = "base_linear_softmax_cross_entropy_loss"

  @override
  def __call__(
      self,
      x: torch.Tensor,
      labels: torch.Tensor,
      w: torch.Tensor,
      reduction: str,
      configs: tuple[Any, Any] | None = None,
  ):
    if configs is None:
      self.configs = torch_utils.get_configs(
          self,
          x,
          labels,
          w,
          reduction=reduction,
          from_autotuning_cache=False,
      )
    else:
      self.configs = configs
    assert (
        self._torch_tokamax_op is not None
    ), "Forward op not registered. This means that self.op_impl_jax is not set"
    " in the constructor."
    loss, lse = self._torch_tokamax_op(
        x,
        labels,
        w,
        reduction=reduction,
        return_residuals=True,
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
  ) -> tuple[jax.Array, jax.Array]:
    reduction = cast(Literal["mean", "none", "sum"], reduction)
    assert self.op_impl_jax is not None, "Forward class not set."
    loss, (lse,) = self.op_impl_jax._fwd(
        x,
        labels,
        w,
        reduction=reduction,
        return_residuals=return_residuals,
        config=self.configs[0],
    )
    return loss, lse

  @override
  def setup_context(
      self,
      ctx: Any,
      inputs: Sequence[Any],
      output: Any,
  ) -> None:
    """Saves tensors to the context for the backward pass."""
    x, labels, w, reduction, _ = inputs
    loss, lse = output
    ctx.save_for_backward(x, labels, w, lse, loss)
    ctx.reduction = reduction

  @override
  def backward(
      self,
      ctx: Any,
      d_loss: torch.Tensor,
      d_lse: torch.Tensor,
  ) -> tuple[torch.Tensor, None, torch.Tensor, None, None]:
    """Callback for torch register_autograd."""
    del d_lse  # Unused
    assert self.backward_op_torch is not None, "Backward op not set."
    x, labels, w, lse, loss = ctx.saved_tensors

    grad_x, grad_w = self.backward_op_torch(
        lse,
        loss,
        d_loss,
        x,
        labels,
        w,
        reduction=ctx.reduction,
        config=self.configs[1],
    )
    return grad_x, None, grad_w, None, None


# Singleton instance of the LinearSoftmaxCrossEntropyLoss.
LinearSoftmaxCrossEntropyLoss = _LinearSoftmaxCrossEntropyLoss[Any]()
