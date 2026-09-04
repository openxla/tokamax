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

"""The Linear Softmax Cross-Entropy Loss PyTorch Op API."""

import jax
from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_tpu_kernel as tokamax_kernel
import torch
import torch_tpu._internal.pallas.pallas


def linear_softmax_cross_entropy_loss_jax(
    x: jax.Array,
    labels: jax.Array,
    weights: jax.Array,
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
    reduction: str,
    preferred_element_type: str,
) -> tuple[jax.Array, jax.Array]:
  """Wrapper for tokamax kernel with type hints for torch_tpu's jax_op."""
  dtype = (
      jax.numpy.float32
      if preferred_element_type == "float32"
      else jax.numpy.bfloat16
  )
  return tokamax_kernel.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
      x,
      labels,
      weights,
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
      reduction=reduction,
      preferred_element_type=dtype,
  )


def linear_softmax_cross_entropy_loss_backward_jax(
    dout: jax.Array,
    lse: jax.Array,
    x: jax.Array,
    labels: jax.Array,
    weights: jax.Array,
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
    reduction: str,
    preferred_element_type: str,
) -> tuple[jax.Array, jax.Array]:
  """Wrapper for tokamax kernel with type hints for torch_tpu...jaxop."""
  dtype = (
      jax.numpy.float32
      if preferred_element_type == "float32"
      else jax.numpy.bfloat16
  )
  return tokamax_kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
      dout,
      lse,
      x,
      labels,
      weights,
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
      reduction=reduction,
      preferred_element_type=dtype,
  )


# pylint: disable=protected-access
linear_softmax_cross_entropy_loss: torch._library.custom_ops.CustomOpDef = (
    torch_tpu._internal.pallas.pallas.jax_op(
        "tokamax::linear_softmax_cross_entropy_loss",
        linear_softmax_cross_entropy_loss_jax,
    )
)

# pylint: disable=protected-access
linear_softmax_cross_entropy_loss_backward: (
    torch._library.custom_ops.CustomOpDef
) = torch_tpu._internal.pallas.pallas.jax_op(
    "tokamax::linear_softmax_cross_entropy_loss_backward",
    linear_softmax_cross_entropy_loss_backward_jax,
)


def setup_context(ctx, inputs, output):
  """Callback for torch register_autograd."""
  (
      x,
      labels,
      weights,
      b_block_size,
      h_block_size,
      v_block_size,
      reduction,
      preferred_element_type,
  ) = inputs
  loss, lse = output
  del loss  # Unused
  ctx.save_for_backward(x, labels, weights, lse)
  ctx.b_block_size = b_block_size
  ctx.h_block_size = h_block_size
  ctx.v_block_size = v_block_size
  ctx.reduction = reduction
  ctx.preferred_element_type = preferred_element_type


def backward(ctx, d_loss, d_lse):
  """Callback for torch register_autograd."""
  del d_lse  # Unused
  x, labels, weights, lse = ctx.saved_tensors
  grad_x, grad_w = linear_softmax_cross_entropy_loss_backward(
      d_loss,
      lse,
      x,
      labels,
      weights,
      ctx.b_block_size,
      ctx.h_block_size,
      ctx.v_block_size,
      ctx.reduction,
      ctx.preferred_element_type,
  )
  return grad_x, None, grad_w, None, None, None, None, None


linear_softmax_cross_entropy_loss.register_autograd(
    backward, setup_context=setup_context
)
