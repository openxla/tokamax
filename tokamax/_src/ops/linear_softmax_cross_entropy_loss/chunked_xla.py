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

"""Chunked XLA implementation of linear softmax cross-entropy loss."""

import dataclasses
import functools
from typing import Annotated, ClassVar, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Integer
from jaxtyping import Real
import pydantic
from tokamax._src.ops import op
from tokamax._src.ops.linear_softmax_cross_entropy_loss import base


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """The configuration specific for the Chunked XLA kernel.

  Attributes:
    b_block_size: The block size for the batch dimension.
    v_block_size: The block size for the vocabulary dimension.
  """

  b_block_size: Annotated[int, pydantic.Field(ge=128, multiple_of=128)] = 1024
  v_block_size: Annotated[int, pydantic.Field(ge=128, multiple_of=128)] = 2048


@functools.partial(
    jax.jit,
    static_argnames=(
        "reduction",
        "b_block_sz",
        "v_block_sz",
        "unroll_factor",
        "preferred_element_type",
    ),
)
def linear_softmax_cross_entropy_loss_fwd_chunked_xla(
    x: Real[Array, "B H"],
    labels: Integer[Array, "B"],
    w: Real[Array, "H V"],
    *,
    reduction: Literal["sum", "mean"] = "sum",
    b_block_sz: int = 1024,
    v_block_sz: int = 1024,
    unroll_factor: int | bool = False,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> tuple[Real[Array, ""], Real[Array, "B"]]:
  """Chunked XLA implementation of linear softmax cross-entropy loss forward.

  This implementation processes the B and V dimensions in blocks to avoid
  materializing the full BxV logits matrix in HBM.

  Args:
    x: The input activations of shape (B, H).
    labels: The ground truth labels of shape (B,).
    w: The weight matrix of shape (H, V).
    reduction: The reduction method ("sum" or "mean").
    b_block_sz: The block size for the batch dimension.
    v_block_sz: The block size for the vocabulary dimension.
    unroll_factor: The unroll factor for the inner V loop.
    preferred_element_type: The preferred element type for computation.

  Returns:
    A tuple (loss, lse).
  """
  if preferred_element_type is not None:
    preferred_element_type = jnp.dtype(preferred_element_type)

  b_dim, h_dim = x.shape
  _, v_dim = w.shape
  dtype = x.dtype

  b_pad = -b_dim % b_block_sz
  v_pad = -v_dim % v_block_sz

  x_padded = jnp.pad(x, ((0, b_pad), (0, 0)))
  labels_padded = jnp.pad(labels, ((0, b_pad),))
  w_padded = jnp.pad(w, ((0, 0), (0, v_pad)))

  b_dim_padded = b_dim + b_pad
  v_dim_padded = v_dim + v_pad

  num_b_blocks = b_dim_padded // b_block_sz
  num_v_blocks = v_dim_padded // v_block_sz

  def b_loop_body(i, carry):
    lse_all, loss_sum = carry
    b_start = i * b_block_sz

    def v_loop_body(j, v_carry):
      lse_b_, b_loss_sum_neg_logits_ = v_carry
      v_start = j * v_block_sz
      labels_b = jax.lax.dynamic_slice(labels_padded, (b_start,), (b_block_sz,))
      x_b = jax.lax.dynamic_slice(x_padded, (b_start, 0), (b_block_sz, h_dim))
      w_j = jax.lax.dynamic_slice(w_padded, (0, v_start), (h_dim, v_block_sz))
      logits_bv = jnp.dot(
          x_b, w_j, preferred_element_type=preferred_element_type
      )

      lse_b__ = jnp.logaddexp(
          lse_b_, jax.nn.logsumexp(logits_bv.astype(dtype), axis=-1)
      )
      # If labels_b - v_start < 0 or labels_b - v_start >= v_block_sz, then
      # one_hot will be all zeros.
      labels_one_hot = jax.nn.one_hot(
          labels_b - v_start, v_block_sz, dtype=dtype
      )
      b_loss_sum_neg_logits__ = b_loss_sum_neg_logits_ - jnp.sum(
          logits_bv.astype(dtype) * labels_one_hot, axis=-1
      )
      return lse_b__, b_loss_sum_neg_logits__

    lse_b, b_loss_sum_neg_logits = jax.lax.fori_loop(
        0,
        num_v_blocks,
        v_loop_body,
        (
            jnp.full((b_block_sz,), -jnp.inf, dtype=dtype),
            jnp.zeros((b_block_sz,), dtype=dtype),
        ),
        unroll=unroll_factor,
    )

    mask_b = jnp.arange(b_dim_padded) < b_dim
    mask_b = jax.lax.dynamic_slice(mask_b, (b_start,), (b_block_sz,))
    mask_b = mask_b.astype(dtype)
    b_loss_sum_neg_logits = b_loss_sum_neg_logits * mask_b
    lse_b_masked = lse_b * mask_b

    lse_all = jax.lax.dynamic_update_slice(lse_all, lse_b, (b_start,))
    loss_sum += jnp.sum(b_loss_sum_neg_logits) + jnp.sum(lse_b_masked)
    return lse_all, loss_sum

  lse_all, loss_sum = jax.lax.fori_loop(
      0,
      num_b_blocks,
      b_loop_body,
      (jnp.zeros(b_dim_padded, dtype=dtype), jnp.array(0.0, dtype=dtype)),
  )

  lse_all = lse_all[:b_dim]

  if reduction == "mean":
    loss_sum /= b_dim

  return loss_sum, lse_all


@functools.partial(
    jax.jit,
    static_argnames=(
        "b_block_sz",
        "v_block_sz",
        "reduction",
        "preferred_element_type",
    ),
)
def linear_softmax_cross_entropy_loss_bwd_chunked_xla(
    dout: Real[Array, ""],
    lse: Real[Array, "B"],
    x: Real[Array, "B H"],
    labels: Integer[Array, "B"],
    w: Real[Array, "H V"],
    *,
    b_block_sz: int = 1024,
    v_block_sz: int = 2048,
    reduction: Literal["sum", "mean"] = "sum",
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> tuple[Real[Array, "B H"], Real[Array, "H V"]]:
  """The chunked XLA implementation of linear softmax cross-entropy loss backward pass.

  This implementation processes the B and V dimensions in blocks to avoid
  materializing the full BxV logits matrix in HBM.

  Args:
    dout: The gradient of the loss.
    lse: The log-sum-exp from the forward pass of shape (B,).
    x: The input activations of shape (B, H).
    labels: The ground truth labels of shape (B,).
    w: The weight matrix of shape (H, V).
    b_block_sz: The block size for the batch dimension.
    v_block_sz: The block size for the vocabulary dimension.
    reduction: The reduction method ("sum" or "mean").
    preferred_element_type: The preferred element type for computation.

  Returns:
    A tuple (dx, dw).
  """
  if preferred_element_type is not None:
    preferred_element_type = jnp.dtype(preferred_element_type)

  b_dim, h_dim = x.shape
  _, v_dim = w.shape
  dtype = x.dtype

  b_pad = -b_dim % b_block_sz
  v_pad = -v_dim % v_block_sz

  x_padded = jnp.pad(x, ((0, b_pad), (0, 0)))
  labels_padded = jnp.pad(labels, ((0, b_pad),))
  w_padded = jnp.pad(w, ((0, 0), (0, v_pad)))
  lse_padded = jnp.pad(lse, ((0, b_pad),))

  b_dim_padded = b_dim + b_pad
  v_dim_padded = v_dim + v_pad

  num_b_blocks = b_dim_padded // b_block_sz
  num_v_blocks = v_dim_padded // v_block_sz

  def b_loop_body(i, b_carry):
    dx, dw = b_carry
    b_start = i * b_block_sz
    lse_b = jax.lax.dynamic_slice(lse_padded, (b_start,), (b_block_sz,))
    labels_b = jax.lax.dynamic_slice(labels_padded, (b_start,), (b_block_sz,))

    x_b = jax.lax.dynamic_slice(x_padded, (b_start, 0), (b_block_sz, h_dim))

    def v_loop_body(j, v_carry):
      dx_b, dw_acc = v_carry
      v_start = j * v_block_sz

      w_v = jax.lax.dynamic_slice(w_padded, (0, v_start), (h_dim, v_block_sz))
      logits_bv = jnp.dot(
          x_b, w_v, preferred_element_type=preferred_element_type
      )

      labels_one_hot_bv = jax.nn.one_hot(
          labels_b - v_start, v_block_sz, dtype=dtype
      )
      s_bv = jnp.exp(logits_bv - lse_b[:, None]) - labels_one_hot_bv

      dx_b += jnp.dot(
          s_bv, w_v.T, preferred_element_type=preferred_element_type
      )

      dw_v = jax.lax.dynamic_slice(
          dw_acc, (0, v_start), (h_dim, v_block_sz)
      ) + jnp.dot(x_b.T, s_bv, preferred_element_type=preferred_element_type)
      dw_acc = jax.lax.dynamic_update_slice(dw_acc, dw_v, (0, v_start))
      return dx_b, dw_acc

    dx_b_init = jnp.zeros((b_block_sz, h_dim), dtype=preferred_element_type)
    dx_b, dw = jax.lax.fori_loop(0, num_v_blocks, v_loop_body, (dx_b_init, dw))

    dx = jax.lax.dynamic_update_slice(dx, dx_b, (b_start, 0))
    return dx, dw

  dx_init = jnp.zeros((b_dim_padded, h_dim), dtype=preferred_element_type)
  dw_init = jnp.zeros((h_dim, v_dim_padded), dtype=preferred_element_type)
  dx, dw = jax.lax.fori_loop(0, num_b_blocks, b_loop_body, (dx_init, dw_init))

  dx = dx[:b_dim, :]
  dw = dw[:, :v_dim]

  if reduction == "mean":
    dx /= b_dim
    dw /= b_dim

  return (dx * dout).astype(x.dtype), (dw * dout).astype(w.dtype)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ChunkedXlaLinearSoftmaxCrossEntropyLoss(
    base.LinearSoftmaxCrossEntropyLoss[Config]
):
  """Linear Softmax Cross-Entropy Loss Op API using chunked XLA backward."""

  config_cls: ClassVar[type[Config]] = Config

  def __post_init__(self):
    object.__setattr__(
        self,
        "vjp",
        ChunkedXlaLinearSoftmaxCrossEntropyLossVjp(config=self.config),
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
    loss, lse = linear_softmax_cross_entropy_loss_fwd_chunked_xla(
        x,
        labels,
        w,
        reduction=reduction,
        b_block_sz=config.b_block_size,
        v_block_sz=config.v_block_size,
    )
    return loss, (lse,)

  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config()


@dataclasses.dataclass(frozen=True, kw_only=True)
class ChunkedXlaLinearSoftmaxCrossEntropyLossVjp(
    base.LinearSoftmaxCrossEntropyLossVjp[Config]
):
  """Linear Softmax Cross-Entropy Loss chunked XLA VJP wrapper."""

  config_cls: ClassVar[type[Config]] = Config

  def _fwd(
      self,
      residuals: base.Residuals,
      out: jax.Array,
      dout: Real[Array, ""],
      x: Real[Array, "B H"],
      labels: Integer[Array, "B"],
      w: Real[Array, "H V"],
      *,
      reduction: Literal["sum", "mean"] = "sum",
      config: Config,
      return_residuals: bool,
  ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
    """Computes Linear Softmax Cross-Entropy Loss chunked XLA VJP `(dx, dlabels, dw)`."""
    del out

    (lse,) = residuals

    x_grad, w_grad = linear_softmax_cross_entropy_loss_bwd_chunked_xla(
        dout,
        lse,
        x,
        labels,
        w,
        b_block_sz=config.b_block_size,
        v_block_sz=config.v_block_size,
        reduction=reduction,
    )

    labels_grad = jnp.zeros_like(labels)
    return (x_grad, labels_grad, w_grad), None

  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config()
