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
"""Chunked XLA implementation of linear softmax cross-entropy loss."""

import dataclasses
import functools
from typing import Any, ClassVar, Literal, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Integer, Real  # pylint: disable=g-importing-member,g-multiple-import
from tokamax._src.ops import op

# Residuals are (LSE,).
type Residuals = tuple[Real[Array, "B"],]

_DEFAULT_B_BLOCK_SIZE = 1024
_DEFAULT_V_BLOCK_SIZE = 2048


def _chunk_dim(dim: int, block_size: int) -> tuple[int, int]:
  chunk = min(dim, block_size) or 1
  return chunk, -dim % chunk


def _chunk_batch(arr: jax.Array, b_pad: int, b_chunk: int) -> jax.Array:
  pad_width = ((0, b_pad),) + ((0, 0),) * (arr.ndim - 1)
  return jnp.pad(arr, pad_width).reshape(-1, b_chunk, *arr.shape[1:])


def _compute_bwd_scale(
    dout: Real[Array, "*"] | float,
    b_dim: int,
    reduction: Literal["sum", "mean", "none"],
) -> Real[Array, "B"]:
  dout_f32 = jnp.asarray(dout, dtype=jnp.float32)
  match reduction:
    case "none":
      return jnp.broadcast_to(dout_f32, (b_dim,))
    case "mean":
      return jnp.full((b_dim,), dout_f32 / b_dim, dtype=jnp.float32)
    case "sum":
      return jnp.full((b_dim,), dout_f32, dtype=jnp.float32)
    case _:
      raise ValueError(f"Unsupported reduction method: {reduction}")


@functools.partial(
    jax.jit,
    static_argnames=("reduction", "b_block_size", "v_block_size"),
)
def linear_softmax_cross_entropy_loss_fwd_xla(
    x: Real[Array, "B H"],
    labels: Integer[Array, "B"],
    w: Real[Array, "H V"],
    *,
    reduction: Literal["sum", "mean", "none"] = "sum",
    b_block_size: int = _DEFAULT_B_BLOCK_SIZE,
    v_block_size: int = _DEFAULT_V_BLOCK_SIZE,
) -> tuple[Real[Array, "*"], Real[Array, "B"]]:
  """Chunked XLA forward pass for linear softmax cross-entropy loss."""
  if reduction not in ("sum", "mean", "none"):
    raise ValueError(f"Unsupported reduction method: {reduction}")

  b_dim, _ = x.shape
  _, v_dim = w.shape
  b_chunk, b_pad = _chunk_dim(b_dim, b_block_size)
  v_chunk, v_pad = _chunk_dim(v_dim, v_block_size)
  num_v_blocks = (v_dim + v_pad) // v_chunk

  x_blocks = _chunk_batch(x, b_pad, b_chunk)
  labels_blocks = _chunk_batch(labels, b_pad, b_chunk)
  w_padded = jnp.pad(w, ((0, 0), (0, v_pad)))

  def b_step(_, inputs):
    x_b, labels_b = inputs

    def v_step(j, carry):
      lse_b, target_logit_b = carry
      v_start = j * v_chunk
      w_v = jax.lax.dynamic_slice_in_dim(w_padded, v_start, v_chunk, axis=1)
      logits = jnp.dot(x_b, w_v, preferred_element_type=jnp.float32)

      v_indices = v_start + jnp.arange(v_chunk)
      valid_v = v_indices < v_dim
      masked_logits = jnp.where(valid_v, logits, -jnp.inf)
      lse_b = jnp.logaddexp(lse_b, jax.nn.logsumexp(masked_logits, axis=-1))

      is_target = (labels_b[:, None] == v_indices) & valid_v
      target_logit_b += jnp.sum(jnp.where(is_target, logits, 0.0), axis=-1)
      return lse_b, target_logit_b

    init = (
        jnp.full((b_chunk,), -jnp.inf, dtype=jnp.float32),
        jnp.zeros((b_chunk,), dtype=jnp.float32),
    )
    lse_b, target_logit_b = jax.lax.fori_loop(0, num_v_blocks, v_step, init)
    return None, (lse_b, lse_b - target_logit_b)

  _, (lse_blocks, loss_blocks) = jax.lax.scan(
      b_step, None, (x_blocks, labels_blocks)
  )
  lse = lse_blocks.reshape(-1)[:b_dim]
  loss_per_sample = loss_blocks.reshape(-1)[:b_dim]

  if reduction == "sum":
    return jnp.sum(loss_per_sample, dtype=jnp.float32), lse
  if reduction == "mean":
    return jnp.mean(loss_per_sample, dtype=jnp.float32), lse
  return loss_per_sample, lse


@functools.partial(
    jax.jit,
    static_argnames=("reduction", "b_block_size", "v_block_size"),
)
def linear_softmax_cross_entropy_loss_bwd_xla(
    dout: Real[Array, "*"] | float,
    lse: Real[Array, "B"],
    x: Real[Array, "B H"],
    labels: Integer[Array, "B"],
    w: Real[Array, "H V"],
    *,
    reduction: Literal["sum", "mean", "none"] = "sum",
    b_block_size: int = _DEFAULT_B_BLOCK_SIZE,
    v_block_size: int = _DEFAULT_V_BLOCK_SIZE,
) -> tuple[Real[Array, "B H"], Real[Array, "H V"]]:
  """Chunked XLA backward pass for linear softmax cross-entropy loss."""
  b_dim, h_dim = x.shape
  _, v_dim = w.shape
  scale = _compute_bwd_scale(dout, b_dim, reduction)

  b_chunk, b_pad = _chunk_dim(b_dim, b_block_size)
  v_chunk, v_pad = _chunk_dim(v_dim, v_block_size)
  num_v_blocks = (v_dim + v_pad) // v_chunk

  x_blocks = _chunk_batch(x, b_pad, b_chunk)
  labels_blocks = _chunk_batch(labels, b_pad, b_chunk)
  lse_blocks = _chunk_batch(lse, b_pad, b_chunk)
  scale_blocks = _chunk_batch(scale, b_pad, b_chunk)
  w_padded = jnp.pad(w, ((0, 0), (0, v_pad)))

  def b_step(dw, inputs):
    x_b, labels_b, lse_b, scale_b = inputs

    def v_step(j, carry):
      dx_b, dw_acc = carry
      v_start = j * v_chunk
      w_v = jax.lax.dynamic_slice_in_dim(w_padded, v_start, v_chunk, axis=1)
      logits = jnp.dot(x_b, w_v, preferred_element_type=jnp.float32)

      v_indices = v_start + jnp.arange(v_chunk)
      valid_v = v_indices < v_dim
      shifted = jnp.where(valid_v, logits - lse_b[:, None], -jnp.inf)
      is_target = (labels_b[:, None] == v_indices) & valid_v
      s_bv = (jnp.exp(shifted) - is_target.astype(jnp.float32)) * scale_b[
          :, None
      ]

      dx_b += s_bv @ w_v.T
      dw_v = jax.lax.dynamic_slice_in_dim(dw_acc, v_start, v_chunk, axis=1)
      dw_acc = jax.lax.dynamic_update_slice_in_dim(
          dw_acc, dw_v + x_b.T @ s_bv, v_start, axis=1
      )
      return dx_b, dw_acc

    dx_b_init = jnp.zeros((b_chunk, h_dim), dtype=jnp.float32)
    dx_b, dw = jax.lax.fori_loop(0, num_v_blocks, v_step, (dx_b_init, dw))
    return dw, dx_b

  dw_init = jnp.zeros((h_dim, v_dim + v_pad), dtype=jnp.float32)
  dw, dx_blocks = jax.lax.scan(
      b_step, dw_init, (x_blocks, labels_blocks, lse_blocks, scale_blocks)
  )

  dx = dx_blocks.reshape(-1, h_dim)[:b_dim]
  dw = dw[:, :v_dim]
  return dx.astype(x.dtype), dw.astype(w.dtype)


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinearSoftmaxCrossEntropyLoss[C](
    op.Op[Any, jax.Array, Residuals, C, None]
):
  """Linear Softmax Cross-Entropy Loss op using chunked XLA."""

  supports_symbolic_shapes: ClassVar[bool] = False

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(
          self, "vjp", LinearSoftmaxCrossEntropyLossVjp(config=self.config)
      )

  @override
  def _fwd(
      self,
      x: Real[Array, "B H"],
      labels: Integer[Array, "B"],
      w: Real[Array, "H V"],
      *,
      reduction: Literal["sum", "mean", "none"] = "sum",
      config: C,
      return_residuals: bool,
  ) -> tuple[jax.Array, Residuals]:
    del config, return_residuals
    loss, lse = linear_softmax_cross_entropy_loss_fwd_xla(
        x, labels, w, reduction=reduction
    )
    return loss, (lse,)


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinearSoftmaxCrossEntropyLossVjp[C](op.Op[Any, Any, None, C, None]):
  """Linear Softmax Cross-Entropy Loss VJP op using chunked XLA."""

  supports_symbolic_shapes: ClassVar[bool] = False

  @override
  def _fwd(
      self,
      residuals: Residuals,
      out: jax.Array,
      dout: Real[Array, "*"],
      x: Real[Array, "B H"],
      labels: Integer[Array, "B"],
      w: Real[Array, "H V"],
      *,
      reduction: Literal["sum", "mean", "none"] = "sum",
      config: C,
      return_residuals: bool,
  ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
    del out, config, return_residuals
    (lse,) = residuals
    x_grad, w_grad = linear_softmax_cross_entropy_loss_bwd_xla(
        dout,
        lse,
        x,
        labels,
        w,
        reduction=reduction,
    )
    labels_grad = jnp.zeros_like(labels)
    return (x_grad, labels_grad, w_grad), None
