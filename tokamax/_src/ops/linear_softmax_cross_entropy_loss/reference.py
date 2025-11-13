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

"""Linear Softmax Cross-Entropy Loss reference implementation."""
from functools import partial
from typing import Literal
import jax
import jax.numpy as jnp
from jaxtyping import Array, Integer, Real, Scalar


@partial(jax.jit, static_argnames=["reduction"])
def linear_softmax_cross_entropy_loss_fwd_reference(
    x: Real[Array, "B H"],
    labels: Integer[Array, "B"],
    w: Real[Array, "H V"],
    *,
    reduction: Literal["sum", "mean"] = "sum",
) -> Real[Scalar, ""]:
  """The reference Jax implementation of the linear softmax cross-entropy loss."""
  logits = x @ w
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  labels_one_hot = jax.nn.one_hot(labels, num_classes=w.shape[1], dtype=x.dtype)
  loss = -labels_one_hot * log_probs
  lse = jax.nn.logsumexp(logits, axis=-1)

  if reduction == "sum":
    return jnp.sum(loss), lse
  elif reduction == "mean":
    num_tokens = x.shape[0]
    mean_loss = jnp.sum(loss) / num_tokens
    return mean_loss, lse
  else:
    raise ValueError(f"Unsupported reduction method: {reduction}")


@partial(jax.jit, static_argnames=["reduction"])
def linear_softmax_cross_entropy_loss_bwd_reference(
    dout: Real[Array, ""],
    lse: Real[Array, "B"],
    x: Real[Array, "B H"],
    labels: Integer[Array, "B"],
    w: Real[Array, "H V"],
    *,
    reduction: Literal["sum", "mean"] = "sum",
) -> tuple[Real[Array, "B H"], Real[Array, "B V"]]:
  """The reference Jax implementation of the linear softmax cross-entropy loss backward kernel."""
  labels_one_hot = jax.nn.one_hot(labels, num_classes=w.shape[1], dtype=x.dtype)
  s = -labels_one_hot + jnp.exp(x @ w - lse[:, None])
  x_grad = s @ w.T
  w_grad = x.T @ s
  num_tokens = x.shape[0]

  if reduction == "mean":
    x_grad /= num_tokens
    w_grad /= num_tokens

  return dout * x_grad, dout * w_grad
