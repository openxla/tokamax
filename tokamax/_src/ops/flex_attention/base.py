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
"""Base for FlexAttention ops."""

from collections.abc import Callable
import functools
from typing import Any, Literal, TypeVar, overload

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src import precision as precision_lib
from tokamax._src import quantization
from tokamax._src.ops import op


QuantizedArray = quantization.QuantizedArray


_Config = TypeVar("_Config")
_Key = TypeVar("_Key")
# The attention residuals come from the softmax calculation:
# `(maximum softmax input values, softmax denominator)`.
Residuals = tuple[Float[Array, "*B H T"], Float[Array, "*B H T"]]
ScoreMod = Callable[[Float[Array, "*B H T t"]], Float[Array, "*B H T t"]]
MaskMod = Callable[[tuple[int, ...]], Bool[Array, "*#B #H #T #t"]]


class FlexAttention(
    op.Op[Any, Float[Array, "*B T H d"], Residuals, _Config, _Key]
):
  """FlexAttention function."""

  # We override `__call__` in order to handle sharding at the top level.
  @overload
  def __call__(
      self,
      q: Float[Array | QuantizedArray, "*B T H D"],
      k: Float[Array | QuantizedArray, "*B t h D"],
      v: Float[Array | QuantizedArray, "*B t h d"],
      *,
      precision: (
          jax.lax.PrecisionLike
          | tuple[jax.lax.PrecisionLike, jax.lax.PrecisionLike]
      ) = ...,
      score_mod: ScoreMod | None = ...,
      mask_mod: MaskMod | None = ...,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None = ...,
      dropout_rate: float = ...,
      q_sharding: jax.sharding.NamedSharding | None = ...,
      k_sharding: jax.sharding.NamedSharding | None = ...,
      normalize_output: bool = ...,
      return_residuals: Literal[False] = ...,
  ) -> Float[Array, "*B T H d"]:
    ...

  @overload
  def __call__(
      self,
      q: Float[Array | QuantizedArray, "*B T H D"],
      k: Float[Array | QuantizedArray, "*B t h D"],
      v: Float[Array | QuantizedArray, "*B t h d"],
      *,
      precision: (
          jax.lax.PrecisionLike
          | tuple[jax.lax.PrecisionLike, jax.lax.PrecisionLike]
      ) = ...,
      score_mod: ScoreMod | None = ...,
      mask_mod: MaskMod | None = ...,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None = ...,
      dropout_rate: float = ...,
      q_sharding: jax.sharding.NamedSharding | None = ...,
      k_sharding: jax.sharding.NamedSharding | None = ...,
      normalize_output: bool = ...,
      return_residuals: Literal[True] = ...,
  ) -> tuple[Float[Array, "*B T H d"], Residuals]:
    ...

  @jaxtyping.jaxtyped
  def __call__(
      self,
      q: Float[Array | QuantizedArray, "*B T H D"],
      k: Float[Array | QuantizedArray, "*B t h D"],
      v: Float[Array | QuantizedArray, "*B t h d"],
      *,
      precision: (
          jax.lax.PrecisionLike
          | tuple[jax.lax.PrecisionLike, jax.lax.PrecisionLike]
      ) = None,
      score_mod: ScoreMod | None = None,
      mask_mod: MaskMod | None = None,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None = None,
      dropout_rate: float = 0.0,
      q_sharding: jax.sharding.NamedSharding | None = None,
      k_sharding: jax.sharding.NamedSharding | None = None,
      normalize_output: bool = True,
      return_residuals: bool = False,
  ) -> Float[Array, "*B T H d"] | tuple[Float[Array, "*B T H d"], Residuals]:
    """Performs FlexAttention.

    Inspired by "FlexAttention: The Flexibility of PyTorch with the Performance
    of FlashAttention" https://pytorch.org/blog/flexattention.

    Scaled dot-product attention from "Attention is all you need"
    https://arxiv.org/abs/1706.03762.

    Computes self- or cross-attention. The following is computed:
    `softmax(score_mod(q @ k.T)) @ v`.

    Supports both multi-head and multi-query attention
    (https://arxiv.org/abs/1911.02150).

    Arguments:
      q: Query array of shape `[batch, seq_len_q, num_heads_q, head_dim]`.
        `num_heads_q` must be a multiple of `num_heads_kv`.
        Here's an example of how q/kv heads are interleaved:
          For 8 query heads and 4 key/value heads:
          - query heads [0, 1] see key/value head 0
          - query heads [2, 3] see key/value head 1
          - etc.
      k: Key array of shape `[batch, seq_len_kv, num_heads_kv, head_dim]`.
      v: Value array of shape `[batch, seq_len_kv, num_heads_kv, head_dim]`.
      precision: The precision for the dot products. Either a tuple `(
        q_k_dot_precision, p_v_dot_precision)` or a single precision applied to
        both dot products.
      score_mod: Score modification function applied to the logits / scores
        before softmax calculation. The scores are passed to the function with
        the dtype of the q-k dot product accumulators and are **not** scaled.
      mask_mod: Mask modification function. The function is passed the shape of
        the scores array and must return a boolean mask array broadcastable to
        that shape. Scores will be set to a large negative value where the mask
        is `False`, before the softmax (such that the corresponding weights
        after the softmax are zero). While the same masking could be achieved
        with `score_mod`, `mask_mod` may allow the computation of scores to be
        skipped where the values are masked and should be preferred / used in
        conjunction with `score_mod`.
      dropout_mask: Optional boolean mask, applied after softmax calculation.
      dropout_rate: If `dropout_mask` is not `None`, weights will be scaled by
        `1 / (1 - dropout_rate)` after the dropout mask is applied.
      q_sharding: NamedSharding for `q`. Sharding information for the other
        tensors will be inferred by this.
      k_sharding: NamedSharding for `k` and also `v`. If `None`,
        `q_sharding` will be used to derive kv_heads_axis.
      normalize_output: If `True`, divide the output by softmax denominator.
      return_residuals: If `True`, also return the residuals.

    Returns:
      The output array.
    """  # fmt: skip

    def fwd_closed(q, k, v, dropout_mask):
      return op.Op.__call__(
          self,
          q,
          k,
          v,
          precision=precision,
          score_mod=score_mod,
          mask_mod=mask_mod,
          dropout_mask=dropout_mask,
          dropout_rate=dropout_rate,
          normalize_output=normalize_output,
          return_residuals=return_residuals,
      )

    if q_sharding is None:
      return fwd_closed(q, k, v, dropout_mask)

    # TODO: Support sharding.
    raise NotImplementedError("Sharding is not supported yet.")

  def bind(
      self,
      q: Float[Array | QuantizedArray, "*B T H D"],
      k: Float[Array | QuantizedArray, "*B t h D"],
      v: Float[Array | QuantizedArray, "*B t h d"],
      *,
      precision: (
          jax.lax.PrecisionLike
          | tuple[jax.lax.PrecisionLike, jax.lax.PrecisionLike]
      ) = None,
      score_mod: ScoreMod | None = None,
      mask_mod: MaskMod | None = None,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None = None,
      dropout_rate: float = 0.0,
      q_sharding: jax.sharding.NamedSharding | None = None,
      k_sharding: jax.sharding.NamedSharding | None = None,
      normalize_output: bool = True,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds the arguments for the attention function."""
    if q_sharding is not None or k_sharding is not None:
      raise NotImplementedError("Sharding is not supported in `bind` yet.")

    if (dropout_rate != 0.0) and (dropout_mask is None):
      raise ValueError("`dropout_mask` can't be None if `dropout_rate` != 0.0")

    if not isinstance(precision, tuple):
      precision = (precision, precision)

    q_k_dot_precision, p_v_dot_precision = precision
    q_k_dot_precision = precision_lib.to_dot_algorithm_preset(
        q.dtype, k.dtype, q_k_dot_precision
    )
    p_v_dot_precision = precision_lib.to_dot_algorithm_preset(
        v.dtype, v.dtype, p_v_dot_precision
    )

    return super().bind(
        q,
        k,
        v,
        precision=(q_k_dot_precision, p_v_dot_precision),
        score_mod=score_mod,
        mask_mod=mask_mod,
        dropout_mask=dropout_mask,
        dropout_rate=dropout_rate,
        normalize_output=normalize_output,
        return_residuals=return_residuals,
    )

  @jaxtyping.jaxtyped
  def _fwd(
      self,
      q: Float[Array | QuantizedArray, "*B T H D"],
      k: Float[Array | QuantizedArray, "*B t h D"],
      v: Float[Array | QuantizedArray, "*B t h d"],
      *,
      precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset],
      score_mod: ScoreMod | None,
      mask_mod: MaskMod | None,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
      dropout_rate: float,
      normalize_output: bool,
      return_residuals: bool,
      config: _Config,
  ) -> tuple[Float[Array, "*B T H d"], Residuals | None]:
    del config  # Unused.

    q, k, v = map(as_array, (q, k, v))
    if k.shape[-2] not in (1, q.shape[-2]):
      repeats = q.shape[-2] // k.shape[-2]
      k = jnp.repeat(k, repeats, axis=-2)
      v = jnp.repeat(v, repeats, axis=-2)

    q_k_dot_precision, weights_v_dot_precision = precision

    logits = jnp.einsum(
        "...qhd,...khd->...hqk",
        q,
        k,
        precision=q_k_dot_precision,
        preferred_element_type=q_k_dot_precision.accumulation_type,
    )

    logits = score_mod(logits)

    if mask_mod is not None:
      # This is not `-inf` as this can lead to `NaN`s when a full softmax row is
      # masked (with stable softmax). This is because the maximum value for the
      # row will be `-inf`, leading to `(-inf) - (-inf)` in the softmax.
      mask_value = float(jnp.finfo(logits.dtype).min)
      logits = jnp.where(mask_mod(logits.shape), logits, mask_value)

    weights, softmax_residuals = _softmax(logits, normalize_output)

    if dropout_mask is not None:
      weights *= dropout_mask.astype(weights.dtype) / (1 - dropout_rate)

    weights = weights.astype(v.dtype)
    out = jnp.einsum(
        "...hqk,...khd->...qhd",
        weights,
        v,
        precision=weights_v_dot_precision,
        preferred_element_type=weights_v_dot_precision.accumulation_type,
    ).astype(q.dtype)
    return out, (softmax_residuals if return_residuals else None)


def as_array(x: jax.Array | QuantizedArray) -> jax.Array:
  return x.recompose() if isinstance(x, QuantizedArray) else x


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _softmax(x: jax.Array, normalize: bool) -> tuple[jax.Array, Residuals]:
  """Computes softmax, possibly using residual from the forward pass."""
  # Always perform reductions in at least f32 precision.
  dtype = jnp.promote_types(x.dtype, jnp.float32)
  x_max = jnp.max(x.astype(dtype), axis=-1, keepdims=True)
  unnormalized = jnp.exp(x - x_max)
  denom = jnp.sum(unnormalized, axis=-1, keepdims=True)
  out = (unnormalized / denom) if normalize else unnormalized
  return out, (jnp.squeeze(x_max, -1), jnp.squeeze(denom, -1))


@_softmax.defjvp
def _softmax_jvp(normalize, primals, tangents):
  (x,) = primals
  (x_dot,) = tangents
  y, residual = _softmax(x, normalize)
  if normalize:
    y_dot = y * (x_dot - (y * x_dot).sum(axis=-1, keepdims=True))
  else:
    x_max = residual[0][..., None]
    x_dot_max_mean = jnp.mean(x_dot, axis=-1, keepdims=True, where=(x == x_max))
    y_dot = y * (x_dot - x_dot_max_mean)
  return (y, residual), (y_dot, jax.tree.map(jnp.zeros_like, residual))
