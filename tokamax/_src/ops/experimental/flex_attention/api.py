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
"""FlexAttention API."""

from collections.abc import Callable, Sequence
import math
from typing import Any, Final, Literal

import immutabledict
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.flex_attention import base

type Implementation = Literal["xla"]

ScoreMod = Callable[[Float[Array, "B H T t"]], Float[Array, "B H T t"]]
MaskMod = Callable[[tuple[int, int, int, int]], Bool[Array, "#B #H #T #t"]]


_IMPLEMENTATIONS = dict(
    xla=base.FlexAttention(),
)
_DEFAULT_IMPLEMENTATIONS = ("xla",)

IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS


@jaxtyping.jaxtyped
def flex_attention(
    query: Float[Array, "B T H D"],
    key: Float[Array, "B t h D"],
    value: Float[Array, "B t h d"],
    *,
    scale: float | None = None,
    score_mod: ScoreMod | None = None,
    mask_mod: MaskMod | None = None,
    precision: (
        jax.lax.PrecisionLike
        | tuple[jax.lax.PrecisionLike, jax.lax.PrecisionLike]
    ) = None,
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> Float[Array, "B T H d"]:
  """Performs FlexAttention.

  Inspired by "FlexAttention: The Flexibility of PyTorch with the Performance
  of FlashAttention" https://pytorch.org/blog/flexattention.

  Args:
    query: Query array of shape `[batch, seq_len_q, num_heads_q, head_dim]`.
      `num_heads_q` must be a multiple of `num_heads_kv`. Here's an example of
      how q/kv heads are interleaved: For 8 query heads and 4 key/value heads: -
      query heads [0, 1] see key/value head 0 - query heads [2, 3] see key/value
      head 1 - etc.
    key: Key array of shape `[batch, seq_len_kv, num_heads_kv, head_dim]`.
    value: Value array of shape `[batch, seq_len_kv, num_heads_kv, head_dim]`.
    scale: Scaling factor applied prior to softmax. If None, defaults to `1.0 /
      math.sqrt(query.shape[-1])`.
    precision: The precision for the dot products. Either a tuple `(
      q_k_dot_precision, p_v_dot_precision)` or a single precision applied to
      both dot products.
    scale: Scaling factor applied to the `query @ key.T` scores prior to
      `score_mod` and softmax. If `None`, defaults to
      `1.0 / math.sqrt(query.shape[-1])`.
    score_mod: Optional score modification function of signature
      `(scores: Float[Array, "B H T t"]) -> Float[Array, "B H T t"]` applied to
      the scaled logits / scores before `mask_mod` and softmax calculation. The
      scores array passed to `score_mod` has shape
      `[batch, num_heads_q, seq_len_q, seq_len_kv]`, has the dtype of the q-k
      dot product accumulators, and is already scaled by `scale`. Must return a
      modified scores array of the same shape.
    mask_mod: Optional mask modification function of signature
      `(shape: tuple[int, int, int, int]) -> Bool[Array, "#B #H #T #t"]`. The
      function is passed the shape
      `(batch, num_heads_q, seq_len_q, seq_len_kv)` of the scores array and must
      return a rank-4 boolean mask array broadcastable to that shape. Scores
      will be set to a large negative value where the mask is `False`, after
      `score_mod` and before the softmax (such that the corresponding weights
      after the softmax are zero). While the same masking could be achieved with
      `score_mod`, `mask_mod` may allow the computation of scores to be skipped
      where the values are masked and should be preferred / used in conjunction
      with `score_mod`.
    implementation: The implementation to use. By default, `None` is used, which
      will automatically select the best available backend, and is guaranteed to
      work on all platforms. If a sequence is passed, the first implementation
      that doesn't raise a `NotImplementedError` is used.

  Returns:
    The output array.

  Raises:
    ValueError: If `implementation` is an empty sequence or unknown.
    ExceptionGroup: If multiple implementations are provided and all of them
      raise `NotImplementedError`.
  """
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError("`implementation` must not be an empty sequence.")

  if scale is None:
    scale = 1.0 / math.sqrt(query.shape[-1])

  if scale != 1.0:

    def effective_score_mod(score):
      score = score * jnp.asarray(scale, dtype=score.dtype)
      if score_mod is not None:
        score = score_mod(score)
      return score

  else:
    effective_score_mod = score_mod

  errors = []
  for impl in implementation:
    if isinstance(impl, str):
      if impl not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown implementation: {impl}.")
      impl = IMPLEMENTATIONS[impl]

    try:
      return impl(
          query,
          key,
          value,
          precision=precision,
          score_mod=effective_score_mod,
          mask_mod=mask_mod,
      )
    except NotImplementedError as e:
      if len(implementation) == 1:
        raise
      errors.append(e)

  raise ExceptionGroup("all implementations failed", errors)
