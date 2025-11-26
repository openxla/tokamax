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
"""`DotProductAttention` wrapper for `FlexAttention`."""

import dataclasses
from typing import Annotated

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import qwix
from tokamax._src import jaxtyping
from tokamax._src import pydantic
from tokamax._src.ops import op
from tokamax._src.ops.attention import base as attn_base
from tokamax._src.ops.flex_attention import base
from typing_extensions import override


Mask = attn_base.Mask
Residuals = attn_base.Residuals
QArray = qwix.QArray
PagingInfo = attn_base.PagingInfo


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class WrappedFlexAttention(attn_base.DotProductAttention[op.NullConfig, None]):
  """`DotProductAttention` implementation wrapping a `FlexAttention`."""

  impl: Annotated[base.FlexAttention, pydantic.AnyInstanceOf]

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array | QArray, "*B T H D"],
      k: Float[Array | QArray, "*B t h D"],
      v: Float[Array | QArray, "*B t h d"],
      *,
      precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset],
      logits_dtype: jnp.dtype,
      logits_scale: float,
      bias: Float[Array, "*#B #H #T #t"] | None,
      logits_soft_cap: float | None,
      mask: Mask,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
      dropout_rate: float,
      paging_info: PagingInfo | None,
      q_indices: Int[Array, "*#B #H T"] | None,
      k_indices: Int[Array, "*#B #h t"] | None,
      normalize_output: bool,
      return_residuals: bool,
      config: op.NullConfig,
  ) -> tuple[Float[Array, "*B T H d"], Residuals | None]:
    del config  # Unused.

    if paging_info is not None:
      raise NotImplementedError("Paged attention not supported.")

    def score_mod(logits):
      logits = logits.astype(logits_dtype) * logits_scale
      if bias is not None:
        logits += bias
      if logits_soft_cap is not None:
        logits = logits_soft_cap * jnp.tanh(logits / logits_soft_cap)
      return logits

    def mask_mod(shape, *, mask=mask, q_indices=q_indices, k_indices=k_indices):
      if q_indices is None:
        q_indices = jnp.arange(shape[-2])
      if k_indices is None:
        k_indices = jnp.arange(shape[-1])
      mask = mask.as_array(q_indices, k_indices)
      return jnp.ones(shape, dtype=jnp.bool_) if mask is None else mask

    out = self.impl(
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
    return out if return_residuals else (out, None)
