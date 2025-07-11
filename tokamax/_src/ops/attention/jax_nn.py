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
"""`jax.nn.dot_product_attention` implementation of `DotProductAttention`."""

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src import quantization
from tokamax._src import shape as shape_lib
from tokamax._src.ops import op
from tokamax._src.ops.attention import base


Mask = base.Mask
QuantizedArray = quantization.QuantizedArray
PagingInfo = base.PagingInfo


@dataclasses.dataclass(frozen=True, kw_only=True)
class JaxNnDotProductAttention(base.DotProductAttention[op.NullConfig, None]):
  """`jax.nn.dot_product_attention` implementation."""

  implementation: Literal["xla", "cudnn"] | None = None

  @jaxtyping.jaxtyped
  def _fwd(
      self,
      q: Float[Array | QuantizedArray, "*B T H D"],
      k: Float[Array | QuantizedArray, "*B t h D"],
      v: Float[Array | QuantizedArray, "*B t h d"],
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
      k_indices: Int[Array, "*#B #H t"] | None,
      normalize_output: bool,
      return_residuals: bool,
      config: op.NullConfig,
  ) -> tuple[Float[Array, "*B T H d"], None]:
    del config  # Unused.

    if precision[0] != precision[1]:
      raise NotImplementedError("Both precisions must be the same.")
    if logits_dtype != jnp.float32:
      raise NotImplementedError("`logits_dtype` must be `jnp.float32`.")
    if logits_soft_cap is not None:
      raise NotImplementedError("`logits_soft_cap` not supported.")
    if dropout_mask is not None or dropout_rate != 0.0:
      raise NotImplementedError("dropout not supported.")
    if not normalize_output:
      raise NotImplementedError("`normalize_output=False` not supported.")
    if return_residuals:
      raise NotImplementedError("`return_residuals=True` not supported.")
    if paging_info is not None:
      raise NotImplementedError("Paged attention not supported.")

    q, k, v = map(base.as_array, (q, k, v))

    is_causal = False
    if q_indices is None and k_indices is None:
      mask, is_causal = mask.take("is_causal")

    query_seq_lengths = None
    if q_indices is None and (q_end := mask.q_end) is not None:
      q_end = jax.lax.broadcast_to_rank(q_end, q.ndim - 1)
      if q_end.shape[-2:] == (1, 1):
        mask, _ = mask.take("q_end")
        query_seq_lengths = q_end[..., 0, 0]

    key_value_seq_lengths = None
    if k_indices is None and (k_end := mask.k_end) is not None:
      k_end = jax.lax.broadcast_to_rank(k_end, q.ndim - 1)
      if k_end.shape[-2:] == (1, 1):
        mask, _ = mask.take("k_end")
        key_value_seq_lengths = k_end[..., 0, 0]

    q_len_or_indices = q_indices if q_indices is not None else q.shape[-3]
    k_len_or_indices = k_indices if k_indices is not None else k.shape[-3]
    mask = mask.as_array(q_len_or_indices, k_len_or_indices)

    *batch, seq_len_q, num_heads, head_dim = q.shape
    *_, seq_len_k, _, head_dim_out = v.shape

    max_head_dim = max(head_dim, head_dim_out)
    pad_head_dim = lambda x: shape_lib.pad_dim_to(x, max_head_dim, -1)
    q, k, v = map(pad_head_dim, (q, k, v))

    def flatten_batch(x, rank=q.ndim):
      if x is None:
        return None
      x = jax.lax.broadcast_to_rank(x, rank)
      x = jnp.broadcast_to(x, (*batch, *x.shape[len(batch) :]))
      return jax.lax.collapse(x, 0, len(batch))

    q, k, v, bias, mask = map(flatten_batch, (q, k, v, bias, mask))
    query_seq_lengths = flatten_batch(query_seq_lengths, len(batch))
    key_value_seq_lengths = flatten_batch(key_value_seq_lengths, len(batch))

    # CuDNN doesn't support mask/bias implicitly broadcast along sequences.
    if self.implementation == "cudnn":
      if bias is not None:
        bias = jnp.broadcast_to(bias, (*bias.shape[:-2], seq_len_q, seq_len_k))
      if mask is not None:
        mask = jnp.broadcast_to(mask, (*mask.shape[:-2], seq_len_q, seq_len_k))

    with jax.default_matmul_precision(str(precision[0])):
      out = jax.nn.dot_product_attention(
          q,
          k,
          v,
          bias=bias,
          mask=mask,
          is_causal=is_causal,
          scale=logits_scale,
          query_seq_lengths=query_seq_lengths,
          key_value_seq_lengths=key_value_seq_lengths,
          implementation=self.implementation,
      )
    out = out.reshape(*batch, seq_len_q, num_heads, out.shape[-1])
    return out[..., :head_dim_out], None
