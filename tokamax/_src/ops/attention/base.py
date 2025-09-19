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
"""Base for attention ops."""

import dataclasses
import functools
import math
import types
from typing import Any, Literal, NotRequired, Self, TypeVar, TypedDict, overload

import jax
from jax import export
from jax.experimental import shard_map
import jax.numpy as jnp
from jax.typing import DTypeLike  # pylint: disable=g-importing-member
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import ad
from tokamax._src import jaxtyping
from tokamax._src import precision as precision_lib
from tokamax._src import quantization
from tokamax._src.ops import op


class AUTO:  # Used as a sentinel value.
  pass


QuantizedArray = quantization.QuantizedArray


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Mask:
  """An attention mask.

  `k_start` (inclusive) and `k_end` (exclusive) define range of enabled
  k-sequence values for each row of logits.

  For example, a local attention mask could be defined as follows:
  ```
  seq_len_q = seq_len_k = 4
  window_size = 2
  k_start = jnp.maximum(0, jnp.arange(seq_len_q) + 1 - window_size)
  mask = Mask(k_start=k_start, is_causal=True)
  assert mask.as_array(seq_len_q, seq_len_k) == jnp.array(
      [[1, 0, 0, 0],
       [1, 1, 0, 0],
       [0, 1, 1, 0],
       [0, 0, 1, 1]], dtype=bool)
  ```
  Or equivalently (but less efficiently):
  ```
  k_end = jnp.arange(seq_len_q) + 1
  k_start = jnp.maximum(0, k_end - window_size)
  mask = Mask(k_start=k_start, k_end=k_end)
  assert mask.as_array(seq_len_q, seq_len_k) == jnp.array(
      [[1, 0, 0, 0],
       [1, 1, 0, 0],
       [0, 1, 1, 0],
       [0, 0, 1, 1]], dtype=bool)
  ```

  A mask for two independent causal sequences could be defined as follows:
  ```
  k_start = jnp.array([0, 0, 2, 2])
  mask = Mask(k_start=k_start, is_causal=True)
  assert mask.as_array(seq_len_q, seq_len_k) == jnp.array(
      [[1, 0, 0, 0],
       [1, 1, 0, 0],
       [0, 0, 1, 0],
       [0, 0, 1, 1]], dtype=bool)
  ```
  """

  bool_mask: Bool[Array, "*#B #T #t"] | None = None
  _: dataclasses.KW_ONLY
  q_start: Int[Array, "*#B #t"] | None = None
  q_end: Int[Array, "*#B #t"] | None = None
  k_start: Int[Array, "*#B #T"] | None = None
  k_end: Int[Array, "*#B #T"] | None = None
  is_causal: bool = dataclasses.field(default=False, metadata=dict(static=True))

  def as_array(
      self,
      q_len_or_indices: int | Int[Array, "*#B T"],
      k_len_or_indices: int | Int[Array, "*#B t"],
  ) -> Bool[Array, "*#B #T #t"] | None:
    """Returns the mask as a boolean array."""
    if isinstance(q_len_or_indices, int) or export.is_symbolic_dim(
        q_len_or_indices
    ):
      q_indices = jnp.arange(q_len_or_indices)
    else:
      q_indices = q_len_or_indices

    if isinstance(k_len_or_indices, int) or export.is_symbolic_dim(
        k_len_or_indices
    ):
      k_indices = jnp.arange(k_len_or_indices)
    else:
      k_indices = k_len_or_indices

    q_indices = q_indices[..., None]
    k_indices = k_indices[..., None, :]

    mask = []
    if self.bool_mask is not None:
      mask.append(self.bool_mask)
      # Check `bool_mask` shape is compatible with `{q,kv}_indices`.
      _ = jnp.broadcast_shapes(
          q_indices.shape, k_indices.shape, self.bool_mask.shape
      )

    if self.q_start is not None:
      mask.append(q_indices >= self.q_start[..., None, :])

    if self.q_end is not None:
      mask.append(q_indices < self.q_end[..., None, :])

    if self.k_start is not None:
      mask.append(k_indices >= self.k_start[..., None])

    if self.k_end is not None:
      mask.append(k_indices < self.k_end[..., None])

    if self.is_causal:
      mask.append(q_indices >= k_indices)

    logical_and = functools.partial(functools.reduce, jnp.logical_and)
    return jax.lax.broadcast_to_rank(logical_and(mask), 3) if mask else None

  def take(self, *attrs: str) -> tuple[Any, ...]:
    """Returns a mask with attrs removed and the removed attrs."""
    default_mask = type(self)()
    replacements = {attr: getattr(default_mask, attr) for attr in attrs}
    values = (getattr(self, attr) for attr in attrs)
    return dataclasses.replace(self, **replacements), *values

  def __bool__(self):
    return (
        self.bool_mask is not None
        or self.is_causal
        or self.q_start is not None
        or self.q_end is not None
        or self.k_start is not None
        or self.k_end is not None
    )

  def __and__(self, other: "Bool[Array, '*#B #T #t'] | Mask") -> "Mask":
    """Returns the intersection of two masks."""
    if not isinstance(other, Mask):
      other = Mask(other)

    def combine(fn):
      return lambda a, b: b if a is None else a if b is None else fn(a, b)

    return Mask(
        bool_mask=combine(jnp.logical_and)(self.bool_mask, other.bool_mask),
        q_start=combine(jnp.maximum)(self.q_start, other.q_start),
        q_end=combine(jnp.minimum)(self.q_end, other.q_end),
        k_start=combine(jnp.maximum)(self.k_start, other.k_start),
        k_end=combine(jnp.minimum)(self.k_end, other.k_end),
        is_causal=self.is_causal or other.is_causal,
    )


CAUSAL_MASK = Mask(is_causal=True)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PagingInfo:
  """Information about the paged kv cache.

  Attributes:
    num_active_pages: Number of pages that each query needs to attend over.
    active_page_indices: Sequence of page indices to attend over for each query.
      MP indicates the maximum number of page indices in a sequence.
    lengths: Length of each page in the paged kv cache, independently of which
      query it belongs to.

  Note:
    - "Each query" means one per batch element in the `q` array.
    - `is_causal`, `k_start` and `k_end` are invalid when paging is used. Each
      page is assumed to be valid from the start of its sequence, up to its
      corresponding `length`.
  """

  num_active_pages: Int[Array, "*#B"]
  active_page_indices: Int[Array, "*#B MP"]
  lengths: Int[Array, "*#b"]


_Config = TypeVar("_Config")
_Key = TypeVar("_Key")
# The attention residuals come from the softmax calculation:
# `(maximum softmax input values, softmax denominator)`.
Residuals = tuple[Float[Array, "*B H T"], Float[Array, "*B H T"]]


class DotProductAttention(
    op.Op[Any, Float[Array, "*B T H d"], Residuals, _Config, _Key]
):
  """Dot product attention function."""

  # We override `__call__` in order to handle sharding at the top level.
  @overload
  def __call__(
      self,
      q: Float[Array | QuantizedArray, "*B T H D"],
      k: Float[Array | QuantizedArray, "*b t h D"],
      v: Float[Array | QuantizedArray, "*b t h d"],
      *,
      precision: (
          jax.lax.PrecisionLike
          | tuple[jax.lax.PrecisionLike, jax.lax.PrecisionLike]
      ) = ...,
      logits_dtype: DTypeLike | type[AUTO] = ...,
      logits_scale: float | type[AUTO] = ...,
      bias: Float[Array, "*#B #H #T #t"] | None = ...,
      logits_soft_cap: float | None = ...,
      mask: Bool[Array, "*#B #H #T #t"] | Mask | None = ...,
      is_causal: bool = ...,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None = ...,
      dropout_rate: float = ...,
      paging_info: PagingInfo | None = None,
      q_sharding: jax.sharding.NamedSharding | None = ...,
      k_sharding: jax.sharding.NamedSharding | None = ...,
      q_indices: Int[Array, "*#B #H T"] | None = ...,
      k_indices: Int[Array, "*#b #H t"] | None = ...,
      normalize_output: bool = ...,
      return_residuals: Literal[False] = ...,
  ) -> Float[Array, "*B T H d"]:
    ...

  @overload
  def __call__(
      self,
      q: Float[Array | QuantizedArray, "*B T H D"],
      k: Float[Array | QuantizedArray, "*b t h D"],
      v: Float[Array | QuantizedArray, "*b t h d"],
      *,
      precision: (
          jax.lax.PrecisionLike
          | tuple[jax.lax.PrecisionLike, jax.lax.PrecisionLike]
      ) = ...,
      logits_dtype: DTypeLike | type[AUTO] = ...,
      logits_scale: float | type[AUTO] = ...,
      bias: Float[Array, "*#B #H #T #t"] | None = ...,
      logits_soft_cap: float | None = ...,
      mask: Bool[Array, "*#B #H #T #t"] | Mask | None = ...,
      is_causal: bool = ...,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None = ...,
      dropout_rate: float = ...,
      paging_info: PagingInfo | None = None,
      q_sharding: jax.sharding.NamedSharding | None = ...,
      k_sharding: jax.sharding.NamedSharding | None = ...,
      q_indices: Int[Array, "*#B #H T"] | None = ...,
      k_indices: Int[Array, "*#b #h t"] | None = ...,
      normalize_output: bool = ...,
      return_residuals: Literal[True] = ...,
  ) -> tuple[Float[Array, "*B T H d"], Residuals]:
    ...

  @jaxtyping.jaxtyped
  def __call__(
      self,
      q: Float[Array | QuantizedArray, "*B T H D"],
      k: Float[Array | QuantizedArray, "*b t h D"],
      v: Float[Array | QuantizedArray, "*b t h d"],
      *,
      precision: (
          jax.lax.PrecisionLike
          | tuple[jax.lax.PrecisionLike, jax.lax.PrecisionLike]
      ) = jax.lax.Precision.DEFAULT,
      logits_dtype: DTypeLike | type[AUTO] = AUTO,
      logits_scale: float | type[AUTO] = AUTO,
      bias: Float[Array, "*#B #H #T #t"] | None = None,
      logits_soft_cap: float | None = None,
      mask: Bool[Array, "*#B #H #T #t"] | Mask | None = None,
      is_causal: bool = False,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None = None,
      dropout_rate: float = 0.0,
      paging_info: PagingInfo | None = None,
      q_sharding: jax.sharding.NamedSharding | None = None,
      k_sharding: jax.sharding.NamedSharding | None = None,
      q_indices: Int[Array, "*#B #H T"] | None = None,
      k_indices: Int[Array, "*#b #h t"] | None = None,
      normalize_output: bool = True,
      return_residuals: bool = False,
  ) -> Float[Array, "*B T H d"] | tuple[Float[Array, "*B T H d"], Residuals]:
    """Performs a scaled dot-product attention.

    Scaled dot-product attention from "Attention is all you need"
    https://arxiv.org/abs/1706.03762.

    Computes self- or cross-attention. The following is computed:
    `softmax(logits_scale * q @ k.T + bias) @ v`.

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
      k: Key array of shape `[batch, seq_len_kv, num_heads_kv, head_dim]`. If
        `paging_info` is not None, the "batch" dimensions correspond to pages.
      v: Value array of shape `[batch, seq_len_kv, num_heads_kv, head_dim]`. If
        `paging_info` is not None, the "batch" dimensions correspond to pages.
      precision: The precision for the dot products. Either a tuple `(
        q_k_dot_precision, p_v_dot_precision)` or a single precision applied to
        both dot products.
      logits_dtype: Data type for attention logits (`q @ k.T`). If `AUTO` is
        passed (the default), the accumulator type from the q-k dot product will
        be used.
      logits_scale: A scaling factor applied to the attention logits
        (`q @ k.T`). If `AUTO` is passed (the default), it is set to `1 /
        sqrt(head_dim)`. NOTE This scaling may be applied in another numerically
        equivalent way (such as scaling `query` before the dot product).
      bias: Optional bias array, broadcastable to shape `[batch, num_heads,
        seq_len_q, seq_len_kv]`.
      logits_soft_cap: If not `None`, perform `logits = logits_soft_cap *
        tanh(logits / logits_soft_cap)`, where `logits` are `logits_scale *
        q @ k.T + bias`.
      mask: Optional boolean mask, broadcastable to `[batch, num_heads,
        seq_len_q, seq_len_kv]`. Attention weights are masked out if the
        corresponding mask value is `False`.
      is_causal: If `True`, apply causal masking. NOTE While this can also be
        achieved using `bias` or `mask`, using this option may allow computation
        of causally-masked values to be avoided.
      dropout_mask: Optional boolean mask, applied after softmax calculation.
      dropout_rate: If `dropout_mask` is not `None`, weights will be scaled by
        `1 / (1 - dropout_rate)` after the dropout mask is applied.
      paging_info: Information about the paged kv cache.
      q_sharding: NamedSharding for `q`. Sharding information for the other
        tensors will be inferred by this.
      k_sharding: NamedSharding for `k` and also `v`. If `None`,
        `q_sharding` will be used to derive kv_heads_axis.
      q_indices: Optional indices for each token in query sequence.
      k_indices: Optional indices for each token in key/value sequence.
      normalize_output: If `True`, divide the output by softmax denominator.
      return_residuals: If `True`, also return the residuals.

    Returns:
      The output array.
    """  # fmt: skip
    # Use `bind` to canonicalize the arguments.
    params = locals().copy()
    params.pop("q_sharding")
    params.pop("k_sharding")
    mask = type(self).bind(**params).kwargs["mask"]
    del is_causal  # Combined into `mask`.

    def fwd_closed(
        q, k, v, bias, mask, dropout_mask, paging_info, q_indices, k_indices
    ):
      return op.Op.__call__(
          self,
          q,
          k,
          v,
          precision=precision,
          logits_dtype=logits_dtype,
          logits_scale=logits_scale,
          bias=bias,
          logits_soft_cap=logits_soft_cap,
          mask=mask,
          dropout_mask=dropout_mask,
          dropout_rate=dropout_rate,
          paging_info=paging_info,
          q_indices=q_indices,
          k_indices=k_indices,
          normalize_output=normalize_output,
          return_residuals=return_residuals,
      )

    if q_sharding is None:
      return fwd_closed(
          q, k, v, bias, mask, dropout_mask, paging_info, q_indices, k_indices
      )

    mesh, q_spec = q_sharding.mesh, q_sharding.spec

    if len(q_spec) > len(q.shape):
      raise ValueError("Too many dimensions in q_sharding's pspec.")

    # Match `shard_map` behaviour, with sharding applied to leading axes.
    q_axes = tuple(q_spec) + (None,) * (q.ndim - len(q_spec))
    *batch_axes, seq_q_axis, heads_axis, head_dim_axis = q_axes
    if k_sharding is None:
      k_batch_axes = batch_axes
      k_heads_axis = heads_axis
      seq_k_axis = None
    else:
      if q_sharding.mesh != k_sharding.mesh:
        raise ValueError("q_sharding and k_sharding must have the same mesh.")
      k_spec = k_sharding.spec
      k_axes = tuple(k_spec) + (None,) * (k.ndim - len(k_spec))
      *k_batch_axes, seq_k_axis, k_heads_axis, k_head_dim_axis = k_axes
      if seq_k_axis is not None and mesh.shape[seq_k_axis] != 1:
        raise ValueError("Sharding along seq_k_axis unsupported.")
      if k_head_dim_axis is not None and mesh.shape[k_head_dim_axis] != 1:
        raise ValueError("Sharding along head_dim unsupported.")

    if head_dim_axis is not None and mesh.shape[head_dim_axis] != 1:
      raise ValueError("Sharding along head_dim unsupported.")

    # Ensure that `q_indices` is specified when sharding the q-sequence with
    # causal masking enabled.
    if seq_q_axis is not None and mesh.shape[seq_q_axis] != 1:
      if mask.is_causal and q_indices is None:
        q_indices = jnp.arange(q.shape[-3])

    def qkv_axes(x, axes):
      if isinstance(x, QuantizedArray):
        return QuantizedArray(axes, axes)  # pytype: disable=wrong-arg-types
      return axes

    # TODO: Derive in_axes from the type annotations.
    in_axes = (
        qkv_axes(q, [*batch_axes, seq_q_axis, heads_axis, head_dim_axis]),
        qkv_axes(k, [*batch_axes, seq_k_axis, k_heads_axis, head_dim_axis]),
        qkv_axes(v, [*batch_axes, seq_k_axis, k_heads_axis, head_dim_axis]),
        [*batch_axes, heads_axis, seq_q_axis, seq_k_axis],  # bias
        dataclasses.replace(
            mask,
            bool_mask=[*batch_axes, heads_axis, seq_q_axis, seq_k_axis],
            q_start=[*batch_axes, heads_axis, seq_k_axis],
            q_end=[*batch_axes, heads_axis, seq_k_axis],
            k_start=[*batch_axes, k_heads_axis, seq_q_axis],
            k_end=[*batch_axes, k_heads_axis, seq_q_axis],
        ),
        [*batch_axes, heads_axis, seq_q_axis, seq_k_axis],  # dropout_mask
        PagingInfo([*batch_axes], [*batch_axes, None], [*k_batch_axes]),
        [*batch_axes, heads_axis, seq_q_axis],  # q_indices
        [*batch_axes, k_heads_axis, seq_k_axis],  # k_indices
    )
    P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
    residual_spec = P(*batch_axes, heads_axis, seq_q_axis)
    out_specs = (P(*q_axes), residual_spec) if return_residuals else P(*q_axes)
    is_array = lambda x: isinstance(x, jax.Array)

    def bcast(x, axes):
      return jax.lax.broadcast_to_rank(x, len(axes)) if is_array(x) else x

    def spec(x, axes):
      if not is_array(x):
        return None
      # Do not shard along axes that are broadcast.
      return P(*(None if d == 1 else a for a, d in zip(axes, x.shape)))

    def fwd_sharded(*args):  # pylint: disable=function-redefined
      is_leaf = lambda x: x is None
      args = jax.tree.map(bcast, args, in_axes, is_leaf=is_leaf)
      in_specs = jax.tree.map(spec, args, in_axes, is_leaf=is_leaf)
      return shard_map.shard_map(
          fwd_closed, mesh, in_specs, out_specs, check_rep=False
      )(*args)

    return fwd_sharded(
        q, k, v, bias, mask, dropout_mask, paging_info, q_indices, k_indices
    )

  def bind(
      self,
      q: Float[Array | QuantizedArray, "*B T H D"],
      k: Float[Array | QuantizedArray, "*b t h D"],
      v: Float[Array | QuantizedArray, "*b t h d"],
      *,
      precision: (
          jax.lax.PrecisionLike
          | tuple[jax.lax.PrecisionLike, jax.lax.PrecisionLike]
      ) = jax.lax.Precision.DEFAULT,
      logits_dtype: DTypeLike | type[AUTO] = AUTO,
      logits_scale: float | type[AUTO] = AUTO,
      bias: Float[Array, "*#B #H #T #t"] | None = None,
      logits_soft_cap: float | None = None,
      mask: Bool[Array, "*#B #H #T #t"] | Mask | None = None,
      is_causal: bool = False,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None = None,
      dropout_rate: float = 0.0,
      paging_info: PagingInfo | None = None,
      q_sharding: jax.sharding.NamedSharding | None = None,
      k_sharding: jax.sharding.NamedSharding | None = None,
      q_indices: Int[Array, "*#B #H T"] | None = None,
      k_indices: Int[Array, "*#b #h t"] | None = None,
      normalize_output: bool = True,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds the arguments for the attention function."""
    if q_sharding is not None or k_sharding is not None:
      raise NotImplementedError("Sharding is not supported in `bind` yet.")

    if (dropout_rate != 0.0) and (dropout_mask is None):
      raise ValueError("`dropout_mask` can't be None if `dropout_rate` != 0.0")

    if paging_info is None and k.shape[:-3] != q.shape[:-3]:
      raise ValueError("`k` batch size must be the same as `q`.")

    if not isinstance(precision, tuple):
      precision = (precision, precision)

    q_k_dot_precision, p_v_dot_precision = precision

    if not isinstance(q_k_dot_precision, jax.lax.DotAlgorithmPreset):
      q_k_dot_precision = precision_lib.to_dot_algorithm_preset(
          q.dtype, k.dtype, q_k_dot_precision
      )

    if not isinstance(p_v_dot_precision, jax.lax.DotAlgorithmPreset):
      p_v_dot_precision = precision_lib.to_dot_algorithm_preset(
          v.dtype, v.dtype, p_v_dot_precision
      )

    if logits_dtype is AUTO:
      logits_dtype = q_k_dot_precision.accumulation_type

    if logits_scale is AUTO:
      logits_scale = 1 / math.sqrt(q.shape[-1])

    if not isinstance(mask, Mask):
      mask = Mask(mask)

    if is_causal:
      mask &= CAUSAL_MASK

    return super().bind(
        q,
        k,
        v,
        precision=(q_k_dot_precision, p_v_dot_precision),
        logits_dtype=jnp.dtype(logits_dtype),
        logits_scale=logits_scale,
        bias=bias,
        logits_soft_cap=logits_soft_cap,
        mask=mask,
        dropout_mask=dropout_mask,
        dropout_rate=dropout_rate,
        paging_info=paging_info,
        q_indices=q_indices,
        k_indices=k_indices,
        normalize_output=normalize_output,
        return_residuals=return_residuals,
    )

  @jaxtyping.jaxtyped
  def _fwd(
      self,
      q: Float[Array | QuantizedArray, "*B T H D"],
      k: Float[Array | QuantizedArray, "*b t h D"],
      v: Float[Array | QuantizedArray, "*b t h d"],
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
      k_indices: Int[Array, "*#b #h t"] | None,
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
    ).astype(logits_dtype)

    logits *= logits_scale

    if bias is not None:
      logits += bias

    if logits_soft_cap is not None:
      logits = logits_soft_cap * jnp.tanh(logits / logits_soft_cap)

    q_len_or_indices = q.shape[-3] if q_indices is None else q_indices
    k_len_or_indices = k.shape[-3] if k_indices is None else k_indices
    mask = mask.as_array(q_len_or_indices, k_len_or_indices)

    if mask is not None:
      # This is not `-inf` as this can lead to `NaN`s when a full softmax row is
      # masked (with stable softmax). This is because the maximum value for the
      # row will be `-inf`, leading to `(-inf) - (-inf)` in the softmax.
      mask_value = float(jnp.finfo(logits.dtype).min)
      logits = jnp.where(jnp.asarray(mask), logits, mask_value)

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


def needs_stable_softmax(
    logits_dtype: DTypeLike, logits_soft_cap: float | None
) -> bool:
  """Returns `True` if stable softmax is needed."""
  # Typically, softmax is calculated using the stable softmax algorithm, which
  # subtracts the maximum value before the exponential. This eliminates the
  # possibility of overflow to infinity, while hopefully keeping underflows to
  # zero minimal.
  #
  # If we are using tanh clipping, the range of (non-masked) softmax inputs is
  # bound to `[-logits_soft_cap, logits_soft_cap]`, so {under,over}flow may be
  # avoided without subtracting the maximum value.
  if jnp.dtype(logits_dtype) not in (jnp.float32, jnp.bfloat16):
    return True
  # NOTE: This value is very conservative and could be relaxed.
  return True if logits_soft_cap is None else logits_soft_cap > 60.0


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


class DotProductAttentionGrads(TypedDict):
  q: Float[Array, "*B T H D"]
  k: Float[Array, "*b t h D"]
  v: Float[Array, "*b t h d"]
  bias: NotRequired[Float[Array, "*#B #H #T #t"] | None]


class DotProductAttentionVjp(
    op.Op[Any, DotProductAttentionGrads, types.NoneType, _Config, _Key]
):
  """Dot product attention VJP."""

  @jaxtyping.jaxtyped
  def _fwd(
      self,
      residuals: Residuals,
      out: Float[Array, "*B T H d"],
      dout: Float[Array, "*B T H d"],
      q: Float[Array, "*B T H D"],
      k: Float[Array, "*b t h D"],
      v: Float[Array, "*b t h d"],
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
      k_indices: Int[Array, "*#b #H t"] | None,
      normalize_output: bool,
      return_residuals: bool,
      config: _Config,
  ) -> tuple[DotProductAttentionGrads, None]:
    """Computes attention VJP."""
    del config  # Unused.

    if return_residuals:
      raise NotImplementedError("`return_residuals` not supported.")
    if paging_info is not None:
      raise NotImplementedError("Paged attention is not supported in VJP.")
    if q.shape[:-3] != k.shape[:-3]:
      raise ValueError("q and k must have the same batch dimensions.")

    def attend(q, k, v, bias):
      out, softmax_residual = DotProductAttention()._fwd(  # pylint: disable=protected-access
          q,
          k,
          v,
          precision=precision,
          logits_dtype=logits_dtype,
          logits_scale=logits_scale,
          bias=bias,
          logits_soft_cap=logits_soft_cap,
          mask=mask,
          dropout_mask=dropout_mask,
          dropout_rate=dropout_rate,
          paging_info=paging_info,
          q_indices=q_indices,
          k_indices=k_indices,
          normalize_output=normalize_output,
          return_residuals=True,
          config=None,
      )
      residuals_ = (
          q,
          k,
          v,
          bias,
          mask,
          dropout_mask,
          paging_info,
          q_indices,
          k_indices,
      )
      residuals_ += (out, softmax_residual)
      return out, residuals_

    vjp = ad.get_vjp_taking_residuals(attend, q, k, v, bias)
    residuals_ = (
        q,
        k,
        v,
        bias,
        mask,
        dropout_mask,
        paging_info,
        q_indices,
        k_indices,
    )
    residuals_ += (out, residuals)
    dq, dk, dv, dbias = vjp(residuals_, dout)
    return DotProductAttentionGrads(q=dq, k=dk, v=dv, bias=dbias), None
