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

"""Base operator for Splash Attention."""

from collections.abc import Callable
import dataclasses
import functools
import math
from typing import Any, ClassVar, Final, Literal, NotRequired, TypeAlias, TypeVar, TypedDict, cast, overload, override
import jax
from jax.experimental import shard_map
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from tokamax._src import batching
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.splash_attention import reference
from tokamax._src.ops.splash_attention import splash_attention_mask as mask_lib

_Config = TypeVar("_Config")

DEFAULT_MASK_VALUE = reference.DEFAULT_MASK_VALUE
QuantizationStrategy = reference.QuantizationStrategy
SegmentIds = reference.SegmentIds
SplashCustomReturnType = reference.SplashCustomReturnType
SplashResidualsType = reference.SplashResidualsType
apply_quantization = reference.apply_quantization
attention_reference = reference.attention_reference
attention_reference_vjp = reference.attention_reference_vjp

Residuals: TypeAlias = tuple[
    Float[Array, "*B H T"],
    Float[Array, "*B H T"],
]


def vmap_batch_dims[**P, T](f: Callable[P, T]) -> Callable[P, T]:
  """Returns `f` vmapped over the batch dims of its first argument."""

  def vmap(f, *args: P.args, **kwargs: P.kwargs):
    assert not kwargs
    for _ in cast(jax.Array, args[0]).shape[:-3]:
      f = batching.vmap_maybe_bcast(f, 0)
    return f(*args)

  return functools.wraps(f)(functools.partial(vmap, f))


def _mesh_axis_size(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh, axis: Any
):
  """Returns the number of shards for a PartitionSpec entry in a mesh."""
  if axis is None:
    return 1
  axis_names = axis if isinstance(axis, tuple) else (axis,)
  return math.prod(mesh.shape[name] for name in axis_names)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class Mask:
  """An attention mask dataclass.

  Attributes:
    bool_mask: An optional explicit 2D boolean mask.
    is_causal: Whether causal masking should be applied.
    offset: Offset of q start with respect to kv. Meaningful when
      `is_causal=True` or `window_size` is set. Semantics match
      splash_attention_mask.CausalMask.offset (q_idx + offset >= kv_idx).
    window_size: Optional `(left, right)` local attention window bounds.
  """

  bool_mask: Bool[jax.Array, "*#B T t"] | None = None
  _: dataclasses.KW_ONLY
  is_causal: bool = dataclasses.field(default=False, metadata=dict(static=True))
  offset: int = dataclasses.field(default=0, metadata=dict(static=True))
  window_size: tuple[int | None, int | None] | None = dataclasses.field(
      default=None, metadata=dict(static=True)
  )
  q_seq_axis: str | tuple[str, ...] | None = dataclasses.field(
      default=None, metadata=dict(static=True)
  )

  def __post_init__(self):
    if self.offset != 0 and not self.is_causal and self.window_size is None:
      raise ValueError(
          "`offset` can only be non-zero when `is_causal=True` or"
          " `window_size` is set."
      )

  def get_q_seq_shard_info(self) -> tuple[int, int | jax.Array]:
    """Returns (q_seq_shards, q_shard_idx) when inside a jax shard_map."""
    if self.q_seq_axis is None:
      return 1, 0
    try:
      return (
          jax.lax.axis_size(self.q_seq_axis),
          jax.lax.axis_index(self.q_seq_axis),
      )
    except (NameError, ValueError):
      return 1, 0

  def as_array(
      self,
      q_seq_len: int,
      kv_seq_len: int,
  ) -> Bool[Array, "*#B T t"]:
    """Returns the mask as a 2D boolean array."""
    _, q_shard_idx = self.get_q_seq_shard_info()
    q_idx = (q_shard_idx * q_seq_len + jnp.arange(q_seq_len))[:, None]
    kv_idx = jnp.arange(kv_seq_len)[None, :]

    mask = jnp.ones((q_seq_len, kv_seq_len), dtype=jnp.bool_)
    if self.window_size is not None:
      left, right = self.window_size
      if left is not None:
        mask = jnp.logical_and(mask, q_idx + self.offset - left <= kv_idx)
      if right is not None:
        mask = jnp.logical_and(mask, q_idx + self.offset + right >= kv_idx)
    if self.is_causal:
      mask = jnp.logical_and(mask, q_idx + self.offset >= kv_idx)
    if self.bool_mask is not None:
      mask = jnp.logical_and(jnp.asarray(self.bool_mask), mask)
    return mask

  def __bool__(self) -> bool:
    return (
        self.bool_mask is not None
        or self.is_causal
        or self.window_size is not None
    )


CAUSAL_MASK: Final[Mask] = Mask(is_causal=True)
FULL_MASK: Final[Mask] = Mask()


def _has_default_q_sequence(mask: mask_lib._ComputableMask) -> bool:  # pylint: disable=protected-access
  return mask.q_sequence is not None and np.array_equal(
      mask.q_sequence, np.arange(mask.shape[0], dtype=mask.q_sequence.dtype)
  )


def _canonicalize_mask(
    mask: Bool[Array, "*#B T t"] | Mask | mask_lib.Mask | None,
    *,
    preserve_local_mask: bool = False,
) -> Mask:
  """Canonicalizes a mask argument."""
  if mask is None:
    return Mask()
  if isinstance(mask, (jax.Array, np.ndarray)):
    return Mask(bool_mask=mask)
  if isinstance(mask, mask_lib.CausalMask) and _has_default_q_sequence(mask):
    return Mask(is_causal=True, offset=mask.offset)
  if (
      preserve_local_mask
      and isinstance(mask, mask_lib.LocalMask)
      and _has_default_q_sequence(mask)
  ):
    return Mask(window_size=mask.window_size, offset=mask.offset)
  if isinstance(mask, mask_lib.FullMask):
    return Mask()
  if isinstance(mask, mask_lib.Mask):
    return Mask(bool_mask=jnp.asarray(mask[:, :]))
  if isinstance(mask, Mask):
    return mask
  raise TypeError(f"Unsupported mask type: {type(mask)}")


def _resolve_bound_mask(
    mask: Bool[Array, "*#B T t"] | Mask | mask_lib.Mask | None,
    *,
    is_causal: bool,
    q_seq_len: int,
    kv_seq_len: int,
    preserve_local_mask: bool,
) -> Mask:
  """Canonicalizes `mask` and materializes unsupported sliding windows."""
  if is_causal:
    return CAUSAL_MASK
  resolved = _canonicalize_mask(mask, preserve_local_mask=preserve_local_mask)
  if not preserve_local_mask and resolved.window_size is not None:
    return Mask(bool_mask=resolved.as_array(q_seq_len, kv_seq_len))
  return resolved


def _canonicalize_segment_ids(
    segment_ids: (
        SegmentIds
        | tuple[Int[Array, "*#B T"], Int[Array, "*#B t"]]
        | Int[Array, "*#B T"]
        | None
    ),
) -> SegmentIds | None:
  """Canonicalizes the segment ids and constructs internal dataclass.

  Args:
    segment_ids: The document segment IDs. Can be of several different types
      including SegmentIds, any object with q and kv attributes, a (q, kv) tuple
      of tensors, or a single tensor used for both q and kv (for
      self-attention).

  Returns:
    A SegmentIds instance or None.
  """
  if segment_ids is None:
    return None
  if isinstance(segment_ids, SegmentIds):
    return segment_ids
  if isinstance(segment_ids, tuple) and len(segment_ids) == 2:
    return SegmentIds(
        q=jnp.asarray(segment_ids[0]), kv=jnp.asarray(segment_ids[1])
    )
  if isinstance(segment_ids, (jax.Array, np.ndarray)):
    # allow single array for self-attention
    return SegmentIds(q=jnp.asarray(segment_ids), kv=jnp.asarray(segment_ids))
  if hasattr(segment_ids, "q") and hasattr(segment_ids, "kv"):
    return SegmentIds(
        q=jnp.asarray(segment_ids.q), kv=jnp.asarray(segment_ids.kv)
    )
  raise TypeError(f"Unsupported segment ids type: {type(segment_ids)}")


def _max_logit_value_kwarg(
    op_self: op.Op[Any, Any, Any, Any, Any],
    max_logit_value: Any,
) -> dict[str, Any]:
  """Returns `{"max_logit_value": ...}` only when supported by `_fwd`."""
  if isinstance(
      max_logit_value, (int, float, np.integer, np.floating)
  ) and not isinstance(max_logit_value, bool):
    max_logit_value = float(max_logit_value)
  if "max_logit_value" in op_self._fwd_signature.parameters:  # pylint: disable=protected-access
    return {"max_logit_value": max_logit_value}
  if max_logit_value is not None:
    raise NotImplementedError("`max_logit_value` is not supported.")
  return {}


@dataclasses.dataclass(frozen=True)
class SplashAttention[_Config](
    op.Op[Any, Float[Array, "*B H T d"], Residuals, _Config, Any]
):
  """Tokamax operator template for Splash Attention."""

  supports_analytic_local_mask: ClassVar[bool] = False

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(
          self,
          "vjp",
          SplashAttentionVjp(),
      )

  @overload
  def __call__(
      self,
      q: Float[Array, "*B H T D"],
      k: Float[Array, "*B h t D"] | Float[Array, "*B t D"],
      v: Float[Array, "*B h t d"] | Float[Array, "*B t d"],
      mask: Bool[Array, "*#B T t"] | Mask | mask_lib.Mask | None = ...,
      segment_ids: (
          SegmentIds
          | tuple[Int[Array, "*#B T"], Int[Array, "*#B t"]]
          | Int[Array, "*#B T"]
          | None
      ) = ...,
      sinks: Float[Array, "*#B H"] | None = ...,
      *,
      is_mqa: bool = ...,
      is_causal: bool = ...,
      mask_value: float = ...,
      attn_logits_soft_cap: float | None = ...,
      max_logit_value: (
          float | Float[Array, ""] | Float[Array, "*#B #H"] | None
      ) = ...,
      dropout_rate: float = ...,
      prng_key: jax.Array | None = ...,
      q_sharding: jax.sharding.NamedSharding | None = ...,
      k_sharding: jax.sharding.NamedSharding | None = ...,
      return_residuals: Literal[False] = ...,
  ) -> Float[Array, "*B H T d"]:
    ...

  @overload
  def __call__(
      self,
      q: Float[Array, "*B H T D"],
      k: Float[Array, "*B h t D"] | Float[Array, "*B t D"],
      v: Float[Array, "*B h t d"] | Float[Array, "*B t d"],
      mask: Bool[Array, "*#B T t"] | Mask | mask_lib.Mask | None = ...,
      segment_ids: (
          SegmentIds
          | tuple[Int[Array, "*#B T"], Int[Array, "*#B t"]]
          | Int[Array, "*#B T"]
          | None
      ) = ...,
      sinks: Float[Array, "*#B H"] | None = ...,
      *,
      is_mqa: bool = ...,
      is_causal: bool = ...,
      mask_value: float = ...,
      attn_logits_soft_cap: float | None = ...,
      max_logit_value: (
          float | Float[Array, ""] | Float[Array, "*#B #H"] | None
      ) = ...,
      dropout_rate: float = ...,
      prng_key: jax.Array | None = ...,
      q_sharding: jax.sharding.NamedSharding | None = ...,
      k_sharding: jax.sharding.NamedSharding | None = ...,
      return_residuals: Literal[True],
  ) -> tuple[Float[Array, "*B H T d"], Residuals]:
    ...

  def __call__(
      self,
      q: Float[Array, "*B H T D"],
      k: Float[Array, "*B h t D"] | Float[Array, "*B t D"],
      v: Float[Array, "*B h t d"] | Float[Array, "*B t d"],
      mask: Bool[Array, "*#B T t"] | Mask | mask_lib.Mask | None = None,
      segment_ids: (
          SegmentIds
          | tuple[Int[Array, "*#B T"], Int[Array, "*#B t"]]
          | Int[Array, "*#B T"]
          | None
      ) = None,
      sinks: Float[Array, "*#B H"] | None = None,
      *,
      is_mqa: bool = False,
      is_causal: bool = False,
      mask_value: float = DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      max_logit_value: (
          float | Float[Array, ""] | Float[Array, "*#B #H"] | None
      ) = None,
      dropout_rate: float = 0.0,
      prng_key: jax.Array | None = None,
      q_sharding: jax.sharding.NamedSharding | None = None,
      k_sharding: jax.sharding.NamedSharding | None = None,
      return_residuals: bool = False,
  ) -> Float[Array, "*B H T d"] | tuple[Float[Array, "*B H T d"], Residuals]:
    """Performs Splash Attention."""
    params = locals().copy()
    params.pop("self")
    params.pop("q_sharding")
    params.pop("k_sharding")
    bound_args = self.bind(**params)
    mask = cast(Mask, bound_args.kwargs["mask"])
    segment_ids = cast(SegmentIds | None, bound_args.kwargs["segment_ids"])
    max_logit_value = bound_args.kwargs.get("max_logit_value", max_logit_value)

    def fwd_closed(q, k, v, mask, segment_ids, sinks, max_logit_value):
      return op.Op.__call__(
          self,
          q,
          k,
          v,
          mask=mask,
          segment_ids=segment_ids,
          sinks=sinks,
          is_mqa=is_mqa,
          mask_value=mask_value,
          attn_logits_soft_cap=attn_logits_soft_cap,
          dropout_rate=dropout_rate,
          prng_key=prng_key,
          return_residuals=return_residuals,
          **_max_logit_value_kwarg(self, max_logit_value),
      )

    if q_sharding is None:
      if k_sharding is not None:
        raise ValueError(
            "q_sharding must be specified if k_sharding is specified."
        )
      return fwd_closed(q, k, v, mask, segment_ids, sinks, max_logit_value)

    mesh, q_spec = q_sharding.mesh, q_sharding.spec

    if len(q_spec) > len(q.shape):
      raise ValueError("Too many dimensions in q_sharding's pspec.")

    q_axes = tuple(q_spec) + (None,) * (q.ndim - len(q_spec))
    *batch_axes, heads_axis, seq_q_axis, head_dim_axis = q_axes
    if seq_q_axis is not None:
      mask = dataclasses.replace(mask, q_seq_axis=seq_q_axis)

    if k_sharding is None:
      k_batch_axes = batch_axes
      k_heads_axis = heads_axis
      seq_k_axis = None
      k_head_dim_axis = head_dim_axis
    else:
      if q_sharding.mesh != k_sharding.mesh:
        raise ValueError("q_sharding and k_sharding must have the same mesh.")
      k_spec = k_sharding.spec
      if len(k_spec) > len(k.shape):
        raise ValueError("Too many dimensions in k_sharding's pspec.")
      k_axes = tuple(k_spec) + (None,) * (k.ndim - len(k_spec))
      if is_mqa and k.ndim == q.ndim - 1:
        *k_batch_axes, seq_k_axis, k_head_dim_axis = k_axes
        k_heads_axis = None
      else:
        *k_batch_axes, k_heads_axis, seq_k_axis, k_head_dim_axis = k_axes
      if _mesh_axis_size(mesh, seq_k_axis) != 1:
        raise ValueError("Sharding along seq_k_axis is unsupported.")
      if _mesh_axis_size(mesh, k_head_dim_axis) != 1:
        raise ValueError("Sharding along head_dim is unsupported.")

    if _mesh_axis_size(mesh, head_dim_axis) != 1:
      raise ValueError("Sharding along head_dim is unsupported.")

    def _create_spec(batch_axes, heads_axis, seq_axis, head_dim_axis):
      if is_mqa:
        if k.ndim == q.ndim - 1:
          axes_spec = [*batch_axes, seq_axis, head_dim_axis]
        else:
          axes_spec = [*batch_axes, None, seq_axis, head_dim_axis]
      else:
        axes_spec = [*batch_axes, heads_axis, seq_axis, head_dim_axis]
      return axes_spec

    k_axes_spec = _create_spec(
        k_batch_axes, k_heads_axis, seq_k_axis, k_head_dim_axis
    )
    v_axes_spec = _create_spec(
        k_batch_axes, k_heads_axis, seq_k_axis, k_head_dim_axis
    )

    in_axes = (
        [*batch_axes, heads_axis, seq_q_axis, head_dim_axis],
        k_axes_spec,
        v_axes_spec,
        dataclasses.replace(mask, bool_mask=[*batch_axes, seq_q_axis, seq_k_axis]),  # pyrefly: ignore[bad-argument-type]
        SegmentIds(q=[*batch_axes, seq_q_axis], kv=[*batch_axes, seq_k_axis]),  # pyrefly: ignore[bad-argument-type]
        [*batch_axes, heads_axis],
        [*batch_axes, heads_axis],
    )
    P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
    residual_spec = P(*batch_axes, heads_axis, seq_q_axis)
    out_specs = (P(*q_axes), residual_spec) if return_residuals else P(*q_axes)
    is_array = lambda x: isinstance(x, (jax.Array, np.ndarray))

    def bcast(x, axes):
      return (
          jax.lax.broadcast_to_rank(jnp.asarray(x), len(axes))
          if is_array(x)
          else x
      )

    def spec(x, axes):
      if not is_array(x):
        return None
      # Do not shard along axes that are broadcast.
      return P(*(None if d == 1 else a for a, d in zip(axes, x.shape)))

    def fwd_sharded(*args):
      is_leaf = lambda x: x is None
      args = jax.tree.map(bcast, args, in_axes, is_leaf=is_leaf)
      in_specs = jax.tree.map(spec, args, in_axes, is_leaf=is_leaf)
      return shard_map.shard_map(
          fwd_closed, mesh, in_specs, out_specs, check_rep=False
      )(*args)

    return fwd_sharded(q, k, v, mask, segment_ids, sinks, max_logit_value)

  def bind(
      self,
      q: Float[Array, "*B H T D"],
      k: Float[Array, "*B h t D"] | Float[Array, "*B t D"],
      v: Float[Array, "*B h t d"] | Float[Array, "*B t d"],
      mask: Bool[Array, "*#B T t"] | Mask | mask_lib.Mask | None = None,
      segment_ids: (
          SegmentIds
          | tuple[Int[Array, "*#B T"], Int[Array, "*#B t"]]
          | Int[Array, "*#B T"]
          | None
      ) = None,
      sinks: Float[Array, "*#B H"] | None = None,
      *,
      is_mqa: bool = False,
      mask_value: float = DEFAULT_MASK_VALUE,
      is_causal: bool = False,
      attn_logits_soft_cap: float | None = None,
      max_logit_value: (
          float | Float[Array, ""] | Float[Array, "*#B #H"] | None
      ) = None,
      dropout_rate: float = 0.0,
      prng_key: jax.Array | None = None,
      q_sharding: jax.sharding.NamedSharding | None = None,
      k_sharding: jax.sharding.NamedSharding | None = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds and validates arguments for Splash Attention."""
    if q_sharding is not None or k_sharding is not None:
      raise NotImplementedError("Sharding is not supported in `bind` yet.")

    if not (0.0 <= dropout_rate < 1.0):
      raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}.")
    if dropout_rate > 0.0 and prng_key is None:
      raise ValueError("prng_key must be provided when dropout_rate > 0.")

    if is_causal and mask is not None:
      raise ValueError(
          "Cannot specify both a mask and the is_causal parameter. Pass only"
          " one."
      )

    if is_mqa:
      if k.ndim == q.ndim and k.shape[-3] != 1:
        raise ValueError(
            "When is_mqa=True and k has a head axis, it must have 1 KV head,"
            f" got {k.shape[-3]}."
        )
    else:
      if k.ndim != q.ndim:
        raise ValueError(
            f"When is_mqa=False, k.ndim ({k.ndim}) must equal q.ndim"
            f" ({q.ndim})."
        )
      if q.shape[-3] % k.shape[-3] != 0:
        raise ValueError(
            f"num_q_heads ({q.shape[-3]}) must be divisible by num_kv_heads"
            f" ({k.shape[-3]})."
        )

    mask = _resolve_bound_mask(
        mask,
        is_causal=is_causal,
        q_seq_len=q.shape[-2],
        kv_seq_len=k.shape[-2],
        preserve_local_mask=self.supports_analytic_local_mask,
    )
    segment_ids = _canonicalize_segment_ids(segment_ids)

    return op.Op.bind(
        self,
        q,
        k,
        v,
        mask=mask,
        segment_ids=segment_ids,
        sinks=sinks,
        is_mqa=is_mqa,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        dropout_rate=dropout_rate,
        prng_key=prng_key,
        return_residuals=return_residuals,
        **_max_logit_value_kwarg(self, max_logit_value),
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      q: Float[Array, "*B H T D"],
      k: Float[Array, "*B h t D"] | Float[Array, "*B t D"],
      v: Float[Array, "*B h t d"] | Float[Array, "*B t d"],
      *,
      mask: Mask,
      segment_ids: SegmentIds | None = None,
      sinks: Float[Array, "*#B H"] | None = None,
      is_mqa: bool = False,
      mask_value: float = DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      max_logit_value: (
          float | Float[Array, ""] | Float[Array, "*#B #H"] | None
      ) = None,
      dropout_rate: float = 0.0,
      prng_key: jax.Array | None = None,
      return_residuals: bool = False,
      config: _Config,
  ) -> tuple[Float[Array, "*B H T d"], Residuals | None]:
    del config

    q_seq_len = q.shape[-2]
    kv_seq_len = k.shape[-2]
    mask_array = mask.as_array(q_seq_len, kv_seq_len)

    dropout_mask = None
    if dropout_rate > 0.0:
      assert prng_key is not None
      dropout_mask = jax.random.bernoulli(
          prng_key, dropout_rate, (*q.shape[:-1], kv_seq_len)
      )

    if is_mqa and k.ndim == q.ndim:
      k_in = jnp.squeeze(k, axis=-3)
      v_in = jnp.squeeze(v, axis=-3)
    else:
      k_in = k
      v_in = v

    broadcast_to_rank = (
        lambda x, rank: None
        if x is None
        else jax.lax.broadcast_to_rank(x, rank)
    )

    mask_array = broadcast_to_rank(mask_array, q.ndim - 1)
    if segment_ids is not None:
      segment_ids = SegmentIds(
          q=jax.lax.broadcast_to_rank(segment_ids.q, q.ndim - 2)
          if isinstance(segment_ids.q, (jax.Array, np.ndarray))
          else segment_ids.q,
          kv=jax.lax.broadcast_to_rank(segment_ids.kv, q.ndim - 2)
          if isinstance(segment_ids.kv, (jax.Array, np.ndarray))
          else segment_ids.kv,
      )
    sinks = broadcast_to_rank(sinks, q.ndim - 2)
    max_logit_val = None
    if max_logit_value is not None:
      max_logit_val = jnp.broadcast_to(
          jnp.asarray(max_logit_value, dtype=jnp.float32), q.shape[:-2]
      )

    def single_fwd(
        q, k, v, mask, segment_ids, sinks, dropout_mask, max_logit_val
    ):
      return reference.attention_reference(
          q=q,
          k=k,
          v=v,
          mask=mask,
          segment_ids=segment_ids,
          sinks=sinks,
          dropout_mask=dropout_mask,
          is_mqa=is_mqa,
          mask_value=mask_value,
          save_residuals=return_residuals,
          attn_logits_soft_cap=attn_logits_soft_cap,
          max_logit_value=max_logit_val,
          dropout_rate=dropout_rate,
      )

    out = vmap_batch_dims(single_fwd)(
        q,
        k_in,
        v_in,
        mask_array,
        segment_ids,
        sinks,
        dropout_mask,
        max_logit_val,
    )
    if return_residuals:
      out, stats = out
      residuals = (stats["max_logits"], stats["logsumexp"])
    else:
      residuals = None
    return out.astype(q.dtype), residuals


class SplashAttentionGrads(TypedDict):
  q: Float[Array, "*B H T D"]
  k: Float[Array, "*B h t D"] | Float[Array, "*B t D"]
  v: Float[Array, "*B h t d"] | Float[Array, "*B t d"]
  sinks: NotRequired[Float[Array, "*#B H"] | None]


class SplashAttentionVjp[_Config](
    op.Op[Any, SplashAttentionGrads, None, _Config, Any]
):
  """Splash attention VJP."""

  supports_analytic_local_mask: ClassVar[bool] = False

  def bind(
      self,
      residuals: Residuals,
      out: Float[Array, "*B H T d"],
      dout: Float[Array, "*B H T d"],
      q: Float[Array, "*B H T D"],
      k: Float[Array, "*B h t D"] | Float[Array, "*B t D"],
      v: Float[Array, "*B h t d"] | Float[Array, "*B t d"],
      mask: Bool[Array, "*#B T t"] | Mask | mask_lib.Mask | None = None,
      segment_ids: (
          SegmentIds
          | tuple[Int[Array, "*#B T"], Int[Array, "*#B t"]]
          | Int[Array, "*#B T"]
          | None
      ) = None,
      sinks: Float[Array, "*#B H"] | None = None,
      *,
      is_mqa: bool = False,
      mask_value: float = DEFAULT_MASK_VALUE,
      is_causal: bool = False,
      attn_logits_soft_cap: float | None = None,
      max_logit_value: (
          float | Float[Array, ""] | Float[Array, "*#B #H"] | None
      ) = None,
      dropout_rate: float = 0.0,
      prng_key: jax.Array | None = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds and validates arguments for Splash Attention VJP."""
    if not (0.0 <= dropout_rate < 1.0):
      raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}.")
    if dropout_rate > 0.0 and prng_key is None:
      raise ValueError("prng_key must be provided when dropout_rate > 0.")

    if is_causal and mask is not None:
      raise ValueError(
          "Cannot specify both a mask and the is_causal parameter. Pass only"
          " one."
      )

    if is_mqa:
      if k.ndim == q.ndim and k.shape[-3] != 1:
        raise ValueError(
            "When is_mqa=True and k has a head axis, it must have 1 KV head,"
            f" got {k.shape[-3]}."
        )
    else:
      if k.ndim != q.ndim:
        raise ValueError(
            f"When is_mqa=False, k.ndim ({k.ndim}) must equal q.ndim"
            f" ({q.ndim})."
        )
      if q.shape[-3] % k.shape[-3] != 0:
        raise ValueError(
            f"num_q_heads ({q.shape[-3]}) must be divisible by num_kv_heads"
            f" ({k.shape[-3]})."
        )

    mask = _resolve_bound_mask(
        mask,
        is_causal=is_causal,
        q_seq_len=q.shape[-2],
        kv_seq_len=k.shape[-2],
        preserve_local_mask=self.supports_analytic_local_mask,
    )
    segment_ids = _canonicalize_segment_ids(segment_ids)

    return op.Op.bind(
        self,
        residuals,
        out,
        dout,
        q,
        k,
        v,
        mask=mask,
        segment_ids=segment_ids,
        sinks=sinks,
        is_mqa=is_mqa,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        dropout_rate=dropout_rate,
        prng_key=prng_key,
        return_residuals=return_residuals,
        **_max_logit_value_kwarg(self, max_logit_value),
    )

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      residuals: Residuals,
      out: Float[Array, "*B H T d"],
      dout: Float[Array, "*B H T d"],
      q: Float[Array, "*B H T D"],
      k: Float[Array, "*B h t D"] | Float[Array, "*B t D"],
      v: Float[Array, "*B h t d"] | Float[Array, "*B t d"],
      *,
      mask: Mask,
      segment_ids: SegmentIds | None = None,
      sinks: Float[Array, "*#B H"] | None = None,
      is_mqa: bool = False,
      mask_value: float = DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      max_logit_value: (
          float | Float[Array, ""] | Float[Array, "*#B #H"] | None
      ) = None,
      dropout_rate: float = 0.0,
      prng_key: jax.Array | None = None,
      return_residuals: bool = False,
      config: _Config,
  ) -> tuple[SplashAttentionGrads, None]:
    """Computes attention VJP."""
    del config, max_logit_value

    if return_residuals:
      raise NotImplementedError("`return_residuals` not supported in VJP.")

    _, lse = residuals

    seq_len_q = q.shape[-2]
    seq_len_kv = k.shape[-2]
    mask_array = mask.as_array(seq_len_q, seq_len_kv)

    dropout_mask = None
    if dropout_rate > 0.0:
      assert prng_key is not None
      dropout_mask = jax.random.bernoulli(
          prng_key, dropout_rate, (*q.shape[:-1], seq_len_kv)
      )

    if is_mqa and k.ndim == q.ndim:
      k_in = jnp.squeeze(k, axis=-3)
      v_in = jnp.squeeze(v, axis=-3)
    else:
      k_in = k
      v_in = v

    orig_sinks_shape = None if sinks is None else sinks.shape
    broadcast_to_rank = (
        lambda x, rank: None
        if x is None
        else jax.lax.broadcast_to_rank(x, rank)
    )

    mask_array = broadcast_to_rank(mask_array, q.ndim - 1)
    if segment_ids is not None:
      segment_ids = SegmentIds(
          q=jax.lax.broadcast_to_rank(segment_ids.q, q.ndim - 2)
          if isinstance(segment_ids.q, (jax.Array, np.ndarray))
          else segment_ids.q,
          kv=jax.lax.broadcast_to_rank(segment_ids.kv, q.ndim - 2)
          if isinstance(segment_ids.kv, (jax.Array, np.ndarray))
          else segment_ids.kv,
      )
    sinks_bcast = broadcast_to_rank(sinks, q.ndim - 2)

    def single_bwd(
        dout, q, k, v, mask, segment_ids, sinks, out, lse, dropout_mask
    ):
      return reference.attention_reference_vjp(
          do=dout,
          q=q,
          k=k,
          v=v,
          mask=mask,
          segment_ids=segment_ids,
          sinks=sinks,
          o=out,
          logsumexp=lse,
          dropout_mask=dropout_mask,
          is_mqa=is_mqa,
          attn_logits_soft_cap=attn_logits_soft_cap,
          dropout_rate=dropout_rate,
      )

    dq, dk, dv, dsinks = vmap_batch_dims(single_bwd)(
        dout,
        q,
        k_in,
        v_in,
        mask_array,
        segment_ids,
        sinks_bcast,
        out,
        lse,
        dropout_mask,
    )

    if is_mqa and k.ndim == q.ndim:
      dk = jnp.expand_dims(dk, axis=-3)
      dv = jnp.expand_dims(dv, axis=-3)

    if sinks is not None and dsinks is not None:
      assert sinks_bcast is not None

      # Need to identify which axes to sum over because of broadcasting
      broadcast_sinks_axes = []
      for i, d in enumerate(sinks_bcast.shape):
        if i < len(sinks_bcast.shape) - 1 and d == 1:
          broadcast_sinks_axes.append(i)

      # sum grads over broadcasted axes
      if broadcast_sinks_axes:
        dsinks = jnp.sum(dsinks, axis=broadcast_sinks_axes)

      dsinks = dsinks.astype(sinks.dtype).reshape(orig_sinks_shape)

    grads = SplashAttentionGrads(q=dq, k=dk, v=dv, sinks=dsinks)
    return grads, None
