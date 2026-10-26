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
from typing import Any, Final, NotRequired, TypeAlias, TypeVar, TypedDict, cast, override
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from tokamax._src import batching
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.splash_attention import reference
from tokamax._src.ops.splash_attention import splash_attention_mask as mask_lib

_Config = TypeVar("_Config")

DEFAULT_MASK_VALUE = reference.DEFAULT_MASK_VALUE
SegmentIds = reference.SegmentIds
SplashCustomReturnType = reference.SplashCustomReturnType
SplashResidualsType = reference.SplashResidualsType
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class Mask:
  """An attention mask dataclass."""

  bool_mask: Bool[jax.Array, "*#B T t"] | None = None
  _: dataclasses.KW_ONLY
  is_causal: bool = dataclasses.field(default=False, metadata=dict(static=True))

  def as_array(
      self,
      q_seq_len: int,
      kv_seq_len: int,
  ) -> Bool[Array, "*#B T t"]:
    """Returns the mask as a 2D boolean array."""

    q_idx = jnp.arange(q_seq_len)[:, None]
    kv_idx = jnp.arange(kv_seq_len)[None, :]

    if self.bool_mask is not None and self.is_causal:
      return jnp.logical_and(jnp.asarray(self.bool_mask), q_idx >= kv_idx)

    if self.bool_mask is not None:
      return jnp.asarray(self.bool_mask)

    if self.is_causal:
      return q_idx >= kv_idx

    return jnp.ones((q_seq_len, kv_seq_len), dtype=jnp.bool_)

  def __bool__(self) -> bool:
    return self.bool_mask is not None or self.is_causal


CAUSAL_MASK: Final[Mask] = Mask(is_causal=True)
FULL_MASK: Final[Mask] = Mask()


def _canonicalize_mask(
    mask: Bool[Array, "*#B T t"] | Mask | mask_lib.Mask | None,
) -> Mask:
  """Canonicalizes a mask argument."""
  if mask is None:
    return Mask()
  if isinstance(mask, (jax.Array, np.ndarray)):
    return Mask(bool_mask=mask)
  if isinstance(mask, mask_lib.CausalMask):
    return Mask(is_causal=True)
  if isinstance(mask, mask_lib.FullMask):
    return Mask()
  if isinstance(mask, mask_lib.Mask):
    return Mask(bool_mask=jnp.asarray(mask[:, :]))
  if isinstance(mask, Mask):
    return mask
  raise TypeError(f"Unsupported mask type: {type(mask)}")


@dataclasses.dataclass(frozen=True)
class SplashAttention[_Config](
    op.Op[Any, Float[Array, "*B H T d"], Residuals, _Config, Any]
):
  """Tokamax operator template for Splash Attention."""

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(
          self,
          "vjp",
          SplashAttentionVjp(),
      )

  def bind(
      self,
      q: Float[Array, "*B H T D"],
      k: Float[Array, "*B h t D"] | Float[Array, "*B t D"],
      v: Float[Array, "*B h t d"] | Float[Array, "*B t d"],
      mask: Bool[Array, "*#B T t"] | Mask | mask_lib.Mask | None = None,
      segment_ids: SegmentIds | None = None,
      sinks: Float[Array, "*#B H"] | None = None,
      *,
      is_mqa: bool = False,
      mask_value: float = DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      dropout_rate: float = 0.0,
      prng_key: jax.Array | None = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds and validates arguments for Splash Attention."""

    if not (0.0 <= dropout_rate < 1.0):
      raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}.")
    if dropout_rate > 0.0 and prng_key is None:
      raise ValueError("prng_key must be provided when dropout_rate > 0.")

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

    mask = _canonicalize_mask(mask)

    return super().bind(
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

    def single_fwd(q, k, v, mask, segment_ids, sinks, dropout_mask):
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
          dropout_rate=dropout_rate,
      )

    out = vmap_batch_dims(single_fwd)(
        q, k_in, v_in, mask_array, segment_ids, sinks, dropout_mask
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

  def bind(
      self,
      residuals: Residuals,
      out: Float[Array, "*B H T d"],
      dout: Float[Array, "*B H T d"],
      q: Float[Array, "*B H T D"],
      k: Float[Array, "*B h t D"] | Float[Array, "*B t D"],
      v: Float[Array, "*B h t d"] | Float[Array, "*B t d"],
      mask: Bool[Array, "*#B T t"] | Mask | mask_lib.Mask | None = None,
      segment_ids: SegmentIds | None = None,
      sinks: Float[Array, "*#B H"] | None = None,
      *,
      is_mqa: bool = False,
      mask_value: float = DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      dropout_rate: float = 0.0,
      prng_key: jax.Array | None = None,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds and validates arguments for Splash Attention VJP."""
    if not (0.0 <= dropout_rate < 1.0):
      raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}.")
    if dropout_rate > 0.0 and prng_key is None:
      raise ValueError("prng_key must be provided when dropout_rate > 0.")

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

    mask = _canonicalize_mask(mask)

    return super().bind(
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
      dropout_rate: float = 0.0,
      prng_key: jax.Array | None = None,
      return_residuals: bool = False,
      config: _Config,
  ) -> tuple[SplashAttentionGrads, None]:
    """Computes attention VJP."""
    del config

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
