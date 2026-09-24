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

import dataclasses
from typing import Any, Final, NotRequired, TypeAlias, TypeVar, TypedDict, override
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.splash_attention import reference
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib

_Config = TypeVar("_Config")

DEFAULT_MASK_VALUE = reference.DEFAULT_MASK_VALUE
SegmentIds = reference.SegmentIds
SplashCustomReturnType = reference.SplashCustomReturnType
SplashResidualsType = reference.SplashResidualsType
attention_reference = reference.attention_reference
attention_reference_vjp = reference.attention_reference_vjp

Residuals: TypeAlias = tuple[
    Float[Array, "num_q_heads q_seq_len"],
    Float[Array, "num_q_heads q_seq_len"],
]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class Mask:
  """An attention mask dataclass."""

  bool_mask: Bool[jax.Array, "q_seq_len kv_seq_len"] | None = None
  _: dataclasses.KW_ONLY
  is_causal: bool = dataclasses.field(default=False, metadata=dict(static=True))

  def as_array(
      self,
      q_seq_len: int,
      kv_seq_len: int,
  ) -> Bool[Array, "q_seq_len kv_seq_len"]:
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
    mask: Bool[Array, "q_seq_len kv_seq_len"] | Mask | mask_lib.Mask | None,
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
class SplashAttention[_Config](op.Op[Any, jax.Array, Residuals, _Config, Any]):
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
      q: Float[Array, "num_q_heads q_seq_len head_dim_qk"],
      k: Float[Array, "..."],
      v: Float[Array, "..."],
      mask: (
          Bool[Array, "q_seq_len kv_seq_len"] | Mask | mask_lib.Mask | None
      ) = None,
      segment_ids: SegmentIds | None = None,
      sinks: Float[Array, "..."] | None = None,
      *,
      is_mqa: bool = False,
      mask_value: float = DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      dropout_rate: float = 0.0,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds and validates arguments for Splash Attention."""

    if not (0.0 <= dropout_rate < 1.0):
      raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}.")

    if not is_mqa and k.ndim == 3 and q.ndim == 3:
      if q.shape[0] % k.shape[0] != 0:
        raise ValueError(
            f"num_q_heads ({q.shape[0]}) must be divisible by num_kv_heads"
            f" ({k.shape[0]})."
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
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      q: Float[Array, "num_q_heads q_seq_len head_dim_qk"],
      k: Float[Array, "..."],
      v: Float[Array, "..."],
      *,
      mask: Mask,
      segment_ids: SegmentIds | None = None,
      sinks: Float[Array, "..."] | None = None,
      is_mqa: bool = False,
      mask_value: float = DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      dropout_rate: float = 0.0,
      return_residuals: bool = False,
      config: _Config,
  ) -> tuple[jax.Array, Residuals | None]:
    del config

    q_seq_len = q.shape[1]
    kv_seq_len = k.shape[0] if is_mqa and k.ndim == 2 else k.shape[1]
    mask_array = mask.as_array(q_seq_len, kv_seq_len)

    if is_mqa and k.ndim == 3:
      k_in = k[0]
      v_in = v[0]
    else:
      k_in = k
      v_in = v

    out = reference.attention_reference(
        q=q,
        k=k_in,
        v=v_in,
        mask=mask_array,
        segment_ids=segment_ids,
        sinks=sinks,
        dropout_mask=None,
        is_mqa=is_mqa,
        mask_value=mask_value,
        save_residuals=return_residuals,
        attn_logits_soft_cap=attn_logits_soft_cap,
        dropout_rate=dropout_rate,
    )
    if return_residuals:
      out, stats = out
      residuals = (stats["max_logits"], stats["logsumexp"])
    else:
      residuals = None
    return out.astype(q.dtype), residuals


class SplashAttentionGrads(TypedDict):
  q: Float[Array, "num_q_heads q_seq_len head_dim_qk"]
  k: Float[Array, "..."]
  v: Float[Array, "..."]
  sinks: NotRequired[Float[Array, "..."] | None]


class SplashAttentionVjp[_Config](
    op.Op[Any, SplashAttentionGrads, None, _Config, Any]
):
  """Splash attention VJP."""

  def bind(
      self,
      residuals: Residuals,
      out: Float[Array, "num_q_heads q_seq_len head_dim_v"],
      dout: Float[Array, "num_q_heads q_seq_len head_dim_v"],
      q: Float[Array, "num_q_heads q_seq_len head_dim_qk"],
      k: Float[Array, "..."],
      v: Float[Array, "..."],
      mask: (
          Bool[Array, "q_seq_len kv_seq_len"] | Mask | mask_lib.Mask | None
      ) = None,
      segment_ids: SegmentIds | None = None,
      sinks: Float[Array, "..."] | None = None,
      *,
      is_mqa: bool = False,
      mask_value: float = DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      dropout_rate: float = 0.0,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds and validates arguments for Splash Attention VJP."""
    if not (0.0 <= dropout_rate < 1.0):
      raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}.")

    if not is_mqa and k.ndim == 3 and q.ndim == 3:
      if q.shape[0] % k.shape[0] != 0:
        raise ValueError(
            f"num_q_heads ({q.shape[0]}) must be divisible by num_kv_heads"
            f" ({k.shape[0]})."
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
        return_residuals=return_residuals,
    )

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      residuals: Residuals,
      out: Float[Array, "num_q_heads q_seq_len head_dim_v"],
      dout: Float[Array, "num_q_heads q_seq_len head_dim_v"],
      q: Float[Array, "num_q_heads q_seq_len head_dim_qk"],
      k: Float[Array, "..."],
      v: Float[Array, "..."],
      *,
      mask: Mask,
      segment_ids: SegmentIds | None = None,
      sinks: Float[Array, "..."] | None = None,
      is_mqa: bool = False,
      mask_value: float = DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      dropout_rate: float = 0.0,
      return_residuals: bool = False,
      config: _Config,
  ) -> tuple[SplashAttentionGrads, None]:
    """Computes attention VJP."""
    del config

    if return_residuals:
      raise NotImplementedError("`return_residuals` not supported.")

    _, lse = residuals

    seq_len_q = q.shape[1]
    seq_len_kv = k.shape[0] if is_mqa and k.ndim == 2 else k.shape[1]
    mask_array = mask.as_array(seq_len_q, seq_len_kv)

    if is_mqa and k.ndim == 3:
      k_in = k[0]
      v_in = v[0]
    else:
      k_in = k
      v_in = v

    dq, dk, dv, dsinks = reference.attention_reference_vjp(
        do=dout,
        q=q,
        k=k_in,
        v=v_in,
        mask=mask_array,
        segment_ids=segment_ids,
        sinks=sinks,
        o=out,
        logsumexp=lse,
        is_mqa=is_mqa,
        attn_logits_soft_cap=attn_logits_soft_cap,
        dropout_rate=dropout_rate,
    )

    if is_mqa and k.ndim == 3:
      dk = dk.reshape(k.shape)
      dv = dv.reshape(v.shape)

    grads = SplashAttentionGrads(q=dq, k=dk, v=dv, sinks=dsinks)
    return grads, None
