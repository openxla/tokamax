# Copyright 2026 Google LLC
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
"""Splash Attention PyTorch Op API using reference implementation."""

from collections.abc import Sequence
import dataclasses
from typing import Any, Final, TypeVar, cast
import jax
import jax.numpy as jnp
from tokamax._src.ops import op as jax_tokamax_op
from tokamax._src.ops.experimental.tpu.splash_attention import base as jax_base
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops import torch_utils
import torch
from typing_extensions import override

_Config = TypeVar("_Config")

DEFAULT_MASK_VALUE = jax_base.DEFAULT_MASK_VALUE


@dataclasses.dataclass(frozen=True, slots=True)
class SegmentIds:
  """SegmentIds for Q and KV sequences in PyTorch."""

  q: torch.Tensor
  kv: torch.Tensor


@dataclasses.dataclass(frozen=True, slots=True)
class Mask:
  """An attention mask dataclass for PyTorch."""

  bool_mask: torch.Tensor | None = None
  _: dataclasses.KW_ONLY
  is_causal: bool = False

  def __bool__(self) -> bool:
    return self.bool_mask is not None or self.is_causal


CAUSAL_MASK: Final[Mask] = Mask(is_causal=True)
FULL_MASK: Final[Mask] = Mask()


def normalize_mask(
    mask: torch.Tensor | Mask | None,
    is_causal: bool = False,
) -> tuple[torch.Tensor | None, bool]:
  """Splits a PyTorch mask input into `(mask_tensor, is_causal)` for custom_op."""
  if mask is None:
    return None, is_causal
  if isinstance(mask, torch.Tensor):
    return mask, is_causal
  if isinstance(mask, Mask):
    return mask.bool_mask, mask.is_causal or is_causal
  raise TypeError(f"Unsupported mask type: {type(mask)}")


def normalize_segment_ids(
    segment_ids: SegmentIds | tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
  """Splits PyTorch `segment_ids` into `(q_segment_ids, kv_segment_ids)`."""
  if segment_ids is None:
    return None, None
  if isinstance(segment_ids, SegmentIds):
    return segment_ids.q, segment_ids.kv
  if isinstance(segment_ids, tuple) and len(segment_ids) == 2:
    return segment_ids
  raise TypeError(f"Unsupported segment_ids type: {type(segment_ids)}")


def populate_default_fwd_kwargs(
    args: Sequence[Any],
    kwargs: dict[str, Any],
    is_vjp: bool = False,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Normalizes SplashAttention arguments for `TorchOp.get_bound_args`.

    `TorchOp.get_bound_args` inspects `op_impl_jax._fwd` directly and already
    populates parameter defaults and converts `torch.Tensor` / `jax.Array`
    leaves
    (including inside registered JAX PyTree dataclasses) into
    `jax.ShapeDtypeStruct`. However, SplashAttention still needs this adapter
    before `TorchOp.get_bound_args` because `_fwd` expects composite JAX types
    (`mask: jax_base.Mask`, `segment_ids: jax_base.SegmentIds | None`), whereas
    `get_bound_args` can be called from two different entry points:

    1. From `op_impl_call_config_setup` (`op_impl_call` signature), where `mask`
       and `segment_ids` have been flattened into `(mask_tensor, is_causal)` and
       `(q_segment_ids, kv_segment_ids)` for PyTorch `custom_op` compatibility.
    2. Directly with user-facing `__call__` arguments (e.g., from
       `torch_utils.get_configs`), where `mask` (optional, defaulting to `None`)
       and `segment_ids` are passed as PyTorch types (`torch.Tensor`, `Mask`,
       `SegmentIds`, or `tuple`) rather than JAX dataclasses.

  To Summarize:

  We're creating a `BoundArguments` object based on the inputs to the `__call__`
  method and then using this `BoundArguments` object to get a config. Because
  SplashAttention has arguments that are JAX-specific composite types (`mask`
  and `segment_ids`), they can't be passed directly so we do the same trick that
  we do with Configs - break these tyes down to their primitive types and then
  pass them into `_op_impl_call`.

  However, there is a differenec between the two call scenarios:

  1. Users call `get_config` from outside of `__call_` and passes in `Mask` and
  `SegmentIDs` objects. We need to create the `BoundArguments` using these two
  paramters. In this scenario we `normalize` them into their JAX equivalents and
  pass them into `get_bound_args`.

  2. `op_impl_call_config_setup` calls `get_config`. At this stage, there is no
  `Mask` and `SegmentIDs` objects -they have been deconstructed to their
  primitive types so that they can be passed into `op_impl_call`. In this
  scenario we take the primitive types and reconstruct the JAX Mask and
  SegmentID objects to create a `BoundArguments`.

  That's what `populate_default_fwd_kwargs` is trying to account for - does it
  need to handle scenario 1 or scenario 2? Both are accomplishing the same goal
  of getting a `BoundArguments` for the sake of getting a config, but what
  `populate_default_fwd_kwargs` needs to reconstruct into a `BoundArguments` is
  different.
  """
  # Case 1: Invoked from `op_impl_call_config_setup` with flattened
  # `op_impl_call` arguments (`mask_tensor`, `q_segment_ids`, etc.).
  if "mask_tensor" in kwargs or len(args) >= 7:
    op_call_names = (
        (
            "max_logits",
            "lse",
            "out",
            "dout",
            "q",
            "k",
            "v",
            "mask_tensor",
            "q_segment_ids",
            "kv_segment_ids",
            "sinks",
            "is_causal",
            "is_mqa",
            "mask_value",
            "attn_logits_soft_cap",
            "dropout_rate",
            "return_residuals",
            "config",
        )
        if is_vjp
        else (
            "q",
            "k",
            "v",
            "mask_tensor",
            "q_segment_ids",
            "kv_segment_ids",
            "sinks",
            "is_causal",
            "is_mqa",
            "mask_value",
            "attn_logits_soft_cap",
            "dropout_rate",
            "return_residuals",
            "config",
            "bwd_config",
        )
    )
    bound = dict(zip(op_call_names, args), **kwargs)
    mask_tensor = bound.get("mask_tensor")
    is_causal = bound.get("is_causal", False)
    q_seg = bound.get("q_segment_ids")
    kv_seg = bound.get("kv_segment_ids")
  # Case 2: Invoked directly with user-facing `__call__` arguments
  # (e.g., from `torch_utils.get_configs`).
  else:
    call_names = ("q", "k", "v", "mask", "segment_ids", "sinks")
    bound = dict(zip(call_names, args), **kwargs)
    mask_tensor, is_causal = normalize_mask(
        bound.get("mask"), is_causal=bound.pop("is_causal", False)
    )
    q_seg, kv_seg = normalize_segment_ids(bound.get("segment_ids"))

  bound["mask"] = jax_base.Mask(
      bool_mask=cast(Any, mask_tensor), is_causal=is_causal
  )
  bound["segment_ids"] = (
      jax_base.SegmentIds(q=cast(Any, q_seg), kv=cast(Any, kv_seg))
      if q_seg is not None and kv_seg is not None
      else None
  )
  if bound.get("mask_value") is None:
    bound.pop("mask_value", None)
  bound.pop("return_residuals", None)
  return (bound.pop("q"), bound.pop("k"), bound.pop("v")), bound


class _SplashAttentionVjp(torch_op.TorchOp[_Config]):
  """Splash Attention PyTorch VJP Op API using reference implementation."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.SplashAttentionVjp()
    self.jax_op_name = "base_splash_attention_vjp"
    self.is_vjp = True
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      max_logits: torch.Tensor,
      lse: torch.Tensor,
      out: torch.Tensor,
      dout: torch.Tensor,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      mask_tensor: torch.Tensor | None = None,
      q_segment_ids: torch.Tensor | None = None,
      kv_segment_ids: torch.Tensor | None = None,
      sinks: torch.Tensor | None = None,
      is_causal: bool = False,
      is_mqa: bool = False,
      mask_value: float | None = DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      dropout_rate: float = 0.0,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del max_logits, lse, out, dout, mask_tensor, q_segment_ids, kv_segment_ids
    del is_causal, is_mqa, mask_value, attn_logits_soft_cap, dropout_rate
    del return_residuals, config
    dsinks = torch.empty_like(sinks) if sinks is not None else q.new_empty(())
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        dsinks,
    )

  @override
  def get_bound_args(
      self, *args: Any, **kwargs: Any
  ) -> jax_tokamax_op.BoundArguments:
    pos_args, updated_kwargs = populate_default_fwd_kwargs(
        args, kwargs, is_vjp=True
    )
    return super().get_bound_args(*pos_args, **updated_kwargs)

  @override
  def __call__(
      self,
      max_logits: torch.Tensor,
      lse: torch.Tensor,
      out: torch.Tensor,
      dout: torch.Tensor,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      mask: torch.Tensor | Mask | None = None,
      segment_ids: SegmentIds | tuple[torch.Tensor, torch.Tensor] | None = None,
      sinks: torch.Tensor | None = None,
      is_mqa: bool = False,
      mask_value: float = DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      dropout_rate: float = 0.0,
      is_causal: bool = False,
      config: Any = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    mask_tensor, effective_causal = normalize_mask(mask, is_causal=is_causal)
    q_segment_ids, kv_segment_ids = normalize_segment_ids(segment_ids)
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    dq, dk, dv, dsinks_arr = self._torch_tokamax_op(
        max_logits,
        lse,
        out,
        dout,
        q,
        k,
        v,
        mask_tensor,
        q_segment_ids,
        kv_segment_ids,
        sinks,
        is_causal=effective_causal,
        is_mqa=is_mqa,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        dropout_rate=dropout_rate,
        return_residuals=False,
        config=self.deconstruct_config(config),
    )
    dsinks = dsinks_arr if sinks is not None else None
    return dq, dk, dv, dsinks

  @override
  def op_impl_call(
      self,
      max_logits: jax.Array,
      lse: jax.Array,
      out: jax.Array,
      dout: jax.Array,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      mask_tensor: jax.Array | None = None,
      q_segment_ids: jax.Array | None = None,
      kv_segment_ids: jax.Array | None = None,
      sinks: jax.Array | None = None,
      is_causal: bool = False,
      is_mqa: bool = False,
      mask_value: float | None = None,
      attn_logits_soft_cap: float | None = None,
      dropout_rate: float = 0.0,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    del config
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    if mask_value is None:
      mask_value = DEFAULT_MASK_VALUE
    jax_mask = jax_base.Mask(bool_mask=mask_tensor, is_causal=is_causal)
    segment_ids = (
        jax_base.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
        if q_segment_ids is not None and kv_segment_ids is not None
        else None
    )
    grads, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        (max_logits, lse),
        out,
        dout,
        q,
        k,
        v,
        mask=jax_mask,
        segment_ids=segment_ids,
        sinks=sinks,
        is_mqa=is_mqa,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        dropout_rate=dropout_rate,
        return_residuals=return_residuals,
        config=None,
    )
    dsinks = grads.get("sinks")
    dsinks_arr = dsinks if dsinks is not None else jnp.zeros((), dtype=q.dtype)
    return grads["q"], grads["k"], grads["v"], dsinks_arr

  @override
  def derive_backward_shapes(
      self, abstract_argument_dict: dict[str, Any]
  ) -> dict[str, Any]:
    """Derives the backward shapes for the SplashAttention VJP pass."""
    if not abstract_argument_dict:
      empty_struct = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)
      return {
          "residuals": (empty_struct, empty_struct),
          "out": empty_struct,
          "dout": empty_struct,
      }

    q = abstract_argument_dict["q"]
    v = abstract_argument_dict["v"]
    res_shape = q.shape[:-1]
    out_shape = (*q.shape[:-1], v.shape[-1])
    res_struct = jax.ShapeDtypeStruct(shape=res_shape, dtype=jnp.float32)
    out_struct = jax.ShapeDtypeStruct(shape=out_shape, dtype=q.dtype)
    return {
        "residuals": (res_struct, res_struct),
        "out": out_struct,
        "dout": out_struct,
    }


# Singleton instance of SplashAttentionVjp.
SplashAttentionVjp = _SplashAttentionVjp[Any]()


class _SplashAttention(torch_op.TorchOp[_Config]):
  """Splash Attention PyTorch Op API using reference implementation."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.SplashAttention()
    self.backward_op_torch = SplashAttentionVjp
    self.jax_op_name = "base_splash_attention"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      mask_tensor: torch.Tensor | None = None,
      q_segment_ids: torch.Tensor | None = None,
      kv_segment_ids: torch.Tensor | None = None,
      sinks: torch.Tensor | None = None,
      is_causal: bool = False,
      is_mqa: bool = False,
      mask_value: float | None = DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      dropout_rate: float = 0.0,
      return_residuals: bool = True,
      config: tuple[int, ...] | None = None,
      bwd_config: tuple[int, ...] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del mask_tensor, q_segment_ids, kv_segment_ids, sinks, is_causal, is_mqa
    del mask_value, attn_logits_soft_cap, dropout_rate, return_residuals
    del config, bwd_config
    assert self.backward_op_torch is not None, "Backward op not set."
    derived = self.backward_op_torch.derive_backward_shapes({
        "q": torch_utils.convert_torch_to_jax_via_meta(q),
        "k": torch_utils.convert_torch_to_jax_via_meta(k),
        "v": torch_utils.convert_torch_to_jax_via_meta(v),
    })
    out_shape = derived["out"].shape
    max_logits_struct, lse_struct = derived["residuals"]
    return (
        q.new_empty(out_shape),
        q.new_empty(max_logits_struct.shape, dtype=torch.float32),
        q.new_empty(lse_struct.shape, dtype=torch.float32),
    )

  @override
  def get_bound_args(
      self, *args: Any, **kwargs: Any
  ) -> jax_tokamax_op.BoundArguments:
    pos_args, updated_kwargs = populate_default_fwd_kwargs(
        args, kwargs, is_vjp=False
    )
    return super().get_bound_args(*pos_args, **updated_kwargs)

  @override
  def __call__(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      mask: torch.Tensor | Mask | None = None,
      segment_ids: SegmentIds | tuple[torch.Tensor, torch.Tensor] | None = None,
      sinks: torch.Tensor | None = None,
      is_mqa: bool = False,
      mask_value: float = DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      dropout_rate: float = 0.0,
      return_residuals: bool = False,
      configs: tuple[Any, Any] | None = None,
  ) -> torch.Tensor:
    del return_residuals
    fwd_config, bwd_config = (None, None) if configs is None else configs
    mask_tensor, effective_causal = normalize_mask(mask)
    q_segment_ids, kv_segment_ids = normalize_segment_ids(segment_ids)
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    assert self.backward_op_torch is not None, "Backward op not set."
    out, _, _ = self._torch_tokamax_op(
        q,
        k,
        v,
        mask_tensor,
        q_segment_ids,
        kv_segment_ids,
        sinks,
        is_causal=effective_causal,
        is_mqa=is_mqa,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        dropout_rate=dropout_rate,
        return_residuals=True,
        config=self.deconstruct_config(fwd_config),
        bwd_config=self.backward_op_torch.deconstruct_config(bwd_config),
    )
    return out

  @override
  def op_impl_call(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      mask_tensor: jax.Array | None = None,
      q_segment_ids: jax.Array | None = None,
      kv_segment_ids: jax.Array | None = None,
      sinks: jax.Array | None = None,
      is_causal: bool = False,
      is_mqa: bool = False,
      mask_value: float | None = None,
      attn_logits_soft_cap: float | None = None,
      dropout_rate: float = 0.0,
      return_residuals: bool = True,
      config: tuple[int, ...] | None = None,
      bwd_config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    del return_residuals, config, bwd_config
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    if mask_value is None:
      mask_value = DEFAULT_MASK_VALUE
    jax_mask = jax_base.Mask(bool_mask=mask_tensor, is_causal=is_causal)
    segment_ids = (
        jax_base.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
        if q_segment_ids is not None and kv_segment_ids is not None
        else None
    )
    out, residuals = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        q,
        k,
        v,
        mask=jax_mask,
        segment_ids=segment_ids,
        sinks=sinks,
        is_mqa=is_mqa,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        dropout_rate=dropout_rate,
        return_residuals=True,
        config=None,
    )
    assert residuals is not None
    max_logits, lse = residuals
    return out, max_logits, lse

  @override
  def setup_context(
      self,
      ctx: Any,
      inputs: Sequence[Any],
      output: Any,
  ) -> None:
    """Saves tensors to the context for the backward pass."""
    (
        q,
        k,
        v,
        mask_tensor,
        q_segment_ids,
        kv_segment_ids,
        sinks,
        is_causal,
        is_mqa,
        mask_value,
        attn_logits_soft_cap,
        dropout_rate,
        _,
        _,
        bwd_config,
    ) = inputs
    out, max_logits, lse = output
    ctx.save_for_backward(
        q,
        k,
        v,
        out,
        max_logits,
        lse,
        mask_tensor,
        q_segment_ids,
        kv_segment_ids,
        sinks,
    )
    ctx.is_causal = is_causal
    ctx.is_mqa = is_mqa
    ctx.mask_value = mask_value
    ctx.attn_logits_soft_cap = attn_logits_soft_cap
    ctx.dropout_rate = dropout_rate
    ctx.bwd_config = bwd_config

  @override
  def backward(
      self,
      ctx: Any,
      dout: torch.Tensor,
      d_max_logits: torch.Tensor,
      d_lse: torch.Tensor,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      None,
      None,
      None,
      torch.Tensor | None,
      None,
      None,
      None,
      None,
      None,
      None,
      None,
      None,
  ]:
    """Callback for torch register_autograd."""
    del d_max_logits, d_lse
    assert self.backward_op_torch is not None, "Backward op not set."
    (
        q,
        k,
        v,
        out,
        max_logits,
        lse,
        mask_tensor,
        q_segment_ids,
        kv_segment_ids,
        sinks,
    ) = ctx.saved_tensors
    mask = Mask(bool_mask=mask_tensor, is_causal=ctx.is_causal)
    segment_ids = (
        SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
        if q_segment_ids is not None and kv_segment_ids is not None
        else None
    )
    dq, dk, dv, dsinks = self.backward_op_torch(
        max_logits,
        lse,
        out,
        dout,
        q,
        k,
        v,
        mask=mask,
        segment_ids=segment_ids,
        sinks=sinks,
        is_mqa=ctx.is_mqa,
        mask_value=ctx.mask_value,
        attn_logits_soft_cap=ctx.attn_logits_soft_cap,
        dropout_rate=ctx.dropout_rate,
        is_causal=ctx.is_causal,
        config=ctx.bwd_config,
    )
    return (
        dq,
        dk,
        dv,
        None,
        None,
        None,
        dsinks,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


# Singleton instance of SplashAttention.
SplashAttention = _SplashAttention[Any]()
