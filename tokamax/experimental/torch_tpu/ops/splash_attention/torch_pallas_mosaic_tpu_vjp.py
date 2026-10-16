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
"""Tokamax operator wrapper for Pallas Mosaic TPU Splash Attention VJP."""

from typing import Any
import jax
import jax.numpy as jnp
from tokamax._src.ops import op as jax_tokamax_op
from tokamax._src.ops.experimental.tpu.splash_attention import base as jax_base
from tokamax._src.ops.experimental.tpu.splash_attention import pallas_mosaic_tpu_vjp as jax_pallas_mosaic_tpu_vjp
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops.splash_attention import torch_base
import torch
from typing_extensions import override

Config = jax_pallas_mosaic_tpu_vjp.Config


class _PallasMosaicTpuSplashAttentionVjp(
    torch_base._SplashAttentionVjp[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax Splash Attention VJP Op wrapped for PyTorch."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = (
        jax_pallas_mosaic_tpu_vjp.PallasMosaicTpuSplashAttentionVjp()
    )
    self.jax_op_name = "pallas_mosaic_tpu_splash_attention_vjp"
    self.is_vjp = True
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(  # pyrefly: ignore[bad-override]
      self, config: Config | None
  ) -> tuple[int, ...] | None:
    if config is None:
      return None
    return (
        config.block_q_dkv,
        config.block_kv_dkv,
        config.block_kv_dkv_compute,
        -1 if config.dropout_block_q is None else config.dropout_block_q,
        -1 if config.dropout_block_kv is None else config.dropout_block_kv,
        int(config.use_base2_exp),
        -1 if config.dq_reduction_steps is None else config.dq_reduction_steps,
    )

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    config = config_parts[0]
    assert config is not None, "Backward config not set."
    assert len(config) == 7, (
        f"Expected 7 elements in bwd config, got {len(config)}."
    )
    return Config(
        block_q_dkv=int(config[0]),
        block_kv_dkv=int(config[1]),
        block_kv_dkv_compute=int(config[2]),
        dropout_block_q=None if config[3] < 0 else int(config[3]),
        dropout_block_kv=None if config[4] < 0 else int(config[4]),
        use_base2_exp=bool(config[5]),
        dq_reduction_steps=None if config[6] < 0 else int(config[6]),
    )

  @override
  def get_bound_args(
      self, *args: Any, **kwargs: Any
  ) -> jax_tokamax_op.BoundArguments:
    pos_args, updated_kwargs = torch_base.populate_default_fwd_kwargs(
        args, kwargs, is_vjp=True
    )
    return torch_op.TorchOp.get_bound_args(self, *pos_args, **updated_kwargs)

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
      mask: torch.Tensor | torch_base.Mask | None = None,
      segment_ids: (
          torch_base.SegmentIds | tuple[torch.Tensor, torch.Tensor] | None
      ) = None,
      sinks: torch.Tensor | None = None,
      is_mqa: bool = False,
      mask_value: float = torch_base.DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      dropout_rate: float = 0.0,
      is_causal: bool = False,
      config: Config | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    mask_tensor, effective_causal = torch_base.normalize_mask(
        mask, is_causal=is_causal
    )
    q_segment_ids, kv_segment_ids = torch_base.normalize_segment_ids(
        segment_ids
    )
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
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    if mask_value is None:
      mask_value = torch_base.DEFAULT_MASK_VALUE
    jax_mask = jax_base.Mask(bool_mask=mask_tensor, is_causal=is_causal)
    segment_ids = (
        jax_base.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
        if q_segment_ids is not None and kv_segment_ids is not None
        else None
    )
    assert config is not None, "Backward config not set."
    kernel_config = self.reconstruct_config(config)

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
        config=kernel_config,
    )
    dsinks = grads.get("sinks")
    dsinks_arr = dsinks if dsinks is not None else jnp.zeros((), dtype=q.dtype)
    return grads["q"], grads["k"], grads["v"], dsinks_arr


# Singleton instance of Pallas Mosaic TPU Splash Attention VJP.
PallasMosaicTpuSplashAttentionVjp = (  # pylint: disable=invalid-name
    _PallasMosaicTpuSplashAttentionVjp()
)
