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
"""Tokamax operator wrapper for Pallas Mosaic TPU Splash Attention."""

from collections.abc import Sequence
import dataclasses
from typing import Any
import jax
from tokamax._src.ops.experimental.tpu.splash_attention import base as jax_base
from tokamax._src.ops.experimental.tpu.splash_attention import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops.splash_attention import torch_base
from tokamax.experimental.torch_tpu.ops.splash_attention import torch_pallas_mosaic_tpu_vjp
import torch
from typing_extensions import override

Config = jax_pallas_mosaic_tpu.Config
VjpConfig = torch_pallas_mosaic_tpu_vjp.Config
QKVLayout = jax_pallas_mosaic_tpu.QKVLayout
PallasMosaicTpuSplashAttentionVjp = (
    torch_pallas_mosaic_tpu_vjp.PallasMosaicTpuSplashAttentionVjp
)


class _PallasMosaicTpuSplashAttention(
    torch_base._SplashAttention[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax Splash Attention Op wrapped for PyTorch."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasMosaicTpuSplashAttention()
    self.backward_op_torch = PallasMosaicTpuSplashAttentionVjp
    self.jax_op_name = "pallas_mosaic_tpu_splash_attention"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(  # pyrefly: ignore[bad-override]
      self, config: Config | None
  ) -> tuple[int, ...] | None:
    if config is None:
      return None
    assert len(dataclasses.fields(config)) == 13, (
        "Expected 13 fields in fwd config, got"
        f" {len(dataclasses.fields(config))}."
    )
    return (
        config.block_q,
        config.block_kv,
        config.block_kv_compute,
        -1 if config.dropout_block_q is None else config.dropout_block_q,
        -1 if config.dropout_block_kv is None else config.dropout_block_kv,
        int(config.q_layout.value),
        int(config.k_layout.value),
        int(config.v_layout.value),
        int(config.use_experimental_scheduler),
        int(config.use_base2_exp),
        int(config.qk_diag_skip),
        config.qk_diag_grid,
        int(config.sv_diag_skip),
    )

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    config = config_parts[0]
    assert config is not None, "Forward config not set."
    assert len(config) == 13, (
        f"Expected 13 elements in fwd config, got {len(config)}."
    )
    return Config(
        block_q=int(config[0]),
        block_kv=int(config[1]),
        block_kv_compute=int(config[2]),
        dropout_block_q=None if config[3] < 0 else int(config[3]),
        dropout_block_kv=None if config[4] < 0 else int(config[4]),
        q_layout=QKVLayout(int(config[5])),
        k_layout=QKVLayout(int(config[6])),
        v_layout=QKVLayout(int(config[7])),
        use_experimental_scheduler=bool(config[8]),
        use_base2_exp=bool(config[9]),
        qk_diag_skip=bool(config[10]),
        qk_diag_grid=int(config[11]),
        sv_diag_skip=bool(config[12]),
    )

  @override
  def __call__(
      self,
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
      return_residuals: bool = False,
      configs: tuple[Any, Any] | None = None,
  ) -> torch.Tensor:
    del return_residuals
    mask_tensor, effective_causal = torch_base.normalize_mask(mask)
    q_segment_ids, kv_segment_ids = torch_base.normalize_segment_ids(
        segment_ids
    )
    fwd_config, bwd_config = (None, None) if configs is None else configs
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
    del return_residuals, bwd_config
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
    assert config is not None, "Forward config not set."
    kernel_config = self.reconstruct_config(config)

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
        config=kernel_config,
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
    """Callback for torch register_autograd."""
    super().setup_context(ctx, inputs, output)

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
    mask = torch_base.Mask(bool_mask=mask_tensor, is_causal=ctx.is_causal)
    segment_ids = (
        torch_base.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
        if q_segment_ids is not None and kv_segment_ids is not None
        else None
    )
    bwd_config = (
        None
        if ctx.bwd_config is None
        else self.backward_op_torch.reconstruct_config(ctx.bwd_config)
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
        config=bwd_config,
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


# Singleton instance of Pallas Mosaic TPU Splash Attention.
PallasMosaicTpuSplashAttention = (  # pylint: disable=invalid-name
    _PallasMosaicTpuSplashAttention()
)
