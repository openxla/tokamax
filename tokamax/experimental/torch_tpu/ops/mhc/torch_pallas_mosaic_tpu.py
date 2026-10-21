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
"""Tokamax operator wrappers for Pallas Mosaic TPU mHC."""

import dataclasses
from typing import Any, override

import jax
from tokamax._src.ops.mhc import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops.mhc import torch_base
import torch

Config = jax_pallas_mosaic_tpu.Config


def _deconstruct_config(config: Config | int | None) -> int | None:
  if config is None:
    return None
  if isinstance(config, Config):
    assert len(dataclasses.fields(config)) == 1, (
        f"Expected 1 field in Config, got {len(dataclasses.fields(config))}."
    )
    return int(config.token_block_size)
  return int(config)


def _reconstruct_config(*config_parts: Any) -> Config:
  assert config_parts and config_parts[0] is not None, (
      "Forward config not set."
  )
  return Config(token_block_size=int(config_parts[0]))


class _PallasMosaicTpuMhcPre(
    torch_base._MhcPre[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax MhcPre Op wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuMhcPre()
    self.jax_op_name = "pallas_mosaic_tpu_mhc_pre"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(  # pyrefly: ignore[bad-override]
      self, config: Config | int | None
  ) -> int | None:
    return _deconstruct_config(config)

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    return _reconstruct_config(*config_parts)

  @override
  def _fake_impl(
      self,
      residual: torch.Tensor,
      fn: torch.Tensor,
      hc_scale: torch.Tensor,
      hc_base: torch.Tensor,
      rms_eps: float,
      hc_pre_eps: float,
      hc_sinkhorn_eps: float,
      hc_post_mult_value: float,
      sinkhorn_repeat: int,
      return_residuals: bool = False,
      config: int | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return super()._fake_impl(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        return_residuals=return_residuals,
        config=config,
    )

  @override
  def __call__(
      self,
      residual: torch.Tensor,
      fn: torch.Tensor,
      hc_scale: torch.Tensor,
      hc_base: torch.Tensor,
      rms_eps: float,
      hc_pre_eps: float,
      hc_sinkhorn_eps: float,
      hc_post_mult_value: float,
      sinkhorn_repeat: int,
      return_residuals: bool = False,
      configs: tuple[Config | int | None, Any] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fwd_config, _ = (None, None) if configs is None else configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        return_residuals=return_residuals,
        config=self.deconstruct_config(fwd_config),
    )

  @override
  def op_impl_call(
      self,
      residual: jax.Array,
      fn: jax.Array,
      hc_scale: jax.Array,
      hc_base: jax.Array,
      rms_eps: float,
      hc_pre_eps: float,
      hc_sinkhorn_eps: float,
      hc_post_mult_value: float,
      sinkhorn_repeat: int,
      return_residuals: bool = False,
      config: int | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    assert config is not None, "Forward config not set."
    kernel_cfg = self.reconstruct_config(config)
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        return_residuals=return_residuals,
        config=kernel_cfg,
    )
    return out


class _PallasMosaicTpuMhcPost(
    torch_base._MhcPost[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax MhcPost Op wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuMhcPost()
    self.jax_op_name = "pallas_mosaic_tpu_mhc_post"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(  # pyrefly: ignore[bad-override]
      self, config: Config | int | None
  ) -> int | None:
    return _deconstruct_config(config)

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    return _reconstruct_config(*config_parts)

  @override
  def _fake_impl(
      self,
      x: torch.Tensor,
      residual: torch.Tensor,
      post_layer_mix: torch.Tensor,
      comb_res_mix: torch.Tensor,
      return_residuals: bool = False,
      config: int | None = None,
  ) -> torch.Tensor:
    return super()._fake_impl(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        return_residuals=return_residuals,
        config=config,
    )

  @override
  def __call__(
      self,
      x: torch.Tensor,
      residual: torch.Tensor,
      post_layer_mix: torch.Tensor,
      comb_res_mix: torch.Tensor,
      return_residuals: bool = False,
      configs: tuple[Config | int | None, Any] | None = None,
  ) -> torch.Tensor:
    fwd_config, _ = (None, None) if configs is None else configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        return_residuals=return_residuals,
        config=self.deconstruct_config(fwd_config),
    )

  @override
  def op_impl_call(
      self,
      x: jax.Array,
      residual: jax.Array,
      post_layer_mix: jax.Array,
      comb_res_mix: jax.Array,
      return_residuals: bool = False,
      config: int | None = None,
  ) -> jax.Array:
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    assert config is not None, "Forward config not set."
    kernel_cfg = self.reconstruct_config(config)
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        return_residuals=return_residuals,
        config=kernel_cfg,
    )
    return out


class _PallasMosaicTpuMhcFusedPostPre(
    torch_base._MhcFusedPostPre[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax MhcFusedPostPre Op wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuMhcFusedPostPre()
    self.jax_op_name = "pallas_mosaic_tpu_mhc_fused_post_pre"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(  # pyrefly: ignore[bad-override]
      self, config: Config | int | None
  ) -> int | None:
    return _deconstruct_config(config)

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    return _reconstruct_config(*config_parts)

  @override
  def _fake_impl(
      self,
      x: torch.Tensor,
      residual: torch.Tensor,
      post_layer_mix: torch.Tensor,
      comb_res_mix: torch.Tensor,
      fn: torch.Tensor,
      hc_scale: torch.Tensor,
      hc_base: torch.Tensor,
      rms_eps: float,
      hc_pre_eps: float,
      hc_sinkhorn_eps: float,
      hc_post_mult_value: float,
      sinkhorn_repeat: int,
      return_residuals: bool = False,
      config: int | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return super()._fake_impl(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        return_residuals=return_residuals,
        config=config,
    )

  @override
  def __call__(
      self,
      x: torch.Tensor,
      residual: torch.Tensor,
      post_layer_mix: torch.Tensor,
      comb_res_mix: torch.Tensor,
      fn: torch.Tensor,
      hc_scale: torch.Tensor,
      hc_base: torch.Tensor,
      rms_eps: float,
      hc_pre_eps: float,
      hc_sinkhorn_eps: float,
      hc_post_mult_value: float,
      sinkhorn_repeat: int,
      return_residuals: bool = False,
      configs: tuple[Config | int | None, Any] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fwd_config, _ = (None, None) if configs is None else configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        return_residuals=return_residuals,
        config=self.deconstruct_config(fwd_config),
    )

  @override
  def op_impl_call(
      self,
      x: jax.Array,
      residual: jax.Array,
      post_layer_mix: jax.Array,
      comb_res_mix: jax.Array,
      fn: jax.Array,
      hc_scale: jax.Array,
      hc_base: jax.Array,
      rms_eps: float,
      hc_pre_eps: float,
      hc_sinkhorn_eps: float,
      hc_post_mult_value: float,
      sinkhorn_repeat: int,
      return_residuals: bool = False,
      config: int | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    assert config is not None, "Forward config not set."
    kernel_cfg = self.reconstruct_config(config)
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        return_residuals=return_residuals,
        config=kernel_cfg,
    )
    return out


# Singleton instances of Pallas Mosaic TPU mHC ops.
PallasMosaicTpuMhcPre = _PallasMosaicTpuMhcPre()  # pylint: disable=invalid-name
PallasMosaicTpuMhcPost = _PallasMosaicTpuMhcPost()  # pylint: disable=invalid-name
PallasMosaicTpuMhcFusedPostPre = _PallasMosaicTpuMhcFusedPostPre()  # pylint: disable=invalid-name
