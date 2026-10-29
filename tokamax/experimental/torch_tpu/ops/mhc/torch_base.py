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
"""Base classes for PyTorch interfaces to Tokamax mHC operators."""

from typing import Any, TypeVar, override

import jax
from tokamax._src.ops.mhc import base as jax_base
from tokamax._src.ops.mhc import reference as jax_reference
from tokamax.experimental.torch_tpu.ops import torch_op
import torch

_Config = TypeVar("_Config")


class _MhcPre(torch_op.TorchOp[_Config]):
  """mHC Pre PyTorch Op API using reference implementation."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.MhcPre()
    self.jax_op_name = "base_mhc_pre"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

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
    """Meta/fake implementation for symbolic shape tracing in torch.compile."""
    del hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps
    del hc_post_mult_value, sinkhorn_repeat, return_residuals, config
    hc_mult = jax_reference.hc_mult_from_mix_dim(int(fn.shape[0]))
    hidden_size = residual.shape[-1] // hc_mult
    num_tokens = residual.shape[0]
    return (
        residual.new_empty((num_tokens, hc_mult), dtype=torch.float32),
        residual.new_empty((num_tokens, hc_mult, hc_mult), dtype=torch.float32),
        residual.new_empty((num_tokens, hidden_size), dtype=residual.dtype),
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
      configs: tuple[Any, Any] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del configs
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
        config=None,
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
    del config
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
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
        config=None,
    )
    return out


class _MhcPost(torch_op.TorchOp[_Config]):
  """mHC Post PyTorch Op API using reference implementation."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.MhcPost()
    self.jax_op_name = "base_mhc_post"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  def _fake_impl(
      self,
      x: torch.Tensor,
      residual: torch.Tensor,
      post_layer_mix: torch.Tensor,
      comb_res_mix: torch.Tensor,
      return_residuals: bool = False,
      config: int | None = None,
  ) -> torch.Tensor:
    """Meta/fake implementation for symbolic shape tracing in torch.compile."""
    del x, post_layer_mix, comb_res_mix, return_residuals, config
    return torch.empty_like(residual)

  @override
  def __call__(
      self,
      x: torch.Tensor,
      residual: torch.Tensor,
      post_layer_mix: torch.Tensor,
      comb_res_mix: torch.Tensor,
      return_residuals: bool = False,
      configs: tuple[Any, Any] | None = None,
  ) -> torch.Tensor:
    del configs
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
        config=None,
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
    del config
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        return_residuals=return_residuals,
        config=None,
    )
    return out


class _MhcFusedPostPre(torch_op.TorchOp[_Config]):
  """mHC Fused Post-Pre PyTorch Op API using reference implementation."""

  def __init__(self) -> None:
    super().__init__()
    self.op_impl_jax = jax_base.MhcFusedPostPre()
    self.jax_op_name = "base_mhc_fused_post_pre"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

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
    """Meta/fake implementation for symbolic shape tracing in torch.compile."""
    del fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps
    del hc_post_mult_value, sinkhorn_repeat, return_residuals, config
    return (
        torch.empty_like(residual),
        torch.empty_like(post_layer_mix),
        torch.empty_like(comb_res_mix),
        torch.empty_like(x),
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
      configs: tuple[Any, Any] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del configs
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
        config=None,
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
    del config
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
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
        config=None,
    )
    return out


# Singleton instances of mHC base ops.
MhcPre = _MhcPre[Any]()  # pylint: disable=invalid-name
MhcPost = _MhcPost[Any]()  # pylint: disable=invalid-name
MhcFusedPostPre = _MhcFusedPostPre[Any]()  # pylint: disable=invalid-name
