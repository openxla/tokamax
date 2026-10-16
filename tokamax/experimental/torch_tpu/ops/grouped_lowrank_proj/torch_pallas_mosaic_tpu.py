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
"""Tokamax operator wrapper for Pallas Mosaic TPU GroupedLowrankProj."""

import dataclasses
from typing import Any, override

import jax
from tokamax._src.ops.grouped_lowrank_proj import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops.grouped_lowrank_proj import torch_base
import torch

Config = jax_pallas_mosaic_tpu.Config


class _PallasMosaicTpuGroupedLowrankProj(
    torch_base._GroupedLowrankProj[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax GroupedLowrankProj Op wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuGroupedLowrankProj()
    self.jax_op_name = "pallas_mosaic_tpu_grouped_lowrank_proj"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(  # pyrefly: ignore[bad-override]
      self, config: Config | None
  ) -> tuple[int, int, int] | None:
    """Deconstructs a Config object into a 3-int tuple for jax_op."""
    if config is None:
      return None
    assert len(dataclasses.fields(config)) == 3, (
        f"Expected 3 fields in Config, got {len(dataclasses.fields(config))}."
    )
    return (
        int(config.tile_t),
        int(config.tile_r),
        int(config.sub_t),
    )

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    """Rebuilds the deconstructed 3-int tuple back into a Config."""
    if not config_parts or config_parts[0] is None:
      raise ValueError("Forward config cannot be None.")
    flat = config_parts[0]
    if isinstance(flat, Config):
      return flat
    assert len(flat) == 3, f"Invalid config: {flat!r}"
    return Config(
        tile_t=int(flat[0]),
        tile_r=int(flat[1]),
        sub_t=int(flat[2]),
    )

  @override
  def __call__(
      self,
      x: torch.Tensor,
      positions: torch.Tensor,
      cos_sin_cache: torch.Tensor,
      wo_a: torch.Tensor,
      wo_a_scale: torch.Tensor,
      inverse: bool = True,
      quantize_activations: bool = True,
      return_residuals: bool = False,
      configs: tuple[Config | None, Any] | None = None,
  ) -> torch.Tensor:
    fwd_config, _ = (None, None) if configs is None else configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        x,
        positions,
        cos_sin_cache,
        wo_a,
        wo_a_scale,
        inverse=inverse,
        quantize_activations=quantize_activations,
        return_residuals=return_residuals,
        config=self.deconstruct_config(fwd_config),
    )

  @override
  def op_impl_call(
      self,
      x: jax.Array,
      positions: jax.Array,
      cos_sin_cache: jax.Array,
      wo_a: jax.Array,
      wo_a_scale: jax.Array,
      inverse: bool = True,
      quantize_activations: bool = True,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    assert config is not None, "Forward config not set."
    kernel_cfg = self.reconstruct_config(config)
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        x,
        positions,
        cos_sin_cache,
        wo_a,
        wo_a_scale,
        inverse=inverse,
        quantize_activations=quantize_activations,
        return_residuals=return_residuals,
        config=kernel_cfg,
    )
    return out


# Singleton instance of Pallas Mosaic TPU GroupedLowrankProj.
PallasMosaicTpuGroupedLowrankProj = _PallasMosaicTpuGroupedLowrankProj()  # pylint: disable=invalid-name
