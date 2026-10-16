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
"""Tokamax operator wrapper for Pallas Mosaic TPU ProjAndSaveState."""

import dataclasses
from typing import Any, override

import jax
from tokamax._src.ops.proj_and_save_state import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops.proj_and_save_state import torch_base
import torch

Config = jax_pallas_mosaic_tpu.Config


class _PallasMosaicTpuProjAndSaveState(
    torch_base._ProjAndSaveState[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax ProjAndSaveState Op wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuProjAndSaveState()
    self.jax_op_name = "pallas_mosaic_tpu_proj_and_save_state"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(  # pyrefly: ignore[bad-override]
      self, config: Config | None
  ) -> tuple[int, int] | None:
    """Deconstructs a Config object into a 2-int tuple for jax_op."""
    if config is None:
      return None
    assert len(dataclasses.fields(config)) == 2, (
        f"Expected 2 fields in Config, got {len(dataclasses.fields(config))}."
    )
    return (
        int(config.tile_k),
        int(config.tile_n),
    )

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    """Rebuilds the deconstructed 2-int tuple back into a Config."""
    if not config_parts or config_parts[0] is None:
      raise ValueError("Forward config cannot be None.")
    flat = config_parts[0]
    if isinstance(flat, Config):
      return flat
    assert len(flat) == 2, f"Invalid config: {flat!r}"
    return Config(
        tile_k=int(flat[0]),
        tile_n=int(flat[1]),
    )

  @override
  def __call__(
      self,
      hidden_states: torch.Tensor,
      wkv_wgate: torch.Tensor,
      ape: torch.Tensor,
      positions: torch.Tensor,
      slot_mapping: torch.Tensor,
      cache: torch.Tensor,
      compress_ratio: int,
      return_residuals: bool = False,
      configs: tuple[Config | None, Any] | None = None,
  ) -> torch.Tensor:
    fwd_config, _ = (None, None) if configs is None else configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        hidden_states,
        wkv_wgate,
        ape,
        positions,
        slot_mapping,
        cache,
        compress_ratio=compress_ratio,
        return_residuals=return_residuals,
        config=self.deconstruct_config(fwd_config),
    )

  @override
  def op_impl_call(
      self,
      hidden_states: jax.Array,
      wkv_wgate: jax.Array,
      ape: jax.Array,
      positions: jax.Array,
      slot_mapping: jax.Array,
      cache: jax.Array,
      compress_ratio: int,
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
        hidden_states,
        wkv_wgate,
        ape,
        positions,
        slot_mapping,
        cache,
        compress_ratio=compress_ratio,
        return_residuals=return_residuals,
        config=kernel_cfg,
    )
    return out


# Singleton instance of Pallas Mosaic TPU ProjAndSaveState.
PallasMosaicTpuProjAndSaveState = _PallasMosaicTpuProjAndSaveState()  # pylint: disable=invalid-name
