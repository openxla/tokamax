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
"""Tokamax operator wrapper for Pallas Mosaic TPU DSv4 Compressor."""

import dataclasses
from typing import Any, override

import jax
from tokamax._src.ops.dsv4_compressor import base as jax_base
from tokamax._src.ops.dsv4_compressor import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops.dsv4_compressor import torch_base
import torch

Config = jax_pallas_mosaic_tpu.Config


class _PallasMosaicTpuDsv4Compressor(
    torch_base._Dsv4Compressor[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax DSv4 Compressor Op wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuDsv4Compressor()
    self.jax_op_name = "pallas_mosaic_tpu_dsv4_compressor"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(  # pyrefly: ignore[bad-override]
      self, config: Config | None
  ) -> tuple[int, ...] | None:
    """Deconstructs a Config object into a 1-int tuple for jax_op."""
    if config is None:
      return None
    assert len(dataclasses.fields(config)) == 1, (
        f"Expected 1 field in Config, got {len(dataclasses.fields(config))}."
    )
    return (config.tile_n,)

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    """Rebuilds the deconstructed 1-int tuple back into a Config."""
    if not config_parts or config_parts[0] is None:
      raise ValueError("Forward config cannot be None.")
    flat = config_parts[0]
    if isinstance(flat, Config):
      return flat
    assert len(flat) == 1, f"Invalid config: {flat!r}"
    return Config(tile_n=int(flat[0]))

  @override
  def __call__(
      self,
      cache: torch.Tensor,
      positions: torch.Tensor,
      block_table: torch.Tensor,
      token_to_req_indices: torch.Tensor,
      kv_slot_mapping: torch.Tensor,
      rms_weight: torch.Tensor,
      cos_sin_cache: torch.Tensor,
      block_table_stride: int,
      state_block_size: int,
      compress_ratio: int,
      overlap: bool,
      state_cache: torch.Tensor | None = None,
      rope_cache: torch.Tensor | None = None,
      quant_block: int = jax_base.CSA_QUANT_BLOCK,
      rms_eps: float = 1e-6,
      return_residuals: bool = False,
      configs: tuple[Config | None, Any] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    fwd_config, _ = (None, None) if configs is None else configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        cache,
        positions,
        block_table,
        token_to_req_indices,
        kv_slot_mapping,
        rms_weight,
        cos_sin_cache,
        block_table_stride=block_table_stride,
        state_block_size=state_block_size,
        compress_ratio=compress_ratio,
        overlap=overlap,
        state_cache=state_cache,
        rope_cache=rope_cache,
        quant_block=quant_block,
        rms_eps=rms_eps,
        return_residuals=return_residuals,
        config=self.deconstruct_config(fwd_config),
    )

  @override
  def op_impl_call(
      self,
      cache: jax.Array,
      positions: jax.Array,
      block_table: jax.Array,
      token_to_req_indices: jax.Array,
      kv_slot_mapping: jax.Array,
      rms_weight: jax.Array,
      cos_sin_cache: jax.Array,
      block_table_stride: int,
      state_block_size: int,
      compress_ratio: int,
      overlap: bool,
      state_cache: jax.Array | None = None,
      rope_cache: jax.Array | None = None,
      quant_block: int = jax_base.CSA_QUANT_BLOCK,
      rms_eps: float = 1e-6,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array | None]:
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    assert config is not None, "Forward config not set."
    kernel_cfg = self.reconstruct_config(config)
    (out_cache, out_rope_cache), _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        cache,
        positions,
        block_table,
        token_to_req_indices,
        kv_slot_mapping,
        rms_weight,
        cos_sin_cache=cos_sin_cache,
        block_table_stride=block_table_stride,
        state_block_size=state_block_size,
        compress_ratio=compress_ratio,
        overlap=overlap,
        state_cache=state_cache,
        rope_cache=rope_cache,
        quant_block=quant_block,
        rms_eps=rms_eps,
        return_residuals=return_residuals,
        config=kernel_cfg,
    )
    return out_cache, out_rope_cache


# Singleton instance of Pallas Mosaic TPU Dsv4Compressor.
PallasMosaicTpuDsv4Compressor = _PallasMosaicTpuDsv4Compressor()  # pylint: disable=invalid-name
