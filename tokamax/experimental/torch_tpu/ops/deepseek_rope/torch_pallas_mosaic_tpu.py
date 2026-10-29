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
"""Tokamax operator wrapper for Pallas Mosaic TPU DeepSeek-V4 RoPE."""

import dataclasses
from typing import Any, cast, override

import jax
import jax.numpy as jnp
from tokamax._src.ops.deepseek_rope import base as jax_base
from tokamax._src.ops.deepseek_rope import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.deepseek_rope import torch_base
import torch

Config = jax_pallas_mosaic_tpu.Config


class _PallasMosaicTpuDeepseekRope(
    torch_base._DeepseekRope[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax DeepSeek-V4 RoPE Op wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuDeepseekRope()
    self.jax_op_name = "pallas_mosaic_tpu_deepseek_rope"
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
    return (int(config.tile_n),)

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
      x: torch.Tensor,
      positions: torch.Tensor,
      cos_sin_cache: torch.Tensor,
      mode: str = "rope",
      inverse: bool = False,
      eps: float = 1e-6,
      quant_dtype: torch.dtype | str | None = None,
      return_residuals: bool = False,
      config: Config | None = None,
      configs: tuple[Config | None, Any] | None = None,
  ) -> Any:
    fwd_config = config if configs is None else configs[0]
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    out, scales = self._torch_tokamax_op(
        x,
        positions,
        cos_sin_cache,
        mode=mode,
        inverse=inverse,
        eps=eps,
        quant_dtype=torch_utils.dtype_to_str(quant_dtype),
        return_residuals=return_residuals,
        config=self.deconstruct_config(fwd_config),
    )
    return (out, scales) if mode == "rope_quant" else out

  @override
  def op_impl_call(
      self,
      x: jax.Array,
      positions: jax.Array,
      cos_sin_cache: jax.Array,
      mode: str = "rope",
      inverse: bool = False,
      eps: float = 1e-6,
      quant_dtype: str | None = None,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array]:
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
        mode=cast(jax_base.Mode, mode),
        inverse=inverse,
        eps=eps,
        quant_dtype=torch_base.canonicalize_jax_quant_dtype(mode, quant_dtype),
        return_residuals=return_residuals,
        config=kernel_cfg,
    )
    if mode == "rope_quant":
      assert isinstance(out, tuple)
      return out
    assert isinstance(out, jax.Array)
    return out, jnp.empty((), dtype=jnp.float32)


# Singleton instance of Pallas Mosaic TPU DeepseekRope.
PallasMosaicTpuDeepseekRope = _PallasMosaicTpuDeepseekRope()  # pylint: disable=invalid-name
