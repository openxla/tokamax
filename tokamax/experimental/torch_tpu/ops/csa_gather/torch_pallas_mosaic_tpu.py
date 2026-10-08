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
"""Tokamax operator wrapper for Pallas Mosaic TPU CSA Gather."""

from typing import Any
import jax
from tokamax._src.ops.experimental.tpu.csa_gather import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops.csa_gather import torch_base
import torch
from typing_extensions import override

Config = jax_pallas_mosaic_tpu.Config


class _PallasMosaicTpuCsaGather(
    torch_base._CsaGather[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax CSA Gather Op wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuCsaGather()
    self.jax_op_name = "pallas_mosaic_tpu_csa_gather"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(  # pyrefly: ignore[bad-override]
      self, config: Config | None
  ) -> tuple[int, ...] | None:
    if config is None:
      return None
    return (config.num_streams,)

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    config = config_parts[0]
    assert config is not None, "Forward config not set."
    return Config(num_streams=int(config[0]))

  # Overridden so default config resolution routes through
  # `self.deconstruct_config` instead of the base class's `dataclasses.astuple`.
  @override
  def op_impl_call_config_setup(
      self, *args: Any, config: Any = None, **kwargs: Any
  ) -> tuple[int, ...] | None:
    if config is None:
      config = self.deconstruct_config(
          self.get_bound_args(*args, **kwargs).get_config(
              check_autotuning_cache=False,
          )
      )
    return config

  @override
  def __call__(
      self,
      nope_cache: torch.Tensor,
      rope_cache: torch.Tensor,
      indices: torch.Tensor,
      num_valid_indices: torch.Tensor | None = None,
      top_k: int = 1024,
      return_residuals: bool = False,
      configs: tuple[Config | None, Any] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    fwd_config, _ = (None, None) if configs is None else configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        nope_cache,
        rope_cache,
        indices,
        num_valid_indices,
        top_k=top_k,
        return_residuals=return_residuals,
        config=self.deconstruct_config(fwd_config),
    )

  @override
  def op_impl_call(
      self,
      nope_cache: jax.Array,
      rope_cache: jax.Array,
      indices: jax.Array,
      num_valid_indices: jax.Array | None = None,
      top_k: int = 1024,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    assert config is not None, "Forward config not set."
    kernel_config = self.reconstruct_config(config)
    (nope_out, rope_out), _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        nope_cache,
        rope_cache,
        indices,
        num_valid_indices,
        top_k=top_k,
        return_residuals=return_residuals,
        config=kernel_config,
    )
    return nope_out, rope_out


# Singleton instance of Pallas Mosaic TPU CsaGather.
PallasMosaicTpuCsaGather = _PallasMosaicTpuCsaGather()  # pylint: disable=invalid-name
