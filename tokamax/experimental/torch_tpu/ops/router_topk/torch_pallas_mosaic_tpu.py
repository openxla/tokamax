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
"""Tokamax operator wrapper for Pallas Mosaic TPU RouterTopK."""

import dataclasses
from typing import Any
import jax
from tokamax._src.ops.router_topk import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops.router_topk import torch_base
import torch
from typing_extensions import override

Config = jax_pallas_mosaic_tpu.Config


class _PallasMosaicTpuRouterTopK(
    torch_base._RouterTopK[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax RouterTopK Op wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuRouterTopK()
    self.jax_op_name = "pallas_mosaic_tpu_router_topk"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(  # pyrefly: ignore[bad-override]
      self, config: Config | None
  ) -> tuple[int, ...] | None:
    if config is None:
      return None
    assert (
        len(dataclasses.fields(config)) == 1
    ), f"Expected 1 fields in Config, got {len(dataclasses.fields(config))}."
    return (config.block_rows,)

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    config = config_parts[0]
    assert config is not None, "Forward config not set."
    return Config(block_rows=int(config[0]))

  @override
  def __call__(
      self,
      scores: torch.Tensor,
      k: int,
      return_residuals: bool = False,
      configs: tuple[Config | None, Any] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    fwd_config, _ = (None, None) if configs is None else configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        scores,
        k,
        return_residuals=return_residuals,
        config=self.deconstruct_config(fwd_config),
    )

  @override
  def op_impl_call(
      self,
      scores: jax.Array,
      k: int,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    assert config is not None, "Forward config not set."
    kernel_config = self.reconstruct_config(config)
    (weights, indices), _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        scores,
        k,
        return_residuals=return_residuals,
        config=kernel_config,
    )
    return weights, indices


# Singleton instance of Pallas Mosaic TPU RouterTopK.
PallasMosaicTpuRouterTopK = _PallasMosaicTpuRouterTopK()  # pylint: disable=invalid-name
