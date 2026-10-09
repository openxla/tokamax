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
"""Tokamax operator wrapper for Pallas Mosaic TPU TopK."""

from typing import Any
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.tpu.topk import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops.topk import torch_base
import torch
from typing_extensions import override

Config = jax_pallas_mosaic_tpu.Config


class _PallasMosaicTpuTopK(
    torch_base._TopK[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax TopK Op wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuTopK()
    self.jax_op_name = "pallas_mosaic_tpu_topk"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(  # pyrefly: ignore[bad-override]
      self, config: Config | None
  ) -> tuple[int, ...] | None:
    if config is None:
      return None
    return (
        -1
        if config.scheduling_group_id is None
        else config.scheduling_group_id,
        -1
        if config.stage2_scheduling_group_id is None
        else config.stage2_scheduling_group_id,
    )

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    config = config_parts[0]
    assert config is not None, "Forward config not set."
    return Config(
        scheduling_group_id=None if config[0] < 0 else int(config[0]),
        stage2_scheduling_group_id=None if config[1] < 0 else int(config[1]),
    )

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
      scores: torch.Tensor,
      k: int,
      row_lengths: torch.Tensor | None = None,
      return_scores: bool = False,
      return_residuals: bool = False,
      configs: tuple[Config | None, Any] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    fwd_config, _ = (None, None) if configs is None else configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    indices, scores_bits = self._torch_tokamax_op(
        scores,
        k,
        row_lengths,
        return_scores=return_scores,
        return_residuals=return_residuals,
        config=self.deconstruct_config(fwd_config),
    )
    if return_scores:
      return indices, scores_bits
    return indices, None

  @override
  def op_impl_call(
      self,
      scores: jax.Array,
      k: int,
      row_lengths: jax.Array | None = None,
      return_scores: bool = False,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    assert config is not None, "Forward config not set."
    kernel_config = self.reconstruct_config(config)
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        scores,
        k,
        row_lengths,
        return_scores=return_scores,
        return_residuals=return_residuals,
        config=kernel_config,
    )
    if isinstance(out, tuple):
      return out
    return out, jnp.empty((), dtype=jnp.int32)


# Singleton instance of Pallas Mosaic TPU TopK.
PallasMosaicTpuTopK = _PallasMosaicTpuTopK()  # pylint: disable=invalid-name
