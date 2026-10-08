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
"""Tokamax operator wrapper for Pallas Mosaic TPU Ragged Gather."""

from typing import Any, TypeVar
import jax
from tokamax._src.ops.ragged_gather import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops.ragged_gather import torch_base
import torch
from typing_extensions import override

_Config = TypeVar("_Config")


class _PallasMosaicTpuRaggedGather(
    torch_base._RaggedGather[_Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax Ragged Gather Op wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuRaggedGather()
    self.jax_op_name = "pallas_mosaic_tpu_ragged_gather"
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def __call__(
      self,
      x: torch.Tensor,
      indices: torch.Tensor,
      start: torch.Tensor,
      end: torch.Tensor,
      return_residuals: bool = False,
      configs: tuple[Any, Any] | None = None,
  ) -> torch.Tensor:
    fwd_config, _ = (None, None) if configs is None else configs
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    return self._torch_tokamax_op(
        x,
        indices,
        start,
        end,
        return_residuals=return_residuals,
        config=self.deconstruct_config(fwd_config),
    )

  @override
  def op_impl_call(
      self,
      x: jax.Array,
      indices: jax.Array,
      start: jax.Array,
      end: jax.Array,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    del config
    assert self.op_impl_jax is not None, (
        "Forward class not set. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        x,
        indices,
        start,
        end,
        return_residuals=return_residuals,
        config=None,
    )
    return out


# Singleton instance of Pallas Mosaic TPU Ragged Gather.
PallasMosaicTpuRaggedGather = _PallasMosaicTpuRaggedGather[Any]()
