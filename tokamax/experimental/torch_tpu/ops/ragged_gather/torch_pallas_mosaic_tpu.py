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
from jax.experimental.pallas import tpu as pltpu
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
  def _fake_impl(
      self,
      x: torch.Tensor,
      indices: torch.Tensor,
      start: torch.Tensor,
      end: torch.Tensor,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
      max_row_subchunks: int = 4,
      trim_rows: bool = True,
  ) -> torch.Tensor:
    if trim_rows:
      return super()._fake_impl(
          x=x,
          indices=indices,
          start=start,
          end=end,
          return_residuals=return_residuals,
          config=config,
      )
    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
      return x.new_empty((indices.shape[0], *x.shape[1:]))
    out_size = indices.shape[0]
    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores
    base_block_size = num_simd_lanes * num_cores
    num_row_subchunks = max(
        1,
        min(
            max_row_subchunks,
            (out_size + base_block_size - 1) // base_block_size,
        ),
    )
    block_size = num_simd_lanes * num_row_subchunks * num_cores
    padded_out_size = ((out_size + block_size - 1) // block_size) * block_size
    return x.new_empty((padded_out_size, *x.shape[1:]))

  @override
  def __call__(
      self,
      x: torch.Tensor,
      indices: torch.Tensor,
      start: torch.Tensor,
      end: torch.Tensor,
      return_residuals: bool = False,
      configs: tuple[Any, Any] | None = None,
      max_row_subchunks: int = 4,
      trim_rows: bool = True,
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
        max_row_subchunks=max_row_subchunks,
        trim_rows=trim_rows,
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
      max_row_subchunks: int = 4,
      trim_rows: bool = True,
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
        max_row_subchunks=max_row_subchunks,
        trim_rows=trim_rows,
        return_residuals=return_residuals,
        config=None,
    )
    return out


# Singleton instance of Pallas Mosaic TPU Ragged Gather.
PallasMosaicTpuRaggedGather = _PallasMosaicTpuRaggedGather[Any]()
