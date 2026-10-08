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
"""Pallas Mosaic TPU Ragged Dot PyTorch Op."""

from collections.abc import Sequence
import dataclasses
from typing import Any
import jax
from tokamax._src.ops.ragged_dot import base as jax_base
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2 as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops import torch_op
from tokamax.experimental.torch_tpu.ops import torch_utils
from tokamax.experimental.torch_tpu.ops.ragged_dot import torch_base
import torch
from typing_extensions import override

Config = jax_pallas_mosaic_tpu.Config


class _PallasMosaicTpuRaggedDot(
    torch_base._RaggedDot[Config]  # pylint: disable=protected-access
):
  """Pallas Mosaic TPU Tokamax Ragged Dot Op wrapped for PyTorch."""

  def __init__(self) -> None:
    torch_op.TorchOp.__init__(self)
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasMosaicTpuV2RaggedDot()
    self.jax_op_name = "pallas_mosaic_tpu_ragged_dot"
    # Ragged dot is a special case where the forward and backward ops are the
    # same.
    self.backward_op_torch = self

  @override
  def deconstruct_config(
      self, config: Config | tuple[Any, ...] | list[Any] | None
  ) -> tuple[int, ...] | None:
    """Deconstructs a Config object into a jax_op-compatible int tuple."""
    if config is None:
      return None
    raw = (
        tuple(config)
        if isinstance(config, (tuple, list))
        else dataclasses.astuple(config)
    )
    return tuple(0 if x is None else int(x) for x in raw)

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    """Reconstructs a Config object from its deconstructed representation."""
    config = config_parts[0]
    assert config is not None, "Config not set."
    tile_m, tile_k, tile_n, bucket_base = (
        None if x is None or x <= 0 else x for x in config
    )
    return Config(
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        bucket_base=bucket_base,
    )

  @override
  def __call__(
      self,
      lhs: torch.Tensor,
      rhs: torch.Tensor,
      group_sizes: torch.Tensor,
      ragged_dot_dimension_numbers: (
          jax.lax.RaggedDotDimensionNumbers | Sequence[int] | None
      ) = None,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: (
          torch.dtype | jax.typing.DTypeLike | str | None
      ) = None,
      return_residuals: bool = False,
      activation: jax_base.ActivationFunction | None = None,
      configs: tuple[Config | None, Config | None] | None = None,
  ) -> Any:
    if activation is not None:
      raise NotImplementedError(
          "activations are not supported on Torch TPU Tokamax"
      )
    dim_nums = torch_base.tuple_to_dim_nums(ragged_dot_dimension_numbers)
    dim_nums_tuple = torch_base.dim_nums_to_tuple(dim_nums)
    precision_str = torch_base.precision_to_str(precision)
    preferred_element_type_str = torch_utils.dtype_to_str(
        preferred_element_type
    )
    fwd_config, bwd_config = (None, None) if configs is None else configs

    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. This means that self.op_impl_jax is not set"
        " in the constructor."
    )
    assert self.backward_op_torch is not None, "Backward op not set."
    is_fwd = dim_nums == jax_base.DEFAULT_RAGGED_DOT_DIM_NUMS
    out, residuals = self._torch_tokamax_op(
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers=dim_nums_tuple,
        precision=precision_str,
        preferred_element_type=preferred_element_type_str,
        return_residuals=is_fwd,
        config=self.deconstruct_config(fwd_config),
        bwd_config=self.backward_op_torch.deconstruct_config(bwd_config),
    )
    return (out, residuals) if return_residuals else out

  @override
  def op_impl_call(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      group_sizes: jax.Array,
      ragged_dot_dimension_numbers: tuple[int, ...] | None = None,
      precision: str | None = None,
      preferred_element_type: str | None = None,
      return_residuals: bool = True,
      config: tuple[int, ...] | None = None,
      bwd_config: tuple[int, ...] | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    del bwd_config
    assert self.op_impl_jax is not None, "Forward class not set."
    assert config is not None, "Config not set."
    kernel_config = self.reconstruct_config(config)
    dim_nums = torch_base.tuple_to_dim_nums(ragged_dot_dimension_numbers)
    canon_precision = torch_base.str_to_precision(precision)
    preferred_element_type_dtype = torch_utils.str_to_jax_dtype(
        preferred_element_type
    )
    op_impl_jax = self.op_impl_jax
    if dim_nums == jax_pallas_mosaic_tpu.DRHS_RAGGED_DOT_DIM_NUMS:
      op_impl_jax = dataclasses.replace(
          self.op_impl_jax, num_actual_groups=group_sizes.shape[0]
      )

    out, residuals = op_impl_jax._fwd(  # pylint: disable=protected-access
        lhs,
        rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=dim_nums,
        precision=canon_precision,
        preferred_element_type=preferred_element_type_dtype,
        return_residuals=return_residuals,
        config=kernel_config,
        activation=None,
    )
    return out, (out if residuals is None else residuals)

  @override
  def setup_context(
      self,
      ctx: Any,
      inputs: Sequence[Any],
      output: tuple[torch.Tensor, torch.Tensor],
  ) -> None:
    """Saves tensors and attributes to the context for the backward pass."""
    (
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers,
        precision,
        preferred_element_type,
        _,
        _,
        bwd_config,
    ) = inputs
    out, residuals = output
    ctx.save_for_backward(lhs, rhs, group_sizes, residuals, out)
    ctx.ragged_dot_dimension_numbers = ragged_dot_dimension_numbers
    ctx.precision = precision
    ctx.preferred_element_type = preferred_element_type
    ctx.bwd_config = bwd_config

  @override
  def backward(
      self,
      ctx: Any,
      dout: torch.Tensor,
      dresiduals: torch.Tensor | None = None,
  ) -> tuple[
      torch.Tensor, torch.Tensor, None, None, None, None, None, None, None
  ]:
    """Callback for torch register_autograd."""
    del dresiduals  # Unused.
    assert self.backward_op_torch is not None, "Backward op not set."
    lhs, rhs, group_sizes, _, _ = ctx.saved_tensors

    dlhs_dim_nums = jax_base.TRANS_RHS_RAGGED_DOT_DIM_NUMS
    drhs_dim_nums = jax_base.RAGGED_CONTRACTING_DOT_DIM_NUMS

    # Execute dlhs and drhs using the Pallas ragged_dot op itself (since
    # self.backward_op_torch is self), retrieving the config for each backward
    # kernel call via torch_utils.get_configs without mutating any attributes
    # on `self`.
    dlhs_config, _ = torch_utils.get_configs(
        self.backward_op_torch,
        dout,
        rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=dlhs_dim_nums,
        precision=ctx.precision,
        preferred_element_type=lhs.dtype,
    )
    dlhs = self.backward_op_torch(
        dout,
        rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=dlhs_dim_nums,
        precision=ctx.precision,
        preferred_element_type=lhs.dtype,
        return_residuals=False,
        configs=(dlhs_config, None),
    )
    drhs_config, _ = torch_utils.get_configs(
        self.backward_op_torch,
        lhs,
        dout,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=drhs_dim_nums,
        precision=ctx.precision,
        preferred_element_type=rhs.dtype,
    )
    drhs = self.backward_op_torch(
        lhs,
        dout,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=drhs_dim_nums,
        precision=ctx.precision,
        preferred_element_type=rhs.dtype,
        return_residuals=False,
        configs=(drhs_config, None),
    )
    # Return gradients for (lhs, rhs, group_sizes, ragged_dot_dimension_numbers,
    # precision, preferred_element_type, return_residuals, config, bwd_config).
    return dlhs, drhs, None, None, None, None, None, None, None


# Singleton instance of the Pallas Mosaic TPU Ragged Dot.
PallasMosaicTpuRaggedDot = (  # pylint: disable=invalid-name
    _PallasMosaicTpuRaggedDot()
)
