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
"""Tokamax operator wrapper for Pallas Mosaic TPU Ragged Gather Reduce."""

from typing import Any, override

import jax
from tokamax._src.ops.ragged_gather_reduce import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops.ragged_gather_reduce import torch_base
import torch

Config = jax_pallas_mosaic_tpu.Config


class _PallasMosaicTpuRaggedGatherReduce(
    torch_base._RaggedGatherReduce[Config]  # pylint: disable=protected-access
):
  """Tokamax operator wrapper for Pallas Mosaic TPU Ragged Gather Reduce."""

  def __init__(self) -> None:
    super().__init__()
    self.jax_op_name = "pallas_mosaic_tpu_ragged_gather_reduce"
    self.op_impl_jax = jax_pallas_mosaic_tpu.PallasTpuRaggedGatherReduce()
    self.is_vjp = False
    self.fake_impl = self._fake_impl

  @override
  def deconstruct_config(
      self, config: Config | tuple[Any, ...] | list[Any] | None
  ) -> tuple[int, ...] | None:
    """Converts Config into a 1-int tuple for jax_op static_argnums."""
    if config is None:
      return None
    if isinstance(config, Config):
      num_column_partitions = config.num_column_partitions
    else:
      num_column_partitions = config[0] if config else None
    return (
        -1 if num_column_partitions is None else int(num_column_partitions),
    )

  @override
  def reconstruct_config(self, *config_parts: Any) -> Config:
    """Rebuilds the 1-int tuple back into a Config."""
    config = config_parts[0]
    assert config is not None, "Forward config not set."
    num_column_partitions = config[0] if config else None
    return Config(
        num_column_partitions=(
            None
            if num_column_partitions is None or num_column_partitions < 0
            else int(num_column_partitions)
        )
    )

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
  def op_impl_call(
      self,
      x: jax.Array,
      indices: jax.Array,
      topk_weights: jax.Array,
      valid_rows_mask: jax.Array,
      reduce_group_size: int,
      return_residuals: bool = False,
      config: tuple[int, ...] | None = None,
  ) -> jax.Array:
    assert (
        self.op_impl_jax is not None
    ), "Forward class not set. self.op_impl_jax was not set in the constructor."
    assert config is not None, "Forward config not set."
    kernel_config = self.reconstruct_config(config)
    out, _ = self.op_impl_jax._fwd(  # pylint: disable=protected-access
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
        return_residuals=return_residuals,
        config=kernel_config,
    )
    return out

  @override
  def __call__(
      self,
      x: torch.Tensor,
      indices: torch.Tensor,
      topk_weights: torch.Tensor,
      valid_rows_mask: torch.Tensor,
      reduce_group_size: int,
      return_residuals: bool = False,
      config: Config | tuple[Any, ...] | list[Any] | None = None,
  ) -> torch.Tensor:
    assert self._torch_tokamax_op is not None, (
        "Forward op not registered. self.op_impl_jax was not set in the"
        " constructor."
    )
    return self._torch_tokamax_op(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
        return_residuals=return_residuals,
        config=self.deconstruct_config(config),
    )


PallasMosaicTpuRaggedGatherReduce = _PallasMosaicTpuRaggedGatherReduce()  # pylint: disable=invalid-name
