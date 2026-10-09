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
"""Pallas/Mosaic operator implementation for Ragged Gather Reduce on TPU."""

from typing import Annotated, ClassVar, override

import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.ragged_gather_reduce import base
from tokamax._src.ops.ragged_gather_reduce import pallas_mosaic_tpu_kernel
from tokamax._src.ops.ragged_gather_reduce import reference

# Widest column slice the autotuner tries beyond the heuristic's choice. The
# kernel keeps a `(64, col_size)` float32 accumulator, two output tiles and the
# double-buffered gather of each subcore's column slice in VMEM, and reduces
# wider slices in column chunks that fit; upstream sizes its heuristic for
# column slices of up to 1024.
_MAX_AUTOTUNING_COL_SIZE = 1024
# Inputs whose `x` (counted twice, for the read and the write) fills less than
# this fraction of TensorCore VMEM run the XLA reference instead: for them the
# kernel's launch and metadata overhead outweighs the gather (upstream's
# threshold).
_SMALL_INPUT_VMEM_FRACTION = 0.6


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning parameters for the Ragged Gather Reduce SparseCore kernel.

  Attributes:
    num_column_partitions: Number of hidden-dimension slices the SparseCore
      subcores split `x` into. The remaining factor of the subcore count splits
      the destination tokens. `None` selects
      `pallas_mosaic_tpu_kernel.default_num_column_partitions`.
  """

  num_column_partitions: Annotated[int, pydantic.Field(gt=0)] | None = None


def _is_small_input(x: jax.Array) -> bool:
  vmem_capacity_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
  return x.size * x.dtype.itemsize * 2 < (
      vmem_capacity_bytes * _SMALL_INPUT_VMEM_FRACTION
  )


class PallasTpuRaggedGatherReduce(base.RaggedGatherReduce[Config]):
  """Tokamax operator invoking the SparseCore Pallas kernel for RGR."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      x: Shaped[Array, "num_rows hidden_size"],
      indices: Int[Array, "input_size"],
      topk_weights: Shaped[Array, "input_size"],
      valid_rows_mask: Shaped[Array, "input_size"],
      *,
      reduce_group_size: int,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[jax.Array, None]:
    """Gathers, weights and sums the routes of every destination token.

    See `reference` for the semantics.

    Args:
      x: `(num_rows, hidden_size)` bf16 expert outputs.
      indices: `(input_size,)` int32 row of `x` read by each route.
      topk_weights: `(input_size,)` weight of each route.
      valid_rows_mask: `(input_size,)` bool. Routes with a false entry are
        skipped.
      reduce_group_size: Routes summed into one output token.
      return_residuals: Unused; the op has no residuals.
      config: The `num_column_partitions` to run the kernel with. Defaults to
        `Config()`.

    Returns:
      `(out, None)`, where `out` is `(input_size // reduce_group_size,
      hidden_size)` bf16.

    Raises:
      NotImplementedError: If the kernel does not support the inputs.
    """
    num_rows, hidden_size = x.shape
    if reason := pallas_mosaic_tpu_kernel.get_unsupported_reason(
        num_rows, hidden_size, x.dtype, reduce_group_size
    ):
      raise NotImplementedError(reason)
    if _is_small_input(x):
      out = reference.ragged_gather_reduce(
          x, indices, topk_weights, valid_rows_mask, reduce_group_size
      )
      return out, None
    if config is None:
      config = Config()
    return (
        pallas_mosaic_tpu_kernel.ragged_gather_reduce(
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            reduce_group_size,
            num_column_partitions=config.num_column_partitions,
        ),
        None,
    )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    hidden_size = ba.arguments["x"].shape[-1]
    num_column_partitions, _ = pallas_mosaic_tpu_kernel.partition_counts(
        hidden_size, pltpu.get_tpu_info()
    )
    return Config(num_column_partitions=num_column_partitions)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    hidden_size = ba.arguments["x"].shape[-1]
    tpu_info = pltpu.get_tpu_info()
    heuristic = self._get_heuristics_config(ba)
    max_col_size = max(
        _MAX_AUTOTUNING_COL_SIZE,
        hidden_size // heuristic.num_column_partitions,
    )
    configs = set()
    n = 1
    while n <= pallas_mosaic_tpu_kernel.num_sparse_cores(tpu_info):
      if (
          pallas_mosaic_tpu_kernel.is_valid_num_column_partitions(
              n, hidden_size, tpu_info
          )
          and hidden_size // n <= max_col_size
      ):
        configs.add(Config(num_column_partitions=n))
      n *= 2
    return configs

  @override
  def supported_on(self, device) -> bool:
    if device.platform != "tpu":
      return False
    sc_info = pltpu.get_tpu_info().sparse_core
    return (
        sc_info is not None
        and sc_info.num_lanes == pallas_mosaic_tpu_kernel.TOKEN_SUBCHUNK
    )
