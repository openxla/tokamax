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
"""Pallas/Mosaic operator implementation v2 for Ragged Gather on TPU."""

from typing import override

import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.ragged_gather import base
from tokamax._src.ops.ragged_gather import pallas_mosaic_tpu_kernel


class PallasTpuRaggedGather[C](base.RaggedGather[C]):
  """Tokamax operator invoking the Pallas kernel for Ragged Gather."""

  @override
  @jaxtyping.jaxtyped
  def bind(
      self,
      x: Shaped[Array | base.AbstractArray, "in_size hidden_size"],
      indices: Int[Array | base.AbstractArray, "out_size"],
      start: Int[Array | base.AbstractArray, "1"],
      end: Int[Array | base.AbstractArray, "1"],
      *,
      max_row_subchunks: int = 4,
      trim_rows: bool = True,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    return op.Op.bind(
        self,
        x=x,
        indices=indices,
        start=start,
        end=end,
        max_row_subchunks=max_row_subchunks,
        trim_rows=trim_rows,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      x: Shaped[Array, "in_size hidden_size"],
      indices: Int[Array, "out_size"],
      start: Int[Array, "1"],
      end: Int[Array, "1"],
      *,
      max_row_subchunks: int = 4,
      trim_rows: bool = True,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[jax.Array, None]:
    return (
        pallas_mosaic_tpu_kernel.ragged_gather_pallas(
            x,
            indices,
            start,
            end,
            max_row_subchunks=max_row_subchunks,
            trim_rows=trim_rows,
        ),
        None,
    )

  @override
  def supported_on(self, device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
