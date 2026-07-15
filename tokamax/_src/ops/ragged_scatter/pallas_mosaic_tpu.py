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
"""Pallas/Mosaic operator implementation for Ragged Scatter on TPU."""

import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Int, Shaped
from tokamax._src import jaxtyping
from tokamax._src.ops import op as tokamax_op
from tokamax._src.ops.ragged_scatter import base
from tokamax._src.ops.ragged_scatter import pallas_mosaic_tpu_kernel
from typing_extensions import override

_Config = pallas_mosaic_tpu_kernel.Config


class PallasTpuRaggedScatter(base.RaggedScatter[_Config]):
  """Pallas Sparse Core implementation of Ragged Scatter on TPU."""

  config_cls = _Config

  @override
  def _get_heuristics_config(self, ba: tokamax_op.BoundArguments) -> _Config:
    x = ba.arguments["x"]
    indices = ba.arguments["indices"]
    return pallas_mosaic_tpu_kernel.create_config(
        x.shape, indices.size, x.dtype
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      x: Shaped[Array, "num_rows hidden_size"],
      indices: Int[Array, "output_size"],
      start: Int[Array, "1"] | Int[Array, ""],
      end: Int[Array, "1"] | Int[Array, ""],
      *,
      return_residuals: bool = False,
      config: _Config | None = None,
  ) -> tuple[jax.Array, None]:
    del return_residuals
    # Direct return of the forward execution output
    return (
        pallas_mosaic_tpu_kernel.ragged_scatter_pallas(
            x, indices, start, end, config=config
        ),
        None,
    )

  @override
  def supported_on(self, device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
