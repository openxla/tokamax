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
"""Pallas/Mosaic operator implementation v2 for Ragged Gather Reduce on TPU."""

from typing import TypeVar
import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.ragged_gather_reduce import base
from tokamax._src.ops.ragged_gather_reduce import pallas_mosaic_v2_tpu_kernel
from typing_extensions import override

_Config = TypeVar("_Config")


class PallasV2TpuRaggedGatherReduce(base.RaggedGatherReduce[_Config]):
  """Tokamax operator invoking the Pallas kernel V2 for Ragged Gather Reduce."""

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      x: Shaped[Array, "input_size hidden_size"],
      indices: Int[Array, "input_size"],
      topk_weights: Shaped[Array, "input_size"],
      valid_rows_mask: Shaped[Array, "input_size"],
      *,
      reduce_group_size: int,
      return_residuals: bool = False,
      config: _Config | None = None,
  ) -> tuple[jax.Array, None]:
    return (
        pallas_mosaic_v2_tpu_kernel.ragged_gather_reduce_pallas(
            x, indices, topk_weights, valid_rows_mask, reduce_group_size
        ),
        None,
    )

  @override
  def supported_on(self, device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
