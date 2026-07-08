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

from typing import TypeVar
import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.ragged_gather import base
from tokamax._src.ops.ragged_gather import pallas_mosaic_v2_tpu_kernel
from typing_extensions import override

_Config = TypeVar("_Config")


class PallasV2TpuRaggedGather(base.RaggedGather[_Config]):
  """Tokamax operator invoking the Pallas kernel V2 for Ragged Gather."""

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      x: Shaped[Array, "in_size hidden_size"],
      indices: Int[Array, "out_size"],
      start: Int[Array, "1"],
      end: Int[Array, "1"],
      *,
      return_residuals: bool = False,
      config: _Config | None = None,
  ) -> tuple[jax.Array, None]:
    return (
        pallas_mosaic_v2_tpu_kernel.ragged_gather_pallas(
            x, indices, start, end
        ),
        None,
    )

  @override
  def supported_on(self, device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
