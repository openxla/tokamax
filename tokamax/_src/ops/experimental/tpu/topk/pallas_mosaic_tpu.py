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
"""Pallas/Mosaic kernel wrapper for TopK on TPU."""

from typing import ClassVar, override

import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.tpu.topk import base
from tokamax._src.ops.experimental.tpu.topk import pallas_mosaic_tpu_kernel


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Execution configuration for SparseCore TopK Pallas TPU kernel."""

  scheduling_group_id: int | None = None
  stage2_scheduling_group_id: int | None = None


class PallasTpuTopK(base.TopK[Config]):
  """Tokamax operator wrapper for Pallas Mosaic TPU SparseCore TopK kernel."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      scores: Float[Array, "b n"] | Int[Array, "b n"],
      k: int,
      row_lengths: Int[Array, "b"] | None = None,
      *,
      return_scores: bool = False,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[jax.Array | tuple[jax.Array, jax.Array], None]:
    del return_residuals
    if config is None:
      config = self._get_heuristics_config(None)  # pyrefly: ignore[bad-argument-type]

    return (
        pallas_mosaic_tpu_kernel.sparsecore_topk(
            scores=scores,
            k=k,
            row_lengths=row_lengths,
            scheduling_group_id=config.scheduling_group_id,
            stage2_scheduling_group_id=config.stage2_scheduling_group_id,
            return_scores=return_scores,
        ),
        None,
    )

  # TODO: Add correct heuristics config and autotuning search space.
  @override
  def _get_heuristics_config(self, ba) -> Config:
    del ba
    return Config()

  @override
  def _get_autotuning_configs(self, ba) -> set[Config]:
    del ba
    return {Config()}

  @override
  def supported_on(self, device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 7
