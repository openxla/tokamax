# Copyright 2026 Google LLC. All Rights Reserved.
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

"""Tokamax operator wrapper for Raven GSA Pallas TPU kernel."""

import dataclasses
from typing import Annotated, Any, ClassVar
import jax
import pydantic
from tokamax._src.ops import op
from tokamax._src.ops.experimental.raven import base
from tokamax._src.ops.experimental.raven import pallas_mosaic_tpu_kernel
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning and execution configuration for Pallas TPU Raven."""

  chunk_size: Annotated[int, pydantic.Field(gt=0)] = 64


class PallasTpuRavenGSA(base.RavenGSA[Config]):
  """Tokamax operator wrapper for Raven GSA Pallas TPU kernel."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  def _fwd(
      self,
      q: jax.Array,
      k: jax.Array,
      s: jax.Array,
      g: jax.Array,
      *,
      chunk_size: int = 64,
      initial_state: jax.Array | None = None,
      output_final_state: bool = False,
      config: Config | None = None,
  ) -> tuple[base.Output, None]:
    resolved_chunk_size = config.chunk_size if config is not None else chunk_size
    out = pallas_mosaic_tpu_kernel.raven_pallas_stage1_fwd(
        q,
        k,
        s,
        g,
        chunk_size=resolved_chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
    return out, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(chunk_size=64)

  @override
  def supported_on(self, device: Any) -> bool:
    return True
