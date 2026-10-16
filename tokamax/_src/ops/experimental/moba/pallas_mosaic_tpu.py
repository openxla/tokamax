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

"""Tokamax operator wrapper for Mixture of Block Attention (MoBA)."""

import dataclasses
from typing import Annotated, Any, ClassVar
import jax
import pydantic
from tokamax._src.ops import op
from tokamax._src.ops.experimental.moba import base
from tokamax._src.ops.experimental.moba import pallas_mosaic_tpu_kernel
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning and execution configuration for Pallas TPU MoBA."""

  chunk_size: Annotated[int, pydantic.Field(gt=0)] = 256
  topk: Annotated[int, pydantic.Field(gt=0)] = 4


class PallasTpuMixtureOfBlockAttention(base.MixtureOfBlockAttention[Config]):
  """Tokamax operator wrapper for MoBA Pallas TPU kernel."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  def _fwd(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      *,
      topk: int = 4,
      chunk_size: int = 256,
      scale: float | None = None,
      config: Config | None = None,
  ) -> tuple[base.Output, None]:
    resolved_chunk_size = config.chunk_size if config is not None else chunk_size
    resolved_topk = config.topk if config is not None else topk
    out, _ = pallas_mosaic_tpu_kernel.moba_pallas_fwd(
        q,
        k,
        v,
        topk=resolved_topk,
        chunk_size=resolved_chunk_size,
        scale=scale,
    )
    return (out, None), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(chunk_size=256, topk=4)

  @override
  def supported_on(self, device: Any) -> bool:
    return True
