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

"""Tokamax operator wrapper for Native Sparse Attention (NSA)."""

import dataclasses
from typing import Annotated, Any, ClassVar
import jax
import pydantic
from tokamax._src.ops import op
from tokamax._src.ops.experimental.nsa import base
from tokamax._src.ops.experimental.nsa import pallas_mosaic_tpu_kernel
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning and execution configuration for Pallas TPU NSA."""

  chunk_size: Annotated[int, pydantic.Field(gt=0)] = 256
  topk: Annotated[int, pydantic.Field(gt=0)] = 4
  window: Annotated[int, pydantic.Field(gt=0)] = 128
  cmp_block_size: Annotated[int, pydantic.Field(gt=0)] = 64


class PallasTpuNativeSparseAttention(base.NativeSparseAttention[Config]):
  """Tokamax operator wrapper for NSA Pallas TPU kernel."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  def _fwd(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g_cmp: jax.Array | None = None,
      g_slc: jax.Array | None = None,
      g_swa: jax.Array | None = None,
      *,
      chunk_size: int = 256,
      topk: int = 4,
      window: int = 128,
      cmp_block_size: int = 64,
      scale: float | None = None,
      config: Config | None = None,
  ) -> tuple[base.Output, None]:
    resolved_chunk_size = config.chunk_size if config is not None else chunk_size
    resolved_topk = config.topk if config is not None else topk
    resolved_window = config.window if config is not None else window
    resolved_cmp_size = config.cmp_block_size if config is not None else cmp_block_size

    out, _ = pallas_mosaic_tpu_kernel.nsa_pallas_fwd(
        q,
        k,
        v,
        g_cmp=g_cmp,
        g_slc=g_slc,
        g_swa=g_swa,
        chunk_size=resolved_chunk_size,
        topk=resolved_topk,
        window=resolved_window,
        cmp_block_size=resolved_cmp_size,
        scale=scale,
    )
    return (out, None), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(chunk_size=256, topk=4, window=128, cmp_block_size=64)

  @override
  def supported_on(self, device: Any) -> bool:
    return True
