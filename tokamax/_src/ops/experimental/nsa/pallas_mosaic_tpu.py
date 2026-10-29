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

"""Pallas/Mosaic TPU operator implementation for Native Sparse Attention (NSA)."""

# Dimensions.
#   B   batch size
#   H   number of attention heads
#   T   sequence length
#   D   head dimension
#   tt  query/KV block tile extent along T (chunk_size)
#   tc  compression block tile extent along T (cmp_block_size)

from typing import ClassVar

import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.nsa import base
from tokamax._src.ops.experimental.nsa import pallas_mosaic_tpu_kernel
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning and execution configuration for Pallas TPU NSA."""

  chunk_size: pydantic.PositiveInt = 256
  topk: pydantic.PositiveInt = 4
  window: pydantic.PositiveInt = 128
  cmp_block_size: pydantic.PositiveInt = 64

  def __post_init__(self):
    if self.chunk_size < self.cmp_block_size:
      raise ValueError(
          f"chunk_size ({self.chunk_size}) must be at least cmp_block_size"
          f" ({self.cmp_block_size})."
      )


class PallasTpuNativeSparseAttention(base.NativeSparseAttention[Config]):
  """Tokamax operator wrapper for NSA Pallas TPU kernel."""

  config_cls: ClassVar[type[Config]] = Config

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array, "B H T D"],
      k: Float[Array, "B H T D"],
      v: Float[Array, "B H T D"],
      g_cmp: Float[Array, "B H T D"] | None = None,
      g_slc: Float[Array, "B H T D"] | None = None,
      g_swa: Float[Array, "B H T D"] | None = None,
      *,
      chunk_size: int = 256,
      topk: int = 4,
      window: int = 128,
      cmp_block_size: int = 64,
      scale: float | None = None,
      config: Config | None = None,
      return_residuals: bool = False,
  ) -> tuple[Float[Array, "B H T D"], None]:
    del return_residuals
    tt = config.chunk_size if config is not None else chunk_size
    resolved_topk = config.topk if config is not None else topk
    resolved_window = config.window if config is not None else window
    tc = config.cmp_block_size if config is not None else cmp_block_size
    if tt <= 0:
      raise NotImplementedError(f"chunk_size must be positive, got {tt}")
    if resolved_topk <= 0:
      raise NotImplementedError(f"topk must be positive, got {resolved_topk}")
    if resolved_window <= 0:
      raise NotImplementedError(
          f"window must be positive, got {resolved_window}"
      )
    if tc <= 0:
      raise NotImplementedError(f"cmp_block_size must be positive, got {tc}")

    out, _ = pallas_mosaic_tpu_kernel.nsa_pallas_fwd(
        q,
        k,
        v,
        g_cmp=g_cmp,
        g_slc=g_slc,
        g_swa=g_swa,
        chunk_size=tt,
        topk=resolved_topk,
        window=resolved_window,
        cmp_block_size=tc,
        scale=scale,
    )
    return out, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config(
        chunk_size=ba.arguments.get("chunk_size", 256),
        topk=ba.arguments.get("topk", 4),
        window=ba.arguments.get("window", 128),
        cmp_block_size=ba.arguments.get("cmp_block_size", 64),
    )

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
