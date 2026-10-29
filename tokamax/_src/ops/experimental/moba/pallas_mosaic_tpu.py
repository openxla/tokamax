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

# Dimensions.
#   B      batch
#   H      number of attention heads
#   T      sequence length
#   D      head dimension
#   tt     query/key block chunk size (chunk_size)
#   nt     grid extent over T: nt = T // tt
#   K      effective number of routed historical blocks per query block

from typing import ClassVar
import jax
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.moba import base
from tokamax._src.ops.experimental.moba import pallas_mosaic_tpu_kernel
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning and execution configuration for Pallas TPU MoBA."""

  chunk_size: pydantic.PositiveInt = 256
  topk: pydantic.PositiveInt = 4


class PallasTpuMixtureOfBlockAttention(base.MixtureOfBlockAttention[Config]):
  """Tokamax operator wrapper for MoBA Pallas TPU kernel."""

  config_cls: ClassVar[type[Config]] = Config

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array, "B H T D"],
      k: Float[Array, "B H T D"],
      v: Float[Array, "B H T D"],
      *,
      topk: int = 4,
      chunk_size: int = 256,
      scale: float | None = None,
      config: Config | None = None,
      return_residuals: bool = False,
  ) -> tuple[Float[Array, "B H T D"], None]:
    del return_residuals
    if q.shape != k.shape or q.shape != v.shape:
      raise ValueError(
          f"Shapes of q, k, v must match, got q={q.shape}, k={k.shape},"
          f" v={v.shape}"
      )
    _, _, T, _ = q.shape
    resolved_chunk_size = (
        config.chunk_size if config is not None else chunk_size
    )
    resolved_topk = config.topk if config is not None else topk
    if resolved_chunk_size <= 0:
      raise NotImplementedError(
          f"chunk_size must be positive, got {resolved_chunk_size}"
      )
    if resolved_topk <= 0:
      raise NotImplementedError(f"topk must be positive, got {resolved_topk}")
    if T % resolved_chunk_size != 0:
      raise ValueError(
          f"Sequence length ({T}) must be divisible by chunk_size"
          f" ({resolved_chunk_size})."
      )

    out, _ = pallas_mosaic_tpu_kernel.moba_pallas_fwd(
        q,
        k,
        v,
        topk=resolved_topk,
        chunk_size=resolved_chunk_size,
        scale=scale,
    )
    return out, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config(
        chunk_size=ba.arguments.get("chunk_size", 256),
        topk=ba.arguments.get("topk", 4),
    )

  @override
  def supported_on(self, device: jax.Device) -> bool:
    try:
      return device.platform == "tpu"
    except Exception:
      return False
