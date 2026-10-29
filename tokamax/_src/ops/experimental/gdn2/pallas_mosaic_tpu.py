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

"""Tokamax operator wrapper for GDN-2 Pallas TPU kernel."""

# Dimensions.
#   B   batch size
#   T   sequence length (nt * tt)
#   H   number of attention heads
#   K   key head dimension
#   V   value head dimension
#   BH  flattened batch-head count (B * H)
#   tt  tile extent of T (chunk_size)
#   nt  grid extent along T (T // tt)

import dataclasses
from typing import Any, ClassVar

import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import (  # pylint: disable=g-multiple-import,g-importing-member
    Array,
    Float,
)
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.gdn2 import (
    base,
    pallas_mosaic_tpu_bwd_kernel,
    pallas_mosaic_tpu_kernel,
)
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning and execution configuration for Pallas TPU GDN-2."""

  chunk_size: pydantic.PositiveInt = 64

  def __post_init__(self) -> None:
    if self.chunk_size & (self.chunk_size - 1) != 0:
      raise ValueError(
          f"chunk_size must be a power of 2, got {self.chunk_size}."
      )


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicTpuGatedDeltaNet2Vjp(base.GatedDeltaNet2Vjp[Config]):
  """Tokamax Op VJP wrapper for GDN-2 Pallas Mosaic TPU backward path."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    chunk_size = ba.arguments.get("chunk_size", 64)
    return Config(chunk_size=chunk_size)

  @override
  def _fwd(
      self,
      residuals: base.Residuals,
      out: base.Output,
      dout: base.Output,
      q: Float[Array, "B T H K"],
      k: Float[Array, "B T H K"],
      v: Float[Array, "B T H V"],
      g: Float[Array, "B T H K"],
      b: Float[Array, "B T H K"],
      w: Float[Array, "B T H V"],
      *,
      chunk_size: int = 64,
      scale: float | None = None,
      initial_state: Float[Array, "B H K V"] | None = None,
      output_final_state: bool = False,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[dict[str, jax.Array], None]:
    del out, output_final_state, return_residuals
    resolved_chunk_size = (
        config.chunk_size if config is not None else chunk_size
    )
    a_mat_c = None
    merged_out_c = None
    s_in_c = None
    if isinstance(residuals, tuple) and len(residuals) == 12:
      (
          q,
          k,
          v,
          g,
          b,
          w,
          initial_state,
          a_mat_c,
          merged_out_c,
          s_in_c,
          resolved_chunk_size,
          scale,
      ) = residuals
    elif isinstance(residuals, tuple) and len(residuals) == 3:
      a_mat_c, merged_out_c, s_in_c = residuals

    if isinstance(dout, tuple):
      grad_out, grad_final_state = dout
    else:
      grad_out, grad_final_state = dout, None

    dq, dk, dv, dg, db, dw, ds0 = pallas_mosaic_tpu_bwd_kernel.gdn2_pallas_bwd(
        q,
        k,
        v,
        g,
        b,
        w,
        grad_out,
        grad_final_state=grad_final_state,
        chunk_size=resolved_chunk_size,
        scale=scale,
        initial_state=initial_state,
        a_mat_c=a_mat_c,
        merged_out_c=merged_out_c,
        s_in_c=s_in_c,
    )

    grads: dict[str, Any] = {
        "q": dq,
        "k": dk,
        "v": dv,
        "g": dg,
        "b": db,
        "w": dw,
    }
    if initial_state is not None and ds0 is not None:
      grads["initial_state"] = ds0
    return grads, None

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicTpuGatedDeltaNet2(base.GatedDeltaNet2[Config]):
  """Tokamax operator wrapper for GDN-2 Pallas Mosaic TPU kernel."""

  config_cls: ClassVar[type[Config]] = Config

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(
          self, "vjp", PallasMosaicTpuGatedDeltaNet2Vjp(config=self.config)
      )

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array, "B T H K"],
      k: Float[Array, "B T H K"],
      v: Float[Array, "B T H V"],
      g: Float[Array, "B T H K"],
      b: Float[Array, "B T H K"],
      w: Float[Array, "B T H V"],
      *,
      chunk_size: int = 64,
      scale: float | None = None,
      initial_state: Float[Array, "B H K V"] | None = None,
      output_final_state: bool = False,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[base.Output, Any]:
    resolved_chunk_size = (
        config.chunk_size if config is not None else chunk_size
    )
    out, (a_mat_c, merged_out_c, s_in_c) = (
        pallas_mosaic_tpu_kernel.gdn2_pallas_fwd(
            q,
            k,
            v,
            g,
            b,
            w,
            chunk_size=resolved_chunk_size,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            return_residuals=True,
        )
    )
    residuals = (a_mat_c, merged_out_c, s_in_c)
    return out, (residuals if return_residuals else None)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    chunk_size = ba.arguments.get("chunk_size", 64)
    return Config(chunk_size=chunk_size)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
