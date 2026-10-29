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

# Dimensions.
#   B      batch size
#   T      sequence length
#   H      number of attention heads
#   K      key/query head dimension
#   M      number of memory slots
#   V      value head dimension
#   tt     tile extent of T (chunk_size)

import dataclasses
from typing import Any, ClassVar
import jax
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.raven import (
    base,
    pallas_mosaic_tpu_bwd_kernel,
    pallas_mosaic_tpu_kernel,
)
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning and execution configuration for Pallas TPU Raven."""

  chunk_size: pydantic.PositiveInt = 64


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTpuRavenGSAVjp(op.Op[Any, dict[str, Any], None, Config, Any]):
  """Tokamax Op VJP wrapper for Raven GSA Pallas TPU backward path."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    chunk_size = ba.arguments.get("chunk_size", 64)
    return Config(chunk_size=chunk_size)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu"

  @override
  def _fwd(
      self,
      residuals: tuple[Any, ...],
      out: base.Output,
      dout: base.Output,
      q: Float[Array, "B T H K"],
      k: Float[Array, "B T H K"],
      s: Float[Array, "B T H M"],
      g: Float[Array, "B T H M"],
      v: Float[Array, "B T H V"] | None = None,
      *,
      chunk_size: int = 64,
      scale: float | None = None,
      initial_state: base.State = None,
      output_final_state: bool = False,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[dict[str, Any], None]:
    del out, q, k, s, g, v, chunk_size, scale
    del initial_state, output_final_state, return_residuals, config
    (
        res_q,
        res_k,
        res_s,
        res_g,
        res_v,
        res_s0,
        res_s1_in_c,
        res_qv,
        res_s2_in_c,
        res_chunk_size,
        res_scale,
    ) = residuals
    grad_out, grad_final_state = dout

    dq, dk, ds, dg, dv, ds0 = pallas_mosaic_tpu_bwd_kernel.raven_pallas_bwd(
        res_q,
        res_k,
        res_s,
        res_g,
        res_v,
        grad_out,
        grad_final_state=grad_final_state,
        chunk_size=res_chunk_size,
        scale=res_scale,
        initial_state=res_s0,
        s1_in_c=res_s1_in_c,
        qv=res_qv,
        s2_in_c=res_s2_in_c,
    )
    grads: dict[str, Any] = {"q": dq, "k": dk, "s": ds, "g": dg}
    if res_v is not None:
      grads["v"] = dv
    if res_s0 is not None:
      grads["initial_state"] = ds0
    return grads, None


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTpuRavenGSA(base.RavenGSA[Config]):
  """Tokamax operator wrapper for Raven GSA Pallas TPU kernel."""

  config_cls: ClassVar[type[Config]] = Config
  vjp: op.Op | None = PallasTpuRavenGSAVjp()

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array, "B T H K"],
      k: Float[Array, "B T H K"],
      s: Float[Array, "B T H M"],
      g: Float[Array, "B T H M"],
      v: Float[Array, "B T H V"] | None = None,
      *,
      chunk_size: int = 64,
      scale: float | None = None,
      initial_state: base.State = None,
      output_final_state: bool = False,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[base.Output, Any]:
    resolved_chunk_size = (
        config.chunk_size if config is not None else chunk_size
    )
    if return_residuals:
      out, (s1_in_c, qv, s2_in_c) = pallas_mosaic_tpu_kernel.raven_pallas_fwd(
          q,
          k,
          s,
          g,
          v,
          chunk_size=resolved_chunk_size,
          scale=scale,
          initial_state=initial_state,
          output_final_state=output_final_state,
          return_residuals=True,
      )
      residuals = (
          q,
          k,
          s,
          g,
          v,
          initial_state,
          s1_in_c,
          qv,
          s2_in_c,
          resolved_chunk_size,
          scale,
      )
      return out, residuals

    out = pallas_mosaic_tpu_kernel.raven_pallas_fwd(
        q,
        k,
        s,
        g,
        v,
        chunk_size=resolved_chunk_size,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        return_residuals=False,
    )
    return out, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    chunk_size = ba.arguments.get("chunk_size", 64)
    return Config(chunk_size=chunk_size)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu"
