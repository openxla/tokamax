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
from tokamax._src.ops.experimental.raven import (
    base,
    pallas_mosaic_tpu_bwd_kernel,
    pallas_mosaic_tpu_kernel,
)
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning and execution configuration for Pallas TPU Raven."""

  chunk_size: Annotated[int, pydantic.Field(gt=0)] = 64


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTpuRavenGSAVjp(op.Op[Any, dict[str, jax.Array], None, Config, Any]):
  """Tokamax Op VJP wrapper for Raven GSA Pallas TPU backward path."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(chunk_size=64)

  def _fwd(
      self,
      residuals: tuple[
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array | None,
          jax.Array,
          int,
      ],
      out: base.Output,
      dout: base.Output,
      q: jax.Array,
      k: jax.Array,
      s: jax.Array,
      g: jax.Array,
      *,
      chunk_size: int = 64,
      initial_state: jax.Array | None = None,
      output_final_state: bool = False,
      config: Config | None = None,
  ) -> tuple[dict[str, jax.Array], None]:
    del out, q, k, s, g, chunk_size, initial_state, output_final_state, config
    res_q, res_k, res_s, res_g, res_s0, res_s1_in_c, res_chunk_size = residuals
    grad_out, grad_final_state = dout

    dq, dk, ds, dg, ds0 = pallas_mosaic_tpu_bwd_kernel.raven_pallas_stage1_bwd(
        res_q,
        res_k,
        res_s,
        res_g,
        grad_out,
        grad_final_state=grad_final_state,
        chunk_size=res_chunk_size,
        initial_state=res_s0,
        s1_in_c=res_s1_in_c,
    )
    grads = {"q": dq, "k": dk, "s": ds, "g": dg}
    if res_s0 is not None:
      grads["initial_state"] = ds0
    return grads, None


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTpuRavenGSA(base.RavenGSA[Config]):
  """Tokamax operator wrapper for Raven GSA Pallas TPU kernel."""

  config_cls: ClassVar[type[Config]] = Config
  vjp: op.Op | None = PallasTpuRavenGSAVjp()

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
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[base.Output, Any]:
    del return_residuals
    resolved_chunk_size = config.chunk_size if config is not None else chunk_size
    out, s1_in_c = pallas_mosaic_tpu_kernel.raven_pallas_stage1_fwd(
        q,
        k,
        s,
        g,
        chunk_size=resolved_chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
        return_residuals=True,
    )
    residuals = (q, k, s, g, initial_state, s1_in_c, resolved_chunk_size)
    return out, residuals

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(chunk_size=64)

  @override
  def supported_on(self, device: Any) -> bool:
    return True
