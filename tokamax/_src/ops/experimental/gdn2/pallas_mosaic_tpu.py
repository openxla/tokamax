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

import dataclasses
from typing import Annotated, Any, ClassVar

import jax
import pydantic
from tokamax._src.ops import op
from tokamax._src.ops.experimental.gdn2 import base, pallas_mosaic_tpu_bwd_kernel, pallas_mosaic_tpu_kernel
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning and execution configuration for Pallas TPU GDN-2."""

  chunk_size: Annotated[int, pydantic.Field(gt=0)] = 64


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTpuGatedDeltaNet2Vjp(
    op.Op[Any, dict[str, jax.Array], None, Config, Any]
):
  """Tokamax Op VJP wrapper for GDN-2 Pallas TPU backward path."""

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
          jax.Array,
          jax.Array,
          jax.Array | None,
          jax.Array,
          jax.Array,
          jax.Array,
          int,
          float | None,
      ],
      out: base.Output,
      dout: base.Output,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g: jax.Array,
      b: jax.Array,
      w: jax.Array,
      *,
      chunk_size: int = 64,
      scale: float | None = None,
      initial_state: jax.Array | None = None,
      output_final_state: bool = False,
      config: Config | None = None,
  ) -> tuple[dict[str, jax.Array], None]:
    del out, q, k, v, g, b, w, initial_state, output_final_state, config
    (
        res_q,
        res_k,
        res_v,
        res_g,
        res_b,
        res_w,
        res_s0,
        res_a_mat_c,
        res_merged_out_c,
        res_s_in_c,
        res_chunk_size,
        res_scale,
    ) = residuals
    grad_out, grad_final_state = dout

    dq, dk, dv, dg, db, dw, ds0 = pallas_mosaic_tpu_bwd_kernel.gdn2_pallas_bwd(
        res_q,
        res_k,
        res_v,
        res_g,
        res_b,
        res_w,
        grad_out,
        grad_final_state=grad_final_state,
        chunk_size=res_chunk_size,
        scale=res_scale,
        initial_state=res_s0,
        a_mat_c=res_a_mat_c,
        merged_out_c=res_merged_out_c,
        s_in_c=res_s_in_c,
    )

    grads = {
        "q": dq,
        "k": dk,
        "v": dv,
        "g": dg,
        "b": db,
        "w": dw,
    }
    if res_s0 is not None:
      grads["initial_state"] = ds0
    return grads, None


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTpuGatedDeltaNet2(base.GatedDeltaNet2[Config]):
  """Tokamax operator wrapper for GDN-2 Pallas TPU kernel."""

  config_cls: ClassVar[type[Config]] = Config
  vjp: op.Op | None = PallasTpuGatedDeltaNet2Vjp()

  @override
  def _fwd(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      g: jax.Array,
      b: jax.Array,
      w: jax.Array,
      *,
      chunk_size: int = 64,
      scale: float | None = None,
      initial_state: jax.Array | None = None,
      output_final_state: bool = False,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[base.Output, Any]:
    del return_residuals
    resolved_chunk_size = config.chunk_size if config is not None else chunk_size
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
    residuals = (
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
    )
    return out, residuals

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(chunk_size=64)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu"
