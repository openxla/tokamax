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

"""Tokamax operator wrapper for RWKV-7 Pallas TPU kernel."""

import dataclasses
from typing import Annotated, Any, ClassVar

import jax
import pydantic
from tokamax._src.ops import op
from tokamax._src.ops.experimental.rwkv7 import base
from tokamax._src.ops.experimental.rwkv7 import pallas_mosaic_tpu_bwd_kernel
from tokamax._src.ops.experimental.rwkv7 import pallas_mosaic_tpu_kernel
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning and execution configuration for Pallas TPU RWKV-7."""

  chunk_size: Annotated[int, pydantic.Field(gt=0)] = 64


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasTpuRWKV7Vjp(op.Op[Any, dict[str, Any], None, Config, None]):
  """Analytical Pallas TPU VJP operator for RWKV-7."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  @override
  def _fwd(
      self,
      residuals: base.Residuals,
      out: Any,
      dout: Any,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      alpha: jax.Array,
      beta: jax.Array,
      gk: jax.Array,
      *,
      chunk_size: int = 64,
      scale: float | None = None,
      initial_state: jax.Array | None = None,
      output_final_state: bool = False,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[dict[str, Any], None]:
    del out, output_final_state, return_residuals
    resolved_chunk_size = (
        config.chunk_size if config is not None else chunk_size
    )
    if isinstance(dout, tuple):
      grad_out, grad_final_state = dout
    else:
      grad_out, grad_final_state = dout, None

    a_ab_inv = None
    merged_out = None
    s_in = None
    if isinstance(residuals, tuple) and len(residuals) == 9:
      q, k, v, alpha, beta, gk, a_ab_inv, merged_out, s_in = residuals
    elif isinstance(residuals, tuple) and len(residuals) == 7:
      q, k, v, alpha, beta, gk, s_in = residuals

    dq, dk, dv, dalpha, dbeta, dgk, d_s0 = (
        pallas_mosaic_tpu_bwd_kernel.rwkv7_pallas_bwd(
            q,
            k,
            v,
            alpha,
            beta,
            gk,
            grad_out,
            chunk_size=resolved_chunk_size,
            scale=scale,
            initial_state=initial_state,
            a_ab_inv=a_ab_inv,
            merged_out=merged_out,
            s_in=s_in,
            d_final_state=grad_final_state,
        )
    )
    grads: dict[str, Any] = {
        "q": dq,
        "k": dk,
        "v": dv,
        "alpha": dalpha,
        "beta": dbeta,
        "gk": dgk,
    }
    if initial_state is not None:
      grads["initial_state"] = d_s0
    return grads, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(chunk_size=64)

  @override
  def supported_on(self, device: Any) -> bool:
    return True


PallasTpuRwkv7Vjp = PallasTpuRWKV7Vjp


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTpuRWKV7(base.RWKV7[Config]):
  """Tokamax operator wrapper for RWKV-7 Pallas TPU kernel."""

  config_cls: ClassVar[type[Config]] = Config

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(self, "vjp", PallasTpuRWKV7Vjp())

  @override
  def _fwd(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      alpha: jax.Array,
      beta: jax.Array,
      gk: jax.Array,
      *,
      chunk_size: int = 64,
      scale: float | None = None,
      initial_state: jax.Array | None = None,
      output_final_state: bool = False,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[base.Output, base.Residuals]:
    resolved_chunk_size = (
        config.chunk_size if config is not None else chunk_size
    )
    if return_residuals:
      (
          out,
          final_state,
          a_ab_inv,
          merged_out,
          s_in,
      ) = pallas_mosaic_tpu_kernel.rwkv7_pallas_fwd(
          q,
          k,
          v,
          alpha,
          beta,
          gk,
          chunk_size=resolved_chunk_size,
          scale=scale,
          initial_state=initial_state,
          output_final_state=output_final_state,
          return_residuals=True,
      )
      return (out, final_state), (
          q,
          k,
          v,
          alpha,
          beta,
          gk,
          a_ab_inv,
          merged_out,
          s_in,
      )

    out, final_state = pallas_mosaic_tpu_kernel.rwkv7_pallas_fwd(
        q,
        k,
        v,
        alpha,
        beta,
        gk,
        chunk_size=resolved_chunk_size,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        return_residuals=False,
    )
    return (out, final_state), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(chunk_size=64)

  @override
  def supported_on(self, device: Any) -> bool:
    return True
