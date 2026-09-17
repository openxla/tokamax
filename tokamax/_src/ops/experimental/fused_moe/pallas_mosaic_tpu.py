# Copyright 2026 Google LLC
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
"""Pallas Mosaic TPU operator implementation for Fused MoE."""

from typing import ClassVar
import jax
import jax.experimental.pallas.tpu as pltpu
from jaxtyping import Array, Float
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.fused_moe import base
from tokamax._src.ops.experimental.fused_moe import host as host_lib
from tokamax._src.ops.experimental.fused_moe import layer as layer_lib
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Autotuning parameters class for Pallas Mosaic TPU Fused MoE."""
  # TODO: Add pydantic bounds and bring in tunable params from
  # host.py
  capacity: int = 128
  block: int | None = None
  ragged_stride: int | None = None
  rhs_qb: int | None = None
  sharded_plan: bool = False


class PallasMosaicTpuFusedMoe(base.FusedMoe):
  """Tokamax operator wrapper for custom Pallas Mosaic TPU Fused MoE kernel."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      x: Float[Array, "T H"],
      w1: Array,
      w2: Array,
      gating: Float[Array, "T E"],
      w1_scale: Array | None = None,
      w2_scale: Array | None = None,
      w1_bias: Array | None = None,
      w2_bias: Array | None = None,
      *,
      topk: int = 2,
      renormalize: bool = True,
      act_fn: str = "silu",
      mesh: jax.sharding.Mesh,
      config: Config | None = None,
  ) -> tuple[jax.Array, None]:
    if config is None:
      config = Config()

    inferred_wf = host_lib.weight_format_of_dtype(w1.dtype)
    weight_format = (
        inferred_wf if inferred_wf is not None else host_lib.WeightFormat.FP8
    )
    # w1 shape: [E, H, 2 * I] (gate+up), w2 shape: [E, I, H] (down projection).
    # Dtype dictates weight format: bfloat16 (unquantized), float8_e4m3fn (fp8),
    # int8 (int8), or uint32 (packed 4-bit fp4 where last dim is packed by 8).
    return layer_lib.fused_ep_moe_v2(
        x=x,
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        gating=gating,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        topk=topk,
        renormalize=renormalize,
        mesh=mesh,
        capacity=config.capacity,
        block=config.block,
        ragged_stride=config.ragged_stride,
        weight_format=weight_format,
        rhs_qb=config.rhs_qb,
        act_fn=act_fn,
        sharded_plan=config.sharded_plan,
    ), None

  @override
  def _get_heuristics_config(self, ba):
    return Config()

  @override
  def _get_autotuning_configs(self, ba):
    return {self._get_heuristics_config(ba)}

  @override
  def supported_on(self, device):
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5

