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
"""Pallas Mosaic TPU implementation of the Routing MoE op."""

import itertools
from typing import ClassVar, Final
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan_lib
from tokamax._src.ops.experimental.routing_moe.layer import modular
from tokamax._src.ops.experimental.routing_moe.layer import plan as moe_plan_lib
from tokamax._src.ops.experimental.routing_moe_op import base
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Autotuning parameters for the routing MoE implementation."""

  remote_scatter: bool | None = None
  tile_m: pydantic.PositiveInt | None = None
  tile_k: pydantic.PositiveInt | None = None
  tile_n: pydantic.PositiveInt | None = None
  num_xo_buffers: pydantic.PositiveInt | None = None
  num_w_buffers: pydantic.PositiveInt | None = None


_TILE_M_CANDIDATES: Final[tuple[int, ...]] = (64, 128, 256)
_BUFFER_CANDIDATES: Final[tuple[tuple[int, int], ...]] = (
    (3, 2),
    (3, 3),
    (4, 2),
    (4, 3),
)
_REMOTE_SCATTER_CANDIDATES: Final[tuple[bool, ...]] = (False, True)


def _tpu_generation() -> int:
  try:
    return pltpu.get_tpu_info().generation
  except (RuntimeError, ValueError, AttributeError):
    return 0


def _is_fp8_weight(w1: Array, w1_scale: Array | None) -> bool:
  return w1.dtype == jnp.float8_e4m3fn or (
      w1_scale is not None and w1.dtype.itemsize == 1
  )


class PallasMosaicTpuRoutingMoe(base.RoutingMoe):
  """Mosaic TPU implementation for routing MoE."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      hidden_states: Float[Array, "T H"],
      topk_weights: Float[Array, "T K"],
      topk_indices: Int[Array, "T K"],
      w1: Array,
      w2: Array,
      w1_scale: Array | None = None,
      w2_scale: Array | None = None,
      w1_bias: Array | None = None,
      w2_bias: Array | None = None,
      mesh: jax.sharding.Mesh | None = None,
      activation: str = "silu",
      tp_axis: str | None = None,
      ep_axis: str | None = None,
      use_sp: bool = False,
      reduce_output: bool = True,
      *,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[jax.Array, None]:
    del return_residuals
    if config is None:
      config = Config()
    if mesh is None:
      raise NotImplementedError("Routing MoE requires a device mesh.")
    if w1_bias is not None or w2_bias is not None:
      raise NotImplementedError("Bias is currently not supported.")

    if _tpu_generation() < 7 and _is_fp8_weight(w1, w1_scale):
      raise NotImplementedError(
          "Routing MoE Mosaic TPU kernel requires TPU generation >= 7 for FP8."
      )

    gmm_plan = gmm_plan_lib.RoutingGmmPlan(
        tile_m=config.tile_m,
        tile_k=config.tile_k,
        tile_n=config.tile_n,
        num_xo_buffers=config.num_xo_buffers,
        num_w_buffers=config.num_w_buffers,
    )
    moe_plan = moe_plan_lib.RoutingMoEPlan(remote_scatter=config.remote_scatter)

    out = modular.routing_moe(
        hidden_states=hidden_states,
        topk_weights=topk_weights.astype(jnp.float32),
        topk_indices=topk_indices.astype(jnp.int32),
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        mesh=mesh,
        activation=activation,
        tp_axis=tp_axis,
        ep_axis=ep_axis,
        use_sp=use_sp,
        reduce_output=reduce_output,
        moe_plan=moe_plan,
        gmm_plan=gmm_plan,
    )
    return out, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config()

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    args = ba.arguments
    allow_remote_scatter = (
        args.get("ep_axis") is not None
        and args.get("tp_axis") is None
        and args.get("reduce_output", True)
    )
    remote_scatter_candidates = (
        _REMOTE_SCATTER_CANDIDATES if allow_remote_scatter else (False,)
    )
    return {
        Config(
            remote_scatter=remote_scatter,
            tile_m=tile_m,
            num_xo_buffers=num_xo_buffers,
            num_w_buffers=num_w_buffers,
        )
        for tile_m, (
            num_xo_buffers,
            num_w_buffers,
        ), remote_scatter in itertools.product(
            _TILE_M_CANDIDATES,
            _BUFFER_CANDIDATES,
            remote_scatter_candidates,
        )
    }

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu"
