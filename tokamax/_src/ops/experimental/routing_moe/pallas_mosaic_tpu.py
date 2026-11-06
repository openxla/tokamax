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
import jax.experimental.pallas.tpu as pltpu
from jaxtyping import Array
from jaxtyping import Float
import pydantic

# Routing gmm (`gmm/memory_ref.py`) still checks `isinstance` against
# `pltpu.BufferedRefBase`, which is deprecated and warns on access. Its refs all
# subclass `pltpu.BufferedRef`, so alias that name before `gmm` is used.
# TODO: Drop this once `memory_ref.py` uses `pltpu.BufferedRef`.
vars(pltpu)["BufferedRefBase"] = pltpu.BufferedRef
from tokamax._src import jaxtyping  # pylint: disable=g-import-not-at-top,g-bad-import-order
from tokamax._src.ops import op
from tokamax._src.ops.experimental.routing_moe import base
from tokamax._src.ops.experimental.routing_moe import gmm
from tokamax._src.ops.experimental.routing_moe import layer
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Autotuning parameters for the routing MoE implementation.

  Every field left as `None` is resolved by routing MoE's own heuristics, so the
  default `Config()` runs the kernel exactly as `gated_routing_moe` would.

  Attributes:
    use_sp: Whether tokens are sharded over the expert-parallel axis (sequence
      parallelism). The Routing MoE argument specifications shard `x` and
      `gating` along tokens, which this matches; with `False` the inputs are
      replicated on entry instead.
    remote_scatter: Whether routing gmm writes each output row straight to the
      device that owns its token, instead of reducing across devices.
    tile_m: Rows per routing gmm tile.
    tile_k: Contraction tile of the gate/up projection.
    tile_n: Output tile of the gate/up projection.
    num_xo_buffers: Activation and output buffers kept in flight.
    num_w_buffers: Weight buffers kept in flight.
  """

  use_sp: bool = True
  remote_scatter: bool | None = None
  tile_m: pydantic.PositiveInt | None = None
  tile_k: pydantic.PositiveInt | None = None
  tile_n: pydantic.PositiveInt | None = None
  num_xo_buffers: pydantic.PositiveInt | None = None
  num_w_buffers: pydantic.PositiveInt | None = None


def _as_block_scale(scale: jax.Array | None) -> jax.Array | None:
  """Returns `scale` in routing MoE's `[E, num_blocks, 1, N]` layout.

  Routing MoE argument specs take per-channel scales as `[E, N]`, which is one
  quantization block spanning the whole contraction dimension, and per-block
  scales as `[E, num_blocks, N]`. Scales already in the 4-D layout pass through
  unchanged.

  Args:
    scale: A weight scale, `[E, N]`, `[E, num_blocks, N]` or
      `[E, num_blocks, 1, N]`, or `None`.

  Returns:
    The scale as `[E, num_blocks, 1, N]`, or `None`.

  Raises:
    NotImplementedError: For any other scale rank.
  """
  if scale is None:
    return None
  match scale.ndim:
    case 2:
      return scale[:, None, None, :]
    case 3:
      return scale[:, :, None, :]
    case 4:
      return scale
  raise NotImplementedError(
      f"Routing MoE does not support weight scales of shape {scale.shape}."
  )


# Autotuning candidates. `tile_k` and `tile_n` stay with routing gmm's own
# heuristics, which size them to the shapes being run.
_TILE_M_CANDIDATES: Final[tuple[int, ...]] = (64, 128, 256)
# (num_xo_buffers, num_w_buffers) pairs.
_BUFFER_CANDIDATES: Final[tuple[tuple[int, int], ...]] = (
    (3, 2),
    (3, 3),
    (4, 2),
    (4, 3),
)
_REMOTE_SCATTER_CANDIDATES: Final[tuple[bool, ...]] = (False, True)


class PallasMosaicTpuRoutingMoe(base.RoutingMoe):
  """Routing MoE computed by routing MoE's routing gmm kernels."""

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
      scoring_fn: str = "softmax",
      mesh: jax.sharding.Mesh | None = None,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[jax.Array, None]:
    del return_residuals  # Unused.
    if config is None:
      config = Config()
    if w1_bias is not None or w2_bias is not None:
      raise NotImplementedError("Routing MoE does not support biases yet.")
    if scoring_fn != "softmax":
      raise NotImplementedError(
          f"Routing MoE only supports softmax scoring, got {scoring_fn!r}."
      )
    if mesh is None:
      raise NotImplementedError("Routing MoE requires an expert-parallel mesh.")
    if len(mesh.axis_names) != 1:
      raise NotImplementedError(
          "Routing MoE supports only a 1-D expert-parallel mesh, got axes"
          f" {mesh.axis_names}."
      )

    gmm_plan = gmm.RoutingGmmPlan(
        tile_m=config.tile_m,
        tile_k=config.tile_k,
        tile_n=config.tile_n,
        num_xo_buffers=config.num_xo_buffers,
        num_w_buffers=config.num_w_buffers,
    )
    moe_plan = layer.RoutingMoEPlan(
        remote_scatter=config.remote_scatter
    )
    out = layer.gated_routing_moe(
        hidden_states=x,
        gating_output=gating,
        w1=w1,
        w2=w2,
        w1_scale=_as_block_scale(w1_scale),
        w2_scale=_as_block_scale(w2_scale),
        w1_bias=None,
        w2_bias=None,
        topk=topk,
        renormalize=renormalize,
        mesh=mesh,
        activation=act_fn,
        scoring_fn=scoring_fn,
        tp_axis=None,
        ep_axis=mesh.axis_names[0],
        use_sp=config.use_sp,
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
    # This is a fixed grid, not filtered by shape: routing gmm rejects invalid
    # configs when it compiles them, and Tokamax's autotuner records those
    # failures and picks the fastest config that works.
    del ba
    return {
        Config(
            remote_scatter=remote_scatter,
            tile_m=tile_m,
            num_xo_buffers=num_xo_buffers,
            num_w_buffers=num_w_buffers,
        )
        for tile_m, (num_xo_buffers, num_w_buffers), remote_scatter in (
            itertools.product(
                _TILE_M_CANDIDATES,
                _BUFFER_CANDIDATES,
                _REMOTE_SCATTER_CANDIDATES,
            )
        )
    }

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 7
