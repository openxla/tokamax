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
"""Pallas Mosaic TPU operator implementation for Fused MoE Decode."""

import functools
from typing import Annotated, ClassVar, override

import jax
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec  # pylint: disable=g-multiple-import,g-importing-member
from jaxtyping import Array, Float, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.fused_moe_decode import base
from tokamax._src.ops.experimental.fused_moe_decode import fused_moe_decode_kernel

_BlockWidth = Annotated[int, pydantic.Field(ge=128, multiple_of=128)]
_NumBuffers = Annotated[int, pydantic.Field(ge=3)]


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning parameters for Pallas Mosaic TPU Fused MoE Decode.

  Attributes:
    i_block: Intermediate block width for blocked weight-ring streaming. When
      `None`, `plan_weight_ring` selects whole-expert or blocked mode
      automatically based on VMEM capacity.
    n_chunk: Output columns per second-matmul step.
    num_buffers: Upper bound on the weight ring depth (at least 3).
    q_block: Columns per FP8 activation quantization scale block.
  """

  i_block: _BlockWidth | None = None
  n_chunk: _BlockWidth = 512
  num_buffers: _NumBuffers = 6
  q_block: _BlockWidth = 512


def _in_mesh_context(axis_name: str = "d") -> bool:
  """Returns whether execution is currently inside a `shard_map` mesh context."""
  if not jax.sharding.get_abstract_mesh().empty:
    return True
  try:
    jax.lax.axis_index(axis_name)
    return True
  except NameError:
    return False


class PallasMosaicTpuFusedMoeDecode(base.FusedMoeDecode[Config]):
  """Tokamax operator wrapper for the Pallas Mosaic TPU Fused MoE Decode kernel."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      rank: Int[Array, "*#r"],
      hidden: Float[Array, "tokens H"],
      w1: Shaped[Array, "E H two_i"],
      w2: Shaped[Array, "E i H"],
      w1_scale: Float[Array, "E 1 1 two_i"],
      w2_scale: Float[Array, "E 1 1 H"],
      shared_w1: Shaped[Array, "H two_is"],
      shared_w1_scale: Float[Array, "1 two_is"],
      shared_w2: Shaped[Array, "i_s H"],
      shared_w2_scale: Float[Array, "1 H"],
      shared_gate_weight: Float[Array, "H 1"] | None,
      topk_weights: Float[Array, "tokens k"],
      topk_ids: Int[Array, "tokens k"],
      *,
      n_ranks: int = 1,
      expert_start: Int[Array, "*#es"] | None = None,
      scatter_rows: bool = False,
      routed_scale: float = 1.0,
      axis_name: str | None = None,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[jax.Array, None]:
    del return_residuals
    if config is None:
      config = Config()

    call_fn = functools.partial(
        fused_moe_decode_kernel.fused_moe_decode,
        n_ranks=n_ranks,
        scatter_rows=scatter_rows,
        routed_scale=routed_scale,
        i_block=config.i_block,
        n_chunk=config.n_chunk,
        num_buffers=config.num_buffers,
        q_block=config.q_block,
    )

    ax = axis_name or "d"
    if not _in_mesh_context(ax):
      if n_ranks > 1:
        raise ValueError(
            f"Multi-rank fused_moe_decode (n_ranks={n_ranks}) must be called"
            " inside a jax.shard_map context."
        )
      mesh = Mesh(np.asarray(jax.devices()[:1]), (ax,))
      spec = PartitionSpec()
      in_specs = (
          spec,
          spec,
          spec,
          spec,
          spec,
          spec,
          spec,
          spec,
          spec,
          spec,
          None if shared_gate_weight is None else spec,
          spec,
          spec,
          None if expert_start is None else spec,
      )
      out = jax.shard_map(
          lambda r, h, w1_, w2_, s1, s2, sw1, ss1, sw2, ss2, sgw, tw, ti, es: (
              call_fn(
                  r,
                  h,
                  w1_,
                  w2_,
                  s1,
                  s2,
                  sw1,
                  ss1,
                  sw2,
                  ss2,
                  sgw,
                  tw,
                  ti,
                  expert_start=es,
              )
          ),
          mesh=mesh,
          in_specs=in_specs,
          out_specs=spec,
          check_vma=False,
      )(
          rank,
          hidden,
          w1,
          w2,
          w1_scale,
          w2_scale,
          shared_w1,
          shared_w1_scale,
          shared_w2,
          shared_w2_scale,
          shared_gate_weight,
          topk_weights,
          topk_ids,
          expert_start,
      )
      return out, None

    out = call_fn(
        rank,
        hidden,
        w1,
        w2,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w2,
        shared_w2_scale,
        shared_gate_weight,
        topk_weights,
        topk_ids,
        expert_start=expert_start,
    )
    return out, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config()

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    i = ba.arguments["w2"].shape[1]
    configs = {Config(i_block=None)}
    for width in (128, 256):
      if width < i and i % width == 0:
        configs.add(Config(i_block=width))
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 7
