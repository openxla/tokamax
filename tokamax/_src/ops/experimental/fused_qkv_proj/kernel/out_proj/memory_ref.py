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
"""fused_out_proj's inputs, outputs and scratch: their shapes and blocks.

No numerics here.
"""

from typing import Generic, NamedTuple, TypeVar

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import memory
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.out_proj import config as config_lib

# Arrays, ShapeDtypeStructs, BlockSpecs or Refs, depending on the use.
Leaf = TypeVar("Leaf")


def _rows(cfg: config_lib.FusedOutProjConfig, width: int) -> pl.BlockSpec:
  return pl.BlockSpec((cfg.tile.tile_size, width), lambda i: (i, 0))


class Inputs(NamedTuple, Generic[Leaf]):
  """Inputs split into token tiles by the pipeline."""

  # batched_rpa's raw output, before prepare_outputs:
  # (Hkv, T_rpa, q_per_kv / q_packing, q_packing, align(D, 128)) bf16.
  o: Leaf
  gate: Leaf | None  # (T, Hq * D) bf16 sigmoid output gate.

  @classmethod
  def block_specs(
      cls, cfg: config_lib.FusedOutProjConfig
  ) -> "Inputs[pl.BlockSpec]":
    """The pipeline's per-tile blocks: whole tokens of each input."""
    m = cfg.model
    o_spec = pl.BlockSpec(
        (m.num_kv_heads, cfg.o_tile_tokens, *cfg.o_shape[2:]),
        lambda i: (0, i, 0, 0, 0),
    )
    return cls(o=o_spec, gate=_rows(cfg, m.q_dim) if m.gate else None)


class Outputs(NamedTuple, Generic[Leaf]):
  """The output, in the layout the TP all-reduce takes."""

  out: Leaf  # (T, K) bf16, ready for the TP all-reduce.

  @classmethod
  def shapes(
      cls, cfg: config_lib.FusedOutProjConfig
  ) -> "Outputs[jax.ShapeDtypeStruct]":
    """The output's shape and dtype."""
    return cls(
        out=jax.ShapeDtypeStruct(
            (cfg.num_tokens, cfg.hidden_size), jnp.bfloat16
        )
    )

  @classmethod
  def block_specs(
      cls, cfg: config_lib.FusedOutProjConfig
  ) -> "Outputs[pl.BlockSpec]":
    """The pipeline's per-tile blocks."""
    return cls(out=_rows(cfg, cfg.hidden_size))


class Scratch(NamedTuple, Generic[Leaf]):
  """VMEM scratch and semaphores. Fields a config doesn't use are None."""

  x_q: Leaf  # (tile, Hq * D) W's dtype: unpacked, gated and quantized o.
  x_scale: Leaf | None  # fp8: (Hq * D / block, tile, lanes) f32.
  w_vmem: Leaf  # (Hq * D, K) W's VMEM copy.
  w_sems: Leaf  # (len(cfg.w_chunks),) its DMA semaphores.

  @classmethod
  def shapes(
      cls, cfg: config_lib.FusedOutProjConfig, w: jax.Array
  ) -> "Scratch[pl.MemoryRef]":
    """The VMEM scratch and semaphores, including W's VMEM copy."""
    m, tile = cfg.model, cfg.tile.tile_size
    num_lanes = pltpu.get_tpu_info().num_lanes
    w_vmem, w_sems = memory.stream_scratch(w, cfg.w_chunks)
    x_scale = None
    if m.fp8:
      num_blocks = m.q_dim // cfg.quant_block
      x_scale = pltpu.VMEM((num_blocks, tile, num_lanes), jnp.float32)
    return cls(
        x_q=pltpu.VMEM((tile, m.q_dim), w.dtype),
        x_scale=x_scale,
        w_vmem=w_vmem,
        w_sems=w_sems,
    )
