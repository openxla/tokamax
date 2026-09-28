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
"""fused_gdn's inputs, outputs and scratch: their shapes and blocks.

No numerics here.
"""

from typing import Generic, NamedTuple, TypeVar

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import memory
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gdn import config as config_lib

# Arrays, ShapeDtypeStructs, BlockSpecs or Refs, depending on the use.
Leaf = TypeVar("Leaf")


def _rows(cfg: config_lib.FusedGdnConfig, width: int) -> pl.BlockSpec:
  return pl.BlockSpec((cfg.tile.tile_size, width), lambda i: (i, 0))


class Inputs(NamedTuple, Generic[Leaf]):
  """Inputs split into token tiles by the pipeline."""

  x: Leaf  # (T, K) bf16 decoder hidden states, before the input norm.
  residual: Leaf | None  # (T, K) bf16.

  @classmethod
  def block_specs(
      cls, cfg: config_lib.FusedGdnConfig
  ) -> "Inputs[pl.BlockSpec]":
    """The pipeline's per-tile blocks: whole rows of each input."""
    rows = _rows(cfg, cfg.hidden_size)
    return cls(x=rows, residual=rows if cfg.model.residual else None)


class Outputs(NamedTuple, Generic[Leaf]):
  """Outputs, in the layouts the GDN kernel (fused_conv1d_gdn) reads."""

  # (T, 1, qkv_cols) f32 [q | k | v] per token, compact (1, 128) tiling.
  mixed_qkv: Leaf
  z: Leaf  # (T, Hv, D) bf16, head-major.
  b: Leaf  # (T, 1, lanes) f32, lanes >= Hv zero.
  a: Leaf  # (T, 1, lanes) f32, lanes >= Hv zero.
  residual: Leaf | None  # (T, K) bf16 x + residual.

  @classmethod
  def shapes(
      cls, cfg: config_lib.FusedGdnConfig
  ) -> "Outputs[jax.ShapeDtypeStruct]":
    """The outputs' shapes and dtypes."""
    m, num_tokens = cfg.model, cfg.num_tokens
    num_lanes = pltpu.get_tpu_info().num_lanes
    return cls(
        mixed_qkv=jax.ShapeDtypeStruct(
            (num_tokens, 1, m.qkv_cols), jnp.float32
        ),
        z=jax.ShapeDtypeStruct(
            (num_tokens, m.num_v_heads, m.head_dim), jnp.bfloat16
        ),
        b=jax.ShapeDtypeStruct((num_tokens, 1, num_lanes), jnp.float32),
        a=jax.ShapeDtypeStruct((num_tokens, 1, num_lanes), jnp.float32),
        residual=(
            jax.ShapeDtypeStruct((num_tokens, cfg.hidden_size), jnp.bfloat16)
            if m.residual
            else None
        ),
    )

  @classmethod
  def block_specs(
      cls, cfg: config_lib.FusedGdnConfig
  ) -> "Outputs[pl.BlockSpec]":
    """The pipeline's per-tile blocks."""
    tile = cfg.tile.tile_size
    shapes = cls.shapes(cfg)

    def tokens(shape: jax.ShapeDtypeStruct) -> pl.BlockSpec:
      return pl.BlockSpec((tile, *shape.shape[1:]), lambda i: (i, 0, 0))

    return cls(
        mixed_qkv=tokens(shapes.mixed_qkv),
        z=tokens(shapes.z),
        b=tokens(shapes.b),
        a=tokens(shapes.a),
        residual=(
            _rows(cfg, cfg.hidden_size) if shapes.residual is not None else None
        ),
    )


class Scratch(NamedTuple, Generic[Leaf]):
  """VMEM scratch and semaphores."""

  x_q: Leaf  # (tile, K) fp8: the normed, quantized x.
  x_scale: Leaf  # (K / block, tile, lanes) f32, lane-broadcast.
  row_scale: Leaf  # (tile, lanes) f32 deferred 1/rms.
  normed: Leaf  # (tile, K) bf16 x * norm weight: in_proj_ba's input.
  w_vmem: Leaf  # (K, N) in_proj_qkvz's VMEM copy.
  w_sems: Leaf  # (len(cfg.w_chunks),) its DMA semaphores.
  ba_vmem: Leaf  # (K, lanes) in_proj_ba's VMEM copy.
  ba_sems: Leaf  # (1,) its DMA semaphore.

  @classmethod
  def shapes(
      cls, cfg: config_lib.FusedGdnConfig, weights: model_config.GdnWeights
  ) -> "Scratch[pl.MemoryRef]":
    """The VMEM scratch and semaphores, including both weights' VMEM copies."""
    m, tile = cfg.model, cfg.tile.tile_size
    num_lanes = pltpu.get_tpu_info().num_lanes
    w_vmem, w_sems = memory.stream_scratch(weights.w, cfg.w_chunks)
    ba_vmem, ba_sems = memory.stream_scratch(weights.ba_w, cfg.ba_chunks)
    return cls(
        x_q=pltpu.VMEM((tile, cfg.hidden_size), weights.w.dtype),
        x_scale=pltpu.VMEM(
            (cfg.hidden_size // m.act_quant_block, tile, num_lanes), jnp.float32
        ),
        row_scale=pltpu.VMEM((tile, num_lanes), jnp.float32),
        normed=pltpu.VMEM((tile, cfg.hidden_size), jnp.bfloat16),
        w_vmem=w_vmem,
        w_sems=w_sems,
        ba_vmem=ba_vmem,
        ba_sems=ba_sems,
    )
