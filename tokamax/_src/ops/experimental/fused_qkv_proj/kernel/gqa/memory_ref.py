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
"""fused_qkv's inputs, outputs and scratch: their shapes, blocks and DMAs.

No numerics here.
"""

import dataclasses
from typing import Generic, NamedTuple, TypeVar

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import layout_util
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import memory
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.gqa import config as config_lib

# Arrays, ShapeDtypeStructs, BlockSpecs or Refs, depending on the use.
Leaf = TypeVar("Leaf")


def _rows(cfg: config_lib.FusedQkvConfig, width: int) -> pl.BlockSpec:
  return pl.BlockSpec((cfg.tile.tile_size, width), lambda i: (i, 0))


def _q_shape(cfg: config_lib.FusedQkvConfig) -> tuple[int, ...]:
  m = cfg.model
  return (
      m.num_kv_heads,
      cfg.num_tokens,
      cfg.model.q_per_kv // m.q_packing,
      m.q_packing,
      layout_util.align(m.head_dim, pltpu.get_tpu_info().num_lanes),
  )


def _kv_shape(cfg: config_lib.FusedQkvConfig) -> tuple[int, ...]:
  m = cfg.model
  if cfg.seq_along_lane:
    return (
        2 * m.num_kv_heads,
        m.head_dim // m.kv_packing,
        m.kv_packing,
        cfg.kv_tokens,
    )
  rows = 2 * m.num_kv_heads * cfg.kv_folds
  return (
      cfg.num_tokens,
      layout_util.align(rows, m.kv_packing) // m.kv_packing,
      m.kv_packing,
      layout_util.align(
          m.head_dim // cfg.kv_folds, pltpu.get_tpu_info().num_lanes
      ),
  )


class Inputs(NamedTuple, Generic[Leaf]):
  """Inputs split into token tiles by the pipeline."""

  x: Leaf  # (T, K) bf16 decoder hidden states, before the input norm.
  residual: Leaf | None  # (T, K) bf16.
  cos: Leaf | None  # (T, n_pairs) bf16 RoPE.
  sin: Leaf | None  # (T, n_pairs) bf16 RoPE.

  @classmethod
  def block_specs(
      cls, cfg: config_lib.FusedQkvConfig
  ) -> "Inputs[pl.BlockSpec]":
    """The pipeline's per-tile blocks: whole rows of each input."""
    m = cfg.model
    rope = _rows(cfg, m.rope.n_pairs) if m.rope is not None else None
    return cls(
        x=_rows(cfg, cfg.hidden_size),
        residual=_rows(cfg, cfg.hidden_size) if m.residual else None,
        cos=rope,
        sin=rope,
    )


class Outputs(NamedTuple, Generic[Leaf]):
  """Outputs, written in batched_rpa's input layouts."""

  # (Hkv, T, q_per_kv / q_packing, q_packing, align(D, 128)) bf16.
  q: Leaf
  # fp8, heads ordered [k_0, v_0, k_1, v_1, ...]:
  #   SEQ_ALONG_LANE: (2 Hkv, D / kv_packing, kv_packing, cfg.kv_tokens),
  #     tokens past T zero.
  #   HEAD_ALONG_SUBLANE: (T, align(2 Hkv F, kv_packing) / kv_packing,
  #     kv_packing, align(D / F, 128)), F = cfg.kv_folds. Row f * 2 Hkv + h
  #     is lanes [f D / F, (f + 1) D / F) of head h; rows zero-padded to whole
  #     groups.
  kv: Leaf
  gate: Leaf | None  # (T, Hq * D) bf16.
  residual: Leaf | None  # (T, K) bf16 x + residual.

  @classmethod
  def shapes(
      cls, cfg: config_lib.FusedQkvConfig
  ) -> "Outputs[jax.ShapeDtypeStruct]":
    """The outputs' shapes and dtypes."""
    m, num_tokens = cfg.model, cfg.num_tokens
    bf16 = jnp.bfloat16
    return cls(
        q=jax.ShapeDtypeStruct(_q_shape(cfg), m.q_dtype),
        kv=jax.ShapeDtypeStruct(_kv_shape(cfg), m.kv_dtype),
        gate=(
            jax.ShapeDtypeStruct((num_tokens, m.q_dim), bf16)
            if m.gate
            else None
        ),
        residual=(
            jax.ShapeDtypeStruct((num_tokens, cfg.hidden_size), bf16)
            if m.residual
            else None
        ),
    )

  @classmethod
  def block_specs(
      cls, cfg: config_lib.FusedQkvConfig
  ) -> "Outputs[pl.BlockSpec]":
    """The pipeline's per-tile blocks."""
    m, tile = cfg.model, cfg.tile.tile_size
    q_shape, kv_shape = _q_shape(cfg), _kv_shape(cfg)
    if cfg.seq_along_lane:
      kv_spec = pl.BlockSpec(
          (*kv_shape[:3], cfg.kv_block_lanes), lambda i: (0, 0, 0, i)
      )
    else:
      kv_spec = pl.BlockSpec((tile, *kv_shape[1:]), lambda i: (i, 0, 0, 0))
    return cls(
        q=pl.BlockSpec(
            (m.num_kv_heads, tile, *q_shape[2:]), lambda i: (0, i, 0, 0, 0)
        ),
        kv=kv_spec,
        gate=_rows(cfg, m.q_dim) if m.gate else None,
        residual=_rows(cfg, cfg.hidden_size) if m.residual else None,
    )


class Scratch(NamedTuple, Generic[Leaf]):
  """VMEM scratch and semaphores. Fields a config doesn't use are None."""

  x_q: Leaf  # (tile, K) W's dtype: the normed, quantized x.
  qkv: Leaf  # (tile, N) bf16 projection output.
  x_scale: Leaf | None  # fp8: (K / block, tile, lanes) f32, lane-broadcast.
  row_scale: Leaf | None  # fp8: (tile, lanes) f32 deferred 1/rms.
  # SEQ_ALONG_LANE with a token pad past the last block: one lane chunk of
  # zero KV, and the semaphore of its DMAs.
  kv_pad_zeros: Leaf | None
  kv_pad_sem: Leaf | None
  w_vmem: Leaf  # (K, N) W's VMEM copy.
  w_sems: Leaf  # (len(cfg.w_chunks),) W's DMA semaphores.

  @classmethod
  def shapes(
      cls, cfg: config_lib.FusedQkvConfig, w: jax.Array
  ) -> "Scratch[pl.MemoryRef | pltpu.SemaphoreType]":
    """The VMEM scratch and semaphores, including W's VMEM copy."""
    m, tile = cfg.model, cfg.tile.tile_size
    num_lanes = pltpu.get_tpu_info().num_lanes
    fp8 = m.fp8
    pad = bool(cfg.kv_pad_starts)
    w_vmem, w_sems = memory.stream_scratch(w, cfg.w_chunks)
    x_scale = kv_pad_zeros = None
    if fp8:
      x_scale = pltpu.VMEM(
          (cfg.hidden_size // m.act_quant_block, tile, num_lanes), jnp.float32
      )
    if pad:
      kv_pad_zeros = pltpu.VMEM((*_kv_shape(cfg)[:3], num_lanes), m.kv_dtype)
    return cls(
        x_q=pltpu.VMEM((tile, cfg.hidden_size), w.dtype),
        qkv=pltpu.VMEM((tile, m.num_cols), jnp.bfloat16),
        x_scale=x_scale,
        row_scale=pltpu.VMEM((tile, num_lanes), jnp.float32) if fp8 else None,
        kv_pad_zeros=kv_pad_zeros,
        kv_pad_sem=pltpu.SemaphoreType.DMA if pad else None,
        w_vmem=w_vmem,
        w_sems=w_sems,
    )


@dataclasses.dataclass
class KvPadZeroing:
  """Zero DMAs over the SEQ_ALONG_LANE KV token pad past the last block."""

  cfg: config_lib.FusedQkvConfig
  kv_hbm: jax.Ref  # Outputs.kv in HBM.
  zeros: jax.Ref  # Scratch.kv_pad_zeros.
  sem: jax.Ref  # Scratch.kv_pad_sem.
  _copies: list[memory.AsyncCopy] = dataclasses.field(default_factory=list)

  def start(self) -> None:
    """Zeroes the slab and starts its copy into each pad lane chunk."""
    m = self.cfg.model
    num_lanes = pltpu.get_tpu_info().num_lanes
    for head in range(self.zeros.shape[0]):
      slab = self.zeros.at[head].bitcast(jnp.uint32)
      slab = slab.reshape(m.head_dim // m.kv_packing, num_lanes)
      slab[...] = jnp.zeros(slab.shape, jnp.uint32)
    self._copies = [
        pltpu.make_async_copy(
            self.zeros,
            self.kv_hbm.at[:, :, :, pl.ds(start, num_lanes)],
            self.sem,
        )
        for start in self.cfg.kv_pad_starts
    ]
    for copy in self._copies:
      copy.start()

  def wait(self) -> None:
    """Waits for the pad copies."""
    for copy in self._copies:
      copy.wait()
