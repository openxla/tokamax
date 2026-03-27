# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Pallas-Triton linear softmax cross-entropy loss configuration."""

from typing import Annotated

import jax
import pydantic
from tokamax._src import pydantic as pydantic_lib


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Tile-size configuration for the Pallas/Triton GPU kernel.

  All block sizes must evenly divide the corresponding tensor dimension.

  Attributes:
    b_block_size: Tile size over the batch/token (B) dimension.
    h_block_size: Tile size for the inner hidden (H) matmul loop. Each
      iteration loads a (b_block_size, h_block_size) slice of x and a
      (h_block_size, v_block_size) slice of w; total HBM data volume is the
      same regardless of this value. It controls register pressure and the
      matmul tile shape presented to tensor cores.
    v_block_size: Tile size over the vocabulary (V) dimension.
    num_warps: Number of Triton warps per program.
  """

  b_block_size: Annotated[int, pydantic.Field(ge=16, multiple_of=16)] = 32
  h_block_size: Annotated[int, pydantic.Field(ge=16, multiple_of=16)] = 64
  v_block_size: Annotated[int, pydantic.Field(ge=16, multiple_of=16)] = 128
  num_warps: pydantic_lib.PowerOfTwo = 4


def get_heuristics_config(
    x: jax.Array,
    w: jax.Array,
) -> Config:
  """Returns a register-safe config based on the input shapes.

  ## v_block_size (fixed at 128)

  v_block_size=256 crashes the Triton-to-PTX compilation stage in JAX 0.9.2's
  bundled Triton: the power-of-2 check in pallas/triton/lowering.py passes
  (total tensor size 8192 is a power of 2) but the Triton compiler then crashes
  with a C++ exception. The check comment explicitly warns: "the Triton lowering
  will fail anyway but it will crash with a C++ exception". The nearest upstream
  fix is jax-ml/jax#35654, which guards the same crash for fp64; the fp32/n=256
  case is not yet guarded. Revisit when JAX upgrades its bundled Triton.

  ## Register budget (SM80+, 65536 regs per SM, num_warps=4, 128 threads)

  With v_block=128, per-thread register cost:
    accumulator:  b_block * v_block / 128 = b_block regs/thread.
    w tile:       h_block * v_block / 128 = h_block regs/thread.
    x tile:       b_block * h_block / 128 regs/thread.
    total:        b_block + h_block + b_block * h_block / 128.

  The 50%-budget constraint (256 regs/thread, allows 2 CTAs/SM) limits
  combined (b_block, h_block) choices:
    b=128, h=64:  128 + 64  + 64  = 256 regs  (50%)  ← 2 CTAs/SM OK
    b=64,  h=128: 64  + 128 + 64  = 256 regs  (50%)  ← 2 CTAs/SM OK
    b=64,  h=64:  64  + 64  + 32  = 160 regs  (31%)  ← safe
    b=32,  h=128: 32  + 128 + 32  = 192 regs  (37%)  ← safe
    b=128, h=128: 128 + 128 + 128 = 384 regs  (75%)  ← 1 CTA/SM, avoided

  ## HBM traffic analysis

  HBM reads scale as (all shapes in elements):
    x traffic: B * H * (V / v_block)  —  x is re-read once per v_block tile.
    w traffic: H * V * (B / b_block)  —  w is re-read once per b_block tile.

  At v_block=128: x traffic = B*H*V/128, w traffic = B*H*V/b_block.
  Traffic is balanced when b_block = v_block = 128. At b_block=64, w traffic
  is 2× x traffic; at b_block=32, 4×. So b_block=128 (when B divisible by 128)
  minimises total HBM reads and measurably outperforms b_block=64 (~4% on
  LLM-scale shapes, bandwidth-bound regime).

  When b_block=128, h_block is capped at 64 to stay within the 50% budget.
  When b_block<=64, h_block=128 (if H divisible by 128) for better tensor-core
  tile efficiency; h_block does not affect HBM traffic.
  """
  b_dim, h_dim = x.shape
  if b_dim % 128 == 0:
    b_block_size = 128
    h_block_size = 64  # b=128,h=128 → 75% regs → 1 CTA/SM; cap at 64.
  elif b_dim % 64 == 0:
    b_block_size = 64
    h_block_size = 128 if h_dim % 128 == 0 else 64
  else:
    b_block_size = 32
    h_block_size = 128 if h_dim % 128 == 0 else 64
  return Config(
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=128,
      num_warps=4,
  )
