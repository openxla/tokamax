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

  Tile-size selection targets the register budget of SM80+ (65536 32-bit
  registers per SM). With num_warps warps the per-thread accumulator cost
  is b_block * v_block / (32 * num_warps) float32 registers.

  When V is divisible by 256 we use larger tiles (b=64, v=256, warps=8):
    accumulator: 64 * 256 / 256 = 64 regs — well within budget.
  Otherwise we fall back to conservative tiles (b=32, v=128, warps=4):
    accumulator: 32 * 128 / 128 = 32 regs.

  h_block scales with H (64 or 128) to improve tensor-core utilisation.
  """
  _, h_dim = x.shape
  v_dim = w.shape[1]
  h_block_size = 128 if h_dim % 128 == 0 else 64
  if v_dim % 256 == 0:
    # h_block capped at 64: with v_block=256 and warps=8 (256 threads), the
    # w tile alone is h_block*256/256 regs; h=128 pushes total over ~200 regs.
    return Config(
        b_block_size=64,
        h_block_size=64,
        v_block_size=256,
        num_warps=8,
    )
  return Config(
      b_block_size=32,
      h_block_size=h_block_size,
      v_block_size=128,
      num_warps=4,
  )
