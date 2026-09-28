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
"""Token tile sizes and the VMEM budget they must fit, shared by the kernels.

Each kernel step works on one token tile. TileConfig.create picks the largest
tile that fits VMEM and still leaves TilePolicy.min_tiles tiles: the first
tile's loads and the last tile's writeback can't overlap compute, and with 4
or more tiles they stay a small share of the run.

The rule is simple on purpose. Kernel us on TPU7x per tile size
(benchmark.py test_tile_sweep, 2026-10-06; -: OOM), and its gap to the best:

  model                  T     128     256     512  best  picked
  qwen3_coder_tp8     1024    28.0    26.2       -   256     256
  qwen3_coder_tp8    16384   333.3   277.7       -   256     256
  qwen3_5_tp8         1024    24.7    23.9    24.9   256     256
  qwen3_5_tp8        16384   289.5   249.5   238.1   512     512
  qwen3_5_tp2         1024    62.7    63.6       -   128     128
  qwen3_5_tp2        16384   764.5   759.2       -   256     128  +0.7%
  gemma4_sliding_tp4  1024    40.3    40.3    40.9   256     256
  gemma4_sliding_tp4 16384   488.6   470.8   463.3   512     512
  gemma4_full_tp4     1024    48.0    48.6    49.5   128     256  +1.4%
  gemma4_full_tp4    16384   594.9   592.6   580.2   512     512
  gpt_oss_tp4         1024    21.9    20.7    19.7   512     256  +5.3%
  gpt_oss_tp4        16384   250.8   217.6   195.6   512     512
  qwen3_5_gdn_tp8     1024    27.2    25.9    27.8   256     256
  qwen3_5_gdn_tp8    16384   318.2   274.2   266.2   512     256  +3.0%

GDN's 512 compiles, but its VMEM estimate overshoots, so it isn't a candidate.
"""

from collections.abc import Callable
import dataclasses
import math
from typing import NamedTuple

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import layout_util
from tokamax._src.ops.experimental.fused_qkv_proj.kernel.common import model_config

MIB = 2**20


@dataclasses.dataclass(frozen=True)
class TilePolicy:
  """The tiling heuristic's knobs, with their measured defaults."""

  # Fewest token tiles worth pipelining (see the module docstring).
  min_tiles: int = 4
  # VMEM outside the scoped limit: the pinned weights sit there too, and so do
  # XLA's register spill slots, which no config describes (up to ~9 MiB at
  # tile 512).
  vmem_margin_bytes: int = 9 * MIB
  # Slack for the per-row VMEM estimate, which is within -2%..+3% of what
  # Mosaic needs.
  vmem_estimate_margin_bytes: int = MIB // 2
  # Column blocks per DMA chunk of a streamed weight. One block per chunk is
  # slower (each wait is a scheduling barrier); one chunk for the whole weight
  # overlaps nothing.
  stream_blocks_per_chunk: int = 2


@dataclasses.dataclass(frozen=True)
class TileConfig:
  """Every tile size a fused kernel uses."""

  num_tokens: int
  tile_size: int  # Tokens per pipeline step.
  matmul_rows: int  # Rows per projection group.
  prep_rows: int  # Rows per input-prep chunk: one fp8 sublane tile.
  norm_cols: int  # bf16 input norm: columns per block (16 f32 vregs).
  vmem_limit_bytes: int  # Scoped VMEM for the tiles and scratch.
  policy: TilePolicy

  @property
  def num_tiles(self) -> int:
    return self.num_tokens // self.tile_size

  def row_groups(self) -> range:
    """First row of each matmul_rows group of a tile."""
    return range(0, self.tile_size, self.matmul_rows)

  def prep_chunks(self, first_row: int) -> list[pl.Slice]:
    """The prep_rows chunks of the group starting at first_row."""
    return [
        pl.ds(start, self.prep_rows)
        for start in range(
            first_row, first_row + self.matmul_rows, self.prep_rows
        )
    ]

  def __post_init__(self) -> None:
    """Asserts the tile sizes nest and fit Mosaic's block rules."""
    assert self.num_tokens % self.tile_size == 0
    assert self.tile_size % self.matmul_rows == 0
    assert self.matmul_rows % self.prep_rows == 0
    # A lone tile's blocks are whole arrays, which Mosaic allows at any size.
    num_lanes = pltpu.get_tpu_info().num_lanes
    assert self.num_tiles == 1 or self.tile_size % num_lanes == 0

  def metadata(self) -> dict[str, str]:
    """The tile sizes, for the kernel's HLO metadata."""
    return {
        "num_tokens": str(self.num_tokens),
        "tile_size": str(self.tile_size),
        "num_tiles": str(self.num_tiles),
        "matmul_rows": str(self.matmul_rows),
        "prep_rows": str(self.prep_rows),
    }

  @classmethod
  def create(
      cls,
      *,
      num_tokens: int,
      fp8: bool,
      resident_bytes: int,
      bytes_per_row: int,
      tile_size: int | None = None,
      policy: TilePolicy = TilePolicy(),
  ) -> "TileConfig":
    """Picks the token tile (module docstring) and the sizes within a tile.

    Args:
      num_tokens: T.
      fp8: whether the projection is fp8.
      resident_bytes: VMEM held outside the scoped limit for the whole call (the
        weights).
      bytes_per_row: the kernel's VMEM bytes per token row (total_bytes of its
        RowBytes).
      tile_size: overrides the token tile.
      policy: the heuristic's knobs.

    Returns:
      The picked tile and the sizes within it.
    """
    tpu_info = pltpu.get_tpu_info()
    vmem_limit = (
        tpu_info.vmem_capacity_bytes - resident_bytes - policy.vmem_margin_bytes
    )
    if tile_size is None:

      def fits(tile: int) -> bool:
        need = bytes_per_row * tile
        return need + policy.vmem_estimate_margin_bytes <= vmem_limit

      tile_size = _pick_tile_size(
          num_tokens, tpu_info.num_lanes, fits, policy.min_tiles
      )
    prep_rows = tpu_info.get_sublane_tiling(jnp.float8_e4m3fn)
    return cls(
        num_tokens=num_tokens,
        tile_size=tile_size,
        # Measured: fewer rows push W to the MXU more often (fp8 at 64 rows is
        # 50% slower than 128; bf16 needs 256).
        matmul_rows=math.gcd(128 if fp8 else 256, tile_size),
        prep_rows=prep_rows,
        norm_cols=16 * tpu_info.num_sublanes * tpu_info.num_lanes // prep_rows,
        vmem_limit_bytes=vmem_limit,
        policy=policy,
    )


def _pick_tile_size(
    num_tokens: int,
    num_lanes: int,
    fits: Callable[[int], bool],
    min_tiles: int,
) -> int:
  """Largest fitting tile with >= min_tiles tiles, else the smallest fitting.

  Tiles of a multi-tile grid must be lane-aligned (SEQ_ALONG_LANE KV blocks
  put tokens on lanes); a single tile can be any size.

  Args:
    num_tokens: T.
    num_lanes: lanes per vreg.
    fits: whether a tile size fits in VMEM.
    min_tiles: the fewest tiles to prefer.

  Returns:
    The tile size.
  """
  candidates = [
      tile
      for tile in range(num_lanes, num_tokens, num_lanes)
      if num_tokens % tile == 0
  ]
  candidates.append(num_tokens)
  fitting = [tile for tile in candidates if fits(tile)]
  # TODO: fails when the weights alone fill VMEM; see
  # memory.stream_scratch.
  assert fitting, "no tile fits in VMEM"
  enough_tiles = [tile for tile in fitting if num_tokens // tile >= min_tiles]
  return max(enough_tiles) if enough_tiles else min(fitting)


def total_bytes(row_bytes: NamedTuple) -> int:
  """The sum of a (possibly nested) RowBytes tuple's terms."""
  return sum(jax.tree.leaves(row_bytes))


class InputRowBytes(NamedTuple):
  """The input stage's VMEM bytes per token row.

  Pipelined blocks are double-buffered; scratch is not. Every block's last dim
  is padded to the lane count; second-minor padding is ignored.
  """

  x: int  # Pipelined input block.
  residual: int  # Pipelined input block.
  residual_out: int  # Pipelined output block.
  x_q: int  # Scratch.
  x_scale: int = 0  # Scratch, fp8 only.
  row_scale: int = 0  # Scratch, fp8 only.


def input_bytes_per_row(
    config: model_config.AttentionConfig | model_config.GdnConfig,
    hidden_size: int,
) -> InputRowBytes:
  """The input stage's VMEM bytes per token row."""
  num_lanes = pltpu.get_tpu_info().num_lanes
  padded_k = layout_util.align(hidden_size, num_lanes)
  bf16 = 2
  residual = padded_k * bf16 if config.residual else 0
  fp8_scales = {}
  if config.fp8:
    fp8_scales = dict(
        x_scale=hidden_size // config.act_quant_block * num_lanes * 4,
        row_scale=num_lanes * 4,
    )
  return InputRowBytes(
      x=2 * padded_k * bf16,
      residual=2 * residual,
      residual_out=2 * residual,
      x_q=padded_k * (1 if config.fp8 else bf16),
      **fp8_scales,
  )
