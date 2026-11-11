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

"""Resolved static configuration of the fused expert-parallel MoE kernel.

Config is the single source of truth that kernel modules read: problem
dimensions, weight format, schedule depth, SMEM footprint validation, and
compile-time constants.
"""

import dataclasses
import logging

from tokamax._src.ops.experimental.dedup_moe.common import arg
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan as dedup_plan_mod
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import plan as plan_lib
from tokamax._src.ops.experimental.fused_moe import host as fused_host
from tokamax._src.ops.experimental.fused_moe import kernel as fused_kernel

logger = logging.getLogger(__name__)

# Kernel constants shared with `fused_moe.kernel`: the single-axis mesh name
# the layer's shard_map uses, the requantization chunk width, the DMA
# priority of the weight refills, and the collective id of the barrier and
# remote copies.
AXIS = fused_kernel.AXIS
QCHUNK = fused_kernel.QCHUNK
WEIGHT_DMA_PRIORITY = fused_kernel.WEIGHT_DMA_PRIORITY
COLLECTIVE_ID = fused_kernel.COLLECTIVE_ID
# Builds that ship results from the HBM contribution slab push each
# (expert, dest) run as soon as the tiles holding it are committed, while
# the expert's remaining tiles still compute. Off, an expert's results go
# out only after its last tile, so the push of a many-tile expert -- and
# of the last expert visited -- sits on the wire with nothing to hide
# behind, and every shard's drain waits for it. Read at build time; the
# VMEM-push builds (small buckets) already ship every tile as it finishes.
PROGRESSIVE_PUSH = True
# The scalar-prefetch and HBM operand counts, in the order the kernel body
# unpacks them (see `fused_moe.kernel`). The two totals and the visit count
# are rows of the count table, and the push source is the same table as the
# commit offset, so it rides once. The shard's index along the
# expert-parallel axis is an i32 operand rather than `lax.axis_index`, so the
# module stays partitionable after a serialize-and-recompile round trip.
# Every build of this kernel also takes the wave dispatch table as an HBM
# operand, right after the token buffer, and the start kernel's receive
# semaphores as a trailing semaphore operand.
N_TRANSPORT_TABLES = fused_kernel.N_TRANSPORT_TABLES
N_PUSH_TABLES = fused_kernel.N_PUSH_TABLES
N_SLAB_TABLES = fused_kernel.N_SLAB_TABLES
N_VISIT_TABLES = fused_kernel.N_VISIT_TABLES
N_COUNT_TABLES = fused_kernel.N_COUNT_TABLES
N_RANK_TABLES = fused_kernel.N_RANK_TABLES
N_PREFETCH = fused_kernel.N_PREFETCH
N_STAGING_INPUTS = fused_kernel.N_STAGING_INPUTS
N_WEIGHT_SLABS = fused_kernel.N_WEIGHT_SLABS
N_SCALE_TABLES = fused_kernel.N_SCALE_TABLES

# The FFN activations this kernel can fuse (shared with `fused_moe.kernel`).
ACT_FNS = fused_kernel.ACT_FNS

# XLA:TPU leaves a Mosaic callee `xla_tpu_impure_smem_callee_ratio` (0.6) of
# the core's scalar memory.
SMEM_CALLEE_BUDGET_BYTES = int(0.6 * 1024 * 1024)
_SMEM_FOOTPRINTS_LOGGED = set()


def smem_padded_bytes(shape):
  """Bytes a 32-bit SMEM array occupies under Mosaic's SMEM tiling.

  Mosaic (`infer_memref_layout`) tiles a rank-1 SMEM memref by 128 words and
  a rank >= 2 one by (8, 128) over its two minor dims, so e.g. a [R, 4]
  table occupies R * 128 words: 32x its data.

  Args:
    shape: Shape of the 32-bit SMEM array.

  Returns:
    Padded bytes.
  """
  shape = tuple(int(d) for d in shape)
  if len(shape) <= 1:
    n = shape[0] if shape else 1
    return 4 * (-(-n // 128) * 128)
  lead = 1
  for d in shape[:-2]:
    lead *= d
  return 4 * lead * (-(-shape[-2] // 8) * 8) * (-(-shape[-1] // 128) * 128)


def check_smem_footprint(shapes, *, what):
  """Log (warn above the callee budget) a kernel's padded SMEM bytes."""
  total = sum(smem_padded_bytes(s) for s in shapes)
  key = (what, total)
  if key not in _SMEM_FOOTPRINTS_LOGGED:
    _SMEM_FOOTPRINTS_LOGGED.add(key)
    log = logger.warning if total > SMEM_CALLEE_BUDGET_BYTES else logger.info
    log(
        "%s: padded SMEM footprint %.1f KiB (XLA callee budget %.0f KiB)",
        what,
        total / 1024,
        SMEM_CALLEE_BUDGET_BYTES / 1024,
    )
  return total


@dataclasses.dataclass(frozen=True)
class FusedKernelConfig:
  """Resolved static configuration of the fused expert-parallel MoE kernel."""

  g_local: int
  capacity: int
  hidden: int
  inter: int
  ep: int
  weight_format: arg.WeightFormat
  ragged_rows_alloc: int
  act_fn: str
  has_w1_bias: bool
  has_w2_bias: bool
  waves: int
  dedup_slots: int
  dedup_fwd_max_runs: int
  dedup_board_size: int
  wait_for_exchange: bool
  form: arg.WeightForm
  slots: plan_lib.WeightSlots
  prefetch_distance: int
  nbuf: int
  prefetches: bool
  push_from_vmem: bool
  progressive_push: bool
  skip_scale_mirror: bool
  has_scales: bool
  has_act_scale: bool
  has_ls_vm: bool
  lane_blocks: int
  tile_m: int
  tile_blocks: int
  tile_block_shift: int
  tile_shift: int
  height_rungs: tuple[int, ...]
  ls_window_rows: int
  gather_tiles: int
  gather_payload_rows: int
  gather_window_rows: int
  wave_bounds_base: int
  nb: int
  fwd_slots: int

  @property
  def dedup(self) -> bool:
    return self.dedup_slots > 0

  @property
  def fwd_max_runs(self) -> int:
    return self.dedup_fwd_max_runs

  @property
  def kernel_name(self) -> str:
    format_tag = {
        arg.WeightFormat.FP8: "",
        arg.WeightFormat.BF16: "_bf16w",
    }[self.weight_format]
    bias_tag = ("_w13bias" if self.has_w1_bias else "") + (
        "_w2bias" if self.has_w2_bias else ""
    )
    return (
        f"fused_ep_moe_v2_w{self.waves}"
        f"{'' if self.wait_for_exchange else '_landed'}"
        f"{f'_dd{self.dedup_slots}' if self.dedup else ''}"
        f"{f'_fr{self.fwd_max_runs}' if self.dedup and self.fwd_max_runs != dedup_plan_mod.MAX_FWD_RUNS else ''}"
        f"{f'_bs{self.dedup_board_size}' if self.dedup else ''}"
        f"{'_pp' if self.progressive_push else ''}"
        f"{format_tag}"
        f"{'' if self.act_fn == 'silu' else '_' + self.act_fn}"
        f"{bias_tag}"
        f"_g{self.g_local}_c{self.capacity}_nb{self.nbuf}"
    )

  @classmethod
  def create(
      cls,
      *,
      g_local: int,
      capacity: int,
      hidden: int,
      inter: int,
      ep: int,
      weight_format: arg.WeightFormat = arg.WeightFormat.FP8,
      ragged_rows_alloc: int | None = None,
      act_fn: str = "silu",
      has_w1_bias: bool = False,
      has_w2_bias: bool = False,
      waves: int,
      dedup_slots: int = 0,
      dedup_fwd_max_runs: int = dedup_plan_mod.MAX_FWD_RUNS,
      dedup_board_size: int = dedup_plan_mod.MAX_BOARD_SIZE,
      wait_for_exchange: bool = True,
  ) -> "FusedKernelConfig":
    """Validates build parameters, resolves the weight slots and builds the config.

    Args:
      g_local: Experts per shard.
      capacity: Tile height in rows.
      hidden: Model width.
      inter: Expert FFN width.
      ep: Expert-parallel width.
      weight_format: fp8 or bf16.
      ragged_rows_alloc: Slab rows per shard; a multiple of ROWBLK and of
        `capacity`.
      act_fn: One of ACT_FNS.
      has_w1_bias: Whether the gate/up bias is supplied.
      has_w2_bias: Whether the down bias is supplied.
      waves: W; must divide g_local.
      dedup_slots: Stage slots per gateway; 0 disables De-Dup forwarding.
      dedup_fwd_max_runs: Gateway forward runs per (served source, wave).
      dedup_board_size: Devices per board of the De-Dup topology; ep must be a
        multiple of it.
      wait_for_exchange: False for builds on an already complete token buffer
        (CPU interpret tests).

    Returns:
      The FusedKernelConfig.

    Raises:
      NotImplementedError: If `act_fn` is not supported.
      ValueError: If a parameter is out of range or the buffers do not fit in
        VMEM.
    """
    if act_fn not in ACT_FNS:
      raise NotImplementedError(
          f"the fused EP MoE kernel fuses {ACT_FNS}; got {act_fn!r}"
      )
    dedup = dedup_slots > 0
    if dedup and not wait_for_exchange:
      raise ValueError(
          "wait_for_exchange=False builds take no De-Dup forwarding"
      )
    if dedup and (
        not 2 <= dedup_board_size <= dedup_plan_mod.MAX_BOARD_SIZE
        or ep % dedup_board_size
    ):
      raise ValueError(
          f"dedup_slots={dedup_slots} requires ep divisible by a board size in"
          f" [2, {dedup_plan_mod.MAX_BOARD_SIZE}]; got ep={ep},"
          f" dedup_board_size={dedup_board_size}"
      )
    if dedup_fwd_max_runs < 1:
      raise ValueError(
          f"dedup_fwd_max_runs must be >= 1; got {dedup_fwd_max_runs}"
      )
    fwd_slots = dedup_plan_mod.FWD_SLOTS
    nb = ep // dedup_board_size if dedup else 0
    if waves < 1 or g_local % waves:
      raise ValueError(
          f"waves must be a positive divisor of g_local={g_local}; got {waves}"
      )
    if ep < 3:
      raise ValueError(
          "the wave bounds ride a dispatch-table padding row: ep >= 3; got"
          f" {ep}"
      )
    wave_bounds_base = (3 * ep + 2) * waves
    form = arg.weight_form(weight_format)
    weight_format = form.name
    skip_scale_mirror = form.wire_dtype is not arg.FP8

    if capacity % fused_host.ROWBLK:
      raise ValueError(
          f"tile height {capacity} is not a whole number of"
          f" {fused_host.ROWBLK}-row blocks, which is the unit every transport"
          " here moves"
      )
    if (
        hidden < fused_host.HIDDEN_LANE_BLOCK
        or hidden % fused_host.HIDDEN_LANE_BLOCK
        or hidden > fused_host.HIDDEN_MAX_BLOCKS * fused_host.HIDDEN_LANE_BLOCK
    ):
      raise ValueError(
          f"hidden {hidden} is not between one and"
          f" {fused_host.HIDDEN_MAX_BLOCKS} whole"
          f" {fused_host.HIDDEN_LANE_BLOCK}-lane blocks, which the per-row"
          " transport geometry requires"
      )
    if inter < fused_host.HIDDEN_LANE_BLOCK:
      raise ValueError(
          f"inter {inter} is narrower than one"
          f" {fused_host.HIDDEN_LANE_BLOCK}-lane block; an expert whose FFN has"
          " no intermediate channels to speak of is not a layer this kernel"
          " can serve"
      )
    lane_blocks = fused_host.row_lane_blocks(hidden)
    has_scales = form.has_scales
    has_act_scale = form.quantized_activations
    has_ls_vm = has_act_scale
    slots = plan_lib.fit_weight_slots(
        g_local,
        capacity,
        hidden,
        inter,
        weight_format=weight_format,
        has_w1_bias=has_w1_bias,
        has_w2_bias=has_w2_bias,
    )
    prefetch_distance = slots.distance
    nbuf = slots.nbuf
    assert 0 <= prefetch_distance < nbuf, (
        f"weight prefetch distance {prefetch_distance} must satisfy "
        f"0 <= distance < nbuf={nbuf}, so that distance + 1 consecutive "
        "experts occupy distinct weight slots"
    )
    prefetches = prefetch_distance > 0
    if not slots.fits:
      raise ValueError(
          "the kernel's VMEM buffers need "
          f"{slots.vmem_bytes / 2**20:.1f}MiB, over the "
          f"{slots.limit / 2**20:.1f}MiB budget, for {g_local} local experts "
          f"of {hidden}x{inter} at capacity={capacity} with the weight "
          f"prefetch already given up (nbuf={nbuf}, one expert of weight "
          "slabs). Nothing further is bought by shortening the runway; "
          "the remaining levers are the tile height, the expert shape and "
          "the number of local experts"
      )
    if prefetch_distance < plan_lib.MAX_WEIGHT_PREFETCH_DISTANCE:
      logger.info(
          "fused EP MoE: %d local experts of %dx%d at capacity=%d fit at "
          "prefetch distance %d (%d weight slots, %.1fMiB of a %.1fMiB "
          "budget), below the ceiling of %d",
          g_local,
          hidden,
          inter,
          capacity,
          prefetch_distance,
          nbuf,
          slots.vmem_bytes / 2**20,
          slots.limit / 2**20,
          plan_lib.MAX_WEIGHT_PREFETCH_DISTANCE,
      )
    if (
        ragged_rows_alloc is None
        or ragged_rows_alloc % fused_host.ROWBLK
        or ragged_rows_alloc % capacity
    ):
      raise ValueError(
          f"ragged_rows_alloc {ragged_rows_alloc} must be a whole number of"
          f" both {fused_host.ROWBLK}-row blocks and {capacity}-row tiles,"
          " because a tail tile reads a full window past the shard's last row"
      )
    tile_m = capacity
    tile_blocks = tile_m // fused_host.ROWBLK
    tile_block_shift = fused_host.pow2_shift(tile_blocks, "tile_m // ROWBLK")
    tile_shift = fused_host.pow2_shift(tile_m, "tile_m")
    height_rungs = plan_lib.tail_height_rungs(tile_m)
    push_from_vmem = ragged_rows_alloc <= ep * g_local * tile_m
    progressive_push = PROGRESSIVE_PUSH and not push_from_vmem
    ls_window_rows = fused_host.act_scale_window_rows(tile_m)
    gather_tiles = fused_host.TOKEN_GATHER_TILES_PER_WINDOW
    gather_payload_rows = gather_tiles * tile_m
    gather_window_rows = fused_host.token_gather_window_rows(tile_m)
    return cls(
        g_local=g_local,
        capacity=capacity,
        hidden=hidden,
        inter=inter,
        ep=ep,
        weight_format=weight_format,
        ragged_rows_alloc=ragged_rows_alloc,
        act_fn=act_fn,
        has_w1_bias=has_w1_bias,
        has_w2_bias=has_w2_bias,
        waves=waves,
        dedup_slots=dedup_slots,
        dedup_fwd_max_runs=dedup_fwd_max_runs,
        dedup_board_size=dedup_board_size if dedup else 0,
        wait_for_exchange=wait_for_exchange,
        form=form,
        slots=slots,
        prefetch_distance=prefetch_distance,
        nbuf=nbuf,
        prefetches=prefetches,
        push_from_vmem=push_from_vmem,
        progressive_push=progressive_push,
        skip_scale_mirror=skip_scale_mirror,
        has_scales=has_scales,
        has_act_scale=has_act_scale,
        has_ls_vm=has_ls_vm,
        lane_blocks=lane_blocks,
        tile_m=tile_m,
        tile_blocks=tile_blocks,
        tile_block_shift=tile_block_shift,
        tile_shift=tile_shift,
        height_rungs=height_rungs,
        ls_window_rows=ls_window_rows,
        gather_tiles=gather_tiles,
        gather_payload_rows=gather_payload_rows,
        gather_window_rows=gather_window_rows,
        wave_bounds_base=wave_bounds_base,
        nb=nb,
        fwd_slots=fwd_slots,
    )
