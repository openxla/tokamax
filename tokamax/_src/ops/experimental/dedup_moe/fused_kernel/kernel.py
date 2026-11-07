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

"""Host-side preprocessing and invocation of the fused expert-parallel kernel.

Consolidates host-side operand validation, SMEM footprint checks,
`pl.pallas_call`
construction, and build-cache memoization outside the Pallas pipeline body.
"""

import logging
import threading
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.dedup_moe.common import arg
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan as dedup_plan_mod
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import config
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import memory_ref
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import pipeline
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import plan as plan_lib
from tokamax._src.ops.experimental.fused_moe import host as fused_host

logger = logging.getLogger(__name__)


def _build_fused_ep_moe_kernel(
    *,
    g_local,
    capacity,
    hidden,
    inter,
    ep,
    weight_format=arg.WeightFormat.FP8,
    ragged_rows_alloc=None,
    act_fn="silu",
    has_w1_bias=False,
    has_w2_bias=False,
    waves,
    dedup_slots=0,
    dedup_fwd_max_runs=dedup_plan_mod.MAX_FWD_RUNS,
    dedup_board_size=dedup_plan_mod.MAX_BOARD_SIZE,
    wait_for_exchange=True,
):
  """Build the pallas_call and the function that invokes it.

  The rows arrive by the wave dispatch (dedup_ep_moe, wave_dispatch.py):
  a pairwise exchange that dispatch_kernels.ragged_a2a_start_waves starts
  BEFORE this kernel, split into `waves` waves of experts laid out in visit
  order. The token buffer operand is the start kernel's receive buffer, and
  before an expert's first tile the kernel waits only for the waves up to
  that expert's, so the first experts compute while later waves are in
  flight. The waves have VARIABLE sizes (wave_dispatch.wave_plan): the
  visit positions each wave ends at come in dispatch-table row
  wave_dispatch.wave_bounds_row(ep) = 3 * ep + 2 (zero padding otherwise),
  and the visit at position p waits through wave
  #{w < W - 1 : bounds[w] <= p}.

  dedup_slots > 0 (De-Dup dispatch) adds gateway forwarding of staged rows
  over the intra-board links: before waiting on `wave_sems[w]`, the gateway
  waits on
  `stage_sems[0..dedup_slots, w]` and issues forwards from `stage_hbm`
  into `lhs_hbm`. The per-target-wave forward schedule
  (`dedup_plan.FwdSchedule`) is carried by the tables: each run signals its
  target's wave issue wave + (dst field >> FWD_DST_SHIFT), and the runs of
  (sigma, issue wave w) come in ready slots -- slot 0 (rows from earlier
  stage waves) first, then slot 1 after the stage run of wave w.
  dedup_fwd_max_runs is the table depth per (sigma, issue wave), and
  dedup_board_size the devices per board of the De-Dup topology.

  wait_for_exchange=False (CPU interpret tests only) builds the same kernel
  for a token buffer that is already complete: no dispatch table or wave
  semaphores, no wave waits and no De-Dup forwarding (dedup_slots must be
  0). Pallas interpret mode cannot run the split-phase exchange kernels, so
  interpret_util.emulate_wave_exchange lands the rows (and the forwards)
  itself and calls this build. Same tiles, matmuls and results.

  Args:
    g_local: Experts per shard.
    capacity: Tile height in rows.
    hidden: Model width.
    inter: Expert FFN width.
    ep: Expert-parallel width.
    weight_format: fp8 or bf16.
    ragged_rows_alloc: Slab rows per shard.
    act_fn: Fused FFN activation.
    has_w1_bias: Whether the gate/up bias is supplied.
    has_w2_bias: Whether the down bias is supplied.
    waves: W.
    dedup_slots: Stage slots per gateway; 0 disables De-Dup forwarding.
    dedup_fwd_max_runs: Gateway forward runs per (served source, wave).
    dedup_board_size: Devices per board of the De-Dup topology.
    wait_for_exchange: False for builds on a complete token buffer.

  Returns:
    `fn(...)`, the host wrapper that validates the operands and calls the
    kernel; it returns (arrivals, arrival scales).
  """
  # Step 1: Validate build parameters, resolve weight slots, build config.
  cfg = config.FusedKernelConfig.create(
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
      dedup_board_size=dedup_board_size,
      wait_for_exchange=wait_for_exchange,
  )

  # Step 2: Construct the Pallas kernel body and scratch shapes.
  kernel = pipeline.make_fused_ep_moe_kernel_body(cfg)
  scratch = memory_ref.build_scratch_shapes(cfg)

  hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
  n_inputs = (
      config.N_STAGING_INPUTS
      + config.N_WEIGHT_SLABS
      + int(cfg.has_ls_vm)
      + config.N_SCALE_TABLES * int(cfg.has_scales)
      + int(cfg.has_w1_bias)
      + int(cfg.has_w2_bias)
      + int(cfg.wait_for_exchange)  # the dispatch table
      + int(cfg.dedup)  # stage_hbm
  )
  n_prefetch = config.N_PREFETCH + 3 * int(cfg.dedup)

  def make_call(recv_rows):
    """The pallas_call for this build, at this arrival-buffer height."""
    if recv_rows % fused_host.ROWBLK:
      raise ValueError(
          f"the arrival buffer height {recv_rows} is not a whole "
          f"number of {fused_host.ROWBLK}-row blocks, which is the unit every "
          "transport into it moves"
      )
    contrib_rows = cfg.ragged_rows_alloc
    wire = cfg.form.wire_dtype
    out_shape = [
        jax.ShapeDtypeStruct(
            (recv_rows, cfg.lane_blocks, fused_host.HIDDEN_LANE_BLOCK), wire
        ),
        jax.ShapeDtypeStruct(
            (
                fused_host.arrival_mirror_rows(
                    recv_rows, ep * g_local, cfg.tile_m
                ),
                cfg.tile_m,
            ),
            jnp.float32,
        ),
        jax.ShapeDtypeStruct(
            (contrib_rows, cfg.lane_blocks, fused_host.HIDDEN_LANE_BLOCK), wire
        ),
        jax.ShapeDtypeStruct(
            (
                fused_host.contrib_mirror_rows(
                    contrib_rows, g_local, cfg.tile_m
                ),
                cfg.tile_m,
            ),
            jnp.float32,
        ),
    ]
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=n_prefetch,
            in_specs=[hbm] * n_inputs
            + (
                [pl.BlockSpec(memory_space=pltpu.SEMAPHORE)]
                if cfg.wait_for_exchange
                else []
            )
            + (
                [pl.BlockSpec(memory_space=pltpu.SEMAPHORE)]
                if cfg.dedup
                else []
            ),
            out_specs=[hbm] * len(out_shape),
            scratch_shapes=scratch,
            grid=(),
        ),
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            collective_id=config.COLLECTIVE_ID,
            vmem_limit_bytes=plan_lib.vmem_limit(),
            # Bounds checks cost time and change no computed value.
            disable_bounds_checks=True,
        ),
        name=cfg.kernel_name,
    )

  # Step 3: Build the host wrapper that validates dynamic operands and
  # invokes `make_call`.
  def fn(
      tables,
      token_gather,
      expert_rows,
      expert_base,
      act_q_gathered,
      act_scales,
      w1,
      w2,
      w1s,
      w2s,
      w1b: Any = None,
      w2b: Any = None,
      *,
      recv_rows,
      visit,
      counts,
      rank,
      disp_tab: Any = None,
      wave_sems: Any = None,
      stage: Any = None,
      stage_tab: Any = None,
      fwd_tab: Any = None,
      fwd_cnt: Any = None,
      stage_sems: Any = None,
  ):
    """The served ragged transport form; returns arrivals and scales.

    act_q_gathered is the start kernel's receive buffer, disp_tab its
    dispatch table and wave_sems its receive semaphores (neither on a
    wait_for_exchange=False build).

    Args:
      tables: Block and row transport tables.
      token_gather: Token-gather table.
      expert_rows: Rows per local expert.
      expert_base: Slab base row per local expert.
      act_q_gathered: Receive buffer of token rows.
      act_scales: Activation row-scale slab, or None.
      w1: Gate/up weights.
      w2: Down weights.
      w1s: Gate/up weight scales, or None.
      w2s: Down weight scales, or None.
      w1b: Optional gate/up bias.
      w2b: Optional down bias.
      recv_rows: Height of the arrival buffer.
      visit: Expert visit list.
      counts: Count vector.
      rank: This shard's index.
      disp_tab: Dispatch table, or None.
      wave_sems: Receive-wave semaphores, or None.
      stage: De-Dup stage buffer, or None.
      stage_tab: De-Dup stage table, or None.
      fwd_tab: De-Dup forward table, or None.
      fwd_cnt: De-Dup forward run counts, or None.
      stage_sems: De-Dup stage semaphores, or None.

    Returns:
      (arrivals, arrival scales).

    Raises:
      ValueError: If the operands do not match the build.
    """
    if (disp_tab is not None) != cfg.wait_for_exchange or (
        wave_sems is not None
    ) != cfg.wait_for_exchange:
      raise ValueError(
          "the kernel was built with"
          f" wait_for_exchange={cfg.wait_for_exchange}, which"
          f" {'needs' if cfg.wait_for_exchange else 'takes no'} disp_tab and"
          " wave_sems"
      )
    for op_name, op_val in (
        ("stage", stage),
        ("stage_tab", stage_tab),
        ("fwd_tab", fwd_tab),
        ("fwd_cnt", fwd_cnt),
        ("stage_sems", stage_sems),
    ):
      if (op_val is not None) != cfg.dedup:
        raise ValueError(
            f"the kernel was built with dedup_slots={dedup_slots}, which "
            f"{'needs' if cfg.dedup else 'takes no'} {op_name}"
        )
    if cfg.dedup:
      dedup_ops: tuple[tuple[str, Any, tuple[int, ...]], ...] = (
          (
              "fwd_tab",
              fwd_tab,
              (
                  dedup_slots,
                  waves,
                  cfg.fwd_max_runs,
                  dedup_plan_mod.FWD_FIELDS,
              ),
          ),
          ("fwd_cnt", fwd_cnt, (dedup_slots, waves, cfg.fwd_slots)),
          ("stage_sems", stage_sems, (dedup_slots * waves,)),
      )
      for op_name, op_val, want in dedup_ops:
        if tuple(op_val.shape) != want:
          raise ValueError(
              f"{op_name} {tuple(op_val.shape)} does not match the "
              f"build (dedup_slots={dedup_slots}, W={waves}, "
              f"fwd_max_runs={cfg.fwd_max_runs}): want {want}"
          )
    for operand, present, what in (
        (act_scales, cfg.has_ls_vm, "activation row scale"),
        (w1s, cfg.has_scales, "gate/up weight scale table"),
        (w2s, cfg.has_scales, "down weight scale table"),
        (w1b, cfg.has_w1_bias, "gate/up bias"),
        (w2b, cfg.has_w2_bias, "down bias"),
    ):
      if (operand is not None) != present:
        raise ValueError(
            "the kernel was built for weight format "
            f"{cfg.weight_format!r} with has_w1_bias={has_w1_bias} "
            f"has_w2_bias={has_w2_bias}, "
            f"which {'wants' if present else 'has no operand for'} the "
            f"{what}, and it is "
            f"{'present' if operand is not None else 'absent'}"
        )
    act_scale_arg = ()
    if cfg.has_ls_vm:
      act_scale_arg = (
          act_scales.reshape(
              fused_host.act_scale_slab_rows(cfg.ragged_rows_alloc),
              fused_host.HIDDEN_LANE_BLOCK,
          ),
      )
    scale_args = (
        (w1s.astype(jnp.float32), w2s.astype(jnp.float32))
        if cfg.has_scales
        else ()
    )
    biases = tuple(b.astype(jnp.float32) for b in (w1b, w2b) if b is not None)
    # Every SMEM table goes in flat (1-D): Mosaic pads the minor dim of a
    # rank >= 2 SMEM array to 128 words, and a [S*W*128, 4] forward table
    # would then exceed XLA's callee share of SMEM.
    dedup_smem_args = ()
    if cfg.dedup:
      dedup_smem_args = (
          stage_tab.astype(jnp.int32).reshape(-1),
          fwd_tab.astype(jnp.int32).reshape(-1),
          fwd_cnt.astype(jnp.int32).reshape(-1),
      )
    dedup_hbm_args = (
        (stage.reshape(-1, cfg.lane_blocks, fused_host.HIDDEN_LANE_BLOCK),)
        if cfg.dedup
        else ()
    )
    prefetch_args = (
        *tables,
        expert_rows.astype(jnp.int32),
        expert_base.astype(jnp.int32),
        visit.astype(jnp.int32),
        counts.astype(jnp.int32),
        rank.astype(jnp.int32),
        *dedup_smem_args,
    )
    smem_scratch_shapes = (
        [(cfg.gather_window_rows,)] * 2
        + ([(4 * ep * waves,), (1,)] if cfg.wait_for_exchange else [])
        + ([(1,)] if cfg.dedup else [])
    )
    config.check_smem_footprint(
        [a.shape for a in prefetch_args] + smem_scratch_shapes,
        what=(
            f"dedup_moe kernel (ep={ep}, W={waves}, dedup_slots={dedup_slots})"
        ),
    )
    call = make_call(recv_rows)
    res = call(
        *prefetch_args,
        token_gather.astype(jnp.int32),
        act_q_gathered.reshape(
            -1, cfg.lane_blocks, fused_host.HIDDEN_LANE_BLOCK
        ),
        *(
            (disp_tab.astype(jnp.int32).reshape(-1),)
            if cfg.wait_for_exchange
            else ()
        ),
        *act_scale_arg,
        w1,
        w2,
        *scale_args,
        *biases,
        *dedup_hbm_args,
        *((wave_sems,) if cfg.wait_for_exchange else ()),
        *((stage_sems,) if cfg.dedup else ()),
    )
    recv, rscl = res[:2]  # The contribution slab and its scales are scratch.
    return recv, rscl

  return fn


# Pallas keys its kernel-to-jaxpr cache on the `kernel` object itself, so
# memoizing the build is what makes pallas hit across builds.
_BUILD_CACHE = {}
# Two threads missing on the same key would each build a kernel, destroying
# the object identity this cache holds, so the miss path is serialized.
_BUILD_CACHE_LOCK = threading.Lock()


def build_fused_ep_moe_kernel(**kwargs):
  """Memoizing front door for _build_fused_ep_moe_kernel.

  The cache is unbounded. fused_host.ragged_stride_bound's docstring states
  the growth law that decides how many entries a deployment ends up holding;
  the size is logged on every insert so a boot that is accumulating
  programs says so, rather than being read off host memory afterwards.

  Args:
    **kwargs: _build_fused_ep_moe_kernel's arguments.

  Returns:
    The (cached) host wrapper of the built kernel.
  """
  key = tuple(sorted(kwargs.items()))
  fn = _BUILD_CACHE.get(key)
  if fn is None:
    with _BUILD_CACHE_LOCK:
      fn = _BUILD_CACHE.get(key)
      if fn is None:
        fn = _build_fused_ep_moe_kernel(**kwargs)
        _BUILD_CACHE[key] = fn
        logger.info(
            "dedup_moe: built program %d for %s",
            len(_BUILD_CACHE),
            ", ".join(f"{k}={v}" for k, v in key),
        )
  return fn
