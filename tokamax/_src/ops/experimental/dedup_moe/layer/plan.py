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
"""Static layer plan: t-bucket sizing and policy resolution."""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan as dedup_plan_mod
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology as topology_lib
from tokamax._src.ops.experimental.dedup_moe.layer import config as layer_config
from tokamax._src.ops.experimental.dedup_moe.layer import mode_registry

DEFAULT_T_BUCKETS = layer_config.DEFAULT_T_BUCKETS
UnifiedRules = layer_config.UnifiedRules
DedupMoeConfig = layer_config.DedupMoeConfig

# Runs on every ep the layer supports (no De-Dup): any config the rules or a
# pinned mode cannot build.
FALLBACK_MODE = mode_registry.FALLBACK_MODE
# Per-target-wave forwarding packs the target wave in a bit field.
MAX_PTW_WAVES = dedup_plan_mod.MAX_PTW_WAVES
POLICIES = ("auto", "rules")
# The mode name the rules policy reports.
RULES_MODE = "rules"

# A measured mode table: per (board_size, num_boards) of a topology with
# mirror gateways and per weight format, ((t_max, mode), ...) rows ordered by
# t_max; the first row whose t_max >= t wins (None: any larger t). Tables are
# only consulted for t buckets inside MEASURED_T_RANGE.
MeasuredPolicy = tuple[tuple[int | None, str], ...]
MEASURED_POLICY: dict[tuple[int, int, str], MeasuredPolicy] = {}
MEASURED_T_RANGE: tuple[int, int] = (0, 0)


@dataclasses.dataclass(frozen=True)
class UnifiedPlan:
  """What `modular.dedup_moe` runs at one t bucket (static).

  Attributes:
    t_bucket: The t bucket.
    mode: The resolved mode name, or RULES_MODE for the rules.
    kwargs: `monolithic.dedup_ep_moe` keyword arguments.
    notes: Derivation notes for logs and tests.
  """

  t_bucket: int
  mode: str
  kwargs: dict[str, Any]
  notes: tuple[tuple[str, Any], ...] = ()


def expected_board_fanout(ep: int, top_k: int, board_size: int) -> float:
  """Returns the expected devices of a remote board a token needs.

  E[# devices of a remote board a token needs | it needs the board], under
  uniform routing: a device is needed with p = 1 - (1 - 1/ep)^K, so a board
  of bs devices is needed with 1 - (1 - p)^bs and then holds
  bs p / (1 - (1 - p)^bs) needing devices. A staged token saves (F - 1) of
  its F cross-board rows.

  Args:
    ep: Expert-parallel width.
    top_k: Experts per token.
    board_size: Devices per board.

  Returns:
    The expected fan-out; 1.0 for fewer than two boards.
  """
  if ep < 2 * board_size:
    return 1.0
  p = 1.0 - (1.0 - 1.0 / ep) ** top_k
  board = 1.0 - (1.0 - p) ** board_size
  return board_size * p / board


def t_bucket(t_local: int, buckets: Sequence[int] = DEFAULT_T_BUCKETS) -> int:
  """Returns the t bucket of a token count.

  Args:
    t_local: Tokens per device.
    buckets: Ascending buckets.

  Returns:
    The smallest bucket >= t_local; above the last, the next power of two.

  Raises:
    ValueError: If t_local < 1.
  """
  if t_local < 1:
    raise ValueError(f"t_local must be >= 1; got {t_local}")
  for b in sorted(buckets):
    if t_local <= b:
      return int(b)
  return 1 << (t_local - 1).bit_length()


def bucket_math(
    t_local: int, buckets: Sequence[int] = DEFAULT_T_BUCKETS
) -> float:
  """Returns the fraction of a bucket's rows that are padding.

  Args:
    t_local: Tokens per device.
    buckets: Ascending buckets.

  Returns:
    (bucket - t_local) / bucket.
  """
  tb = t_bucket(t_local, buckets)
  return (tb - t_local) / tb if tb else math.nan


def static_fwd_max_runs(
    num_slots: int,
    waves: int,
    t_local: int,
    factor: float = 1.0,
) -> int:
  """Static per-target-wave forward-table depth (no routing at trace time).

  The depth is max(MAX_FWD_RUNS, next_pow2(factor * t_local)), capped by the
  SMEM table budget (mode_registry.FWD_TAB_SMEM_BUDGET_BYTES). It sets the
  table SMEM and the size of the run-compaction sorts, so it is kept as
  small as typical needs allow: the forward runs a (gateway slot, issue
  wave) needs are about t_local, while the provable bound (t_local times the
  board size) would make the tables and sorts much larger.

  Routing-proof: a (slot, wave) that would need more runs than the depth
  demotes its excess staged tokens to direct sends (same bits; it only
  loses the De-Dup savings).

  Args:
    num_slots: Gateway stage slots (topology tables).
    waves: Waves.
    t_local: Tokens per device (the bucket).
    factor: Depth / t_local (UnifiedRules.fwd_runs_factor).

  Returns:
    max_runs (>= dedup_plan.MAX_FWD_RUNS).
  """
  per_run = num_slots * waves * dedup_plan_mod.FWD_FIELDS * 4
  budget = mode_registry.FWD_TAB_SMEM_BUDGET_BYTES // per_run
  budget = 1 << (budget.bit_length() - 1) if budget > 0 else 1
  want = max(1, math.ceil(factor * t_local))
  want = 1 << (want - 1).bit_length()
  return max(dedup_plan_mod.MAX_FWD_RUNS, min(want, budget))


def derive_config(
    topology: topology_lib.BoardTopology | None,
    ep: int,
    g_local: int,
    t_local: int,
    *,
    top_k: int,
    hidden: int,
    rules: UnifiedRules = UnifiedRules(),
) -> tuple[str, dict[str, Any], dict[str, Any]]:
  """The rules config: (mode name, dedup_ep_moe kwargs, derivation notes).

  Args:
    topology: The board topology, or None (direct wave dispatch only).
    ep: Expert-parallel width.
    g_local: Experts per device.
    t_local: Tokens per device (the t bucket).
    top_k: K.
    hidden: D.
    rules: The formulas.

  Returns:
    (mode name, kwargs, notes).
  """
  boards = topology.num_boards if topology is not None else 1
  board_size = topology.board_size if topology is not None else ep
  fanout = expected_board_fanout(ep, top_k, board_size) if boards >= 2 else 1.0
  if boards >= 2 and rules.board_fanout > 0:
    fanout = rules.board_fanout
  savings = t_local * (1.0 - 1.0 / fanout)
  dedup = (
      topology is not None
      and boards >= max(2, rules.dedup_min_boards)
      and fanout >= rules.dedup_min_board_fanout
      and savings >= rules.dedup_min_savings
  )
  notes: dict[str, Any] = dict(
      boards=boards,
      rows_per_device=t_local * top_k,
      board_fanout=round(fanout, 3),
      dedup_savings=round(savings, 1),
  )
  num_slots = 0
  if dedup:
    assert topology is not None
    num_slots = topology.tables().num_slots
    static_mib = num_slots * t_local * hidden * 2 / 2**20
    notes["static_stage_mib"] = round(static_mib, 1)
    if static_mib > rules.static_stage_max_mib:
      # The static stage buffer does not fit: send direct (every size).
      dedup = False
      notes["dedup_off"] = (
          f"static stage {static_mib:.1f} MiB > {rules.static_stage_max_mib}"
      )
  if dedup:
    waves = rules.waves_dedup
  else:
    waves = (
        rules.waves_large if g_local >= rules.waves_g_cut else rules.waves_small
    )
  if dedup and waves > MAX_PTW_WAVES:
    waves = min(rules.waves_small, MAX_PTW_WAVES)
  while waves > 1 and g_local % waves:
    waves //= 2
  kw: dict[str, Any] = dict(
      waves=waves, dedup_topology=topology if dedup else None
  )
  if dedup:
    min_fanout = (
        rules.min_fanout_small
        if boards <= rules.min_fanout_boards_cut
        else rules.min_fanout_large
    )
    kw.update(
        dedup_min_fanout=min_fanout,
        dedup_fwd_max_runs=static_fwd_max_runs(
            num_slots, waves, t_local, factor=rules.fwd_runs_factor
        ),
    )
    notes.update(min_fanout=min_fanout, num_slots=num_slots)
  notes.update(waves=waves, dedup=dedup)
  return mode_registry.mode_name(dedup=dedup, waves=waves), kw, notes


def measured_mode(
    topology: topology_lib.BoardTopology | None,
    t_bucket_: int,
    weight_format: str,
) -> str | None:
  """The measured mode for (topology, t bucket, format), if there is one.

  Args:
    topology: The board topology, or None.
    t_bucket_: The t bucket.
    weight_format: `fp8` or `bf16`.

  Returns:
    A mode name, or None where no measured table covers the point.
  """
  if topology is None or topology.gateways is not None:
    return None
  lo, hi = MEASURED_T_RANGE
  if not lo <= t_bucket_ <= hi:
    return None
  rows = MEASURED_POLICY.get(
      (topology.board_size, topology.num_boards, str(weight_format))
  )
  if rows is None:
    return None
  for t_max, mode in rows:
    if t_max is None or t_bucket_ <= t_max:
      return mode
  return None


def fallback_mode(g_local: int) -> str:
  """The direct mode with the most supported waves that divide g_local."""
  waves = max(w for w in mode_registry.SUPPORTED_WAVES if g_local % w == 0)
  return mode_registry.mode_name(dedup=False, waves=min(waves, 4))


def resolve_policy(cfg: DedupMoeConfig, t_bucket_: int) -> str:
  """The concrete policy (`rules` or a mode name) at this bucket.

  `auto` takes the measured table where it covers (topology, format, t) and
  the rules everywhere else.

  Args:
    cfg: The layer config.
    t_bucket_: The t bucket.

  Returns:
    `rules` or a mode name.

  Raises:
    ValueError: If cfg.policy is neither a policy nor a mode name.
  """
  if cfg.policy == "auto":
    mode = measured_mode(cfg.topology, t_bucket_, cfg.weight_format)
    return "rules" if mode is None else mode
  if cfg.policy == "rules":
    return "rules"
  mode_registry.parse_mode(cfg.policy)  # Raises on an unknown name.
  return cfg.policy


def _static_schedule(
    kwargs: dict[str, Any], cfg: DedupMoeConfig, t_local: int
) -> dict[str, Any]:
  """A mode's kwargs with its De-Dup forward-table depth sized statically."""
  topology = kwargs.get("dedup_topology")
  if topology is None:
    return kwargs
  kwargs = dict(kwargs)
  kwargs["dedup_fwd_max_runs"] = static_fwd_max_runs(
      topology.tables().num_slots,
      kwargs["waves"],
      t_local,
      factor=cfg.rules.fwd_runs_factor,
  )
  return kwargs


def plan(cfg: DedupMoeConfig, t_local: int) -> UnifiedPlan:
  """Returns the static plan `modular.dedup_moe` runs for a token count.

  Args:
    cfg: The layer config.
    t_local: Tokens per device; rounded up to a t bucket.

  Returns:
    The UnifiedPlan.
  """
  tb = t_bucket(t_local, cfg.t_buckets)
  policy = resolve_policy(cfg, tb)
  auto_notes = (("policy", f"auto->{policy}"),) if cfg.policy == "auto" else ()
  if policy == "rules":
    mode, kw, notes = derive_config(
        cfg.topology,
        cfg.ep,
        cfg.g_local,
        tb,
        top_k=cfg.top_k,
        hidden=cfg.hidden,
        rules=cfg.rules,
    )
    return UnifiedPlan(
        tb,
        RULES_MODE,
        kw,
        tuple(sorted(notes.items())) + (("rules_mode", mode),) + auto_notes,
    )
  notes = list(auto_notes)
  try:
    kw = mode_registry.resolve_mode(
        policy, ep=cfg.ep, g_local=cfg.g_local, topology=cfg.topology
    )
    mode = policy
  except mode_registry.SkipCaseError as e:
    notes.append(("fallback_reason", f"{policy}: {e}"))
    mode = fallback_mode(cfg.g_local)
    kw = mode_registry.resolve_mode(
        mode, ep=cfg.ep, g_local=cfg.g_local, topology=cfg.topology
    )
  return UnifiedPlan(tb, mode, _static_schedule(kw, cfg, tb), tuple(notes))
