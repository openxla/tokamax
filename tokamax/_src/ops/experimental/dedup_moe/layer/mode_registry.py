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
"""Named layer modes of the De-Dup MoE layer and their resolution.

A mode names one TensorCore configuration of `monolithic.dedup_ep_moe`:

  `direct_w<W>`  direct wave dispatch: each (token, device) pair is sent once,
                 in W ramp waves (wave_dispatch.wave_plan).
  `dedup_w<W>`   De-Dup wave dispatch: tokens needed by several devices of a
                 remote board are sent once to that board's gateway, which
                 forwards them (needs a `BoardTopology`).

W is one of SUPPORTED_WAVES. Every mode computes the same arithmetic, so all
of them give bit-identical outputs; they differ only in how the tokens move.
"""

import dataclasses
import itertools
import re
from typing import Any

from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan as dedup_plan_mod
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology as topology_lib

DIRECT = "direct"
DEDUP = "dedup"
SUPPORTED_WAVES = (1, 2, 4, 8)
# The mode every configuration supports (no topology needed).
FALLBACK_MODE = "direct_w4"

# SMEM a per-target-wave forward table may take: the fused kernel's whole
# XLA callee share is fused_kernel.config.SMEM_CALLEE_BUDGET_BYTES, and its
# other tables need much less than the rest.
FWD_TAB_SMEM_BUDGET_BYTES = 384 * 1024

_MODE_RE = re.compile(r"^(direct|dedup)_w(\d+)$")


class SkipCaseError(Exception):
  """A mode that does not apply to this shape or topology."""


@dataclasses.dataclass(frozen=True)
class ModeSpec:
  """One layer mode.

  Attributes:
    dedup: Whether the mode De-Dups the dispatch.
    waves: Expert waves of the dispatch.
  """

  dedup: bool
  waves: int

  @property
  def name(self) -> str:
    """The mode name."""
    return mode_name(dedup=self.dedup, waves=self.waves)

  @property
  def label(self) -> str:
    """Human-readable description."""
    kind = "De-Dup" if self.dedup else "Direct"
    return f"{kind} wave dispatch, W={self.waves}, TensorCore sender"


def mode_name(*, dedup: bool, waves: int) -> str:
  """Returns the name of the mode with these settings."""
  return f"{DEDUP if dedup else DIRECT}_w{waves}"


def parse_mode(name: str) -> ModeSpec:
  """Parses a mode name.

  Args:
    name: A mode name such as `direct_w4` or `dedup_w4`.

  Returns:
    The ModeSpec.

  Raises:
    ValueError: If the name is not a mode.
  """
  m = _MODE_RE.match(name)
  if m is None or int(m.group(2)) not in SUPPORTED_WAVES:
    raise ValueError(
        f"unknown mode {name!r}: want direct_w<W> or dedup_w<W> with W in"
        f" {SUPPORTED_WAVES}"
    )
  return ModeSpec(dedup=m.group(1) == DEDUP, waves=int(m.group(2)))


MODES: dict[str, ModeSpec] = {
    spec.name: spec
    for spec in (
        ModeSpec(dedup=d, waves=w)
        for d, w in itertools.product((False, True), SUPPORTED_WAVES)
    )
}


def resolve_mode(
    name: str,
    *,
    ep: int,
    g_local: int,
    topology: topology_lib.BoardTopology | None,
) -> dict[str, Any]:
  """Resolves a mode name to `dedup_ep_moe` keyword arguments.

  A De-Dup mode's forward-table depth (dedup_fwd_max_runs) is left at its
  default; layer/plan.py sizes it statically.

  Args:
    name: A mode name.
    ep: Expert-parallel width.
    g_local: Experts per device.
    topology: The board topology, or None.

  Returns:
    The keyword arguments (`waves` and `dedup_topology`).

  Raises:
    SkipCaseError: If the mode cannot run on this shape or topology.
    ValueError: If the name is not a mode.
  """
  spec = parse_mode(name)
  if g_local % spec.waves:
    raise SkipCaseError(f"waves={spec.waves} does not divide g_local={g_local}")
  if spec.dedup:
    if topology is None:
      raise SkipCaseError(f"{name} needs a BoardTopology")
    if topology.num_devices != ep:
      raise SkipCaseError(
          f"the topology covers {topology.num_devices} devices, not ep={ep}"
      )
    if spec.waves > dedup_plan_mod.MAX_PTW_WAVES:
      raise SkipCaseError(
          f"De-Dup supports at most {dedup_plan_mod.MAX_PTW_WAVES} waves"
      )
  return dict(waves=spec.waves, dedup_topology=topology if spec.dedup else None)
