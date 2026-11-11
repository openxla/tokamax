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
"""Layer configuration and the prepared weight pytree of the De-Dup MoE layer."""

from collections.abc import Sequence
import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology as topology_lib
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import config as kernel_config

AXIS = kernel_config.AXIS

# Decode-friendly t_local buckets: one compile per bucket a model hits.
DEFAULT_T_BUCKETS = (8, 16, 32, 64, 128, 256, 512, 1024)


@dataclasses.dataclass(frozen=True)
class UnifiedRules:
  """The formulas `layer.plan.derive_config` turns topology and size into knobs.

  Fixed in every config (not knobs, built into the layer): ramp waves, no
  result scale mirror on a bf16 wire, fast send lists, HBM-pinned exchange
  buffers, 2 tail-tile height rungs, one-hop result push with a Pallas
  combine, and per-target-wave forwarding when De-Dup is on. The formulas do
  not depend on the weight format.

  Attributes:
    dedup_min_board_fanout: De-Dup only when the expected fan-out per needed
      remote board F (plan.expected_board_fanout) is >= this, so that one staged
      row replaces at least this many cross-board rows on average.
    board_fanout: Per-layer measured F (offline, from the layer's routing
      statistics or expert placement) replacing the uniform-routing estimate; 0
      = plan.expected_board_fanout. Skewed routing has a lower effective F.
    dedup_min_savings: ... and t_local * (1 - 1/F), the cross-board rows saved
      per device and expert slot, is >= this: below it the De-Dup plan's fixed
      cost outweighs the saved rows.
    dedup_min_boards: Never De-Dup below this many boards.
    min_fanout_small: dedup_min_fanout up to `min_fanout_boards_cut` boards.
    min_fanout_boards_cut: See above.
    min_fanout_large: dedup_min_fanout above the cut.
    waves_dedup: Waves with De-Dup.
    waves_large: Waves without De-Dup when g_local >= waves_g_cut.
    waves_small: Waves without De-Dup otherwise.
    waves_g_cut: See above.
    static_stage_max_mib: De-Dup only while the gateway's static stage buffer
      (num_slots * t_local * D * 2 bytes, pinned HBM) fits this; above it the
      config sends direct, which works at every size.
    fwd_runs_factor: Static forward-table depth / t_local
      (plan.static_fwd_max_runs). An overflowing forward table demotes the
      excess staged tokens to direct sends (same bits, less savings).
  """

  dedup_min_board_fanout: float = 2.0
  board_fanout: float = 0.0
  dedup_min_savings: float = 85.0
  dedup_min_boards: int = 2
  min_fanout_small: int = 2
  min_fanout_boards_cut: int = 4
  min_fanout_large: int = 3
  waves_dedup: int = 4
  waves_large: int = 8
  waves_small: int = 4
  waves_g_cut: int = 16
  static_stage_max_mib: float = 64.0
  fwd_runs_factor: float = 2.0


@dataclasses.dataclass(frozen=True)
class DedupMoeConfig:
  """Static description of one MoE layer deployment.

  Attributes:
    ep: Expert-parallel width (devices in the mesh axis).
    hidden: Model width D.
    num_experts: Routed experts E.
    top_k: Experts per token K.
    ffn: Expert FFN width F.
    topology: The board topology of the mesh axis, or None. Without one the
      layer always runs the direct wave dispatch.
    weight_format: `fp8` (W8A8, default) or `bf16` (W16A16).
    capacity: Kernel tile height (a multiple of 128 on TPU).
    include_shared_expert: The prepared weights carry a replicated shared expert
      (ws1, ws2) that the layer adds.
    placement: Optional per-layer expert placement: slot -> original expert id
      (a permutation of range(E)); prepare_weights applies it.
    policy: `auto` (default), `rules` (plan.derive_config) or a mode_registry
      mode name. `auto` uses a measured per-topology mode table where one is
      available for (topology, format, t bucket), and the rules everywhere else.
    t_buckets: Ascending t_local buckets; a t_local above the last rounds up to
      the next power of two.
    rules: The plan.derive_config formulas.
    canonical_routing: Score the router in ORIGINAL expert order (softmax here,
      then the layer's elementwise sigmoid on the matching logits), so an expert
      placement is invisible bit for bit. None: on exactly when the prepared
      weights carry a placement.
  """

  ep: int
  hidden: int
  num_experts: int
  top_k: int
  ffn: int
  topology: topology_lib.BoardTopology | None = None
  weight_format: str = "fp8"
  capacity: int = 128
  include_shared_expert: bool = False
  placement: tuple[int, ...] | None = None
  policy: str = "auto"
  t_buckets: tuple[int, ...] = DEFAULT_T_BUCKETS
  rules: UnifiedRules = UnifiedRules()
  canonical_routing: bool | None = None

  def __post_init__(self):
    if self.topology is not None and self.topology.num_devices != self.ep:
      raise ValueError(
          f"topology covers {self.topology.num_devices} devices; ep={self.ep}"
      )
    if self.num_experts % self.ep:
      raise ValueError(
          f"num_experts={self.num_experts} does not split over ep={self.ep}"
      )

  @property
  def g_local(self) -> int:
    """Experts per device."""
    return self.num_experts // self.ep


# Alias matching Tokamax's `routing_moe/layer/config.py` naming convention.
Config = DedupMoeConfig


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "w1",
        "w2",
        "w1_scale",
        "w2_scale",
        "ws1",
        "ws2",
        "slot_to_orig",
        "orig_to_slot",
    ],
    meta_fields=["mesh"],
)
@dataclasses.dataclass
class PreparedMoe:
  """Expert weights in slot order and the placement maps (a pytree)."""

  w1: Any
  w2: Any
  w1_scale: Any
  w2_scale: Any
  ws1: Any
  ws2: Any
  slot_to_orig: Any  # [E] int32 replicated, or None (identity)
  orig_to_slot: Any  # [E] int32 replicated, or None (identity)
  mesh: Any


def check_placement(placement, e_total: int) -> np.ndarray | None:
  """Validates an expert placement.

  Args:
    placement: [E] slot -> original expert id, or None.
    e_total: E.

  Returns:
    The placement as int32 numpy, or None for no placement or the identity.

  Raises:
    ValueError: If the placement has the wrong shape.
    NotImplementedError: If the placement is not a permutation.
  """
  if placement is None:
    return None
  p = np.asarray(placement, np.int32)
  if p.shape != (e_total,):
    raise ValueError(f"placement must be [{e_total}]; got {p.shape}")
  if not np.array_equal(np.sort(p), np.arange(e_total, dtype=np.int32)):
    raise NotImplementedError(
        "placement must be a permutation (slot -> original expert); replicated"
        " experts need a per-token replica choice that this layer does not"
        " make yet"
    )
  if np.array_equal(p, np.arange(e_total, dtype=np.int32)):
    return None
  return p


def prepare_weights(
    mesh: Any,
    w1: Any,
    w2: Any,
    w1_scale: Any = None,
    w2_scale: Any = None,
    *,
    cfg: DedupMoeConfig,
    ws1: Any = None,
    ws2: Any = None,
    placement: Sequence[int] | np.ndarray | None = None,
) -> PreparedMoe:
  """One-time offline step: weights into slot order plus placement maps.

  Applies the slot -> original expert permutation (`placement`, default
  cfg.placement) to the expert axis of w1, w2 and their scales, and keeps
  the maps `modular.dedup_moe` uses to bring the router's logits and expert
  ids into slot order. Identity (weights returned as given) without a
  placement.

  Args:
    mesh: One-axis EP mesh; expert stacks are sharded on it.
    w1: [E, D, 2F] expert gate+up weights (original expert order).
    w2: [E, F, D].
    w1_scale: [E, 2F] f32 per-channel scales (fp8), or None (bf16).
    w2_scale: [E, D] or None.
    cfg: The layer config.
    ws1: Optional shared expert [D, 2Fs] (replicated).
    ws2: Optional shared expert [Fs, D].
    placement: [E] slot -> original expert id, or None for cfg.placement.

  Returns:
    The prepared weights.

  Raises:
    ValueError: If cfg.include_shared_expert does not match ws1 and ws2.
  """
  if cfg.include_shared_expert != (ws1 is not None):
    raise ValueError("cfg.include_shared_expert must match passing ws1/ws2")
  e_total = cfg.num_experts
  p = check_placement(
      cfg.placement if placement is None else placement, e_total
  )
  shard = NamedSharding(mesh, P(AXIS))
  rep = NamedSharding(mesh, P())
  if p is None:
    w1_s, w2_s, w1_scale_s, w2_scale_s = w1, w2, w1_scale, w2_scale
    s2o = o2s = None
  else:
    idx = jax.device_put(p, rep)
    take = jax.jit(lambda w, i: jnp.take(w, i, axis=0), out_shardings=shard)
    w1_s, w2_s, w1_scale_s, w2_scale_s = (
        None if w is None else take(w, idx)
        for w in (w1, w2, w1_scale, w2_scale)
    )
    s2o = idx
    o2s = jax.device_put(np.argsort(p).astype(np.int32), rep)
  return PreparedMoe(
      w1_s, w2_s, w1_scale_s, w2_scale_s, ws1, ws2, s2o, o2s, mesh
  )
