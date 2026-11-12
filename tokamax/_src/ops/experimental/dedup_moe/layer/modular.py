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
r"""Model-facing entry point of the De-Dup MoE layer.

A model's MoE layer calls

    topology = BoardTopology.uniform(num_devices=ep, board_size=bs)  # optional
    cfg = DedupMoeConfig(ep=ep, hidden=D, num_experts=E, top_k=K, ffn=F,
                         topology=topology)
    prepared = prepare_weights(mesh, w1, w2, w1_scale, w2_scale, cfg=cfg)
    y = dedup_moe(x, router_logits, prepared, cfg)   # inside jax.jit

and always runs ONE kernel configuration per (topology, t bucket, weight
format), chosen statically at trace time by `cfg.policy` (layer/plan.py):

  `auto` (default): a measured mode table where one covers the point, and
      the formula config (`plan.derive_config`) everywhere else.
  `rules`: always `plan.derive_config`.
  `<mode name>`: pins a mode (mode_registry.MODES).

Without a topology every policy runs the direct wave dispatch. All policies
run the same arithmetic and give bit-identical outputs.

Everything here is static at trace time: t_local comes from x's shape and is
rounded up to a t bucket, and the layer is `monolithic.dedup_ep_moe` with
the plan's keyword arguments.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from tokamax._src.ops.experimental.dedup_moe.common import arg
from tokamax._src.ops.experimental.dedup_moe.layer import config as layer_config
from tokamax._src.ops.experimental.dedup_moe.layer import monolithic
from tokamax._src.ops.experimental.dedup_moe.layer import plan as layer_plan

AXIS = layer_config.AXIS

# Re-export config and plan symbols so `layer.modular` is a complete surface.
DEFAULT_T_BUCKETS = layer_config.DEFAULT_T_BUCKETS
UnifiedRules = layer_config.UnifiedRules
DedupMoeConfig = layer_config.DedupMoeConfig
Config = layer_config.Config
PreparedMoe = layer_config.PreparedMoe
prepare_weights = layer_config.prepare_weights
UnifiedPlan = layer_plan.UnifiedPlan
plan = layer_plan.plan

# A pad row of the gating logits: 0 at the row's K chosen (local) experts,
# this elsewhere, so softmax top-k picks exactly those K.
_PAD_LOGIT = -1.0e4


def _pad_expert_ids(me, n_pad: int, *, g_local: int, board: int, topk: int):
  """[n_pad, K] expert ids for pad rows: this device's experts first.

  Pad j takes local experts (j + i) % g_local for i < g_local, then the
  experts of the next devices on the same board, so pad rows send no rows
  off the board (none off the device when g_local >= K).

  Args:
    me: This device's index (traced).
    n_pad: Pad rows.
    g_local: Experts per device.
    board: Devices per board (ep without a topology).
    topk: K.

  Returns:
    [n_pad, K] int32 expert ids.

  Raises:
    ValueError: If K exceeds the experts on one board.
  """
  j = jnp.arange(n_pad, dtype=jnp.int32)[:, None]
  i = jnp.arange(topk, dtype=jnp.int32)[None, :]
  dev_in_board = me % board
  hop = i // g_local
  dev = (me - dev_in_board) + (dev_in_board + hop) % board
  if topk > g_local * board:
    raise ValueError(f"top_k={topk} > experts on one board")
  return dev * g_local + (j + i) % g_local


def dedup_moe(
    x: jax.Array,
    router_logits: jax.Array,
    prepared: PreparedMoe,
    cfg: DedupMoeConfig,
    *,
    topk_ids: jax.Array | None = None,
) -> jax.Array:
  """One MoE layer: [T, D] tokens -> [T, D], T = ep * t_local.

  jit-friendly; every decision is static (from shapes and cfg). x, the
  router logits and topk_ids are token-sharded over the mesh axis (each
  device holds its t_local rows).

  Args:
    x: [T, D] bf16 activations.
    router_logits: [T, E] router logits in ORIGINAL expert order (softmax top-k
      with renormalization).
    prepared: From prepare_weights.
    cfg: The layer config.
    topk_ids: Optional [T, K] expert ids (original order) chosen by the caller;
      the weights are then the softmax scores at those ids.

  Returns:
    [T, D], sharded like x.

  Raises:
    ValueError: If the tokens do not split over the mesh axis.
  """
  ep = cfg.ep
  n, _ = x.shape
  if n % ep:
    raise ValueError(f"{n} tokens do not split over ep={ep}")
  t_local = n // ep
  p = plan(cfg, t_local)
  return _run_plan(x, router_logits, prepared, cfg, p, t_local, topk_ids)


def _run_plan(x, gating, prepared, cfg, p, t_local, topk_ids):
  """Pads to the bucket, remaps to slot order, runs the plan, unpads."""
  mesh, ep, k = prepared.mesh, cfg.ep, cfg.top_k
  tb, g_local = p.t_bucket, cfg.g_local
  board = cfg.topology.board_size if cfg.topology is not None else ep
  n_pad = tb - t_local
  remap = prepared.slot_to_orig is not None
  canonical = (
      remap if cfg.canonical_routing is None else bool(cfg.canonical_routing)
  )
  # ------------------------------------------------------------------
  # Step 1: Canonical softmax scoring, slot remap, and bucket pad.
  # ------------------------------------------------------------------
  if n_pad or remap or canonical:
    has_ids = topk_ids is not None

    def prep(x_l, g_l, *rest):
      ids_l = rest[0] if has_ids else None
      maps = rest[1 if has_ids else 0 :]
      if canonical:
        # Softmax in original expert order (the reduction order of the
        # placement-free layer), then the logits whose sigmoid is that
        # probability; the layer scores them elementwise (scoring_fn
        # "sigmoid"), so the slot order no longer matters.
        prob = jax.nn.softmax(g_l.astype(jnp.float32), axis=-1)
        g_l = jnp.log(prob) - jnp.log1p(-prob)
      if remap:
        s2o, o2s = maps[0], maps[1]
        g_l = jnp.take(g_l, s2o, axis=1)
        if ids_l is not None:
          ids_l = jnp.take(o2s, ids_l.astype(jnp.int32), axis=0)
      if n_pad:
        me = jax.lax.axis_index(AXIS)
        pad_ids = _pad_expert_ids(
            me, n_pad, g_local=g_local, board=board, topk=k
        )
        x_l = jnp.concatenate(
            [x_l, jnp.zeros((n_pad, x_l.shape[1]), x_l.dtype)], axis=0
        )
        g_pad = jnp.full((n_pad, g_l.shape[1]), _PAD_LOGIT, g_l.dtype)
        g_pad = g_pad.at[jnp.arange(n_pad)[:, None], pad_ids].set(0.0)
        g_l = jnp.concatenate([g_l, g_pad], axis=0)
        if ids_l is not None:
          ids_l = jnp.concatenate([ids_l.astype(jnp.int32), pad_ids], axis=0)
      return (x_l, g_l) + ((ids_l,) if has_ids else ())

    args = (x, gating) + ((topk_ids,) if has_ids else ())
    specs = (P(AXIS),) * len(args)
    if remap:
      args += (prepared.slot_to_orig, prepared.orig_to_slot)
      specs += (P(), P())
    out = jax.shard_map(
        prep,
        mesh=mesh,
        in_specs=specs,
        out_specs=(P(AXIS),) * (3 if has_ids else 2),
        check_vma=False,
    )(*args)
    x, gating = out[0], out[1]
    topk_ids = out[2] if has_ids else None

  # ------------------------------------------------------------------
  # Step 2: Run the monolithic layer with the plan's arguments.
  # ------------------------------------------------------------------
  kwargs = dict(p.kwargs)
  if canonical:
    kwargs["scoring_fn"] = "sigmoid"
  if topk_ids is not None:
    kwargs["hash_topk_ids"] = topk_ids
  if prepared.ws1 is not None:
    kwargs.update(ws1=prepared.ws1, ws2=prepared.ws2)
  y = monolithic.dedup_ep_moe(
      x,
      prepared.w1,
      prepared.w2,
      prepared.w1_scale,
      prepared.w2_scale,
      gating,
      topk=k,
      renormalize=True,
      mesh=mesh,
      capacity=cfg.capacity,
      weight_format=arg.WeightFormat(cfg.weight_format),
      **kwargs,
  )

  # ------------------------------------------------------------------
  # Step 3: Slice off bucket padding rows on each shard.
  # ------------------------------------------------------------------
  if n_pad:
    y = jax.shard_map(
        lambda y_l: y_l[:t_local],
        mesh=mesh,
        in_specs=(P(AXIS),),
        out_specs=P(AXIS),
        check_vma=False,
    )(y)
  return y


def make_layer(cfg: DedupMoeConfig, mesh: Any, shared_weights=(None, None)):
  """Returns `layer(x, w1, w2, w1s, w2s, gating, hash_ids)` running dedup_moe.

  The weights and hash_ids are already in slot order (no placement remap).

  Args:
    cfg: The layer config.
    mesh: One-axis EP mesh.
    shared_weights: (ws1, ws2) replicated shared expert, or (None, None).

  Returns:
    The layer function.
  """

  def layer(x, w1, w2, w1s, w2s, gating, hash_ids):
    prepared = PreparedMoe(
        w1, w2, w1s, w2s, shared_weights[0], shared_weights[1], None, None, mesh
    )
    return dedup_moe(x, gating, prepared, cfg, topk_ids=hash_ids)

  return layer
