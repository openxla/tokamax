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
"""Monolithic shard-mapped De-Dup MoE layer.

`dedup_ep_moe` wraps the fused kernel in a `shard_map` over the
expert-parallel mesh axis in seven stages:
  1. Token-parallel split and input row quantization.
  2. Router scoring, top-k selection (`pallas_select`), and renormalization.
  3. Single-collective packed routing exchange (`all_gather` over EP).
  4. Wave dispatch start (`wave_dispatch.start`) and optimization barrier.
  5. Overlapped sharded routing-table construction and `tg_assemble`.
  6. Fused EP MoE Pallas kernel (`build_fused_ep_moe_kernel`) and send drain.
  7. Arrival combine (`combine_arrivals`), shared expert FFN, and TP restore.

Stages 1-3 and the routing tables of stage 5 are shared with
`fused_moe.layer.fused_ep_moe_v2`; the wave dispatch, De-Dup, kernel and
combine are this layer's own.
"""

# Roadmap of upcoming optimizations (see also README.md):
# TODO: SparseCore gateway forwarding: forward the staged rows
#   on the SparseCore instead of the TensorCore. Blocked on compiler support
#   for zero-output SparseCore kernels and on the JAX writes-to-array-input
#   check; re-enable once available and re-measure against TensorCore
#   forwarding.
# TODO: SparseCore sender path (SparseCore gather and
#   contiguous senders): upstream in a follow-up once it no longer needs
#   private JAX patches.
# TODO: Optimize the combine phase with a hierarchical
#   reduce-scatter: first try a SparseCore collective-offload hierarchical
#   reduce-scatter; if that does not pan out, implement a TensorCore
#   hierarchical reduce-scatter.
# TODO: De-Dup on the return path (De-Dup collect): reduce
#   partial expert outputs at the gateway before sending them back across
#   boards.
# TODO: Offload the routing exchange (all-gather) to the
#   SparseCore collectives engine with multicast, once the compiler support
#   lands.
# TODO: Re-validate and retune the auto policy table (for
#   example, whether SparseCore forwarding helps), and add decode-specific
#   (very small t_local) and larger-slice configurations.
# TODO: Potential: EPLB / expert placement integration, and
#   more weight formats (W4A8, sub-channel, fp4).

import threading
from typing import Any, cast

import jax
from jax import lax
import jax.numpy as jnp
from tokamax._src.ops.experimental.dedup_moe.common import arg
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan as dedup_plan_mod
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology as topology_lib
from tokamax._src.ops.experimental.dedup_moe.dispatch import tg_assemble
from tokamax._src.ops.experimental.dedup_moe.dispatch import wave_dispatch
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import kernel as kernel_lib
from tokamax._src.ops.experimental.dedup_moe.io_relayout import input_pack
from tokamax._src.ops.experimental.dedup_moe.io_relayout import output_reduce
from tokamax._src.ops.experimental.fused_moe import host as fused_host
from tokamax._src.ops.experimental.fused_moe import layer as fused_layer
from tokamax._src.ops.experimental.fused_moe import router_ops

WeightFormat = arg.WeightFormat
weight_form = arg.weight_form
weight_format_of_dtype = arg.weight_format_of_dtype
BoardTopology = topology_lib.BoardTopology

HIDDEN_LANE_BLOCK = fused_host.HIDDEN_LANE_BLOCK
MAX_ROUTING_BLOCK = fused_host.MAX_ROUTING_BLOCK
ROWBLK = fused_host.ROWBLK
U32_SUBLANE_TILE = fused_host.U32_SUBLANE_TILE
act_scale_slab_rows = fused_host.act_scale_slab_rows
align_up = fused_host.align_up
ragged_stride_bound = fused_host.ragged_stride_bound
routing_block = fused_host.routing_block
token_gather_window_rows = fused_host.token_gather_window_rows

GatheredPairs = fused_host.GatheredPairs
build_routing_tables_sharded = fused_host.build_routing_tables_sharded
pair_block_hist = fused_host.pair_block_hist
shard_count_vector = fused_host.shard_count_vector
shard_expert_slabs = fused_host.shard_expert_slabs
shard_push_tables_in_rows = fused_host.shard_push_tables_in_rows
shard_transport_tables_in_blocks = fused_host.shard_transport_tables_in_blocks

build_fused_ep_moe_kernel = kernel_lib.build_fused_ep_moe_kernel

TokenReplicaLayout = input_pack.TokenReplicaLayout
rowquant_fp8 = input_pack.rowquant_fp8
shared_expert_ffn = input_pack.shared_expert_ffn
combine_arrivals = output_reduce.combine_arrivals
pallas_select = router_ops.pallas_select

# Router scoring and the packed routing exchange, shared with
# `fused_moe.layer`.
# pylint: disable=protected-access
SCORING_FNS = fused_layer.SCORING_FNS
_apply_scoring_fn = fused_layer._apply_scoring_fn
_exchange_layout = fused_layer._exchange_layout
_pack_routing_exchange = fused_layer._pack_routing_exchange
_unpack_routing_exchange = fused_layer._unpack_routing_exchange
_relabel_expert_ids_to_mesh_order = (
    fused_layer._relabel_expert_ids_to_mesh_order
)
# pylint: enable=protected-access

# On a real TPU the token-gather DMA windows are 128-row aligned, so the
# kernel tile height (capacity) must be a multiple of this.
TPU_CAPACITY_ALIGNMENT = 128


def _loop_variant_one(x_l):
  """f32 1.0 that depends on this shard's own activations.

  XLA does not fold a float `0 * v` (v may be inf or NaN), so the result
  stays an op on x_l: per call, and per iteration of any loop whose x
  varies. The select keeps a non-finite first element from turning the
  scales into NaN, which `1 + 0 * v` alone would. x_l is the shard_map's
  local block, so this is a local slice with no collective.

  Args:
    x_l: This shard's activations.

  Returns:
    A f32 scalar equal to 1.0.
  """
  v = x_l[0, 0].astype(jnp.float32)
  return 1.0 + 0.0 * jnp.where(jnp.isfinite(v), v, 0.0)


def _on_tpu(mesh) -> bool:
  return all(d.platform == "tpu" for d in mesh.devices.flat)


# Per-config cache of the shard_map'd MoE callable; a fresh shard_map
# body would re-trace the whole kernel for every hidden layer.
_LAYER_SM_CACHE = {}
_LAYER_SM_CACHE_LOCK = threading.Lock()


def dedup_ep_moe(
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    gating,
    w1_bias=None,
    w2_bias=None,
    *,
    topk,
    renormalize,
    mesh,
    capacity,
    block=None,
    ragged_stride=None,
    weight_format=WeightFormat.FP8,
    act_fn="silu",
    rank=None,
    mesh_ep_ranks=None,
    scoring_fn="softmax",
    score_bias=None,
    hash_topk_ids=None,
    routed_scaling_factor=1.0,
    token_replica_groups=None,
    waves=4,
    dedup_topology: BoardTopology | None = None,
    dedup_min_fanout=2,
    dedup_fwd_max_runs=dedup_plan_mod.MAX_FWD_RUNS,
    ws1=None,
    ws2=None,
):
  """Run one MoE layer through the wave-dispatch (and De-Dup) fused kernel.

  x [tokens, hidden], w1 [experts, hidden, 2 * inter], w2 [experts,
  inter, hidden] and gating [tokens, experts], all sharded over the mesh
  axis. Returns [tokens, hidden], sharded like x.

  weight_format names one of the kernel's accepted weight forms (fp8 W8A8
  or bf16 W16A16). w1_scale and w2_scale are per output channel for fp8
  (e4m3) weights, [experts, 2 * inter] and [experts, hidden]; on an
  unquantized weight (bf16) there are no scales and both must be None. The
  token rows are quantized to fp8 only where the format's matmuls take fp8
  rows.

  w1_bias [experts, 1, 2 * inter] and w2_bias [experts, 1, hidden] are
  optional and independent: the gate and up halves are added to the first
  matmul's post-scale accumulator before the activation, and the down bias
  to the second matmul's post-scale row before it is quantized for the wire.

  block and ragged_stride default to the values this function derives;
  an explicitly passed one still has to divide tokens // ep * topk and to
  reach the no-drop bound respectively. act_fn selects the fused FFN
  activation from the kernel's ACT_FNS and is a build-time constant.

  token_replica_groups lists ordered groups of device-mesh indices whose
  x and gating rows are identical (for example, a TP group). Each member
  routes and computes a disjoint slice, then gathers the combined outputs
  within its group to restore the input row order. None means singleton
  groups with no token replication.

  mesh_ep_ranks names the EP rank at every device-mesh index. A non-identity
  order relabels only the selected expert ids after top-k.

  hash_topk_ids [tokens, topk] i32 replaces the router's selection with
  expert ids the caller already chose. The gate logits are still scored: the
  WEIGHTS are the scores at the selected ids.

  Wave dispatch (wave_dispatch.py): each shard sends every (token,
  destination shard) pair once, over a split-phase ragged all-to-all, in
  `waves` groups ("waves") of each destination's experts. Experts are
  visited fewest rows first and cut into waves of growing modeled compute
  (1 : 2 : 4 : ..., wave_dispatch.wave_plan), and the fused kernel starts
  each expert as soon as its wave has landed, so the later waves arrive
  under the first experts' matmuls. `waves` must divide the experts per
  shard; the mesh needs at least 3 devices. Every buffer is sized for the
  worst case, so every routing is dropless.

  dedup_topology adds De-Dup to the dispatch: a token needed by at least
  `dedup_min_fanout` devices of a remote board is sent once to that board's
  gateway device, which forwards it to those devices inside the fused
  kernel. A forwarded row counts in its target's own first wave
  (per-target-wave forwarding), so a receiver waits for exactly the rows the
  direct send would have given it in each wave. dedup_fwd_max_runs is the
  gateway forward-table depth per (served source, wave): size it statically
  (layer.plan.static_fwd_max_runs) or from the routing
  (wave_dispatch.forward_runs_needed); a gateway that would need more runs
  sends the excess staged tokens direct instead. Without a topology the
  layer runs the direct wave dispatch. Both give the same bits.

  The weight scale tables (and the optional bias tables) reach the kernel
  as `w * one` with `one = 1 + 0 * x_l[0, 0]` (`_loop_variant_one`). Inside
  an XLA while loop a Mosaic operand that is a loop-invariant while
  parameter can lose its HBM memory-space annotation, and memory-space
  assignment may then prefetch the small f32 scale tables into VMEM while
  the kernel DMAs them as HBM. The product keeps them per-call values. Same
  values bit for bit: `w * 1.0 == w`.

  ws1 [hidden, 2 * Fs] and ws2 [Fs, hidden] add a replicated shared expert
  (`shared_expert_ffn`) to every token's routed output.

  Args:
    x: [tokens, hidden] activations, sharded over the mesh axis.
    w1: [experts, hidden, 2 * inter] gate/up weights.
    w2: [experts, inter, hidden] down weights.
    w1_scale: [experts, 2 * inter] f32 scales (fp8), or None (bf16).
    w2_scale: [experts, hidden] f32 scales, or None.
    gating: [tokens, experts] router logits.
    w1_bias: Optional [experts, 1, 2 * inter] gate/up bias.
    w2_bias: Optional [experts, 1, hidden] down bias.
    topk: Experts per token.
    renormalize: Renormalize the top-k weights.
    mesh: One-axis EP mesh.
    capacity: Kernel tile height; a multiple of 128 on TPU.
    block: Routing block width, or None to derive it.
    ragged_stride: Slab rows per shard, or None to derive it.
    weight_format: fp8 or bf16.
    act_fn: Fused FFN activation.
    rank: Optional [ep, 1] i32 shard index of every device, sharded over the
      mesh; None uses arange(ep).
    mesh_ep_ranks: EP rank at every mesh index, or None for identity.
    scoring_fn: Router scoring function.
    score_bias: Optional selection bias added to the scores.
    hash_topk_ids: Optional [tokens, topk] precomputed expert ids.
    routed_scaling_factor: Scale applied to the routed weights.
    token_replica_groups: Groups of mesh indices with identical rows, or None.
    waves: Expert waves of the dispatch.
    dedup_topology: The board topology to De-Dup over, or None for the direct
      wave dispatch.
    dedup_min_fanout: Staging threshold.
    dedup_fwd_max_runs: Gateway forward runs per (served source, wave).
    ws1: Optional [hidden, 2 * Fs] shared expert gate/up weights.
    ws2: Optional [Fs, hidden] shared expert down weights.

  Returns:
    [tokens, hidden] output, sharded like x.

  Raises:
    ValueError: If the operands, formats or options are inconsistent.
  """
  form = weight_form(weight_format)
  weight_format = form.name
  for name, w in (("w1", w1), ("w2", w2)):
    actual = weight_format_of_dtype(w.dtype)
    if actual != weight_format:
      raise ValueError(
          f"weight format {weight_format!r} takes "
          f"{jnp.dtype(cast(Any, form.weight_dtype)).name} expert weights and "
          f"{name} is {jnp.dtype(w.dtype).name}"
          f"{f', which is the {actual!r} form' if actual else ''}"
      )
  if form.has_scales != (w1_scale is not None):
    raise ValueError(
        f"weight format {weight_format!r} carries "
        f"{form.scale_layout} weight scales and w1_scale is "
        f"{'present' if w1_scale is not None else 'absent'}"
    )
  if (w1_scale is None) != (w2_scale is None):
    raise ValueError("both weight scales are supplied together or neither is")
  n_tokens, hidden = x.shape
  e_total = w1.shape[0]
  inter = w1.shape[2] // 2
  # The scales are constants of the weights, so whatever prepares the
  # weights gives them the kernel's [E, N] layout once; a table carrying a
  # loader's singleton axes is refused rather than reshaped on every call.
  if form.has_scales:
    for name, s, n in (
        ("w1_scale", w1_scale, 2 * inter),
        ("w2_scale", w2_scale, hidden),
    ):
      if s.ndim != 2 or s.shape[0] != e_total or s.shape[-1] != n:
        want = (e_total, n)
        raise ValueError(
            f"{name} layout {tuple(s.shape)} is not the "
            f"{form.scale_layout} form {want} the kernel "
            "takes"
        )
  if _on_tpu(mesh) and capacity % TPU_CAPACITY_ALIGNMENT:
    raise ValueError(
        f"capacity={capacity} must be a multiple of {TPU_CAPACITY_ALIGNMENT}"
        " on TPU: the kernel's token-gather DMA windows are"
        f" {TPU_CAPACITY_ALIGNMENT}-row aligned"
    )
  (ax,) = mesh.axis_names
  ep = mesh.shape[ax]
  if mesh_ep_ranks is not None:
    mesh_ep_ranks = tuple(int(rank) for rank in mesh_ep_ranks)
    if tuple(sorted(mesh_ep_ranks)) != tuple(range(ep)):
      raise ValueError(
          "mesh_ep_ranks must be a permutation of "
          f"range({ep}); got {mesh_ep_ranks}"
      )
  g_local = e_total // ep
  if n_tokens % ep or gating.shape != (n_tokens, e_total):
    raise ValueError(
        "input and gating rows must agree and partition evenly over EP"
    )
  token_layout = TokenReplicaLayout.create(
      ep=ep,
      input_rows=n_tokens // ep,
      groups=token_replica_groups,
      row_alignment=U32_SUBLANE_TILE,
  )
  t_local = token_layout.local_rows
  n_tokens = ep * t_local
  pspec = jax.sharding.PartitionSpec
  stride_bound = ragged_stride_bound(n_tokens, topk, e_total, capacity)
  if ragged_stride is None:
    ragged_stride = stride_bound
  if block is None:
    block = routing_block(t_local, topk)
  if ragged_stride % capacity != 0:
    raise ValueError(
        f"ragged_stride must be a multiple of capacity {capacity}; got "
        f"{ragged_stride}. Pass {stride_bound}."
    )
  if ragged_stride < stride_bound:
    raise ValueError(
        f"ragged_stride {ragged_stride} is below the no-drop bound "
        f"{stride_bound}, so one shard's slab could bleed into the "
        f"next. Pass {stride_bound} or more."
    )
  has_w1_bias = w1_bias is not None
  has_w2_bias = w2_bias is not None
  has_score_bias = score_bias is not None
  has_hash_ids = hash_topk_ids is not None
  if scoring_fn not in SCORING_FNS:
    raise ValueError(
        f"scoring_fn must be one of {SCORING_FNS}; got {scoring_fn!r}"
    )
  if has_score_bias and score_bias.shape != (e_total,):
    raise ValueError(
        f"score_bias must be one value per global expert, [{e_total}]; "
        f"got {tuple(score_bias.shape)}"
    )
  if has_hash_ids:
    if hash_topk_ids.shape != (x.shape[0], topk):
      raise ValueError(
          "hash_topk_ids must be one expert id per token slot, "
          f"[{x.shape[0]}, {topk}]; got {tuple(hash_topk_ids.shape)}"
      )
    if not jnp.issubdtype(hash_topk_ids.dtype, jnp.integer):
      raise ValueError(
          "hash_topk_ids are expert ids and must be an integer dtype; "
          f"got {jnp.dtype(hash_topk_ids.dtype).name}"
      )
  waves = int(waves)
  if waves < 1 or g_local % waves:
    raise ValueError(
        "waves must be a positive divisor of the experts per shard "
        f"{g_local}; got {waves}"
    )
  if ep < 3:
    raise ValueError(
        "wave dispatch carries its wave cuts in the dispatch table's "
        f"padding rows, which needs ep >= 3; got ep={ep}"
    )
  if (ws1 is None) != (ws2 is None):
    raise ValueError(
        "both shared expert weights (ws1, ws2) are supplied together or"
        " neither is"
    )
  has_shared_expert = ws1 is not None
  if ws1 is not None and ws2 is not None:
    if (
        ws1.ndim != 2
        or ws2.ndim != 2
        or ws1.shape[0] != hidden
        or ws2.shape[1] != hidden
        or ws1.shape[1] != 2 * ws2.shape[0]
    ):
      raise ValueError(
          f"shared expert weights must have shapes ws1=[{hidden}, 2*Fs] and"
          f" ws2=[Fs, {hidden}]; got ws1={tuple(ws1.shape)},"
          f" ws2={tuple(ws2.shape)}"
      )
  dedup = dedup_topology is not None
  dedup_fwd_max_runs = int(dedup_fwd_max_runs)
  if dedup:
    assert dedup_topology is not None
    if dedup_topology.num_devices != ep:
      raise ValueError(
          f"dedup_topology covers {dedup_topology.num_devices} devices; the"
          f" mesh axis has ep={ep}"
      )
    if waves > dedup_plan_mod.MAX_PTW_WAVES:
      raise ValueError(
          "per-target-wave forwarding packs the target wave in 4 bits: "
          f"waves <= {dedup_plan_mod.MAX_PTW_WAVES}; got {waves}"
      )
    if dedup_fwd_max_runs < 1:
      raise ValueError(
          f"dedup_fwd_max_runs must be >= 1; got {dedup_fwd_max_runs}"
      )
    dedup_tables = dedup_topology.tables()
    dedup_slots = dedup_tables.max_served
    dedup_board_size = dedup_tables.board_size
  else:
    dedup_tables = None
    dedup_slots = 0
    dedup_board_size = dedup_plan_mod.MAX_BOARD_SIZE
  kfn = build_fused_ep_moe_kernel(
      g_local=g_local,
      capacity=capacity,
      hidden=hidden,
      inter=inter,
      ep=ep,
      ragged_rows_alloc=ragged_stride,
      weight_format=weight_format,
      act_fn=act_fn,
      has_w1_bias=has_w1_bias,
      has_w2_bias=has_w2_bias,
      waves=waves,
      dedup_slots=dedup_slots,
      dedup_fwd_max_runs=dedup_fwd_max_runs,
      dedup_board_size=dedup_board_size,
  )

  def local_fn(x_l, rank_l, w1_l, w2_l, *scales_gating_biases):
    """One shard's half of the layer: route, dispatch, kernel, combine."""
    # The scales come before the gating logits and the biases after
    # them, which is the operand order the scaled formats have always
    # had: a format that supplies no scales drops them out of the middle
    # rather than moving anything that stayed.
    operands = iter(scales_gating_biases)
    w1s_l = next(operands) if form.has_scales else None
    w2s_l = next(operands) if form.has_scales else None
    gate_l = next(operands)
    hash_l = next(operands) if has_hash_ids else None
    score_bias_l = next(operands) if has_score_bias else None

    # ------------------------------------------------------------------
    # Step 1: Token-parallel split and input row quantization.
    # ------------------------------------------------------------------
    # This shard's index arrives as data, not from `lax.axis_index`: that
    # lowers to `partition-id`, which the SPMD partitioner rejects when the
    # module is serialized and recompiled (for example through jax.export).
    me = rank_l[0, 0]
    x_l = token_layout.split(x_l, me)
    gate_l = token_layout.split(gate_l, me, fill_value=-jnp.inf)
    if hash_l is not None:
      hash_l = token_layout.split(hash_l, me)
    if form.quantized_activations:
      # Whatever produced x_l -- the replica slice above, the caller's t
      # bucket padding (modular.dedup_moe concatenates pad rows in the same
      # XLA program), any model op -- must not fuse into the row
      # quantization: on TPU the fused form changes the BF16 rounding of the
      # row quantization, so fp8 rows (and every output of their tokens)
      # would change with the producer. Materialize x_l first.
      x_l = lax.optimization_barrier(x_l)
    rows_bf16 = x_l.astype(jnp.bfloat16)
    if form.quantized_activations:
      q_l, row_scale_l = rowquant_fp8(rows_bf16)
    else:
      # The rows go on the wire as they arrived. There is no scale to
      # carry, so nothing downstream builds or ships one.
      q_l, row_scale_l = rows_bf16, None
    q_v = q_l.reshape(t_local, hidden // HIDDEN_LANE_BLOCK, HIDDEN_LANE_BLOCK)

    # ------------------------------------------------------------------
    # Step 2: Router scoring, top-k selection, and renormalization.
    # ------------------------------------------------------------------
    scores = _apply_scoring_fn(gate_l, scoring_fn)
    select_rows = MAX_ROUTING_BLOCK
    while t_local % select_rows:
      select_rows //= 2
    if hash_l is not None:
      # The selection was made by the caller's table, so the selector
      # does not run at all; the weights are still the scores at the
      # selected ids, which is the only thing the selector would have
      # contributed here.
      topk_idx = hash_l.astype(jnp.int32)
      topk_weights = jnp.take_along_axis(scores, topk_idx, axis=-1)
    elif score_bias_l is None:
      topk_weights, topk_idx = pallas_select(
          scores, topk=topk, block_rows=select_rows
      )
    else:
      # Selection is based on (scores + score_bias_l); the weights come
      # from the scores.
      topk_weights, topk_idx = pallas_select(
          scores + score_bias_l.astype(jnp.float32)[None, :],
          topk=topk,
          block_rows=select_rows,
          weight_scores=scores,
      )
    topk_idx = _relabel_expert_ids_to_mesh_order(
        topk_idx, g_local=g_local, mesh_ep_ranks=mesh_ep_ranks
    )
    # A row of scores carrying no real value routes nowhere. It was
    # dropped by arithmetic accident: the selector gives every slot of
    # such a row a large negative sentinel, and the renormalization below
    # divided by their sum, which OVERFLOWED float32 to -inf and so
    # returned exactly zero. That outcome is a property of the
    # accumulation dtype and the association order, not of the design --
    # a wider accumulator, a reassociating compiler pass, or a different
    # renormalization would silently turn such a row from dropped into
    # routed at full weight to expert zero. Mask it explicitly instead.
    row_routes = jnp.any(jnp.isfinite(scores), axis=-1, keepdims=True)
    if renormalize:
      denom = jnp.maximum(topk_weights.sum(axis=-1, keepdims=True), 1e-20)
      topk_weights = topk_weights / jnp.where(row_routes, denom, 1.0)
    if routed_scaling_factor != 1.0:
      topk_weights = topk_weights * jnp.float32(routed_scaling_factor)
    topk_weights = jnp.where(row_routes, topk_weights, 0.0)

    # ------------------------------------------------------------------
    # Step 3: Single-collective packed routing exchange across EP.
    # ------------------------------------------------------------------
    # A count of the masked rows would be the other half of this: an
    # incident today is degraded output against a completely clean log.
    # It is not added here because the only in-trace channel for it,
    # jax.debug.print, makes the program refuse to lower for TPU from a
    # host, which is how this repository verifies that a change served
    # the same program. A counter belongs in a metrics channel outside
    # the trace.
    # One all-gather carries the indices, the row scales where the rows
    # were quantized, and the per-expert pair counts the routing plan
    # would otherwise exchange in a collective of its own.
    local_blocks = topk_idx.astype(jnp.int32).reshape(
        t_local * topk // block, block
    )
    block_hist = pair_block_hist(local_blocks, e_total)
    scale_bits_l = None
    if row_scale_l is not None:
      scale_bits_l = lax.bitcast_convert_type(
          row_scale_l[:, 0].astype(jnp.float32), jnp.int32
      )
    layout = _exchange_layout(
        t_local, topk, e_total, form.quantized_activations
    )
    exchange = lax.all_gather(
        _pack_routing_exchange(
            local_blocks.reshape(-1),
            scale_bits_l,
            block_hist.sum(axis=0),
            layout,
        ),
        ax,
        axis=0,
        tiled=True,
    ).reshape(ep, layout[-1])
    pairs_g, _, rows_by_dest = _unpack_routing_exchange(
        exchange, layout, t_local, topk, e_total
    )

    # ------------------------------------------------------------------
    # Step 4: Wave dispatch start and optimization barrier.
    # ------------------------------------------------------------------
    wd = wave_dispatch.start(
        q_v,
        pairs_g.reshape(n_tokens, topk),
        rows_by_dest,
        me,
        ep=ep,
        g_local=g_local,
        waves=waves,
        scale_bits=scale_bits_l,
        dedup_tables=dedup_tables,
        min_fanout=dedup_min_fanout,
        fwd_max_runs=dedup_fwd_max_runs,
    )
    # The barrier makes every table op below consume a value the start
    # kernel produced, so none can be hoisted in front of it. The
    # weight-scale relayouts ride it too, so they also land under the
    # exchange.
    # pairs_g rides it rather than its expert_blocks view: pinned at
    # the barrier, the view was relaid out for its consumers after it
    # (three copies per layer); formed after it, XLA lays it out for
    # them directly.
    g_arrays = (pairs_g, local_blocks, block_hist, rows_by_dest)
    if dedup:
      wave_of_b = wd.wave_of
      (g_arrays, wave_of_b, w1s_l, w2s_l), fut = lax.optimization_barrier(
          ((g_arrays, wave_of_b, w1s_l, w2s_l), wd.fut)
      )
      wd = wd._replace(fut=fut, wave_of=wave_of_b)
      # The gateway's forward tables are built after the start kernel,
      # so they overlap the dispatch DMAs in flight.
      wd = wave_dispatch.materialize_smem_ops(
          wd,
          g_arrays[0].reshape(n_tokens, topk),
          me,
          t_local=t_local,
          waves=waves,
          dedup_tables=dedup_tables,
          min_fanout=dedup_min_fanout,
      )
    else:
      (g_arrays, w1s_l, w2s_l), fut = lax.optimization_barrier(
          ((g_arrays, w1s_l, w2s_l), wd.fut)
      )
      wd = wd._replace(fut=fut)
    gathered = GatheredPairs(
        expert_blocks=g_arrays[0].reshape(n_tokens * topk // block, block),
        local_blocks=g_arrays[1],
        block_hist=g_arrays[2],
        rows_by_dest=g_arrays[3],
        n_tokens=n_tokens,
        topk=topk,
    )
    q_g = wd.fut[3]  # the receive buffer, filled wave by wave

    # ------------------------------------------------------------------
    # Step 5: Overlapped sharded routing tables and token-gather assembly.
    # ------------------------------------------------------------------
    # tg_assemble builds token_gather and scale_slab from the sender
    # lists, so the routing plan's slab rows (and, on an unquantized wire,
    # its mirror positions) are unused and dead-code-eliminated.
    routing = build_routing_tables_sharded(
        None,
        me,
        gathered=gathered,
        e_total=e_total,
        ep=ep,
        t_local=t_local,
        block=block,
        tile_m=capacity,
        shard_stride=ragged_stride,
    )
    block_tables = shard_transport_tables_in_blocks(
        routing, me, e_total=e_total, ep=ep
    )
    row_tables = shard_push_tables_in_rows(routing, me, e_total=e_total, ep=ep)
    expert_rows, slab_base = shard_expert_slabs(
        routing, me, e_total=e_total, ep=ep
    )
    # token_gather (and the fp8 activation scale slab when the rows are
    # quantized) come from the sender lists landed in wd.tg_land, with
    # no XLA scatter.
    nw = 2 if form.quantized_activations else 1
    scale_rows = act_scale_slab_rows(ragged_stride) if nw == 2 else None
    run_start_me = lax.dynamic_slice(
        routing.run_start_aligned, (me * g_local, 0), (g_local, ep)
    )
    tg_runs, tg_meta = tg_assemble.assemble_runs(
        rows_by_dest,
        me,
        ep=ep,
        g_local=g_local,
        t_local=t_local,
        topk=topk,
        nw=nw,
        slab_base=slab_base,
        run_start=run_start_me,
    )
    tg_rows = tg_assemble.gather_rows(ragged_stride, capacity)
    # The list semaphore is a DMA semaphore from the start kernel; an
    # emulated exchange (CPU tests) hands over complete lists and an integer
    # placeholder instead, and then there is nothing to wait for.
    tg_sem = (
        None
        if (wd.tg_sem is None or jnp.issubdtype(wd.tg_sem.dtype, jnp.integer))
        else wd.tg_sem
    )
    tg_flat, scale_bits_slab = tg_assemble.tg_assemble(
        wd.tg_land,
        tg_runs,
        tg_meta,
        tg_sem,
        ep=ep,
        nw=nw,
        n_tokens=n_tokens,
        tg_rows=tg_rows,
        scale_rows=scale_rows,
    )
    token_gather = tg_flat[: ragged_stride + token_gather_window_rows(capacity)]
    scale_slab = (
        None
        if scale_bits_slab is None
        else lax.bitcast_convert_type(scale_bits_slab, jnp.float32)
    )
    recv_rows = align_up(t_local * topk + (ROWBLK - 1) * e_total, ROWBLK)
    # The bias tables are per expert and per output channel on every
    # weight format, so they take one layout: [G, N].
    bias_it = operands
    w1b_k = next(bias_it).reshape(g_local, 2 * inter) if has_w1_bias else None
    w2b_k = next(bias_it).reshape(g_local, hidden) if has_w2_bias else None
    ws1_l = next(bias_it) if has_shared_expert else None
    ws2_l = next(bias_it) if has_shared_expert else None
    # Fewest rows first, the order the waves were laid out in, which the
    # kernel's per-wave wait keys on; empty experts come last.
    visit = lax.dynamic_index_in_dim(wd.visit_order, me, axis=0, keepdims=False)
    # One pass builds every count the kernel needs, still spread over
    # the expert-parallel axis; the kernel's scalar core closes them.
    counts = shard_count_vector(
        routing, expert_rows, me, e_total=e_total, ep=ep
    )

    # ------------------------------------------------------------------
    # Step 6: Fused EP MoE Pallas kernel and outgoing send drain.
    # ------------------------------------------------------------------
    # The kernel takes four of the seven transport tables and one of the
    # three push tables. The two totals are rows of `counts`, and three
    # more are the same array as one already here -- push_src is the
    # commit offset, push_len is the commit length, and the receive row
    # offset is the push destination in rows -- so each goes over once and
    # the kernel derives the rest. Every table shipped is another blocking
    # HBM->SMEM DMA at the head of the kernel, and the prefetch also has to
    # fit in 1 MB of SMEM.
    kernel_tables = (*block_tables[:3], block_tables[5], row_tables[0])
    # q_g is the exchange's receive buffer, and the kernel waits for each
    # wave on the semaphores the start kernel gave.
    wave_kw = dict(disp_tab=wd.disp_tab, wave_sems=wd.fut[1])
    if dedup:
      wave_kw.update(
          stage=wd.stage,
          stage_tab=wd.smem_ops.stage_tab,
          fwd_tab=wd.smem_ops.fwd_tab,
          fwd_cnt=wd.smem_ops.fwd_cnt,
          stage_sems=wd.stage_sems,
      )
    # Loop-safe scales (see the docstring): the small per-expert tables
    # leave this function as products with a per-call value, never as the
    # bare (possibly loop-invariant) parameters an enclosing XLA while
    # loop would let memory-space assignment move into VMEM. Exact: * 1.0.
    one = _loop_variant_one(x_l)
    w1s_l, w2s_l, w1b_k, w2b_k = (
        None if a is None else a * one.astype(a.dtype)
        for a in (w1s_l, w2s_l, w1b_k, w2b_k)
    )
    arrivals, arrival_scales = kfn(
        kernel_tables,
        token_gather,
        expert_rows,
        slab_base,
        q_g,
        scale_slab,
        w1_l,
        w2_l,
        w1s_l,
        w2s_l,
        w1b_k,
        w2b_k,
        recv_rows=recv_rows,
        visit=visit,
        counts=counts,
        rank=rank_l,
        **wave_kw,
    )
    arrivals = wave_dispatch.finish(wd, arrivals, ep=ep)

    # ------------------------------------------------------------------
    # Step 7: Arrival combine, shared expert FFN, and TP restore.
    # ------------------------------------------------------------------
    # The destination's own table: one arrival row per selection slot.
    combined = combine_arrivals(
        arrivals,
        arrival_scales,
        routing.pos,
        routing.mirror_pos,
        topk_weights,
        x_l.dtype,
    )
    if has_shared_expert:
      assert ws1_l is not None and ws2_l is not None
      shared_y = shared_expert_ffn(x_l, ws1_l, ws2_l, act_fn=act_fn).reshape(
          t_local, hidden
      )
      combined = (combined.astype(jnp.float32) + shared_y).astype(x_l.dtype)
    return token_layout.restore(combined, ax)

  # The biases ride the same expert-axis sharding as the weights they
  # belong to, so no shard ever holds a bias for an expert it does not own.
  bias_args = tuple(b for b in (w1_bias, w2_bias) if b is not None)
  shared_args = (ws1, ws2) if has_shared_expert else ()
  scale_args = (w1_scale, w2_scale) if form.has_scales else ()
  # `rank` rides the mesh axis so each shard reads its own index out of it.
  # A caller that serializes and recompiles the module MUST pass it as a
  # real operand:
  # an arange built here is a constant of the traced function, and sharding a
  # jit-internal constant compiles to `dynamic-slice(constant, partition-id)`
  # -- which XLA then refuses when the module is recompiled for one device
  # ("partitionId instruction is not supported for SPMD partitioning").
  # Under plain `jax.jit` with a mesh the constant form is fine, so the
  # default keeps the device tests and any JAX-native caller working.
  if rank is None:
    rank = jnp.arange(ep, dtype=jnp.int32).reshape(ep, 1)
  # The score bias is the one operand that is NOT sharded on the mesh axis:
  # it is indexed by GLOBAL expert id and every shard scores the full
  # expert set, so `pspec()` hands each shard the whole [e_total] vector.
  # Sharding it would give each shard a 1/ep slice of the experts and
  # silently bias the wrong ones.
  score_bias_args = (score_bias,) if has_score_bias else ()
  hash_args = (hash_topk_ids,) if has_hash_ids else ()
  in_specs = (
      (pspec(ax),) * (5 + len(scale_args) + len(hash_args))
      + (pspec(),) * len(score_bias_args)
      + (pspec(ax),) * len(bias_args)
      + (pspec(),) * len(shared_args)
  )
  args = (
      (x, rank, w1, w2)
      + scale_args
      + (gating,)
      + hash_args
      + score_bias_args
      + bias_args
      + shared_args
  )
  named_sharding = jax.sharding.NamedSharding
  args = tuple(
      jax.device_put(a, named_sharding(mesh, sp))
      for a, sp in zip(args, in_specs)
  )
  # local_fn closes over config-static values only -- the per-call data
  # are the shard_map arguments -- so this key is exact.
  key = (
      mesh,
      n_tokens,
      hidden,
      e_total,
      inter,
      topk,
      bool(renormalize),
      capacity,
      block,
      ragged_stride,
      weight_format,
      act_fn,
      mesh_ep_ranks,
      token_layout,
      x.dtype,
      w1.dtype,
      w2.dtype,
      form.has_scales and w1_scale.dtype,
      gating.dtype,
      has_w1_bias and w1_bias.dtype,
      has_w2_bias and w2_bias.dtype,
      scoring_fn,
      score_bias.dtype if score_bias is not None else False,
      has_hash_ids and hash_topk_ids.dtype,
      float(routed_scaling_factor),
      waves,
      dedup_topology,
      int(dedup_min_fanout),
      dedup_fwd_max_runs,
      has_shared_expert and ws1.shape,
      has_shared_expert and ws1.dtype,
  )
  sm = _LAYER_SM_CACHE.get(key)
  if sm is None:
    with _LAYER_SM_CACHE_LOCK:
      sm = _LAYER_SM_CACHE.get(key)
      if sm is None:
        sm = jax.shard_map(
            local_fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=pspec(ax),
            check_vma=False,
        )
        # The exchange's start kernel hands its DMA semaphores to the
        # fused kernel and to the drain, which is only valid inside
        # one XLA program. Under a caller's trace this jit is inlined;
        # an eager call would otherwise run the shard_map one
        # primitive at a time, a program each.
        sm = jax.jit(sm)
        _LAYER_SM_CACHE[key] = sm
  return sm(*args)
