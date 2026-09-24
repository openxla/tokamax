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
"""Serving layer for the fused expert-parallel MoE kernel.

fused_ep_moe_v2 wraps the kernel builder in a shard_map over the
expert-parallel mesh axis: quantize, dispatch, kernel, combine.
"""
import threading

import jax
import jax.numpy as jnp
from jax import lax

from .host import (
    HIDDEN_LANE_BLOCK, MAX_ROUTING_BLOCK, QB4, ROWBLK, U32_SUBLANE_TILE,
    GatheredPairs, WeightFormat, act_scale_slab_rows, align_up,
    build_routing_tables, build_routing_tables_sharded, expert_visit_list,
    local_slab_rows, pair_block_hist, pow2_shift, ragged_stride_bound,
    routing_block, shard_count_vector, shard_expert_slabs,
    shard_push_tables_in_rows, shard_token_gather,
    shard_transport_tables_in_blocks, token_gather_window_rows, weight_form,
    weight_format_of_dtype)
from .kernel import (
    build_combine_kernel, build_fused_ep_moe_kernel, combine_step_tokens,
    rowquant_fp8)
from .router_ops import pallas_select
from .token_parallel import TokenReplicaLayout

# The scoring functions the router can apply to the gate logits. These
# mirror `layers/adapter/moe_routing._apply_scoring_fn` exactly.
SCORING_FNS = ("softmax", "sigmoid", "sqrtsoftplus")


def _apply_scoring_fn(logits, scoring_fn):
    """Gate logits -> f32 routing scores. """
    scores = logits.astype(jnp.float32)
    if scoring_fn == "softmax":
        return jax.nn.softmax(scores, axis=-1)
    if scoring_fn == "sigmoid":
        return jax.nn.sigmoid(scores)
    if scoring_fn == "sqrtsoftplus":
        return jnp.sqrt(jax.nn.softplus(scores))
    raise NotImplementedError(
        f"the fused EP MoE router scores with {SCORING_FNS}; got "
        f"{scoring_fn!r}")


def _combine_arrivals(arrivals, arrival_scales, pos, mirror_pos, topk_weights,
                      out_dtype):
    """Combine arrivals [rows, lane blocks, 128] into [t_local, hidden].

    Two tables index two different objects, so each owns its own clamp.
    `pos` is the arrival ROW: a row of `arrivals`, and on the kernel path
    below it is a DMA address. `mirror_pos` is a position in the wire's scale
    mirror, which is [sublanes, tile_m] because a transport can only address
    the mirror a sublane at a time and a sublane holds tile_m f32 whatever is
    written in it -- so a whole source TILE's scales are one sublane and the
    mirror position is NOT the arrival row.

    Where the kernel path applies (kernel.combine_step_tokens) the row gather
    and the weighted sum are one Mosaic call: XLA answers `arrivals[rows]`
    with a SparseCore gather whose output is in HBM at any size, so the rows
    are written out and read straight back before the sum sees them.
    Everywhere else the gather is not offloaded and there is no round trip to
    remove, so the fused XLA epilogue below stays.

    Both paths fold the arrival row's scale and the router weight into one
    coefficient over K * t_local values, so each of the K * t_local * hidden
    elements is multiplied once rather than twice.
    """
    t_local, topk = pos.shape
    recv_rows, lane_blocks, lanes = arrivals.shape
    mirror_sublanes, tile_m = arrival_scales.shape
    hidden = lane_blocks * lanes
    # A gather clamps an out-of-range index to the last one; a DMA does not,
    # and a sublane index clamped on its own would pair the last sublane with
    # the wrong lane. Neither position is ever negative.
    rows = jnp.minimum(pos, recv_rows - 1)
    shift = pow2_shift(tile_m, "the mirror's lane count")
    scl = jnp.minimum(mirror_pos, mirror_sublanes * tile_m - 1)
    row_scales = arrival_scales[jnp.right_shift(scl, shift),
                                scl & (tile_m - 1)]
    coef = row_scales * topk_weights.astype(jnp.float32)
    if combine_step_tokens(t_local):
        combine = build_combine_kernel(t_local=t_local,
                                       topk=topk,
                                       hidden=hidden,
                                       lane_blocks=lane_blocks,
                                       wire_dtype=arrivals.dtype,
                                       out_dtype=out_dtype)
        return combine(arrivals, coef, rows)
    rows_fp8 = arrivals[pos.T.reshape(-1)]  # f8 [K*t, lane blocks, 128]
    coef_k = coef.T.reshape(-1)  # k-major [K*t_local]
    terms = []
    for k in range(topk):
        slot_rows = lax.slice(rows_fp8, (k * t_local, 0, 0),
                              ((k + 1) * t_local, lane_blocks, lanes))
        slot_coef = lax.slice(coef_k, (k * t_local, ), ((k + 1) * t_local, ))
        terms.append(slot_rows.astype(jnp.float32) * slot_coef[:, None, None])
    return sum(terms).astype(out_dtype).reshape(t_local, hidden)


def _exchange_layout(t_local, topk, e_total, quantized):
    """(scale offset, count offset, width) of the flat routing payload.

    ONE all-gather carries everything the other shards need to know about
    this shard's tokens: the selections, the row scales the wire quantized
    them by, and the per-expert pair counts the routing plan exchanges. The
    plan's own count collective disappears into this one, and the selections
    arrive already flat -- the pair grid the plan reads is a reshape of the
    gathered payload rather than a [T, topk] array whose 10 live lanes sit in
    a 128-lane tile.

    Section starts are lane-block aligned so every unpack slice is a view.
    """
    idx = align_up(t_local * topk, HIDDEN_LANE_BLOCK)
    scale = align_up(t_local, HIDDEN_LANE_BLOCK) if quantized else 0
    return idx, idx + scale, idx + scale + e_total


def _pack_routing_exchange(pairs, scale_bits, counts, layout):
    """The flat i32 payload: pairs [t*K], row-scale bits [t], counts [E].

    The router weight stays local: the destination combine applies it in f32
    from its own copy, so the wire never carries it.
    """
    scale_at, count_at, _ = layout
    parts = [pairs, jnp.zeros((scale_at - pairs.size, ), jnp.int32)]
    if scale_bits is not None:
        parts += [
            scale_bits,
            jnp.zeros((count_at - scale_at - scale_bits.size, ), jnp.int32)
        ]
    parts.append(counts)
    return jnp.concatenate([p for p in parts if p.size])


def _unpack_routing_exchange(exchange, layout, t_local, topk, e_total):
    """Inverse of _pack_routing_exchange over the gathered [ep, width] rows.

    Returns the gathered pairs [ep, t*K], the row-scale BIT PATTERNS [T] (the
    scale slab wants them as int32, so nothing is bitcast back) and the
    [ep, e_total] count table. The bits are None where the payload has none.
    """
    scale_at, count_at, _ = layout
    ep = exchange.shape[0]
    pairs = lax.slice(exchange, (0, 0), (ep, t_local * topk))
    scale_bits = None if scale_at == count_at else lax.slice(
        exchange, (0, scale_at), (ep, scale_at + t_local)).reshape(-1)
    counts = lax.slice(exchange, (0, count_at), (ep, count_at + e_total))
    return pairs, scale_bits, counts


def _relabel_expert_ids_to_mesh_order(topk_idx, *, g_local, mesh_ep_ranks):
    """Map selected experts from EP-rank order to device-mesh order.

    vLLM numbers contiguous expert blocks by EP rank, while the Mosaic remote
    DMA kernel addresses peers by their position in the device-id-ordered JAX
    mesh. ``mesh_ep_ranks[i]`` names the EP rank at mesh index ``i``.

    Apply the permutation after top-k selection. Reordering the full router
    logits before selection materializes a ``[tokens, experts]`` gather at the
    Torch/JAX boundary, and can also change which expert wins an exact tie.
    Here only the ``[tokens, topk]`` integer IDs move; the selected weights keep
    the arithmetic and tie behavior of the unpermuted router.
    """
    if mesh_ep_ranks is None:
        return topk_idx
    ep = len(mesh_ep_ranks)
    if tuple(sorted(mesh_ep_ranks)) != tuple(range(ep)):
        raise ValueError("mesh_ep_ranks must be a permutation of "
                         f"range({ep}); got {mesh_ep_ranks}")
    owning_rank = topk_idx // g_local
    mesh_block = jnp.zeros_like(topk_idx)
    for mesh_index, ep_rank in enumerate(mesh_ep_ranks):
        mesh_block = jnp.where(owning_rank == ep_rank, jnp.int32(mesh_index),
                               mesh_block)
    return mesh_block * g_local + topk_idx % g_local


# Local tokens at or above which the activation all-gather is enqueued ahead
# of the slab scatters. Measured, not chosen: +2.80% at t_local 2048 and
# -0.01 / -7.81 / -1.92 / -1.94% at 1024 / 256 / 64 / 32, because the scatters
# lose ~90 us of cover to buy a ~129 us hoist and only the largest shape's
# plan is long enough to pay for it.
AGQ_MIN_LOCAL_TOKENS = 1024

# Per-config cache of the shard_map'd MoE callable; a fresh shard_map
# body would re-trace the whole kernel for every hidden layer.
_LAYER_SM_CACHE = {}
_LAYER_SM_CACHE_LOCK = threading.Lock()


def fused_ep_moe_v2(x,
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
                    rhs_qb=None,
                    act_fn="silu",
                    rank=None,
                    mesh_ep_ranks=None,
                    sharded_plan=False,
                    scoring_fn="softmax",
                    score_bias=None,
                    routed_scaling_factor=1.0,
                    token_replica_groups=None):
    """Run one MoE layer through the fused expert-parallel kernel.

    x [tokens, hidden], w1 [experts, hidden, 2 * inter], w2 [experts,
    inter, hidden] and gating [tokens, experts], all sharded over the mesh
    axis. Returns [tokens, hidden], sharded like x.

    weight_format names one of the kernel's accepted weight forms and
    decides everything that follows from it. w1_scale and w2_scale are per
    output channel for the per-channel forms (fp8 e4m3, int8), [experts,
    2 * inter] and [experts, hidden], and per contraction block for fp4
    e2m1, [experts, blocks, 2 * inter] and [experts, blocks, hidden]; on an
    unquantized weight (bf16) there are no scales and both must be None.
    Both are taken in the kernel's own operand layout, so a table shaped
    for a loader rather than for the kernel is refused. The token rows are
    quantized to fp8 only where the format's matmuls take fp8 rows: an
    unquantized weight gets unquantized activations, with no quantize and
    dequantize pair inserted around a model that never asked for one.

    w1_bias [experts, 1, 2 * inter] and w2_bias [experts, 1, hidden] are
    optional and independent. The gate and up halves are added to the first
    matmul's post-scale accumulator before the activation; the down bias is
    added to the second matmul's post-scale row before it is quantized for
    the wire. Under expert parallelism the second matmul is sharded on the
    expert axis rather than on its contraction, so a routed row's whole down
    projection is produced on one shard and the bias enters it exactly once;
    the combine then weights it by that row's router weight, which is the
    per-selected-expert weighting the sum wants.

    block and ragged_stride default to the values this function derives;
    an explicitly passed one still has to divide tokens // ep * topk and to
    reach the no-drop bound respectively. act_fn selects the fused FFN
    activation from the kernel's ACT_FNS and is a build-time constant.

    sharded_plan computes the same kernel operands from the two complementary
    1/ep slices their consumers read, plus one small expert-count all-gather.
    It changes routing-plan construction only, never the core kernel.

    token_replica_groups lists ordered groups of device-mesh indices whose
    x and gating rows are identical (for example, a TP group). Each member
    routes and computes a disjoint slice, then gathers the combined outputs
    within its group to restore the input row order. Expert sharding stays
    unchanged. None means singleton groups with no token replication.

    mesh_ep_ranks names the EP rank at every device-mesh index. A non-identity
    order relabels only the selected expert ids after top-k, keeping the full
    router-logit tensor in its original order.
    """
    form = weight_form(weight_format)
    rhs_qb = QB4 if rhs_qb is None else int(rhs_qb)
    # The weights themselves, against the format they were named as. This
    # validated only that the SCALES matched, which was largely
    # self-limiting while fp8 and fp4 were the whole table: four-bit weights
    # stream as packed u32 words, so a wrong format name died on the
    # ref-level bitcast or on a shape mismatch soon after. int8 and bf16
    # declare IDENTICAL slab shapes and differ only in element size, one
    # byte against two, so naming one and passing the other is a half-slab
    # read with correct-looking shapes all the way down. This is the
    # documented public entry, so the check belongs where the operands come
    # in rather than only in the serving adapter above it.
    for name, w in (("w1", w1), ("w2", w2)):
        actual = weight_format_of_dtype(w.dtype)
        if actual != weight_format:
            raise ValueError(
                f"weight format {weight_format!r} takes "
                f"{jnp.dtype(form.weight_dtype).name} expert weights and "
                f"{name} is {jnp.dtype(w.dtype).name}"
                f"{f', which is the {actual!r} form' if actual else ''}")
    if form.has_scales != (w1_scale is not None):
        raise ValueError(f"weight format {weight_format!r} carries "
                         f"{form.scale_layout} weight scales and w1_scale is "
                         f"{'present' if w1_scale is not None else 'absent'}")
    if (w1_scale is None) != (w2_scale is None):
        raise ValueError(
            "both weight scales are supplied together or neither is")
    T, hidden = x.shape
    e_total = w1.shape[0]
    inter = w1.shape[2] // 2
    # Where the weight scales' layout is fixed: [E, N] per output channel,
    # [E, blocks, N] per contraction block, which is what the kernel's own
    # operands are. The scales are constants of the weights, so whatever
    # prepares the weights gives them that shape once; a reshape here would
    # instead stand between the parameter and the kernel call on every call,
    # and it is a layout change rather than free. A table carrying a
    # loader's singleton axes is refused rather than reshaped.
    if form.has_scales:
        ndim = 3 if form.scale_layout == "per_contraction_block" else 2
        for name, s, n in (("w1_scale", w1_scale, 2 * inter),
                           ("w2_scale", w2_scale, hidden)):
            if s.ndim != ndim or s.shape[0] != e_total or s.shape[-1] != n:
                want = (e_total, "blocks", n) if ndim == 3 else (e_total, n)
                raise ValueError(f"{name} layout {tuple(s.shape)} is not the "
                                 f"{form.scale_layout} form {want} the kernel "
                                 f"takes")
    (ax, ) = mesh.axis_names
    ep = mesh.shape[ax]
    if mesh_ep_ranks is not None:
        mesh_ep_ranks = tuple(int(rank) for rank in mesh_ep_ranks)
        if tuple(sorted(mesh_ep_ranks)) != tuple(range(ep)):
            raise ValueError("mesh_ep_ranks must be a permutation of "
                             f"range({ep}); got {mesh_ep_ranks}")
    g_local = e_total // ep
    if T % ep or gating.shape != (T, e_total):
        raise ValueError(
            "input and gating rows must agree and partition evenly over EP")
    token_layout = TokenReplicaLayout.create(ep=ep,
                                             input_rows=T // ep,
                                             groups=token_replica_groups,
                                             row_alignment=U32_SUBLANE_TILE)
    t_local = token_layout.local_rows
    T = ep * t_local
    P = jax.sharding.PartitionSpec
    stride_bound = ragged_stride_bound(T, topk, e_total, capacity)
    if ragged_stride is None:
        ragged_stride = stride_bound
    if block is None:
        block = routing_block(t_local, topk)
    if ragged_stride % capacity != 0:
        raise ValueError(
            f"ragged_stride must be a multiple of capacity {capacity}; got "
            f"{ragged_stride}. Pass {stride_bound}.")
    if ragged_stride < stride_bound:
        raise ValueError(
            f"ragged_stride {ragged_stride} is below the no-drop bound "
            f"{stride_bound}, so one shard's slab could bleed into the "
            f"next. Pass {stride_bound} or more.")
    has_w1_bias = w1_bias is not None
    has_w2_bias = w2_bias is not None
    has_score_bias = score_bias is not None
    if scoring_fn not in SCORING_FNS:
        raise ValueError(f"scoring_fn must be one of {SCORING_FNS}; got "
                         f"{scoring_fn!r}")
    if has_score_bias and score_bias.shape != (e_total, ):
        raise ValueError(
            f"score_bias must be one value per global expert, [{e_total}]; "
            f"got {tuple(score_bias.shape)}")
    kfn = build_fused_ep_moe_kernel(g_local=g_local,
                                    capacity=capacity,
                                    hidden=hidden,
                                    inter=inter,
                                    ep=ep,
                                    ragged_rows_alloc=ragged_stride,
                                    weight_format=weight_format,
                                    rhs_qb=rhs_qb,
                                    act_fn=act_fn,
                                    has_w1_bias=has_w1_bias,
                                    has_w2_bias=has_w2_bias)

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
        score_bias_l = next(operands) if has_score_bias else None
        # This shard's index arrives as data, not from `lax.axis_index`: that
        # lowers to `partition-id`, which the SPMD partitioner rejects when
        # torch_tpu recompiles this module through its jax.export bridge.
        me = rank_l[0, 0]
        x_l = token_layout.split(x_l, me)
        gate_l = token_layout.split(gate_l, me, fill_value=-jnp.inf)
        if len(token_layout.groups[0]) > 1 and form.quantized_activations:
            # Slicing fused into rowquant changes BF16 rounding on TPU.
            # Keep this boundary only for quantized replicated inputs.
            x_l = lax.optimization_barrier(x_l)
        rows_bf16 = x_l.astype(jnp.bfloat16)
        if form.quantized_activations:
            q_l, row_scale_l = rowquant_fp8(rows_bf16)
        else:
            # The rows go on the wire as they arrived. There is no scale to
            # carry, so nothing downstream builds or ships one.
            q_l, row_scale_l = rows_bf16, None
        q_g = lax.all_gather(q_l.reshape(t_local, hidden // HIDDEN_LANE_BLOCK,
                                         HIDDEN_LANE_BLOCK),
                             ax,
                             axis=0,
                             tiled=True)

        scores = _apply_scoring_fn(gate_l, scoring_fn)
        select_rows = MAX_ROUTING_BLOCK
        while t_local % select_rows:
            select_rows //= 2
        if score_bias_l is None:
            topk_weights, topk_idx = pallas_select(scores,
                                                   topk=topk,
                                                   block_rows=select_rows)
        else:
            # Selection is based on (scores + score_bias_l); the weights come
            # from the scores.
            topk_weights, topk_idx = pallas_select(
                scores + score_bias_l.astype(jnp.float32)[None, :],
                topk=topk,
                block_rows=select_rows,
                weight_scores=scores)
        topk_idx = _relabel_expert_ids_to_mesh_order(
            topk_idx, g_local=g_local, mesh_ep_ranks=mesh_ep_ranks)
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
            denom = jnp.maximum(topk_weights.sum(axis=-1, keepdims=True),
                                1e-20)
            topk_weights = topk_weights / jnp.where(row_routes, denom, 1.0)
        if routed_scaling_factor != 1.0:
            topk_weights = topk_weights * jnp.float32(routed_scaling_factor)
        topk_weights = jnp.where(row_routes, topk_weights, 0.0)
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
            t_local * topk // block, block)
        block_hist = pair_block_hist(local_blocks, e_total)
        scale_bits_l = None if row_scale_l is None else (
            lax.bitcast_convert_type(row_scale_l[:, 0].astype(jnp.float32),
                                     jnp.int32))
        layout = _exchange_layout(t_local, topk, e_total,
                                  form.quantized_activations)
        exchange = lax.all_gather(_pack_routing_exchange(
            local_blocks.reshape(-1), scale_bits_l, block_hist.sum(axis=0),
            layout),
                                  ax,
                                  axis=0,
                                  tiled=True).reshape(ep, layout[-1])
        pairs_g, scale_bits_g, rows_by_dest = _unpack_routing_exchange(
            exchange, layout, t_local, topk, e_total)
        gathered = GatheredPairs(expert_blocks=pairs_g.reshape(
            T * topk // block, block),
                                 local_blocks=local_blocks,
                                 block_hist=block_hist,
                                 rows_by_dest=rows_by_dest,
                                 n_tokens=T,
                                 topk=topk)

        plan_kw = dict(e_total=e_total,
                       ep=ep,
                       t_local=t_local,
                       block=block,
                       tile_m=capacity,
                       shard_stride=ragged_stride)
        if sharded_plan:
            routing = build_routing_tables_sharded(None,
                                                   me,
                                                   gathered=gathered,
                                                   **plan_kw)
        else:
            routing = build_routing_tables(pairs_g.reshape(T, topk), **plan_kw)
        block_tables = shard_transport_tables_in_blocks(routing,
                                                        me,
                                                        e_total=e_total,
                                                        ep=ep)
        row_tables = shard_push_tables_in_rows(routing,
                                               me,
                                               e_total=e_total,
                                               ep=ep)
        # Both slab tables scatter onto this shard's own slab. The rows of
        # the other shards are dropped where they are computed rather than
        # built into a replicated slab and sliced away afterwards.
        slab_row = local_slab_rows(routing, me, shard_stride=ragged_stride)
        # The two slab scatters and the activation all-gather are offloaded to
        # the SAME SparseCore. Ordering the scatters behind the collective
        # enqueues it 129 us earlier and costs them the cover they had; that
        # trade measures +2.80% at 2048 local tokens and negative at every
        # smaller shape, so it is taken only where the routing plan in front
        # of the collective is longer than the scatters behind it. Same bits
        # either way; only the queue moves.
        if t_local >= AGQ_MIN_LOCAL_TOKENS:
            slab_row, q_g = lax.optimization_barrier((slab_row, q_g))
        token_gather = shard_token_gather(routing,
                                          me,
                                          shard_stride=ragged_stride,
                                          rows=slab_row)
        # The kernel DMA-aligns each logical expert window down by as much as
        # 127 int32 rows and always copies a full fixed window. Pad after the
        # scatter (rather than enlarging its destination) so the off-shard
        # sentinel remains out of range and cannot write a live padding row.
        token_gather = jnp.pad(token_gather,
                               (0, token_gather_window_rows(capacity)))
        # The activation row scale, scattered onto the slab row each routed
        # pair computes on. It exists only where the rows were quantized.
        #
        # The slab is built flat and handed over in the dense lane-block
        # view, which is the same bytes in the same order and so costs
        # nothing. A column would cost a copy of the whole slab: [rows, 1]
        # is padded out to a full lane block on the way to the kernel, and
        # that copy is the largest single piece of glue above a thousand
        # rows. The kernel rebuilds the column a tile at a time instead.
        #
        # The scatter runs the length of the view rather than of the slab.
        # Those extra elements are past the slab's last row, so a pair the
        # rebase sent past the end lands on one of them instead of being
        # dropped; nothing reads them, and the slab's own rows are the same
        # either way.
        if form.quantized_activations:
            scale_bits = jnp.repeat(scale_bits_g, topk)
            scale_rows = act_scale_slab_rows(ragged_stride)
            scale_slab = lax.bitcast_convert_type(
                jnp.zeros((scale_rows * HIDDEN_LANE_BLOCK, ),
                          jnp.int32).at[slab_row].add(scale_bits, mode="drop"),
                jnp.float32).reshape(scale_rows, HIDDEN_LANE_BLOCK)
        else:
            scale_slab = None
        expert_rows, slab_base = shard_expert_slabs(routing,
                                                    me,
                                                    e_total=e_total,
                                                    ep=ep)
        recv_rows = align_up(t_local * topk + (ROWBLK - 1) * e_total, ROWBLK)
        # The bias tables are per expert and per output channel on every
        # weight format, so they take one layout: [G, N].
        bias_it = operands
        w1b_k = next(bias_it).reshape(g_local, 2 *
                                      inter) if has_w1_bias else None
        w2b_k = next(bias_it).reshape(g_local, hidden) if has_w2_bias else None
        # The empty experts drop out of the visit list, so an empty
        # expert's weight slab never streams. How MANY are visited is a
        # row of the count table rather than its own reduction, so this
        # builder's second value is not read; it leaves the program with
        # the rest of the dead code, and its own test still sees it.
        visit, _ = expert_visit_list(expert_rows, g_local)
        # One pass builds every count the kernel needs, still spread over
        # the expert-parallel axis; the kernel's scalar core closes them.
        counts = shard_count_vector(routing,
                                    expert_rows,
                                    me,
                                    e_total=e_total,
                                    ep=ep)

        # The kernel takes four of the seven transport tables and one of the
        # three push tables. The two totals are rows of `counts`, and three
        # more are the same array as one already here -- push_src is the
        # commit offset, push_len is the commit length, and the receive row
        # offset is the push destination in rows -- so each goes over once and
        # the kernel derives the rest. Every table shipped is another blocking
        # HBM->SMEM DMA at the head of the kernel, and the prefetch also has to
        # fit in 1 MB of SMEM.
        kernel_tables = (*block_tables[:3], block_tables[5], row_tables[0])
        arrivals, arrival_scales = kfn(kernel_tables,
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
                                       rank=rank_l)

        # The destination's own table: one arrival row per selection slot.
        pos = routing.pos if sharded_plan else lax.dynamic_slice(
            routing.arrival_row, (me * t_local, 0), (t_local, topk))
        mirror_pos = routing.mirror_pos if sharded_plan else \
            lax.dynamic_slice(routing.mirror_row, (me * t_local, 0),
                              (t_local, topk))
        combined = _combine_arrivals(arrivals, arrival_scales, pos, mirror_pos,
                                     topk_weights, x_l.dtype)
        return token_layout.restore(combined, ax)

    # The biases ride the same expert-axis sharding as the weights they
    # belong to, so no shard ever holds a bias for an expert it does not own.
    bias_args = tuple(b for b in (w1_bias, w2_bias) if b is not None)
    scale_args = (w1_scale, w2_scale) if form.has_scales else ()
    # `rank` rides the mesh axis so each shard reads its own index out of it.
    # A caller that is going through torch_tpu MUST pass it as a real operand:
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
    # expert set, so `P()` hands each shard the whole [e_total] vector.
    # Sharding it would give each shard a 1/ep slice of the experts and
    # silently bias the wrong ones.
    score_bias_args = (score_bias, ) if has_score_bias else ()
    in_specs = ((P(ax), ) * (5 + len(scale_args)) +
                (P(), ) * len(score_bias_args) + (P(ax), ) * len(bias_args))
    args = ((x, rank, w1, w2) + scale_args + (gating, ) + score_bias_args +
            bias_args)
    NS = jax.sharding.NamedSharding
    args = tuple(
        jax.device_put(a, NS(mesh, sp)) for a, sp in zip(args, in_specs))
    # local_fn closes over config-static values only -- the per-call data
    # are the shard_map arguments -- so this key is exact.
    key = (mesh, T, hidden, e_total, inter, topk, bool(renormalize), capacity,
           block, ragged_stride,
           weight_format, rhs_qb, act_fn, mesh_ep_ranks, token_layout,
           bool(sharded_plan), x.dtype, w1.dtype, w2.dtype, form.has_scales
           and w1_scale.dtype, gating.dtype, has_w1_bias
           and w1_bias.dtype, has_w2_bias
           and w2_bias.dtype, scoring_fn, has_score_bias and score_bias.dtype,
           float(routed_scaling_factor))
    sm = _LAYER_SM_CACHE.get(key)
    if sm is None:
        with _LAYER_SM_CACHE_LOCK:
            sm = _LAYER_SM_CACHE.get(key)
            if sm is None:
                sm = jax.shard_map(local_fn,
                                   mesh=mesh,
                                   in_specs=in_specs,
                                   out_specs=P(ax),
                                   check_vma=False)
                _LAYER_SM_CACHE[key] = sm
    return sm(*args)
