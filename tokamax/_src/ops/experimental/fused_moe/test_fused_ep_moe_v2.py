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
"""Device tests for the fused expert-parallel MoE kernel, in pure JAX.

``fused_ep_moe_v2`` is the whole MoE layer: it routes, quantizes, dispatches
the routed rows over the expert-parallel mesh axis, runs the expert FFN and
combines the arrivals back onto the tokens that asked for them. So the only
useful gate is the layer's output against a plain dense reference -- softmax
router, top-k, per-expert dense matmuls, weighted sum -- computed over the
same weights with no fp8 transport anywhere.

The reference and the tolerances are the ones the kernel was originally
validated against: a dense ``ref_moe``, and its two bands -- a batch-wide
relative L2 and a worst-single-token relative L2. The reference here is the
same arithmetic spread over the expert axis rather than replicated, which is
what makes the 512-expert served shape affordable to check at all.

The mesh axis carries the name the kernel body reads
(``vllm_torchtpu.kernels.fused_moe.v2.AXIS``, "d") rather than a name of this
test's choosing; ``test_the_mesh_axis_name_is_the_one_the_kernel_reads`` pins
that constraint, which the layer's own signature does not state.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tokamax._src.ops.experimental.fused_moe.host import WeightFormat
from tokamax._src.ops.experimental.fused_moe.kernel import AXIS
from tokamax._src.ops.experimental.fused_moe.layer import fused_ep_moe_v2

pytestmark = pytest.mark.multichip


@pytest.fixture(scope="module", autouse=True)
def _require_eight_devices():
    """Skip the module unless this host really has the eight-way slice.

    A fixture rather than a `skipif`, because a `skipif` is evaluated while
    pytest IMPORTS the module -- during collection, before markers deselect
    anything -- and `jax.device_count()` builds the PJRT client to answer.
    libtpu is process-exclusive, so a client built in the pytest parent locks
    every child out: `tests/conftest.py` puts vLLM workers in SPAWNED
    processes, and they all die with "pjrt client not initialized". Merely
    collecting this file was enough to take down every engine-starting test in
    the same pytest run, whether or not any test here was selected.

    A fixture runs at setup, after `-m` deselection, so on a host that is not
    the eight-chip agent nothing here touches the accelerator at all.
    """
    if jax.device_count() < 8:
        pytest.skip("the fused EP MoE kernel is an eight-way expert-parallel "
                    "collective and this host reports fewer than 8 JAX "
                    "devices")


# The expert-parallel width, and the tile height the serving adapter passes
# the kernel as its capacity (moe_fused_ep._TILE_M upstream).
EP = 8
CAPACITY = 128

FP8 = jnp.float8_e4m3fn

# The small structurally-complete shape: expert count divisible by the shard
# count, hidden a whole number of 128-lane blocks, intermediate over one.
SMALL = dict(hidden=512, inter=256, e_total=32, topk=4, tokens=256)

# Qwen3.5-397B's MoE block, which is also the served geometry the kernel was
# written for: 512 experts over 8 shards is 64 local experts a shard.
PROD = dict(hidden=4096, inter=1024, e_total=512, topk=10)
PROD_TOKENS = (2048, 16384)

# How far the layer may sit from the dense reference, as a batch-wide
# relative L2 and as the worst single token's. Taken from the kernel's own
# suite, where they are stated against an fp8 wire: the routed rows travel
# between shards quantized to e4m3 with one scale a row, and the intermediate
# is requantized the same way, so the band is the transport's and not the
# expert weights'. Measured on this host, as (batch, worst token): small fp8
# 0.0534/0.0636, T=2048 0.0540/0.0583, T=16384 0.0540/0.0615. The worst token
# clears the batch band on the small shape, which is what the second bound is
# for -- 256 tokens leave one unlucky row nowhere to hide.
FP8_RELATIVE_BOUND = 0.06
FP8_TOKEN_BOUND = 0.075

# The unquantized weight format quantizes nothing -- no fp8 rows, no fp8 wire
# -- so what is left is bf16 rounding in the matmuls and in the intermediate.
# Measured 0.00233/0.00261 on the small shape; the band is an order of
# magnitude under the fp8 one, which is the point of stating it separately.
BF16_RELATIVE_BOUND = 0.01
BF16_TOKEN_BOUND = 0.01

# How much further a reference computed over rotated experts has to sit than
# the reference itself, so that a band this wide still means something: a
# layer that routed to the wrong experts would not clear it. The kernel's own
# suite uses the same control and the same factor.
ROTATED_CONTROL_FACTOR = 10


def _mesh():
    """The single-axis expert-parallel mesh, over the first eight devices."""
    return Mesh(np.asarray(jax.devices()[:EP]), axis_names=(AXIS, ))


def _make_weights(mesh, seed, *, hidden, inter, e_total, weight_format):
    """(w1, w2, w1_scale, w2_scale) as global arrays sharded on the experts.

    Built a shard at a time rather than on one device and handed out: the
    served shape's f32 draw is 17GB where its fp8 form is 6.4GB, and only the
    latter has to exist at once. fp8 scales are per output channel, taken
    along each matmul's contraction axis, which is the layout the kernel
    takes: w1_scale [E, 2 * inter], w2_scale [E, hidden].
    """
    (axis, ) = mesh.axis_names
    g_local = e_total // mesh.shape[axis]
    quantized = weight_format == WeightFormat.FP8

    def local(_):
        me = lax.axis_index(axis)
        k1, k2 = jax.random.split(jax.random.fold_in(jax.random.key(seed), me))
        weights = []
        for key, shape in ((k1, (g_local, hidden, 2 * inter)),
                           (k2, (g_local, inter, hidden))):
            w = jax.random.normal(key, shape, jnp.float32) / 10
            if not quantized:
                weights.append((w.astype(jnp.bfloat16), None))
                continue
            amax = jnp.max(jnp.abs(w), axis=1, keepdims=True)
            scale = jnp.where(amax == 0, 1.0, amax / float(jnp.finfo(FP8).max))
            weights.append(
                ((w / scale).astype(FP8),
                 scale.astype(jnp.float32).reshape(shape[0], shape[2])))
        (w1, w1s), (w2, w2s) = weights
        return (w1, w2) if not quantized else (w1, w2, w1s, w2s)

    out_specs = (P(axis), ) * (4 if quantized else 2)
    built = jax.jit(
        jax.shard_map(local,
                      mesh=mesh,
                      in_specs=(P(axis), ),
                      out_specs=out_specs,
                      check_vma=False))(jnp.zeros((EP, ), jnp.float32))
    return built if quantized else (built[0], built[1], None, None)


def _make_inputs(mesh, seed, *, tokens, hidden, e_total):
    """(x, gating) as global arrays sharded on the tokens."""
    (axis, ) = mesh.axis_names
    shard = NamedSharding(mesh, P(axis))

    @jax.jit
    def build():
        kx, kg = jax.random.split(jax.random.key(seed))
        x = (jax.random.normal(kx, (tokens, hidden), jnp.float32) / 10).astype(
            jnp.bfloat16)
        gating = jax.random.normal(kg, (tokens, e_total), jnp.float32)
        return (lax.with_sharding_constraint(x, shard),
                lax.with_sharding_constraint(gating, shard))

    return build()


def _dense_reference(mesh, x, w1, w2, w1_scale, w2_scale, gating, *, topk):
    """The plain MoE: softmax, top-k, dense per-expert FFN, weighted sum.

    Every expert is applied to every token in f32 and weighted by the router
    weight of the slots that chose it, which is zero for a token that did
    not. Nothing is quantized: the weights are dequantized once and the whole
    body runs in f32, so the difference from the kernel is the kernel's
    transport rather than a second implementation of the same shortcuts.

    The loop is spread over the same expert axis the layer shards on -- each
    shard applies its own experts to the whole batch and the results are
    summed across shards -- which is a distribution of the reference, not a
    change of it: the summands and their order are the same.
    """
    (axis, ) = mesh.axis_names
    g_local = w1.shape[0] // mesh.shape[axis]
    inter = w1.shape[2] // 2
    scaled = w1_scale is not None

    def local(x_g, gating_g, w1_l, w2_l, *scales):
        s1, s2 = scales if scaled else (None, None)
        scores = jax.nn.softmax(gating_g.astype(jnp.float32), axis=-1)
        topk_weights, topk_idx = lax.top_k(scores, topk)
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
        first = lax.axis_index(axis) * g_local
        x32 = x_g.astype(jnp.float32)

        def expert(e, acc):
            w1e = lax.dynamic_index_in_dim(w1_l, e, 0,
                                           keepdims=False).astype(jnp.float32)
            w2e = lax.dynamic_index_in_dim(w2_l, e, 0,
                                           keepdims=False).astype(jnp.float32)
            if scaled:
                w1e = w1e * lax.dynamic_index_in_dim(s1, e, 0, keepdims=True)
                w2e = w2e * lax.dynamic_index_in_dim(s2, e, 0, keepdims=True)
            weight = jnp.sum(jnp.where(topk_idx == first + e, topk_weights,
                                       0.0),
                             axis=-1)[:, None]
            acc1 = x32 @ w1e
            row = (jax.nn.silu(acc1[:, :inter]) * acc1[:, inter:]) @ w2e
            return acc + weight * row

        mine = lax.fori_loop(0, g_local, expert, jnp.zeros_like(x32))
        return lax.psum(mine, axis)

    in_specs = (P(), P(), P(axis), P(axis)) + ((P(axis), ) * 2 if scaled else
                                               ())
    args = (x, gating, w1, w2) + ((w1_scale, w2_scale) if scaled else ())
    return jax.jit(
        jax.shard_map(local,
                      mesh=mesh,
                      in_specs=in_specs,
                      out_specs=P(),
                      check_vma=False))(*args)


def _relative_l2(actual, want):
    a = np.asarray(actual, np.float64)
    w = np.asarray(want, np.float64)
    return float(np.linalg.norm(a - w) / np.linalg.norm(w))


def _worst_token_relative_l2(actual, want):
    """The largest per-token relative error. A routing failure is per token,
    and a batch-wide norm divides one bad row by every other row's size."""
    a = np.asarray(actual, np.float64)
    w = np.asarray(want, np.float64)
    per_token = np.linalg.norm(a - w, axis=-1)
    scale = np.linalg.norm(w, axis=-1)
    return float(np.max(per_token / np.where(scale == 0, 1.0, scale)))


def _check_against_the_reference(mesh, x, w1, w2, w1_scale, w2_scale, gating,
                                 *, topk, batch_bound, token_bound):
    """Run the layer, hold it to the dense reference, and rotate the experts.

    The rotation is the control the bands need: both are wide enough to
    cover an fp8 wire, so without it a layer that sent every token to the
    wrong expert could still pass one of them.
    """
    out = fused_ep_moe_v2(x,
                          w1,
                          w2,
                          w1_scale,
                          w2_scale,
                          gating,
                          topk=topk,
                          renormalize=True,
                          mesh=mesh,
                          capacity=CAPACITY,
                          weight_format=(WeightFormat.FP8 if w1.dtype == FP8
                                         else WeightFormat.BF16))
    # [tokens, hidden] sharded the way x is: the combine leaves each token's
    # row on the shard that owns the token, so nothing has to be gathered.
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.sharding.spec == x.sharding.spec

    want = _dense_reference(mesh,
                            x,
                            w1,
                            w2,
                            w1_scale,
                            w2_scale,
                            gating,
                            topk=topk)
    error = _relative_l2(out, want)
    assert error < batch_bound, (
        f"relative L2 {error:.4f} past the {batch_bound} band")
    worst = _worst_token_relative_l2(out, want)
    assert worst < token_bound, (
        f"worst token's relative error {worst:.4f} past the {token_bound} "
        f"per-token band; the batch norm was {error:.4f}, so this is a few "
        f"tokens rather than the whole batch")

    def roll(a):
        return None if a is None else jnp.roll(a, 1, axis=0)

    rotated = _dense_reference(mesh,
                               x,
                               roll(w1),
                               roll(w2),
                               roll(w1_scale),
                               roll(w2_scale),
                               gating,
                               topk=topk)
    assert _relative_l2(out, rotated) > ROTATED_CONTROL_FACTOR * error
    return error, worst


@pytest.mark.parametrize("weight_format",
                         [WeightFormat.FP8, WeightFormat.BF16])
def test_small_shape_tracks_a_dense_reference(weight_format):
    """The layer end to end at a small but structurally complete shape."""
    mesh = _mesh()
    w1, w2, w1_scale, w2_scale = _make_weights(mesh,
                                               0,
                                               hidden=SMALL["hidden"],
                                               inter=SMALL["inter"],
                                               e_total=SMALL["e_total"],
                                               weight_format=weight_format)
    x, gating = _make_inputs(mesh,
                             1,
                             tokens=SMALL["tokens"],
                             hidden=SMALL["hidden"],
                             e_total=SMALL["e_total"])
    quantized = weight_format == WeightFormat.FP8
    _check_against_the_reference(
        mesh,
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        gating,
        topk=SMALL["topk"],
        batch_bound=FP8_RELATIVE_BOUND if quantized else BF16_RELATIVE_BOUND,
        token_bound=FP8_TOKEN_BOUND if quantized else BF16_TOKEN_BOUND)


def test_sharded_routing_plan_is_bit_exact_with_the_replicated_plan():
    """Plan decomposition changes only how the kernel operands are built."""
    mesh = _mesh()
    w1, w2, w1_scale, w2_scale = _make_weights(mesh,
                                               20,
                                               hidden=SMALL["hidden"],
                                               inter=SMALL["inter"],
                                               e_total=SMALL["e_total"],
                                               weight_format=WeightFormat.FP8)
    x, gating = _make_inputs(mesh,
                             21,
                             tokens=SMALL["tokens"],
                             hidden=SMALL["hidden"],
                             e_total=SMALL["e_total"])
    common = dict(topk=SMALL["topk"],
                  renormalize=True,
                  mesh=mesh,
                  capacity=CAPACITY,
                  weight_format=WeightFormat.FP8)
    replicated = fused_ep_moe_v2(x,
                                 w1,
                                 w2,
                                 w1_scale,
                                 w2_scale,
                                 gating,
                                 sharded_plan=False,
                                 **common)
    sharded = fused_ep_moe_v2(x,
                              w1,
                              w2,
                              w1_scale,
                              w2_scale,
                              gating,
                              sharded_plan=True,
                              **common)
    np.testing.assert_array_equal(np.asarray(sharded), np.asarray(replicated))


def test_nonidentity_mesh_order_relabels_after_topk_bit_exactly():
    """Physical expert placement may follow device order, not EP-rank order.

    Keep the logical router logits untouched, place each logical expert block
    on the mesh index that owns its EP rank, and relabel only selected IDs. The
    result must match the identity placement bit for bit with the sharded plan
    enabled, which is the served configuration.
    """
    mesh = _mesh()
    w1, w2, w1_scale, w2_scale = _make_weights(mesh,
                                               22,
                                               hidden=SMALL["hidden"],
                                               inter=SMALL["inter"],
                                               e_total=SMALL["e_total"],
                                               weight_format=WeightFormat.FP8)
    x, gating = _make_inputs(mesh,
                             23,
                             tokens=SMALL["tokens"],
                             hidden=SMALL["hidden"],
                             e_total=SMALL["e_total"])
    order = (0, 1, 6, 7, 2, 3, 4, 5)
    g_local = SMALL["e_total"] // EP
    expert_sharding = NamedSharding(mesh, P(AXIS))

    def place_in_mesh_order(a):
        host = np.asarray(a)
        by_rank = host.reshape((EP, g_local) + host.shape[1:])
        placed = by_rank[np.asarray(order)].reshape(host.shape)
        return jax.device_put(placed, expert_sharding)

    common = dict(topk=SMALL["topk"],
                  renormalize=True,
                  mesh=mesh,
                  capacity=CAPACITY,
                  weight_format=WeightFormat.FP8,
                  sharded_plan=True)
    identity = fused_ep_moe_v2(x, w1, w2, w1_scale, w2_scale, gating, **common)
    placed = tuple(
        place_in_mesh_order(a) for a in (w1, w2, w1_scale, w2_scale))
    remapped = fused_ep_moe_v2(x,
                               *placed,
                               gating,  # pyrefly: ignore[bad-argument-count]
                               mesh_ep_ranks=order,
                               **common)
    np.testing.assert_array_equal(np.asarray(remapped), np.asarray(identity))


@pytest.fixture(scope="module")
def production_weights():
    """The Qwen3.5-397B MoE weights, fp8 e4m3 with per-channel scales.

    6.4GB over the eight devices, so they are built once and both batch
    shapes below run against them.
    """
    mesh = _mesh()
    weights = _make_weights(mesh,
                            7,
                            hidden=PROD["hidden"],
                            inter=PROD["inter"],
                            e_total=PROD["e_total"],
                            weight_format=WeightFormat.FP8)
    jax.block_until_ready(weights)
    yield weights
    del weights


@pytest.mark.parametrize("tokens", PROD_TOKENS)
def test_production_shape_tracks_a_dense_reference(production_weights, tokens):
    """Qwen3.5-397B's MoE block at the decode and prefill batch shapes."""
    mesh = _mesh()
    w1, w2, w1_scale, w2_scale = production_weights
    x, gating = _make_inputs(mesh,
                             2,
                             tokens=tokens,
                             hidden=PROD["hidden"],
                             e_total=PROD["e_total"])
    _check_against_the_reference(mesh,
                                 x,
                                 w1,
                                 w2,
                                 w1_scale,
                                 w2_scale,
                                 gating,
                                 topk=PROD["topk"],
                                 batch_bound=FP8_RELATIVE_BOUND,
                                 token_bound=FP8_TOKEN_BOUND)


def test_any_mesh_axis_name_works():
    """The kernel no longer cares what the mesh axis is called.

    Upstream reads this shard's index with ``lax.axis_index(kernel.AXIS)``,
    with AXIS a module constant, so a mesh named anything else traced the whole
    kernel and then died on "unbound axis name: d". vllm-torchtpu passes the
    index in as an i32 operand instead -- it has to, because ``lax.axis_index``
    lowers to ``partition-id`` and XLA rejects that when torch_tpu recompiles
    the exported module. The layer's own collectives take their axis from the
    mesh it is handed, so nothing is left that hardcodes a name. Held here so
    the freedom is stated, and so a revert to axis_index fails loudly.
    """
    mesh = Mesh(np.asarray(jax.devices()[:EP]), axis_names=("ep", ))
    w1, w2, w1_scale, w2_scale = _make_weights(mesh,
                                               0,
                                               hidden=SMALL["hidden"],
                                               inter=SMALL["inter"],
                                               e_total=SMALL["e_total"],
                                               weight_format=WeightFormat.FP8)
    x, gating = _make_inputs(mesh,
                             1,
                             tokens=SMALL["tokens"],
                             hidden=SMALL["hidden"],
                             e_total=SMALL["e_total"])
    _check_against_the_reference(mesh,
                                 x,
                                 w1,
                                 w2,
                                 w1_scale,
                                 w2_scale,
                                 gating,
                                 topk=SMALL["topk"],
                                 batch_bound=FP8_RELATIVE_BOUND,
                                 token_bound=FP8_TOKEN_BOUND)


def test_fp4_block512_tracks_dense_reference_and_routing_plan():
    """FP4 block scales and FP8 transport, including a single-block W2."""
    mesh = _mesh()
    shard = NamedSharding(mesh, P(AXIS))
    rng = np.random.default_rng(42)
    weights, scales, decoded = [], [], []
    for shape in ((32, 1024, 1024), (32, 512, 1024)):
        raw = rng.normal(0, 0.1, shape).astype(np.float32)
        grouped = raw.reshape(shape[0], shape[1] // 512, 512, shape[2])
        scale = np.maximum(np.max(np.abs(grouped), axis=2) / 6, 1e-12)
        normalized = (grouped / scale[:, :, None, :]).reshape(shape)
        w = jax.device_put(normalized, shard).astype(jnp.float4_e2m1fn)
        s = jax.device_put(scale, shard)
        dequant = (w.astype(jnp.float32).reshape(grouped.shape) *
                   s[:, :, None, :]).reshape(shape)
        weights.append(w)
        scales.append(s)
        decoded.append(dequant)
    x, gating = _make_inputs(mesh, 43, tokens=256, hidden=1024, e_total=32)
    common = dict(topk=4,
                  renormalize=True,
                  mesh=mesh,
                  capacity=CAPACITY,
                  weight_format=WeightFormat.FP4,
                  rhs_qb=512)
    out = fused_ep_moe_v2(x,
                          *weights,
                          *scales,
                          gating,  # pyrefly: ignore[bad-argument-count]
                          sharded_plan=True,
                          **common)
    replicated = fused_ep_moe_v2(x,
                                 *weights,
                                 *scales,
                                 gating,  # pyrefly: ignore[bad-argument-count]
                                 sharded_plan=False,
                                 **common)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(replicated))
    ref = _dense_reference(mesh, x, *decoded, None, None, gating, topk=4)  # pyrefly: ignore[bad-argument-count]
    error = _relative_l2(out, ref)
    worst = _worst_token_relative_l2(out, ref)
    print(f"FP4 block512 relative_l2={error} worst_token={worst}")
    assert error < FP8_RELATIVE_BOUND
    assert worst < FP8_TOKEN_BOUND
    wrong = _dense_reference(mesh,
                             x,
                             jnp.roll(decoded[0], 1, axis=0),
                             jnp.roll(decoded[1], 1, axis=0),
                             None,
                             None,
                             gating,
                             topk=4)
    assert _relative_l2(out, wrong) > ROTATED_CONTROL_FACTOR * error


if __name__ == "__main__":
    import sys
    from absl import app
    app.run(lambda argv: sys.exit(pytest.main([__file__] + argv[1:])))
