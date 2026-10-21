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
import jax.numpy as jnp
import pytest

from tokamax._src.ops.experimental.fused_moe import host, kernel

BIG = dict(g_local=8, capacity=128, hidden=7168, inter=2048)
SMALL = dict(g_local=32, capacity=128, hidden=512, inter=256)
SHAPES = [pytest.param(BIG, id="dsv4_pro"), pytest.param(SMALL, id="small")]

FORMATS = [
    pytest.param(host.WeightFormat.FP4, id="fp4"),
    pytest.param(host.WeightFormat.FP8, id="fp8"),
]


class _FakeChip:
    """The fields the VMEM accounting reads off a device record.

    The sublane tiling is the 32-byte rule the real chips report -- f32 tiles
    to 8 sublanes, bf16 to 16, eight-bit to 32 -- so the byte figures here are
    the shape of the real ones, not just self-consistent.
    """

    num_lanes = 128
    generation = host.MIN_GENERATION

    def __init__(self, vmem_capacity_bytes=128 * 2**20):
        self.vmem_capacity_bytes = vmem_capacity_bytes

    def get_sublane_tiling(self, dtype):
        return 32 // jnp.dtype(dtype).itemsize


@pytest.fixture
def chip():
    return _FakeChip()


def _estimate(shape, nbuf, chip, weight_format=host.WeightFormat.FP8, **kw):
    """One candidate's cost, at `fit_weight_slots`'s own defaults."""
    return host.vmem_estimate_bytes(
        **shape, nbuf=nbuf, weight_format=weight_format, info=chip, **kw
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("weight_format", FORMATS)
def test_scale_tables_are_slot_deep_not_expert_deep(shape, weight_format):
    """A scale table is staged with the slab it scales, so it is nbuf deep.

    While they were `g_local` deep they were fixed cost, and giving a weight
    slot back bought proportionally less -- on the four-bit path, where the
    block-scale tables are some of the largest buffers, it bought almost
    nothing.
    """
    nbuf = 2
    assert nbuf != shape["g_local"]
    arrays = dict(
        (name, s)
        for name, s, _ in host.vmem_scratch_arrays(
            **shape, nbuf=nbuf, weight_format=weight_format
        )
    )
    for name in ("w1_vm", "w2_vm", "w1s_vm", "w2s_vm"):
        assert arrays[name][0] == nbuf, f"{name} is not slot deep"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("weight_format", FORMATS)
def test_a_shallower_schedule_costs_strictly_less(shape, weight_format, chip):
    """What the descent trades on: fewer slots, fewer bytes, monotonically."""
    costs = [
        _estimate(shape, host.nbuf_for(d), chip, weight_format)
        for d in range(host.MAX_WEIGHT_PREFETCH_DISTANCE + 1)
    ]
    assert costs == sorted(costs)
    assert len(set(costs)) == len(costs)


@pytest.mark.parametrize("shape", SHAPES)
def test_room_at_the_ceiling_is_taken(shape, chip):
    """No descent when the full runway fits: the ceiling is the default."""
    ceiling = _estimate(shape, host.MAX_NBUF, chip)
    slots = host.fit_weight_slots(**shape, info=chip, limit=ceiling)
    assert slots.fits
    assert slots.distance == host.MAX_WEIGHT_PREFETCH_DISTANCE
    assert slots.nbuf == host.MAX_NBUF
    assert (slots.vmem_bytes, slots.limit) == (ceiling, ceiling)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("distance", range(host.MAX_WEIGHT_PREFETCH_DISTANCE + 1))
def test_the_descent_stops_at_the_first_depth_that_fits(shape, distance, chip):
    """One slot at a time, and it stops the moment it clears.

    The budget is a byte under what the next depth up costs, so the deepest
    schedule that fits is exactly `distance` -- a descent that skipped a rung,
    or dropped straight to zero, lands somewhere else.
    """
    want = _estimate(shape, host.nbuf_for(distance), chip)
    over = (
        _estimate(shape, host.nbuf_for(distance + 1), chip)
        if distance < host.MAX_WEIGHT_PREFETCH_DISTANCE
        else want + 1
    )

    slots = host.fit_weight_slots(**shape, info=chip, limit=over - 1)

    assert slots.fits
    assert slots.distance == distance
    assert slots.nbuf == host.nbuf_for(distance)
    assert slots.vmem_bytes == want


@pytest.mark.parametrize("shape", SHAPES)
def test_over_budget_at_distance_zero_is_reported_not_raised(shape, chip):
    """The builder owns the refusal, so the walk hands back the bottom rung.

    Its figures are the ones the build quotes by name: the cost with the
    runway already given up, which is what says no further shortening helps.
    """
    slots = host.fit_weight_slots(**shape, info=chip, limit=1)

    assert not slots.fits
    assert (slots.distance, slots.nbuf) == (0, 1)
    assert slots.vmem_bytes == _estimate(shape, 1, chip)
    assert slots.limit == 1


# tile_m, and a rows_alloc small enough that the ladder is buildable at all:
# the ladder is for shards whose routed experts fit inside one tile.
TILE_M = 512
ROWS_ALLOC = 1024
LADDER_ARGS = (TILE_M, ROWS_ALLOC, 8, 8)


@pytest.mark.parametrize(
    "override,want",
    [
        (None, tuple(range(128, 513, 128))),
        ("0", tuple(range(128, 513, 128))),
        ("2", (256, 512)),
        ("8", tuple(range(64, 513, 64))),
        ("1", (TILE_M,)),
        ("3", (TILE_M,)),
        ("512", (TILE_M,)),
        ("-1", (TILE_M,)),
    ],
)
def test_the_height_ladder_override(monkeypatch, override, want):
    """The override is a request the ladder still has to be able to honour.

    Unset and zero leave the built-in count. One is the ladder off, by the same
    path a count that cannot be built takes: a step that is not whole row
    blocks, or that does not divide the tile height into equal rungs, or a
    nonsense count. All of them fall back to the single full height rather than
    emitting a ladder whose rungs do not cover the tile.
    """
    monkeypatch.delenv("MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE", raising=False)
    if override is not None:
        monkeypatch.setenv("MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE", override)

    rungs = kernel.tile_height_rungs(*LADDER_ARGS)

    assert rungs == want
    # Whatever the count, the top rung is the full tile: a full tile's body is
    # never the short one.
    assert rungs[-1] == TILE_M
    assert all(step % host.ROWBLK == 0 for step in rungs)
