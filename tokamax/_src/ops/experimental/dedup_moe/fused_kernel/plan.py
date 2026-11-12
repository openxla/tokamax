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


"""VMEM estimate and weight-slot schedule of the fused expert-parallel kernel.

The chip-record helpers (`vmem_limit`, `chip_generation`, `array_vmem_bytes`)
and the weight-slot schedule types (`nbuf_for`, `WeightSlots`) are shared with
`fused_moe.host`. The scratch list and the VMEM estimate are this kernel's own:
its wave dispatch and combine change which buffers the kernel declares.
"""

from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.dedup_moe.common import arg
from tokamax._src.ops.experimental.fused_moe import host as fused_host

HIDDEN_LANE_BLOCK = fused_host.HIDDEN_LANE_BLOCK
OUT_PARITIES = fused_host.OUT_PARITIES
ROWBLK = fused_host.ROWBLK
act_scale_window_rows = fused_host.act_scale_window_rows
row_lane_blocks = fused_host.row_lane_blocks

MAX_WEIGHT_PREFETCH_DISTANCE = fused_host.MAX_WEIGHT_PREFETCH_DISTANCE
nbuf_for = fused_host.nbuf_for
MAX_NBUF = fused_host.MAX_NBUF
VMEM_FRACTION = fused_host.VMEM_FRACTION
vmem_limit = fused_host.vmem_limit
MIN_GENERATION = fused_host.MIN_GENERATION
chip_generation = fused_host.chip_generation
array_vmem_bytes = fused_host.array_vmem_bytes
WeightSlots = fused_host.WeightSlots

# How many tile heights the FFN body is emitted at. A tile runs the smallest
# rung that covers its live rows, so an expert's short tail tile computes
# tile_m / 2 rows instead of a full tile_m of rows nothing will commit. Every
# rung is another copy of the body in the program text (instruction-memory
# overlay swaps measured against 4 rungs), and the top rung is always tile_m,
# so a full tile is unchanged.
TAIL_HEIGHT_RUNGS = 2


def tail_height_rungs(tile_m):
  """The ascending static tile heights a build emits, ending at tile_m.

  TAIL_HEIGHT_RUNGS equally spaced heights, so the tail tile of every
  expert computes only the smallest one covering its live rows (with 2:
  tile_m / 2 or tile_m rows). A tile height must be a whole number of
  ROWBLK -- it is the unit every commit offset is in -- so a tile_m that
  does not divide into whole ROWBLK steps keeps the single full height.
  Same bits either way: rows past the live ones are never committed.

  Args:
    tile_m: Tile height in rows.

  Returns:
    Ascending tile heights ending at `tile_m`.
  """
  rungs = TAIL_HEIGHT_RUNGS
  step = tile_m // rungs
  if step < 1 or step % ROWBLK or step * rungs != tile_m:
    return (tile_m,)
  return tuple(range(step, tile_m + 1, step))


def vmem_scratch_arrays(
    g_local,
    capacity,
    hidden,
    inter,
    *,
    nbuf,
    weight_format=arg.WeightFormat.FP8,
    has_w1_bias=False,
    has_w2_bias=False,
):
  """The kernel's VMEM scratch buffers, as (name, shape, dtype).

  _build_fused_ep_moe_kernel declares its scratch from this list, in this
  order, and the VMEM accounting below sums this same list, so a buffer
  cannot be resized on one side only.

  Every element type here comes off the weight format's record: the
  weight slabs, the row staging the activations arrive in, the wire
  buffer results leave in, and whether the scale tables and the
  activation row scale exist at all.

  The staging buffers are three-dimensional on purpose. An eight-bit
  array's second-minor dimension tiles to 32 sublanes, so a flat
  [rows, hidden] buffer could only be sliced at row offsets that are
  multiples of 32; splitting a token row across the two minor dimensions
  leaves the row axis untiled, which is what makes an 8-row transport
  offset legal. It does that for every element type, which is why a
  sixteen-bit wire takes the same geometry at twice the bytes.

  Args:
    g_local: Experts per shard.
    capacity: Tile height in rows.
    hidden: Model width.
    inter: Expert FFN width.
    nbuf: Weight slots.
    weight_format: fp8 or bf16.
    has_w1_bias: Whether the gate/up bias is resident.
    has_w2_bias: Whether the down bias is resident.

  Returns:
    A list of (name, shape, dtype).
  """
  form = arg.weight_form(weight_format)
  lane_blocks = row_lane_blocks(hidden)
  # The tile height IS the capacity, and the out pair is in tile units.
  tile_m = capacity
  # A scale table rides the weight slots, not the expert list: it is read
  # only alongside the slab it scales, so it is staged with that slab and
  # is nbuf deep.
  weights = [
      ("w1_vm", (nbuf, hidden, 2 * inter), form.weight_dtype),
      ("w2_vm", (nbuf, inter, hidden), form.weight_dtype),
  ]
  if form.has_scales:
    weights += [
        ("w1s_vm", (nbuf, 2 * inter), jnp.float32),
        ("w2s_vm", (nbuf, hidden), jnp.float32),
    ]
  # The optional expert biases are resident for every local expert, one
  # row per expert on each matmul's output channels. They carry no block
  # dimension, so they take the same shape on every weight format -- the
  # shape and the cost of the per-channel scale tables beside them.
  biases = []
  if has_w1_bias:
    biases.append(("w1b_vm", (g_local, 2 * inter), jnp.float32))
  if has_w2_bias:
    biases.append(("w2b_vm", (g_local, hidden), jnp.float32))
  # The activation row scale exists where the rows were quantized.
  # It rides per tile, so it is sized by the OUT-PARITY count: a tile's
  # stream is issued into, waited on and read at the parity its results
  # will be staged in, and never at any other index. nbuf is the depth the
  # WEIGHT slabs need so a refill and the live readers of the prefetch
  # distance's worth of earlier experts occupy distinct slots, which is a
  # different question and a larger answer.
  act_scale = []
  if form.quantized_activations:
    act_scale.append((
        "ls_vm",
        (OUT_PARITIES, act_scale_window_rows(capacity), HIDDEN_LANE_BLOCK),
        jnp.float32,
    ))
  return [
      # The indirect form keeps each token row as one lane-block row.
      # Parity-deep for the same reason ls_vm is.
      (
          "lhs_vm",
          (OUT_PARITIES, capacity, lane_blocks, HIDDEN_LANE_BLOCK),
          form.act_dtype,
      ),
      *weights,
      *biases,
      *act_scale,
      (
          "out_vm",
          (OUT_PARITIES, tile_m, lane_blocks, HIDDEN_LANE_BLOCK),
          form.wire_dtype,
      ),
      ("oscl_vm", (OUT_PARITIES, 1, capacity), jnp.float32),
  ]


def vmem_tile_body_arrays(
    capacity, hidden, inter, *, weight_format=arg.WeightFormat.FP8
):
  """What one tile body keeps live, as (name, shape, dtype).

  These are not declared scratch: the compiler places them, in the same
  memory as the buffers above. In order, the first matmul's accumulator,
  the bf16 intermediate the activation chunk loop builds up before it is
  concatenated, that concatenation -- as an fp8 requantization where the
  format has one and as a second bf16 array where it does not -- the
  second matmul's accumulator, its bf16 result, and the wire row that
  result is quantized into where the wire is eight-bit.

  The row staging a tile reads is a view of lhs_vm, not a second array,
  so it is not counted again here.

  Args:
    capacity: Tile height in rows.
    hidden: Model width.
    inter: Expert FFN width.
    weight_format: fp8 or bf16.

  Returns:
    A list of (name, shape, dtype).
  """
  form = arg.weight_form(weight_format)
  tile_m = capacity
  arrays = [
      ("acc1", (tile_m, 2 * inter), jnp.float32),
      ("mid_chunks", (tile_m, inter), jnp.bfloat16),
  ]
  # The second matmul takes the intermediate quantized only where its
  # weights are. The other formats contract the bf16 rows directly -- but
  # they still materialize a second copy: the non-requantizing path
  # concatenates the chunk list into one [tile_m, inter] bf16 array while
  # the chunks that feed it are still live. The requantizing path does not,
  # because it quantizes straight out of the chunk list into mid_q.
  if form.quantized_activations:
    arrays.append(("mid_q", (tile_m, inter), arg.FP8))
  else:
    arrays.append(("mid_concat", (tile_m, inter), jnp.bfloat16))
  arrays += [
      ("acc2", (tile_m, hidden), jnp.float32),
      ("down_bf16", (tile_m, hidden), jnp.bfloat16),
  ]
  # An eight-bit wire quantizes the down projection into a fresh array
  # while down_bf16, the bf16 source it reduces over, is still live. A
  # sixteen-bit wire ships down_bf16 itself and materializes nothing.
  if form.wire_dtype is arg.FP8:
    arrays.append(("wire_rows", (tile_m, hidden), arg.FP8))
  return arrays


def vmem_estimate_bytes(
    g_local,
    capacity,
    hidden,
    inter,
    nbuf,
    weight_format=arg.WeightFormat.FP8,
    has_w1_bias=False,
    has_w2_bias=False,
    info=None,
):
  """VMEM one built kernel occupies, from the arrays it declares.

  An upper bound, and deliberately so: the declared scratch is live for
  the whole call, while the tile body's values are counted as though all
  of them were live at once even though several die before the next is
  born. Nothing is left out, which is what a caller asking "will this
  fit" needs; a caller wanting the exact high-water mark should read the
  figure the compiler reports for a built kernel.

  Reads the chip's lane count and per-dtype sublane tiling off the device
  record, so a host with no chip to name raises rather than answering.

  Args:
    g_local: Experts per shard.
    capacity: Tile height in rows.
    hidden: Model width.
    inter: Expert FFN width.
    nbuf: Weight slots.
    weight_format: fp8 or bf16.
    has_w1_bias: Whether the gate/up bias is resident.
    has_w2_bias: Whether the down bias is resident.
    info: Device record, or None to query it.

  Returns:
    An upper bound on the VMEM bytes.
  """
  if info is None:
    info = pltpu.get_tpu_info()
  arrays = vmem_scratch_arrays(
      g_local,
      capacity,
      hidden,
      inter,
      nbuf=nbuf,
      weight_format=weight_format,
      has_w1_bias=has_w1_bias,
      has_w2_bias=has_w2_bias,
  )
  arrays += vmem_tile_body_arrays(
      capacity, hidden, inter, weight_format=weight_format
  )
  return sum(array_vmem_bytes(shape, dtype, info) for _, shape, dtype in arrays)


def fit_weight_slots(
    g_local,
    capacity,
    hidden,
    inter,
    *,
    max_distance=MAX_WEIGHT_PREFETCH_DISTANCE,
    weight_format=arg.WeightFormat.FP8,
    has_w1_bias=False,
    has_w2_bias=False,
    info=None,
    limit=None,
):
  """The deepest prefetch schedule this shape's VMEM leaves room for.

  The weight slabs are the only buffer sized by the schedule rather than
  by the model: they are `nbuf` copies of one expert, and `nbuf` is ours
  to pick. So a shape that misses at the ceiling often fits one slot
  shallower. Walk down a slot at a time rather than dropping to zero --
  even distance one keeps a refill in flight, which is most of the win.

  Distance zero has nothing left to give back. A shape still over budget
  there is returned NOT FITTING rather than raised on, carrying the
  distance-zero figures for the builder to quote in its refusal.

  Queries the device record once and reuses it for every candidate.

  Args:
    g_local: Experts per shard.
    capacity: Tile height in rows.
    hidden: Model width.
    inter: Expert FFN width.
    max_distance: Prefetch distance to start from.
    weight_format: fp8 or bf16.
    has_w1_bias: Whether the gate/up bias is resident.
    has_w2_bias: Whether the down bias is resident.
    info: Device record, or None to query it.
    limit: VMEM budget in bytes, or None for vmem_limit(info).

  Returns:
    The WeightSlots of the deepest fitting distance, or the distance-zero
    WeightSlots if none fits.

  Raises:
    ValueError: If `max_distance` is negative.
  """
  if max_distance < 0:
    raise ValueError(
        f"the maximum prefetch distance is {max_distance}; a negative "
        "distance is not a schedule, and zero already means no "
        "prefetching"
    )
  if info is None:
    info = pltpu.get_tpu_info()
  if limit is None:
    limit = vmem_limit(info)
  slots = None
  for distance in range(max_distance, -1, -1):
    nbuf = nbuf_for(distance)
    slots = WeightSlots(
        distance,
        nbuf,
        vmem_estimate_bytes(
            g_local,
            capacity,
            hidden,
            inter,
            nbuf=nbuf,
            weight_format=weight_format,
            has_w1_bias=has_w1_bias,
            has_w2_bias=has_w2_bias,
            info=info,
        ),
        limit,
    )
    if slots.fits:
      break
  # The first depth that fits, or distance zero over budget for the
  # caller to refuse by name.
  return slots
