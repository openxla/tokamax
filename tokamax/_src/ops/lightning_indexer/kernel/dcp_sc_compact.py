# Copyright 2026 Google LLC. All Rights Reserved.
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
"""Left-pack the merged DCP top-k winners that one rank owns, on SparseCore.

The winners are rank-local already: only scores cross the all-gather, so the
indices never leave the rank that produced them.
"""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp

SCAN_UNROLL = 8


def _pack_body(
    vals_hbm,  # i32[padded_b * width]
    out_hbm,  # i32[padded_b * width]
    rowbuf_vmem,  # i32[width + lanes]
    *,
    b: int,
    width: int,
    num_waves: int,
    lanes: int,
    num_cores: int,
    num_subcores: int,
):
  flat = lax.axis_index("core") * num_subcores + lax.axis_index("subcore")
  chunks = width // lanes

  def pipeline_step(vals_vmem, out_vmem):
    row = pl.program_id(0) * (num_cores * num_subcores) + flat

    def fill(i):
      rowbuf_vmem[pl.ds(i * lanes, lanes)] = jnp.full((lanes,), -1, jnp.int32)

    # Pad up front; the scan below only overwrites the kept prefix.
    plsc.parallel_loop(0, chunks, unroll=4)(fill)

    def emit(i, woff):
      v = vals_vmem[pl.ds(i * lanes, lanes)]
      keep = v >= 0
      plsc.store_compressed(rowbuf_vmem.at[pl.ds(woff, lanes)], v, mask=keep)
      return woff + plsc.all_reduce_population_count(keep)[0]

    # Carried `woff` makes this sequential: ascending order, and `woff <= width`.
    plsc.parallel_loop(
        0, jnp.where(row < b, chunks, 0), unroll=SCAN_UNROLL, carry=jnp.int32(0)
    )(emit)

    def copy_out(i):
      out_vmem[pl.ds(i * lanes, lanes)] = rowbuf_vmem[pl.ds(i * lanes, lanes)]

    plsc.parallel_loop(0, chunks, unroll=4)(copy_out)

  cores_per_chip = num_cores * num_subcores
  buf_count = 1 if num_waves == 1 else 2
  in_specs = (
      pl.BlockSpec(
          (width,),
          lambda w: (w * cores_per_chip + flat,),
          pipeline_mode=pl.Buffered(buf_count),
      ),
  )
  out_specs = pl.BlockSpec(
      (width,),
      lambda w: (w * cores_per_chip + flat,),
      pipeline_mode=pl.Buffered(buf_count),
  )
  pltpu.emit_pipeline(
      pipeline_step, grid=(num_waves,), in_specs=in_specs, out_specs=out_specs
  )(vals_hbm, out_hbm)


def pack_nonnegative(vals: jax.Array) -> jax.Array:  # i32[num_rows, width]
  """Left-packs each row's non-negative entries into a `-1` padded prefix.

  Relative order is preserved, and a row of all non-negatives fills exactly
  `width`, so this never truncates.
  """
  b, width = vals.shape
  if vals.dtype != jnp.int32:
    raise ValueError(f"vals must be int32, got {vals.dtype}")
  sc = pltpu.get_tpu_info().sparse_core
  if sc is None:
    raise NotImplementedError("SparseCore is not available")

  # Geometry off the chip: a row per subcore, a chunk per lane.
  lanes = sc.num_lanes
  rows_per_wave = sc.num_cores * sc.num_subcores
  if width % lanes != 0:
    raise ValueError(f"width ({width}) must be a multiple of {lanes}")

  num_waves = (b + rows_per_wave - 1) // rows_per_wave
  padded_b = num_waves * rows_per_wave
  if padded_b > b:
    # `-1`, not the default 0: 0 is a real position.
    vals = jnp.pad(vals, ((0, padded_b - b), (0, 0)), constant_values=-1)

  mesh = plsc.VectorSubcoreMesh(
      num_cores=sc.num_cores,
      num_subcores=sc.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  out = pl.kernel(
      functools.partial(
          _pack_body,
          b=b,
          width=width,
          num_waves=num_waves,
          lanes=lanes,
          num_cores=sc.num_cores,
          num_subcores=sc.num_subcores,
      ),
      out_type=jax.ShapeDtypeStruct((padded_b * width,), jnp.int32),
      compiler_params=pltpu.CompilerParams(
          disable_bounds_checks=True, needs_layout_passes=False
      ),
      scratch_types=(pltpu.VMEM((width + lanes,), jnp.int32),),
      mesh=mesh,
      name=f"sc_pack_nonneg_b{b}_w{width}",
  )(vals.reshape(-1))
  return out.reshape(padded_b, width)[:b]


def resolve_owned_winners(
    local_idxs: jax.Array,  # i32[num_rows, k], this rank's rank-local candidates
    slots: jax.Array,  # i32[num_rows, k], flat columns of the merged score list
    k: int,
    dcp_rank: jax.Array | int,
) -> jax.Array:
  """Picks the merged winners this rank produced out of its own list, packed.

  The candidate all-gather is rank-major, so flat column `c` came from rank
  `c // k` at that rank's own slot `c % k`; both are functions of the chunk
  width alone, not of the order the chunks arrived in. Every surviving entry
  is rank-local already, because `local_idxs` never crossed the gather --
  which is also why this is the only place ownership is decided.
  """
  safe = jnp.maximum(slots, 0)
  ours = (slots >= 0) & (jnp.floor_divide(safe, k) == dcp_rank)
  mine = jnp.take_along_axis(local_idxs, jnp.mod(safe, k), axis=1)
  return pack_nonnegative(jnp.where(ours, mine, -1))
