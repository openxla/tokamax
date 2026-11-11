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
"""Reference direct-send rows for tests of the De-Dup plan.

The first-wave rows `dest`, `wave_pair`, `fw` and `need` of one source, built
exactly like the reference direct wave plan (`wave_dispatch.dispatch_plan`)
restricted to the t tokens of that source (a `dynamic_slice`, no gather or
scatter), plus the receiver-side view of every source. The production plan
builds the gateway view (`dedup_plan.served_fw`) from the same row builder,
`dedup_plan.direct_send_rows`.

Optionally the rows are built for a block of `num_dests` consecutive
destination devices only (a gateway needs the devices of its own board for
each source it serves, and a receiver the devices of its board for every
source), which costs O(K * num_dests * t) instead of O(K * ep * t).
"""

import jax
from jax import lax
import jax.numpy as jnp
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan

Phase1Rows = dedup_plan.DirectSendRows
served_fw = dedup_plan.served_fw
_rows_for_pairs = dedup_plan.direct_send_rows


# Identical to wave_dispatch.ascending_expert_order; inlined so the plan
# tests do not depend on the kernel library.
def _ascending_expert_order(hist, *, ep, g_local):
  """Each shard's local experts, fewest rows first, empties last.

  Ties break by ascending index, as in expert_visit_list.

  Args:
    hist: [ep * g_local] i32 rows per global expert (the exchanged routing
      histogram, identical on every shard, so every sender and receiver derives
      the same order).
    ep: Expert-parallel width.
    g_local: Experts per shard.

  Returns:
    (pos [ep, g_local] i32: the visit position of local expert e on shard d;
    order [ep, g_local] i32: the local expert at visit position p on d, so
    order[me] is a drop-in visit list with the nonempty experts first).
  """
  h = hist.reshape(ep, g_local).astype(jnp.int32)
  idx = jnp.arange(g_local, dtype=jnp.int32)
  # 2**24 > any row count; 2**24 * g_local + g_local stays below 2**31.
  key = jnp.where(h > 0, jnp.minimum(h, 2**24 - 1), 2**24) * g_local + idx
  pos = jnp.sum((key[:, None, :] < key[:, :, None]).astype(jnp.int32), axis=2)
  order = jnp.sum(
      jnp.where(pos[:, None, :] == idx[None, :, None], idx[None, None, :], 0),
      axis=2,
  )
  return pos, order


def wave_of_from_rows_by_dest(
    rows_by_dest: jax.Array, *, g: int, num_waves: int
) -> jax.Array:
  """Equal-width expert waves from the exchanged counts (a test input).

  Every device visits its local experts fewest rows first, and wave =
  visit position // (g / W). The layer cuts ramp
  waves instead (wave_dispatch.wave_plan); the plan functions take any
  `wave_of`, and the golden tests use this one.

  Args:
    rows_by_dest: [ep, E] int32 per-expert pair counts of every source (or any
      [.., E] array whose column sums are the global expert histogram).
    g: Local experts per device.
    num_waves: W (must divide g).

  Returns:
    wave_of [ep, g] int32.
  """
  hist = jnp.sum(rows_by_dest, axis=0).astype(jnp.int32)
  ep = hist.shape[0] // g
  pos, _ = _ascending_expert_order(hist, ep=ep, g_local=g)
  return pos // (g // num_waves)


def phase1_inputs(
    pairs_g: jax.Array,
    wave_of: jax.Array,
    s: jax.Array | int,
    *,
    t: int,
    num_waves: int,
    dest_base: jax.Array | int = 0,
    num_dests: int | None = None,
) -> Phase1Rows:
  """Direct-send rows `dest`, `wave_pair`, `fw` and `need` for source s.

  Args:
    pairs_g: [T, K] int32 global expert ids of every token in mesh order (T = ep
      * t; dest = id // g).
    wave_of: [ep, g] int32 wave of each device's local experts.
    s: Source device (Python int or traced int32 scalar).
    t: Tokens per device.
    num_waves: W.
    dest_base: First destination device of the block (traced is fine).
    num_dests: Block size (static); None = all ep devices.

  Returns:
    The rows; `fw`/`need` cover devices [dest_base, dest_base + nd).
  """
  ep = wave_of.shape[0]
  s = jnp.asarray(s, jnp.int32)
  pk = lax.dynamic_slice_in_dim(pairs_g.astype(jnp.int32), s * t, t, axis=0).T
  if num_dests is None:
    dest_base, num_dests = 0, ep
  return _rows_for_pairs(
      pk,
      wave_of,
      num_waves=num_waves,
      dest_base=dest_base,
      num_dests=num_dests,
  )


def board_fw(
    pairs_g: jax.Array,
    wave_of: jax.Array,
    me: jax.Array | int,
    *,
    t: int,
    num_waves: int,
    board_size: int,
) -> jax.Array:
  """Receiver input: fw of every source for the devices of board(me).

  One pass over all T tokens, O(K * board_size * T): the input of
  `dedup_plan.receiver_counts`.

  Args:
    pairs_g: [T, K] int32 global expert ids.
    wave_of: [ep, g] int32.
    me: Receiving device (Python int or traced int32 scalar).
    t: Tokens per device.
    num_waves: W.
    board_size: Devices per board.

  Returns:
    [ep, board_size, t] int32.
  """
  ep = wave_of.shape[0]
  me = jnp.asarray(me, jnp.int32)
  rows = _rows_for_pairs(
      pairs_g.astype(jnp.int32).T,
      wave_of,
      num_waves=num_waves,
      dest_base=(me // board_size) * board_size,
      num_dests=board_size,
  )
  return jnp.transpose(rows.fw.reshape(board_size, ep, t), (1, 0, 2))
