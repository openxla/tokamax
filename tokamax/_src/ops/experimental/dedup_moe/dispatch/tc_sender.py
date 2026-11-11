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
"""A wave-major ragged all-to-all as a pair of split-phase Pallas kernels.

XLA builds an async ragged all-to-all, but its scheduler places start and done
back to back and `AsyncCollectiveMerger` turns the pair back into one sync op,
so nothing can run under the exchange. Splitting it into custom calls -- a
start kernel that only ISSUES the remote DMAs and returns their semaphores, and
consumers that wait on them -- puts the overlap under the caller's control
(see the Pallas TPU async design note in the JAX documentation).

The exchange is issued in `waves` groups: the caller lays the dispatch plan out
by expert group, and the start kernel issues group 0 to every destination, then
group 1, and so on. Every row of group w lands on the receiver's semaphore
`recv_sems[w]`, so a receiver can start computing on group 0
while later groups are still in flight. The fused MoE kernel does exactly that:
it takes the receive semaphores as an operand and waits for group w only before
its first expert in group w.

Table: the [4 * ep, waves] int32 dispatch plan. Rows (k * ep + d, w): k=0 send
offset, k=1 send rows, k=2 offset in d's receive buffer; row 3 * ep holds the
rows this shard receives per group, row 3 * ep + 1 col 0 the total rows it
sends to OTHER shards. A sender signals the receiver's semaphore for group w,
so the receiver's count for group w is the sum over sources -- exactly row
3 * ep -- and its waits are byte-exact (every semaphore returns to zero).

Invariants:
  * Senders only `wait_send` (never `wait`), once, after the consumer kernel
    (`ragged_a2a_send_done`): every outgoing DMA had the whole kernel to
    finish, and `wait_send` never blocks on a peer.
  * The start kernel opens with an all-pairs barrier, so no shard writes a
    peer's receive buffer before that peer is executing the same program
    point (its buffer is live there; SPMD buffer assignment is identical).
  * `send` and `recv` are threaded through the kernels with
    input_output_aliases, and the send drain threads a value the caller uses,
    so XLA can neither free, reuse nor dead-code-eliminate them while DMAs
    are in flight.
  * Every exchange buffer is pinned to HBM, operands and outputs both, so
    memory-space assignment cannot stage it through VMEM around these calls.

With De-Dup, the start kernel also packs and sends each (source, remote
board) stage run to that board's gateway device; boards are contiguous
blocks of `board_size` devices on the mesh axis.
"""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Distinct from the fused kernel's COLLECTIVE_ID (0): its own barrier semaphore.
A2A_COLLECTIVE_ID = 1


def _all_pairs_barrier(ep):
  sem = pltpu.get_barrier_semaphore()
  for i in range(ep):
    pl.semaphore_signal(sem, inc=1, device_id=(jnp.int32(i),))
  pl.semaphore_wait(sem, ep)


def _head_tail_piece(n, cs, head):
  """Rows [lo, lo + m) of an n-row run sent in this pass.

  The head pass sends the first `cs` rows and the tail pass the rest.

  Args:
    n: Rows of the run.
    cs: Rows of the head piece (chunk_rows).
    head: Whether this is the head pass.

  Returns:
    (lo, m): the first row and the row count of this pass's piece.
  """
  lo = jnp.int32(0) if head else cs
  m = jnp.minimum(cs, n) if head else n - cs
  return lo, m


def _start_waves_kernel(
    tab,
    me_ref,
    send_in,
    recv_in,
    *refs,
    ep,
    waves,
    chunk_rows,
    list_rows,
    has_stage=False,
    num_boards=0,
    board_size=0,
):
  """Start kernel body: pack and issue the lists, stage runs and waves."""
  del send_in, recv_in  # aliased to send_out / recv_out
  refs = list(refs)
  perm_sm = refs.pop(0)
  rows = refs.pop(0)
  list_tab = refs.pop(0)
  tg_lists_in = refs.pop(0)
  tg_land_in = refs.pop(0)
  del tg_lists_in, tg_land_in
  if has_stage:
    stage_send_tab = refs.pop(0)
    stage_in = refs.pop(0)
    del stage_in
  else:
    stage_send_tab = None
  send_sem = refs.pop(0)
  recv_sems = refs.pop(0)
  send_out = refs.pop(0)
  recv_out = refs.pop(0)
  list_send_sem = refs.pop(0)
  tg_sem = refs.pop(0)
  tg_lists_out = refs.pop(0)
  tg_land_out = refs.pop(0)
  if has_stage:
    stage_sems = refs.pop(0)
    stage_out = refs.pop(0)
  else:
    stage_sems = stage_out = None
  pack_sem = refs.pop(0)
  assert not refs

  me = me_ref[0]
  _all_pairs_barrier(ep)
  # Issue the token-gather list DMAs before wave 0's activation rows so the
  # lists land before the rows that the receiver's token gather indexes.
  for i in range(ep):
    d = lax.rem(me + i, ep)
    n_list = list_tab[d]
    src_list = tg_lists_out.at[pl.ds(d * list_rows, n_list)]
    dst_list = tg_land_out.at[pl.ds(me * list_rows, n_list)]
    if i == 0:

      @pl.when(n_list > 0)
      def _(src_list=src_list, dst_list=dst_list):
        pltpu.make_async_copy(src_list, dst_list, tg_sem).start()

    else:

      @pl.when(n_list > 0)
      def _(src_list=src_list, dst_list=dst_list, d=d):
        pltpu.make_async_remote_copy(
            src_ref=src_list,
            dst_ref=dst_list,
            send_sem=list_send_sem,
            recv_sem=tg_sem,
            device_id=(d,),
        ).start()

  cs = jnp.int32(chunk_rows)

  def pack_rows(base, m):
    """Pack send rows [base, base + m) from `rows` by `perm`."""

    def _pack_one(r, _):
      idx = base + r
      pltpu.make_async_copy(
          rows.at[pl.ds(perm_sm[idx], 1)],
          send_out.at[pl.ds(idx, 1)],
          pack_sem,
      ).start()
      return jnp.int32(0)

    lax.fori_loop(0, m, _pack_one, jnp.int32(0))

  def pack_stage(w, head):
    # Pack the De-Dup stage runs of this (w, head) pass into send_out
    # and returns the rows started on `pack_sem`.
    assert stage_send_tab is not None
    m_stage = jnp.int32(0)
    for bi in range(1, num_boards):
      b = lax.rem(me // board_size + bi, num_boards)
      n = stage_send_tab[num_boards + b, w]
      lo, m = _head_tail_piece(n, cs, head)
      m = jnp.maximum(jnp.int32(0), m)
      pack_rows(stage_send_tab[b, w] + lo, m)
      m_stage = m_stage + m
    return m_stage

  def pack_direct(w, head):
    # Wave-pipelined local pack of this (w, head) pass's direct runs into
    # send_out, so no XLA `rows[perm]` gather runs before the kernel
    # and returns the rows started on `pack_sem`.
    m_dir = jnp.int32(0)
    for i in range(ep):
      d = lax.rem(me + i, ep)
      n = tab[ep + d, w]
      lo, m = _head_tail_piece(n, cs, head)
      m = jnp.maximum(jnp.int32(0), m)
      pack_rows(tab[d, w] + lo, m)
      m_dir = m_dir + m
    return m_dir

  def wait_pack(m_rows):
    @pl.when(m_rows > 0)
    def _():
      pltpu.make_async_copy(
          send_out.at[pl.ds(0, m_rows)],
          send_out.at[pl.ds(0, m_rows)],
          pack_sem,
      ).wait()

  def send_stage(w, head):
    # Issue the De-Dup stage runs to the remote gateways. A stage run never
    # targets this shard's own board (`b != me // board_size`), so every
    # stage DMA is a remote copy on `send_sem` signalling
    # `stage_sems.at[sigma * W + w]` on gateway `gw`; the head and tail
    # pieces share it.
    assert stage_send_tab is not None
    assert stage_out is not None
    assert stage_sems is not None
    for bi in range(1, num_boards):
      b = lax.rem(me // board_size + bi, num_boards)
      n = stage_send_tab[num_boards + b, w]
      gw = stage_send_tab[3 * num_boards + b, w]
      sig = stage_send_tab[4 * num_boards + b, w]
      lo, m = _head_tail_piece(n, cs, head)
      src = send_out.at[pl.ds(stage_send_tab[b, w] + lo, m)]
      dst = stage_out.at[pl.ds(stage_send_tab[2 * num_boards + b, w] + lo, m)]

      @pl.when(m > 0)
      def _(src=src, dst=dst, gw=gw, sig=sig, w=w):
        pltpu.make_async_remote_copy(
            src_ref=src,
            dst_ref=dst,
            send_sem=send_sem,
            recv_sem=stage_sems.at[sig * waves + w],
            device_id=(gw,),
        ).start()

  def send_direct(w, head):
    # Destination order rotated by `me` (as XLA's own a2a does) so the
    # shards do not all hit peer 0 first.
    for i in range(ep):
      d = lax.rem(me + i, ep)
      n = tab[ep + d, w]
      lo, m = _head_tail_piece(n, cs, head)
      src = send_out.at[pl.ds(tab[d, w] + lo, m)]
      dst = recv_out.at[pl.ds(tab[2 * ep + d, w] + lo, m)]
      if i == 0:  # d == me: a local copy on this shard's own slot

        @pl.when(m > 0)
        def _(src=src, dst=dst, w=w):
          pltpu.make_async_copy(src, dst, recv_sems.at[w]).start()

      else:

        @pl.when(m > 0)
        def _(src=src, dst=dst, d=d, w=w):
          pltpu.make_async_remote_copy(
              src_ref=src,
              dst_ref=dst,
              send_sem=send_sem,
              recv_sem=recv_sems.at[w],
              device_id=(d,),
          ).start()

  for w in range(waves):
    # The first chunk_rows rows of every destination, then the rest of
    # each destination's run in one copy: every peer's first piece is in
    # flight early, and a wave costs at most 2 * ep copies however
    # skewed it is (chunking every run cost a loop body per chunk of the
    # busiest destination, times ep).
    for head in (True, False):
      # Stage rows take 2 hops (sender -> gateway -> target), so each
      # (w, head) pass packs and issues its cross-board stage runs
      # FIRST, at the head of the DMA queue, and only then the ep 1-hop
      # direct runs.
      if has_stage:
        wait_pack(pack_stage(w, head))
        send_stage(w, head)
      wait_pack(pack_direct(w, head))
      send_direct(w, head)


def ragged_a2a_start_waves(
    send,
    recv,
    disp_tab,
    me,
    *,
    ep,
    waves,
    chunk_rows,
    rows,
    perm,
    list_tab,
    tg_lists,
    tg_land,
    stage_send_tab=None,
    stage=None,
    num_boards=0,
    num_slots=0,
    board_size=0,
):
  """Issue the wave-major exchange, its token-gather lists and De-Dup stage.

  send [rows, ...] is an uninitialized send buffer that the kernel packs
  from `rows` by `perm`, wave by wave; recv (uninitialized HBM) is the
  receive buffer; disp_tab is the [4 * ep, waves] plan described in the
  module docstring and me this shard's index. Each (me -> d) token-gather
  list goes into `tg_land` on `tg_sem`. When `stage_send_tab` and `stage`
  are supplied (De-Dup), also packs and DMAs each (me -> remote board b)
  stage run into the gateway's `stage` buffer, on
  `stage_sems[sigma * waves + w]`.

  Args:
    send: Uninitialized send buffer the kernel packs from `rows`.
    recv: Uninitialized receive buffer in HBM.
    disp_tab: [4 * ep, W] dispatch table.
    me: This shard's index.
    ep: Expert-parallel width.
    waves: W.
    chunk_rows: Rows per DMA chunk.
    rows: This shard's token rows.
    perm: Local token of each packed send row.
    list_tab: Token-gather list table.
    tg_lists: Token-gather lists to send.
    tg_land: Landing buffer of the lists.
    stage_send_tab: De-Dup stage table, or None.
    stage: Gateway stage buffer, or None.
    num_boards: Boards (De-Dup).
    num_slots: Stage slots per gateway (De-Dup).
    board_size: Devices per board (De-Dup).

  Returns:
    The `fut` tuple of semaphores and buffers the receivers and the drain wait
    on.

  Raises:
    ValueError: If the arguments are inconsistent.
  """
  hbm = functools.partial(
      pltpu.with_memory_space_constraint, memory_space=pltpu.HBM
  )
  send, recv = hbm(send), hbm(recv)
  tg_lists = hbm(tg_lists)
  tg_land = hbm(tg_land)
  if stage is not None:
    stage = hbm(stage)
  has_stage = stage is not None
  if has_stage != (stage_send_tab is not None):
    raise ValueError("pass stage_send_tab and stage together or neither")
  if has_stage and (num_boards < 1 or num_slots < 1 or board_size < 1):
    raise ValueError(
        "has_stage requires num_boards, num_slots and board_size >= 1"
    )
  list_rows = tg_lists.shape[0] // ep
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  anyspec = pl.BlockSpec(memory_space=pl.ANY)
  semspec = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
  in_specs = [smem, smem, anyspec, anyspec, smem, anyspec]
  args = [
      disp_tab.astype(jnp.int32),
      jnp.reshape(me, (1,)).astype(jnp.int32),
      send,
      recv,
      perm.astype(jnp.int32),
      rows,
  ]
  aliases = {2: 2, 3: 3}
  out_shape = [
      pltpu.SemaphoreType.DMA(()),
      pltpu.SemaphoreType.DMA((waves,)),
      jax.ShapeDtypeStruct(send.shape, send.dtype),
      jax.ShapeDtypeStruct(recv.shape, recv.dtype),
  ]
  out_specs = [semspec, semspec, anyspec, anyspec]
  base_in = len(in_specs)
  base_out = len(out_shape)
  in_specs.extend([smem, anyspec, anyspec])
  args.extend([list_tab.astype(jnp.int32), tg_lists, tg_land])
  out_shape.extend([
      pltpu.SemaphoreType.DMA(()),
      pltpu.SemaphoreType.DMA(()),
      jax.ShapeDtypeStruct(tg_lists.shape, tg_lists.dtype),
      jax.ShapeDtypeStruct(tg_land.shape, tg_land.dtype),
  ])
  out_specs.extend([semspec, semspec, anyspec, anyspec])
  aliases[base_in + 1] = base_out + 2
  aliases[base_in + 2] = base_out + 3
  if has_stage:
    assert stage_send_tab is not None
    assert stage is not None
    base_in = len(in_specs)
    base_out = len(out_shape)
    in_specs.extend([smem, anyspec])
    args.extend([stage_send_tab.astype(jnp.int32), stage])
    out_shape.extend([
        pltpu.SemaphoreType.DMA((num_slots * waves,)),
        jax.ShapeDtypeStruct(stage.shape, stage.dtype),
    ])
    out_specs.extend([semspec, anyspec])
    aliases[base_in + 1] = base_out + 1
  return pl.pallas_call(
      functools.partial(
          _start_waves_kernel,
          ep=ep,
          waves=waves,
          chunk_rows=chunk_rows,
          list_rows=list_rows,
          has_stage=has_stage,
          num_boards=num_boards,
          board_size=board_size,
      ),
      out_shape=tuple(out_shape),
      in_specs=in_specs,
      out_specs=tuple(out_specs),
      scratch_shapes=(pltpu.SemaphoreType.DMA(()),),
      input_output_aliases=aliases,
      compiler_params=pltpu.CompilerParams(collective_id=A2A_COLLECTIVE_ID),
      name=f"wave_a2a_start_ep{ep}_w{waves}{'_dedup' if has_stage else ''}",
  )(*args)


def _send_done_kernel(
    tab,
    send,
    send_sem,
    list_tab,
    tg_lists,
    list_send_sem,
    carry,
    carry_out,
    *,
    ep,
    axis=None,
):
  """Drain kernel body: wait for this shard's outgoing rows and lists."""
  del carry, carry_out  # aliased pass-through (orders + keeps this alive)
  total = tab[3 * ep + 1, 0]
  # Any peer works: the drain only waits on this shard's own send semaphore
  # (upstream passes mesh coordinate (0,)). The standalone drain kernel
  # (ragged_a2a_send_done) names the shard_map axis instead, because JAX
  # lowers `wait_send` with the real device id against the kernel's mesh,
  # and Pallas gives a kernel its mesh
  # only when the kernel has a named-axis or comms effect. That kernel only
  # waits, and a DMA with a tuple id adds neither, so the tuple failed to
  # lower ("got 1 ids for a 0D mesh"); the dict form adds the named-axis
  # effect. Same wait, same semaphore. axis=None keeps the tuple for callers
  # that inline this body into a kernel that already communicates.
  peer = (jnp.int32(0),) if axis is None else {axis: jnp.int32(0)}

  @pl.when(total > 0)
  def _():
    # One byte-exact wait for every row this shard sent to other shards
    # (all on the one send semaphore); `wait_send` never blocks on peers.
    pltpu.make_async_remote_copy(
        src_ref=send.at[pl.ds(0, total)],
        dst_ref=send.at[pl.ds(0, total)],
        send_sem=send_sem,
        recv_sem=send_sem,
        device_id=peer,
    ).wait_send()

  list_total = list_tab[ep]

  @pl.when(list_total > 0)
  def _():
    pltpu.make_async_remote_copy(
        src_ref=tg_lists.at[pl.ds(0, list_total)],
        dst_ref=tg_lists.at[pl.ds(0, list_total)],
        send_sem=list_send_sem,
        recv_sem=list_send_sem,
        device_id=peer,
    ).wait_send()


def _shard_map_axis():
  """The single mesh axis of the enclosing shard_map (the layer's)."""
  names = jax.sharding.get_abstract_mesh().axis_names
  if len(names) != 1:
    raise ValueError(
        "ragged_a2a_send_done must run inside the layer's single-axis"
        f" shard_map; the current mesh axes are {names}"
    )
  return names[0]


def ragged_a2a_send_done(
    send_sem,
    send,
    disp_tab,
    carry,
    *,
    ep,
    list_tab,
    tg_lists,
    list_send_sem,
):
  """Drain the start kernel's outgoing DMAs; returns `carry` (aliased).

  `carry` must be a value the caller USES (e.g. the fused kernel's
  arrivals): threading it through makes the drain impossible to
  dead-code-eliminate and orders it after `carry`'s producer. Without it
  the drain was eliminated, so the send semaphore and send buffer died
  right after the start kernel and XLA reused them while the DMAs were
  still in flight (the fused kernel's semaphores ended nonzero and garbage
  rows were sent).

  `carry` (and so the aliased output) is pinned to HBM. Unconstrained,
  memory-space assignment can stage the arrivals through VMEM around this
  call, a full extra copy in and out.

  The start kernel sends De-Dup stage rows on the same send semaphore and
  counts them in disp_tab[3 * ep + 1, 0], so this one wait covers them.

  Args:
    send_sem: The start kernel's send semaphore.
    send: The send buffer.
    disp_tab: [4 * ep, W] dispatch table.
    carry: A value the caller uses, threaded through and aliased.
    ep: Expert-parallel width.
    list_tab: Token-gather list table.
    tg_lists: The sent lists.
    list_send_sem: The lists' send semaphore.

  Returns:
    `carry`, aliased and pinned to HBM.
  """
  carry = pltpu.with_memory_space_constraint(carry, pltpu.HBM)
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  anyspec = pl.BlockSpec(memory_space=pl.ANY)
  semspec = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
  in_specs = [smem, anyspec, semspec, smem, anyspec, semspec, anyspec]
  args = [
      disp_tab.astype(jnp.int32),
      send,
      send_sem,
      list_tab.astype(jnp.int32),
      tg_lists,
      list_send_sem,
      carry,
  ]
  return pl.pallas_call(
      functools.partial(_send_done_kernel, ep=ep, axis=_shard_map_axis()),
      out_shape=jax.ShapeDtypeStruct(carry.shape, carry.dtype),
      in_specs=in_specs,
      out_specs=anyspec,
      input_output_aliases={len(args) - 1: 0},
      name=f"wave_a2a_send_done_ep{ep}",
  )(*args)
