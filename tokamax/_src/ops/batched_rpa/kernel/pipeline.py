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
"""Manual 3-input / 2-output VMEM buffering pipeline loop for Batched RPA."""

from collections.abc import Callable
from typing import Any

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa.kernel import memory_ref


def _in_buffer_count(bref: memory_ref.BypassRef) -> int:
  try:
    return bref.in_buffer_count
  except AttributeError:
    return bref.buffer_count


def _out_buffer_count(bref: memory_ref.BypassRef) -> int:
  try:
    return bref.out_buffer_count
  except AttributeError:
    return bref.buffer_count


def _rem(val: int | jax.Array, n: int) -> int | jax.Array:
  """Computes remainder modulo `n` for Python ints and JAX arrays/tracers."""
  if isinstance(val, int):
    return val % n
  return jax.lax.rem(val.astype(jnp.int32), n)


def run_manual_triple_buffer(
    total_steps: int | jax.Array,
    compute_fn: Callable[..., Any],
    q_bref: memory_ref.BypassRef,
    o_bref: memory_ref.BypassRef,
    kv_bref: memory_ref.BypassRef,
    lse_bref: memory_ref.BypassRef | None = None,
    *,
    init_slot_fn: Callable[[int], None] | None = None,
    post_init_fn: Callable[[], None] | None = None,
) -> None:
  """Executes an explicit 3-input / 2-output buffered pipeline over `total_steps`.

  Inputs rotate over `q_bref.in_buffer_count` (3) physical slots:
  - comp_slot = step % in_slots       (slot being computed on)
  - comm_slot = (step + 2) % in_slots (slot fetched two steps ahead)

  Outputs rotate independently over `o_bref.out_buffer_count` (2) slots:
  - out_slot = step % out_slots            (slot being written)
  - wait_out_slot = (step - 2) % out_slots (slot whose writeback is awaited
    before it is reused)

  Args:
    total_steps: Total number of pipeline steps to execute.
    compute_fn: Callable of the per-step kernel body.
    q_bref: Fetch-only ref for Q blocks.
    o_bref: Writeback-only ref for attention output.
    kv_bref: Input/output ref for KV cache.
    lse_bref: Optional writeback-only ref for log-sum-exp values, or `None` when
      LSE output is disabled.
    init_slot_fn: Optional callback `init_slot_fn(slot)` invoked right before
      the prologue `copy_in` of input slot `slot` (`slot` in `{0, 1}`), so slot
      1 initialization overlaps with slot 0's prologue DMAs.
    post_init_fn: Optional callback invoked right after both prologue `copy_in`
      pairs are issued, so the remaining initialization overlaps with both
      in-flight prologue DMAs.
  """
  in_slots = _in_buffer_count(q_bref)
  out_slots = _out_buffer_count(o_bref)
  assert (in_slots, _in_buffer_count(kv_bref), out_slots) == (3, 3, 2)
  if lse_bref is not None:
    assert _out_buffer_count(lse_bref) == out_slots

  for bref in (q_bref, kv_bref, o_bref, lse_bref):
    if bref is not None:
      bref.open()

  with jax.named_scope("ep_initialize"):
    for slot in range(2):
      if init_slot_fn is not None:
        init_slot_fn(slot)
      q_bref.copy_in(slot, slot, total_steps)
      kv_bref.copy_in(slot, slot, total_steps)
    if post_init_fn is not None:
      post_init_fn()

  @pl.loop(0, total_steps)
  def steady_state(step):
    comp_slot = _rem(step, in_slots)
    comm_slot = _rem(step + 2, in_slots)
    out_slot = _rem(step, out_slots)
    wait_out_slot = _rem(step - 2, out_slots)

    with jax.named_scope("wait_in_q_kv"):
      q_bref.wait_in(step, comp_slot, total_steps)
      kv_bref.wait_in(step, comp_slot, total_steps)

    with jax.named_scope("wait_out_o"):
      o_bref.wait_out(step - 2, wait_out_slot, total_steps)
      if lse_bref is not None:
        lse_bref.wait_out(step - 2, wait_out_slot, total_steps)

    with jax.named_scope("ep_run_kernel"):

      def start_next_fetch():
        q_bref.copy_in(step + 2, comm_slot, total_steps)
        kv_bref.copy_in(step + 2, comm_slot, total_steps)

      def start_kv_writeback():
        kv_bref.copy_out(step, comp_slot, total_steps)

      compute_fn(
          step,
          comp_slot,
          total_steps,
          out_slot=out_slot,
          start_next_fetch=start_next_fetch,
          start_kv_writeback=start_kv_writeback,
      )

      with jax.named_scope("wait_out_kv"):
        kv_bref.wait_out(step, comp_slot, total_steps)

      o_bref.copy_out(step, out_slot, total_steps)
      if lse_bref is not None:
        lse_bref.copy_out(step, out_slot, total_steps)

  with jax.named_scope("epilogue"):
    last_two = total_steps - 2
    last_one = total_steps - 1
    o_bref.wait_out(last_two, _rem(last_two, out_slots), total_steps)
    if lse_bref is not None:
      lse_bref.wait_out(last_two, _rem(last_two, out_slots), total_steps)
    o_bref.wait_out(last_one, _rem(last_one, out_slots), total_steps)
    if lse_bref is not None:
      lse_bref.wait_out(last_one, _rem(last_one, out_slots), total_steps)

  for bref in (q_bref, kv_bref, o_bref, lse_bref):
    if bref is not None:
      bref.close()
