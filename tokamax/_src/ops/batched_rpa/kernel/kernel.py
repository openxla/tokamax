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
"""Top-level Batched RPA attention Pallas kernel orchestrator."""

from collections.abc import Callable, Sequence
import dataclasses
import typing

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa.kernel import attention
from tokamax._src.ops.batched_rpa.kernel import configs
from tokamax._src.ops.batched_rpa.kernel import memory_ref
from tokamax._src.ops.batched_rpa.kernel import pipeline
from tokamax._src.ops.batched_rpa.kernel import schedule
from tokamax._src.ops.batched_rpa.kernel import vmem_ldst


def _attend_subchunks(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    k_scale: jax.Array | float | None,
    v_scale: jax.Array | float | None,
    causal_offset: Sequence[jax.Array],
    k_min: Sequence[jax.Array] | None,
    k_max: Sequence[jax.Array] | None,
    is_last_k: Sequence[jax.Array | bool],
    m_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    acc_scratch_ref: jax.Ref,
    attention_sink_ref: jax.Ref | None,
    cfgs: configs.RpaConfigs,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
  """Pipelines `QK[c]` with `PV[c - 1]` across `bq_c_sz` sub-chunks."""
  m_val = m_scratch_ref[...]
  l_val = l_scratch_ref[...]
  acc_val = acc_scratch_ref[...]
  sink_val = None
  if attention_sink_ref is not None:
    sink_val = jnp.concat([attention_sink_ref[...]] * cfgs.bq_c_sz, axis=1)

  l_new_list = []
  m_new_list = []
  acc_new_list = []

  prev_p = prev_alpha_list = prev_q_slice = None
  for bq_start in range(0, cfgs.bq_sz, cfgs.bq_c_sz):
    bq_end = min(bq_start + cfgs.bq_c_sz, cfgs.bq_sz)
    q_start = bq_start * cfgs.aligned_num_q_heads_per_kv_head
    q_end = bq_end * cfgs.aligned_num_q_heads_per_kv_head
    q_slice = slice(q_start, q_end)

    custom_mask = attention.build_qk_mask(
        shape=(
            cfgs.batch_size,
            cfgs.model.num_kv_heads,
            q_end - q_start,
            cfgs.bkv_sz,
        ),
        bq_start=bq_start,
        causal_offset=causal_offset,
        k_min=k_min,
        k_max=k_max,
        cfgs=cfgs,
    )

    p, alpha_list, m_next_chunk, l_next_chunk, m_carry, l_carry = (
        attention.qk_softmax_step(
            q[:, :, q_slice],
            k,
            m_val[:, q_slice],
            l_val[:, q_slice],
            is_last_k,
            custom_mask=custom_mask,
            cfgs=cfgs,
            k_scale=k_scale,
            attention_sink=sink_val,
        )
    )
    m_scratch_ref[:, q_slice] = m_carry
    l_scratch_ref[:, q_slice] = l_carry
    if cfgs.serve.return_lse:
      m_new_list.append(m_next_chunk)
    l_new_list.append(l_next_chunk)

    if prev_p is not None:
      assert prev_alpha_list is not None and prev_q_slice is not None
      o_next_chunk, o_carry = attention.pv_step(
          prev_p,
          v,
          prev_alpha_list,
          acc_val[:, prev_q_slice],
          is_last_k,
          cfgs=cfgs,
          v_scale=v_scale,
          attention_sink=sink_val,
      )
      acc_scratch_ref[:, prev_q_slice] = o_carry
      acc_new_list.append(o_next_chunk)

    prev_p = p
    prev_alpha_list = alpha_list
    prev_q_slice = q_slice

  assert (
      prev_p is not None
      and prev_alpha_list is not None
      and prev_q_slice is not None
  )
  o_next_chunk, o_carry = attention.pv_step(
      prev_p,
      v,
      prev_alpha_list,
      acc_val[:, prev_q_slice],
      is_last_k,
      cfgs=cfgs,
      v_scale=v_scale,
      attention_sink=sink_val,
  )
  acc_scratch_ref[:, prev_q_slice] = o_carry
  acc_new_list.append(o_next_chunk)

  m_next = (
      jnp.concatenate(m_new_list, axis=2) if cfgs.serve.return_lse else None
  )
  l_next = jnp.concatenate(l_new_list, axis=2)
  acc_next = jnp.concatenate(acc_new_list, axis=2)
  return acc_next, l_next, m_next


def _store_outputs(
    is_last_k_done: Sequence[jax.Array],
    acc_next: jax.Array,
    l_next: jax.Array,
    m_next: jax.Array | None,
    o_vref: jax.Ref,
    lse_o_vref: jax.Ref | None,
    *,
    cfgs: configs.RpaConfigs,
) -> None:
  """Normalizes and writes completed batch lanes to `o_vref` / `lse_o_vref`."""

  def _store_normalized_out(
      b_idx: int | jax.Array, batch_acc: jax.Array, batch_l: jax.Array
  ):
    out = attention.normalize_out(batch_acc, batch_l, cfgs=cfgs)
    vmem_ldst.store_out_lane(o_vref, b_idx, out, cfgs=cfgs)

  def _store_lse(
      b_idx: int | jax.Array, batch_m: jax.Array, batch_l: jax.Array
  ):
    assert lse_o_vref is not None
    lse_val = attention.compute_lse(batch_m, batch_l)
    vmem_ldst.store_lse_lane(lse_o_vref, b_idx, lse_val, cfgs=cfgs)

  if cfgs.fuse_accum:
    for b in range(cfgs.batch_size):
      _store_normalized_out(b, acc_next[b], l_next[b])
  else:
    for b in range(cfgs.batch_size):
      accum_named_call = jax.named_call(
          _store_normalized_out, name=f"accum_{b}"
      )
      jax.lax.cond(
          is_last_k_done[b],
          accum_named_call,
          lambda *_: None,
          b,
          acc_next[b],
          l_next[b],
      )

  if cfgs.serve.return_lse:
    assert m_next is not None
    for b in range(cfgs.batch_size):
      jax.lax.cond(
          is_last_k_done[b],
          _store_lse,
          lambda *_: None,
          b,
          m_next[b],
          l_next[b],
      )


def rpa_body(
    step: int | jax.Array,
    slot: int | jax.Array,
    q_vref: jax.Ref,
    kv_in_vref: typing.Any,
    o_vref: jax.Ref,
    lse_o_vref: jax.Ref | None,
    schedule_ref: schedule.RpaSchedule,
    m_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    acc_scratch_ref: jax.Ref,
    *,
    attention_sink_ref: jax.Ref | None,
    cfgs: configs.RpaConfigs,
    start_next_fetch: Callable[[], None],
    start_kv_writeback: Callable[[], None],
) -> None:
  """Executes one step of the Batched RPA attention pipeline."""
  del slot
  kv_in_vref = pltpu.annotate(kv_in_vref, no_hazard_no_deps=True)
  metas = [schedule_ref.tile_meta[step, b] for b in range(cfgs.batch_size)]

  # 1. Stitch staged new KV into cache prefix (SEQ_ALONG_LANE) and write back.
  if cfgs.is_seq_along_lane:
    seq_metas = [typing.cast(schedule.SeqAlongLaneTileMeta, m) for m in metas]
    stitch_dst = [m.stitch_dst for m in seq_metas]
    stitch_src = [m.stitch_src for m in seq_metas]
    stitch_len = [m.stitch_len for m in seq_metas]
  else:
    stitch_dst = stitch_src = stitch_len = ()

  always_true = schedule_ref.actual_steps[0] >= 0

  @pl.when(always_true)
  @jax.named_scope("stitch_and_issue_writeback")
  def _kv_store_fence():
    if cfgs.is_seq_along_lane:
      vmem_ldst.stitch_kv_along_lane(
          kv_in_vref, stitch_dst, stitch_src, stitch_len, cfgs=cfgs
      )
    start_kv_writeback()

  # 2. Prefetch Q/KV for step + 2 and load current Q/KV from VMEM into VREGs.
  start_next_fetch()
  causal_offset = [m.causal_offset for m in metas]
  scope = cfgs.serve.attention_scope
  k_min = (
      [m.k_min for m in metas]
      if scope == configs.AttentionScope.NEW_TOKENS_ONLY
      else None
  )
  k_max = (
      [m.k_max for m in metas]
      if scope == configs.AttentionScope.CACHE_ONLY
      else None
  )
  is_last_k_any = [m.is_last_k != 0 for m in metas]
  is_last_k_done = [m.is_final_k for m in metas]

  q = vmem_ldst.load_q_tile(q_vref, cfgs=cfgs)
  k, v, k_scale, v_scale = vmem_ldst.load_kv_tile(kv_in_vref, cfgs=cfgs)

  # 3. Run FlashAttention over the loaded Q/KV tile.
  acc_next, l_next, m_next = _attend_subchunks(
      q,
      k,
      v,
      k_scale=k_scale,
      v_scale=v_scale,
      causal_offset=causal_offset,
      k_min=k_min,
      k_max=k_max,
      is_last_k=is_last_k_any,
      m_scratch_ref=m_scratch_ref,
      l_scratch_ref=l_scratch_ref,
      acc_scratch_ref=acc_scratch_ref,
      attention_sink_ref=attention_sink_ref,
      cfgs=cfgs,
  )

  # 4. Normalize and store completed lanes into output VMEM slots.
  _store_outputs(
      is_last_k_done,
      acc_next,
      l_next,
      m_next,
      o_vref,
      lse_o_vref,
      cfgs=cfgs,
  )


def get_kernel_name(cfgs: configs.RpaConfigs) -> str:
  """Returns the XProf/Percale kernel name for `cfgs`."""
  serve = cfgs.serve
  name = f"RPA{cfgs.mode.symbol}-{serve.kv_layout.symbol}-p{serve.page_size}"
  name += f"-b{cfgs.batch_size}-q{cfgs.bq_sz}-k{cfgs.bkv_sz}"
  if cfgs.model.sliding_window:
    name += f"-sw{cfgs.model.sliding_window}"
  if serve.attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY:
    name += "-new_only"
  elif serve.attention_scope == configs.AttentionScope.CACHE_ONLY:
    name += "-cache_only"
  elif serve.attention_scope == configs.AttentionScope.FULL:
    name += "-full"
  return name


def get_kernel_metadata(
    cfgs: configs.RpaConfigs,
) -> dict[str, str | int | float]:
  """Serializes `cfgs` into flat Pallas compiler metadata."""
  cfgs_dict = dataclasses.asdict(cfgs)
  ret = {}
  for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
    key = jax.tree_util.keystr(path, simple=True, separator=".")
    if not isinstance(val, str | int | float):
      val = str(val)
    ret[key] = val
  return ret


def rpa_kernel(
    schedule_hbm: schedule.RpaSchedule,
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    kv_cache_hbm: jax.Array,
    lse_hbm: jax.Array | None,
    attention_sink_hbm: jax.Array | None,
    *,
    cfgs: configs.RpaConfigs,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
  """Launches the Batched RPA attention Pallas kernel over `schedule_hbm`."""
  return_lse = cfgs.serve.return_lse

  def ragged_paged_attention_pipeline(
      schedule_hbm_ref: schedule.RpaSchedule,
      q_hbm_ref: jax.Ref,
      new_kv_hbm_ref: jax.Ref,
      kv_cache_hbm_ref: jax.Ref,
      lse_hbm_ref: jax.Ref | None,
      attention_sink_ref: jax.Ref | None,
      o_hbm_ref: jax.Ref,
      o_kv_cache_hbm_ref: jax.Ref,
      o_lse_hbm_ref: jax.Ref | None = None,
  ):
    del o_kv_cache_hbm_ref, o_lse_hbm_ref

    q_alloc, kv_cache_alloc, o_alloc, lse_alloc = memory_ref.create_allocs(
        kv_cache_hbm_ref, q_hbm_ref, lse_hbm_ref, cfgs
    )

    max_steps_ub = schedule.RpaSchedule.max_steps_ub(cfgs)
    actual_steps = schedule_hbm_ref.actual_steps[0]
    num_safe_step_iterations = pl.cdiv(actual_steps, max_steps_ub)

    @pl.with_scoped(
        final_allocs=(q_alloc, kv_cache_alloc, o_alloc, lse_alloc),
        schedule_ref=schedule.RpaSchedule.create_shape_dtype(
            cfgs
        ).scratch_shapes(),
        dma_sem=pltpu.SemaphoreType.DMA((1,)),
        scratches=vmem_ldst.SoftmaxScratch.scratch_shapes(cfgs),
    )
    def _run(final_allocs, schedule_ref, dma_sem, scratches):
      q_alloc, kv_cache_alloc, o_alloc, lse_alloc = final_allocs
      schedule_ref.actual_steps[0] = actual_steps
      vmem_ldst.SoftmaxScratch.init(
          scratches[0],
          scratches[1],
          scratches[2],
          attention_sink_ref,
          cfgs=cfgs,
      )
      memory_ref.zero_initialize_buffers(
          q_alloc, kv_cache_alloc, num_lanes=cfgs.layout.num_lanes
      )

      def execute_schedule_chunk(start_step, num_steps):
        total_steps = schedule_ref.read_chunk_from_hbm(
            schedule_hbm_ref, start_step, num_steps, dma_sem
        )
        kv_bound = kv_cache_alloc.bind(
            src_ref=(kv_cache_hbm_ref, new_kv_hbm_ref, schedule_ref),
            dst_ref=(kv_cache_hbm_ref, new_kv_hbm_ref, schedule_ref),
        )
        q_bound = q_alloc.bind(src_ref=(q_hbm_ref, schedule_ref))
        o_bound = o_alloc.bind(dst_ref=(o_hbm_ref, schedule_ref))
        lse_bound = (
            lse_alloc.bind(dst_ref=(lse_hbm_ref, schedule_ref))
            if return_lse
            else None
        )

        def compute_fn(
            step,
            slot,
            total_steps,
            out_slot,
            start_next_fetch,
            start_kv_writeback,
        ):
          del total_steps
          rpa_body(
              step,
              slot,
              q_alloc.window_ref.at[slot],
              kv_cache_alloc.window_ref.at[slot],
              o_alloc.window_ref.at[out_slot],
              lse_alloc.window_ref.at[out_slot] if return_lse else None,
              schedule_ref,
              scratches[0],
              scratches[1],
              scratches[2],
              attention_sink_ref=attention_sink_ref,
              cfgs=cfgs,
              start_next_fetch=start_next_fetch,
              start_kv_writeback=start_kv_writeback,
          )

        pipeline.run_manual_triple_buffer(
            total_steps,
            compute_fn,
            q_bound,
            o_bound,
            kv_bound,
            lse_bref=lse_bound,
        )

      @pl.loop(0, num_safe_step_iterations)
      def loop_body(step_idx):
        start = step_idx * max_steps_ub
        rem = actual_steps % max_steps_ub
        last_step_size = jnp.where(rem == 0, max_steps_ub, rem)
        is_last_step = step_idx == num_safe_step_iterations - 1
        size = jnp.where(is_last_step, last_step_size, max_steps_ub)
        execute_schedule_chunk(start, size)

    _run()

  out_shape = [q_hbm, kv_cache_hbm, lse_hbm if return_lse else None]

  schedule_leaves = len(jax.tree_util.tree_leaves(schedule_hbm))
  input_output_aliases = {
      schedule_leaves: 0,
      schedule_leaves + 2: 1,
  }
  if return_lse:
    input_output_aliases[schedule_leaves + 3] = 2

  return pl.pallas_call(
      ragged_paged_attention_pipeline,
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              schedule_hbm.in_specs(),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM) if return_lse else None,
              pl.BlockSpec(memory_space=pltpu.VMEM)
              if attention_sink_hbm is not None
              else None,
          ],
          out_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM) if return_lse else None,
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=cfgs.vmem_limit_bytes,
          disable_bounds_checks=True,
          flags={
              "XLA_TPU_EVENLY_DISTRIBUTE_CRITICAL_NODES": True,
          },
      ),
      input_output_aliases=input_output_aliases,
      name=get_kernel_name(cfgs),
      metadata=typing.cast(typing.Any, get_kernel_metadata(cfgs)),
  )(
      schedule_hbm,
      q_hbm,
      new_kv_hbm,
      kv_cache_hbm,
      lse_hbm if return_lse else None,
      None
      if attention_sink_hbm is None
      else attention_sink_hbm.astype(cfgs.layout.softmax_dtype),
  )
