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
"""Wrapper for MLA v3 kernel to orchestrate multi-phase batch execution."""

from absl import logging
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.mla.v3 import configs
from tokamax._src.ops.experimental.mla.v3 import kernel
from tokamax._src.ops.experimental.mla.v3 import schedule

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def calculate_block_sizes(
    serve_cfgs: configs.ServingConfigs,
    *,
    num_kv_pages_per_block: int = 2,
    num_queries_per_block: int = 4,
    batch_size: int = 2,
    n_buffer: int = 2,
    q_split: int = 1,
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
  """Calculates default block sizes for decode and prefill/mixed passes.

  `q_split` divides the prefill/mixed query block into that many sub-chunks.
  """
  page_size = serve_cfgs.page_size
  bkv_sz = num_kv_pages_per_block * page_size

  decode_blocks = configs.BlockSizes(
      bq_sz=1,
      bq_c_sz=1,
      bkv_sz=bkv_sz,
      batch_size=batch_size,
      n_buffer=n_buffer,
  )
  prefill_blocks = configs.BlockSizes(
      bq_sz=num_queries_per_block,
      bq_c_sz=max(1, num_queries_per_block // max(1, q_split)),
      bkv_sz=bkv_sz,
      batch_size=1,
      n_buffer=n_buffer,
  )
  return decode_blocks, prefill_blocks


# Canonical pass order. Later passes must see earlier passes' writes, so the
# order is part of the contract, not just an iteration detail.
ALL_MODES = (
    configs.MlaCase.DECODE,
    configs.MlaCase.PREFILL,
    configs.MlaCase.MIXED,
)


def mode_configs(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int = 2,
    num_queries_per_block: int = 4,
    batch_size: int = 2,
    n_buffer: int = 2,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    q_split: int = 1,
    kv_layout: configs.KVLayout = configs.KVLayout.SEQ_ALONG_LANE,
    modes: tuple[configs.MlaCase, ...] = ALL_MODES,
) -> dict[configs.MlaCase, configs.MlaConfigs]:
  """Returns the `MlaConfigs` each pass runs under, keyed by mode.

  Shared by `mla_ragged_paged_attention` and `build_schedules` so a caller that
  precomputes schedules cannot drift from the configs the kernel then runs
  under -- a mismatch would silently produce a schedule for different block
  sizes.

  `modes` selects which passes exist at all. A caller that knows a band is
  always empty can drop it here and the pass vanishes from the graph -- no
  schedule built, no kernel launched. That is strictly better than leaving it
  in: an empty band still costs a schedule build plus a zero-trip
  `pallas_call`. At least one mode must be selected; DECODE need not be one.
  """
  actual_num_q_heads, total_q_tokens, actual_lkv_dim = ql_nope.shape
  actual_r_dim = q_pe.shape[-1]
  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]

  if vmem_limit_bytes is None:
    vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  # [pages, kv_dim, page_size] vs [pages, page_size, kv_dim]
  page_size = cache_kv.shape[-1]

  model_cfgs = configs.MlaModelConfigs(
      num_q_heads=actual_num_q_heads,
      lkv_dim=actual_lkv_dim,
      r_dim=actual_r_dim,
      mask_value=mask_value,
      sm_scale=sm_scale,
      soft_cap=soft_cap,
      sliding_window=sliding_window,
  )
  serve_cfgs = configs.ServingConfigs(
      num_seqs=max_num_seqs,
      num_page_indices=num_page_indices,
      total_q_tokens=total_q_tokens,
      dtype_q=ql_nope.dtype,
      dtype_kv=cache_kv.dtype,
      dtype_out=ql_nope.dtype,
      page_size=page_size,
      scale_q=q_scale,
      scale_k=k_scale,
      scale_v=v_scale,
      kv_layout=kv_layout,
  )
  default_decode, default_prefill = calculate_block_sizes(
      serve_cfgs,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      batch_size=batch_size,
      n_buffer=n_buffer,
      q_split=q_split,
  )

  out = {}
  for mode in ALL_MODES:
    if mode not in modes:
      continue
    if mode == configs.MlaCase.DECODE:
      blocks = decode_block_sizes or default_decode
    else:
      blocks = prefill_block_sizes or default_prefill
    out[mode] = configs.MlaConfigs(
        block=blocks,
        model=model_cfgs,
        serve=serve_cfgs,
        vmem_limit_bytes=vmem_limit_bytes,
        mode=mode,
    )
  if not out:
    raise ValueError(f"{modes=} must include at least one of {ALL_MODES}.")
  return out


def build_schedules(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    distribution: jax.Array,
    cfgs_by_mode: dict[configs.MlaCase, configs.MlaConfigs],
) -> dict[configs.MlaCase, schedule.MlaSchedule]:
  """Builds each pass's schedule once, for reuse across all attention layers."""
  return {
      mode: schedule.generate_mla_metadata(
          cu_q_lens, kv_lens, page_indices, distribution, cfgs=cfgs
      )
      for mode, cfgs in cfgs_by_mode.items()
  }


def mla_ragged_paged_attention(
    ql_nope: jax.Array,  # [actual_num_q_heads, max_num_tokens, actual_lkv_dim] or [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
    q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
    cache_kv: jax.Array,  # [total_num_pages, aligned_kv_dim, page_size]
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int = 2,
    num_queries_per_block: int = 4,
    batch_size: int = 2,
    n_buffer: int = 2,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    q_split: int = 1,
    kv_layout: configs.KVLayout = configs.KVLayout.SEQ_ALONG_LANE,
    schedules: dict[configs.MlaCase, schedule.MlaSchedule] | None = None,
    modes: tuple[configs.MlaCase, ...] = ALL_MODES,
    debug_mode: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """MLA Ragged paged attention, orchestrated as Decode -> Prefill -> Mixed.

  `schedules` lets the caller supply metadata built by `build_schedules`. The
  schedule depends only on `cu_q_lens`, `kv_lens`, `page_indices`,
  `distribution` and the block sizes -- none of which vary across the model's
  attention layers -- so leaving it to be built here runs the
  `mla_metadata_schedule` kernel once per layer. XLA does not CSE the Pallas
  call, so on a 61-layer model that is 61 launches per step instead of one.
  Left as None it is built inline, which keeps the standalone call sites
  working.
  """
  # Before anything below unpacks a shape, so a wrong rank gets a message
  # rather than a tuple-unpacking error.
  kernel.validate_input_ranks(
      ql_nope,
      q_pe,
      new_kv_c,
      new_k_pe,
      cache_kv,
      kv_lens,
      page_indices,
      cu_q_lens,
  )
  actual_num_q_heads, total_q_tokens, actual_lkv_dim = ql_nope.shape

  if vmem_limit_bytes is None:
    vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

  cfgs_by_mode = mode_configs(
      ql_nope,
      q_pe,
      cache_kv,
      kv_lens,
      page_indices,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      q_scale=q_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      batch_size=batch_size,
      n_buffer=n_buffer,
      decode_block_sizes=decode_block_sizes,
      prefill_block_sizes=prefill_block_sizes,
      vmem_limit_bytes=vmem_limit_bytes,
      q_split=q_split,
      kv_layout=kv_layout,
      modes=modes,
  )

  # Every pass, not just DECODE: `run_mla_kernel` calls the kernel directly,
  # which skips validation, and the passes can have different block sizes.
  for pass_cfgs in cfgs_by_mode.values():
    kernel.static_validate_inputs(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        cfgs=pass_cfgs,
    )

  ql_nope_prep = kernel.prepare_q_nope_inputs(
      ql_nope,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  # All three stay at 128-lane alignment. `q_pe` in particular *must*: it is
  # reshaped with `aligned_r_dim` as its minor dimension, which is the lane
  # axis. See `MlaConfigs.kv_dim_align`. It is the same for every mode.
  head_align = next(iter(cfgs_by_mode.values())).kv_dim_align
  q_pe_prep = kernel.prepare_q_pe_inputs(q_pe, head_align=head_align)
  new_kv_c_prep = kernel.prepare_kv_inputs_for_transposed_kv_cache(
      new_kv_c,
      head_align=head_align,
  )
  new_k_pe_prep = kernel.prepare_kv_inputs_for_transposed_kv_cache(
      new_k_pe,
      head_align=head_align,
  )

  def run_mla_kernel(
      mode: configs.MlaCase,
      ql_nope_in: jax.Array,
      cache_kv_in: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    cfgs = cfgs_by_mode[mode]
    if debug_mode:
      logging.info("blocks: %s, mode: %s", cfgs.block, mode)
    if schedules is not None and mode in schedules:
      schedule_hbm = schedules[mode]
    else:
      schedule_hbm = schedule.generate_mla_metadata(
          cu_q_lens, kv_lens, page_indices, distribution, cfgs=cfgs
      )
    return kernel._mla_ragged_paged_attention_kernel(
        cu_q_lens,
        kv_lens,
        schedule_hbm,
        ql_nope_in,
        q_pe_prep,
        new_kv_c_prep,
        new_k_pe_prep,
        cache_kv_in,
        cfgs=cfgs,
    )

  if debug_mode:
    logging.info(
        "Prepared inputs for MLA: ql_nope=%s, q_pe=%s, new_kv_c=%s,"
        " new_k_pe=%s, cache_kv=%s",
        ql_nope_prep.shape,
        q_pe_prep.shape,
        new_kv_c_prep.shape,
        new_k_pe_prep.shape,
        cache_kv.shape,
    )

  # `distribution` splits the batch into decode / prefill / mixed bands. Each
  # gets its own pass, chaining `ql_nope_prep` and `cache_kv` so later passes
  # see earlier writes.
  o_hbm = ql_nope_prep
  for mode in ALL_MODES:
    if mode in cfgs_by_mode:
      o_hbm, cache_kv = run_mla_kernel(mode, o_hbm, cache_kv)

  output = kernel.prepare_outputs(
      o_hbm,
      actual_num_q_heads,
      total_q_tokens,
      actual_lkv_dim,
      vmem_limit_bytes=vmem_limit_bytes,
  )

  return output, cache_kv
