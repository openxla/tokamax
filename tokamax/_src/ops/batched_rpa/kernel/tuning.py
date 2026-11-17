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
"""Hardware VMEM/MXU cost models and block-size selection for Batched RPA."""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from tokamax._src.ops.batched_rpa.kernel import configs
from tokamax._src.ops.batched_rpa.kernel import utils

# Prefill tile caps on SFS. EMPIRICAL, not derived: the VMEM-driven search in
# find_best_block_sizes picks bkv=2304 on SFS, which is well past the measured
# optimum, because prefill uses only ~6% of HBM bandwidth and a larger tile buys
# no bandwidth relief while causal-mask waste and the bq_sz * bkv_sz softmax
# intermediate both grow. The optimum depends on matmul precision.
# Measured kernel latency, Qwen3.5, TP=8, one 8500-token sequence,
# page_size=256, fp8 kv cache, prefill batch_size=2:
#
#   bkv            512       1024      2048      2304 (uncapped pick)
#   bf16, c512     366.2 us  414.5 us  480.3 us  536.6 us
#   fp8,  c512     266.4 us  250.2 us  288.7 us  458.4 us
#   fp8,  c128     259.2 us  244.1 us  237.1 us
#
# (c128 is what the threshold below picks for fp8 on this shape; the c512 rows
# are from the earlier hand-tuned sweep.) Batch size at a fixed tile is flat:
# fp8 c128 bkv=1024 gives 240.1 / 244.1 / 240.6 us at batch_size 1 / 2 / 4, and
# bkv=512 at batch_size 4 (same VMEM as bkv=1024 at 2) is 252.9 us, so more
# batch is not a substitute for a bigger kv tile.
#
# Caps must be multiples of the bkv stride (256 here). Other chips are
# unmeasured and stay uncapped.
# TODO: the cap is not the optimum at c128: bkv=1536 / 2048 at
# batch_size=1 measured 225.1 / 229.0 us (-8% / -6% vs this cap), while
# bkv=2048 at batch_size=4 is 281.2 us and bkv=4096 at 1 is 316.1 us, so the
# surface has cliffs. Revisit the cap and prefill batch_size together, on
# multi-sequence prefill and the other models, before moving either; and
# re-measure when the per-block cost of the kernel changes (b/503493392
# softmax work).
SFS_PREFILL_MAX_BQ_SZ = 512
SFS_PREFILL_MAX_BKV_SZ_8BIT = 1024
SFS_PREFILL_MAX_BKV_SZ_16BIT = 512


def is_sfs(tpu_info) -> bool:
  """Returns True on TPU v8i."""
  # Accepts an enum or a string, since unit tests mock chip_version with a
  # plain string and a strict `==` against the enum would never match.
  return tpu_info.generation == 8 and tpu_info.chip_version in (
      pltpu.ChipVersion.TPU_8I,
      "v8i",
      "8i",
  )


def prefill_tile_caps(
    serve_cfgs: configs.ServingConfigs, tpu_info
) -> tuple[int | None, int | None]:
  """Returns (max_bq_sz, max_bkv_sz) for prefill tile search, or (None, None)."""
  if not is_sfs(tpu_info):
    return None, None
  if serve_cfgs.matmuls_are_8bit:
    return SFS_PREFILL_MAX_BQ_SZ, SFS_PREFILL_MAX_BKV_SZ_8BIT
  return SFS_PREFILL_MAX_BQ_SZ, SFS_PREFILL_MAX_BKV_SZ_16BIT


def calculate_vmem_usage(
    layout: configs.LayoutConfig,
    serve_cfgs: configs.ServingConfigs,
    batch_size: int,
    bq_sz: int,
    bkv_sz: int,
    *,
    num_input_slots: int = 3,
    num_output_slots: int = 2,
) -> float:
  """Given tile size, calculate VMEM usage of the kernel in bytes."""
  aligned_head_dim = layout.aligned_q_head_dim
  aligned_num_q_heads = layout.aligned_num_q_heads

  q_bytes = jax.dtypes.itemsize_bits(layout.dtype_q) / 8
  kv_bytes = jax.dtypes.itemsize_bits(layout.dtype_kv) / 8
  out_bytes = jax.dtypes.itemsize_bits(layout.dtype_out) / 8

  # Step 1: Calculate buffer sizes.
  bq_array_size = bq_sz * aligned_num_q_heads * aligned_head_dim
  vmem_bkv_sz = (
      layout.kv_vmem_lanes(bkv_sz) if layout.is_seq_along_lane else bkv_sz
  )
  bkv_array_size = (
      vmem_bkv_sz * layout.vmem_kv_heads_x2 * layout.storage_kv_head_dim
  )
  bo_array_size = bq_array_size

  bq_bytes = bq_array_size * q_bytes * num_input_slots
  bkv_bytes = bkv_array_size * kv_bytes * num_input_slots
  bo_bytes = bo_array_size * out_bytes * num_output_slots

  # Sum up all buffer memory usage (scales are co-located inside bkv_bytes).
  buffer_bytes = bq_bytes + bkv_bytes + bo_bytes

  if serve_cfgs.return_lse:
    lse_array_size = bq_sz * aligned_num_q_heads * layout.num_lanes
    lse_bytes = lse_array_size * out_bytes * num_output_slots
    buffer_bytes += lse_bytes

  # Step 2: Calculate worst case memory usage during computation.
  loaded_bq_size = bq_sz * layout.num_q_heads * aligned_head_dim
  loaded_bkv_size = bkv_sz * layout.num_kv_heads * aligned_head_dim
  qk_size = bq_sz * bkv_sz * layout.num_q_heads

  loaded_bq_bytes = loaded_bq_size * q_bytes
  loaded_bkv_bytes = loaded_bkv_size * kv_bytes
  qk_bytes = qk_size * out_bytes

  # Active unpacked k_scale and v_scale each have shape (num_kv_heads, bkv_sz).
  loaded_scale_bytes = 0
  if layout.per_token_scale:
    assert layout.per_token_scale_dtype is not None
    scale_bits = jax.dtypes.itemsize_bits(layout.per_token_scale_dtype)
    scale_bytes = scale_bits // 8
    loaded_scale_bytes = 2 * bkv_sz * layout.num_kv_heads * scale_bytes

  compute_bytes = (
      loaded_bq_bytes + loaded_bkv_bytes + qk_bytes + loaded_scale_bytes
  )

  # Step 3: Sum up all memory usage across batch lanes + online-softmax scratch.
  total_bytes = (buffer_bytes + compute_bytes) * batch_size

  lm_array_size = (
      layout.num_kv_heads * bq_sz * layout.aligned_num_q_heads_per_kv_head
  )
  acc_array_size = (
      layout.num_kv_heads * bq_sz * layout.aligned_num_q_heads_per_kv_head
  )
  scratch_bytes = (2 * lm_array_size + acc_array_size) * out_bytes
  total_bytes += scratch_bytes

  return total_bytes


def calculate_compute_buffer_time(
    model_cfgs: configs.ModelConfigs,
    mxu_column_size: int,
    batch_size: int,
    bq_c_sz: int,
    bkv_sz: int,
) -> int:
  """Calculate computational complexity of a single compute block."""
  num_k_rows = pl.cdiv(bkv_sz, mxu_column_size)
  num_k_cols = pl.cdiv(model_cfgs.head_dim, mxu_column_size)
  num_k = num_k_rows * num_k_cols
  num_muls = bq_c_sz * num_k * model_cfgs.num_q_heads
  return batch_size * num_muls


def find_best_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    layout: configs.LayoutConfig,
    vmem_limit_bytes: int,
    *,
    max_batch_size: int,
    fixed_bq_sz: int | None = None,
    max_bkv_sz: int | None = None,
    max_bq_sz: int | None = None,
    num_input_slots: int = 3,
    num_output_slots: int = 2,
) -> configs.BlockSizes:
  """Sweep candidate block sizes under an 80% VMEM budget."""
  tpu_info = pltpu.get_tpu_info()
  mxu_column_size = tpu_info.mxu_column_size

  # Conservatively use 80% of the VMEM budget to prevent OOM.
  capped_vmem_limit_bytes = vmem_limit_bytes * 0.8

  # To maximize MXU, we want bkv_sz to be at least MXU size.
  bkv_default = mxu_column_size
  if layout.is_seq_along_lane:
    # SEQ_ALONG_LANE expects bkv_sz to be page-aligned.
    bkv_default = utils.align_to(bkv_default, serve_cfgs.page_size)
  bkv_sz = bkv_stride = bkv_default
  if fixed_bq_sz is None:
    bq_sz = bq_stride = bkv_sz
  else:
    bq_sz = fixed_bq_sz
    bq_stride = 0
  batch_size = max_batch_size

  def _vmem_bytes(b: int, q: int, kv: int) -> float:
    return calculate_vmem_usage(
        layout,
        serve_cfgs,
        b,
        q,
        kv,
        num_input_slots=num_input_slots,
        num_output_slots=num_output_slots,
    )

  # Step 1: Lower batch_size if even the smallest bq and bkv trigger OOM.
  while (
      batch_size > 1
      and _vmem_bytes(batch_size, bq_sz, bkv_sz) > capped_vmem_limit_bytes
  ):
    batch_size -= 1

  if batch_size == 0:
    raise ValueError("Cannot find batch size that fits within VMEM limit.")

  # Step 2: Increase block sizes until the kernel exceeds VMEM.
  max_seq_len = serve_cfgs.pages_per_seq * serve_cfgs.page_size
  while (
      _vmem_bytes(batch_size, bq_sz, bkv_sz) < capped_vmem_limit_bytes
      and bkv_sz <= max_seq_len
      and bkv_sz <= 4096
  ):
    bkv_sz += bkv_stride
    bq_sz += bq_stride

  # Rollback one step since the last attempted value triggered OOM.
  if bkv_sz > bkv_stride:
    bkv_sz -= bkv_stride
    if fixed_bq_sz is None:
      bq_sz -= bq_stride

  # Apply chip-specific caps after rollback (caps are multiples of bkv_stride).
  if max_bkv_sz is not None and bkv_sz > max_bkv_sz:
    bkv_sz = max_bkv_sz
    if fixed_bq_sz is None:
      bq_sz = max_bkv_sz
  if fixed_bq_sz is None and max_bq_sz is not None and bq_sz > max_bq_sz:
    bq_sz = max_bq_sz

  if bkv_sz == 0:
    raise ValueError("Cannot find block sizes that fit within VMEM limit.")

  # Step 2b: When many Q heads bound bq at 256, keep growing bkv up to 512.
  if fixed_bq_sz is None:
    step2b_max_bkv_sz = 512
    if max_bkv_sz is not None:
      step2b_max_bkv_sz = min(step2b_max_bkv_sz, max_bkv_sz)
    while (
        bkv_sz + bkv_stride <= max_seq_len
        and bkv_sz + bkv_stride <= step2b_max_bkv_sz
        and _vmem_bytes(batch_size, bq_sz, bkv_sz + bkv_stride)
        < capped_vmem_limit_bytes
    ):
      bkv_sz += bkv_stride

  # Step 3: Calculate compute sub-chunk size bq_c_sz (prefill only).
  bq_c_sz = bq_sz
  if fixed_bq_sz is None:
    match tpu_info.generation:
      case 8 if is_sfs(tpu_info):
        threshold = 800
      case 7:
        threshold = 1500
      case _:
        threshold = 1500

    if serve_cfgs.matmuls_are_8bit:
      flops_ratio = tpu_info.fp8_ops_per_second // tpu_info.bf16_ops_per_second
      threshold *= flops_ratio

    num_bq_c = 1
    bq_c_rem = 0
    while (
        calculate_compute_buffer_time(
            model_cfgs, mxu_column_size, batch_size, bq_c_sz, bkv_sz
        )
        > threshold
        or bq_c_rem != 0
    ) and num_bq_c < bq_sz:
      num_bq_c += 1
      bq_c_sz, bq_c_rem = divmod(bq_sz, num_bq_c)

  return configs.BlockSizes(
      bq_sz=bq_sz,
      bq_c_sz=bq_c_sz,
      bkv_sz=bkv_sz,
      batch_size=batch_size,
      num_input_slots=num_input_slots,
      num_output_slots=num_output_slots,
  )


def calculate_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
    decode_query_size: int = 1,
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
  """Calculate optimal block sizes for decode and prefill."""
  tpu_info = pltpu.get_tpu_info()
  layout = configs.LayoutConfig.from_configs(
      model_cfgs,
      serve_cfgs,
      num_lanes=tpu_info.num_lanes,
      num_sublanes=tpu_info.num_sublanes,
  )

  decode_batch_size = 8
  prefill_batch_size = 2

  decode_block_sizes = find_best_block_sizes(
      model_cfgs,
      serve_cfgs,
      layout,
      vmem_limit_bytes,
      max_batch_size=decode_batch_size,
      fixed_bq_sz=decode_query_size,
  )
  prefill_max_bq_sz, prefill_max_bkv_sz = prefill_tile_caps(
      serve_cfgs, tpu_info
  )
  prefill_block_sizes = find_best_block_sizes(
      model_cfgs,
      serve_cfgs,
      layout,
      vmem_limit_bytes,
      max_batch_size=prefill_batch_size,
      max_bkv_sz=prefill_max_bkv_sz,
      max_bq_sz=prefill_max_bq_sz,
  )

  return decode_block_sizes, prefill_block_sizes
