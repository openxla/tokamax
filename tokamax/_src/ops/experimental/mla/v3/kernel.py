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
"""TPU-Friendly MLA Ragged Paged Attention kernel v3 (Transposed KV Cache)."""

import dataclasses
import functools
from typing import Any

from absl import logging
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.mla.v2 import transpose as v2_transpose
from tokamax._src.ops.experimental.mla.v3 import bref_override
from tokamax._src.ops.experimental.mla.v3 import configs
from tokamax._src.ops.experimental.mla.v3 import flash_attention
from tokamax._src.ops.experimental.mla.v3 import schedule
from tokamax._src.ops.experimental.mla.v3 import stitch_utils
from tokamax._src.ops.experimental.mla.v3 import utils

MlaCase = configs.MlaCase

# Tiles for the layout-transpose helper (v2 `xpose_pipeline`), which runs
# outside the attention kernel to move q_nope into and the output out of
# token-major layout. These are not attention-kernel tunables, and
# `xpose_pipeline` snaps them down to valid sublane-aligned divisors. The
# output-side values are carried over from v2.
_Q_XPOSE_N_TILE = 128
_Q_XPOSE_M_TILE = 32
_OUT_XPOSE_N_TILE = 160
_OUT_XPOSE_M_TILE = 64

# ==============================================================================
# Input Validation & Preparation (Transposed KV Cache Only)
# ==============================================================================


def validate_input_ranks(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
) -> None:
  """Validates input ranks only, so it needs no configs.

  Split out of `static_validate_inputs` so `mla_wrapper` can run it before it
  unpacks the input shapes to build the configs.
  """
  if len(ql_nope.shape) != 3:
    raise ValueError(f"Expected 3D array for {ql_nope.shape=}")
  if len(q_pe.shape) != 3:
    raise ValueError(f"Expected 3D array for {q_pe.shape=}")
  if len(new_kv_c.shape) != 2:
    raise ValueError(f"Expected 2D array for {new_kv_c.shape=}")
  if len(new_k_pe.shape) != 2:
    raise ValueError(f"Expected 2D array for {new_k_pe.shape=}")
  if cache_kv.ndim != 3:
    raise ValueError(
        "Expected 3D array [pages, aligned_kv_dim, page_size] for cache_kv,"
        f" got {cache_kv.shape=}"
    )
  if not (
      len(kv_lens.shape) == len(page_indices.shape) == len(cu_q_lens.shape) == 1
  ):
    raise ValueError(
        f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
        f" {cu_q_lens.shape=}"
    )


def static_validate_inputs(
    ql_nope: jax.Array,  # [actual_num_q_heads, max_num_tokens, actual_lkv_dim]
    q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
    cache_kv: jax.Array,  # [total_num_pages, aligned_kv_dim, page_size]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    cfgs: configs.MlaConfigs,
) -> None:
  """Statically validates input shapes, dtypes, and 3D transposed layout constraints."""
  validate_input_ranks(
      ql_nope,
      q_pe,
      new_kv_c,
      new_k_pe,
      cache_kv,
      kv_lens,
      page_indices,
      cu_q_lens,
  )

  if ql_nope.shape[0] != q_pe.shape[1]:
    raise ValueError(
        f"Expected ql_nope num_heads {ql_nope.shape[0]=} to equal"
        f" q_pe num_heads {q_pe.shape[1]=}"
    )
  if ql_nope.shape[1] != q_pe.shape[0]:
    raise ValueError(
        f"Expected ql_nope num_tokens {ql_nope.shape[1]=} to equal"
        f" q_pe num_tokens {q_pe.shape[0]=}"
    )
  if ql_nope.shape[1] != new_kv_c.shape[0]:
    raise ValueError(
        f"Expected {ql_nope.shape[1]=} to be equal to {new_kv_c.shape[0]=}"
    )
  if new_kv_c.shape[0] != new_k_pe.shape[0]:
    raise ValueError(
        f"Expected {new_kv_c.shape[0]=} to be equal to {new_k_pe.shape[0]=}"
    )
  if ql_nope.shape[2] != new_kv_c.shape[1]:
    raise ValueError(
        f"Expected {ql_nope.shape[2]=} to be equal to {new_kv_c.shape[1]=}"
    )
  if q_pe.shape[2] != new_k_pe.shape[1]:
    raise ValueError(
        f"Expected {q_pe.shape[2]=} to be equal to {new_k_pe.shape[1]=}"
    )

  _, kv_dim, page_size = cache_kv.shape

  num_lanes = utils.get_tpu_num_lanes()
  if page_size % num_lanes != 0:
    raise ValueError(
        f"Expected {page_size=} to be a multiple of {num_lanes=}: tokens sit on"
        " the lane axis."
    )

  if kv_dim != cfgs.aligned_kv_dim:
    raise ValueError(
        f"cache_kv {kv_dim=} does not match {cfgs.aligned_kv_dim=}"
        f" (= aligned_lkv_dim {cfgs.aligned_lkv_dim} + aligned_r_dim"
        f" {cfgs.aligned_r_dim})"
    )

  if cache_kv.dtype != new_kv_c.dtype:
    raise ValueError(
        f"Expected {cache_kv.dtype=} to be equal to {new_kv_c.dtype=}."
    )
  if cache_kv.dtype != new_k_pe.dtype:
    raise ValueError(
        f"Expected {cache_kv.dtype=} to be equal to {new_k_pe.dtype=}."
    )

  if not jnp.issubdtype(cache_kv.dtype, jnp.floating):
    raise ValueError(f"Expected {cache_kv.dtype=} to be a floating point.")

  if not (
      jnp.int32
      == kv_lens.dtype
      == page_indices.dtype
      == cu_q_lens.dtype
      == distribution.dtype
  ):
    raise ValueError(
        f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
        f" {cu_q_lens.dtype=}, {distribution.dtype=}"
    )

  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]
  if num_page_indices % max_num_seqs != 0:
    raise ValueError(
        f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
    )
  if cu_q_lens.shape != (max_num_seqs + 1,):
    raise ValueError(
        f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)."
    )
  if distribution.shape != (3,):
    raise ValueError(f"Expected {distribution.shape=} to be (3,).")

  # `page_size_log2` implements `//` as a shift, which is only equivalent for
  # powers of two. A multiple of 128 is not enough (e.g. 384).
  if page_size & (page_size - 1) != 0:
    raise ValueError(f"Expected {page_size=} to be a power of two.")

  if cfgs.serve.page_size != page_size:
    raise ValueError(
        f"Config {cfgs.serve.page_size=} disagrees with the cache_kv page"
        f" dimension {page_size=}."
    )

  # `bkv_p` is `cdiv(bkv_sz, page_size)`, so a non-multiple would silently round
  # the block up and make every DMA offset computed from `bkv_p * page_size`
  # disagree with the `bkv_sz` the softmax masks against.
  if cfgs.bkv_sz % page_size != 0:
    raise ValueError(
        f"Expected {cfgs.bkv_sz=} to be a multiple of {page_size=}."
    )

  # `BatchingQRef.wait_in` and `BatchingORef.wait_out` wait on a u32 view with
  # `aligned_num_q_heads * itemsize // 4` rows per token, which must be a whole
  # number of 8-row sublane tiles (they assert it). Reject here with a message
  # instead. That is a multiple of 16 heads for bf16, 32 for 8-bit types.
  for name, dtype in (
      ("dtype_q", cfgs.serve.dtype_q),
      ("dtype_out", cfgs.serve.dtype_out),
  ):
    itemsize = jnp.dtype(dtype).itemsize
    if (cfgs.aligned_num_q_heads * itemsize) % 32 != 0:
      raise ValueError(
          f"{cfgs.aligned_num_q_heads=} is unsupported for"
          f" {name}={jnp.dtype(dtype).name}: the Q/O DMA waits need a multiple"
          f" of {32 // itemsize} heads."
      )

  if cfgs.model.sliding_window is not None and cfgs.model.sliding_window <= 0:
    raise ValueError(f"{cfgs.model.sliding_window=} must be positive.")
  if cfgs.model.soft_cap is not None and cfgs.model.soft_cap == 0.0:
    raise ValueError(f"{cfgs.model.soft_cap=} must not be 0.0.")
  if cfgs.vmem_limit_bytes <= 0:
    raise ValueError(f"{cfgs.vmem_limit_bytes=} must be positive.")


def prepare_q_pe_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
    head_align: int | None = None,
) -> jax.Array:
  """Pads q to [max_num_tokens, num_q_heads, head_dim]."""
  # `head_align` must match `MlaConfigs.kv_dim_align` for q_pe: the QK-PE dot
  # contracts q_pe against k_pe over this dimension, so the two must agree.
  # Both default to the lane count.
  if head_align is None:
    head_align = utils.get_tpu_num_lanes()
  _, actual_num_q_heads, actual_head_dim = q.shape
  packing_q = utils.get_dtype_packing(q.dtype)
  num_q_heads = utils.align_to(actual_num_q_heads, packing_q)
  head_dim = utils.align_to(actual_head_dim, head_align)
  return jnp.pad(
      q,
      (
          (0, 0),
          (0, num_q_heads - actual_num_q_heads),
          (0, head_dim - actual_head_dim),
      ),
      constant_values=0,
  )


def _physical_transpose(
    x: jax.Array,
    *,
    n_tile: int,
    m_tile: int,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
  """(N, T, D) <-> (T, N, D) via v2's pipelined transpose kernel.

  A plain `jnp.transpose` feeding a Pallas call cannot be folded into the
  producing fusion -- Pallas fixes its operand layouts -- so XLA materializes a
  layout-changing copy of the whole tensor, once per attention layer on both the
  q_nope and the output side. `xpose_pipeline` instead tiles and double-buffers
  the movement, which is why v2 treats `jnp.transpose` as the fallback it warns
  about (v2/kernel.py `prepare_q_nope_inputs` / `prepare_outputs`).

  `vmem_limit_bytes` must be forwarded: left as None the transpose is capped at
  XLA's scoped-VMEM default (32 MiB), which the double-buffered in/out tiles
  (e.g. 160x64x512 bf16 on the output side) exceed at compile time -- a
  compile error, not a ValueError, so the fallback below would not catch it.

  Falls back on ValueError for the same reason v2 does: the kernel rejects
  shapes it cannot tile.
  """
  try:
    return v2_transpose.xpose_pipeline(
        x,
        transpose_axes=(1, 0, 2),
        n_tile=n_tile,
        m_tile=m_tile,
        vmem_limit_bytes=vmem_limit_bytes,
    )[0]
  except ValueError as e:
    logging.warning(
        "xpose_pipeline failed for shape=%s dtype=%s: %s. Falling back to"
        " jnp.transpose -- this materializes a full copy per layer.",
        x.shape,
        x.dtype,
        e,
    )
    return jnp.transpose(x, (1, 0, 2))


def prepare_q_nope_inputs(
    q: jax.Array,  # [actual_num_q_heads, max_num_tokens, actual_head_dim]
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
  """Pads and physically transposes q_nope to token-major layout.

  Returns: [max_num_tokens, num_q_heads, head_dim], kept 3D for the same
  reason as `prepare_q_pe_inputs`.
  """
  actual_num_q_heads, _, actual_head_dim = q.shape
  packing_q = utils.get_dtype_packing(q.dtype)
  num_q_heads = utils.align_to(actual_num_q_heads, packing_q)
  head_dim = utils.align_to(actual_head_dim, utils.get_tpu_num_lanes())

  head_pad = (0, num_q_heads - actual_num_q_heads)
  dim_pad = (0, head_dim - actual_head_dim)
  q = jnp.pad(
      q,
      (head_pad, (0, 0), dim_pad),
      constant_values=0,
  )
  return _physical_transpose(
      q,
      n_tile=_Q_XPOSE_N_TILE,
      m_tile=_Q_XPOSE_M_TILE,
      vmem_limit_bytes=vmem_limit_bytes,
  )


def prepare_kv_inputs_for_transposed_kv_cache(
    kv: jax.Array,
    head_align: int | None = None,
) -> jax.Array:
  """Pads new KV inputs to match the cache layout.

  The cache wants [aligned_head_dim, max_num_tokens], so the incoming
  [tokens, head_dim] is transposed.
  """
  max_num_tokens, actual_head_dim = kv.shape
  num_lanes = utils.get_tpu_num_lanes()
  if head_align is None:
    head_align = num_lanes

  # The lane tile, not the page size (see `MlaConfigs.kv_token_align`). The
  # kernel fetches new KV as one lane-aligned span ending at most at
  # align_to(num_tokens, num_lanes), so nothing reads past that. Padding a
  # 4-token decode bucket to a 1024-token page was a [640, 1024] pad+transpose
  # per layer for 4 real tokens.
  pad_multiple = num_lanes
  if max_num_tokens % pad_multiple != 0:
    pad = pad_multiple - (max_num_tokens % pad_multiple)
    kv = jnp.pad(kv, ((0, pad), (0, 0)), constant_values=0)

  aligned_head_dim = utils.align_to(actual_head_dim, head_align)
  if aligned_head_dim != actual_head_dim:
    pad = aligned_head_dim - actual_head_dim
    kv = jnp.pad(kv, ((0, 0), (0, pad)), constant_values=0)

  return jnp.transpose(kv, (1, 0))


def prepare_outputs(
    out: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
    actual_num_q_heads: int,
    actual_max_num_tokens: int,
    actual_head_dim: int,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
  """Physically transposes output activations back to head-major layout."""
  packing_q = utils.get_dtype_packing(out.dtype)
  num_q_heads = utils.align_to(actual_num_q_heads, packing_q)
  head_dim = utils.align_to(actual_head_dim, utils.get_tpu_num_lanes())
  out = out.reshape((out.shape[0], num_q_heads, head_dim))
  out = _physical_transpose(
      out,
      n_tile=_OUT_XPOSE_N_TILE,
      m_tile=_OUT_XPOSE_M_TILE,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  return out[:actual_num_q_heads, :actual_max_num_tokens, :actual_head_dim]


# ==============================================================================
# Pallas Allocation & Compute Pipeline
# ==============================================================================


def create_allocs(
    cache_kv_hbm_ref: jax.Ref,
    ql_nope_hbm_ref: jax.Ref,
    q_pe_hbm_ref: jax.Ref,
    o_hbm_ref: jax.Ref,
    cfgs: configs.MlaConfigs,
):
  """Instantiates BufferedRef overrides for MLA query, KV cache, and output."""
  kv_cache_spec = pl.BlockSpec(
      block_shape=cfgs.kv_vmem_shape,
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i,),
      pipeline_mode=pl.Buffered(
          buffer_count=cfgs.kv_buffer_count, use_lookahead=False
      ),
  )
  q_spec = pl.BlockSpec(
      block_shape=cfgs.q_vmem_shape,
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i,),
      pipeline_mode=pl.Buffered(
          buffer_count=cfgs.n_buffer, use_lookahead=False
      ),
  )
  o_spec = pl.BlockSpec(
      block_shape=cfgs.o_vmem_shape,
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i,),
      pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=False),
  )

  kv_cache_alloc = bref_override.KVBufferedRefSeqAlongLane.input_output(
      spec=kv_cache_spec,
      dtype_or_type=cache_kv_hbm_ref,
      buffer_count=cfgs.kv_buffer_count,
      use_lookahead=False,
      cfgs=cfgs,
  )
  q_alloc = bref_override.BatchingQRef.input(
      spec=q_spec,
      dtype_or_type=ql_nope_hbm_ref,
      buffer_count=cfgs.n_buffer,
      use_lookahead=False,
      cfgs=cfgs,
  )
  o_alloc = bref_override.BatchingORef.output(
      spec=o_spec,
      dtype_or_type=o_hbm_ref,
      buffer_count=2,
      use_lookahead=False,
      cfgs=cfgs,
  )

  return q_alloc, kv_cache_alloc, o_alloc


def mla_body(
    q_vref: Any,
    kv_in_vref: Any,
    o_vref: Any,
    schedule_ref: schedule.MlaSchedule,
    m_scratch_ref: Any,
    l_scratch_ref: Any,
    acc_scratch_ref: Any,
    *,
    cu_q_lens_ref: Any,
    kv_lens_ref: Any,
    cfgs: configs.MlaConfigs,
):
  """Inner step execution body of the MLA Pallas pipeline."""
  step = pl.program_id(0)

  with jax.named_scope("prep_math"):
    q_abs_start = []
    kv_abs_start = []
    bkv_sz_frm_cache_list = []
    new_kv_len_start_list = []
    has_new_kv_list = []
    for b_idx in range(cfgs.batch_size):
      s_idx = schedule_ref.s_idx[step, b_idx]
      is_valid = s_idx != -1
      q_idx = schedule_ref.q_idx[step, b_idx]
      k_idx = schedule_ref.k_idx[step, b_idx]
      k_id = jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)
      kv_len = jnp.where(is_valid, kv_lens_ref[s_idx], 0)
      q_start = jnp.where(is_valid, cu_q_lens_ref[s_idx], 0)
      q_end = jnp.where(is_valid, cu_q_lens_ref[s_idx + 1], 0)
      q_len = q_end - q_start
      offset = kv_len - q_len

      q_abs_start.append((q_idx * cfgs.bq_sz + offset))
      kv_abs_start.append(k_id)

      # Stitching metadata
      bkv_sz_frm_cache = jnp.clip(offset - k_id, 0, cfgs.bkv_sz)
      new_kv_len_start = q_start + jnp.maximum(k_id - offset, 0)
      _, _, fetch_sz = schedule_ref.get_dma_kv_new_fetch(step, b_idx)

      bkv_sz_frm_cache_list.append(bkv_sz_frm_cache)
      new_kv_len_start_list.append(new_kv_len_start)
      has_new_kv_list.append(fetch_sz > 0)

  # Skip the whole merge -- strided loads, roll, strided stores -- on lanes
  # with no new tokens.
  for b_idx in range(cfgs.batch_size):

    @pl.when(has_new_kv_list[b_idx])
    def _stitch_and_store(b_idx=b_idx):
      with jax.named_scope("stitch_and_store"):
        stitch_utils.store_new_kv_lane(
            kv_in_vref,
            b_idx,
            stitch_utils.stitch_new_kv_lane(
                kv_in_vref,
                b_idx,
                bkv_sz_frm_cache_list[b_idx],
                new_kv_len_start_list[b_idx],
                cfgs=cfgs,
            ),
            cfgs=cfgs,
        )

  with jax.named_scope("load_q"):
    q_flat = q_vref.reshape((cfgs.batch_size, -1, cfgs.aligned_q_dim))

  with jax.named_scope("load_kv"):
    k_fused = kv_in_vref.at[:, :, : cfgs.bkv_sz]

  is_last_k_list = [
      schedule_ref.is_last_k[step, b] == 1 for b in range(cfgs.batch_size)
  ]

  flash_attention.chunked_flash_attention(
      q=q_flat,
      k=k_fused,
      m_scratch_ref=m_scratch_ref,
      l_scratch_ref=l_scratch_ref,
      acc_scratch_ref=acc_scratch_ref,
      o_vref=o_vref,
      is_last_k=is_last_k_list,
      q_abs_start=q_abs_start,
      kv_abs_start=kv_abs_start,
      cfgs=cfgs,
  )


def get_kernel_name(cfgs: configs.MlaConfigs) -> str:
  serve = cfgs.serve
  name = f"V3-MLA{cfgs.mode.symbol}-{serve.kv_layout.symbol}-p{serve.page_size}"
  name += f"-b{cfgs.batch_size}-q{cfgs.bq_sz}-k{cfgs.bkv_sz}"
  if cfgs.model.sliding_window:
    name += f"-sw{cfgs.model.sliding_window}"
  return name


def get_kernel_metadata(
    cfgs: configs.MlaConfigs,
) -> dict[str, str | int | float]:
  cfgs_dict = dataclasses.asdict(cfgs)
  ret = {}
  for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
    key = jax.tree_util.keystr(path, simple=True, separator=".")
    if not isinstance(val, (str, int, float)):
      val = str(val)
    ret[key] = val
  return ret


def _mla_ragged_paged_attention_kernel(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    schedule_hbm: schedule.MlaSchedule,
    ql_nope_hbm: jax.Array,
    q_pe_hbm: jax.Array,
    new_kv_c_hbm: jax.Array,
    new_k_pe_hbm: jax.Array,
    cache_kv_hbm: jax.Array,
    *,
    cfgs: configs.MlaConfigs,
) -> tuple[jax.Array, jax.Array]:
  """Executes the Pallas MLA attention pipeline with HBM schedule data."""

  # The 1D 32-bit SMEM/DMA tile is one lane-row of words. Every schedule DMA
  # below (and `MlaConfigs.max_steps_ub`) is aligned to it.
  num_lanes = utils.get_tpu_num_lanes()
  aligned_cu_q_len = utils.align_to(cu_q_lens.shape[0], num_lanes)
  aligned_kv_len = utils.align_to(kv_lens.shape[0], num_lanes)
  cu_q_lens_padded = jnp.pad(
      cu_q_lens, (0, aligned_cu_q_len - cu_q_lens.shape[0])
  )
  kv_lens_padded = jnp.pad(kv_lens, (0, aligned_kv_len - kv_lens.shape[0]))
  # The schedule kernel already emits `actual_steps` into SMEM. Constraining the
  # operand keeps XLA from staging it through HBM before this kernel starts.
  # That only matters, and is only allowed by some JAX releases, while tracing;
  # called eagerly there is no XLA graph to stage it through.
  if isinstance(schedule_hbm.actual_steps, jax.core.Tracer):
    schedule_hbm = dataclasses.replace(
        schedule_hbm,
        actual_steps=pltpu.with_memory_space_constraint(
            schedule_hbm.actual_steps, pltpu.SMEM
        ),
    )

  def ragged_paged_attention_pipeline(
      # Inputs.
      cu_q_lens_hbm_ref: jax.Ref,
      kv_lens_hbm_ref: jax.Ref,
      schedule_hbm_ref: schedule.MlaSchedule,
      ql_nope_hbm_ref: jax.Ref,
      q_pe_hbm_ref: jax.Ref,
      new_kv_c_hbm_ref: jax.Ref,
      new_k_pe_hbm_ref: jax.Ref,
      cache_kv_hbm_ref: jax.Ref,
      # Outputs.
      o_hbm_ref: jax.Ref,
      o_kv_cache_hbm_ref: jax.Ref,
  ):
    del o_kv_cache_hbm_ref

    q_alloc, kv_cache_alloc, o_alloc = create_allocs(
        cache_kv_hbm_ref, ql_nope_hbm_ref, q_pe_hbm_ref, o_hbm_ref, cfgs
    )

    actual_steps = schedule_hbm_ref.actual_steps[0]
    num_safe_step_iterations = pl.cdiv(actual_steps, cfgs.max_steps_ub)

    sched_template = schedule.MlaSchedule.create_shape_dtype(cfgs)

    @pl.with_scoped(
        final_allocs=(q_alloc, kv_cache_alloc, o_alloc),
        schedule_ref=sched_template.scratch_shapes(),
        schedule_wait_ref=pltpu.SMEM(
            (
                sched_template.total_hbm_words
                + aligned_cu_q_len
                + aligned_kv_len,
            ),
            jnp.int32,
        ),
        cu_q_lens_smem_ref=pltpu.SMEM((aligned_cu_q_len,), jnp.int32),
        kv_lens_smem_ref=pltpu.SMEM((aligned_kv_len,), jnp.int32),
        dma_sem=pltpu.SemaphoreType.DMA((1,)),
        # One flag per KV buffer slot: set once that slot's window holds only
        # finite values. See KVBufferedRefSeqAlongLane.
        # Sized by physical slots (`kv_num_slots`), not `n_buffer`.
        kv_finite_ref=pltpu.SMEM((cfgs.kv_num_slots,), jnp.int32),
        scratches=(
            pltpu.VMEM(
                cfgs.lm_scratch_shape,
                dtype=jnp.float32,
            ),  # m
            pltpu.VMEM(
                cfgs.lm_scratch_shape,
                dtype=jnp.float32,
            ),  # l
            pltpu.VMEM(
                cfgs.acc_scratch_shape,
                dtype=jnp.float32,
            ),  # acc
        ),
    )
    def _run(
        final_allocs,
        schedule_ref,
        schedule_wait_ref,
        cu_q_lens_smem_ref,
        kv_lens_smem_ref,
        dma_sem,
        kv_finite_ref,
        scratches,
    ):

      def execute_schedule_chunk(start_step, num_steps):
        aligned_start_step = (start_step // num_lanes) * num_lanes
        prefix_steps = start_step - aligned_start_step

        @pl.when(start_step == 0)
        def _():
          copy_q = pltpu.make_async_copy(
              cu_q_lens_hbm_ref,
              cu_q_lens_smem_ref,
              dma_sem.at[0],
          )
          copy_kv = pltpu.make_async_copy(
              kv_lens_hbm_ref,
              kv_lens_smem_ref,
              dma_sem.at[0],
          )
          copy_q.start()
          copy_kv.start()

        total_words = 0
        for h, s in zip(
            schedule_hbm_ref.step_leaves(), schedule_ref.step_leaves()
        ):
          element_size = s.shape[0] // cfgs.max_steps_ub
          read_size = element_size * (num_steps + prefix_steps)
          read_size = utils.align_to(read_size, num_lanes)
          read_size = jnp.minimum(read_size, s.shape[0])
          src_off = pl.multiple_of(element_size * aligned_start_step, num_lanes)

          copy = pltpu.make_async_copy(
              h.at[pl.ds(src_off, read_size)],
              s.at[pl.ds(0, read_size)],
              dma_sem.at[0],
          )
          copy.start()
          total_words += read_size

        total_words = pl.multiple_of(jnp.asarray(total_words), num_lanes)
        extra_lens_words = jnp.where(
            start_step == 0, aligned_cu_q_len + aligned_kv_len, 0
        )

        # Zero scratches here:
        @pl.when(start_step == 0)
        def _zero_scratches():
          scratches[0][...] = jnp.full_like(scratches[0], -jnp.inf)
          scratches[1][...] = jnp.zeros_like(scratches[1])
          scratches[2][...] = jnp.zeros_like(scratches[2])
          for i in range(cfgs.kv_num_slots):
            kv_finite_ref[i] = 0

        total_wait_words = total_words + extra_lens_words
        total_wait_words = pl.multiple_of(
            jnp.asarray(total_wait_words), num_lanes
        )
        dummy_slice = schedule_wait_ref.at[pl.ds(0, total_wait_words)]
        pltpu.make_async_copy(dummy_slice, dummy_slice, dma_sem.at[0]).wait()

        pipeline_func = pltpu.emit_pipeline(
            body=functools.partial(
                mla_body,
                cfgs=cfgs,
                cu_q_lens_ref=cu_q_lens_smem_ref,
                kv_lens_ref=kv_lens_smem_ref,
            ),
            grid=(num_steps + prefix_steps,),
            in_specs=(
                q_alloc.spec,
                kv_cache_alloc.spec,
            ),
            out_specs=(o_alloc.spec,),
        )
        pipeline_func(
            (ql_nope_hbm_ref, q_pe_hbm_ref, schedule_ref),
            (
                cache_kv_hbm_ref,
                new_kv_c_hbm_ref,
                new_k_pe_hbm_ref,
                schedule_ref,
                kv_finite_ref,
            ),
            (o_hbm_ref, schedule_ref),
            scratches=(schedule_ref,) + scratches,
            allocations=final_allocs,
        )

      @pl.loop(0, num_safe_step_iterations)
      def loop_body(step_idx):
        start = step_idx * cfgs.max_steps_ub
        rem = actual_steps % cfgs.max_steps_ub
        last_step_size = jnp.where(rem == 0, cfgs.max_steps_ub, rem)
        is_last_step = step_idx == num_safe_step_iterations - 1
        size = jnp.where(is_last_step, last_step_size, cfgs.max_steps_ub)

        execute_schedule_chunk(start, size)

    _run()

  num_pre_leaves = 2  # cu_q_lens_padded, kv_lens_padded
  num_sched_leaves = len(jax.tree_util.tree_leaves(schedule_hbm))
  ql_nope_hbm_idx = num_pre_leaves + num_sched_leaves
  cache_kv_hbm_idx = ql_nope_hbm_idx + 4

  return pl.pallas_call(
      ragged_paged_attention_pipeline,
      out_shape=[ql_nope_hbm, cache_kv_hbm],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM),  # cu_q_lens_padded
              pl.BlockSpec(memory_space=pltpu.HBM),  # kv_lens_padded
              schedule_hbm.in_specs(),
              pl.BlockSpec(memory_space=pltpu.HBM),  # ql_nope_hbm_ref
              pl.BlockSpec(memory_space=pltpu.HBM),  # q_pe_hbm_ref
              pl.BlockSpec(memory_space=pltpu.HBM),  # new_kv_c_hbm_ref
              pl.BlockSpec(memory_space=pltpu.HBM),  # new_k_pe_hbm_ref
              pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_hbm_ref
          ],
          out_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM),  # aliased_o_hbm_ref
              pl.BlockSpec(memory_space=pltpu.HBM),  # aliased_cache_kv_hbm_ref
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=cfgs.vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      input_output_aliases={ql_nope_hbm_idx: 0, cache_kv_hbm_idx: 1},
      name=get_kernel_name(cfgs),
      metadata={k: str(v) for k, v in get_kernel_metadata(cfgs).items()},
  )(
      cu_q_lens_padded,
      kv_lens_padded,
      schedule_hbm,
      ql_nope_hbm,
      q_pe_hbm,
      new_kv_c_hbm,
      new_k_pe_hbm,
      cache_kv_hbm,
  )


# ==============================================================================
# Outermost Kernel Entrypoint
# ==============================================================================


@functools.partial(
    jax.jit,
    static_argnames=(
        "cfgs",
        "debug_mode",
    ),
    donate_argnames=("cache_kv",),
)
def mla_ragged_paged_attention(
    ql_nope: jax.Array,  # [actual_num_q_heads, max_num_tokens, actual_lkv_dim]
    q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
    cache_kv: jax.Array,  # [total_num_pages, aligned_kv_dim, page_size]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    cfgs: configs.MlaConfigs,
    schedule_hbm: schedule.MlaSchedule | None = None,
    debug_mode: bool = False,
) -> tuple[
    jax.Array,  # [actual_num_q_heads, max_num_tokens, actual_lkv_dim]
    jax.Array,  # updated_cache_kv: [total_num_pages, aligned_kv_dim, page_size]
]:
  """MLA Ragged paged attention with 3D transposed KV cache support."""
  static_validate_inputs(
      ql_nope,
      q_pe,
      new_kv_c,
      new_k_pe,
      cache_kv,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
      cfgs=cfgs,
  )

  actual_num_q_heads, actual_max_num_tokens, actual_lkv_dim = ql_nope.shape

  if schedule_hbm is None:
    schedule_hbm = schedule.generate_mla_metadata(
        cu_q_lens, kv_lens, page_indices, distribution, cfgs=cfgs
    )

  ql_nope = prepare_q_nope_inputs(
      ql_nope,
      vmem_limit_bytes=cfgs.vmem_limit_bytes,
  )  # [max_num_tokens, num_q_heads, lkv_dim]
  q_pe = prepare_q_pe_inputs(q_pe)  # [max_num_tokens, num_q_heads, r_dim]
  new_kv_c = prepare_kv_inputs_for_transposed_kv_cache(
      new_kv_c,
  )  # [lkv_dim, max_num_tokens]
  new_k_pe = prepare_kv_inputs_for_transposed_kv_cache(
      new_k_pe,
  )  # [r_dim, max_num_tokens]

  if debug_mode:
    logging.info(
        "Prepared inputs for MLA: ql_nope=%s, q_pe=%s, new_kv_c=%s,"
        " new_k_pe=%s, cache_kv=%s",
        ql_nope.shape,
        q_pe.shape,
        new_kv_c.shape,
        new_k_pe.shape,
        cache_kv.shape,
    )

  out_hbm, updated_cache_kv = _mla_ragged_paged_attention_kernel(
      cu_q_lens,
      kv_lens,
      schedule_hbm,
      ql_nope,
      q_pe,
      new_kv_c,
      new_k_pe,
      cache_kv,
      cfgs=cfgs,
  )

  output = prepare_outputs(
      out_hbm,
      actual_num_q_heads,
      actual_max_num_tokens,
      actual_lkv_dim,
      vmem_limit_bytes=cfgs.vmem_limit_bytes,
  )  # [actual_num_q_heads, max_num_tokens, actual_lkv_dim]

  return output, updated_cache_kv
