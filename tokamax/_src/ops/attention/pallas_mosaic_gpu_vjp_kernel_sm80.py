# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Flash Attention Pallas-Mosaic-GPU VJP implementation for SM80 (Ampere/A100).

Implements two deterministic, non-atomic backward kernels (`kernel_dq` and
`kernel_dkv`) tailored to SM80 hardware (163 KB SMEM/CTA, `cp.async`,
`mma.sync.aligned.m16n8k16`, `ldmatrix`, swizzled `SMEM_GMEM_COPY` stores):
1. `kernel_dq`: Computes `dQ` (and optional `dS` for `dbias`) with cooperative
   multi-stage `cp.async` KV buffering, causal LPT scheduling, and tile-level
   early exit + boundary mask peeling for `query_seq_lengths` and
   `key_value_seq_lengths`.
2. `kernel_dkv`: Computes `dK` and `dV` (accumulating GQA query heads directly
   in FP32 registers):
   - KV-parallel mode (`head_dim + head_dim_out <= 256`): 1 or 2 warpgroups per
     CTA sharing `Q` and `dO` in SMEM with `P^T` and `dP^T` in registers.
   - Cooperative Split-dK/dV 2-WG mode (`head_dim + head_dim_out > 256`, e.g.,
     DeepSeek-V2 `192/128` and FlashAttention-3 `256/256`): WG0 holds `K` and
     `dK_acc` in registers (`S^T` + `dK`) while WG1 holds `V` and `dV_acc` in
     registers (`dP^T` + `dV`), sharing `Q` and `dO` in SMEM and exchanging
     compact `P^T` / `dP^T` tiles via SMEM with zero register spills.
"""

import math
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.experimental.mosaic.gpu as mgpu
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src import shape as shape_lib
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_mosaic_gpu_common as common
from tokamax._src.ops.attention import pallas_mosaic_gpu_vjp_common as vjp_common

# pylint: disable=cell-var-from-loop

Config = vjp_common.Config
Residuals = base.Residuals

_SMEM_SIZE_BYTES: int = 162 * 1024
_F32_BYTES: int = 4


def _get_bound_args(ba: op.BoundArguments):
  """Extracts attention input arguments from BoundArguments."""
  args_dict = getattr(ba, "arguments", {})

  def _get(name, pos):
    if name in args_dict:
      return args_dict[name]
    if pos >= 0 and len(ba.args) > pos:
      return ba.args[pos]
    return ba.kwargs.get(name)

  q, k, v = _get("q", 3), _get("k", 4), _get("v", 5)
  bias, mask = _get("bias", -1), _get("mask", -1)
  q_indices, k_indices = _get("q_indices", -1), _get("k_indices", -1)
  return q, k, v, bias, mask, q_indices, k_indices


def _pad_head_dim(x):
  return common.pad_head_dim_to_next_multiple_of_min_swizzle(x)


def _eval_mask_shape(mask, q, k, q_indices, k_indices):
  def _decomp(m, q_, k_, qi_, ki_):
    m, _, _ = common.extract_q_range(m, q_.ndim, qi_, ki_)
    arr, *_ = common.decompose_mask(m, q_, k_, qi_, ki_)
    return arr

  return jax.eval_shape(_decomp, mask, q, k, q_indices, k_indices)


def _align_1k(n_bytes: int) -> int:
  return pl.cdiv(n_bytes, 1024) * 1024


def _estimate_dq_smem_bytes(
    ba: op.BoundArguments,
    block_q: int,
    block_kv: int,
    num_stages: int,
    ds_dtype: jax.typing.DTypeLike | None,
    compute_wgs: int = 2,
) -> int:
  """Estimates `kernel_dq` SMEM usage in bytes for `config`."""
  q, k, v, bias, mask, q_indices, k_indices = _get_bound_args(ba)
  head_dim = pl.cdiv(q.shape[-1], 16) * 16
  head_dim_out = pl.cdiv(v.shape[-1], 16) * 16
  if head_dim + head_dim_out > 256 and block_q > 64:
    return _SMEM_SIZE_BYTES + 1

  tile_q = compute_wgs * block_q
  dtype_bytes = 2  # f32 inputs are downcast to bf16/f16 before kernel call.

  q_do_bytes = _align_1k(tile_q * head_dim * dtype_bytes) + _align_1k(
      tile_q * head_dim_out * dtype_bytes
  )
  kv_bytes = _align_1k(
      num_stages * block_kv * head_dim * dtype_bytes
  ) + _align_1k(num_stages * block_kv * head_dim_out * dtype_bytes)

  extra_stage_bytes = 0
  if (
      bias is not None
      and bias.ndim >= 2
      and bias.shape[-2] != 1
      and bias.shape[-1] != 1
  ):
    bias_bytes = jnp.finfo(bias.dtype).bits // 8
    extra_stage_bytes += _align_1k(num_stages * tile_q * block_kv * bias_bytes)

  mask_arr = _eval_mask_shape(mask, q, k, q_indices, k_indices)
  if (
      mask_arr is not None
      and mask_arr.shape[-2] != 1
      and mask_arr.shape[-1] != 1
  ):
    extra_stage_bytes += _align_1k(num_stages * tile_q * block_kv)

  q_in_regs = (head_dim + head_dim_out <= 192) and (block_q == 64)
  if q_in_regs:
    loop_bytes = max(q_do_bytes, kv_bytes + extra_stage_bytes)
  else:
    loop_bytes = q_do_bytes + kv_bytes + extra_stage_bytes

  if ds_dtype is not None:
    ds_elem_bytes = jnp.finfo(ds_dtype).bits // 8
    loop_bytes += _align_1k(tile_q * block_kv * ds_elem_bytes)

  epi_bytes = _align_1k(tile_q * head_dim * dtype_bytes)
  return max(loop_bytes, epi_bytes)


def _estimate_dkv_smem_bytes(
    ba: op.BoundArguments,
    block_q: int,
    block_kv: int,
    num_stages: int,
    compute_wgs: int = 2,
) -> int:
  """Estimates `kernel_dkv` SMEM usage in bytes for `config`."""
  q, k, v, bias, mask, q_indices, k_indices = _get_bound_args(ba)
  head_dim = pl.cdiv(q.shape[-1], 16) * 16
  head_dim_out = pl.cdiv(v.shape[-1], 16) * 16
  dtype_bytes = 2

  split_mode = (head_dim + head_dim_out) > 256
  if split_mode and (block_kv > 64 or block_q > 64):
    return _SMEM_SIZE_BYTES + 1
  if not split_mode and (head_dim + head_dim_out > 160) and block_kv > 64:
    return _SMEM_SIZE_BYTES + 1

  tile_kv = block_kv if split_mode else (compute_wgs * block_kv)
  kv_bytes = _align_1k(tile_kv * head_dim * dtype_bytes) + _align_1k(
      tile_kv * head_dim_out * dtype_bytes
  )
  q_do_bytes = _align_1k(
      num_stages * block_q * head_dim * dtype_bytes
  ) + _align_1k(num_stages * block_q * head_dim_out * dtype_bytes)

  extra_stage_bytes = 0
  if (
      bias is not None
      and bias.ndim >= 2
      and bias.shape[-2] != 1
      and bias.shape[-1] != 1
  ):
    bias_bytes = jnp.finfo(bias.dtype).bits // 8
    extra_stage_bytes += _align_1k(num_stages * tile_kv * block_q * bias_bytes)

  mask_arr = _eval_mask_shape(mask, q, k, q_indices, k_indices)
  if (
      mask_arr is not None
      and mask_arr.shape[-2] != 1
      and mask_arr.shape[-1] != 1
  ):
    extra_stage_bytes += _align_1k(num_stages * tile_kv * block_q)

  if split_mode:
    pt_dpt_bytes = _align_1k(block_kv * block_q * dtype_bytes) + _align_1k(
        block_kv * block_q * _F32_BYTES
    )
    loop_bytes = max(kv_bytes, q_do_bytes + extra_stage_bytes) + pt_dpt_bytes
  else:
    loop_bytes = kv_bytes + q_do_bytes + extra_stage_bytes

  return max(loop_bytes, kv_bytes)


def get_heuristics_config(ba: op.BoundArguments) -> Config:
  """Returns a heuristic VJP configuration for SM80 GPUs."""
  q, k, v, bias, *_ = _get_bound_args(ba)
  dbias_intermediate_dtype = getattr(ba.op, "dbias_intermediate_dtype", None)
  ds_dtype = vjp_common.get_ds_dtype(q, k, bias, dbias_intermediate_dtype)

  batch_size = math.prod(q.shape[:-3]) * math.prod(ba.vmap_axis_sizes)
  q_seq_len = q.shape[-3]
  kv_seq_len, num_kv_heads = k.shape[-3], k.shape[-2]
  head_dim = pl.cdiv(q.shape[-1], 16) * 16
  head_dim_out = pl.cdiv(v.shape[-1], 16) * 16

  # Choose `compute_wgs`: use `1` for large head dims (`D + d > 256`) or when
  # KV tiles are scarce across 108 SMs (e.g. short sequence or small GQA KV).
  kv_ctas_at_2wgs = batch_size * num_kv_heads * pl.cdiv(kv_seq_len, 128)
  prefer_2wgs = (
      (head_dim + head_dim_out <= 256)
      and q_seq_len >= 128
      and kv_seq_len >= 128
      and kv_ctas_at_2wgs >= 54
  )
  wgs_candidates = [2, 1] if prefer_2wgs else [1, 2]

  for compute_wgs in wgs_candidates:
    for num_stages in (2, 1):
      if (
          _estimate_dq_smem_bytes(ba, 64, 64, num_stages, ds_dtype, compute_wgs)
          <= _SMEM_SIZE_BYTES
          and _estimate_dkv_smem_bytes(ba, 64, 64, num_stages, compute_wgs)
          <= _SMEM_SIZE_BYTES
      ):
        return Config(
            block_q_dkv=64,
            block_kv_dkv=64,
            block_q_dq=64,
            block_kv_dq=64,
            num_stages=num_stages,
            compute_wgs=compute_wgs,
        )

  return Config(
      block_q_dkv=64,
      block_kv_dkv=64,
      block_q_dq=64,
      block_kv_dq=64,
      num_stages=1,
      compute_wgs=1,
  )


def get_autotuning_configs(ba: op.BoundArguments) -> set[Config]:
  """Returns autotuning VJP configs filtered by SM80's 163 KB SMEM limit."""
  q, k, _, bias, *_ = _get_bound_args(ba)
  dbias_intermediate_dtype = getattr(ba.op, "dbias_intermediate_dtype", None)
  ds_dtype = vjp_common.get_ds_dtype(q, k, bias, dbias_intermediate_dtype)

  configs: set[Config] = set()
  for block_q in (64, 128):
    for block_kv in (64, 128):
      for num_stages in (1, 2, 3):
        for compute_wgs in (1, 2):
          if (
              _estimate_dq_smem_bytes(
                  ba, block_q, block_kv, num_stages, ds_dtype, compute_wgs
              )
              <= _SMEM_SIZE_BYTES
              and _estimate_dkv_smem_bytes(
                  ba, block_q, block_kv, num_stages, compute_wgs
              )
              <= _SMEM_SIZE_BYTES
          ):
            configs.add(
                Config(
                    block_q_dkv=block_q,
                    block_kv_dkv=block_kv,
                    block_q_dq=block_q,
                    block_kv_dq=block_kv,
                    num_stages=num_stages,
                    compute_wgs=compute_wgs,
                )
            )
  return configs


def _async_copy(src, dst):
  common.cp_async_gmem_to_smem(src, dst)


def _cta_barrier():
  mgpu_lib.bar_sync(8, 256)


def _relayout_acc_to_lhs(x: jax.Array, mma_acc, mma_lhs) -> jax.Array:
  """Zero-cost register relayout from SM80 MMA_ACC to MMA_LHS."""
  @plgpu.inline_mgpu(
      arg_types=(mma_acc,),
      return_type=plgpu.ShapeDtypeStruct(x.shape, x.dtype, layout=mma_lhs),
  )
  def relabel(ctx, fa):
    del ctx
    with mgpu_lib.ir.Context():
      target = mma_lhs.to_mgpu()
    return mgpu.FragmentedArray(
        _registers=fa.registers.reshape(target.registers_shape(fa.shape)),
        _layout=target,
        _is_signed=fa.is_signed,
    )

  return relabel(x)


def _choose_rhs_swizzle(minor_dim: int, n_dim: int, elem_bits: int) -> int:
  del n_dim
  return min(64, plgpu.find_swizzle(minor_dim * elem_bits))


def _choose_row_step_k(k_total: int, cols: int, max_regs: int = 32) -> int:
  max_step = max(16, (max_regs * 64 // cols) // 16 * 16)
  for step in range(min(k_total, max_step), 15, -16):
    if k_total % step == 0:
      return step
  return 16


def _mma_transposed_rhs(
    lhs_reg: jax.Array | None,
    lhs_smem,
    rhs_smem,
    m: int,
    n: int,
    k: int,
    step_k: int,
    mma_acc,
    mma_lhs,
    mma_rhs,
) -> jax.Array:
  """Computes `LHS @ RHS.T` chunked along `k` (`minor` dim of `rhs_smem`)."""
  acc = plgpu.layout_cast(jnp.zeros((m, n), jnp.float32), mma_acc)
  for d0 in range(0, k, step_k):
    d_slice = pl.ds(d0, step_k)
    a_chunk = (
        lhs_reg[:, d0 : d0 + step_k]
        if lhs_reg is not None
        else plgpu.load(lhs_smem.at[:, d_slice], layout=mma_lhs)
    )
    b_chunk = plgpu.load(rhs_smem.at[:, d_slice].T, layout=mma_rhs)
    acc = plgpu.mma(acc, a_chunk, b_chunk)
  return acc


def _mma_accumulate_row_rhs(
    acc: jax.Array,
    lhs: jax.Array,
    rhs_smem,
    step_k: int,
    mma_acc,
    mma_rhs,
) -> jax.Array:
  """Accumulates `acc += lhs @ rhs_smem` chunked along `major` dim of `rhs_smem`."""
  acc = plgpu.layout_cast(acc, mma_acc)
  k = lhs.shape[1]
  for k0 in range(0, k, step_k):
    b_chunk = plgpu.load(rhs_smem.at[pl.ds(k0, step_k), :], layout=mma_rhs)
    acc = plgpu.mma(acc, lhs[:, k0 : k0 + step_k], b_chunk)
  return acc


def _store_swizzled(
    gmem_ref, val: jax.Array, smem_buf, out_dtype: jnp.dtype, swizzle: int
):
  """Coalesced 128-bit store of an `MMA_ACC` register tile via swizzled SMEM."""
  rows, cols = val.shape
  buf_cols = smem_buf.shape[-1]
  copy_layout = plgpu.Layout.SMEM_GMEM_COPY(
      (rows, buf_cols), out_dtype, swizzle=swizzle
  )
  for c0 in range(0, cols, buf_cols):
    plgpu.store(
        smem_buf,
        val[:, c0 : c0 + buf_cols].astype(out_dtype),
        optimized=False,
    )
    mgpu_lib.warpgroup_barrier()
    staged = plgpu.load(smem_buf, layout=copy_layout, optimized=False)
    plgpu.store(gmem_ref.at[:, pl.ds(c0, buf_cols)], staged, optimized=False)
    if c0 + buf_cols < cols:
      mgpu_lib.warpgroup_barrier()


@jaxtyping.jaxtyped
def flash_attention_vjp_kernel(
    q: Float[Array, "T H D"],
    k: Float[Array, "t h D"],
    v: Float[Array, "t h d"],
    residuals: Residuals,
    out: Float[Array, "T H d"],
    dout: Float[Array, "T H d"],
    bias: Float[Array, "#H #T #t"] | None,
    mask: Bool[Array, "#H #T #t"] | None,
    k_start: Int[Array, "#H #T"] | None,
    k_end: Int[Array, "#H #T"] | None,
    q_start: Int[Array, "#H 1"] | None = None,
    q_end: Int[Array, "#H 1"] | None = None,
    *,
    logits_scale: float,
    logits_soft_cap: float | None,
    is_causal: bool,
    ds_dtype: jax.typing.DTypeLike | None,
    config: Config,
) -> tuple[
    Float[Array, "T H D"],  # dq
    Float[Array, "t h D"],  # dk
    Float[Array, "t h d"],  # dv
    Float[Array, "H T t"] | None,  # ds
]:
  """SM80 Pallas Mosaic GPU Flash Attention VJP."""
  orig_q_seq_len, num_q_heads, orig_head_dim = q.shape
  orig_kv_seq_len, num_kv_heads, orig_head_dim_out = v.shape

  if (dtype := q.dtype) != k.dtype or dtype != v.dtype:
    raise ValueError(
        f"q, k, and v should all have the same dtype, got: {q.dtype},"
        f" {k.dtype}, {v.dtype}"
    )
  if jnp.dtype(dtype) not in map(jnp.dtype, [jnp.float16, jnp.bfloat16]):
    raise NotImplementedError(
        f"Only f16 and bf16 are supported on SM80 VJP, got {dtype}"
    )
  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  delta = jnp.einsum(
      "qhd,qhd->hq",
      out.astype(jnp.float32),
      dout.astype(jnp.float32),
      precision=jax.lax.Precision.HIGHEST,
  )

  q, k, v, dout = map(_pad_head_dim, (q, k, v, dout))
  head_dim = q.shape[-1]
  head_dim_out = v.shape[-1]
  if head_dim > 256 or head_dim_out > 256:
    raise NotImplementedError(
        "SM80 VJP supports head_dim, head_dim_out <= 256, got"
        f" ({head_dim}, {head_dim_out})."
    )

  block_q_dq = config.block_q_dq
  block_kv_dq = config.block_kv_dq
  block_q_dkv = config.block_q_dkv
  block_kv_dkv = config.block_kv_dkv

  if block_q_dq % 64 or block_kv_dq % 64:
    raise NotImplementedError(
        "SM80 VJP requires block_q_dq, block_kv_dq % 64 == 0."
    )
  if head_dim + head_dim_out > 256 and block_q_dq > 64:
    raise NotImplementedError("block_q_dq > 64 unsupported for large head_dim.")

  dkv_split_mode = (head_dim + head_dim_out) > 256
  if dkv_split_mode and (block_kv_dkv > 64 or block_q_dkv > 64):
    raise NotImplementedError("Split-dK/dV requires 64x64 blocks.")
  if (
      not dkv_split_mode
      and (head_dim + head_dim_out > 160)
      and block_kv_dkv > 64
  ):
    raise NotImplementedError(
        "block_kv_dkv > 64 unsupported for head_dim > 80."
    )

  dq_wgs = config.compute_wgs
  dkv_wgs = 1 if dkv_split_mode else config.compute_wgs
  tile_q_dq = dq_wgs * block_q_dq
  tile_kv_dkv = block_kv_dkv if dkv_split_mode else (dkv_wgs * block_kv_dkv)

  pad_q = math.lcm(tile_q_dq, block_q_dkv)
  pad_kv = math.lcm(block_kv_dq, tile_kv_dkv)

  pad_seq_q = lambda x, ax: shape_lib.pad_to_next_multiple_of(x, pad_q, ax)
  pad_seq_kv = lambda x, ax: shape_lib.pad_to_next_multiple_of(x, pad_kv, ax)
  q, dout = map(pad_seq_q, (q, dout), (-3, -3))
  k, v = map(pad_seq_kv, (k, v), (-3, -3))
  q_seq_len = q.shape[-3]
  kv_seq_len = k.shape[-3]

  m, l = residuals
  finfo_min = float(jnp.finfo(jnp.float32).min)
  m = jnp.where((m == -jnp.inf) | (l == 0.0) | (m == finfo_min), jnp.inf, m)
  m = shape_lib.pad_to_next_multiple_of(m, pad_q, -1, pad_value=jnp.inf)
  l = shape_lib.pad_to_next_multiple_of(l, pad_q, -1, pad_value=0.0)
  delta = shape_lib.pad_to_next_multiple_of(delta, pad_q, -1, pad_value=0.0)

  as_2d = lambda x: None if x is None else jax.lax.broadcast_to_rank(x, 2)
  k_start, k_end, q_start, q_end = map(as_2d, (k_start, k_end, q_start, q_end))

  if orig_kv_seq_len < kv_seq_len:
    if k_end is None:
      k_end = jnp.full((1, 1), orig_kv_seq_len, jnp.int32)
    else:
      k_end = jnp.minimum(k_end, orig_kv_seq_len)

  if orig_q_seq_len < q_seq_len:
    if q_end is None:
      q_end = jnp.full((1, 1), orig_q_seq_len, jnp.int32)
    else:
      q_end = jnp.minimum(q_end, orig_q_seq_len)

  # Compute per-CTA `(min, max)` before padding `k_start` / `k_end`.
  k_start_minmax = common.k_range_block_minmax(k_start, kv_seq_len, tile_q_dq)
  k_end_minmax = common.k_range_block_minmax(k_end, kv_seq_len, tile_q_dq)

  if k_start is not None and k_start.shape[-1] != 1:
    k_start = shape_lib.pad_to_next_multiple_of(
        k_start, pad_q, -1, pad_value=kv_seq_len
    )
  if k_end is not None and k_end.shape[-1] != 1:
    k_end = shape_lib.pad_to_next_multiple_of(
        k_end, pad_q, -1, pad_value=0
    )

  if bias is not None:
    if bias.shape[-2] != 1:
      bias = pad_seq_q(bias, -2)
    if bias.shape[-1] != 1:
      bias = pad_seq_kv(bias, -1)

  if mask is not None:
    if mask.shape[-2] != 1:
      mask = pad_seq_q(mask, -2)
    if mask.shape[-1] != 1:
      mask = pad_seq_kv(mask, -1)
    mask = mask.astype(jnp.int8)

  elem_bits = mgpu_lib.num_bits(dtype)
  mma_lhs = plgpu.Layout.MMA_LHS(dtype)
  mma_rhs = plgpu.Layout.MMA_RHS(dtype)
  mma_acc = plgpu.Layout.MMA_ACC(dtype)
  mma_row = mma_acc.reduce(1)
  mma_col = mma_acc.reduce(0)
  tiled_smem = mgpu_lib.tiled_swizzled_smem

  has_index_mask = (
      is_causal
      or q_start is not None
      or q_end is not None
      or k_start is not None
      or k_end is not None
  )

  # ===========================================================================
  # Kernel 1: `kernel_dq` (computes `dQ` and optional `dS`)
  # ===========================================================================
  num_q_tiles_dq = q_seq_len // tile_q_dq
  num_kv_tiles_dq = kv_seq_len // block_kv_dq
  stages_dq = max(1, min(config.num_stages, num_kv_tiles_dq))

  k_dq_swizzle = _choose_rhs_swizzle(head_dim, block_kv_dq, elem_bits)
  v_dq_swizzle = _choose_rhs_swizzle(head_dim_out, block_kv_dq, elem_bits)
  step_d_dq = 8 * k_dq_swizzle // elem_bits
  step_d_out_dq = 8 * v_dq_swizzle // elem_bits
  step_kv_dq = _choose_row_step_k(block_kv_dq, head_dim)
  q_in_regs_dq = (head_dim + head_dim_out <= 192) and (block_q_dq == 64)

  stage_bias_dq = (
      bias is not None and bias.shape[-2] != 1 and bias.shape[-1] != 1
  )
  stage_mask_dq = (
      mask is not None and mask.shape[-2] != 1 and mask.shape[-1] != 1
  )

  def _build_dq_scratch(use_bias_smem: bool, use_mask_smem: bool):
    q_sc = tiled_smem(
        (dq_wgs, block_q_dq, head_dim), dtype, "q_dq", swizzle=k_dq_swizzle
    )
    do_sc = tiled_smem(
        (dq_wgs, block_q_dq, head_dim_out),
        dtype,
        "do_dq",
        swizzle=v_dq_swizzle,
    )
    k_sc = tiled_smem(
        (stages_dq, block_kv_dq, head_dim),
        dtype,
        "k_dq",
        swizzle=k_dq_swizzle,
    )
    v_sc = tiled_smem(
        (stages_dq, block_kv_dq, head_dim_out),
        dtype,
        "v_dq",
        swizzle=v_dq_swizzle,
    )
    epi_sc = tiled_smem((dq_wgs, block_q_dq, head_dim), dtype, "epi_dq")
    extras: dict[str, Any] = {}
    if use_bias_smem:
      assert bias is not None
      extras["bias_smem"] = tiled_smem(
          (stages_dq, dq_wgs, block_q_dq, block_kv_dq), bias.dtype, "bias_dq"
      )
    if use_mask_smem:
      extras["mask_smem"] = tiled_smem(
          (stages_dq, dq_wgs, block_q_dq, block_kv_dq), jnp.int8, "mask_dq"
      )
    if q_in_regs_dq:
      main_union = plgpu.RefUnion(
          (q_sc, do_sc), (k_sc, v_sc, extras), (epi_sc,)
      )
    else:
      main_union = plgpu.RefUnion(
          (q_sc, do_sc, k_sc, v_sc, extras), (epi_sc,)
      )

    if bias is None or ds_dtype is None:
      ds_sc = None
    else:
      ds_sc = tiled_smem(
          (dq_wgs, block_q_dq, block_kv_dq), ds_dtype, "ds_dq"
      )
    return dict(
        main_epi_union=main_union,
        ds_smems=ds_sc,
    )

  dq_scratch = _build_dq_scratch(stage_bias_dq, stage_mask_dq)
  if mgpu_lib.estimate_smem_bytes(dq_scratch) > _SMEM_SIZE_BYTES:
    stage_mask_dq = False
    dq_scratch = _build_dq_scratch(stage_bias_dq, stage_mask_dq)
  if mgpu_lib.estimate_smem_bytes(dq_scratch) > _SMEM_SIZE_BYTES:
    stage_bias_dq = False
    dq_scratch = _build_dq_scratch(stage_bias_dq, stage_mask_dq)

  epi_dq_swizzle = plgpu.find_swizzle(head_dim * elem_bits, "epi_dq")
  if bias is None or ds_dtype is None:
    ds_swizzle = 0
  else:
    ds_swizzle = plgpu.find_swizzle(
        block_kv_dq * mgpu_lib.num_bits(ds_dtype), "ds_dq"
    )

  def kernel_dq(
      q_gmem,
      k_gmem,
      v_gmem,
      dout_gmem,
      m_gmem,
      l_gmem,
      delta_gmem,
      bias_gmem,
      mask_gmem,
      k_start_gmem,
      k_end_gmem,
      q_start_gmem,
      q_end_gmem,
      k_start_minmax_gmems,
      k_end_minmax_gmems,
      *out_gmems,
      main_epi_union,
      ds_smems=None,
  ):
    if bias is None:
      (dq_gmem,) = out_gmems
      ds_gmem = None
    else:
      dq_gmem, ds_gmem = out_gmems

    wg = lax.axis_index("wg") if dq_wgs > 1 else 0
    hi = lax.axis_index("heads")
    qi_raw = lax.axis_index("q_tiles")
    qi = (num_q_tiles_dq - 1 - qi_raw) if is_causal else qi_raw
    hi_kv = lax.div(hi, jnp.asarray(q_heads_per_kv_head, hi.dtype))

    cta_q_base = qi * tile_q_dq
    q_base = cta_q_base + wg * block_q_dq
    qs = pl.ds(q_base, block_q_dq)

    q_start_val = jnp.int32(0)
    if q_start_gmem is not None:
      q_hi = 0 if q_start_gmem.shape[0] == 1 else hi
      q_start_val = q_start_gmem[q_hi, 0]
    q_end_val = jnp.int32(orig_q_seq_len)
    if q_end_gmem is not None:
      q_hi = 0 if q_end_gmem.shape[0] == 1 else hi
      q_end_val = lax.min(q_end_val, q_end_gmem[q_hi, 0])

    eff_q_max = lax.min(cta_q_base + tile_q_dq, q_end_val)
    ub = (
        lax.min(num_kv_tiles_dq, pl.cdiv(lax.max(eff_q_max, 0), block_kv_dq))
        if is_causal
        else num_kv_tiles_dq
    )
    lb = jnp.int32(0)
    k_start_max = jnp.int32(0)
    k_end_min = jnp.int32(kv_seq_len)
    if k_start_minmax_gmems is not None:
      ks_min_g, ks_max_g = k_start_minmax_gmems
      ks_min = common.load_bcast(ks_min_g, (hi, qi), layout=None)
      k_start_max = common.load_bcast(ks_max_g, (hi, qi), layout=None)
      lb = lax.max(lb, lax.div(lax.max(ks_min, 0), block_kv_dq))
    if k_end_minmax_gmems is not None:
      ke_min_g, ke_max_g = k_end_minmax_gmems
      k_end_min = common.load_bcast(ke_min_g, (hi, qi), layout=None)
      ke_max = common.load_bcast(ke_max_g, (hi, qi), layout=None)
      ub = lax.min(ub, pl.cdiv(lax.max(ke_max, 0), block_kv_dq))

    cta_q_valid = (cta_q_base + tile_q_dq > q_start_val) & (
        cta_q_base < q_end_val
    )
    cta_active = cta_q_valid & (ub > lb)

    q_cta_fully_valid = True
    if q_start_gmem is not None:
      q_cta_fully_valid = q_cta_fully_valid & (cta_q_base >= q_start_val)
    if q_end_gmem is not None:
      q_cta_fully_valid = q_cta_fully_valid & (
          cta_q_base + tile_q_dq <= q_end_val
      )

    if q_in_regs_dq:
      (q_smem, dout_smem), (k_smem, v_smem, extras), (epi_smem,) = (
          main_epi_union
      )
    else:
      (q_smem, dout_smem, k_smem, v_smem, extras), (epi_smem,) = main_epi_union
    bias_smem = extras.get("bias_smem")
    mask_smem = extras.get("mask_smem")

    def _sync_dq():
      if dq_wgs > 1:
        _cta_barrier()
      else:
        mgpu_lib.warpgroup_barrier()

    def _zero_ds_range(lo_ki, hi_ki):
      if ds_gmem is None:
        return
      assert ds_smems is not None
      z_ds = plgpu.layout_cast(
          jnp.zeros((block_q_dq, block_kv_dq), jnp.float32), mma_acc
      )
      plgpu.store(ds_smems.at[wg], z_ds.astype(ds_dtype), optimized=False)
      mgpu_lib.warpgroup_barrier()
      copy_layout = plgpu.Layout.SMEM_GMEM_COPY(
          (block_q_dq, block_kv_dq), ds_dtype, swizzle=ds_swizzle
      )
      staged_zero = plgpu.load(
          ds_smems.at[wg], layout=copy_layout, optimized=False
      )

      def _z_body(ki, _):
        ks = pl.ds(ki * block_kv_dq, block_kv_dq)
        plgpu.store(ds_gmem.at[hi, qs, ks], staged_zero, optimized=False)
        return None

      lax.fori_loop(lo_ki, hi_ki, _z_body, None)
      mgpu_lib.warpgroup_barrier()

    @pl.when(jnp.logical_not(cta_active))
    def _dq_early_exit():
      z_dq = plgpu.layout_cast(
          jnp.zeros((block_q_dq, head_dim), jnp.float32), mma_acc
      )
      _store_swizzled(
          dq_gmem.at[qs, hi], z_dq, epi_smem.at[wg], dtype, epi_dq_swizzle
      )
      _zero_ds_range(0, num_kv_tiles_dq)

    @pl.when(cta_active)
    def _dq_active():
      _zero_ds_range(0, lb)
      _zero_ds_range(ub, num_kv_tiles_dq)

      m_row = plgpu.load(m_gmem.at[hi, qs], layout=mma_row, optimized=False)
      l_row = plgpu.load(l_gmem.at[hi, qs], layout=mma_row, optimized=False)
      delta_row = plgpu.load(
          delta_gmem.at[hi, qs], layout=mma_row, optimized=False
      )
      m_log2 = m_row * math.log2(math.e)
      l_rcp = jnp.where(
          l_row == 0.0, 0.0, 1.0 / (l_row + float(jnp.finfo(jnp.float32).tiny))
      )

      def _load_k_bound(ref):
        if ref is None:
          return None
        r_hi = 0 if ref.shape[0] == 1 else hi
        if ref.shape[-1] == 1:
          return ref[r_hi, 0]
        return plgpu.load(ref.at[r_hi, qs], layout=mma_row, optimized=False)

      k_start_bound = _load_k_bound(k_start_gmem)
      k_end_bound = _load_k_bound(k_end_gmem)

      _async_copy(q_gmem.at[qs, hi], q_smem.at[wg])
      _async_copy(dout_gmem.at[qs, hi], dout_smem.at[wg])
      plgpu.wait_gmem_to_smem(0)
      if q_in_regs_dq:
        q_reg = plgpu.load(q_smem.at[wg], layout=mma_lhs)
        do_reg = plgpu.load(dout_smem.at[wg], layout=mma_lhs)
        _sync_dq()
      else:
        q_reg = None
        do_reg = None
        _sync_dq()

      num_bm = int(bias_smem is not None) + int(mask_smem is not None)
      copies_per_wg = (1 if dq_wgs == 2 else 2) + num_bm

      def _issue_kv_stage(fetch_ki, slot):
        ks = pl.ds(fetch_ki * block_kv_dq, block_kv_dq)

        def _do_cp():
          if dq_wgs == 1:
            _async_copy(k_gmem.at[ks, hi_kv], k_smem.at[slot])
            _async_copy(v_gmem.at[ks, hi_kv], v_smem.at[slot])
          else:
            @pl.when(wg == 0)
            def _cp_k():
              _async_copy(k_gmem.at[ks, hi_kv], k_smem.at[slot])

            @pl.when(wg == 1)
            def _cp_v():
              _async_copy(v_gmem.at[ks, hi_kv], v_smem.at[slot])

          if bias_smem is not None:
            b_hi = 0 if bias_gmem.shape[0] == 1 else hi
            _async_copy(bias_gmem.at[b_hi, qs, ks], bias_smem.at[slot, wg])
          if mask_smem is not None:
            m_hi = 0 if mask_gmem.shape[0] == 1 else hi
            _async_copy(mask_gmem.at[m_hi, qs, ks], mask_smem.at[slot, wg])

        def _do_empty():
          for _ in range(copies_per_wg):
            common.commit_cp_async_group()

        lax.cond(fetch_ki < ub, _do_cp, _do_empty)

      for s_idx in range(stages_dq):
        _issue_kv_stage(lb + s_idx, s_idx)

      bcast_row = lambda x: plgpu.layout_cast(
          lax.broadcast_in_dim(x, (block_q_dq, block_kv_dq), (0,)), mma_acc
      )

      def _make_dq_loop_body(apply_idx: bool):
        def _dq_loop_body(step, dq_acc):
          ki = lb + step
          k_base = ki * block_kv_dq
          ks = pl.ds(k_base, block_kv_dq)
          slot = lax.rem(step, stages_dq)

          plgpu.wait_gmem_to_smem((stages_dq - 1) * copies_per_wg)
          _sync_dq()

          dp = _mma_transposed_rhs(
              do_reg,
              None if q_in_regs_dq else dout_smem.at[wg],
              v_smem.at[slot],
              block_q_dq,
              block_kv_dq,
              head_dim_out,
              step_d_out_dq,
              mma_acc,
              mma_lhs,
              mma_rhs,
          )
          s = _mma_transposed_rhs(
              q_reg,
              None if q_in_regs_dq else q_smem.at[wg],
              k_smem.at[slot],
              block_q_dq,
              block_kv_dq,
              head_dim,
              step_d_dq,
              mma_acc,
              mma_lhs,
              mma_rhs,
          )

          scale = logits_scale
          if bias_gmem is not None:
            if bias_smem is not None:
              b_tile = plgpu.load(
                  bias_smem.at[slot, wg], layout=mma_acc, optimized=False
              )
            else:
              b_tile = common.load_bcast(
                  bias_gmem, (hi, qs, ks), layout=mma_acc, optimized=False
              )
            s = s * scale + b_tile.astype(jnp.float32)
            scale = 1.0

          if logits_soft_cap is not None:
            s = jnp.tanh(s * (scale / logits_soft_cap))
            scale = logits_soft_cap
          logits = s

          p = jnp.exp2(
              s * (scale * math.log2(math.e)) - bcast_row(m_log2)
          ) * bcast_row(l_rcp)

          if apply_idx and has_index_mask:
            q_pos = q_base + plgpu.broadcasted_iota(
                jnp.int32, p.shape, 0, layout=mma_acc
            )
            k_pos = k_base + plgpu.broadcasted_iota(
                jnp.int32, p.shape, 1, layout=mma_acc
            )
            keep = None
            if q_start_gmem is not None:
              keep = q_pos >= q_start_val
            if q_end_gmem is not None:
              c = q_pos < q_end_val
              keep = c if keep is None else (keep & c)
            if is_causal:
              c = q_pos >= k_pos
              keep = c if keep is None else (keep & c)
            if k_start_bound is not None:
              bnd = (
                  k_start_bound
                  if k_start_bound.ndim == 0
                  else bcast_row(k_start_bound)
              )
              c = k_pos >= bnd
              keep = c if keep is None else (keep & c)
            if k_end_bound is not None:
              bnd = (
                  k_end_bound
                  if k_end_bound.ndim == 0
                  else bcast_row(k_end_bound)
              )
              c = k_pos < bnd
              keep = c if keep is None else (keep & c)
            assert keep is not None
            p = jnp.where(keep, p, 0.0)

          if mask_gmem is not None:
            if mask_smem is not None:
              m_tile = (
                  plgpu.load(
                      mask_smem.at[slot, wg], layout=mma_acc, optimized=False
                  )
                  != 0
              )
            else:
              m_tile = (
                  common.load_bcast(
                      mask_gmem, (hi, qs, ks), layout=mma_acc, optimized=False
                  )
                  != 0
              )
            p = jnp.where(m_tile, p, 0.0)

          ds = p * (dp - bcast_row(delta_row))
          if logits_soft_cap is not None:
            ds = ds * (1.0 - logits * logits)

          if ds_gmem is not None:
            assert ds_smems is not None
            _store_swizzled(
                ds_gmem.at[hi, qs, ks],
                ds,
                ds_smems.at[wg],
                jnp.dtype(ds_dtype),
                ds_swizzle,
            )

          ds_lhs = _relayout_acc_to_lhs(ds.astype(dtype), mma_acc, mma_lhs)
          dq_acc = _mma_accumulate_row_rhs(
              dq_acc, ds_lhs, k_smem.at[slot], step_kv_dq, mma_acc, mma_rhs
          )
          _sync_dq()
          _issue_kv_stage(ki + stages_dq, slot)
          return dq_acc

        return _dq_loop_body

      dq_init = plgpu.layout_cast(
          jnp.zeros((block_q_dq, head_dim), jnp.float32), mma_acc
      )
      total_dq_steps = ub - lb
      if not has_index_mask:
        dq_acc = lax.fori_loop(
            0, total_dq_steps, _make_dq_loop_body(False), dq_init
        )
      elif k_start_gmem is not None and k_start_gmem.shape[-1] != 1:
        dq_acc = lax.fori_loop(
            0, total_dq_steps, _make_dq_loop_body(True), dq_init
        )
      else:
        ki_unmasked_lo = (
            lax.max(lb, pl.cdiv(lax.max(k_start_max, 0), block_kv_dq))
            if k_start_gmem is not None
            else lb
        )
        ki_unmasked_hi = ub
        if is_causal:
          ki_unmasked_hi = lax.min(
              ki_unmasked_hi, lax.div(cta_q_base, block_kv_dq)
          )
        if k_end_gmem is not None:
          ki_unmasked_hi = lax.min(
              ki_unmasked_hi, lax.div(lax.max(k_end_min, 0), block_kv_dq)
          )
        ki_unmasked_lo = lax.clamp(lb, ki_unmasked_lo, ub)
        ki_unmasked_hi = lax.clamp(ki_unmasked_lo, ki_unmasked_hi, ub)
        if q_start_gmem is not None or q_end_gmem is not None:
          ki_unmasked_hi = lax.select(
              q_cta_fully_valid, ki_unmasked_hi, ki_unmasked_lo
          )
        dq_start_unmasked = ki_unmasked_lo - lb
        dq_end_unmasked = ki_unmasked_hi - lb
        dq_acc = dq_init
        if k_start_gmem is not None:
          dq_acc = lax.fori_loop(
              0, dq_start_unmasked, _make_dq_loop_body(True), dq_acc
          )
        dq_acc = lax.fori_loop(
            dq_start_unmasked,
            dq_end_unmasked,
            _make_dq_loop_body(False),
            dq_acc,
        )
        dq_acc = lax.fori_loop(
            dq_end_unmasked,
            total_dq_steps,
            _make_dq_loop_body(True),
            dq_acc,
        )
      plgpu.wait_gmem_to_smem(0)
      _sync_dq()
      _store_swizzled(
          dq_gmem.at[qs, hi],
          dq_acc * logits_scale,
          epi_smem.at[wg],
          dtype,
          epi_dq_swizzle,
      )

  if bias is None:
    dq_out_types = (jax.ShapeDtypeStruct(q.shape, q.dtype),)
  else:
    ds_out_shape = (num_q_heads, q_seq_len, kv_seq_len)
    dq_out_types = (
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(ds_out_shape, ds_dtype),
    )

  dq_outputs = plgpu.kernel(
      kernel_dq,
      out_type=dq_out_types,
      scratch_types=dq_scratch,  # pyrefly: ignore[bad-argument-type]
      compiler_params=plgpu.CompilerParams(
          approx_math=True,
          unsafe_no_auto_barriers=True,
          lowering_semantics=plgpu.LoweringSemantics.Lane,
      ),
      grid=(num_q_heads, num_q_tiles_dq),
      grid_names=("heads", "q_tiles"),
      num_threads=dq_wgs if dq_wgs > 1 else None,
      thread_name="wg" if dq_wgs > 1 else None,
      kernel_name=f"flash_attention_vjp_dq_sm80_{dq_wgs}wg",
  )(
      q,
      k,
      v,
      dout,
      m,
      l,
      delta,
      bias,
      mask,
      k_start,
      k_end,
      q_start,
      q_end,
      k_start_minmax,
      k_end_minmax,
  )
  if bias is None:
    (dq,) = dq_outputs
    ds = None
  else:
    dq, ds = dq_outputs

  # ===========================================================================
  # Kernel 2: `kernel_dkv` (computes `dK` and `dV`)
  # ===========================================================================
  bias_t = None if bias is None else bias.mT
  mask_t = None if mask is None else mask.mT

  num_kv_tiles_dkv = kv_seq_len // tile_kv_dkv
  num_q_tiles_dkv = q_seq_len // block_q_dkv
  stages_dkv = max(
      1, min(config.num_stages, num_q_tiles_dkv * q_heads_per_kv_head)
  )

  q_dkv_swizzle = _choose_rhs_swizzle(head_dim, block_q_dkv, elem_bits)
  do_dkv_swizzle = _choose_rhs_swizzle(head_dim_out, block_q_dkv, elem_bits)
  step_d_dkv = 8 * q_dkv_swizzle // elem_bits
  step_d_out_dkv = 8 * do_dkv_swizzle // elem_bits
  step_q_dk = _choose_row_step_k(block_q_dkv, head_dim)
  step_q_dv = _choose_row_step_k(block_q_dkv, head_dim_out)
  kv_in_regs_dkv = False

  stage_bias_dkv = (
      bias_t is not None and bias_t.shape[-2] != 1 and bias_t.shape[-1] != 1
  )
  stage_mask_dkv = (
      mask_t is not None and mask_t.shape[-2] != 1 and mask_t.shape[-1] != 1
  )

  def _build_dkv_scratch(use_bias_smem: bool, use_mask_smem: bool):
    wgs_dim = 1 if dkv_split_mode else dkv_wgs
    k_sc = tiled_smem(
        (wgs_dim, block_kv_dkv, head_dim),
        dtype,
        "k_dkv",
        swizzle=q_dkv_swizzle,
    )
    v_sc = tiled_smem(
        (wgs_dim, block_kv_dkv, head_dim_out),
        dtype,
        "v_dkv",
        swizzle=do_dkv_swizzle,
    )
    q_sc = tiled_smem(
        (stages_dkv, block_q_dkv, head_dim),
        dtype,
        "q_dkv",
        swizzle=q_dkv_swizzle,
    )
    do_sc = tiled_smem(
        (stages_dkv, block_q_dkv, head_dim_out),
        dtype,
        "do_dkv",
        swizzle=do_dkv_swizzle,
    )
    epi_k_sc = tiled_smem(
        (wgs_dim, block_kv_dkv, head_dim), dtype, "epi_dk"
    )
    epi_v_sc = tiled_smem(
        (wgs_dim, block_kv_dkv, head_dim_out), dtype, "epi_dv"
    )
    extras: dict[str, Any] = {}
    if use_bias_smem:
      assert bias_t is not None
      extras["bias_smem"] = tiled_smem(
          (stages_dkv, wgs_dim, block_kv_dkv, block_q_dkv),
          bias_t.dtype,
          "bias_dkv",
      )
    if use_mask_smem:
      extras["mask_smem"] = tiled_smem(
          (stages_dkv, wgs_dim, block_kv_dkv, block_q_dkv),
          jnp.int8,
          "mask_dkv",
      )

    if dkv_split_mode:
      pt_sc = tiled_smem((block_kv_dkv, block_q_dkv), dtype, "pt_dkv")
      dpt_sc = tiled_smem((block_kv_dkv, block_q_dkv), jnp.float32, "dpt_dkv")
      main_union = plgpu.RefUnion(
          (k_sc, v_sc),
          (q_sc, do_sc, pt_sc, dpt_sc, extras),
          (epi_k_sc, epi_v_sc),
      )
    elif kv_in_regs_dkv:
      main_union = plgpu.RefUnion(
          (k_sc, v_sc),
          (q_sc, do_sc, extras),
          (epi_k_sc, epi_v_sc),
      )
    else:
      main_union = plgpu.RefUnion(
          (k_sc, v_sc, q_sc, do_sc, extras),
          (epi_k_sc, epi_v_sc),
      )

    return dict(main_epi_union=main_union)

  dkv_scratch = _build_dkv_scratch(stage_bias_dkv, stage_mask_dkv)
  if mgpu_lib.estimate_smem_bytes(dkv_scratch) > _SMEM_SIZE_BYTES:
    stage_mask_dkv = False
    dkv_scratch = _build_dkv_scratch(stage_bias_dkv, stage_mask_dkv)
  if mgpu_lib.estimate_smem_bytes(dkv_scratch) > _SMEM_SIZE_BYTES:
    stage_bias_dkv = False
    dkv_scratch = _build_dkv_scratch(stage_bias_dkv, stage_mask_dkv)

  epi_dk_swizzle = plgpu.find_swizzle(head_dim * elem_bits, "epi_dk")
  epi_dv_swizzle = plgpu.find_swizzle(head_dim_out * elem_bits, "epi_dv")

  def kernel_dkv(
      q_gmem,
      k_gmem,
      v_gmem,
      dout_gmem,
      m_gmem,
      l_gmem,
      delta_gmem,
      bias_t_gmem,
      mask_t_gmem,
      k_start_gmem,
      k_end_gmem,
      q_start_gmem,
      q_end_gmem,
      dk_gmem,
      dv_gmem,
      *,
      main_epi_union,
  ):
    dkv_threads = 2 if dkv_split_mode else dkv_wgs
    wg = lax.axis_index("wg") if dkv_threads > 1 else 0
    hi_kv = lax.axis_index("heads")
    ki = lax.axis_index("kv_tiles")

    cta_kv_base = ki * tile_kv_dkv
    wg_kv_idx = 0 if dkv_split_mode else wg
    kv_base = cta_kv_base + wg_kv_idx * block_kv_dkv
    ks = pl.ds(kv_base, block_kv_dkv)

    # Scalar KV-range early exit (`key_value_seq_lengths` / scalar `k_start`).
    kv_tile_valid = True
    k_start_fully_in = k_start_gmem is None
    if k_start_gmem is not None and k_start_gmem.shape[-1] == 1:
      if k_start_gmem.shape[0] == 1:
        ks_min_scalar = ks_max_scalar = k_start_gmem[0, 0]
      else:
        ks_min_scalar = ks_max_scalar = k_start_gmem[
            hi_kv * q_heads_per_kv_head, 0
        ]
        for g_idx in range(1, q_heads_per_kv_head):
          val = k_start_gmem[hi_kv * q_heads_per_kv_head + g_idx, 0]
          ks_min_scalar = lax.min(ks_min_scalar, val)
          ks_max_scalar = lax.max(ks_max_scalar, val)
      kv_tile_valid = kv_tile_valid & (
          cta_kv_base + tile_kv_dkv > ks_min_scalar
      )
      k_start_fully_in = cta_kv_base >= ks_max_scalar

    k_end_fully_in = k_end_gmem is None
    if k_end_gmem is not None and k_end_gmem.shape[-1] == 1:
      if k_end_gmem.shape[0] == 1:
        ke_min_scalar = ke_max_scalar = k_end_gmem[0, 0]
      else:
        ke_min_scalar = ke_max_scalar = k_end_gmem[
            hi_kv * q_heads_per_kv_head, 0
        ]
        for g_idx in range(1, q_heads_per_kv_head):
          val = k_end_gmem[hi_kv * q_heads_per_kv_head + g_idx, 0]
          ke_min_scalar = lax.min(ke_min_scalar, val)
          ke_max_scalar = lax.max(ke_max_scalar, val)
      kv_tile_valid = kv_tile_valid & (cta_kv_base < ke_max_scalar)
      k_end_fully_in = cta_kv_base + tile_kv_dkv <= ke_min_scalar

    kv_cta_fully_in_k_range = k_start_fully_in & k_end_fully_in

    # Loop-invariant 1D KV mask check (e.g. AlphaFold `(B, 1, 1, S)` mask).
    has_1d_kv_mask = (
        mask_t_gmem is not None
        and mask_t_gmem.shape[0] == 1
        and mask_t_gmem.shape[1] != 1
        and mask_t_gmem.shape[2] == 1
    )
    if has_1d_kv_mask:
      raw_kv_mask = plgpu.load(
          mask_t_gmem.at[0, ks, 0], layout=mma_row, optimized=False
      )
      kv_mask_row = raw_kv_mask != 0
      if not dkv_split_mode and dkv_wgs == 1:
        kv_tile_valid = kv_tile_valid & (
            jnp.max(raw_kv_mask.astype(jnp.int32)) > 0
        )
    else:
      kv_mask_row = None

    # Q-loop bounds `[lb_q, ub_q)` from `is_causal` and `q_start`/`q_end`.
    lb_q = (
        lax.min(num_q_tiles_dkv, lax.div(cta_kv_base, block_q_dkv))
        if is_causal
        else jnp.int32(0)
    )
    ub_q = jnp.int32(num_q_tiles_dkv)
    q_start_group_max = jnp.int32(0)
    q_end_group_min = jnp.int32(orig_q_seq_len)

    if q_start_gmem is not None:
      if q_start_gmem.shape[0] == 1:
        qs_min = qs_max = q_start_gmem[0, 0]
      else:
        qs_min = qs_max = q_start_gmem[hi_kv * q_heads_per_kv_head, 0]
        for g_idx in range(1, q_heads_per_kv_head):
          val = q_start_gmem[hi_kv * q_heads_per_kv_head + g_idx, 0]
          qs_min = lax.min(qs_min, val)
          qs_max = lax.max(qs_max, val)
      q_start_group_max = qs_max
      lb_q = lax.max(lb_q, lax.div(lax.max(qs_min, 0), block_q_dkv))

    if q_end_gmem is not None:
      if q_end_gmem.shape[0] == 1:
        qe_min = qe_max = q_end_gmem[0, 0]
      else:
        qe_min = qe_max = q_end_gmem[hi_kv * q_heads_per_kv_head, 0]
        for g_idx in range(1, q_heads_per_kv_head):
          val = q_end_gmem[hi_kv * q_heads_per_kv_head + g_idx, 0]
          qe_min = lax.min(qe_min, val)
          qe_max = lax.max(qe_max, val)
      q_end_group_min = lax.min(orig_q_seq_len, qe_min)
      ub_q = lax.min(
          ub_q,
          pl.cdiv(lax.max(lax.min(qe_max, orig_q_seq_len), 0), block_q_dkv),
      )

    cta_active = kv_tile_valid & (ub_q > lb_q)
    if dkv_split_mode:
      (
          (k_smem, v_smem),
          (q_smem, dout_smem, pt_smem, dpt_smem, extras),
          (epi_k_smem, epi_v_smem),
      ) = main_epi_union
    elif kv_in_regs_dkv:
      (
          (k_smem, v_smem),
          (q_smem, dout_smem, extras),
          (epi_k_smem, epi_v_smem),
      ) = main_epi_union
      pt_smem = dpt_smem = None
    else:
      (
          (k_smem, v_smem, q_smem, dout_smem, extras),
          (epi_k_smem, epi_v_smem),
      ) = main_epi_union
      pt_smem = dpt_smem = None

    def _sync_dkv():
      if dkv_threads > 1:
        _cta_barrier()
      else:
        mgpu_lib.warpgroup_barrier()

    @pl.when(jnp.logical_not(cta_active))
    def _dkv_early_exit():
      if dkv_split_mode:
        @pl.when(wg == 0)
        def _zk():
          zk = plgpu.layout_cast(
              jnp.zeros((block_kv_dkv, head_dim), jnp.float32), mma_acc
          )
          _store_swizzled(
              dk_gmem.at[ks, hi_kv], zk, epi_k_smem.at[0], dtype, epi_dk_swizzle
          )

        @pl.when(wg == 1)
        def _zv():
          zv = plgpu.layout_cast(
              jnp.zeros((block_kv_dkv, head_dim_out), jnp.float32), mma_acc
          )
          _store_swizzled(
              dv_gmem.at[ks, hi_kv], zv, epi_v_smem.at[0], dtype, epi_dv_swizzle
          )
      else:
        zk = plgpu.layout_cast(
            jnp.zeros((block_kv_dkv, head_dim), jnp.float32), mma_acc
        )
        zv = plgpu.layout_cast(
            jnp.zeros((block_kv_dkv, head_dim_out), jnp.float32), mma_acc
        )
        _store_swizzled(
            dk_gmem.at[ks, hi_kv], zk, epi_k_smem.at[wg], dtype, epi_dk_swizzle
        )
        _store_swizzled(
            dv_gmem.at[ks, hi_kv], zv, epi_v_smem.at[wg], dtype, epi_dv_swizzle
        )

    @pl.when(cta_active)
    def _dkv_active():
      total_steps = (ub_q - lb_q) * q_heads_per_kv_head
      bcast_col = lambda x: plgpu.layout_cast(
          lax.broadcast_in_dim(x, (block_kv_dkv, block_q_dkv), (1,)), mma_acc
      )
      bcast_row_dkv = lambda x: plgpu.layout_cast(
          lax.broadcast_in_dim(x, (block_kv_dkv, block_q_dkv), (0,)), mma_acc
      )

      def _step_to_qi_hi(step):
        if q_heads_per_kv_head == 1:
          return lb_q + step, hi_kv
        q_idx = lax.div(step, q_heads_per_kv_head)
        g_idx = lax.rem(step, q_heads_per_kv_head)
        return lb_q + q_idx, hi_kv * q_heads_per_kv_head + g_idx

      def _reconstruct_pt(
          s_t, qi, hi, slot, bias_smem, mask_smem, *, apply_idx: bool
      ):
        q_base = qi * block_q_dkv
        qs = pl.ds(q_base, block_q_dkv)
        scale = logits_scale
        if bias_t_gmem is not None:
          if bias_smem is not None:
            b_tile = plgpu.load(
                bias_smem.at[slot, wg_kv_idx], layout=mma_acc, optimized=False
            )
          else:
            b_tile = common.load_bcast(
                bias_t_gmem, (hi, ks, qs), layout=mma_acc, optimized=False
            )
          s_t = s_t * scale + b_tile.astype(jnp.float32)
          scale = 1.0

        if logits_soft_cap is not None:
          s_t = jnp.tanh(s_t * (scale / logits_soft_cap))
          scale = logits_soft_cap
        logits_t = s_t

        m_col = plgpu.load(
            m_gmem.at[hi, qs], layout=mma_col, optimized=False
        )
        l_col = plgpu.load(
            l_gmem.at[hi, qs], layout=mma_col, optimized=False
        )
        m_log2 = m_col * math.log2(math.e)
        l_rcp = jnp.where(
            l_col == 0.0,
            0.0,
            1.0 / (l_col + float(jnp.finfo(jnp.float32).tiny)),
        )
        p_t = jnp.exp2(
            s_t * (scale * math.log2(math.e)) - bcast_col(m_log2)
        ) * bcast_col(l_rcp)

        if apply_idx and has_index_mask:
          k_pos = kv_base + plgpu.broadcasted_iota(
              jnp.int32, p_t.shape, 0, layout=mma_acc
          )
          q_pos = q_base + plgpu.broadcasted_iota(
              jnp.int32, p_t.shape, 1, layout=mma_acc
          )
          keep = None
          if is_causal:
            keep = k_pos <= q_pos
          if q_start_gmem is not None:
            q_hi = 0 if q_start_gmem.shape[0] == 1 else hi
            c = q_pos >= q_start_gmem[q_hi, 0]
            keep = c if keep is None else (keep & c)
          if q_end_gmem is not None:
            q_hi = 0 if q_end_gmem.shape[0] == 1 else hi
            c = q_pos < lax.min(orig_q_seq_len, q_end_gmem[q_hi, 0])
            keep = c if keep is None else (keep & c)
          if k_start_gmem is not None:
            r_hi = 0 if k_start_gmem.shape[0] == 1 else hi
            if k_start_gmem.shape[-1] == 1:
              bnd = k_start_gmem[r_hi, 0]
            else:
              bnd = bcast_col(
                  plgpu.load(
                      k_start_gmem.at[r_hi, qs],
                      layout=mma_col,
                      optimized=False,
                  )
              )
            c = k_pos >= bnd
            keep = c if keep is None else (keep & c)
          if k_end_gmem is not None:
            r_hi = 0 if k_end_gmem.shape[0] == 1 else hi
            if k_end_gmem.shape[-1] == 1:
              bnd = k_end_gmem[r_hi, 0]
            else:
              bnd = bcast_col(
                  plgpu.load(
                      k_end_gmem.at[r_hi, qs],
                      layout=mma_col,
                      optimized=False,
                  )
              )
            c = k_pos < bnd
            keep = c if keep is None else (keep & c)
          assert keep is not None
          p_t = jnp.where(keep, p_t, 0.0)

        if mask_t_gmem is not None:
          if mask_smem is not None:
            m_tile = (
                plgpu.load(
                    mask_smem.at[slot, wg_kv_idx],
                    layout=mma_acc,
                    optimized=False,
                )
                != 0
            )
          elif kv_mask_row is not None:
            m_tile = bcast_row_dkv(kv_mask_row)
          else:
            m_tile = (
                common.load_bcast(
                    mask_t_gmem, (hi, ks, qs), layout=mma_acc, optimized=False
                )
                != 0
            )
          p_t = jnp.where(m_tile, p_t, 0.0)

        return p_t, logits_t

      if not has_index_mask:
        def _run_dkv_loops(make_body, init_carry):
          return lax.fori_loop(0, total_steps, make_body(False), init_carry)
      else:
        qi_unmasked_lo = lb_q
        if is_causal:
          qi_unmasked_lo = lax.max(
              qi_unmasked_lo,
              pl.cdiv(cta_kv_base + tile_kv_dkv, block_q_dkv),
          )
        if q_start_gmem is not None:
          qi_unmasked_lo = lax.max(
              qi_unmasked_lo,
              pl.cdiv(lax.max(q_start_group_max, 0), block_q_dkv),
          )
        qi_unmasked_hi = ub_q
        if q_end_gmem is not None:
          qi_unmasked_hi = lax.min(
              qi_unmasked_hi,
              lax.div(lax.max(q_end_group_min, 0), block_q_dkv),
          )
        qi_unmasked_lo = lax.clamp(lb_q, qi_unmasked_lo, ub_q)
        qi_unmasked_hi = lax.clamp(qi_unmasked_lo, qi_unmasked_hi, ub_q)
        if k_start_gmem is not None or k_end_gmem is not None:
          qi_unmasked_hi = lax.select(
              jnp.asarray(kv_cta_fully_in_k_range),
              qi_unmasked_hi,
              qi_unmasked_lo,
          )
        step_unmasked_lo = (qi_unmasked_lo - lb_q) * q_heads_per_kv_head
        step_unmasked_hi = (qi_unmasked_hi - lb_q) * q_heads_per_kv_head

        def _run_dkv_loops(make_body, init_carry):
          carry = init_carry
          if is_causal or q_start_gmem is not None:
            carry = lax.fori_loop(0, step_unmasked_lo, make_body(True), carry)
          carry = lax.fori_loop(
              step_unmasked_lo, step_unmasked_hi, make_body(False), carry
          )
          if (
              q_end_gmem is not None
              or k_start_gmem is not None
              or k_end_gmem is not None
          ):
            carry = lax.fori_loop(
                step_unmasked_hi, total_steps, make_body(True), carry
            )
          return carry

      # -----------------------------------------------------------------------
      # Sub-mode 2A: Cooperative Split-dK/dV 2-WG (D + d > 256)
      # -----------------------------------------------------------------------
      if dkv_split_mode:
        assert pt_smem is not None and dpt_smem is not None
        bias_smem = extras.get("bias_smem")
        mask_smem = extras.get("mask_smem")

        wg0_copies = (
            1 + int(bias_smem is not None) + int(mask_smem is not None)
        )
        wg1_copies = 1

        def _issue_wg0_stage(step_idx, slot):
          fetch_qi, fetch_hi = _step_to_qi_hi(step_idx)
          f_qs = pl.ds(fetch_qi * block_q_dkv, block_q_dkv)

          def _do():
            _async_copy(q_gmem.at[f_qs, fetch_hi], q_smem.at[slot])
            if bias_smem is not None:
              b_hi = 0 if bias_t_gmem.shape[0] == 1 else fetch_hi
              _async_copy(
                  bias_t_gmem.at[b_hi, ks, f_qs], bias_smem.at[slot, 0]
              )
            if mask_smem is not None:
              m_hi = 0 if mask_t_gmem.shape[0] == 1 else fetch_hi
              _async_copy(
                  mask_t_gmem.at[m_hi, ks, f_qs], mask_smem.at[slot, 0]
              )

          def _empty():
            for _ in range(wg0_copies):
              common.commit_cp_async_group()

          lax.cond(step_idx < total_steps, _do, _empty)

        def _issue_wg1_stage(step_idx, slot):
          fetch_qi, fetch_hi = _step_to_qi_hi(step_idx)
          f_qs = pl.ds(fetch_qi * block_q_dkv, block_q_dkv)
          lax.cond(
              step_idx < total_steps,
              lambda: _async_copy(
                  dout_gmem.at[f_qs, fetch_hi], dout_smem.at[slot]
              ),
              common.commit_cp_async_group,
          )

        @pl.when(wg == 0)
        def _run_dk_wg():
          _async_copy(k_gmem.at[ks, hi_kv], k_smem.at[0])
          plgpu.wait_gmem_to_smem(0)
          k_reg = plgpu.load(k_smem.at[0], layout=mma_lhs)
          _cta_barrier()
          for s_idx in range(stages_dkv):
            _issue_wg0_stage(s_idx, s_idx)

          def _make_dk_body(apply_idx: bool):
            def _dk_body(step, dk_acc):
              qi, hi = _step_to_qi_hi(step)
              qs = pl.ds(qi * block_q_dkv, block_q_dkv)
              slot = lax.rem(step, stages_dkv)

              plgpu.wait_gmem_to_smem((stages_dkv - 1) * wg0_copies)
              _cta_barrier()

              s_t = _mma_transposed_rhs(
                  k_reg,
                  None,
                  q_smem.at[slot],
                  block_kv_dkv,
                  block_q_dkv,
                  head_dim,
                  step_d_dkv,
                  mma_acc,
                  mma_lhs,
                  mma_rhs,
              )
              p_t, logits_t = _reconstruct_pt(
                  s_t,
                  qi,
                  hi,
                  slot,
                  bias_smem,
                  mask_smem,
                  apply_idx=apply_idx,
              )
              plgpu.store(pt_smem, p_t.astype(dtype), optimized=False)
              _cta_barrier()

              dp_t = plgpu.load(dpt_smem, layout=mma_acc, optimized=False)
              delta_col = plgpu.load(
                  delta_gmem.at[hi, qs], layout=mma_col, optimized=False
              )
              ds_t = p_t * (dp_t - bcast_col(delta_col))
              if logits_soft_cap is not None:
                ds_t = ds_t * (1.0 - logits_t * logits_t)

              ds_t_lhs = _relayout_acc_to_lhs(
                  ds_t.astype(dtype), mma_acc, mma_lhs
              )
              dk_acc = _mma_accumulate_row_rhs(
                  dk_acc, ds_t_lhs, q_smem.at[slot], step_q_dk, mma_acc, mma_rhs
              )
              _cta_barrier()
              _issue_wg0_stage(step + stages_dkv, slot)
              return dk_acc

            return _dk_body

          dk_init = plgpu.layout_cast(
              jnp.zeros((block_kv_dkv, head_dim), jnp.float32), mma_acc
          )
          dk_acc = _run_dkv_loops(_make_dk_body, dk_init)
          plgpu.wait_gmem_to_smem(0)
          _cta_barrier()
          _store_swizzled(
              dk_gmem.at[ks, hi_kv],
              dk_acc * logits_scale,
              epi_k_smem.at[0],
              dtype,
              epi_dk_swizzle,
          )

        @pl.when(wg == 1)
        def _run_dv_wg():
          _async_copy(v_gmem.at[ks, hi_kv], v_smem.at[0])
          plgpu.wait_gmem_to_smem(0)
          v_reg = plgpu.load(v_smem.at[0], layout=mma_lhs)
          _cta_barrier()
          for s_idx in range(stages_dkv):
            _issue_wg1_stage(s_idx, s_idx)

          def _dv_body(step, dv_acc):
            slot = lax.rem(step, stages_dkv)

            plgpu.wait_gmem_to_smem((stages_dkv - 1) * wg1_copies)
            _cta_barrier()

            dp_t = _mma_transposed_rhs(
                v_reg,
                None,
                dout_smem.at[slot],
                block_kv_dkv,
                block_q_dkv,
                head_dim_out,
                step_d_out_dkv,
                mma_acc,
                mma_lhs,
                mma_rhs,
            )
            plgpu.store(dpt_smem, dp_t, optimized=False)
            _cta_barrier()

            pt_lhs = plgpu.load(pt_smem, layout=mma_lhs)
            dv_acc = _mma_accumulate_row_rhs(
                dv_acc, pt_lhs, dout_smem.at[slot], step_q_dv, mma_acc, mma_rhs
            )
            _cta_barrier()
            _issue_wg1_stage(step + stages_dkv, slot)
            return dv_acc

          dv_init = plgpu.layout_cast(
              jnp.zeros((block_kv_dkv, head_dim_out), jnp.float32), mma_acc
          )
          dv_acc = lax.fori_loop(0, total_steps, _dv_body, dv_init)
          plgpu.wait_gmem_to_smem(0)
          _cta_barrier()
          _store_swizzled(
              dv_gmem.at[ks, hi_kv],
              dv_acc,
              epi_v_smem.at[0],
              dtype,
              epi_dv_swizzle,
          )
        return

      # -----------------------------------------------------------------------
      # Sub-mode 2B: KV-Parallel Mode (`head_dim + head_dim_out <= 256`)
      # -----------------------------------------------------------------------
      bias_smem = extras.get("bias_smem")
      mask_smem = extras.get("mask_smem")

      _async_copy(k_gmem.at[ks, hi_kv], k_smem.at[wg])
      _async_copy(v_gmem.at[ks, hi_kv], v_smem.at[wg])
      plgpu.wait_gmem_to_smem(0)
      if kv_in_regs_dkv:
        k_reg = plgpu.load(k_smem.at[wg], layout=mma_lhs)
        v_reg = plgpu.load(v_smem.at[wg], layout=mma_lhs)
        _sync_dkv()
      else:
        k_reg = None
        v_reg = None
        _sync_dkv()

      num_bm = int(bias_smem is not None) + int(mask_smem is not None)
      copies_per_wg = (1 if dkv_wgs == 2 else 2) + num_bm

      def _issue_q_stage(step_idx, slot):
        fetch_qi, fetch_hi = _step_to_qi_hi(step_idx)
        f_qs = pl.ds(fetch_qi * block_q_dkv, block_q_dkv)

        def _do_cp():
          if dkv_wgs == 1:
            _async_copy(q_gmem.at[f_qs, fetch_hi], q_smem.at[slot])
            _async_copy(dout_gmem.at[f_qs, fetch_hi], dout_smem.at[slot])
          else:
            @pl.when(wg == 0)
            def _cp_q():
              _async_copy(q_gmem.at[f_qs, fetch_hi], q_smem.at[slot])

            @pl.when(wg == 1)
            def _cp_do():
              _async_copy(dout_gmem.at[f_qs, fetch_hi], dout_smem.at[slot])

          if bias_smem is not None:
            b_hi = 0 if bias_t_gmem.shape[0] == 1 else fetch_hi
            _async_copy(
                bias_t_gmem.at[b_hi, ks, f_qs], bias_smem.at[slot, wg]
            )
          if mask_smem is not None:
            m_hi = 0 if mask_t_gmem.shape[0] == 1 else fetch_hi
            _async_copy(
                mask_t_gmem.at[m_hi, ks, f_qs], mask_smem.at[slot, wg]
            )

        def _do_empty():
          for _ in range(copies_per_wg):
            common.commit_cp_async_group()

        lax.cond(step_idx < total_steps, _do_cp, _do_empty)

      for s_idx in range(stages_dkv):
        _issue_q_stage(s_idx, s_idx)

      def _make_dkv_loop_body(apply_idx: bool):
        def _dkv_loop_body(step, carry):
          dk_acc, dv_acc = carry
          qi, hi = _step_to_qi_hi(step)
          qs = pl.ds(qi * block_q_dkv, block_q_dkv)
          slot = lax.rem(step, stages_dkv)

          plgpu.wait_gmem_to_smem((stages_dkv - 1) * copies_per_wg)
          _sync_dkv()

          dp_t = _mma_transposed_rhs(
              v_reg,
              None if kv_in_regs_dkv else v_smem.at[wg],
              dout_smem.at[slot],
              block_kv_dkv,
              block_q_dkv,
              head_dim_out,
              step_d_out_dkv,
              mma_acc,
              mma_lhs,
              mma_rhs,
          )
          s_t = _mma_transposed_rhs(
              k_reg,
              None if kv_in_regs_dkv else k_smem.at[wg],
              q_smem.at[slot],
              block_kv_dkv,
              block_q_dkv,
              head_dim,
              step_d_dkv,
              mma_acc,
              mma_lhs,
              mma_rhs,
          )
          p_t, logits_t = _reconstruct_pt(
              s_t,
              qi,
              hi,
              slot,
              bias_smem,
              mask_smem,
              apply_idx=apply_idx,
          )

          pt_lhs = _relayout_acc_to_lhs(p_t.astype(dtype), mma_acc, mma_lhs)
          dv_acc = _mma_accumulate_row_rhs(
              dv_acc, pt_lhs, dout_smem.at[slot], step_q_dv, mma_acc, mma_rhs
          )

          delta_col = plgpu.load(
              delta_gmem.at[hi, qs], layout=mma_col, optimized=False
          )
          ds_t = p_t * (dp_t - bcast_col(delta_col))
          if logits_soft_cap is not None:
            ds_t = ds_t * (1.0 - logits_t * logits_t)

          ds_t_lhs = _relayout_acc_to_lhs(ds_t.astype(dtype), mma_acc, mma_lhs)
          dk_acc = _mma_accumulate_row_rhs(
              dk_acc, ds_t_lhs, q_smem.at[slot], step_q_dk, mma_acc, mma_rhs
          )
          _sync_dkv()
          _issue_q_stage(step + stages_dkv, slot)
          return dk_acc, dv_acc

        return _dkv_loop_body

      dk_init = plgpu.layout_cast(
          jnp.zeros((block_kv_dkv, head_dim), jnp.float32), mma_acc
      )
      dv_init = plgpu.layout_cast(
          jnp.zeros((block_kv_dkv, head_dim_out), jnp.float32), mma_acc
      )
      dk_acc, dv_acc = _run_dkv_loops(
          _make_dkv_loop_body, (dk_init, dv_init)
      )
      plgpu.wait_gmem_to_smem(0)
      _sync_dkv()
      _store_swizzled(
          dk_gmem.at[ks, hi_kv],
          dk_acc * logits_scale,
          epi_k_smem.at[wg],
          dtype,
          epi_dk_swizzle,
      )
      _store_swizzled(
          dv_gmem.at[ks, hi_kv],
          dv_acc,
          epi_v_smem.at[wg],
          dtype,
          epi_dv_swizzle,
      )

  dkv_num_threads = 2 if dkv_split_mode else dkv_wgs
  dk, dv = plgpu.kernel(
      kernel_dkv,
      out_type=(
          jax.ShapeDtypeStruct(k.shape, k.dtype),
          jax.ShapeDtypeStruct(v.shape, v.dtype),
      ),
      scratch_types=dkv_scratch,  # pyrefly: ignore[bad-argument-type]
      compiler_params=plgpu.CompilerParams(
          approx_math=True,
          unsafe_no_auto_barriers=True,
          lowering_semantics=plgpu.LoweringSemantics.Lane,
      ),
      grid=(num_kv_heads, num_kv_tiles_dkv),
      grid_names=("heads", "kv_tiles"),
      num_threads=dkv_num_threads if dkv_num_threads > 1 else None,
      thread_name="wg" if dkv_num_threads > 1 else None,
      kernel_name=(
          "flash_attention_vjp_dkv_sm80_split"
          if dkv_split_mode
          else f"flash_attention_vjp_dkv_sm80_{dkv_wgs}wg"
      ),
  )(
      q,
      k,
      v,
      dout,
      m,
      l,
      delta,
      bias_t,
      mask_t,
      k_start,
      k_end,
      q_start,
      q_end,
  )

  dq = dq[:orig_q_seq_len, :, :orig_head_dim]
  dk = dk[:orig_kv_seq_len, :, :orig_head_dim]
  dv = dv[:orig_kv_seq_len, :, :orig_head_dim_out]
  ds = None if ds is None else ds[:, :orig_q_seq_len, :orig_kv_seq_len]
  return dq, dk, dv, ds
