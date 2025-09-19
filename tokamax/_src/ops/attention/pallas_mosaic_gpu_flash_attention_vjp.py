# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Pallas-Mosaic-GPU FlashAttention VJP implementation."""

# pylint: disable=invalid-name

import dataclasses
import functools
import math
from typing import ClassVar, TypeAlias

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src import shape as shape_lib
from tokamax._src.ops import op
from tokamax._src.ops.attention import base

Mask = base.Mask
Residuals = base.Residuals
PagingInfo = base.PagingInfo

L: TypeAlias = plgpu.Layout

_WGMMA = plgpu.Layout.WGMMA
_WGMMA_COL = plgpu.Layout.WGMMA.reduce(0)
_WGMMA_ROW = plgpu.Layout.WGMMA.reduce(1)


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  block_q_dkv: pydantic.conint(multiple_of=64, gt=0)
  block_kv_dkv: pydantic.conint(multiple_of=64, gt=0)
  block_q_dq: pydantic.PositiveInt
  block_kv_dq: pydantic.PositiveInt
  num_stages: pydantic.conint(gt=1) = 2
  compute_wgs: pydantic.PositiveInt = 2


@jaxtyping.jaxtyped
def _bwd(
    q: Float[Array, "*B T H D"],
    k: Float[Array, "*B t h D"],
    v: Float[Array, "*B t h d"],
    residuals: Residuals,
    out: Float[Array, "*B T H d"],
    dout: Float[Array, "*B T H d"],
    *,
    bias: Float[Array, "*#B #H #T #t"] | None,
    mask: Bool[Array, "*#B #H #T #t"] | None,
    k_start: Int[Array, "*#B #H #T"] | None,
    k_end: Int[Array, "*#B #H #T"] | None,
    logits_scale: float,
    logits_soft_cap: float | None,
    use_base2: bool,
    config: Config,
) -> tuple[
    Float[Array, "*B T H D"],  # dq
    Float[Array, "*B t h D"],  # dk
    Float[Array, "*B t h d"],  # dv
    Float[Array, "*#B #H #T #t"] | None,  # dbias
]:
  orig_q_shape = q.shape
  orig_k_shape = k.shape
  orig_v_shape = v.shape
  as_ndim = lambda x, ndim: jax.lax.collapse(
      jax.lax.broadcast_to_rank(x, ndim), 0, -ndim + 1
  )
  as_3d = lambda x: as_ndim(x, 3)
  as_4d = lambda x: as_ndim(x, 4)
  pad_head_dim = lambda x: shape_lib.pad_to_next_multiple_of(x, 64, -1)

  q, k, v, out, dout = map(as_4d, (q, k, v, out, dout))
  q, k, v, out, dout = map(pad_head_dim, (q, k, v, out, dout))
  m, l = map(as_3d, residuals)

  batch_size, q_seq_len, num_q_heads, head_dim = q.shape
  _, kv_seq_len, num_kv_heads, head_dim_out = v.shape
  if (dtype := q.dtype) != k.dtype or dtype != v.dtype:
    raise ValueError(
        f"q, k, and v should all have the same dtype, got: {q.dtype},"
        f" {k.dtype}, {v.dtype}"
    )
  if jnp.dtype(dtype) not in (jnp.float16, jnp.bfloat16):
    raise NotImplementedError(
        f"Only f16 and bf16 are supported, got dtype: {dtype}"
    )
  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by and {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  compute_wgs = config.compute_wgs
  num_q_tiles, rem = divmod(q_seq_len, config.block_q_dq * compute_wgs)
  if rem:
    raise NotImplementedError(
        f"{q_seq_len=} must be a multiple of {config.block_q_dq=} *"
        f" {compute_wgs=}"
    )

  num_kv_tiles, rem = divmod(kv_seq_len, config.block_kv_dkv * compute_wgs)
  if rem:
    raise NotImplementedError(
        f"{kv_seq_len=} must be a multiple of {config.block_kv_dkv=} *"
        f" {compute_wgs=}"
    )

  num_q_tiles_in_dkv, rem = divmod(q_seq_len, config.block_q_dkv)
  if rem:
    raise NotImplementedError(
        f"{q_seq_len=} must be a multiple of {config.block_q_dkv=}"
    )

  num_kv_tiles_in_dq, rem = divmod(kv_seq_len, config.block_kv_dq)
  if rem:
    raise NotImplementedError(
        f"{kv_seq_len=} must be a multiple of {config.block_kv_dq=}"
    )

  if bias is not None:
    orig_bias_shape = bias.shape
    bias = as_4d(bias)
  if mask is not None:
    mask = as_4d(mask).astype(jnp.int8)

  # TODO: Avoid broadcast.
  bcast = lambda x: jnp.broadcast_to(x, (batch_size, x.shape[-2], q_seq_len))
  k_start = None if k_start is None else bcast(k_start)
  k_end = None if k_end is None else bcast(k_end)

  swizzle = 128
  transforms = (
      plgpu.TilingTransform((8, swizzle // q.dtype.itemsize)),
      plgpu.SwizzleTransform(swizzle),
  )
  delta = jnp.einsum(
      "bqhd,bqhd->bhq", out.astype(jnp.float32), dout.astype(jnp.float32)
  )

  exp = jnp.exp2 if use_base2 else jnp.exp

  def bias_mask_info(x_ref, b_idx, q_head, name):
    if x_ref is None:
      return (None,) * 4 + (False,)
    bcast_b, bcast_h, bcast_q, bcast_k = [d == 1 for d in x_ref.shape]
    if bcast_q and bcast_k:
      raise NotImplementedError(f"{name} broadcast on both sequences.")
    b_idx = 0 if bcast_b else b_idx
    h_idx = 0 if bcast_h else q_head
    return b_idx, h_idx, bcast_q, bcast_k, not (bcast_q or bcast_k)

  def load_bias_mask(
      s, x_ref, smems, smem_idx, barrier, bcast_q_slice, bcast_k_slice
  ):
    if x_ref is None:
      return None
    if bcast_q_slice is not None:
      x = plgpu.load(x_ref, bcast_q_slice, layout=_WGMMA_COL, optimized=False)
      return lax.broadcast_in_dim(x, s.shape, [1])
    if bcast_k_slice is not None:
      x = plgpu.load(x_ref, bcast_k_slice, layout=_WGMMA_ROW, optimized=False)
      return lax.broadcast_in_dim(x, s.shape, [0])
    if barrier is not None:
      plgpu.barrier_arrive(barrier)
    return smems[smem_idx]

  def bias_mask_async_spec(x_ref, is_async, block_q, block_kv, idx, name):
    if not is_async:
      return None
    bytes_ = jnp.dtype(x_ref.dtype).itemsize
    swizzle = plgpu.find_swizzle(8 * bytes_ * block_kv, name)
    transforms = (
        plgpu.TilingTransform((8, swizzle // bytes_)),
        plgpu.SwizzleTransform(swizzle),
    )
    return plgpu.BlockSpec(
        block_shape=(compute_wgs * block_q, block_kv),
        index_map=lambda i: (idx, i),
        transforms=transforms,
    )

  def kernel_dq(
      q_ref,
      k_ref,
      v_ref,
      dout_ref,
      m_ref,
      l_ref,
      delta_ref,
      bias_ref,
      mask_ref,
      k_start_ref,
      k_end_ref,
      dq_ref,
      ds_ref,
      smem_buffers,
      buffer_barriers,
      block_q: int,
      block_kv: int,
  ):
    b_idx = lax.axis_index("batch")
    q_idx = lax.axis_index("q_tiles")
    q_head = lax.axis_index("heads")
    wg_idx = lax.axis_index("wg")
    kv_head = lax.div(q_head, jnp.array(q_heads_per_kv_head, q_head.dtype))

    q_smems, dout_smems, m_smems, l_smems, delta_smems = smem_buffers
    q_barriers, dout_barriers, m_barriers, l_barriers, delta_barriers = (
        buffer_barriers
    )

    bias_b_idx, bias_h_idx, bcast_bias_q, bcast_bias_k, async_bias = (
        bias_mask_info(bias_ref, b_idx, q_head, "bias")
    )
    mask_b_idx, mask_h_idx, bcast_mask_q, bcast_mask_k, async_mask = (
        bias_mask_info(mask_ref, b_idx, q_head, "mask")
    )

    def compute_thread(pipeline_callback):
      q_smem = q_smems.at[wg_idx]
      dout_smem = dout_smems.at[wg_idx]
      m_smem = m_smems.at[wg_idx]
      l_smem = l_smems.at[wg_idx]
      delta_smem = delta_smems.at[wg_idx]

      q_seq_base = q_idx * (compute_wgs * block_q) + wg_idx * block_q
      q_slice = (b_idx, pl.ds(q_seq_base, block_q), q_head)
      res_slice = (b_idx, q_head, pl.ds(q_seq_base, block_q))

      plgpu.copy_gmem_to_smem(q_ref.at[q_slice], q_smem, q_barriers.at[wg_idx])
      plgpu.copy_gmem_to_smem(
          dout_ref.at[q_slice], dout_smem, dout_barriers.at[wg_idx]
      )
      plgpu.copy_gmem_to_smem(
          delta_ref.at[res_slice], delta_smem, delta_barriers.at[wg_idx]
      )
      plgpu.copy_gmem_to_smem(
          m_ref.at[res_slice], m_smem, m_barriers.at[wg_idx]
      )
      plgpu.copy_gmem_to_smem(
          l_ref.at[res_slice], l_smem, l_barriers.at[wg_idx]
      )
      _ = [plgpu.barrier_wait(buffer.at[wg_idx]) for buffer in buffer_barriers]

      delta = plgpu.load(delta_smem, (), layout=L.WGMMA.reduce(1))  # [block_q]
      m = plgpu.load(m_smem, (), layout=L.WGMMA.reduce(1))  # [block_q]
      if use_base2:
        m *= math.log2(math.e)
      l = plgpu.load(l_smem, (), layout=L.WGMMA.reduce(1))  # [block_q]
      dq_acc = plgpu.layout_cast(
          jnp.full((block_q, head_dim), 0, dtype=jnp.float32),
          L.WGMMA,
      )

      def load_k_range(ref):
        if ref is None:
          return None
        qs = pl.ds(q_seq_base, block_q)
        idx = (b_idx, 0 if (ref.shape[1] == 1) else q_head, qs)
        return plgpu.load(ref, idx, layout=_WGMMA_ROW, optimized=False)

      k_start = load_k_range(k_start_ref)
      k_end = load_k_range(k_end_ref)
      dq, _, _, _, _, _ = pipeline_callback(
          (dq_acc, m, l, delta, k_start, k_end)
      )
      q_smem[...] = dq.astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(q_smem, dq_ref.at[q_slice])
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    # TODO: If bias/mask are broadcast along k, we can load outside the
    # pipeline as they are not dependent on kv_step.
    def kv_pipeline(
        index,
        k_smem,
        v_smem,
        bias_smems,
        mask_smems,
        k_consumed_barrier,
        v_consumed_barrier,
        bias_consumed_barrier,
        mask_consumed_barrier,
        carry,
    ):
      kv_step = index[0]
      kv_base = kv_step * block_kv
      q_seq_base = q_idx * (compute_wgs * block_q) + wg_idx * block_q
      q_smem, dout_smem = q_smems.at[wg_idx], dout_smems.at[wg_idx]
      (dq_acc, m, l, delta, k_start, k_end) = carry

      qs = pl.ds(q_seq_base, block_q)
      ks = pl.ds(kv_base, block_kv)

      def compute_s(acc_ref):
        plgpu.wgmma(acc_ref, q_smem, plgpu.transpose_ref(k_smem, (1, 0)))
        return acc_ref[...]

      s = pl.run_scoped(compute_s, plgpu.ACC((block_q, block_kv), jnp.float32))
      s *= logits_scale

      bias = load_bias_mask(
          s=s,
          x_ref=bias_ref,
          smems=bias_smems,
          smem_idx=pl.ds(wg_idx * block_q, block_q),
          barrier=bias_consumed_barrier,
          bcast_q_slice=(0, ks) if bcast_bias_q else None,
          bcast_k_slice=(qs, 0) if bcast_bias_k else None,
      )
      if bias is not None:
        s += bias

      if logits_soft_cap is not None:
        logits = jnp.tanh(s / logits_soft_cap)
        s = logits_soft_cap * logits

      # NOTE: This rescaling must happen after bias and soft-cap but before the
      # attention masking (as the multiplication will cause `-inf`s).
      if use_base2:
        s *= math.log2(math.e)

      mask_value = float(jnp.finfo(jnp.float32).min)

      def iota(d):
        return plgpu.broadcasted_iota(jnp.int32, s.shape, d, layout=_WGMMA)

      if k_start is not None:
        _start = lax.broadcast_in_dim(k_start, s.shape, [0])
        s = jnp.where(kv_base + iota(1) >= _start, s, mask_value)

      if k_end is not None:
        _end = lax.broadcast_in_dim(k_end, s.shape, [0])
        s = jnp.where(kv_base + iota(1) < _end, s, mask_value)

      mask = load_bias_mask(
          s=s,
          x_ref=mask_ref,
          smems=mask_smems,
          smem_idx=pl.ds(wg_idx * block_q, block_q),
          barrier=mask_consumed_barrier,
          bcast_q_slice=(0, ks) if bcast_mask_q else None,
          bcast_k_slice=(qs, 0) if bcast_mask_k else None,
      )
      if mask is not None:
        s = jnp.where(mask, s, mask_value)

      broadcast = lambda x: lax.broadcast_in_dim(x, (block_q, block_kv), [0])
      p = exp(s - broadcast(m)) / broadcast(l)

      def compute_dp(acc_ref):
        plgpu.wgmma(acc_ref, dout_smem, plgpu.transpose_ref(v_smem, (1, 0)))
        return acc_ref[...]

      dp = pl.run_scoped(
          compute_dp, plgpu.ACC((block_q, block_kv), jnp.float32)
      )
      plgpu.barrier_arrive(v_consumed_barrier)

      ds = p * (dp - lax.broadcast_in_dim(delta, (block_q, block_kv), [0]))
      if logits_soft_cap is not None:
        ds *= 1 - jnp.pow(logits, 2)

      if ds_ref is not None:
        # TODO: Make this store non-blocking.
        ds_ref[b_idx, q_head, qs, ks] = ds.astype(ds_ref.dtype)

      ds *= logits_scale

      def compute_dq(acc_ref):
        plgpu.wgmma(acc_ref, ds.astype(k_ref.dtype), k_smem)

      dq_acc = pl.run_state(compute_dq)(plgpu.ACC.init(dq_acc))
      plgpu.barrier_arrive(k_consumed_barrier)

      return (dq_acc, m, l, delta, k_start, k_end)

    bias_in_spec = bias_mask_async_spec(
        bias_ref, async_bias, block_q, block_kv, q_idx, "bias"
    )
    mask_in_spec = bias_mask_async_spec(
        mask_ref, async_mask, block_q, block_kv, q_idx, "mask"
    )

    pipeline = plgpu.emit_pipeline_warp_specialized(
        kv_pipeline,
        grid=(num_kv_tiles_in_dq,),
        max_concurrent_steps=min(config.num_stages, num_kv_tiles_in_dq),
        num_compute_wgs=compute_wgs,
        memory_registers=40,
        wg_axis="wg",
        manual_consumed_barriers=True,
        compute_context=compute_thread,
        in_specs=[
            plgpu.BlockSpec(  # k
                block_shape=(block_kv, head_dim),
                index_map=lambda i: (i, 0),
                transforms=transforms,
            ),
            plgpu.BlockSpec(  # v
                block_shape=(block_kv, head_dim_out),
                index_map=lambda i: (i, 0),
                transforms=transforms,
            ),
            bias_in_spec,
            mask_in_spec,
        ],
    )
    k_ref = k_ref.at[b_idx, :, kv_head, :]
    v_ref = v_ref.at[b_idx, :, kv_head, :]
    if bias_ref is not None:
      bias_ref = bias_ref.at[bias_b_idx, bias_h_idx, :, :]
    if mask_ref is not None:
      mask_ref = mask_ref.at[mask_b_idx, mask_h_idx, :, :]

    pipeline(
        k_ref,
        v_ref,
        bias_ref if async_bias else None,
        mask_ref if async_mask else None,
    )

  def kernel_dkv(
      q_ref,
      k_ref,
      v_ref,
      dout_ref,
      m_ref,
      l_ref,
      delta_ref,
      bias_ref,
      mask_ref,
      k_start_ref,
      k_end_ref,
      dk_ref,
      dv_ref,
      smem_buffers,
      buffer_barriers,
      block_q: int,
      block_kv: int,
  ):
    b_idx = lax.axis_index("batch")
    kv_idx = lax.axis_index("num_kv_tiles")
    q_head = lax.axis_index("heads")
    wg_idx = lax.axis_index("wg")
    (k_smems, v_smems) = smem_buffers
    (k_barriers, v_barriers) = buffer_barriers

    bias_b_idx, bias_h_idx, bcast_bias_k, bcast_bias_q, async_bias = (
        bias_mask_info(bias_ref, b_idx, q_head, "bias")
    )
    mask_b_idx, mask_h_idx, bcast_mask_k, bcast_mask_q, async_mask = (
        bias_mask_info(mask_ref, b_idx, q_head, "mask")
    )

    def compute_thread(pipeline_callback):
      k_smem, v_smem = k_smems.at[wg_idx], v_smems.at[wg_idx]
      kv_seq_base = kv_idx * (compute_wgs * block_kv) + wg_idx * block_kv
      kv_head = lax.div(q_head, jnp.array(q_heads_per_kv_head, q_head.dtype))
      kv_slice = (b_idx, pl.ds(kv_seq_base, block_kv), kv_head)
      plgpu.copy_gmem_to_smem(k_ref.at[kv_slice], k_smem, k_barriers.at[wg_idx])
      plgpu.copy_gmem_to_smem(v_ref.at[kv_slice], v_smem, v_barriers.at[wg_idx])
      plgpu.barrier_wait(k_barriers.at[wg_idx])
      plgpu.barrier_wait(v_barriers.at[wg_idx])
      dk_acc = plgpu.layout_cast(
          jnp.full((block_kv, head_dim), 0, dtype=jnp.float32),
          L.WGMMA,
      )
      dv_acc = plgpu.layout_cast(
          jnp.full((block_kv, head_dim_out), 0, dtype=jnp.float32),
          L.WGMMA,
      )
      dk, dv = pipeline_callback((dk_acc, dv_acc))
      k_smem[...] = dk.astype(k.dtype)
      v_smem[...] = dv.astype(v.dtype)

      plgpu.commit_smem()
      kv_out_slice = (b_idx, pl.ds(kv_seq_base, block_kv), q_head)
      plgpu.copy_smem_to_gmem(
          k_smem, dk_ref.at[kv_out_slice], commit_group=False
      )
      plgpu.copy_smem_to_gmem(
          v_smem, dv_ref.at[kv_out_slice], commit_group=False
      )
      plgpu.commit_smem_to_gmem_group()
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    # TODO: If bias/mask are broadcast along q, we can load outside the
    # pipeline as they are not dependent on q_step.
    def q_pipeline(
        index,
        q_smem,
        dout_smem,
        m_smem,
        l_smem,
        delta_smem,
        bias_smems,
        mask_smems,
        q_consumed_barrier,
        dout_consumed_barrier,
        m_consumed_barrier,
        l_consumed_barrier,
        delta_consumed_barrier,
        bias_consumed_barrier,
        mask_consumed_barrier,
        carry,
    ):
      q_step = index[0]
      q_seq_base = q_step * block_q
      kv_seq_base = kv_idx * (compute_wgs * block_kv) + wg_idx * block_kv
      k_smem, v_smem = k_smems.at[wg_idx], v_smems.at[wg_idx]
      dk_acc, dv_acc = carry

      qs = pl.ds(q_seq_base, block_q)
      ks = pl.ds(kv_seq_base, block_kv)

      def compute_sT(acc_ref):
        plgpu.wgmma(acc_ref, k_smem, plgpu.transpose_ref(q_smem, (1, 0)))
        return acc_ref[...]

      m = plgpu.load(m_smem, (), layout=L.WGMMA.reduce(0))
      l = plgpu.load(l_smem, (), layout=L.WGMMA.reduce(0))
      plgpu.barrier_arrive(m_consumed_barrier)
      plgpu.barrier_arrive(l_consumed_barrier)

      broadcast = lambda x: lax.broadcast_in_dim(x, (block_kv, block_q), [1])
      sT = pl.run_scoped(
          compute_sT, plgpu.ACC((block_kv, block_q), jnp.float32)
      )
      sT *= logits_scale

      bias = load_bias_mask(
          s=sT,
          x_ref=bias_ref,
          smems=bias_smems,
          smem_idx=pl.ds(wg_idx * block_q, block_q),
          barrier=bias_consumed_barrier,
          bcast_q_slice=(0, qs) if bcast_bias_k else None,
          bcast_k_slice=(ks, 0) if bcast_bias_q else None,
      )
      if bias is not None:
        sT += bias

      if logits_soft_cap is not None:
        logits = jnp.tanh(sT / logits_soft_cap)
        sT = logits_soft_cap * logits

      # NOTE: This rescaling must happen after bias and soft-cap but before the
      # attention masking (as the multiplication will cause `-inf`s).
      if use_base2:
        sT *= math.log2(math.e)
        m *= math.log2(math.e)

      mask_value = float(jnp.finfo(jnp.float32).min)

      def load_k_range(ref):
        qs = pl.ds(q_seq_base, block_q)
        idx = (b_idx, 0 if (ref.shape[1] == 1) else q_head, qs)
        return plgpu.load(ref, idx, layout=_WGMMA_COL, optimized=False)

      def iota(d):
        return plgpu.broadcasted_iota(jnp.int32, sT.shape, d, layout=_WGMMA)

      if k_start_ref is not None:
        k_start = load_k_range(k_start_ref)
        _start = lax.broadcast_in_dim(k_start, sT.shape, [1])
        sT = jnp.where(kv_seq_base + iota(0) >= _start, sT, mask_value)

      if k_end_ref is not None:
        k_end = load_k_range(k_end_ref)
        _end = lax.broadcast_in_dim(k_end, sT.shape, [1])
        sT = jnp.where(kv_seq_base + iota(0) < _end, sT, mask_value)

      mask = load_bias_mask(
          s=sT,
          x_ref=mask_ref,
          smems=mask_smems,
          smem_idx=pl.ds(wg_idx * block_q, block_q),
          barrier=mask_consumed_barrier,
          bcast_q_slice=(0, qs) if bcast_mask_k else None,
          bcast_k_slice=(ks, 0) if bcast_mask_q else None,
      )
      if mask is not None:
        sT = jnp.where(mask, sT, mask_value)

      pT = exp(sT - broadcast(m)) / broadcast(l)

      def _compute(refs):
        # Combining two WGMMA calls in one block to avoid the unnecessary
        # synchronization from two `wgmma.wait_group` calls.
        dv_acc_ref, dpT_acc_ref = refs
        plgpu.wgmma(dv_acc_ref, pT.astype(dtype), dout_smem)
        plgpu.wgmma(dpT_acc_ref, v_smem, plgpu.transpose_ref(dout_smem, (1, 0)))

      zeros = plgpu.layout_cast(
          jnp.full((block_kv, block_q), 0, dtype=jnp.float32),
          L.WGMMA,
      )
      dv_acc, dpT = pl.run_state(_compute)(
          (plgpu.ACC.init(dv_acc), plgpu.ACC.init(zeros))
      )
      plgpu.barrier_arrive(dout_consumed_barrier)

      delta = plgpu.load(delta_smem, (), layout=L.WGMMA.reduce(0))
      plgpu.barrier_arrive(delta_consumed_barrier)

      dsT = pT * (dpT - broadcast(delta))  # pytype: disable=wrong-arg-types  # jax-operator-types
      if logits_soft_cap is not None:
        dsT *= 1 - jnp.pow(logits, 2)
      dsT *= logits_scale

      def compute_dk(acc_ref):
        plgpu.wgmma(acc_ref, dsT.astype(dtype), q_smem)

      dk_acc = pl.run_state(compute_dk)(plgpu.ACC.init(dk_acc))
      plgpu.barrier_arrive(q_consumed_barrier)

      return (dk_acc, dv_acc)

    bias_in_spec = bias_mask_async_spec(
        bias_ref, async_bias, block_kv, block_q, kv_idx, "bias"
    )
    mask_in_spec = bias_mask_async_spec(
        mask_ref, async_mask, block_kv, block_q, kv_idx, "mask"
    )

    pipeline = plgpu.emit_pipeline_warp_specialized(
        q_pipeline,
        grid=(num_q_tiles_in_dkv,),
        max_concurrent_steps=min([config.num_stages, num_q_tiles_in_dkv]),
        num_compute_wgs=compute_wgs,
        memory_registers=40,
        wg_axis="wg",
        manual_consumed_barriers=True,
        compute_context=compute_thread,
        in_specs=[
            plgpu.BlockSpec(  # q
                block_shape=(block_q, head_dim),
                index_map=lambda i: (i, 0),
                transforms=transforms,
            ),
            plgpu.BlockSpec(  # dout
                block_shape=(block_q, head_dim_out),
                index_map=lambda i: (i, 0),
                transforms=transforms,
            ),
            plgpu.BlockSpec(block_shape=(block_q,), index_map=lambda i: (i,)),
            plgpu.BlockSpec(block_shape=(block_q,), index_map=lambda i: (i,)),
            plgpu.BlockSpec(block_shape=(block_q,), index_map=lambda i: (i,)),
            bias_in_spec,
            mask_in_spec,
        ],
    )
    q_ref = q_ref.at[b_idx, :, q_head, :]
    dout_ref = dout_ref.at[b_idx, :, q_head, :]
    m_ref = m_ref.at[b_idx, q_head, :]
    l_ref = l_ref.at[b_idx, q_head, :]
    delta_ref = delta_ref.at[b_idx, q_head, :]
    if bias_ref is not None:
      bias_ref = bias_ref.at[bias_b_idx, bias_h_idx, :, :]
    if mask_ref is not None:
      mask_ref = mask_ref.at[mask_b_idx, mask_h_idx, :, :]

    pipeline(
        q_ref,
        dout_ref,
        m_ref,
        l_ref,
        delta_ref,
        bias_ref if async_bias else None,
        mask_ref if async_mask else None,
    )

  q_scratch = plgpu.SMEM(
      (compute_wgs, config.block_q_dq, head_dim),
      q.dtype,
      transforms=transforms,
  )
  dout_scratch = plgpu.SMEM(
      (compute_wgs, config.block_q_dq, head_dim_out),
      dout.dtype,
      transforms=transforms,
  )
  m_scratch = l_scratch = delta_scratch = plgpu.SMEM(
      (compute_wgs, config.block_q_dq), jnp.float32
  )
  if bias is None:
    ds_out_shape = None
  else:
    ds_out_shape = (batch_size, num_q_heads, kv_seq_len, q_seq_len)
    ds_out_shape = jax.ShapeDtypeStruct(ds_out_shape, bias.dtype)
  # TODO: Optionally fuse the dq and dkv kernels.
  dq, ds = plgpu.kernel(
      functools.partial(
          kernel_dq, block_q=config.block_q_dq, block_kv=config.block_kv_dq
      ),
      out_shape=(q, ds_out_shape),
      scratch_shapes=[
          (q_scratch, dout_scratch, m_scratch, l_scratch, delta_scratch),  # type: ignore
          (plgpu.Barrier(num_barriers=compute_wgs),) * 5,  # type: ignore
      ],
      compiler_params=plgpu.CompilerParams(approx_math=True),
      grid=(num_q_heads, num_q_tiles, batch_size),
      grid_names=("heads", "q_tiles", "batch"),
      num_threads=compute_wgs + 1,
      thread_name="wg",
  )(q, k, v, dout, m, l, delta, bias, mask, k_start, k_end)

  k_scratch = plgpu.SMEM(
      (compute_wgs, config.block_kv_dkv, head_dim),
      k.dtype,
      transforms=transforms,
  )
  v_scratch = plgpu.SMEM(
      (compute_wgs, config.block_kv_dkv, head_dim_out),
      v.dtype,
      transforms=transforms,
  )
  # `dk` and `dv` outputs have `num_q_heads` heads (reduced below if necessary).
  dk_shape = (batch_size, kv_seq_len, num_q_heads, head_dim)
  dv_shape = (batch_size, kv_seq_len, num_q_heads, head_dim_out)

  # TODO: Fuse transpose in the kernel.
  bias_ = None if bias is None else bias.swapaxes(-1, -2)
  if mask is not None:
    mask = mask.swapaxes(-1, -2)

  dk, dv = plgpu.kernel(
      functools.partial(
          kernel_dkv, block_q=config.block_q_dkv, block_kv=config.block_kv_dkv
      ),
      out_shape=(
          jax.ShapeDtypeStruct(dk_shape, k.dtype),
          jax.ShapeDtypeStruct(dv_shape, v.dtype),
      ),
      scratch_shapes=[
          (k_scratch, v_scratch),  # type: ignore
          (plgpu.Barrier(num_barriers=compute_wgs),) * 2,  # type: ignore
      ],
      compiler_params=plgpu.CompilerParams(approx_math=True),
      grid=(num_q_heads, num_kv_tiles, batch_size),
      grid_names=("heads", "num_kv_tiles", "batch"),
      num_threads=compute_wgs + 1,
      thread_name="wg",
  )(q, k, v, dout, m, l, delta, bias_, mask, k_start, k_end)

  if q_heads_per_kv_head > 1:
    dk = dk.reshape(*k.shape[:-1], q_heads_per_kv_head, -1).sum(axis=-2)
    dv = dv.reshape(*v.shape[:-1], q_heads_per_kv_head, -1).sum(axis=-2)

  dq = dq[..., : orig_q_shape[-1]].reshape(*orig_q_shape)
  dk = dk[..., : orig_k_shape[-1]].reshape(*orig_k_shape)
  dv = dv[..., : orig_v_shape[-1]].reshape(*orig_v_shape)

  if bias is None:
    dbias = None
  else:
    broadcast_bias_axes = [i for i, d in enumerate(bias.shape) if d == 1]
    dbias = jnp.sum(ds, axis=broadcast_bias_axes).reshape(orig_bias_shape)
  return dq, dk, dv, dbias


def _decompose_mask(mask, q, k, q_indices, k_indices):
  """Decomposes `mask` into a mask array, `k_start` and `k_end`."""
  if mask is None:
    return None, None, None

  k_start = None
  k_end = None

  if k_indices is None:
    mask, is_causal, k_start, k_end = mask.take("is_causal", "k_start", "k_end")

    # TODO: Support not folding `is_causal` into `k_end`.
    # Fold `is_causal` into `k_end`.
    if is_causal:
      if q_indices is None:
        q_indices = jnp.arange(q.shape[-3])
      k_end_ = q_indices + 1
      k_end = k_end_ if k_end is None else jnp.minimum(k_end, k_end_)

    if k_start is not None:
      k_start = jax.lax.broadcast_to_rank(k_start, 2)
    if k_end is not None:
      k_end = jax.lax.broadcast_to_rank(k_end, 2)

  q_len_or_indices = q.shape[-3] if q_indices is None else q_indices
  k_len_or_indices = k.shape[-3] if k_indices is None else k_indices
  mask = mask.as_array(q_len_or_indices, k_len_or_indices)
  return mask, k_start, k_end


_SUPPORTED_PRECISIONS = (
    lax.DotAlgorithmPreset.DEFAULT,
    lax.DotAlgorithmPreset.BF16_BF16_F32,
    lax.DotAlgorithmPreset.F16_F16_F32,
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicGpuFlashAttentionVjp(
    base.DotProductAttentionVjp[Config, None]
):
  """Pallas-Triton FlashAttention VJP implementation."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  use_base2: bool = False

  @jaxtyping.jaxtyped
  def _fwd(
      self,
      residuals: Residuals,
      out: Float[Array, "*B T H d"],
      dout: Float[Array, "*B T H d"],
      q: Float[Array, "*B T H D"],
      k: Float[Array, "*B t h D"],
      v: Float[Array, "*B t h d"],
      *,
      precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset],
      logits_dtype: jnp.dtype,
      logits_scale: float,
      bias: Float[Array, "*#B #H #T #t"] | None,
      logits_soft_cap: float | None,
      mask: base.Mask,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
      dropout_rate: float,
      paging_info: PagingInfo | None,
      q_indices: Int[Array, "*#B #H T"] | None,
      k_indices: Int[Array, "*#B #H t"] | None,
      normalize_output: bool,
      return_residuals: bool,
      config: Config,
  ) -> tuple[base.DotProductAttentionGrads, None]:
    del dropout_rate

    if paging_info is not None:
      raise NotImplementedError("Paged attention not supported.")

    if not normalize_output:
      raise NotImplementedError("`normalize_output=False` not supported.")

    if logits_dtype != jnp.float32:
      raise NotImplementedError("`logits_dtype` must be float32.")

    if dropout_mask is not None:
      raise NotImplementedError("dropout is not supported.")

    if return_residuals:
      raise NotImplementedError("`return_residuals` not supported.")

    mask, k_start, k_end = _decompose_mask(mask, q, k, q_indices, k_indices)

    q_k_dot_precision, weights_v_dot_precision = precision
    if q_k_dot_precision not in _SUPPORTED_PRECISIONS:
      raise NotImplementedError(f"{q_k_dot_precision=} not supported")
    if weights_v_dot_precision not in _SUPPORTED_PRECISIONS:
      raise NotImplementedError(f" {weights_v_dot_precision=} not supported")

    f = functools.partial(
        _bwd,
        bias=bias,
        mask=mask,
        k_start=k_start,
        k_end=k_end,
        logits_scale=logits_scale,
        logits_soft_cap=logits_soft_cap,
        use_base2=self.use_base2,
        config=config,
    )

    args = (q, k, v, residuals, out, dout)

    dq, dk, dv, dbias = f(*args)
    return base.DotProductAttentionGrads(q=dq, k=dk, v=dv, bias=dbias), None

  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config(
        block_q_dkv=64,
        block_kv_dkv=64,
        block_q_dq=64,
        block_kv_dq=64,
        num_stages=2,
    )
