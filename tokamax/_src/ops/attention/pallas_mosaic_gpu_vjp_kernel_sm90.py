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
"""Flash Attention Pallas-Mosaic-GPU VJP implementation."""

# pylint: disable=invalid-name

import functools
import math

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src import shape as shape_lib
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_mosaic_gpu_vjp_common as vjp_common

Config = vjp_common.Config
Residuals = base.Residuals

_WGMMA = plgpu.Layout.WGMMA
_WGMMA_COL = plgpu.Layout.WGMMA.reduce(0)
_WGMMA_ROW = plgpu.Layout.WGMMA.reduce(1)


@jaxtyping.jaxtyped
def flash_attention_vjp_kernel(
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
    dbias_intermediate_dtype: jax.typing.DTypeLike | None,
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
  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by and {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  compute_wgs = config.compute_wgs
  num_q_tiles = pl.cdiv(q_seq_len, config.block_q_dq * compute_wgs)
  num_kv_tiles = pl.cdiv(kv_seq_len, config.block_kv_dkv * compute_wgs)
  num_q_tiles_in_dkv = pl.cdiv(q_seq_len, config.block_q_dkv)
  num_kv_tiles_in_dq = pl.cdiv(kv_seq_len, config.block_kv_dq)

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
  transforms = lambda dt: (
      plgpu.TilingTransform((8, swizzle // dt.itemsize)),
      plgpu.SwizzleTransform(swizzle),
  )
  delta = jnp.einsum(
      "bqhd,bqhd->bhq", out, dout, preferred_element_type=jnp.float32
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
      dq_gmem,
      ds_gmem,
      smem_buffers,
      buffer_barriers,
      block_q: int,
      block_kv: int,
  ):
    bi = lax.axis_index("batch")
    qi = lax.axis_index("q_tiles")
    hi = lax.axis_index("heads")
    wg = lax.axis_index("wg")
    hi_kv = lax.div(hi, jnp.array(q_heads_per_kv_head, hi.dtype))

    q_base = qi * (compute_wgs * block_q) + wg * block_q
    qs = pl.ds(q_base, block_q)

    at_wg = lambda x: x.at[wg]
    q_smem, dout_smem, m_smem, l_smem, delta_smem = map(at_wg, smem_buffers)
    q_barrier, dout_barrier, m_barrier, l_barrier, delta_barrier = map(
        at_wg, buffer_barriers
    )

    bias_b_idx, bias_h_idx, bcast_bias_q, bcast_bias_k, async_bias = (
        bias_mask_info(bias_gmem, bi, hi, "bias")
    )
    mask_b_idx, mask_h_idx, bcast_mask_q, bcast_mask_k, async_mask = (
        bias_mask_info(mask_gmem, bi, hi, "mask")
    )

    def compute_thread(pipeline_callback):
      plgpu.copy_gmem_to_smem(q_gmem.at[bi, qs, hi], q_smem, q_barrier)
      plgpu.copy_gmem_to_smem(dout_gmem.at[bi, qs, hi], dout_smem, dout_barrier)
      plgpu.copy_gmem_to_smem(
          delta_gmem.at[bi, hi, qs], delta_smem, delta_barrier
      )
      plgpu.copy_gmem_to_smem(m_gmem.at[bi, hi, qs], m_smem, m_barrier)
      plgpu.copy_gmem_to_smem(l_gmem.at[bi, hi, qs], l_smem, l_barrier)
      _ = [plgpu.barrier_wait(buffer.at[wg]) for buffer in buffer_barriers]

      delta = plgpu.load(delta_smem, (), layout=_WGMMA_ROW)
      m = plgpu.load(m_smem, (), layout=_WGMMA_ROW)
      l = plgpu.load(l_smem, (), layout=_WGMMA_ROW)
      if use_base2:
        m *= math.log2(math.e)
      acc = plgpu.layout_cast(jnp.zeros(q_smem.shape, jnp.float32), _WGMMA)

      def load_k_range(ref):
        if ref is None:
          return None
        idx = (bi, 0 if ref.shape[1] == 1 else hi, qs)
        return plgpu.load(ref, idx, layout=_WGMMA_ROW, optimized=False)

      k_start = load_k_range(k_start_gmem)
      k_end = load_k_range(k_end_gmem)
      dq, _, _, _, _, _ = pipeline_callback((acc, m, l, delta, k_start, k_end))
      q_smem[...] = dq.astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(q_smem, dq_gmem.at[bi, qs, hi])
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
      ki = index[0]
      kv_base = ki * block_kv
      ks = pl.ds(kv_base, block_kv)
      acc, m, l, delta, k_start, k_end = carry

      def compute_s(acc_ref):
        plgpu.wgmma(acc_ref, q_smem, k_smem.T)
        return acc_ref[...]

      s = pl.run_scoped(compute_s, plgpu.ACC((block_q, block_kv), jnp.float32))
      s *= logits_scale

      bias = load_bias_mask(
          s=s,
          x_ref=bias_gmem,
          smems=bias_smems,
          smem_idx=pl.ds(wg * block_q, block_q),
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
        k_start_ = lax.broadcast_in_dim(k_start, s.shape, [0])
        s = jnp.where(kv_base + iota(1) >= k_start_, s, mask_value)

      if k_end is not None:
        k_end_ = lax.broadcast_in_dim(k_end, s.shape, [0])
        s = jnp.where(kv_base + iota(1) < k_end_, s, mask_value)

      mask = load_bias_mask(
          s=s,
          x_ref=mask_gmem,
          smems=mask_smems,
          smem_idx=pl.ds(wg * block_q, block_q),
          barrier=mask_consumed_barrier,
          bcast_q_slice=(0, ks) if bcast_mask_q else None,
          bcast_k_slice=(qs, 0) if bcast_mask_k else None,
      )
      if mask is not None:
        s = jnp.where(mask, s, mask_value)

      broadcast = lambda x: lax.broadcast_in_dim(x, s.shape, [0])
      epsilon = jnp.finfo(jnp.float32).tiny  # Avoid division by zero.
      p = exp(s - broadcast(m)) / broadcast(l + epsilon)

      def compute_dp(acc_ref):
        plgpu.wgmma(acc_ref, dout_smem, v_smem.T)
        return acc_ref[...]

      dp = pl.run_scoped(
          compute_dp, plgpu.ACC((block_q, block_kv), jnp.float32)
      )
      plgpu.barrier_arrive(v_consumed_barrier)

      ds = p * (dp - lax.broadcast_in_dim(delta, p.shape, [0]))
      if logits_soft_cap is not None:
        ds *= 1 - logits * logits

      # If we have an attention mask, it is possible that the entire row is
      # masked out. In that case, the forwards pass will calculate `p`'s values
      # as `1 / seq_len_k`. The corresponding `ds` values must be zeroed.
      if mask is not None:
        ds = jnp.where(mask, ds, 0.0)

      if ds_gmem is not None:
        # TODO: Make this store non-blocking.
        ds_gmem[bi, hi, qs, ks] = ds.astype(ds_gmem.dtype)

      ds *= logits_scale

      def compute_dq(acc_ref):
        plgpu.wgmma(acc_ref, ds.astype(k_smem.dtype), k_smem)

      acc = pl.run_state(compute_dq)(plgpu.ACC.init(acc))
      plgpu.barrier_arrive(k_consumed_barrier)

      return (acc, m, l, delta, k_start, k_end)

    bias_in_spec = bias_mask_async_spec(
        bias_gmem, async_bias, block_q, block_kv, qi, "bias"
    )
    mask_in_spec = bias_mask_async_spec(
        mask_gmem, async_mask, block_q, block_kv, qi, "mask"
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
                transforms=transforms(k.dtype),
            ),
            plgpu.BlockSpec(  # v
                block_shape=(block_kv, head_dim_out),
                index_map=lambda i: (i, 0),
                transforms=transforms(v.dtype),
            ),
            bias_in_spec,
            mask_in_spec,
        ],
    )
    k_gmem = k_gmem.at[bi, :, hi_kv, :]
    v_gmem = v_gmem.at[bi, :, hi_kv, :]
    if bias_gmem is not None:
      bias_gmem = bias_gmem.at[bias_b_idx, bias_h_idx, :, :]
    if mask_gmem is not None:
      mask_gmem = mask_gmem.at[mask_b_idx, mask_h_idx, :, :]

    pipeline(
        k_gmem,
        v_gmem,
        bias_gmem if async_bias else None,
        mask_gmem if async_mask else None,
    )

  def kernel_dkv(
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
      dk_gmem,
      dv_gmem,
      smem_buffers,
      buffer_barriers,
      block_q: int,
      block_kv: int,
  ):
    bi = lax.axis_index("batch")
    ki = lax.axis_index("kv_tiles")
    hi = lax.axis_index("heads")
    wg = lax.axis_index("wg")

    kv_base = ki * (compute_wgs * block_kv) + wg * block_kv
    ks = pl.ds(kv_base, block_kv)
    hi_kv = lax.div(hi, jnp.array(q_heads_per_kv_head, hi.dtype))

    k_smem, v_smem = map(lambda x: x.at[wg], smem_buffers)
    k_barrier, v_barrier = map(lambda x: x.at[wg], buffer_barriers)

    bias_b_idx, bias_h_idx, bcast_bias_k, bcast_bias_q, async_bias = (
        bias_mask_info(bias_gmem, bi, hi, "bias")
    )
    mask_b_idx, mask_h_idx, bcast_mask_k, bcast_mask_q, async_mask = (
        bias_mask_info(mask_gmem, bi, hi, "mask")
    )
    if bias_gmem is not None:
      bias_gmem = bias_gmem.at[bias_b_idx, bias_h_idx, :, :]
    if mask_gmem is not None:
      mask_gmem = mask_gmem.at[mask_b_idx, mask_h_idx, :, :]

    def compute_thread(pipeline_callback):
      plgpu.copy_gmem_to_smem(k_gmem.at[bi, ks, hi_kv], k_smem, k_barrier)
      plgpu.copy_gmem_to_smem(v_gmem.at[bi, ks, hi_kv], v_smem, v_barrier)
      plgpu.barrier_wait(k_barrier)
      plgpu.barrier_wait(v_barrier)
      dk_acc = plgpu.layout_cast(jnp.zeros(k_smem.shape, jnp.float32), _WGMMA)
      dv_acc = plgpu.layout_cast(jnp.zeros(v_smem.shape, jnp.float32), _WGMMA)
      dk, dv = pipeline_callback((dk_acc, dv_acc))
      k_smem[...] = dk.astype(k.dtype)
      v_smem[...] = dv.astype(v.dtype)

      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(
          k_smem, dk_gmem.at[bi, ks, hi], commit_group=False
      )
      plgpu.copy_smem_to_gmem(
          v_smem, dv_gmem.at[bi, ks, hi], commit_group=False
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
      qi = index[0]
      q_base = qi * block_q
      qs = pl.ds(q_base, block_q)
      dk_acc, dv_acc = carry

      def compute_sT(acc_ref):
        plgpu.wgmma(acc_ref, k_smem, q_smem.T)
        return acc_ref[...]

      m = plgpu.load(m_smem, (), layout=_WGMMA_COL)
      l = plgpu.load(l_smem, (), layout=_WGMMA_COL)
      plgpu.barrier_arrive(m_consumed_barrier)
      plgpu.barrier_arrive(l_consumed_barrier)

      broadcast = lambda x: lax.broadcast_in_dim(x, (block_kv, block_q), [1])
      sT = pl.run_scoped(
          compute_sT, plgpu.ACC((block_kv, block_q), jnp.float32)
      )
      sT *= logits_scale

      bias = load_bias_mask(
          s=sT,
          x_ref=bias_gmem,
          smems=bias_smems,
          smem_idx=pl.ds(wg * block_q, block_q),
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
        idx = (bi, 0 if (ref.shape[1] == 1) else hi, qs)
        return plgpu.load(ref, idx, layout=_WGMMA_COL, optimized=False)

      def iota(d):
        return plgpu.broadcasted_iota(jnp.int32, sT.shape, d, layout=_WGMMA)

      if k_start_gmem is not None:
        k_start = load_k_range(k_start_gmem)
        k_start = lax.broadcast_in_dim(k_start, sT.shape, [1])
        sT = jnp.where(kv_base + iota(0) >= k_start, sT, mask_value)

      if k_end_gmem is not None:
        k_end = load_k_range(k_end_gmem)
        k_end = lax.broadcast_in_dim(k_end, sT.shape, [1])
        sT = jnp.where(kv_base + iota(0) < k_end, sT, mask_value)

      mask = load_bias_mask(
          s=sT,
          x_ref=mask_gmem,
          smems=mask_smems,
          smem_idx=pl.ds(wg * block_q, block_q),
          barrier=mask_consumed_barrier,
          bcast_q_slice=(0, qs) if bcast_mask_k else None,
          bcast_k_slice=(ks, 0) if bcast_mask_q else None,
      )
      if mask is not None:
        sT = jnp.where(mask, sT, mask_value)

      epsilon = float(jnp.finfo(jnp.float32).tiny)  # Avoid division by zero.
      pT = exp(sT - broadcast(m)) / broadcast(l + epsilon)

      def _compute(refs):
        # Combining two WGMMA calls in one block to avoid the unnecessary
        # synchronization from two `wgmma.wait_group` calls.
        dv_acc_ref, dpT_acc_ref = refs
        plgpu.wgmma(dv_acc_ref, pT.astype(dtype), dout_smem)
        plgpu.wgmma(dpT_acc_ref, v_smem, dout_smem.T)

      zeros = plgpu.layout_cast(
          jnp.full((block_kv, block_q), 0, dtype=jnp.float32), _WGMMA
      )
      dv_acc, dpT = pl.run_state(_compute)(
          (plgpu.ACC.init(dv_acc), plgpu.ACC.init(zeros))
      )
      plgpu.barrier_arrive(dout_consumed_barrier)

      delta = plgpu.load(delta_smem, (), layout=_WGMMA_COL)
      plgpu.barrier_arrive(delta_consumed_barrier)

      dsT = pT * (dpT - broadcast(delta))  # pytype: disable=wrong-arg-types  # jax-operator-types
      if logits_soft_cap is not None:
        dsT *= 1 - logits * logits
      dsT *= logits_scale

      def compute_dk(acc_ref):
        plgpu.wgmma(acc_ref, dsT.astype(dtype), q_smem)

      dk_acc = pl.run_state(compute_dk)(plgpu.ACC.init(dk_acc))
      plgpu.barrier_arrive(q_consumed_barrier)

      return (dk_acc, dv_acc)

    bias_in_spec = bias_mask_async_spec(
        bias_gmem, async_bias, block_kv, block_q, ki, "bias"
    )
    mask_in_spec = bias_mask_async_spec(
        mask_gmem, async_mask, block_kv, block_q, ki, "mask"
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
                transforms=transforms(q.dtype),
            ),
            plgpu.BlockSpec(  # dout
                block_shape=(block_q, head_dim_out),
                index_map=lambda i: (i, 0),
                transforms=transforms(dout.dtype),
            ),
            plgpu.BlockSpec(block_shape=(block_q,), index_map=lambda i: (i,)),
            plgpu.BlockSpec(block_shape=(block_q,), index_map=lambda i: (i,)),
            plgpu.BlockSpec(block_shape=(block_q,), index_map=lambda i: (i,)),
            bias_in_spec,
            mask_in_spec,
        ],
    )

    pipeline(
        q_gmem.at[bi, :, hi, :],
        dout_gmem.at[bi, :, hi, :],
        m_gmem.at[bi, hi, :],
        l_gmem.at[bi, hi, :],
        delta_gmem.at[bi, hi, :],
        bias_gmem if async_bias else None,
        mask_gmem if async_mask else None,
    )

  q_scratch = plgpu.SMEM(
      (compute_wgs, config.block_q_dq, head_dim),
      q.dtype,
      transforms=transforms(q.dtype),
  )
  dout_scratch = plgpu.SMEM(
      (compute_wgs, config.block_q_dq, head_dim_out),
      dout.dtype,
      transforms=transforms(dout.dtype),
  )
  m_scratch = l_scratch = delta_scratch = plgpu.SMEM(
      (compute_wgs, config.block_q_dq), jnp.float32
  )
  if bias is None:
    ds_out_shape = None
  else:
    # NOTE: TMA stores to GMEM do not mask out-of-bounds writes, so we must pad
    # the output to a multiple of the block size.
    q_seq_len_ = num_q_tiles * compute_wgs * config.block_q_dq
    kv_seq_len_ = num_kv_tiles_in_dq * config.block_kv_dq
    ds_out_shape = (batch_size, num_q_heads, q_seq_len_, kv_seq_len_)
    if dbias_intermediate_dtype is None or (ds_out_shape == bias.shape):
      dbias_intermediate_dtype = bias.dtype
    ds_out_shape = jax.ShapeDtypeStruct(ds_out_shape, dbias_intermediate_dtype)
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
      grid=(batch_size, num_q_heads, num_q_tiles),
      grid_names=("batch", "heads", "q_tiles"),
      num_threads=compute_wgs + 1,
      thread_name="wg",
  )(q, k, v, dout, m, l, delta, bias, mask, k_start, k_end)

  k_scratch = plgpu.SMEM(
      (compute_wgs, config.block_kv_dkv, head_dim),
      k.dtype,
      transforms=transforms(k.dtype),
  )
  v_scratch = plgpu.SMEM(
      (compute_wgs, config.block_kv_dkv, head_dim_out),
      v.dtype,
      transforms=transforms(v.dtype),
  )
  # `dk` and `dv` outputs have `num_q_heads` heads (reduced below if necessary).
  dk_shape = (batch_size, kv_seq_len, num_q_heads, head_dim)
  dv_shape = (batch_size, kv_seq_len, num_q_heads, head_dim_out)

  # TODO: Fuse transpose in the kernel.
  bias_ = None if bias is None else bias.mT
  if mask is not None:
    mask = mask.mT

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
      grid=(batch_size, num_q_heads, num_kv_tiles),
      grid_names=("batch", "heads", "kv_tiles"),
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
    ds = ds[..., :q_seq_len, :kv_seq_len]
    broadcast_bias_axes = [i for i, d in enumerate(bias.shape) if d == 1]
    dbias = jnp.sum(ds, axis=broadcast_bias_axes)
    dbias = dbias.astype(bias.dtype).reshape(orig_bias_shape)
  return dq, dk, dv, dbias
