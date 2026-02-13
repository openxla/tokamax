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
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_mosaic_gpu_common as common
from tokamax._src.ops.attention import pallas_mosaic_gpu_vjp_common as vjp_common

Config = vjp_common.Config
Residuals = base.Residuals

_WGMMA = plgpu.Layout.WGMMA
_WGMMA_COL = plgpu.Layout.WGMMA.reduce(0)
_WGMMA_ROW = plgpu.Layout.WGMMA.reduce(1)
_WGMMA_TRANSPOSED = plgpu.Layout.WGMMA_TRANSPOSED
_SMEM_SIZE_BYTES = 227 * 1024
_F32_BYTES = 4
_load_bcast = common.load_bcast


def _estimate_dq_smem_bytes(ba, block_q, block_kv, num_stages):
  """Estimates the dq kernel smem usage in bytes for a given configuration."""
  _, _, _, q, k, v = ba.args
  tile_q = 2 * block_q

  # 32-bit floats are downcast to 16-bit before the kernel call.
  dtype_bits = jnp.finfo(jnp.bfloat16).bits
  block_d = pl.cdiv(q.shape[-1], 64) * 64
  block_d_out = v.shape[-1]
  bytes_per_stage = block_kv * (block_d + block_d_out) * dtype_bits // 8  # k,v
  if (bias := ba.kwargs["bias"]) is not None:
    if bias.shape[-2] != 1 and bias.shape[-1] != 1:
      bytes_per_stage += tile_q * block_kv * jnp.finfo(bias.dtype).bits // 8

  mask = ba.kwargs["mask"]
  q_indices = ba.kwargs["q_indices"]
  k_indices = ba.kwargs["k_indices"]
  mask, *_ = jax.eval_shape(
      common.decompose_mask, mask, q, k, q_indices, k_indices
  )
  if mask is not None and mask.shape[-2] != 1 and mask.shape[-1] != 1:
    bytes_per_stage += tile_q * block_kv

  return (
      (tile_q * (block_d + block_d_out) * dtype_bits // 8)  # q, dout
      + 3 * tile_q * _F32_BYTES  # m, l, delta
      + num_stages * bytes_per_stage
  )


def _estimate_dkv_smem_bytes(ba, block_q, block_kv, num_stages):
  """Estimates the dkv kernel smem usage in bytes for a given configuration."""
  _, _, _, q, k, v = ba.args
  tile_kv = 2 * block_kv

  # 32-bit floats are downcast to 16-bit before the kernel call.
  dtype_bits = jnp.finfo(jnp.bfloat16).bits
  block_d = pl.cdiv(q.shape[-1], 64) * 64
  block_d_out = v.shape[-1]
  bytes_per_stage = block_q * (block_d + block_d_out) * dtype_bits // 8  # q, do
  bytes_per_stage += 3 * block_q * _F32_BYTES  # m, l, delta
  if (bias := ba.kwargs["bias"]) is not None:
    if bias.shape[-2] != 1 and bias.shape[-1] != 1:
      bytes_per_stage += block_q * tile_kv * jnp.finfo(bias.dtype).bits // 8

  mask = ba.kwargs["mask"]
  q_indices = ba.kwargs["q_indices"]
  k_indices = ba.kwargs["k_indices"]
  mask, *_ = jax.eval_shape(
      common.decompose_mask, mask, q, k, q_indices, k_indices
  )
  if mask is not None and mask.shape[-2] != 1 and mask.shape[-1] != 1:
    bytes_per_stage += block_q * tile_kv

  return (
      tile_kv * (block_d + block_d_out) * dtype_bits // 8
  ) + num_stages * bytes_per_stage  # k, v


def get_heuristics_config(ba: op.BoundArguments) -> Config:
  num_stages = 2
  if _estimate_dq_smem_bytes(ba, 64, 64, 2) > _SMEM_SIZE_BYTES:
    num_stages = 1
  if _estimate_dkv_smem_bytes(ba, 64, 64, 2) > _SMEM_SIZE_BYTES:
    num_stages = 1

  return Config(
      block_q_dkv=64,
      block_kv_dkv=64,
      block_q_dq=64,
      block_kv_dq=64,
      num_stages=num_stages,
  )


def get_autotuning_configs(ba: op.BoundArguments) -> set[Config]:
  # TODO: Expand vjp search space.
  del ba
  configs = set()
  for block_q in [64, 128]:
    for block_kv in [64, 128]:
      for num_stages in [2, 3]:
        for compute_wgs in [1, 2]:
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
    *,
    logits_scale: float,
    logits_soft_cap: float | None,
    is_causal: bool,
    use_base2: bool,
    ds_dtype: jax.typing.DTypeLike | None,
    config: Config,
) -> tuple[
    Float[Array, "T H D"],  # dq
    Float[Array, "t h D"],  # dk
    Float[Array, "t h d"],  # dv
    Float[Array, "H T t"] | None,  # ds
]:
  orig_head_dim = q.shape[-1]
  orig_head_dim_out = v.shape[-1]
  pad_head_dim = lambda x: shape_lib.pad_to_next_multiple_of(x, 64, -1)
  q, k, v, out, dout = map(pad_head_dim, (q, k, v, out, dout))
  m, l = residuals

  q_seq_len, num_q_heads, head_dim = q.shape
  kv_seq_len, num_kv_heads, head_dim_out = v.shape
  if (dtype := q.dtype) != k.dtype or dtype != v.dtype:
    raise ValueError(
        f"q, k, and v should all have the same dtype, got: {q.dtype},"
        f" {k.dtype}, {v.dtype}"
    )
  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by and {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  block_q_dq = config.block_q_dq
  block_q_dkv = config.block_q_dkv
  block_kv_dkv = config.block_kv_dkv
  block_kv_dq = config.block_kv_dq
  compute_wgs = config.compute_wgs

  if mask is not None:
    mask = mask.astype(jnp.int8)

  # TODO: Avoid broadcast.
  bcast = lambda x: jnp.broadcast_to(x, (x.shape[-2], q_seq_len))
  k_start = None if k_start is None else bcast(k_start)
  k_end = None if k_end is None else bcast(k_end)
  delta = jnp.einsum(
      "qhd,qhd->hq", out, dout, preferred_element_type=jnp.float32
  )
  exp = jnp.exp2 if use_base2 else jnp.exp

  def tiled_spec(block_shape, dtype, index_map, what=""):
    transforms = common.tile_swizzle_transforms(block_shape, dtype, what)
    return plgpu.BlockSpec(block_shape, index_map, transforms=transforms)

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
      q_smems,
      dout_smems,
      barriers,
      block_q: int,
      block_kv: int,
  ):
    qi = lax.axis_index("q_tiles")
    hi = lax.axis_index("heads")
    wg = lax.axis_index("wg")
    hi_kv = lax.div(hi, jnp.array(q_heads_per_kv_head, hi.dtype))

    tile_q = compute_wgs * block_q
    q_base = qi * tile_q + wg * block_q
    qs = pl.ds(q_base, block_q)

    lb = 0
    ub = num_kv_tiles = pl.cdiv(kv_seq_len, block_kv_dq)
    if is_causal:
      ub = lax.min(ub, pl.cdiv(qi * tile_q + tile_q, block_kv))

    k_gmem = k_gmem.at[:, hi_kv]
    v_gmem = v_gmem.at[:, hi_kv]
    if bias_gmem is not None:
      bias_gmem = bias_gmem.at[0 if bias_gmem.shape[0] == 1 else hi]
    if mask_gmem is not None:
      mask_gmem = mask_gmem.at[0 if mask_gmem.shape[0] == 1 else hi]

    q_smem = q_smems.at[wg]
    dout_smem = dout_smems.at[wg]
    barrier = barriers.at[wg]

    def compute_thread(pipeline_callback):
      plgpu.copy_gmem_to_smem(q_gmem.at[qs, hi], q_smem, barrier)
      plgpu.copy_gmem_to_smem(dout_gmem.at[qs, hi], dout_smem, barrier)

      delta = plgpu.load(
          delta_gmem, (hi, qs), layout=_WGMMA_ROW, optimized=False
      )
      m = plgpu.load(m_gmem, (hi, qs), layout=_WGMMA_ROW, optimized=False)
      l = plgpu.load(l_gmem, (hi, qs), layout=_WGMMA_ROW, optimized=False)

      def load_k_range(ref):
        if ref is None:
          return None
        idx = (0 if ref.shape[0] == 1 else hi, qs)
        return plgpu.load(ref, idx, layout=_WGMMA_ROW, optimized=False)

      k_start = load_k_range(k_start_gmem)
      k_end = load_k_range(k_end_gmem)

      if use_base2:
        m *= math.log2(math.e)

      plgpu.barrier_wait(barrier)

      def compute_dq(dq_acc):
        _ = pipeline_callback((dq_acc, m, l, delta, k_start, k_end))
        return dq_acc[...] * logits_scale

      dq = pl.run_scoped(compute_dq, plgpu.ACC(q_smem.shape, jnp.float32))
      q_smem[...] = dq.astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(q_smem, dq_gmem.at[qs, hi])

      if ds_gmem is not None:  # Zero `ds` for the kv tiles that are not used.

        @pl.loop(ub, num_kv_tiles)
        def zero_ds(ki):
          ks = pl.ds(ki * block_kv, block_kv)
          ds_gmem_ = ds_gmem.at[hi, qs, ks]
          ds_gmem_[...] = jnp.zeros_like(ds_gmem_)  # TODO

      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    # TODO: If bias/mask are broadcast along k, we can load outside the
    # pipeline as they are not dependent on kv_step.
    def kv_pipeline(
        index,
        bias_smem,
        mask_smem,
        v_smem,
        k_smem,
        bias_consumed_barrier,
        mask_consumed_barrier,
        v_consumed_barrier,
        k_consumed_barrier,
        carry,
    ):
      (i,) = index
      ki = lb + i
      kv_base = ki * block_kv
      ks = pl.ds(kv_base, block_kv)
      dq_acc, m, l, delta, k_start, k_end = carry

      def compute_s(acc):
        plgpu.wgmma(acc, q_smem, k_smem.T)
        if bias_gmem is None:
          bias = None
        elif bias_smem is None:
          bias = _load_bcast(bias_gmem, (qs, ks), layout=_WGMMA)
        else:
          bias = bias_smem[pl.ds(wg * block_q, block_q)]
        return acc[...], bias

      acc_type = plgpu.ACC((block_q, block_kv), jnp.float32)
      s, bias = pl.run_scoped(compute_s, acc_type)
      scale = logits_scale

      if bias is not None:
        s = s * scale + bias.astype(s.dtype)
        if bias_smem is not None:
          plgpu.barrier_arrive(bias_consumed_barrier)
        scale = 1.0

      if logits_soft_cap is not None:
        s = logits = jnp.tanh(s * (scale / logits_soft_cap))
        scale = logits_soft_cap

      # NOTE: This rescaling must happen after bias and soft-cap but before the
      # attention masking (as the multiplication will cause `-inf`s).
      if use_base2:
        scale *= math.log2(math.e)

      s *= scale

      mask_value = float(jnp.finfo(jnp.float32).min)

      def iota(d):
        return plgpu.broadcasted_iota(jnp.int32, s.shape, d, layout=_WGMMA)

      if is_causal:

        def apply_causal_mask():
          return jnp.where(q_base + iota(0) >= kv_base + iota(1), s, mask_value)

        s = lax.cond(kv_base + block_kv > q_base, apply_causal_mask, lambda: s)

      if k_start is not None:
        k_start_ = lax.broadcast_in_dim(k_start, s.shape, [0])
        s = jnp.where(kv_base + iota(1) >= k_start_, s, mask_value)

      if k_end is not None:
        k_end_ = lax.broadcast_in_dim(k_end, s.shape, [0])
        s = jnp.where(kv_base + iota(1) < k_end_, s, mask_value)

      if mask_gmem is not None:
        if mask_smem is None:
          mask = _load_bcast(mask_gmem, (qs, ks), layout=_WGMMA)
        elif mask_smem.ndim == 1:
          mask = plgpu.load(mask_smem, (), layout=_WGMMA_COL)
          mask = lax.broadcast_in_dim(mask, s.shape, [1])
          plgpu.barrier_arrive(mask_consumed_barrier)
        else:
          mask = mask_smem[pl.ds(wg * block_q, block_q)]
          plgpu.barrier_arrive(mask_consumed_barrier)

        s = jnp.where(mask, s, mask_value)

      broadcast = lambda x: lax.broadcast_in_dim(x, s.shape, [0])
      epsilon = jnp.finfo(jnp.float32).tiny  # Avoid division by zero.
      p = exp(s - broadcast(m)) / broadcast(l + epsilon)

      def compute_dp(acc):
        plgpu.wgmma(acc, dout_smem, v_smem.T)
        return acc[...]

      dp = pl.run_scoped(compute_dp, plgpu.ACC(p.shape, jnp.float32))
      plgpu.barrier_arrive(v_consumed_barrier)

      ds = p * (dp - lax.broadcast_in_dim(delta, p.shape, [0]))
      if logits_soft_cap is not None:
        ds *= 1 - logits * logits

      # If we have an attention mask, it is possible that the entire row is
      # masked out. In that case, the forwards pass will calculate `p`'s values
      # as `1 / seq_len_k`. The corresponding `ds` values must be zeroed.
      if mask_gmem is not None:
        ds = jnp.where(mask, ds, 0.0)

      if ds_gmem is not None:
        # TODO: Make this store non-blocking.
        ds_gmem[hi, qs, ks] = ds.astype(ds_gmem.dtype)

      plgpu.wgmma(dq_acc, ds.astype(k_smem.dtype), k_smem)
      plgpu.wgmma_wait(0)
      plgpu.barrier_arrive(k_consumed_barrier)
      return dq_acc, m, l, delta, k_start, k_end

    k_spec = tiled_spec(
        (block_kv, head_dim), k.dtype, lambda i: (lb + i, 0), "k"
    )
    v_spec = tiled_spec(
        (block_kv, head_dim_out), v.dtype, lambda i: (lb + i, 0), "v"
    )

    if bias is not None and bias.shape[-2] != 1 and bias.shape[-1] != 1:
      bias_gmem_ = bias_gmem
      bias_spec = tiled_spec(
          (tile_q, block_kv), bias.dtype, lambda i: (qi, lb + i), "bias"
      )
    else:
      bias_gmem_ = bias_spec = None

    mask_gmem_ = mask_spec = None
    if mask is not None and mask.shape[-1] != 1:
      if mask.shape[-2] == 1:
        if block_kv >= 128:  # Minimum transfer size is 128 bytes.
          mask_gmem_ = mask_gmem
          mask_spec = plgpu.BlockSpec((None, block_kv), lambda i: (0, lb + i))
      else:
        mask_gmem_ = mask_gmem
        mask_spec = tiled_spec(
            (tile_q, block_kv), mask.dtype, lambda i: (qi, lb + i), "mask"
        )

    plgpu.emit_pipeline_warp_specialized(
        kv_pipeline,
        grid=(ub - lb,),
        max_concurrent_steps=min(config.num_stages, num_kv_tiles),
        num_compute_wgs=compute_wgs,
        memory_registers=40,
        wg_axis="wg",
        manual_consumed_barriers=True,
        compute_context=compute_thread,
        in_specs=[bias_spec, mask_spec, v_spec, k_spec],
    )(bias_gmem_, mask_gmem_, v_gmem, k_gmem)

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
      k_smems,
      v_smems,
      barriers,
      block_q: int,
      block_kv: int,
  ):
    ki = lax.axis_index("kv_tiles")
    hi = lax.axis_index("heads")
    wg = lax.axis_index("wg")

    tile_kv = compute_wgs * block_kv
    kv_base = ki * tile_kv + wg * block_kv
    ks = pl.ds(kv_base, block_kv)
    hi_kv = lax.div(hi, jnp.array(q_heads_per_kv_head, hi.dtype))

    lb = lax.div(ki * tile_kv, block_q) if is_causal else 0
    ub = num_q_tiles = pl.cdiv(q_seq_len, block_q)

    q_gmem = q_gmem.at[:, hi]
    dout_gmem = dout_gmem.at[:, hi]
    m_gmem = m_gmem.at[hi]
    l_gmem = l_gmem.at[hi]
    delta_gmem = delta_gmem.at[hi]

    if bias_gmem is not None:
      bias_gmem = bias_gmem.at[0 if bias_gmem.shape[0] == 1 else hi]
    if mask_gmem is not None:
      mask_gmem = mask_gmem.at[0 if mask_gmem.shape[0] == 1 else hi]

    k_smem = k_smems.at[wg]
    v_smem = v_smems.at[wg]
    barrier = barriers.at[wg]

    def compute_thread(pipeline_callback):
      plgpu.copy_gmem_to_smem(k_gmem.at[ks, hi_kv], k_smem, barrier)
      plgpu.copy_gmem_to_smem(v_gmem.at[ks, hi_kv], v_smem, barrier)
      if mask_gmem is None or mask_gmem.shape[-1] != 1:
        mask = None
      else:  # The mask is loop invariant.
        idx = (ks, 0)
        mask = plgpu.load(mask_gmem, idx, layout=_WGMMA_ROW, optimized=False)
      plgpu.barrier_wait(barrier)

      def compute_dk_dv(dk_acc, dv_acc):
        _ = pipeline_callback((dk_acc, dv_acc, mask))
        return dk_acc[...], dv_acc[...]

      dk_acc_type = plgpu.ACC(k_smem.shape, jnp.float32)
      dv_acc_type = plgpu.ACC(v_smem.shape, jnp.float32)
      dk, dv = pl.run_scoped(compute_dk_dv, dk_acc_type, dv_acc_type)
      k_smem[...] = (dk * logits_scale).astype(k.dtype)
      v_smem[...] = dv.astype(v.dtype)

      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(k_smem, dk_gmem.at[ks, hi], commit_group=False)
      plgpu.copy_smem_to_gmem(v_smem, dv_gmem.at[ks, hi], commit_group=False)
      plgpu.commit_smem_to_gmem_group()
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    # TODO: If bias/mask are broadcast along q, we can load outside the
    # pipeline as they are not dependent on q_step.
    def q_pipeline(
        index,
        bias_smem,
        m_smem,
        l_smem,
        mask_smem,
        dout_smem,
        delta_smem,
        q_smem,
        bias_consumed_barrier,
        m_consumed_barrier,
        l_consumed_barrier,
        mask_consumed_barrier,
        dout_consumed_barrier,
        delta_consumed_barrier,
        q_consumed_barrier,
        carry,
    ):
      (i,) = index
      qi = lb + i
      q_base = qi * block_q
      qs = pl.ds(q_base, block_q)
      dk_acc, dv_acc, loop_invariant_mask = carry

      def compute_sT(acc):
        plgpu.wgmma(acc, k_smem, q_smem.T)
        if bias_gmem is None:
          biasT = None
        elif bias_smem is None:
          biasT = _load_bcast(bias_gmem.T, (ks, qs), layout=_WGMMA)
        else:
          idx = pl.ds(wg * block_kv, block_kv)
          biasT = plgpu.load(bias_smem.T, idx, layout=_WGMMA_TRANSPOSED)
          plgpu.barrier_arrive(bias_consumed_barrier)

        m = plgpu.load(m_smem, (), layout=_WGMMA_COL)
        l = plgpu.load(l_smem, (), layout=_WGMMA_COL)
        plgpu.barrier_arrive(m_consumed_barrier)
        plgpu.barrier_arrive(l_consumed_barrier)
        return acc[...], biasT, m, l

      acc_type = plgpu.ACC((block_kv, block_q), jnp.float32)
      sT, biasT, m, l = pl.run_scoped(compute_sT, acc_type)
      scale = logits_scale

      if biasT is not None:
        sT = sT * scale + plgpu.layout_cast(biasT.astype(sT.dtype), _WGMMA)
        scale = 1.0

      if logits_soft_cap is not None:
        sT = logits = jnp.tanh(sT * (scale / logits_soft_cap))
        scale = logits_soft_cap

      # NOTE: This rescaling must happen after bias and soft-cap but before the
      # attention masking (as the multiplication will cause `-inf`s).
      if use_base2:
        scale *= math.log2(math.e)
        m *= math.log2(math.e)

      sT *= scale

      mask_value = float(jnp.finfo(jnp.float32).min)

      def iota(d):
        return plgpu.broadcasted_iota(jnp.int32, sT.shape, d, layout=_WGMMA)

      if is_causal:

        def apply_causal_mask():
          mask = kv_base + iota(0) <= q_base + iota(1)
          return jnp.where(mask, sT, mask_value)

        needs_causal_mask = kv_base + block_kv > q_base
        sT = lax.cond(needs_causal_mask, apply_causal_mask, lambda: sT)

      def load_k_range(ref):
        idx = (0 if (ref.shape[0] == 1) else hi, qs)
        return plgpu.load(ref, idx, layout=_WGMMA_COL, optimized=False)

      if k_start_gmem is not None:
        k_start = load_k_range(k_start_gmem)
        k_start = lax.broadcast_in_dim(k_start, sT.shape, [1])
        sT = jnp.where(kv_base + iota(0) >= k_start, sT, mask_value)

      if k_end_gmem is not None:
        k_end = load_k_range(k_end_gmem)
        k_end = lax.broadcast_in_dim(k_end, sT.shape, [1])
        sT = jnp.where(kv_base + iota(0) < k_end, sT, mask_value)

      if mask_gmem is not None:
        if mask_smem is None:
          if loop_invariant_mask is None:
            maskT = _load_bcast(mask_gmem, (ks, qs), layout=_WGMMA)
          else:
            maskT = lax.broadcast_in_dim(loop_invariant_mask, sT.shape, [0])
        else:
          maskT = mask_smem[pl.ds(wg * block_kv, block_kv)]
          plgpu.barrier_arrive(mask_consumed_barrier)

        sT = jnp.where(maskT, sT, mask_value)

      broadcast = lambda x: lax.broadcast_in_dim(x, sT.shape, [1])
      epsilon = float(jnp.finfo(jnp.float32).tiny)  # Avoid division by zero.
      pT = exp(sT - broadcast(m)) / broadcast(l + epsilon)

      def compute_dpT(acc):
        plgpu.wgmma(acc, v_smem, dout_smem.T)
        plgpu.wgmma(dv_acc, pT.astype(dtype), dout_smem)
        # TODO: Load this without waiting for the DV matmul to complete.
        return acc[...]

      dpT = pl.run_scoped(compute_dpT, plgpu.ACC(pT.shape, jnp.float32))
      plgpu.barrier_arrive(dout_consumed_barrier)

      delta = plgpu.load(delta_smem, (), layout=_WGMMA_COL)
      plgpu.barrier_arrive(delta_consumed_barrier)

      dsT = pT * (dpT - broadcast(delta))  # pytype: disable=wrong-arg-types  # jax-operator-types
      if logits_soft_cap is not None:
        dsT *= 1 - logits * logits

      plgpu.wgmma(dk_acc, dsT.astype(dtype), q_smem)
      plgpu.wgmma_wait(0)
      plgpu.barrier_arrive(q_consumed_barrier)
      return dk_acc, dv_acc, loop_invariant_mask

    q_spec = tiled_spec(
        (block_q, head_dim), q.dtype, lambda i: (lb + i, 0), "q"
    )
    dout_spec = tiled_spec(
        (block_q, head_dim_out), dout.dtype, lambda i: (lb + i, 0), "dout"
    )
    m_spec = l_spec = delta_spec = plgpu.BlockSpec(
        (block_q,), lambda i: (lb + i,)
    )

    if bias is not None and bias.shape[-2] != 1 and bias.shape[-1] != 1:
      bias_gmem_ = bias_gmem
      bias_spec = tiled_spec(
          (block_q, tile_kv), bias.dtype, lambda i: (lb + i, ki), "bias"
      )
    else:
      bias_gmem_ = bias_spec = None

    if mask is not None and mask.shape[-2] != 1 and mask.shape[-1] != 1:
      mask_gmem_ = mask_gmem
      mask_spec = tiled_spec(
          (tile_kv, block_q), mask.dtype, lambda i: (ki, lb + i), "mask"
      )
    else:
      mask_gmem_ = mask_spec = None

    plgpu.emit_pipeline_warp_specialized(
        q_pipeline,
        grid=(ub - lb,),
        max_concurrent_steps=min(config.num_stages, num_q_tiles),
        num_compute_wgs=compute_wgs,
        memory_registers=40,
        wg_axis="wg",
        manual_consumed_barriers=True,
        compute_context=compute_thread,
        in_specs=[
            bias_spec,
            m_spec,
            l_spec,
            mask_spec,
            dout_spec,
            delta_spec,
            q_spec,
        ],
    )(bias_gmem_, m_gmem, l_gmem, mask_gmem_, dout_gmem, delta_gmem, q_gmem)

  def tiled_wgs_smem(shape, dtype, what=""):
    transforms = common.tile_swizzle_transforms(shape, dtype, what)
    return plgpu.SMEM((compute_wgs, *shape), dtype, transforms=transforms)

  tile_q_dq = compute_wgs * block_q_dq
  if bias is None:
    ds_out_shape = None
  else:
    # NOTE: TMA stores to GMEM do not mask out-of-bounds writes, so we must pad
    # the output to a multiple of the block size.
    q_seq_len_ = pl.cdiv(q_seq_len, tile_q_dq) * tile_q_dq
    kv_seq_len_ = pl.cdiv(kv_seq_len, block_kv_dq) * block_kv_dq
    ds_out_shape = (num_q_heads, q_seq_len_, kv_seq_len_)
    ds_out_shape = jax.ShapeDtypeStruct(ds_out_shape, ds_dtype)
  # TODO: Optionally fuse the dq and dkv kernels.
  dq, ds = plgpu.kernel(
      functools.partial(kernel_dq, block_q=block_q_dq, block_kv=block_kv_dq),
      out_shape=(jax.ShapeDtypeStruct(q.shape, q.dtype), ds_out_shape),
      scratch_shapes=[
          tiled_wgs_smem((block_q_dq, head_dim), q.dtype, "q"),
          tiled_wgs_smem((block_q_dq, head_dim_out), dout.dtype, "dout"),
          plgpu.Barrier(num_barriers=compute_wgs, num_arrivals=2),
      ],
      compiler_params=plgpu.CompilerParams(approx_math=True),
      grid=(num_q_heads, pl.cdiv(q_seq_len, tile_q_dq)),
      grid_names=("heads", "q_tiles"),
      num_threads=compute_wgs + 1,
      thread_name="wg",
  )(q, k, v, dout, m, l, delta, bias, mask, k_start, k_end)

  # TODO: Fuse transpose in the kernel.
  if mask is not None:
    mask = mask.mT

  # `dk` and `dv` outputs have `num_q_heads` heads (reduced below if necessary).
  out_shape = (
      jax.ShapeDtypeStruct((kv_seq_len, num_q_heads, head_dim), k.dtype),
      jax.ShapeDtypeStruct((kv_seq_len, num_q_heads, head_dim_out), v.dtype),
  )

  tile_kv_dkv = compute_wgs * block_kv_dkv
  dk, dv = plgpu.kernel(
      functools.partial(kernel_dkv, block_q=block_q_dkv, block_kv=block_kv_dkv),
      out_shape=out_shape,
      scratch_shapes=[
          tiled_wgs_smem((block_kv_dkv, head_dim), k.dtype, "k"),
          tiled_wgs_smem((block_kv_dkv, head_dim_out), v.dtype, "v"),
          plgpu.Barrier(num_barriers=compute_wgs, num_arrivals=2),
      ],
      compiler_params=plgpu.CompilerParams(approx_math=True),
      grid=(num_q_heads, pl.cdiv(kv_seq_len, tile_kv_dkv)),
      grid_names=("heads", "kv_tiles"),
      num_threads=compute_wgs + 1,
      thread_name="wg",
  )(q, k, v, dout, m, l, delta, bias, mask, k_start, k_end)

  if q_heads_per_kv_head > 1:
    dk = dk.reshape(*k.shape[:-1], q_heads_per_kv_head, -1).sum(axis=-2)
    dv = dv.reshape(*v.shape[:-1], q_heads_per_kv_head, -1).sum(axis=-2)

  dq = dq[..., :orig_head_dim]
  dk = dk[..., :orig_head_dim]
  dv = dv[..., :orig_head_dim_out]
  ds = None if ds is None else ds[:, :q_seq_len, :kv_seq_len]
  return dq, dk, dv, ds
