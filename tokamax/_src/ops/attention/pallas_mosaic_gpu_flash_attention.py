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
"""Flash attention with Mosaic GPU."""

import dataclasses
import math
from typing import Any, TypeAlias

import immutabledict
import jax
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src import quantization
from tokamax._src.ops import op
from tokamax._src.ops.attention import base


# pylint: disable=cell-var-from-loop


DotPrecisionLike = lax.Precision | lax.DotAlgorithmPreset
QuantizedArray = quantization.QuantizedArray
Residuals = base.Residuals


@dataclasses.dataclass(frozen=True)
class Config:
  block_q: int = 64
  block_kv: int = 64
  num_stages: int = 2

  def __post_init__(self):
    if self.block_q % 64:
      raise ValueError(f"{self.block_q=} must be a multiple of 64")
    if self.block_kv % 64:
      raise ValueError(f"{self.block_kv=} must be a multiple of 64")
    if self.num_stages < 2:
      raise ValueError(f"{self.num_stages=} must be at least 2")


@jaxtyping.jaxtyped
def _attention_fwd(
    q: Float[Array, "*B T H D"],
    k: Float[Array, "*B t h D"],
    v: Float[Array, "*B t h d"],
    *,
    bias: Float[Array, "*#B #H #T #t"] | None,
    mask: Bool[Array, "*#B #H #T #t"] | None,
    k_start: Int[Array, "*#B #H #T"] | None,
    k_end: Int[Array, "*#B #H #T"] | None,
    is_causal: bool,
    logits_soft_cap: float | None,
    logits_scale: float,
    normalize_output: bool,
    return_residuals: bool,
    use_base2: bool,
    config: Config,
) -> tuple[Float[Array, "*B T H d"], Residuals | None]:
  """Flash attention with Mosaic GPU."""

  orig_q_shape = q.shape
  as_4d = lambda x: jax.lax.collapse(jax.lax.broadcast_to_rank(x, 4), 0, -3)
  q, k, v = map(as_4d, (q, k, v))

  batch_size, q_seq_len, num_q_heads, head_dim = q.shape
  _, kv_seq_len, num_kv_heads, head_dim_out = v.shape
  orig_head_dim = head_dim
  orig_head_dim_out = head_dim_out

  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by and {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  if head_dim % 64:
    head_dim = pl.cdiv(orig_head_dim, 64) * 64
    pad = lambda x: jnp.pad(x, (*[(0, 0)] * 3, (0, head_dim - orig_head_dim)))
    q, k = map(pad, (q, k))
  if head_dim_out % 64:
    head_dim_out = pl.cdiv(orig_head_dim_out, 64) * 64
    v = jnp.pad(v, (*[(0, 0)] * 3, (0, head_dim_out - orig_head_dim_out)))

  max_concurrent_steps = min(config.num_stages, kv_seq_len // config.block_kv)
  block_q_kv = block_q, block_kv = config.block_q, config.block_kv
  num_q_tiles, rem = divmod(q_seq_len, block_q * 2)
  if rem:
    raise NotImplementedError(
        f"{q_seq_len=} must be a multiple of {block_q * 2=}"
    )

  # Fold the batch dimension into the seq dimension and we will do the index
  # arithmetic by hand in the kernel because mgpu can't handle too high rank
  # slices.
  # TODO(giorgioa): Fold num_heads as well.
  q, k, v = map(lambda x: jax.lax.collapse(x, 0, 2), (q, k, v))

  logits_shape = (batch_size, num_q_heads, q_seq_len, kv_seq_len)
  if bias is not None:
    bias = jnp.broadcast_to(bias, logits_shape)
    bias = as_4d(bias)
    bias = jnp.swapaxes(bias, -2, -3)  # [B, T, H, t]
    bias = jax.lax.collapse(bias, 0, 2)  # [B*T, H, t]

  if mask is not None:
    mask = as_4d(mask)
    bcast_mask_dims = [d == 1 for d in mask.shape]
    bcast_mask_b, bcast_mask_h, bcast_mask_q, bcast_mask_k = bcast_mask_dims
    if bcast_mask_q and bcast_mask_k:
      raise NotImplementedError("Broadcast on both sequences not supported.")
    mask = jnp.swapaxes(mask, -2, -3)  # [B, T, H, t]
    mask = jax.lax.collapse(mask, 0, 2)  # [B*T, H, t]
    mask = mask.astype(jnp.int8)

  # TODO(giorgioa): Avoid broadcast.
  def broadcast_and_collapse(x):
    if x.shape[-2] == 1:
      return jnp.broadcast_to(x, (batch_size, 1, q_seq_len)).flatten()  # [b*t]
    x = jnp.broadcast_to(x, (batch_size, num_q_heads, q_seq_len))  # [b, h, t]
    x = jnp.swapaxes(x, -2, -3)  # [h, b, t]
    return jax.lax.collapse(x, -2)  # [h, b*t]

  k_start = None if k_start is None else broadcast_and_collapse(k_start)
  k_end = None if k_end is None else broadcast_and_collapse(k_end)

  L: TypeAlias = plgpu.Layout

  def kernel(*refs, scoped):
    bidx = lax.axis_index("batch")
    qidx = lax.axis_index("q_tiles")
    q_head = lax.axis_index("heads")
    wg_idx = lax.axis_index("wg")

    (
        q_ref,
        k_ref,
        v_ref,
        bias_ref,
        mask_ref,
        k_start_ref,
        k_end_ref,
        k_start_min_ref,
        k_end_max_ref,
        out_ref,
        *residual_gmem_refs,
    ) = refs

    (
        smem_buffers,
        residual_smem_buffers,
        buffer_barriers,
        consumed_barriers,
        schedule_barrier,
    ) = scoped
    (q_smems, o_smems), k_smem, v_smem, bias_smem, mask_smem = smem_buffers
    q_barriers, k_barriers, v_barriers, bias_barriers, mask_barriers = (
        buffer_barriers
    )
    (
        k_consumed_barriers,
        v_consumed_barriers,
        bias_consumed_barriers,
        mask_consumed_barriers,
    ) = consumed_barriers

    use_k_ranges = k_start_ref is not None or k_end_ref is not None

    def perform_schedule_barrier():
      plgpu.barrier_arrive(schedule_barrier)
      plgpu.barrier_wait(schedule_barrier)

    min_kv_step = 0
    max_kv_step = kv_seq_len // block_kv

    if is_causal:
      max_kv_step = lax.min(max_kv_step, 2 * (qidx + 1))
    if use_k_ranges:
      idx = bidx * num_q_tiles + qidx
      load_kwargs = dict(idx=idx, layout=L.WG_SPLAT)

      if k_start_min_ref is not None:
        bcast_num_heads = k_start_min_ref.ndim == 1
        ref = k_start_min_ref if bcast_num_heads else k_start_min_ref.at[q_head]
        k_start_min = plgpu.load(ref, **load_kwargs)
        assert isinstance(k_start_min, jax.Array)
        # TODO(giorgioa): Why do we need to subtract 2?
        min_kv_step = lax.max(min_kv_step, k_start_min - 2)
      if k_end_max_ref is not None:
        bcast_num_heads = k_end_max_ref.ndim == 1
        ref = k_end_max_ref if bcast_num_heads else k_end_max_ref.at[q_head]
        k_end_max = plgpu.load(ref, **load_kwargs)
        assert isinstance(k_end_max, jax.Array)
        max_kv_step = lax.min(max_kv_step, k_end_max)

    stages = lax.min(max_concurrent_steps, max_kv_step - min_kv_step)
    stages = lax.max(1, stages)

    q_seq_off = bidx * q_seq_len + qidx * (2 * block_q)
    bias_mask_barriers_idx = wg_idx * max_concurrent_steps

    if mask_ref is not None:
      mask_bidx = 0 if bcast_mask_b else bidx
      mask_hidx = 0 if bcast_mask_h else q_head
      mask_q_seq_off = mask_bidx * q_seq_len + qidx * (2 * block_q)

    if bias_ref is not None:
      bias_bidx = bidx
      bias_hidx = q_head
      bias_q_seq_off = bias_bidx * q_seq_len + qidx * (2 * block_q)

    @pl.when(wg_idx < 2)
    def _compute_wg():

      plgpu.set_max_registers(232, action="increase")
      q_smem, o_smem = q_smems.at[wg_idx], o_smems.at[wg_idx]
      residual_smem = [ref.at[wg_idx] for ref in residual_smem_buffers]

      q_slice = pl.ds(q_seq_off + wg_idx * block_q, block_q)

      plgpu.copy_gmem_to_smem(
          q_ref.at[q_slice, q_head], q_smem, barrier=q_barriers.at[wg_idx]
      )
      plgpu.barrier_wait(q_barriers.at[wg_idx])

      m_i = plgpu.layout_cast(
          jnp.full((block_q,), -jnp.inf, dtype=jnp.float32), L.WGMMA_ROW
      )
      l_i = plgpu.layout_cast(
          jnp.full((block_q,), 0, dtype=jnp.float32), L.WGMMA_ROW
      )
      acc = plgpu.layout_cast(
          jnp.full((block_q, head_dim_out), 0, dtype=jnp.float32), L.WGMMA
      )

      ranges_load_kwargs = dict(layout=L.WGMMA_ROW, optimized=False)
      if k_start_ref is not None:
        slice_ = (q_slice,) if k_start_ref.ndim == 1 else (q_head, q_slice)
        k_start = plgpu.load(k_start_ref, slice_, **ranges_load_kwargs)
      if k_end_ref is not None:
        slice_ = (q_slice,) if k_end_ref.ndim == 1 else (q_head, q_slice)
        k_end = plgpu.load(k_end_ref, slice_, **ranges_load_kwargs)

      first_tma_slot = lax.rem(min_kv_step, stages)
      plgpu.barrier_wait(k_barriers.at[first_tma_slot])
      pl.when(wg_idx == 1)(perform_schedule_barrier)

      def kv_loop(kv_step, carry, *, do_causal=False):
        acc, m_i, l_i = carry
        slot = lax.rem(kv_step, stages)

        def compute_scores(acc_ref):
          plgpu.wgmma(
              acc_ref, q_smem, plgpu.transpose_ref(k_smem.at[slot], (1, 0))
          )
          perform_schedule_barrier()
          acc = acc_ref[...] * logits_scale

          if bias_ref is not None:
            bias_barrier_idx = bias_mask_barriers_idx + slot
            plgpu.barrier_wait(bias_barriers.at[bias_barrier_idx])
            bias = bias_smem.at[wg_idx, slot][...]
            plgpu.barrier_arrive(bias_consumed_barriers.at[bias_barrier_idx])
            acc += bias.astype(acc_ref.dtype)

          if logits_soft_cap is not None:
            acc = logits_soft_cap * jnp.tanh(acc / logits_soft_cap)

          if use_base2:
            acc *= math.log2(math.e)

          if do_causal or use_k_ranges or mask_ref is not None:
            if acc_ref.shape[0] != acc_ref.shape[1]:
              raise NotImplementedError(
                  "Only masks for square accumulators for now."
              )

            iota = lambda d: plgpu.broadcasted_iota(
                acc_ref.dtype, acc_ref.shape, dimension=d, layout=L.WGMMA
            )

            zero = jnp.zeros(acc_ref.shape, dtype=jnp.float32)
            f32_min = jnp.full(acc_ref.shape, float(jnp.finfo(jnp.float32).min))
            mask_fn = lambda cond, mask: jnp.where(cond, mask, f32_min)

            mask = zero
            if do_causal:
              mask = mask_fn(iota(0) >= iota(1), mask)
            if use_k_ranges:
              bc_range = lambda x: lax.broadcast_in_dim(x, acc_ref.shape, (0,))
              block_kv_iota = iota(1) + (kv_step * block_kv)

              if k_start_ref is not None:
                assert isinstance(k_start, jax.Array)
                mask = mask_fn(bc_range(k_start) <= block_kv_iota, mask)
              if k_end_ref is not None:
                assert isinstance(k_end, jax.Array)
                mask = mask_fn(bc_range(k_end) > block_kv_iota, mask)
            if mask_ref is not None:
              if bcast_mask_q:
                i = (mask_bidx, mask_hidx, pl.ds(kv_step * block_kv, block_kv))
                m = plgpu.load(mask_ref, i, layout=L.WGMMA_COL, optimized=False)
                m = lax.broadcast_in_dim(m, (block_q, block_kv), (1,))
                mask = mask_fn(m, mask)
              elif bcast_mask_k:
                mask_slice_q = pl.ds(mask_q_seq_off + wg_idx * block_q, block_q)
                i = (mask_slice_q, mask_hidx, 0)
                m = plgpu.load(mask_ref, i, layout=L.WGMMA_ROW, optimized=False)
                m = lax.broadcast_in_dim(m, (block_q, block_kv), (0,))
                mask = mask_fn(m, mask)
              else:
                barrier_idx = bias_mask_barriers_idx + slot
                plgpu.barrier_wait(mask_barriers.at[barrier_idx])
                mask = mask_fn(mask_smem.at[wg_idx, slot][...], mask)
                plgpu.barrier_arrive(mask_consumed_barriers.at[barrier_idx])
            acc += mask
          return acc

        s = pl.run_scoped(compute_scores, plgpu.ACC(block_q_kv, jnp.float32))

        plgpu.barrier_arrive(k_consumed_barriers.at[slot])

        # Softmax
        exp = jnp.exp2 if use_base2 else jnp.exp
        m_ij = jnp.maximum(m_i, s.max(axis=1))
        alpha = exp(m_i - m_ij)
        m_i = m_ij
        p = exp(s - lax.broadcast_in_dim(m_ij, (block_q, block_kv), [0]))
        acc *= lax.broadcast_in_dim(alpha, (block_q, head_dim_out), [0])
        l_i *= alpha
        p16 = p.astype(q.dtype)

        plgpu.barrier_wait(v_barriers.at[slot])
        perform_schedule_barrier()

        l_i += p.sum(axis=1)

        # PV
        def compute_pv(acc_ref):
          plgpu.wgmma(acc_ref, p16, v_smem.at[slot])

          wait_step = kv_step + 1
          wait_slot = lax.rem(wait_step, stages)

          @pl.when(wait_step < max_kv_step)
          def _wait():
            plgpu.barrier_wait(k_barriers.at[wait_slot])

        acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
        plgpu.barrier_arrive(v_consumed_barriers.at[slot])
        return acc, m_i, l_i

      if kv_seq_len % block_kv:
        raise ValueError(f"{kv_seq_len=} must be a multiple of {block_kv=}")

      acc, m_i, l_i = lax.fori_loop(
          min_kv_step,
          max_kv_step - 2 * is_causal,
          kv_loop,
          (acc, m_i, l_i),
      )
      if is_causal:

        def noop_kv_loop(kv_step):
          slot = kv_step % stages
          perform_schedule_barrier()
          plgpu.barrier_arrive(k_consumed_barriers.at[slot])
          perform_schedule_barrier()
          if bias_ref is not None:
            plgpu.barrier_arrive(
                bias_consumed_barriers.at[bias_mask_barriers_idx + slot]
            )
            perform_schedule_barrier()

          if mask_ref is not None:
            plgpu.barrier_arrive(
                mask_consumed_barriers.at[bias_mask_barriers_idx + slot]
            )
            perform_schedule_barrier()
          plgpu.barrier_arrive(v_consumed_barriers.at[slot])

        def _wg0_kv_epilogue():
          carry = kv_loop(max_kv_step - 2, (acc, m_i, l_i), do_causal=True)
          noop_kv_loop(max_kv_step - 1)
          return carry

        def _wg1_kv_epilogue():
          carry = kv_loop(max_kv_step - 2, (acc, m_i, l_i), do_causal=False)
          return kv_loop(max_kv_step - 1, carry, do_causal=True)

        acc, m_i, l_i = lax.cond(
            wg_idx == 0, _wg0_kv_epilogue, _wg1_kv_epilogue
        )

      pl.when(wg_idx == 0)(perform_schedule_barrier)

      if return_residuals:
        residual_smem[0][...], residual_smem[1][...] = m_i, l_i
        plgpu.commit_smem()
        for smem, gmem in zip(residual_smem, residual_gmem_refs):
          plgpu.copy_smem_to_gmem(smem, gmem.at[q_head, q_slice])

      l_i += float(jnp.finfo(jnp.float32).tiny)

      if normalize_output:
        # TODO(apaszke): Invert and multiply to avoid expensive divisions.
        acc /= lax.broadcast_in_dim(l_i, (block_q, head_dim_out), [0])

      o_smem[...] = acc.astype(q.dtype)  # pytype: disable=attribute-error
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(o_smem, out_ref.at[q_slice, q_head])
      plgpu.wait_smem_to_gmem(0)

    @pl.when(wg_idx == 2)
    def _memory_wg():
      plgpu.set_max_registers(40, action="decrease")
      kv_batch_off = bidx * kv_seq_len
      kv_head = lax.div(q_head, q_heads_per_kv_head)

      def async_load(ds, ref, smem, barrs, cons_barrs, smem_slot, barr_slot):
        barr_slot = smem_slot if barr_slot is None else barr_slot
        if cons_barrs is not None:
          plgpu.barrier_wait(cons_barrs.at[barr_slot])
        plgpu.copy_gmem_to_smem(
            ref.at[*ds], smem.at[smem_slot], barrier=barrs.at[barr_slot]
        )

      def kv_async_load(step, ref, smem, barrs, cons_barrs=None, slot=None):
        slot = step if slot is None else slot
        slice_ = (pl.ds(kv_batch_off + step * block_kv, block_kv), kv_head)
        async_load(slice_, ref, smem, barrs, cons_barrs, slot, None)

      def mask_async_load(step, ref, smem, barrs, cons_barrs=None, slot=None):
        if ref is None or bcast_mask_q or bcast_mask_k:
          return
        slot = step if slot is None else slot
        for wg in range(2):
          qs = pl.ds(mask_q_seq_off + wg * block_q, block_q)
          slice_ = (qs, mask_hidx, pl.ds(step * block_kv, block_kv))
          smem_slot = (wg, slot)
          barr_slot = wg * max_concurrent_steps + slot
          async_load(slice_, ref, smem, barrs, cons_barrs, smem_slot, barr_slot)

      def bias_async_load(step, ref, smem, barrs, cons_barrs=None, slot=None):
        if ref is None:
          return
        slot = step if slot is None else slot
        for wg in range(2):
          qs = pl.ds(bias_q_seq_off + wg * block_q, block_q)
          slice_ = (qs, bias_hidx, pl.ds(step * block_kv, block_kv))
          smem_slot = (wg, slot)
          barr_slot = wg * max_concurrent_steps + slot
          async_load(slice_, ref, smem, barrs, cons_barrs, smem_slot, barr_slot)

      for i in range(max_concurrent_steps):

        @pl.when(i < stages)
        def _preload_kv_bias_mask():
          tma_slot = lax.rem(min_kv_step + i, stages)
          kv_async_load(tma_slot, k_ref, k_smem, k_barriers)
          bias_async_load(tma_slot, bias_ref, bias_smem, bias_barriers)
          mask_async_load(tma_slot, mask_ref, mask_smem, mask_barriers)
          kv_async_load(tma_slot, v_ref, v_smem, v_barriers)

      @pl.loop(min_kv_step, max_kv_step - stages)
      def _kv_loop(kv_step):
        tma_step = kv_step + stages
        tma_slot = lax.rem(kv_step, stages)
        closed = lambda fn, ref, smem, barrs, cons_barrs: fn(
            tma_step, ref, smem, barrs, cons_barrs, tma_slot
        )

        closed(kv_async_load, k_ref, k_smem, k_barriers, k_consumed_barriers)
        # Load bias before mask
        bias_args = bias_ref, bias_smem, bias_barriers, bias_consumed_barriers
        closed(bias_async_load, *bias_args)
        mask_args = mask_ref, mask_smem, mask_barriers, mask_consumed_barriers
        closed(mask_async_load, *mask_args)
        closed(kv_async_load, v_ref, v_smem, v_barriers, v_consumed_barriers)

  def run(refs):
    mesh = plgpu.Mesh(
        grid=(batch_size, num_q_tiles, num_q_heads),
        grid_names=("batch", "q_tiles", "heads"),
        num_threads=3,
        thread_name="wg",
    )

    compiler_params = plgpu.CompilerParams(approx_math=True)

    @pl.core_map(mesh, compiler_params=compiler_params)
    def _kernel_entry():
      compute_wgs = 2
      tiling = plgpu.TilingTransform((8, 64))
      swizzle = plgpu.SwizzleTransform(128)

      # SMEM scratch buffers
      l_scratch = m_scratch = plgpu.SMEM((compute_wgs, block_q), jnp.float32)
      qo_scratch = plgpu.RefUnion(
          plgpu.SMEM(
              (compute_wgs, block_q, head_dim),
              q.dtype,
              transforms=(tiling, swizzle),
          ),
          plgpu.SMEM(
              (compute_wgs, block_q, head_dim_out),
              q.dtype,
              transforms=(tiling, swizzle),
          )
      )
      k_scratch = plgpu.SMEM(
          (max_concurrent_steps, block_kv, head_dim),
          k.dtype,
          transforms=(tiling, swizzle),
      )
      v_scratch = plgpu.SMEM(
          (max_concurrent_steps, block_kv, head_dim_out),
          v.dtype,
          transforms=(tiling, swizzle),
      )

      # Barriers
      q_barriers = plgpu.Barrier(num_barriers=compute_wgs)
      kv_barriers = plgpu.Barrier(num_barriers=max_concurrent_steps)
      kv_consumed_barriers = plgpu.Barrier(
          num_arrivals=compute_wgs, num_barriers=max_concurrent_steps
      )
      schedule_barrier = plgpu.Barrier(num_arrivals=compute_wgs)

      # Mask scratch buffer/barriers
      num_barriers = compute_wgs * max_concurrent_steps
      if mask is None:
        mask_scratch = None
        mask_barriers = None
        mask_consumed_barriers = None
      else:
        mask_scratch = plgpu.SMEM(
            (compute_wgs, max_concurrent_steps, block_q, block_kv),
            jnp.int8,
            transforms=(tiling, plgpu.SwizzleTransform(block_kv)),
        )
        mask_barriers = plgpu.Barrier(num_barriers=num_barriers)
        mask_consumed_barriers = plgpu.Barrier(num_barriers=num_barriers)

      # Bias scratch buffer/barriers
      if bias is None:
        bias_scratch = None
        bias_barriers = None
        bias_consumed_barriers = None
      else:
        bias_swizzle_elems = 128 // jnp.dtype(bias.dtype).itemsize
        bias_scratch = plgpu.SMEM(
            (compute_wgs, max_concurrent_steps, block_q, block_kv),
            bias.dtype,
            # Bias has the same type as logits.
            transforms=(
                plgpu.TilingTransform((8, bias_swizzle_elems)),
                swizzle,
            ),
        )
        bias_barriers = plgpu.Barrier(num_barriers=num_barriers)
        bias_consumed_barriers = plgpu.Barrier(num_barriers=num_barriers)
      pl.run_scoped(
          lambda *args: kernel(*refs, scoped=args),
          (qo_scratch, k_scratch, v_scratch, bias_scratch, mask_scratch),
          (l_scratch, m_scratch) if return_residuals else (),
          (q_barriers, kv_barriers, kv_barriers, bias_barriers, mask_barriers),
          (
              kv_consumed_barriers,
              kv_consumed_barriers,
              bias_consumed_barriers,
              mask_consumed_barriers,
          ),
          schedule_barrier,
          collective_axes="wg",
      )

  outs = (jnp.full((*q.shape[:-1], head_dim_out), jnp.inf, q.dtype),)
  if return_residuals:
    residuals_shape = (num_q_heads, batch_size * q_seq_len)
    m_i = jnp.full(residuals_shape, -jnp.inf, dtype=jnp.float32)
    l_i = jnp.full(residuals_shape, 0, dtype=jnp.float32)
    outs += (m_i, l_i)

  def preprocess_k_range(reduction_fn, x):
    if x is None:
      return None
    reshape = lambda x, d: x.reshape((*x.shape[:-1], x.shape[-1] // d, d))
    # Pre-reduce the k_start/k_end to a single value per q_block.
    x = reduction_fn(reshape(x, block_q), axis=-1)
    # The kernel processes 2 q_blocks per thread-block (one per warpgroup), so
    # we need to reduce over two q_blocks.
    x = reduction_fn(reshape(x, 2), axis=-1)
    # Finally, perform the division by block_kv to get the number of steps.
    round_ = jnp.floor if reduction_fn == jnp.min else jnp.ceil
    return round_(x / block_kv).astype(jnp.int32)

  k_start_min = preprocess_k_range(jnp.min, k_start)
  k_end_max = preprocess_k_range(jnp.max, k_end)

  _, _, _, _, _, _, _, _, _, out, *residuals = pl.run_state(run)(
      (q, k, v, bias, mask, k_start, k_end, k_start_min, k_end_max, *outs)
  )
  out = out.reshape(*orig_q_shape[:-1], out.shape[-1])[..., :orig_head_dim_out]
  residuals = tuple(
      jnp.moveaxis(res.reshape(-1, *orig_q_shape[:-3], q_seq_len), 0, -2)
      for res in residuals
  )
  return out, (residuals if return_residuals else None)


Key: TypeAlias = immutabledict.immutabledict[str, Any]


def _decompose_mask(mask, q, k):
  """Decomposes `mask` into a mask array, `is_causal`, `k_start` and `k_end`."""
  if mask is None:
    return None, False, None, None

  mask, is_causal, k_start, k_end = mask.take("is_causal", "k_start", "k_end")

  # Fold is_causal into k_end to avoid conflicts (and allow sharding).
  # TODO(giorgioa): Fold is_causal into k_end only if k_end is not None.
  if is_causal:
    k_end_ = jnp.arange(q.shape[-3]) + 1
    k_end = k_end_ if k_end is None else jnp.minimum(k_end, k_end_)
    is_causal = False

  bcast_rank2 = lambda x: None if x is None else jax.lax.broadcast_to_rank(x, 2)
  k_start = bcast_rank2(k_start)
  k_end = bcast_rank2(k_end)

  mask = mask.as_array(q.shape[-3], k.shape[-3])
  return mask, is_causal, k_start, k_end


_SUPPORTED_PRECISIONS = (
    lax.DotAlgorithmPreset.DEFAULT,
    lax.DotAlgorithmPreset.BF16_BF16_F32,
    lax.DotAlgorithmPreset.F16_F16_F32,
)


@dataclasses.dataclass(frozen=True)
class PallasMosaicGpuFlashAttention(base.DotProductAttention[Config, Key]):
  """Flash attention with Mosaic GPU."""

  use_base2: bool = False

  @jaxtyping.jaxtyped
  def _fwd(
      self,
      q: Float[Array | QuantizedArray, "*B T H D"],
      k: Float[Array | QuantizedArray, "*B t h D"],
      v: Float[Array | QuantizedArray, "*B t h d"],
      *,
      precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset],
      logits_dtype: jnp.dtype,
      logits_scale: float,
      bias: Float[Array, "*#B #H #T #t"] | None,
      logits_soft_cap: float | None,
      mask: base.Mask,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
      dropout_rate: float,
      q_indices: Int[Array, "*#B #H T"] | None,
      k_indices: Int[Array, "*#B #H t"] | None,
      normalize_output: bool,
      return_residuals: bool,
      config: Config,
  ) -> tuple[Float[Array, "*B T H d"], Residuals | None]:
    """Performs attention, optionally returning softmax residuals."""

    if jnp.dtype(q.dtype) not in map(jnp.dtype, [jnp.float16, jnp.bfloat16]):
      raise NotImplementedError(
          f"Only f16 and bf16 are supported, got dtype: {q.dtype}"
      )

    if logits_dtype != jnp.float32:
      raise NotImplementedError("`logits_dtype` must be float32.")
    if dropout_mask is not None:
      raise NotImplementedError("dropout is not supported.")
    if q_indices is not None:
      raise NotImplementedError("q_indices is not implemented")
    if k_indices is not None:
      raise NotImplementedError("k_indices is not implemented")

    # TODO(giorgioa): Support in-kernel dequantization.
    q, k, v = map(base.as_array, (q, k, v))
    # FIXME(cjfj): We shouldn't silently downcast types.
    k = k.astype(q.dtype)
    v = v.astype(q.dtype)

    q_k_dot_precision, weights_v_dot_precision = precision
    if q_k_dot_precision not in _SUPPORTED_PRECISIONS:
      raise NotImplementedError(f"{q_k_dot_precision=} not supported")
    if weights_v_dot_precision not in _SUPPORTED_PRECISIONS:
      raise NotImplementedError(f" {weights_v_dot_precision=} not supported")

    mask, is_causal, k_start, k_end = _decompose_mask(mask, q, k)

    return _attention_fwd(
        q,
        k,
        v,
        bias=bias,
        mask=mask,
        k_start=k_start,
        k_end=k_end,
        is_causal=is_causal,
        logits_soft_cap=logits_soft_cap,
        logits_scale=logits_scale,
        normalize_output=normalize_output,
        return_residuals=return_residuals,
        use_base2=self.use_base2,
        config=config,
    )

  def _get_heuristics_config(self, ba: op.BoundArguments):
    del ba
    # This is a pretty good option that works for most cases.
    # TODO(cperivol): for larger embed dimensions we can probably guess better.
    return Config(block_q=64, block_kv=64, num_stages=2)
