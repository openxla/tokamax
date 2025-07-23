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
from tokamax._src import mosaic_gpu
from tokamax._src import quantization
from tokamax._src import shape as shape_lib
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_mosaic_gpu_flash_attention_vjp as vjp
from tokamax._src.pallas import block


# pylint: disable=cell-var-from-loop


DotPrecisionLike = lax.Precision | lax.DotAlgorithmPreset
QuantizedArray = quantization.QuantizedArray
Residuals = base.Residuals
PagingInfo = base.PagingInfo
_WGMMA = plgpu.Layout.WGMMA
_WGMMA_ROW = plgpu.Layout.WGMMA_ROW
_WGMMA_COL = plgpu.Layout.WGMMA_COL
_WG_SPLAT = plgpu.Layout.WG_SPLAT


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
def _fwd(
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
    use_stable_softmax: bool,
    config: Config,
) -> tuple[Float[Array, "*B T H d"], Residuals | None]:
  """Flash attention with Mosaic GPU."""

  orig_q_shape = q.shape
  as_4d = lambda x: jax.lax.collapse(jax.lax.broadcast_to_rank(x, 4), 0, -3)
  q, k, v = map(as_4d, (q, k, v))

  batch_size, q_seq_len, num_q_heads, _ = q.shape
  _, kv_seq_len, num_kv_heads, head_dim_out = v.shape
  orig_head_dim_out = head_dim_out

  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by and {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  pad_head_dim = lambda x: shape_lib.pad_to_next_multiple_of(x, 64, -1)
  q, k, v = map(pad_head_dim, (q, k, v))
  head_dim = q.shape[-1]
  head_dim_out = v.shape[-1]

  max_stages = min(config.num_stages, kv_seq_len // config.block_kv)
  block_q_kv = block_q, block_kv = config.block_q, config.block_kv
  num_q_tiles, rem = divmod(q_seq_len, block_q * 2)
  if rem:
    raise NotImplementedError(
        f"{q_seq_len=} must be a multiple of {block_q * 2=}"
    )

  logits_shape = (batch_size, num_q_heads, q_seq_len, kv_seq_len)
  if bias is not None:
    bias = jnp.broadcast_to(as_4d(bias), logits_shape)

  if mask is not None:
    mask = as_4d(mask).astype(jnp.int8)

  # TODO: Avoid broadcast.
  bcast = lambda x: jnp.broadcast_to(x, (batch_size, x.shape[-2], q_seq_len))
  k_start = None if k_start is None else bcast(k_start)
  k_end = None if k_end is None else bcast(k_end)

  def kernel(
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
      *residual_refs,
      scoped,
  ):
    b_idx = lax.axis_index("batch")
    q_idx = lax.axis_index("q_tiles")
    h_idx = lax.axis_index("heads")
    wg_idx = lax.axis_index("wg")

    (
        (q_smems, k_smem, v_smem, o_smems, *residual_smems),
        (bias_smems, mask_smems),
        (q_barriers, k_barriers, v_barriers, bias_barriers, mask_barriers),
        (
            k_consumed_barriers,
            v_consumed_barriers,
            bias_consumed_barriers,
            mask_consumed_barriers,
        ),
        schedule_barrier,
    ) = scoped

    def perform_schedule_barrier():
      plgpu.barrier_arrive(schedule_barrier)
      plgpu.barrier_wait(schedule_barrier)

    min_kv_step = 0
    max_kv_step = kv_seq_len // block_kv

    if is_causal:
      max_kv_step = lax.min(max_kv_step, 2 * (q_idx + 1))
    if k_start_min_ref is not None:
      idx = (b_idx, 0 if (k_start_min_ref.shape[1] == 1) else h_idx, q_idx)
      k_start_min = plgpu.load(k_start_min_ref, idx, layout=_WG_SPLAT)
      min_kv_step = lax.max(min_kv_step, k_start_min)
    if k_end_max_ref is not None:
      idx = (b_idx, 0 if (k_end_max_ref.shape[1] == 1) else h_idx, q_idx)
      k_end_max = plgpu.load(k_end_max_ref, idx, layout=_WG_SPLAT)
      max_kv_step = lax.min(max_kv_step, k_end_max)

    if mask_ref is None:
      bcast_mask_q = bcast_mask_k = mask_b_idx = mask_h_idx = None
    else:
      bcast_dims = [d == 1 for d in mask_ref.shape]
      bcast_mask_b, bcast_mask_h, bcast_mask_q, bcast_mask_k = bcast_dims
      if bcast_mask_q and bcast_mask_k:
        raise NotImplementedError("Mask broadcast on both sequences.")

      mask_b_idx = 0 if bcast_mask_b else b_idx
      mask_h_idx = 0 if bcast_mask_h else h_idx

    @pl.when(wg_idx < 2)
    def _compute_wg():
      qs = block.ds(2 * q_idx + wg_idx, block_q)

      plgpu.set_max_registers(232, action="increase")
      q_smem, o_smem = q_smems.at[wg_idx], o_smems.at[wg_idx]
      residual_smem = [ref.at[wg_idx] for ref in residual_smems]

      q_barrier = q_barriers.at[wg_idx]
      plgpu.copy_gmem_to_smem(q_ref.at[b_idx, qs, h_idx], q_smem, q_barrier)

      l_i = plgpu.layout_cast(jnp.zeros((block_q,), jnp.float32), _WGMMA_ROW)
      if use_stable_softmax:
        m_i = plgpu.layout_cast(jnp.full_like(l_i, -jnp.inf), _WGMMA_ROW)
      else:
        m_i = 0.0
      acc = jnp.zeros((block_q, head_dim_out), jnp.float32)
      acc = plgpu.layout_cast(acc, _WGMMA)

      def load_k_range(ref):
        if ref is None:
          return None
        idx = (b_idx, 0 if (ref.shape[1] == 1) else h_idx, qs)
        return plgpu.load(ref, idx, layout=_WGMMA_ROW, optimized=False)

      k_start = load_k_range(k_start_ref)
      k_end = load_k_range(k_end_ref)

      plgpu.barrier_wait(q_barrier)
      @pl.when(max_kv_step > min_kv_step)
      def _():
        plgpu.barrier_wait(k_barriers.at[lax.rem(min_kv_step, max_stages)])

      pl.when(wg_idx == 1)(perform_schedule_barrier)

      def kv_loop(kv_step, carry, *, do_causal=False):
        acc, m_i, l_i = carry
        slot = lax.rem(kv_step, max_stages)

        def compute_qk(acc_ref):
          k_smem_T = plgpu.transpose_ref(k_smem.at[slot], (1, 0))  # pylint: disable=invalid-name
          plgpu.wgmma(acc_ref, q_smem, k_smem_T)
          plgpu.barrier_arrive(schedule_barrier)
          return acc_ref[...]

        s = pl.run_scoped(compute_qk, plgpu.ACC(block_q_kv, jnp.float32))
        plgpu.barrier_arrive(k_consumed_barriers.at[slot])
        plgpu.barrier_wait(schedule_barrier)
        s *= logits_scale

        if bias_ref is not None:
          bias_smem = bias_smems.at[slot, block.ds(wg_idx, block_q)]
          plgpu.barrier_wait(bias_barriers.at[slot])
          bias = bias_smem[...]
          plgpu.barrier_arrive(bias_consumed_barriers.at[slot])
          s += bias.astype(s.dtype)

        if logits_soft_cap is not None:
          s = logits_soft_cap * jnp.tanh(s / logits_soft_cap)

        if use_base2:
          s *= math.log2(math.e)

        def iota(d):
          return plgpu.broadcasted_iota(jnp.int32, s.shape, d, layout=_WGMMA)

        q_idxs = (2 * q_idx + wg_idx) * block_q + iota(0)
        k_idxs = kv_step * block_kv + iota(1)
        mask_value = float(jnp.finfo(jnp.float32).min)

        if do_causal:
          s = jnp.where(q_idxs >= k_idxs, s, mask_value)
        if k_start_ref is not None:
          k_start_ = lax.broadcast_in_dim(k_start, s.shape, [0])
          s = jnp.where(k_idxs >= k_start_, s, mask_value)
        if k_end_ref is not None:
          k_end_ = lax.broadcast_in_dim(k_end, s.shape, [0])
          s = jnp.where(k_idxs < k_end_, s, mask_value)

        if mask_ref is not None:
          if bcast_mask_q:
            idx = (mask_b_idx, mask_h_idx, 0, block.ds(kv_step, block_kv))
            mask = plgpu.load(mask_ref, idx, layout=_WGMMA_COL, optimized=False)
          elif bcast_mask_k:
            idx = (mask_b_idx, mask_h_idx, qs, 0)
            mask = plgpu.load(mask_ref, idx, layout=_WGMMA_ROW, optimized=False)
            mask = lax.broadcast_in_dim(mask, s.shape, [0])
          else:
            mask_smem = mask_smems.at[slot, block.ds(wg_idx, block_q)]
            plgpu.barrier_wait(mask_barriers.at[slot])
            mask = mask_smem[...]
            plgpu.barrier_arrive(mask_consumed_barriers.at[slot])
          s = jnp.where(mask, s, mask_value)

        exp = jnp.exp2 if use_base2 else jnp.exp
        if use_stable_softmax:
          m_ij = jnp.maximum(m_i, s.max(axis=1))
          alpha = exp(m_i - m_ij)
          m_i = m_ij
          p = exp(s - lax.broadcast_in_dim(m_ij, s.shape, [0]))
          acc *= lax.broadcast_in_dim(alpha, acc.shape, [0])
          l_i *= alpha
        else:
          p = exp(s)
        p_ = p.astype(q.dtype)

        # Can't fully explain why, but empirically the ordering here influences
        # the performance of the final kernel quite significantly.
        if p_sum_before_barriers := (head_dim <= 128):
          l_i += p.sum(axis=1)
          acc, l_i, m_i, p_ = lax.optimization_barrier((acc, l_i, m_i, p_))

        plgpu.barrier_arrive(schedule_barrier)
        plgpu.barrier_wait(v_barriers.at[slot])
        plgpu.barrier_wait(schedule_barrier)

        if not p_sum_before_barriers:
          l_i += p.sum(axis=1)

        def compute_pv(acc_ref):
          plgpu.wgmma(acc_ref, p_, v_smem.at[slot])
          wait_step = kv_step + 1

          @pl.when(wait_step < max_kv_step)
          def _():
            plgpu.barrier_wait(k_barriers.at[lax.rem(wait_step, max_stages)])

        acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
        plgpu.barrier_arrive(v_consumed_barriers.at[slot])
        return acc, m_i, l_i

      if kv_seq_len % block_kv:
        raise ValueError(f"{kv_seq_len=} must be a multiple of {block_kv=}")

      # If `is_causal`, the last iteration is split out, with masking enabled.
      hi = max_kv_step - (2 - wg_idx) * is_causal
      acc, m_i, l_i = lax.fori_loop(min_kv_step, hi, kv_loop, (acc, m_i, l_i))

      if is_causal:
        if block_q != block_kv:  # TODO: Fix this.
          raise NotImplementedError("Causal masking requires square blocks.")

        acc, m_i, l_i = kv_loop(hi, (acc, m_i, l_i), do_causal=True)

        @pl.when(wg_idx == 0)
        def _unblock_wg1():
          perform_schedule_barrier()
          perform_schedule_barrier()

      pl.when(wg_idx == 0)(perform_schedule_barrier)

      if return_residuals:
        if use_base2:
          m_i *= (1 / math.log2(math.e))
        residual_smem[0][...], residual_smem[1][...] = m_i, l_i
        plgpu.commit_smem()
        for smem, gmem in zip(residual_smem, residual_refs):
          plgpu.copy_smem_to_gmem(smem, gmem.at[b_idx, h_idx, qs])

      l_i += float(jnp.finfo(jnp.float32).tiny)

      if normalize_output:
        # TODO: Invert and multiply to avoid expensive divisions.
        acc /= lax.broadcast_in_dim(l_i, acc.shape, [0])

      o_smem[...] = acc.astype(q.dtype)  # pytype: disable=attribute-error
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(o_smem, out_ref.at[b_idx, qs, h_idx])
      plgpu.wait_smem_to_gmem(0)

    @pl.when(wg_idx == 2)
    def _memory_wg():
      plgpu.set_max_registers(40, action="decrease")
      kv_head = lax.div(h_idx, q_heads_per_kv_head)

      def kv_async_load(slot, ref, ks, smem, barriers):
        idx = (b_idx, ks, kv_head)
        plgpu.copy_gmem_to_smem(ref.at[idx], smem.at[slot], barriers.at[slot])

      def bias_async_load(slot, ref, ks, smem, barriers):
        idx = (b_idx, h_idx, block.ds(q_idx, 2 * block_q), ks)
        plgpu.copy_gmem_to_smem(ref.at[idx], smem.at[slot], barriers.at[slot])

      def mask_async_load(slot, ref, ks, smem, barriers):
        idx = (mask_b_idx, mask_h_idx, block.ds(q_idx, 2 * block_q), ks)
        plgpu.copy_gmem_to_smem(ref.at[idx], smem.at[slot], barriers.at[slot])

      async_mask = not (mask_ref is None or bcast_mask_q or bcast_mask_k)

      # TODO: Reorder loads to match order of consumption.
      for i in range(max_stages):

        @pl.when(i < (max_kv_step - min_kv_step))
        def _preload_kv_bias_mask():
          step = min_kv_step + i
          slot = lax.rem(step, max_stages)
          ks = block.ds(step, block_kv)
          kv_async_load(slot, k_ref, ks, k_smem, k_barriers)
          if bias_ref is not None:
            bias_async_load(slot, bias_ref, ks, bias_smems, bias_barriers)
          if async_mask:
            mask_async_load(slot, mask_ref, ks, mask_smems, mask_barriers)
          kv_async_load(slot, v_ref, ks, v_smem, v_barriers)

      @pl.loop(min_kv_step, max_kv_step - max_stages)
      def _kv_loop(kv_step):
        slot = lax.rem(kv_step, max_stages)
        ks = block.ds(kv_step + max_stages, block_kv)
        plgpu.barrier_wait(k_consumed_barriers.at[slot])
        kv_async_load(slot, k_ref, ks, k_smem, k_barriers)
        if bias_ref is not None:
          plgpu.barrier_wait(bias_consumed_barriers.at[slot])
          bias_async_load(slot, bias_ref, ks, bias_smems, bias_barriers)
        if async_mask:
          plgpu.barrier_wait(mask_consumed_barriers.at[slot])
          mask_async_load(slot, mask_ref, ks, mask_smems, mask_barriers)
        plgpu.barrier_wait(v_consumed_barriers.at[slot])
        kv_async_load(slot, v_ref, ks, v_smem, v_barriers)

  def entry(*refs):
    compute_wgs = 2

    def tiled_smem(shape, dtype):
      elem_bytes = jnp.dtype(dtype).itemsize
      swizzle_elems = min(shape[-1], 128 // elem_bytes)
      tiling = plgpu.TilingTransform((8, swizzle_elems))
      swizzle = plgpu.SwizzleTransform(swizzle_elems * elem_bytes)
      return plgpu.SMEM(shape, dtype, transforms=(tiling, swizzle))

    q_scratch = tiled_smem((compute_wgs, block_q, head_dim), q.dtype)
    k_scratch = tiled_smem((max_stages, block_kv, head_dim), k.dtype)
    v_scratch = tiled_smem((max_stages, block_kv, head_dim_out), v.dtype)
    o_scratch = tiled_smem((compute_wgs, block_q, head_dim_out), q.dtype)
    l_scratch = m_scratch = plgpu.SMEM((compute_wgs, block_q), jnp.float32)

    q_barriers = plgpu.Barrier(num_barriers=compute_wgs)
    kv_barriers = plgpu.Barrier(num_barriers=max_stages)
    kv_consumed_barriers = plgpu.Barrier(
        num_arrivals=compute_wgs, num_barriers=max_stages
    )
    schedule_barrier = plgpu.Barrier(num_arrivals=compute_wgs)

    if bias is None:
      bias_scratch = None
      bias_barriers = None
      bias_consumed_barriers = None
    else:
      bias_scratch = tiled_smem(
          (max_stages, compute_wgs * block_q, block_kv), bias.dtype
      )
      bias_barriers = plgpu.Barrier(num_barriers=max_stages)
      bias_consumed_barriers = plgpu.Barrier(
          num_arrivals=compute_wgs, num_barriers=max_stages
      )

    if mask is None or mask.shape[-2] == 1 or mask.shape[-1] == 1:
      mask_scratch = None
      mask_barriers = None
      mask_consumed_barriers = None
    else:
      mask_scratch = tiled_smem(
          (max_stages, compute_wgs * block_q, block_kv), jnp.int8
      )
      mask_barriers = plgpu.Barrier(num_barriers=max_stages)
      mask_consumed_barriers = plgpu.Barrier(
          num_arrivals=compute_wgs, num_barriers=max_stages
      )

    pl.run_scoped(
        lambda *args: kernel(*refs, scoped=args),
        plgpu.RefUnion(
            (q_scratch, k_scratch, v_scratch),
            (o_scratch, ((l_scratch, m_scratch) if return_residuals else ())),
        ),
        (bias_scratch, mask_scratch),
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

  def preprocess_k_range(reduction_fn, x):
    if x is None:
      return None
    reshape = lambda x, d: x.reshape((*x.shape[:-1], x.shape[-1] // d, d))
    # Pre-reduce the k_start/k_end to a single value per `2 * block_q` (as each
    # warpgroup processes a q-block and share the same k/v blocks).
    x = reduction_fn(reshape(x, 2 * block_q), axis=-1)
    return (x // block_kv) if reduction_fn == jnp.min else pl.cdiv(x, block_kv)

  k_start_min = preprocess_k_range(jnp.min, k_start)
  k_end_max = preprocess_k_range(jnp.max, k_end)

  out_shape = [jax.ShapeDtypeStruct((*q.shape[:-1], head_dim_out), q.dtype)]
  if return_residuals:
    residuals_shape = (batch_size, num_q_heads, q_seq_len)
    out_shape += [jax.ShapeDtypeStruct(residuals_shape, jnp.float32)] * 2

  out, *residuals = plgpu.kernel(
      entry,
      out_shape=out_shape,
      grid=(num_q_heads, num_q_tiles, batch_size),
      grid_names=("heads", "q_tiles", "batch"),
      num_threads=3,
      thread_name="wg",
      compiler_params=plgpu.CompilerParams(approx_math=True),
  )(q, k, v, bias, mask, k_start, k_end, k_start_min, k_end_max)

  out = out.reshape(*orig_q_shape[:-1], out.shape[-1])[..., :orig_head_dim_out]
  residuals = tuple(
      res.reshape(*orig_q_shape[:-3], num_q_heads, q_seq_len)
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
  # TODO: Fold is_causal into k_end only if k_end is not None.
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

  use_base2: bool = True
  use_stable_softmax: bool | type[base.AUTO] = base.AUTO

  def __post_init__(self):
    if self.vjp is None:
      vjp_ = vjp.PallasMosaicGpuFlashAttentionVjp(use_base2=self.use_base2)
      object.__setattr__(self, "vjp", vjp_)

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
      paging_info: PagingInfo | None,
      q_indices: Int[Array, "*#B #H T"] | None,
      k_indices: Int[Array, "*#B #H t"] | None,
      normalize_output: bool,
      return_residuals: bool,
      config: Config,
  ) -> tuple[Float[Array, "*B T H d"], Residuals | None]:
    """Performs attention, optionally returning softmax residuals."""
    if not mosaic_gpu.has_mosaic_gpu_support():
      raise NotImplementedError("Mosaic GPU not supported on this platform.")

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
    if paging_info is not None:
      raise NotImplementedError("Paged attention not supported.")

    # TODO: Support in-kernel dequantization.
    q, k, v = map(base.as_array, (q, k, v))
    # FIXME: We shouldn't silently downcast types.
    k = k.astype(q.dtype)
    v = v.astype(q.dtype)

    q_k_dot_precision, weights_v_dot_precision = precision
    if q_k_dot_precision not in _SUPPORTED_PRECISIONS:
      raise NotImplementedError(f"{q_k_dot_precision=} not supported")
    if weights_v_dot_precision not in _SUPPORTED_PRECISIONS:
      raise NotImplementedError(f" {weights_v_dot_precision=} not supported")

    mask, is_causal, k_start, k_end = _decompose_mask(mask, q, k)

    use_stable_softmax = self.use_stable_softmax
    if use_stable_softmax is base.AUTO:
      use_stable_softmax = base.needs_stable_softmax(
          logits_dtype, logits_soft_cap
      )

    return _fwd(
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
        use_stable_softmax=use_stable_softmax,
        config=config,
    )

  def _get_heuristics_config(self, ba: op.BoundArguments):
    del ba
    # This is a pretty good option that works for most cases.
    # TODO: for larger embed dimensions we can probably guess better.
    return Config(block_q=64, block_kv=64, num_stages=2)
