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
import functools
import math
from typing import Any, ClassVar, TypeAlias

import immutabledict
import jax
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
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
_WGMMA_ROW = plgpu.Layout.WGMMA.reduce(1)
_WGMMA_COL = plgpu.Layout.WGMMA.reduce(0)


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  block_q: pydantic.conint(multiple_of=64, gt=0) = 64
  block_kv: pydantic.conint(multiple_of=64, gt=0) = 64
  num_stages: pydantic.conint(gt=1) = 2


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
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
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

  if bias is not None:
    bias = as_4d(bias)

  if mask is not None:
    mask = as_4d(mask).astype(jnp.int8)

  def bcast_k_range(x):
    if x is None:
      return None
    x = jax.lax.collapse(jax.lax.broadcast_to_rank(x, 3), 0, -2)
    # TODO: Avoid broadcast in q-sequence dim.
    return jnp.broadcast_to(x, (*x.shape[:-1], q_seq_len))

  k_start, k_end = map(bcast_k_range, (k_start, k_end))

  def kernel(
      q_gmem,
      k_gmem,
      v_gmem,
      bias_gmem,
      mask_gmem,
      k_start_gmem,
      k_end_gmem,
      k_start_minmax_gmems,
      k_end_minmax_gmems,
      out_gmem,
      *residual_gmems,
      scoped,
  ):
    bi = lax.axis_index("batch")
    qi = lax.axis_index("q_tiles")
    hi = lax.axis_index("heads")
    wg = lax.axis_index("wg")

    (
        (q_smems, k_smems, o_smems, *residual_smems),
        v_smems,
        q_barriers,
        bias_smems,
        mask_smems,
        (k_barriers, k_consumed_barriers),
        (v_barriers, v_consumed_barriers),
        bias_barriers,
        (mask_barriers, mask_consumed_barriers),
        schedule_barrier,
    ) = scoped

    def perform_schedule_barrier():
      plgpu.barrier_arrive(schedule_barrier)
      plgpu.barrier_wait(schedule_barrier)

    def get_kv_ranges():
      lb = 0
      ub = kv_seq_len // block_kv

      if is_causal:
        q_max = (qi + 1) * (2 * block_q)
        ub = lax.min(ub, pl.cdiv(q_max, block_kv))

      def load_k_minmax(x):
        return x[0 if x.shape[0] == 1 else bi, 0 if x.shape[1] == 1 else hi, qi]

      if k_start_minmax_gmems is None:
        k_start_max = None
      else:
        k_start_min, k_start_max = map(load_k_minmax, k_start_minmax_gmems)
        lb = lax.max(lb, lax.div(k_start_min, block_kv))

      if k_end_minmax_gmems is None:
        k_end_min = None
      else:
        k_end_min, k_end_max = map(load_k_minmax, k_end_minmax_gmems)
        ub = lax.min(ub, pl.cdiv(k_end_max, block_kv))

      return lb, ub, k_start_max, k_end_min

    @pl.when(wg < 2)
    def _compute_wg():
      q_base = (2 * qi + wg) * block_q
      qs = pl.ds(q_base, block_q)

      plgpu.set_max_registers(232, action="increase")
      q_smem = q_smems.at[wg]
      q_barrier = q_barriers.at[wg]
      plgpu.copy_gmem_to_smem(q_gmem.at[bi, qs, hi], q_smem, q_barrier)

      l_i = plgpu.layout_cast(jnp.zeros((block_q,), jnp.float32), _WGMMA_ROW)
      if use_stable_softmax:
        m_i = plgpu.layout_cast(jnp.full_like(l_i, -jnp.inf), _WGMMA_ROW)
      else:
        m_i = plgpu.layout_cast(jnp.full_like(l_i, 0.0), _WGMMA_ROW)
      acc = jnp.zeros((block_q, head_dim_out), jnp.float32)
      acc = plgpu.layout_cast(acc, _WGMMA)

      def load_k_range(ref):
        if ref is None:
          return None
        b_idx_ = 0 if ref.shape[0] == 1 else bi
        h_idx_ = 0 if ref.shape[1] == 1 else hi
        idx = (b_idx_, h_idx_, qs)
        return plgpu.load(ref, idx, layout=_WGMMA_ROW, optimized=False)

      k_start = load_k_range(k_start_gmem)
      k_end = load_k_range(k_end_gmem)
      lb, ub, k_start_max, k_end_min = get_kv_ranges()

      plgpu.barrier_wait(q_barrier)

      @pl.when(ub > lb)
      def _():
        plgpu.barrier_wait(k_barriers.at[lax.rem(lb, max_stages)])

      pl.when(wg == 1)(perform_schedule_barrier)

      def loop_body(ki, carry, *, do_causal=False):
        acc, m_i, l_i = carry
        si = lax.rem(ki, max_stages)
        k_base = ki * block_kv

        def iota(d):
          return plgpu.broadcasted_iota(jnp.int32, block_q_kv, d, layout=_WGMMA)

        def load_bias_mask(gmem, smems, barriers):
          if gmem is None:
            return None
          bi_ = 0 if gmem.shape[0] == 1 else bi
          hi_ = 0 if gmem.shape[1] == 1 else hi
          if gmem.shape[-2] == 1:
            if gmem.shape[-1] == 1:
              raise NotImplementedError("Broadcast on both sequences.")
            idx = (bi_, hi_, 0, pl.ds(k_base, block_kv))
            x = plgpu.load(gmem, idx, layout=_WGMMA_COL, optimized=False)
            return lax.broadcast_in_dim(x, block_q_kv, [1])
          if gmem.shape[-1] == 1:
            idx = (bi_, hi_, qs, 0)
            x = plgpu.load(gmem, idx, layout=_WGMMA_ROW, optimized=False)
            return lax.broadcast_in_dim(x, block_q_kv, [0])
          plgpu.barrier_wait(barriers.at[si])
          return smems[si, block.ds(wg, block_q)]

        def compute_qk(acc_ref):
          k_smem_T = plgpu.transpose_ref(k_smems.at[si], (1, 0))  # pylint: disable=invalid-name
          plgpu.wgmma(acc_ref, q_smem, k_smem_T)
          bias = load_bias_mask(bias_gmem, bias_smems, bias_barriers)
          plgpu.barrier_arrive(schedule_barrier)
          mask = (q_base + iota(0) >= k_base + iota(1)) if do_causal else None
          return acc_ref[...], bias, mask

        acc_type = plgpu.ACC(block_q_kv, jnp.float32)
        s, bias, mask = pl.run_scoped(compute_qk, acc_type)
        plgpu.barrier_arrive(k_consumed_barriers.at[si])
        plgpu.barrier_wait(schedule_barrier)

        scale = logits_scale

        if bias is not None:
          s = s * scale + bias.astype(s.dtype)
          scale = 1.0

        if logits_soft_cap is not None:
          s = jnp.tanh(s * (scale / logits_soft_cap))
          scale = logits_soft_cap

        if use_base2:
          scale *= math.log2(math.e)

        s *= scale

        mask_value = float(jnp.finfo(jnp.float32).min)

        if mask is not None:
          s = jnp.where(mask, s, mask_value)

        def apply_k_start():
          k_start_ = lax.broadcast_in_dim(k_start, s.shape, [0])
          return jnp.where(k_base + iota(1) >= k_start_, s, mask_value)

        if k_start is not None:
          s = lax.cond(k_base < k_start_max, apply_k_start, lambda: s)

        def apply_k_end():
          k_end_ = lax.broadcast_in_dim(k_end, s.shape, [0])
          return jnp.where(k_base + iota(1) < k_end_, s, mask_value)

        if k_end is not None:
          s = lax.cond(k_base + block_kv > k_end_min, apply_k_end, lambda: s)

        mask = load_bias_mask(mask_gmem, mask_smems, mask_barriers)
        if mask is not None:
          if mask_consumed_barriers is not None:
            plgpu.barrier_arrive(mask_consumed_barriers.at[si])
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
        plgpu.barrier_wait(v_barriers.at[si])
        plgpu.barrier_wait(schedule_barrier)

        if not p_sum_before_barriers:
          l_i += p.sum(axis=1)

        def compute_pv(acc_ref):
          plgpu.wgmma(acc_ref, p_, v_smems.at[si])

          @pl.when(ki + 1 < ub)
          def _():
            plgpu.barrier_wait(k_barriers.at[lax.rem(ki + 1, max_stages)])

        acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
        plgpu.barrier_arrive(v_consumed_barriers.at[si])
        return acc, m_i, l_i

      if kv_seq_len % block_kv:
        raise ValueError(f"{kv_seq_len=} must be a multiple of {block_kv=}")

      if is_causal:
        causal_loop_body = functools.partial(loop_body, do_causal=True)
        ub_no_causal = lax.min(ub, lax.div(q_base, block_kv))
        carry = lax.fori_loop(lb, ub_no_causal, loop_body, (acc, m_i, l_i))
        # TODO: This cond should be redundant, but without it we
        # hit a weird compiler bug.
        acc, m_i, l_i = lax.cond(
            ub_no_causal < ub,
            lambda: lax.fori_loop(ub_no_causal, ub, causal_loop_body, carry),
            lambda: carry,
        )
      else:
        acc, m_i, l_i = lax.fori_loop(lb, ub, loop_body, (acc, m_i, l_i))

      pl.when(wg == 0)(perform_schedule_barrier)

      if return_residuals:
        m_smem, l_smem = (smems.at[wg] for smems in residual_smems)
        m_smem[...] = (m_i * (1 / math.log2(math.e))) if use_base2 else m_i
        l_smem[...] = l_i
        plgpu.commit_smem()
        m_gmem, l_gmem = residual_gmems
        plgpu.copy_smem_to_gmem(m_smem, m_gmem.at[bi, hi, qs])
        plgpu.copy_smem_to_gmem(l_smem, l_gmem.at[bi, hi, qs])

      l_i += float(jnp.finfo(jnp.float32).tiny)

      if normalize_output:
        # TODO: Use `reciprocal`?
        acc *= lax.broadcast_in_dim(1 / l_i, acc.shape, [0])

      o_smem = o_smems.at[wg]
      o_smem[...] = acc.astype(q.dtype)  # pytype: disable=attribute-error
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(o_smem, out_gmem.at[bi, qs, hi])
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    @pl.when(wg == 2)
    def _memory_wg():
      plgpu.set_max_registers(40, action="decrease")
      hi_kv = lax.div(hi, q_heads_per_kv_head)
      qs = block.ds(qi, 2 * block_q)

      if bias_smems is None:
        bias_gmem_ = None
      else:
        bias_bi = 0 if bias_gmem.shape[0] == 1 else bi
        bias_hi = 0 if bias_gmem.shape[1] == 1 else hi
        bias_gmem_ = bias_gmem.at[bias_bi, bias_hi, qs]

      if mask_smems is None:
        mask_gmem_ = None
      else:
        mask_bi = 0 if mask_gmem.shape[0] == 1 else bi
        mask_hi = 0 if mask_gmem.shape[1] == 1 else hi
        mask_gmem_ = mask_gmem.at[mask_bi, mask_hi, qs]

      def cp(gmem, smems, barriers, si):
        plgpu.copy_gmem_to_smem(gmem, smems.at[si], barriers.at[si])

      lb, ub, _, _ = get_kv_ranges()

      for i in range(max_stages):

        @pl.when(i < (ub - lb))
        def _preload_kv_bias_mask():
          ki = lb + i
          ks = block.ds(ki, block_kv)
          si = lax.rem(ki, max_stages)
          cp(k_gmem.at[bi, ks, hi_kv], k_smems, k_barriers, si)
          if bias_gmem_ is not None:
            cp(bias_gmem_.at[:, ks], bias_smems, bias_barriers, si)
          if mask_gmem_ is not None:
            cp(mask_gmem_.at[:, ks], mask_smems, mask_barriers, si)
          cp(v_gmem.at[bi, ks, hi_kv], v_smems, v_barriers, si)

      @pl.loop(lb, ub - max_stages)
      def _kv_loop(ki):
        si = lax.rem(ki, max_stages)
        ks = block.ds(ki + max_stages, block_kv)
        plgpu.barrier_wait(k_consumed_barriers.at[si])
        cp(k_gmem.at[bi, ks, hi_kv], k_smems, k_barriers, si)
        if bias_gmem_ is not None:
          cp(bias_gmem_.at[:, ks], bias_smems, bias_barriers, si)
        if mask_gmem_ is not None:
          plgpu.barrier_wait(mask_consumed_barriers.at[si])
          cp(mask_gmem_.at[:, ks], mask_smems, mask_barriers, si)
        plgpu.barrier_wait(v_consumed_barriers.at[si])
        cp(v_gmem.at[bi, ks, hi_kv], v_smems, v_barriers, si)

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
    kv_barriers = (
        plgpu.Barrier(num_barriers=max_stages),
        plgpu.Barrier(num_barriers=max_stages, num_arrivals=compute_wgs),
    )
    schedule_barrier = plgpu.Barrier(num_arrivals=compute_wgs)

    bias_mask_smem_shape = (max_stages, compute_wgs * block_q, block_kv)
    no_async_bias = bias is None or bias.shape[-2] == 1 or bias.shape[-1] == 1
    no_async_mask = mask is None or mask.shape[-2] == 1 or mask.shape[-1] == 1

    pl.run_scoped(
        lambda *args: kernel(*refs, scoped=args),
        plgpu.RefUnion(
            (q_scratch, k_scratch),
            (o_scratch, ((l_scratch, m_scratch) if return_residuals else ())),
        ),
        v_scratch,  # wg1 may still access v as wg0 writes to {o,l,m}_scratch.
        q_barriers,
        None if no_async_bias else tiled_smem(bias_mask_smem_shape, bias.dtype),
        None if no_async_mask else tiled_smem(bias_mask_smem_shape, jnp.int8),
        kv_barriers,
        kv_barriers,
        # bias doesn't need a consumed barrier as it is implied by k consumed.
        None if no_async_bias else kv_barriers[0],
        (None, None) if no_async_mask is None else kv_barriers,
        schedule_barrier,
        collective_axes="wg",
    )

  # Pre-reduce the k_start/k_end to a single value per `2 * block_q` (as compute
  # warpgroups share the same k/v blocks).
  if k_start is None:
    k_start_minmax = None
  else:
    k_start_ = shape_lib.einshape("...(qb)->...qb", b=2 * block_q)(k_start)
    k_start_minmax = (jnp.min(k_start_, -1), jnp.max(k_start_, -1))

  if k_end is None:
    k_end_minmax = None
  else:
    k_end_ = shape_lib.einshape("...(qb)->...qb", b=2 * block_q)(k_end)
    k_end_minmax = (jnp.min(k_end_, -1), jnp.max(k_end_, -1))

  out_shape = [jax.ShapeDtypeStruct((*q.shape[:-1], head_dim_out), q.dtype)]
  if return_residuals:
    residuals_shape = (batch_size, num_q_heads, q_seq_len)
    out_shape += [jax.ShapeDtypeStruct(residuals_shape, jnp.float32)] * 2

  out, *residuals = plgpu.kernel(
      entry,
      out_shape=out_shape,
      grid=(batch_size, num_q_heads, num_q_tiles),
      grid_names=("batch", "heads", "q_tiles"),
      num_threads=3,
      thread_name="wg",
      compiler_params=plgpu.CompilerParams(approx_math=True),
  )(q, k, v, bias, mask, k_start, k_end, k_start_minmax, k_end_minmax)

  out = out.reshape(*orig_q_shape[:-1], out.shape[-1])[..., :orig_head_dim_out]
  residuals = tuple(
      res.reshape(*orig_q_shape[:-3], num_q_heads, q_seq_len)
      for res in residuals
  )
  return out, (residuals if return_residuals else None)


Key: TypeAlias = immutabledict.immutabledict[str, Any]


def _decompose_mask(mask, q, k, q_indices, k_indices):
  """Decomposes `mask` into a mask array, `is_causal`, `k_start` and `k_end`."""
  if mask is None:
    return None, False, None, None

  is_causal = False
  k_start = None
  k_end = None

  if k_indices is None:
    mask, is_causal, k_start, k_end = mask.take("is_causal", "k_start", "k_end")

    # Fold `is_causal` into `k_end`. If `q_indices` is not `None`, then this is
    # necessary for correctness. Otherwise, it is a performance optimization.
    if is_causal and (q_indices is not None or k_end is not None):
      if q_indices is None:
        q_indices = jnp.arange(q.shape[-3])
      k_end_ = q_indices + 1
      k_end = k_end_ if k_end is None else jnp.minimum(k_end, k_end_)
      is_causal = False

    if k_start is not None:
      k_start = jax.lax.broadcast_to_rank(k_start, 2)
    if k_end is not None:
      k_end = jax.lax.broadcast_to_rank(k_end, 2)

  q_len_or_indices = q.shape[-3] if q_indices is None else q_indices
  k_len_or_indices = k.shape[-3] if k_indices is None else k_indices
  mask = mask.as_array(q_len_or_indices, k_len_or_indices)
  return mask, is_causal, k_start, k_end


_SUPPORTED_PRECISIONS = (
    lax.DotAlgorithmPreset.DEFAULT,
    lax.DotAlgorithmPreset.BF16_BF16_F32,
    lax.DotAlgorithmPreset.F16_F16_F32,
)


@dataclasses.dataclass(frozen=True)
class PallasMosaicGpuFlashAttention(base.DotProductAttention[Config, Key]):
  """Flash attention with Mosaic GPU."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
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
    if paging_info is not None:
      raise NotImplementedError("Paged attention not supported.")

    # TODO: Support in-kernel dequantization.
    q, k, v = map(base.as_array, (q, k, v))
    # TODO: Avoid silently downcasting types.
    k = k.astype(q.dtype)
    v = v.astype(q.dtype)

    q_k_dot_precision, weights_v_dot_precision = precision
    if q_k_dot_precision not in _SUPPORTED_PRECISIONS:
      raise NotImplementedError(f"{q_k_dot_precision=} not supported")
    if weights_v_dot_precision not in _SUPPORTED_PRECISIONS:
      raise NotImplementedError(f" {weights_v_dot_precision=} not supported")

    mask, is_causal, k_start, k_end = _decompose_mask(
        mask, q, k, q_indices, k_indices
    )

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
    q, k, v = ba.batched.args
    seq_len_k, _, head_dim = k.shape[-3:]
    head_dim_out = v.shape[-1]

    mask = ba.batched.kwargs["mask"]
    q_indices = ba.batched.kwargs["q_indices"]
    k_indices = ba.batched.kwargs["k_indices"]
    mask, *_ = jax.eval_shape(_decompose_mask, mask, q, k, q_indices, k_indices)

    def shared_mem_usage_bytes(block_q, block_kv, num_stages):
      bytes_per_stage = (
          block_kv * head_dim * jnp.finfo(k.dtype).bits // 8
          + block_kv * head_dim_out * jnp.finfo(v.dtype).bits // 8
      )
      if (bias := ba.kwargs["bias"]) is not None:
        bytes_per_stage += (
            2 * block_q * block_kv * jnp.finfo(bias.dtype).bits // 8
        )
      # FIXME: This is an overestimate for broadcast masks.
      if mask is not None:
        bytes_per_stage += 2 * block_q * block_kv
      return (
          2 * block_q * head_dim * jnp.finfo(q.dtype).bits // 8
          + num_stages * bytes_per_stage
          + 1000  # Add some extra for barriers.
      )

    if seq_len_k % 128 == 0 and shared_mem_usage_bytes(64, 128, 2) < 227 * 1024:
      return Config(block_q=64, block_kv=128, num_stages=2)

    # This is a pretty good option that works for most cases.
    return Config(block_q=64, block_kv=64, num_stages=2)
