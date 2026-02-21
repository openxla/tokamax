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
"""H100 Flash attention with Mosaic GPU."""

import functools
import math

import jax
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src import shape as shape_lib
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_mosaic_gpu_common as common
from tokamax._src.pallas import block


# pylint: disable=cell-var-from-loop

Residuals = base.Residuals

_MIN_SWIZZLE = 32
_WGMMA = plgpu.Layout.WGMMA
_WGMMA_ROW = plgpu.Layout.WGMMA.reduce(1)
_WGMMA_COL = plgpu.Layout.WGMMA.reduce(0)
_load_bcast = common.load_bcast


@pydantic.dataclasses.dataclass(
    frozen=True, kw_only=True, slots=True, config=dict(extra="forbid")
)  # pytype: disable=wrong-keyword-args
class Config(common.ConfigBase):
  """Configuration parameters for Pallas-Mosaic-GPU kernels on SM90 GPUs."""

  pass


def _estimate_shared_mem_usage_bytes(ba, block_q, block_kv, num_stages):
  """Estimates the shared memory usage in bytes for a given configuration."""
  q, k, v = ba.args
  tile_q = 2 * block_q

  # 32-bit floats are downcast to 16-bit before the kernel call.
  dtype_bits = jnp.finfo(jnp.bfloat16).bits
  m = 8 * _MIN_SWIZZLE // dtype_bits
  block_d = pl.cdiv(q.shape[-1], m) * m
  block_d_out = v.shape[-1]
  bytes_per_stage = block_kv * block_d_out * dtype_bits // 8
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

  # `q`/`k` and the outputs are in a union.
  q_k_elems = (tile_q + num_stages * block_kv) * block_d
  out_elems = tile_q * block_d_out
  return (
      (max(q_k_elems, out_elems) * dtype_bits // 8)
      + num_stages * bytes_per_stage
      + 2200  # Add some extra for barriers, etc. (slight overestimate).
  )


def get_heuristics_config(ba: op.BoundArguments) -> Config:
  if _estimate_shared_mem_usage_bytes(ba, 64, 128, 2) < 227 * 1024:
    return Config(block_q=64, block_kv=128, num_stages=2)

  # This is a pretty good option that works for most cases.
  return Config(block_q=64, block_kv=64, num_stages=2)


def get_autotuning_configs(ba: op.BoundArguments) -> set[Config]:
  """Returns a set of configs for autotuning flash attention on SM90 GPUs."""
  q, k, _ = ba.args
  block_qs = set([
      min(x, pl.next_power_of_2(q.shape[-3] // 2))
      for x in [64, 128, 256]
      if q.shape[-3] % (x * 2) == 0  # 2 * block_q must divide seq_len_q.
  ])
  block_kvs = set([
      min(x, pl.next_power_of_2(k.shape[-3]))
      for x in [64, 128, 256]
      if k.shape[-3] % x == 0  # block_kv must divide seq_len_kv.
  ])
  configs = set()
  for block_q in block_qs:
    for block_kv in block_kvs:
      for num_stages in [2, 3, 4]:
        configs.add(
            Config(block_q=block_q, block_kv=block_kv, num_stages=num_stages)
        )
  return configs


@jaxtyping.jaxtyped
def flash_attention_kernel(
    q: Float[Array, "T H D"],
    k: Float[Array, "t h D"],
    v: Float[Array, "t h d"],
    bias: Float[Array, "#H #T #t"] | None,
    mask: Bool[Array, "#H #T #t"] | None,
    k_start: Int[Array, "#H #T"] | None,
    k_end: Int[Array, "#H #T"] | None,
    *,
    is_causal: bool,
    logits_soft_cap: float | None,
    logits_scale: float,
    out_dtype: jnp.dtype,
    normalize_output: bool,
    return_residuals: bool,
    use_base2: bool,
    use_stable_softmax: bool,
    config: Config,
) -> tuple[Float[Array, "T H d"], Residuals | None]:
  """Flash attention with Mosaic GPU."""

  _, num_q_heads, _ = q.shape
  _, num_kv_heads, _ = v.shape

  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  # The sequence dimensions must be a multiple of 8.
  orig_q_seq_len, _, _ = q.shape
  pad_seq_len = lambda x: shape_lib.pad_to_next_multiple_of(x, 8, 0)
  q, k, v = map(pad_seq_len, (q, k, v))

  q_seq_len, num_q_heads, _ = q.shape
  kv_seq_len, _, orig_head_dim_out = v.shape

  # The contracting dimension for `wgmma` must be a multiple of the minimum
  # swizzle size (in number of elements).
  def pad_head_dim(x):
    m = 8 * _MIN_SWIZZLE // common.num_bits(x.dtype)
    return shape_lib.pad_to_next_multiple_of(x, m, -1)

  q, k, v = map(pad_head_dim, (q, k, v))
  head_dim = q.shape[-1]
  head_dim_out = v.shape[-1]

  block_q_kv = block_q, block_kv = config.block_q, config.block_kv
  max_stages = min(config.num_stages, pl.cdiv(kv_seq_len, block_kv))
  num_q_tiles = pl.cdiv(q_seq_len, block_q * 2)

  if mask is not None:
    mask = mask.astype(jnp.int8)

  as_2d = lambda x: None if x is None else jax.lax.broadcast_to_rank(x, 2)
  k_start, k_end = map(as_2d, (k_start, k_end))

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
    qi = lax.axis_index("q_tiles")
    hi = lax.axis_index("heads")
    wg = lax.axis_index("wg")

    (
        ((q_smems, k_smems), (o_smems, *residual_smems)),
        v_smems,
        q_barriers,
        bias_smems,
        mask_smems,
        (k_barriers, k_consumed_barriers),
        (v_barriers, v_consumed_barriers),
        bias_barriers,
        (mask_barriers, mask_consumed_barriers),
    ) = scoped

    at_wg = lambda x: x.at[wg]
    q_smem, q_barrier, o_smem = map(at_wg, (q_smems, q_barriers, o_smems))

    def get_kv_ranges():
      lb = 0
      ub = pl.cdiv(kv_seq_len, block_kv)

      if is_causal:
        q_max = (qi + 1) * (2 * block_q)
        ub = lax.min(ub, pl.cdiv(q_max, block_kv))

      load_k_minmax = lambda x: _load_bcast(x, (hi, qi), layout=None)

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
    def compute_wg():
      q_base = (2 * qi + wg) * block_q
      qs = pl.ds(q_base, block_q)

      plgpu.set_max_registers(232, action="increase")
      plgpu.copy_gmem_to_smem(q_gmem.at[qs, hi], q_smem, q_barrier)

      m_init_value = -jnp.inf if use_stable_softmax else 0.0
      l_i = plgpu.layout_cast(jnp.zeros((block_q,), jnp.float32), _WGMMA_ROW)
      m_i = plgpu.layout_cast(jnp.full_like(l_i, m_init_value), _WGMMA_ROW)
      acc = plgpu.layout_cast(jnp.zeros_like(o_smem, jnp.float32), _WGMMA)

      load_k_range = lambda r: _load_bcast(r, (hi, qs), layout=_WGMMA_ROW)
      k_start = None if k_start_gmem is None else load_k_range(k_start_gmem)
      k_end = None if k_end_gmem is None else load_k_range(k_end_gmem)
      lb, ub, k_start_max, k_end_min = get_kv_ranges()

      plgpu.barrier_wait(q_barrier)

      @pl.when(ub > lb)
      def _():
        plgpu.barrier_wait(k_barriers.at[lax.rem(lb, max_stages)])

      # MGPU uses the lower barrier IDs, so use barriers 8 and 9 for scheduling.
      schedule_barrier_arrive = functools.partial(
          common.bar_arrive, barrier_id=9 - wg, num_threads=256
      )
      schedule_barrier_arrive_and_wait = functools.partial(
          common.bar_sync, barrier_id=8 + wg, num_threads=256
      )

      pl.when(wg == 1)(schedule_barrier_arrive_and_wait)

      def loop_body(ki, carry, *, do_causal=False):
        acc, m_i, l_i = carry
        si = lax.rem(ki, max_stages)
        k_base = ki * block_kv
        ks = pl.ds(k_base, block_kv)

        def iota(d):
          return plgpu.broadcasted_iota(jnp.int32, block_q_kv, d, layout=_WGMMA)

        def compute_qk(acc):
          plgpu.wgmma(acc, q_smem, k_smems.at[si].T)
          schedule_barrier_arrive()
          if bias_gmem is None:
            bias = None
          elif bias_smems is None:
            bias = _load_bcast(bias_gmem, (hi, qs, ks), layout=_WGMMA)
          else:
            plgpu.barrier_wait(bias_barriers.at[si])
            bias = bias_smems[si, block.ds(wg, block_q)]
          mask = (q_base + iota(0) >= k_base + iota(1)) if do_causal else None
          return acc[...], bias, mask

        acc_type = plgpu.ACC(block_q_kv, jnp.float32)
        s, bias, mask = pl.run_scoped(compute_qk, acc_type)
        plgpu.barrier_arrive(k_consumed_barriers.at[si])

        scale = logits_scale

        if bias is not None:
          s, scale = s * scale + bias.astype(s.dtype), 1.0

        if logits_soft_cap is not None:
          s, scale = jnp.tanh(s * (scale / logits_soft_cap)), logits_soft_cap

        if use_base2:
          scale *= math.log2(math.e)

        # Defer scaling to the softmax computation below, if possible (allowing
        # FMA to be used).

        mask_value = float(jnp.finfo(jnp.float32).min)

        if mask is not None:
          s, scale = jnp.where(mask, s * scale, mask_value), 1.0

        if k_start is not None:

          def apply_k_start(k_start=k_start):
            k_start = lax.broadcast_in_dim(k_start, s.shape, [0])
            s_ = s * scale
            return jnp.where(k_base + iota(1) >= k_start, s_, mask_value), 1.0

          s, scale = lax.cond(
              k_base < k_start_max, apply_k_start, lambda: (s, scale)
          )

        if k_end is not None:

          def apply_k_end(k_end=k_end):
            k_end = lax.broadcast_in_dim(k_end, s.shape, [0])
            s_ = s * scale
            return jnp.where(k_base + iota(1) < k_end, s_, mask_value), 1.0

          s, scale = lax.cond(
              k_base + block_kv > k_end_min, apply_k_end, lambda: (s, scale)
          )

        if mask_gmem is not None:
          if mask_smems is None:
            mask = _load_bcast(mask_gmem, (hi, qs, ks), layout=_WGMMA)
          else:
            plgpu.barrier_wait(mask_barriers.at[si])
            if mask_smems.ndim == 2:
              mask = plgpu.load(mask_smems, si, layout=_WGMMA_COL)
              mask = lax.broadcast_in_dim(mask, s.shape, [1])
            else:
              mask = mask_smems[si, block.ds(wg, block_q)]
            plgpu.barrier_arrive(mask_consumed_barriers.at[si])
          s, scale = jnp.where(mask, s * scale, mask_value), 1.0

        exp = jnp.exp2 if use_base2 else jnp.exp
        if use_stable_softmax:
          m_ij = jnp.maximum(m_i, s.max(axis=1) * scale)
          alpha = exp(m_i - m_ij)
          m_i = m_ij
          p = exp(s * scale - lax.broadcast_in_dim(m_ij, s.shape, [0]))
          acc = jnp.where(
              lax.broadcast_in_dim(alpha != 1.0, acc.shape, [0]),
              acc * lax.broadcast_in_dim(alpha, acc.shape, [0]),
              acc,
          )
          l_i *= alpha
        else:
          p = exp(s * scale)
        p_ = p.astype(v.dtype)

        # Can't fully explain why, but empirically the ordering here influences
        # the performance of the final kernel quite significantly.
        if p_sum_before_barriers := (head_dim <= 128):
          l_i += p.sum(axis=1)
          acc, l_i, m_i, p_ = lax.optimization_barrier((acc, l_i, m_i, p_))

        plgpu.barrier_wait(v_barriers.at[si])
        schedule_barrier_arrive_and_wait()

        def compute_pv(refs):
          acc, l_i = refs
          plgpu.wgmma(acc, p_, v_smems.at[si])

          if not p_sum_before_barriers:
            l_i[...] += p.sum(axis=1)

          @pl.when(ki + 1 < ub)
          def _():
            plgpu.barrier_wait(k_barriers.at[lax.rem(ki + 1, max_stages)])

        acc, l_i = pl.run_state(compute_pv)((plgpu.ACC.init(acc), l_i))
        plgpu.barrier_arrive(v_consumed_barriers.at[si])
        return acc, m_i, l_i

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

      pl.when(wg == 0)(schedule_barrier_arrive)

      if return_residuals:
        m_smem, l_smem = map(at_wg, residual_smems)
        m_smem[...] = (m_i * (1 / math.log2(math.e))) if use_base2 else m_i
        l_smem[...] = l_i
        plgpu.commit_smem()
        m_gmem, l_gmem = residual_gmems
        plgpu.copy_smem_to_gmem(m_smem, m_gmem.at[hi, qs], commit_group=False)
        plgpu.copy_smem_to_gmem(l_smem, l_gmem.at[hi, qs], commit_group=False)

      l_i += float(jnp.finfo(jnp.float32).tiny)

      if normalize_output:
        # TODO: Use `reciprocal`?
        acc *= lax.broadcast_in_dim(1 / l_i, acc.shape, [0])

      o_smem[...] = acc.astype(o_smem.dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(o_smem, out_gmem.at[qs, hi])
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    @pl.when(wg == 2)
    def memory_wg():
      plgpu.set_max_registers(40, action="decrease")
      hi_kv = lax.div(hi, q_heads_per_kv_head)
      qs = block.ds(qi, 2 * block_q)

      if bias_smems is None:
        bias_gmem_ = None
      else:
        bias_gmem_ = bias_gmem.at[0 if bias_gmem.shape[0] == 1 else hi, qs]

      if mask_smems is None:
        mask_gmem_ = None
      else:
        mask_gmem_ = mask_gmem.at[
            0 if mask_gmem.shape[0] == 1 else hi,
            0 if mask_gmem.shape[1] == 1 else qs,
        ]

      def cp(gmem, smems, barriers, si):
        plgpu.copy_gmem_to_smem(gmem, smems.at[si], barriers.at[si])

      lb, ub, _, _ = get_kv_ranges()

      @pl.loop(lb, lax.min(lb + max_stages, ub))
      def prologue(ki):
        si = lax.rem(ki, max_stages)
        ks = block.ds(ki, block_kv)
        cp(k_gmem.at[ks, hi_kv], k_smems, k_barriers, si)
        if bias_gmem_ is not None:
          cp(bias_gmem_.at[:, ks], bias_smems, bias_barriers, si)
        if mask_gmem_ is not None:
          cp(mask_gmem_.at[..., ks], mask_smems, mask_barriers, si)
        cp(v_gmem.at[ks, hi_kv], v_smems, v_barriers, si)

      @pl.loop(lb, ub - max_stages)
      def kv_loop(ki):
        si = lax.rem(ki, max_stages)
        ks = block.ds(ki + max_stages, block_kv)
        plgpu.barrier_wait(k_consumed_barriers.at[si])
        cp(k_gmem.at[ks, hi_kv], k_smems, k_barriers, si)
        if bias_gmem_ is not None:
          cp(bias_gmem_.at[:, ks], bias_smems, bias_barriers, si)
        if mask_gmem_ is not None:
          plgpu.barrier_wait(mask_consumed_barriers.at[si])
          cp(mask_gmem_.at[..., ks], mask_smems, mask_barriers, si)
        plgpu.barrier_wait(v_consumed_barriers.at[si])
        cp(v_gmem.at[ks, hi_kv], v_smems, v_barriers, si)

  def entry(*refs):
    compute_wgs = 2

    def tiled_smem(shape, dtype, what=""):
      transforms = common.tile_swizzle_transforms(shape, dtype, what)
      return plgpu.SMEM(shape, dtype, transforms=transforms)

    q_scratch = tiled_smem((compute_wgs, block_q, head_dim), q.dtype, "q")
    k_scratch = tiled_smem((max_stages, block_kv, head_dim), k.dtype, "k")
    v_scratch = tiled_smem((max_stages, block_kv, head_dim_out), v.dtype, "v")
    o_scratch = tiled_smem((compute_wgs, block_q, head_dim_out), out_dtype, "o")
    l_scratch = m_scratch = plgpu.SMEM((compute_wgs, block_q), jnp.float32)

    q_barriers = plgpu.Barrier(num_barriers=compute_wgs)
    kv_barriers = (
        plgpu.Barrier(num_barriers=max_stages),
        plgpu.Barrier(num_barriers=max_stages, num_arrivals=compute_wgs),
    )

    bias_mask_smem_shape = (max_stages, compute_wgs * block_q, block_kv)
    # bias doesn't need a consumed barrier as it is implied by k consumed.
    if bias is not None and bias.shape[-2] != 1 and bias.shape[-1] != 1:
      bias_scratch = tiled_smem(bias_mask_smem_shape, bias.dtype, "bias")
      bias_barrier = kv_barriers[0]
    else:
      bias_scratch = bias_barrier = None

    mask_scratch = None
    if mask is not None and mask.shape[-1] != 1:
      if mask.shape[-2] == 1:
        if block_kv >= 128:  # Minimum transfer size is 128 bytes.
          mask_scratch = plgpu.SMEM((max_stages, block_kv), jnp.int8)
      else:
        mask_scratch = tiled_smem(bias_mask_smem_shape, jnp.int8, "mask")
    mask_barriers = (None, None) if mask_scratch is None else kv_barriers

    pl.run_scoped(
        lambda *args: kernel(*refs, scoped=args),
        plgpu.RefUnion(
            (q_scratch, k_scratch),
            (o_scratch, *((l_scratch, m_scratch) if return_residuals else ())),
        ),
        v_scratch,  # wg1 may still access v as wg0 writes to {o,l,m}_scratch.
        q_barriers,
        bias_scratch,
        mask_scratch,
        kv_barriers,
        kv_barriers,
        bias_barrier,
        mask_barriers,
        collective_axes="wg",
    )

  # Pre-reduce the k_start/k_end to a single value per `2 * block_q` (as compute
  # warpgroups share the same k/v blocks).
  if k_start is None:
    k_start_minmax = None
  elif k_start.shape[-1] == 1:
    k_start_minmax = (k_start, k_start)
  else:
    k_start_ = shape_lib.einshape("...(qb)->...qb", b=2 * block_q)(k_start)
    k_start_minmax = (jnp.min(k_start_, -1), jnp.max(k_start_, -1))

  if k_end is None:
    k_end_minmax = None
  elif k_end.shape[-1] == 1:
    k_end_minmax = (k_end, k_end)
  else:
    k_end_ = shape_lib.einshape("...(qb)->...qb", b=2 * block_q)(k_end)
    k_end_minmax = (jnp.min(k_end_, -1), jnp.max(k_end_, -1))

  out_shape = [jax.ShapeDtypeStruct((*q.shape[:-1], head_dim_out), out_dtype)]
  if return_residuals:
    residuals_shape = (num_q_heads, q_seq_len)
    out_shape += [jax.ShapeDtypeStruct(residuals_shape, jnp.float32)] * 2

  out, *residuals = plgpu.kernel(
      entry,
      out_shape=out_shape,
      grid=(num_q_heads, num_q_tiles),
      grid_names=("heads", "q_tiles"),
      num_threads=3,
      thread_name="wg",
      compiler_params=plgpu.CompilerParams(approx_math=True),
  )(q, k, v, bias, mask, k_start, k_end, k_start_minmax, k_end_minmax)
  if return_residuals:
    residuals = tuple(r[:, :orig_q_seq_len] for r in residuals)
  else:
    residuals = None
  return out[:orig_q_seq_len, :, :orig_head_dim_out], residuals
