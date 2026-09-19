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

import dataclasses
import functools
import math
from typing import Any, cast

import jax
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src import shape as shape_lib
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_mosaic_gpu_common as common
from tokamax._src.pallas import block


# pylint: disable=cell-var-from-loop

Residuals = base.Residuals

_WGMMA = plgpu.Layout.WGMMA
_WGMMA_ROW = plgpu.Layout.WGMMA.reduce(1)
_WGMMA_COL = plgpu.Layout.WGMMA.reduce(0)
_load_bcast = common.load_bcast
_COMPUTE_WGS = 2
_FORBID_EXTRA = pydantic.ConfigDict(extra="forbid")
_MAX_SMEM = 227 * 1024


@pydantic.dataclasses.dataclass(
    frozen=True, kw_only=True, slots=True, config=_FORBID_EXTRA
)
class Config(common.ConfigBase):
  """Configuration parameters for Pallas-Mosaic-GPU kernels on SM90 GPUs."""
  persistent: bool = False


def _pad_qkv(q, k, v):
  # The sequence dimensions must be a multiple of 8.
  k, v = map(lambda x: shape_lib.pad_to_next_multiple_of(x, 8, 0), (k, v))
  q, k, v = map(common.pad_head_dim_to_next_multiple_of_min_swizzle, (q, k, v))
  return q, k, v


def _get_scratch_types(
    q, k, v, bias, mask, out_dtype, return_residuals, config
) -> dict[str, Any]:
  """Returns the scratch types for the kernel."""
  q, k, v = jax.eval_shape(_pad_qkv, q, k, v)
  head_dim = q.shape[-1]
  head_dim_out = v.shape[-1]

  block_q = config.block_q
  block_kv = config.block_kv
  num_stages = config.num_stages

  epi_tile_q = 64
  epi_tile_d = 1024 // mgpu_lib.num_bits(out_dtype)
  assert block_q % epi_tile_q == 0
  if head_dim_out % epi_tile_d != 0:
    epi_tile_d = head_dim_out
  num_epi_slots = min(2, (block_q // epi_tile_q) * (head_dim_out // epi_tile_d))

  tiled_smem = mgpu_lib.tiled_swizzled_smem
  q_scratch = tiled_smem((_COMPUTE_WGS * block_q, head_dim), q.dtype, "q")
  k_scratch = tiled_smem((num_stages, block_kv, head_dim), k.dtype, "k")
  o_scratch = tiled_smem(
      (_COMPUTE_WGS, num_epi_slots, epi_tile_q, epi_tile_d), out_dtype, "o"
  )
  l_scratch = m_scratch = plgpu.SMEM((_COMPUTE_WGS, block_q), jnp.float32)
  k_bias_produced = plgpu.Barrier(num_barriers=num_stages)

  residuals_scratch = (l_scratch, m_scratch) if return_residuals else ()

  if config.persistent:
    qko_scratch = (q_scratch, k_scratch, o_scratch, *residuals_scratch)
  else:
    qko_scratch = plgpu.RefUnion(
        (q_scratch, k_scratch), (o_scratch, *residuals_scratch)
    )

  scratch = dict(
      # wg1 may still access v as wg0 writes to {o,l,m}_scratch.
      qko_smems=qko_scratch,
      v_smem=tiled_smem((num_stages, block_kv, head_dim_out), v.dtype, "v"),
      q_produced=plgpu.Barrier(),
      v_produced=plgpu.Barrier(num_barriers=num_stages),
  )

  # bias doesn't need a consumed barrier as it is implied by k consumed.
  if bias is not None and bias.shape[-2] != 1 and bias.shape[-1] != 1:
    shape = (num_stages, _COMPUTE_WGS * block_q, block_kv)
    scratch["bias_smem"] = tiled_smem(shape, bias.dtype, "bias")
    k_bias_produced = dataclasses.replace(
        k_bias_produced, num_arrivals=k_bias_produced.num_arrivals + 1
    )

  if mask is not None and mask.shape[-1] != 1:
    if mask.shape[-2] == 1:
      if block_kv >= 128:  # Minimum transfer size is 128 bytes.
        scratch["mask_smem"] = plgpu.SMEM((num_stages, block_kv), jnp.int8)
        k_bias_produced = dataclasses.replace(
            k_bias_produced, num_arrivals=k_bias_produced.num_arrivals + 1
        )
    else:
      shape = (num_stages, _COMPUTE_WGS * block_q, block_kv)
      scratch["mask_smem"] = tiled_smem(shape, jnp.int8, "mask")
      scratch["mask_produced"] = plgpu.Barrier(num_barriers=num_stages)

  scratch["k_bias_produced"] = k_bias_produced
  return scratch


def _estimate_shared_mem_usage_bytes(ba, config) -> int:
  """Estimates the shared memory usage in bytes for a given configuration."""
  q, k, v, bias, mask = common.eval_input_shapes(
      ba,
      fold_q_sequence_heads=config.fold_q_sequence_heads,
      split_k=config.split_k,
  )
  out_dtype = ba.args[0].dtype
  return_residuals = ba.kwargs["return_residuals"]
  scratch_types = _get_scratch_types(
      q, k, v, bias, mask, out_dtype, return_residuals, config
  )
  return mgpu_lib.estimate_smem_bytes(scratch_types)


def get_heuristics_config(
    ba: op.BoundArguments, fold_q_sequence_heads: bool
) -> Config:
  """Returns a heuristic configuration for flash attention on SM90 GPUs."""
  config = Config(
      block_q=64,
      block_kv=128,
      num_stages=2,
      persistent=not ba.kwargs["mask"].is_causal,
      fold_q_sequence_heads=fold_q_sequence_heads,
  )
  if _estimate_shared_mem_usage_bytes(ba, config) < _MAX_SMEM:
    return config

  config = dataclasses.replace(config, block_kv=64)
  if _estimate_shared_mem_usage_bytes(ba, config) < _MAX_SMEM:
    return config

  return dataclasses.replace(config, persistent=False)


def get_autotuning_configs(ba: op.BoundArguments) -> set[Config]:
  """Returns a set of configs for autotuning flash attention on SM90 GPUs."""
  q, k, _ = ba.args
  block_kvs = [x for x in [128, 256] if x <= pl.next_power_of_2(k.shape[-3])]

  configs = set()
  for block_q in [64, 128] if pl.next_power_of_2(q.shape[-3]) > 128 else [64]:
    for block_kv in [64] + block_kvs:
      for persistent in (True, False):
        for num_stages in (2, 3, 4):
          if not persistent and num_stages > pl.cdiv(k.shape[-3], block_kv):
            continue
          config = Config(
              block_q=block_q,
              block_kv=block_kv,
              num_stages=num_stages,
              persistent=persistent,
          )
          if _estimate_shared_mem_usage_bytes(ba, config) <= _MAX_SMEM:
            configs.add(config)
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
    use_stable_softmax: bool,
    rescale_threshold: float,
    config: Config,
) -> tuple[Float[Array, "T H d"], Residuals | None]:
  """Flash attention with Mosaic GPU."""

  _, num_q_heads, _ = q.shape
  _, num_kv_heads, orig_head_dim_out = v.shape

  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  q, k, v = _pad_qkv(q, k, v)
  q_seq_len, _, head_dim = q.shape
  kv_seq_len, _, head_dim_out = v.shape

  block_q = config.block_q
  block_kv = config.block_kv
  num_stages = config.num_stages
  num_q_tiles = pl.cdiv(q_seq_len, block_q * 2)

  if mask is not None:
    mask = mask.astype(jnp.int8)

  as_2d = lambda x: None if x is None else jax.lax.broadcast_to_rank(x, 2)
  k_start, k_end = map(as_2d, (k_start, k_end))

  def kernel(
      grid_loop,
      *,
      qko_smems,
      v_smem,
      bias_smem=None,
      mask_smem=None,
      q_produced,
      k_bias_produced,
      v_produced,
      mask_produced=None,
  ):
    wg = lax.axis_index("wg")

    if config.persistent:
      q_smem, k_smem, o_smem, *residual_smems = qko_smems
    else:
      (q_smem, k_smem), (o_smem, *residual_smems) = qko_smems

    if mask_produced is None:
      mask_produced = k_bias_produced

    @grid_loop(init_carry=0)
    def grid_loop_body(gmems, loop_info: plgpu.NDLoopInfo, carry):
      (
          q_gmem,
          k_gmem,
          v_gmem,
          bias_gmem,
          mask_gmem,
          k_start_gmem,
          k_end_gmem,
          k_start_minmax_gmems,
          k_end_minmax_gmems,
          o_gmem,
          *residual_gmems,
      ) = gmems
      hi, qi = loop_info.index
      if is_causal:
        qi = num_q_tiles - 1 - qi
      prev_iters = carry

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

      # MGPU uses the lower barrier IDs.
      schedule_barrier = 4
      q_consumed_barrier = schedule_barrier + 2
      k_consumed_barrier = q_consumed_barrier + 1
      v_consumed_barrier = k_consumed_barrier + num_stages
      mask_consumed_barrier = v_consumed_barrier + num_stages
      if mask_smem is None:
        assert v_consumed_barrier + num_stages <= 16
      else:
        assert mask_consumed_barrier + num_stages <= 16

      schedule_barrier_arrive = functools.partial(
          mgpu_lib.bar_arrive, schedule_barrier + 1 - wg, num_threads=256
      )
      schedule_barrier_arrive_and_wait = functools.partial(
          mgpu_lib.bar_sync, schedule_barrier + wg, num_threads=256
      )

      def iota(d):
        return plgpu.broadcasted_iota(
            jnp.int32, (block_q, block_kv), d, layout=_WGMMA
        )

      def compute_wg():
        plgpu.set_max_registers(232, action="increase")

        q_base = (2 * qi + wg) * block_q
        qs = cast(pl.Slice, pl.ds(q_base, block_q))

        @pl.when(loop_info.local_index == 0)
        def prologue():
          @pl.when(wg == 0)
          def load_q():
            qs = block.ds(qi, 2 * block_q)
            plgpu.copy_gmem_to_smem(q_gmem.at[qs, hi], q_smem, q_produced)

          for si in range(num_stages):
            mgpu_lib.bar_arrive(k_consumed_barrier + si, num_threads=288)
            if mask_smem is not None:
              mgpu_lib.bar_arrive(mask_consumed_barrier + si, num_threads=288)
            mgpu_lib.bar_arrive(v_consumed_barrier + si, num_threads=288)

        m_init_value = -jnp.inf if use_stable_softmax else 0.0
        l_i = plgpu.layout_cast(jnp.zeros((block_q,), jnp.float32), _WGMMA_ROW)
        m_i = plgpu.layout_cast(jnp.full_like(l_i, m_init_value), _WGMMA_ROW)
        acc = jnp.zeros((block_q, head_dim_out), jnp.float32)
        acc = plgpu.layout_cast(acc, _WGMMA)

        load_k_range = lambda r: _load_bcast(r, (hi, qs), layout=_WGMMA_ROW)
        k_start = None if k_start_gmem is None else load_k_range(k_start_gmem)
        k_end = None if k_end_gmem is None else load_k_range(k_end_gmem)
        lb, ub, k_start_max, k_end_min = get_kv_ranges()

        def await_k():
          @pl.when(ub > lb)
          def _():
            si = lax.rem(prev_iters, num_stages)
            plgpu.barrier_wait(k_bias_produced.at[si])

        # If there is a k-range, we expect the first `k` load to be slower than
        # the `q`, as the `k` load isn't issued until the loop bounds are known.
        # Subsequent `k` loads should be faster, as the memory WG runs ahead.
        has_k_range = k_start is not None or k_end is not None
        lax.cond(
            has_k_range and (loop_info.local_index == 0),
            lambda: (plgpu.barrier_wait(q_produced), await_k()),
            lambda: (await_k(), plgpu.barrier_wait(q_produced)),
        )

        pl.when(wg == 1)(schedule_barrier_arrive_and_wait)

        def kv_loop(ki, carry, *, do_causal=False):
          acc, m_scale, m_i, l_i = carry
          si = lax.rem(prev_iters + ki - lb, num_stages)
          k_base = ki * block_kv
          ks = cast(pl.Slice, pl.ds(k_base, block_kv))

          def compute_qk(acc):
            plgpu.wgmma(acc, q_smem.at[block.ds(wg, block_q)], k_smem.at[si].T)
            schedule_barrier_arrive()
            if bias_gmem is None:
              bias = None
            elif bias_smem is None:
              bias = _load_bcast(bias_gmem, (hi, qs, ks), layout=_WGMMA)
            else:
              bias = bias_smem[si, block.ds(wg, block_q)]
            mask = (q_base + iota(0) >= k_base + iota(1)) if do_causal else None
            return acc[...], bias, mask

          acc_type = plgpu.ACC((block_q, block_kv), jnp.float32)
          s, bias, mask = pl.run_scoped(compute_qk, acc_type)
          mgpu_lib.bar_arrive(k_consumed_barrier + si, num_threads=288)

          scale = logits_scale

          if bias is not None:
            s, scale = s * scale + bias.astype(s.dtype), 1.0

          if logits_soft_cap is not None:
            s, scale = jnp.tanh(s * (scale / logits_soft_cap)), logits_soft_cap

          if mask is not None:
            s = jnp.where(mask, s, -jnp.inf)

          bcast = lambda x, tgt=s: lax.broadcast_in_dim(x, tgt.shape, [0])

          if k_start is not None:

            def apply_k_start():
              return jnp.where(k_base + iota(1) >= bcast(k_start), s, -jnp.inf)

            s = lax.cond(k_base < k_start_max, apply_k_start, lambda: s)

          if k_end is not None:

            def apply_k_end():
              return jnp.where(k_base + iota(1) < bcast(k_end), s, -jnp.inf)

            s = lax.cond(k_base + block_kv > k_end_min, apply_k_end, lambda: s)

          if mask_gmem is not None:
            if mask_smem is None:
              mask = _load_bcast(mask_gmem, (hi, qs, ks), layout=_WGMMA)
            else:
              if mask_produced is not k_bias_produced:
                plgpu.barrier_wait(mask_produced.at[si])
              if mask_smem.ndim == 2:
                mask = plgpu.load(mask_smem.at[si], layout=_WGMMA_COL)
                mask = lax.broadcast_in_dim(mask, s.shape, [1])
              else:
                mask = mask_smem[si, block.ds(wg, block_q)]
              mgpu_lib.bar_arrive(mask_consumed_barrier + si, num_threads=288)
            s = jnp.where(mask, s, -jnp.inf)

          scale *= math.log2(math.e)
          if use_stable_softmax:
            m_i = jnp.maximum(m_i, s.max(axis=1) * scale)
            m_valid = (mask is None and not has_k_range) | (m_i != -jnp.inf)
            alpha = jnp.where(m_valid, jnp.exp2(m_scale - m_i), 1.0)
            threshold_is_1 = rescale_threshold == 1.0
            needs_rescale = alpha < rescale_threshold
            m_scale = jnp.where(needs_rescale | threshold_is_1, m_i, m_scale)
            p = jnp.exp2(s * scale - bcast(jnp.where(m_valid, m_scale, 0.0)))
            acc = jnp.where(
                bcast(needs_rescale, acc), acc * bcast(alpha, acc), acc
            )
            l_i = jnp.where(needs_rescale | threshold_is_1, l_i * alpha, l_i)
          else:
            p = jnp.exp2(s * scale)
          p_ = p.astype(v.dtype)

          # Can't fully explain why, but empirically the ordering here
          # influences the performance of the final kernel quite significantly.
          if p_sum_before_barriers := (head_dim <= 128):
            l_i += p.sum(axis=1)
            acc, p_ = lax.optimization_barrier((acc, p_))
            l_i, m_i, m_scale = lax.optimization_barrier((l_i, m_i, m_scale))

          plgpu.barrier_wait(v_produced.at[si])
          schedule_barrier_arrive_and_wait()

          def compute_pv(refs):
            acc, l_i = refs
            plgpu.wgmma(acc, p_, v_smem.at[si])

            if not p_sum_before_barriers:
              l_i[...] += p.sum(axis=1)

            @pl.when(ki + 1 < ub)
            def _():
              si_next = lax.rem(prev_iters + ki + 1 - lb, num_stages)
              plgpu.barrier_wait(k_bias_produced.at[si_next])

          acc, l_i = pl.run_state(compute_pv)((plgpu.ACC.init(acc), l_i))
          mgpu_lib.bar_arrive(v_consumed_barrier + si, num_threads=288)
          return acc, m_scale, m_i, l_i

        carry = (acc, m_i, m_i, l_i)

        if is_causal:
          causal_loop_body = functools.partial(kv_loop, do_causal=True)
          ub_no_causal = lax.min(ub, lax.div(q_base, block_kv))
          carry = lax.fori_loop(lb, ub_no_causal, kv_loop, carry)
          # TODO: This cond should be redundant, but without it we
          # hit a weird compiler bug.
          acc, m_scale, m_i, l_i = lax.cond(
              ub_no_causal < ub,
              lambda: lax.fori_loop(ub_no_causal, ub, causal_loop_body, carry),
              lambda: carry,
          )
        else:
          acc, m_scale, m_i, l_i = lax.fori_loop(lb, ub, kv_loop, carry)

        @pl.when(wg == 0)
        def unblock_wg1_and_q_load():
          if config.persistent:
            # We must sync here to ensure that WG1 is finished with `q_smem`.
            mgpu_lib.bar_sync(schedule_barrier + 1, num_threads=256)
            mgpu_lib.bar_arrive(q_consumed_barrier, num_threads=160)
          else:
            schedule_barrier_arrive()

        m_valid = (mask is None and not has_k_range) | (m_i != -jnp.inf)
        alpha = jnp.where(m_valid, jnp.exp2(m_scale - m_i), 1.0)

        if return_residuals:
          m_smem, l_smem = residual_smems
          m_smem[wg] = m_i * (1 / math.log2(math.e))
          if use_stable_softmax and rescale_threshold != 1.0:
            l_smem[wg] = l_i * alpha
          else:
            l_smem[wg] = l_i
          plgpu.commit_smem()
          m_gmem, l_gmem = residual_gmems
          plgpu.copy_smem_to_gmem(
              m_smem.at[wg], m_gmem.at[hi, qs], commit_group=False
          )
          plgpu.copy_smem_to_gmem(
              l_smem.at[wg], l_gmem.at[hi, qs], commit_group=False
          )

        if normalize_output:
          l_i += float(jnp.finfo(jnp.float32).tiny)
          acc *= lax.broadcast_in_dim(1 / l_i, acc.shape, [0])
        elif use_stable_softmax and rescale_threshold != 1.0:
          acc *= lax.broadcast_in_dim(alpha, acc.shape, [0])

        _, num_epi_slots, epi_tile_q, epi_tile_d = o_smem.shape
        o_gmem_ = o_gmem.at[qs, hi]
        for qj in range(block_q // epi_tile_q):
          for dj in range(head_dim_out // epi_tile_d):
            si = lax.rem(qj * (head_dim_out // epi_tile_d) + dj, num_epi_slots)
            epi_qs = slice(qj * epi_tile_q, (qj + 1) * epi_tile_q)
            epi_ds = slice(dj * epi_tile_d, (dj + 1) * epi_tile_d)
            plgpu.wait_smem_to_gmem(num_epi_slots - 1, wait_read_only=True)
            o_smem[wg, si] = acc[epi_qs, epi_ds].astype(o_smem.dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(
                o_smem.at[wg, si], o_gmem_.at[epi_qs, epi_ds]
            )
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        return lb, ub

      def memory_wg():
        plgpu.set_max_registers(40, action="decrease")
        hi_kv = lax.div(hi, q_heads_per_kv_head)
        qs = block.ds(qi, 2 * block_q)

        if bias_smem is None:
          bias_gmem_ = None
        else:
          bias_gmem_ = bias_gmem.at[0 if bias_gmem.shape[0] == 1 else hi]

        if mask_smem is None:
          mask_gmem_ = None
        else:
          mask_gmem_ = mask_gmem.at[
              0 if mask_gmem.shape[0] == 1 else hi,
              0 if mask_gmem.shape[1] == 1 else qs,
          ]

        lb, ub, _, _ = get_kv_ranges()

        @plgpu.warp_map
        def per_warp(warp_id):

          @pl.when(warp_id == 0)
          def tma_k_v_bias_mask_load_warp():

            def cp(gmem, smem, barrier, si):
              plgpu.copy_gmem_to_smem(gmem, smem.at[si], barrier.at[si])

            @pl.loop(lb, ub)
            def kv_loop(ki):
              si = lax.rem(prev_iters + ki - lb, num_stages)
              ks = block.ds(ki, block_kv)
              mgpu_lib.bar_sync(k_consumed_barrier + si, num_threads=288)
              cp(k_gmem.at[ks, hi_kv], k_smem, k_bias_produced, si)
              if bias_gmem_ is not None:
                assert bias_smem is not None
                mgpu_lib.fence_async_shared_cta()
                cp(bias_gmem_.at[qs, ks], bias_smem, k_bias_produced, si)
              if mask_gmem_ is not None:
                assert mask_smem is not None
                mgpu_lib.bar_sync(mask_consumed_barrier + si, num_threads=288)
                mgpu_lib.fence_async_shared_cta()
                cp(mask_gmem_.at[..., ks], mask_smem, mask_produced, si)
              mgpu_lib.bar_sync(v_consumed_barrier + si, num_threads=288)
              cp(v_gmem.at[ks, hi_kv], v_smem, v_produced, si)

          if config.persistent:

            @pl.when((warp_id == 1) & (loop_info.local_index > 0))
            def tma_q_load_warp():
              plgpu.async_prefetch(q_gmem.at[qs, hi])
              mgpu_lib.bar_sync(q_consumed_barrier, num_threads=160)
              plgpu.copy_gmem_to_smem(q_gmem.at[qs, hi], q_smem, q_produced)

        return lb, ub

      lb, ub = lax.cond(wg < 2, compute_wg, memory_wg)
      return prev_iters + lax.max(0, ub - lb)

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
    residuals_shape = (num_q_heads, num_q_tiles * 2 * block_q)
    out_shape += [jax.ShapeDtypeStruct(residuals_shape, jnp.float32)] * 2

  scratch_types = _get_scratch_types(
      q, k, v, bias, mask, out_dtype, return_residuals, config
  )

  if config.persistent:
    maybe_persistent_kernel = mgpu_lib.static_scheduling_persistent_kernel
  else:
    maybe_persistent_kernel = mgpu_lib.not_persistent_grid_loop_kernel

  out, *residuals = maybe_persistent_kernel(
      kernel,
      out_type=out_shape,
      scratch_types=scratch_types,
      grid=(num_q_heads, num_q_tiles),
      grid_names=("heads", "q_tiles"),
      num_threads=_COMPUTE_WGS + 1,
      thread_name="wg",
      compiler_params=plgpu.CompilerParams(
          approx_math=True,
          unsafe_no_auto_barriers=True,
          reduction_scratch_bytes=0,
      ),
      kernel_name="flash_attention_sm90",
  )(q, k, v, bias, mask, k_start, k_end, k_start_minmax, k_end_minmax)

  residuals = tuple(res[..., :q_seq_len] for res in residuals)
  return (out[..., :orig_head_dim_out], residuals if residuals else None)
