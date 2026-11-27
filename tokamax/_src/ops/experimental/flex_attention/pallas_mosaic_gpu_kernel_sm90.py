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
"""H100 FlexAttention with Mosaic GPU."""

import dataclasses
import functools
import math
from typing import Annotated, Any, cast

import jax
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src import quantization
from tokamax._src import shape as shape_lib
from tokamax._src.ops import op
from tokamax._src.ops.attention import pallas_mosaic_gpu_common as attn_common
from tokamax._src.ops.experimental.flex_attention import base
from tokamax._src.ops.experimental.flex_attention import pallas_mosaic_gpu_common as common

MaskMod = base.MaskMod
Residuals = base.Residuals
ScoreMod = base.ScoreMod

_WGMMA = plgpu.Layout.WGMMA
_WGMMA_ROW = plgpu.Layout.WGMMA.reduce(1)
_COMPUTE_WGS = 2
_FORBID_EXTRA = pydantic.ConfigDict(extra="forbid")
_MAX_SMEM = 227 * 1024


@pydantic.dataclasses.dataclass(
    frozen=True, kw_only=True, slots=True, config=_FORBID_EXTRA
)
class Config:
  """Configuration parameters for Pallas-Mosaic-GPU kernels on SM90 GPUs.

  Attributes:
    block_q: Block size along Q sequence length.
    block_kv: Block size along K/V sequence length.
    num_stages: Number of TMA stages for loading KV.
    persistent: Whether to use a persistent kernel.
  """

  block_q: Annotated[int, pydantic.Field(multiple_of=64, gt=0)] = 64
  block_kv: Annotated[int, pydantic.Field(multiple_of=64, gt=0)] = 64
  num_stages: pydantic.PositiveInt = 2
  persistent: bool = False


def _bcast_rows(x: jax.Array, shape: tuple[int, ...]) -> jax.Array:
  return lax.broadcast_in_dim(x, shape, [0])


def _pad_qkv(q, k, v, block_kv: int):
  # The Q sequence dimension must be a multiple of 8; K/V padded to block_kv.
  q = shape_lib.pad_to_next_multiple_of(q, 8, -3)
  k = shape_lib.pad_to_next_multiple_of(k, block_kv, -3)
  v = shape_lib.pad_to_next_multiple_of(v, block_kv, -3)
  pad_head_dim = attn_common.pad_head_dim_to_next_multiple_of_min_swizzle
  return tuple(map(pad_head_dim, (q, k, v)))


def _get_scratch_types(
    q, k, v, out_dtype, return_residuals: bool, config: Config
) -> dict[str, Any]:
  """Returns the scratch types for the kernel."""
  q, k, v = jax.eval_shape(
      functools.partial(_pad_qkv, block_kv=config.block_kv), q, k, v
  )
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

  residuals_scratch = (m_scratch, l_scratch) if return_residuals else ()

  if config.persistent:
    qko_scratch = (q_scratch, k_scratch, o_scratch, *residuals_scratch)
  else:
    qko_scratch = plgpu.RefUnion(
        (q_scratch, k_scratch), (o_scratch, *residuals_scratch)
    )

  return dict(
      qko_smems=qko_scratch,
      v_smem=tiled_smem((num_stages, block_kv, head_dim_out), v.dtype, "v"),
      q_produced=plgpu.Barrier(),
      k_produced=plgpu.Barrier(num_barriers=num_stages),
      v_produced=plgpu.Barrier(num_barriers=num_stages),
  )


def _estimate_shared_mem_usage_bytes(
    ba: op.BoundArguments, config: Config
) -> int:
  """Estimates the shared memory usage in bytes for a given configuration."""
  q, k, v = jax.eval_shape(
      functools.partial(common.prepare_qkv, precision=ba.kwargs["precision"]),
      *ba.args,
  )
  out_dtype = jax.eval_shape(quantization.as_array, ba.args[0]).dtype
  return_residuals = ba.kwargs["return_residuals"]
  scratch_types = _get_scratch_types(
      q, k, v, out_dtype, return_residuals, config
  )
  return mgpu_lib.estimate_smem_bytes(scratch_types)


def get_heuristics_config(ba: op.BoundArguments) -> Config:
  """Returns a heuristic configuration for FlexAttention on SM90 GPUs."""
  config = Config(
      block_q=64,
      block_kv=128,
      num_stages=2,
      persistent=True,
  )
  if _estimate_shared_mem_usage_bytes(ba, config) < _MAX_SMEM:
    return config

  config = dataclasses.replace(config, block_kv=64)
  if _estimate_shared_mem_usage_bytes(ba, config) < _MAX_SMEM:
    return config

  return dataclasses.replace(config, persistent=False)


def get_autotuning_configs(ba: op.BoundArguments) -> set[Config]:
  """Returns a set of configs for autotuning FlexAttention on SM90 GPUs."""
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
def flex_attention_kernel(
    q: Float[Array, "*B T H D"],
    k: Float[Array, "*B t h D"],
    v: Float[Array, "*B t h d"],
    *,
    score_mod: ScoreMod | None,
    mask_mod: MaskMod | None,
    dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
    dropout_rate: float,
    out_dtype: jnp.dtype,
    normalize_output: bool,
    return_residuals: bool,
    use_base2: bool,
    use_stable_softmax: bool,
    rescale_threshold: float,
    config: Config,
) -> tuple[Float[Array, "*B T H d"], Residuals | None]:
  """FlexAttention with Mosaic GPU on SM90."""
  *batch, orig_q_seq_len, num_q_heads, _ = q.shape
  *_, orig_kv_seq_len, num_kv_heads, orig_head_dim_out = v.shape

  if k.shape[:-3] != q.shape[:-3] or v.shape[:-3] != q.shape[:-3]:
    raise ValueError("`k` and `v` batch sizes must be the same as `q`.")
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  block_q = config.block_q
  block_kv = config.block_kv
  num_stages = config.num_stages

  scores_shape = (*batch, num_q_heads, orig_q_seq_len, orig_kv_seq_len)
  score_mod_fn, score_mod_values, score_mod_specs = common.tile_score_mod(
      score_mod, scores_shape, block_q, block_kv
  )
  mask_mod_fn, mask_mod_values, mask_mod_specs = common.tile_mask_mod(
      mask_mod, scores_shape, block_q, block_kv
  )
  score_mod_values = common.pad_mod_values(
      score_mod_values, score_mod_specs, 2 * block_q, block_kv
  )
  mask_mod_values = common.pad_mod_values(
      mask_mod_values, mask_mod_specs, 2 * block_q, block_kv
  )

  q, k, v = _pad_qkv(q, k, v, block_kv)
  *_, q_seq_len, _, head_dim = q.shape
  *_, kv_seq_len, _, head_dim_out = v.shape
  num_q_tiles = pl.cdiv(q_seq_len, block_q * 2)
  use_mask_k = orig_kv_seq_len % block_kv != 0
  has_mask = (
      (score_mod_fn is not None) or (mask_mod_fn is not None) or use_mask_k
  )
  lazy_rescale = use_stable_softmax and rescale_threshold != 1.0
  scale = math.log2(math.e) if use_base2 else 1.0
  exp = jnp.exp2 if use_base2 else jnp.exp

  def where_m_valid(m_i, x, fallback):
    return jnp.where(m_i != -jnp.inf, x, fallback) if has_mask else x

  if dropout_mask is not None:
    dropout_mask = jax.lax.broadcast_to_rank(dropout_mask, len(scores_shape))
    for axis, m in ((-2, 2 * block_q), (-1, block_kv)):
      if dropout_mask.shape[axis] > 1:
        dropout_mask = shape_lib.pad_to_next_multiple_of(dropout_mask, m, axis)
    dropout_mask = dropout_mask.astype(jnp.int8)

  def kernel(
      grid_loop,
      *,
      qko_smems,
      v_smem,
      q_produced,
      k_produced,
      v_produced,
  ):
    wg = lax.axis_index("wg")

    if config.persistent:
      q_smem, k_smem, o_smem, *residual_smems = qko_smems
    else:
      (q_smem, k_smem), (o_smem, *residual_smems) = qko_smems

    @grid_loop(init_carry=0)
    def grid_loop_body(gmems, loop_info: plgpu.NDLoopInfo, carry):
      (
          q_gmem,
          k_gmem,
          v_gmem,
          score_mod_value_refs,
          mask_mod_value_refs,
          dropout_mask_gmem,
          o_gmem,
          *residual_gmems,
      ) = gmems
      *batch_pids, hi, qi = loop_info.index
      prev_iters = carry
      lb = 0
      ub = pl.cdiv(kv_seq_len, block_kv)

      # MGPU uses the lower barrier IDs.
      schedule_barrier = 4
      q_consumed_barrier = schedule_barrier + 2
      k_consumed_barrier = q_consumed_barrier + 1
      v_consumed_barrier = k_consumed_barrier + num_stages
      assert v_consumed_barrier + num_stages <= 16

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

        qi_wg = 2 * qi + wg
        q_base = qi_wg * block_q
        qs = cast(pl.Slice, pl.ds(q_base, block_q))

        @pl.when(loop_info.local_index == 0)
        def prologue():
          @pl.when(wg == 0)
          def load_q():
            qs_both = pl.ds(qi * (2 * block_q), 2 * block_q)
            plgpu.copy_gmem_to_smem(
                q_gmem.at[*batch_pids, qs_both, hi], q_smem, q_produced
            )

          for si in range(num_stages):
            mgpu_lib.bar_arrive(k_consumed_barrier + si, num_threads=288)
            mgpu_lib.bar_arrive(v_consumed_barrier + si, num_threads=288)

        m_init_value = -jnp.inf if use_stable_softmax else 0.0
        l_i = plgpu.layout_cast(jnp.zeros((block_q,), jnp.float32), _WGMMA_ROW)
        m_i = plgpu.layout_cast(jnp.full_like(l_i, m_init_value), _WGMMA_ROW)
        acc = jnp.zeros((block_q, head_dim_out), jnp.float32)
        acc = plgpu.layout_cast(acc, _WGMMA)

        pre_pids = (*batch_pids, hi, qi_wg, 0)
        preloaded_score_vals = common.preload_mod_values(
            score_mod_value_refs, score_mod_specs, pre_pids
        )
        preloaded_mask_vals = common.preload_mod_values(
            mask_mod_value_refs, mask_mod_specs, pre_pids
        )

        @pl.when(ub > lb)
        def await_k():
          si = lax.rem(prev_iters, num_stages)
          plgpu.barrier_wait(k_produced.at[si])

        plgpu.barrier_wait(q_produced)
        schedule_barrier_arrive_and_wait(predicate=wg == 1)

        def kv_loop(ki, carry):
          acc, m_scale, m_i, l_i = carry
          si = lax.rem(prev_iters + ki - lb, num_stages)
          k_base = ki * block_kv
          ks = cast(pl.Slice, pl.ds(k_base, block_kv))

          def compute_qk(acc_ref):
            q_smem_wg = q_smem.at[pl.ds(wg * block_q, block_q)]
            plgpu.wgmma(acc_ref, q_smem_wg, k_smem.at[si].T)
            schedule_barrier_arrive()
            return acc_ref[...]

          acc_type = plgpu.ACC((block_q, block_kv), jnp.float32)
          s = pl.run_scoped(compute_qk, acc_type)
          mgpu_lib.bar_arrive(k_consumed_barrier + si, num_threads=288)

          mod_fn_pids = (*batch_pids, hi, qi_wg, ki)
          if score_mod_fn is not None:
            score_vals = common.load_mod_values(
                preloaded_score_vals,
                score_mod_value_refs,
                score_mod_specs,
                mod_fn_pids,
            )
            s = score_mod_fn(mod_fn_pids, (), score_vals, s)

          if mask_mod_fn is not None:
            mask_vals = common.load_mod_values(
                preloaded_mask_vals,
                mask_mod_value_refs,
                mask_mod_specs,
                mod_fn_pids,
            )
            mask = mask_mod_fn(mod_fn_pids, (), mask_vals, s)
            s = jnp.where(mask, s, -jnp.inf)

          if use_mask_k:
            s = jnp.where(k_base + iota(1) < orig_kv_seq_len, s, -jnp.inf)

          if use_stable_softmax:
            m_i = jnp.maximum(m_i, s.max(axis=1) * scale)
            alpha = where_m_valid(m_i, exp(m_scale - m_i), 1.0)
            if lazy_rescale:
              needs_rescale = alpha < rescale_threshold
              m_scale = jnp.where(needs_rescale, m_i, m_scale)
              alpha = jnp.where(needs_rescale, alpha, 1.0)
            else:
              m_scale = m_i
            m_sub = _bcast_rows(where_m_valid(m_i, m_scale, 0.0), s.shape)
            p = exp(s * scale - m_sub)
            acc *= _bcast_rows(alpha, acc.shape)
            l_i *= alpha
          else:
            p = exp(s * scale)

          if dropout_mask_gmem is not None:
            dmask = attn_common.load_bcast(
                dropout_mask_gmem, (*batch_pids, hi, qs, ks), layout=_WGMMA
            )
            p_dropped = p * (dmask.astype(p.dtype) / (1.0 - dropout_rate))
          else:
            p_dropped = p
          p_ = p_dropped.astype(v.dtype)

          if p_sum_before_barriers := (head_dim <= 128):
            l_i += p.sum(axis=1)
            acc, p_ = lax.optimization_barrier((acc, p_))
            l_i, m_i, m_scale = lax.optimization_barrier((l_i, m_i, m_scale))

          plgpu.barrier_wait(v_produced.at[si])
          schedule_barrier_arrive_and_wait()

          def compute_pv(refs):
            acc_ref, l_i_ref = refs
            plgpu.wgmma(acc_ref, p_, v_smem.at[si])

            if not p_sum_before_barriers:
              l_i_ref[...] += p.sum(axis=1)

            @pl.when(ki + 1 < ub)
            def _():
              si_next = lax.rem(prev_iters + ki + 1 - lb, num_stages)
              plgpu.barrier_wait(k_produced.at[si_next])

          acc, l_i = pl.run_state(compute_pv)((plgpu.ACC.init(acc), l_i))
          mgpu_lib.bar_arrive(v_consumed_barrier + si, num_threads=288)
          return acc, m_scale, m_i, l_i

        carry = (acc, m_i, m_i, l_i)
        acc, m_scale, m_i, l_i = lax.fori_loop(lb, ub, kv_loop, carry)

        # Sync so WG1 finishes reading q_smem/k_smem before RefUnion reuse or
        # q_consumed_barrier.
        mgpu_lib.bar_sync(
            schedule_barrier + 1, num_threads=256, predicate=wg == 0
        )
        if config.persistent:
          mgpu_lib.bar_arrive(
              q_consumed_barrier, num_threads=160, predicate=wg == 0
          )

        if lazy_rescale:
          alpha = where_m_valid(m_i, exp(m_scale - m_i), 1.0)
          l_i_out = l_i * alpha
          if not normalize_output:
            acc *= _bcast_rows(alpha, acc.shape)
        else:
          l_i_out = l_i

        if return_residuals:
          m_smem, l_smem = residual_smems
          m_smem[wg] = m_i * (1.0 / scale)
          l_smem[wg] = l_i_out
          plgpu.commit_smem()
          m_gmem, l_gmem = residual_gmems
          plgpu.copy_smem_to_gmem(
              m_smem.at[wg],
              m_gmem.at[*batch_pids, hi, qs],
              commit_group=False,
          )
          plgpu.copy_smem_to_gmem(
              l_smem.at[wg],
              l_gmem.at[*batch_pids, hi, qs],
              commit_group=False,
          )

        if normalize_output:
          l_i += float(jnp.finfo(jnp.float32).tiny)
          acc *= _bcast_rows(1 / l_i, acc.shape)

        _, num_epi_slots, epi_tile_q, epi_tile_d = o_smem.shape
        o_gmem_ = o_gmem.at[*batch_pids, qs, hi]
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
        qs = pl.ds(qi * (2 * block_q), 2 * block_q)

        @plgpu.warp_map
        def per_warp(warp_id):

          @pl.when(warp_id == 0)
          def tma_kv_load_warp():

            def cp(gmem, smem, barrier, si):
              plgpu.copy_gmem_to_smem(gmem, smem.at[si], barrier.at[si])

            @pl.loop(lb, ub)
            def kv_loop(ki):
              si = lax.rem(prev_iters + ki - lb, num_stages)
              ks = pl.ds(ki * block_kv, block_kv)
              mgpu_lib.bar_sync(k_consumed_barrier + si, num_threads=288)
              cp(k_gmem.at[*batch_pids, ks, hi_kv], k_smem, k_produced, si)
              mgpu_lib.bar_sync(v_consumed_barrier + si, num_threads=288)
              cp(v_gmem.at[*batch_pids, ks, hi_kv], v_smem, v_produced, si)

          if config.persistent:

            @pl.when((warp_id == 1) & (loop_info.local_index > 0))
            def tma_q_load_warp():
              plgpu.async_prefetch(q_gmem.at[*batch_pids, qs, hi])
              mgpu_lib.bar_sync(q_consumed_barrier, num_threads=160)
              plgpu.copy_gmem_to_smem(
                  q_gmem.at[*batch_pids, qs, hi], q_smem, q_produced
              )

        return lb, ub

      lb, ub = lax.cond(wg < 2, compute_wg, memory_wg)
      return prev_iters + lax.max(0, ub - lb)

  out_shape = [jax.ShapeDtypeStruct((*q.shape[:-1], head_dim_out), out_dtype)]
  if return_residuals:
    residuals_shape = (*batch, num_q_heads, num_q_tiles * 2 * block_q)
    out_shape += [jax.ShapeDtypeStruct(residuals_shape, jnp.float32)] * 2

  scratch_types = _get_scratch_types(
      q, k, v, out_dtype, return_residuals, config
  )

  if config.persistent:
    maybe_persistent_kernel = mgpu_lib.static_scheduling_persistent_kernel
  else:
    maybe_persistent_kernel = mgpu_lib.not_persistent_grid_loop_kernel

  grid = (*batch, num_q_heads, num_q_tiles)
  grid_names = (
      *(f"batch_{i}" for i in range(len(batch))),
      "heads",
      "q_tiles",
  )

  out, *residuals = maybe_persistent_kernel(
      kernel,
      out_type=out_shape,
      scratch_types=scratch_types,
      grid=grid,
      grid_names=grid_names,
      num_threads=_COMPUTE_WGS + 1,
      thread_name="wg",
      compiler_params=plgpu.CompilerParams(
          approx_math=True,
          unsafe_no_auto_barriers=True,
          reduction_scratch_bytes=0,
      ),
      kernel_name="flex_attention_sm90",
  )(q, k, v, score_mod_values, mask_mod_values, dropout_mask)

  out = out[..., :orig_q_seq_len, :, :orig_head_dim_out]
  residuals = tuple(res[..., :orig_q_seq_len] for res in residuals)
  return (out, residuals if residuals else None)
