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
"""Flash Attention Pallas-Mosaic-GPU VJP implementation (SM100)."""

# pylint: disable=invalid-name

import functools
import math
from typing import Any, cast

import jax
from jax import lax
from jax.experimental import pallas as pl
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
from tokamax._src.ops.attention import pallas_mosaic_gpu_vjp_common as vjp_common

_SMEM_SIZE_LIMIT = 227 * 1024
_TCGEN05 = plgpu.Layout.TCGEN05
_TCGEN05_COL = plgpu.Layout.TCGEN05.reduce(0)
_TCGEN05_ROW = plgpu.Layout.TCGEN05.reduce(1)
_load_bcast = common.load_bcast
_tiled_smem = mgpu_lib.tiled_swizzled_smem


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config(vjp_common.Config):
  """Configuration for the VJP.

  Attributes:
    eltwise_stages: The number of pipeline stages for elementwise ops
      (bias/mask).
    residual_stages: The number of stages for residual data (m, l, delta).
  """
  residual_stages: pydantic.PositiveInt = 2


def _get_dq_scratch_shapes(
    config: Config,
    head_dim: int,
    head_dim_out: int,
    q_dtype,
    dout_dtype,
    k_dtype,
    v_dtype,
    ds_dtype,
    bias,
    mask,
):
  block_q = config.block_q_dq
  block_kv = config.block_kv_dq
  num_stages = config.num_stages

  shapes = dict(
      q_smem=_tiled_smem((block_q, head_dim), q_dtype),
      do_smem=_tiled_smem((block_q, head_dim_out), dout_dtype),
      k_smem=_tiled_smem((num_stages, block_kv, head_dim), k_dtype),
      v_smem=_tiled_smem((num_stages, block_kv, head_dim_out), v_dtype),
      s_tmem=plgpu.TMEM((block_q, block_kv), jnp.float32),
      dp_ds_tmems=plgpu.RefUnion(
          plgpu.TMEM((block_q, block_kv), jnp.float32),
          plgpu.TMEM((block_q, block_kv), k_dtype, packed=True),
      ),
      dq_tmem=plgpu.TMEM((block_q, head_dim), jnp.float32),
      q_do_produced=plgpu.Barrier(num_arrivals=2),
      kv_produced=plgpu.Barrier(num_arrivals=2, num_barriers=num_stages),
      k_consumed=plgpu.Barrier(
          num_barriers=num_stages, orders_tensor_core=True
      ),
      v_consumed=plgpu.Barrier(
          num_barriers=num_stages, orders_tensor_core=True
      ),
      s_produced=plgpu.Barrier(orders_tensor_core=True),
      s_consumed=plgpu.Barrier(orders_tensor_core=True),
      dp_produced=plgpu.Barrier(orders_tensor_core=True),
      ds_produced=plgpu.Barrier(orders_tensor_core=True),
      dq_mma_finished=plgpu.Barrier(orders_tensor_core=True),
  )

  if bias is not None:
    shapes["ds_smem"] = _tiled_smem((block_q, block_kv), ds_dtype)
    if bias.shape[-2] != 1 and bias.shape[-1] != 1:
      shapes["bias_smem"] = _tiled_smem((block_q, block_kv), bias.dtype)
      shapes["bias_produced"] = plgpu.Barrier()
      shapes["bias_consumed"] = plgpu.Barrier()

  if mask is not None and mask.shape[-1] != 1:
    if mask.shape[-2] == 1:
      shapes["mask_smem"] = plgpu.SMEM((block_kv,), jnp.int8)
    else:
      shapes["mask_smem"] = _tiled_smem((block_q, block_kv), jnp.int8)
    shapes["mask_produced"] = plgpu.Barrier()
    shapes["mask_consumed"] = plgpu.Barrier()

  return shapes


def _get_dkv_scratch_shapes(
    config: Config,
    head_dim: int,
    head_dim_out: int,
    q_dtype,
    dout_dtype,
    k_dtype,
    v_dtype,
    bias,
    mask,
):
  block_q = config.block_q_dkv
  block_kv = config.block_kv_dkv
  num_stages = config.num_stages
  residual_stages = config.residual_stages
  shapes = dict(
      k_smem=_tiled_smem((block_kv, head_dim), k_dtype),
      v_smem=_tiled_smem((block_kv, head_dim_out), v_dtype),
      q_smem=_tiled_smem((num_stages, block_q, head_dim), q_dtype),
      do_smem=_tiled_smem((num_stages, block_q, head_dim_out), dout_dtype),
      s_p_tmems=plgpu.RefUnion(
          plgpu.TMEM((block_kv, block_q), jnp.float32),
          plgpu.TMEM((block_kv, block_q), dout_dtype, packed=True),
      ),
      dp_ds_tmems=plgpu.RefUnion(
          plgpu.TMEM((block_kv, block_q), jnp.float32),
          plgpu.TMEM((block_kv, block_q), q_dtype, packed=True),
      ),
      dk_tmem=plgpu.TMEM((block_kv, head_dim), jnp.float32),
      dv_tmem=plgpu.TMEM((block_kv, head_dim_out), jnp.float32),
      kv_produced=plgpu.Barrier(num_arrivals=2),
      q_do_produced=plgpu.Barrier(num_barriers=num_stages, num_arrivals=2),
      q_consumed=plgpu.Barrier(
          num_barriers=num_stages, orders_tensor_core=True
      ),
      do_consumed=plgpu.Barrier(
          num_barriers=num_stages, orders_tensor_core=True
      ),
      s_produced=plgpu.Barrier(orders_tensor_core=True),
      p_produced=plgpu.Barrier(orders_tensor_core=True),
      dp_produced=plgpu.Barrier(orders_tensor_core=True),
      ds_produced=plgpu.Barrier(orders_tensor_core=True),
      kv_mma_finished=plgpu.Barrier(orders_tensor_core=True),
      residuals_smem=plgpu.SMEM((3, residual_stages, block_q), jnp.float32),
      residual_produced=plgpu.Barrier(
          num_barriers=residual_stages, num_arrivals=3
      ),
      residual_consumed=plgpu.Barrier(num_barriers=residual_stages),
  )
  if bias is not None and bias.shape[-2] != 1 and bias.shape[-1] != 1:
    shapes["bias_smem"] = _tiled_smem((block_kv, block_q), bias.dtype, "bias")
    shapes["bias_produced"] = plgpu.Barrier()
    shapes["bias_consumed"] = plgpu.Barrier()

  if mask is not None and mask.shape[-2] != 1 and mask.shape[-1] != 1:
    shapes["mask_smem"] = _tiled_smem((block_kv, block_q), jnp.int8, "mask")
    shapes["mask_produced"] = plgpu.Barrier()
    shapes["mask_consumed"] = plgpu.Barrier()

  return shapes


def get_autotuning_configs(ba: op.BoundArguments) -> set[Config]:
  args_dict = getattr(ba, "arguments", {})

  def _get(name, pos):
    if name in args_dict:
      return args_dict[name]
    if pos >= 0 and len(ba.args) > pos:
      return ba.args[pos]
    return ba.kwargs.get(name)

  q, k, v, dout = _get("q", 3), _get("k", 4), _get("v", 5), _get("dout", 2)
  # Satisfy pytype
  assert q is not None and k is not None and v is not None and dout is not None
  bias, mask_obj = _get("bias", -1), _get("mask", -1)
  q_indices, k_indices = _get("q_indices", -1), _get("k_indices", -1)
  precision = _get("precision", -1)

  def _downcast_if_needed(dtype, prec):
    if dtype == jnp.float32 and prec is not None:
      if prec == jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
        return jnp.bfloat16
      if prec == jax.lax.DotAlgorithmPreset.F16_F16_F32:
        return jnp.float16
    return dtype

  q_k_prec = precision[0] if precision is not None and precision != -1 else None
  v_prec = precision[1] if precision is not None and precision != -1 else None

  mask, *_ = jax.eval_shape(
      common.decompose_mask, mask_obj, q, k, q_indices, k_indices
  )

  head_dim, head_dim_out = _get_input_metadata(q, v)
  dbias_intermediate_dtype = getattr(ba.op, "dbias_intermediate_dtype", None)
  ds_dtype = vjp_common.get_ds_dtype(q, k, bias, dbias_intermediate_dtype)

  configs = set()
  min_dq_smem = float("inf")
  min_dkv_smem = float("inf")
  min_total_smem = float("inf")
  fallback_dq_smem = 0
  fallback_dkv_smem = 0
  q_dtype = _downcast_if_needed(q.dtype, q_k_prec)
  k_dtype = _downcast_if_needed(k.dtype, q_k_prec)
  v_dtype = _downcast_if_needed(v.dtype, v_prec)
  dout_dtype = _downcast_if_needed(dout.dtype, v_prec)

  for q_kv_block_size in (128, 64):
    for residual_stages in (1, 2):
      for num_stages in (2, 3, 4):
        config = Config(
            block_kv_dkv=128,
            block_q_dkv=q_kv_block_size,
            block_kv_dq=q_kv_block_size,
            block_q_dq=128,
            residual_stages=residual_stages,
            num_stages=num_stages,
        )
        dq_shapes = _get_dq_scratch_shapes(
            config=config,
            head_dim=head_dim,
            head_dim_out=head_dim_out,
            q_dtype=q_dtype,
            dout_dtype=dout_dtype,
            k_dtype=k_dtype,
            v_dtype=v_dtype,
            ds_dtype=ds_dtype,
            bias=bias,
            mask=mask,
        )
        dkv_shapes = _get_dkv_scratch_shapes(
            config=config,
            head_dim=head_dim,
            head_dim_out=head_dim_out,
            q_dtype=q_dtype,
            dout_dtype=dout_dtype,
            k_dtype=k_dtype,
            v_dtype=v_dtype,
            bias=bias,
            mask=mask,
        )
        dq_smem = _estimate_smem_bytes(dq_shapes)
        dkv_smem = _estimate_smem_bytes(dkv_shapes)
        if dq_smem + dkv_smem < min_total_smem:
          min_total_smem = dq_smem + dkv_smem
          fallback_dq_smem = dq_smem
          fallback_dkv_smem = dkv_smem
        min_dq_smem = min(min_dq_smem, dq_smem)
        min_dkv_smem = min(min_dkv_smem, dkv_smem)
        if dq_smem <= _SMEM_SIZE_LIMIT and dkv_smem <= _SMEM_SIZE_LIMIT:
          configs.add(config)
    # If we found a good config for q_kv_block_size 128 there is no point
    # looking into 64 which is strictly worse for use of TC and
    # SMEM/TMEM.
    if configs:
      break
  if not configs:
    raise ValueError(
        "Could not find any SM100 dual kernel configuration that fits in"
        f" shared memory (limit: {_SMEM_SIZE_LIMIT} bytes). The smallest"
        f" configuration requires {fallback_dq_smem} bytes for the `dq` kernel"
        f" and {fallback_dkv_smem} bytes for the `dkv` kernel."
    )
  return configs


def get_heuristics_config(ba: op.BoundArguments) -> Config:
  """Returns a heuristic configuration for flash attention VJP on SM100 GPUs."""
  configs = get_autotuning_configs(ba)
  if len(configs) == 1:
    return next(iter(configs))

  def _score(c: Config):
    return (c.num_stages, c.residual_stages)

  return max(configs, key=_score)


def _get_input_metadata(q, v):
  """Normalizes and returns head dimensions and datatypes."""
  head_dim = pl.cdiv(q.shape[-1], 64) * 64
  head_dim_out = pl.cdiv(v.shape[-1], 64) * 64
  return head_dim, head_dim_out


def _estimate_smem_bytes(scratch_shapes: dict[str, Any]) -> int:
  return mgpu_lib.estimate_smem_bytes(scratch_shapes) + 4096 * 2


def _kernel_dq(
    q_gmem,
    k_gmem,
    v_gmem,
    dout_gmem,
    m_gmem,
    l_gmem,
    delta_gmem,
    bias_gmem,
    k_start_gmem,
    k_end_gmem,
    mask_gmem,
    dq_gmem,
    ds_gmem,
    *,
    q_smem,
    do_smem,
    k_smem,
    v_smem,
    s_tmem,
    dp_ds_tmems,
    dq_tmem,
    q_do_produced,
    kv_produced,
    k_consumed,
    v_consumed,
    s_produced,
    s_consumed,
    dp_produced,
    ds_produced,
    dq_mma_finished,
    bias_smem=None,
    mask_smem=None,
    ds_smem=None,
    bias_produced=None,
    bias_consumed=None,
    mask_produced=None,
    mask_consumed=None,
    config: Config,
    is_causal,
    logits_scale,
    logits_soft_cap,
):
  """Computes dq."""
  wg = lax.axis_index("wg")
  qi = lax.axis_index("q_tiles")
  hi = lax.axis_index("heads")

  block_q = config.block_q_dq
  block_kv = config.block_kv_dq
  num_stages = config.num_stages

  # We assume MHA or simple mapping here to respect boundaries.
  q_heads_per_kv_head = q_gmem.shape[-2] // k_gmem.shape[-2]
  hi_kv = lax.div(hi, jnp.array(q_heads_per_kv_head, hi.dtype))

  q_base = qi * block_q
  qs = cast(pl.Slice, pl.ds(q_base, block_q))

  lb = 0
  ub = k_gmem.shape[-3] // block_kv
  if is_causal:
    ub = lax.min(ub, pl.cdiv(q_base + block_q, block_kv))

  dp_tmem, ds_tmem = dp_ds_tmems

  @pl.when((wg == 0) & (ub > lb))
  def mma_tma_wg():

    @plgpu.warp_map
    def per_warp(warp_id):
      def cp(gmem, smem, barrier, si=()):
        plgpu.copy_gmem_to_smem(gmem, smem.at[si], barrier.at[si])

      @pl.when(warp_id == 0)
      def tma_q_warp():
        cp(q_gmem.at[qs, hi], q_smem, q_do_produced)
        cp(dout_gmem.at[qs, hi], do_smem, q_do_produced)

      @pl.when(warp_id == 1)
      def tma_kv_warp():
        @pl.loop(lb, lax.min(lb + num_stages, ub))
        def prologue(ki):
          si = lax.rem(ki - lb, num_stages)
          ks = pl.ds(ki * block_kv, block_kv)
          cp(v_gmem.at[ks, hi_kv], v_smem, kv_produced, si)
          cp(k_gmem.at[ks, hi_kv], k_smem, kv_produced, si)

        @pl.loop(lb + num_stages, ub)
        def kv_loop(ki):
          si = lax.rem(ki - lb, num_stages)
          ks = pl.ds(ki * block_kv, block_kv)
          plgpu.barrier_wait(v_consumed.at[si])
          cp(v_gmem.at[ks, hi_kv], v_smem, kv_produced, si)
          plgpu.barrier_wait(k_consumed.at[si])
          cp(k_gmem.at[ks, hi_kv], k_smem, kv_produced, si)

      if bias_gmem is not None or mask_gmem is not None:

        @pl.when(warp_id == 3)
        def tma_eltwise_warp():
          if bias_smem is not None:
            plgpu.barrier_arrive(bias_consumed)  # pyrefly: ignore[bad-argument-type]
          if mask_smem is not None:
            plgpu.barrier_arrive(mask_consumed)  # pyrefly: ignore[bad-argument-type]

          @pl.loop(lb, ub)
          def kv_loop(ki):
            ks = pl.ds(ki * block_kv, block_kv)

            if bias_smem is not None:
              plgpu.barrier_wait(bias_consumed)  # pyrefly: ignore[bad-argument-type]
              mgpu_lib.fence_async_shared_cta()
              bias_hi = 0 if bias_gmem.shape[-3] == 1 else hi
              cp(bias_gmem.at[bias_hi, qs, ks], bias_smem, bias_produced)

            if mask_smem is not None:
              plgpu.barrier_wait(mask_consumed)  # pyrefly: ignore[bad-argument-type]
              mgpu_lib.fence_async_shared_cta()
              mask_hi = 0 if mask_gmem.shape[-3] == 1 else hi
              mask_qs = 0 if mask_gmem.shape[-2] == 1 else qs
              cp(mask_gmem.at[mask_hi, mask_qs, ks], mask_smem, mask_produced)

      @pl.when(warp_id == 2)
      def mma_warp():

        def q_k_mma(ki, wait_s_consumed=True):
          si = lax.rem(ki - lb, num_stages)
          plgpu.barrier_wait(kv_produced.at[si])
          if wait_s_consumed:
            plgpu.barrier_wait(s_consumed)
          plgpu.tcgen05_mma(s_tmem, q_smem, k_smem.at[si].T, accumulate=False)
          plgpu.tcgen05_commit_arrive(s_produced)

        def do_v_mma(ki):
          si = lax.rem(ki - lb, num_stages)
          plgpu.tcgen05_mma(dp_tmem, do_smem, v_smem.at[si].T, accumulate=False)
          plgpu.tcgen05_commit_arrive(dp_produced)
          plgpu.tcgen05_commit_arrive(v_consumed.at[si])

        def ds_k_mma(ki):
          si = lax.rem(ki - lb, num_stages)
          plgpu.barrier_wait(ds_produced)
          plgpu.tcgen05_mma(dq_tmem, ds_tmem, k_smem.at[si], accumulate=ki > lb)
          plgpu.tcgen05_commit_arrive(k_consumed.at[si])

        plgpu.barrier_wait(q_do_produced)
        q_k_mma(lb, wait_s_consumed=False)
        do_v_mma(lb)

        @pl.loop(lb, ub - 1)
        def mma_loop(ki):
          q_k_mma(ki + 1)
          ds_k_mma(ki)
          do_v_mma(ki + 1)

        ds_k_mma(ub - 1)
        plgpu.tcgen05_commit_arrive(dq_mma_finished)

    plgpu.barrier_wait(dq_mma_finished)
    dq = plgpu.async_load_tmem(dq_tmem, layout=_TCGEN05)
    plgpu.wait_load_tmem()
    q_smem[...] = (dq * logits_scale).astype(q_smem.dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(q_smem, dq_gmem.at[qs, hi])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  @pl.when((wg == 1) & (ub > lb))
  def softmax_wg():
    if bias_gmem is None and mask_gmem is None:
      layout = plgpu.Layout.TCGEN05_TMEM_NATIVE
    else:
      layout = plgpu.Layout.TCGEN05
    row_layout = layout.reduce(1)

    def load_k_range(ref):
      if ref is None:
        return None
      hi_ = 0 if ref.shape[0] == 1 else hi
      return plgpu.load(ref.at[hi_, qs], layout=row_layout, optimized=False)

    k_start = load_k_range(k_start_gmem)
    k_end = load_k_range(k_end_gmem)

    m = plgpu.load(m_gmem.at[hi, qs], layout=row_layout, optimized=False)
    l_rcp = plgpu.load(l_gmem.at[hi, qs], layout=row_layout, optimized=False)
    delta = plgpu.load(
        delta_gmem.at[hi, qs], layout=row_layout, optimized=False
    )
    m *= math.log2(math.e)

    @pl.loop(lb, ub)
    def kv_loop(ki):
      kv_base = ki * block_kv
      ks = cast(pl.Slice, pl.ds(kv_base, block_kv))

      plgpu.barrier_wait(s_produced)
      s = plgpu.async_load_tmem(s_tmem, layout=layout)
      scale = logits_scale

      if bias_gmem is not None:
        if bias_smem is None:
          bias = _load_bcast(bias_gmem, (hi, qs, ks), layout=layout)
        else:
          plgpu.barrier_wait(bias_produced)  # pyrefly: ignore[bad-argument-type]
          bias = plgpu.load(bias_smem, layout=layout)
          plgpu.barrier_arrive(bias_consumed)  # pyrefly: ignore[bad-argument-type]
        s, scale = s * scale + bias.astype(s.dtype), 1.0

      mgpu_lib.tcgen05_wait_ld()
      plgpu.barrier_arrive(s_consumed)

      if logits_soft_cap is not None:
        s, scale = jnp.tanh(s * (scale / logits_soft_cap)), logits_soft_cap
      logits = s

      def iota(d):
        return plgpu.broadcasted_iota(jnp.int32, s.shape, d, layout=layout)

      def apply_causal_mask():
        return jnp.where(q_base + iota(0) >= kv_base + iota(1), s, -jnp.inf)

      if is_causal:
        s = lax.cond(kv_base + block_kv > q_base, apply_causal_mask, lambda: s)

      bcast = lambda x: lax.broadcast_in_dim(x, s.shape, [0])
      p = jnp.exp2(s * (scale * math.log2(math.e)) - bcast(m)) * bcast(l_rcp)

      if k_start is not None:
        p = jnp.where(kv_base + iota(1) >= bcast(k_start), p, 0.0)

      if k_end is not None:
        p = jnp.where(kv_base + iota(1) < bcast(k_end), p, 0.0)

      if mask_gmem is not None:
        if mask_smem is None:
          mask = _load_bcast(mask_gmem, (hi, qs, ks), layout=layout)
        else:
          plgpu.barrier_wait(mask_produced)  # pyrefly: ignore[bad-argument-type]
          if mask_smem.ndim == 1:
            mask = plgpu.load(mask_smem, layout=_TCGEN05_COL)
            mask = lax.broadcast_in_dim(mask, s.shape, [1])
          else:
            mask = plgpu.load(mask_smem, layout=layout)
          plgpu.barrier_arrive(mask_consumed)  # pyrefly: ignore[bad-argument-type]

        p = jnp.where(mask, p, 0.0)

      plgpu.barrier_wait(dp_produced)
      dp = plgpu.async_load_tmem(dp_tmem, layout=layout)
      ds = p * (dp - bcast(delta))

      if logits_soft_cap is not None:
        ds *= 1.0 - logits * logits

      plgpu.async_store_tmem(ds_tmem, ds.astype(ds_tmem.dtype))
      if ds_gmem is not None:
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
      mgpu_lib.tcgen05_wait_st()
      plgpu.barrier_arrive(ds_produced)

      if ds_gmem is not None:
        assert ds_smem is not None
        ds_smem[...] = ds.astype(ds_smem.dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(ds_smem, ds_gmem.at[hi, qs, ks])

    if ds_gmem is not None:
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)


def _kernel_dkv(
    q_gmem,
    k_gmem,
    v_gmem,
    dout_gmem,
    m_gmem,
    l_gmem,
    delta_gmem,
    bias_gmem,
    k_start_gmem,
    k_end_gmem,
    mask_gmem,
    dk_gmem,
    dv_gmem,
    *,
    k_smem,
    v_smem,
    q_smem,
    do_smem,
    residuals_smem,
    s_p_tmems,
    dp_ds_tmems,
    dk_tmem,
    dv_tmem,
    kv_produced,
    q_do_produced,
    q_consumed,
    do_consumed,
    residual_produced,
    residual_consumed,
    s_produced,
    p_produced,
    dp_produced,
    ds_produced,
    kv_mma_finished,
    bias_smem=None,
    mask_smem=None,
    bias_produced=None,
    bias_consumed=None,
    mask_produced=None,
    mask_consumed=None,
    config,
    is_causal,
    logits_scale,
    logits_soft_cap,
):
  """Computes dkv."""
  wg = lax.axis_index("wg")
  ki = lax.axis_index("kv_tiles")
  hi_kv = lax.axis_index("heads")

  block_q = config.block_q_dkv
  block_kv = config.block_kv_dkv
  num_stages = config.num_stages
  residual_stages = config.residual_stages

  num_q_heads = q_gmem.shape[-2]
  q_heads_per_kv_head = num_q_heads // k_gmem.shape[-2]

  kv_base = ki * block_kv
  ks = pl.ds(kv_base, block_kv)

  lb = lax.div(kv_base, block_q) if is_causal else 0
  ub = q_gmem.shape[-3] // block_q
  num_q_tiles = ub - lb
  safe_num_q_tiles = lax.max(num_q_tiles, 1)
  total_steps = q_heads_per_kv_head * num_q_tiles

  if residuals_smem is not None:
    # Pack m, l, and delta into the same buffer to avoid SMEM padding overhead.
    # Note that they represent different residuals.
    m_smem = residuals_smem.at[0]
    l_smem = residuals_smem.at[1]
    delta_smem = residuals_smem.at[2]

  s_tmem, p_tmem = s_p_tmems
  dp_tmem, ds_tmem = dp_ds_tmems

  @pl.when((wg == 0) & (total_steps > 0))
  def mma_tma_wg():

    @plgpu.warp_map
    def per_warp(warp_id):

      def cp(gmem, smem, barrier, si=()):
        plgpu.copy_gmem_to_smem(gmem, smem.at[si], barrier.at[si])

      @pl.when(warp_id == 0)
      def tma_kv_warp():
        cp(k_gmem.at[ks, hi_kv], k_smem, kv_produced)
        cp(v_gmem.at[ks, hi_kv], v_smem, kv_produced)

        @pl.loop(0, total_steps)
        def q_loop(step):
          si = lax.rem(step, residual_stages)
          qi = lb + lax.rem(step, safe_num_q_tiles)
          qs = pl.ds(qi * block_q, block_q)
          hi = hi_kv * q_heads_per_kv_head + lax.div(step, safe_num_q_tiles)

          @pl.when(step >= residual_stages)
          def wait_res():
            plgpu.barrier_wait(residual_consumed.at[si])
            mgpu_lib.fence_async_shared_cta()

          cp(m_gmem.at[hi, qs], m_smem, residual_produced, si)
          cp(l_gmem.at[hi, qs], l_smem, residual_produced, si)
          cp(delta_gmem.at[hi, qs], delta_smem, residual_produced, si)

      @pl.when(warp_id == 1)
      def tma_q_warp():
        @pl.loop(0, total_steps)
        def q_loop(step):
          si = lax.rem(step, num_stages)
          qi = lb + lax.rem(step, safe_num_q_tiles)
          qs = pl.ds(qi * block_q, block_q)
          hi = hi_kv * q_heads_per_kv_head + lax.div(step, safe_num_q_tiles)
          do_wait = step >= num_stages

          pl.when(do_wait)(lambda: plgpu.barrier_wait(do_consumed.at[si]))
          cp(dout_gmem.at[qs, hi], do_smem, q_do_produced, si)
          pl.when(do_wait)(lambda: plgpu.barrier_wait(q_consumed.at[si]))
          cp(q_gmem.at[qs, hi], q_smem, q_do_produced, si)

      if bias_gmem is not None or mask_gmem is not None:

        @pl.when(warp_id == 3)
        def tma_eltwise_warp():
          if bias_smem is not None:
            plgpu.barrier_arrive(bias_consumed)  # pyrefly: ignore[bad-argument-type]
          if mask_smem is not None:
            plgpu.barrier_arrive(mask_consumed)  # pyrefly: ignore[bad-argument-type]

          @pl.loop(0, total_steps)
          def q_loop(step):
            qi = lb + lax.rem(step, safe_num_q_tiles)
            qs = pl.ds(qi * block_q, block_q)
            hi = hi_kv * q_heads_per_kv_head + lax.div(step, safe_num_q_tiles)

            if bias_smem is not None:
              plgpu.barrier_wait(bias_consumed)  # pyrefly: ignore[bad-argument-type]
              mgpu_lib.fence_async_shared_cta()
              bias_hi = 0 if bias_gmem.shape[-3] == 1 else hi
              cp(bias_gmem.at[bias_hi, ks, qs], bias_smem, bias_produced)

            if mask_smem is not None:
              plgpu.barrier_wait(mask_consumed)  # pyrefly: ignore[bad-argument-type]
              mgpu_lib.fence_async_shared_cta()
              mask_hi = 0 if mask_gmem.shape[-3] == 1 else hi
              mask_qs = 0 if mask_gmem.shape[-1] == 1 else qs
              cp(mask_gmem.at[mask_hi, ks, mask_qs], mask_smem, mask_produced)

      @pl.when(warp_id == 2)
      def mma_warp():
        plgpu.barrier_wait(kv_produced)

        @pl.loop(0, total_steps)
        def q_loop(step):
          si = lax.rem(step, num_stages)
          plgpu.barrier_wait(q_do_produced.at[si])
          plgpu.tcgen05_mma(s_tmem, k_smem, q_smem.at[si].T, accumulate=False)
          plgpu.tcgen05_commit_arrive(s_produced)

          plgpu.tcgen05_mma(dp_tmem, v_smem, do_smem.at[si].T, accumulate=False)
          plgpu.tcgen05_commit_arrive(dp_produced)

          plgpu.barrier_wait(p_produced)
          plgpu.tcgen05_mma(
              dv_tmem, p_tmem, do_smem.at[si], accumulate=(step > 0)
          )
          plgpu.tcgen05_commit_arrive(do_consumed.at[si])

          plgpu.barrier_wait(ds_produced)
          plgpu.tcgen05_mma(
              dk_tmem, ds_tmem, q_smem.at[si], accumulate=(step > 0)
          )
          plgpu.tcgen05_commit_arrive(q_consumed.at[si])

        plgpu.tcgen05_commit_arrive(kv_mma_finished)

    plgpu.barrier_wait(kv_mma_finished)
    dk = plgpu.async_load_tmem(dk_tmem, layout=_TCGEN05)
    dv = plgpu.async_load_tmem(dv_tmem, layout=_TCGEN05)
    k_smem[...] = (dk * logits_scale).astype(k_smem.dtype)
    v_smem[...] = dv.astype(v_smem.dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(k_smem, dk_gmem.at[ks, hi_kv])
    plgpu.copy_smem_to_gmem(v_smem, dv_gmem.at[ks, hi_kv])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  @pl.when((wg == 1) & (total_steps > 0))
  def softmax_wg():
    if mask_gmem is None:
      loop_invariant_mask = None
    elif mask_gmem.shape[-3] != 1 or mask_gmem.shape[-1] != 1:
      loop_invariant_mask = None
    else:
      loop_invariant_mask = plgpu.load(
          mask_gmem.at[0, ks, 0], layout=_TCGEN05_ROW, optimized=False
      )

    @pl.loop(0, total_steps)
    def q_loop(step):
      si_res = lax.rem(step, residual_stages)
      hi = hi_kv * q_heads_per_kv_head + lax.div(step, safe_num_q_tiles)
      qi = lb + lax.rem(step, safe_num_q_tiles)
      q_base = qi * block_q
      qs = pl.ds(q_base, block_q)
      plgpu.barrier_wait(residual_produced.at[si_res])

      if bias_gmem is None:
        bias = None
      elif bias_smem is None:
        bias = _load_bcast(bias_gmem, (hi, ks, qs), layout=_TCGEN05)  # pyrefly: ignore[bad-argument-type]
      else:
        plgpu.barrier_wait(bias_produced)  # pyrefly: ignore[bad-argument-type]
        bias = plgpu.load(bias_smem, layout=_TCGEN05)

      plgpu.barrier_wait(s_produced)
      s = plgpu.async_load_tmem(s_tmem, layout=_TCGEN05)
      scale = logits_scale

      if bias is not None:
        if bias_smem is not None:
          plgpu.barrier_arrive(bias_consumed)  # pyrefly: ignore[bad-argument-type]
        s, scale = s * scale + bias.astype(s.dtype), 1.0

      if logits_soft_cap is not None:
        s, scale = jnp.tanh(s * (scale / logits_soft_cap)), logits_soft_cap
      logits = s

      m = plgpu.load(m_smem.at[si_res], layout=_TCGEN05_COL)
      l_rcp = plgpu.load(l_smem.at[si_res], layout=_TCGEN05_COL)

      def iota(d):
        return plgpu.broadcasted_iota(jnp.int32, s.shape, d, layout=_TCGEN05)

      def apply_causal_mask():
        return jnp.where(kv_base + iota(0) <= q_base + iota(1), s, -jnp.inf)

      if is_causal:
        s = lax.cond(kv_base + block_kv > q_base, apply_causal_mask, lambda: s)

      bcast = lambda x: lax.broadcast_in_dim(x, s.shape, [1])
      m *= math.log2(math.e)
      p = jnp.exp2(s * (scale * math.log2(math.e)) - bcast(m)) * bcast(l_rcp)

      def load_k_range(ref):
        hi_ = 0 if ref.shape[0] == 1 else hi
        return plgpu.load(ref.at[hi_, qs], layout=_TCGEN05_COL, optimized=False)

      if k_start_gmem is not None:
        k_start = bcast(load_k_range(k_start_gmem))
        p = jnp.where(kv_base + iota(0) >= k_start, p, 0.0)

      if k_end_gmem is not None:
        k_end = bcast(load_k_range(k_end_gmem))
        p = jnp.where(kv_base + iota(0) < k_end, p, 0.0)

      if mask_gmem is not None:
        if mask_smem is None:
          if loop_invariant_mask is None:
            mask = _load_bcast(mask_gmem, (hi, ks, qs), layout=_TCGEN05)  # pyrefly: ignore[bad-argument-type]
          else:
            mask = lax.broadcast_in_dim(loop_invariant_mask, s.shape, [0])
        else:
          plgpu.barrier_wait(mask_produced)  # pyrefly: ignore[bad-argument-type]
          mask = plgpu.load(mask_smem, layout=_TCGEN05)
          plgpu.barrier_arrive(mask_consumed)  # pyrefly: ignore[bad-argument-type]

        p = jnp.where(mask, p, 0.0)

      plgpu.async_store_tmem(p_tmem, p.astype(p_tmem.dtype))
      mgpu_lib.tcgen05_wait_st()
      plgpu.barrier_arrive(p_produced)

      delta = plgpu.load(delta_smem.at[si_res], layout=_TCGEN05_COL)
      plgpu.barrier_arrive(residual_consumed.at[si_res])

      plgpu.barrier_wait(dp_produced)
      dp = plgpu.async_load_tmem(dp_tmem, layout=_TCGEN05)
      ds = p * (dp - bcast(delta))

      if logits_soft_cap is not None:
        ds *= 1.0 - logits * logits

      plgpu.async_store_tmem(ds_tmem, ds.astype(ds_tmem.dtype))
      mgpu_lib.tcgen05_wait_st()
      plgpu.barrier_arrive(ds_produced)


def _pad_maybe_bcast(x, m, axis):
  if x.shape[axis] == 1:
    return x
  return shape_lib.pad_to_next_multiple_of(x, m, axis)


@jaxtyping.jaxtyped
def flash_attention_vjp_kernel(
    q: Float[Array, "T H D"],
    k: Float[Array, "t h D"],
    v: Float[Array, "t h d"],
    residuals: base.Residuals,
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
    ds_dtype: jax.typing.DTypeLike | None,
    config: Config,
) -> tuple[
    Float[Array, "T H D"],  # dq
    Float[Array, "t h D"],  # dk
    Float[Array, "t h d"],  # dv
    Float[Array, "#H #T #t"] | None,  # ds
]:
  """SM100 Pallas Mosaic GPU Flash Attention VJP."""
  orig_q_seq_len, _, orig_head_dim = q.shape
  orig_kv_seq_len, _, orig_head_dim_out = v.shape

  block_q_dq = config.block_q_dq
  block_kv_dq = config.block_kv_dq
  block_kv_dkv = config.block_kv_dkv

  # TODO: Remove explicit padding in favor of TMA out-of-bounds zero-filling and in-kernel -inf masking.
  q = shape_lib.pad_to_next_multiple_of(q, block_q_dq, -3)
  out = shape_lib.pad_to_next_multiple_of(out, block_q_dq, -3)
  dout = shape_lib.pad_to_next_multiple_of(dout, block_q_dq, -3)
  k = shape_lib.pad_to_next_multiple_of(k, block_kv_dkv, -3)
  v = shape_lib.pad_to_next_multiple_of(v, block_kv_dkv, -3)

  # TODO: Avoid broadcast.
  bcast = lambda x: jnp.broadcast_to(x, (q.shape[-2], orig_q_seq_len))
  k_start = None if k_start is None else bcast(k_start)
  k_end = None if k_end is None else bcast(k_end)
  if mask is not None:
    # Mask shape is usually broadcasted
    mask = mask.astype(jnp.int8)

  if mask is not None:
    mask = _pad_maybe_bcast(mask, block_q_dq, -2)
    mask = _pad_maybe_bcast(mask, block_kv_dkv, -1)
  if bias is not None:
    bias = _pad_maybe_bcast(bias, block_q_dq, -2)
    bias = _pad_maybe_bcast(bias, block_kv_dkv, -1)
  if k_start is not None:
    k_start = shape_lib.pad_to_next_multiple_of(k_start, block_q_dq, -1)
  if k_end is not None:
    k_end = shape_lib.pad_to_next_multiple_of(k_end, block_q_dq, -1)

  pad_dim = lambda x: shape_lib.pad_to_next_multiple_of(x, 64, -1)
  q, k, v, out, dout = map(pad_dim, (q, k, v, out, dout))
  head_dim, head_dim_out = _get_input_metadata(q, v)

  m, l = residuals
  m = shape_lib.pad_to_next_multiple_of(m, block_q_dq, -1, pad_value=1e9)
  l = shape_lib.pad_to_next_multiple_of(l, block_q_dq, -1, pad_value=1)
  l_rcp = jnp.reciprocal(l + jnp.finfo(jnp.float32).tiny)

  delta = jnp.einsum(
      "...qhd,...qhd->...hq", out.astype(jnp.float32), dout.astype(jnp.float32)
  )
  delta = shape_lib.pad_to_next_multiple_of(delta, block_q_dq, -1)

  compiler_params = plgpu.CompilerParams(
      approx_math=True,
      unsafe_no_auto_barriers=True,
      reduction_scratch_bytes=0,
  )

  dq_scratch_shapes = _get_dq_scratch_shapes(
      config=config,
      head_dim=head_dim,
      head_dim_out=head_dim_out,
      q_dtype=q.dtype,
      dout_dtype=dout.dtype,
      k_dtype=k.dtype,
      v_dtype=v.dtype,
      ds_dtype=ds_dtype,
      bias=bias,
      mask=mask,
  )

  if bias is None:
    ds_shape = None
  else:
    q_seq_len_ = pl.cdiv(q.shape[-3], block_q_dq) * block_q_dq
    kv_seq_len_ = pl.cdiv(k.shape[-3], block_kv_dq) * block_kv_dq
    ds_shape = (q.shape[-2], q_seq_len_, kv_seq_len_)
    ds_shape = jax.ShapeDtypeStruct(ds_shape, ds_dtype)

  kernel_dq = functools.partial(
      _kernel_dq,
      config=config,
      is_causal=is_causal,
      logits_scale=logits_scale,
      logits_soft_cap=logits_soft_cap,
  )

  dq, ds = plgpu.kernel(
      kernel_dq,
      out_type=(jax.ShapeDtypeStruct(q.shape, q.dtype), ds_shape),
      kernel_name="sm100_dq_kernel",
      grid=(q.shape[-2], q.shape[-3] // block_q_dq),
      grid_names=("heads", "q_tiles"),
      num_threads=2,
      thread_name="wg",
      compiler_params=compiler_params,
      scratch_types=dq_scratch_shapes,
  )(q, k, v, dout, m, l_rcp, delta, bias, k_start, k_end, mask)

  dkv_shape = (
      jax.ShapeDtypeStruct(k.shape, k.dtype),
      jax.ShapeDtypeStruct(v.shape, v.dtype),
  )

  dkv_scratch_shapes = _get_dkv_scratch_shapes(
      config=config,
      head_dim=head_dim,
      head_dim_out=head_dim_out,
      q_dtype=q.dtype,
      dout_dtype=dout.dtype,
      k_dtype=k.dtype,
      v_dtype=v.dtype,
      bias=bias,
      mask=mask,
  )

  bias_dkv = None if bias is None else bias.mT
  mask_dkv = None if mask is None else mask.mT

  kernel_dkv = functools.partial(
      _kernel_dkv,
      config=config,
      is_causal=is_causal,
      logits_scale=logits_scale,
      logits_soft_cap=logits_soft_cap,
  )

  dk, dv = plgpu.kernel(
      kernel_dkv,
      out_type=dkv_shape,
      kernel_name="sm100_dkv_kernel",
      grid=(k.shape[-2], k.shape[-3] // block_kv_dkv),
      grid_names=("heads", "kv_tiles"),
      num_threads=2,
      thread_name="wg",
      compiler_params=compiler_params,
      scratch_types=dkv_scratch_shapes,
  )(q, k, v, dout, m, l_rcp, delta, bias_dkv, k_start, k_end, mask_dkv)

  dq = dq[:orig_q_seq_len, :, :orig_head_dim]
  dk = dk[:orig_kv_seq_len, :, :orig_head_dim]
  dv = dv[:orig_kv_seq_len, :, :orig_head_dim_out]
  ds = None if ds is None else ds[:, :orig_q_seq_len, :orig_kv_seq_len]
  return dq, dk, dv, ds
