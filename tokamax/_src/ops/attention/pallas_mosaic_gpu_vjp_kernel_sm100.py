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
from typing import Annotated, cast

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
    double_buffer: Whether to use double buffering for SMEM allocations.
    residual_stages: The number of stages for residual data (m, l, delta).
    chunk_size: The chunk size for processing along the sequence dimension.
  """

  eltwise_stages: pydantic.PositiveInt = 1
  double_buffer: bool = False
  residual_stages: pydantic.PositiveInt = 2
  chunk_size: Annotated[int, pydantic.Field(multiple_of=32, ge=32)] = 64


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
  eltwise_stages = config.eltwise_stages

  shapes = dict(
      q_smem=_tiled_smem((block_q, head_dim), q_dtype),
      do_smem=_tiled_smem((block_q, head_dim_out), dout_dtype),
      k_smem=_tiled_smem((num_stages, block_kv, head_dim), k_dtype, swizzle=64),
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
      shape = (eltwise_stages, block_q, block_kv)
      shapes["bias_smem"] = _tiled_smem(shape, bias.dtype)
      shapes["bias_produced"] = plgpu.Barrier(num_barriers=eltwise_stages)
      shapes["bias_consumed"] = plgpu.Barrier(num_barriers=eltwise_stages)

  if mask is not None and mask.shape[-1] != 1:
    if mask.shape[-2] == 1:
      shapes["mask_smem"] = plgpu.SMEM((eltwise_stages, block_kv), jnp.int8)
    else:
      shape = (eltwise_stages, block_q, block_kv)
      shapes["mask_smem"] = _tiled_smem(shape, jnp.int8)
    shapes["mask_produced"] = plgpu.Barrier(num_barriers=eltwise_stages)
    shapes["mask_consumed"] = plgpu.Barrier(num_barriers=eltwise_stages)

  return shapes


def _get_dkv_scratch_shapes(
    config: Config,
    head_dim: int,
    head_dim_out: int,
    chunk_size: int,
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
  eltwise_stages = config.eltwise_stages
  ds_stages = 2 if config.double_buffer else 1
  residual_stages = config.residual_stages
  shapes = dict(
      k_smem=_tiled_smem((block_kv, head_dim), k_dtype),
      v_smem=_tiled_smem((block_kv, head_dim_out), v_dtype),
      q_smem=_tiled_smem((num_stages, block_q, head_dim), q_dtype, swizzle=64),
      do_smem=_tiled_smem(
          (num_stages, block_q, head_dim_out), dout_dtype, swizzle=64
      ),
      ds_smem=_tiled_smem((ds_stages, block_kv, chunk_size), q_dtype),
      p_smem=_tiled_smem((ds_stages, block_kv, chunk_size), dout_dtype),
      s_tmem=plgpu.TMEM((block_kv, block_q), jnp.float32),
      dp_tmem=plgpu.TMEM((block_kv, block_q), jnp.float32),
      dk_tmem=plgpu.TMEM((block_kv, head_dim), jnp.float32),
      dv_tmem=plgpu.TMEM((block_kv, head_dim_out), jnp.float32),
      kv_produced=plgpu.Barrier(num_arrivals=2),
      q_do_produced=plgpu.Barrier(num_barriers=num_stages, num_arrivals=2),
      q_do_consumed=plgpu.Barrier(
          num_barriers=num_stages, num_arrivals=2, orders_tensor_core=True
      ),
      s_produced=plgpu.Barrier(orders_tensor_core=True),
      s_consumed=plgpu.Barrier(),
      p_produced=plgpu.Barrier(num_barriers=ds_stages, orders_tensor_core=True),
      p_consumed=plgpu.Barrier(num_barriers=ds_stages, orders_tensor_core=True),
      dp_produced=plgpu.Barrier(orders_tensor_core=True),
      dp_consumed=plgpu.Barrier(),
      ds_produced=plgpu.Barrier(num_barriers=ds_stages),
      ds_consumed=plgpu.Barrier(
          num_barriers=ds_stages, orders_tensor_core=True
      ),
      kv_mma_finished=plgpu.Barrier(orders_tensor_core=True),
      residuals_smem=plgpu.SMEM((3, residual_stages, block_q), jnp.float32),
      residual_produced=plgpu.Barrier(
          num_barriers=residual_stages, num_arrivals=3
      ),
      residual_consumed=plgpu.Barrier(num_barriers=residual_stages),
  )
  if bias is not None and bias.shape[-2] != 1 and bias.shape[-1] != 1:
    shape = (eltwise_stages, block_kv, block_q)
    swizzle = min(
        plgpu.find_swizzle(block_q * mgpu_lib.num_bits(bias.dtype), "bias"),
        chunk_size * 2,
    )
    shapes["bias_smem"] = _tiled_smem(shape, bias.dtype, swizzle=swizzle)
    shapes["bias_produced"] = plgpu.Barrier(num_barriers=eltwise_stages)
    shapes["bias_consumed"] = plgpu.Barrier(num_barriers=eltwise_stages)

  if mask is not None and mask.shape[-2] != 1 and mask.shape[-1] != 1:
    shape = (eltwise_stages, block_kv, block_q)
    swizzle = min(plgpu.find_swizzle(8 * block_q, "mask"), chunk_size)
    shapes["mask_smem"] = _tiled_smem(shape, jnp.int8, swizzle=swizzle)
    shapes["mask_produced"] = plgpu.Barrier(num_barriers=eltwise_stages)
    shapes["mask_consumed"] = plgpu.Barrier(num_barriers=eltwise_stages)

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
    for double_buffer in (False, True):
      for eltwise_stages in (1, 2):
        for residual_stages in (1, 2):
          for num_stages in (2, 3, 4):
            for chunk_size in (32, 64):
              if q_kv_block_size < chunk_size:
                continue
              config = Config(
                  block_kv_dkv=128,
                  block_q_dkv=q_kv_block_size,
                  block_kv_dq=q_kv_block_size,
                  block_q_dq=128,
                  double_buffer=double_buffer,
                  eltwise_stages=eltwise_stages,
                  residual_stages=residual_stages,
                  num_stages=num_stages,
                  chunk_size=chunk_size,
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
                  chunk_size=config.chunk_size,
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
    return (c.double_buffer, c.num_stages, c.eltwise_stages, c.residual_stages)

  return max(configs, key=_score)


def _get_input_metadata(q, v):
  """Normalizes and returns head dimensions and datatypes."""
  head_dim = pl.cdiv(q.shape[-1], 64) * 64
  head_dim_out = pl.cdiv(v.shape[-1], 64) * 64
  return head_dim, head_dim_out


def _estimate_smem_bytes(scratch_shapes: dict) -> int:
  """Estimates the total SMEM usage in bytes for a given scratch shapes dict."""
  total_bytes = 0
  for val in scratch_shapes.values():
    if isinstance(val, plgpu.RefUnion):
      max_size = 0
      for ref in val.refs:
        if (
            hasattr(ref, "memory_space")
            and getattr(ref.memory_space, "value", "") == "smem"
        ):
          size = math.prod(ref.shape) * jnp.dtype(ref.dtype).itemsize
          max_size = max(max_size, size)
      total_bytes += (max_size + 1023) // 1024 * 1024
    elif (
        hasattr(val, "memory_space")
        and getattr(val.memory_space, "value", "") == "smem"
    ):
      size = math.prod(val.shape) * jnp.dtype(val.dtype).itemsize
      total_bytes += (size + 1023) // 1024 * 1024
    elif isinstance(val, (plgpu.Barrier, plgpu.ClusterBarrier)):
      num_barriers = val.num_barriers
      if isinstance(num_barriers, tuple):
        num_barriers = math.prod(num_barriers)
      total_bytes += num_barriers * 8
  # Add a 4096 byte compiler safety margin.
  return total_bytes + 4096 * 2


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
  eltwise_stages = config.eltwise_stages

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

    @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
    def per_warp():
      warp_id = lax.axis_index("warp")

      @pl.when(warp_id == 0)
      def tma_q_warp():
        plgpu.copy_gmem_to_smem(q_gmem.at[qs, hi], q_smem, q_do_produced)
        plgpu.copy_gmem_to_smem(dout_gmem.at[qs, hi], do_smem, q_do_produced)

      @pl.when(warp_id == 1)
      def tma_kv_warp():
        @pl.loop(lb, lax.min(lb + num_stages, ub))
        def prologue(ki):
          si = lax.rem(ki - lb, num_stages)
          ks = pl.ds(ki * block_kv, block_kv)
          plgpu.copy_gmem_to_smem(
              v_gmem.at[ks, hi_kv], v_smem.at[si], barrier=kv_produced.at[si]
          )
          plgpu.copy_gmem_to_smem(
              k_gmem.at[ks, hi_kv], k_smem.at[si], barrier=kv_produced.at[si]
          )

        @pl.loop(lb + num_stages, ub)
        def kv_loop(ki):
          si = lax.rem(ki - lb, num_stages)
          ks = pl.ds(ki * block_kv, block_kv)
          plgpu.barrier_wait(v_consumed.at[si])
          plgpu.copy_gmem_to_smem(
              v_gmem.at[ks, hi_kv], v_smem.at[si], barrier=kv_produced.at[si]
          )
          plgpu.barrier_wait(k_consumed.at[si])
          plgpu.copy_gmem_to_smem(
              k_gmem.at[ks, hi_kv], k_smem.at[si], barrier=kv_produced.at[si]
          )

      if bias_gmem is not None or mask_gmem is not None:

        @pl.when(warp_id == 3)
        def tma_eltwise_warp():

          @pl.loop(lb, ub)
          def kv_loop(ki):
            si = lax.rem(ki - lb, eltwise_stages)
            ks = pl.ds(ki * block_kv, block_kv)

            if bias_smem is not None:

              @pl.when(ki - lb >= eltwise_stages)
              def wait_bias():
                plgpu.barrier_wait(bias_consumed.at[si])
                mgpu_lib.fence_async_shared_cta()

              plgpu.copy_gmem_to_smem(
                  bias_gmem.at[0 if bias_gmem.shape[-3] == 1 else hi, qs, ks],
                  bias_smem.at[si],
                  bias_produced.at[si],
              )

            if mask_smem is not None:

              @pl.when(ki - lb >= eltwise_stages)
              def wait_mask():
                plgpu.barrier_wait(mask_consumed.at[si])
                mgpu_lib.fence_async_shared_cta()

              mask_hi = 0 if mask_gmem.shape[-3] == 1 else hi
              mask_qs = 0 if mask_gmem.shape[-2] == 1 else qs
              plgpu.copy_gmem_to_smem(
                  mask_gmem.at[mask_hi, mask_qs, ks],
                  mask_smem.at[si],
                  mask_produced.at[si],
              )

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
    l = plgpu.load(l_gmem.at[hi, qs], layout=row_layout, optimized=False)
    delta = plgpu.load(
        delta_gmem.at[hi, qs], layout=row_layout, optimized=False
    )
    m *= math.log2(math.e)

    @pl.loop(lb, ub)
    def kv_loop(ki):
      si = lax.rem(ki - lb, eltwise_stages)
      kv_base = ki * block_kv
      ks = cast(pl.Slice, pl.ds(kv_base, block_kv))

      plgpu.barrier_wait(s_produced)
      s = plgpu.async_load_tmem(s_tmem, layout=layout)
      mgpu_lib.tcgen05_wait_ld()
      plgpu.barrier_arrive(s_consumed)
      scale = logits_scale

      if bias_gmem is not None:
        if bias_smem is None:
          bias = _load_bcast(bias_gmem, (hi, qs, ks), layout=layout)
        else:
          plgpu.barrier_wait(bias_produced.at[si])
          bias = plgpu.load(bias_smem.at[si], layout=layout)
          plgpu.barrier_arrive(bias_consumed.at[si])
        s = s * scale + bias
        scale = 1.0

      if logits_soft_cap is not None:
        s = jnp.tanh(s * (scale / logits_soft_cap))
        scale = logits_soft_cap
      logits = s

      # NOTE: This rescaling must happen after bias and soft-cap but before the
      # attention masking (as the multiplication will cause `-inf`s).
      scale *= math.log2(math.e)
      mask_value = float(jnp.finfo(jnp.float32).min)

      def iota(d):
        return plgpu.broadcasted_iota(jnp.int32, s.shape, d, layout=layout)

      if is_causal:

        def apply_causal_mask():
          is_causal = q_base + iota(0) >= kv_base + iota(1)
          return jnp.where(is_causal, s * scale, mask_value), 1.0

        do_causal = kv_base + block_kv > q_base
        s, scale = lax.cond(do_causal, apply_causal_mask, lambda: (s, scale))

      broadcast = lambda x: lax.broadcast_in_dim(x, s.shape, [0])

      if k_start is not None:
        s *= scale
        s = jnp.where(kv_base + iota(1) >= broadcast(k_start), s, mask_value)
        scale = 1.0

      if k_end is not None:
        s *= scale
        s = jnp.where(kv_base + iota(1) < broadcast(k_end), s, mask_value)
        scale = 1.0

      if mask_gmem is None:
        mask = None
      else:
        if mask_smem is None:
          mask = _load_bcast(mask_gmem, (hi, qs, ks), layout=layout)
        else:
          plgpu.barrier_wait(mask_produced.at[si])
          if mask_smem.ndim == 2:
            mask = plgpu.load(mask_smem.at[si], layout=_TCGEN05_COL)
            mask = lax.broadcast_in_dim(mask, s.shape, [1])
          else:
            mask = plgpu.load(mask_smem.at[si], layout=layout)
          plgpu.barrier_arrive(mask_consumed.at[si])

        s = jnp.where(mask, s * scale, mask_value)
        scale = 1.0

      epsilon = float(jnp.finfo(jnp.float32).tiny)
      p = jnp.exp2(s * scale - broadcast(m)) / (broadcast(l) + epsilon)

      plgpu.barrier_wait(dp_produced)
      dp = plgpu.async_load_tmem(dp_tmem, layout=layout)
      ds = p * (dp - broadcast(delta))

      if logits_soft_cap is not None:
        ds *= 1.0 - logits * logits

      # If we have an attention mask, it is possible that the entire row is
      # masked out. In that case, the forwards pass will calculate `p`'s values
      # as `1 / seq_len_k`. The corresponding `ds` values must be zeroed.
      if mask is not None:
        ds = jnp.where(mask, ds, 0.0)

      if ds_gmem is not None:
        assert ds_smem is not None
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        ds_smem[...] = ds.astype(ds_smem.dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(ds_smem, ds_gmem.at[hi, qs, ks])

      plgpu.async_store_tmem(ds_tmem, ds.astype(ds_tmem.dtype))
      mgpu_lib.tcgen05_wait_st()
      plgpu.barrier_arrive(ds_produced)

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
    ds_smem,
    p_smem,
    s_tmem,
    dp_tmem,
    dk_tmem,
    dv_tmem,
    kv_produced,
    q_do_produced,
    q_do_consumed,
    residual_produced,
    residual_consumed,
    s_produced,
    s_consumed,
    p_produced,
    p_consumed,
    dp_produced,
    dp_consumed,
    ds_produced,
    ds_consumed,
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
  eltwise_stages = config.eltwise_stages
  residual_stages = config.residual_stages
  ds_stages = 2 if config.double_buffer else 1

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

  @pl.when((wg == 0) & (total_steps > 0))
  def mma_tma_wg():

    @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
    def per_warp():
      warp_id = lax.axis_index("warp")

      @pl.when(warp_id == 0)
      def tma_kv_warp():
        plgpu.copy_gmem_to_smem(k_gmem.at[ks, hi_kv], k_smem, kv_produced)
        plgpu.copy_gmem_to_smem(v_gmem.at[ks, hi_kv], v_smem, kv_produced)

        @pl.loop(0, total_steps)
        def q_loop(step):
          si = lax.rem(step, residual_stages)
          qi = lb + lax.rem(step, safe_num_q_tiles)
          qs = pl.ds(qi * block_q, block_q)
          hi = hi_kv * q_heads_per_kv_head + lax.div(step, safe_num_q_tiles)

          @pl.when(step >= residual_stages)
          def wait_res():
            plgpu.barrier_wait(residual_consumed.at[si])

          plgpu.copy_gmem_to_smem(
              m_gmem.at[hi, qs], m_smem.at[si], barrier=residual_produced.at[si]
          )
          plgpu.copy_gmem_to_smem(
              l_gmem.at[hi, qs], l_smem.at[si], barrier=residual_produced.at[si]
          )
          plgpu.copy_gmem_to_smem(
              delta_gmem.at[hi, qs],
              delta_smem.at[si],
              barrier=residual_produced.at[si],
          )

      @pl.when(warp_id == 1)
      def tma_q_warp():
        @pl.loop(0, total_steps)
        def q_loop(step):
          si = lax.rem(step, num_stages)
          qi = lb + lax.rem(step, safe_num_q_tiles)
          qs = pl.ds(qi * block_q, block_q)
          hi = hi_kv * q_heads_per_kv_head + lax.div(step, safe_num_q_tiles)

          @pl.when(step >= num_stages)
          def wait_q():
            plgpu.barrier_wait(q_do_consumed.at[si])

          plgpu.copy_gmem_to_smem(
              q_gmem.at[qs, hi], q_smem.at[si], barrier=q_do_produced.at[si]
          )
          plgpu.copy_gmem_to_smem(
              dout_gmem.at[qs, hi], do_smem.at[si], barrier=q_do_produced.at[si]
          )

      if bias_gmem is not None or mask_gmem is not None:

        @pl.when(warp_id == 3)
        def tma_eltwise_warp():

          @pl.loop(0, total_steps)
          def q_loop(step):
            si = lax.rem(step, eltwise_stages)
            qi = lb + lax.rem(step, safe_num_q_tiles)
            qs = pl.ds(qi * block_q, block_q)
            hi = hi_kv * q_heads_per_kv_head + lax.div(step, safe_num_q_tiles)

            if bias_smem is not None:

              @pl.when(step >= eltwise_stages)
              def wait_bias():
                plgpu.barrier_wait(bias_consumed.at[si])
                mgpu_lib.fence_async_shared_cta()

              plgpu.copy_gmem_to_smem(
                  bias_gmem.at[0 if bias_gmem.shape[-3] == 1 else hi, ks, qs],
                  bias_smem.at[si],
                  bias_produced.at[si],
              )

            if mask_smem is not None:

              @pl.when(step >= eltwise_stages)
              def wait_mask():
                plgpu.barrier_wait(mask_consumed.at[si])
                mgpu_lib.fence_async_shared_cta()

              mask_hi = 0 if mask_gmem.shape[-3] == 1 else hi
              mask_qs = 0 if mask_gmem.shape[-1] == 1 else qs
              plgpu.copy_gmem_to_smem(
                  mask_gmem.at[mask_hi, ks, mask_qs],
                  mask_smem.at[si],
                  mask_produced.at[si],
              )

      @pl.when(warp_id == 2)
      def mma_warp():
        plgpu.barrier_wait(kv_produced)

        @pl.loop(0, total_steps)
        def q_loop(step):
          si = lax.rem(step, num_stages)
          plgpu.barrier_wait(q_do_produced.at[si])
          plgpu.barrier_wait(s_consumed)
          plgpu.tcgen05_mma(s_tmem, k_smem, q_smem.at[si].T, accumulate=False)
          plgpu.tcgen05_commit_arrive(s_produced)

          plgpu.barrier_wait(dp_consumed)
          plgpu.tcgen05_mma(dp_tmem, v_smem, do_smem.at[si].T, accumulate=False)
          plgpu.tcgen05_commit_arrive(dp_produced)

          num_chunks = block_q // config.chunk_size
          for chunk_idx in range(num_chunks):
            gci = step * num_chunks + chunk_idx
            ci = lax.rem(gci, ds_stages)
            c_start = chunk_idx * config.chunk_size
            chunk_slice = pl.ds(c_start, config.chunk_size)
            accumulate = (step > 0) | (chunk_idx > 0)
            plgpu.barrier_wait(p_produced.at[ci])
            plgpu.tcgen05_mma(
                dv_tmem,
                p_smem.at[ci],
                do_smem.at[si, chunk_slice, :],
                accumulate=accumulate,
            )
            plgpu.tcgen05_commit_arrive(p_consumed.at[ci])
            plgpu.barrier_wait(ds_produced.at[ci])
            plgpu.tcgen05_mma(
                dk_tmem,
                ds_smem.at[ci],
                q_smem.at[si, chunk_slice, :],
                accumulate=accumulate,
            )
            plgpu.tcgen05_commit_arrive(ds_consumed.at[ci])

          plgpu.tcgen05_commit_arrive(q_do_consumed.at[si])

        plgpu.barrier_wait(s_consumed)
        plgpu.barrier_wait(dp_consumed)

        plgpu.tcgen05_commit_arrive(kv_mma_finished)

    plgpu.barrier_wait(kv_mma_finished)
    dk = plgpu.async_load_tmem(dk_tmem, layout=_TCGEN05)
    dv = plgpu.async_load_tmem(dv_tmem, layout=_TCGEN05)
    plgpu.wait_load_tmem()
    k_smem[...] = (dk * logits_scale).astype(k_smem.dtype)
    v_smem[...] = dv.astype(v_smem.dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(k_smem, dk_gmem.at[ks, hi_kv])
    plgpu.copy_smem_to_gmem(v_smem, dv_gmem.at[ks, hi_kv])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  @pl.when((wg == 1) & (total_steps > 0))
  def softmax_wg():
    pl.loop(0, ds_stages)(lambda i: plgpu.barrier_arrive(p_consumed.at[i]))
    pl.loop(0, ds_stages)(lambda i: plgpu.barrier_arrive(ds_consumed.at[i]))
    plgpu.barrier_arrive(s_consumed)
    plgpu.barrier_arrive(dp_consumed)

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
      qi = lb + lax.rem(step, safe_num_q_tiles)
      si = lax.rem(step, num_stages)
      si_elt = lax.rem(step, eltwise_stages)
      si_res = lax.rem(step, residual_stages)
      hi = hi_kv * q_heads_per_kv_head + lax.div(step, safe_num_q_tiles)
      plgpu.barrier_wait(q_do_produced.at[si])
      plgpu.barrier_wait(residual_produced.at[si_res])

      num_chunks = block_q // config.chunk_size

      for chunk_idx in range(num_chunks):
        gci = step * num_chunks + chunk_idx
        ci = lax.rem(gci, ds_stages)
        c_start = chunk_idx * config.chunk_size
        chunk_slice = pl.ds(c_start, config.chunk_size)
        q_base = qi * block_q + c_start
        qs = pl.ds(q_base, config.chunk_size)

        if chunk_idx == 0:
          plgpu.barrier_wait(s_produced)
        s = plgpu.async_load_tmem(s_tmem.at[:, chunk_slice], layout=_TCGEN05)
        if chunk_idx == num_chunks - 1:
          plgpu.wait_load_tmem()
          plgpu.barrier_arrive(s_consumed)

        scale = logits_scale

        if bias_gmem is not None:
          if bias_smem is None:
            bias = _load_bcast(bias_gmem, (hi, ks, qs), layout=_TCGEN05)
          else:
            if chunk_idx == 0:
              plgpu.barrier_wait(bias_produced.at[si_elt])
            bias = plgpu.load(
                bias_smem.at[si_elt, :, chunk_slice], layout=_TCGEN05
            )
            if chunk_idx == num_chunks - 1:
              plgpu.barrier_arrive(bias_consumed.at[si_elt])

          s = s * scale + bias
          scale = 1.0

        if logits_soft_cap is not None:
          s = jnp.tanh(s * (scale / logits_soft_cap))
          scale = logits_soft_cap
        logits = s

        m = plgpu.load(m_smem.at[si_res, chunk_slice], layout=_TCGEN05_COL)
        l = plgpu.load(l_smem.at[si_res, chunk_slice], layout=_TCGEN05_COL)

        # NOTE: This rescaling must happen after bias and soft-cap but before
        # the attention masking (as the multiplication will cause `-inf`s).
        scale *= math.log2(math.e)
        m *= math.log2(math.e)

        mask_value = float(jnp.finfo(jnp.float32).min)

        def iota(d):
          return plgpu.broadcasted_iota(jnp.int32, s.shape, d, layout=_TCGEN05)

        if is_causal:

          def apply_causal_mask():
            mask = kv_base + iota(0) <= q_base + iota(1)
            return jnp.where(mask, s * scale, mask_value), 1.0

          do_causal = kv_base + block_kv > q_base
          s, scale = lax.cond(do_causal, apply_causal_mask, lambda: (s, scale))

        broadcast = lambda x, s=s: lax.broadcast_in_dim(x, s.shape, [1])

        def load_k_range(ref):
          hi_ = 0 if ref.shape[0] == 1 else hi
          return plgpu.load(
              ref.at[hi_, qs], layout=_TCGEN05_COL, optimized=False
          )

        if k_start_gmem is not None:
          k_start = broadcast(load_k_range(k_start_gmem))
          s = jnp.where(kv_base + iota(0) >= k_start, s * scale, mask_value)
          scale = 1.0

        if k_end_gmem is not None:
          k_end = broadcast(load_k_range(k_end_gmem))
          s = jnp.where(kv_base + iota(0) < k_end, s * scale, mask_value)
          scale = 1.0

        if mask_gmem is not None:
          if mask_smem is None:
            if loop_invariant_mask is None:
              mask = _load_bcast(mask_gmem, (hi, ks, qs), layout=_TCGEN05)
            else:
              mask = lax.broadcast_in_dim(loop_invariant_mask, s.shape, [0])
          else:
            if chunk_idx == 0:
              plgpu.barrier_wait(mask_produced.at[si_elt])
            mask = plgpu.load(
                mask_smem.at[si_elt, :, chunk_slice], layout=_TCGEN05
            )
            if chunk_idx == num_chunks - 1:
              plgpu.barrier_arrive(mask_consumed.at[si_elt])

          s = jnp.where(mask, s * scale, mask_value)
          scale = 1.0

        epsilon = float(jnp.finfo(jnp.float32).tiny)
        p = jnp.exp2(s * scale - broadcast(m)) / (broadcast(l) + epsilon)

        plgpu.barrier_wait(p_consumed.at[ci])
        p_smem[ci] = p.astype(p_smem.dtype)
        plgpu.commit_smem()
        plgpu.barrier_arrive(p_produced.at[ci])

        delta = plgpu.load(
            delta_smem.at[si_res, chunk_slice], layout=_TCGEN05_COL
        )

        if chunk_idx == 0:
          plgpu.barrier_wait(dp_produced)
        dp = plgpu.async_load_tmem(dp_tmem.at[:, chunk_slice], layout=_TCGEN05)
        if chunk_idx == num_chunks - 1:
          plgpu.wait_load_tmem()
          plgpu.barrier_arrive(dp_consumed)

        ds = p * (dp - broadcast(delta))

        if logits_soft_cap is not None:
          ds *= 1.0 - logits * logits

        plgpu.barrier_wait(ds_consumed.at[ci])
        ds_smem[ci] = ds.astype(ds_smem.dtype)
        plgpu.commit_smem()
        plgpu.barrier_arrive(ds_produced.at[ci])

      plgpu.barrier_arrive(q_do_consumed.at[si])
      plgpu.barrier_arrive(residual_consumed.at[si_res])


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
  chunk_size = config.chunk_size

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
  )(q, k, v, dout, m, l, delta, bias, k_start, k_end, mask)

  dkv_shape = (
      jax.ShapeDtypeStruct(k.shape, k.dtype),
      jax.ShapeDtypeStruct(v.shape, v.dtype),
  )

  dkv_scratch_shapes = _get_dkv_scratch_shapes(
      config=config,
      head_dim=head_dim,
      head_dim_out=head_dim_out,
      chunk_size=chunk_size,
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
  )(q, k, v, dout, m, l, delta, bias_dkv, k_start, k_end, mask_dkv)

  dq = dq[:orig_q_seq_len, :, :orig_head_dim]
  dk = dk[:orig_kv_seq_len, :, :orig_head_dim]
  dv = dv[:orig_kv_seq_len, :, :orig_head_dim_out]
  ds = None if ds is None else ds[:, :orig_q_seq_len, :orig_kv_seq_len]
  return dq, dk, dv, ds
