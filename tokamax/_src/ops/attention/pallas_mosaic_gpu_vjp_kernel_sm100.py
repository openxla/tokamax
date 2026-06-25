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

import math

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_mosaic_gpu_common as common
from tokamax._src.ops.attention import pallas_mosaic_gpu_vjp_common as vjp_common

_SMEM_SIZE_LIMIT = 227 * 1024


_TMEM = plgpu.Layout.TCGEN05
_TMEM_COL = plgpu.Layout.TCGEN05.reduce(0)
_TMEM_ROW = plgpu.Layout.TCGEN05.reduce(1)


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config(vjp_common.Config):
  """Configuration for the VJP.

  Attributes:
    eltwise_stages: The number of pipeline stages for elementwise ops
      (bias/mask).
    double_buffer: Whether to use double buffering for SMEM allocations.
    residual_stages: The number of stages for residual data (m, l, delta).
    chunk_size: The chunk size for processing along the sequence dimension.
    load_residuals_in_regs: Whether to load residuals (m, l, delta) into
      registers instead of SMEM in the dq kernel to save on SMEM.
  """

  eltwise_stages: pydantic.PositiveInt = 1
  double_buffer: bool = False
  residual_stages: pydantic.PositiveInt = 2
  chunk_size: pydantic.conint(multiple_of=32, ge=32) = 64
  load_residuals_in_regs: bool = False


def _get_dq_scratch_shapes(
    config: Config,
    head_dim: int,
    head_dim_out: int,
    chunk_size: int,
    q_dtype,
    dout_dtype,
    k_dtype,
    v_dtype,
    ds_dtype,
    bias_shape,
    bias_dtype,
    mask_shape,
    mask_dtype,
    orig_ds_dtype=None,
):
  ds_stages = 2 if config.double_buffer else 1
  if orig_ds_dtype == jnp.float32:
    ds_smem_ref = plgpu.RefUnion(
        plgpu.SMEM(
            (ds_stages, config.block_q_dq, chunk_size),
            jnp.float32,
            transforms=_smem_transforms(jnp.float32, swizzle=64),
        ),
        plgpu.SMEM(
            (ds_stages, config.block_q_dq, chunk_size),
            ds_dtype,
            transforms=_smem_transforms(ds_dtype, swizzle=64),
        ),
    )
  else:
    ds_smem_ref = plgpu.SMEM(
        (ds_stages, config.block_q_dq, chunk_size),
        ds_dtype,
        transforms=_smem_transforms(ds_dtype, swizzle=64),
    )
  shapes = dict(
      q_smem=plgpu.SMEM(
          (config.block_q_dq, head_dim),
          q_dtype,
          transforms=_smem_transforms(q_dtype, swizzle=64),
      ),
      do_smem=plgpu.SMEM(
          (config.block_q_dq, head_dim_out),
          dout_dtype,
          transforms=_smem_transforms(dout_dtype, swizzle=64),
      ),
      k_smem=plgpu.SMEM(
          (config.num_stages, config.block_kv_dq, head_dim),
          k_dtype,
          transforms=_smem_transforms(k_dtype, swizzle=64),
      ),
      v_smem=plgpu.SMEM(
          (config.num_stages, config.block_kv_dq, head_dim_out),
          v_dtype,
          transforms=_smem_transforms(v_dtype, swizzle=64),
      ),
      ds_smem=ds_smem_ref,
      s_tmem=plgpu.TMEM((config.block_q_dq, config.block_kv_dq), jnp.float32),
      dp_tmem=plgpu.TMEM((config.block_q_dq, config.block_kv_dq), jnp.float32),
      dq_tmem=plgpu.TMEM((config.block_q_dq, head_dim), jnp.float32),
      k_produced=plgpu.Barrier(num_barriers=config.num_stages),
      v_produced=plgpu.Barrier(num_barriers=config.num_stages),
      k_consumed=plgpu.Barrier(
          num_barriers=config.num_stages, orders_tensor_core=True
      ),
      v_consumed=plgpu.Barrier(
          num_barriers=config.num_stages, orders_tensor_core=True
      ),
      s_produced=plgpu.Barrier(orders_tensor_core=True),
      s_consumed=plgpu.Barrier(),
      dp_produced=plgpu.Barrier(orders_tensor_core=True),
      dp_consumed=plgpu.Barrier(),
      ds_produced=plgpu.Barrier(num_barriers=ds_stages),
      ds_consumed=plgpu.Barrier(
          num_barriers=ds_stages, orders_tensor_core=True
      ),
      dq_mma_finished=plgpu.Barrier(orders_tensor_core=True),
  )
  if config.load_residuals_in_regs:
    shapes["q_do_produced"] = plgpu.Barrier(num_arrivals=2)
  else:
    shapes["q_do_produced"] = plgpu.Barrier(num_arrivals=5)
    shapes["residuals_smem"] = plgpu.SMEM((3, config.block_q_dq), jnp.float32)

  shapes.update(
      _add_eltwise_scratch_shapes(
          "bias",
          bias_shape,
          bias_dtype,
          config,
          config.block_q_dq,
          config.block_kv_dq,
          chunk_size * 2,
      )
  )
  shapes.update(
      _add_eltwise_scratch_shapes(
          "mask",
          mask_shape,
          mask_dtype,
          config,
          config.block_q_dq,
          config.block_kv_dq,
          chunk_size,
      )
  )
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
    ds_dtype,
    bias_shape,
    bias_dtype,
    mask_shape,
    mask_dtype,
):
  ds_stages = 2 if config.double_buffer else 1
  shapes = dict(
      k_smem=plgpu.SMEM(
          (config.block_kv_dkv, head_dim),
          k_dtype,
          transforms=_smem_transforms(k_dtype, swizzle=64),
      ),
      v_smem=plgpu.SMEM(
          (config.block_kv_dkv, head_dim_out),
          v_dtype,
          transforms=_smem_transforms(v_dtype, swizzle=64),
      ),
      q_smem=plgpu.SMEM(
          (config.num_stages, config.block_q_dkv, head_dim),
          q_dtype,
          transforms=_smem_transforms(q_dtype, swizzle=64),
      ),
      do_smem=plgpu.SMEM(
          (config.num_stages, config.block_q_dkv, head_dim_out),
          dout_dtype,
          transforms=_smem_transforms(dout_dtype, swizzle=64),
      ),
      ds_smem=plgpu.SMEM(
          (ds_stages, config.block_kv_dkv, chunk_size),
          ds_dtype,
          transforms=_smem_transforms(ds_dtype, swizzle=64),
      ),
      p_smem=plgpu.SMEM(
          (ds_stages, config.block_kv_dkv, chunk_size),
          ds_dtype,
          transforms=_smem_transforms(ds_dtype, swizzle=64),
      ),
      s_tmem=plgpu.TMEM((config.block_kv_dkv, config.block_q_dkv), jnp.float32),
      dp_tmem=plgpu.TMEM(
          (config.block_kv_dkv, config.block_q_dkv), jnp.float32
      ),
      dk_tmem=plgpu.TMEM((config.block_kv_dkv, head_dim), jnp.float32),
      dv_tmem=plgpu.TMEM((config.block_kv_dkv, head_dim_out), jnp.float32),
      kv_produced=plgpu.Barrier(num_arrivals=2),
      q_do_produced=plgpu.Barrier(
          num_barriers=config.num_stages, num_arrivals=2
      ),
      q_do_consumed=plgpu.Barrier(
          num_barriers=config.num_stages,
          num_arrivals=2,
          orders_tensor_core=True,
      ),
      s_produced=plgpu.Barrier(orders_tensor_core=True),
      s_consumed=plgpu.Barrier(),
      dp_produced=plgpu.Barrier(orders_tensor_core=True),
      dp_consumed=plgpu.Barrier(),
      ds_produced=plgpu.Barrier(num_barriers=ds_stages),
      ds_consumed=plgpu.Barrier(
          num_barriers=ds_stages, orders_tensor_core=True
      ),
      kv_mma_finished=plgpu.Barrier(orders_tensor_core=True),
      residuals_smem=plgpu.SMEM(
          (3, config.residual_stages, config.block_q_dkv), jnp.float32
      ),
      residual_produced=plgpu.Barrier(
          num_barriers=config.residual_stages, num_arrivals=3
      ),
      residual_consumed=plgpu.Barrier(num_barriers=config.residual_stages),
  )
  shapes.update(
      _add_eltwise_scratch_shapes(
          "bias",
          bias_shape,
          bias_dtype,
          config,
          config.block_q_dkv,
          config.block_kv_dkv,
          chunk_size * 2,
      )
  )
  shapes.update(
      _add_eltwise_scratch_shapes(
          "mask",
          mask_shape,
          mask_dtype,
          config,
          config.block_q_dkv,
          config.block_kv_dkv,
          chunk_size,
      )
  )
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

  head_dim, head_dim_out, ds_dtype = _get_input_metadata(q, v)


  configs = set()
  min_dq_smem = float("inf")
  min_dkv_smem = float("inf")
  min_total_smem = float("inf")
  fallback_dq_smem = 0
  fallback_dkv_smem = 0
  orig_ds_dtype = bias.dtype if bias is not None else ds_dtype
  q_dtype = _downcast_if_needed(q.dtype, q_k_prec)
  k_dtype = _downcast_if_needed(k.dtype, q_k_prec)
  v_dtype = _downcast_if_needed(v.dtype, v_prec)
  dout_dtype = _downcast_if_needed(dout.dtype, v_prec)

  if bias is not None and ((bias.ndim >= 4 and bias.shape[-4] == 1) or (bias.ndim >= 3 and bias.shape[-3] == 1)):
    orig_ds_dtype = jnp.float32

  for double_buffer in [False, True]:
    for eltwise_stages in [1, 2]:
      for residual_stages in [1, 2]:
        for num_stages in [2, 3, 4]:
          for load_residuals_in_regs in [False, True]:
            for chunk_size in [32, 64]:
              config = Config(
                  block_kv_dkv=128,
                  block_q_dkv=128,
                  block_kv_dq=128,
                  block_q_dq=128,
                  double_buffer=double_buffer,
                  eltwise_stages=eltwise_stages,
                  residual_stages=residual_stages,
                  num_stages=num_stages,
                  load_residuals_in_regs=load_residuals_in_regs,
                  chunk_size=chunk_size,
              )
              dq_shapes = _get_dq_scratch_shapes(
                  config=config,
                  head_dim=head_dim,
                  head_dim_out=head_dim_out,
                  chunk_size=config.chunk_size,
                  q_dtype=q_dtype,
                  dout_dtype=dout_dtype,
                  k_dtype=k_dtype,
                  v_dtype=v_dtype,
                  ds_dtype=ds_dtype,
                  bias_shape=bias.shape if bias is not None else None,
                  bias_dtype=bias.dtype if bias is not None else None,
                  mask_shape=mask.shape if mask is not None else None,
                  mask_dtype=mask.dtype if mask is not None else None,
                  orig_ds_dtype=orig_ds_dtype,
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
                  ds_dtype=ds_dtype,
                  bias_shape=bias.shape if bias is not None else None,
                  bias_dtype=bias.dtype if bias is not None else None,
                  mask_shape=mask.shape if mask is not None else None,
                  mask_dtype=mask.dtype if mask is not None else None,
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


def _kernel(body, out_type, **kernel_kwargs):
  """Interface for SM100 attention VJP kernel.

  Unwraps the custom_vmap() operator because tokamax handles vmap
  already via the batched arguments.

  Args:
    body: The kernel body.
    out_shape: The output shapes.
    **kernel_kwargs: Additional kernel arguments.

  Returns:
    The kernel results.

  """

  if singleton_out := not isinstance(out_type, (tuple, list)):
    out_type = (out_type,)

  @jax.custom_batching.custom_vmap
  def wrapped(*kernel_args):
    @pl.run_state
    def stateful(out_refs):
      def _body(*args, **kwargs):
        body(*args, *out_refs, **kwargs)
      _body.__name__ = body.__name__

      plgpu.kernel(
          _body,
          out_type=(),
          **kernel_kwargs,
      ).fun(*kernel_args)

    out = stateful(
        jax.tree.map(
            lambda s: s
            if isinstance(s, jax.Array)
            else jax.lax.empty(s.shape, s.dtype),
            out_type,
        )
    )
    return tuple(out)

  @wrapped.def_vmap
  def _wrapped_vmap(axis_size, in_batched, *kernel_args):
    # Handles broadcasting for inputs that are not batched across the
    # mapped axis.
    mapped_arrays = []
    none_indices = []
    for i, (batched, arg) in enumerate(zip(in_batched, kernel_args)):
      if arg is None:
        none_indices.append(i)
      elif batched:
        mapped_arrays.append(arg)
      else:
        arg_arr = jnp.asarray(arg)
        mapped_arrays.append(
            jnp.broadcast_to(arg_arr, (axis_size,) + arg_arr.shape)
        )

    def loop_body(arrays_tuple):
      args_list = list(arrays_tuple)
      for i in none_indices:
        args_list.insert(i, None)
      return wrapped(*args_list)

    out = jax.lax.map(loop_body, tuple(mapped_arrays))
    return out, tuple([True] * len(out))

  def unwrap(*args):
    out = wrapped(*args)
    return out[0] if singleton_out else out

  return unwrap


def _pad(x, axis, block, constant_values=0, broadcastable=False):
  """Pads an array to a multiple of `block` along `axis`."""
  if x is None:
    return None
  if axis is None:
    return x

  if isinstance(axis, int):
    axis = (axis,)
    block = (block,)

  padding = [(0, 0)] * x.ndim
  should_pad = False
  for ax, blk in zip(axis, block):
    if broadcastable and x.shape[ax] == 1:
      continue
    if x.shape[ax] % blk == 0:
      continue
    pad_len = blk - (x.shape[ax] % blk)
    padding[ax] = (0, pad_len)
    should_pad = True

  if not should_pad:
    return x

  return jnp.pad(x, padding, constant_values=constant_values)


def _smem_transforms(dtype, swizzle=128):
  return (
      plgpu.TilingTransform((8, swizzle // jnp.dtype(dtype).itemsize)),
      plgpu.SwizzleTransform(swizzle),
  )


def _load_bcast_smem(
    ref, smem, b, hi, elt_bi, s_shape, chunk_slice, is_dq, ref_4d_shape
):
  if smem is None:
    b_m = 0 if ref_4d_shape[-4] == 1 else b
    hi_m = 0 if ref_4d_shape[-3] == 1 else hi
    if len(ref.shape) == 2:
      val_0d = ref[b_m, hi_m]
    else:
      val_0d = ref[b_m, hi_m, 0, 0]
    val = lax.broadcast_in_dim(val_0d, s_shape, [])
  else:
    if ref_4d_shape[-1] == 1:
      if is_dq:
        val_1d = plgpu.load(smem.at[elt_bi], (), layout=_TMEM_ROW)
        val = lax.broadcast_in_dim(val_1d, s_shape, [0])
      else:
        val_1d = plgpu.load(smem.at[elt_bi, chunk_slice], (), layout=_TMEM_COL)
        val = lax.broadcast_in_dim(val_1d, s_shape, [1])
    elif ref_4d_shape[-2] == 1:
      if is_dq:
        val_1d = plgpu.load(smem.at[elt_bi, chunk_slice], (), layout=_TMEM_COL)
        val = lax.broadcast_in_dim(val_1d, s_shape, [1])
      else:
        val_1d = plgpu.load(smem.at[elt_bi], (), layout=_TMEM_ROW)
        val = lax.broadcast_in_dim(val_1d, s_shape, [0])
    else:
      val = plgpu.load(
          smem.at[elt_bi], (slice(None), chunk_slice), layout=_TMEM
      )
  return plgpu.layout_cast(val, _TMEM)


def _get_input_metadata(q, v):
  """Normalizes and returns head dimensions and datatypes."""
  head_dim = pl.cdiv(q.shape[-1], 64) * 64
  head_dim_out = pl.cdiv(v.shape[-1], 64) * 64

  dtype = jnp.dtype(q.dtype)
  # We downcast float32 to float16 because 32-bit types will always exceed
  # the shared memory capacity in the dual kernel.
  if dtype == jnp.float32:
    dtype = jnp.dtype(jnp.float16)

  return head_dim, head_dim_out, dtype


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


def _add_eltwise_scratch_shapes(
    name, shape, dtype, config, block_q, block_kv, swizzle_limit
):
  if shape is None:
    return {}
  smem_q = block_q if shape[-2] != 1 else 1
  smem_kv = block_kv if shape[-1] != 1 else 1
  if smem_q <= 1 and smem_kv <= 1:
    return {}
  smem_shape = (config.eltwise_stages,)
  if smem_q > 1:
    smem_shape += (smem_q,)
  if smem_kv > 1:
    smem_shape += (smem_kv,)
  transforms = (
      _smem_transforms(dtype, swizzle=min(128, swizzle_limit))
      if (smem_q > 1 and smem_kv > 1)
      else ()
  )
  return {
      f"{name}_smem": plgpu.SMEM(smem_shape, dtype, transforms=transforms),
      f"{name}_produced": plgpu.Barrier(num_barriers=config.eltwise_stages),
      f"{name}_consumed": plgpu.Barrier(num_barriers=config.eltwise_stages),
  }


def _kernel_dq(
    q_ref,
    k_ref,
    v_ref,
    dout_ref,
    m_ref,
    l_ref,
    delta_ref,
    bias_ref,
    k_start_ref,
    k_end_ref,
    mask_ref,
    dq_ref,
    ds_ref=None,
    bias_4d_shape=None,
    mask_4d_shape=None,
    *,
    q_smem=None,
    do_smem=None,
    residuals_smem=None,
    k_smem=None,
    v_smem=None,
    ds_smem=None,
    s_tmem=None,
    dp_tmem=None,
    dq_tmem=None,
    q_do_produced,
    k_produced,
    v_produced,
    k_consumed,
    v_consumed,
    s_produced,
    s_consumed,
    dp_produced,
    dp_consumed,
    ds_produced,
    ds_consumed,
    dq_mma_finished,
    bias_smem=None,
    mask_smem=None,
    bias_produced=None,
    bias_consumed=None,
    mask_produced=None,
    mask_consumed=None,
    config: Config,
    is_causal,
    logits_scale,
    logits_soft_cap,
    orig_ds_dtype=None,
    reduce_ds: bool = False,
):
  """Computes dq."""
  ds_stages = 2 if config.double_buffer else 1
  wg_id = lax.axis_index("wg")
  qi = lax.axis_index("q_tiles")
  hi = lax.axis_index("heads")
  b = 0 if q_ref.ndim == 3 else lax.axis_index("batch")

  # We assume MHA or simple mapping here to respect boundaries.
  q_heads_per_kv_head = q_ref.shape[-2] // k_ref.shape[-2]
  hi_kv = lax.div(hi, jnp.array(q_heads_per_kv_head, hi.dtype))

  q_base = qi * config.block_q_dq
  qs = pl.ds(q_base, config.block_q_dq)

  lb = 0
  ub = k_ref.shape[-3] // config.block_kv_dq
  if is_causal:
    ub = lax.min(ub, pl.cdiv(q_base + config.block_q_dq, config.block_kv_dq))

  if residuals_smem is not None:
    # Pack m, l, and delta into the same buffer to avoid SMEM padding overhead.
    # Note that they represent different residuals.
    m_smem = residuals_smem.at[0]
    l_smem = residuals_smem.at[1]
    delta_smem = residuals_smem.at[2]

  @pl.when((wg_id == 0) & (ub > lb))
  def mma_tma_wg():
    plgpu.set_max_registers(112, action="decrease")

    @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
    def per_warp():
      warp_id = lax.axis_index("warp")

      @pl.when(warp_id == 0)
      def tma_q():
        qs = pl.ds(qi * config.block_q_dq, config.block_q_dq)

        plgpu.copy_gmem_to_smem(
            q_ref.at[b, qs, hi] if q_ref.ndim == 4 else q_ref.at[qs, hi],
            q_smem,
            barrier=q_do_produced,
        )
        plgpu.copy_gmem_to_smem(
            dout_ref.at[b, qs, hi]
            if dout_ref.ndim == 4
            else dout_ref.at[qs, hi],
            do_smem,
            barrier=q_do_produced,
        )
        if not config.load_residuals_in_regs:
          plgpu.copy_gmem_to_smem(
              m_ref.at[b, hi, qs] if m_ref.ndim == 3 else m_ref.at[hi, qs],
              m_smem,
              barrier=q_do_produced,
          )
          plgpu.copy_gmem_to_smem(
              l_ref.at[b, hi, qs] if l_ref.ndim == 3 else l_ref.at[hi, qs],
              l_smem,
              barrier=q_do_produced,
          )
          plgpu.copy_gmem_to_smem(
              delta_ref.at[b, hi, qs]
              if delta_ref.ndim == 3
              else delta_ref.at[hi, qs],
              delta_smem,
              barrier=q_do_produced,
          )

      @pl.when(warp_id == 1)
      def tma_kv():
        @pl.loop(lb, lax.min(lb + config.num_stages, ub))
        def prologue(ki):
          si = lax.rem(ki - lb, config.num_stages)
          ks = pl.ds(ki * config.block_kv_dq, config.block_kv_dq)
          plgpu.copy_gmem_to_smem(
              k_ref.at[b, ks, hi_kv]
              if k_ref.ndim == 4
              else k_ref.at[ks, hi_kv],
              k_smem.at[si],
              barrier=k_produced.at[si],
          )
          plgpu.copy_gmem_to_smem(
              v_ref.at[b, ks, hi_kv]
              if v_ref.ndim == 4
              else v_ref.at[ks, hi_kv],
              v_smem.at[si],
              barrier=v_produced.at[si],
          )

        @pl.loop(lb + config.num_stages, ub)
        def kv_loop(ki):
          si = lax.rem(ki - lb, config.num_stages)
          bi = lax.rem(ki - lb, config.eltwise_stages)
          ks = pl.ds(ki * config.block_kv_dq, config.block_kv_dq)
          # BLOCKING: Waiting for MMA to signal that previous tile in this slot is consumed.
          plgpu.barrier_wait(k_consumed.at[si])
          plgpu.copy_gmem_to_smem(
              k_ref.at[b, ks, hi_kv]
              if k_ref.ndim == 4
              else k_ref.at[ks, hi_kv],
              k_smem.at[si],
              barrier=k_produced.at[si],
          )
          plgpu.barrier_wait(v_consumed.at[si])
          plgpu.copy_gmem_to_smem(
              v_ref.at[b, ks, hi_kv]
              if v_ref.ndim == 4
              else v_ref.at[ks, hi_kv],
              v_smem.at[si],
              barrier=v_produced.at[si],
          )

        @pl.loop(lax.max(lb, ub - config.num_stages), ub)
        def kv_epilogue(ki):
          si = lax.rem(ki - lb, config.num_stages)
          plgpu.barrier_wait(k_consumed.at[si])
          plgpu.barrier_wait(v_consumed.at[si])

      @pl.when(warp_id == 3)
      def tma_eltwise():
        if bias_ref is not None or mask_ref is not None:

          @pl.loop(lb, ub)
          def eltwise_loop(ki):
            bi = lax.rem(ki - lb, config.eltwise_stages)
            ks = pl.ds(ki * config.block_kv_dq, config.block_kv_dq)
            if bias_ref is not None:
              if bias_smem is not None:
                b_m = 0 if bias_4d_shape[-4] == 1 else b
                hi_m = 0 if bias_4d_shape[-3] == 1 else hi
                if bias_4d_shape[-1] == 1:
                  if bias_4d_shape[-2] == 1:
                    bias_slice = bias_ref.at[b_m, hi_m]
                  else:
                    bias_slice = bias_ref.at[b_m, hi_m, qs]
                elif bias_4d_shape[-2] == 1:
                  bias_slice = bias_ref.at[b_m, hi_m, 0, ks]
                else:
                  bias_slice = bias_ref.at[b_m, hi_m, qs, ks]

                @pl.when(ki - lb >= config.eltwise_stages)
                def wait_bias():
                  plgpu.barrier_wait(bias_consumed.at[bi])

                plgpu.copy_gmem_to_smem(
                    bias_slice, bias_smem.at[bi], barrier=bias_produced.at[bi]
                )
            if mask_ref is not None:
              if mask_smem is not None:
                b_m = 0 if mask_4d_shape[-4] == 1 else b
                hi_m = 0 if mask_4d_shape[-3] == 1 else hi
                if mask_4d_shape[-1] == 1:
                  if mask_4d_shape[-2] == 1:
                    mask_slice = mask_ref.at[b_m, hi_m]
                  else:
                    mask_slice = mask_ref.at[b_m, hi_m, qs]
                elif mask_4d_shape[-2] == 1:
                  mask_slice = mask_ref.at[b_m, hi_m, 0, ks]
                else:
                  mask_slice = mask_ref.at[b_m, hi_m, qs, ks]

                @pl.when(ki - lb >= config.eltwise_stages)
                def wait_mask():
                  plgpu.barrier_wait(mask_consumed.at[bi])

                plgpu.copy_gmem_to_smem(
                    mask_slice, mask_smem.at[bi], barrier=mask_produced.at[bi]
                )

          @pl.loop(lax.max(lb, ub - config.eltwise_stages), ub)
          def eltwise_epilogue(ki):
            bi = lax.rem(ki - lb, config.eltwise_stages)
            if bias_ref is not None:
              plgpu.barrier_wait(bias_consumed.at[bi])
            if mask_ref is not None:
              plgpu.barrier_wait(mask_consumed.at[bi])

      @pl.when(warp_id == 2)
      def mma():
        plgpu.barrier_wait(q_do_produced)

        @pl.loop(lb, ub)
        def mma_loop(ki):
          si = lax.rem(ki - lb, config.num_stages)
          # BLOCKING: Waiting for TMA to signal that tile data is produced in SMEM.
          plgpu.barrier_wait(k_produced.at[si])
          plgpu.barrier_wait(v_produced.at[si])
          plgpu.barrier_wait(s_consumed)

          plgpu.tcgen05_mma(s_tmem, q_smem, k_smem.at[si].T, accumulate=False)
          plgpu.tcgen05_commit_arrive(s_produced)

          plgpu.barrier_wait(dp_consumed)
          plgpu.tcgen05_mma(dp_tmem, do_smem, v_smem.at[si].T, accumulate=False)
          plgpu.tcgen05_commit_arrive(dp_produced)

          num_chunks = config.block_kv_dq // config.chunk_size
          for chunk_idx in range(num_chunks):
            gci = (ki - lb) * num_chunks + chunk_idx
            ci = lax.rem(gci, ds_stages)
            plgpu.barrier_wait(ds_produced.at[ci])
            c_start = chunk_idx * config.chunk_size
            chunk_slice = pl.ds(c_start, config.chunk_size)

            if orig_ds_dtype == jnp.float32:
              _, ds_smem_bf16 = ds_smem
            else:
              ds_smem_bf16 = ds_smem

            plgpu.tcgen05_mma(
                dq_tmem,
                ds_smem_bf16.at[ci],
                k_smem.at[si, chunk_slice, :],
                accumulate=(ki > lb) | (chunk_idx > 0),
            )
            plgpu.tcgen05_commit_arrive(ds_consumed.at[ci])

          plgpu.tcgen05_commit_arrive(k_consumed.at[si])
          plgpu.tcgen05_commit_arrive(v_consumed.at[si])

        plgpu.barrier_wait(s_consumed)
        plgpu.barrier_wait(dp_consumed)

        plgpu.tcgen05_commit_arrive(dq_mma_finished)

    plgpu.barrier_wait(dq_mma_finished)
    dq_val = plgpu.async_load_tmem(dq_tmem, layout=_TMEM)
    plgpu.wait_load_tmem()
    q_smem[...] = (dq_val * logits_scale).astype(q_smem.dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(
        q_smem,
        dq_ref.at[b, qs, hi] if dq_ref.ndim == 4 else dq_ref.at[qs, hi],
    )
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  @pl.when((wg_id == 1) & (ub > lb))
  def sfu_wg():
    plgpu.set_max_registers(216, action="increase")
    plgpu.barrier_wait(q_do_produced)

    @pl.loop(0, ds_stages)
    def ds_prologue(i):
      plgpu.barrier_arrive(ds_consumed.at[i])

    plgpu.barrier_arrive(s_consumed)
    plgpu.barrier_arrive(dp_consumed)

    if k_start_ref is not None:
      if k_start_ref.ndim == 3:
        b_m = 0 if k_start_ref.shape[-3] == 1 else b
        hi_m = 0 if k_start_ref.shape[-2] == 1 else hi
        k_start_slice = k_start_ref.at[b_m, hi_m, qs]
      else:
        hi_m = 0 if k_start_ref.shape[-2] == 1 else hi
        k_start_slice = k_start_ref.at[hi_m, qs]
      k_start_val = plgpu.load(
          k_start_slice, (), layout=_TMEM_ROW, optimized=False
      )
    if k_end_ref is not None:
      if k_end_ref.ndim == 3:
        b_m = 0 if k_end_ref.shape[-3] == 1 else b
        hi_m = 0 if k_end_ref.shape[-2] == 1 else hi
        k_end_slice = k_end_ref.at[b_m, hi_m, qs]
      else:
        hi_m = 0 if k_end_ref.shape[-2] == 1 else hi
        k_end_slice = k_end_ref.at[hi_m, qs]
      k_end_val = plgpu.load(k_end_slice, (), layout=_TMEM_ROW, optimized=False)

    if config.load_residuals_in_regs:
      if m_ref.ndim == 3:
        m_slice = m_ref.at[b, hi, qs]
        l_slice = l_ref.at[b, hi, qs]
        delta_slice = delta_ref.at[b, hi, qs]
      else:
        m_slice = m_ref.at[hi, qs]
        l_slice = l_ref.at[hi, qs]
        delta_slice = delta_ref.at[hi, qs]
      m_val_full = plgpu.load(m_slice, (), layout=_TMEM_ROW, optimized=False)
      l_val_full = plgpu.load(l_slice, (), layout=_TMEM_ROW, optimized=False)
      delta_val_full = plgpu.load(
          delta_slice, (), layout=_TMEM_ROW, optimized=False
      )
      m_val_full = m_val_full * math.log2(math.e)

    @pl.loop(lb, ub)
    def sfu_loop(ki):
      # BLOCKING: Waiting for MMA to signal that S matmul is complete in TMEM.
      elt_bi = lax.rem(ki - lb, config.eltwise_stages)
      plgpu.barrier_wait(s_produced)
      plgpu.barrier_wait(dp_produced)
      if bias_ref is not None:
        plgpu.barrier_wait(bias_produced.at[elt_bi])
      if mask_ref is not None:
        plgpu.barrier_wait(mask_produced.at[elt_bi])

      if not config.load_residuals_in_regs:
        m_val = plgpu.load(m_smem, (), layout=_TMEM_ROW)
        l_val = plgpu.load(l_smem, (), layout=_TMEM_ROW)
        delta_val = plgpu.load(delta_smem, (), layout=_TMEM_ROW)
        m_val = m_val * math.log2(math.e)
      else:
        m_val = m_val_full
        l_val = l_val_full
        delta_val = delta_val_full

      num_chunks = config.block_kv_dq // config.chunk_size

      m_val_bc = plgpu.layout_cast(
          lax.broadcast_in_dim(
              m_val, (config.block_q_dq, config.chunk_size), [0]
          ),
          _TMEM,
      )
      l_val_bc = plgpu.layout_cast(
          lax.broadcast_in_dim(
              l_val, (config.block_q_dq, config.chunk_size), [0]
          ),
          _TMEM,
      )
      delta_val_bc = plgpu.layout_cast(
          lax.broadcast_in_dim(
              delta_val, (config.block_q_dq, config.chunk_size), [0]
          ),
          _TMEM,
      )

      for chunk_idx in range(num_chunks):
        gci = (ki - lb) * num_chunks + chunk_idx
        ci = lax.rem(gci, ds_stages)
        plgpu.barrier_wait(ds_consumed.at[ci])
        c_start = chunk_idx * config.chunk_size
        chunk_slice = pl.ds(c_start, config.chunk_size)

        s_val = plgpu.async_load_tmem(s_tmem.at[:, chunk_slice], layout=_TMEM)
        dp_val = plgpu.async_load_tmem(dp_tmem.at[:, chunk_slice], layout=_TMEM)
        plgpu.wait_load_tmem()
        if chunk_idx == num_chunks - 1:
          plgpu.barrier_arrive(s_consumed)
          plgpu.barrier_arrive(dp_consumed)

        if bias_ref is not None:
          s_val *= logits_scale
          bias_val = _load_bcast_smem(
              bias_ref,
              bias_smem,
              b,
              hi,
              elt_bi,
              s_val.shape,
              chunk_slice,
              True,
              bias_4d_shape,
          )
          s_val += bias_val

        if logits_soft_cap is not None:
          if bias_ref is None:
            s_val *= logits_scale
          s_val = jnp.tanh(s_val / logits_soft_cap) * logits_soft_cap
          base_val = s_val * math.log2(math.e) - m_val_bc
        else:
          combined_scale = logits_scale * math.log2(math.e)
          if bias_ref is None:
            base_val = s_val * combined_scale - m_val_bc
          else:
            base_val = s_val * math.log2(math.e) - m_val_bc

        def iota(shape, d):
          return plgpu.broadcasted_iota(jnp.int32, shape, d, layout=_TMEM)

        k_iota = iota(base_val.shape, 1) + ki * config.block_kv_dq + c_start
        q_iota = iota(base_val.shape, 0) + q_base
        causal_mask = None
        k_start_val_bc = None
        k_end_val_bc = None

        if is_causal:
          causal_mask = k_iota <= q_iota
          base_val = jnp.where(
              causal_mask, base_val, float(jnp.finfo(jnp.float32).min)
          )

        if k_start_ref is not None:
          k_start_val_bc = lax.broadcast_in_dim(
              k_start_val, base_val.shape, [0]
          )
          k_start_val_bc = plgpu.layout_cast(k_start_val_bc, _TMEM)
          base_val = jnp.where(
              k_iota >= k_start_val_bc,
              base_val,
              float(jnp.finfo(jnp.float32).min),
          )

        if k_end_ref is not None:
          k_end_val_bc = lax.broadcast_in_dim(k_end_val, base_val.shape, [0])
          k_end_val_bc = plgpu.layout_cast(k_end_val_bc, _TMEM)
          base_val = jnp.where(
              k_iota < k_end_val_bc,
              base_val,
              float(jnp.finfo(jnp.float32).min),
          )

        mask_val = None
        if mask_ref is not None:
          mask_val = _load_bcast_smem(
              mask_ref,
              mask_smem,
              b,
              hi,
              elt_bi,
              s_val.shape,
              chunk_slice,
              True,
              mask_4d_shape,
          )
          mask_val = mask_val != 0
          base_val = jnp.where(
              mask_val, base_val, float(jnp.finfo(jnp.float32).min)
          )

        epsilon = float(jnp.finfo(jnp.float32).tiny)
        p_val = jnp.exp2(base_val) / (l_val_bc + epsilon)
        ds_val = p_val * (dp_val - delta_val_bc)

        if logits_soft_cap is not None:
          ds_val *= 1.0 - jnp.square(s_val / logits_soft_cap)

        if is_causal:
          ds_val = jnp.where(causal_mask, ds_val, 0.0)
        if k_start_ref is not None:
          ds_val = jnp.where(k_iota >= k_start_val_bc, ds_val, 0.0)
        if k_end_ref is not None:
          ds_val = jnp.where(k_iota < k_end_val_bc, ds_val, 0.0)
        if mask_ref is not None:
          ds_val = jnp.where(mask_val, ds_val, 0.0)

        if orig_ds_dtype == jnp.float32 and ds_ref is not None:
          ds_smem_f32, ds_smem_bf16 = ds_smem
          ds_smem_f32.at[ci].set(ds_val)
          plgpu.commit_smem()

          red_op = "add" if reduce_ds else None
          if ds_ref.ndim == 4:
            b_m = 0 if ds_ref.shape[-4] == 1 else b
            hi_m = 0 if ds_ref.shape[-3] == 1 else hi
            plgpu.copy_smem_to_gmem(
                ds_smem_f32.at[ci],
                ds_ref.at[
                    b_m,
                    hi_m,
                    qs,
                    pl.ds(ki * config.block_kv_dq + c_start, config.chunk_size),
                ],
                reduction_op=red_op,
            )
          else:
            hi_m = 0 if ds_ref.shape[-3] == 1 else hi
            plgpu.copy_smem_to_gmem(
                ds_smem_f32.at[ci],
                ds_ref.at[
                    hi_m,
                    qs,
                    pl.ds(ki * config.block_kv_dq + c_start, config.chunk_size),
                ],
                reduction_op=red_op,
            )
          plgpu.wait_smem_to_gmem(0, wait_read_only=True)

          ds_smem_bf16.at[ci].set(ds_val.astype(ds_smem_bf16.dtype))
          plgpu.commit_smem()
        else:
          if orig_ds_dtype == jnp.float32:
            _, ds_smem_bf16 = ds_smem
          else:
            ds_smem_bf16 = ds_smem
          if ds_ref is not None:
            ds_smem_bf16.at[ci].set(ds_val.astype(ds_smem_bf16.dtype))
            plgpu.commit_smem()

            red_op = "add" if reduce_ds else None
            if ds_ref.ndim == 4:
              b_m = 0 if ds_ref.shape[-4] == 1 else b
              hi_m = 0 if ds_ref.shape[-3] == 1 else hi
              plgpu.copy_smem_to_gmem(
                  ds_smem_bf16.at[ci],
                  ds_ref.at[
                      b_m,
                      hi_m,
                      qs,
                      pl.ds(
                          ki * config.block_kv_dq + c_start, config.chunk_size
                      ),
                  ],
                  reduction_op=red_op,
              )
            else:
              hi_m = 0 if ds_ref.shape[-3] == 1 else hi
              plgpu.copy_smem_to_gmem(
                  ds_smem_bf16.at[ci],
                  ds_ref.at[
                      hi_m,
                      qs,
                      pl.ds(
                          ki * config.block_kv_dq + c_start, config.chunk_size
                      ),
                  ],
                  reduction_op=red_op,
              )
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)

          ds_smem_bf16.at[ci].set(ds_val.astype(ds_smem_bf16.dtype))
          plgpu.commit_smem()

        plgpu.barrier_arrive(ds_produced.at[ci])

      if bias_ref is not None:
        plgpu.barrier_arrive(bias_consumed.at[elt_bi])
      if mask_ref is not None:
        plgpu.barrier_arrive(mask_consumed.at[elt_bi])

    @pl.loop(0, ds_stages)
    def ds_cleanup(i):
      plgpu.barrier_wait(ds_consumed.at[i])


def _kernel_dkv(
    q_ref,
    k_ref,
    v_ref,
    dout_ref,
    m_ref,
    l_ref,
    delta_ref,
    bias_ref,
    k_start_ref,
    k_end_ref,
    mask_ref,
    dk_ref,
    dv_ref,
    bias_4d_shape=None,
    mask_4d_shape=None,
    *,
    k_smem=None,
    v_smem=None,
    q_smem=None,
    do_smem=None,
    residuals_smem=None,
    ds_smem=None,
    p_smem=None,
    s_tmem=None,
    dp_tmem=None,
    dk_tmem=None,
    dv_tmem=None,
    kv_produced,
    q_do_produced,
    q_do_consumed,
    residual_produced,
    residual_consumed,
    s_produced,
    s_consumed,
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
  ds_stages = 2 if config.double_buffer else 1
  wg_id = lax.axis_index("wg")
  ki = lax.axis_index("kv_tiles")
  hi_kv = lax.axis_index("heads")
  b = 0 if k_ref.ndim == 3 else lax.axis_index("batch")

  num_q_heads = q_ref.shape[-2]
  q_heads_per_kv_head = num_q_heads // k_ref.shape[-2]

  kv_base = ki * config.block_kv_dkv
  ks = pl.ds(kv_base, config.block_kv_dkv)

  lb = lax.div(kv_base, config.block_q_dkv) if is_causal else 0
  ub = q_ref.shape[-3] // config.block_q_dkv
  num_q_tiles = ub - lb
  safe_num_q_tiles = lax.max(num_q_tiles, 1)
  num_chunks = config.block_q_dkv // config.chunk_size
  total_steps = q_heads_per_kv_head * num_q_tiles

  if residuals_smem is not None:
    # Pack m, l, and delta into the same buffer to avoid SMEM padding overhead.
    # Note that they represent different residuals.
    m_smem = residuals_smem.at[0]
    l_smem = residuals_smem.at[1]
    delta_smem = residuals_smem.at[2]

  @pl.when((wg_id == 0) & (total_steps > 0))
  def mma_tma_wg():
    plgpu.set_max_registers(96, action="decrease")

    @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
    def per_warp():
      warp_id = lax.axis_index("warp")

      @pl.when(warp_id == 0)
      def tma_kv():
        plgpu.copy_gmem_to_smem(
            k_ref.at[b, ks, hi_kv] if k_ref.ndim == 4 else k_ref.at[ks, hi_kv],
            k_smem,
            barrier=kv_produced,
        )
        plgpu.copy_gmem_to_smem(
            v_ref.at[b, ks, hi_kv] if v_ref.ndim == 4 else v_ref.at[ks, hi_kv],
            v_smem,
            barrier=kv_produced,
        )

        @pl.loop(0, total_steps)
        def residual_loop(step):
          qh = lax.div(step, safe_num_q_tiles)
          qi = lb + lax.rem(step, safe_num_q_tiles)
          hi = hi_kv * q_heads_per_kv_head + qh
          li = lax.rem(step, config.residual_stages)
          qs = pl.ds(qi * config.block_q_dkv, config.block_q_dkv)

          @pl.when(step >= config.residual_stages)
          def wait_res():
            plgpu.barrier_wait(residual_consumed.at[li])

          plgpu.copy_gmem_to_smem(
              m_ref.at[b, hi, qs] if m_ref.ndim == 3 else m_ref.at[hi, qs],
              m_smem.at[li],
              barrier=residual_produced.at[li],
          )
          plgpu.copy_gmem_to_smem(
              l_ref.at[b, hi, qs] if l_ref.ndim == 3 else l_ref.at[hi, qs],
              l_smem.at[li],
              barrier=residual_produced.at[li],
          )
          plgpu.copy_gmem_to_smem(
              delta_ref.at[b, hi, qs]
              if delta_ref.ndim == 3
              else delta_ref.at[hi, qs],
              delta_smem.at[li],
              barrier=residual_produced.at[li],
          )

        @pl.loop(lax.max(0, total_steps - config.residual_stages), total_steps)
        def residual_epilogue(step):
          li = lax.rem(step, config.residual_stages)
          plgpu.barrier_wait(residual_consumed.at[li])

      @pl.when(warp_id == 1)
      def tma_q():
        @pl.loop(0, total_steps)
        def q_loop(step):
          qh = lax.div(step, safe_num_q_tiles)
          qi = lb + lax.rem(step, safe_num_q_tiles)
          hi = hi_kv * q_heads_per_kv_head + qh
          si = lax.rem(step, config.num_stages)
          qs = pl.ds(qi * config.block_q_dkv, config.block_q_dkv)

          @pl.when(step >= config.num_stages)
          def wait_q():
            plgpu.barrier_wait(q_do_consumed.at[si])

          plgpu.copy_gmem_to_smem(
              q_ref.at[b, qs, hi] if q_ref.ndim == 4 else q_ref.at[qs, hi],
              q_smem.at[si],
              barrier=q_do_produced.at[si],
          )
          plgpu.copy_gmem_to_smem(
              dout_ref.at[b, qs, hi]
              if dout_ref.ndim == 4
              else dout_ref.at[qs, hi],
              do_smem.at[si],
              barrier=q_do_produced.at[si],
          )

        @pl.loop(lax.max(0, total_steps - config.num_stages), total_steps)
        def q_epilogue(step):
          si = lax.rem(step, config.num_stages)
          plgpu.barrier_wait(q_do_consumed.at[si])

      @pl.when(warp_id == 3)
      def tma_eltwise():
        if bias_ref is not None or mask_ref is not None:

          @pl.loop(0, total_steps)
          def eltwise_loop(step):
            qh = lax.div(step, safe_num_q_tiles)
            qi = lb + lax.rem(step, safe_num_q_tiles)
            hi = hi_kv * q_heads_per_kv_head + qh
            bi = lax.rem(step, config.eltwise_stages)
            qs = pl.ds(qi * config.block_q_dkv, config.block_q_dkv)
            if bias_ref is not None:
              if bias_smem is not None:
                b_m = 0 if bias_4d_shape[-4] == 1 else b
                hi_m = 0 if bias_4d_shape[-3] == 1 else hi
                if bias_4d_shape[-1] == 1:
                  if bias_4d_shape[-2] == 1:
                    bias_slice = bias_ref.at[b_m, hi_m]
                  else:
                    bias_slice = bias_ref.at[b_m, hi_m, qs]
                elif bias_4d_shape[-2] == 1:
                  bias_slice = bias_ref.at[b_m, hi_m, 0, ks]
                else:
                  bias_slice = bias_ref.at[b_m, hi_m, ks, qs]

                @pl.when(step >= config.eltwise_stages)
                def wait_bias():
                  plgpu.barrier_wait(bias_consumed.at[bi])

                plgpu.copy_gmem_to_smem(
                    bias_slice, bias_smem.at[bi], barrier=bias_produced.at[bi]
                )
            if mask_ref is not None:
              if mask_smem is not None:
                b_m = 0 if mask_4d_shape[-4] == 1 else b
                hi_m = 0 if mask_4d_shape[-3] == 1 else hi
                if mask_4d_shape[-1] == 1:
                  if mask_4d_shape[-2] == 1:
                    mask_slice = mask_ref.at[b_m, hi_m]
                  else:
                    mask_slice = mask_ref.at[b_m, hi_m, qs]
                elif mask_4d_shape[-2] == 1:
                  mask_slice = mask_ref.at[b_m, hi_m, 0, ks]
                else:
                  mask_slice = mask_ref.at[b_m, hi_m, ks, qs]

                @pl.when(step >= config.eltwise_stages)
                def wait_mask():
                  plgpu.barrier_wait(mask_consumed.at[bi])

                plgpu.copy_gmem_to_smem(
                    mask_slice, mask_smem.at[bi], barrier=mask_produced.at[bi]
                )

          @pl.loop(lax.max(0, total_steps - config.eltwise_stages), total_steps)
          def eltwise_epilogue(step):
            bi = lax.rem(step, config.eltwise_stages)
            if bias_ref is not None:
              plgpu.barrier_wait(bias_consumed.at[bi])
            if mask_ref is not None:
              plgpu.barrier_wait(mask_consumed.at[bi])

      @pl.when(warp_id == 2)
      def mma():
        plgpu.barrier_wait(kv_produced)

        @pl.loop(0, total_steps)
        def mma_loop(step):
          si = lax.rem(step, config.num_stages)
          plgpu.barrier_wait(q_do_produced.at[si])

          plgpu.barrier_wait(s_consumed)
          plgpu.tcgen05_mma(s_tmem, k_smem, q_smem.at[si].T, accumulate=False)
          plgpu.tcgen05_commit_arrive(s_produced)

          plgpu.barrier_wait(dp_consumed)
          plgpu.tcgen05_mma(dp_tmem, v_smem, do_smem.at[si].T, accumulate=False)
          plgpu.tcgen05_commit_arrive(dp_produced)

          num_chunks = config.block_q_dkv // config.chunk_size
          for chunk_idx in range(num_chunks):
            gci = step * num_chunks + chunk_idx
            ci = lax.rem(gci, ds_stages)
            plgpu.barrier_wait(ds_produced.at[ci])
            c_start = chunk_idx * config.chunk_size
            chunk_slice = pl.ds(c_start, config.chunk_size)
            accumulate = (step > 0) | (chunk_idx > 0)
            plgpu.tcgen05_mma(
                dk_tmem,
                ds_smem.at[ci],
                q_smem.at[si, chunk_slice, :],
                accumulate=accumulate,
            )
            plgpu.tcgen05_mma(
                dv_tmem,
                p_smem.at[ci],
                do_smem.at[si, chunk_slice, :],
                accumulate=accumulate,
            )
            plgpu.tcgen05_commit_arrive(ds_consumed.at[ci])

          plgpu.tcgen05_commit_arrive(q_do_consumed.at[si])

        plgpu.barrier_wait(s_consumed)
        plgpu.barrier_wait(dp_consumed)

        plgpu.tcgen05_commit_arrive(kv_mma_finished)

    plgpu.barrier_wait(kv_mma_finished)
    dk_val = plgpu.async_load_tmem(dk_tmem, layout=_TMEM)
    dv_val = plgpu.async_load_tmem(dv_tmem, layout=_TMEM)
    plgpu.wait_load_tmem()
    k_smem[...] = (dk_val * logits_scale).astype(k_smem.dtype)
    v_smem[...] = dv_val.astype(v_smem.dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(
        k_smem,
        dk_ref.at[b, ks, hi_kv] if dk_ref.ndim == 4 else dk_ref.at[ks, hi_kv],
    )
    plgpu.copy_smem_to_gmem(
        v_smem,
        dv_ref.at[b, ks, hi_kv] if dv_ref.ndim == 4 else dv_ref.at[ks, hi_kv],
    )
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  @pl.when((wg_id == 1) & (total_steps > 0))
  def sfu_wg():
    plgpu.set_max_registers(216, action="increase")

    @pl.loop(0, ds_stages)
    def ds_prologue(i):
      plgpu.barrier_arrive(ds_consumed.at[i])

    plgpu.barrier_arrive(s_consumed)
    plgpu.barrier_arrive(dp_consumed)

    @pl.loop(0, total_steps)
    def sfu_loop(step):
      qh = lax.div(step, safe_num_q_tiles)
      qi = lb + lax.rem(step, safe_num_q_tiles)
      si = lax.rem(step, config.num_stages)
      elt_bi = lax.rem(step, config.eltwise_stages)
      li = lax.rem(step, config.residual_stages)
      hi = hi_kv * q_heads_per_kv_head + qh
      plgpu.barrier_wait(q_do_produced.at[si])
      plgpu.barrier_wait(residual_produced.at[li])
      plgpu.barrier_wait(s_produced)
      plgpu.barrier_wait(dp_produced)
      if bias_ref is not None:
        plgpu.barrier_wait(bias_produced.at[elt_bi])
      if mask_ref is not None:
        plgpu.barrier_wait(mask_produced.at[elt_bi])

      qs = pl.ds(qi * config.block_q_dkv, config.block_q_dkv)

      num_chunks = config.block_q_dkv // config.chunk_size

      for chunk_idx in range(num_chunks):
        gci = step * num_chunks + chunk_idx
        ci = lax.rem(gci, ds_stages)
        plgpu.barrier_wait(ds_consumed.at[ci])
        c_start = chunk_idx * config.chunk_size
        chunk_slice = pl.ds(c_start, config.chunk_size)

        s_val = plgpu.async_load_tmem(s_tmem.at[:, chunk_slice], layout=_TMEM)
        dp_val = plgpu.async_load_tmem(dp_tmem.at[:, chunk_slice], layout=_TMEM)
        plgpu.wait_load_tmem()
        if chunk_idx == num_chunks - 1:
          plgpu.barrier_arrive(s_consumed)
          plgpu.barrier_arrive(dp_consumed)

        if bias_ref is not None:
          s_val *= logits_scale
          bias_val = _load_bcast_smem(
              bias_ref,
              bias_smem,
              b,
              hi,
              elt_bi,
              s_val.shape,
              chunk_slice,
              False,
              bias_4d_shape,
          )
          s_val += bias_val

        if logits_soft_cap is not None:
          if bias_ref is None:
            s_val *= logits_scale
          s_val = jnp.tanh(s_val / logits_soft_cap) * logits_soft_cap

        m_val = plgpu.load(m_smem.at[li, chunk_slice], (), layout=_TMEM_COL)
        m_val = lax.broadcast_in_dim(m_val, s_val.shape, [1])
        m_val = plgpu.layout_cast(m_val, _TMEM)
        l_val = plgpu.load(l_smem.at[li, chunk_slice], (), layout=_TMEM_COL)
        l_val = lax.broadcast_in_dim(l_val, s_val.shape, [1])
        l_val = plgpu.layout_cast(l_val, _TMEM)

        if logits_soft_cap is not None or bias_ref is not None:
          base_val = s_val * math.log2(math.e) - m_val * math.log2(math.e)
        else:
          combined_scale = logits_scale * math.log2(math.e)
          base_val = s_val * combined_scale - m_val * math.log2(math.e)

        def iota(shape, d):
          return plgpu.broadcasted_iota(jnp.int32, shape, d, layout=_TMEM)

        k_iota = iota(base_val.shape, 0) + kv_base
        q_iota = iota(base_val.shape, 1) + qi * config.block_q_dkv + c_start
        causal_mask = None
        k_start_val = None
        k_end_val = None

        if is_causal:
          causal_mask = k_iota <= q_iota
          base_val = jnp.where(
              causal_mask, base_val, float(jnp.finfo(jnp.float32).min)
          )

        if k_start_ref is not None:
          if k_start_ref.ndim == 3:
            b_m = 0 if k_start_ref.shape[-3] == 1 else b
            hi_m = 0 if k_start_ref.shape[-2] == 1 else hi
            k_start_slice = k_start_ref.at[
                b_m,
                hi_m,
                pl.ds(qi * config.block_q_dkv + c_start, config.chunk_size),
            ]
          else:
            hi_m = 0 if k_start_ref.shape[-2] == 1 else hi
            k_start_slice = k_start_ref.at[
                hi_m,
                pl.ds(qi * config.block_q_dkv + c_start, config.chunk_size),
            ]
          k_start_val = plgpu.load(
              k_start_slice, (), layout=_TMEM_COL, optimized=False
          )
          k_start_val = lax.broadcast_in_dim(k_start_val, base_val.shape, [1])
          k_start_val = plgpu.layout_cast(k_start_val, _TMEM)
          base_val = jnp.where(
              k_iota >= k_start_val,
              base_val,
              float(jnp.finfo(jnp.float32).min),
          )

        if k_end_ref is not None:
          if k_end_ref.ndim == 3:
            b_m = 0 if k_end_ref.shape[-3] == 1 else b
            hi_m = 0 if k_end_ref.shape[-2] == 1 else hi
            k_end_slice = k_end_ref.at[
                b_m,
                hi_m,
                pl.ds(qi * config.block_q_dkv + c_start, config.chunk_size),
            ]
          else:
            hi_m = 0 if k_end_ref.shape[-2] == 1 else hi
            k_end_slice = k_end_ref.at[
                hi_m,
                pl.ds(qi * config.block_q_dkv + c_start, config.chunk_size),
            ]
          k_end_val = plgpu.load(
              k_end_slice, (), layout=_TMEM_COL, optimized=False
          )
          k_end_val = lax.broadcast_in_dim(k_end_val, base_val.shape, [1])
          k_end_val = plgpu.layout_cast(k_end_val, _TMEM)
          base_val = jnp.where(
              k_iota < k_end_val,
              base_val,
              float(jnp.finfo(jnp.float32).min),
          )

        mask_val = None
        if mask_ref is not None:
          mask_val = _load_bcast_smem(
              mask_ref,
              mask_smem,
              b,
              hi,
              elt_bi,
              s_val.shape,
              chunk_slice,
              False,
              mask_4d_shape,
          )
          mask_val = mask_val != 0
          base_val = jnp.where(
              mask_val, base_val, float(jnp.finfo(jnp.float32).min)
          )

        epsilon = float(jnp.finfo(jnp.float32).tiny)
        p_val = jnp.exp2(base_val) / (l_val + epsilon)

        delta_val = plgpu.load(
            delta_smem.at[li, chunk_slice], (), layout=_TMEM_COL
        )
        delta_val = lax.broadcast_in_dim(delta_val, p_val.shape, [1])
        delta_val = plgpu.layout_cast(delta_val, _TMEM)
        ds_val = p_val * (dp_val - delta_val)

        if logits_soft_cap is not None:
          ds_val *= 1.0 - jnp.square(s_val / logits_soft_cap)

        if is_causal:
          ds_val = jnp.where(causal_mask, ds_val, 0.0)
          p_val = jnp.where(causal_mask, p_val, 0.0)
        if k_start_ref is not None:
          ds_val = jnp.where(k_iota >= k_start_val, ds_val, 0.0)
          p_val = jnp.where(k_iota >= k_start_val, p_val, 0.0)
        if k_end_ref is not None:
          ds_val = jnp.where(k_iota < k_end_val, ds_val, 0.0)
          p_val = jnp.where(k_iota < k_end_val, p_val, 0.0)
        if mask_ref is not None:
          ds_val = jnp.where(mask_val, ds_val, 0.0)
          p_val = jnp.where(mask_val, p_val, 0.0)

        ds_smem.at[ci].set(ds_val.astype(ds_smem.dtype))
        p_smem.at[ci].set(p_val.astype(p_smem.dtype))
        plgpu.commit_smem()
        plgpu.barrier_arrive(ds_produced.at[ci])

      if bias_ref is not None:
        plgpu.barrier_arrive(bias_consumed.at[elt_bi])
      if mask_ref is not None:
        plgpu.barrier_arrive(mask_consumed.at[elt_bi])
      plgpu.barrier_arrive(q_do_consumed.at[si])
      plgpu.barrier_arrive(residual_consumed.at[li])

    @pl.loop(0, ds_stages)
    def ds_cleanup(i):
      plgpu.barrier_wait(ds_consumed.at[i])


@jaxtyping.jaxtyped
def flash_attention_vjp_kernel(
    q: Float[Array, "*B T H D"],
    k: Float[Array, "*B t h D"],
    v: Float[Array, "*B t h d"],
    residuals: base.Residuals,
    out: Float[Array, "*B T H d"],
    dout: Float[Array, "*B T H d"],
    bias: Float[Array, "*#B #H #T #t"] | None,
    mask: Bool[Array, "*#B #H #T #t"] | None,
    k_start: Int[Array, "*#B #H #T"] | None,
    k_end: Int[Array, "*#B #H #T"] | None,
    *,
    logits_scale: float,
    logits_soft_cap: float | None,
    is_causal: bool,
    ds_dtype: jax.typing.DTypeLike | None,
    config: Config,
) -> tuple[
    Float[Array, "*B T H D"],  # dq
    Float[Array, "*B t h D"],  # dk
    Float[Array, "*B t h d"],  # dv
    Float[Array, "*#B #H #T #t"] | None,  # ds
]:
  """SM100 Pallas Mosaic GPU Flash Attention VJP."""
  *_, orig_q_seq_len, _, orig_head_dim = q.shape
  *_, orig_kv_seq_len, _, orig_head_dim_out = v.shape

  block_q_dq = config.block_q_dq
  block_kv_dkv = config.block_kv_dkv
  chunk_size = config.chunk_size

  batch_shape = q.shape[:-3]
  orig_bias_shape = bias.shape if bias is not None else None

  def _reshape_4d(arr, core_ndim):
    if arr is None:
      return None
    return arr.reshape(-1, *arr.shape[-core_ndim:])

  def _squeeze_trailing_1s(arr):
    if arr is None:
      return None
    while len(arr.shape) > 1 and arr.shape[-1] == 1:
      arr = arr[..., 0]
    return arr

  q, k, v, out, dout = [_reshape_4d(x, 3) for x in (q, k, v, out, dout)]
  m, l = [_reshape_4d(x, 2) for x in residuals]
  residuals = (m, l)
  bias, mask = [_reshape_4d(x, 3) for x in (bias, mask)]
  bias_4d_shape = bias.shape if bias is not None else None
  k_start, k_end = [_reshape_4d(x, 2) for x in (k_start, k_end)]

  assert q is not None  # To make pytype happy
  orig_ds_dtype = (
      ds_dtype
      if ds_dtype is not None
      else (bias.dtype if bias is not None else q.dtype)
  )

  # If there are many elements in the batch or head dimensions, we accumulate
  # from multiple thread blocks, which requires float32 reduction for precision.
  if bias is not None and (bias_4d_shape[-4] == 1 or bias_4d_shape[-3] == 1):
    orig_ds_dtype = jnp.float32

  # TODO: Remove explicit padding in favor of TMA out-of-bounds zero-filling and in-kernel -inf masking.
  q = _pad(q, -3, block_q_dq)
  out = _pad(out, -3, block_q_dq)
  dout = _pad(dout, -3, block_q_dq)
  k = _pad(k, -3, block_kv_dkv)
  v = _pad(v, -3, block_kv_dkv)

  # TODO: Avoid broadcast.
  bcast = lambda x: jnp.broadcast_to(
      x, (*x.shape[:-2], q.shape[-2], orig_q_seq_len)
  )
  k_start = None if k_start is None else bcast(k_start)
  k_end = None if k_end is None else bcast(k_end)
  if mask is not None:
    # Mask shape is usually broadcasted
    mask = mask.astype(jnp.int8)

  if mask is not None:
    mask = _pad(mask, -2, block_q_dq, broadcastable=True)
    mask = _pad(mask, -1, block_kv_dkv, broadcastable=True)
  if bias is not None:
    bias = _pad(bias, -2, block_q_dq, broadcastable=True)
    bias = _pad(bias, -1, block_kv_dkv, broadcastable=True)
  if k_start is not None:
    k_start = _pad(k_start, -1, block_q_dq)
  if k_end is not None:
    k_end = _pad(k_end, -1, block_q_dq)

  bias_4d_shape = bias.shape if bias is not None else None
  mask_4d_shape = mask.shape if mask is not None else None

  bias = _squeeze_trailing_1s(bias)
  mask = _squeeze_trailing_1s(mask)

  pad_dim = lambda x: _pad(x, -1, 64)
  q, k, v, out, dout = map(pad_dim, (q, k, v, out, dout))

  # ds_dtype must match `dtype` (compute precision) for TCGEN05 MMA
  # compatibility.
  head_dim, head_dim_out, ds_dtype = _get_input_metadata(q, v)

  m, l = residuals
  m = _pad(m, -1, block_q_dq, constant_values=1e9)
  l = _pad(l, -1, block_q_dq, constant_values=1)

  delta = jnp.einsum(
      "...qhd,...qhd->...hq", out.astype(jnp.float32), dout.astype(jnp.float32)
  )
  delta = _pad(delta, -1, block_q_dq)

  compiler_params = plgpu.CompilerParams(
      approx_math=True,
      unsafe_no_auto_barriers=True,
      reduction_scratch_bytes=0,
  )

  dq_shape = jax.ShapeDtypeStruct(q.shape, q.dtype)

  dq_scratch_shapes = _get_dq_scratch_shapes(
      config=config,
      head_dim=head_dim,
      head_dim_out=head_dim_out,
      chunk_size=chunk_size,
      q_dtype=q.dtype,
      dout_dtype=dout.dtype,
      k_dtype=k.dtype,
      v_dtype=v.dtype,
      ds_dtype=ds_dtype,
      bias_shape=bias_4d_shape,
      bias_dtype=bias.dtype if bias is not None else None,
      mask_shape=mask_4d_shape,
      mask_dtype=mask.dtype if mask is not None else None,
      orig_ds_dtype=orig_ds_dtype,
  )

  if bias is None:
    dq_out_shape = dq_shape

    def dq_body(
        q_ref,
        k_ref,
        v_ref,
        dout_ref,
        m_ref,
        l_ref,
        delta_ref,
        bias_ref,
        k_start_ref,
        k_end_ref,
        mask_ref,
        dq_ref,
        **scratches,
    ):
      return _kernel_dq(
          q_ref,
          k_ref,
          v_ref,
          dout_ref,
          m_ref,
          l_ref,
          delta_ref,
          bias_ref,
          k_start_ref,
          k_end_ref,
          mask_ref,
          dq_ref,
          bias_4d_shape=bias_4d_shape,
          mask_4d_shape=mask_4d_shape,
          ds_ref=None,
          **scratches,
          config=config,
          is_causal=is_causal,
          logits_scale=logits_scale,
          logits_soft_cap=logits_soft_cap,
          orig_ds_dtype=orig_ds_dtype,
          reduce_ds=False,
      )

    dq = _kernel(
        dq_body,
        out_type=dq_out_shape,
        kernel_name="sm100_dq_kernel",
        grid=(
            q.shape[-4] if q.ndim == 4 else 1,
            q.shape[-2],
            q.shape[-3] // block_q_dq,
        ),
        grid_names=("batch", "heads", "q_tiles"),
        num_threads=2,
        thread_name="wg",
        compiler_params=compiler_params,
        scratch_types=dq_scratch_shapes,
    )(q, k, v, dout, m, l, delta, bias, k_start, k_end, mask)
    ds = None
  else:
    q_seq_len_ = pl.cdiv(q.shape[-3], config.block_q_dq) * config.block_q_dq
    kv_seq_len_ = pl.cdiv(k.shape[-3], config.block_kv_dq) * config.block_kv_dq
    full_ds_shape = (q.shape[-4], q.shape[-2], q_seq_len_, kv_seq_len_)

    b_size = bias_4d_shape[-4] if bias_4d_shape[-4] == 1 else full_ds_shape[-4]
    h_size = bias_4d_shape[-3] if bias_4d_shape[-3] == 1 else full_ds_shape[-3]
    ds_out_shape = (b_size, h_size, full_ds_shape[2], full_ds_shape[3])

    reduce_ds = ds_out_shape != full_ds_shape
    ds_shape_or_array = (
        jnp.zeros(ds_out_shape, orig_ds_dtype)
        if reduce_ds
        else jax.ShapeDtypeStruct(ds_out_shape, orig_ds_dtype)
    )
    dq_out_shape = (dq_shape, ds_shape_or_array)

    def dq_body_bias(
        q_ref,
        k_ref,
        v_ref,
        dout_ref,
        m_ref,
        l_ref,
        delta_ref,
        bias_ref,
        k_start_ref,
        k_end_ref,
        mask_ref,
        dq_ref,
        ds_ref,
        **scratches,
    ):
      return _kernel_dq(
          q_ref,
          k_ref,
          v_ref,
          dout_ref,
          m_ref,
          l_ref,
          delta_ref,
          bias_ref,
          k_start_ref,
          k_end_ref,
          mask_ref,
          dq_ref,
          bias_4d_shape=bias_4d_shape,
          mask_4d_shape=mask_4d_shape,
          ds_ref=ds_ref,
          **scratches,
          config=config,
          is_causal=is_causal,
          logits_scale=logits_scale,
          logits_soft_cap=logits_soft_cap,
          orig_ds_dtype=orig_ds_dtype,
          reduce_ds=reduce_ds,
      )

    dq, ds = _kernel(
        dq_body_bias,
        out_type=dq_out_shape,
        kernel_name="sm100_dq_kernel_bias",
        grid=(
            q.shape[-4] if q.ndim == 4 else 1,
            q.shape[-2],
            q.shape[-3] // block_q_dq,
        ),
        grid_names=("batch", "heads", "q_tiles"),
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
      ds_dtype=ds_dtype,
      bias_shape=bias_4d_shape,
      bias_dtype=bias.dtype if bias is not None else None,
      mask_shape=mask_4d_shape,
      mask_dtype=mask.dtype if mask is not None else None,
  )

  # Only transpose 2D arrays; transposing 1D arrays creates invalid
  # strides for TMA.
  bias_dkv = (
      bias.mT
      if (
          bias is not None and bias_4d_shape[-1] != 1 and bias_4d_shape[-2] != 1
      )
      else bias
  )
  mask_dkv = (
      mask.mT
      if (
          mask is not None and mask_4d_shape[-1] != 1 and mask_4d_shape[-2] != 1
      )
      else mask
  )

  def dkv_body(
      q_ref,
      k_ref,
      v_ref,
      dout_ref,
      m_ref,
      l_ref,
      delta_ref,
      bias_ref,
      k_start_ref,
      k_end_ref,
      mask_ref,
      dk_ref,
      dv_ref,
      **scratches,
  ):
    return _kernel_dkv(
        q_ref,
        k_ref,
        v_ref,
        dout_ref,
        m_ref,
        l_ref,
        delta_ref,
        bias_ref,
        k_start_ref,
        k_end_ref,
        mask_ref,
        dk_ref,
        bias_4d_shape=bias_4d_shape,
        mask_4d_shape=mask_4d_shape,
        dv_ref=dv_ref,
        config=config,
        is_causal=is_causal,
        logits_scale=logits_scale,
        logits_soft_cap=logits_soft_cap,
        **scratches,
    )

  dk, dv = _kernel(
      dkv_body,
      out_type=dkv_shape,
      kernel_name="sm100_dkv_kernel",
      grid=(
          q.shape[-4] if q.ndim == 4 else 1,
          k.shape[-2],
          k.shape[-3] // block_kv_dkv,
      ),
      grid_names=("batch", "heads", "kv_tiles"),
      num_threads=2,
      thread_name="wg",
      compiler_params=compiler_params,
      scratch_types=dkv_scratch_shapes,
  )(q, k, v, dout, m, l, delta, bias_dkv, k_start, k_end, mask_dkv)

  dq = dq[..., :orig_q_seq_len, :, :orig_head_dim]
  dk = dk[..., :orig_kv_seq_len, :, :orig_head_dim]
  dv = dv[..., :orig_kv_seq_len, :, :orig_head_dim_out]

  dq = dq.reshape(*batch_shape, *dq.shape[-3:])
  dk = dk.reshape(*batch_shape, *dk.shape[-3:])
  dv = dv.reshape(*batch_shape, *dv.shape[-3:])

  if ds is not None:
    ds = ds[..., :orig_q_seq_len, :orig_kv_seq_len]
    if is_causal:
      q_idx = jnp.arange(orig_q_seq_len)[:, None]
      k_idx = jnp.arange(orig_kv_seq_len)[None, :]
      ds = jnp.where(q_idx >= k_idx, ds, 0.0)

    if orig_bias_shape is not None:
      broadcast_axes = []
      if bias_4d_shape[-2] == 1:
        broadcast_axes.append(2)
      if bias_4d_shape[-1] == 1:
        broadcast_axes.append(3)
      if broadcast_axes:
        # Reduce sequence dimensions Python-side if they were
        # broadcasted in the original bias.
        ds = jnp.sum(ds, axis=tuple(broadcast_axes), keepdims=True)
    ds = ds.reshape(orig_bias_shape).astype(orig_ds_dtype)
  return dq, dk, dv, ds
