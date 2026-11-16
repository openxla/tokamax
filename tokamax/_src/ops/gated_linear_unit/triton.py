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
"""Triton gated linear unit."""

from collections.abc import Callable
import dataclasses
import math
from typing import ClassVar, override

import jax
from jax.extend import backend
import jax.numpy as jnp
import jax_triton as jt
from jaxtyping import Array, Float  # pylint: disable=g-importing-member,g-multiple-import
import pydantic
from tokamax._src import gpu_utils
from tokamax._src import jaxtyping
from tokamax._src import pydantic as pydantic_lib
from tokamax._src import triton_utils
from tokamax._src.ops import op
from tokamax._src.ops.gated_linear_unit import base
from tokamax._src.ops.gated_linear_unit.base import FusedWeights, UnfusedWeights  # pylint: disable=g-importing-member,g-multiple-import
import triton
import triton.language as tl

Residuals = base.Residuals


@triton.jit
def _and_masks(a, b):
  if a is None:
    return b
  elif b is None:
    return a
  else:
    return a & b


@triton.jit
def _gated_linear_unit_kernel(
    x_ptr,
    w_ptr,
    v_ptr,
    out_ptr,
    res_ptr,
    batch_shape: tl.constexpr,
    batch_strides: tl.constexpr,
    is_batched: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    x_strides: tl.constexpr,
    w_strides: tl.constexpr,
    v_strides: tl.constexpr,
    out_strides: tl.constexpr,
    res_strides: tl.constexpr,
    grid_m: tl.constexpr,
    grid_n: tl.constexpr,
    group_size: tl.constexpr,
    get_pids_fn: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
    activation: tl.constexpr,
    return_residuals: tl.constexpr,
    use_int64_k_step: tl.constexpr,
):
  """Triton GLU kernel."""
  pid = tl.program_id(axis=0)
  if is_batched:
    num_mn_blocks: tl.constexpr = grid_m * grid_n
    pid_b = (pid // num_mn_blocks).to(tl.int64)
    pid = pid % num_mn_blocks
    x_bstrides, w_bstrides, v_bstrides = batch_strides
    x_ptr += triton_utils.batch_offset(pid_b, batch_shape, x_bstrides)
    w_ptr += triton_utils.batch_offset(pid_b, batch_shape, w_bstrides)
    if v_ptr is not None:
      v_ptr += triton_utils.batch_offset(pid_b, batch_shape, v_bstrides)
    out_ptr += pid_b * m * n
    if return_residuals:
      res_ptr += pid_b * m * 2 * n
  pid_m, pid_n = get_pids_fn(pid, grid_m, grid_n, group_size)

  offs_m = (
      pid_m.to(tl.int64) if grid_m * block_m >= 2**31 else pid_m
  ) * block_m + tl.arange(0, block_m)
  offs_n = (
      pid_n.to(tl.int64) if grid_n * block_n >= 2**31 else pid_n
  ) * block_n + tl.arange(0, block_n)
  offs_k = tl.arange(0, block_k)

  stride_xm: tl.constexpr = x_strides[0]
  stride_xk: tl.constexpr = x_strides[1]
  stride_om: tl.constexpr = out_strides[0]
  stride_on: tl.constexpr = out_strides[1]

  x_ptrs = (
      x_ptr
      + triton_utils.row_offset(offs_m, m, stride_xm)
      + offs_k[None, :] * stride_xk
  )
  if v_ptr is None:
    stride_wk: tl.constexpr = w_strides[0]
    stride_wg: tl.constexpr = w_strides[1]
    stride_wn: tl.constexpr = w_strides[2]
    stride_vk: tl.constexpr = stride_wk
    w_ptrs = (
        w_ptr
        + triton_utils.row_offset(offs_k, 0, stride_wk)
        + offs_n[None, :] * stride_wn
    )
    v_ptrs = w_ptrs + stride_wg
  else:
    stride_wk: tl.constexpr = w_strides[0]
    stride_wn: tl.constexpr = w_strides[1]
    stride_vk: tl.constexpr = v_strides[0]
    stride_vn: tl.constexpr = v_strides[1]
    w_ptrs = (
        w_ptr
        + triton_utils.row_offset(offs_k, 0, stride_wk)
        + offs_n[None, :] * stride_wn
    )
    v_ptrs = (
        v_ptr
        + triton_utils.row_offset(offs_k, 0, stride_vk)
        + offs_n[None, :] * stride_vn
    )

  m_mask = (offs_m[:, None] < m) if m % block_m != 0 else None
  n_mask = (offs_n[None, :] < n) if n % block_n != 0 else None
  gates = tl.zeros((block_m, block_n), dtype=tl.float32)
  proj = tl.zeros((block_m, block_n), dtype=tl.float32)

  x_k_step = (
      tl.full((), block_k * stride_xk, tl.int64)
      if use_int64_k_step
      else block_k * stride_xk
  )
  w_k_step = (
      tl.full((), block_k * stride_wk, tl.int64)
      if use_int64_k_step
      else block_k * stride_wk
  )
  v_k_step = (
      tl.full((), block_k * stride_vk, tl.int64)
      if use_int64_k_step
      else block_k * stride_vk
  )

  for start_k in range(0, k, block_k):
    start_k = tl.multiple_of(start_k, block_k)
    k_mask = ((start_k + offs_k) < k) if k % block_k != 0 else None
    x_mask = _and_masks(m_mask, None if k_mask is None else k_mask[None, :])
    w_mask = _and_masks(None if k_mask is None else k_mask[:, None], n_mask)
    x = tl.load(x_ptrs, mask=x_mask, other=None if x_mask is None else 0.0)
    w = tl.load(w_ptrs, mask=w_mask, other=None if w_mask is None else 0.0)
    v = tl.load(v_ptrs, mask=w_mask, other=None if w_mask is None else 0.0)

    x = x.to(dot_dtype)
    w = w.to(dot_dtype)
    v = v.to(dot_dtype)

    gates = triton_utils.dot(x, w, acc=gates, input_precision=input_precision)
    proj = triton_utils.dot(x, v, acc=proj, input_precision=input_precision)

    x_ptrs += x_k_step
    w_ptrs += w_k_step
    v_ptrs += v_k_step

  out_mask = _and_masks(m_mask, n_mask)

  if return_residuals:
    stride_res_m: tl.constexpr = res_strides[0]
    stride_res_g: tl.constexpr = res_strides[1]
    stride_res_n: tl.constexpr = res_strides[2]
    res_ptrs_0 = (
        res_ptr
        + triton_utils.row_offset(offs_m, m, stride_res_m)
        + offs_n[None, :] * stride_res_n
    )
    res_ptrs_1 = res_ptrs_0 + stride_res_g
    tl.store(res_ptrs_0, gates.to(res_ptr.dtype.element_ty), mask=out_mask)
    tl.store(res_ptrs_1, proj.to(res_ptr.dtype.element_ty), mask=out_mask)

  if activation is not None:
    gates = activation(gates)

  out = proj * gates
  out_ptrs = (
      out_ptr
      + triton_utils.row_offset(offs_m, m, stride_om)
      + offs_n[None, :] * stride_on
  )
  tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=out_mask)


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_m: pydantic_lib.PowerOfTwo
  block_n: pydantic_lib.PowerOfTwo
  block_k: pydantic_lib.PowerOfTwo
  num_warps: pydantic_lib.PowerOfTwo
  num_stages: pydantic.PositiveInt


def _unbatched_weights_x_shape(x, weights) -> tuple[int, ...]:
  """Returns effective `x` shape after collapsing unbatched-weights vmap axes."""
  x_vmap_axes = getattr(x, 'vmap_axes', ())
  if not x_vmap_axes:
    return x.shape
  w_leaves = jax.tree.leaves(weights)
  shape = list(x.shape)
  for i in range(len(x_vmap_axes) - 1, -1, -1):
    x_axis = x_vmap_axes[i]
    w_batched = any(
        i < len(w_axes := getattr(w, 'vmap_axes', ()))
        and w_axes[i] is not None
        for w in w_leaves
    )
    if x_axis is not None and not w_batched:
      shape.insert(*x_axis)
  return tuple(shape)


def _max_smem_bytes() -> int:
  if gpu_utils.is_sm90() or gpu_utils.is_sm100():
    return 224 * 1024
  return 160 * 1024


def _smem_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    x_itemsize: int,
    w_pair_itemsize: int,
) -> int:
  return (
      max(1, num_stages)
      * block_k
      * (block_m * x_itemsize + block_n * w_pair_itemsize)
  )


def _is_valid_config(
    cfg: Config,
    *,
    x_itemsize: int,
    w_pair_itemsize: int,
    input_precision: str | None,
    max_smem: int,
) -> bool:
  is_multi_or_simt = input_precision in (
      'ieee',
      'bf16x3',
      'bf16x6',
      'tf32x3',
  )
  max_area = 64 * 128 if is_multi_or_simt else 128 * 128
  if cfg.block_m * cfg.block_n > max_area:
    return False
  if cfg.block_m * cfg.block_n >= 128 * 128 and cfg.num_warps < 8:
    return False
  if (
      _smem_bytes(
          cfg.block_m,
          cfg.block_n,
          cfg.block_k,
          cfg.num_stages,
          x_itemsize,
          w_pair_itemsize,
      )
      > max_smem
  ):
    return False
  return True


def _extract_problem_info(
    ba: op.BoundArguments,
) -> tuple[int, int, int, int, int, str | None]:
  """Extracts `(m, n, k, x_itemsize, w_pair_itemsize, input_precision)`."""
  x = ba.arguments['x']
  weights = ba.arguments['weights']
  x_shape = _unbatched_weights_x_shape(x, weights)
  m = math.prod(x_shape[:-1])
  k = x_shape[-1]
  w_leaves = jax.tree.leaves(weights)
  n = w_leaves[0].shape[-1]
  x_itemsize = jnp.dtype(x.dtype).itemsize
  w_pair_itemsize = (
      sum(jnp.dtype(w.dtype).itemsize for w in w_leaves)
      if isinstance(weights, tuple)
      else 2 * jnp.dtype(weights.dtype).itemsize
  )
  rhs_dtype = jnp.result_type(*(w.dtype for w in w_leaves))
  try:
    _, input_precision = triton_utils.get_dot_dtype_and_precision(
        ba.arguments['precision'], x.dtype, rhs_dtype
    )
  except (NotImplementedError, KeyError):
    input_precision = None
  return m, n, k, x_itemsize, w_pair_itemsize, input_precision


@dataclasses.dataclass(frozen=True, slots=True)
class TritonGatedLinearUnit(base.GatedLinearUnit[Config, None]):
  """Triton gated linear unit."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      x: Float[Array, '*B M K'],
      weights: FusedWeights | UnfusedWeights,
      *,
      activation: Callable[[jax.Array], jax.Array] | None,
      precision: base.CanonicalPrecision,
      return_residuals: bool,
      config: Config,
  ) -> tuple[Float[Array, '*B M N'], Residuals | None]:
    supported_dtypes = {jnp.float16, jnp.bfloat16, jnp.float32}
    for arr in (x, *jax.tree.leaves(weights)):
      if arr.dtype.type not in supported_dtypes:
        raise NotImplementedError(
            f'Triton kernel does not support input datatype {arr.dtype}. Must'
            f' be one of {supported_dtypes}.'
        )

    block_m = config.block_m
    block_n = config.block_n
    block_k = config.block_k

    triton_act = triton_utils.get_triton_activation(activation)
    rhs_dtype = jnp.result_type(*jax.tree.leaves(weights))
    res_dtype = jnp.result_type(x.dtype, rhs_dtype)
    dot_dtype, input_precision = triton_utils.get_dot_dtype_and_precision(
        precision, x.dtype, rhs_dtype
    )

    def fn(x, weights):
      if isinstance(weights, tuple):
        w, v = weights
      else:
        w, v = weights, None
      n = w.shape[-1]
      out_shape = x.shape[:-1] + (n,)
      x = jax.lax.collapse(x, 0, -1)
      m, k = x.shape
      if v is not None:
        if w.shape != (k, n) or v.shape != (k, n):
          raise ValueError(
              f'Incompatible shapes: x={x.shape}, weights=({w.shape},'
              f' {v.shape})'
          )
      elif w.shape != (k, 2, n):
        raise ValueError(f'Incompatible shapes: x={x.shape}, weights={w.shape}')
      if m == 0 or n == 0 or k == 0:
        out = jnp.zeros(out_shape, dtype=x.dtype)
        residuals = (
            jnp.zeros(out_shape[:-1] + (2, n), dtype=res_dtype)
            if return_residuals
            else None
        )
        return out, residuals

      # We re-order the program IDs to minimize cache usage.
      grid_m = triton.cdiv(m, block_m)
      grid_n = triton.cdiv(n, block_n)

      block_m_cost = block_m * k * jnp.dtype(x.dtype).itemsize
      w_itemsize = (
          jnp.dtype(w.dtype).itemsize + jnp.dtype(v.dtype).itemsize
          if v is not None
          else jnp.dtype(w.dtype).itemsize * 2
      )
      block_n_cost = block_n * k * w_itemsize
      group_size, get_pids_fn = triton_utils.get_cheapest_grid_pids(
          grid_m=grid_m,
          grid_n=grid_n,
          block_m_cost=block_m_cost,
          block_n_cost=block_n_cost,
      )

      name = 'triton_glu'
      if activation is not None:
        name += f'_{getattr(activation, "__name__", repr(activation))}'
      if return_residuals:
        name += '_fwd_res'

      x_strides = jt.strides_from_shape(x.shape)
      w_strides = jt.strides_from_shape(w.shape)
      v_strides = () if v is None else jt.strides_from_shape(v.shape)
      out_strides = jt.strides_from_shape((m, n))
      res_strides = (
          jt.strides_from_shape((m, 2, n)) if return_residuals else ()
      )

      k_padded = triton.cdiv(k, block_k) * block_k
      use_int64_k_step = (
          max(
              k_padded * x_strides[1],
              k_padded * w_strides[0],
              0 if v is None else k_padded * v_strides[0],
          )
          > 2**31 - 1
      )

      @triton_utils.vmap_kernel
      def launch(batch_shape, batch_strides, x, w, v):
        out_shapes = (
            jax.ShapeDtypeStruct((*batch_shape, m, n), x.dtype),
            jax.ShapeDtypeStruct(
                (*batch_shape, m, 2, n)
                if return_residuals
                else (*batch_shape, 0),
                res_dtype,
            ),
        )
        return jt.triton_call(
            x,
            w,
            v_ptr=v,
            kernel=_gated_linear_unit_kernel,
            out_type=out_shapes,
            grid=(math.prod(batch_shape) * grid_m * grid_n,),
            name=name,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
            batch_shape=batch_shape,
            batch_strides=batch_strides,
            is_batched=bool(batch_shape),
            m=m,
            n=n,
            k=k,
            x_strides=x_strides,
            w_strides=w_strides,
            v_strides=v_strides,
            out_strides=out_strides,
            res_strides=res_strides,
            grid_m=grid_m,
            grid_n=grid_n,
            group_size=group_size,
            get_pids_fn=get_pids_fn,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            dot_dtype=dot_dtype,
            input_precision=input_precision,
            activation=triton_act,
            return_residuals=return_residuals,
            use_int64_k_step=use_int64_k_step,
        )

      out, residuals = launch(x, w, v)

      if return_residuals:
        residuals = residuals.reshape(out_shape[:-1] + (2, n))
      else:
        residuals = None

      return out.reshape(out_shape), residuals

    fn = self._with_vmap(fn, fallback_to_sequential=False)
    return fn(x, weights)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    m, n, k, x_itemsize, w_pair_itemsize, input_precision = (
        _extract_problem_info(ba)
    )
    if m == 0 or n == 0 or k == 0:
      return Config(
          block_m=32, block_n=32, block_k=32, num_warps=4, num_stages=1
      )

    is_wgmma_or_tcgen05 = (
        gpu_utils.is_sm90() or gpu_utils.is_sm100()
    ) and input_precision != 'ieee'
    min_block_m = 64 if is_wgmma_or_tcgen05 else 32
    min_block_n = 32
    is_multi_or_simt = input_precision in (
        'ieee',
        'bf16x3',
        'bf16x6',
        'tf32x3',
    )
    max_area = 64 * 128 if is_multi_or_simt else 128 * 128

    block_m = min(max(min_block_m, triton.next_power_of_2(m)), 128)
    # Gate and projection hold separate RHS tiles and float32 accumulators.
    block_n = min(
        max(min_block_n, triton.next_power_of_2(n) // 2),
        128,
        max_area // block_m,
    )
    num_sms = getattr(backend.get_default_device(), 'core_count', 1)
    while (
        triton.cdiv(m, block_m) * triton.cdiv(n, block_n) < num_sms
        and block_m > min_block_m
    ):
      block_m //= 2
    while (
        triton.cdiv(m, block_m) * triton.cdiv(n, block_n) < num_sms
        and block_n > max(min_block_n, 64)
    ):
      block_n //= 2

    block_k = (
        64 if (k >= 64 and x_itemsize <= 2 and w_pair_itemsize <= 4) else 32
    )
    max_smem = _max_smem_bytes()
    num_stages = min(4, max(1, triton.cdiv(k, block_k)))
    while (
        num_stages > 1
        and _smem_bytes(
            block_m, block_n, block_k, num_stages, x_itemsize, w_pair_itemsize
        )
        > max_smem
    ):
      num_stages -= 1
    if (
        _smem_bytes(
            block_m, block_n, block_k, num_stages, x_itemsize, w_pair_itemsize
        )
        > max_smem
        and block_k > 32
    ):
      block_k = 32

    num_warps = 8 if block_m * block_n >= 128 * 64 else 4
    return Config(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    m, n, k, x_itemsize, w_pair_itemsize, input_precision = (
        _extract_problem_info(ba)
    )
    if m == 0 or n == 0 or k == 0:
      return set()

    is_wgmma_or_tcgen05 = (
        gpu_utils.is_sm90() or gpu_utils.is_sm100()
    ) and input_precision != 'ieee'
    min_block_m = 64 if is_wgmma_or_tcgen05 else 32
    m_pow2 = max(min_block_m, triton.next_power_of_2(m))
    n_pow2 = max(32, triton.next_power_of_2(n))
    block_m_cands = tuple(
        bm for bm in (32, 64, 128) if min_block_m <= bm <= m_pow2
    )
    block_n_cands = tuple(bn for bn in (32, 64, 128) if bn <= n_pow2)
    block_k_cands = (
        (32, 64)
        if (k >= 64 and x_itemsize <= 2 and w_pair_itemsize <= 4)
        else (32,)
    )
    max_smem = _max_smem_bytes()

    autotuning_configs = set()
    for block_m in block_m_cands:
      for block_n in block_n_cands:
        for block_k in block_k_cands:
          max_stages = min(5, max(1, triton.cdiv(k, block_k)))
          for num_warps in (4, 8):
            for num_stages in range(1, max_stages + 1):
              cfg = Config(
                  block_m=block_m,
                  block_n=block_n,
                  block_k=block_k,
                  num_warps=num_warps,
                  num_stages=num_stages,
              )
              if _is_valid_config(
                  cfg,
                  x_itemsize=x_itemsize,
                  w_pair_itemsize=w_pair_itemsize,
                  input_precision=input_precision,
                  max_smem=max_smem,
              ):
                autotuning_configs.add(cfg)
    return autotuning_configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)
