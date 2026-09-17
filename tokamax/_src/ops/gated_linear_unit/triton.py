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
import functools
import math
from typing import ClassVar, override

import jax
import jax.numpy as jnp
import jax_triton as jt
from jaxtyping import Array, Float  # pylint: disable=g-importing-member,g-multiple-import
from tokamax._src import gpu_utils
from tokamax._src import triton_utils
from tokamax._src.ops import op
from tokamax._src.ops.gated_linear_unit import base
from tokamax._src.ops.gated_linear_unit.base import FusedWeights, UnfusedWeights  # pylint: disable=g-importing-member,g-multiple-import
import triton
import triton.language as tl

Residuals = base.Residuals


@triton.jit
def _gated_linear_unit_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    res_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    x_strides: tl.constexpr,
    w_strides: tl.constexpr,
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
):
  """Triton GLU kernel."""
  pid = tl.program_id(axis=0)
  pid_m, pid_n = get_pids_fn(pid, grid_m, grid_n, group_size)

  offs_m = pid_m.to(tl.int64) * block_m + tl.arange(0, block_m)
  offs_n = pid_n.to(tl.int64) * block_n + tl.arange(0, block_n)
  offs_k = tl.arange(0, block_k)

  stride_xm, stride_xk = x_strides
  stride_wk, stride_wg, stride_wn = w_strides
  stride_om, stride_on = out_strides

  x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
  w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
  v_ptrs = w_ptrs + stride_wg

  gates = tl.zeros((block_m, block_n), dtype=tl.float32)
  proj = tl.zeros((block_m, block_n), dtype=tl.float32)

  for start_k in range(0, k, block_k):
    start_k = tl.multiple_of(start_k, block_k)
    k_mask = (start_k + offs_k) < k
    x = tl.load(x_ptrs, mask=k_mask[None, :], other=0.0)
    w = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)

    x = x.to(dot_dtype)
    w = w.to(dot_dtype)
    v = v.to(dot_dtype)

    gates = tl.dot(x, w, acc=gates, input_precision=input_precision)
    proj = tl.dot(x, v, acc=proj, input_precision=input_precision)

    x_ptrs += block_k * stride_xk
    w_ptrs += block_k * stride_wk
    v_ptrs += block_k * stride_wk

  out_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)

  if return_residuals:
    stride_res_m, stride_res_g, stride_res_n = res_strides
    res_ptrs_0 = (
        res_ptr
        + offs_m[:, None] * stride_res_m
        + offs_n[None, :] * stride_res_n
    )
    res_ptrs_1 = res_ptrs_0 + stride_res_g
    tl.store(res_ptrs_0, gates.to(x_ptr.dtype.element_ty), mask=out_mask)
    tl.store(res_ptrs_1, proj.to(x_ptr.dtype.element_ty), mask=out_mask)

  if activation is not None:
    gates = activation(gates)

  out = proj * gates
  out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
  tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=out_mask)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_m: int
  block_n: int
  block_k: int
  num_warps: int
  num_stages: int


def _get_best_block_size(m: int, n: int) -> tuple[int, int, int]:
  """Returns the best block size for the given shape."""
  min_block_dim = 32
  block_m = min(max(min_block_dim, triton.next_power_of_2(m)), 128)
  block_n = min(max(min_block_dim, triton.next_power_of_2(n)), 256)
  block_n = min(block_n, (128 * 128) // block_m)
  block_k = 32
  num_sms = jax.devices()[0].core_count
  num_blocks = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
  while num_blocks < num_sms:
    if block_m == min_block_dim:
      break
    block_m //= 2
    num_blocks = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
  return block_m, block_n, block_k


@dataclasses.dataclass(frozen=True, slots=True)
class TritonGatedLinearUnit(base.GatedLinearUnit[Config, None]):
  """Triton gated linear unit."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

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
    if x.dtype.type not in supported_dtypes:
      raise NotImplementedError(
          f'Triton kernel does not support input datatype {x.dtype}. Must be'
          f' one of {supported_dtypes}.'
      )

    block_m = config.block_m
    block_n = config.block_n
    block_k = config.block_k

    # TODO: Avoid stacking weights.
    weights = (
        jnp.stack(weights, axis=1) if isinstance(weights, tuple) else weights
    )

    triton_act = triton_utils.get_triton_activation(activation)
    dot_dtype, input_precision = triton_utils.get_dot_dtype_and_precision(
        precision, x.dtype, weights.dtype
    )

    def fn(x, weights):
      out_shape = x.shape[:-1] + (weights.shape[-1],)
      x = jax.lax.collapse(x, 0, -1)
      m, k = x.shape
      n = weights.shape[-1]
      # We re-order the program IDs to minimize cache usage.
      grid_m = triton.cdiv(m, block_m)
      grid_n = triton.cdiv(n, block_n)

      block_m_cost = block_m * k * jnp.dtype(x.dtype).itemsize
      block_n_cost = block_n * k * jnp.dtype(weights.dtype).itemsize * 2
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

      out_shapes = (
          jax.ShapeDtypeStruct((m, n), x.dtype),
          jax.ShapeDtypeStruct(
              (m, 2, n) if return_residuals else (0,), x.dtype
          ),
      )

      out, residuals = jt.triton_call(
          x,
          weights,
          kernel=_gated_linear_unit_kernel,
          out_type=out_shapes,
          grid=(grid_m * grid_n,),
          name=name,
          num_warps=config.num_warps,
          num_stages=config.num_stages,
          m=m,
          n=n,
          k=k,
          x_strides=jt.strides_from_shape(x.shape),
          w_strides=jt.strides_from_shape(weights.shape),
          out_strides=jt.strides_from_shape(out_shapes[0].shape),
          res_strides=(
              jt.strides_from_shape(out_shapes[1].shape)
              if return_residuals
              else ()
          ),
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
      )

      if return_residuals:
        residuals = residuals.reshape(out_shape[:-1] + (2, n))
      else:
        residuals = None

      return out.reshape(out_shape), residuals

    fn = self._with_vmap(fn, fallback_to_sequential=False)
    return fn(x, weights)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    x, weights = ba.args  # TODO: Use batched args.
    m = math.prod(x.shape[:-1])
    n = weights[0].shape[-1] if isinstance(weights, tuple) else weights.shape[2]
    if n >= m:  # Prefer `block_n` > `block_m`.
      block_m, block_n, block_k = _get_best_block_size(m, n)
    else:
      block_n, block_m, block_k = _get_best_block_size(n, m)
    return Config(
        block_m=block_m,
        block_n=block_n // 2,  # We have two blocks for RHS, so halve `block_n`.
        block_k=block_k,
        num_warps=4,
        num_stages=4,
    )

  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    # Simple autotuning search space that can be improved upon.
    x, weights = ba.args  # TODO: Use batched args.
    m_pow2 = triton.next_power_of_2(math.prod(x.shape[:-1]))
    n = weights[0].shape[-1] if isinstance(weights, tuple) else weights.shape[2]
    n_pow2 = triton.next_power_of_2(n)
    autotuning_configs = set()
    for block_m in (32, *filter(lambda x: x <= m_pow2, (64, 128))):
      for block_n in (32, *filter(lambda x: x <= n_pow2, (64, 128, 256))):
        for num_warps in (4, 8):
          for num_stages in range(1, 6):
            autotuning_configs.add(
                Config(
                    block_m=block_m,
                    block_n=block_n,
                    block_k=32,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            )
    return autotuning_configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)
