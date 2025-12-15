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
"""Pallas-Triton gated linear unit."""

from collections.abc import Callable
import dataclasses
import functools
import math
from typing import ClassVar

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-importing-member,g-multiple-import
from tokamax._src import gpu_utils
from tokamax._src.ops import op
from tokamax._src.ops.gated_linear_unit import base
from tokamax._src.ops.gated_linear_unit.base import FusedWeights, UnfusedWeights  # pylint: disable=g-importing-member,g-multiple-import
from tokamax._src.pallas import block
from tokamax._src.pallas import grid
from typing_extensions import override


Residuals = base.Residuals


def _gated_linear_unit_kernel(
    x_ref,
    weights_ref,
    out_ref,
    residuals_ref,
    *,
    block_m,
    block_n,
    block_k,
    activation,
    precision,
):
  """Pallas GLU kernel."""

  def body(i, acc):
    k_span = block.ds(i, block_k)
    x = x_ref.at[:, k_span].load(bounds_check=(False, True))
    w = weights_ref.at[k_span, 0].load(bounds_check=(True, False))
    v = weights_ref.at[k_span, 1].load(bounds_check=(True, False))
    acc[0] += pl.dot(x, w.astype(x.dtype), precision=precision)
    acc[1] += pl.dot(x, v.astype(x.dtype), precision=precision)
    return acc

  num_iters = pl.cdiv(x_ref.shape[-1], block_k)
  acc0 = jnp.zeros((block_m, block_n), dtype=jnp.float32)
  acc1 = jnp.zeros((block_m, block_n), dtype=jnp.float32)
  gates, proj = jax.lax.fori_loop(0, num_iters, body, init_val=[acc0, acc1])
  gates = gates.astype(x_ref.dtype)
  proj = proj.astype(x_ref.dtype)

  if residuals_ref is not None:
    residuals_ref.at[:, 0].store(gates)
    residuals_ref.at[:, 1].store(proj)

  proj = proj.astype(jnp.float32)
  gates = gates.astype(jnp.float32)
  out = proj * (gates if activation is None else activation(gates))
  out_ref.store(out.astype(out_ref.dtype))


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
  block_m = min(max(min_block_dim, pl.next_power_of_2(m)), 128)
  block_n = min(max(min_block_dim, pl.next_power_of_2(n)), 256)
  block_n = min(block_n, (128 * 128) // block_m)
  block_k = 32
  num_blocks = pl.cdiv(m, block_m) * pl.cdiv(n, block_n)
  while num_blocks < jax.devices()[0].core_count:
    if block_m == min_block_dim:
      break
    block_m //= 2
    num_blocks = pl.cdiv(m, block_m) * pl.cdiv(n, block_n)
  return block_m, block_n, block_k


@dataclasses.dataclass(frozen=True, slots=True)
class PallasTritonGatedLinearUnit(base.GatedLinearUnit[Config, None]):
  """Pallas-Triton gated linear unit."""

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

    def fn(x, weights):
      out_shape = x.shape[:-1] + (weights.shape[-1],)
      x = jax.lax.collapse(x, 0, -1)
      m, k = x.shape
      n = weights.shape[-1]
      # We re-order the program IDs to minimize cache usage.
      get_pids = functools.partial(
          grid.get_cheapest_grid_pids,
          grid_m=pl.cdiv(m, block_m),
          grid_n=pl.cdiv(n, block_n),
          block_m_cost=block_m * k * jnp.dtype(x.dtype).itemsize,
          block_n_cost=block_n * k * jnp.dtype(weights.dtype).itemsize * 2,
      )
      wrap_index_map = lambda fn: lambda pid: fn(*get_pids(pid))

      name = 'pallas_glu'
      if activation is not None:
        name += f'_{getattr(activation, "__name__", repr(activation))}'
      if return_residuals:
        name += '_fwd_res'

      kernel = functools.partial(
          _gated_linear_unit_kernel,
          block_m=block_m,
          block_n=block_n,
          block_k=block_k,
          activation=activation,
          precision=precision,
      )

      res_shape = jax.ShapeDtypeStruct((m, 2, n), x.dtype)
      out, residuals = block.pallas_call(
          kernel,
          name=name,
          grid=(pl.cdiv(m, block_m) * pl.cdiv(n, block_n),),
          out_shape=(
              jax.ShapeDtypeStruct((m, n), x.dtype),
              res_shape if return_residuals else None,
          ),
          in_specs=(
              pl.BlockSpec((block_m, k), wrap_index_map(lambda i, j: (i, 0))),
              pl.BlockSpec(
                  (k, 2, block_n), wrap_index_map(lambda i, j: (0, 0, j))
              ),
          ),
          out_specs=(
              pl.BlockSpec(
                  (block_m, block_n), wrap_index_map(lambda i, j: (i, j))
              ),
              pl.BlockSpec(
                  (block_m, 2, block_n), wrap_index_map(lambda i, j: (i, 0, j))
              ),
          ),
          filter_specs=True,
          compiler_params=plgpu.CompilerParams(
              num_warps=config.num_warps, num_stages=config.num_stages
          ),
      )(x, weights)
      if residuals is not None:
        residuals = residuals.reshape(out_shape[:-1] + (2, n))
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
    m_pow2 = pl.next_power_of_2(math.prod(x.shape[:-1]))
    n_pow2 = pl.next_power_of_2(weights.shape[2])
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
