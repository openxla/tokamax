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
"""Ragged dot Pallas-Triton implementation."""

import dataclasses
import functools
import math
from typing import Callable, ClassVar

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import qwix
from tokamax._src import batching
from tokamax._src import gpu_utils
from tokamax._src import quantization
from tokamax._src.ops import op
from tokamax._src.ops.ragged_dot import base
from tokamax._src.pallas import block
from typing_extensions import override


Residuals = base.Residuals
QArray = base.QArray
AsQArray = base.AsQArray
GroupSizes = base.GroupSizes


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_m: int
  block_n: int
  block_k: int
  split_k: int = 1
  num_warps: int = 4
  num_stages: int


def _ragged_dot_kernel(
    a_ref,
    a_scales_ref,
    b_ref,
    b_scales_ref,
    lo_ref,
    hi_ref,
    out_ref,
    *,
    n,
    block_m,
    block_k,
    precision,
    activation,
):
  """Pallas-Triton ragged dot kernel."""
  lo = lo_ref.load()
  hi = hi_ref.load()
  block_n = out_ref.shape[1]
  start_m = lo + pl.program_id(0) * block_m
  start_n = pl.program_id(1) * block_n

  @pl.when(start_m < hi)
  def f():  # pylint: disable=unused-variable
    span_m = pl.ds(start_m, block_m)
    acc = jnp.zeros((block_m, out_ref.shape[1]), dtype=jnp.float32)

    def body(i, acc):
      start_k = i * block_k
      span_k = pl.ds(start_k, block_k)
      a = a_ref.at[span_m, span_k].load(bounds_check=(False, True))
      b = b_ref.at[span_k].load(bounds_check=(True, False))

      a_scales = None
      if a_scales_ref is not None:
        tile_k = a_ref.shape[1] // a_scales_ref.shape[1]
        if tile_k % block_k == 0:
          offs_k = pl.ds(start_k // tile_k, 1)
          a_scales = plgpu.load(a_scales_ref.at[span_m, offs_k])
        else:
          offs_m = (start_m + jnp.arange(block_m))[:, None]
          offs_k = ((start_k + jnp.arange(block_k)) // tile_k)[None, :]
          a_scales = plgpu.load(a_scales_ref.at[offs_m, offs_k])

      b_scales = None
      if b_scales_ref is not None:
        tile_k = b_ref.shape[0] // b_scales_ref.shape[0]
        tile_n = n // b_scales_ref.shape[1]
        if tile_k % block_k == 0:
          offs_k = start_k // tile_k
        elif tile_k == 1:
          offs_k = pl.ds(start_k, block_k)
        else:
          offs_k = (start_k + jnp.arange(block_k)) // tile_k
        if tile_n % block_n == 0:
          offs_n = pl.ds(start_n // tile_n, 1)
          b_scales = plgpu.load(b_scales_ref.at[offs_k, offs_n])
        elif tile_n == 1:
          offs_n = pl.ds(start_n, block_n)
          b_scales = plgpu.load(b_scales_ref.at[offs_k, offs_n])
        else:
          offs_k = ((start_k + jnp.arange(block_k)) // tile_k)[:, None]
          offs_n = ((start_n + jnp.arange(block_n)) // tile_n)[None, :]
          b_scales = plgpu.load(b_scales_ref.at[offs_k, offs_n])

      if (
          a_scales is not None
          and b_scales is not None
          and a_scales.shape[1] == b_scales.shape[0] == 1
      ):
        dot_precision = None
      else:
        if a_scales is not None:
          a = a.astype(a_scales.dtype) * a_scales
        if b_scales is not None:
          b = b.astype(b_scales.dtype) * b_scales
        a_scales = b_scales = 1
        dot_precision = precision

      dtype = jnp.result_type(a, b)
      a = a.astype(dtype)
      b = b.astype(dtype)
      return acc + pl.dot(a, b, precision=dot_precision) * a_scales * b_scales

    acc = jax.lax.fori_loop(0, pl.cdiv(a_ref.shape[1], block_k), body, acc)
    mask = (start_m + jnp.arange(block_m)) < hi
    if activation is not None:
      acc = activation(acc)
    out_ref.at[span_m].store(acc.astype(out_ref.dtype), mask=mask[:, None])


def _ragged_dot(
    lhs: jax.Array | QArray,
    rhs: jax.Array | QArray,
    *,
    group_sizes: jax.Array,
    ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
    precision: base.CanonicalPrecision,
    out_dtype: jnp.dtype,
    split_k_intermediate_dtype: jax.typing.DTypeLike | None,
    config: Config,
    activation: base.ActivationFunction | None = None,
) -> jax.Array:
  """Pallas-Triton ragged dot."""
  assert ragged_dot_dimension_numbers == base.DEFAULT_RAGGED_DOT_DIM_NUMS

  if config.split_k != 1:
    if split_k_intermediate_dtype is None:
      # if not provided, promote intermediate dtype to avoid precision loss
      # when reducing.
      split_k_out_dtype = jnp.result_type(out_dtype, jnp.float32)
    else:
      split_k_out_dtype = jnp.dtype(split_k_intermediate_dtype)

    def f(lhs, rhs):
      return _ragged_dot(
          lhs,
          rhs,
          group_sizes=group_sizes,
          ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
          precision=precision,
          out_dtype=split_k_out_dtype,
          split_k_intermediate_dtype=None,
          config=dataclasses.replace(config, split_k=1),
      )

    f = batching.vmap_split(f, in_axes=(1, 1), num_parts=config.split_k)
    return f(lhs, rhs).sum(axis=0).astype(out_dtype)

  m, k = lhs.shape
  num_groups, _, n = rhs.shape
  cum_rows = jnp.cumulative_sum(group_sizes, include_initial=True)

  block_m = config.block_m
  block_k = config.block_k
  block_n = config.block_n

  lhs_scales = None
  lhs_scales_spec = None
  if isinstance(lhs, QArray):
    if (lhs.scale.shape[0] == lhs.shape[0]) and lhs.zero_point is None:
      lhs, lhs_scales = lhs.qvalue, lhs.scale
      index_map = lambda _, __, e: (0, 0)
      lhs_scales_spec = pl.BlockSpec(lhs_scales.shape, index_map)
    else:
      lhs = qwix.dequantize(lhs)

  rhs_scales = None
  rhs_scales_spec = None
  if isinstance(rhs, QArray):
    if (rhs.scale.shape[0] == rhs.shape[0]) and rhs.zero_point is None:
      rhs, rhs_scales = rhs.qvalue, rhs.scale
      index_map = lambda _, __, e: (e, 0, 0)
      rhs_scales_spec = pl.BlockSpec((None, *rhs_scales.shape[1:]), index_map)
    else:
      rhs = qwix.dequantize(rhs)

  kernel = functools.partial(
      _ragged_dot_kernel,
      n=n,
      block_m=block_m,
      block_k=block_k,
      precision=precision,
      activation=activation,
  )

  return block.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
      in_specs=[
          pl.no_block_spec,
          lhs_scales_spec,
          pl.BlockSpec((None, k, block_n), lambda _, j, e: (e, 0, j)),
          rhs_scales_spec,
          pl.BlockSpec((None,), lambda _, __, e: (e,)),
          pl.BlockSpec((None,), lambda _, __, e: (e,)),
      ],
      out_specs=pl.BlockSpec((m, block_n), lambda _, j, __: (0, j)),
      grid=(pl.cdiv(m, block_m), pl.cdiv(n, block_n), num_groups),
      compiler_params=plgpu.CompilerParams(
          num_warps=config.num_warps, num_stages=config.num_stages
      ),
  )(lhs, lhs_scales, rhs, rhs_scales, cum_rows[:-1], cum_rows[1:])


def _ragged_contracting_dim_dot_kernel(
    a_ref,
    b_ref,
    lo_ref,
    hi_ref,
    out_ref,
    *,
    block_m,
    block_k,
    precision,
    activation,
):
  """Pallas-Triton ragged dot kernel for ragged contracting dimension."""
  lo = lo_ref.load()
  hi = hi_ref.load()

  def body(i, acc, mask_k=False):
    start_k = lo + i * block_k
    span_k = pl.ds(start_k, block_k)
    mask = None
    other = None
    if mask_k:
      mask = (jnp.arange(block_k) < hi - start_k)[:, None]
      other = 0.0
    a = plgpu.load(a_ref.at[span_k], mask=mask, other=other)
    b = plgpu.load(b_ref.at[span_k], mask=mask, other=other)
    dtype = jnp.result_type(a, b)
    a = a.astype(dtype)
    b = b.astype(dtype)
    return acc + pl.dot(a.T, b, precision=precision)

  num_iters = pl.cdiv(jnp.int32(hi - lo), block_k)
  acc = jnp.zeros((block_m, out_ref.shape[1]), dtype=jnp.float32)
  acc = jax.lax.fori_loop(0, num_iters - 1, body, acc)
  acc = body(num_iters - 1, acc, mask_k=True)  # Mask final iteration.
  if activation is not None:
    acc = activation(acc)
  out_ref.store(acc.astype(out_ref.dtype))


def _ragged_contracting_dim_dot(
    lhs: jax.Array | QArray,
    rhs: jax.Array | QArray,
    *,
    group_sizes: jax.Array,
    ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
    precision: base.CanonicalPrecision,
    out_dtype: jnp.dtype,
    config: Config,
    activation: base.ActivationFunction | None = None,
) -> jax.Array:
  """Pallas-Triton ragged dot for ragged contracting dimension."""
  assert ragged_dot_dimension_numbers == base.RAGGED_CONTRACTING_DOT_DIM_NUMS

  if config.split_k != 1:
    raise NotImplementedError(
        "`split_k != 1` not supported with ragged contracting dim."
    )

  k, m = lhs.shape
  _, n = rhs.shape
  cum_rows = jnp.cumulative_sum(group_sizes, include_initial=True)

  block_m = config.block_m
  block_k = config.block_k
  block_n = config.block_n

  lhs, rhs = map(quantization.as_array, (lhs, rhs))

  def f(lhs, rhs, lo, hi):
    kernel = functools.partial(
        _ragged_contracting_dim_dot_kernel,
        block_m=block_m,
        block_k=block_k,
        precision=precision,
        activation=activation,
    )

    return block.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        in_specs=[
            pl.BlockSpec((k, block_m), lambda i, j: (0, i)),
            pl.BlockSpec((k, block_n), lambda i, j: (0, j)),
            pl.no_block_spec,
            pl.no_block_spec,
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        grid=(pl.cdiv(m, block_m), pl.cdiv(n, block_n)),
        compiler_params=plgpu.CompilerParams(
            num_warps=config.num_warps, num_stages=config.num_stages
        ),
    )(lhs, rhs, lo, hi)

  f = jax.vmap(f, in_axes=(None, None, 0, 0))
  return f(lhs, rhs, cum_rows[:-1], cum_rows[1:])


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasTritonRaggedDot(base.RaggedDot[Config, None]):
  """Pallas-Triton ragged dot implementation."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  split_k_intermediate_dtype: jax.typing.DTypeLike | None = None

  def __post_init__(self):
    if self.vjp is None:
      # Avoid infinite recursion.
      f = lambda *a, **kw: PallasTritonRaggedDot()(*a, **kw)  # pylint: disable=unnecessary-lambda
      vjp = functools.partial(base.vjp, dlhs_ragged_dot=f, drhs_ragged_dot=f)
      object.__setattr__(self, "vjp", vjp)

  @override
  def _fwd(
      self,
      lhs: jax.Array | QArray | AsQArray,
      rhs: jax.Array | QArray | AsQArray,
      *,
      group_sizes: jax.Array | GroupSizes,
      ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
      precision: base.CanonicalPrecision,
      preferred_element_type: jnp.dtype | None,
      return_residuals: bool,
      config: Config,
      activation: Callable[[jax.Array], jax.Array] | None = None,
  ) -> tuple[jax.Array, base.Residuals]:
    lhs, rhs = map(quantization.as_array_or_qarray, (lhs, rhs))

    if preferred_element_type is None:
      out_dtype = jnp.promote_types(lhs.dtype, rhs.dtype)
    else:
      out_dtype = preferred_element_type

    if ragged_dot_dimension_numbers == base.TRANS_RHS_RAGGED_DOT_DIM_NUMS:
      rhs = rhs.mT  # TODO: Fuse transpose into kernel.
      ragged_dot_dimension_numbers = base.DEFAULT_RAGGED_DOT_DIM_NUMS

    def handle_residuals(
        dot_out: jax.Array,
    ) -> tuple[jax.Array, base.Residuals]:
      """Handles the residuals for the ragged dot operation.

      Args:
        dot_out: The result of the dot product.

      Returns:
        A tuple containing the (potentially activated) output and the residuals
        if `return_residuals` is True, otherwise None.
      """
      residuals = dot_out
      if activation is not None and return_residuals:
        dot_out = activation(dot_out)

      return dot_out, residuals if return_residuals else None

    if ragged_dot_dimension_numbers == base.DEFAULT_RAGGED_DOT_DIM_NUMS:
      out = _ragged_dot(
          lhs,
          rhs,
          group_sizes=group_sizes,
          ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
          precision=precision,
          out_dtype=out_dtype,
          split_k_intermediate_dtype=self.split_k_intermediate_dtype,
          config=config,
          activation=activation if not return_residuals else None,
      )
      return handle_residuals(out)

    if ragged_dot_dimension_numbers == base.RAGGED_CONTRACTING_DOT_DIM_NUMS:
      out = _ragged_contracting_dim_dot(
          lhs,
          rhs,
          group_sizes=group_sizes,
          ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
          precision=precision,
          out_dtype=out_dtype,
          config=config,
          activation=activation if not return_residuals else None,
      )
      return handle_residuals(out)

    raise NotImplementedError("Unsupported `ragged_dot_dimension_numbers`.")

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    m = ba.args[0].shape[0]  # TODO: Respect ragged dot dim nums.
    return Config(  # TODO: Create heuristics.
        block_m=min(128, pl.next_power_of_2(m)),
        block_n=128,
        block_k=32,
        num_warps=4,
        num_stages=4,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    lhs, rhs = ba.args
    m, k = lhs.shape
    n = rhs.shape[-1]
    batch_size = math.prod(ba.vmap_axis_sizes)
    # This is unnecessary high to ensure good load balancing.
    min_num_blocks = 4 * jax.local_devices()[0].core_count
    clamp = lambda lo, x, hi: max(lo, min(x, hi))
    configs = set()
    for block_m in [64, 128, 256]:
      block_m = clamp(32, block_m, pl.next_power_of_2(m))
      for block_n in [64, 128, 256]:
        block_n = clamp(32, block_n, pl.next_power_of_2(n))
        # This is num blocks for one expert (so an underestimate).
        num_blocks = batch_size * pl.cdiv(m, block_m) * pl.cdiv(n, block_n)

        for block_k in [32, 64, 128]:
          block_k = clamp(32, block_k, pl.next_power_of_2(k))
          for split_k in [1, 2, 4, 8, 16]:
            split_k = min(split_k, pl.cdiv(min_num_blocks, num_blocks))

            for num_warps in [4, 8]:
              for num_stages in [2, 3, 4, 5, 6]:
                configs.add(
                    Config(
                        block_m=block_m,
                        block_n=block_n,
                        block_k=block_k,
                        split_k=split_k,
                        num_stages=min(num_stages, pl.cdiv(k, block_k)),
                        num_warps=num_warps,
                    )
                )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)
