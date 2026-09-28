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
"""Ragged dot Triton implementation."""

from collections.abc import Callable
import dataclasses
import functools
import math
from typing import ClassVar, override

import jax
from jax import numpy as jnp
import jax_triton as jt
import qwix
from tokamax._src import batching
from tokamax._src import gpu_utils
from tokamax._src import quantization
from tokamax._src import triton_utils
from tokamax._src.ops import op
from tokamax._src.ops.ragged_dot import base
import triton
import triton.language as tl

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


def _vmap_kernel(launch_fn: Callable[..., jax.Array]):
  def call(batch_size, in_batched, *args):
    strides = [
        () if x is None else (0,) * (not b) + jt.strides_from_shape(x.shape)
        for x, b in zip(args, in_batched)
    ]
    return launch_fn(batch_size, strides, *args)

  @jax.custom_batching.custom_vmap
  def f(*args):
    return call(1, [False] * len(args), *args)[0]

  @f.def_vmap
  def _f_vmap(axis_size, in_batched, *args):
    return call(axis_size, in_batched, *args), True

  return f


@triton.jit
def _ragged_dot_kernel(
    a_ptr,
    a_scales_ptr,
    b_ptr,
    b_scales_ptr,
    cum_rows_ptr,
    out_ptr,
    n: tl.constexpr,
    k: tl.constexpr,
    grid_m: tl.constexpr,
    a_strides: tl.constexpr,
    a_scales_strides: tl.constexpr,
    b_strides: tl.constexpr,
    b_scales_strides: tl.constexpr,
    cum_rows_strides: tl.constexpr,
    out_strides: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    a_is_int4: tl.constexpr,
    b_is_int4: tl.constexpr,
    a_tile_k: tl.constexpr,
    b_tile_k: tl.constexpr,
    b_tile_n: tl.constexpr,
    use_post_scale: tl.constexpr,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
    activation: tl.constexpr,
):
  """Triton ragged dot kernel."""
  pid_b = (tl.program_id(0) // grid_m).to(tl.int64)
  pid_m = tl.program_id(0) % grid_m
  pid_n = tl.program_id(1)
  pid_e = tl.program_id(2).to(tl.int64)

  stride_cb, stride_ce = cum_rows_strides
  cum_rows_ptr += pid_b * stride_cb + pid_e * stride_ce
  lo = tl.load(cum_rows_ptr)
  hi = tl.load(cum_rows_ptr + stride_ce)

  start_m = (lo + pid_m * block_m).to(tl.int64)
  if start_m >= hi:
    return

  start_n = (pid_n * block_n).to(tl.int64)
  a_pack: tl.constexpr = tl.constexpr(2 if a_is_int4 else 1)
  b_pack: tl.constexpr = tl.constexpr(2 if b_is_int4 else 1)
  offs_m = start_m + tl.arange(0, block_m)
  offs_n = start_n + tl.arange(0, block_n)
  offs_n_b = start_n // b_pack + tl.arange(0, block_n // b_pack)
  offs_k = tl.arange(0, block_k)
  offs_k_a = tl.arange(0, block_k // a_pack)

  stride_ab, stride_am, stride_ak = a_strides
  stride_bb, stride_be, stride_bk, stride_bn = b_strides
  a_ptr += pid_b * stride_ab
  b_ptr += pid_b * stride_bb + pid_e * stride_be
  a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k_a[None, :] * stride_ak
  b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n_b[None, :] * stride_bn

  if a_scales_ptr is not None:
    stride_asb, stride_asm, _ = a_scales_strides
    a_scales_ptr += pid_b * stride_asb + offs_m[:, None] * stride_asm

  if b_scales_ptr is not None:
    stride_bsb, stride_bse, _, stride_bsn = b_scales_strides
    bs_offs_n = (
        start_n // b_tile_n
        if b_tile_n % block_n == 0
        else (offs_n // b_tile_n)[None, :]
    )
    b_scales_ptr += (
        pid_b * stride_bsb + pid_e * stride_bse + bs_offs_n * stride_bsn
    )

  m_mask = (offs_m < hi)[:, None]
  n_b_mask = True if n % block_n == 0 else (offs_n_b < n // b_pack)[None, :]
  bs_n_mask = (
      True
      if (n % block_n == 0 or b_tile_n % block_n == 0)
      else (offs_n < n)[None, :]
  )
  acc = tl.zeros((block_m, block_n), dtype=tl.float32)

  for start_k in range(0, k, block_k):
    start_k = tl.multiple_of(start_k, block_k)
    k_rem = k - start_k

    k_a_mask = (
        True if k % block_k == 0 else (offs_k_a < k_rem // a_pack)[None, :]
    )
    a = tl.load(a_ptrs, mask=m_mask & k_a_mask, other=0.0)
    a_ptrs += (block_k // a_pack) * stride_ak
    if a_is_int4:
      a = tl.join((a << 4) >> 4, a >> 4).reshape(block_m, block_k)

    k_mask = True if k % block_k == 0 else (offs_k < k_rem)[:, None]
    b = tl.load(b_ptrs, mask=k_mask & n_b_mask, other=0.0)
    b_ptrs += block_k * stride_bk
    if b_is_int4:
      b = tl.join((b << 4) >> 4, b >> 4).reshape(block_k, block_n)

    a_scales = None
    if a_scales_ptr is not None:
      _, _, stride_ask = a_scales_strides
      as_offs_k = (
          start_k // a_tile_k
          if a_tile_k % block_k == 0
          else ((start_k + offs_k) // a_tile_k)[None, :]
      )
      as_k_mask = (
          True
          if (k % block_k == 0 or a_tile_k % block_k == 0)
          else (offs_k < k_rem)[None, :]
      )
      a_scales = tl.load(
          a_scales_ptr + as_offs_k * stride_ask,
          mask=m_mask & as_k_mask,
          other=0.0,
      )
      if not use_post_scale:
        a = a.to(a_scales.dtype) * a_scales

    b_scales = None
    if b_scales_ptr is not None:
      _, _, stride_bsk, _ = b_scales_strides
      bs_offs_k = (
          start_k // b_tile_k
          if b_tile_k % block_k == 0
          else ((start_k + offs_k) // b_tile_k)[:, None]
      )
      bs_k_mask = (
          True
          if (k % block_k == 0 or b_tile_k % block_k == 0)
          else (offs_k < k_rem)[:, None]
      )
      b_scales = tl.load(
          b_scales_ptr + bs_offs_k * stride_bsk,
          mask=bs_k_mask & bs_n_mask,
          other=0.0,
      )
      if not use_post_scale:
        b = b.to(b_scales.dtype) * b_scales

    a = a.to(dot_dtype)
    b = b.to(dot_dtype)
    if use_post_scale:
      assert a_scales is not None and b_scales is not None
      acc += (
          tl.dot(a, b, input_precision=input_precision).to(tl.float32)
          * a_scales.to(tl.float32)
          * b_scales.to(tl.float32)
      )
    elif dot_dtype == tl.int8:
      acc += tl.dot(a, b).to(tl.float32)
    else:
      acc = tl.dot(a, b, acc=acc, input_precision=input_precision)

  if activation is not None:
    acc = activation(acc)
  stride_ob, stride_om, stride_on = out_strides
  out_ptr += pid_b * stride_ob
  out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
  n_mask = True if n % block_n == 0 else (offs_n[None, :] < n)
  tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask & n_mask)


def _prepare_input(
    x: jax.Array | QArray,
) -> tuple[jax.Array, jax.Array | None, bool]:
  scales = None
  if isinstance(x, QArray):
    if (x.scale.shape[0] == x.shape[0]) and x.zero_point is None:
      x, scales = x.qvalue, x.scale
    else:
      x = qwix.dequantize(x)
  if is_int4 := (x.dtype == jnp.int4):
    x = jax.lax.bitcast_convert_type(
        x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2), jnp.int8
    )
  return x, scales, is_int4


def _ragged_dot(
    lhs: jax.Array | QArray,
    rhs: jax.Array | QArray,
    *,
    group_sizes: jax.Array,
    precision: base.CanonicalPrecision,
    out_dtype: jnp.dtype,
    split_k_intermediate_dtype: jax.typing.DTypeLike | None,
    config: Config,
    activation: base.ActivationFunction | None = None,
) -> jax.Array:
  """Triton ragged dot."""
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
          precision=precision,
          out_dtype=split_k_out_dtype,
          split_k_intermediate_dtype=None,
          config=dataclasses.replace(config, split_k=1),
      )

    f = batching.vmap_split(f, in_axes=(1, 1), num_parts=config.split_k)
    out = f(lhs, rhs).sum(axis=0)
    if activation is not None:
      out = activation(out)
    return out.astype(out_dtype)

  m, k = lhs.shape
  num_groups, _, n = rhs.shape
  cum_rows = jnp.cumulative_sum(group_sizes, include_initial=True)

  block_m = config.block_m
  block_k = config.block_k
  block_n = config.block_n

  lhs, lhs_scales, a_is_int4 = _prepare_input(lhs)
  rhs, rhs_scales, b_is_int4 = _prepare_input(rhs)

  a_tile_k = k // lhs_scales.shape[-1] if lhs_scales is not None else 1
  b_tile_k = k // rhs_scales.shape[-2] if rhs_scales is not None else 1
  b_tile_n = n // rhs_scales.shape[-1] if rhs_scales is not None else 1

  use_post_scale = (
      lhs_scales is not None
      and rhs_scales is not None
      and lhs.dtype == rhs.dtype
      and a_tile_k % block_k == 0
      and b_tile_k % block_k == 0
      and b_tile_n % block_n == 0
  )

  if use_post_scale:
    dot_dtype = triton_utils.jnp_to_tl_dtype(lhs.dtype)
    input_precision = None
  else:
    dot_dtype, input_precision = triton_utils.get_dot_dtype_and_precision(
        precision,
        lhs.dtype if lhs_scales is None else lhs_scales.dtype,
        rhs.dtype if rhs_scales is None else rhs_scales.dtype,
    )
  triton_act = triton_utils.get_triton_activation(activation)
  grid_m = triton.cdiv(m, block_m)
  grid_n = triton.cdiv(n, block_n)

  @_vmap_kernel
  def f(batch_size, strides, lhs, lhs_scales, rhs, rhs_scales, cum_rows):
    (
        a_strides,
        a_scales_strides,
        b_strides,
        b_scales_strides,
        cum_rows_strides,
    ) = strides
    out_struct = jax.ShapeDtypeStruct((batch_size, m, n), out_dtype)
    return jt.triton_call(
        lhs,
        lhs_scales,
        rhs,
        rhs_scales,
        cum_rows,
        kernel=_ragged_dot_kernel,
        out_type=out_struct,
        grid=(batch_size * grid_m, grid_n, num_groups),
        name="triton_ragged_dot",
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        n=n,
        k=k,
        grid_m=grid_m,
        a_strides=a_strides,
        a_scales_strides=a_scales_strides,
        b_strides=b_strides,
        b_scales_strides=b_scales_strides,
        cum_rows_strides=cum_rows_strides,
        out_strides=jt.strides_from_shape(out_struct.shape),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        a_is_int4=a_is_int4,
        b_is_int4=b_is_int4,
        a_tile_k=a_tile_k,
        b_tile_k=b_tile_k,
        b_tile_n=b_tile_n,
        use_post_scale=use_post_scale,
        dot_dtype=dot_dtype,
        input_precision=input_precision,
        activation=triton_act,
    )

  return f(lhs, lhs_scales, rhs, rhs_scales, cum_rows)


@triton.jit
def _ragged_contracting_dim_dot_kernel(
    a_ptr,
    b_ptr,
    cum_rows_ptr,
    out_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    grid_m: tl.constexpr,
    a_strides: tl.constexpr,
    b_strides: tl.constexpr,
    cum_rows_strides: tl.constexpr,
    out_strides: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    dot_dtype: tl.constexpr,
    input_precision: tl.constexpr,
    activation: tl.constexpr,
):
  """Triton ragged dot kernel for ragged contracting dimension."""
  pid_b = (tl.program_id(0) // grid_m).to(tl.int64)
  pid_m = tl.program_id(0) % grid_m
  pid_n = tl.program_id(1)
  pid_e = tl.program_id(2).to(tl.int64)

  stride_cb, stride_ce = cum_rows_strides
  cum_rows_ptr += pid_b * stride_cb + pid_e * stride_ce
  lo = tl.load(cum_rows_ptr)
  hi = tl.load(cum_rows_ptr + stride_ce)

  offs_m = (pid_m * block_m).to(tl.int64) + tl.arange(0, block_m)
  offs_n = (pid_n * block_n).to(tl.int64) + tl.arange(0, block_n)
  offs_k = lo.to(tl.int64) + tl.arange(0, block_k)

  stride_ab, stride_ak, stride_am = a_strides
  stride_bb, stride_bk, stride_bn = b_strides
  a_ptr += pid_b * stride_ab
  b_ptr += pid_b * stride_bb
  a_ptrs = a_ptr + offs_k[:, None] * stride_ak + offs_m[None, :] * stride_am
  b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

  m_mask = True if m % block_m == 0 else (offs_m[None, :] < m)
  n_mask = True if n % block_n == 0 else (offs_n[None, :] < n)

  acc = tl.zeros((block_m, block_n), dtype=tl.float32)
  for start_k in range(lo, hi, block_k):
    k_mask = (start_k + tl.arange(0, block_k))[:, None] < hi
    a = tl.load(a_ptrs, mask=k_mask & m_mask, other=0.0).to(dot_dtype)
    b = tl.load(b_ptrs, mask=k_mask & n_mask, other=0.0).to(dot_dtype)
    acc = tl.dot(tl.trans(a), b, acc=acc, input_precision=input_precision)
    a_ptrs += block_k * stride_ak
    b_ptrs += block_k * stride_bk

  if activation is not None:
    acc = activation(acc)
  stride_ob, stride_oe, stride_om, stride_on = out_strides
  out_ptr += pid_b * stride_ob + pid_e * stride_oe
  out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
  out_m_mask = True if m % block_m == 0 else (offs_m[:, None] < m)
  tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_m_mask & n_mask)


def _ragged_contracting_dim_dot(
    lhs: jax.Array | QArray,
    rhs: jax.Array | QArray,
    *,
    group_sizes: jax.Array,
    precision: base.CanonicalPrecision,
    out_dtype: jnp.dtype,
    config: Config,
    activation: base.ActivationFunction | None = None,
) -> jax.Array:
  """Triton ragged dot for ragged contracting dimension."""
  if config.split_k != 1:
    raise NotImplementedError(
        "`split_k != 1` not supported with ragged contracting dim."
    )

  _, m = lhs.shape
  _, n = rhs.shape
  [num_groups] = group_sizes.shape
  cum_rows = jnp.cumulative_sum(group_sizes, include_initial=True)

  block_m = config.block_m
  block_k = config.block_k
  block_n = config.block_n

  lhs, rhs = map(quantization.as_array, (lhs, rhs))

  dot_dtype, input_precision = triton_utils.get_dot_dtype_and_precision(
      precision, lhs.dtype, rhs.dtype
  )
  triton_act = triton_utils.get_triton_activation(activation)
  grid_m = triton.cdiv(m, block_m)
  grid_n = triton.cdiv(n, block_n)

  @_vmap_kernel
  def f(batch_size, strides, lhs, rhs, cum_rows):
    a_strides, b_strides, cum_rows_strides = strides
    out_struct = jax.ShapeDtypeStruct((batch_size, num_groups, m, n), out_dtype)

    return jt.triton_call(
        lhs,
        rhs,
        cum_rows,
        kernel=_ragged_contracting_dim_dot_kernel,
        out_type=out_struct,
        grid=(batch_size * grid_m, grid_n, num_groups),
        name="triton_ragged_contracting_dim_dot",
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        m=m,
        n=n,
        grid_m=grid_m,
        a_strides=a_strides,
        b_strides=b_strides,
        cum_rows_strides=cum_rows_strides,
        out_strides=jt.strides_from_shape(out_struct.shape),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        dot_dtype=dot_dtype,
        input_precision=input_precision,
        activation=triton_act,
    )

  return f(lhs, rhs, cum_rows)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class TritonRaggedDot(base.RaggedDot[Config, None]):
  """Triton ragged dot implementation."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  split_k_intermediate_dtype: jax.typing.DTypeLike | None = None

  def __post_init__(self):
    if self.vjp is None:
      # Avoid infinite recursion.
      f = lambda *a, **kw: TritonRaggedDot()(*a, **kw)  # pylint: disable=unnecessary-lambda
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
      manual_axis_type: jax.sharding.ManualAxisType | None = None,
      group_offset: jax.Array | None = None,
      rhs_scale: jax.Array | None = None,
      rhs_bias: jax.Array | None = None,
      maybe_quantize_lhs: bool = False,
      lhs_scale: jax.Array | None = None,
      zero_initialize: bool = True,
      fuse_gateup_activation: str | None = None,
      lhs_quantization_dtype: jax.typing.DTypeLike | None = None,
      rhs_quantization_dtype: jax.typing.DTypeLike | None = None,
  ) -> tuple[jax.Array, base.Residuals]:
    if (
        group_offset is not None
        or rhs_scale is not None
        or rhs_bias is not None
        or maybe_quantize_lhs
        or lhs_scale is not None
        or not zero_initialize
        or fuse_gateup_activation is not None
        or lhs_quantization_dtype is not None
        or rhs_quantization_dtype is not None
    ):
      raise NotImplementedError(
          "The Triton implementation does not support"
          " group_offset, rhs_scale, rhs_bias, maybe_quantize_lhs, lhs_scale,"
          " zero_initialize, fuse_gateup_activation, lhs_quantization_dtype,"
          " or rhs_quantization_dtype."
      )

    lhs, rhs = map(quantization.as_array_or_qarray, (lhs, rhs))
    # `QArray.dtype` is the scale dtype; the stored values are in `qvalue`.
    dtypes = [
        x.qvalue.dtype if isinstance(x, QArray) else x.dtype for x in (lhs, rhs)
    ]
    if gpu_utils.is_sm80() and jnp.float8_e4m3fn in dtypes:
      raise NotImplementedError("float8_e4m3fn is not supported on SM80.")

    if preferred_element_type is None:
      out_dtype = jnp.promote_types(lhs.dtype, rhs.dtype)
    else:
      out_dtype = preferred_element_type

    if ragged_dot_dimension_numbers == base.TRANS_RHS_RAGGED_DOT_DIM_NUMS:
      rhs = rhs.mT  # TODO: Fuse transpose into kernel.
      ragged_dot_dimension_numbers = base.DEFAULT_RAGGED_DOT_DIM_NUMS

    match ragged_dot_dimension_numbers:
      case base.DEFAULT_RAGGED_DOT_DIM_NUMS:
        dot_fn = functools.partial(
            _ragged_dot,
            split_k_intermediate_dtype=self.split_k_intermediate_dtype,
        )
      case base.RAGGED_CONTRACTING_DOT_DIM_NUMS:
        dot_fn = _ragged_contracting_dim_dot
      case _:
        raise NotImplementedError(
            f"Unsupported {ragged_dot_dimension_numbers=}"
        )

    out = dot_fn(
        lhs,
        rhs,
        group_sizes=jnp.asarray(group_sizes),
        precision=precision,
        out_dtype=out_dtype,
        config=config,
        activation=None if return_residuals else activation,
    )
    residuals = out if return_residuals else None
    if activation is not None and return_residuals:
      out = activation(out)
    return out, residuals

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    lhs = ba.args[0]
    dim_nums = ba.kwargs["ragged_dot_dimension_numbers"]
    m = lhs.shape[1 if dim_nums == base.RAGGED_CONTRACTING_DOT_DIM_NUMS else 0]
    return Config(  # TODO: Create heuristics.
        block_m=max(16, min(128, triton.next_power_of_2(m))),
        block_n=128,
        block_k=32,
        num_warps=4,
        num_stages=4,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    lhs, rhs = ba.args
    dim_nums = ba.kwargs["ragged_dot_dimension_numbers"]
    match dim_nums:
      case base.DEFAULT_RAGGED_DOT_DIM_NUMS:
        (m, k), (_, _, n) = lhs.shape, rhs.shape
      case base.TRANS_RHS_RAGGED_DOT_DIM_NUMS:
        (m, k), (_, n, _) = lhs.shape, rhs.shape
      case base.RAGGED_CONTRACTING_DOT_DIM_NUMS:
        (k, m), (_, n) = lhs.shape, rhs.shape
      case _:
        raise NotImplementedError(f"Unsupported {dim_nums=}")
    batch_size = math.prod(ba.vmap_axis_sizes)
    # This is unnecessary high to ensure good load balancing.
    min_num_blocks = 4 * jax.local_devices()[0].core_count
    clamp = lambda lo, x, hi: max(lo, min(x, hi))
    configs = set()
    for block_m in [64, 128, 256]:
      block_m = clamp(32, block_m, triton.next_power_of_2(m))
      for block_n in [64, 128, 256]:
        block_n = clamp(32, block_n, triton.next_power_of_2(n))
        # This is num blocks for one expert (so an underestimate).
        num_blocks = (
            batch_size * triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
        )

        for block_k in [32, 64, 128]:
          block_k = clamp(32, block_k, triton.next_power_of_2(k))
          split_ks = (
              [1]
              if dim_nums == base.RAGGED_CONTRACTING_DOT_DIM_NUMS
              or ba.vmap_axis_sizes
              else [1, 2, 4, 8, 16]
          )
          for split_k in split_ks:
            split_k = min(split_k, triton.cdiv(min_num_blocks, num_blocks))

            for num_warps in [4, 8]:
              for num_stages in [2, 3, 4, 5, 6]:
                configs.add(
                    Config(
                        block_m=block_m,
                        block_n=block_n,
                        block_k=block_k,
                        split_k=split_k,
                        num_stages=min(num_stages, triton.cdiv(k, block_k)),
                        num_warps=num_warps,
                    )
                )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)
