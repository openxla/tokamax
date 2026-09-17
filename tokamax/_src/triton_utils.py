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
"""Triton utilities."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from tokamax._src import precision as precision_lib
from tokamax._src.pallas import grid
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


get_cheapest_grid_grouping = grid.get_cheapest_grid_grouping


@triton.jit
def _get_grid_pids_m(
    pid,
    grid_m: tl.constexpr,
    grid_n: tl.constexpr,
    group_size: tl.constexpr,
):
  num_progs_in_group: tl.constexpr = group_size * grid_n
  group_start_m = (pid // num_progs_in_group) * group_size
  cur_group_size = tl.minimum(grid_m - group_start_m, group_size)
  pid_m = group_start_m + (pid % cur_group_size)
  pid_n = (pid % num_progs_in_group) // cur_group_size
  return pid_m, pid_n


@triton.jit
def _get_grid_pids_n(
    pid,
    grid_m: tl.constexpr,
    grid_n: tl.constexpr,
    group_size: tl.constexpr,
):
  pid_n, pid_m = _get_grid_pids_m(pid, grid_n, grid_m, group_size)
  return pid_m, pid_n


def get_cheapest_grid_pids(
    *,
    grid_m: int,
    grid_n: int,
    block_m_cost: int,
    block_n_cost: int,
) -> tuple[int, triton.runtime.jit.JITFunction]:
  """Returns (group_size, get_pids_fn) that minimizes total cache cost."""
  group_size, group_by_m = grid.get_cheapest_grid_grouping(
      grid_m=grid_m,
      grid_n=grid_n,
      block_m_cost=block_m_cost,
      block_n_cost=block_n_cost,
  )
  return group_size, (_get_grid_pids_m if group_by_m else _get_grid_pids_n)


@triton.jit
def _swish(x):
  return x * tl.sigmoid(x)


@triton.jit
def _relu(x):
  return tl.maximum(x, 0.0)


def get_triton_activation(
    activation: Callable[[jax.Array], jax.Array] | None,
) -> Callable[..., tl.tensor] | None:
  """Returns the Triton device function corresponding to a JAX activation."""
  if activation is None:
    return None
  if isinstance(activation, triton.runtime.jit.JITFunction):
    return activation
  if activation in (jax.nn.swish, jax.nn.silu):
    return _swish
  if activation == jax.nn.sigmoid:
    return tl.sigmoid
  if activation == jax.nn.tanh:
    return libdevice.tanh
  if activation == jax.nn.relu:
    return _relu
  raise NotImplementedError(
      f'Unsupported activation for Triton kernel: {activation}'
  )


_JNP_TO_TL_DTYPES = {
    jnp.bool_: tl.int1,
    jnp.int8: tl.int8,
    jnp.int16: tl.int16,
    jnp.int32: tl.int32,
    jnp.int64: tl.int64,
    jnp.uint8: tl.uint8,
    jnp.uint16: tl.uint16,
    jnp.uint32: tl.uint32,
    jnp.uint64: tl.uint64,
    jnp.float8_e4m3fn: tl.float8e4nv,
    jnp.float8_e5m2: tl.float8e5,
    jnp.float16: tl.float16,
    jnp.bfloat16: tl.bfloat16,
    jnp.float32: tl.float32,
    jnp.float64: tl.float64,
}


def jnp_to_tl_dtype(dtype: jax.typing.DTypeLike) -> tl.dtype:
  """Returns the Triton dtype corresponding to a JAX dtype."""
  return _JNP_TO_TL_DTYPES[jnp.dtype(dtype).type]


def get_input_precision(
    precision: precision_lib.CanonicalPrecision, dtype: jnp.dtype
) -> str | None:
  """Returns the Triton `tl.dot` `input_precision` string for a given precision."""
  if dtype != jnp.float32:
    return None
  if isinstance(precision, tuple):
    p = precision[0]
    if p in (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGH):
      return 'tf32'
    return 'ieee'
  if isinstance(precision, jax.lax.DotAlgorithmPreset):
    match precision:
      case jax.lax.DotAlgorithmPreset.TF32_TF32_F32:
        return 'tf32'
      case jax.lax.DotAlgorithmPreset.TF32_TF32_F32_X3:
        return 'tf32x3'
      case jax.lax.DotAlgorithmPreset.F32_F32_F32:
        return 'ieee'
      case _:
        raise NotImplementedError(
            f'Unsupported precision for Triton kernel: {precision}'
        )
  return 'tf32'


def get_dot_dtype_and_precision(
    precision: precision_lib.CanonicalPrecision,
    lhs_dtype: jax.typing.DTypeLike,
    rhs_dtype: jax.typing.DTypeLike,
) -> tuple[tl.dtype, str | None]:
  """Returns the Triton dot operand dtype and `input_precision` string."""
  if (
      isinstance(precision, jax.lax.DotAlgorithmPreset)
      and precision.supported_lhs_types is not None
  ):
    dot_jnp_dtype = jnp.dtype(precision.supported_lhs_types[0])
  else:
    dot_jnp_dtype = jnp.dtype(jnp.result_type(lhs_dtype, rhs_dtype))
  input_precision = get_input_precision(precision, dot_jnp_dtype)
  return jnp_to_tl_dtype(dot_jnp_dtype), input_precision

