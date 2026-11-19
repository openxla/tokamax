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
"""Pallas Mosaic TPU backward kernel for the gated linear unit op."""

from collections.abc import Callable
import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

DotPrecision = tuple[jax.lax.Precision, jax.lax.Precision]

# Contracts dim 1 of both operands: (tm, 2N) x (K, 2N) -> (tm, K).
_DX_DIM_NUMS = (((1,), (1,)), ((), ()))
# Contracts dim 0 of both operands: (tm, K) x (tm, 2N) -> (K, 2N).
_DW_DIM_NUMS = (((0,), (0,)), ((), ()))
_TILE_M_CANDIDATES = (1024, 512, 256, 128)


def _align_up(val: int, alignment: int) -> int:
  return ((val + alignment - 1) // alignment) * alignment


def _bwd_kernel(
    x_ref: Any,
    y_ref: Any,
    dout_ref: Any,
    w_ref: Any,
    dx_ref: Any,
    dw_ref: Any,
    *,
    n: int,
    activation: Callable[[jax.Array], jax.Array] | None,
    dot_precision: DotPrecision,
    dot_dtype: jnp.dtype,
) -> None:
  """Computes dx for one M tile and accumulates dW over the M tiles."""

  @pl.when(pl.program_id(1) == 0)
  def _():
    dw_ref[...] = jnp.zeros_like(dw_ref)

  y = y_ref[...].astype(jnp.float32)
  gate, up = y[:, :n], y[:, n:]
  dout = dout_ref[...].astype(jnp.float32)
  if activation is None:
    act, d_gate = gate, dout * up
  else:
    act, act_vjp = jax.vjp(activation, gate)
    (d_gate,) = act_vjp(dout * up)
  dy = jnp.concatenate([d_gate, dout * act], axis=-1).astype(dot_dtype)

  dx = jax.lax.dot_general(
      dy,
      w_ref[...].astype(dot_dtype),
      dimension_numbers=_DX_DIM_NUMS,
      precision=dot_precision,
      preferred_element_type=jnp.float32,
  )
  dx_ref[...] = dx.astype(dx_ref.dtype)
  dw_ref[0] += jax.lax.dot_general(
      x_ref[...].astype(dot_dtype),
      dy,
      dimension_numbers=_DW_DIM_NUMS,
      precision=dot_precision,
      preferred_element_type=jnp.float32,
  )


def _vmem_bytes(tile_m: int, k: int, n: int, itemsize: int) -> int:
  """Estimates the kernel's VMEM use, with double-buffered inputs/outputs."""
  per_tile = tile_m * (2 * k + 3 * n) * itemsize  # x, dx, y, dout.
  weights = k * 2 * n * itemsize
  dw = k * 2 * n * 4
  temps = tile_m * 2 * n * 4 * 4  # f32 gate/up/dy intermediates.
  return 2 * (per_tile + weights + dw) + temps


def select_tile_m(
    m: int, k: int, n: int, itemsize: int, vmem_budget: int
) -> int | None:
  """Returns the largest M tile that fits in VMEM, or None if none does."""
  if n % 128:
    return None
  k = _align_up(k, 128)
  for tile_m in _TILE_M_CANDIDATES:
    if tile_m > max(m, 128):
      continue
    if _vmem_bytes(tile_m, k, n, itemsize) <= vmem_budget:
      return tile_m
  return None


def gated_linear_unit_bwd(
    x: jax.Array,
    y: jax.Array,
    dout: jax.Array,
    w: jax.Array,
    *,
    activation: Callable[[jax.Array], jax.Array] | None,
    dot_precision: DotPrecision,
    dot_dtype: jnp.dtype,
    tile_m: int,
    num_cores: int,
    vmem_limit_bytes: int,
) -> tuple[jax.Array, jax.Array]:
  """Returns (dx, dW) for a GLU from the (M, 2N) residual y = [gate, up].

  Args:
    x: (M, K) input.
    y: (M, 2N) pre-activation gate and up projections, side by side.
    dout: (M, N) output cotangent.
    w: (K, 2N) gate and up weights, side by side.
    activation: The gate activation.
    dot_precision: Precision for both matmuls.
    dot_dtype: Operand dtype for both matmuls.
    tile_m: Rows per grid step.
    num_cores: TensorCores to split the rows over; each accumulates its own dW.
    vmem_limit_bytes: Scoped VMEM limit for the kernel.

  Returns:
    dx with x's shape and dtype, and dW as an f32 (K, 2N) array.
  """
  m, k = x.shape
  n = dout.shape[-1]
  k_padded = _align_up(k, 128)
  num_splits = max(1, min(num_cores, pl.cdiv(m, tile_m)))
  m_padded = _align_up(m, tile_m * num_splits)
  steps = m_padded // (tile_m * num_splits)

  # Zero rows and columns contribute nothing to either gradient.
  if m_padded != m or k_padded != k:
    x = jnp.pad(x, ((0, m_padded - m), (0, k_padded - k)))
  if m_padded != m:
    y = jnp.pad(y, ((0, m_padded - m), (0, 0)))
    dout = jnp.pad(dout, ((0, m_padded - m), (0, 0)))
  if k_padded != k:
    w = jnp.pad(w, ((0, k_padded - k), (0, 0)))

  row = lambda s, i: (s * steps + i, 0)
  kernel = functools.partial(
      _bwd_kernel,
      n=n,
      activation=activation,
      dot_precision=dot_precision,
      dot_dtype=dot_dtype,
  )
  itemsize = jnp.dtype(x.dtype).itemsize
  dx, dw = pl.pallas_call(
      kernel,
      grid=(num_splits, steps),
      in_specs=[
          pl.BlockSpec((tile_m, k_padded), row),
          pl.BlockSpec((tile_m, 2 * n), row),
          pl.BlockSpec((tile_m, n), row),
          pl.BlockSpec((k_padded, 2 * n), lambda s, i: (0, 0)),
      ],
      out_specs=[
          pl.BlockSpec((tile_m, k_padded), row),
          pl.BlockSpec((1, k_padded, 2 * n), lambda s, i: (s, 0, 0)),
      ],
      out_shape=[
          jax.ShapeDtypeStruct((m_padded, k_padded), x.dtype),
          jax.ShapeDtypeStruct((num_splits, k_padded, 2 * n), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      cost_estimate=pl.CostEstimate(
          flops=8 * m * k * n + 6 * m * n,
          transcendentals=m * n,
          bytes_accessed=itemsize * (2 * m * k + 3 * m * n + 2 * k * n)
          + 4 * 2 * k * n,
      ),
      name="pallas_mosaic_tpu_glu_bwd",
  )(x, y, dout, w)
  return dx[:m, :k], jnp.sum(dw, axis=0)[:k]
