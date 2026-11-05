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
"""Pallas Mosaic TPU implementation of the gated linear unit op."""

from collections.abc import Callable
import dataclasses
import functools
import math
from typing import Any, ClassVar, Final, override

import immutabledict
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import jaxtyping as jt
import pydantic
from tokamax._src import batching
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.gated_linear_unit import base

Array = jt.Array
Float = jt.Float


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Configuration parameters for Pallas Mosaic TPU GatedLinearUnit kernel.

  Attributes:
    tile_m: Tile size along the M (batch/token) dimension. Must be a multiple of
      8.
    tile_k: Tile size along the K (input channel) dimension loaded into VMEM per
      grid step. Must be a multiple of 64.
    tile_n: Tile size along the N (output channel) dimension. Must be a multiple
      of 128.
    input_buffer_count: Number of pipeline buffers for inputs in VMEM.
  """

  tile_m: pydantic.PositiveInt = 128
  tile_k: pydantic.PositiveInt = 128
  tile_n: pydantic.PositiveInt = 128
  input_buffer_count: pydantic.PositiveInt = 2

  def __post_init__(self):
    if self.tile_m % 8 != 0:
      raise ValueError(f"{self.tile_m=} must be a multiple of 8.")
    if self.tile_k % 64 != 0:
      raise ValueError(f"{self.tile_k=} must be a multiple of 64.")
    if self.tile_n % 128 != 0:
      raise ValueError(f"{self.tile_n=} must be a multiple of 128.")


type Key = immutabledict.immutabledict[str, Any]

_SUPPORTED_DTYPES: Final[frozenset[jnp.dtype]] = frozenset({
    jnp.dtype(jnp.bfloat16),
    jnp.dtype(jnp.float32),
    jnp.dtype(jnp.float16),
})

_DEFAULT_VMEM_BUDGET_BYTES: Final[int] = 14 * 1024 * 1024
# XLA keeps part of VMEM for itself (64 KiB on TPU v7x), and the kernel uses a
# few KiB that `_estimate_vmem_bytes` does not count.
_VMEM_RESERVED_BYTES: Final[int] = 1024 * 1024
_DOT_DIM_NUMS: Final[
    tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[()], tuple[()]]]
] = (
    ((1,), (0,)),
    ((), ()),
)


def _get_vmem_budget_bytes() -> int:
  """Returns the VMEM bytes available to the kernel's tiles."""
  if jax.default_backend() == "tpu":
    return pltpu.get_tpu_info().vmem_capacity_bytes - _VMEM_RESERVED_BYTES
  return _DEFAULT_VMEM_BUDGET_BYTES


def _resolve_tpu_dot_precision(
    precision: base.CanonicalPrecision,
    input_dtype: jnp.dtype,
) -> tuple[tuple[jax.lax.Precision, jax.lax.Precision], jnp.dtype]:
  """Resolves CanonicalPrecision into a TPU-compatible lax.Precision tuple."""
  if isinstance(precision, jax.lax.DotAlgorithmPreset):
    match precision:
      case jax.lax.DotAlgorithmPreset.DEFAULT:
        return (
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
            input_dtype,
        )
      case jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
        return (
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
            jnp.dtype(jnp.bfloat16),
        )
      case jax.lax.DotAlgorithmPreset.F16_F16_F32:
        return (
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
            jnp.dtype(jnp.float16),
        )
      case (
          jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3
          | jax.lax.DotAlgorithmPreset.TF32_TF32_F32
      ):
        return (
            (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
            input_dtype,
        )
      case (
          jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6
          | jax.lax.DotAlgorithmPreset.F32_F32_F32
          | jax.lax.DotAlgorithmPreset.TF32_TF32_F32_X3
      ):
        return (
            (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
            input_dtype,
        )
      case _:
        raise NotImplementedError(
            f"Unsupported DotAlgorithmPreset on Mosaic TPU: {precision}."
        )
  if (
      isinstance(precision, tuple)
      and len(precision) == 2
      and isinstance(precision[0], jax.lax.Precision)
      and isinstance(precision[1], jax.lax.Precision)
  ):
    return (precision[0], precision[1]), input_dtype
  raise NotImplementedError(
      f"Unsupported precision on Mosaic TPU: {precision}."
  )


def _contract_tile(
    x_ref: Any,
    w_gate_ref: Any,
    w_up_ref: Any,
    *,
    dot_precision: tuple[jax.lax.Precision, jax.lax.Precision],
    dot_dtype: jnp.dtype,
    init_gate: jax.Array | None = None,
    init_up: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Contracts (tile_m, tile_k) with gate and up weights on the MXU."""
  lhs = x_ref[...]
  rhs_gate = w_gate_ref[...]
  rhs_up = w_up_ref[...]

  if lhs.dtype != dot_dtype:
    lhs = lhs.astype(dot_dtype)
  if rhs_gate.dtype != dot_dtype:
    rhs_gate = rhs_gate.astype(dot_dtype)
  if rhs_up.dtype != dot_dtype:
    rhs_up = rhs_up.astype(dot_dtype)

  gate_prod = jax.lax.dot_general(
      lhs,
      rhs_gate,
      dimension_numbers=_DOT_DIM_NUMS,
      precision=dot_precision,
      preferred_element_type=jnp.float32,
  )
  up_prod = jax.lax.dot_general(
      lhs,
      rhs_up,
      dimension_numbers=_DOT_DIM_NUMS,
      precision=dot_precision,
      preferred_element_type=jnp.float32,
  )
  if init_gate is not None and init_up is not None:
    return init_gate + gate_prod, init_up + up_prod
  return gate_prod, up_prod


def _glu_single_k_no_res_kernel(
    x_ref: Any,
    w_gate_ref: Any,
    w_up_ref: Any,
    out_ref: Any,
    *,
    activation: Callable[[jax.Array], jax.Array] | None,
    dot_precision: tuple[jax.lax.Precision, jax.lax.Precision],
    dot_dtype: jnp.dtype,
) -> None:
  """Single-K-step fused GLU kernel without residuals."""
  gate_acc, up_acc = _contract_tile(
      x_ref,
      w_gate_ref,
      w_up_ref,
      dot_precision=dot_precision,
      dot_dtype=dot_dtype,
  )
  if activation is not None:
    gate_acc = activation(gate_acc)
  out = gate_acc * up_acc
  out_ref[...] = out.astype(out_ref.dtype)


def _glu_single_k_fused_res_kernel(
    x_ref: Any,
    w_gate_ref: Any,
    w_up_ref: Any,
    out_ref: Any,
    res_ref: Any,
    *,
    activation: Callable[[jax.Array], jax.Array] | None,
    dot_precision: tuple[jax.lax.Precision, jax.lax.Precision],
    dot_dtype: jnp.dtype,
) -> None:
  """Single-K-step fused GLU kernel writing a single fused (M, 2*N) residual."""
  gate_acc, up_acc = _contract_tile(
      x_ref,
      w_gate_ref,
      w_up_ref,
      dot_precision=dot_precision,
      dot_dtype=dot_dtype,
  )
  res_ref[...] = jnp.concatenate([gate_acc, up_acc], axis=-1).astype(
      res_ref.dtype
  )
  if activation is not None:
    gate_acc = activation(gate_acc)
  out = gate_acc * up_acc
  out_ref[...] = out.astype(out_ref.dtype)


def _glu_single_k_with_res_kernel(
    x_ref: Any,
    w_gate_ref: Any,
    w_up_ref: Any,
    out_ref: Any,
    gate_res_ref: Any,
    up_res_ref: Any,
    *,
    activation: Callable[[jax.Array], jax.Array] | None,
    dot_precision: tuple[jax.lax.Precision, jax.lax.Precision],
    dot_dtype: jnp.dtype,
) -> None:
  """Single-K-step fused GLU kernel with residuals for VJP."""
  gate_acc, up_acc = _contract_tile(
      x_ref,
      w_gate_ref,
      w_up_ref,
      dot_precision=dot_precision,
      dot_dtype=dot_dtype,
  )
  gate_res_ref[...] = gate_acc.astype(gate_res_ref.dtype)
  up_res_ref[...] = up_acc.astype(up_res_ref.dtype)
  if activation is not None:
    gate_acc = activation(gate_acc)
  out = gate_acc * up_acc
  out_ref[...] = out.astype(out_ref.dtype)


def _init_multi_k_acc(
    x_ref: Any,
    w_gate_ref: Any,
    w_up_ref: Any,
    gate_acc_ref: Any,
    up_acc_ref: Any,
    *,
    dot_precision: tuple[jax.lax.Precision, jax.lax.Precision],
    dot_dtype: jnp.dtype,
) -> None:
  """Initializes multi-K VMEM accumulators at k_idx == 0."""
  gate_init, up_init = _contract_tile(
      x_ref,
      w_gate_ref,
      w_up_ref,
      dot_precision=dot_precision,
      dot_dtype=dot_dtype,
  )
  gate_acc_ref[...] = gate_init
  up_acc_ref[...] = up_init


def _accum_multi_k_acc(
    x_ref: Any,
    w_gate_ref: Any,
    w_up_ref: Any,
    gate_acc_ref: Any,
    up_acc_ref: Any,
    *,
    dot_precision: tuple[jax.lax.Precision, jax.lax.Precision],
    dot_dtype: jnp.dtype,
) -> None:
  """Accumulates into multi-K VMEM accumulators at 0 < k_idx < grid_k - 1."""
  gate_next, up_next = _contract_tile(
      x_ref,
      w_gate_ref,
      w_up_ref,
      dot_precision=dot_precision,
      dot_dtype=dot_dtype,
      init_gate=gate_acc_ref[...],
      init_up=up_acc_ref[...],
  )
  gate_acc_ref[...] = gate_next
  up_acc_ref[...] = up_next


def _epilogue_multi_k_acc(
    x_ref: Any,
    w_gate_ref: Any,
    w_up_ref: Any,
    out_ref: Any,
    gate_res_ref: Any | None,
    up_res_ref: Any | None,
    gate_acc_ref: Any,
    up_acc_ref: Any,
    *,
    activation: Callable[[jax.Array], jax.Array] | None,
    dot_precision: tuple[jax.lax.Precision, jax.lax.Precision],
    dot_dtype: jnp.dtype,
) -> None:
  """Applies final K step, activation, and store at k_idx == grid_k - 1."""
  gate_final, up_final = _contract_tile(
      x_ref,
      w_gate_ref,
      w_up_ref,
      dot_precision=dot_precision,
      dot_dtype=dot_dtype,
      init_gate=gate_acc_ref[...],
      init_up=up_acc_ref[...],
  )
  if gate_res_ref is not None and up_res_ref is not None:
    gate_res_ref[...] = gate_final.astype(gate_res_ref.dtype)
    up_res_ref[...] = up_final.astype(up_res_ref.dtype)
  if activation is not None:
    gate_final = activation(gate_final)
  out = gate_final * up_final
  out_ref[...] = out.astype(out_ref.dtype)


def _glu_multi_k_step(
    k_idx: jax.Array,
    grid_k: int,
    x_ref: Any,
    w_gate_ref: Any,
    w_up_ref: Any,
    out_ref: Any,
    gate_res_ref: Any | None,
    up_res_ref: Any | None,
    gate_acc_ref: Any,
    up_acc_ref: Any,
    *,
    activation: Callable[[jax.Array], jax.Array] | None,
    dot_precision: tuple[jax.lax.Precision, jax.lax.Precision],
    dot_dtype: jnp.dtype,
) -> None:
  """Multi-K-step fused GLU accumulation and epilogue."""
  pl.when(k_idx == 0)(
      functools.partial(
          _init_multi_k_acc,
          x_ref,
          w_gate_ref,
          w_up_ref,
          gate_acc_ref,
          up_acc_ref,
          dot_precision=dot_precision,
          dot_dtype=dot_dtype,
      )
  )
  if grid_k > 2:
    pl.when((k_idx > 0) & (k_idx < grid_k - 1))(
        functools.partial(
            _accum_multi_k_acc,
            x_ref,
            w_gate_ref,
            w_up_ref,
            gate_acc_ref,
            up_acc_ref,
            dot_precision=dot_precision,
            dot_dtype=dot_dtype,
        )
    )
  pl.when(k_idx == grid_k - 1)(
      functools.partial(
          _epilogue_multi_k_acc,
          x_ref,
          w_gate_ref,
          w_up_ref,
          out_ref,
          gate_res_ref,
          up_res_ref,
          gate_acc_ref,
          up_acc_ref,
          activation=activation,
          dot_precision=dot_precision,
          dot_dtype=dot_dtype,
      )
  )


def _glu_multi_k_no_res_kernel(
    x_ref: Any,
    w_gate_ref: Any,
    w_up_ref: Any,
    out_ref: Any,
    gate_acc_ref: Any,
    up_acc_ref: Any,
    *,
    grid_k: int,
    activation: Callable[[jax.Array], jax.Array] | None,
    dot_precision: tuple[jax.lax.Precision, jax.lax.Precision],
    dot_dtype: jnp.dtype,
) -> None:
  """Multi-K-step fused GLU kernel without residuals."""
  _glu_multi_k_step(
      pl.program_id(2),
      grid_k,
      x_ref,
      w_gate_ref,
      w_up_ref,
      out_ref,
      None,
      None,
      gate_acc_ref,
      up_acc_ref,
      activation=activation,
      dot_precision=dot_precision,
      dot_dtype=dot_dtype,
  )


def _glu_multi_k_with_res_kernel(
    x_ref: Any,
    w_gate_ref: Any,
    w_up_ref: Any,
    out_ref: Any,
    gate_res_ref: Any,
    up_res_ref: Any,
    gate_acc_ref: Any,
    up_acc_ref: Any,
    *,
    grid_k: int,
    activation: Callable[[jax.Array], jax.Array] | None,
    dot_precision: tuple[jax.lax.Precision, jax.lax.Precision],
    dot_dtype: jnp.dtype,
) -> None:
  """Multi-K-step fused GLU kernel with residuals."""
  _glu_multi_k_step(
      pl.program_id(2),
      grid_k,
      x_ref,
      w_gate_ref,
      w_up_ref,
      out_ref,
      gate_res_ref,
      up_res_ref,
      gate_acc_ref,
      up_acc_ref,
      activation=activation,
      dot_precision=dot_precision,
      dot_dtype=dot_dtype,
  )


def _align_up(val: int, alignment: int) -> int:
  return ((val + alignment - 1) // alignment) * alignment


def _gated_linear_unit_tpu(
    x: jax.Array,
    weights: base.FusedWeights | base.UnfusedWeights,
    *,
    activation: Callable[[jax.Array], jax.Array] | None,
    dot_precision: tuple[jax.lax.Precision, jax.lax.Precision],
    dot_dtype: jnp.dtype,
    return_residuals: bool,
    config: Config,
) -> tuple[jax.Array, jax.Array | None]:
  """Executes the Pallas Mosaic TPU GatedLinearUnit kernel."""
  orig_dtype = x.dtype
  compute_dtype = (
      jnp.dtype(jnp.float32)
      if orig_dtype == jnp.dtype(jnp.float16)
      else orig_dtype
  )
  effective_dot_dtype = (
      jnp.dtype(jnp.float32)
      if dot_dtype == jnp.dtype(jnp.float16)
      else dot_dtype
  )

  batch_shape = x.shape[:-1]
  x_2d = jax.lax.collapse(x, 0, -1)
  if x_2d.dtype != compute_dtype:
    x_2d = x_2d.astype(compute_dtype)

  m_orig, k_orig = x_2d.shape
  n_orig = (
      weights[0].shape[-1] if isinstance(weights, tuple) else weights.shape[-1]
  )

  tile_m = config.tile_m
  tile_k = config.tile_k
  tile_n = config.tile_n

  m_padded = _align_up(m_orig, tile_m)
  k_padded = _align_up(k_orig, tile_k)
  n_padded = _align_up(n_orig, tile_n)
  # When tile_k < 128 (e.g. K=64), TPU 2D BlockSpecs require minor dim >= 128.
  if k_padded < 128:
    k_padded = 128
    tile_k = 128

  needs_m_pad = m_padded != m_orig
  needs_k_pad = k_padded != k_orig
  needs_n_pad = n_padded != n_orig

  if needs_m_pad or needs_k_pad:
    x_2d = jnp.pad(x_2d, ((0, m_padded - m_orig), (0, k_padded - k_orig)))

  grid_m = m_padded // tile_m
  grid_k = k_padded // tile_k
  grid_n = n_padded // tile_n

  use_fused_2d_weights = (
      not isinstance(weights, tuple) and not needs_k_pad and not needs_n_pad
  )
  if use_fused_2d_weights:
    assert not isinstance(weights, tuple)
    w_2d = jax.lax.collapse(weights, 1, 3)
    if w_2d.dtype != compute_dtype:
      w_2d = w_2d.astype(compute_dtype)
    w_gate_in = w_2d
    w_up_in = w_2d
    up_col_offset = grid_n
  else:
    if isinstance(weights, tuple):
      w_gate, w_up = weights
    else:
      w_gate = weights[:, 0, :]
      w_up = weights[:, 1, :]
    if w_gate.dtype != compute_dtype:
      w_gate = w_gate.astype(compute_dtype)
      w_up = w_up.astype(compute_dtype)
    if needs_k_pad or needs_n_pad:
      pad_width = ((0, k_padded - k_orig), (0, n_padded - n_orig))
      w_gate = jnp.pad(w_gate, pad_width)
      w_up = jnp.pad(w_up, pad_width)
    w_gate_in = w_gate
    w_up_in = w_up
    up_col_offset = 0

  # Choose outer loop order (M-major vs N-major) to minimize HBM re-reads when
  # grid_k == 1.
  n_major = (
      grid_k == 1 and (grid_n - 1) * m_padded < (grid_m - 1) * 2 * n_padded
  )

  if n_major:
    grid = (grid_n, grid_m, grid_k)
    x_index_map = lambda j, i, k: (i, k)
    w_gate_index_map = lambda j, i, k: (k, j)
    w_up_index_map = lambda j, i, k: (k, j + up_col_offset)
    out_index_map = lambda j, i, k: (i, j)
  else:
    grid = (grid_m, grid_n, grid_k)
    x_index_map = lambda i, j, k: (i, k)
    w_gate_index_map = lambda i, j, k: (k, j)
    w_up_index_map = lambda i, j, k: (k, j + up_col_offset)
    out_index_map = lambda i, j, k: (i, j)

  in_specs = [
      pl.BlockSpec(
          (tile_m, tile_k),
          x_index_map,
          pipeline_mode=pl.Buffered(buffer_count=config.input_buffer_count),
      ),
      pl.BlockSpec(
          (tile_k, tile_n),
          w_gate_index_map,
          pipeline_mode=pl.Buffered(buffer_count=config.input_buffer_count),
      ),
      pl.BlockSpec(
          (tile_k, tile_n),
          w_up_index_map,
          pipeline_mode=pl.Buffered(buffer_count=config.input_buffer_count),
      ),
  ]

  out_block_spec = pl.BlockSpec((tile_m, tile_n), out_index_map)
  out_struct = jax.ShapeDtypeStruct((m_padded, n_padded), compute_dtype)
  use_fused_res_buffer = return_residuals and grid_k == 1 and grid_n == 1

  if grid_k == 1:
    scratch_shapes = ()
    if use_fused_res_buffer:
      kernel_fn = functools.partial(
          _glu_single_k_fused_res_kernel,
          activation=activation,
          dot_precision=dot_precision,
          dot_dtype=effective_dot_dtype,
      )
      res_struct = jax.ShapeDtypeStruct((m_padded, 2 * n_padded), compute_dtype)
      res_block_spec = pl.BlockSpec((tile_m, 2 * tile_n), out_index_map)
      out_shape = (out_struct, res_struct)
      out_specs = (out_block_spec, res_block_spec)
    elif return_residuals:
      kernel_fn = functools.partial(
          _glu_single_k_with_res_kernel,
          activation=activation,
          dot_precision=dot_precision,
          dot_dtype=effective_dot_dtype,
      )
      out_shape = (out_struct, out_struct, out_struct)
      out_specs = (out_block_spec, out_block_spec, out_block_spec)
    else:
      kernel_fn = functools.partial(
          _glu_single_k_no_res_kernel,
          activation=activation,
          dot_precision=dot_precision,
          dot_dtype=effective_dot_dtype,
      )
      out_shape = out_struct
      out_specs = out_block_spec
  else:
    acc_scratch = pltpu.VMEM((tile_m, tile_n), jnp.float32)
    scratch_shapes = (acc_scratch, acc_scratch)
    if return_residuals:
      kernel_fn = functools.partial(
          _glu_multi_k_with_res_kernel,
          grid_k=grid_k,
          activation=activation,
          dot_precision=dot_precision,
          dot_dtype=effective_dot_dtype,
      )
      out_shape = (out_struct, out_struct, out_struct)
      out_specs = (out_block_spec, out_block_spec, out_block_spec)
    else:
      kernel_fn = functools.partial(
          _glu_multi_k_no_res_kernel,
          grid_k=grid_k,
          activation=activation,
          dot_precision=dot_precision,
          dot_dtype=effective_dot_dtype,
      )
      out_shape = out_struct
      out_specs = out_block_spec

  itemsize = jnp.dtype(compute_dtype).itemsize
  flops = 4 * m_orig * k_orig * n_orig + 2 * m_orig * n_orig
  num_out_buffers = 3 if return_residuals else 1
  bytes_accessed = itemsize * (
      m_orig * k_orig + 2 * k_orig * n_orig + num_out_buffers * m_orig * n_orig
  )
  cost_estimate = pl.CostEstimate(
      flops=flops,
      transcendentals=m_orig * n_orig,
      bytes_accessed=bytes_accessed,
  )

  kernel_name = "pallas_mosaic_tpu_glu"
  if return_residuals:
    kernel_name += "_fwd_res"

  res = pl.pallas_call(
      kernel_fn,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      out_shape=out_shape,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
      ),
      cost_estimate=cost_estimate,
      name=kernel_name,
  )(x_2d, w_gate_in, w_up_in)

  if use_fused_res_buffer:
    out, res_2d = res
    if needs_m_pad or needs_n_pad:
      out = out[:m_orig, :n_orig]
      res_3d = res_2d.reshape(m_padded, 2, n_padded)[:m_orig, :, :n_orig]
    else:
      res_3d = res_2d.reshape(m_orig, 2, n_orig)
    if out.dtype != orig_dtype:
      out = out.astype(orig_dtype)
      res_3d = res_3d.astype(orig_dtype)
    return out.reshape(*batch_shape, n_orig), res_3d.reshape(
        *batch_shape, 2, n_orig
    )

  if return_residuals:
    out, gate_res, up_res = res
    if needs_m_pad or needs_n_pad:
      out = out[:m_orig, :n_orig]
      gate_res = gate_res[:m_orig, :n_orig]
      up_res = up_res[:m_orig, :n_orig]
    if out.dtype != orig_dtype:
      out = out.astype(orig_dtype)
      gate_res = gate_res.astype(orig_dtype)
      up_res = up_res.astype(orig_dtype)
    out = out.reshape(*batch_shape, n_orig)
    residuals = jnp.stack([gate_res, up_res], axis=-2).reshape(
        *batch_shape, 2, n_orig
    )
    return out, residuals

  out = res
  if needs_m_pad or needs_n_pad:
    out = out[:m_orig, :n_orig]
  if out.dtype != orig_dtype:
    out = out.astype(orig_dtype)
  return out.reshape(*batch_shape, n_orig), None


def _estimate_vmem_bytes(
    tile_m: int,
    tile_k: int,
    tile_n: int,
    input_buffer_count: int,
    itemsize: int,
    *,
    single_k_step: bool,
    return_residuals: bool,
) -> int:
  """Estimates VMEM bytes required by a tile configuration.

  Counts the pipeline buffers, the multi-K accumulators, and the f32 values the
  kernel body keeps live: the gate and up products and the gated output, plus
  the concatenated residual when residuals are returned.
  """
  x_bytes = input_buffer_count * tile_m * tile_k * itemsize
  w_bytes = 2 * input_buffer_count * tile_k * tile_n * itemsize
  num_out = 3 if return_residuals else 1
  out_bytes = 2 * num_out * tile_m * tile_n * itemsize
  acc_bytes = 0 if single_k_step else (2 * tile_m * tile_n * 4)
  num_f32_values = 5 if return_residuals else 3
  body_bytes = num_f32_values * tile_m * tile_n * 4
  return x_bytes + w_bytes + out_bytes + acc_bytes + body_bytes


def _select_heuristic_config(
    m_dim: int,
    k_dim: int,
    n_dim: int,
    itemsize: int,
    *,
    return_residuals: bool,
    num_cores: int,
) -> Config:
  """Selects high-performance tile sizes for TPU v5+."""
  vmem_budget = _get_vmem_budget_bytes()
  k_padded = max(128, _align_up(k_dim, 128))
  n_padded = max(128, _align_up(n_dim, 128))
  m_padded = max(128, _align_up(m_dim, 128))

  if k_padded <= 4096:
    tile_k = k_padded
    if m_padded >= 8192 and n_padded <= 512:
      # Large-M, small-N regime (e.g., AlphaFold): keep full N in one tile so
      # weights stay stationary in VMEM and x streams once with large tile_m.
      mn_candidates = (
          (1024, n_padded),
          (512, n_padded),
          (256, n_padded),
          (128, n_padded),
          (512, 256),
          (256, 256),
          (128, 128),
      )
    else:
      # Prioritize tiles with both tile_m >= 256 and tile_n >= 256 to saturate
      # the 256x256 MXU on TPU v6e+ as well as 128x128 MXUs on TPU v5.
      mn_candidates = (
          (512, 512),
          (512, 1024),
          (1024, 512),
          (256, 512),
          (512, 256),
          (256, 256),
          (256, 128),
          (128, 256),
          (128, 128),
      )

    for tile_m, tile_n in mn_candidates:
      if tile_m > m_padded or tile_n > n_padded or n_padded % tile_n != 0:
        continue
      grid_m = math.ceil(m_padded / tile_m)
      grid_n = n_padded // tile_n
      if (grid_m * grid_n) < num_cores and (tile_m > 128 or tile_n > 128):
        continue
      vmem = _estimate_vmem_bytes(
          tile_m,
          tile_k,
          tile_n,
          2,
          itemsize,
          single_k_step=True,
          return_residuals=return_residuals,
      )
      if vmem <= vmem_budget:
        return Config(
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
            input_buffer_count=2,
        )

  # Multi-K fallback when full K exceeds VMEM with 256x256+ tiles.
  for tile_k in (2048, 1024, 512, 256, 128):
    if tile_k > k_padded or k_padded % tile_k != 0:
      continue
    for tile_m, tile_n in (
        (512, 512),
        (256, 512),
        (512, 256),
        (256, 256),
        (128, 256),
        (256, 128),
        (128, 128),
    ):
      if tile_m > m_padded or tile_n > n_padded or n_padded % tile_n != 0:
        continue
      vmem = _estimate_vmem_bytes(
          tile_m,
          tile_k,
          tile_n,
          2,
          itemsize,
          single_k_step=(tile_k == k_padded),
          return_residuals=return_residuals,
      )
      if vmem <= vmem_budget:
        return Config(
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
            input_buffer_count=2,
        )

  return Config(
      tile_m=128,
      tile_k=128,
      tile_n=128,
      input_buffer_count=2,
  )


@dataclasses.dataclass(frozen=True, slots=True)
class PallasMosaicTpuGatedLinearUnit(base.GatedLinearUnit[Config, Key]):
  """Pallas-Mosaic-TPU gated linear unit implementation."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      x: Float[Array, "*B M K"],
      weights: base.FusedWeights | base.UnfusedWeights,
      *,
      activation: Callable[[jax.Array], jax.Array] | None,
      precision: base.CanonicalPrecision,
      return_residuals: bool,
      config: Config,
  ) -> tuple[Float[Array, "*B M N"], base.Residuals | None]:
    if jax.default_backend() != "tpu":
      raise NotImplementedError("Mosaic TPU is only supported on TPU.")

    x_dtype = jnp.dtype(x.dtype)
    if isinstance(weights, tuple):
      w_dtype = jnp.dtype(weights[0].dtype)
      if jnp.dtype(weights[1].dtype) != w_dtype:
        raise NotImplementedError(
            "Tuple weights must have identical dtypes, got"
            f" {weights[0].dtype} and {weights[1].dtype}."
        )
    else:
      w_dtype = jnp.dtype(weights.dtype)

    if x_dtype not in _SUPPORTED_DTYPES:
      raise NotImplementedError(
          f"Mosaic TPU GLU only supports {_SUPPORTED_DTYPES}, got {x_dtype}."
      )
    if x_dtype != w_dtype:
      raise NotImplementedError(
          "Mosaic TPU GLU requires matching x and weights dtypes, got"
          f" {x_dtype} and {w_dtype}."
      )

    dot_precision, dot_dtype = _resolve_tpu_dot_precision(precision, x_dtype)

    if isinstance(weights, tuple):
      return _gated_linear_unit_tpu(
          x,
          weights,
          activation=activation,
          dot_precision=dot_precision,
          dot_dtype=dot_dtype,
          return_residuals=return_residuals,
          config=config,
      )

    fn = functools.partial(
        _gated_linear_unit_tpu,
        activation=activation,
        dot_precision=dot_precision,
        dot_dtype=dot_dtype,
        return_residuals=return_residuals,
        config=config,
    )
    fn = self._with_vmap(fn, fallback_to_sequential=False)
    return fn(x, weights)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    input_tensor, weights = ba.args
    x_shape = (
        input_tensor.vmap_shape
        if isinstance(input_tensor, batching.BatchedShapeDtype)
        else input_tensor.shape
    )
    m_dim = math.prod(x_shape[:-1])
    k_dim = x_shape[-1]
    n_dim = (
        weights[0].shape[-1]
        if isinstance(weights, tuple)
        else weights.shape[-1]
    )
    itemsize = jnp.dtype(input_tensor.dtype).itemsize
    return_residuals = bool(ba.arguments.get("return_residuals", False))
    num_cores = pltpu.get_tpu_info().num_cores
    return _select_heuristic_config(
        m_dim,
        k_dim,
        n_dim,
        itemsize,
        return_residuals=return_residuals,
        num_cores=num_cores,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    input_tensor, weights = ba.args
    x_shape = (
        input_tensor.vmap_shape
        if isinstance(input_tensor, batching.BatchedShapeDtype)
        else input_tensor.shape
    )
    m_dim = math.prod(x_shape[:-1])
    k_dim = x_shape[-1]
    n_dim = (
        weights[0].shape[-1]
        if isinstance(weights, tuple)
        else weights.shape[-1]
    )
    itemsize = jnp.dtype(input_tensor.dtype).itemsize
    return_residuals = bool(ba.arguments.get("return_residuals", False))

    m_padded = max(128, _align_up(m_dim, 128))
    k_padded = max(128, _align_up(k_dim, 128))
    n_padded = max(128, _align_up(n_dim, 128))

    vmem_budget = _get_vmem_budget_bytes()
    configs: set[Config] = {self._get_heuristics_config(ba)}
    for tile_m in (128, 256, 512, 1024):
      if tile_m > m_padded:
        continue
      for tile_n in (128, 256, 512, 1024):
        if tile_n > n_padded or n_padded % tile_n != 0:
          continue
        for tile_k in (128, 256, 512, 1024, 2048, 4096):
          if tile_k > k_padded or k_padded % tile_k != 0:
            continue
          vmem = _estimate_vmem_bytes(
              tile_m,
              tile_k,
              tile_n,
              2,
              itemsize,
              single_k_step=(tile_k == k_padded),
              return_residuals=return_residuals,
          )
          if vmem <= vmem_budget:
            configs.add(
                Config(
                    tile_m=tile_m,
                    tile_k=tile_k,
                    tile_n=tile_n,
                    input_buffer_count=2,
                )
            )
    return configs
