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
import enum
import functools
import math
from typing import Any, ClassVar, Final, Literal, override

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
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_tpu_vjp

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
    xla_residuals: Whether a forward pass with residuals uses XLA instead, for
      shapes that train faster with XLA's forward and VJP.
  """

  tile_m: pydantic.PositiveInt = 128
  tile_k: pydantic.PositiveInt = 128
  tile_n: pydantic.PositiveInt = 128
  input_buffer_count: pydantic.PositiveInt = 2
  xla_residuals: bool = False

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
_DOT_DIM_NUMS: Final[
    tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[()], tuple[()]]]
] = (
    ((1,), (0,)),
    ((), ()),
)


# Mosaic's smallest default scoped VMEM limit (TPU v5e).
_DEFAULT_SCOPED_VMEM_BYTES: Final[int] = 16 * 1024 * 1024


def _vmem_limit_bytes(estimate: int) -> int | None:
  """Returns the scoped VMEM limit: all of VMEM, but only for tiles that need it.

  Reserving all of VMEM for every call measured 10% slower on TPU v5e and v7x,
  with a few microseconds added even to tiny calls, so tiles that fit the
  default limit keep it.

  Args:
    estimate: The tiles' estimated VMEM use in bytes.
  """
  if jax.default_backend() != "tpu" or estimate <= _DEFAULT_SCOPED_VMEM_BYTES:
    return None
  return pltpu.get_tpu_info().vmem_capacity_bytes


def _get_vmem_budget_bytes() -> int:
  """Returns 85% of hardware VMEM capacity in bytes."""
  if jax.default_backend() == "tpu":
    return int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.85)
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


def _align_up(val: int, alignment: int) -> int:
  return ((val + alignment - 1) // alignment) * alignment


class _Residuals(enum.Enum):
  """How the forward kernel writes the residuals for the VJP."""

  NONE = enum.auto()
  # One (M, 2N) output, gate then up. Needs a single N tile.
  FUSED = enum.auto()
  # Separate (M, N) gate and up outputs.
  SPLIT = enum.auto()


@dataclasses.dataclass(frozen=True, kw_only=True)
class _KernelPlan:
  """Static parameters of one forward kernel call.

  `Config` holds the tunable tile sizes. The plan resolves everything that
  follows from them and the shapes (padding, grid, loop order, weight and
  residual layouts), so the kernel body and its specs only read the plan.
  """

  m: int
  k: int
  n: int
  tile_m: int
  tile_k: int
  tile_n: int
  m_padded: int
  k_padded: int
  n_padded: int
  input_buffer_count: int
  compute_dtype: jnp.dtype
  dot_dtype: jnp.dtype
  dot_precision: tuple[jax.lax.Precision, jax.lax.Precision]
  activation: Callable[[jax.Array], jax.Array] | None
  residuals: _Residuals
  # Gate and up weights are read from one (K, 2N) array.
  fused_weights: bool
  # N tiles are the outer loop (only with a single K step).
  n_major: bool

  @property
  def grid_m(self) -> int:
    return self.m_padded // self.tile_m

  @property
  def grid_k(self) -> int:
    return self.k_padded // self.tile_k

  @property
  def grid_n(self) -> int:
    return self.n_padded // self.tile_n

  @property
  def num_outputs(self) -> int:
    return {_Residuals.NONE: 1, _Residuals.FUSED: 2, _Residuals.SPLIT: 3}[
        self.residuals
    ]

  @classmethod
  def create(
      cls,
      x: jax.Array,
      weights: base.FusedWeights | base.UnfusedWeights,
      *,
      activation: Callable[[jax.Array], jax.Array] | None,
      dot_precision: tuple[jax.lax.Precision, jax.lax.Precision],
      dot_dtype: jnp.dtype,
      return_residuals: bool,
      config: Config,
  ) -> "_KernelPlan":
    """Resolves the plan for `x` and `weights` from `config`."""
    m, k = math.prod(x.shape[:-1]), x.shape[-1]
    n = (
        weights[0].shape[-1]
        if isinstance(weights, tuple)
        else weights.shape[-1]
    )
    tile_k = config.tile_k
    m_padded = _align_up(m, config.tile_m)
    k_padded = _align_up(k, tile_k)
    n_padded = _align_up(n, config.tile_n)
    if k_padded < 128:  # TPU BlockSpecs need a minor dim >= 128 (e.g. K=64).
      k_padded = tile_k = 128
    grid_m, grid_n = m_padded // config.tile_m, n_padded // config.tile_n
    single_k = k_padded == tile_k
    if not return_residuals:
      residuals = _Residuals.NONE
    elif single_k and grid_n == 1:
      residuals = _Residuals.FUSED
    else:
      residuals = _Residuals.SPLIT
    f16, f32 = jnp.dtype(jnp.float16), jnp.dtype(jnp.float32)
    return cls(
        m=m,
        k=k,
        n=n,
        tile_m=config.tile_m,
        tile_k=tile_k,
        tile_n=config.tile_n,
        m_padded=m_padded,
        k_padded=k_padded,
        n_padded=n_padded,
        input_buffer_count=config.input_buffer_count,
        compute_dtype=f32 if x.dtype == f16 else jnp.dtype(x.dtype),
        dot_dtype=f32 if dot_dtype == f16 else jnp.dtype(dot_dtype),
        dot_precision=dot_precision,
        activation=activation,
        residuals=residuals,
        fused_weights=(
            not isinstance(weights, tuple) and (k_padded, n_padded) == (k, n)
        ),
        # Minimizes HBM re-reads of x (N-major) vs. the weights (M-major).
        n_major=(
            single_k and (grid_n - 1) * m_padded < (grid_m - 1) * 2 * n_padded
        ),
    )


def _glu_kernel(
    x_ref: Any, w_gate_ref: Any, w_up_ref: Any, *refs: Any, plan: _KernelPlan
) -> None:
  """Computes one output tile, accumulating over K tiles if there are several.

  Args:
    x_ref: The (tile_m, tile_k) input tile.
    w_gate_ref: The (tile_k, tile_n) gate weight tile.
    w_up_ref: The (tile_k, tile_n) up weight tile.
    *refs: The output refs (out, then the residuals per `plan.residuals`),
      followed by the f32 gate and up accumulators if there are several K tiles.
    plan: The kernel plan.
  """
  out_ref, *res_refs = refs[: plan.num_outputs]
  dot = functools.partial(
      jax.lax.dot_general,
      dimension_numbers=_DOT_DIM_NUMS,
      precision=plan.dot_precision,
      preferred_element_type=jnp.float32,
  )

  def contract() -> tuple[jax.Array, jax.Array]:
    x = x_ref[...].astype(plan.dot_dtype)
    return (
        dot(x, w_gate_ref[...].astype(plan.dot_dtype)),
        dot(x, w_up_ref[...].astype(plan.dot_dtype)),
    )

  def store(gate: jax.Array, up: jax.Array) -> None:
    if plan.residuals == _Residuals.FUSED:
      (res_ref,) = res_refs
      res_ref[...] = jnp.concatenate([gate, up], axis=-1).astype(res_ref.dtype)
    elif plan.residuals == _Residuals.SPLIT:
      gate_res_ref, up_res_ref = res_refs
      gate_res_ref[...] = gate.astype(gate_res_ref.dtype)
      up_res_ref[...] = up.astype(up_res_ref.dtype)
    if plan.activation is not None:
      gate = plan.activation(gate)
    out_ref[...] = (gate * up).astype(out_ref.dtype)

  if plan.grid_k == 1:
    store(*contract())
    return

  gate_acc_ref, up_acc_ref = refs[plan.num_outputs :]
  k_idx = pl.program_id(2)

  @pl.when(k_idx == 0)
  def _init():
    gate_acc_ref[...], up_acc_ref[...] = contract()

  if plan.grid_k > 2:

    @pl.when((k_idx > 0) & (k_idx < plan.grid_k - 1))
    def _accumulate():
      gate, up = contract()
      gate_acc_ref[...] += gate
      up_acc_ref[...] += up

  @pl.when(k_idx == plan.grid_k - 1)
  def _epilogue():
    gate, up = contract()
    store(gate_acc_ref[...] + gate, up_acc_ref[...] + up)


def _kernel_inputs(
    x: jax.Array,
    weights: base.FusedWeights | base.UnfusedWeights,
    plan: _KernelPlan,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Returns x as a padded (M, K) array and the gate and up weight arrays."""
  x_2d = jax.lax.collapse(x, 0, -1).astype(plan.compute_dtype)
  if (plan.m_padded, plan.k_padded) != (plan.m, plan.k):
    x_2d = jnp.pad(
        x_2d, ((0, plan.m_padded - plan.m), (0, plan.k_padded - plan.k))
    )
  if plan.fused_weights:
    assert not isinstance(weights, tuple)
    w_2d = jax.lax.collapse(weights, 1, 3).astype(plan.compute_dtype)
    return x_2d, w_2d, w_2d
  if isinstance(weights, tuple):
    w_gate, w_up = weights
  else:
    w_gate, w_up = weights[:, 0, :], weights[:, 1, :]
  pad = ((0, plan.k_padded - plan.k), (0, plan.n_padded - plan.n))
  w_gate = w_gate.astype(plan.compute_dtype)
  w_up = w_up.astype(plan.compute_dtype)
  if pad != ((0, 0), (0, 0)):
    w_gate, w_up = jnp.pad(w_gate, pad), jnp.pad(w_up, pad)
  return x_2d, w_gate, w_up


def _grid_spec(plan: _KernelPlan) -> pltpu.PrefetchScalarGridSpec:
  """Returns the grid, block specs and scratch buffers for `plan`."""
  if plan.n_major:
    grid = (plan.grid_n, plan.grid_m, plan.grid_k)
    mnk = lambda j, i, k: (i, j, k)
  else:
    grid = (plan.grid_m, plan.grid_n, plan.grid_k)
    mnk = lambda i, j, k: (i, j, k)
  # In the fused (K, 2N) weights, the up columns follow the gate columns.
  up_offset = plan.grid_n if plan.fused_weights else 0
  tm, tk, tn = plan.tile_m, plan.tile_k, plan.tile_n
  buffered = pl.Buffered(buffer_count=plan.input_buffer_count)

  def x_map(*g):
    i, _, k = mnk(*g)
    return i, k

  def w_gate_map(*g):
    _, j, k = mnk(*g)
    return k, j

  def w_up_map(*g):
    _, j, k = mnk(*g)
    return k, j + up_offset

  def out_map(*g):
    i, j, _ = mnk(*g)
    return i, j

  out_spec = pl.BlockSpec((tm, tn), out_map)
  out_specs = {
      _Residuals.NONE: out_spec,
      _Residuals.FUSED: (out_spec, pl.BlockSpec((tm, 2 * tn), out_map)),
      _Residuals.SPLIT: (out_spec, out_spec, out_spec),
  }[plan.residuals]
  acc = pltpu.VMEM((tm, tn), jnp.float32)
  return pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      in_specs=[
          pl.BlockSpec((tm, tk), x_map, pipeline_mode=buffered),
          pl.BlockSpec((tk, tn), w_gate_map, pipeline_mode=buffered),
          pl.BlockSpec((tk, tn), w_up_map, pipeline_mode=buffered),
      ],
      out_specs=out_specs,
      grid=grid,
      scratch_shapes=() if plan.grid_k == 1 else (acc, acc),
  )


def _out_shape(
    plan: _KernelPlan,
) -> jax.ShapeDtypeStruct | tuple[jax.ShapeDtypeStruct, ...]:
  out = jax.ShapeDtypeStruct((plan.m_padded, plan.n_padded), plan.compute_dtype)
  if plan.residuals == _Residuals.FUSED:
    res = jax.ShapeDtypeStruct(
        (plan.m_padded, 2 * plan.n_padded), plan.compute_dtype
    )
    return out, res
  if plan.residuals == _Residuals.SPLIT:
    return out, out, out
  return out


def _unpack_outputs(
    outs: Any, plan: _KernelPlan, x: jax.Array
) -> tuple[jax.Array, jax.Array | None]:
  """Crops the kernel outputs and returns (out, residuals) in x's shape."""
  m, n, batch_shape = plan.m, plan.n, x.shape[:-1]
  crop = lambda a: a if a.shape == (m, n) else a[:m, :n]
  if plan.residuals == _Residuals.NONE:
    return crop(outs).astype(x.dtype).reshape(*batch_shape, n), None
  if plan.residuals == _Residuals.FUSED:
    out, res = outs
    res = res.reshape(plan.m_padded, 2, plan.n_padded)
    if res.shape != (m, 2, n):
      res = res[:m, :, :n]
  else:
    out, gate, up = outs
    # Same values as stacking on axis -2, but built from a lane-dense (M, 2N)
    # array, so the VJP's reshape back to (M, 2N) folds away.
    res = jnp.concatenate([crop(gate), crop(up)], axis=-1).reshape(m, 2, n)
  out = crop(out).astype(x.dtype).reshape(*batch_shape, n)
  return out, res.astype(x.dtype).reshape(*batch_shape, 2, n)


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
  plan = _KernelPlan.create(
      x,
      weights,
      activation=activation,
      dot_precision=dot_precision,
      dot_dtype=dot_dtype,
      return_residuals=return_residuals,
      config=config,
  )
  m, k, n = plan.m, plan.k, plan.n
  itemsize = plan.compute_dtype.itemsize
  vmem = _estimate_vmem_bytes(
      plan.tile_m,
      plan.tile_k,
      plan.tile_n,
      plan.input_buffer_count,
      itemsize,
      single_k_step=plan.grid_k == 1,
      return_residuals=return_residuals,
  )
  num_outputs = 3 if return_residuals else 1
  outs = pl.pallas_call(
      functools.partial(_glu_kernel, plan=plan),
      grid_spec=_grid_spec(plan),
      out_shape=_out_shape(plan),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          vmem_limit_bytes=_vmem_limit_bytes(vmem),
      ),
      cost_estimate=pl.CostEstimate(
          flops=4 * m * k * n + 2 * m * n,
          transcendentals=m * n,
          bytes_accessed=itemsize * (m * k + 2 * k * n + num_outputs * m * n),
      ),
      name="pallas_mosaic_tpu_glu" + ("_fwd_res" if return_residuals else ""),
  )(*_kernel_inputs(x, weights, plan))
  return _unpack_outputs(outs, plan, x)


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
  """Estimates VMEM bytes required by a tile configuration."""
  x_bytes = input_buffer_count * tile_m * tile_k * itemsize
  w_bytes = 2 * input_buffer_count * tile_k * tile_n * itemsize
  num_out = 3 if return_residuals else 1
  out_bytes = 2 * num_out * tile_m * tile_n * itemsize
  acc_bytes = 0 if single_k_step else (2 * tile_m * tile_n * 4)
  return x_bytes + w_bytes + out_bytes + acc_bytes


# Row tiles that pad M by more than this fraction waste more MXU work than the
# larger tile saves. On 256x256-MXU chips (v6+), halving a tile with N < 512
# costs more in per-step overhead than the padding, so it only applies there to
# tiles with N >= 512.
_MAX_M_PADDING_WASTE: Final[float] = 0.1


def _single_k_tiles_ok(
    m_dim: int,
    n_padded: int,
    tile_m: int,
    tile_n: int,
    *,
    num_cores: int,
    generation: int,
) -> bool:
  """Returns whether an (M, N) tile pair suits a single-K-step forward."""
  m_padded = max(128, _align_up(m_dim, 128))
  if tile_m > m_padded or tile_n > n_padded or n_padded % tile_n != 0:
    return False
  grid_m = math.ceil(m_padded / tile_m)
  if grid_m * (n_padded // tile_n) < num_cores and max(tile_m, tile_n) > 128:
    return False
  if tile_m == 128 or (generation >= 6 and tile_n < 512):
    return True
  return (grid_m * tile_m - m_dim) / m_dim <= _MAX_M_PADDING_WASTE


# Mid-size shapes (below `_WIDE_N_MAX_ROWS` rows, K and N up to these limits)
# are fastest with M tiles that divide M exactly (padding copies x) and N tiles
# as wide as possible, so x is read from HBM fewer times. TPU v5 prefers
# 256-row tiles, with VMEM raised as needed; TPU v6+ prefers the largest M tile
# whose working set fits the default scoped VMEM.
_WIDE_N_MAX_ROWS: Final[int] = 8192
_WIDE_N_MAX_K: Final[int] = 1024
_WIDE_N_MAX_N: Final[int] = 1536


def _wide_n_config(
    m_dim: int,
    k_padded: int,
    n_padded: int,
    itemsize: int,
    *,
    return_residuals: bool,
    generation: int,
) -> Config | None:
  """Returns the mid-size shape config, or None if the shape isn't one."""
  if (
      m_dim >= _WIDE_N_MAX_ROWS
      or k_padded > _WIDE_N_MAX_K
      or n_padded > _WIDE_N_MAX_N
  ):
    return None
  if generation == 5:
    tile_ms, vmem_limit = (256, 128), _get_vmem_budget_bytes()
  else:
    tile_ms = (1024, 768, 512, 384, 256, 128)
    vmem_limit = _DEFAULT_SCOPED_VMEM_BYTES
  tile_ns = [t for t in (1536, 768, 512, 256, 128) if n_padded % t == 0]
  for tile_n in tile_ns:
    for tile_m in tile_ms:
      if m_dim % tile_m:
        continue
      vmem = _estimate_vmem_bytes(
          tile_m,
          k_padded,
          tile_n,
          2,
          itemsize,
          single_k_step=True,
          return_residuals=return_residuals,
      )
      if vmem <= vmem_limit:
        return Config(
            tile_m=tile_m, tile_k=k_padded, tile_n=tile_n, input_buffer_count=2
        )
  return None


def _select_heuristic_config(
    m_dim: int,
    k_dim: int,
    n_dim: int,
    itemsize: int,
    *,
    return_residuals: bool,
    num_cores: int,
    generation: int,
) -> Config:
  """Selects high-performance tile sizes for TPU v5+."""
  if m_dim == 0 or k_dim == 0 or n_dim == 0:
    return Config()
  vmem_budget = _get_vmem_budget_bytes()
  k_padded = max(128, _align_up(k_dim, 128))
  n_padded = max(128, _align_up(n_dim, 128))
  m_padded = max(128, _align_up(m_dim, 128))

  wide_n = _wide_n_config(
      m_dim,
      k_padded,
      n_padded,
      itemsize,
      return_residuals=return_residuals,
      generation=generation,
  )
  if wide_n is not None:
    return wide_n

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
      if not _single_k_tiles_ok(
          m_dim,
          n_padded,
          tile_m,
          tile_n,
          num_cores=num_cores,
          generation=generation,
      ):
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


def _collapse_weights(
    weights: base.FusedWeights | base.UnfusedWeights,
) -> jax.Array:
  """Returns the gate and up weights side by side, as a (K, 2N) array."""
  if isinstance(weights, tuple):
    return jnp.concatenate(weights, axis=-1)
  return jax.lax.collapse(weights, -2)


def _expand_weight_grads(
    dw_2d: jax.Array, weights: base.FusedWeights | base.UnfusedWeights
) -> base.FusedWeights | base.UnfusedWeights:
  """Inverts `_collapse_weights` for the weight gradients."""
  if isinstance(weights, tuple):
    d_gate, d_up = jnp.split(dw_2d, 2, axis=-1)
    return d_gate.astype(weights[0].dtype), d_up.astype(weights[1].dtype)
  return dw_2d.reshape(weights.shape).astype(weights.dtype)


# The fused backward only beats XLA once there are enough rows to amortize the
# full-weight VMEM accumulation of dW: in total, and per weight element.
_MIN_TRAINING_ROWS: Final[int] = 2048
_MIN_TRAINING_ROWS_F32: Final[int] = 32768
_MIN_TRAINING_ROWS_PER_WEIGHT: Final[float] = 1 / 16
# Below this many rows, XLA's f32 forward is faster than the kernel's.
_MIN_FORWARD_ROWS_F32: Final[int] = 1024
# Below this many rows, f32 trains faster with XLA than with the lane-dense VJP.
_MIN_LANE_DENSE_ROWS_F32: Final[int] = 16384
# On TPU v5, bf16 shapes the fused kernel doesn't take train faster with the
# lane-dense VJP up to this many rows, or with K at least this wide.
_MAX_LANE_DENSE_ROWS_BF16: Final[int] = 2560
_MIN_LANE_DENSE_K_BF16: Final[int] = 2048
# XLA fails to compile the lane-dense VJP's elementwise fusions on v7x for
# ~10M rows, so larger shapes keep the base VJP.
_MAX_LANE_DENSE_ROWS: Final[int] = 1 << 22

type TrainingStrategy = Literal["fused", "lane_dense", "xla"]


def _xla_forward_is_faster(
    x_shape: tuple[int, ...], dtype: jax.typing.DTypeLike
) -> bool:
  rows = math.prod(x_shape[:-1])
  return jnp.dtype(dtype).itemsize > 2 and rows < _MIN_FORWARD_ROWS_F32


def _fused_bwd_tile_m(
    x_shape: tuple[int, ...], n: int, dtype: jax.typing.DTypeLike
) -> int | None:
  """Returns the backward kernel's M tile, or None to train with XLA."""
  if jax.default_backend() != "tpu":
    return None
  m, k = math.prod(x_shape[:-1]), x_shape[-1]
  itemsize = 4 if jnp.dtype(dtype).itemsize > 2 else 2
  min_rows = _MIN_TRAINING_ROWS if itemsize == 2 else _MIN_TRAINING_ROWS_F32
  if m < max(min_rows, _MIN_TRAINING_ROWS_PER_WEIGHT * k * 2 * n):
    return None
  return pallas_mosaic_tpu_vjp.select_tile_m(
      m, k, n, itemsize, _get_vmem_budget_bytes()
  )


def _training_strategy(x: jax.ShapeDtypeStruct, n: int) -> TrainingStrategy:
  """Returns how to train a GLU on `x`, possibly a `BatchedShapeDtype`.

  The base VJP is a custom VJP, which under `vmap` is slower than XLA's own
  autodiff, so vmapped shapes the fused kernel doesn't take use the lane-dense
  VJP instead.

  Args:
    x: The input (per `vmap` instance, with its `vmap` axes if batched).
    n: The output width.
  """
  if _fused_bwd_tile_m(x.shape, n, x.dtype) is not None:
    return "fused"
  if jax.default_backend() != "tpu":
    return "xla"
  rows, k = math.prod(x.shape[:-1]), x.shape[-1]
  vmap_size = 1
  if isinstance(x, batching.BatchedShapeDtype):
    vmap_size = math.prod(s for s in x.vmap_axis_sizes if s is not None)
  if rows * vmap_size > _MAX_LANE_DENSE_ROWS:
    return "xla"
  if vmap_size > 1:
    return "lane_dense"
  if jnp.dtype(x.dtype).itemsize > 2:
    return "lane_dense" if rows >= _MIN_LANE_DENSE_ROWS_F32 else "xla"
  # On TPU v6+, XLA's own bf16 VJP is as fast for these shapes.
  if pltpu.get_tpu_info().generation == 5 and (
      rows <= _MAX_LANE_DENSE_ROWS_BF16 or k >= _MIN_LANE_DENSE_K_BF16
  ):
    return "lane_dense"
  return "xla"


def _xla_residuals(x: jax.ShapeDtypeStruct, n: int) -> bool:
  """Returns whether XLA's forward should compute the training residuals."""
  strategy = _training_strategy(x, n)
  if strategy == "lane_dense":
    # The lane-dense VJP reads XLA's `(..., 2, N)` residuals just as well.
    return _xla_forward_is_faster(x.shape, x.dtype)
  return strategy == "xla"


def _lane_dense_bwd(
    residuals: base.Residuals,
    dout: jax.Array,
    x: jax.Array,
    weights: base.FusedWeights | base.UnfusedWeights,
    *,
    activation: Callable[[jax.Array], jax.Array] | None,
    precision: base.CanonicalPrecision,
) -> base.Grads:
  """Returns the gradients via XLA, with gate and up side by side in (M, 2N)."""
  n = dout.shape[-1]
  gate_up = jax.lax.collapse(residuals, 0, -2).reshape(-1, 2 * n)
  gate, up = jnp.split(gate_up.astype(jnp.float32), 2, axis=-1)
  dy = jax.lax.collapse(dout, 0, -1).astype(jnp.float32)
  if activation is None:
    act, d_gate = gate, dy * up
  else:
    act, act_vjp = jax.vjp(activation, gate)
    (d_gate,) = act_vjp(dy * up)
  dot_precision, dot_dtype = _resolve_tpu_dot_precision(precision, x.dtype)
  if dot_dtype == jnp.dtype(jnp.float16):
    dot_dtype = jnp.dtype(jnp.float32)
  dot = functools.partial(
      jnp.matmul, precision=dot_precision, preferred_element_type=jnp.float32
  )
  d_gate_up = jnp.concatenate([d_gate, dy * act], axis=-1).astype(dot_dtype)
  w = _collapse_weights(weights).astype(dot_dtype)
  dx = dot(d_gate_up, w.T)
  dw = dot(jax.lax.collapse(x, 0, -1).astype(dot_dtype).T, d_gate_up)
  return dx.reshape(x.shape).astype(x.dtype), _expand_weight_grads(dw, weights)


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class VjpConfig:
  """Configuration of the Pallas Mosaic TPU GatedLinearUnit VJP.

  Attributes:
    strategy: `fused` for the Pallas backward kernel, `lane_dense` for XLA
      matmuls on the gate and up gradients side by side, or `xla` for the base
      VJP.
    tile_m: The fused kernel's M tile.
  """

  strategy: TrainingStrategy = "xla"
  tile_m: pydantic.PositiveInt | None = None


class PallasMosaicTpuGatedLinearUnitVjp(
    base.GatedLinearUnitVjp[VjpConfig, None]
):
  """VJP for the Pallas Mosaic TPU GLU.

  The base VJP differentiates through `(..., 2, N)` intermediates, whose size-2
  second-minor dimension is heavily padded on TPU. The Pallas kernel keeps the
  gate and up columns side by side, computes d(gate, up) in VMEM, and does both
  matmuls from it. Shapes it doesn't help use the same layout with XLA matmuls,
  or the base VJP (see `_training_strategy`).
  """

  config_cls: ClassVar[type[VjpConfig]] = VjpConfig
  supports_symbolic_shapes: ClassVar[bool] = False

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> VjpConfig:
    x, dout = ba.arguments["x"], ba.arguments["dout"]
    n = dout.shape[-1]
    strategy = _training_strategy(x, n)
    if strategy != "fused":
      return VjpConfig(strategy=strategy)
    return VjpConfig(
        strategy=strategy, tile_m=_fused_bwd_tile_m(x.shape, n, x.dtype)
    )

  @override
  def _fwd(
      self,
      residuals: base.Residuals,
      out: Float[Array, "*B M N"],
      dout: Float[Array, "*B M N"],
      x: Float[Array, "*B M K"],
      weights: base.FusedWeights | base.UnfusedWeights,
      *,
      activation: Callable[[jax.Array], jax.Array] | None,
      precision: base.CanonicalPrecision,
      return_residuals: bool,
      config: VjpConfig,
  ) -> tuple[base.Grads, None]:
    n = dout.shape[-1]
    tile_m = config.tile_m or _fused_bwd_tile_m(x.shape, n, x.dtype)
    if config.strategy == "lane_dense" and not return_residuals:
      grads = _lane_dense_bwd(
          residuals,
          dout,
          x,
          weights,
          activation=activation,
          precision=precision,
      )
      return grads, None
    if config.strategy != "fused" or tile_m is None or return_residuals:
      return super()._fwd(
          residuals,
          out,
          dout,
          x,
          weights,
          activation=activation,
          precision=precision,
          return_residuals=return_residuals,
          config=config,
      )
    compute_dtype = jnp.promote_types(x.dtype, jnp.bfloat16)
    dot_precision, dot_dtype = _resolve_tpu_dot_precision(precision, x.dtype)
    if dot_dtype == jnp.dtype(jnp.float16):
      dot_dtype = jnp.dtype(jnp.float32)
    info = pltpu.get_tpu_info()
    dx, dw_2d = pallas_mosaic_tpu_vjp.gated_linear_unit_bwd(
        jax.lax.collapse(x, 0, -1).astype(compute_dtype),
        jax.lax.collapse(residuals, 0, -2)
        .reshape(-1, 2 * n)
        .astype(compute_dtype),
        jax.lax.collapse(dout, 0, -1).astype(compute_dtype),
        _collapse_weights(weights).astype(compute_dtype),
        activation=activation,
        dot_precision=dot_precision,
        dot_dtype=dot_dtype,
        tile_m=tile_m,
        num_cores=info.num_cores,
        vmem_limit_bytes=info.vmem_capacity_bytes,
    )
    dx = dx.reshape(x.shape).astype(x.dtype)
    return (dx, _expand_weight_grads(dw_2d, weights)), None


@dataclasses.dataclass(frozen=True, slots=True)
class PallasMosaicTpuGatedLinearUnit(base.GatedLinearUnit[Config, Key]):
  """Pallas-Mosaic-TPU gated linear unit implementation."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(self, "vjp", PallasMosaicTpuGatedLinearUnitVjp())

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
      if weights[0].shape != weights[1].shape:
        raise ValueError(
            "Tuple weights must have identical shapes, got"
            f" {weights[0].shape} and {weights[1].shape}."
        )
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

    n = (
        weights[0].shape[-1]
        if isinstance(weights, tuple)
        else weights.shape[-1]
    )
    if x.size == 0 or n == 0:
      raise NotImplementedError("Mosaic TPU GLU does not support empty dims.")

    dot_precision, dot_dtype = _resolve_tpu_dot_precision(precision, x_dtype)
    if (
        config.xla_residuals
        if return_residuals
        else _xla_forward_is_faster(x.shape, x_dtype)
    ):
      return base.GatedLinearUnit._fwd(  # pylint: disable=protected-access
          self,
          x,
          weights,
          activation=activation,
          precision=precision,
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
    info = pltpu.get_tpu_info()
    config = _select_heuristic_config(
        m_dim,
        k_dim,
        n_dim,
        itemsize,
        return_residuals=return_residuals,
        num_cores=info.num_cores,
        generation=info.generation,
    )
    if return_residuals and _xla_residuals(input_tensor, n_dim):
      return dataclasses.replace(config, xla_residuals=True)
    return config

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
          for input_buffer_count in (2, 3):
            vmem = _estimate_vmem_bytes(
                tile_m,
                tile_k,
                tile_n,
                input_buffer_count,
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
                      input_buffer_count=input_buffer_count,
                  )
              )
    return configs
