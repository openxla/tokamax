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
"""Native Triton normalization op via jax-triton."""

import dataclasses
from typing import ClassVar, TypeAlias

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl
from tokamax._src import gpu_utils
from tokamax._src.ops import op
from tokamax._src.ops.normalization import base
from tokamax._src.ops.normalization import pallas_triton_config
from tokamax._src.ops.normalization import pallas_triton_vjp_config
from typing_extensions import override


Config: TypeAlias = pallas_triton_config.Config
Key: TypeAlias = pallas_triton_config.Key
VjpConfig: TypeAlias = pallas_triton_vjp_config.Config
VjpKey: TypeAlias = pallas_triton_vjp_config.Key
FusedInputArray = base.FusedInputArray
Residuals = base.Residuals
_NUM_REGISTERS_PER_SM = gpu_utils.NUM_REGISTERS_PER_SM


# ── Forward kernel ─────────────────────────────────────────────────────────────


@triton.jit
def _normalization_kernel(
    # --- pointer args: inputs then outputs ---
    X, SCALE, OFFSET,
    Y, MEAN, RSTD,
    # --- scalar args ---
    M_dim, A, N_dim, stride_m, stride_a, epsilon, scale_offset,
    # --- constexpr args ---
    BLOCK_M: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_OFFSET: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
    RETURN_MEAN: tl.constexpr,
    RETURN_RSTD: tl.constexpr,
    SINGLE_N: tl.constexpr,
):
  """Normalization forward kernel (layer norm / RMS norm).

  Operates on a canonicalized (M, A, N) layout where A is the (middle) norm
  axis and N is contiguous. 2D grid (pid_m, pid_n) tiles over M and N; each
  program reduces a full (BLOCK_M, A, BLOCK_N) tile along A.

  When the N grid is degenerate (`SINGLE_N`, i.e. `cdiv(N, BLOCK_N) == 1`) the N
  program id is folded to a compile-time 0. Otherwise `n_off = program_id(1) *
  BLOCK_N` is a runtime value on the contiguous (stride-1) N axis of the block
  pointer, which defeats Triton's alignment/divisibility proof and blocks
  vectorization of the reduction axis (scalar, cross-warp layout). Folding it —
  as Pallas/XLA do for size-1 grid dims — restores the vectorized intra-warp
  layout.
  """
  pid_m = tl.program_id(0)
  pid_n = 0 if SINGLE_N else tl.program_id(1)
  m_off = pid_m * BLOCK_M
  n_off = pid_n * BLOCK_N

  # (M, A, N) tile: N contiguous (stride 1), A strided by N.
  x_blk = tl.make_block_ptr(
      X, shape=(M_dim, A, N_dim), strides=(stride_m, stride_a, 1),
      offsets=(m_off, 0, n_off),
      block_shape=(BLOCK_M, BLOCK_A, BLOCK_N), order=(2, 1, 0),
  )
  x = tl.load(x_blk, boundary_check=(0, 1, 2), padding_option='zero').to(tl.float32)

  a_offs = tl.arange(0, BLOCK_A)
  a_mask = a_offs < A

  if SUBTRACT_MEAN:
    mean = tl.sum(x, axis=1) / A
    if RETURN_MEAN:
      mean_blk = tl.make_block_ptr(
          MEAN, shape=(M_dim, N_dim), strides=(N_dim, 1),
          offsets=(m_off, n_off),
          block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
      )
      tl.store(mean_blk, mean, boundary_check=(0, 1))
    x = x - tl.expand_dims(mean, 1)
    # Zero invalid values (when A is not a power of two).
    x = tl.where(a_mask[None, :, None], x, 0.0)

  var = tl.sum(x * x, axis=1) / A
  rstd = 1.0 / tl.sqrt(var + epsilon)
  if RETURN_RSTD:
    rstd_blk = tl.make_block_ptr(
        RSTD, shape=(M_dim, N_dim), strides=(N_dim, 1),
        offsets=(m_off, n_off),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    tl.store(rstd_blk, rstd, boundary_check=(0, 1))
  x = x * tl.expand_dims(rstd, 1)

  if HAS_SCALE:
    s = tl.load(SCALE + a_offs, mask=a_mask, other=0.0).to(tl.float32)
    x = x * (s[None, :, None] + scale_offset)
  if HAS_OFFSET:
    o = tl.load(OFFSET + a_offs, mask=a_mask, other=0.0).to(tl.float32)
    x = x + o[None, :, None]

  y_blk = tl.make_block_ptr(
      Y, shape=(M_dim, A, N_dim), strides=(stride_m, stride_a, 1),
      offsets=(m_off, 0, n_off),
      block_shape=(BLOCK_M, BLOCK_A, BLOCK_N), order=(2, 1, 0),
  )
  tl.store(y_blk, x.to(Y.dtype.element_ty), boundary_check=(0, 1, 2))


# ── VJP kernel ──────────────────────────────────────────────────────────────────


@triton.jit
def _normalization_vjp_kernel(
    # --- pointer args: inputs then outputs ---
    DOUT, X, SCALE, MEAN, RSTD,
    DX, DSCALE, DOFFSET,
    # --- scalar args ---
    M_dim, A, N_dim, stride_m, stride_a, scale_offset, grid_n,
    # --- constexpr args ---
    BLOCK_M: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_OFFSET: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
    SINGLE_N: tl.constexpr,
):
  """Normalization VJP kernel.

  2D grid (pid_m, pid_n). Produces per-program partial sums for dscale and
  doffset (shape (grid_m * grid_n, A)), reduced outside the kernel. Operates on
  the canonical (M, A, N) layout, reducing along the middle axis A. `SINGLE_N`
  folds a degenerate N grid id to 0 to keep loads vectorizable (see the forward
  kernel docstring).
  """
  pid_m = tl.program_id(0)
  pid_n = 0 if SINGLE_N else tl.program_id(1)
  m_off = pid_m * BLOCK_M
  n_off = pid_n * BLOCK_N

  a_offs = tl.arange(0, BLOCK_A)
  a_mask = a_offs < A

  # Compute x_norm = (x - mean) * rstd. (M, A, N) tile, N contiguous.
  x_norm = tl.load(tl.make_block_ptr(
      X, shape=(M_dim, A, N_dim), strides=(stride_m, stride_a, 1),
      offsets=(m_off, 0, n_off),
      block_shape=(BLOCK_M, BLOCK_A, BLOCK_N), order=(2, 1, 0),
  ), boundary_check=(0, 1, 2), padding_option='zero').to(tl.float32)
  if SUBTRACT_MEAN:
    mean = tl.load(tl.make_block_ptr(
        MEAN, shape=(M_dim, N_dim), strides=(N_dim, 1),
        offsets=(m_off, n_off),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    ), boundary_check=(0, 1), padding_option='zero').to(tl.float32)
    x_norm = x_norm - tl.expand_dims(mean, 1)
  rstd = tl.load(tl.make_block_ptr(
      RSTD, shape=(M_dim, N_dim), strides=(N_dim, 1),
      offsets=(m_off, n_off),
      block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
  ), boundary_check=(0, 1), padding_option='zero').to(tl.float32)
  x_norm = x_norm * tl.expand_dims(rstd, 1)

  # Load dout tile.
  dout = tl.load(tl.make_block_ptr(
      DOUT, shape=(M_dim, A, N_dim), strides=(stride_m, stride_a, 1),
      offsets=(m_off, 0, n_off),
      block_shape=(BLOCK_M, BLOCK_A, BLOCK_N), order=(2, 1, 0),
  ), boundary_check=(0, 1, 2), padding_option='zero').to(tl.float32)

  # Partial doffset/dscale: reduce (BM, BA, BN) over M and N → (BA,).
  # Store as (linear_pid, A); host sums axis 0 to get (A,).
  linear_pid = pid_m * grid_n + pid_n

  if HAS_OFFSET:
    doffset = tl.sum(tl.sum(dout, axis=2), axis=0)
    tl.store(DOFFSET + linear_pid * A + a_offs, doffset, mask=a_mask)

  if HAS_SCALE:
    dscale = tl.sum(tl.sum(dout * x_norm, axis=2), axis=0)
    tl.store(DSCALE + linear_pid * A + a_offs, dscale, mask=a_mask)
    s = tl.load(SCALE + a_offs, mask=a_mask, other=0.0).to(tl.float32)
    dout = dout * (s[None, :, None] + scale_offset)

  # dx = (dout - mean_a(dout·x_norm)·x_norm [- mean_a(dout)]) * rstd
  # Reductions over axis 1 (the A dimension).
  dx1 = tl.expand_dims(-(tl.sum(dout * x_norm, axis=1) / A), 1)
  dx = dout + dx1 * x_norm
  if SUBTRACT_MEAN:
    dx2 = tl.expand_dims(-(tl.sum(dout, axis=1) / A), 1)
    dx = dx + dx2
  dx = dx * tl.expand_dims(rstd, 1)

  tl.store(tl.make_block_ptr(
      DX, shape=(M_dim, A, N_dim), strides=(stride_m, stride_a, 1),
      offsets=(m_off, 0, n_off),
      block_shape=(BLOCK_M, BLOCK_A, BLOCK_N), order=(2, 1, 0),
  ), dx.to(DX.dtype.element_ty), boundary_check=(0, 1, 2))


# ── Forward op ──────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class JaxTritonNormalization(base.Normalization[Config, Key]):
  """Native Triton normalization op via jax-triton."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  # If `None`, `input_output_alias = not return_residuals`.
  input_output_alias: bool | None = None

  def __post_init__(self):
    if self.vjp is None:
      vjp = JaxTritonNormalizationVjp()
      object.__setattr__(self, 'vjp', vjp)

  @override
  def _fwd(
      self,
      x: jax.Array | FusedInputArray,
      scale: jax.Array | None,
      offset: jax.Array | None,
      *,
      axis: int,
      epsilon: float,
      scale_offset: float,
      subtract_mean: bool,
      return_residuals: bool,
      config: Config,
  ) -> tuple[jax.Array, base.Residuals | None]:
    input_output_alias = self.input_output_alias
    if input_output_alias is None:
      input_output_alias = not return_residuals and not callable(x)
    elif input_output_alias and callable(x):
      raise NotImplementedError(
          '`input_output_alias` not supported when `x` is a callable.'
      )

    # Materialize fused inputs — no Pallas fusion support in native Triton.
    if callable(x):
      x = x()

    orig_x_shape = x.shape
    x_shape = pallas_triton_config.canonicalize_shape_3d(orig_x_shape, axis)
    x = x.reshape(x_shape)
    M, A, N = x_shape
    num_rows = M * N
    block_a = pl.next_power_of_2(A)
    block_m = config.block_m
    block_n = 1 if config.block_n is None else config.block_n

    # Dummy arrays for optional inputs (never dereferenced by the kernel).
    dummy = jnp.zeros(1, dtype=x.dtype)
    scale_arr = scale if scale is not None else dummy
    offset_arr = offset if offset is not None else dummy

    alias_kwargs = {}
    if input_output_alias:
      alias_kwargs['input_output_aliases'] = {0: 0}

    grid = (pl.cdiv(M, block_m), pl.cdiv(N, block_n))
    single_n = pl.cdiv(N, block_n) == 1
    out_shapes = [
        jax.ShapeDtypeStruct(x_shape, x.dtype),
        jax.ShapeDtypeStruct((num_rows,), jnp.float32),
        jax.ShapeDtypeStruct((num_rows,), jnp.float32),
    ]
    y, mean_flat, rstd_flat = jt.triton_call(
        x, scale_arr, offset_arr,
        kernel=_normalization_kernel,
        out_shape=out_shapes,
        grid=grid,
        num_warps=config.num_warps,
        **alias_kwargs,
        M_dim=M,
        A=A,
        N_dim=N,
        stride_m=A * N,
        stride_a=N,
        epsilon=epsilon,
        scale_offset=scale_offset,
        BLOCK_M=block_m,
        BLOCK_A=block_a,
        BLOCK_N=block_n,
        HAS_SCALE=scale is not None,
        HAS_OFFSET=offset is not None,
        SUBTRACT_MEAN=subtract_mean,
        RETURN_MEAN=return_residuals and subtract_mean,
        RETURN_RSTD=return_residuals,
        SINGLE_N=single_n,
    )

    y = y.reshape(orig_x_shape)

    if return_residuals:
      stat_shape = list(orig_x_shape)
      stat_shape[axis] = 1
      rstd = rstd_flat.reshape(stat_shape)
      mean = mean_flat.reshape(stat_shape) if subtract_mean else None
      return y, (mean, rstd)

    return y, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return pallas_triton_config.get_heuristics_config(
        *ba.args, vmap_axis_sizes=ba.vmap_axis_sizes, **ba.kwargs
    )

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> Key:
    return pallas_triton_config.get_key(*ba.args, **ba.kwargs)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    x = ba.args[0]
    axis = ba.kwargs['axis']
    x_shape = pallas_triton_config.canonicalize_shape(x.shape, axis)
    configs = set()
    for num_warps in [1, 2, 4, 8, 16]:
      for block_m in [1, 2, 4, 8, 16, 32, 64]:
        block_m = min(block_m, pl.next_power_of_2(x_shape[0]))
        if len(x_shape) > 2:
          for block_n in [16, 32, 64, 128]:
            block_n = min(block_n, pl.next_power_of_2(x_shape[2]))
            if (block_m * x_shape[1] * block_n <= _NUM_REGISTERS_PER_SM) or (
                block_m == 1 and block_n <= 16
            ):
              configs.add(
                  Config(block_m=block_m, block_n=block_n, num_warps=num_warps)
              )
        elif block_m * x_shape[1] <= _NUM_REGISTERS_PER_SM or block_m == 1:
          configs.add(Config(block_m=block_m, block_n=None, num_warps=num_warps))
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)


# ── VJP op ──────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, slots=True)
class JaxTritonNormalizationVjp(base.NormalizationVjp[VjpConfig, VjpKey]):
  """Native Triton normalization VJP via jax-triton."""

  config_cls: ClassVar[type[VjpConfig]] = VjpConfig

  @override
  def _fwd(
      self,
      residuals: Residuals,
      out: jax.Array,
      dout: jax.Array,
      x: jax.Array,
      scale: jax.Array | None,
      offset: jax.Array | None,
      *,
      axis: int,
      epsilon: float,
      scale_offset: float,
      subtract_mean: bool,
      return_residuals: bool,
      config: VjpConfig,
  ) -> tuple[tuple[jax.Array, jax.Array | None, jax.Array | None], None]:
    """Computes normalization VJP `(dx, dscale, doffset)`."""
    del out, epsilon  # Unused.

    if return_residuals:
      raise NotImplementedError('`return_residuals` not supported.')

    mean, rstddev = residuals
    if (mean is not None) != subtract_mean:
      raise ValueError('`mean` residual inconsistent with `subtract_mean`.')

    orig_x_shape = x.shape
    x_shape = pallas_triton_config.canonicalize_shape_3d(x.shape, axis)
    x = x.reshape(x_shape)
    dout = dout.reshape(x_shape)
    M, A, N = x_shape
    block_a = pl.next_power_of_2(A)
    block_m = config.block_m
    block_n = 1 if config.block_n is None else config.block_n

    # Flatten stats — same layout the forward kernel wrote.
    mean_flat = (
        mean.reshape(-1) if mean is not None
        else jnp.zeros(1, dtype=jnp.float32)
    )
    rstd_flat = rstddev.reshape(-1)

    dummy = jnp.zeros(1, dtype=x.dtype)
    scale_arr = scale if scale is not None else dummy

    grid_m = pl.cdiv(M, block_m)
    grid_n = pl.cdiv(N, block_n)
    grid = (grid_m, grid_n)
    num_programs = grid_m * grid_n
    dparam_shape = jax.ShapeDtypeStruct((num_programs, A), jnp.float32)
    dummy_shape = jax.ShapeDtypeStruct((1,), jnp.float32)
    out_shapes = [
        jax.ShapeDtypeStruct(x_shape, x.dtype),
        dparam_shape if scale is not None else dummy_shape,
        dparam_shape if offset is not None else dummy_shape,
    ]
    dx, dscale_partial, doffset_partial = jt.triton_call(
        dout, x, scale_arr, mean_flat, rstd_flat,
        kernel=_normalization_vjp_kernel,
        out_shape=out_shapes,
        grid=grid,
        num_warps=config.num_warps,
        input_output_aliases={1: 0},
        M_dim=M,
        A=A,
        N_dim=N,
        stride_m=A * N,
        stride_a=N,
        scale_offset=scale_offset,
        grid_n=grid_n,
        BLOCK_M=block_m,
        BLOCK_A=block_a,
        BLOCK_N=block_n,
        HAS_SCALE=scale is not None,
        HAS_OFFSET=offset is not None,
        SUBTRACT_MEAN=subtract_mean,
        SINGLE_N=grid_n == 1,
    )

    dscale = None
    if scale is not None:
      dscale = jnp.sum(dscale_partial, axis=0).astype(scale.dtype)

    doffset = None
    if offset is not None:
      doffset = jnp.sum(doffset_partial, axis=0).astype(offset.dtype)

    return (dx.reshape(orig_x_shape), dscale, doffset), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> VjpConfig:
    return pallas_triton_vjp_config.get_heuristics_config(
        *ba.args, vmap_axis_sizes=ba.vmap_axis_sizes, **ba.kwargs
    )

  @override
  def _get_autotuning_cache_key(self, ba: op.BoundArguments) -> VjpKey:
    return pallas_triton_vjp_config.get_key(*ba.args, **ba.kwargs)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[VjpConfig]:
    axis = ba.kwargs['axis']
    dout_shape = pallas_triton_config.canonicalize_shape(ba.args[2].shape, axis)
    configs = set()
    for num_warps in [1, 2, 4, 8, 16]:
      for block_m in [1, 2, 4, 8, 16, 32, 64, 128]:
        if block_m > pl.next_power_of_2(dout_shape[0]):
          break
        config = VjpConfig(block_m=block_m, block_n=None, num_warps=num_warps)
        if len(dout_shape) > 2:
          for block_n in [16, 32, 64, 128]:
            if block_n > max(pl.next_power_of_2(dout_shape[2]), 16):
              break
            if 2 * block_m * dout_shape[1] * block_n <= _NUM_REGISTERS_PER_SM:
              configs.add(dataclasses.replace(config, block_n=block_n))
        elif 2 * block_m * dout_shape[1] <= _NUM_REGISTERS_PER_SM:
          configs.add(config)
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)
