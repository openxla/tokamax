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
"""Common definitions and helpers for Pallas-Mosaic-GPU TriangleMult."""

from typing import Annotated, Any, Final, Literal

import immutabledict
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
import pydantic
from tokamax._src import gpu_utils
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src import precision as precision_lib
from tokamax._src import pydantic as pydantic_lib
from tokamax._src.ops.triangle_multiplication import base

Residuals = base.Residuals
type Key = immutabledict.immutabledict[str, Any]

SUPPORTED_DTYPES: Final[frozenset[jnp.dtype]] = frozenset(
    {jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)}
)
SUPPORTED_CHANNEL_DIMS: Final[frozenset[int]] = frozenset({16, 32, 64, 128})
MIN_KERNEL_CHANNEL_DIM: Final[int] = 64
MAX_INT32: Final[int] = 2**31 - 1

_BlockM = Annotated[pydantic_lib.PowerOfTwo, pydantic.Field(ge=32, le=256)]
_NumWarps = Annotated[pydantic_lib.PowerOfTwo, pydantic.Field(le=16)]


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Configuration for Pallas-Mosaic-GPU Triangle Multiplication kernels.

  Attributes:
    block_m: Tile size along the flattened token-pair dimension M = N * N. Must
      be a power of two in [32, 256].
    num_warps: Number of warps per CTA. Must be a power of two, at most 16.
    num_stages: Number of software pipeline stages.
    epilogue_block_m: Optional tile size override for the epilogue kernel.
    epilogue_num_warps: Optional num_warps override for the epilogue kernel.
    epilogue_num_stages: Optional num_stages override for the epilogue kernel.
    prologue_persistent: Whether to use persistent prologue kernels.
    epilogue_persistent: Whether to use persistent epilogue kernels.
    ctas_per_sm: Number of persistent CTAs per SM when persistent is enabled.
  """

  block_m: _BlockM = 128
  num_warps: _NumWarps = 4
  num_stages: pydantic.PositiveInt = 3
  epilogue_block_m: _BlockM | None = None
  epilogue_num_warps: _NumWarps | None = None
  epilogue_num_stages: pydantic.PositiveInt | None = None
  prologue_persistent: bool | None = None
  epilogue_persistent: bool | None = None
  ctas_per_sm: pydantic.PositiveInt = 2


def is_interpret_mode() -> bool:
  """Returns whether Pallas GPU interpret mode is currently enabled."""
  return (
      jax._src.config.pallas_tpu_interpret_mode_context_manager.value  # pylint: disable=protected-access
      is not None
  )


def infer_device(x: Any) -> jax.Device | None:
  """Infers the target device from a concrete or traced JAX array."""
  sharding = getattr(x, "sharding", None)
  if sharding is not None:
    device_set = getattr(sharding, "device_set", None)
    if device_set:
      return next(iter(device_set))
  if isinstance(x, jax.Array):
    try:
      return next(iter(x.devices()), None)
    except jax.errors.ConcretizationTypeError:
      pass
  return None


def validate_fwd_inputs(
    x: jax.Array,
    projection_in_weights: jax.Array,
    gate_in_weights: jax.Array,
    projection_out_weights: jax.Array,
    gate_out_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    layernorm_out_scale: jax.Array,
    layernorm_out_offset: jax.Array,
    precision: base.CanonicalPrecision,
) -> None:
  """Validates inputs for Pallas Mosaic GPU forward pass."""
  device = infer_device(x)
  if not gpu_utils.has_mosaic_gpu_support(device):
    raise NotImplementedError("Mosaic GPU is not supported on this platform.")
  if not (gpu_utils.is_sm90(device) or gpu_utils.is_sm100(device)):
    raise NotImplementedError("Only supported for sm90 and sm100 GPUs.")

  if not precision_lib.is_default(
      x.dtype, projection_in_weights.dtype, precision
  ):
    raise NotImplementedError(f"{precision=} is not supported.")

  if jnp.dtype(x.dtype) not in SUPPORTED_DTYPES:
    raise NotImplementedError(
        "Mosaic GPU triangle_multiplication only supports"
        f" {SUPPORTED_DTYPES}, got {x.dtype}."
    )

  for w in (
      projection_in_weights,
      gate_in_weights,
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
  ):
    if w.dtype != x.dtype:
      raise NotImplementedError(
          f"All weight and norm arrays must match x.dtype={x.dtype}, got"
          f" {w.dtype}."
      )

  n, _, c = x.shape
  h, d = projection_out_weights.shape
  if not {c, h, d} <= SUPPORTED_CHANNEL_DIMS:
    raise NotImplementedError(
        "Mosaic GPU TriangleMultiplication requires C, H, D in"
        f" {sorted(SUPPORTED_CHANNEL_DIMS)}, got C={c}, H={h}, D={d}."
    )
  if n == 0:
    raise NotImplementedError("Empty inputs are not supported.")
  m_pad = pl.cdiv(n * n, 512) * 512
  max_chd_pad = max(MIN_KERNEL_CHANNEL_DIM, c, h, d)
  if m_pad * max_chd_pad - 1 > MAX_INT32:
    raise NotImplementedError(
        f"N={n} (padded M={m_pad}) with max(C, H, D)={max_chd_pad} would"
        " overflow the int32 offsets used by the Mosaic GPU kernels."
    )


def validate_vjp_inputs(
    residuals: base.Residuals,
    dout: jax.Array,
    x: jax.Array,
    projection_in_weights: jax.Array,
    gate_in_weights: jax.Array,
    projection_out_weights: jax.Array,
    gate_out_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    layernorm_out_scale: jax.Array,
    layernorm_out_offset: jax.Array,
    precision: base.CanonicalPrecision,
    return_residuals: bool,
) -> None:
  """Validates inputs for Pallas Mosaic GPU backward pass."""
  if return_residuals:
    raise NotImplementedError("`return_residuals` not supported.")

  validate_fwd_inputs(
      x,
      projection_in_weights,
      gate_in_weights,
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
      precision,
  )

  n, _, _ = x.shape
  h, d = projection_out_weights.shape
  if dout.shape != (n, n, d):
    raise NotImplementedError(
        f"Expected dout shape {(n, n, d)}, got {dout.shape}."
    )
  if len(residuals) != 3:
    raise NotImplementedError(
        f"Expected 3 residual arrays, got {len(residuals)}."
    )
  for r in residuals:
    if r.shape != (h, n, n):
      raise NotImplementedError(
          f"Expected residual shape {(h, n, n)}, got {r.shape}."
      )

  for w in (*residuals, dout):
    if w.dtype != x.dtype:
      raise NotImplementedError(
          "All residual, weight, norm, and dout arrays must match"
          f" x.dtype={x.dtype}, got {w.dtype}."
      )


def _pad_spatial_16(x: jax.Array, n_pad: int) -> jax.Array:
  pad_n = n_pad - x.shape[1]
  return jnp.pad(x, ((0, 0), (0, pad_n), (0, pad_n)))


def contract(
    left: jax.Array,
    right: jax.Array,
    triangle_type: Literal["incoming", "outgoing"],
    precision: base.CanonicalPrecision,
) -> jax.Array:
  """Batched (over H) triangle contraction of two (H, N, N) planes."""
  n = left.shape[1]
  if n % 16 != 0:
    n_pad = pl.cdiv(n, 16) * 16
    left = _pad_spatial_16(left, n_pad)
    right = _pad_spatial_16(right, n_pad)
  if triangle_type == "outgoing":
    dims = (((2,), (2,)), ((0,), (0,)))
    out = lax.dot_general(left, right, dims, precision=precision)
  else:
    dims = (((1,), (1,)), ((0,), (0,)))
    out = lax.dot_general(right, left, dims, precision=precision)
  return out[:, :n, :n] if n % 16 != 0 else out


def contract_bwd(
    dt3: jax.Array,
    left: jax.Array,
    right: jax.Array,
    triangle_type: Literal["incoming", "outgoing"],
    precision: base.CanonicalPrecision,
) -> tuple[jax.Array, jax.Array]:
  """Backward pass through the batched triangle contraction."""
  n = dt3.shape[1]
  if n % 16 != 0:
    n_pad = pl.cdiv(n, 16) * 16
    dt3 = _pad_spatial_16(dt3, n_pad)
    left = _pad_spatial_16(left, n_pad)
    right = _pad_spatial_16(right, n_pad)
  if triangle_type == "outgoing":
    dleft = lax.dot_general(
        dt3, right, (((2,), (1,)), ((0,), (0,))), precision=precision
    )
    dright = lax.dot_general(
        dt3, left, (((1,), (1,)), ((0,), (0,))), precision=precision
    )
  else:
    dleft = lax.dot_general(
        right, dt3, (((2,), (1,)), ((0,), (0,))), precision=precision
    )
    dright = lax.dot_general(
        left, dt3, (((2,), (2,)), ((0,), (0,))), precision=precision
    )
  if n % 16 != 0:
    return dleft[:, :n, :n], dright[:, :n, :n]
  return dleft, dright


def pad_dim(x: jax.Array, target: int, axis: int = 0) -> jax.Array:
  """Zero-pads `x` along `axis` to `target` if `x.shape[axis] < target`."""
  cur = x.shape[axis]
  if cur >= target:
    return x
  pad_width = [(0, 0)] * x.ndim
  pad_width[axis] = (0, target - cur)
  return jnp.pad(x, pad_width)


# ------------------------------------------------------------------------------
# Device-side helpers used inside Pallas Mosaic GPU kernels.
# ------------------------------------------------------------------------------


def _ln_thread_acc_layout() -> Any:
  """Per-thread `(32, dim)` FP32 layout (`0` shuffles, valid TiledLayoutAttr)."""
  fa_mod = mgpu_lib.mgpu.fragmented_array
  tl = fa_mod.TiledLayout(
      fa_mod.Tiling(((32, 8), (8, 8), (2,))),
      warp_dims=(-5,),
      lane_dims=(-3, -2),
      vector_dim=-1,
  )
  return plgpu.Layout.TILED(
      tl.tiling,
      warp_dims=tl.warp_dims,
      lane_dims=tl.lane_dims,
      vector_dim=tl.vector_dim,
  )


def init_ln_grad_acc(
    dim: int, layout: Any = None
) -> tuple[jax.Array, jax.Array]:
  """Initializes per-thread `(32, dim)` FP32 `(dscale, doffset)` accumulators."""
  del layout
  if is_interpret_mode():
    z = jnp.zeros((32, dim), jnp.float32)
    return z, z
  acc_layout = _ln_thread_acc_layout()
  mgpu = mgpu_lib.mgpu

  @plgpu.inline_mgpu(
      return_type=(
          plgpu.ShapeDtypeStruct((32, dim), jnp.dtype(jnp.float32), acc_layout),
          plgpu.ShapeDtypeStruct((32, dim), jnp.dtype(jnp.float32), acc_layout),
      )
  )
  def _zeros(_):
    f32 = mgpu_lib.ir.F32Type.get()
    z = mgpu.FragmentedArray.splat(
        mgpu.utils.c(0.0, f32), (32, dim), acc_layout.to_mgpu()
    )
    return z, z

  return _zeros()


def warpgroup_barrier() -> None:
  """Synchronizes threads within a warpgroup (no-op in interpret mode)."""
  if not is_interpret_mode():
    mgpu_lib.warpgroup_barrier()


def warpgroup_barrier_and_commit_smem() -> None:
  """Synchronizes warpgroup SMEM reads/writes across generic & async proxies."""
  if is_interpret_mode():
    plgpu.commit_smem()
  else:
    mgpu_lib.fence_async_shared_cta()
    mgpu_lib.warpgroup_barrier()


def copy_gmem_to_smem(
    src: Any,
    dst: Any,
    barrier: Any,
    *,
    predicate: jax.Array | bool | None = None,
) -> None:
  """Copies GMEM to SMEM, using `@pl.when` only in interpret mode."""
  if predicate is not None and is_interpret_mode():

    @pl.when(predicate)
    def _():
      plgpu.copy_gmem_to_smem(src, dst, barrier)

  else:
    plgpu.copy_gmem_to_smem(src, dst, barrier, predicate=predicate)


def mask_rows_1d(
    val: jax.Array,
    tile_start: jax.Array,
    m_real: int,
    m_padded: int,
    layout: Any,
) -> jax.Array:
  """Masks 1D entries `>= m_real` to zero/False when `m_real < m_padded`."""
  if m_real >= m_padded:
    return val
  row_idx = prevent_cse(
      plgpu.broadcasted_iota(jnp.int32, val.shape, 0, layout=layout), layout
  )
  start_b = plgpu.layout_cast(
      lax.broadcast_in_dim(tile_start, val.shape, ()), layout
  )
  return jnp.where(
      (start_b + row_idx) < m_real, val, jnp.zeros((), dtype=val.dtype)
  )


def mask_rows_2d(
    val: jax.Array,
    tile_start: jax.Array,
    m_real: int,
    m_padded: int,
    layout: Any,
) -> jax.Array:
  """Masks 2D rows `>= m_real` to zero when `m_real < m_padded`."""
  if m_real >= m_padded:
    return val
  layout_1d = layout.reduce(1)
  row_idx_1d = prevent_cse(
      plgpu.broadcasted_iota(
          jnp.int32, (val.shape[0],), 0, layout=layout_1d
      ),
      layout_1d,
  )
  start_1d = plgpu.layout_cast(
      lax.broadcast_in_dim(tile_start, (val.shape[0],), ()), layout_1d
  )
  valid_1d = (start_1d + row_idx_1d) < m_real
  valid_2d = plgpu.layout_cast(
      lax.broadcast_in_dim(valid_1d, val.shape, (0,)), layout
  )
  return jnp.where(valid_2d, val, jnp.zeros((), dtype=val.dtype))


def layer_norm_hat_rstd(
    x: jax.Array,
    eps: float,
    real_dim: int,
    layout: Any,
) -> tuple[jax.Array, jax.Array]:
  """Returns normalized `(x - mean) * rstd` and `rstd` along axis 1 in FP32."""
  pad_dim_size = x.shape[1]
  x_f32 = x.astype(jnp.float32)
  mean = jnp.sum(x_f32, axis=1) * (1.0 / real_dim)
  mean_b = plgpu.layout_cast(lax.broadcast_in_dim(mean, x.shape, (0,)), layout)
  centered = x_f32 - mean_b
  if real_dim < pad_dim_size:
    layout_col = layout.reduce(0)
    col_valid_1d = prevent_cse(
        plgpu.broadcasted_iota(
            jnp.int32, (pad_dim_size,), 0, layout=layout_col
        )
        < real_dim,
        layout_col,
    )
    col_valid_2d = plgpu.layout_cast(
        lax.broadcast_in_dim(col_valid_1d, x.shape, (1,)), layout
    )
    centered = jnp.where(col_valid_2d, centered, 0.0)
  var = jnp.sum(centered * centered, axis=1) * (1.0 / real_dim)
  if eps == 0.0:
    rstd = jnp.where(var > 0.0, lax.rsqrt(var), 0.0)
  else:
    rstd = lax.rsqrt(var + eps)
  rstd_b = plgpu.layout_cast(lax.broadcast_in_dim(rstd, x.shape, (0,)), layout)
  return centered * rstd_b, rstd


def prevent_cse(x: jax.Array, layout: Any) -> jax.Array:
  """Prevents MLIR LICM/CSE from hoisting broadcasted vectors across a loop."""
  if is_interpret_mode():
    return x

  @plgpu.inline_mgpu(
      arg_types=(layout,),
      return_type=plgpu.ShapeDtypeStruct(x.shape, x.dtype, layout),
  )
  def _id(_, fa):
    return fa

  return _id(x)


def apply_norm_affine(
    x_hat: jax.Array,
    scale: jax.Array,
    offset: jax.Array,
    layout: Any,
) -> jax.Array:
  """Applies `x_hat * scale + offset` in FP32."""
  scale = prevent_cse(scale, layout.reduce(0))
  offset = prevent_cse(offset, layout.reduce(0))
  scale_b = plgpu.layout_cast(
      lax.broadcast_in_dim(scale.astype(jnp.float32), x_hat.shape, (1,)),
      layout,
  )
  offset_b = plgpu.layout_cast(
      lax.broadcast_in_dim(offset.astype(jnp.float32), x_hat.shape, (1,)),
      layout,
  )
  return x_hat * scale_b + offset_b


def layer_norm_2d(
    x: jax.Array,
    scale: jax.Array,
    offset: jax.Array,
    eps: float,
    real_dim: int,
    layout: Any,
) -> jax.Array:
  """Computes LayerNorm along axis 1 of a 2D tile in FP32."""
  x_hat, _ = layer_norm_hat_rstd(x, eps, real_dim, layout)
  return apply_norm_affine(x_hat, scale, offset, layout)


def layer_norm_2d_bwd_accum(
    dy: jax.Array,
    x_hat: jax.Array,
    rstd: jax.Array,
    scale: jax.Array,
    real_dim: int,
    layout: Any,
    acc_s: jax.Array,
    acc_o: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes `dx` and accumulates `(dscale, doffset)` purely in registers."""
  dy_f32 = dy.astype(jnp.float32)
  x_hat_f32 = x_hat.astype(jnp.float32)
  if is_interpret_mode():
    acc_s = acc_s.at[0].add(jnp.sum(dy_f32 * x_hat_f32, axis=0))
    acc_o = acc_o.at[0].add(jnp.sum(dy_f32, axis=0))
  else:
    acc_layout = _ln_thread_acc_layout()
    mgpu = mgpu_lib.mgpu
    np = mgpu_lib.np
    from jaxlib.mlir.dialects import arith  # pylint: disable=g-import-not-at-top

    @plgpu.inline_mgpu(
        arg_types=(acc_layout, acc_layout, layout, layout),
        return_type=(
            plgpu.ShapeDtypeStruct(
                acc_s.shape, jnp.dtype(jnp.float32), acc_layout
            ),
            plgpu.ShapeDtypeStruct(
                acc_o.shape, jnp.dtype(jnp.float32), acc_layout
            ),
        ),
    )
    def _accum(_, as_fa, ao_fa, dy_fa, xh_fa):
      m_tiles = dy_fa.registers.shape[0]
      k_tiles = dy_fa.registers.shape[1]
      dy_regs = dy_fa.registers.reshape(m_tiles, k_tiles, -1)
      xh_regs = xh_fa.registers.reshape(m_tiles, k_tiles, -1)
      sub_m = dy_regs.shape[2]
      out_s = np.empty_like(as_fa.registers)
      out_o = np.empty_like(ao_fa.registers)
      for kt in range(k_tiles):
        ds = as_fa.registers.flat[kt]
        do = ao_fa.registers.flat[kt]
        for mt in range(m_tiles):
          for sm in range(sub_m):
            d_reg = dy_regs[mt, kt, sm]
            ds = arith.addf(ds, arith.mulf(d_reg, xh_regs[mt, kt, sm]))
            do = arith.addf(do, d_reg)
        out_s.flat[kt] = ds
        out_o.flat[kt] = do
      return (
          mgpu.FragmentedArray(
              _registers=out_s, _layout=as_fa.layout, _is_signed=as_fa.is_signed
          ),
          mgpu.FragmentedArray(
              _registers=out_o, _layout=ao_fa.layout, _is_signed=ao_fa.is_signed
          ),
      )

    acc_s, acc_o = _accum(acc_s, acc_o, dy_f32, x_hat_f32)

  scale = prevent_cse(scale, layout.reduce(0))
  scale_b = plgpu.layout_cast(
      lax.broadcast_in_dim(scale.astype(jnp.float32), dy.shape, (1,)),
      layout,
  )
  dx_hat = dy_f32 * scale_b
  c1 = jnp.sum(dx_hat, axis=1) * (1.0 / real_dim)
  c2 = jnp.sum(dx_hat * x_hat_f32, axis=1) * (1.0 / real_dim)
  c1_b = plgpu.layout_cast(lax.broadcast_in_dim(c1, dy.shape, (0,)), layout)
  c2_b = plgpu.layout_cast(lax.broadcast_in_dim(c2, dy.shape, (0,)), layout)
  rstd_b = plgpu.layout_cast(lax.broadcast_in_dim(rstd, dy.shape, (0,)), layout)
  dx = (dx_hat - c1_b - x_hat_f32 * c2_b) * rstd_b
  return dx, acc_s, acc_o


def sigmoid_f32(x: jax.Array) -> jax.Array:
  """Full-precision vectorized FP32 sigmoid (`ex2.approx` + `rcp.approx`)."""
  return jax.nn.sigmoid(x)


def sigmoid_fwd_f32(x: jax.Array, layout: Any) -> jax.Array:
  """Fast single-MUFU (`tanh.approx.f32`) FP32 sigmoid for forward kernels."""
  if is_interpret_mode() or layout == plgpu.Layout.TCGEN05:
    return 0.5 + 0.5 * lax.tanh(0.5 * x)

  ir = mgpu_lib.ir
  llvm = mgpu_lib.llvm
  np = mgpu_lib.np

  @plgpu.inline_mgpu(
      arg_types=(layout,),
      return_type=plgpu.ShapeDtypeStruct(
          x.shape, jnp.dtype(jnp.float32), layout
      ),
  )
  def _sigmoid(_, fa):
    f32 = ir.F32Type.get()
    i32 = ir.IntegerType.get_signless(32)
    vec_shape = ir.VectorType(fa.registers.flat[0].type).shape
    vec_len = vec_shape[0]
    vec_ty = ir.VectorType.get((vec_len,), f32)
    ptx = (
        "{\n"
        "  .reg .f32 t;\n"
        "  mul.f32 t, $1, 0f3F000000;\n"
        "  tanh.approx.f32 t, t;\n"
        "  fma.rn.f32 $0, t, 0f3F000000, 0f3F000000;\n"
        "}"
    )
    new_regs = np.empty_like(fa.registers)
    for idx_r, reg in np.ndenumerate(fa.registers):
      v = llvm.mlir_undef(vec_ty)
      for i in range(vec_len):
        ci = mgpu_lib.mgpu.utils.c(i, i32)
        elem = llvm.extractelement(reg, ci)
        res = llvm.inline_asm(f32, [elem], ptx, "=f,f", has_side_effects=False)
        assert isinstance(res, ir.Value)
        v = llvm.insertelement(v, res, ci)
      new_regs[idx_r] = v
    return mgpu_lib.mgpu.FragmentedArray(
        _registers=new_regs, _layout=fa.layout, _is_signed=fa.is_signed
    )

  return _sigmoid(x)

