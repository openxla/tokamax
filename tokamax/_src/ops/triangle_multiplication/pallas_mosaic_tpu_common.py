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
"""Common helpers for Pallas-Mosaic-TPU Triangle Multiplication."""

import math
from typing import Annotated, Any, Final, Literal

import immutabledict
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import pydantic
from tokamax._src import precision as precision_lib
from tokamax._src import pydantic as pydantic_lib
from tokamax._src.ops.triangle_multiplication import base

Residuals = base.Residuals
type Key = immutabledict.immutabledict[str, Any]

SUPPORTED_DTYPES: Final[frozenset[jnp.dtype]] = frozenset(
    {jnp.dtype(jnp.bfloat16)}
)
SUPPORTED_CHANNEL_DIMS: Final[frozenset[int]] = frozenset({16, 32, 64, 128})
MIN_KERNEL_CHANNEL_DIM: Final[int] = 16
MAX_INT32: Final[int] = 2**31 - 1

_BlockM = Annotated[pydantic_lib.PowerOfTwo, pydantic.Field(ge=128, le=1024)]
_BlockI = Annotated[
    pydantic.PositiveInt, pydantic.Field(ge=8, le=4096, multiple_of=8)
]
_BlockJ = Annotated[
    pydantic.PositiveInt, pydantic.Field(ge=128, le=4096, multiple_of=128)
]


@pydantic.dataclasses.dataclass(
    frozen=True,
    kw_only=True,
    slots=True,
    config=pydantic.ConfigDict(extra="forbid"),
)
class Config:
  """Configuration for Pallas-Mosaic-TPU Triangle Multiplication kernels.

  Attributes:
    block_m: Tile size control parameter in [128, 1024] (power of two). Mapped
      to 3D `(block_i, block_j)` tile dimensions dividing `N_pad`.
    epilogue_block_m: Optional tile size override for the epilogue kernel.
    block_i: Optional explicit tile height override along axis `i`.
    block_j: Optional explicit tile width override along axis `j`.
    epilogue_block_i: Optional explicit `block_i` override for the epilogue.
    epilogue_block_j: Optional explicit `block_j` override for the epilogue.
  """

  block_m: _BlockM = 512
  epilogue_block_m: _BlockM | None = None
  block_i: _BlockI | None = None
  block_j: _BlockJ | None = None
  epilogue_block_i: _BlockI | None = None
  epilogue_block_j: _BlockJ | None = None


def _infer_explicit_device(x: Any) -> jax.Device | None:
  """Returns concrete device of `x` if explicitly placed, else `None`."""
  sharding = getattr(x, "sharding", None)
  if sharding is not None:
    device_set = getattr(sharding, "device_set", None)
    if device_set:
      return next(iter(device_set))
  if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer):
    try:
      return next(iter(x.devices()), None)
    except jax.errors.ConcretizationTypeError:
      return None
  return None


def infer_tpu_device(x: Any = None) -> jax.Device | None:
  """Infers target TPU `jax.Device` from `x`, default device, or local TPUs."""
  if x is not None:
    explicit_dev = _infer_explicit_device(x)
    if explicit_dev is not None:
      return explicit_dev if explicit_dev.platform == "tpu" else None
  default_dev = backend.get_default_device()
  if default_dev.platform == "tpu":
    return default_dev
  try:
    tpu_devs = jax.devices("tpu")
    return tpu_devs[0] if tpu_devs else None
  except RuntimeError:
    return None


def has_mosaic_tpu_support(device: jax.Device | None = None) -> bool:
  """Returns True if Mosaic TPU is supported on `device` (or default device)."""
  if device is None:
    device = backend.get_default_device()
  if device.platform != "tpu":
    return False
  try:
    if (
        isinstance(device, jax.Device)
        and backend.get_default_device() != device
    ):
      with jax.default_device(device):
        return pltpu.get_tpu_info().generation >= 4
    return pltpu.get_tpu_info().generation >= 4
  except Exception:  # pylint: disable=broad-exception-caught
    kind = device.device_kind.lower()
    return any(k in kind for k in ("v4", "v5", "v6", "tpu7", "7x"))


def get_tpu_info(
    x: Any = None,
) -> tuple[int, int, int, int | None, int | None]:
  """Returns `(gen, vmem_mib, num_cores, vmem_limit_bytes, vmem_budget_bytes)`."""
  tpu_dev = infer_tpu_device(x)
  try:
    if (
        isinstance(tpu_dev, jax.Device)
        and backend.get_default_device() != tpu_dev
    ):
      with jax.default_device(tpu_dev):
        info = pltpu.get_tpu_info()
    else:
      info = pltpu.get_tpu_info()
    return (
        info.generation,
        info.vmem_capacity_bytes // (1024 * 1024),
        info.num_cores,
        int(0.90 * info.vmem_capacity_bytes),
        int(0.80 * info.vmem_capacity_bytes),
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return 6, 64, 1, None, 50 * 1024 * 1024


def validate_fwd_inputs(
    x: jax.Array,
    mask: jax.Array,
    projection_in_weights: jax.Array,
    gate_in_weights: jax.Array,
    projection_out_weights: jax.Array,
    gate_out_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    layernorm_out_scale: jax.Array,
    layernorm_out_offset: jax.Array,
    triangle_type: str,
    precision: base.CanonicalPrecision,
    epsilon: float,
) -> None:
  """Validates inputs for Pallas Mosaic TPU forward pass."""
  if triangle_type not in ("incoming", "outgoing"):
    raise NotImplementedError(
        f"Unsupported triangle_type={triangle_type!r}; expected 'incoming' or"
        " 'outgoing'."
    )

  if (
      isinstance(epsilon, bool)
      or not isinstance(epsilon, (int, float))
      or not math.isfinite(epsilon)
      or epsilon < 0.0
  ):
    raise NotImplementedError(
        f"epsilon must be a finite non-negative float, got {epsilon!r}."
    )

  all_arrays = (
      x,
      mask,
      projection_in_weights,
      gate_in_weights,
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
  )
  for arr in all_arrays:
    if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
      raise NotImplementedError(f"Expected array input, got {type(arr)}.")
    if not all(isinstance(dim, int) for dim in arr.shape):
      raise NotImplementedError("Mosaic TPU does not support symbolic shapes.")

  if precision not in (
      (lax.Precision.DEFAULT, lax.Precision.DEFAULT),
      lax.DotAlgorithmPreset.DEFAULT,
      lax.DotAlgorithmPreset.BF16_BF16_F32,
  ) or not precision_lib.is_default(
      x.dtype, projection_in_weights.dtype, precision
  ):
    raise NotImplementedError(f"{precision=} is not supported.")

  if jnp.dtype(x.dtype) not in SUPPORTED_DTYPES:
    raise NotImplementedError(
        "Mosaic TPU triangle_multiplication only supports"
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
    if jnp.dtype(w.dtype) != jnp.dtype(x.dtype):
      raise NotImplementedError(
          f"All weight and norm arrays must match x.dtype={x.dtype}, got"
          f" {w.dtype}."
      )

  if x.ndim != 3 or x.shape[0] != x.shape[1]:
    raise NotImplementedError(f"Expected x of shape (N, N, C), got {x.shape}.")
  n, _, c = x.shape

  if mask.shape != (n, n) or jnp.dtype(mask.dtype) != jnp.dtype(jnp.bool_):
    raise NotImplementedError(
        f"Expected bool mask of shape {(n, n)}, got shape={mask.shape},"
        f" dtype={mask.dtype}."
    )

  if projection_out_weights.ndim != 2:
    raise NotImplementedError(
        "Expected 2D projection_out_weights (H, D), got"
        f" {projection_out_weights.shape}."
    )
  h, d = projection_out_weights.shape

  if not {c, h, d} <= SUPPORTED_CHANNEL_DIMS:
    raise NotImplementedError(
        "Mosaic TPU TriangleMultiplication requires C, H, D in"
        f" {sorted(SUPPORTED_CHANNEL_DIMS)}, got C={c}, H={h}, D={d}."
    )

  expected_shapes = (
      ("projection_in_weights", projection_in_weights.shape, (c, 2, h)),
      ("gate_in_weights", gate_in_weights.shape, (c, 2, h)),
      ("gate_out_weights", gate_out_weights.shape, (c, d)),
      ("layernorm_in_scale", layernorm_in_scale.shape, (c,)),
      ("layernorm_in_offset", layernorm_in_offset.shape, (c,)),
      ("layernorm_out_scale", layernorm_out_scale.shape, (h,)),
      ("layernorm_out_offset", layernorm_out_offset.shape, (h,)),
  )
  for name, got_shape, exp_shape in expected_shapes:
    if got_shape != exp_shape:
      raise NotImplementedError(
          f"Expected {name} shape {exp_shape}, got {got_shape}."
      )

  if n == 0:
    raise NotImplementedError("Empty inputs are not supported.")
  n_pad = pl.cdiv(n, 128) * 128
  if n_pad * n_pad * max(c, h, d) - 1 > MAX_INT32:
    raise NotImplementedError(
        f"N={n} (padded to {n_pad}) with max(C, H, D)={max(c, h, d)} would"
        " overflow the int32 offsets used by the Mosaic TPU kernels."
    )


def validate_vjp_inputs(
    residuals: base.Residuals,
    out: jax.Array,
    dout: jax.Array,
    x: jax.Array,
    mask: jax.Array,
    projection_in_weights: jax.Array,
    gate_in_weights: jax.Array,
    projection_out_weights: jax.Array,
    gate_out_weights: jax.Array,
    layernorm_in_scale: jax.Array,
    layernorm_in_offset: jax.Array,
    layernorm_out_scale: jax.Array,
    layernorm_out_offset: jax.Array,
    triangle_type: str,
    precision: base.CanonicalPrecision,
    epsilon: float,
    return_residuals: bool,
) -> None:
  """Validates inputs for Pallas Mosaic TPU backward pass."""
  if return_residuals:
    raise NotImplementedError("`return_residuals` not supported.")

  validate_fwd_inputs(
      x,
      mask,
      projection_in_weights,
      gate_in_weights,
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
      triangle_type,
      precision,
      epsilon,
  )

  if not isinstance(residuals, (tuple, list)) or len(residuals) != 3:
    raise NotImplementedError(
        "Expected residuals to be a 3-tuple (left, right, t3), got"
        f" {type(residuals).__name__} of length"
        f" {len(residuals) if isinstance(residuals, (tuple, list)) else 'N/A'}."
    )

  for arr in (*residuals, out, dout):
    if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
      raise NotImplementedError(f"Expected array input, got {type(arr)}.")
    if not all(isinstance(dim, int) for dim in arr.shape):
      raise NotImplementedError("Mosaic TPU does not support symbolic shapes.")

  n = x.shape[0]
  h, d = projection_out_weights.shape
  if out.shape != (n, n, d):
    raise NotImplementedError(
        f"Expected out shape {(n, n, d)}, got {out.shape}."
    )
  if dout.shape != (n, n, d):
    raise NotImplementedError(
        f"Expected dout shape {(n, n, d)}, got {dout.shape}."
    )
  for idx, res in enumerate(residuals):
    if res.shape != (h, n, n):
      raise NotImplementedError(
          f"Expected residuals[{idx}] shape {(h, n, n)}, got {res.shape}."
      )

  for w in (*residuals, out, dout):
    if jnp.dtype(w.dtype) != jnp.dtype(x.dtype):
      raise NotImplementedError(
          "All residual, out, and dout arrays must match"
          f" x.dtype={x.dtype}, got {w.dtype}."
      )


def contract(
    left: jax.Array,
    right: jax.Array,
    triangle_type: Literal["incoming", "outgoing"],
    precision: base.CanonicalPrecision,
) -> jax.Array:
  """Batched (over H) triangle contraction of two (H, N, N) planes."""
  if triangle_type == "outgoing":
    dims = (((2,), (2,)), ((0,), (0,)))
    return lax.dot_general(left, right, dims, precision=precision)
  else:
    dims = (((1,), (1,)), ((0,), (0,)))
    return lax.dot_general(right, left, dims, precision=precision)


def contract_bwd(
    dt3: jax.Array,
    left: jax.Array,
    right: jax.Array,
    triangle_type: Literal["incoming", "outgoing"],
    precision: base.CanonicalPrecision,
) -> tuple[jax.Array, jax.Array]:
  """Backward pass through the batched triangle contraction."""
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
  return dleft, dright


def pad_dim(x: jax.Array, target: int, axis: int = 0) -> jax.Array:
  """Zero-pads `x` along `axis` to `target` if `x.shape[axis] < target`."""
  cur = x.shape[axis]
  if cur >= target:
    return x
  pad_width = [(0, 0)] * x.ndim
  pad_width[axis] = (0, target - cur)
  return jnp.pad(x, pad_width)


UNPIN_XPAD_MIN_BYTES: Final[int] = 1024 * 1024 * 1024


def should_unpin_xpad(x: jax.Array, n_pad: int) -> bool:
  """Returns True if `x_pad` (`>= 1 GiB`) should be unpinned across contract."""
  return n_pad * x.shape[2] * n_pad * x.dtype.itemsize >= UNPIN_XPAD_MIN_BYTES


def transpose_and_pad_icj(x: jax.Array, n_pad: int) -> jax.Array:
  """Transposes `(N, N, K)` -> `(N_pad, K, N_pad)` for 128-lane `Bj` kernels."""
  x_icj = jnp.transpose(x, (0, 2, 1))
  return pad_dim(pad_dim(x_icj, n_pad, 0), n_pad, 2)


def cast_and_pad_mask(
    mask: jax.Array, n_pad: int, dtype: Any, x: Any = None
) -> jax.Array:
  """Casts boolean `(N, N)` mask to `dtype` and pads to `(N_pad, N_pad)`."""
  gen, _, num_cores, _, _ = get_tpu_info(mask if x is None else x)
  if gen == 5 and num_cores > 1:
    m, k = mask.shape
    mask_bcast = jax.lax.optimization_barrier(
        lax.broadcast_in_dim(mask, (2, m, k), (1, 2))
    )
    mask_f = lax.index_in_dim(
        jax.lax.optimization_barrier(mask_bcast.astype(dtype)),
        0,
        axis=0,
        keepdims=False,
    )
  else:
    mask_f = mask.astype(dtype)
  return pad_dim(pad_dim(mask_f, n_pad, 0), n_pad, 1)


# ------------------------------------------------------------------------------
# Device-side helpers used inside Pallas Mosaic TPU kernels (128-lane dense).
# ------------------------------------------------------------------------------


def pack_icj_to_2d(x_ikj: jax.Array) -> jax.Array:
  """Packs `bf16`/`f32` `(Bi, K, Bj)` into `(K, Bi * Bj)` with zero VPU ops."""
  bi, k, bj = x_ikj.shape
  if bi == 1:
    return x_ikj.reshape(k, bj)
  return jnp.concatenate([x_ikj[i : i + 1] for i in range(bi)], axis=2).reshape(
      k, bi * bj
  )


def unpack_2d_to_icj(y_2d: jax.Array, bi: int, bj: int) -> jax.Array:
  """Unpacks `(M, Bi * Bj)` into `(Bi, M, Bj)` with zero VPU ops."""
  m = y_2d.shape[0]
  if bi == 1:
    return y_2d.reshape(1, m, bj)
  y_3d = y_2d.reshape(1, m, bi * bj)
  return jnp.concatenate(
      [y_3d[:, :, i * bj : (i + 1) * bj] for i in range(bi)], axis=0
  )


def matmul_2d(w_mk: jax.Array, x_kt: jax.Array) -> jax.Array:
  """Computes unbatched `(M, K) @ (K, T) -> (M, T)` in FP32 on the TPU MXU."""
  return lax.dot_general(
      w_mk,
      x_kt,
      (((1,), (0,)), ((), ())),
      preferred_element_type=jnp.float32,
  )


def matmul_wt_2d(w_mk: jax.Array, dy_mt: jax.Array) -> jax.Array:
  """Computes unbatched `(M, K)^T @ (M, T) -> (K, T)` in FP32 on the TPU MXU."""
  return lax.dot_general(
      w_mk,
      dy_mt,
      (((0,), (0,)), ((), ())),
      preferred_element_type=jnp.float32,
  )


def matmul_dw_2d(dy_mt: jax.Array, x_kt: jax.Array) -> jax.Array:
  """Computes unbatched `(M, T) @ (K, T)^T -> (M, K)` in FP32 on the TPU MXU."""
  return lax.dot_general(
      dy_mt,
      x_kt,
      (((1,), (1,)), ((), ())),
      preferred_element_type=jnp.float32,
  )


def matmul_icj(w_mk: jax.Array, x_ikj: jax.Array) -> jax.Array:
  """Computes `(M, K) @ (Bi, K, Bj) -> (Bi, M, Bj)` via unbatched 2D MXU dot."""
  bi, _, bj = x_ikj.shape
  return unpack_2d_to_icj(matmul_2d(w_mk, pack_icj_to_2d(x_ikj)), bi, bj)


def matmul_wt_icj(w_mk: jax.Array, dy_imj: jax.Array) -> jax.Array:
  """Computes `(M, K)^T @ (Bi, M, Bj) -> (Bi, K, Bj)` via 2D MXU dot."""
  bi, _, bj = dy_imj.shape
  return unpack_2d_to_icj(matmul_wt_2d(w_mk, pack_icj_to_2d(dy_imj)), bi, bj)


def matmul_dw_icj(dy_imj: jax.Array, x_ikj: jax.Array) -> jax.Array:
  """Computes `sum_{i,j} dy[i,m,j] * x[i,k,j] -> (M, K)` via 2D MXU dot."""
  return matmul_dw_2d(pack_icj_to_2d(dy_imj), pack_icj_to_2d(x_ikj))


def matmul_3d(w_mk: jax.Array, x_ikj: jax.Array) -> jax.Array:
  """Computes `(Bi, M, Bj)` directly on MXU without VMEM concat packing."""
  w_bmk = lax.broadcast_in_dim(w_mk, (x_ikj.shape[0], *w_mk.shape), (1, 2))
  return lax.dot_general(
      w_bmk,
      x_ikj,
      (((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )


def matmul_wt_3d(w_mk: jax.Array, dy_imj: jax.Array) -> jax.Array:
  """Computes `(Bi, K, Bj)` via a single weight transpose and `matmul_3d`."""
  w_km = jnp.transpose(w_mk)
  if dy_imj.ndim == 2:
    return matmul_2d(w_km, dy_imj)
  return matmul_3d(w_km, dy_imj)


def matmul_dw_3d(dy_imj: jax.Array, x_ikj: jax.Array) -> jax.Array:
  """Computes `(M, K)` weight gradient streamed across `Bi` in registers."""
  acc = lax.dot_general(
      dy_imj[0],
      x_ikj[0],
      (((1,), (1,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  for i in range(1, dy_imj.shape[0]):
    acc = acc + lax.dot_general(
        dy_imj[i],
        x_ikj[i],
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
  return acc


def layer_norm_3d_axis0_hat_rstd(
    x: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
  """Returns `(x - mean) * rstd` and `rstd` along axis 0 of `(H, Bi, Bj)`."""
  h = x.shape[0]
  x_f32 = x.astype(jnp.float32)
  mean = jnp.sum(x_f32, axis=0, keepdims=True) * (1.0 / h)
  centered = x_f32 - mean
  var = jnp.sum(centered * centered, axis=0, keepdims=True) * (1.0 / h)
  var_eps = var if eps == 0.0 else var + jnp.asarray(eps, jnp.float32)
  rstd = jnp.where(var_eps == 0.0, 0.0, lax.rsqrt(var_eps))
  return centered * rstd, rstd


def apply_norm_affine_axis0(
    x_hat: jax.Array,
    scale: jax.Array,
    offset: jax.Array,
) -> jax.Array:
  """Applies `x_hat * scale + offset` along axis 0 of `(H, Bi, Bj)` in FP32."""
  scale_b = lax.broadcast_in_dim(scale.astype(jnp.float32), x_hat.shape, (0,))
  offset_b = lax.broadcast_in_dim(offset.astype(jnp.float32), x_hat.shape, (0,))
  return x_hat * scale_b + offset_b


def layer_norm_3d_axis1_hat_rstd(
    x: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
  """Returns `(x - mean) * rstd` and `rstd` along axis 1 of `(Bi, K, Bj)`."""
  k = x.shape[1]
  x_f32 = x.astype(jnp.float32)
  mean = jnp.sum(x_f32, axis=1, keepdims=True) * (1.0 / k)
  centered = x_f32 - mean
  var = jnp.sum(centered * centered, axis=1, keepdims=True) * (1.0 / k)
  var_eps = var if eps == 0.0 else var + jnp.asarray(eps, jnp.float32)
  rstd = jnp.where(var_eps == 0.0, 0.0, lax.rsqrt(var_eps))
  return centered * rstd, rstd


def apply_norm_affine_axis1(
    x_hat: jax.Array,
    scale: jax.Array,
    offset: jax.Array,
) -> jax.Array:
  """Applies `x_hat * scale + offset` along sublane axis 1 of `(Bi, K, Bj)`."""
  scale_b = lax.broadcast_in_dim(scale.astype(jnp.float32), x_hat.shape, (1,))
  offset_b = lax.broadcast_in_dim(offset.astype(jnp.float32), x_hat.shape, (1,))
  return x_hat * scale_b + offset_b


def layer_norm_3d_axis1(
    x: jax.Array,
    scale: jax.Array,
    offset: jax.Array,
    eps: float,
) -> jax.Array:
  """Computes FP32 LayerNorm along sublane axis 1 of `(Bi, K, Bj)`."""
  x_hat, _ = layer_norm_3d_axis1_hat_rstd(x, eps)
  return apply_norm_affine_axis1(x_hat, scale, offset).astype(x.dtype)


def layer_norm_3d_axis1_bwd(
    dact_f32: jax.Array,
    x_hat: jax.Array,
    rstd: jax.Array,
    scale: jax.Array,
    dtype: jnp.dtype,
    *,
    from_hat_rstd: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Computes `dx` `(Bi, K, Bj)` and `(1, 2 * K, 128)` partial LN gradients."""
  k = x_hat.shape[1]
  rstd_fin = rstd * 0.0 == 0.0
  rstd_s = jnp.where(rstd_fin, rstd, 0.0)
  if from_hat_rstd:
    x_hat_ok = rstd_fin & ~(rstd == 0.0)
    x_hat_s = jnp.where(
        lax.broadcast_in_dim(x_hat_ok, x_hat.shape, (0, 1, 2)), x_hat, 0.0
    )
  else:
    x_hat_s = jnp.where(x_hat * 0.0 == 0.0, x_hat, 0.0)
  ds_do_2k_bj = jnp.concatenate(
      [jnp.sum(dact_f32 * x_hat_s, axis=0), jnp.sum(dact_f32, axis=0)], axis=0
  )
  bj = ds_do_2k_bj.shape[1]
  acc_128 = ds_do_2k_bj[:, :128]
  for col in range(128, bj, 128):
    acc_128 = acc_128 + ds_do_2k_bj[:, col : col + 128]
  ds_do_128 = acc_128.reshape(1, 2 * k, 128)
  scale_b = lax.broadcast_in_dim(
      scale.astype(jnp.float32), dact_f32.shape, (1,)
  )
  dact_zero = dact_f32 == 0.0
  dx_hat = jnp.where(dact_zero, 0.0, dact_f32 * scale_b)
  c1 = jnp.sum(dx_hat, axis=1, keepdims=True) * (1.0 / k)
  c2 = jnp.sum(dx_hat * x_hat_s, axis=1, keepdims=True) * (1.0 / k)
  dx = ((dx_hat - c1 - x_hat_s * c2) * rstd_s).astype(dtype)
  return dx, ds_do_128
