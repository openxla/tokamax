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
_BlockDim = Annotated[pydantic.PositiveInt, pydantic.Field(ge=8, le=4096)]


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
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
  block_i: _BlockDim | None = None
  block_j: _BlockDim | None = None
  epilogue_block_i: _BlockDim | None = None
  epilogue_block_j: _BlockDim | None = None


def has_mosaic_tpu_support(device: jax.Device | None = None) -> bool:
  """Returns True if Mosaic TPU is supported on `device` (or default device)."""
  if device is None:
    device = backend.get_default_device()
  if device.platform != "tpu":
    return False
  try:
    return pltpu.get_tpu_info().generation >= 4
  except Exception:  # pylint: disable=broad-exception-caught
    kind = device.device_kind.lower()
    return any(k in kind for k in ("v4", "v5", "v6", "tpu7", "7x"))


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
  """Validates inputs for Pallas Mosaic TPU forward pass."""
  if not has_mosaic_tpu_support():
    raise NotImplementedError("Mosaic TPU is not supported on this platform.")

  if not precision_lib.is_default(
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
    if w.dtype != x.dtype:
      raise NotImplementedError(
          f"All weight and norm arrays must match x.dtype={x.dtype}, got"
          f" {w.dtype}."
      )

  n, _, c = x.shape
  h, d = projection_out_weights.shape
  if not {c, h, d} <= SUPPORTED_CHANNEL_DIMS:
    raise NotImplementedError(
        "Mosaic TPU TriangleMultiplication requires C, H, D in"
        f" {sorted(SUPPORTED_CHANNEL_DIMS)}, got C={c}, H={h}, D={d}."
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
  """Validates inputs for Pallas Mosaic TPU backward pass."""
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

  n, _, d = dout.shape
  h = projection_out_weights.shape[0]
  if dout.shape != (n, n, d):
    raise NotImplementedError(
        f"Expected dout shape {(n, n, d)}, got {dout.shape}."
    )
  for idx, res in enumerate(residuals):
    if res.shape != (h, n, n):
      raise NotImplementedError(
          f"Expected residuals[{idx}] shape {(h, n, n)}, got {res.shape}."
      )

  for w in (*residuals, dout):
    if w.dtype != x.dtype:
      raise NotImplementedError(
          "All residual, weight, norm, and dout arrays must match"
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
    return dleft, dright
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


# ------------------------------------------------------------------------------
# Device-side helpers used inside Pallas Mosaic TPU kernels (128-lane dense).
# ------------------------------------------------------------------------------


def matmul_icj(w_mk: jax.Array, x_ikj: jax.Array) -> jax.Array:
  """Computes `(Bi, M, Bj)` directly on MXU without VMEM reshape spills."""
  w_bmk = lax.broadcast_in_dim(w_mk, (x_ikj.shape[0], *w_mk.shape), (1, 2))
  return lax.dot_general(
      w_bmk,
      x_ikj,
      (((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )


def matmul_wt_icj(w_mk: jax.Array, dy_imj: jax.Array) -> jax.Array:
  """Computes `(Bi, K, Bj)` directly with hardware-fused LHS transpose."""
  w_bmk = lax.broadcast_in_dim(w_mk, (dy_imj.shape[0], *w_mk.shape), (1, 2))
  return lax.dot_general(
      w_bmk,
      dy_imj,
      (((1,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )


def matmul_dw_icj(dy_imj: jax.Array, x_ikj: jax.Array) -> jax.Array:
  """Computes `(M, K)` weight gradient streamed across `Bi` in registers."""
  dw_bmk = lax.dot_general(
      dy_imj,
      x_ikj,
      (((2,), (2,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )
  return jnp.sum(dw_bmk, axis=0)


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
  rstd = (
      jnp.where(var > 0.0, lax.rsqrt(var), 0.0)
      if eps <= 0.0
      else lax.rsqrt(var + eps)
  )
  return centered * rstd, rstd


def apply_norm_affine_axis0(
    x_hat: jax.Array,
    scale: jax.Array,
    offset: jax.Array,
) -> jax.Array:
  """Applies `x_hat * scale + offset` along axis 0 of `(H, Bi, Bj)` in FP32."""
  scale_b = lax.broadcast_in_dim(
      scale.astype(jnp.float32), x_hat.shape, (0,)
  )
  offset_b = lax.broadcast_in_dim(
      offset.astype(jnp.float32), x_hat.shape, (0,)
  )
  return x_hat * scale_b + offset_b


def layer_norm_3d_axis1_hat_rstd(
    x: jax.Array,
    eps: float,
    *,
    sanitize_nonfinite: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Returns `(x - mean) * rstd` and `rstd` along axis 1 of `(Bi, K, Bj)`."""
  k = x.shape[1]
  x_f32 = x.astype(jnp.float32)
  if sanitize_nonfinite:
    x_f32 = jnp.where(jnp.isfinite(x_f32), x_f32, 0.0)
  mean = jnp.sum(x_f32, axis=1, keepdims=True) * (1.0 / k)
  centered = x_f32 - mean
  var = jnp.sum(centered * centered, axis=1, keepdims=True) * (1.0 / k)
  rstd = (
      jnp.where(var > 0.0, lax.rsqrt(var), 0.0)
      if eps <= 0.0
      else lax.rsqrt(var + eps)
  )
  return centered * rstd, rstd


def apply_norm_affine_axis1(
    x_hat: jax.Array,
    scale: jax.Array,
    offset: jax.Array,
) -> jax.Array:
  """Applies `x_hat * scale + offset` along sublane axis 1 of `(Bi, K, Bj)`."""
  scale_b = lax.broadcast_in_dim(
      scale.astype(jnp.float32), x_hat.shape, (1,)
  )
  offset_b = lax.broadcast_in_dim(
      offset.astype(jnp.float32), x_hat.shape, (1,)
  )
  return x_hat * scale_b + offset_b


def layer_norm_3d_axis1(
    x: jax.Array,
    scale: jax.Array,
    offset: jax.Array,
    eps: float,
    *,
    sanitize_nonfinite: bool = False,
) -> jax.Array:
  """Computes FP32 LayerNorm along sublane axis 1 of `(Bi, K, Bj)`."""
  x_hat, _ = layer_norm_3d_axis1_hat_rstd(
      x, eps, sanitize_nonfinite=sanitize_nonfinite
  )
  return apply_norm_affine_axis1(x_hat, scale, offset).astype(x.dtype)


def layer_norm_3d_axis1_bwd(
    dact_f32: jax.Array,
    x_hat: jax.Array,
    rstd: jax.Array,
    scale: jax.Array,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
  """Computes `dx` `(Bi, K, Bj)` and `(1, 2, K)` LayerNorm gradients."""
  k = x_hat.shape[1]
  ds_do_2k_bj = jnp.sum(
      jnp.concatenate([dact_f32 * x_hat, dact_f32], axis=1), axis=0
  )
  bj = ds_do_2k_bj.shape[1]
  acc_128 = ds_do_2k_bj[:, :128]
  for col in range(128, bj, 128):
    acc_128 = acc_128 + ds_do_2k_bj[:, col : col + 128]
  ds_do = jnp.sum(jnp.transpose(acc_128), axis=0).reshape(1, 2, k)
  dx_hat = dact_f32 * lax.broadcast_in_dim(
      scale.astype(jnp.float32), dact_f32.shape, (1,)
  )
  c1 = jnp.sum(dx_hat, axis=1, keepdims=True) * (1.0 / k)
  c2 = jnp.sum(dx_hat * x_hat, axis=1, keepdims=True) * (1.0 / k)
  dx = ((dx_hat - c1 - x_hat * c2) * rstd).astype(dtype)
  return dx, ds_do
