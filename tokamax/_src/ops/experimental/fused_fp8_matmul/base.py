# Copyright 2026 Google LLC
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
"""Base class and XLA reference for the fused FP8 matmul.

The op computes `lhs @ rhs` with both operands in `float8_e4m3fn` on the MXU:

* `lhs` (activations, shape `(m, k)`) is quantized *inside* the op with one
  absmax scale per row. Row scales are the layout that lets the scale be
  factored out of the contraction over `k`.
* `rhs` (weights, shape `(k, n)`) is quantized with one absmax scale per output
  column. It can be passed either already quantized, as a `qwix.QArray`, or as
  a plain array that the op quantizes per column before the matmul.

The output is `(lhs_q @ rhs_q) * (lhs_scale * rhs_scale)` cast to `lhs.dtype`.

The default VJP is a plain-XLA reference. The gradient with respect to a plain
`rhs` array is the usual `dout`-weighted product; a `QArray` `rhs` receives a
zero gradient (pass the unquantized weight if you want to train it).
"""

import dataclasses
from typing import Any, override

import jax
import jax.numpy as jnp
from tokamax._src import quantization
from tokamax._src.ops import op

QArray = quantization.QArray
AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray

FP8_DTYPE = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8_DTYPE).max)  # 448.0

# `(lhs_q, lhs_scale, rhs_q, rhs_scale)`: the FP8 operands and their scales, so
# the backward pass does not have to re-quantize anything.
#   lhs_q:     `(m, k)` float8_e4m3fn
#   lhs_scale: `(m, 1)` float32
#   rhs_q:     `(k, n)` float8_e4m3fn
#   rhs_scale: `(n,)`   float32
type Residuals = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


def quantize_rows_fp8(
    x: jax.Array, eps: float = 1e-7
) -> tuple[jax.Array, jax.Array]:
  """Quantizes `x` to FP8 with one absmax scale per row (last axis).

  This is the XLA counterpart of the in-kernel quantization: the scaling is
  done in `x`'s dtype, the scale is float32.

  Args:
    x: The array to quantize.
    eps: Lower bound on the per-row absmax, so an all-zero row gets a finite
      scale.

  Returns:
    `(xq, scale)` with `scale` of shape `x.shape[:-1] + (1,)` in float32 such
    that `x ~= xq * scale`.
  """
  max_abs = jnp.maximum(jnp.max(jnp.abs(x), axis=-1, keepdims=True), eps)
  scale = (max_abs / FP8_MAX).astype(jnp.float32)
  inv = (FP8_MAX / max_abs).astype(x.dtype)
  return (x * inv).astype(FP8_DTYPE), scale


def quantize_cols_fp8(
    w: jax.Array, eps: float = 1e-7
) -> tuple[jax.Array, jax.Array]:
  """Quantizes `w` to FP8 with one absmax scale per column (axis 0 reduced).

  Args:
    w: The array to quantize.
    eps: Lower bound on the per-column absmax, so an all-zero column gets a
      finite scale.

  Returns:
    `(wq, scale)` with `scale` of shape `w.shape[1:]` in float32 such that
    `w ~= wq * scale`.
  """
  max_abs = jnp.maximum(jnp.max(jnp.abs(w), axis=0), eps)
  scale = (max_abs / FP8_MAX).astype(jnp.float32)
  inv = (FP8_MAX / max_abs).astype(w.dtype)
  return (w * inv).astype(FP8_DTYPE), scale


def rhs_as_fp8(rhs: jax.Array | QArray) -> tuple[jax.Array, jax.Array]:
  """Returns `(rhs_q, rhs_scale)` with `rhs_scale` of shape `(n,)`."""
  if isinstance(rhs, QArray):
    return rhs.qvalue, rhs.scale.reshape(-1)
  return quantize_cols_fp8(rhs)


def fused_fp8_matmul_reference(
    lhs: jax.Array, rhs: jax.Array | QArray
) -> tuple[jax.Array, Residuals]:
  """XLA reference. Returns `(out, residuals)`."""
  lhs_q, lhs_scale = quantize_rows_fp8(lhs)
  rhs_q, rhs_scale = rhs_as_fp8(rhs)
  acc = jnp.matmul(lhs_q, rhs_q, preferred_element_type=jnp.float32)
  out = (acc * (lhs_scale * rhs_scale)).astype(lhs.dtype)
  return out, (lhs_q, lhs_scale, rhs_q, rhs_scale)


def fused_fp8_matmul_vjp_reference(
    residuals: Residuals, dout: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """XLA reference VJP from the FP8 residuals.

  `dlhs = dout @ rhs^T` and `drhs = lhs^T @ dout`, with `lhs` and `rhs`
  reconstructed from their FP8 values and scales. Both gradients are
  accumulated in float32.

  Args:
    residuals: `(lhs_q, lhs_scale, rhs_q, rhs_scale)` from the forward pass.
    dout: Cotangent of the output, shape `(m, n)`.

  Returns:
    `(dlhs, drhs)` in `dout.dtype`.
  """
  lhs_q, lhs_scale, rhs_q, rhs_scale = residuals
  rhs = rhs_q.astype(jnp.float32) * rhs_scale
  lhs = lhs_q.astype(jnp.float32) * lhs_scale
  dout_f32 = dout.astype(jnp.float32)
  dlhs = jnp.matmul(dout_f32, rhs.T).astype(dout.dtype)
  drhs = jnp.matmul(lhs.T, dout_f32).astype(dout.dtype)
  return dlhs, drhs


def _validate_rhs_qarray(rhs: QArray, k: int, n: int) -> None:
  """Checks that `rhs` is an FP8 weight with one scale per output column."""
  if rhs.qvalue.dtype != FP8_DTYPE:
    raise ValueError(f"rhs must be quantized to {FP8_DTYPE}, got {rhs.qtype}.")
  if rhs.zero_point is not None:
    raise ValueError("rhs must be symmetrically quantized (no zero point).")
  if rhs.scale.shape not in ((1, n), (n,)):
    raise ValueError(
        "rhs must have one scale per output column, i.e. scale shape"
        f" (1, {n}); got {rhs.scale.shape} for a ({k}, {n}) weight."
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class FusedFp8Matmul[C](op.Op[Any, jax.Array, Residuals | None, C, None]):
  """Fused FP8 matmul: `lhs @ rhs` with in-op row quantization of `lhs`."""

  def __post_init__(self):
    # The reference VJP takes no config. A backend with its own backward kernel
    # overrides this with a VJP op carrying that kernel's config.
    object.__setattr__(self, "vjp", FusedFp8MatmulVjp())

  @override
  def bind(
      self,
      lhs: jax.Array | AbstractArray,
      rhs: jax.Array | QArray | AbstractArray,
      *,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Validates and binds the arguments.

    Args:
      lhs: Activations of shape `(m, k)` in `bfloat16` or `float32`. Quantized
        to `float8_e4m3fn` with one scale per row inside the op.
      rhs: Weights of shape `(k, n)`. Either a `qwix.QArray` already quantized
        to `float8_e4m3fn` with one scale per output column (scale shape `(1,
        n)`), or a plain floating-point array which the op quantizes per column
        before the matmul.
      return_residuals: Whether to also return the residuals.

    Returns:
      The bound arguments.
    """
    if lhs.ndim != 2:
      raise ValueError(f"lhs must be 2-D (m, k), got shape {lhs.shape}.")
    if rhs.ndim != 2:
      raise ValueError(f"rhs must be 2-D (k, n), got shape {rhs.shape}.")
    m, k = lhs.shape
    k_rhs, n = rhs.shape
    if k_rhs != k:
      raise ValueError(
          f"Contracting dims differ: lhs is ({m}, {k}), rhs is ({k_rhs}, {n})."
      )
    if lhs.dtype not in (jnp.bfloat16, jnp.float32):
      raise ValueError(f"lhs must be bfloat16 or float32, got {lhs.dtype}.")
    if isinstance(rhs, QArray):
      _validate_rhs_qarray(rhs, k, n)
    elif not jnp.issubdtype(rhs.dtype, jnp.floating):
      raise ValueError(f"rhs must be floating point, got {rhs.dtype}.")
    elif rhs.dtype == FP8_DTYPE:
      raise ValueError(
          "An already-quantized rhs must be passed as a qwix.QArray so its"
          " scale is known."
      )
    return super().bind(lhs, rhs, return_residuals=return_residuals)

  @override
  def _fwd(
      self,
      lhs: jax.Array,
      rhs: jax.Array | QArray,
      *,
      return_residuals: bool,
      config: C,
  ) -> tuple[jax.Array, Residuals | None]:
    del config  # Unused.
    out, residuals = fused_fp8_matmul_reference(lhs, rhs)
    return out, (residuals if return_residuals else None)


@dataclasses.dataclass(frozen=True, kw_only=True)
class FusedFp8MatmulVjp[C](op.Op[Any, Any, None, C, None]):
  """VJP of `FusedFp8Matmul`, as an op so backends can override it."""

  @override
  def _fwd(
      self,
      residuals: Residuals | None,
      out: jax.Array,
      dout: jax.Array,
      lhs: jax.Array,
      rhs: jax.Array | QArray,
      *,
      return_residuals: bool,
      config: C,
  ) -> tuple[tuple[jax.Array, Any], None]:
    """Returns `((dlhs, drhs), None)`.

    `drhs` is zeros when `rhs` is a `QArray`: the gradient of a pre-quantized
    weight is not defined by this op. Pass the unquantized weight to train it.
    """
    del out, config  # Unused.
    if residuals is None:
      _, residuals = fused_fp8_matmul_reference(lhs, rhs)
    dlhs, drhs = fused_fp8_matmul_vjp_reference(residuals, dout)
    dlhs = dlhs.astype(lhs.dtype)
    if isinstance(rhs, QArray):
      drhs = jax.tree.map(jnp.zeros_like, rhs)
    else:
      drhs = drhs.astype(rhs.dtype)
    return (dlhs, drhs), None
