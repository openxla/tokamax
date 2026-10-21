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
"""Triangle multiplication op."""

from collections.abc import Hashable, Sequence
import functools
from typing import Any, Literal, TypedDict, override

import jax
from jax import lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src import precision as precision_lib
from tokamax._src.ops import op

type AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray
type Implementation = Literal["xla", "triton"]
type Residuals = tuple[
    Float[Array, "H N N"],
    Float[Array, "H N N"],
    Float[Array, "H N N"],
]
CanonicalPrecision = precision_lib.CanonicalPrecision


class Grads(TypedDict):
  x: Float[Array, "N N C"]
  projection_in_weights: Float[Array, "C 2 H"]
  gate_in_weights: Float[Array, "C 2 H"]
  projection_out_weights: Float[Array, "H D"]
  gate_out_weights: Float[Array, "C D"]
  layernorm_in_scale: Float[Array, "C"]
  layernorm_in_offset: Float[Array, "C"]
  layernorm_out_scale: Float[Array, "H"]
  layernorm_out_offset: Float[Array, "H"]


def _contract(
    left: jax.Array,
    right: jax.Array,
    triangle_type: Literal["incoming", "outgoing"],
    precision: CanonicalPrecision,
) -> jax.Array:
  """Batched (over H) triangle contraction of two (H, N, N) planes."""
  if triangle_type == "outgoing":
    dims = (((2,), (2,)), ((0,), (0,)))
    return lax.dot_general(left, right, dims, precision=precision)
  dims = (((1,), (1,)), ((0,), (0,)))
  return lax.dot_general(right, left, dims, precision=precision)


def _contract_bwd(
    dt3: jax.Array,
    left: jax.Array,
    right: jax.Array,
    triangle_type: Literal["incoming", "outgoing"],
    precision: CanonicalPrecision,
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


def _sum_01(x: jax.Array) -> jax.Array:
  """Sums `x` of shape (N, N, K) over axes 0 and 1 via 1D reductions."""
  return jnp.sum(jnp.sum(x, axis=0), axis=0)


def _layer_norm_in_fwd(
    x: jax.Array,
    scale: jax.Array,
    offset: jax.Array,
    epsilon: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes LayerNorm along axis -1 of `x` (N, N, C)."""
  acc_dtype = jnp.promote_types(x.dtype, jnp.float32)
  x_f32 = x.astype(acc_dtype)
  mean = jnp.mean(x_f32, axis=-1, keepdims=True)
  y = x_f32 - mean
  var = jnp.mean(jnp.square(y), axis=-1, keepdims=True)
  rstd = (
      jnp.where(var > 0, lax.rsqrt(var), 0.0)
      if epsilon <= 0.0
      else lax.rsqrt(var + epsilon)
  )
  x_hat = y * rstd
  left_act = (
      x_hat * scale.astype(acc_dtype) + offset.astype(acc_dtype)
  ).astype(x.dtype)
  return left_act, mean, rstd


def _to_contract_layout(x: jax.Array) -> jax.Array:
  """Ensures (H, N, N) tensor uses matrix-minor {2, 1, 0} layout."""
  h, n, _ = x.shape
  return lax.optimization_barrier(lax.collapse(x, 0, 2)).reshape(h, n, n)


def _from_contract_layout(x: jax.Array) -> jax.Array:
  """Transitions (H, N, N) contraction output to channel-sublane layout."""
  h, n, _ = x.shape
  return lax.optimization_barrier(lax.collapse(x, 1, 3)).reshape(h, n, n)


def _prologue_fwd(
    left_act: jax.Array,
    mask: jax.Array,
    projection_in_weights: jax.Array,
    gate_in_weights: jax.Array,
    precision: CanonicalPrecision,
) -> tuple[jax.Array, jax.Array]:
  """Computes masked left and right GLU projections of shape (H, N, N)."""
  dtype = left_act.dtype
  acc_dtype = jnp.promote_types(dtype, jnp.float32)
  masked_left_act = jnp.where(mask[:, :, None], left_act, 0)
  w_proj_l = projection_in_weights[:, 0, :]
  w_proj_r = projection_in_weights[:, 1, :]
  w_gate_l = gate_in_weights[:, 0, :]
  w_gate_r = gate_in_weights[:, 1, :]
  dot_dims = (((0,), (2,)), ((), ()))

  proj_l = lax.dot_general(
      w_proj_l, masked_left_act, dot_dims, precision=precision
  )
  gate_l = lax.dot_general(
      w_gate_l, masked_left_act, dot_dims, precision=precision
  )
  left_proj_act = _to_contract_layout(
      (
          proj_l.astype(acc_dtype) * jax.nn.sigmoid(gate_l.astype(acc_dtype))
      ).astype(dtype)
  )

  proj_r = lax.dot_general(
      w_proj_r, masked_left_act, dot_dims, precision=precision
  )
  gate_r = lax.dot_general(
      w_gate_r, masked_left_act, dot_dims, precision=precision
  )
  right_proj_act = _to_contract_layout(
      (
          proj_r.astype(acc_dtype) * jax.nn.sigmoid(gate_r.astype(acc_dtype))
      ).astype(dtype)
  )
  return left_proj_act, right_proj_act


def _act_stats(
    act: jax.Array,
    epsilon: float,
) -> tuple[jax.Array, jax.Array]:
  """Computes two-pass centered mean and rstd of `act` along axis 0."""
  acc_dtype = jnp.promote_types(act.dtype, jnp.float32)
  # Optimization barrier prevents XLA:TPU from fusing the axis-0 reduction into
  # the preceding cubic (H, N, N) contraction, which would shrink its MXU tile.
  act_f32 = lax.optimization_barrier(act).astype(acc_dtype)
  mean = jnp.mean(act_f32, axis=0)
  centered = act_f32 - mean[None, :, :]
  var = jnp.mean(jnp.square(centered), axis=0)
  rstd = (
      jnp.where(var > 0, lax.rsqrt(var), 0.0)
      if epsilon <= 0.0
      else lax.rsqrt(var + epsilon)
  )
  return mean, rstd


def _epilogue_fwd(
    act: jax.Array,
    left_act: jax.Array,
    mean_out: jax.Array,
    rstd_out: jax.Array,
    projection_out_weights: jax.Array,
    gate_out_weights: jax.Array,
    layernorm_out_scale: jax.Array,
    layernorm_out_offset: jax.Array,
    precision: CanonicalPrecision,
) -> jax.Array:
  """Computes folded FP32 LayerNorm_out + projection_out + gate_out."""
  dtype = left_act.dtype
  acc_dtype = jnp.promote_types(dtype, jnp.float32)
  s_out_f32 = layernorm_out_scale.astype(acc_dtype)
  b_out_f32 = layernorm_out_offset.astype(acc_dtype)
  w_out_f32 = projection_out_weights.astype(acc_dtype)
  w_out_scaled = (s_out_f32[:, None] * w_out_f32).astype(
      projection_out_weights.dtype
  )
  w_out_sum = jnp.sum(w_out_scaled.astype(acc_dtype), axis=0)
  b_out_proj = jnp.dot(
      b_out_f32, w_out_f32, precision=jax.lax.Precision.HIGHEST
  )

  raw_proj_out = lax.dot_general(
      act,
      w_out_scaled,
      (((0,), (0,)), ((), ())),
      precision=precision,
      preferred_element_type=acc_dtype,
  )
  proj_out_f32 = (
      raw_proj_out - mean_out[:, :, None] * w_out_sum[None, None, :]
  ) * rstd_out[:, :, None] + b_out_proj[None, None, :]
  gate_values = jnp.dot(left_act, gate_out_weights, precision=precision)
  return (
      proj_out_f32 * jax.nn.sigmoid(gate_values.astype(acc_dtype))
  ).astype(dtype)


def _fwd_core(
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
    triangle_type: Literal["incoming", "outgoing"],
    precision: CanonicalPrecision,
    epsilon: float,
):
  """Computes forward pass and internal residuals."""
  left_act, mean_in, rstd_in = _layer_norm_in_fwd(
      x, layernorm_in_scale, layernorm_in_offset, epsilon
  )
  left_proj_act, right_proj_act = _prologue_fwd(
      left_act,
      mask,
      projection_in_weights,
      gate_in_weights,
      precision,
  )

  act = _contract(left_proj_act, right_proj_act, triangle_type, precision)
  mean_out, rstd_out = _act_stats(act, epsilon)
  out = _epilogue_fwd(
      act,
      left_act,
      mean_out,
      rstd_out,
      projection_out_weights,
      gate_out_weights,
      layernorm_out_scale,
      layernorm_out_offset,
      precision,
  )
  res = (
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
      left_act,
      left_proj_act,
      right_proj_act,
      act,
      mean_in,
      rstd_in,
      mean_out,
      rstd_out,
      out,
  )
  return out, res


def _bwd_core(
    triangle_type: Literal["incoming", "outgoing"],
    precision: CanonicalPrecision,
    res,
    dout: jax.Array,
):
  """Computes backward pass from internal residuals with FP32 accumulations."""
  (
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
      left_act,
      left_proj_act,
      right_proj_act,
      act,
      mean_in,
      rstd_in,
      mean_out,
      rstd_out,
      out,
  ) = res

  (
      mask,
      projection_in_weights,
      gate_in_weights,
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
      left_proj_act,
      right_proj_act,
      act,
      mean_in,
      rstd_in,
      mean_out,
      rstd_out,
  ) = lax.optimization_barrier((
      mask,
      projection_in_weights,
      gate_in_weights,
      projection_out_weights,
      gate_out_weights,
      layernorm_in_scale,
      layernorm_in_offset,
      layernorm_out_scale,
      layernorm_out_offset,
      left_proj_act,
      right_proj_act,
      act,
      mean_in,
      rstd_in,
      mean_out,
      rstd_out,
  ))

  dtype = x.dtype
  acc_dtype = jnp.promote_types(dtype, jnp.float32)

  # 1. Epilogue backward.
  s_out_f32 = layernorm_out_scale.astype(acc_dtype)
  b_out_f32 = layernorm_out_offset.astype(acc_dtype)
  act_b, mean_out_b, rstd_out_b = lax.optimization_barrier(
      (act, mean_out, rstd_out)
  )
  t_hat = (act_b.astype(acc_dtype) - mean_out_b[None, :, :]) * rstd_out_b[
      None, :, :
  ]
  act_out = (
      t_hat * s_out_f32[:, None, None] + b_out_f32[:, None, None]
  ).astype(dtype)
  proj_out = lax.dot_general(
      act_out,
      projection_out_weights,
      (((0,), (0,)), ((), ())),
      precision=precision,
  )

  gate_values = jnp.dot(left_act, gate_out_weights, precision=precision)
  gate_f32 = jax.nn.sigmoid(gate_values.astype(acc_dtype))

  dout_f32 = dout.astype(acc_dtype)
  dout_nz = dout_f32 != 0
  dp_f32 = jnp.where(dout_nz, dout_f32 * gate_f32, 0.0)
  dp = dp_f32.astype(dtype)
  dg_out = jnp.where(
      dout_nz,
      dout_f32 * proj_out.astype(acc_dtype) * gate_f32 * (1.0 - gate_f32),
      0.0,
  ).astype(dtype)

  safe_left_act = jnp.where(jnp.isfinite(left_act), left_act, 0)
  dw_gate_out = lax.dot_general(
      safe_left_act,
      dg_out,
      (((0, 1), (0, 1)), ((), ())),
      precision=precision,
      preferred_element_type=acc_dtype,
  ).astype(gate_out_weights.dtype)

  dw_proj_out = lax.dot_general(
      act_out,
      dp,
      (((1, 2), (0, 1)), ((), ())),
      precision=precision,
      preferred_element_type=acc_dtype,
  ).astype(projection_out_weights.dtype)

  d_act_out = lax.dot_general(
      projection_out_weights,
      dp,
      (((1,), (2,)), ((), ())),
      precision=precision,
  ).astype(acc_dtype)
  dln_out_offset = jnp.sum(d_act_out, axis=(1, 2)).astype(
      layernorm_out_offset.dtype
  )
  dln_out_scale = jnp.sum(d_act_out * t_hat, axis=(1, 2)).astype(
      layernorm_out_scale.dtype
  )
  dx_hat = d_act_out * s_out_f32[:, None, None]
  c1_tilde = jnp.mean(dx_hat, axis=0, keepdims=True)
  c2_tilde = jnp.mean(dx_hat * t_hat, axis=0, keepdims=True)
  d_act = (
      (dx_hat - c1_tilde - t_hat * c2_tilde) * rstd_out_b[None, :, :]
  ).astype(dtype)

  # 2. Contraction backward.
  dt3 = lax.optimization_barrier(d_act)
  dleft, dright = _contract_bwd(
      dt3, left_proj_act, right_proj_act, triangle_type, precision
  )
  dleft = _from_contract_layout(dleft)
  dright = _from_contract_layout(dright)

  # 3. Prologue + LayerNorm_in backward.
  w_proj_l = projection_in_weights[:, 0, :]
  w_proj_r = projection_in_weights[:, 1, :]
  w_gate_l = gate_in_weights[:, 0, :]
  w_gate_r = gate_in_weights[:, 1, :]
  dot_dims_fwd = (((0,), (2,)), ((), ()))
  mask_l = mask[None, :, :]
  mask_r = lax.optimization_barrier(mask_l)

  proj_l = lax.dot_general(
      w_proj_l, safe_left_act, dot_dims_fwd, precision=precision
  )
  gate_l = lax.dot_general(
      w_gate_l, safe_left_act, dot_dims_fwd, precision=precision
  )
  s_l_f32 = jax.nn.sigmoid(gate_l.astype(acc_dtype))
  dleft_m = jnp.where(mask_l, dleft.astype(acc_dtype), 0.0)
  dp_l = (dleft_m * s_l_f32).astype(dtype)
  dg_l = (
      dleft_m * proj_l.astype(acc_dtype) * s_l_f32 * (1.0 - s_l_f32)
  ).astype(dtype)

  proj_r = lax.dot_general(
      w_proj_r, safe_left_act, dot_dims_fwd, precision=precision
  )
  gate_r = lax.dot_general(
      w_gate_r, safe_left_act, dot_dims_fwd, precision=precision
  )
  s_r_f32 = jax.nn.sigmoid(gate_r.astype(acc_dtype))
  dright_m = jnp.where(mask_r, dright.astype(acc_dtype), 0.0)
  dp_r = (dright_m * s_r_f32).astype(dtype)
  dg_r = (
      dright_m * proj_r.astype(acc_dtype) * s_r_f32 * (1.0 - s_r_f32)
  ).astype(dtype)

  dot_dims_dw = (((0, 1), (1, 2)), ((), ()))
  dw_proj_l = lax.dot_general(
      safe_left_act,
      dp_l,
      dot_dims_dw,
      precision=precision,
      preferred_element_type=acc_dtype,
  )
  dw_proj_r = lax.dot_general(
      safe_left_act,
      dp_r,
      dot_dims_dw,
      precision=precision,
      preferred_element_type=acc_dtype,
  )
  dw_proj_in = jnp.stack([dw_proj_l, dw_proj_r], axis=1).astype(
      projection_in_weights.dtype
  )

  dw_gate_l = lax.dot_general(
      safe_left_act,
      dg_l,
      dot_dims_dw,
      precision=precision,
      preferred_element_type=acc_dtype,
  )
  dw_gate_r = lax.dot_general(
      safe_left_act,
      dg_r,
      dot_dims_dw,
      precision=precision,
      preferred_element_type=acc_dtype,
  )
  dw_gate_in = jnp.stack([dw_gate_l, dw_gate_r], axis=1).astype(
      gate_in_weights.dtype
  )

  dot_dims_dx = (((0,), (1,)), ((), ()))
  d_left_act = (
      lax.dot_general(
          dg_out,
          gate_out_weights,
          (((2,), (1,)), ((), ())),
          precision=precision,
          preferred_element_type=acc_dtype,
      )
      + lax.dot_general(
          dp_l,
          w_proj_l,
          dot_dims_dx,
          precision=precision,
          preferred_element_type=acc_dtype,
      )
      + lax.dot_general(
          dp_r,
          w_proj_r,
          dot_dims_dx,
          precision=precision,
          preferred_element_type=acc_dtype,
      )
      + lax.dot_general(
          dg_l,
          w_gate_l,
          dot_dims_dx,
          precision=precision,
          preferred_element_type=acc_dtype,
      )
      + lax.dot_general(
          dg_r,
          w_gate_r,
          dot_dims_dx,
          precision=precision,
          preferred_element_type=acc_dtype,
      )
  )

  dy_in = d_left_act
  dln_in_offset = _sum_01(dy_in).astype(layernorm_in_offset.dtype)
  mean_in_b, rstd_in_b, s_in_b = lax.optimization_barrier(
      (mean_in, rstd_in, layernorm_in_scale)
  )
  x_hat = jnp.where(
      dy_in != 0, (x.astype(acc_dtype) - mean_in_b) * rstd_in_b, 0.0
  )
  dln_in_scale = _sum_01(dy_in * x_hat).astype(layernorm_in_scale.dtype)

  s_in_f32 = s_in_b.astype(acc_dtype)
  dx_hat = dy_in * s_in_f32
  c1_in = jnp.mean(dx_hat, axis=-1, keepdims=True)
  c2_in = jnp.mean(dx_hat * x_hat, axis=-1, keepdims=True)
  raw_dx = dx_hat - c1_in - x_hat * c2_in
  dx = jnp.where(raw_dx != 0, raw_dx * rstd_in_b, 0.0).astype(dtype)

  return (
      dx,
      jnp.zeros_like(mask),
      dw_proj_in,
      dw_gate_in,
      dw_proj_out,
      dw_gate_out,
      dln_in_scale,
      dln_in_offset,
      dln_out_scale,
      dln_out_offset,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(10, 11, 12))
def _triangle_multiplication_xla(
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
    triangle_type: Literal["incoming", "outgoing"],
    precision: CanonicalPrecision,
    epsilon: float,
) -> jax.Array:
  """XLA triangle multiplication with custom VJP."""
  out, _ = _fwd_core(
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
  return out


def _triangle_multiplication_xla_fwd(
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
    triangle_type: Literal["incoming", "outgoing"],
    precision: CanonicalPrecision,
    epsilon: float,
):
  """Forward rule for `_triangle_multiplication_xla`."""
  return _fwd_core(
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


def _triangle_multiplication_xla_bwd(
    triangle_type: Literal["incoming", "outgoing"],
    precision: CanonicalPrecision,
    epsilon: float,
    res,
    dout: jax.Array,
):
  """Backward rule for `_triangle_multiplication_xla`."""
  del epsilon  # Unused (already captured in rstd_in and rstd_out).
  return _bwd_core(triangle_type, precision, res, dout)


_triangle_multiplication_xla.defvjp(
    _triangle_multiplication_xla_fwd, _triangle_multiplication_xla_bwd
)


class TriangleMultiplication[C, K](op.Op[Any, jax.Array, Residuals, C, K]):
  """Triangle multiplicative update."""

  @override
  def bind(
      self,
      x: Float[Array | AbstractArray, "N N C"],
      mask: Bool[Array | AbstractArray, "N N"],
      projection_in_weights: Float[Array | AbstractArray, "C 2 H"],
      gate_in_weights: Float[Array | AbstractArray, "C 2 H"],
      projection_out_weights: Float[Array | AbstractArray, "H D"],
      gate_out_weights: Float[Array | AbstractArray, "C D"],
      layernorm_in_scale: Float[Array | AbstractArray, "C"],
      layernorm_in_offset: Float[Array | AbstractArray, "C"],
      layernorm_out_scale: Float[Array | AbstractArray, "H"],
      layernorm_out_offset: Float[Array | AbstractArray, "H"],
      triangle_type: Literal["incoming", "outgoing"],
      *,
      precision: jax.lax.PrecisionLike = None,
      epsilon: float = 1e-6,
      return_residuals: bool = False,
      implementation: Implementation | Sequence[Implementation] | None = None,
  ) -> op.BoundArguments:
    """Binds the arguments for the triangle multiplication function."""
    return super().bind(
        x=x,
        mask=mask,
        projection_in_weights=projection_in_weights,
        gate_in_weights=gate_in_weights,
        projection_out_weights=projection_out_weights,
        gate_out_weights=gate_out_weights,
        layernorm_in_scale=layernorm_in_scale,
        layernorm_in_offset=layernorm_in_offset,
        layernorm_out_scale=layernorm_out_scale,
        layernorm_out_offset=layernorm_out_offset,
        triangle_type=triangle_type,
        precision=precision_lib.canonicalize_precision(precision),
        epsilon=epsilon,
        return_residuals=return_residuals,
        implementation=implementation,
    )

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      x: Float[Array, "N N C"],
      mask: Bool[Array, "N N"],
      projection_in_weights: Float[Array, "C 2 H"],
      gate_in_weights: Float[Array, "C 2 H"],
      projection_out_weights: Float[Array, "H D"],
      gate_out_weights: Float[Array, "C D"],
      layernorm_in_scale: Float[Array, "C"],
      layernorm_in_offset: Float[Array, "C"],
      layernorm_out_scale: Float[Array, "H"],
      layernorm_out_offset: Float[Array, "H"],
      triangle_type: Literal["incoming", "outgoing"],
      *,
      precision: CanonicalPrecision,
      epsilon: float,
      return_residuals: bool,
      implementation: Implementation | Sequence[Implementation] | None,
      config: C,
  ) -> tuple[Float[Array, "N N D"], Residuals | None]:
    """Triangle multiplicative update."""
    del config, implementation  # Unused.

    if not return_residuals:
      out = _triangle_multiplication_xla(
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
      return out, None

    out, res = _fwd_core(
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
    residuals = (res[11], res[12], res[13])
    return out, residuals


class TriangleMultiplicationVjp[C, K: Hashable](
    op.Op[Any, Grads, None, C, K]
):
  """Triangle multiplicative update VJP."""

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      residuals: Residuals,
      out: Float[Array, "N N D"],
      dout: Float[Array, "N N D"],
      x: Float[Array, "N N C"],
      mask: Bool[Array, "N N"],
      projection_in_weights: Float[Array, "C 2 H"],
      gate_in_weights: Float[Array, "C 2 H"],
      projection_out_weights: Float[Array, "H D"],
      gate_out_weights: Float[Array, "C D"],
      layernorm_in_scale: Float[Array, "C"],
      layernorm_in_offset: Float[Array, "C"],
      layernorm_out_scale: Float[Array, "H"],
      layernorm_out_offset: Float[Array, "H"],
      triangle_type: Literal["incoming", "outgoing"],
      *,
      precision: CanonicalPrecision,
      epsilon: float,
      return_residuals: bool,
      implementation: Implementation | Sequence[Implementation] | None = None,
      config: C,
  ) -> tuple[Grads, None]:
    """Computes triangle multiplication VJP."""
    del config, implementation  # Unused.

    if return_residuals:
      raise NotImplementedError("`return_residuals` not supported.")

    left_proj_act, right_proj_act, act = residuals
    left_act, mean_in, rstd_in = _layer_norm_in_fwd(
        x, layernorm_in_scale, layernorm_in_offset, epsilon
    )
    mean_out, rstd_out = _act_stats(act, epsilon)

    res = (
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
        left_act,
        left_proj_act,
        right_proj_act,
        act,
        mean_in,
        rstd_in,
        mean_out,
        rstd_out,
        out,
    )
    (
        dx,
        _,
        dw_proj_in,
        dw_gate_in,
        dw_proj_out,
        dw_gate_out,
        dln_in_scale,
        dln_in_offset,
        dln_out_scale,
        dln_out_offset,
    ) = _bwd_core(triangle_type, precision, res, dout)
    return (
        Grads(
            x=dx,
            projection_in_weights=dw_proj_in,
            gate_in_weights=dw_gate_in,
            projection_out_weights=dw_proj_out,
            gate_out_weights=dw_gate_out,
            layernorm_in_scale=dln_in_scale,
            layernorm_in_offset=dln_in_offset,
            layernorm_out_scale=dln_out_scale,
            layernorm_out_offset=dln_out_offset,
        ),
        None,
    )
