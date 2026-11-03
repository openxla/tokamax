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
"""Tests for Pallas-Mosaic-TPU Triangle Multiplication."""

from collections.abc import Sequence
import contextlib
import functools
import math
import sys
from typing import Any, Literal, cast

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import hlo_utils
from tokamax._src import mosaic_tpu
from tokamax._src.ops.triangle_multiplication import api
from tokamax._src.ops.triangle_multiplication import base
from tokamax._src.ops.triangle_multiplication import pallas_mosaic_tpu as mtpu_tri_mul
from tokamax._src.ops.triangle_multiplication import pallas_mosaic_tpu_common as common
from tokamax._src.ops.triangle_multiplication import pallas_mosaic_tpu_vjp as mtpu_tri_mul_vjp

_GRAD_KEYS = (
    "x",
    "projection_in_weights",
    "gate_in_weights",
    "projection_out_weights",
    "gate_out_weights",
    "layernorm_in_scale",
    "layernorm_in_offset",
    "layernorm_out_scale",
    "layernorm_out_offset",
)
_DEFAULT_PREC = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)


@contextlib.contextmanager
def _disable_jaxtyping():
  jt = sys.modules.get("jaxtyping")
  if jt is None:
    yield
    return
  prev = getattr(jt.config, "jaxtyping_disable", False)
  jt.config.update("jaxtyping_disable", True)
  try:
    yield
  finally:
    jt.config.update("jaxtyping_disable", prev)


def _zeros_params(
    n: int,
    c: int,
    h: int,
    d: int,
    dtype: Any = jnp.bfloat16,
) -> dict[str, Any]:
  return dict(
      x=jnp.zeros((n, n, c), dtype=dtype),
      mask=jnp.zeros((n, n), dtype=jnp.bool_),
      projection_in_weights=jnp.zeros((c, 2, h), dtype=dtype),
      gate_in_weights=jnp.zeros((c, 2, h), dtype=dtype),
      projection_out_weights=jnp.zeros((h, d), dtype=dtype),
      gate_out_weights=jnp.zeros((c, d), dtype=dtype),
      layernorm_in_scale=jnp.zeros((c,), dtype=dtype),
      layernorm_in_offset=jnp.zeros((c,), dtype=dtype),
      layernorm_out_scale=jnp.zeros((h,), dtype=dtype),
      layernorm_out_offset=jnp.zeros((h,), dtype=dtype),
  )


def _get_params(
    n: int,
    c: int,
    h: int,
    d: int,
    dtype: jax.typing.DTypeLike,
    seed: int = 0,
) -> dict[str, Any]:
  dt = jnp.dtype(dtype)
  rngs = np.random.default_rng(seed).spawn(10)
  return jax.device_put(
      dict(
          gate_in_weights=rngs[0]
          .standard_normal((c, 2, h), dtype=np.float32)
          .astype(dt),
          gate_out_weights=rngs[1]
          .standard_normal((c, d), dtype=np.float32)
          .astype(dt),
          layernorm_in_offset=rngs[2]
          .standard_normal((c,), dtype=np.float32)
          .astype(dt),
          layernorm_in_scale=rngs[3]
          .standard_normal((c,), dtype=np.float32)
          .astype(dt),
          layernorm_out_offset=rngs[4]
          .standard_normal((h,), dtype=np.float32)
          .astype(dt),
          layernorm_out_scale=rngs[5]
          .standard_normal((h,), dtype=np.float32)
          .astype(dt),
          mask=rngs[6].binomial(n=1, p=0.5, size=(n, n)).astype(np.bool_),
          projection_in_weights=rngs[7]
          .standard_normal((c, 2, h), dtype=np.float32)
          .astype(dt),
          projection_out_weights=rngs[8]
          .standard_normal((h, d), dtype=np.float32)
          .astype(dt),
          x=rngs[9].standard_normal((n, n, c), dtype=np.float32).astype(dt),
      )
  )


def _make_random_array(
    shape: tuple[int, ...], dtype: jax.typing.DTypeLike, seed: int
) -> jax.Array:
  dt = jnp.dtype(dtype)
  rng = np.random.default_rng(seed).spawn(1)[0]
  return jax.device_put(rng.standard_normal(shape, dtype=np.float32).astype(dt))


def _make_seeded_params(
    seed: int,
    n: int,
    c: int,
    h: int,
    d: int,
    *,
    dtype: Any = jnp.bfloat16,
    scale_mode: str = "unit",
    mask_prob: float = 0.5,
) -> dict[str, Any]:
  dt = jnp.dtype(dtype)
  if scale_mode == "unit" and mask_prob == 0.5:
    return _get_params(n, c, h, d, dt, seed=seed)
  rng = np.random.default_rng(seed)
  in_s = np.float32(1.0 if scale_mode == "unit" else 1.0 / math.sqrt(max(c, 1)))
  h_s = np.float32(1.0 if scale_mode == "unit" else 1.0 / math.sqrt(max(h, 1)))
  x = rng.standard_normal((n, n, c), dtype=np.float32).astype(dt)
  mask = rng.random((n, n), dtype=np.float32) < mask_prob
  w_pi = (rng.standard_normal((c, 2, h), dtype=np.float32) * in_s).astype(dt)
  w_gi = (rng.standard_normal((c, 2, h), dtype=np.float32) * in_s).astype(dt)
  if scale_mode == "af_zero_init":
    w_po = np.zeros((h, d), dtype=dt)
    w_go = np.zeros((c, d), dtype=dt)
  else:
    w_po = (rng.standard_normal((h, d), dtype=np.float32) * h_s).astype(dt)
    w_go = (rng.standard_normal((c, d), dtype=np.float32) * in_s).astype(dt)
  if scale_mode == "unit":
    ln_is = rng.standard_normal((c,), dtype=np.float32).astype(dt)
    ln_ib = rng.standard_normal((c,), dtype=np.float32).astype(dt)
    ln_os = rng.standard_normal((h,), dtype=np.float32).astype(dt)
    ln_ob = rng.standard_normal((h,), dtype=np.float32).astype(dt)
  else:
    ln_is = (
        np.float32(1.0)
        + np.float32(0.1) * rng.standard_normal((c,), dtype=np.float32)
    ).astype(dt)
    ln_ib = (
        np.float32(0.1) * rng.standard_normal((c,), dtype=np.float32)
    ).astype(dt)
    ln_os = (
        np.float32(1.0)
        + np.float32(0.1) * rng.standard_normal((h,), dtype=np.float32)
    ).astype(dt)
    ln_ob = (
        np.float32(0.1) * rng.standard_normal((h,), dtype=np.float32)
    ).astype(dt)
  return jax.device_put(
      dict(
          x=x,
          mask=mask,
          projection_in_weights=w_pi,
          gate_in_weights=w_gi,
          projection_out_weights=w_po,
          gate_out_weights=w_go,
          layernorm_in_scale=ln_is,
          layernorm_in_offset=ln_ib,
          layernorm_out_scale=ln_os,
          layernorm_out_offset=ln_ob,
      )
  )


def _make_op(
    fwd_cfg: dict[str, int] | None = None,
    vjp_cfg: dict[str, int] | None = None,
) -> mtpu_tri_mul.PallasMosaicTpuTriangleMultiplication:
  cfg = mtpu_tri_mul.Config(**fwd_cfg) if fwd_cfg is not None else None
  vjp = None
  if vjp_cfg is not None:
    vjp = mtpu_tri_mul_vjp.PallasMosaicTpuTriangleMultiplicationVjp(
        config=mtpu_tri_mul_vjp.Config(**vjp_cfg)
    )
  return mtpu_tri_mul.PallasMosaicTpuTriangleMultiplication(config=cfg, vjp=vjp)


_DEFAULT_OP = mtpu_tri_mul.PallasMosaicTpuTriangleMultiplication()
_DEFAULT_REF_OP = base.TriangleMultiplication()
_FTZ_MIN_NORMAL_F32 = 1.1754944e-38


def _f6_safe_ln(
    x_in: jax.Array, scale: jax.Array, offset: jax.Array, eps_f32: float
) -> jax.Array:
  y = x_in.astype(jnp.float32)
  mean = jnp.mean(y, axis=-1, keepdims=True)
  centered = y - mean
  var = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
  var_eps = var + eps_f32
  active = var_eps > 0.0
  centered_ad = jnp.where(active, centered, 0.0)
  rstd = jnp.where(active, jax.lax.rsqrt(jnp.where(active, var_eps, 1.0)), 0.0)
  out = centered_ad * rstd * scale.astype(jnp.float32) + offset.astype(
      jnp.float32
  )
  return out.astype(x_in.dtype)


def _f6_safe_triangle_mult_with_res(
    params: dict[str, Any],
    triangle_type: Literal["incoming", "outgoing"],
    *,
    precision: Any = None,
    epsilon: float = 1e-6,
):
  eps_f32 = 0.0 if epsilon < _FTZ_MIN_NORMAL_F32 else float(epsilon)
  c = params["gate_in_weights"].shape[0]
  h = params["layernorm_out_scale"].shape[0]
  left_act = _f6_safe_ln(
      params["x"],
      params["layernorm_in_scale"],
      params["layernorm_in_offset"],
      eps_f32,
  )
  w_g = params["gate_in_weights"].reshape(c, 2 * h)
  w_p = params["projection_in_weights"].reshape(c, 2 * h)
  gate = jax.nn.sigmoid(
      jnp.matmul(left_act, w_g, precision=precision).astype(jnp.float32)
  )
  proj = jnp.matmul(left_act, w_p, precision=precision).astype(jnp.float32)
  proj_act = (gate * proj).astype(left_act.dtype)
  proj_act = proj_act.reshape(proj_act.shape[0], proj_act.shape[1], 2, h)
  proj_act = params["mask"] * jnp.transpose(proj_act, (2, 3, 0, 1))
  left_proj, right_proj = [
      jnp.squeeze(t, axis=0) for t in jnp.split(proj_act, 2, axis=0)
  ]
  eq = "cik,cjk->ijc" if triangle_type == "outgoing" else "ckj,cki->ijc"
  act_raw = jnp.einsum(eq, left_proj, right_proj, precision=precision)
  act = _f6_safe_ln(
      act_raw,
      params["layernorm_out_scale"],
      params["layernorm_out_offset"],
      eps_f32,
  )
  act = jnp.dot(act, params["projection_out_weights"], precision=precision)
  gate_values = jnp.dot(
      left_act, params["gate_out_weights"], precision=precision
  )
  out = (act * jax.nn.sigmoid(gate_values)).astype(left_act.dtype)
  return out, (left_proj, right_proj, act_raw)


@functools.partial(jax.jit, static_argnames=("triangle_type", "epsilon"))
def _jit_op_fwd(
    params: dict[str, Any],
    triangle_type: Literal["incoming", "outgoing"],
    epsilon: float = 1e-6,
) -> jax.Array:
  return _DEFAULT_OP(**params, triangle_type=triangle_type, epsilon=epsilon)


@functools.partial(jax.jit, static_argnames=("triangle_type", "epsilon"))
def _jit_op_fwd_with_res(
    params: dict[str, Any],
    triangle_type: Literal["incoming", "outgoing"],
    epsilon: float = 1e-6,
):
  return _DEFAULT_OP(
      **params,
      triangle_type=triangle_type,
      epsilon=epsilon,
      return_residuals=True,
  )


@functools.partial(jax.jit, static_argnames=("triangle_type", "cfg"))
def _jit_op_cfg_fwd(
    params: dict[str, Any],
    triangle_type: Literal["incoming", "outgoing"],
    cfg: common.Config,
) -> jax.Array:
  op = mtpu_tri_mul.PallasMosaicTpuTriangleMultiplication(config=cfg)
  return op(**params, triangle_type=triangle_type)


@functools.partial(
    jax.jit,
    static_argnames=("triangle_type", "epsilon", "fwd_cfg", "vjp_cfg"),
)
def _jit_op_fwd_and_vjp(
    diff_params: dict[str, Any],
    mask: jax.Array,
    dout: jax.Array,
    triangle_type: Literal["incoming", "outgoing"],
    epsilon: float = 1e-6,
    fwd_cfg: common.Config | None = None,
    vjp_cfg: common.Config | None = None,
):
  vjp_op = (
      mtpu_tri_mul_vjp.PallasMosaicTpuTriangleMultiplicationVjp(config=vjp_cfg)
      if vjp_cfg is not None
      else None
  )
  op = mtpu_tri_mul.PallasMosaicTpuTriangleMultiplication(
      config=fwd_cfg, vjp=vjp_op
  )
  out, vjp_fn = jax.vjp(
      lambda q: op(
          **q, mask=mask, triangle_type=triangle_type, epsilon=epsilon
      ),
      diff_params,
  )
  (grads,) = vjp_fn(dout)
  return out, grads


def _should_use_cpu_ref(
    x: jax.Array, max_n: int = 512
) -> tuple[bool, jax.Device | None]:
  """Uses CPU XLA for N % 8 == 0, N <= max_n references on TPU v5p."""
  if (
      isinstance(x, jax.core.Tracer)
      or x.shape[0] > max_n
      or x.shape[0] % 8 != 0
  ):
    return False, None
  dev = next(iter(x.devices()), None)
  if dev is None or dev.platform != "tpu":
    return False, None
  gen, _, num_cores, _, _ = common.get_tpu_info(x)
  if gen == 5 and num_cores > 1:
    return True, dev
  return False, None


def _run_ref_fn(
    jit_fn,
    x_sample: jax.Array,
    *args,
    _max_cpu_n: int = 512,
    **kwargs,
):
  use_cpu, orig_dev = _should_use_cpu_ref(x_sample, max_n=_max_cpu_n)
  if not use_cpu:
    return jit_fn(*args, **kwargs)
  cpu_dev = jax.local_devices(backend="cpu")[0]
  args_cpu = jax.device_put(args, cpu_dev)
  with jax.default_device(cpu_dev):
    res_cpu = jit_fn(*args_cpu, **kwargs)
  return jax.device_put(res_cpu, orig_dev)


@functools.partial(jax.jit, static_argnames=("triangle_type", "epsilon"))
def _jit_ref_fwd_and_vjp_fn(
    diff_params: dict[str, Any],
    mask: jax.Array,
    dout: jax.Array,
    triangle_type: Literal["incoming", "outgoing"],
    epsilon: float = 1e-6,
):
  def primal_fn(q):
    if epsilon < _FTZ_MIN_NORMAL_F32:
      out, _ = _f6_safe_triangle_mult_with_res(
          dict(q, mask=mask), triangle_type, epsilon=epsilon
      )
      return out
    return _DEFAULT_REF_OP(
        **q,
        mask=mask,
        triangle_type=triangle_type,
        epsilon=epsilon,
        implementation="xla",
    )

  out, vjp_fn = jax.vjp(primal_fn, diff_params)
  (grads,) = vjp_fn(dout)
  return out, grads


def _jit_ref_fwd_and_vjp(
    diff_params: dict[str, Any],
    mask: jax.Array,
    dout: jax.Array,
    triangle_type: Literal["incoming", "outgoing"],
    epsilon: float = 1e-6,
):
  return _run_ref_fn(
      _jit_ref_fwd_and_vjp_fn,
      diff_params["x"],
      diff_params,
      mask,
      dout,
      triangle_type=triangle_type,
      epsilon=epsilon,
  )


@functools.partial(jax.jit, static_argnames=("triangle_type", "scale_by_n"))
def _jit_op_loss_grad(
    diff_params: dict[str, Any],
    mask: jax.Array,
    dout: jax.Array,
    triangle_type: Literal["incoming", "outgoing"],
    scale_by_n: bool = True,
):
  n = diff_params["x"].shape[0]
  denom = float(n) if scale_by_n else 1.0

  def loss_fn(p):
    out = _DEFAULT_OP(**p, mask=mask, triangle_type=triangle_type)
    return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32)) / denom

  return jax.grad(loss_fn)(diff_params)


@functools.partial(jax.jit, static_argnames=("triangle_type", "scale_by_n"))
def _jit_ref_loss_grad_fn(
    diff_params: dict[str, Any],
    mask: jax.Array,
    dout: jax.Array,
    triangle_type: Literal["incoming", "outgoing"],
    scale_by_n: bool = True,
):
  n = diff_params["x"].shape[0]
  denom = float(n) if scale_by_n else 1.0

  def loss_fn(p):
    out = _DEFAULT_REF_OP(
        **p,
        mask=mask,
        triangle_type=triangle_type,
        implementation="xla",
    )
    return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32)) / denom

  return jax.grad(loss_fn)(diff_params)


def _jit_ref_loss_grad(
    diff_params: dict[str, Any],
    mask: jax.Array,
    dout: jax.Array,
    triangle_type: Literal["incoming", "outgoing"],
    scale_by_n: bool = True,
):
  return _run_ref_fn(
      _jit_ref_loss_grad_fn,
      diff_params["x"],
      diff_params,
      mask,
      dout,
      triangle_type=triangle_type,
      scale_by_n=scale_by_n,
  )


@functools.partial(jax.jit, static_argnames=("triangle_type", "vjp_cfg"))
def _jit_standalone_vjp(
    residuals: Any,
    out: jax.Array,
    dout: jax.Array,
    params: dict[str, Any],
    triangle_type: Literal["incoming", "outgoing"],
    vjp_cfg: common.Config | None = None,
):
  vjp_op = mtpu_tri_mul_vjp.PallasMosaicTpuTriangleMultiplicationVjp(
      config=vjp_cfg
  )
  return vjp_op(
      residuals,
      out,
      dout,
      **params,
      triangle_type=triangle_type,
      precision=_DEFAULT_PREC,
      epsilon=1e-6,
  )


@functools.partial(jax.jit, static_argnames=("triangle_type",))
def _jit_api_fallback_fwd(
    params: dict[str, Any],
    triangle_type: Literal["incoming", "outgoing"],
) -> jax.Array:
  return api.triangle_multiplication(
      **params, triangle_type=triangle_type, implementation=None
  )


@functools.partial(jax.jit, static_argnames=("triangle_type", "epsilon"))
def _jit_ref_fwd_with_res_fn(
    params: dict[str, Any],
    triangle_type: Literal["incoming", "outgoing"],
    epsilon: float = 1e-6,
):
  if epsilon < _FTZ_MIN_NORMAL_F32:
    return _f6_safe_triangle_mult_with_res(
        params, triangle_type, epsilon=epsilon
    )
  return _DEFAULT_REF_OP(
      **params,
      triangle_type=triangle_type,
      epsilon=epsilon,
      return_residuals=True,
      implementation="xla",
  )


def _jit_ref_fwd_with_res(
    params: dict[str, Any],
    triangle_type: Literal["incoming", "outgoing"],
    epsilon: float = 1e-6,
):
  return _run_ref_fn(
      _jit_ref_fwd_with_res_fn,
      params["x"],
      params,
      _max_cpu_n=127,
      triangle_type=triangle_type,
      epsilon=epsilon,
  )


@functools.partial(jax.jit, static_argnames=("triangle_type", "epsilon"))
def _jit_ref_fp32_highest_fn(
    params: dict[str, Any],
    triangle_type: Literal["incoming", "outgoing"],
    epsilon: float = 1e-6,
) -> jax.Array:
  p_f32: dict[str, Any] = {
      k: v if k == "mask" else v.astype(jnp.float32) for k, v in params.items()
  }
  if epsilon < _FTZ_MIN_NORMAL_F32:
    out_f32, _ = _f6_safe_triangle_mult_with_res(
        p_f32,
        triangle_type,
        precision=jax.lax.Precision.HIGHEST,
        epsilon=epsilon,
    )
    return out_f32.astype(params["x"].dtype)
  return _DEFAULT_REF_OP(
      **p_f32,
      triangle_type=triangle_type,
      precision=jax.lax.Precision.HIGHEST,
      epsilon=epsilon,
      implementation="xla",
  ).astype(params["x"].dtype)


def _jit_ref_fp32_highest(
    params: dict[str, Any],
    triangle_type: Literal["incoming", "outgoing"],
    epsilon: float = 1e-6,
) -> jax.Array:
  return _run_ref_fn(
      _jit_ref_fp32_highest_fn,
      params["x"],
      params,
      _max_cpu_n=512,
      triangle_type=triangle_type,
      epsilon=epsilon,
  )


@functools.partial(
    jax.jit,
    static_argnames=(
        "fwd_cfg",
        "vjp_cfg",
        "triangle_type",
        "epsilon",
        "use_api",
        "api_impl",
        "has_valid_mask",
    ),
)
def _jit_tpu_val_and_grad(
    diff_params: dict[str, Any],
    mask: jax.Array,
    target: jax.Array,
    valid_mask: jax.Array,
    *,
    fwd_cfg: common.Config | None,
    vjp_cfg: common.Config | None,
    triangle_type: Literal["incoming", "outgoing"],
    epsilon: float,
    use_api: bool,
    api_impl: Any,
    has_valid_mask: bool,
):
  n = diff_params["x"].shape[0]
  vjp_op = (
      mtpu_tri_mul_vjp.PallasMosaicTpuTriangleMultiplicationVjp(config=vjp_cfg)
      if vjp_cfg is not None
      else None
  )
  tpu_op = mtpu_tri_mul.PallasMosaicTpuTriangleMultiplication(
      config=fwd_cfg, vjp=vjp_op
  )

  def loss_fn(dp):
    if use_api:
      out = api.triangle_multiplication(
          **dp,
          mask=mask,
          triangle_type=triangle_type,
          epsilon=epsilon,
          implementation=api_impl,
      )
    else:
      out = tpu_op(
          **dp,
          mask=mask,
          triangle_type=triangle_type,
          epsilon=epsilon,
      )
    out_eff = (
        jnp.where(valid_mask[:, :, None], out, 0.0) if has_valid_mask else out
    )
    loss = jnp.sum(
        out_eff.astype(jnp.float32) * target.astype(jnp.float32)
    ) / float(max(n, 1))
    return loss, out

  return jax.value_and_grad(loss_fn, has_aux=True)(diff_params)


@functools.partial(
    jax.jit,
    static_argnames=(
        "triangle_type",
        "epsilon",
        "has_valid_mask",
        "f32_gemm",
    ),
)
def _jit_xla_val_and_grad_fn(
    diff_params: dict[str, Any],
    mask: jax.Array,
    target: jax.Array,
    valid_mask: jax.Array,
    *,
    triangle_type: Literal["incoming", "outgoing"],
    epsilon: float,
    has_valid_mask: bool,
    f32_gemm: bool = False,
):
  n = diff_params["x"].shape[0]
  prec = jax.lax.Precision.HIGHEST if f32_gemm else None

  def loss_fn(dp):
    if epsilon < _FTZ_MIN_NORMAL_F32:
      out, _ = _f6_safe_triangle_mult_with_res(
          dict(dp, mask=mask), triangle_type, precision=prec, epsilon=epsilon
      )
    else:
      out = _DEFAULT_REF_OP(
          **dp,
          mask=mask,
          triangle_type=triangle_type,
          precision=prec,
          epsilon=epsilon,
          implementation="xla",
      )
    out_eff = (
        jnp.where(valid_mask[:, :, None], out, 0.0) if has_valid_mask else out
    )
    loss = jnp.sum(
        out_eff.astype(jnp.float32) * target.astype(jnp.float32)
    ) / float(max(n, 1))
    return loss, out

  return jax.value_and_grad(loss_fn, has_aux=True)(diff_params)


def _jit_xla_val_and_grad(
    diff_params: dict[str, Any],
    mask: jax.Array,
    target: jax.Array,
    valid_mask: jax.Array,
    *,
    triangle_type: Literal["incoming", "outgoing"],
    epsilon: float,
    has_valid_mask: bool,
):
  use_cpu, orig_dev = _should_use_cpu_ref(diff_params["x"], max_n=512)
  if not use_cpu:
    return _jit_xla_val_and_grad_fn(
        diff_params,
        mask,
        target,
        valid_mask,
        triangle_type=triangle_type,
        epsilon=epsilon,
        has_valid_mask=has_valid_mask,
    )
  cpu_dev = jax.local_devices(backend="cpu")[0]
  f32_gemm = diff_params["x"].shape[0] >= 512
  dp_c, mask_c, target_c, vm_c = jax.device_put(
      (diff_params, mask, target, valid_mask), cpu_dev
  )
  with jax.default_device(cpu_dev):
    res_cpu = _jit_xla_val_and_grad_fn(
        dp_c,
        mask_c,
        target_c,
        vm_c,
        triangle_type=triangle_type,
        epsilon=epsilon,
        has_valid_mask=has_valid_mask,
        f32_gemm=f32_gemm,
    )
  return jax.device_put(res_cpu, orig_dev)


def _pairformer_block(
    z: jax.Array,
    mask: jax.Array,
    w_out: dict[str, Any],
    w_in: dict[str, Any],
    *,
    implementation: Any,
    epsilon: float = 1e-6,
) -> jax.Array:
  z = z + api.triangle_multiplication(
      z,
      mask,
      **w_out,
      triangle_type="outgoing",
      epsilon=epsilon,
      implementation=implementation,
  )
  z = z + api.triangle_multiplication(
      z,
      mask,
      **w_in,
      triangle_type="incoming",
      epsilon=epsilon,
      implementation=implementation,
  )
  return z


@jax.jit
def _jit_mosaic_pairformer_fwd(
    z: jax.Array,
    mask: jax.Array,
    w_out: dict[str, Any],
    w_in: dict[str, Any],
) -> jax.Array:
  return _pairformer_block(z, mask, w_out, w_in, implementation="mosaic")


@functools.partial(jax.jit, static_argnames=("impl",))
def _jit_cropped_pairformer_vg(
    z_in: jax.Array,
    w_out: dict[str, Any],
    w_in: dict[str, Any],
    mask: jax.Array,
    target: jax.Array,
    inv_denom: jax.Array,
    *,
    impl: Any,
):
  def loss_and_z2(za, woa, wia):
    z2 = _pairformer_block(za, mask, woa, wia, implementation=impl)
    z2_v = jnp.where(mask[:, :, None], z2, 0.0)
    loss = (
        jnp.sum(z2_v.astype(jnp.float32) * target.astype(jnp.float32))
        * inv_denom
    )
    return loss, z2_v

  return jax.value_and_grad(loss_and_z2, argnums=(0, 1, 2), has_aux=True)(
      z_in, w_out, w_in
  )


@jax.jit
def _jit_poison_vmem_smem(fill_vec_u32: jax.Array) -> jax.Array:
  pl = common.pl
  pltpu = common.pltpu
  tpu_info = pltpu.get_tpu_info()
  num_cores = tpu_info.num_cores
  lanes = mosaic_tpu.LANES
  vmem_words = (4 * 1024 * 1024) // 4
  smem_words = min(4096, max(256, tpu_info.smem_capacity_bytes // 4))

  def _poison_kernel(in_ref, out_ref, v_scratch, s_scratch):
    bits_u32 = in_ref[0, 0]
    fill_f32 = jax.lax.bitcast_convert_type(bits_u32, jnp.float32)
    fill_i32 = jax.lax.bitcast_convert_type(bits_u32, jnp.int32)
    v_scratch[...] = jnp.full(v_scratch.shape, fill_f32, dtype=jnp.float32)

    @pl.loop(0, s_scratch.shape[0])
    def _fill_smem(i):
      s_scratch[i] = fill_i32

    out_ref[0] = in_ref[...]

  return pl.pallas_call(
      _poison_kernel,
      out_shape=jax.ShapeDtypeStruct((num_cores, 16, lanes), jnp.uint32),
      grid=(num_cores,),
      in_specs=[pl.BlockSpec((16, lanes), lambda c: (0, 0))],
      out_specs=pl.BlockSpec((1, 16, lanes), lambda c: (c, 0, 0)),
      scratch_shapes=[
          pltpu.VMEM((vmem_words // lanes, lanes), jnp.float32),
          pltpu.SMEM((smem_words,), jnp.int32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          disable_bounds_checks=True,
      ),
  )(fill_vec_u32)


def _poison_vmem_smem(fill_bits_u32: int = 0x7FC00000) -> None:
  if jax.default_backend() != "tpu":
    return
  vec = jnp.full((16, mosaic_tpu.LANES), jnp.uint32(fill_bits_u32), jnp.uint32)
  _jit_poison_vmem_smem(vec).block_until_ready()


class PallasMosaicTpuTriangleMultiplicationTest(parameterized.TestCase):

  def setUp(self):
    if not common.has_mosaic_tpu_support():
      self.skipTest("Mosaic TPU not supported on this platform.")
    super().setUp()
    self._op = _DEFAULT_OP
    self._ref_op = _DEFAULT_REF_OP

  def _assert_matches_reference(self, out, params, triangle_type):
    expected, _ = _jit_ref_fwd_with_res(params, triangle_type)
    chex.assert_trees_all_close(out, expected, atol=0.3, rtol=0.05)

  def _check_fwd_and_9_grads(
      self,
      params: dict[str, Any],
      triangle_type: Literal["incoming", "outgoing"],
      *,
      fwd_cfg: dict[str, int] | None = None,
      vjp_cfg: dict[str, int] | None = None,
      use_api: bool = False,
      api_impl: Any = "mosaic",
      epsilon: float = 1e-6,
      fwd_atol: float = 0.15,
      fwd_rtol: float = 0.05,
      grad_atol: float = 0.50,
      grad_rtol: float = 0.05,
      valid_mask: jax.Array | None = None,
      vs_fp32_highest: bool = False,
      seed_target: int = 777,
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    n = params["x"].shape[0]
    d = params["projection_out_weights"].shape[1]
    mask = params["mask"]
    diff_params = {k: params[k] for k in _GRAD_KEYS}
    target = _make_random_array((n, n, d), jnp.bfloat16, seed_target)
    has_vm = valid_mask is not None
    vm = valid_mask if valid_mask is not None else mask
    if has_vm:
      target = jnp.where(vm[:, :, None], target, 0.0)

    cfg_f = mtpu_tri_mul.Config(**fwd_cfg) if fwd_cfg is not None else None
    cfg_b = mtpu_tri_mul_vjp.Config(**vjp_cfg) if vjp_cfg is not None else None

    (loss_tpu, out_tpu), grads_tpu = _jit_tpu_val_and_grad(
        diff_params,
        mask,
        target,
        vm,
        fwd_cfg=cfg_f,
        vjp_cfg=cfg_b,
        triangle_type=triangle_type,
        epsilon=epsilon,
        use_api=use_api,
        api_impl=api_impl,
        has_valid_mask=has_vm,
    )
    diff_clean = (
        dict(diff_params, x=jnp.where(vm[:, :, None], diff_params["x"], 0.0))
        if has_vm
        else diff_params
    )
    (_, out_xla), grads_xla = _jit_xla_val_and_grad(
        diff_clean,
        mask,
        target,
        vm,
        triangle_type=triangle_type,
        epsilon=epsilon,
        has_valid_mask=has_vm,
    )

    self.assertEqual(out_tpu.shape, (n, n, d))
    self.assertEqual(out_tpu.dtype, jnp.bfloat16)
    self.assertEqual(set(grads_tpu.keys()), set(_GRAD_KEYS))
    self.assertTrue(bool(jnp.isfinite(loss_tpu)))
    for g in grads_tpu.values():
      self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

    if has_vm:
      out_tpu_cmp = jnp.where(vm[:, :, None], out_tpu, 0.0)
      out_xla_cmp = jnp.where(vm[:, :, None], out_xla, 0.0)
    else:
      out_tpu_cmp, out_xla_cmp = out_tpu, out_xla
      self.assertTrue(bool(jnp.all(jnp.isfinite(out_tpu))))

    if vs_fp32_highest and not has_vm:
      out_ref = _jit_ref_fp32_highest(params, triangle_type, epsilon)
      chex.assert_trees_all_close(
          out_tpu.astype(jnp.float32),
          out_ref.astype(jnp.float32),
          atol=fwd_atol,
          rtol=fwd_rtol,
      )
    else:
      chex.assert_trees_all_close(
          out_tpu_cmp.astype(jnp.float32),
          out_xla_cmp.astype(jnp.float32),
          atol=fwd_atol,
          rtol=fwd_rtol,
      )

    for k in _GRAD_KEYS:
      chex.assert_trees_all_close(
          grads_tpu[k].astype(jnp.float32),
          grads_xla[k].astype(jnp.float32),
          atol=grad_atol,
          rtol=grad_rtol,
      )
    return out_tpu, grads_tpu

  @parameterized.product(
      n=[100, 128, 384],
      triangle_type=["incoming", "outgoing"],
      dtype=[jnp.bfloat16],
  )
  def test_triangle_multiplication(self, n, triangle_type, dtype):
    params = _get_params(n=n, c=64, h=64, d=64, dtype=dtype)
    out = _jit_op_fwd(params, triangle_type)
    expected = jnp.asarray(
        _jit_ref_fp32_highest(params, triangle_type), dtype=dtype
    )

    self.assertEqual(out.shape, (n, n, 64))
    self.assertEqual(out.dtype, dtype)
    atol = 0.15 if n < 128 else 0.45
    rtol = 0.05
    chex.assert_trees_all_close(out, expected, atol=atol, rtol=rtol)

  @parameterized.parameters("incoming", "outgoing")
  def test_non_uniform_channel_dims(self, triangle_type):
    params = _get_params(n=64, c=16, h=32, d=64, dtype=jnp.bfloat16)
    out = _jit_op_fwd(params, triangle_type)
    expected = jnp.asarray(
        _jit_ref_fp32_highest(params, triangle_type), dtype=jnp.bfloat16
    )

    self.assertEqual(out.shape, (64, 64, 64))
    self.assertEqual(out.dtype, jnp.bfloat16)
    chex.assert_trees_all_close(out, expected, atol=0.15, rtol=0.05)

  @parameterized.parameters("incoming", "outgoing")
  def test_api_dispatch(self, triangle_type):
    params = _get_params(n=128, c=64, h=64, d=64, dtype=jnp.bfloat16)

    @jax.jit
    def f_mosaic(p):
      return api.triangle_multiplication(
          **p, triangle_type=triangle_type, implementation="mosaic"
      )

    out_mosaic = f_mosaic(params)
    out_xla, _ = _jit_ref_fwd_with_res(params, triangle_type)

    chex.assert_trees_all_close(out_mosaic, out_xla, atol=0.15, rtol=0.05)

    bound_args = hlo_utils.get_bound_args(f_mosaic.lower(params))
    self.assertLen(bound_args, 1)
    self.assertIsInstance(
        bound_args[0].op, mtpu_tri_mul.PallasMosaicTpuTriangleMultiplication
    )

  def test_autotuning_configs(self):
    n, c, h, d = 128, 64, 64, 64
    dtype = jnp.bfloat16
    params = _get_params(n=n, c=c, h=h, d=d, dtype=dtype)
    ba = self._op.bind(**params, triangle_type="outgoing")
    configs = self._op._get_autotuning_configs(ba)
    self.assertNotEmpty(configs)
    self.assertIn(self._op._get_heuristics_config(ba), configs)
    self.assertLen(
        {
            mtpu_tri_mul.kernel.resolve_config_blocks(n, cfg, x=params["x"])
            for cfg in configs
        },
        len(configs),
    )
    expected, _ = _jit_ref_fwd_with_res(params, "outgoing")
    for cfg in configs:
      out = _jit_op_cfg_fwd(params, "outgoing", cfg)
      chex.assert_trees_all_close(out, expected, atol=0.15, rtol=0.05)

    vjp_op = mtpu_tri_mul_vjp.PallasMosaicTpuTriangleMultiplicationVjp()
    vjp_ba = vjp_op.bind(**ba.vjp_arg_spec)
    vjp_configs = vjp_op._get_autotuning_configs(vjp_ba)
    self.assertNotEmpty(vjp_configs)
    self.assertIn(vjp_op._get_heuristics_config(vjp_ba), vjp_configs)
    self.assertLen(
        {
            mtpu_tri_mul.kernel.resolve_config_blocks(n, cfg, x=params["x"])
            for cfg in vjp_configs
        },
        len(vjp_configs),
    )
    dout = _make_random_array((n, n, d), dtype, 0)
    diff_params = {k: params[k] for k in _GRAD_KEYS}
    _, expected_grads = _jit_ref_fwd_and_vjp(
        diff_params, params["mask"], dout, "outgoing"
    )
    out_res, residuals = _jit_op_fwd_with_res(params, "outgoing")
    for vjp_cfg in vjp_configs:
      grads = _jit_standalone_vjp(
          residuals, out_res, dout, params, "outgoing", vjp_cfg
      )
      for k in params:
        if k == "mask":
          continue
        with self.subTest(vjp_cfg=str(vjp_cfg), grad=k):
          chex.assert_trees_all_close(
              grads[k].astype(jnp.float32),
              expected_grads[k].astype(jnp.float32),
              atol=0.5 * n,
              rtol=0.05,
          )

  @parameterized.parameters(
      dict(c=128, block_m=128),
      dict(c=128, block_m=256),
      dict(c=64, block_m=512),
  )
  def test_channel_and_block_sizes(self, c, block_m):
    params = _get_params(n=64, c=c, h=c, d=c, dtype=jnp.bfloat16)
    config = mtpu_tri_mul.Config(block_m=block_m)
    out = _jit_op_cfg_fwd(params, "outgoing", config)
    self._assert_matches_reference(out, params, "outgoing")

  def test_epilogue_config_overrides(self):
    params = _get_params(n=64, c=64, h=64, d=64, dtype=jnp.bfloat16)
    config = mtpu_tri_mul.Config(
        block_m=512,
        epilogue_block_m=256,
    )
    out = _jit_op_cfg_fwd(params, "incoming", config)
    self._assert_matches_reference(out, params, "incoming")

  @parameterized.product(
      triangle_type=["incoming", "outgoing"],
      n=[64, 100],
  )
  def test_sparse_mask(self, triangle_type, n):
    params = _get_params(n=n, c=64, h=64, d=64, dtype=jnp.bfloat16)
    mask = jax.random.uniform(jax.random.key(1), (n, n)) < 0.1
    params["mask"] = mask.at[: n // 2].set(False)

    out = _jit_op_fwd(params, triangle_type)
    self._assert_matches_reference(out, params, triangle_type)

  @parameterized.parameters(jnp.nan, jnp.inf, -jnp.inf)
  def test_nan_in_masked_tokens_does_not_propagate(self, bad_val):
    _poison_vmem_smem()
    n = 64
    valid = jnp.arange(n) < 48
    mask = valid[:, None] & valid[None, :]
    params = _get_params(n=n, c=64, h=64, d=64, dtype=jnp.bfloat16)
    params.pop("mask")
    params_nan = dict(params)
    params_nan["x"] = jnp.where(mask[:, :, None], params["x"], bad_val)

    out_clean = _jit_op_fwd(dict(params, mask=mask), "outgoing")
    out_nan = _jit_op_fwd(dict(params_nan, mask=mask), "outgoing")
    chex.assert_trees_all_close(out_nan[:48, :48], out_clean[:48, :48])

    dout = jnp.where(
        mask[:, :, None],
        _make_random_array((n, n, 64), jnp.bfloat16, 3),
        0.0,
    )
    _, grads_clean = _jit_op_fwd_and_vjp(params, mask, dout, "outgoing")
    _, grads_nan = _jit_op_fwd_and_vjp(params_nan, mask, dout, "outgoing")
    for k in grads_nan:
      with self.subTest(k):
        self.assertTrue(bool(jnp.all(jnp.isfinite(grads_nan[k]))))
        chex.assert_trees_all_close(grads_nan[k], grads_clean[k])

  def test_zero_epsilon_padded_rows_no_nan(self):
    n = 20
    params = _get_params(n=n, c=64, h=64, d=64, dtype=jnp.bfloat16)
    mask = params.pop("mask")
    dout = _make_random_array((n, n, 64), jnp.bfloat16, 0)
    out, grads = _jit_op_fwd_and_vjp(
        params, mask, dout, "outgoing", epsilon=0.0
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    for g in jax.tree.leaves(grads):
      self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

  @parameterized.parameters("incoming", "outgoing")
  def test_large_mean_shift_stability(self, triangle_type):
    n = 64
    params = _get_params(n=n, c=64, h=64, d=64, dtype=jnp.bfloat16)
    params["layernorm_in_offset"] = (
        params["layernorm_in_offset"].astype(jnp.float32) + 20.0
    ).astype(jnp.bfloat16)
    params["projection_in_weights"] = (
        jnp.abs(params["projection_in_weights"].astype(jnp.float32)) + 0.5
    ).astype(jnp.bfloat16)
    mask = params.pop("mask")
    dout = _make_random_array((n, n, 64), jnp.bfloat16, 5) / n

    full_params = dict(params, mask=mask)
    out = _jit_op_fwd(full_params, triangle_type)
    out_ref, _ = _jit_ref_fwd_with_res(full_params, triangle_type)
    chex.assert_trees_all_close(out, out_ref, atol=0.35, rtol=0.05)

    grads = _jit_op_loss_grad(
        params, mask, dout, triangle_type, scale_by_n=False
    )
    grads_ref = _jit_ref_loss_grad(
        params, mask, dout, triangle_type, scale_by_n=False
    )
    for k in params:
      with self.subTest(k):
        chex.assert_trees_all_close(
            grads[k].astype(jnp.float32),
            grads_ref[k].astype(jnp.float32),
            atol=0.75,
            rtol=0.06,
        )

  def test_config_validation(self):
    mtpu_tri_mul.Config(
        block_m=1024,
        epilogue_block_m=128,
    )
    mtpu_tri_mul_vjp.Config(
        block_m=256,
        epilogue_block_m=512,
    )
    for kwargs in (
        dict(block_m=96),
        dict(block_m=64),
        dict(block_m=2048),
        dict(epilogue_block_m=64),
        dict(epilogue_block_m=192),
    ):
      with self.subTest(str(kwargs)), self.assertRaises(ValueError):
        mtpu_tri_mul.Config(**kwargs)
      with self.subTest(f"vjp_{kwargs}"), self.assertRaises(ValueError):
        mtpu_tri_mul_vjp.Config(**kwargs)
    for bad_bi in (0, -8, 4, 7, 10, 12, 15, 20, 4104):
      with self.subTest(f"bi_{bad_bi}"), self.assertRaises(ValueError):
        mtpu_tri_mul.Config(block_i=bad_bi)
    for bad_ebi in (9, 12, 18, 4104):
      with self.subTest(f"ebi_{bad_ebi}"), self.assertRaises(ValueError):
        mtpu_tri_mul.Config(epilogue_block_i=bad_ebi)
    for bad_bj in (0, -128, 8, 64, 96, 192, 300, 4224):
      with self.subTest(f"bj_{bad_bj}"), self.assertRaises(ValueError):
        mtpu_tri_mul.Config(block_j=bad_bj)
    for bad_ebj in (64, 192, 250, 4224):
      with self.subTest(f"ebj_{bad_ebj}"), self.assertRaises(ValueError):
        mtpu_tri_mul.Config(epilogue_block_j=bad_ebj)
    cfg = mtpu_tri_mul.Config(block_m=256)
    with self.assertRaises((AttributeError, TypeError, ValueError)):
      setattr(cfg, "block_m", 128)
    with self.assertRaises((TypeError, ValueError)):
      getattr(mtpu_tri_mul, "Config")(**{"unknown_field": 128})

  @parameterized.product(
      triangle_type=["incoming", "outgoing"],
      use_api=[False, True],
      n=[64, 100],
      dtype=[jnp.bfloat16],
  )
  def test_gradients(self, triangle_type, use_api, n, dtype):
    params = _get_params(n=n, c=64, h=64, d=64, dtype=dtype)
    mask = params.pop("mask")
    dout = _make_random_array((n, n, 64), dtype, 0)

    if use_api:
      fn = api.triangle_multiplication
    else:
      fn = self._op

    def loss(p, fn, **kwargs):
      out = fn(**p, mask=mask, triangle_type=triangle_type, **kwargs)
      return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32)) / n

    grad_fn = jax.jit(jax.grad(lambda p: loss(p, fn)))
    grads = grad_fn(params)
    expected = _jit_ref_loss_grad(
        params, mask, dout, triangle_type, scale_by_n=True
    )

    bound_args = hlo_utils.get_bound_args(grad_fn.lower(params))
    abstract_params, abstract_mask = jax.eval_shape(
        lambda p, m: (p, m), params, mask
    )
    expected_fwd_ba = self._op.bind(
        **abstract_params,
        mask=abstract_mask,
        triangle_type=triangle_type,
        return_residuals=True,
    )
    expected_vjp_ba = self._op.vjp.bind(**expected_fwd_ba.vjp_arg_spec)  # pyrefly: ignore[missing-attribute]
    self.assertCountEqual(bound_args, (expected_fwd_ba, expected_vjp_ba))

    self.assertEqual(jax.tree.structure(grads), jax.tree.structure(expected))
    for k in params:
      with self.subTest(k):
        chex.assert_trees_all_close(
            grads[k].astype(jnp.float32),
            expected[k].astype(jnp.float32),
            atol=0.5,
            rtol=0.05,
        )

  @parameterized.product(
      triangle_type=["incoming", "outgoing"],
      dims=[
          dict(n=64, c=32, h=16, d=16),
          dict(n=64, c=16, h=32, d=64),
          dict(n=64, c=64, h=64, d=64),
          dict(n=64, c=128, h=128, d=128),
      ],
  )
  def test_gradients_channel_dims_and_sparse_mask(self, triangle_type, dims):
    n, c, h, d = dims["n"], dims["c"], dims["h"], dims["d"]
    params = _get_params(n=n, c=c, h=h, d=d, dtype=jnp.bfloat16)
    mask = jax.random.uniform(jax.random.key(1), (n, n)) < 0.25
    mask = mask.at[: n // 4].set(False)
    params.pop("mask")
    dout = _make_random_array((n, n, d), jnp.bfloat16, 0)

    grads = _jit_op_loss_grad(
        params, mask, dout, triangle_type, scale_by_n=True
    )
    expected = _jit_ref_loss_grad(
        params, mask, dout, triangle_type, scale_by_n=True
    )

    for k in params:
      with self.subTest(k):
        chex.assert_trees_all_close(
            grads[k].astype(jnp.float32),
            expected[k].astype(jnp.float32),
            atol=0.5,
            rtol=0.05,
        )

  @parameterized.parameters("incoming", "outgoing")
  def test_gradients_bwd_config_overrides(self, triangle_type):
    n = 64
    params = _get_params(n=n, c=64, h=64, d=64, dtype=jnp.bfloat16)
    dout = _make_random_array((n, n, 64), jnp.bfloat16, 0)
    config = mtpu_tri_mul.Config(block_m=256)
    vjp_config = mtpu_tri_mul_vjp.Config(
        block_m=256,
        epilogue_block_m=128,
    )
    diff_params = {k: params[k] for k in _GRAD_KEYS}
    out, grads = _jit_op_fwd_and_vjp(
        diff_params,
        params["mask"],
        dout,
        triangle_type,
        fwd_cfg=config,
        vjp_cfg=vjp_config,
    )
    out_ref, expected = _jit_ref_fwd_and_vjp(
        diff_params, params["mask"], dout, triangle_type
    )

    chex.assert_trees_all_close(out, out_ref, atol=0.3, rtol=0.05)
    for k in params:
      if k == "mask":
        continue
      with self.subTest(k):
        chex.assert_trees_all_close(
            grads[k].astype(jnp.float32),
            expected[k].astype(jnp.float32),
            atol=0.5 * n,
            rtol=0.05,
        )

  def test_return_residuals(self):
    params = _get_params(n=64, c=16, h=32, d=64, dtype=jnp.bfloat16)
    out, residuals = _jit_op_fwd_with_res(params, "outgoing")
    expected_out, expected_residuals = _jit_ref_fwd_with_res(params, "outgoing")
    self.assertIsInstance(residuals, tuple)
    self.assertLen(residuals, 3)
    for res in residuals:
      self.assertEqual(res.shape, (32, 64, 64))
      self.assertEqual(res.dtype, jnp.bfloat16)
    self.assertEqual(out.shape, (64, 64, 64))
    chex.assert_trees_all_close(
        residuals[0], expected_residuals[0], atol=0.15, rtol=0.05
    )
    chex.assert_trees_all_close(
        residuals[1], expected_residuals[1], atol=0.15, rtol=0.05
    )
    chex.assert_trees_all_close(
        residuals[2],
        jnp.moveaxis(expected_residuals[2], -1, 0),
        atol=0.2 * 64,
        rtol=0.05,
    )
    chex.assert_trees_all_close(out, expected_out, atol=0.3, rtol=0.05)

  @parameterized.parameters("incoming", "outgoing")
  def test_small_n_boundary(self, triangle_type):
    params = _get_params(n=1, c=64, h=64, d=64, dtype=jnp.bfloat16)
    mask = params.pop("mask")
    dout = _make_random_array((1, 1, 64), jnp.bfloat16, 0)

    out, grads = _jit_op_fwd_and_vjp(params, mask, dout, triangle_type)
    out_ref, expected = _jit_ref_fwd_and_vjp(params, mask, dout, triangle_type)

    chex.assert_trees_all_close(out, out_ref, atol=0.15, rtol=0.05)
    for k in params:
      with self.subTest(k):
        chex.assert_trees_all_close(
            grads[k].astype(jnp.float32),
            expected[k].astype(jnp.float32),
            atol=0.5,
            rtol=0.05,
        )

  def test_int32_offset_overflow_raises(self):
    vjp_op = mtpu_tri_mul_vjp.PallasMosaicTpuTriangleMultiplicationVjp()

    def get_shapes(n, c):
      dtype = jnp.bfloat16
      return dict(
          x=jax.ShapeDtypeStruct((n, n, c), dtype),
          mask=jax.ShapeDtypeStruct((n, n), jnp.bool_),
          projection_in_weights=jax.ShapeDtypeStruct((c, 2, c), dtype),
          gate_in_weights=jax.ShapeDtypeStruct((c, 2, c), dtype),
          projection_out_weights=jax.ShapeDtypeStruct((c, c), dtype),
          gate_out_weights=jax.ShapeDtypeStruct((c, c), dtype),
          layernorm_in_scale=jax.ShapeDtypeStruct((c,), dtype),
          layernorm_in_offset=jax.ShapeDtypeStruct((c,), dtype),
          layernorm_out_scale=jax.ShapeDtypeStruct((c,), dtype),
          layernorm_out_offset=jax.ShapeDtypeStruct((c,), dtype),
      )

    f = lambda p: self._op(**p, triangle_type="outgoing")
    out = jax.eval_shape(f, get_shapes(n=4096, c=128))
    self.assertEqual(out.shape, (4096, 4096, 128))
    with self.assertRaisesRegex(NotImplementedError, "int32"):
      jax.eval_shape(f, get_shapes(n=4097, c=128))
    with self.assertRaisesRegex(NotImplementedError, "int32"):
      jax.eval_shape(f, get_shapes(n=5792, c=64))

    def f_vjp(p, res):
      return vjp_op(
          res,
          p["x"],
          p["x"],
          **p,
          triangle_type="outgoing",
          precision=(jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
          epsilon=1e-6,
      )

    res_4097 = (jax.ShapeDtypeStruct((128, 4097, 4097), jnp.bfloat16),) * 3
    with self.assertRaisesRegex(NotImplementedError, "int32"):
      jax.eval_shape(f_vjp, get_shapes(n=4097, c=128), res_4097)

    out_5760 = jax.eval_shape(f, get_shapes(n=5760, c=64))
    self.assertEqual(out_5760.shape, (5760, 5760, 64))
    res_5760 = (jax.ShapeDtypeStruct((64, 5760, 5760), jnp.bfloat16),) * 3
    grads_5760 = jax.eval_shape(f_vjp, get_shapes(n=5760, c=64), res_5760)
    self.assertLen(grads_5760, 9)

    for n_ov, c_ov, h_ov, d_ov in (
        (4097, 16, 128, 16),
        (4097, 16, 16, 128),
        (5761, 64, 64, 64),
    ):
      asym = dict(
          x=jax.ShapeDtypeStruct((n_ov, n_ov, c_ov), jnp.bfloat16),
          mask=jax.ShapeDtypeStruct((n_ov, n_ov), jnp.bool_),
          projection_in_weights=jax.ShapeDtypeStruct(
              (c_ov, 2, h_ov), jnp.bfloat16
          ),
          gate_in_weights=jax.ShapeDtypeStruct((c_ov, 2, h_ov), jnp.bfloat16),
          projection_out_weights=jax.ShapeDtypeStruct(
              (h_ov, d_ov), jnp.bfloat16
          ),
          gate_out_weights=jax.ShapeDtypeStruct((c_ov, d_ov), jnp.bfloat16),
          layernorm_in_scale=jax.ShapeDtypeStruct((c_ov,), jnp.bfloat16),
          layernorm_in_offset=jax.ShapeDtypeStruct((c_ov,), jnp.bfloat16),
          layernorm_out_scale=jax.ShapeDtypeStruct((h_ov,), jnp.bfloat16),
          layernorm_out_offset=jax.ShapeDtypeStruct((h_ov,), jnp.bfloat16),
      )
      with self.subTest(f"fwd_{n_ov}_{c_ov}_{h_ov}_{d_ov}"):
        with self.assertRaisesRegex(NotImplementedError, "int32"):
          jax.eval_shape(f, asym)
      res_ov = (jax.ShapeDtypeStruct((h_ov, n_ov, n_ov), jnp.bfloat16),) * 3
      out_ov = jax.ShapeDtypeStruct((n_ov, n_ov, d_ov), jnp.bfloat16)
      with self.subTest(f"vjp_{n_ov}_{c_ov}_{h_ov}_{d_ov}"):
        with self.assertRaisesRegex(NotImplementedError, "int32"):
          jax.eval_shape(
              lambda p, r, o: vjp_op(
                  r,
                  o,
                  o,
                  **p,
                  triangle_type="outgoing",
                  precision=_DEFAULT_PREC,
                  epsilon=1e-6,
              ),
              asym,
              res_ov,
              out_ov,
          )

  def test_unsupported_raises_not_implemented(self):
    vjp_op = mtpu_tri_mul_vjp.PallasMosaicTpuTriangleMultiplicationVjp()
    default_prec = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)

    with self.subTest("empty_inputs"):
      params = dict(
          x=jnp.zeros((0, 0, 64), dtype=jnp.bfloat16),
          mask=jnp.zeros((0, 0), dtype=jnp.bool_),
          projection_in_weights=jnp.zeros((64, 2, 64), dtype=jnp.bfloat16),
          gate_in_weights=jnp.zeros((64, 2, 64), dtype=jnp.bfloat16),
          projection_out_weights=jnp.zeros((64, 64), dtype=jnp.bfloat16),
          gate_out_weights=jnp.zeros((64, 64), dtype=jnp.bfloat16),
          layernorm_in_scale=jnp.zeros((64,), dtype=jnp.bfloat16),
          layernorm_in_offset=jnp.zeros((64,), dtype=jnp.bfloat16),
          layernorm_out_scale=jnp.zeros((64,), dtype=jnp.bfloat16),
          layernorm_out_offset=jnp.zeros((64,), dtype=jnp.bfloat16),
      )
      res = (jnp.zeros((64, 0, 0), dtype=jnp.bfloat16),) * 3
      with self.assertRaises(NotImplementedError):
        self._op(**params, triangle_type="outgoing")
      with self.assertRaises(NotImplementedError):
        vjp_op(
            res,
            params["x"],
            params["x"],
            **params,
            triangle_type="outgoing",
            precision=default_prec,
            epsilon=1e-6,
        )

    with self.subTest("unsupported_dtype"):
      params = _zeros_params(128, 64, 64, 64, dtype=jnp.float32)
      res = (jnp.zeros((64, 128, 128), dtype=jnp.float32),) * 3
      with self.assertRaises(NotImplementedError):
        self._op(**params, triangle_type="outgoing")
      with self.assertRaises(NotImplementedError):
        vjp_op(
            res,
            params["x"],
            params["x"],
            **params,
            triangle_type="outgoing",
            precision=default_prec,
            epsilon=1e-6,
        )

    with self.subTest("unsupported_channel_dims"):
      params = _zeros_params(128, 48, 48, 64, dtype=jnp.bfloat16)
      res = (jnp.zeros((48, 128, 128), dtype=jnp.bfloat16),) * 3
      with self.assertRaises(NotImplementedError):
        self._op(**params, triangle_type="outgoing")
      out = jnp.zeros((128, 128, 64), dtype=jnp.bfloat16)
      with self.assertRaises(NotImplementedError):
        vjp_op(
            res,
            out,
            out,
            **params,
            triangle_type="outgoing",
            precision=default_prec,
            epsilon=1e-6,
        )

    with self.subTest("unsupported_precision"):
      params = _zeros_params(128, 64, 64, 64, dtype=jnp.bfloat16)
      res = (jnp.zeros((64, 128, 128), dtype=jnp.bfloat16),) * 3
      for bad_prec in (
          jax.lax.DotAlgorithmPreset.F32_F32_F32,
          jax.lax.Precision.HIGH,
          jax.lax.Precision.HIGHEST,
      ):
        with self.assertRaises(NotImplementedError):
          self._op(
              **params,
              triangle_type="outgoing",
              precision=bad_prec,
          )
      with _disable_jaxtyping():
        with self.assertRaises(NotImplementedError):
          self._op(
              **params,
              triangle_type="outgoing",
              precision=cast(Any, ("F32_F32_F32", "F32_F32_F32")),
          )
      for bad_vjp_prec in (
          jax.lax.DotAlgorithmPreset.F32_F32_F32,
          (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
      ):
        with self.assertRaises(NotImplementedError):
          vjp_op(
              res,
              params["x"],
              params["x"],
              **params,
              triangle_type="outgoing",
              precision=bad_vjp_prec,
              epsilon=1e-6,
          )

    with self.subTest("mismatched_weight_dtype"):
      params = _zeros_params(128, 64, 64, 64, dtype=jnp.bfloat16)
      res = (jnp.zeros((64, 128, 128), dtype=jnp.bfloat16),) * 3
      params["projection_in_weights"] = params["projection_in_weights"].astype(
          jnp.float16
      )
      with self.assertRaises(NotImplementedError):
        self._op(**params, triangle_type="outgoing")
      with self.assertRaises(NotImplementedError):
        vjp_op(
            res,
            params["x"],
            params["x"],
            **params,
            triangle_type="outgoing",
            precision=default_prec,
            epsilon=1e-6,
        )

    with self.subTest("mismatched_residual_dtype"):
      params = _zeros_params(128, 64, 64, 64, dtype=jnp.bfloat16)
      res = (jnp.zeros((64, 128, 128), dtype=jnp.float16),) * 3
      with self.assertRaises(NotImplementedError):
        vjp_op(
            res,
            params["x"],
            params["x"],
            **params,
            triangle_type="outgoing",
            precision=default_prec,
            epsilon=1e-6,
        )

    with self.subTest("vjp_return_residuals"):
      params = _zeros_params(64, 64, 64, 64, dtype=jnp.bfloat16)
      res = (jnp.zeros((64, 64, 64), dtype=jnp.bfloat16),) * 3
      with self.assertRaises(NotImplementedError):
        vjp_op(
            res,
            params["x"],
            params["x"],
            **params,
            triangle_type="outgoing",
            precision=default_prec,
            epsilon=1e-6,
            return_residuals=True,
        )

    with self.subTest("mismatched_residual_shape"):
      params = _zeros_params(64, 32, 32, 64, dtype=jnp.bfloat16)
      bad_res = (
          jnp.zeros((32, 64, 64), dtype=jnp.bfloat16),
          jnp.zeros((32, 64, 64), dtype=jnp.bfloat16),
          jnp.zeros((64, 64, 32), dtype=jnp.bfloat16),
      )
      out = jnp.zeros((64, 64, 64), dtype=jnp.bfloat16)
      with self.assertRaises(NotImplementedError):
        vjp_op(
            bad_res,
            out,
            out,
            **params,
            triangle_type="outgoing",
            precision=default_prec,
            epsilon=1e-6,
        )

    with self.subTest("invalid_residual_tuple_length"):
      params = _zeros_params(64, 32, 64, 16, dtype=jnp.bfloat16)
      r0 = jnp.zeros((64, 64, 64), dtype=jnp.bfloat16)
      out = jnp.zeros((64, 64, 16), dtype=jnp.bfloat16)
      with _disable_jaxtyping():
        for bad_tuple in ((), (r0, r0), (r0, r0, r0, r0)):
          with self.assertRaises(NotImplementedError):
            vjp_op(
                cast(Any, bad_tuple),
                out,
                out,
                **params,
                triangle_type="outgoing",
                precision=default_prec,
                epsilon=1e-6,
            )

    with self.subTest("mismatched_dout_shape"):
      params = _zeros_params(64, 64, 64, 64, dtype=jnp.bfloat16)
      res = (jnp.zeros((64, 64, 64), dtype=jnp.bfloat16),) * 3
      out = jnp.zeros((64, 64, 64), dtype=jnp.bfloat16)
      with _disable_jaxtyping():
        for bad_dout in (
            jnp.ones((128, 128, 64), dtype=jnp.bfloat16),
            jnp.ones((64, 64, 32), dtype=jnp.bfloat16),
        ):
          with self.assertRaises(NotImplementedError):
            vjp_op(
                res,
                out,
                bad_dout,
                **params,
                triangle_type="outgoing",
                precision=default_prec,
                epsilon=1e-6,
            )

    with self.subTest("negative_epsilon"):
      params = _zeros_params(64, 64, 64, 64, dtype=jnp.bfloat16)
      res = (jnp.zeros((64, 64, 64), dtype=jnp.bfloat16),) * 3
      for neg_eps in (-1e-6, -1.0):
        with self.assertRaises(NotImplementedError):
          self._op(**params, triangle_type="outgoing", epsilon=neg_eps)
        with self.assertRaises(NotImplementedError):
          vjp_op(
              res,
              params["x"],
              params["x"],
              **params,
              triangle_type="outgoing",
              precision=default_prec,
              epsilon=neg_eps,
          )

    with self.subTest("symbolic_shape"):
      (n_sym,) = jax.export.symbolic_shape("n")
      sym_params = dict(
          x=jax.ShapeDtypeStruct((n_sym, n_sym, 64), jnp.bfloat16),
          mask=jax.ShapeDtypeStruct((n_sym, n_sym), jnp.bool_),
          projection_in_weights=jax.ShapeDtypeStruct((64, 2, 64), jnp.bfloat16),
          gate_in_weights=jax.ShapeDtypeStruct((64, 2, 64), jnp.bfloat16),
          projection_out_weights=jax.ShapeDtypeStruct((64, 64), jnp.bfloat16),
          gate_out_weights=jax.ShapeDtypeStruct((64, 64), jnp.bfloat16),
          layernorm_in_scale=jax.ShapeDtypeStruct((64,), jnp.bfloat16),
          layernorm_in_offset=jax.ShapeDtypeStruct((64,), jnp.bfloat16),
          layernorm_out_scale=jax.ShapeDtypeStruct((64,), jnp.bfloat16),
          layernorm_out_offset=jax.ShapeDtypeStruct((64,), jnp.bfloat16),
      )
      with self.assertRaisesRegex(NotImplementedError, "symbolic"):
        jax.eval_shape(
            lambda p: self._op(**p, triangle_type="outgoing"), sym_params
        )

  # ---------------------------------------------------------------------------
  # R2 Feature Regression Tests (F3, F5, F6, F7, F8, F9, F10, F11)
  # ---------------------------------------------------------------------------

  def test_return_residuals_unaligned_and_standalone_vjp(self):
    n = 67
    p = _make_seeded_params(303, n, 64, 32, 16)
    out, res = _jit_op_fwd_with_res(p, "outgoing")
    exp_out, exp_res = _jit_ref_fwd_with_res(p, "outgoing")
    self.assertLen(res, 3)
    for r in res:
      self.assertEqual(r.shape, (32, n, n))
    chex.assert_trees_all_close(out, exp_out, atol=0.30, rtol=0.05)
    chex.assert_trees_all_close(res[0], exp_res[0], atol=0.15, rtol=0.05)
    chex.assert_trees_all_close(res[1], exp_res[1], atol=0.15, rtol=0.05)
    chex.assert_trees_all_close(
        res[2], jnp.moveaxis(exp_res[2], -1, 0), atol=0.2 * n, rtol=0.05
    )

    out_no_res = _jit_op_fwd(p, "outgoing")
    chex.assert_trees_all_equal(out, out_no_res)

    dout = _make_random_array((n, n, 16), jnp.bfloat16, 3050) / float(n)
    dp = {k: p[k] for k in _GRAD_KEYS}
    grads_vjp = _jit_standalone_vjp(res, out, dout, p, "outgoing")
    _, grads_ref = _jit_ref_fwd_and_vjp(dp, p["mask"], dout, "outgoing")
    for k in _GRAD_KEYS:
      chex.assert_trees_all_close(
          grads_vjp[k].astype(jnp.float32),
          grads_ref[k].astype(jnp.float32),
          atol=0.50,
          rtol=0.05,
      )

  def test_non_uniform_channel_dims_2h_and_api_fallback(self):
    p_2h = _make_seeded_params(503, 35, 64, 128, 32)
    self._check_fwd_and_9_grads(p_2h, "outgoing", fwd_atol=0.30, grad_atol=0.50)

    p_zero_m = _make_seeded_params(506, 35, 64, 128, 32)
    p_zero_m["mask"] = jnp.zeros((35, 35), dtype=jnp.bool_)
    p_zero_m["layernorm_out_offset"] = jnp.ones((128,), dtype=jnp.bfloat16)
    out_zm, grads_zm = self._check_fwd_and_9_grads(
        p_zero_m, "outgoing", fwd_atol=0.15, grad_atol=0.50
    )
    self.assertGreater(float(jnp.max(jnp.abs(out_zm))), 1e-2)
    self.assertGreater(
        float(jnp.max(jnp.abs(grads_zm["gate_out_weights"]))), 1e-3
    )

    p_asym = _make_seeded_params(504, 35, 128, 64, 16)
    out_asym = _jit_op_fwd(p_asym, "incoming")
    ref_asym = _jit_ref_fp32_highest(p_asym, "incoming")
    chex.assert_trees_all_close(out_asym, ref_asym, atol=0.15, rtol=0.05)

    p_f32 = _make_seeded_params(521, 35, 32, 32, 32, dtype=jnp.float32)
    out_fb_f32 = _jit_api_fallback_fwd(p_f32, "outgoing")
    out_ref_f32, _ = _jit_ref_fwd_with_res_fn(p_f32, "outgoing")
    chex.assert_trees_all_close(out_fb_f32, out_ref_f32, atol=1e-3, rtol=1e-3)

    p_f16 = _make_seeded_params(522, 35, 32, 32, 32, dtype=jnp.float16)
    out_fb_f16 = _jit_api_fallback_fwd(p_f16, "outgoing")
    out_ref_f16, _ = _jit_ref_fwd_with_res_fn(p_f16, "outgoing")
    chex.assert_trees_all_close(
        out_fb_f16.astype(jnp.float32),
        out_ref_f16.astype(jnp.float32),
        atol=0.15,
        rtol=0.02,
    )

  def test_unmasked_nan_and_masked_cotangent_semantics(self):
    dout = jnp.ones((64, 64, 64), dtype=jnp.bfloat16) / 64.0
    mask_ones = jnp.ones((64, 64), dtype=jnp.bool_)
    # 1. Unmasked NaN/+Inf/-Inf in x must propagate (never silently zeroed).
    for sentinel in (jnp.nan, jnp.inf, -jnp.inf):
      p = _make_seeded_params(621, 64, 64, 64, 64)
      p["x"] = p["x"].at[5, 9, 0].set(sentinel)
      out, grads = _jit_op_fwd_and_vjp(
          {k: p[k] for k in _GRAD_KEYS}, mask_ones, dout, "outgoing"
      )
      self.assertFalse(bool(jnp.all(jnp.isfinite(out[5, 9]))))
      self.assertFalse(bool(jnp.all(jnp.isfinite(grads["x"][5, 9]))))
      self.assertFalse(bool(jnp.all(jnp.isfinite(grads["gate_out_weights"]))))

    # 2. NaN in projection_in_weights must propagate.
    p_w = _make_seeded_params(626, 64, 64, 64, 64)
    p_w["projection_in_weights"] = (
        p_w["projection_in_weights"].at[0, 0, 0].set(jnp.nan)
    )
    out_w, grads_w = _jit_op_fwd_and_vjp(
        {k: p_w[k] for k in _GRAD_KEYS}, mask_ones, dout, "outgoing"
    )
    self.assertTrue(bool(jnp.any(jnp.isnan(out_w))))
    self.assertTrue(bool(jnp.any(jnp.isnan(grads_w["x"]))))

    # 3. Masked x[50, 50] = NaN with non-zero dout[50, 50] propagates via
    #    gate_out, while projection_in_weights and gate_in_weights stay finite.
    p_m = _make_seeded_params(623, 64, 64, 64, 64)
    v1d = jnp.arange(64) < 48
    mask_48 = v1d[:, None] & v1d[None, :]
    p_m["x"] = p_m["x"].at[50, 50, :].set(jnp.nan)
    out_m, grads_m = _jit_op_fwd_and_vjp(
        {k: p_m[k] for k in _GRAD_KEYS}, mask_48, dout, "outgoing"
    )
    self.assertTrue(bool(jnp.all(jnp.isnan(out_m[50, 50]))))
    self.assertTrue(bool(jnp.any(jnp.isnan(grads_m["x"][50, 50]))))
    self.assertTrue(bool(jnp.any(jnp.isnan(grads_m["gate_out_weights"]))))
    self.assertTrue(
        bool(jnp.all(jnp.isfinite(grads_m["projection_in_weights"])))
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(grads_m["gate_in_weights"]))))

    # 4. Corrupted residual t3[:, 48:, :] = NaN isolated when dout[48:] == 0.
    p_r = _make_seeded_params(624, 64, 64, 64, 64)
    p_r["mask"] = mask_48
    out_r, res_r = _jit_op_fwd_with_res(p_r, "outgoing")
    dout_r = jnp.where(mask_48[:, :, None], dout, 0.0)
    g_clean = _jit_standalone_vjp(res_r, out_r, dout_r, p_r, "outgoing")
    res_bad = (res_r[0], res_r[1], res_r[2].at[:, 48:, :].set(jnp.nan))
    g_bad = _jit_standalone_vjp(res_bad, out_r, dout_r, p_r, "outgoing")
    for k in _GRAD_KEYS:
      self.assertTrue(bool(jnp.all(jnp.isfinite(g_bad[k]))))
      chex.assert_trees_all_close(g_bad[k], g_clean[k], atol=1e-4, rtol=1e-4)

  def test_zero_and_custom_epsilon_variance_edges(self):
    p = _make_seeded_params(702, 35, 32, 32, 32, scale_mode="lecun")
    p["x"] = (p["x"].astype(jnp.float32) * 0.05).astype(jnp.bfloat16)
    out_small = _jit_op_fwd(p, "outgoing", epsilon=1e-5)
    exp_small, _ = _jit_ref_fwd_with_res(p, "outgoing", epsilon=1e-5)
    chex.assert_trees_all_close(
        out_small.astype(jnp.float32),
        exp_small.astype(jnp.float32),
        atol=0.15,
        rtol=0.05,
    )
    out_large = _jit_op_fwd(p, "outgoing", epsilon=1e-1)
    exp_large, _ = _jit_ref_fwd_with_res(p, "outgoing", epsilon=1e-1)
    chex.assert_trees_all_close(
        out_large.astype(jnp.float32),
        exp_large.astype(jnp.float32),
        atol=0.15,
        rtol=0.05,
    )
    diff_max = jnp.max(
        jnp.abs(out_large.astype(jnp.float32) - out_small.astype(jnp.float32))
    )
    self.assertGreater(float(diff_max), 0.05)

    for val in (3.5, 1e4):
      p_c = _make_seeded_params(722, 35, 32, 32, 32)
      p_c["x"] = jnp.full((35, 35, 32), val, dtype=jnp.bfloat16)
      _, grads_c = self._check_fwd_and_9_grads(
          p_c, "outgoing", epsilon=0.0, fwd_atol=0.25, grad_atol=0.50
      )
      self.assertTrue(bool(jnp.all(grads_c["layernorm_in_scale"] == 0.0)))
      self.assertTrue(bool(jnp.all(grads_c["x"] == 0.0)))

    p_sub = _make_seeded_params(726, 35, 32, 32, 32)
    p_sub["x"] = jnp.full((35, 35, 32), 2.0, dtype=jnp.bfloat16)
    out_sub = _jit_op_fwd(p_sub, "outgoing", epsilon=1e-45)
    exp_sub, _ = _jit_ref_fwd_with_res(p_sub, "outgoing", epsilon=1e-45)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out_sub))))
    chex.assert_trees_all_close(
        out_sub.astype(jnp.float32),
        exp_sub.astype(jnp.float32),
        atol=0.15,
        rtol=0.05,
    )

  def test_single_axis_and_clamped_config_overrides(self):
    p = _make_seeded_params(904, 132, 32, 32, 32)
    out_bj128 = _jit_op_cfg_fwd(p, "outgoing", mtpu_tri_mul.Config(block_j=128))
    exp_p, _ = _jit_ref_fwd_with_res(p, "outgoing")
    chex.assert_trees_all_close(out_bj128, exp_p, atol=0.30, rtol=0.05)

    p_spec = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), p)
    op_128 = _make_op(dict(block_j=128))
    op_256 = _make_op(dict(block_j=256))
    hlo_128 = (
        jax.jit(lambda kw: op_128(**kw, triangle_type="outgoing"))
        .lower(p_spec)
        .as_text(dialect="hlo")
    )
    hlo_256 = (
        jax.jit(lambda kw: op_256(**kw, triangle_type="outgoing"))
        .lower(p_spec)
        .as_text(dialect="hlo")
    )
    self.assertNotEqual(hlo_128, hlo_256)

    op_epi_m = _make_op(dict(block_i=16, block_j=256, epilogue_block_m=128))
    op_epi_ij = _make_op(
        dict(block_i=16, block_j=256, epilogue_block_i=16, epilogue_block_j=256)
    )
    hlo_em = (
        jax.jit(lambda kw: op_epi_m(**kw, triangle_type="outgoing"))
        .lower(p_spec)
        .as_text(dialect="hlo")
    )
    hlo_eij = (
        jax.jit(lambda kw: op_epi_ij(**kw, triangle_type="outgoing"))
        .lower(p_spec)
        .as_text(dialect="hlo")
    )
    self.assertNotEqual(hlo_em, hlo_eij)

    p36 = _make_seeded_params(924, 36, 32, 32, 32)
    self._check_fwd_and_9_grads(
        p36,
        "outgoing",
        fwd_cfg=dict(block_i=8, block_j=1024),
        vjp_cfg=dict(epilogue_block_i=8, block_j=1024),
        fwd_atol=0.30,
        grad_atol=0.50,
    )

  def test_vmap_modes_and_nested_batching(self):
    # 1. Prime B=3 all-10-arrays batched on unaligned N=20.
    items = [_make_seeded_params(1021 + i, 20, 16, 32, 16) for i in range(3)]
    batched = {k: jnp.stack([it[k] for it in items], axis=0) for k in items[0]}
    out_tpu = jax.jit(
        jax.vmap(lambda kw: self._op(**kw, triangle_type="outgoing"))
    )(batched)
    out_xla = jax.jit(
        jax.vmap(
            lambda kw: self._ref_op(
                **kw, triangle_type="outgoing", implementation="xla"
            )
        )
    )(batched)
    chex.assert_trees_all_close(
        out_tpu.astype(jnp.float32),
        out_xla.astype(jnp.float32),
        atol=0.30,
        rtol=0.05,
    )

    # 2. Nested 2-level vmap (2, 3, 20, 20, 16) forward + value_and_grad.
    p0 = _make_seeded_params(1023, 20, 16, 16, 16)
    sw = {k: p0[k] for k in _GRAD_KEYS if k != "x"}
    x2 = _make_random_array((2, 3, 20, 20, 16), jnp.bfloat16, seed=10231)
    m2 = jnp.asarray(np.random.default_rng(10232).random((2, 3, 20, 20)) < 0.5)
    t2 = _make_random_array((2, 3, 20, 20, 16), jnp.bfloat16, seed=10233)

    def nested_loss(use_ref: bool, x_in, weights):
      def leaf(xi, mi):
        if use_ref:
          return self._ref_op(
              x=xi,
              mask=mi,
              **weights,
              triangle_type="outgoing",
              implementation="xla",
          )
        return self._op(x=xi, mask=mi, **weights, triangle_type="outgoing")

      out = jax.vmap(jax.vmap(leaf))(x_in, m2)
      loss = jnp.sum(out.astype(jnp.float32) * t2.astype(jnp.float32)) / 120.0
      return loss, out

    nested_tpu = jax.jit(
        jax.value_and_grad(
            functools.partial(nested_loss, False), argnums=(0, 1), has_aux=True
        )
    )
    nested_xla = jax.jit(
        jax.value_and_grad(
            functools.partial(nested_loss, True), argnums=(0, 1), has_aux=True
        )
    )
    (_, no_tpu), ng_tpu = nested_tpu(x2, sw)
    (_, no_xla), ng_xla = nested_xla(x2, sw)
    chex.assert_trees_all_close(
        no_tpu.astype(jnp.float32),
        no_xla.astype(jnp.float32),
        atol=0.15,
        rtol=0.05,
    )
    chex.assert_trees_all_close(
        jax.tree.map(lambda a: a.astype(jnp.float32), ng_tpu),
        jax.tree.map(lambda a: a.astype(jnp.float32), ng_xla),
        atol=0.50,
        rtol=0.05,
    )

  def test_default_device_and_supported_on(self):
    vjp_op = mtpu_tri_mul_vjp.PallasMosaicTpuTriangleMultiplicationVjp()
    tpu_dev = jax.devices("tpu")[0]
    cpu_dev = jax.devices("cpu")[0]
    self.assertTrue(self._op.supported_on(tpu_dev))
    self.assertTrue(vjp_op.supported_on(tpu_dev))
    self.assertFalse(self._op.supported_on(cpu_dev))
    self.assertFalse(vjp_op.supported_on(cpu_dev))

    items = [
        jax.device_put(_make_seeded_params(1024 + i, 32, 32, 32, 32), tpu_dev)
        for i in range(2)
    ]
    batched = {k: jnp.stack([it[k] for it in items], axis=0) for k in items[0]}
    with jax.default_device(cpu_dev):
      out_eager = api.triangle_multiplication(
          **items[0], triangle_type="outgoing", implementation=None
      )
      self.assertEqual(out_eager.devices(), {tpu_dev})
      with self.assertRaises(NotImplementedError):
        jax.jit(
            jax.vmap(
                lambda kw: api.triangle_multiplication(
                    **kw, triangle_type="outgoing", implementation="mosaic"
                )
            )
        ).lower(batched)

    hlo_vmap = (
        jax.jit(
            jax.vmap(
                lambda kw: api.triangle_multiplication(
                    **kw, triangle_type="outgoing", implementation="mosaic"
                )
            )
        )
        .lower(batched)
        .as_text(dialect="hlo")
    )
    self.assertIn("tpu_custom_call", hlo_vmap)

    p_cpu = jax.device_put(items[0], cpu_dev)
    with self.assertRaises(NotImplementedError):
      self._op(**p_cpu, triangle_type="outgoing")
    with self.assertRaises(NotImplementedError):
      api.triangle_multiplication(
          **p_cpu, triangle_type="outgoing", implementation="mosaic"
      )
    with self.assertRaises(NotImplementedError):
      api.triangle_multiplication(
          **p_cpu, triangle_type="outgoing", implementation=("mosaic",)
      )
    out_cpu = api.triangle_multiplication(
        **p_cpu, triangle_type="outgoing", implementation=None
    )
    self.assertEqual(out_cpu.devices(), {cpu_dev})
    self.assertEqual(out_cpu.shape, (32, 32, 32))
    chex.assert_trees_all_close(
        jax.device_put(out_eager, cpu_dev), out_cpu, atol=0.15, rtol=0.05
    )

  def test_vmem_safe_shapes_and_heuristics(self):
    vjp_op = mtpu_tri_mul_vjp.PallasMosaicTpuTriangleMultiplicationVjp()
    for n in (128, 384, 512, 640, 1024, 2048):
      for c in (16, 32, 64, 128):
        spec: dict[str, Any] = dict(
            x=jax.ShapeDtypeStruct((n, n, c), jnp.bfloat16),
            mask=jax.ShapeDtypeStruct((n, n), jnp.bool_),
            projection_in_weights=jax.ShapeDtypeStruct((c, 2, c), jnp.bfloat16),
            gate_in_weights=jax.ShapeDtypeStruct((c, 2, c), jnp.bfloat16),
            projection_out_weights=jax.ShapeDtypeStruct((c, c), jnp.bfloat16),
            gate_out_weights=jax.ShapeDtypeStruct((c, c), jnp.bfloat16),
            layernorm_in_scale=jax.ShapeDtypeStruct((c,), jnp.bfloat16),
            layernorm_in_offset=jax.ShapeDtypeStruct((c,), jnp.bfloat16),
            layernorm_out_scale=jax.ShapeDtypeStruct((c,), jnp.bfloat16),
            layernorm_out_offset=jax.ShapeDtypeStruct((c,), jnp.bfloat16),
        )
        ba = self._op.bind(
            **spec, triangle_type="outgoing", return_residuals=True
        )
        self.assertIn(ba.heuristics_config, ba.autotuning_configs)
        vjp_ba = vjp_op.bind(**ba.vjp_arg_spec)
        self.assertIn(vjp_ba.heuristics_config, vjp_ba.autotuning_configs)

    for n, c, h, d in (
        (640, 64, 64, 64),
        (512, 16, 128, 16),
    ):
      low_spec: dict[str, Any] = dict(
          x=jax.ShapeDtypeStruct((n, n, c), jnp.bfloat16),
          mask=jax.ShapeDtypeStruct((n, n), jnp.bool_),
          projection_in_weights=jax.ShapeDtypeStruct((c, 2, h), jnp.bfloat16),
          gate_in_weights=jax.ShapeDtypeStruct((c, 2, h), jnp.bfloat16),
          projection_out_weights=jax.ShapeDtypeStruct((h, d), jnp.bfloat16),
          gate_out_weights=jax.ShapeDtypeStruct((c, d), jnp.bfloat16),
          layernorm_in_scale=jax.ShapeDtypeStruct((c,), jnp.bfloat16),
          layernorm_in_offset=jax.ShapeDtypeStruct((c,), jnp.bfloat16),
          layernorm_out_scale=jax.ShapeDtypeStruct((h,), jnp.bfloat16),
          layernorm_out_offset=jax.ShapeDtypeStruct((h,), jnp.bfloat16),
      )
      hlo = (
          jax.jit(lambda kw: self._op(**kw, triangle_type="outgoing"))
          .lower(low_spec)
          .as_text(dialect="hlo")
      )
      self.assertIn("tpu_custom_call", hlo)

    p_400 = _make_seeded_params(1122, 260, 128, 128, 128)
    out_400 = _jit_op_fwd(p_400, "outgoing")
    ref_400, _ = _jit_ref_fwd_with_res(p_400, "outgoing")
    chex.assert_trees_all_close(out_400, ref_400, atol=0.45, rtol=0.05)

  # ---------------------------------------------------------------------------
  # Tier 3 Cross-Feature Pairwise Regression Tests (24 Methods)
  # ---------------------------------------------------------------------------

  def test_vmem_stress_n512_c128_outgoing(self):
    p = _make_seeded_params(3001, 512, 128, 128, 128)
    v1d = jnp.arange(512) < 480
    mask = v1d[:, None] & v1d[None, :]
    p["mask"] = mask
    self._check_fwd_and_9_grads(
        p,
        "outgoing",
        use_api=True,
        api_impl="mosaic",
        valid_mask=mask,
        fwd_atol=0.45,
        grad_atol=0.60,
    )

  def test_vmem_stress_n512_c128_incoming(self):
    p = _make_seeded_params(3002, 512, 128, 128, 128, mask_prob=0.2)
    p["mask"] = p["mask"].at[256:, 256:].set(False)
    self._check_fwd_and_9_grads(p, "incoming", fwd_atol=0.45, grad_atol=0.60)

  def test_vmem_stress_n384_c64(self):
    p = _make_seeded_params(3003, 384, 64, 64, 64)
    self._check_fwd_and_9_grads(
        p,
        "outgoing",
        use_api=True,
        api_impl=None,
        fwd_atol=0.45,
        grad_atol=0.50,
    )

  def test_vmem_stress_n512_asym(self):
    p = _make_seeded_params(3004, 512, 128, 64, 128)
    v1d = jnp.arange(512) < 500
    p["mask"] = v1d[:, None] & v1d[None, :]
    out = _jit_op_fwd(p, "incoming")
    ref_out, _ = _jit_ref_fwd_with_res(p, "incoming")
    chex.assert_trees_all_close(out, ref_out, atol=0.45, rtol=0.05)

  def test_asym_expand_16_128_32(self):
    p = _make_seeded_params(3005, 67, 16, 128, 32)
    self._check_fwd_and_9_grads(
        p,
        "outgoing",
        fwd_cfg=dict(block_i=8),
        vjp_cfg=dict(block_i=8),
        fwd_atol=0.30,
        grad_atol=0.50,
    )

  def test_asym_shrink_128_16_64(self):
    p = _make_seeded_params(3006, 67, 128, 16, 64, mask_prob=0.25)
    self._check_fwd_and_9_grads(
        p,
        "incoming",
        fwd_cfg=dict(block_j=128),
        vjp_cfg=dict(block_m=128, epilogue_block_j=128),
        fwd_atol=0.30,
        grad_atol=0.50,
    )

  def test_cfg_no_shadow_epilogue(self):
    p = _make_seeded_params(3007, 67, 32, 32, 16)
    v1d = jnp.arange(67) < 55
    mask = v1d[:, None] & v1d[None, :]
    p["mask"] = mask
    self._check_fwd_and_9_grads(
        p,
        "outgoing",
        fwd_cfg=dict(block_i=8, epilogue_block_m=128),
        vjp_cfg=dict(block_i=8, epilogue_block_m=128),
        valid_mask=mask,
        fwd_atol=0.30,
        grad_atol=0.50,
    )

  def test_cfg_swapped_2d_pro_epi(self):
    p = _make_seeded_params(3008, 135, 32, 16, 32, mask_prob=0.15)
    self._check_fwd_and_9_grads(
        p,
        "incoming",
        fwd_cfg=dict(
            block_i=16, block_j=128, epilogue_block_i=8, epilogue_block_j=256
        ),
        vjp_cfg=dict(
            block_i=8, block_j=256, epilogue_block_i=16, epilogue_block_j=128
        ),
        fwd_atol=0.35,
        grad_atol=0.50,
    )

  def test_vmap_shared_weights_n70(self):
    p0 = _make_seeded_params(3009, 36, 32, 32, 64)
    sw = {k: p0[k] for k in _GRAD_KEYS if k != "x"}
    xb = _make_random_array((3, 36, 36, 32), jnp.bfloat16, seed=30091)
    mb = jnp.stack(
        [
            (jnp.arange(36) < l)[:, None] & (jnp.arange(36) < l)[None, :]
            for l in (36, 27, 11)
        ],
        axis=0,
    )
    target = _make_random_array((3, 36, 36, 64), jnp.bfloat16, seed=30092)

    def run_vmap(impl: Literal["mosaic", "xla"], x_in, w_in):
      out_b = jax.vmap(
          lambda xi, mi: api.triangle_multiplication(
              x=xi,
              mask=mi,
              **w_in,
              triangle_type="outgoing",
              implementation=impl,
          )
      )(x_in, mb)
      out_eff = jnp.where(mb[:, :, :, None], out_b, 0.0)
      loss = (
          jnp.sum(out_eff.astype(jnp.float32) * target.astype(jnp.float32))
          / 108.0
      )
      return loss, out_eff

    vg_tpu = jax.jit(
        jax.value_and_grad(
            functools.partial(run_vmap, "mosaic"), argnums=(0, 1), has_aux=True
        )
    )
    vg_xla = jax.jit(
        jax.value_and_grad(
            functools.partial(run_vmap, "xla"), argnums=(0, 1), has_aux=True
        )
    )
    (_, o_tpu), g_tpu = vg_tpu(xb, sw)
    (_, o_xla), g_xla = vg_xla(xb, sw)
    chex.assert_trees_all_close(
        o_tpu.astype(jnp.float32),
        o_xla.astype(jnp.float32),
        atol=0.30,
        rtol=0.05,
    )
    chex.assert_trees_all_close(
        jax.tree.map(lambda a: a.astype(jnp.float32), g_tpu),
        jax.tree.map(lambda a: a.astype(jnp.float32), g_xla),
        atol=0.50,
        rtol=0.05,
    )

  def test_vmap_all_batched_n37(self):
    items = [_make_seeded_params(3010 + i, 37, 64, 64, 64) for i in range(2)]
    for i, (l, s) in enumerate(((29, jnp.nan), (17, jnp.inf))):
      m = (jnp.arange(37) < l)[:, None] & (jnp.arange(37) < l)[None, :]
      items[i]["mask"] = m
      items[i]["x"] = jnp.where(m[:, :, None], items[i]["x"], s)
    items[1]["x"] = items[1]["x"].at[30:, 30:, :].set(-jnp.inf)
    batched = {k: jnp.stack([it[k] for it in items], axis=0) for k in items[0]}
    mb = batched["mask"]
    diff_b = {k: batched[k] for k in _GRAD_KEYS}
    diff_clean = dict(diff_b, x=jnp.where(mb[:, :, :, None], diff_b["x"], 0.0))
    target = jnp.where(
        mb[:, :, :, None],
        _make_random_array((2, 37, 37, 64), jnp.bfloat16, seed=30109),
        0.0,
    )

    def run_all(use_ref: bool, dp_b):
      def single(dp_i, m_i):
        if use_ref:
          return self._ref_op(
              **dp_i,
              mask=m_i,
              triangle_type="incoming",
              implementation="xla",
          )
        return self._op(**dp_i, mask=m_i, triangle_type="incoming")

      out_b = jax.vmap(single)(dp_b, mb)
      out_eff = jnp.where(mb[:, :, :, None], out_b, 0.0)
      loss = (
          jnp.sum(out_eff.astype(jnp.float32) * target.astype(jnp.float32))
          / 74.0
      )
      return loss, out_eff

    vg_tpu = jax.jit(
        jax.value_and_grad(functools.partial(run_all, False), has_aux=True)
    )
    vg_xla = jax.jit(
        jax.value_and_grad(functools.partial(run_all, True), has_aux=True)
    )
    (_, o_tpu), g_tpu = vg_tpu(diff_b)
    (_, o_xla), g_xla = vg_xla(diff_clean)
    self.assertTrue(bool(jnp.all(jnp.isfinite(o_tpu))))
    for k in _GRAD_KEYS:
      self.assertTrue(bool(jnp.all(jnp.isfinite(g_tpu[k]))))
      chex.assert_trees_all_close(
          g_tpu[k].astype(jnp.float32),
          g_xla[k].astype(jnp.float32),
          atol=0.50,
          rtol=0.05,
      )
    chex.assert_trees_all_close(
        o_tpu.astype(jnp.float32),
        o_xla.astype(jnp.float32),
        atol=0.30,
        rtol=0.05,
    )

  def test_vmap_axis1_return_residuals(self):
    n = 36
    items = [_make_seeded_params(3011 + i, n, 64, 128, 32) for i in range(2)]
    x_ax1 = jnp.stack([it["x"] for it in items], axis=1)
    mb = jnp.stack([it["mask"] for it in items], axis=0)
    sw = {k: items[0][k] for k in _GRAD_KEYS if k != "x"}
    op = _make_op(dict(block_m=128, epilogue_block_i=8))
    out_b, res_b = jax.jit(
        jax.vmap(
            lambda xi, mi: op(
                x=xi,
                mask=mi,
                **sw,
                triangle_type="outgoing",
                return_residuals=True,
            ),
            in_axes=(1, 0),
            out_axes=0,
        )
    )(x_ax1, mb)
    ref_o, ref_r = jax.jit(
        jax.vmap(
            lambda xi, mi: self._ref_op(
                x=xi,
                mask=mi,
                **sw,
                triangle_type="outgoing",
                return_residuals=True,
                implementation="xla",
            ),
            in_axes=(1, 0),
            out_axes=0,
        )
    )(x_ax1, mb)
    self.assertEqual(out_b.shape, (2, 36, 36, 32))
    self.assertEqual(res_b[2].shape, (2, 128, 36, 36))
    chex.assert_trees_all_close(out_b, ref_o, atol=0.30, rtol=0.05)
    chex.assert_trees_all_close(res_b[0], ref_r[0], atol=0.15, rtol=0.05)
    chex.assert_trees_all_close(res_b[1], ref_r[1], atol=0.15, rtol=0.05)
    chex.assert_trees_all_close(
        res_b[2], jnp.moveaxis(ref_r[2], -1, 1), atol=7.0, rtol=0.05
    )

  def test_cpu_default_device_tpu_arrays(self):
    tpu_dev = jax.devices("tpu")[0]
    cpu_dev = jax.devices("cpu")[0]
    p = _make_seeded_params(3012, 35, 16, 64, 32)
    v1d = jnp.arange(35) < 25
    mask = v1d[:, None] & v1d[None, :]
    p["mask"] = mask
    p_tpu = jax.device_put(p, tpu_dev)
    vjp_op = mtpu_tri_mul_vjp.PallasMosaicTpuTriangleMultiplicationVjp()
    with jax.default_device(cpu_dev):
      out_m_eager = api.triangle_multiplication(
          **p_tpu, triangle_type="incoming", implementation="mosaic"
      )
      self.assertEqual(out_m_eager.devices(), {tpu_dev})
      out_n_eager = api.triangle_multiplication(
          **p_tpu, triangle_type="incoming", implementation=None
      )
      self.assertEqual(out_n_eager.devices(), {tpu_dev})
      chex.assert_trees_all_equal(out_m_eager, out_n_eager)

      out_seq_eager = api.triangle_multiplication(
          **p_tpu, triangle_type="incoming", implementation=("mosaic", "xla")
      )
      self.assertEqual(out_seq_eager.devices(), {tpu_dev})
      chex.assert_trees_all_equal(out_m_eager, out_seq_eager)

      out_res, res_tpu = self._op(
          **p_tpu,
          triangle_type="incoming",
          precision=_DEFAULT_PREC,
          return_residuals=True,
      )
      self.assertEqual(out_res.devices(), {tpu_dev})
      grads_vjp_eager = vjp_op(
          res_tpu,
          out_res,
          out_res,
          **p_tpu,
          triangle_type="incoming",
          precision=_DEFAULT_PREC,
          epsilon=1e-6,
      )
      self.assertEqual(grads_vjp_eager["x"].devices(), {tpu_dev})

      def _lower_jit_hlo(
          impl: api.Implementation | Sequence[api.Implementation] | None,
      ) -> str:
        return (
            jax.jit(
                lambda kw: api.triangle_multiplication(
                    **kw, triangle_type="incoming", implementation=impl
                )
            )
            .lower(p_tpu)
            .as_text(dialect="hlo")
        )

      self.assertNotIn("tpu_custom_call", _lower_jit_hlo(None))
      self.assertNotIn("tpu_custom_call", _lower_jit_hlo(("mosaic", "xla")))

      def _call_jit(
          impl: api.Implementation | Sequence[api.Implementation],
      ) -> None:
        jax.jit(
            lambda kw: api.triangle_multiplication(
                **kw, triangle_type="incoming", implementation=impl
            )
        )(p_tpu)

      with self.assertRaises(NotImplementedError):
        _call_jit("mosaic")
      with self.assertRaises(NotImplementedError):
        _call_jit(("mosaic",))
    ref_n, _ = _jit_ref_fwd_with_res(p_tpu, "incoming")
    chex.assert_trees_all_close(
        out_n_eager[:25, :25], ref_n[:25, :25], atol=0.30, rtol=0.05
    )

  def test_zero_eps_all_false_mask(self):
    p = _make_seeded_params(3013, 37, 64, 64, 32)
    p["mask"] = jnp.zeros((37, 37), dtype=jnp.bool_)
    _, grads = self._check_fwd_and_9_grads(
        p, "outgoing", epsilon=0.0, fwd_atol=0.30, grad_atol=0.50
    )
    self.assertTrue(bool(jnp.all(grads["projection_in_weights"] == 0.0)))
    self.assertTrue(bool(jnp.all(grads["gate_in_weights"] == 0.0)))
    self.assertTrue(bool(jnp.all(grads["layernorm_out_scale"] == 0.0)))

  def test_zero_eps_const_x_var0(self):
    p = _make_seeded_params(3014, 37, 64, 32, 16)
    v1d = jnp.arange(37) < 25
    mask = v1d[:, None] & v1d[None, :]
    c_ij = _make_random_array((37, 37, 1), jnp.bfloat16, seed=30140)
    x_const = jnp.broadcast_to(c_ij, (37, 37, 64))
    p_inf = dict(p, mask=mask, x=jnp.where(mask[:, :, None], x_const, -jnp.inf))
    out_inf, _ = self._check_fwd_and_9_grads(
        p_inf,
        "incoming",
        epsilon=0.0,
        valid_mask=mask,
        fwd_atol=0.15,
        grad_atol=0.50,
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(out_inf[:25, :25]))))

  def test_large_x_mean_shift_1e4(self):
    p = _make_seeded_params(3015, 37, 64, 32, 32)
    v1d = jnp.arange(37) < 29
    mask = v1d[:, None] & v1d[None, :]
    p["mask"] = mask
    p["x"] = (p["x"].astype(jnp.float32) * 128.0 + 1e4).astype(jnp.bfloat16)
    self._check_fwd_and_9_grads(
        p,
        "outgoing",
        use_api=True,
        api_impl="mosaic",
        valid_mask=mask,
        fwd_atol=0.30,
        grad_atol=0.50,
    )

  def test_large_act_mean_shift_2h(self):
    p = _make_seeded_params(3016, 67, 128, 128, 128, mask_prob=0.8)
    p["layernorm_in_offset"] = (
        p["layernorm_in_offset"].astype(jnp.float32) + 20.0
    ).astype(jnp.bfloat16)
    p["projection_in_weights"] = (
        jnp.abs(p["projection_in_weights"].astype(jnp.float32)) + 0.5
    ).astype(jnp.bfloat16)
    self._check_fwd_and_9_grads(
        p,
        "incoming",
        fwd_cfg=dict(block_m=128),
        vjp_cfg=dict(block_m=128, epilogue_block_m=128),
        fwd_atol=0.35,
        fwd_rtol=0.05,
        grad_atol=0.75,
        grad_rtol=0.06,
    )

  def test_unmasked_nan_propagates_in_batch0(self):
    items = [
        _make_seeded_params(3017 + i, 32, 32, 32, 32, mask_prob=1.0)
        for i in range(2)
    ]
    items[0]["mask"] = jnp.ones((32, 32), dtype=jnp.bool_)
    items[1]["mask"] = jnp.ones((32, 32), dtype=jnp.bool_)
    items[0]["x"] = items[0]["x"].at[5, 5, 0].set(jnp.nan)
    batched = {k: jnp.stack([it[k] for it in items], axis=0) for k in items[0]}
    mb = batched["mask"]
    diff_b = {k: batched[k] for k in _GRAD_KEYS}
    target = jnp.ones((2, 32, 32, 32), dtype=jnp.bfloat16) / 32.0

    def run_b(use_ref: bool, dp_b):
      def one(dp_i, m_i):
        if use_ref:
          return self._ref_op(
              **dp_i,
              mask=m_i,
              triangle_type="outgoing",
              implementation="xla",
          )
        return self._op(**dp_i, mask=m_i, triangle_type="outgoing")

      out_b = jax.vmap(one)(dp_b, mb)
      loss = jnp.sum(out_b.astype(jnp.float32) * target.astype(jnp.float32))
      return loss, out_b

    vg_tpu = jax.jit(
        jax.value_and_grad(functools.partial(run_b, False), has_aux=True)
    )
    vg_xla = jax.jit(
        jax.value_and_grad(functools.partial(run_b, True), has_aux=True)
    )
    (_, out_tpu), g_tpu = vg_tpu(diff_b)
    (_, out_xla), g_xla = vg_xla(diff_b)
    self.assertTrue(bool(jnp.any(jnp.isnan(out_tpu[0]))))
    self.assertTrue(bool(jnp.any(jnp.isnan(g_tpu["x"][0]))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(out_tpu[1]))))
    chex.assert_trees_all_close(
        out_tpu[1].astype(jnp.float32),
        out_xla[1].astype(jnp.float32),
        atol=0.30,
        rtol=0.05,
    )
    for k in _GRAD_KEYS:
      self.assertTrue(bool(jnp.all(jnp.isfinite(g_tpu[k][1]))))
      chex.assert_trees_all_close(
          g_tpu[k][1].astype(jnp.float32),
          g_xla[k][1].astype(jnp.float32),
          atol=0.50,
          rtol=0.05,
      )

  def test_zero_cotangent_everywhere(self):
    p = _make_seeded_params(3018, 36, 32, 64, 32)
    v1d = jnp.arange(36) < 28
    mask = v1d[:, None] & v1d[None, :]
    p["x"] = (
        jnp.where(mask[:, :, None], p["x"], jnp.nan)
        .at[32:, 32:, :]
        .set(jnp.inf)
    )
    dout_zero = jnp.zeros((36, 36, 32), dtype=jnp.bfloat16)
    _, grads = _jit_op_fwd_and_vjp(
        {k: p[k] for k in _GRAD_KEYS}, mask, dout_zero, "incoming"
    )
    for g in grads.values():
      self.assertTrue(bool(jnp.all(g == 0.0)))
      self.assertFalse(bool(jnp.any(jnp.isnan(g))))

  def test_diagonal_mask_prime_n19(self):
    p = _make_seeded_params(3019, 19, 16, 32, 64)
    p["mask"] = jnp.eye(19, dtype=jnp.bool_)
    out, res = _jit_op_fwd_with_res(p, "outgoing")
    ref_out, ref_res = _jit_ref_fwd_with_res(p, "outgoing")
    chex.assert_trees_all_close(out, ref_out, atol=0.15, rtol=0.05)
    chex.assert_trees_all_close(res[0], ref_res[0], atol=0.15, rtol=0.05)
    chex.assert_trees_all_close(res[1], ref_res[1], atol=0.15, rtol=0.05)
    chex.assert_trees_all_close(
        res[2], jnp.moveaxis(ref_res[2], -1, 0), atol=3.8, rtol=0.05
    )
    dout = _make_random_array((19, 19, 64), jnp.bfloat16, seed=30190) / 19.0
    grads_vjp = _jit_standalone_vjp(res, out, dout, p, "outgoing")
    _, grads_xla = _jit_ref_fwd_and_vjp(
        {k: p[k] for k in _GRAD_KEYS}, p["mask"], dout, "outgoing"
    )
    for k in _GRAD_KEYS:
      chex.assert_trees_all_close(
          grads_vjp[k].astype(jnp.float32),
          grads_xla[k].astype(jnp.float32),
          atol=0.50,
          rtol=0.05,
      )

  def test_strict_upper_tri_mask(self):
    p = _make_seeded_params(3020, 35, 16, 32, 16)
    p["mask"] = jnp.triu(jnp.ones((35, 35), dtype=jnp.bool_), k=1)
    out_out, _ = self._check_fwd_and_9_grads(
        p,
        "outgoing",
        use_api=True,
        api_impl="mosaic",
        epsilon=1e-5,
        fwd_atol=0.15,
        grad_atol=0.50,
    )
    out_in = _jit_op_fwd(p, "incoming", epsilon=1e-5)
    exp_in, _ = _jit_ref_fwd_with_res(p, "incoming", epsilon=1e-5)
    chex.assert_trees_all_close(out_in, exp_in, atol=0.15, rtol=0.05)
    diff_max = jnp.max(
        jnp.abs(out_out.astype(jnp.float32) - out_in.astype(jnp.float32))
    )
    self.assertGreater(float(diff_max), 1e-2)

  def test_cross_tile_boundary_active_cell(self):
    p = _make_seeded_params(3021, 129, 32, 32, 64)
    mask = (
        jnp.zeros((129, 129), dtype=jnp.bool_)
        .at[0, 0]
        .set(True)
        .at[0, 128]
        .set(True)
        .at[128, 128]
        .set(True)
    )
    p["mask"] = mask
    p["x"] = jnp.where(mask[:, :, None], p["x"], jnp.nan)
    self._check_fwd_and_9_grads(
        p,
        "outgoing",
        fwd_cfg=dict(block_i=16, block_j=128),
        vjp_cfg=dict(block_i=16, block_j=128),
        valid_mask=mask,
        fwd_atol=0.15,
        grad_atol=0.50,
    )

  def test_min_channel_16_16_16(self):
    p = _make_seeded_params(3022, 67, 16, 16, 16)
    self._check_fwd_and_9_grads(
        p,
        "incoming",
        fwd_cfg=dict(epilogue_block_i=16),
        vjp_cfg=dict(epilogue_block_i=8),
        epsilon=1e-12,
        fwd_atol=0.35,
        grad_atol=0.50,
    )

  def test_api_fallback_seq_unsupported(self):
    items_48 = [_make_seeded_params(3071 + i, 32, 48, 48, 32) for i in range(2)]
    batched_48 = {
        k: jnp.stack([it[k] for it in items_48], axis=0) for k in items_48[0]
    }
    mb_48 = batched_48["mask"]
    diff_48 = {k: batched_48[k] for k in _GRAD_KEYS}
    target_48 = _make_random_array((2, 32, 32, 32), jnp.bfloat16, seed=30230)

    def loss_48(impl_arg, dp_b):
      out_b = jax.vmap(
          lambda dp_i, m_i: api.triangle_multiplication(
              **dp_i,
              mask=m_i,
              triangle_type="outgoing",
              implementation=impl_arg,
          )
      )(dp_b, mb_48)
      loss = (
          jnp.sum(out_b.astype(jnp.float32) * target_48.astype(jnp.float32))
          / 64.0
      )
      return loss, out_b

    (_, o_fb), g_fb = jax.jit(
        jax.value_and_grad(
            functools.partial(loss_48, ("mosaic", "xla")), has_aux=True
        )
    )(diff_48)
    (_, o_xla), g_xla = jax.jit(
        jax.value_and_grad(functools.partial(loss_48, "xla"), has_aux=True)
    )(diff_48)
    o_none = jax.jit(
        jax.vmap(
            lambda dp_i, m_i: api.triangle_multiplication(
                **dp_i,
                mask=m_i,
                triangle_type="outgoing",
                implementation=None,
            )
        )
    )(diff_48, mb_48)
    o_xla_fwd = jax.jit(
        jax.vmap(
            lambda dp_i, m_i: api.triangle_multiplication(
                **dp_i,
                mask=m_i,
                triangle_type="outgoing",
                implementation="xla",
            )
        )
    )(diff_48, mb_48)
    chex.assert_trees_all_close(
        o_fb.astype(jnp.float32),
        o_xla.astype(jnp.float32),
        atol=1e-4,
        rtol=1e-4,
    )
    chex.assert_trees_all_close(
        o_none.astype(jnp.float32),
        o_xla_fwd.astype(jnp.float32),
        atol=1e-4,
        rtol=1e-4,
    )
    for k in _GRAD_KEYS:
      chex.assert_trees_all_close(
          g_fb[k].astype(jnp.float32),
          g_xla[k].astype(jnp.float32),
          atol=1e-3,
          rtol=1e-3,
      )

    items_hp = [_make_seeded_params(3087 + i, 32, 32, 32, 32) for i in range(2)]
    batched_hp = {
        k: jnp.stack([it[k] for it in items_hp], axis=0) for k in items_hp[0]
    }
    prec_hi = jax.lax.Precision.HIGHEST
    o_hp_ref = jax.jit(
        jax.vmap(
            lambda kw: api.triangle_multiplication(
                **kw,
                triangle_type="outgoing",
                precision=prec_hi,
                implementation="xla",
            )
        )
    )(batched_hp)
    o_hp_fb = jax.jit(
        jax.vmap(
            lambda kw: api.triangle_multiplication(
                **kw,
                triangle_type="outgoing",
                precision=prec_hi,
                implementation=None,
            )
        )
    )(batched_hp)
    chex.assert_trees_all_close(
        o_hp_fb.astype(jnp.float32),
        o_hp_ref.astype(jnp.float32),
        atol=1e-4,
        rtol=1e-4,
    )

  def test_negative_pairwise_invalid(self):
    vjp_op = mtpu_tri_mul_vjp.PallasMosaicTpuTriangleMultiplicationVjp()
    for bad_cfg in (
        dict(block_i=12, epilogue_block_m=256),
        dict(block_i=16, block_j=192),
        dict(epilogue_block_i=20, block_m=256),
        dict(epilogue_block_j=320, block_m=256),
    ):
      with self.assertRaises(ValueError):
        mtpu_tri_mul.Config(**bad_cfg)
    p = _make_seeded_params(3024, 64, 32, 32, 64)
    out = jnp.zeros((64, 64, 64), jnp.bfloat16)
    bad_res = (
        jnp.zeros((32, 64, 64), jnp.bfloat16),
        jnp.zeros((32, 64, 64), jnp.bfloat16),
        jnp.zeros((64, 64, 32), jnp.bfloat16),
    )
    with self.assertRaises(NotImplementedError):
      vjp_op(
          bad_res,
          out,
          out,
          **p,
          triangle_type="outgoing",
          precision=_DEFAULT_PREC,
          epsilon=1e-6,
      )
    with _disable_jaxtyping():
      with self.assertRaises(NotImplementedError):
        vjp_op(
            cast(Any, bad_res[:2]),
            out,
            out,
            **p,
            triangle_type="outgoing",
            precision=_DEFAULT_PREC,
            epsilon=1e-6,
        )
    with self.assertRaises(NotImplementedError):
      self._op(**p, triangle_type="outgoing", epsilon=-1e-6)

  # ---------------------------------------------------------------------------
  # Tier 4 AlphaFold Pairformer Scenario Regression Tests (9 Methods)
  # ---------------------------------------------------------------------------

  def _run_cropped_pairformer(
      self,
      seed: int,
      n: int,
      l_valid: int,
      ch: int,
      pad_val: jax.Array,
      *,
      impl_tpu: Any = "mosaic",
      fwd_atol: float = 0.45,
      grad_atol: float = 0.50,
      grad_rtol: float = 0.05,
  ) -> None:
    p_z = _make_seeded_params(seed, n, ch, ch, ch, scale_mode="lecun")
    w_out = {
        k: v
        for k, v in _make_seeded_params(
            seed + 1, 8, ch, ch, ch, scale_mode="lecun"
        ).items()
        if k not in ("x", "mask")
    }
    w_in = {
        k: v
        for k, v in _make_seeded_params(
            seed + 2, 8, ch, ch, ch, scale_mode="lecun"
        ).items()
        if k not in ("x", "mask")
    }
    v1d = jnp.arange(n) < l_valid
    mask = v1d[:, None] & v1d[None, :]
    z0_clean = jnp.where(mask[:, :, None], p_z["x"], 0.0)
    z0_pad = jnp.where(mask[:, :, None], p_z["x"], pad_val)
    target = jnp.where(
        mask[:, :, None],
        _make_random_array((n, n, ch), jnp.bfloat16, seed=seed + 9),
        0.0,
    )

    inv_denom = jnp.float32(1.0 / l_valid)
    (val_m, z2_m), (dz0_m, dwo_m, dwi_m) = _jit_cropped_pairformer_vg(
        z0_pad, w_out, w_in, mask, target, inv_denom, impl=impl_tpu
    )
    (_, z2_x), (dz0_x, dwo_x, dwi_x) = _jit_cropped_pairformer_vg(
        z0_clean, w_out, w_in, mask, target, inv_denom, impl="xla"
    )

    self.assertTrue(bool(jnp.isfinite(val_m)))
    for leaf in jax.tree.leaves((z2_m, dz0_m, dwo_m, dwi_m)):
      self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))
    self.assertTrue(
        bool(jnp.all(jnp.where(mask[:, :, None], 0.0, dz0_m) == 0.0))
    )
    chex.assert_trees_all_close(
        z2_m.astype(jnp.float32),
        z2_x.astype(jnp.float32),
        atol=fwd_atol,
        rtol=0.05,
    )
    chex.assert_trees_all_close(
        jax.tree.map(lambda a: a.astype(jnp.float32), (dz0_m, dwo_m, dwi_m)),
        jax.tree.map(lambda a: a.astype(jnp.float32), (dz0_x, dwo_x, dwi_x)),
        atol=grad_atol,
        rtol=grad_rtol,
    )

  def test_af3_pairformer_c128_n256(self):
    self._run_cropped_pairformer(
        4001,
        68,
        53,
        64,
        jnp.full((68, 68, 64), jnp.nan, jnp.bfloat16),
        fwd_atol=0.35,
        grad_atol=0.50,
    )

  def test_af3_pairformer_c64_n384(self):
    pad = jnp.broadcast_to(
        jnp.where(
            ((jnp.arange(68)[:, None] + jnp.arange(68)[None, :]) % 2) == 0,
            1e4,
            -1e4,
        )[:, :, None].astype(jnp.bfloat16),
        (68, 68, 64),
    )
    self._run_cropped_pairformer(
        4002, 68, 55, 64, pad, fwd_atol=0.35, grad_atol=0.50
    )

  def test_af2_multimer_c128_n512(self):
    ij_mod = (jnp.arange(68)[:, None] + jnp.arange(68)[None, :]) % 4
    pad = jnp.broadcast_to(
        jnp.select(
            [ij_mod == 0, ij_mod == 1, ij_mod == 2],
            [jnp.nan, jnp.inf, -jnp.inf],
            default=1e4,
        )[:, :, None].astype(jnp.bfloat16),
        (68, 68, 128),
    )
    self._run_cropped_pairformer(
        4003,
        68,
        55,
        128,
        pad,
        impl_tpu=None,
        fwd_atol=0.35,
        grad_atol=0.60,
        grad_rtol=0.06,
    )

  def test_af3_deep_2block_gapped(self):
    n, ch = 36, 32
    idx = jnp.arange(n)
    v1d = (idx < 14) | ((idx >= 18) & (idx < 30))
    mask = v1d[:, None] & v1d[None, :]
    z0_raw = _make_seeded_params(4004, n, ch, ch, ch, scale_mode="lecun")["x"]
    z0_clean = jnp.where(mask[:, :, None], z0_raw, 0.0)
    z0_nan = jnp.where(mask[:, :, None], z0_raw, jnp.nan)
    blocks = []
    for b in range(2):
      wo_b = {
          k: v
          for k, v in _make_seeded_params(
              40040 + b, 8, ch, ch, ch, scale_mode="lecun"
          ).items()
          if k not in ("x", "mask")
      }
      blocks.append(wo_b)
    blk_stack = jax.tree.map(
        lambda a, b: jnp.stack([a, b], axis=0), blocks[0], blocks[1]
    )
    target = jnp.where(
        mask[:, :, None],
        _make_random_array((n, n, ch), jnp.bfloat16, seed=40049),
        0.0,
    )

    def run_2blocks(impl: Literal["mosaic", "xla"], z, blk_weights):
      def step_fn(z_carry, wo_step):
        z_next = z_carry + api.triangle_multiplication(
            z_carry,
            mask,
            **wo_step,
            triangle_type="outgoing",
            epsilon=1e-5,
            implementation=impl,
        )
        return z_next, None

      z, _ = jax.lax.scan(step_fn, z, blk_weights)
      zv = jnp.where(mask[:, :, None], z, 0.0)
      loss = jnp.sum(zv.astype(jnp.float32) * target.astype(jnp.float32)) / 26.0
      return loss, zv

    vg_m = jax.jit(
        jax.value_and_grad(
            functools.partial(run_2blocks, "mosaic"),
            argnums=(0, 1),
            has_aux=True,
        )
    )
    vg_x = jax.jit(
        jax.value_and_grad(
            functools.partial(run_2blocks, "xla"),
            argnums=(0, 1),
            has_aux=True,
        )
    )
    (val_m, z4_m), (dz0_m, dblk_m) = vg_m(z0_nan, blk_stack)
    (_, z4_x), (dz0_x, dblk_x) = vg_x(z0_clean, blk_stack)

    self.assertTrue(bool(jnp.isfinite(val_m)))
    for leaf in jax.tree.leaves((z4_m, dz0_m, dblk_m)):
      self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))
    self.assertTrue(bool(jnp.all(dz0_m[14:18, :, :] == 0.0)))
    self.assertTrue(bool(jnp.all(dz0_m[30:, :, :] == 0.0)))
    chex.assert_trees_all_close(
        z4_m.astype(jnp.float32),
        z4_x.astype(jnp.float32),
        atol=0.50,
        rtol=0.06,
    )
    chex.assert_trees_all_close(
        jax.tree.map(lambda a: a.astype(jnp.float32), (dz0_m, dblk_m)),
        jax.tree.map(lambda a: a.astype(jnp.float32), (dz0_x, dblk_x)),
        atol=0.75,
        rtol=0.07,
    )

  def test_af_template_stack_vmap4(self):
    b, n, ch = 4, 36, 32
    mb = jnp.stack(
        [
            (jnp.arange(n) < l)[:, None] & (jnp.arange(n) < l)[None, :]
            for l in (32, 24, 13, 0)
        ],
        axis=0,
    )
    z0_raw = _make_random_array((b, n, n, ch), jnp.bfloat16, seed=4005)
    z0_clean = jnp.where(mb[:, :, :, None], z0_raw, 0.0)
    z0_nan = jnp.where(mb[:, :, :, None], z0_raw, jnp.nan).at[3].set(0.0)
    w_out = {
        k: v
        for k, v in _make_seeded_params(
            40051, 8, ch, ch, ch, scale_mode="lecun"
        ).items()
        if k not in ("x", "mask")
    }
    target = jnp.where(
        mb[:, :, :, None],
        _make_random_array((b, n, n, ch), jnp.bfloat16, seed=40059),
        0.0,
    )

    def tpl_loss(impl: Literal["mosaic", "xla"], zb, wo):
      z2b = jax.vmap(
          lambda zi, mi: zi
          + api.triangle_multiplication(
              zi, mi, **wo, triangle_type="outgoing", implementation=impl
          )
      )(zb, mb)
      z2v = jnp.where(mb[:, :, :, None], z2b, 0.0)
      loss = (
          jnp.sum(z2v.astype(jnp.float32) * target.astype(jnp.float32)) / 69.0
      )
      return loss, z2v

    vg_m = jax.jit(
        jax.value_and_grad(
            functools.partial(tpl_loss, "mosaic"),
            argnums=(0, 1),
            has_aux=True,
        )
    )
    vg_x = jax.jit(
        jax.value_and_grad(
            functools.partial(tpl_loss, "xla"),
            argnums=(0, 1),
            has_aux=True,
        )
    )
    (val_m, z2_m), (dz0_m, dwo_m) = vg_m(z0_nan, w_out)
    (_, z2_x), (dz0_x, dwo_x) = vg_x(z0_clean, w_out)

    self.assertTrue(bool(jnp.isfinite(val_m)))
    self.assertTrue(bool(jnp.all(dz0_m[3] == 0.0)))
    chex.assert_trees_all_close(
        z2_m[:3].astype(jnp.float32),
        z2_x[:3].astype(jnp.float32),
        atol=0.35,
        rtol=0.05,
    )
    chex.assert_trees_all_close(
        jax.tree.map(lambda a: a.astype(jnp.float32), (dz0_m, dwo_m)),
        jax.tree.map(lambda a: a.astype(jnp.float32), (dz0_x, dwo_x)),
        atol=0.50,
        rtol=0.05,
    )

  def test_af_zero_init_step0(self):
    n, l_valid, ch = 68, 55, 128
    v1d = jnp.arange(n) < l_valid
    mask = v1d[:, None] & v1d[None, :]
    z0_init = _make_seeded_params(4006, n, ch, ch, ch, scale_mode="lecun")["x"]
    z0 = jnp.where(mask[:, :, None], z0_init, 0.0)
    w_out = {
        k: v
        for k, v in _make_seeded_params(
            40061, 8, ch, ch, ch, scale_mode="af_zero_init"
        ).items()
        if k not in ("x", "mask")
    }
    w_in = {
        k: v
        for k, v in _make_seeded_params(
            40062, 8, ch, ch, ch, scale_mode="af_zero_init"
        ).items()
        if k not in ("x", "mask")
    }
    dout_masked = jnp.where(
        mask[:, :, None],
        _make_random_array((n, n, ch), jnp.bfloat16, seed=40069)
        / float(l_valid),
        0.0,
    )

    def step0(impl: str, z_in, m_in, dout_in, wo, wi):
      z2 = _pairformer_block(z_in, m_in, wo, wi, implementation=impl)
      loss = jnp.sum(z2.astype(jnp.float32) * dout_in.astype(jnp.float32))
      return loss, z2

    vg_m = jax.jit(
        jax.value_and_grad(
            functools.partial(step0, "mosaic"),
            argnums=(0, 3, 4),
            has_aux=True,
        )
    )
    (_, z2_m), (dz0_m, dwo_m, dwi_m) = vg_m(z0, mask, dout_masked, w_out, w_in)

    # Evaluate XLA reference on CPU float32 to avoid TPU v5p XLA::TPU
    # Megacore miscompilation when projection_out_weights == 0.
    cpu = jax.devices("cpu")[0]
    tpu = jax.devices("tpu")[0]
    with jax.default_device(cpu):
      z0_c = jax.device_put(z0, cpu).astype(jnp.float32)
      mask_c = jax.device_put(mask, cpu)
      dout_c = jax.device_put(dout_masked, cpu).astype(jnp.float32)
      w_out_c = jax.tree.map(
          lambda a: jax.device_put(a, cpu).astype(jnp.float32), w_out
      )
      w_in_c = jax.tree.map(
          lambda a: jax.device_put(a, cpu).astype(jnp.float32), w_in
      )
      vg_x = jax.jit(
          jax.value_and_grad(
              functools.partial(step0, "xla"),
              argnums=(0, 3, 4),
              has_aux=True,
          )
      )
      (_, _), (_, dwo_x_c, dwi_x_c) = vg_x(
          z0_c, mask_c, dout_c, w_out_c, w_in_c
      )
    dwo_x = jax.device_put(dwo_x_c, tpu)
    dwi_x = jax.device_put(dwi_x_c, tpu)

    chex.assert_trees_all_equal(z2_m, z0)
    chex.assert_trees_all_equal(dz0_m, dout_masked)
    for dw, dw_ref in ((dwo_m, dwo_x), (dwi_m, dwi_x)):
      norm_w = jnp.linalg.norm(dw["projection_out_weights"].astype(jnp.float32))
      self.assertGreater(float(norm_w), 1e-2)
      chex.assert_trees_all_close(
          dw["projection_out_weights"].astype(jnp.float32),
          dw_ref["projection_out_weights"].astype(jnp.float32),
          atol=0.35,
          rtol=0.05,
      )
      for zk in (
          "gate_out_weights",
          "projection_in_weights",
          "gate_in_weights",
          "layernorm_in_scale",
          "layernorm_in_offset",
          "layernorm_out_scale",
          "layernorm_out_offset",
      ):
        self.assertTrue(bool(jnp.all(dw[zk] == 0.0)))

  def test_af_remat_checkpoint(self):
    n, l_valid, c, h, d = 35, 27, 32, 16, 32
    v1d = jnp.arange(n) < l_valid
    mask = v1d[:, None] & v1d[None, :]
    z0_init = _make_seeded_params(4007, n, c, h, d, scale_mode="lecun")["x"]
    z0 = jnp.where(mask[:, :, None], z0_init, 0.0)
    w_out = {
        k: v
        for k, v in _make_seeded_params(
            40071, 8, c, h, d, scale_mode="lecun"
        ).items()
        if k not in ("x", "mask")
    }
    target = jnp.where(
        mask[:, :, None],
        _make_random_array((n, n, d), jnp.bfloat16, seed=40079),
        0.0,
    )
    denom = float(l_valid * l_valid * d)

    def mse_loss(use_remat: bool, impl: Any, state):
      z_in, wo = state
      fn = lambda za, woa: za + api.triangle_multiplication(
          za, mask, **woa, triangle_type="outgoing", implementation=impl
      )
      if use_remat:
        fn = jax.checkpoint(fn)
      z2 = fn(z_in, wo)
      z2v = jnp.where(mask[:, :, None], z2, 0.0)
      diff = z2v.astype(jnp.float32) - target.astype(jnp.float32)
      return 0.5 * jnp.sum(diff * diff) / denom

    s0 = (z0, w_out)
    vg_remat = jax.jit(
        jax.value_and_grad(functools.partial(mse_loss, True, "mosaic"))
    )
    loss0_remat, g_remat = vg_remat(s0)
    loss0_plain, g_plain = jax.jit(
        jax.value_and_grad(functools.partial(mse_loss, False, "mosaic"))
    )(s0)
    _, g_xla = jax.jit(
        jax.value_and_grad(functools.partial(mse_loss, False, "xla"))
    )(s0)

    chex.assert_trees_all_equal((loss0_remat, g_remat), (loss0_plain, g_plain))
    chex.assert_trees_all_close(
        jax.tree.map(lambda a: a.astype(jnp.float32), g_remat),
        jax.tree.map(lambda a: a.astype(jnp.float32), g_xla),
        atol=0.50,
        rtol=0.05,
    )
    s1_m = jax.tree.map(
        lambda p, g: (
            p.astype(jnp.float32) - 2.0 * g.astype(jnp.float32)
        ).astype(p.dtype),
        s0,
        g_remat,
    )
    loss1_m, _ = vg_remat(s1_m)
    self.assertLess(float(loss1_m), float(loss0_remat))

  def test_poisoned_vmem_smem(self):
    n, ch = 68, 64
    inv_denom = jnp.float32(1.0 / (n - 15))
    v1d = jnp.arange(n) < (n - 15)
    mask = v1d[:, None] & v1d[None, :]
    z0_init = _make_seeded_params(4008 + n, n, ch, ch, ch, scale_mode="lecun")[
        "x"
    ]
    z0 = jnp.where(mask[:, :, None], z0_init, 0.0)
    w_out = {
        k: v
        for k, v in _make_seeded_params(
            40081 + n, 8, ch, ch, ch, scale_mode="lecun"
        ).items()
        if k not in ("x", "mask")
    }
    w_in = {
        k: v
        for k, v in _make_seeded_params(
            40082 + n, 8, ch, ch, ch, scale_mode="lecun"
        ).items()
        if k not in ("x", "mask")
    }
    target = jnp.where(
        mask[:, :, None],
        _make_random_array((n, n, ch), jnp.bfloat16, seed=40089 + n),
        0.0,
    )

    _poison_vmem_smem(0x00000000)
    r0 = _jit_cropped_pairformer_vg(
        z0, w_out, w_in, mask, target, inv_denom, impl="mosaic"
    )
    jax.block_until_ready(r0)
    for leaf in jax.tree.leaves(r0):
      self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))
    for bits in (0x7FC00000, 0xFF800000, 0x7F7FFFFF):
      _poison_vmem_smem(bits)
      r_poison = _jit_cropped_pairformer_vg(
          z0, w_out, w_in, mask, target, inv_denom, impl="mosaic"
      )
      jax.block_until_ready(r_poison)
      chex.assert_trees_all_equal(r0, r_poison)

  def test_af3_large_complex_seq1024(self):
    n, l_valid, ch = 1024, 915, 32
    v1d = jnp.arange(n) < l_valid
    mask = v1d[:, None] & v1d[None, :]
    z0_raw = (
        _make_random_array((n, n, ch), jnp.bfloat16, seed=4009 + n)
        * (1.0 / math.sqrt(ch))
    ).astype(jnp.bfloat16)
    z0_clean = jnp.where(mask[:, :, None], z0_raw, 0.0)
    z0_nan = jnp.where(mask[:, :, None], z0_raw, jnp.nan)
    wo = {
        k: v
        for k, v in _make_seeded_params(
            40091 + n, 8, ch, ch, ch, scale_mode="lecun"
        ).items()
        if k not in ("x", "mask")
    }
    wi = {
        k: v
        for k, v in _make_seeded_params(
            40092 + n, 8, ch, ch, ch, scale_mode="lecun"
        ).items()
        if k not in ("x", "mask")
    }
    z2_m = _jit_mosaic_pairformer_fwd(z0_nan, mask, wo, wi)
    # Evaluate XLA reference via HLO-DCE-safe _jit_ref_fwd_with_res steps.
    u_out, _ = _jit_ref_fwd_with_res(
        dict(x=z0_clean, mask=mask, **wo), "outgoing"
    )
    z1_x = z0_clean + u_out
    u_in, _ = _jit_ref_fwd_with_res(dict(x=z1_x, mask=mask, **wi), "incoming")
    z2_x = z1_x + u_in
    z2_mv = z2_m[:l_valid, :l_valid]
    z2_xv = z2_x[:l_valid, :l_valid]
    self.assertTrue(bool(jnp.all(jnp.isfinite(z2_mv))))
    chex.assert_trees_all_close(
        z2_mv.astype(jnp.float32),
        z2_xv.astype(jnp.float32),
        atol=0.45,
        rtol=0.05,
    )

  def test_m3_2d_mxu_pack_unpack_and_vmem_clamping(self):
    for bi in (1, 16):
      x3d = _make_random_array((bi, 64, 128), jnp.bfloat16, seed=5001 + bi)
      w = _make_random_array((128, 64), jnp.bfloat16, seed=5011 + bi)
      dy3d = _make_random_array((bi, 128, 128), jnp.bfloat16, seed=5021 + bi)

      packed = common.pack_icj_to_2d(x3d)
      self.assertEqual(packed.shape, (64, bi * 128))
      unpacked = common.unpack_2d_to_icj(packed, bi, 128)
      chex.assert_trees_all_equal(unpacked, x3d)

      y_2d = common.unpack_2d_to_icj(common.matmul_2d(w, packed), bi, 128)
      y_icj = common.matmul_icj(w, x3d)
      y_ref = jnp.einsum(
          "oc,icj->ioj",
          w.astype(jnp.float32),
          x3d.astype(jnp.float32),
          precision=jax.lax.Precision.HIGHEST,
      )
      chex.assert_trees_all_equal(y_2d, y_icj)
      chex.assert_trees_all_close(y_2d, y_ref, atol=1e-4, rtol=1e-4)

      dy_packed = common.pack_icj_to_2d(dy3d)
      dx_2d = common.unpack_2d_to_icj(
          common.matmul_wt_2d(w, dy_packed), bi, 128
      )
      dx_icj = common.matmul_wt_icj(w, dy3d)
      dx_ref = jnp.einsum(
          "oc,ioj->icj",
          w.astype(jnp.float32),
          dy3d.astype(jnp.float32),
          precision=jax.lax.Precision.HIGHEST,
      )
      chex.assert_trees_all_equal(dx_2d, dx_icj)
      chex.assert_trees_all_close(dx_2d, dx_ref, atol=1e-4, rtol=1e-4)

      dw_2d = common.matmul_dw_2d(dy_packed, packed)
      dw_icj = common.matmul_dw_icj(dy3d, x3d)
      dw_ref = jnp.einsum(
          "ioj,icj->oc",
          dy3d.astype(jnp.float32),
          x3d.astype(jnp.float32),
          precision=jax.lax.Precision.HIGHEST,
      )
      chex.assert_trees_all_equal(dw_2d, dw_icj)
      chex.assert_trees_all_close(dw_2d, dw_ref, atol=1e-3, rtol=1e-4)

    kern = mtpu_tri_mul.kernel
    budgets = (
        int(16 * 1024 * 1024 * 0.80),
        int(64 * 1024 * 1024 * 0.875),
        int(128 * 1024 * 1024 * 0.85),
    )
    for n_pad in (384, 1024, 2048):
      for ch in (64, 128):
        estimators = (
            lambda bi, bj, ch=ch: kern._estimate_prologue_vmem(bi, bj, ch, ch),
            lambda bi, bj, ch=ch: kern._estimate_epilogue_vmem(
                bi, bj, ch, ch, ch
            ),
            lambda bi, bj, ch=ch: kern._estimate_epilogue_bwd_vmem(
                bi, bj, ch, ch, ch
            ),
            lambda bi, bj, ch=ch: kern._estimate_prologue_bwd_vmem(
                bi, bj, ch, ch
            ),
        )
        for budget in budgets:
          for est_fn in estimators:
            bi, bj = kern._clamp_blocks_to_vmem(
                n_pad, min(64, n_pad), min(1024, n_pad), budget, est_fn
            )
            self.assertLessEqual(est_fn(bi, bj), budget)
            if budget == int(64 * 1024 * 1024 * 0.875) and ch == 64:
              self.assertEqual(bi % 16, 0)

    ce_gp = kern._epilogue_cost_estimate(1024, 64, 64, 64)
    ce_split = kern._epilogue_cost_estimate(1024, 128, 128, 128)
    self.assertEqual(
        ce_split.flops,
        1024 * 1024 * (2 * 128 * 256 + 7 * 256 + 2 * 128),
    )
    for ce in (
        kern._prologue_cost_estimate(1024, 64, 64),
        ce_gp,
        ce_split,
    ):
      self.assertGreater(ce.flops, 0)
      self.assertGreater(ce.transcendentals, 0)
      self.assertGreater(ce.bytes_accessed, 0)

  def test_m3_unpin_xpad_aliasing_and_accuracy(self):
    # Verify x_pad unpinning adds optimization_barrier >= 1 GiB vs < 1 GiB.
    small_shapes = _make_seeded_params(5100, 384, 64, 64, 64)
    small_dp = {k: small_shapes[k] for k in _GRAD_KEYS}
    small_dout = _make_random_array((384, 384, 64), jnp.bfloat16, seed=5101)
    small_hlo = _jit_op_fwd_and_vjp.lower(
        small_dp, small_shapes["mask"], small_dout, "outgoing"
    ).as_text(dialect="hlo")
    gen, _, num_cores, _, _ = common.get_tpu_info(small_dp["x"])
    if not (gen == 5 and num_cores > 1):
      self.assertNotIn("opt-barrier", small_hlo)

    large_dp_sds = dict(
        x=jax.ShapeDtypeStruct((2048, 2048, 128), jnp.bfloat16),
        projection_in_weights=jax.ShapeDtypeStruct((128, 2, 64), jnp.bfloat16),
        gate_in_weights=jax.ShapeDtypeStruct((128, 2, 64), jnp.bfloat16),
        projection_out_weights=jax.ShapeDtypeStruct((64, 64), jnp.bfloat16),
        gate_out_weights=jax.ShapeDtypeStruct((128, 64), jnp.bfloat16),
        layernorm_in_scale=jax.ShapeDtypeStruct((128,), jnp.bfloat16),
        layernorm_in_offset=jax.ShapeDtypeStruct((128,), jnp.bfloat16),
        layernorm_out_scale=jax.ShapeDtypeStruct((64,), jnp.bfloat16),
        layernorm_out_offset=jax.ShapeDtypeStruct((64,), jnp.bfloat16),
    )
    large_mask_sds = jax.ShapeDtypeStruct((2048, 2048), jnp.bool_)
    large_dout_sds = jax.ShapeDtypeStruct((2048, 2048, 64), jnp.bfloat16)
    self.assertFalse(common.should_unpin_xpad(small_dp["x"], 384))
    self.assertTrue(
        common.should_unpin_xpad(cast(Any, large_dp_sds["x"]), 2048)
    )
    large_hlo = _jit_op_fwd_and_vjp.lower(
        large_dp_sds, large_mask_sds, large_dout_sds, "outgoing"
    ).as_text(dialect="hlo")
    self.assertGreaterEqual(
        large_hlo.count("opt-barrier(") - small_hlo.count("opt-barrier("), 1
    )

    n, c, h, d = 512, 128, 64, 64
    params = _make_seeded_params(5102, n, c, h, d, scale_mode="lecun")
    v1d = jnp.arange(n) < 460
    mask = v1d[:, None] & v1d[None, :]
    params["mask"] = mask
    params["x"] = jnp.where(mask[:, :, None], params["x"], 0.0)
    dp = {k: params[k] for k in _GRAD_KEYS}
    target = jnp.where(
        mask[:, :, None],
        _make_random_array((n, n, d), jnp.bfloat16, seed=5103),
        0.0,
    )

    # Verify alias_x_pad=True in epilogue_bwd is bit-exact vs alias_x_pad=False.
    epi_n = 128
    epi_p = _make_seeded_params(5104, epi_n, c, h, d, scale_mode="lecun")
    dout_pad = common.transpose_and_pad_icj(
        _make_random_array((epi_n, epi_n, d), jnp.bfloat16, seed=5105), epi_n
    )
    t3_pad = _make_random_array((h, epi_n, epi_n), jnp.bfloat16, seed=5106)
    x_pad_a = common.transpose_and_pad_icj(epi_p["x"], epi_n)
    x_pad_b = jnp.copy(x_pad_a)
    epi_fn = lambda xp, alias_flag: mtpu_tri_mul.kernel.epilogue_bwd(
        dout_pad,
        t3_pad,
        xp,
        epi_p["projection_out_weights"],
        epi_p["gate_out_weights"],
        epi_p["layernorm_in_scale"],
        epi_p["layernorm_in_offset"],
        epi_p["layernorm_out_scale"],
        epi_p["layernorm_out_offset"],
        eps=1e-6,
        block_m=128,
        alias_x_pad=alias_flag,
    )
    res_no_alias = jax.jit(functools.partial(epi_fn, alias_flag=False))(x_pad_a)
    res_alias = jax.jit(functools.partial(epi_fn, alias_flag=True))(x_pad_b)
    chex.assert_trees_all_equal(res_alias, res_no_alias)

    # Verify full forward + VJP and standalone VJP accuracy at n=512, c=128.
    (loss_m, out_m), grads_m = _jit_tpu_val_and_grad(
        dp,
        mask,
        target,
        mask,
        fwd_cfg=None,
        vjp_cfg=None,
        triangle_type="outgoing",
        epsilon=1e-6,
        use_api=False,
        api_impl=None,
        has_valid_mask=True,
    )
    (loss_x, out_x), grads_x = _jit_xla_val_and_grad(
        dp,
        mask,
        target,
        mask,
        triangle_type="outgoing",
        epsilon=1e-6,
        has_valid_mask=True,
    )
    out_res, residuals = _jit_op_fwd_with_res(params, "outgoing")
    chex.assert_trees_all_close(
        jnp.where(mask[:, :, None], out_res, 0.0).astype(jnp.float32),
        jnp.where(mask[:, :, None], out_x, 0.0).astype(jnp.float32),
        atol=0.45,
        rtol=0.05,
    )
    out_fwd_only = _jit_op_fwd(params, "outgoing")
    chex.assert_trees_all_close(
        jnp.where(mask[:, :, None], out_fwd_only, 0.0).astype(jnp.float32),
        jnp.where(mask[:, :, None], out_x, 0.0).astype(jnp.float32),
        atol=0.45,
        rtol=0.05,
    )
    self.assertTrue(bool(jnp.isfinite(loss_m)))
    chex.assert_trees_all_close(
        loss_m / float(460 * d),
        loss_x / float(460 * d),
        atol=1e-2,
        rtol=0.05,
    )
    chex.assert_trees_all_close(
        jnp.where(mask[:, :, None], out_m, 0.0).astype(jnp.float32),
        jnp.where(mask[:, :, None], out_x, 0.0).astype(jnp.float32),
        atol=0.45,
        rtol=0.05,
    )
    for k in _GRAD_KEYS:
      chex.assert_trees_all_close(
          grads_m[k].astype(jnp.float32),
          grads_x[k].astype(jnp.float32),
          atol=0.50,
          rtol=0.05,
      )
    dout_standalone = jnp.where(
        mask[:, :, None], target.astype(jnp.float32) / float(n), 0.0
    ).astype(jnp.bfloat16)
    grads_standalone = _jit_standalone_vjp(
        residuals,
        out_res,
        dout_standalone,
        params,
        "outgoing",
    )
    for k in _GRAD_KEYS:
      chex.assert_trees_all_close(
          grads_standalone[k].astype(jnp.float32),
          grads_x[k].astype(jnp.float32),
          atol=0.50,
          rtol=0.05,
      )

    # Verify n=512, c=64, h=64, d=64 (aligned 64-channel unpin_x_pad threshold).
    p64 = _make_seeded_params(5107, 512, 64, 64, 64, scale_mode="lecun")
    p64["mask"] = mask
    p64["x"] = jnp.where(mask[:, :, None], p64["x"], 0.0)
    dp64 = {k: p64[k] for k in _GRAD_KEYS}
    (_, out_m64), grads_m64 = _jit_tpu_val_and_grad(
        dp64,
        mask,
        target,
        mask,
        fwd_cfg=None,
        vjp_cfg=None,
        triangle_type="outgoing",
        epsilon=1e-6,
        use_api=False,
        api_impl=None,
        has_valid_mask=True,
    )
    (_, out_x64), grads_x64 = _jit_xla_val_and_grad(
        dp64,
        mask,
        target,
        mask,
        triangle_type="outgoing",
        epsilon=1e-6,
        has_valid_mask=True,
    )
    chex.assert_trees_all_close(
        jnp.where(mask[:, :, None], out_m64, 0.0).astype(jnp.float32),
        jnp.where(mask[:, :, None], out_x64, 0.0).astype(jnp.float32),
        atol=0.45,
        rtol=0.05,
    )
    for k in _GRAD_KEYS:
      chex.assert_trees_all_close(
          grads_m64[k].astype(jnp.float32),
          grads_x64[k].astype(jnp.float32),
          atol=0.50,
          rtol=0.05,
      )

    # End-to-end TPU execution at n=2048, c=128, h=16, d=16 (x_pad == 1 GiB):
    # exercises unpin_x_pad=True + optimization_barrier + epi_aliases={2: 1}
    # + pro_aliases={2: 0} at high tile coordinates [1984:2048, 1984:2048].
    n_big, c_big, h_big, d_big = 2048, 128, 16, 16
    p_sub = _make_seeded_params(
        5110, 64, c_big, h_big, d_big, scale_mode="lecun"
    )
    p_sub["mask"] = jnp.ones((64, 64), dtype=jnp.bool_)
    dout_sub = (
        _make_random_array((64, 64, d_big), jnp.bfloat16, seed=5111) / 64.0
    )
    dp_sub = {k: p_sub[k] for k in _GRAD_KEYS}
    out_sub_ref, grads_sub_ref = _jit_ref_fwd_and_vjp(
        dp_sub, p_sub["mask"], dout_sub, "outgoing"
    )

    valid_big = jnp.arange(n_big) >= (n_big - 64)
    mask_big = valid_big[:, None] & valid_big[None, :]
    x_big = (
        jnp.zeros((n_big, n_big, c_big), dtype=jnp.bfloat16)
        .at[1984:, 1984:, :]
        .set(p_sub["x"])
    )
    dout_big = (
        jnp.zeros((n_big, n_big, d_big), dtype=jnp.bfloat16)
        .at[1984:, 1984:, :]
        .set(dout_sub)
    )
    self.assertTrue(common.should_unpin_xpad(x_big, n_big))
    dp_big = dict(dp_sub, x=x_big)
    out_big, grads_big = _jit_op_fwd_and_vjp(
        dp_big, mask_big, dout_big, "outgoing"
    )
    chex.assert_trees_all_close(
        out_big[1984:, 1984:, :].astype(jnp.float32),
        out_sub_ref.astype(jnp.float32),
        atol=0.35,
        rtol=0.05,
    )
    chex.assert_trees_all_close(
        grads_big["x"][1984:, 1984:, :].astype(jnp.float32),
        grads_sub_ref["x"].astype(jnp.float32),
        atol=0.50,
        rtol=0.05,
    )
    self.assertTrue(bool(jnp.all(grads_big["x"][:1984, :, :] == 0.0)))
    self.assertTrue(bool(jnp.all(grads_big["x"][1984:, :1984, :] == 0.0)))
    for k in _GRAD_KEYS:
      if k == "x":
        continue
      chex.assert_trees_all_close(
          grads_big[k].astype(jnp.float32),
          grads_sub_ref[k].astype(jnp.float32),
          atol=0.50,
          rtol=0.05,
      )

  def test_m3_masked_and_residual_ln_variance_overflow_bwd(self):
    n = 64
    v1d = jnp.arange(n) < 48
    mask = v1d[:, None] & v1d[None, :]
    hi_a = jnp.bfloat16(3.38e38)
    lo_a = jnp.bfloat16(-3.38e38)
    hi_b = jnp.bfloat16(3.389e38)
    lo_b = jnp.bfloat16(-3.389e38)

    for idx, ch in enumerate((64, 128)):
      p_clean = _make_seeded_params(
          5200 + idx, n, ch, ch, ch, scale_mode="lecun"
      )
      p_clean["mask"] = mask
      p_clean["x"] = jnp.where(mask[:, :, None], p_clean["x"], 0.0)
      dout_raw = _make_random_array((n, n, ch), jnp.bfloat16, 5210 + idx) / n
      dout = jnp.where(mask[:, :, None], dout_raw, 0.0)

      vec_a = jnp.full((ch,), lo_a, dtype=jnp.bfloat16).at[0].set(hi_a)
      vec_b = (
          jnp.zeros((ch,), dtype=jnp.bfloat16).at[0].set(hi_b).at[1:3].set(lo_b)
      )
      row_pat = jnp.where(
          (jnp.arange(n) < 56)[:, None, None],
          vec_a[None, None, :],
          vec_b[None, None, :],
      )
      bad_x = jnp.where(mask[:, :, None], p_clean["x"], row_pat)
      self.assertTrue(bool(jnp.all(jnp.isfinite(bad_x))))

      dp_clean = {k: p_clean[k] for k in _GRAD_KEYS}
      dp_bad = dict(dp_clean, x=bad_x)
      for tri in ("outgoing", "incoming"):
        out_c, grads_c = _jit_op_fwd_and_vjp(dp_clean, mask, dout, tri)
        out_b, grads_b = _jit_op_fwd_and_vjp(dp_bad, mask, dout, tri)
        self.assertTrue(bool(jnp.all(jnp.isfinite(out_b[:48, :48]))))
        chex.assert_trees_all_close(
            out_b[:48, :48], out_c[:48, :48], atol=1e-4, rtol=1e-4
        )
        for k in _GRAD_KEYS:
          self.assertTrue(bool(jnp.all(jnp.isfinite(grads_b[k]))))
          chex.assert_trees_all_close(
              grads_b[k], grads_c[k], atol=1e-4, rtol=1e-4
          )

      # Corrupted finite residual t3 overflow on zero-cotangent slices.
      out_r, res_r = _jit_op_fwd_with_res(p_clean, "outgoing")
      g_res_clean = _jit_standalone_vjp(res_r, out_r, dout, p_clean, "outgoing")
      t3_pat_a = jnp.broadcast_to(vec_a[:, None, None], (ch, 8, n))
      t3_pat_b = jnp.broadcast_to(vec_b[:, None, None], (ch, 8, n))
      t3_bad = (
          res_r[2].at[:, 48:56, :].set(t3_pat_a).at[:, 56:64, :].set(t3_pat_b)
      )
      self.assertTrue(bool(jnp.all(jnp.isfinite(t3_bad))))
      res_bad = (res_r[0], res_r[1], t3_bad)
      g_res_bad = _jit_standalone_vjp(res_bad, out_r, dout, p_clean, "outgoing")
      for k in _GRAD_KEYS:
        self.assertTrue(bool(jnp.all(jnp.isfinite(g_res_bad[k]))))
        chex.assert_trees_all_close(
            g_res_bad[k], g_res_clean[k], atol=1e-4, rtol=1e-4
        )

  def test_m3_zero_cotangent_affine_and_projection_overflow(self):
    n = 32
    mask_all = jnp.ones((n, n), dtype=jnp.bool_)
    for idx, ch in enumerate((64, 128)):
      dout_zero = jnp.zeros((n, n, ch), dtype=jnp.bfloat16)
      res_zero = (jnp.zeros((ch, n, n), dtype=jnp.bfloat16),) * 3

      # 1. Finite-bf16 LayerNorm affine scale/offset overflow (dout == 0).
      p_aff = _make_seeded_params(5300 + idx, n, ch, ch, ch)
      big_s = jnp.bfloat16(2.0e38)
      p_aff["layernorm_in_scale"] = jnp.full((ch,), big_s, dtype=jnp.bfloat16)
      p_aff["layernorm_in_offset"] = jnp.full((ch,), big_s, dtype=jnp.bfloat16)
      p_aff["layernorm_out_scale"] = jnp.full((ch,), big_s, dtype=jnp.bfloat16)
      p_aff["layernorm_out_offset"] = jnp.full((ch,), big_s, dtype=jnp.bfloat16)
      dp_aff = {k: p_aff[k] for k in _GRAD_KEYS}
      for arr in dp_aff.values():
        self.assertTrue(bool(jnp.all(jnp.isfinite(arr))))
      grads_aff = _jit_standalone_vjp(
          res_zero, dout_zero, dout_zero, p_aff, "outgoing"
      )
      for g in grads_aff.values():
        self.assertFalse(bool(jnp.any(jnp.isnan(g))))
        self.assertTrue(bool(jnp.all(g == 0.0)))

      # 2. Finite left_act (1e38) + overflowing prologue projection p -> +Inf.
      half = ch // 2
      x_sign = jnp.concatenate(
          [
              jnp.ones((n, n, half), dtype=jnp.bfloat16),
              -jnp.ones((n, n, half), dtype=jnp.bfloat16),
          ],
          axis=-1,
      )
      ln_sign = jnp.concatenate(
          [
              jnp.full((half,), jnp.bfloat16(1.0e38), dtype=jnp.bfloat16),
              jnp.full((half,), jnp.bfloat16(-1.0e38), dtype=jnp.bfloat16),
          ],
          axis=0,
      )
      p_pro = _make_seeded_params(5310 + idx, n, ch, ch, ch)
      p_pro["x"] = x_sign
      p_pro["layernorm_in_scale"] = ln_sign
      p_pro["layernorm_in_offset"] = jnp.zeros((ch,), dtype=jnp.bfloat16)
      p_pro["projection_in_weights"] = jnp.full(
          (ch, 2, ch), jnp.bfloat16(16.0), dtype=jnp.bfloat16
      )
      p_pro["gate_in_weights"] = jnp.ones((ch, 2, ch), dtype=jnp.bfloat16)
      dp_pro = {k: p_pro[k] for k in _GRAD_KEYS}
      for arr in dp_pro.values():
        self.assertTrue(bool(jnp.all(jnp.isfinite(arr))))
      grads_pro = _jit_standalone_vjp(
          res_zero, dout_zero, dout_zero, p_pro, "incoming"
      )
      for g in grads_pro.values():
        self.assertFalse(bool(jnp.any(jnp.isnan(g))))
        self.assertTrue(bool(jnp.all(g == 0.0)))

      # 3. Finite t3 (0.0) & finite act_out (1e38) + overflowing epilogue p.
      p_epi = _make_seeded_params(5320 + idx, n, ch, ch, ch)
      p_epi["projection_in_weights"] = jnp.zeros(
          (ch, 2, ch), dtype=jnp.bfloat16
      )
      p_epi["layernorm_out_scale"] = jnp.ones((ch,), dtype=jnp.bfloat16)
      p_epi["layernorm_out_offset"] = jnp.full(
          (ch,), jnp.bfloat16(1.0e38), dtype=jnp.bfloat16
      )
      p_epi["projection_out_weights"] = jnp.full(
          (ch, ch), jnp.bfloat16(16.0), dtype=jnp.bfloat16
      )
      p_epi["gate_out_weights"] = jnp.zeros((ch, ch), dtype=jnp.bfloat16)
      dp_epi = {k: p_epi[k] for k in _GRAD_KEYS}
      for arr in dp_epi.values():
        self.assertTrue(bool(jnp.all(jnp.isfinite(arr))))
      _, grads_epi = _jit_op_fwd_and_vjp(
          dp_epi, mask_all, dout_zero, "outgoing"
      )
      for g in grads_epi.values():
        self.assertFalse(bool(jnp.any(jnp.isnan(g))))
        self.assertTrue(bool(jnp.all(g == 0.0)))

  def test_m3_epilogue_split_3d_and_autotune_dedup(self):
    # 1. Forward + VJP + residuals at C=H=D=128 (c + h = 256 > 128: kernel_3d).
    n = 67
    p128 = _make_seeded_params(5400, n, 128, 128, 128, scale_mode="lecun")
    for tri in ("outgoing", "incoming"):
      self._check_fwd_and_9_grads(p128, tri, fwd_atol=0.35, grad_atol=0.50)
    out_res, res = _jit_op_fwd_with_res(p128, "outgoing")
    out_fwd = _jit_op_fwd(p128, "outgoing")
    chex.assert_trees_all_equal(out_res, out_fwd)
    exp_out, exp_res = _jit_ref_fwd_with_res(p128, "outgoing")
    chex.assert_trees_all_close(out_res, exp_out, atol=0.35, rtol=0.05)
    chex.assert_trees_all_close(res[0], exp_res[0], atol=0.15, rtol=0.05)
    chex.assert_trees_all_close(res[1], exp_res[1], atol=0.15, rtol=0.05)
    chex.assert_trees_all_close(
        res[2], jnp.moveaxis(exp_res[2], -1, 0), atol=0.2 * n, rtol=0.05
    )
    dout = _make_random_array((n, n, 128), jnp.bfloat16, seed=5401) / float(n)
    grads_st = _jit_standalone_vjp(res, out_res, dout, p128, "outgoing")
    _, grads_ref = _jit_ref_fwd_and_vjp(
        {k: p128[k] for k in _GRAD_KEYS}, p128["mask"], dout, "outgoing"
    )
    for k in _GRAD_KEYS:
      chex.assert_trees_all_close(
          grads_st[k].astype(jnp.float32),
          grads_ref[k].astype(jnp.float32),
          atol=0.50,
          rtol=0.05,
      )

    # 2. Verify _dedup_autotuning_configs retains (16, 1024), (16, 2048), and
    #    epilogue-override candidates without pruning against 1024 / seen_def.
    sds_params = dict(
        x=jax.ShapeDtypeStruct((2048, 2048, 64), jnp.bfloat16),
        mask=jax.ShapeDtypeStruct((2048, 2048), jnp.bool_),
        projection_in_weights=jax.ShapeDtypeStruct((64, 2, 64), jnp.bfloat16),
        gate_in_weights=jax.ShapeDtypeStruct((64, 2, 64), jnp.bfloat16),
        projection_out_weights=jax.ShapeDtypeStruct((64, 64), jnp.bfloat16),
        gate_out_weights=jax.ShapeDtypeStruct((64, 64), jnp.bfloat16),
        layernorm_in_scale=jax.ShapeDtypeStruct((64,), jnp.bfloat16),
        layernorm_in_offset=jax.ShapeDtypeStruct((64,), jnp.bfloat16),
        layernorm_out_scale=jax.ShapeDtypeStruct((64,), jnp.bfloat16),
        layernorm_out_offset=jax.ShapeDtypeStruct((64,), jnp.bfloat16),
    )
    ba_2k = self._op.bind(**sds_params, triangle_type="outgoing")
    fwd_cfgs = self._op._get_autotuning_configs(ba_2k)
    vjp_op = mtpu_tri_mul_vjp.PallasMosaicTpuTriangleMultiplicationVjp()
    vjp_ba_2k = vjp_op.bind(**ba_2k.vjp_arg_spec)
    vjp_cfgs = vjp_op._get_autotuning_configs(vjp_ba_2k)

    fwd_pairs = {
        mtpu_tri_mul.kernel.resolve_config_blocks(2048, c, x=p128["x"])
        for c in fwd_cfgs
    }
    vjp_pro_tiles = {
        mtpu_tri_mul.kernel.resolve_config_blocks(2048, c, x=p128["x"])[0]
        for c in vjp_cfgs
    }
    gen, vmem_mib, _, _, _ = common.get_tpu_info(p128["x"])
    if vmem_mib > 64 and gen > 4:
      self.assertIn(((16, 1024), (16, 1024)), fwd_pairs)
      self.assertIn(((16, 2048), (16, 2048)), fwd_pairs)
      self.assertIn(((16, 2048), (32, 2048)), fwd_pairs)
      self.assertIn(((64, 512), (64, 1024)), fwd_pairs)
      self.assertIn((16, 1024), vjp_pro_tiles)
      self.assertIn((16, 2048), vjp_pro_tiles)
    elif vmem_mib > 16 and gen > 4:
      self.assertIn(((16, 1024), (16, 1024)), fwd_pairs)
      self.assertIn(((16, 1024), (32, 1024)), fwd_pairs)
      self.assertIn((16, 1024), vjp_pro_tiles)


if __name__ == "__main__":
  absltest.main()
