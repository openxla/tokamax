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
"""Base classes for DeepSeek-V4 manifold-constrained hyper-connections (mHC).

There are three ops:

* `MhcPre`: run before each sublayer. Computes the gates and collapses the
  residual streams into the sublayer input.
* `MhcPost`: run after each sublayer. Writes its output back into the streams.
* `MhcFusedPostPre`: one sublayer's `MhcPost` followed by the next sublayer's
  `MhcPre`, sharing one pass over the residual streams.

See `api` for the math, the call order and the shapes.
"""

from typing import Any, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.mhc import reference

AbstractArray = jax.ShapeDtypeStruct | jax.core.ShapedArray | np.ndarray

PreOutputs = tuple[jax.Array, jax.Array, jax.Array]
FusedOutputs = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


def _check_dtype(name: str, x: Any, dtype: Any) -> None:
  if x.dtype != dtype:
    raise ValueError(f"{name} must be {jnp.dtype(dtype).name}, got {x.dtype}.")


def _check_gate_params(
    residual: Any,
    fn: Any,
    hc_scale: Any,
    hc_base: Any,
    sinkhorn_repeat: int,
) -> int:
  """Validates the `mhc_pre` arguments shared by `MhcPre` and the fused op.

  Args:
    residual: `(T, M * H)` residual streams.
    fn: `(M * (M + 2), M * H)` gate projection.
    hc_scale: `(3,)` logit scales.
    hc_base: `(M * (M + 2),)` logit biases.
    sinkhorn_repeat: Number of Sinkhorn normalization rounds.

  Returns:
    The number of streams `M`.
  """
  _check_dtype("residual", residual, jnp.bfloat16)
  for name, x in (("fn", fn), ("hc_scale", hc_scale), ("hc_base", hc_base)):
    _check_dtype(name, x, jnp.float32)
  hc_mult = reference.hc_mult_from_mix_dim(fn.shape[0])
  if fn.shape[1] != residual.shape[-1]:
    raise ValueError(
        f"fn must have shape {(fn.shape[0], residual.shape[-1])} to match"
        f" residual's width, got {fn.shape}."
    )
  if hc_base.shape != (fn.shape[0],):
    raise ValueError(
        f"hc_base must have shape {(fn.shape[0],)}, got {hc_base.shape}."
    )
  if residual.shape[-1] % hc_mult:
    raise ValueError(
        f"residual width {residual.shape[-1]} must be a multiple of"
        f" hc_mult={hc_mult} (from fn's {fn.shape[0]} rows)."
    )
  if sinkhorn_repeat < 1:
    raise ValueError(f"sinkhorn_repeat must be >= 1, got {sinkhorn_repeat}.")
  return hc_mult


def _check_post_params(
    x: Any, residual: Any, post_layer_mix: Any, comb_res_mix: Any
) -> int:
  """Validates the `mhc_post` arguments shared by `MhcPost` and the fused op.

  Args:
    x: `(T, H)` sublayer output.
    residual: `(T, M * H)` residual streams.
    post_layer_mix: `(T, M)` post gates.
    comb_res_mix: `(T, M, M)` stream-mixing matrix.

  Returns:
    The number of streams `M`.
  """
  _check_dtype("x", x, jnp.bfloat16)
  _check_dtype("residual", residual, jnp.bfloat16)
  _check_dtype("post_layer_mix", post_layer_mix, jnp.float32)
  _check_dtype("comb_res_mix", comb_res_mix, jnp.float32)
  hc_mult = comb_res_mix.shape[-1]
  if residual.shape[-1] != hc_mult * x.shape[-1]:
    raise ValueError(
        f"residual must have width hc_mult * hidden_size = {hc_mult} *"
        f" {x.shape[-1]}, got {residual.shape[-1]}."
    )
  return hc_mult


class MhcPre[C](op.Op[Any, PreOutputs, None, C, Any]):
  """Tokamax operator for the mHC pre step."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      residual: Float[Array | AbstractArray, "T MH"],
      fn: Float[Array | AbstractArray, "M3 MH"],
      hc_scale: Float[Array | AbstractArray, "3"],
      hc_base: Float[Array | AbstractArray, "M3"],
      rms_eps: float,
      hc_pre_eps: float,
      hc_sinkhorn_eps: float,
      hc_post_mult_value: float,
      sinkhorn_repeat: int,
      *,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds the arguments. See `reference.mhc_pre` for their meaning."""
    _check_gate_params(residual, fn, hc_scale, hc_base, sinkhorn_repeat)
    return super().bind(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      residual: Float[Array, "T MH"],
      fn: Float[Array, "M3 MH"],
      hc_scale: Float[Array, "3"],
      hc_base: Float[Array, "M3"],
      rms_eps: float,
      hc_pre_eps: float,
      hc_sinkhorn_eps: float,
      hc_post_mult_value: float,
      sinkhorn_repeat: int,
      *,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[PreOutputs, None]:
    return (
        reference.mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        ),
        None,
    )


class MhcPost[C](op.Op[Any, jax.Array, None, C, Any]):
  """Tokamax operator for the mHC post step."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      x: Float[Array | AbstractArray, "T H"],
      residual: Float[Array | AbstractArray, "T MH"],
      post_layer_mix: Float[Array | AbstractArray, "T M"],
      comb_res_mix: Float[Array | AbstractArray, "T M M"],
      *,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds the arguments. See `reference.mhc_post` for their meaning."""
    _check_post_params(x, residual, post_layer_mix, comb_res_mix)
    return super().bind(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      x: Float[Array, "T H"],
      residual: Float[Array, "T MH"],
      post_layer_mix: Float[Array, "T M"],
      comb_res_mix: Float[Array, "T M M"],
      *,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[jax.Array, None]:
    return (
        reference.mhc_post(x, residual, post_layer_mix, comb_res_mix),
        None,
    )


class MhcFusedPostPre[C](op.Op[Any, FusedOutputs, None, C, Any]):
  """Tokamax operator for one layer's mHC post fused with the next one's pre."""

  @jaxtyping.jaxtyped
  def bind(
      self,
      x: Float[Array | AbstractArray, "T H"],
      residual: Float[Array | AbstractArray, "T MH"],
      post_layer_mix: Float[Array | AbstractArray, "T M"],
      comb_res_mix: Float[Array | AbstractArray, "T M M"],
      fn: Float[Array | AbstractArray, "M3 MH"],
      hc_scale: Float[Array | AbstractArray, "3"],
      hc_base: Float[Array | AbstractArray, "M3"],
      rms_eps: float,
      hc_pre_eps: float,
      hc_sinkhorn_eps: float,
      hc_post_mult_value: float,
      sinkhorn_repeat: int,
      *,
      return_residuals: bool = False,
  ) -> op.BoundArguments:
    """Binds the arguments. See `reference.mhc_fused_post_pre`."""
    post_hc_mult = _check_post_params(x, residual, post_layer_mix, comb_res_mix)
    pre_hc_mult = _check_gate_params(
        residual, fn, hc_scale, hc_base, sinkhorn_repeat
    )
    if post_hc_mult != pre_hc_mult:
      raise ValueError(
          f"comb_res_mix has hc_mult={post_hc_mult} but fn has"
          f" {fn.shape[0]} rows, i.e. hc_mult={pre_hc_mult}."
      )
    return super().bind(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        return_residuals=return_residuals,
    )

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      x: Float[Array, "T H"],
      residual: Float[Array, "T MH"],
      post_layer_mix: Float[Array, "T M"],
      comb_res_mix: Float[Array, "T M M"],
      fn: Float[Array, "M3 MH"],
      hc_scale: Float[Array, "3"],
      hc_base: Float[Array, "M3"],
      rms_eps: float,
      hc_pre_eps: float,
      hc_sinkhorn_eps: float,
      hc_post_mult_value: float,
      sinkhorn_repeat: int,
      *,
      return_residuals: bool = False,
      config: C | None = None,
  ) -> tuple[FusedOutputs, None]:
    return (
        reference.mhc_fused_post_pre(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        ),
        None,
    )
