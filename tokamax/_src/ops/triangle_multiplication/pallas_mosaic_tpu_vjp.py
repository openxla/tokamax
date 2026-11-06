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
"""Pallas-Mosaic-TPU Triangle Multiplication VJP implementation."""

from collections.abc import Sequence
import dataclasses
from typing import Any, ClassVar, Literal, override

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float  # pylint: disable=g-importing-member,g-multiple-import
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.triangle_multiplication import base
from tokamax._src.ops.triangle_multiplication import pallas_mosaic_tpu_common as common
from tokamax._src.ops.triangle_multiplication import pallas_mosaic_tpu_kernel as kernel

Config = common.Config
Key = common.Key


@dataclasses.dataclass(frozen=True, slots=True)
class PallasMosaicTpuTriangleMultiplicationVjp(
    base.TriangleMultiplicationVjp[Config, Key]
):
  """Pallas-Mosaic-TPU triangle multiplicative update VJP."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  @override
  def bind(self, *args: Any, **kwargs: Any) -> op.BoundArguments:
    if args and isinstance(args[0], list):
      args = (tuple(args[0]), *args[1:])
    elif "residuals" in kwargs and isinstance(kwargs["residuals"], list):
      kwargs = dict(kwargs, residuals=tuple(kwargs["residuals"]))
    try:
      ba = base.TriangleMultiplicationVjp.bind(self, *args, **kwargs)
    except ValueError as e:
      if "precision" in kwargs or "precision" in str(e).lower():
        raise NotImplementedError(
            f"precision={kwargs.get('precision')!r} is not supported."
        ) from e
      raise
    res = ba.arguments["residuals"]
    if not isinstance(res, tuple) or len(res) != 3:
      raise NotImplementedError(
          "Expected residuals to be a 3-tuple (left, right, t3)."
      )
    for r in res:
      if getattr(r, "ndim", None) != 3:
        raise NotImplementedError("Expected 3D residual arrays.")
    prec = ba.arguments["precision"]
    if prec not in (
        (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        jax.lax.DotAlgorithmPreset.DEFAULT,
        jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
    ):
      raise NotImplementedError(f"{prec=} is not supported.")
    x = ba.arguments["x"]
    w_out = ba.arguments["projection_out_weights"]
    if getattr(x, "ndim", None) != 3 or getattr(w_out, "ndim", None) != 2:
      raise NotImplementedError("Unsupported input rank.")
    return ba

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return common.has_mosaic_tpu_support(device)

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      residuals: base.Residuals,
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
      precision: base.CanonicalPrecision,
      epsilon: float,
      return_residuals: bool,
      implementation: (
          base.Implementation | Sequence[base.Implementation] | None
      ) = None,
      config: Config,
  ) -> tuple[base.Grads, None]:
    """Computes gradients with fused Pallas-Mosaic-TPU backward kernels."""
    common.validate_vjp_inputs(
        residuals,
        out,
        dout,
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
        return_residuals,
    )
    del out, implementation  # Unused.

    n, _, c = x.shape
    dtype = x.dtype
    n_pad = pl.cdiv(n, 128) * 128
    unpin_x_pad = common.should_unpin_xpad(x, n_pad)

    left, right, t3 = residuals

    dout_pad = common.transpose_and_pad_icj(dout, n_pad)
    t3_pad = common.pad_dim(common.pad_dim(t3, n_pad, 1), n_pad, 2)
    x_pad_epi = common.transpose_and_pad_icj(x, n_pad)

    epi_block_m = config.epilogue_block_m or config.block_m
    epi_block_i = (
        config.epilogue_block_i
        if config.epilogue_block_m is not None
        else (config.epilogue_block_i or config.block_i)
    )
    epi_block_j = (
        config.epilogue_block_j
        if config.epilogue_block_m is not None
        else (config.epilogue_block_j or config.block_j)
    )
    (
        dt3_pad,
        dleft_act_epi_pad,
        dw_proj_out,
        dw_gate_out,
        dln_out_scale,
        dln_out_offset,
    ) = kernel.epilogue_bwd(
        dout_pad,
        t3_pad,
        x_pad_epi,
        projection_out_weights,
        gate_out_weights,
        layernorm_in_scale,
        layernorm_in_offset,
        layernorm_out_scale,
        layernorm_out_offset,
        eps=epsilon,
        block_m=epi_block_m,
        block_i=epi_block_i,
        block_j=epi_block_j,
        alias_x_pad=unpin_x_pad,
    )

    dt3 = dt3_pad[:, :n, :n] if n != n_pad else dt3_pad
    dleft, dright = common.contract_bwd(
        dt3, left, right, triangle_type, precision
    )

    dleft_pad = common.pad_dim(common.pad_dim(dleft, n_pad, 1), n_pad, 2)
    dright_pad = common.pad_dim(common.pad_dim(dright, n_pad, 1), n_pad, 2)

    if unpin_x_pad:
      del x_pad_epi
      x_for_pro, mask_for_pro, dleft_pad, dright_pad = (
          jax.lax.optimization_barrier((x, mask, dleft_pad, dright_pad))
      )
      x_pad_pro = common.transpose_and_pad_icj(x_for_pro, n_pad)
    else:
      x_pad_pro = x_pad_epi
      mask_for_pro = mask
    mask_pad = common.cast_and_pad_mask(mask_for_pro, n_pad, dtype, x)

    (
        dx_pad,
        dw_proj_in,
        dw_gate_in,
        dln_in_scale,
        dln_in_offset,
    ) = kernel.prologue_bwd(
        dleft_pad,
        dright_pad,
        dleft_act_epi_pad,
        x_pad_pro,
        mask_pad,
        projection_in_weights,
        gate_in_weights,
        layernorm_in_scale,
        layernorm_in_offset,
        eps=epsilon,
        block_m=config.block_m,
        block_i=config.block_i,
        block_j=config.block_j,
    )

    dx_icj = dx_pad[:n, :c, :n] if n != n_pad else dx_pad
    dx = jnp.transpose(dx_icj, (0, 2, 1))

    return (
        base.Grads(
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

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return kernel.get_vjp_heuristics_config(ba)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    return kernel.get_vjp_autotuning_configs(ba)
