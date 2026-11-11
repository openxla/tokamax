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
"""Pallas-Mosaic-GPU Triangle Multiplication VJP implementation."""

from collections.abc import Sequence
import dataclasses
from typing import ClassVar, Literal, override

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float  # pylint: disable=g-importing-member,g-multiple-import
from tokamax._src import gpu_utils
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.triangle_multiplication import base
from tokamax._src.ops.triangle_multiplication import pallas_mosaic_gpu_common as common
from tokamax._src.ops.triangle_multiplication import pallas_mosaic_gpu_kernel_sm100 as sm100
from tokamax._src.ops.triangle_multiplication import pallas_mosaic_gpu_kernel_sm90 as sm90

Config = common.Config
Key = common.Key


def _get_kernel_module():
  if not gpu_utils.has_mosaic_gpu_support():
    raise NotImplementedError("Mosaic GPU not supported on this platform.")
  if gpu_utils.is_sm100():
    return sm100
  if gpu_utils.is_sm90():
    return sm90
  raise NotImplementedError("Only supported for sm90 and sm100 GPUs.")


@dataclasses.dataclass(frozen=True, slots=True)
class PallasMosaicGpuTriangleMultiplicationVjp(
    base.TriangleMultiplicationVjp[Config, Key]
):
  """Pallas-Mosaic-GPU triangle multiplicative update VJP."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_mosaic_gpu_support(device) and (
        gpu_utils.is_sm90(device) or gpu_utils.is_sm100(device)
    )

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
    """Computes gradients with fused Pallas-Mosaic-GPU backward kernels."""
    del out, implementation  # Unused.

    common.validate_vjp_inputs(
        residuals,
        dout,
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
        return_residuals,
    )

    kernel_mod = _get_kernel_module()
    n, _, c = x.shape
    h, d = projection_out_weights.shape
    m = n * n

    c_pad = max(common.MIN_KERNEL_CHANNEL_DIM, c)
    h_pad = max(common.MIN_KERNEL_CHANNEL_DIM, h)
    d_pad = max(common.MIN_KERNEL_CHANNEL_DIM, d)
    m_pad = pl.cdiv(m, 128) * 128

    x_flat = lax.collapse(x, 0, 2)
    mask_flat = lax.collapse(mask, 0, 2)
    dout_flat = lax.collapse(dout, 0, 2)
    left, right, t3 = residuals
    t3_flat = lax.collapse(t3, 1, 3)

    x_pad = common.pad_dim(common.pad_dim(x_flat, m_pad, 0), c_pad, 1)
    mask_pad = common.pad_dim(mask_flat, m_pad, 0).view(jnp.int8)
    dout_pad = common.pad_dim(common.pad_dim(dout_flat, m_pad, 0), d_pad, 1)
    t3_pad = common.pad_dim(common.pad_dim(t3_flat, h_pad, 0), m_pad, 1)

    w_proj_in_pad = common.pad_dim(
        common.pad_dim(projection_in_weights, c_pad, 0), h_pad, 2
    )
    w_gate_in_pad = common.pad_dim(
        common.pad_dim(gate_in_weights, c_pad, 0), h_pad, 2
    )
    w_proj_out_pad = common.pad_dim(
        common.pad_dim(projection_out_weights, h_pad, 0), d_pad, 1
    )
    w_gate_out_pad = common.pad_dim(
        common.pad_dim(gate_out_weights, c_pad, 0), d_pad, 1
    )
    ln_in_scale_pad = common.pad_dim(layernorm_in_scale, c_pad, 0)
    ln_in_offset_pad = common.pad_dim(layernorm_in_offset, c_pad, 0)
    ln_out_scale_pad = common.pad_dim(layernorm_out_scale, h_pad, 0)
    ln_out_offset_pad = common.pad_dim(layernorm_out_offset, h_pad, 0)

    epi_block_m = config.epilogue_block_m or config.block_m
    epi_num_warps = config.epilogue_num_warps or config.num_warps
    epi_num_stages = config.epilogue_num_stages or config.num_stages
    epi_persistent = (
        True
        if config.epilogue_persistent is None
        else config.epilogue_persistent
    )
    pro_block_m = config.block_m
    pro_num_warps = config.num_warps
    pro_num_stages = config.num_stages
    pro_persistent = (
        True
        if config.prologue_persistent is None
        else config.prologue_persistent
    )

    (
        dt3_pad,
        dleft_act_epi_pad,
        dw_proj_out_pad,
        dw_gate_out_pad,
        dln_out_scale_pad,
        dln_out_offset_pad,
    ) = kernel_mod.epilogue_bwd(
        dout_pad,
        t3_pad,
        x_pad,
        w_proj_out_pad,
        w_gate_out_pad,
        ln_in_scale_pad,
        ln_in_offset_pad,
        ln_out_scale_pad,
        ln_out_offset_pad,
        eps=epsilon,
        c_real=c,
        h_real=h,
        m_real=m,
        block_m=epi_block_m,
        num_warps=epi_num_warps,
        num_stages=epi_num_stages,
        persistent=epi_persistent,
        ctas_per_sm=config.ctas_per_sm,
    )

    if (h, m) != dt3_pad.shape:
      dt3_pad = dt3_pad[:h, :m]
    dt3 = dt3_pad.reshape(h, n, n)
    dleft, dright = common.contract_bwd(
        dt3, left, right, triangle_type, precision
    )

    dleft_pad = common.pad_dim(
        common.pad_dim(dleft.reshape(h, m), h_pad, 0), m_pad, 1
    )
    dright_pad = common.pad_dim(
        common.pad_dim(dright.reshape(h, m), h_pad, 0), m_pad, 1
    )

    (
        dx_pad,
        dw_proj_in_pad,
        dw_gate_in_pad,
        dln_in_scale_pad,
        dln_in_offset_pad,
    ) = kernel_mod.prologue_bwd(
        dleft_pad,
        dright_pad,
        dleft_act_epi_pad,
        x_pad,
        mask_pad,
        w_proj_in_pad,
        w_gate_in_pad,
        ln_in_scale_pad,
        ln_in_offset_pad,
        eps=epsilon,
        c_real=c,
        h_real=h,
        m_real=m,
        block_m=pro_block_m,
        num_warps=pro_num_warps,
        num_stages=pro_num_stages,
        persistent=pro_persistent,
        ctas_per_sm=config.ctas_per_sm,
    )

    if (m, c) != dx_pad.shape:
      dx_pad = dx_pad[:m, :c]
    dx = dx_pad.reshape(n, n, c)
    dw_proj_in = (
        dw_proj_in_pad[:c, :, :h]
        if (c, 2, h) != dw_proj_in_pad.shape
        else dw_proj_in_pad
    )
    dw_gate_in = (
        dw_gate_in_pad[:c, :, :h]
        if (c, 2, h) != dw_gate_in_pad.shape
        else dw_gate_in_pad
    )
    dw_proj_out = (
        dw_proj_out_pad[:h, :d]
        if (h, d) != dw_proj_out_pad.shape
        else dw_proj_out_pad
    )
    dw_gate_out = (
        dw_gate_out_pad[:c, :d]
        if (c, d) != dw_gate_out_pad.shape
        else dw_gate_out_pad
    )
    dln_in_scale = (
        dln_in_scale_pad[:c]
        if (c,) != dln_in_scale_pad.shape
        else dln_in_scale_pad
    )
    dln_in_offset = (
        dln_in_offset_pad[:c]
        if (c,) != dln_in_offset_pad.shape
        else dln_in_offset_pad
    )
    dln_out_scale = (
        dln_out_scale_pad[:h]
        if (h,) != dln_out_scale_pad.shape
        else dln_out_scale_pad
    )
    dln_out_offset = (
        dln_out_offset_pad[:h]
        if (h,) != dln_out_offset_pad.shape
        else dln_out_offset_pad
    )

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
    return _get_kernel_module().get_vjp_heuristics_config(ba)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    return _get_kernel_module().get_vjp_autotuning_configs(ba)
