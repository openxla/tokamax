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
"""Pallas-Mosaic-GPU implementation of the triangle multiplication op."""

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
from tokamax._src.ops.triangle_multiplication import pallas_mosaic_gpu_vjp as vjp

Residuals = base.Residuals
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
class PallasMosaicGpuTriangleMultiplication(
    base.TriangleMultiplication[Config, Key]
):
  """Pallas-Mosaic-GPU triangle multiplicative update.

  Both the forward pass and the backward pass (`vjp`) are computed by fused
  Pallas-Mosaic-GPU kernels (prologue and epilogue) around an XLA batched
  matmul.
  """

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(
          self, "vjp", vjp.PallasMosaicGpuTriangleMultiplicationVjp()
      )

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_mosaic_gpu_support(device) and (
        gpu_utils.is_sm90(device) or gpu_utils.is_sm100(device)
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
      precision: base.CanonicalPrecision,
      epsilon: float,
      return_residuals: bool,
      implementation: (
          base.Implementation | Sequence[base.Implementation] | None
      ) = None,
      config: Config,
  ) -> tuple[Float[Array, "N N D"], Residuals | None]:
    del implementation  # Unused.

    common.validate_fwd_inputs(
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

    x_pad = common.pad_dim(common.pad_dim(x_flat, m_pad, 0), c_pad, 1)
    mask_pad = common.pad_dim(mask_flat, m_pad, 0).view(jnp.int8)
    w_proj_in_pad = common.pad_dim(
        common.pad_dim(projection_in_weights, c_pad, 0), h_pad, 2
    )
    w_gate_in_pad = common.pad_dim(
        common.pad_dim(gate_in_weights, c_pad, 0), h_pad, 2
    )
    ln_in_scale_pad = common.pad_dim(layernorm_in_scale, c_pad, 0)
    ln_in_offset_pad = common.pad_dim(layernorm_in_offset, c_pad, 0)

    pro_persistent = (
        True
        if config.prologue_persistent is None
        else config.prologue_persistent
    )
    epi_persistent = (
        True
        if config.epilogue_persistent is None
        else config.epilogue_persistent
    )

    left_pad, right_pad = kernel_mod.prologue(
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
        block_m=config.block_m,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        persistent=pro_persistent,
        ctas_per_sm=config.ctas_per_sm,
    )

    if (h, m) != left_pad.shape:
      left_pad = left_pad[:h, :m]
      right_pad = right_pad[:h, :m]
    left = left_pad.reshape(h, n, n)
    right = right_pad.reshape(h, n, n)
    t3 = common.contract(left, right, triangle_type, precision)

    t3_pad = common.pad_dim(
        common.pad_dim(t3.reshape(h, m), h_pad, 0), m_pad, 1
    )
    w_proj_out_pad = common.pad_dim(
        common.pad_dim(projection_out_weights, h_pad, 0), d_pad, 1
    )
    w_gate_out_pad = common.pad_dim(
        common.pad_dim(gate_out_weights, c_pad, 0), d_pad, 1
    )
    ln_out_scale_pad = common.pad_dim(layernorm_out_scale, h_pad, 0)
    ln_out_offset_pad = common.pad_dim(layernorm_out_offset, h_pad, 0)

    out_pad = kernel_mod.epilogue(
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
        d_real=d,
        m_real=m,
        block_m=config.epilogue_block_m or config.block_m,
        num_warps=config.epilogue_num_warps or config.num_warps,
        num_stages=config.epilogue_num_stages or config.num_stages,
        persistent=epi_persistent,
        ctas_per_sm=config.ctas_per_sm,
    )

    out_flat = out_pad[:m, :d] if (m, d) != out_pad.shape else out_pad
    residuals = (left, right, t3) if return_residuals else None
    return out_flat.reshape(n, n, d), residuals

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return _get_kernel_module().get_heuristics_config(ba)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    return _get_kernel_module().get_autotuning_configs(ba)
