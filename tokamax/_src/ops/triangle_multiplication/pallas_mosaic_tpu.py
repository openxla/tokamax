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
"""Pallas-Mosaic-TPU implementation of the triangle multiplication op."""

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
from tokamax._src.ops.triangle_multiplication import pallas_mosaic_tpu_vjp as vjp

Residuals = base.Residuals
Config = common.Config
Key = common.Key


@dataclasses.dataclass(frozen=True, slots=True)
class PallasMosaicTpuTriangleMultiplication(
    base.TriangleMultiplication[Config, Key]
):
  """Pallas-Mosaic-TPU triangle multiplicative update.

  Both the forward pass and the backward pass (`vjp`) are computed by fused
  Pallas-Mosaic-TPU kernels (prologue and epilogue) around an XLA batched
  matmul.
  """

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(
          self, "vjp", vjp.PallasMosaicTpuTriangleMultiplicationVjp()
      )

  @override
  def bind(self, *args: Any, **kwargs: Any) -> op.BoundArguments:
    try:
      ba = base.TriangleMultiplication.bind(self, *args, **kwargs)
    except ValueError as e:
      if "precision" in kwargs or "precision" in str(e).lower():
        raise NotImplementedError(
            f"precision={kwargs.get('precision')!r} is not supported."
        ) from e
      raise
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

    n, _, _ = x.shape
    _, d = projection_out_weights.shape
    dtype = x.dtype
    n_pad = pl.cdiv(n, 128) * 128

    x_pad = common.transpose_and_pad_icj(x, n_pad)
    mask_pad = common.cast_and_pad_mask(mask, n_pad, dtype, x)

    left_pad, right_pad = kernel.prologue(
        x_pad,
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

    if n != n_pad:
      left = left_pad[:, :n, :n]
      right = right_pad[:, :n, :n]
    else:
      left, right = left_pad, right_pad
    t3 = common.contract(left, right, triangle_type, precision)
    t3_pad = common.pad_dim(common.pad_dim(t3, n_pad, 1), n_pad, 2)

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
    out_pad = kernel.epilogue(
        t3_pad,
        x_pad,
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
    )

    out_idj = out_pad[:n, :d, :n] if n != n_pad else out_pad
    out = jnp.transpose(out_idj, (0, 2, 1))
    residuals = (left, right, t3) if return_residuals else None
    return out, residuals

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return kernel.get_heuristics_config(ba)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    return kernel.get_autotuning_configs(ba)
