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
"""Pallas/Mosaic TPU operator implementations of DeepSeek-V4 mHC."""

from typing import Annotated, ClassVar, override

import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.mhc import base
from tokamax._src.ops.experimental.tpu.mhc import fused_post_pre_kernel
from tokamax._src.ops.experimental.tpu.mhc import post_kernel
from tokamax._src.ops.experimental.tpu.mhc import pre_kernel
from tokamax._src.ops.experimental.tpu.mhc import reference
from tokamax._src.ops.experimental.tpu.mhc import utils

# Token block sizes the autotuner tries. The pre and post kernels halve a block
# that doesn't fit in VMEM, so their large candidates are safe. The fused kernel
# keeps the block it is given, so its candidates stop at 64, the largest block
# `fused_post_pre_kernel` documents as fitting at DeepSeek-V4 shapes.
_PRE_POST_TOKEN_BLOCK_CANDIDATES = (32, 64, 128, 256)
_FUSED_TOKEN_BLOCK_CANDIDATES = (16, 32, 64)


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning parameters for the mHC Pallas kernels.

  Attributes:
    token_block_size: Tokens per grid step, a multiple of 16 (the bf16 sublane
      tile). Smaller token counts use one block covering all tokens.
  """

  token_block_size: Annotated[
      int, pydantic.Field(gt=0, multiple_of=utils.SUBLANE)
  ] = 64


def _token_block_configs(ba: op.BoundArguments, candidates) -> set[Config]:
  """Returns the candidates, capped at the (sublane-padded) token count."""
  num_tokens = ba.arguments["residual"].shape[0]
  max_block = max(
      utils.SUBLANE,
      -(-num_tokens // utils.SUBLANE) * utils.SUBLANE,
  )
  return {Config(token_block_size=min(c, max_block)) for c in candidates}


def _supported_on(device: jax.Device) -> bool:
  """TPU v5 and newer, the generations the kernels are tested on."""
  return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5


def _pre_gates(
    mixes: jax.Array,
    sqrsum: jax.Array,
    residual: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array]:
  """XLA gates on the kernel's mixes; returns `(post_mix, comb_mix)`."""
  hc_mult = reference.hc_mult_from_mix_dim(fn.shape[0])
  _, post_mix, comb_mix = utils.mhc_pre_gates(
      mixes,
      sqrsum,
      hc_mult,
      residual.shape[-1] // hc_mult,
      hc_scale,
      hc_base,
      rms_eps,
      hc_pre_eps,
      hc_sinkhorn_eps,
      hc_post_mult_value,
      sinkhorn_repeat,
  )
  return post_mix, comb_mix


class PallasTpuMhcPre(base.MhcPre[Config]):
  """Pallas TPU `MhcPre`: the mix GEMM and the stream collapse in one kernel."""

  config_cls: ClassVar[type[Config]] = Config

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
      config: Config | None = None,
  ) -> tuple[base.PreOutputs, None]:
    if config is None:
      config = Config()
    mixes, sqrsum, layer_input = pre_kernel.mhc_pre_mixes_collapse(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        token_block_size=config.token_block_size,
    )
    post_mix, comb_mix = _pre_gates(
        mixes,
        sqrsum,
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    return (post_mix, comb_mix, layer_input), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config(token_block_size=64)  # Upstream's default.

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    return _token_block_configs(ba, _PRE_POST_TOKEN_BLOCK_CANDIDATES)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return _supported_on(device)


class PallasTpuMhcPost(base.MhcPost[Config]):
  """Pallas TPU `MhcPost`: the unrolled stream recombine."""

  config_cls: ClassVar[type[Config]] = Config

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
      config: Config | None = None,
  ) -> tuple[jax.Array, None]:
    if config is None:
      config = Config()
    return (
        post_kernel.mhc_post_2d(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            token_block_size=config.token_block_size,
        ),
        None,
    )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config(token_block_size=64)  # Upstream's default.

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    return _token_block_configs(ba, _PRE_POST_TOKEN_BLOCK_CANDIDATES)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return _supported_on(device)


class PallasTpuMhcFusedPostPre(base.MhcFusedPostPre[Config]):
  """Pallas TPU `MhcFusedPostPre`: post recombine, mix GEMM and collapse."""

  config_cls: ClassVar[type[Config]] = Config

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
      config: Config | None = None,
  ) -> tuple[base.FusedOutputs, None]:
    if config is None:
      config = Config(token_block_size=32)
    num_tokens, hc_mult = post_layer_mix.shape
    new_residual, mixes, sqrsum, layer_input = (
        fused_post_pre_kernel.fused_post_pre_mixes(
            x,
            residual,
            post_layer_mix,
            comb_res_mix.reshape(num_tokens, hc_mult * hc_mult),
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            token_block_size=config.token_block_size,
        )
    )
    post_mix, comb_mix = _pre_gates(
        mixes,
        sqrsum,
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    return (new_residual, post_mix, comb_mix, layer_input), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config(token_block_size=32)  # Upstream's default.

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    return _token_block_configs(ba, _FUSED_TOKEN_BLOCK_CANDIDATES)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return _supported_on(device)
