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
"""Pallas Mosaic TPU kernel implementation for Causal Conv1D Gated Delta Rule."""

import dataclasses
import itertools
from typing import Annotated, ClassVar, Optional, override

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import pydantic
from tokamax._src.ops import op
from tokamax._src.ops.causal_conv1d_gated_delta_rule import base
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config as gdn_config
from tokamax._src.ops.causal_conv1d_gated_delta_rule import tiling
from tokamax._src.ops.causal_conv1d_gated_delta_rule import wrapper

GDNConfig = gdn_config.GDNConfig


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Tile sizes for the kernel. `None` falls back to the VMEM heuristic."""

  decode_tile_size: Annotated[int, pydantic.Field(gt=0)] | None = None
  mixed_tile_size: Annotated[int, pydantic.Field(gt=0)] | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicTpuCausalConv1dGatedDeltaRule(
    base.CausalConv1dGatedDeltaRule[Config]
):
  """Wrapper for the tokamax Op API for Pallas Mosaic TPU kernel."""

  config_cls: ClassVar[type[Config]] = Config

  def _fwd(
      self,
      qkv: jax.Array,
      b: jax.Array,
      a: jax.Array,
      conv_state: jax.Array,
      recurrent_state: jax.Array,
      conv_weight: jax.Array,
      conv_bias: Optional[jax.Array],
      a_log: jax.Array,
      dt_bias: jax.Array,
      query_start_loc: jax.Array,
      state_indices: jax.Array,
      distribution: jax.Array,
      seq_lens: jax.Array,
      read_state_indices: Optional[jax.Array] = None,
      read_offsets: Optional[jax.Array] = None,
      *,
      n_kq: int,
      n_v: int,
      d_k: int,
      d_v: int,
      kernel_size: int,
      num_spec_tokens: int = 0,
      zero_initialize_out: bool = True,
      compute_precision: jnp.dtype = jnp.float32.dtype,
      decode_tile_size: int | None = None,
      mixed_tile_size: int | None = None,
      config: Config | None = None,
      return_residuals: bool = False,
  ) -> tuple[tuple[tuple[jax.Array, jax.Array], jax.Array], None]:
    del return_residuals
    # Precedence: explicit kwarg > autotuned config > VMEM heuristic.
    if config is not None:
      if decode_tile_size is None:
        decode_tile_size = config.decode_tile_size
      if mixed_tile_size is None:
        mixed_tile_size = config.mixed_tile_size

    return (
        wrapper.fused_conv1d_gdn(
            qkv=qkv,
            b=b,
            a=a,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            read_state_indices=read_state_indices,
            read_offsets=read_offsets,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            num_spec_tokens=num_spec_tokens,
            zero_initialize_out=zero_initialize_out,
            compute_precision=compute_precision,
            decode_tile_size=decode_tile_size,
            mixed_tile_size=mixed_tile_size,
        ),
        None,
    )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba  # Unused.
    return Config()

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    # Tiles larger than the token count get clamped down to it, so skip them
    # to avoid benchmarking duplicate kernels.
    num_tokens = ba.arguments["qkv"].shape[0]
    return {
        Config(decode_tile_size=decode_size, mixed_tile_size=mixed_size)
        for decode_size, mixed_size in itertools.product(
            tiling.DECODE_TILE_SIZES, tiling.MIXED_TILE_SIZES
        )
        if decode_size <= num_tokens and mixed_size <= num_tokens
    }

  @override
  def supported_on(self, device: jax.Device) -> bool:
    try:
      return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 6
    except Exception:
      return False
