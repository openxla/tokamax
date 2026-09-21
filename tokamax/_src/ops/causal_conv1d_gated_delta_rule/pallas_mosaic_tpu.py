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
import functools
import itertools
from typing import Annotated, ClassVar, Optional, override

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import pydantic
from tokamax._src.ops import op
from tokamax._src.ops.causal_conv1d_gated_delta_rule import base
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config as config_lib
from tokamax._src.ops.causal_conv1d_gated_delta_rule import tiling
from tokamax._src.ops.causal_conv1d_gated_delta_rule import wrapper

# The candidates searched by `tiling.calculate_decode_tile_size` and
# `tiling.calculate_mixed_tile_size`, which stop at the first that fits in
# VMEM. The autotuner benchmarks all of them.
_DECODE_TILE_CANDIDATES = (1, 2, 4, 8, 16, 32)
_MIXED_TILE_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128)


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Pallas Mosaic TPU fused Conv1D + GDN config.

  A tile size of `None` is unset; `_get_heuristics_config` derives both from the
  input shapes.
  """

  decode_tile_size: Annotated[int, pydantic.Field(gt=0)] | None = None
  mixed_tile_size: Annotated[int, pydantic.Field(gt=0)] | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicTpuCausalConv1dGatedDeltaRule(
    base.CausalConv1dGatedDeltaRule[Config]
):
  """Wrapper for the tokamax Op API for Pallas Mosaic TPU kernel."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    # The same derivation `wrapper.fused_conv1d_gdn` applies when a tile size
    # is `None`, hoisted here so that the chosen sizes are visible to the
    # autotuning cache rather than being derived deeper in the call stack.
    args = ba.arguments
    qkv, conv_state = args["qkv"], args["conv_state"]
    batch_size = qkv.shape[0]
    packing = 4 // jnp.dtype(qkv.dtype).itemsize
    decode_tile_size, mixed_tile_size = tiling.get_tile_sizes(
        batch_size=batch_size,
        num_seqs=args["state_indices"].size,
        padded_batch_size=tiling.align_to(batch_size, packing),
        n_kq=args["n_kq"],
        n_v=args["n_v"],
        d_k=args["d_k"],
        d_v=args["d_v"],
        kernel_size=args["kernel_size"],
        conv_state_dim_size=conv_state.shape[-1],
        act_in_dtype=qkv.dtype,
        act_out_dtype=qkv.dtype,
        conv_state_dtype=conv_state.dtype,
        recurrent_state_dtype=args["recurrent_state"].dtype,
        num_lanes=pltpu.get_tpu_info().num_lanes,
    )
    return Config(
        decode_tile_size=decode_tile_size, mixed_tile_size=mixed_tile_size
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    args = ba.arguments
    qkv, conv_state = args["qkv"], args["conv_state"]
    n_v, distribution = args["n_v"], args["distribution"]
    vmem_limit = config_lib.get_vmem_limit_bytes()

    vmem_bytes = functools.partial(
        tiling.get_vmem_estimate_bytes,
        n_kq=args["n_kq"],
        n_v=n_v,
        d_k=args["d_k"],
        d_v=args["d_v"],
        kernel_size=args["kernel_size"],
        act_in_bytes=jnp.dtype(qkv.dtype).itemsize,
        act_out_bytes=jnp.dtype(qkv.dtype).itemsize,
        conv_state_bytes=jnp.dtype(conv_state.dtype).itemsize,
        rec_state_bytes=jnp.dtype(args["recurrent_state"].dtype).itemsize,
        num_lanes=pltpu.get_tpu_info().num_lanes,
        conv_state_dim_size=conv_state.shape[-1],
    )

    # `wrapper.fused_conv1d_gdn` runs the decode pass only when
    # `distribution[0] > 0` and the prefill pass only when
    # `distribution[-1] > distribution[0]`. Searching the tile size of a pass
    # that does not run would only multiply compile time.
    decode_tiles = [None]
    if distribution[0] > 0:
      # Mirrors the batch padding in `wrapper.fused_conv1d_gdn`.
      packing = 4 // jnp.dtype(qkv.dtype).itemsize
      padded_batch_size = tiling.align_to(qkv.shape[0], packing)
      decode_tiles = [
          c
          for c in _DECODE_TILE_CANDIDATES
          if c <= padded_batch_size
          and vmem_bytes(tile_b=c, chunk_sz=1, is_decode=True) <= vmem_limit
      ] or [1]

    mixed_tiles = [None]
    if distribution[-1] > distribution[0]:
      # The same chunk cap as `tiling.calculate_mixed_tile_size`.
      max_chunk = 64 if n_v >= 64 else 128
      mixed_tiles = [
          c
          for c in _MIXED_TILE_CANDIDATES
          if c <= max_chunk
          and vmem_bytes(tile_b=1, chunk_sz=c, is_decode=False) <= vmem_limit
      ] or [1]

    return {
        Config(decode_tile_size=d, mixed_tile_size=m)
        for d, m in itertools.product(decode_tiles, mixed_tiles)
    }

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
      *,
      n_kq: int,
      n_v: int,
      d_k: int,
      d_v: int,
      kernel_size: int,
      zero_initialize_out: bool = True,
      compute_precision: jnp.dtype = jnp.float32.dtype,
      # Must have a default: the base `_fwd` signature defaults `config`, so an
      # override cannot make it required. Unused in practice, as `Op.__call__`
      # always passes a config explicitly.
      config: Config = Config(),
      return_residuals: bool = False,
  ) -> tuple[tuple[tuple[jax.Array, jax.Array], jax.Array], None]:
    del return_residuals
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
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            zero_initialize_out=zero_initialize_out,
            compute_precision=compute_precision,
            decode_tile_size=config.decode_tile_size,
            mixed_tile_size=config.mixed_tile_size,
        ),
        None,
    )

  @override
  def supported_on(self, device: jax.Device) -> bool:
    try:
      return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 6
    except Exception:
      return False
