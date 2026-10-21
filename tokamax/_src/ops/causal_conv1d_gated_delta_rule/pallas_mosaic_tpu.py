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
from jax.experimental import pallas as pl
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
    # Resolve the VMEM heuristic to concrete tile sizes, mirroring
    # `wrapper.fused_conv1d_gdn`. Returning `Config()` instead would make the
    # heuristic a separate autotuning candidate that duplicates one of the
    # explicit configs and is serialized as null tile sizes in the cache.
    args = ba.arguments
    qkv = args["qkv"]
    batch_size = qkv.shape[0]
    num_seqs = args["state_indices"].shape[0]
    window_size = args["num_spec_tokens"] + 1
    packing = 4 // jnp.dtype(qkv.dtype).itemsize
    decode_tile_size, mixed_tile_size = tiling.get_tile_sizes(
        batch_size=batch_size,
        num_seqs=num_seqs,
        padded_batch_size=pl.cdiv(batch_size, packing) * packing,
        n_kq=args["n_kq"],
        n_v=args["n_v"],
        d_k=args["d_k"],
        d_v=args["d_v"],
        kernel_size=args["kernel_size"],
        conv_state_dim_size=args["conv_state"].shape[-1],
        act_out_dtype=qkv.dtype,
        recurrent_state_dtype=args["recurrent_state"].dtype,
        num_lanes=pltpu.get_tpu_info().num_lanes,
        decode_vmem_limit_bytes=GDNConfig.get_vmem_limit_bytes(window_size),
        mixed_vmem_limit_bytes=GDNConfig.get_vmem_limit_bytes(),
        window_size=window_size,
    )
    return Config(
        decode_tile_size=decode_tile_size, mixed_tile_size=mixed_tile_size
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    # Skip tiles that exceed the token count (which clamp to duplicates) or
    # exceed the VMEM budget.
    args = ba.arguments
    qkv = args["qkv"]
    num_tokens, dim = qkv.shape
    num_seqs = args["state_indices"].shape[0]
    recurrent_state = args["recurrent_state"]
    n_kq = args["n_kq"]
    n_v = args["n_v"]
    d_k = args["d_k"]
    d_v = args["d_v"]
    kernel_size = args["kernel_size"]
    window_size = args["num_spec_tokens"] + 1
    num_lanes = pltpu.get_tpu_info().num_lanes
    decode_vmem_limit_bytes = GDNConfig.get_vmem_limit_bytes(window_size)
    mixed_vmem_limit_bytes = GDNConfig.get_vmem_limit_bytes()
    max_decode_size = min(
        num_tokens,
        tiling.calculate_decode_tile_size(
            batch_size=min(num_tokens, num_seqs),
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            conv_state_dim_size=dim,
            act_out_dtype=qkv.dtype,
            recurrent_state_dtype=recurrent_state.dtype,
            num_lanes=num_lanes,
            vmem_capacity_limit_bytes=decode_vmem_limit_bytes,
            kernel_size=kernel_size,
            window_size=window_size,
        ),
    )
    max_mixed_size = min(
        num_tokens,
        tiling.calculate_mixed_tile_size(
            seq_len=num_tokens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            conv_state_dim_size=dim,
            act_out_dtype=qkv.dtype,
            recurrent_state_dtype=recurrent_state.dtype,
            num_lanes=num_lanes,
            vmem_capacity_limit_bytes=mixed_vmem_limit_bytes,
            kernel_size=kernel_size,
        ),
    )
    return {
        Config(decode_tile_size=decode_size, mixed_tile_size=mixed_size)
        for decode_size, mixed_size in itertools.product(
            GDNConfig.DECODE_TILE_SIZES, GDNConfig.MIXED_TILE_SIZES
        )
        if decode_size <= max_decode_size and mixed_size <= max_mixed_size
    }

  @override
  def supported_on(self, device: jax.Device) -> bool:
    try:
      return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 6
    except Exception:
      return False
