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
"""Pallas/Mosaic kernel for Multi-Head Latent Attention (MLA) on TPU."""

import dataclasses
import itertools
from typing import Annotated, ClassVar

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.mla import base
from tokamax._src.ops.experimental.mla import pallas_mosaic_tpu_kernel
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  num_kv_pages_per_block: Annotated[int, pydantic.Field(multiple_of=2, gt=0)]
  num_queries_per_block: Annotated[int, pydantic.Field(multiple_of=1, gt=0)]
  vmem_limit_bytes: Annotated[int, pydantic.Field(multiple_of=16, gt=0)]
  chunk_prefill_size: Annotated[int, pydantic.Field(multiple_of=256, ge=0)]
  decode_batch_size: Annotated[int, pydantic.Field(multiple_of=1, gt=0)]


class PallasTpuMultiHeadLatentAttention(base.MultiHeadLatentAttention):
  """Tokamax operator that invokes the Pallas kernel for Multi-Head Latent Attention."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      ql_nope: Float[
          Array, "max_num_tokens actual_num_q_heads actual_lkv_dim"
      ],
      q_pe: Float[
          Array, "max_num_tokens actual_num_q_heads actual_r_dim"
      ],
      new_kv_c: Float[Array, "max_num_tokens actual_lkv_dim"],
      new_k_pe: Float[Array, "max_num_tokens actual_r_dim"],
      cache_kv: Float[
          Array,
          "total_num_pages page_size_per_kv_packing kv_packing lkv_dim",
      ],
      kv_lens: Int[Array, "max_num_seqs"],
      page_indices: Int[Array, "num_page_indices"],
      cu_q_lens: Int[Array, "max_num_seqs_plus_1"],
      distribution: Int[Array, "3"],
      *,
      sm_scale: float = 1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      mask_value: float | None = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      s_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      debug_mode: bool = False,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array], None]:

    assert config is not None, "Config must be specified."

    return (
        pallas_mosaic_tpu_kernel.mla_ragged_paged_attention(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            mask_value=mask_value,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            chunk_prefill_size=config.chunk_prefill_size
            if config.chunk_prefill_size > 0
            else None,
            num_kv_pages_per_block=config.num_kv_pages_per_block,
            num_queries_per_block=config.num_queries_per_block,
            vmem_limit_bytes=config.vmem_limit_bytes,
            decode_batch_size=config.decode_batch_size,
            s_dtype=s_dtype,
        ),
        None,
    )

  @override
  def _get_heuristics_config(self, ba) -> Config:
    return Config(
        num_kv_pages_per_block=16,
        num_queries_per_block=1,
        vmem_limit_bytes=64 * 1024 * 1024,
        chunk_prefill_size=0,
        decode_batch_size=1,
    )

  @override
  def _get_autotuning_configs(self, ba) -> set[Config]:
    configs = set()
    for decode_batch_size, kv, q, vmem_size in itertools.product(
        [8],
        [1, 4, 8, 16, 32, 48, 64],
        [1],
        [48, 56, 64],
    ):
      configs.add(
          Config(
              num_kv_pages_per_block=kv,
              num_queries_per_block=q,
              vmem_limit_bytes=vmem_size * 1024 * 1024,
              decode_batch_size=decode_batch_size,
              chunk_prefill_size=0,
          )
      )
    return configs

  @override
  def supported_on(self, device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
