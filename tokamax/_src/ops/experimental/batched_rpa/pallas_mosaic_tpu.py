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
"""Tokamax operator wrapper for Pallas Mosaic TPU Batched RPA."""

from typing import Annotated, Any, ClassVar, Literal
import jax
from jax.experimental.pallas import tpu as pltpu
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.batched_rpa import base
from tokamax._src.ops.experimental.batched_rpa.kernel import configs as rpa_configs
from tokamax._src.ops.experimental.batched_rpa.kernel import wrapper as rpa_wrapper
from typing_extensions import override


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Autotuning parameters for Batched RPA Mosaic TPU kernel."""

  # Prefill block tuning
  prefill_bq_sz: Annotated[int, pydantic.Field(gt=0, multiple_of=16)] = 128
  prefill_bq_c_sz: Annotated[int, pydantic.Field(gt=0, multiple_of=16)] = 64
  # Shared KV tile and buffer configuration
  bkv_sz: Annotated[int, pydantic.Field(gt=0, multiple_of=128)] = 512
  n_buffer: Annotated[int, pydantic.Field(ge=1, le=4)] = 3
  # Batch sizes (decode fixed at bq_sz=1, bq_c_sz=1 for fast single-token path)
  decode_batch_size: Annotated[int, pydantic.Field(gt=0)] = 8
  prefill_batch_size: Annotated[int, pydantic.Field(gt=0)] = 2
  # KV memory layout
  kv_layout: Literal["HEAD_ALONG_SUBLANE", "SEQ_ALONG_LANE"] = (
      "HEAD_ALONG_SUBLANE"
  )

  def __post_init__(self):
    if self.prefill_bq_c_sz > self.prefill_bq_sz:
      raise ValueError(
          f"{self.prefill_bq_c_sz=} cannot be greater than"
          f" {self.prefill_bq_sz=}."
      )


class PallasTpuBatchedRpa(base.BatchedRpa[Config]):
  """Tokamax operator wrapper for Batched Ragged Paged Attention on TPU."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      queries: jax.Array,
      keys: jax.Array,
      values: jax.Array,
      kv_cache: jax.Array,
      kv_lens: jax.Array,
      page_indices: jax.Array,
      cu_q_lens: jax.Array,
      distribution: jax.Array,
      *,
      use_causal_mask: bool = True,
      sm_scale: float = 1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      mask_value: float | None = None,
      out_dtype: Any = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      decode_query_size: int = 1,
      skip_kv_update: bool = True,
      kv_layout: str | Any | None = None,
      cp_group_size: int | None = None,
      cp_rank: jax.Array | None = None,
      attention_scope: str | Any = "FULL",
      return_lse: bool = False,
      decode_block_sizes: Any | None = None,
      prefill_block_sizes: Any | None = None,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array], None]:
    del return_residuals
    assert config is not None, "Config must be supplied."

    if isinstance(attention_scope, str):
      effective_attention_scope = rpa_configs.AttentionScope[attention_scope]
    else:
      effective_attention_scope = attention_scope

    if isinstance(kv_layout, str):
      effective_kv_layout = rpa_configs.KVLayout[kv_layout]
    elif isinstance(kv_layout, rpa_configs.KVLayout):
      effective_kv_layout = kv_layout
    else:
      effective_kv_layout = rpa_configs.KVLayout[config.kv_layout]

    if decode_block_sizes is None:
      decode_block_sizes = rpa_configs.BlockSizes(
          bq_sz=decode_query_size,
          bq_c_sz=1,
          bkv_sz=config.bkv_sz,
          batch_size=config.decode_batch_size,
          n_buffer=config.n_buffer,
      )
    if prefill_block_sizes is None:
      prefill_block_sizes = rpa_configs.BlockSizes(
          bq_sz=config.prefill_bq_sz,
          bq_c_sz=config.prefill_bq_c_sz,
          bkv_sz=config.bkv_sz,
          batch_size=config.prefill_batch_size,
          n_buffer=config.n_buffer,
      )

    result = rpa_wrapper.ragged_paged_attention(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        use_causal_mask=use_causal_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        out_dtype=out_dtype,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        mask_value=mask_value,
        decode_block_sizes=decode_block_sizes,
        prefill_block_sizes=prefill_block_sizes,
        skip_kv_update=skip_kv_update,
        kv_layout=effective_kv_layout,
        decode_query_size=decode_query_size,
        cp_group_size=cp_group_size,
        cp_rank=cp_rank,
        attention_scope=effective_attention_scope,
        return_lse=return_lse,
    )

    if return_lse:
      return (result[0], result[1], result[2]), None
    return (result[0], result[1]), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    """Default heuristic configuration based on input parameters."""
    del ba  # Unused for basic heuristics.
    return Config(
        prefill_bq_sz=128,
        prefill_bq_c_sz=64,
        bkv_sz=512,
        decode_batch_size=8,
        prefill_batch_size=2,
        n_buffer=3,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    """Search space for autotuning configurations across TPU generations."""
    del ba
    configs = set()
    for prefill_bq in (64, 128, 256):
      for bkv in (256, 512, 1024, 2048):
        for prefill_bs in (1, 2):
          for decode_bs in (4, 8):
            for nbuf in (2, 3):
              configs.add(
                  Config(
                      prefill_bq_sz=prefill_bq,
                      prefill_bq_c_sz=min(64, prefill_bq),
                      bkv_sz=bkv,
                      decode_batch_size=decode_bs,
                      prefill_batch_size=prefill_bs,
                      n_buffer=nbuf,
                  )
              )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    """Checks device hardware compatibility."""
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
