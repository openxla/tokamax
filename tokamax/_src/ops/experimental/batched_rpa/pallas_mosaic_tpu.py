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

from typing import Any, ClassVar
import jax
from jax.experimental.pallas import tpu as pltpu
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.batched_rpa import base
from tokamax._src.ops.experimental.batched_rpa import pallas_mosaic_tpu_kernel
from tokamax._src.ops.experimental.batched_rpa.kernel import configs as rpa_configs
from typing_extensions import override
@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Autotuning parameters for Batched RPA Mosaic TPU kernel."""

  # Prefill block tuning
  prefill_bq_sz: int = 128
  prefill_bq_c_sz: int = 64
  # Shared KV tile and buffer configuration
  bkv_sz: int = 512
  n_buffer: int = 3
  # Batch sizes (decode fixed at bq_sz=1, bq_c_sz=1 for fast single-token path)
  decode_batch_size: int = 8
  prefill_batch_size: int = 2
  # KV memory layout
  kv_layout: str = "HEAD_ALONG_SUBLANE"
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
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array], None]:
    del return_residuals
    assert config is not None, "Config must be supplied."

    kv_layout = rpa_configs.KVLayout[config.kv_layout]

    # Decode fast-paths require bq_sz=1 and bq_c_sz=1.
    decode_block_sizes = rpa_configs.BlockSizes(
        bq_sz=1,
        bq_c_sz=1,
        bkv_sz=config.bkv_sz,
        batch_size=config.decode_batch_size,
        n_buffer=config.n_buffer,
    )
    prefill_block_sizes = rpa_configs.BlockSizes(
        bq_sz=config.prefill_bq_sz,
        bq_c_sz=config.prefill_bq_c_sz,
        bkv_sz=config.bkv_sz,
        batch_size=config.prefill_batch_size,
        n_buffer=config.n_buffer,
    )

    result = pallas_mosaic_tpu_kernel.batched_rpa_mosaic_tpu_kernel(
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
        kv_layout=kv_layout,
    )

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
