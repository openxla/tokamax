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
"""Pallas/Mosaic operator implementation for MLA-SWA on TPU."""

import itertools
from typing import Annotated, ClassVar, override

import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int, UInt8  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.mla_swa import base
from tokamax._src.ops.experimental.mla_swa import pallas_mosaic_tpu_kernel


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning parameters for the MLA-SWA Pallas kernel.

  Attributes:
    num_kv_pages_per_block: KV pages per streamed block for (decode, prefill,
      mixed) cases.
    num_queries_per_block: Query tokens per block for (decode, prefill, mixed)
      cases.
    q_compute_block_size: Optional query sub-block size within a flash attention
      step.
    vmem_limit_bytes: VMEM capacity limit in bytes for the Pallas kernel.
  """

  num_kv_pages_per_block: tuple[int, int, int] | int = (2, 2, 2)
  num_queries_per_block: tuple[int, int, int] | int = (1, 32, 32)
  q_compute_block_size: Annotated[int, pydantic.Field(gt=0)] | None = 4
  vmem_limit_bytes: Annotated[int, pydantic.Field(gt=0)] = (
      pallas_mosaic_tpu_kernel.DEFAULT_VMEM_LIMIT_BYTES
  )


class PallasTpuMlaSwa(base.MlaSwa[Config]):
  """Tokamax operator invoking the Pallas kernel for MLA-SWA."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      q: Float[Array, "T H D"],
      new_kv: Float[Array, "T D"],
      cache_kv: UInt8[Array, "P S 4 128"] | Int[Array, "P S 128"],
      kv_lens: Int[Array, "M"],
      page_indices: Int[Array, "N"],
      cu_q_lens: Int[Array, "B"],
      distribution: Int[Array, "3"],
      attention_sinks: Float[Array, "H"],
      *,
      sliding_window: int,
      logical_page_size: int,
      sm_scale: float = 1.0,
      chunk_prefill_size: int | None = None,
      unnormalized_output: bool = False,
      non_causal_block: bool = False,
      return_residuals: bool = False,
      config: Config,
  ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
    del return_residuals
    return (
        pallas_mosaic_tpu_kernel.mla_sliding_window_ragged_paged_attention(
            q,
            new_kv,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            attention_sinks,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            logical_page_size=logical_page_size,
            chunk_prefill_size=chunk_prefill_size,
            num_kv_pages_per_block=config.num_kv_pages_per_block,
            num_queries_per_block=config.num_queries_per_block,
            q_compute_block_size=config.q_compute_block_size,
            vmem_limit_bytes=config.vmem_limit_bytes,
            unnormalized_output=unnormalized_output,
            non_causal_block=non_causal_block,
        ),
        None,
    )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 32, 32),
        q_compute_block_size=4,
        vmem_limit_bytes=pallas_mosaic_tpu_kernel.DEFAULT_VMEM_LIMIT_BYTES,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    del ba
    configs = set()
    kv_pages_per_block = [1, 2, 4]
    queries_per_block = [8, 16, 32]
    for bkv_p, bq in itertools.product(kv_pages_per_block, queries_per_block):
      configs.add(
          Config(
              num_kv_pages_per_block=(bkv_p, bkv_p, bkv_p),
              num_queries_per_block=(1, bq, bq),
              q_compute_block_size=4,
          )
      )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 7
