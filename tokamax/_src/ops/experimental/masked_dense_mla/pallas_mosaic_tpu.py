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
"""Pallas/Mosaic operator implementation for MaskedDenseMLA on TPU."""

import itertools
from typing import Annotated, ClassVar, override

import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int, UInt8  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.masked_dense_mla import base
from tokamax._src.ops.experimental.masked_dense_mla import pallas_mosaic_tpu_kernel

_KV_PAGES_PER_BLOCK_CANDIDATES = (1, 2, 4)
_QUERIES_PER_BLOCK_CANDIDATES = (8, 16, 32)


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning parameters for the MaskedDenseMLA Pallas kernel.

  Attributes:
    num_kv_pages_per_block: KV pages per streamed block for (decode, prefill,
      mixed) cases.
    num_queries_per_block: Query tokens per block for (decode, prefill, mixed)
      cases.
    vmem_limit_bytes: VMEM capacity limit in bytes for the Pallas kernel.
  """

  num_kv_pages_per_block: tuple[int, int, int] | int = (1, 1, 1)
  num_queries_per_block: tuple[int, int, int] | int = (1, 16, 16)
  vmem_limit_bytes: Annotated[int, pydantic.Field(gt=0)] = (
      pallas_mosaic_tpu_kernel.DEFAULT_VMEM_LIMIT_BYTES
  )


class PallasTpuMaskedDenseMla(base.MaskedDenseMla[Config]):
  """Tokamax operator invoking the Pallas kernel for MaskedDenseMLA."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      q: Float[Array, "T H D"],
      cache_kv_nope: UInt8[Array, "P S 4 128"],
      cache_kv_rope: UInt8[Array, "P R 4 128"],
      kv_lens: Int[Array, "M"],
      topk_indices: Int[Array, "T K"],
      page_indices: Int[Array, "N"],
      cu_q_lens: Int[Array, "B"],
      distribution: Int[Array, "3"],
      *,
      sm_scale: float = 1.0,
      k_scale: float = 1.0,
      mask_value: float | None = None,
      max_kv_len: int | None = None,
      chunk_prefill_size: int | None = None,
      sequence_start: Int[Array, ""] | None = None,
      return_residuals: bool = False,
      config: Config,
  ) -> tuple[jax.Array, None]:
    return (
        pallas_mosaic_tpu_kernel.masked_dense_ragged_paged_attention(
            q,
            cache_kv_nope,
            cache_kv_rope,
            kv_lens,
            topk_indices,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            k_scale=k_scale,
            mask_value=mask_value,
            max_kv_len=max_kv_len,
            chunk_prefill_size=chunk_prefill_size,
            num_kv_pages_per_block=config.num_kv_pages_per_block,
            num_queries_per_block=config.num_queries_per_block,
            vmem_limit_bytes=config.vmem_limit_bytes,
            sequence_start=sequence_start,
        ),
        None,
    )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(
        num_kv_pages_per_block=(1, 1, 1),
        num_queries_per_block=(1, 16, 16),
        vmem_limit_bytes=pallas_mosaic_tpu_kernel.DEFAULT_VMEM_LIMIT_BYTES,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    page_size = ba.arguments["cache_kv_nope"].shape[1]
    max_num_seqs = ba.arguments["kv_lens"].shape[0]
    pages_per_seq = ba.arguments["page_indices"].shape[0] // max_num_seqs
    max_kv_len = ba.arguments.get("max_kv_len") or (pages_per_seq * page_size)

    configs = set()
    for bkv_p, bq in itertools.product(
        _KV_PAGES_PER_BLOCK_CANDIDATES, _QUERIES_PER_BLOCK_CANDIDATES
    ):
      if max_kv_len % (bkv_p * page_size) == 0:
        configs.add(
            Config(
                num_kv_pages_per_block=(bkv_p, bkv_p, bkv_p),
                num_queries_per_block=(1, bq, bq),
            )
        )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return (
        device.platform == "tpu"
        and pltpu.get_tpu_info().generation >= 7
        and pltpu.get_tpu_info().sparse_core is not None
    )
