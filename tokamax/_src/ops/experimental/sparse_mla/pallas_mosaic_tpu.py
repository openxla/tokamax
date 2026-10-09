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
"""Pallas/Mosaic operator implementation for SparseMLA on TPU."""

import itertools
from typing import Annotated, ClassVar, override

import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.sparse_mla import base
from tokamax._src.ops.experimental.sparse_mla import pallas_mosaic_tpu_kernel

_CHUNK_SIZE_CANDIDATES = (32, 64, 128)
_BATCH_SIZE_CANDIDATES = (4, 8, 16)


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning parameters for the SparseMLA Pallas kernel.

  Attributes:
    gather_and_attention_chunk_size: Number of query tokens processed per
      SparseCore gather + TensorCore attention chunk.
    attention_kernel_batch_size: Number of query tokens processed per TensorCore
      Pallas grid step inside a chunk.
    vmem_limit_bytes: VMEM capacity limit in bytes for the Pallas kernel.
  """

  gather_and_attention_chunk_size: Annotated[
      int, pydantic.Field(gt=0, multiple_of=4)
  ] = 128
  attention_kernel_batch_size: Annotated[int, pydantic.Field(gt=0)] = 16
  vmem_limit_bytes: Annotated[int, pydantic.Field(gt=0)] = (
      pallas_mosaic_tpu_kernel.DEFAULT_VMEM_LIMIT_BYTES
  )


class PallasTpuSparseMla(base.SparseMla[Config]):
  """Tokamax operator invoking the Pallas kernel for SparseMLA."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      q: Float[Array, "T H D"],
      cache_kv_nope: Int[Array, "P S 128"],
      cache_kv_rope: Int[Array, "P R 128"],
      topk_indices: Int[Array, "T K"],
      page_indices: Int[Array, "N"],
      cu_q_lens: Int[Array, "B"],
      distribution: Int[Array, "3"],
      attention_sinks: Float[Array, "H"],
      swa_accumution: Float[Array, "T H D"],
      swa_l: Float[Array, "T H"],
      swa_m: Float[Array, "T H"],
      *,
      sm_scale: float = 1.0,
      return_residuals: bool = False,
      config: Config,
  ) -> tuple[jax.Array, None]:
    return (
        pallas_mosaic_tpu_kernel.sparse_ragged_paged_attention(
            q,
            cache_kv_nope,
            cache_kv_rope,
            topk_indices,
            page_indices,
            cu_q_lens,
            distribution,
            attention_sinks,
            swa_accumution,
            swa_l,
            swa_m,
            sm_scale=sm_scale,
            gather_and_attention_chunk_size=(
                config.gather_and_attention_chunk_size
            ),
            attention_kernel_batch_size=config.attention_kernel_batch_size,
            vmem_limit_bytes=config.vmem_limit_bytes,
        ),
        None,
    )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(
        gather_and_attention_chunk_size=128,
        attention_kernel_batch_size=16,
        vmem_limit_bytes=pallas_mosaic_tpu_kernel.DEFAULT_VMEM_LIMIT_BYTES,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    del ba
    configs = set()
    for chunk_size, batch_size in itertools.product(
        _CHUNK_SIZE_CANDIDATES, _BATCH_SIZE_CANDIDATES
    ):
      if chunk_size >= batch_size and chunk_size % batch_size == 0:
        configs.add(
            Config(
                gather_and_attention_chunk_size=chunk_size,
                attention_kernel_batch_size=batch_size,
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
