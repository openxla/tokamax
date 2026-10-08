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
"""Pallas/Mosaic operator implementation for compress-and-store on TPU."""

from typing import Annotated, ClassVar, override

import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.compress_store import base
from tokamax._src.ops.experimental.tpu.compress_store import pallas_mosaic_tpu_kernel

# `tile_n` values the autotuner tries. `tile_n` must be a multiple of 4 in CSA
# and indexer modes (4 tokens share a cache row). 4 is the value upstream
# vllm-torchtpu hardcodes and the default. All of these compile and match the
# reference on TPU7x in every mode.
#
# As upstream, the kernel visits whole tiles up to the last token with a slot
# and reads the token arrays for every token of those tiles without bounds
# checks. A tile that runs past the last token faults the TPU, so only `tile_n`
# values that divide the token count are tried (upstream's `tile_n = 4` is
# always kept).
_TILE_N_CANDIDATES = (4, 8, 12)
# Only CSA and the indexer (`overlap=True`) also try these: HCA buffers larger
# state pages, and at `tile_n >= 16` fails to compile on TPU7x with a scoped
# VMEM OOM (E1001: CompileTimeScopedVmemOom).
_OVERLAP_TILE_N_CANDIDATES = (16, 32)


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning parameters for the compress-and-store kernel.

  Attributes:
    tile_n: Tokens processed per grid step. Larger tiles amortize the per-step
      pipeline overhead but buffer more state pages in VMEM.
  """

  tile_n: Annotated[int, pydantic.Field(gt=0, multiple_of=4)] = (
      pallas_mosaic_tpu_kernel.DEFAULT_TILE_N
  )


class PallasTpuCompressStore(base.CompressStore[Config]):
  """Tokamax operator invoking the Pallas kernel for compress-and-store."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      cache: Shaped[Array, "num_pages page_size *lanes"],
      positions: Int[Array, "N"],
      block_table: Int[Array, "B"],
      token_to_req_indices: Int[Array, "N"],
      kv_slot_mapping: Int[Array, "N"],
      rms_weight: Float[Array, "head_dim"],
      *,
      cos_sin_cache: Float[Array, "max_pos rope_head_dim"],
      block_table_stride: int,
      state_block_size: int,
      compress_ratio: int,
      overlap: bool,
      state_cache: (
          Shaped[Array, "state_pages state_page_size 4 128"] | None
      ) = None,
      rope_cache: Int[Array, "num_pages rope_rows 128"] | None = None,
      quant_block: int = base.CSA_QUANT_BLOCK,
      rms_eps: float = 1e-6,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array | None], None]:
    """Compresses each boundary token's window and stores it into the cache.

    See `base.CompressStore._fwd` for the arguments and `reference` for the
    modes and cache layouts.

    Args:
      cache: The compressed KV cache.
      positions: `(N,)` int32 token positions.
      block_table: `(num_reqs * block_table_stride,)` int32 state pages.
      token_to_req_indices: `(N,)` int32 request of each token.
      kv_slot_mapping: `(N,)` int32 compressed-KV slot of each token, or -1.
      rms_weight: `(head_dim,)` f32 RMSNorm weight.
      cos_sin_cache: `(max_pos, 64)` f32 RoPE `[cos | sin]` table.
      block_table_stride: Row stride of `block_table`.
      state_block_size: Token states per state page.
      compress_ratio: Tokens compressed into one record.
      overlap: Whether windows overlap.
      state_cache: The separate state array (HCA), or `None`.
      rope_cache: The CSA RoPE cache, or `None`.
      quant_block: FP8 quantization block.
      rms_eps: RMSNorm epsilon.
      return_residuals: Unused; the op has no residuals.
      config: The `tile_n` to run the kernel with. Defaults to `Config()`.

    Returns:
      `((cache, rope_cache), None)` with the updated caches; `rope_cache` is
      `None` outside CSA.
    """
    if config is None:
      config = Config()
    # As upstream, the kernel donates `cache` and `rope_cache` and updates them
    # in place.
    return (
        pallas_mosaic_tpu_kernel.compress_norm_rope_store(
            cache,
            positions,
            block_table,
            token_to_req_indices,
            kv_slot_mapping,
            rms_weight,
            block_table_stride=block_table_stride,
            state_block_size=state_block_size,
            compress_ratio=compress_ratio,
            overlap=overlap,
            state_cache=state_cache,
            rope_cache=rope_cache,
            cos_sin_cache=cos_sin_cache,
            quant_block=quant_block,
            rms_eps=rms_eps,
            tile_n=config.tile_n,
        ),
        None,
    )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    # Upstream's choice; valid in every mode.
    return Config()

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    tile_ns: tuple[int, ...] = _TILE_N_CANDIDATES
    if ba.arguments["overlap"]:
      tile_ns += _OVERLAP_TILE_N_CANDIDATES
    num_tokens = ba.arguments["positions"].shape[0]
    return {Config()} | {
        Config(tile_n=n) for n in tile_ns if num_tokens % n == 0
    }

  @override
  def supported_on(self, device) -> bool:
    # Upstream only passes its whole suite on TPU7x: a large CSA decode batch
    # fails on TPU v6e.
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 7
