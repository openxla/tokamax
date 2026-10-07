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
"""Pallas/Mosaic operator implementation of the compressor projection on TPU."""

from typing import Annotated, ClassVar, override

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int, Shaped  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.proj_and_save_state import base
from tokamax._src.ops.experimental.tpu.proj_and_save_state import pallas_mosaic_tpu_kernel

# The autotuner tries a few tile sizes around upstream's heuristic. There are
# few because each candidate is a full compile (up to ~15 s with
# `tile_n = 512`, as the kernel unrolls one DMA per token of a tile).
#
# `tile_k` candidates are the largest 128-aligned divisors of `hidden_size` up
# to each cap below; 3584 is upstream's cap. A larger `tile_k` means fewer
# pipeline steps but a larger weight block, double-buffered in VMEM.
_TILE_K_CAPS = (3584, 2048, 1024)
# `tile_n` candidates, besides upstream's heuristic. The kernel streams
# all of `wkv_wgate` from HBM once per token tile, so prefill-sized inputs run
# faster with more tokens per tile: on TPU7x with 4096 tokens, `tile_n = 512`
# is 1.4-2x faster than upstream's 128. A `tile_n` that does not cover all the
# tokens must be a multiple of 128: Mosaic requires a partial `(tile_n,)`
# positions block to be 128-aligned.
_TILE_N_CANDIDATES = (256, 512)
# Extra autotuning candidates whose estimated VMEM use (`_vmem_bytes`) exceeds
# this are skipped, so they compile under XLA's default 32 MiB scoped VMEM limit:
# on TPU7x an estimate of 30.25 MiB compiles and one of 32.5 MiB runs out of
# VMEM. Upstream's heuristic config is always kept; for CSA with
# `hidden_size = 7168` and f32 weights it needs a larger limit, which upstream
# vllm-torchtpu sets (`--xla_tpu_scoped_vmem_limit_kib=65536`).
_VMEM_BUDGET_BYTES = 31 * 2**20


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Tile sizes for the compressor projection TensorCore kernel.

  The output tile is always one state field (`state_width` columns), as the
  kernel addresses its two output tiles as `kv` and `score`.

  Attributes:
    tile_k: Hidden-dimension tile, a multiple of 128 that divides `hidden_size`.
    tile_n: Token tile, a multiple of 8. The tokens are padded to a multiple of
      it, and the weights are streamed from HBM once per token tile.
  """

  tile_k: Annotated[
      int,
      pydantic.Field(gt=0, multiple_of=128),
  ]
  tile_n: Annotated[
      int,
      pydantic.Field(gt=0, multiple_of=8),
  ]


def _upstream_tile_n(num_tokens: int) -> int:
  """Upstream's `tile_n`: `num_tokens` rounded up to 8, at most 128."""
  return min(128, pl.cdiv(num_tokens, 8) * 8)


def _vmem_bytes(config: Config, ba: op.BoundArguments) -> int:
  """Estimates the kernel's VMEM use: its blocks, accumulator and APE."""
  hidden_states = ba.arguments["hidden_states"]
  wkv_wgate = ba.arguments["wkv_wgate"]
  ape = ba.arguments["ape"]
  state_width = wkv_wgate.shape[1] // 2
  # Double-buffered weight and hidden-state blocks, and the f32 accumulator's
  # two buffers.
  weights = 2 * config.tile_k * state_width * wkv_wgate.dtype.itemsize
  hidden = 2 * config.tile_n * config.tile_k * hidden_states.dtype.itemsize
  acc = 2 * config.tile_n * state_width * 4
  return weights + hidden + acc + ape.size * ape.dtype.itemsize


class PallasTpuProjAndSaveState(base.ProjAndSaveState[Config]):
  """Tokamax operator invoking the Pallas kernel for the compressor projection."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      hidden_states: Float[Array, "T H"],
      wkv_wgate: Float[Array, "H D"],
      ape: Float[Array, "R W"],
      positions: Int[Array, "T"],
      slot_mapping: Int[Array, "T"],
      cache: Shaped[Array, "num_pages page_size *slab"],
      *,
      compress_ratio: int,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[jax.Array, None]:
    """Projects the tokens and writes their state into the cache.

    See `reference` for the state and cache layouts.

    Args:
      hidden_states: `(num_tokens, hidden_size)` hidden states.
      wkv_wgate: `(hidden_size, 2 * state_width)` fused `kv` and `score`
        projection weights.
      ape: `(compress_ratio, state_width)` absolute position embeddings, added
        to `score`.
      positions: `(num_tokens,)` int32 token positions; token `t` adds APE row
        `positions[t] % compress_ratio`.
      slot_mapping: `(num_tokens,)` int32 first cache row (`page * page_size +
        row`) of each token's state, a multiple of the rows per token. Negative
        entries skip the token.
      cache: `(num_pages, page_size, 4, lanes)` uint8 or `(num_pages, page_size,
        lanes)` int32 state cache.
      compress_ratio: Number of APE rows.
      return_residuals: Unused; the op has no residuals.
      config: The tile sizes to run the kernel with. Defaults to the heuristic
        config.

    Returns:
      `(new_cache, None)`, where `new_cache` is `cache` with the state of every
      token with a non-negative slot written.
    """
    tile_k = tile_n = None
    if config is not None:
      tile_k, tile_n = config.tile_k, config.tile_n
    return (
        pallas_mosaic_tpu_kernel.proj_and_save_state(
            hidden_states,
            wkv_wgate,
            ape,
            positions,
            slot_mapping,
            cache,
            compress_ratio,
            tile_k=tile_k,
            tile_n=tile_n,
        ),
        None,
    )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    # Upstream's choice: the largest 128-aligned divisor of `hidden_size` up to
    # 3584, and the tokens rounded up to 8 rows, at most 128.
    num_tokens, hidden_size = ba.arguments["hidden_states"].shape
    return Config(
        tile_k=pallas_mosaic_tpu_kernel._select_tile_k(hidden_size),  # pylint: disable=protected-access
        tile_n=_upstream_tile_n(num_tokens),
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    num_tokens, hidden_size = ba.arguments["hidden_states"].shape
    tile_ks = {
        pallas_mosaic_tpu_kernel._select_tile_k(hidden_size, cap=cap)  # pylint: disable=protected-access
        for cap in _TILE_K_CAPS
    }
    # A token tile past the 128-padded token count only adds padding.
    max_tile_n = -(-num_tokens // 128) * 128
    tile_ns = {_upstream_tile_n(num_tokens)}
    tile_ns.update(n for n in _TILE_N_CANDIDATES if n <= max_tile_n)
    return {
        config
        for k in tile_ks
        for n in tile_ns
        if _vmem_bytes(config := Config(tile_k=k, tile_n=n), ba)
        <= _VMEM_BUDGET_BYTES
    }

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 6
