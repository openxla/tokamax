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
"""Pallas/Mosaic operator implementation for CSA Gather on TPU SparseCore."""

from typing import Annotated, ClassVar, override

import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.csa_gather import base
from tokamax._src.ops.experimental.tpu.csa_gather import pallas_mosaic_tpu_kernel

# `num_streams` values the autotuner tries. More streams keep more gather DMAs
# in flight, which hides the latency of the random-access reads. 4 is both the
# default (the value upstream vllm-torchtpu hardcodes) and the cap
# (`pallas_mosaic_tpu_kernel.MAX_NUM_STREAMS`): with 8 streams the kernel
# compiles but never finishes running on TPU7x, even for a single `top_k`
# period, so the autotuner never tries more than 4. 1 and 2 stay as candidates
# for shapes or chips where fewer in-flight gathers are faster, though 4 has
# won on every arg spec measured on TPU7x so far.
_NUM_STREAMS_CANDIDATES = (1, 2, 4)


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning parameters for the CSA Gather SparseCore kernel.

  Attributes:
    num_streams: Number of independent indirect gathers issued per pipeline
      step. More streams keep more gather DMAs in flight.
  """

  num_streams: Annotated[
      int,
      pydantic.Field(gt=0, le=pallas_mosaic_tpu_kernel.MAX_NUM_STREAMS),
  ] = 4


def _num_lanes() -> int:
  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "SparseCore info is missing."
  return sc_info.num_lanes


def _is_valid(config: Config, top_k: int, num_lanes: int) -> bool:
  return (top_k // 2) % (config.num_streams * num_lanes) == 0


class PallasTpuCsaGather(base.CsaGather[Config]):
  """Tokamax operator invoking the SparseCore Pallas kernel for CSA Gather."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      nope_cache: Int[Array, "num_pages page_size 128"],
      rope_cache: Int[Array, "num_pages rope_rows 128"],
      indices: Int[Array, "N"],
      num_valid_indices: Int[Array, "*#nv"] | None = None,
      *,
      top_k: int = 1024,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array], None]:
    """Gathers the NoPE and RoPE cache rows of the requested tokens.

    See `reference` for the exact cache and output layouts.

    Args:
      nope_cache: `(num_pages, page_size, 128)` int32 NoPE cache, one row per
        token holding its packed `(4, 128)` uint8 slab.
      rope_cache: `(num_pages, page_size // 4, 128)` int32 RoPE cache, 4 tokens
        per row and 32 words (64 bf16 values) per token.
      indices: `(N,)` int32 flat token indices (`page * page_size + slot`) into
        the caches, e.g. the indexer's top-k selections for each query. `N` must
        be a multiple of `top_k`.
      num_valid_indices: Optional scalar or `(1,)` int32 count of valid leading
        indices, a multiple of `top_k`. `top_k` periods starting at or past it
        are skipped and their output rows are left unwritten.
      top_k: The block size of the sparse MLA kernel that consumes the output, a
        multiple of 128. The gather does not select anything by it: `indices` is
        split into periods of `top_k` entries, and in `rope_out` entry `i` of
        each period is packed next to entry `i + top_k // 2` (lanes `0:64` and
        `64:128` of one bf16 row) so the consumer reads a period as one
        lane-dense `(top_k // 2, 128)` block. It also sets how the kernel splits
        the work across SparseCore subcores.
      return_residuals: Unused; the op has no residuals.
      config: The `num_streams` to run the kernel with. Defaults to `Config()`.

    Returns:
      `((nope_out, rope_out), None)`, where `nope_out` is `(N, 128)` int32 and
      `rope_out` is `(N // 4, 128)` int32.
    """
    if config is None:
      config = Config()
    return (
        pallas_mosaic_tpu_kernel.csa_gather(
            nope_cache,
            rope_cache,
            indices,
            num_valid_indices,
            top_k=top_k,
            num_streams=config.num_streams,
        ),
        None,
    )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    # Upstream's choice; valid for every top_k that is a multiple of 128.
    return Config()

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    top_k = ba.arguments.get("top_k", 1024)
    num_lanes = _num_lanes()
    return {
        config
        for n in _NUM_STREAMS_CANDIDATES
        if _is_valid(config := Config(num_streams=n), top_k, num_lanes)
    }

  @override
  def supported_on(self, device) -> bool:
    return (
        device.platform == "tpu"
        and pltpu.get_tpu_info().sparse_core is not None
    )
