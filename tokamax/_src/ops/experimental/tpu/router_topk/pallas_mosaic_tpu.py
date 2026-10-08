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
"""Pallas/Mosaic operator implementation of the MoE router top-k on TPU."""

from typing import Annotated, ClassVar, override

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.router_topk import base
from tokamax._src.ops.experimental.tpu.router_topk import pallas_mosaic_tpu_kernel

# A block smaller than the row count must be a whole number of float32 sublane
# tiles.
_SUBLANES = 8
_LANES = 128
# The autotuner tries `block_rows = min(cap, num_tokens)` for each of these
# caps, besides upstream's heuristic (cap 512). Every cap ran, and matched the
# reference bit for bit, on TPU7x and TPU v6e for 8 to 16384 tokens and 256 to
# 2048 experts. Smaller blocks are slower on many tokens: on TPU7x at 512
# experts, 64 rows are 4-5x slower than 512, and 8 rows 30-40x (so 8 is not a
# candidate). Caps of 2048 and 4096 were never faster than 1024 and run out of
# VMEM at 2048 experts.
_BLOCK_ROWS_CAPS = (64, 128, 256, 512, 1024)
# The largest float32 scores block (lane padded) the op runs. On TPU7x every
# 8 MiB block measured ran (1024 x 2048, 512 x 4096, 256 x 8192), while 16 MiB
# blocks mostly run out of VMEM at compile time (2048 x 2048, 1024 x 4096 and
# upstream's 512 x 8192): the k unrolled passes spill whole blocks. TPU v6e
# also ran 1024 x 2048 and failed 2048 x 2048.
_MAX_BLOCK_BYTES = 8 * 2**20


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning parameters for the MoE router top-k kernel.

  Attributes:
    block_rows: Tokens per grid step, capped at the number of tokens. Below the
      number of tokens it must be a multiple of 8; it need not divide the number
      of tokens (the last block is partial). Larger blocks amortize the per-step
      pipeline overhead but buffer more scores in VMEM.
  """

  block_rows: Annotated[int, pydantic.Field(gt=0)]


def _default_block_rows(num_tokens: int) -> int:
  # Upstream's choice: `min(MAX_BLOCK_ROWS, rows)`.
  return min(pallas_mosaic_tpu_kernel.MAX_BLOCK_ROWS, num_tokens)


def _block_bytes(block_rows: int, num_tokens: int, num_experts: int) -> int:
  """The VMEM footprint of one float32 scores block, padded to whole lanes."""
  padded_experts = pl.cdiv(num_experts, _LANES) * _LANES
  return min(block_rows, num_tokens) * padded_experts * 4


class PallasTpuRouterTopK(base.RouterTopK[Config]):
  """Tokamax operator invoking the Pallas kernel for the MoE router top-k.

  Configs whose scores block exceeds 8 MiB (`block_rows * num_experts * 4`
  bytes, lane padded) raise `NotImplementedError`, so `api.router_topk` falls
  back to XLA. With upstream's heuristic that is more than 4096 experts.
  """

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      scores: Float[Array, "T E"],
      k: int,
      *,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[base.RouterTopKOutput, None]:
    """Selects the top `k` experts of each token.

    See `base.RouterTopK` for the semantics.

    Args:
      scores: `(num_tokens, num_experts)` router scores, cast to float32.
      k: The number of experts to select per token.
      return_residuals: Unused; the op has no residuals.
      config: The `block_rows` to run the kernel with. Defaults to upstream's
        heuristic (see `_get_heuristics_config`).

    Returns:
      `((weights, indices), None)`, see `base.RouterTopK._fwd`.

    Raises:
      ValueError: If `config.block_rows` is below the number of tokens and not
        a multiple of 8.
      NotImplementedError: If the scores block exceeds 8 MiB.
    """
    del return_residuals  # Unused.
    num_tokens, num_experts = scores.shape
    if config is None:
      config = Config(block_rows=_default_block_rows(num_tokens))
    block_rows = config.block_rows
    if block_rows < num_tokens and block_rows % _SUBLANES:
      raise ValueError(
          f"block_rows must be a multiple of {_SUBLANES} or at least"
          f" num_tokens={num_tokens}, got {block_rows}."
      )
    if _block_bytes(block_rows, num_tokens, num_experts) > _MAX_BLOCK_BYTES:
      raise NotImplementedError(
          f"A ({min(block_rows, num_tokens)}, {num_experts}) scores block"
          f" exceeds the {_MAX_BLOCK_BYTES // 2**20} MiB VMEM budget."
      )
    out = pallas_mosaic_tpu_kernel.select(
        scores.astype(jnp.float32), k, block_rows=block_rows
    )
    return out, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    num_tokens = ba.arguments["scores"].shape[0]
    return Config(block_rows=_default_block_rows(num_tokens))

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    num_tokens, num_experts = ba.arguments["scores"].shape
    configs = {
        Config(block_rows=min(cap, num_tokens))
        for cap in _BLOCK_ROWS_CAPS
        if _block_bytes(cap, num_tokens, num_experts) <= _MAX_BLOCK_BYTES
    }
    # Always keep upstream's heuristic.
    return configs | {self._get_heuristics_config(ba)}

  @override
  def supported_on(self, device: jax.Device) -> bool:
    # Upstream runs this kernel on TPU7x; it was also verified on TPU v6e. On
    # TPU v5e it is barely faster than XLA and runs out of VMEM sooner.
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 6
