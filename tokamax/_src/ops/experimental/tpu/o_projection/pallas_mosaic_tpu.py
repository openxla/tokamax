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
"""Pallas/Mosaic implementation of the DeepSeek-V4 reverse-RoPE `wo_a` projection."""

from typing import Annotated, ClassVar, override

import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Shaped  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.o_projection import base
from tokamax._src.ops.experimental.tpu.o_projection import pallas_mosaic_tpu_kernel
from tokamax._src.ops.experimental.tpu.rope import pallas_mosaic_tpu_kernel as rope_kernel

_LANE = pallas_mosaic_tpu_kernel.LANE
_SUBLANE = pallas_mosaic_tpu_kernel.SUBLANE
# Upstream's tile caps: `gather_cos_sin` uses the largest divisor of the number
# of tokens up to 128 as its `tile_n`; `wo_a_projection` defaults `tile_t` to
# the largest divisor of the number of tokens up to 1024 and `sub_t` to the
# largest divisor of `tile_t` up to 128.
_GATHER_TILE_N_CAP = 128
_TILE_T_CAP = 1024
_SUB_T_CAP = 128
# The autotuner tries `tile_t = largest_divisor(num_tokens, cap)` for each of
# these caps (1024 is upstream's), keeping only tiles that are a multiple of 8
# or all the tokens.
_TILE_T_CAPS = (256, 512, 1024)
# For each `tile_t`, `sub_t = largest_divisor(tile_t, cap)` for each of these
# caps (128 is upstream's).
_SUB_T_CAPS = (64, 128, 256)
# Candidates whose estimated VMEM use (`_vmem_bytes`) exceeds this are skipped.
_VMEM_BUDGET_BYTES = 48 * 2**20


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Tile sizes for the `wo_a` projection kernel.

  Attributes:
    tile_t: Tokens per grid step, a divisor of the number of tokens.
    tile_r: Output columns per grid step, a divisor of the LoRA rank `R` (the
      output columns of one group).
    sub_t: Tokens per sub-chunk within a grid step, a divisor of `tile_t`.
      Smaller sub-chunks give the compiler more instruction-level parallelism to
      exploit.
  """

  tile_t: Annotated[int, pydantic.Field(gt=0)]
  tile_r: Annotated[int, pydantic.Field(gt=0)]
  sub_t: Annotated[int, pydantic.Field(gt=0)]


def _lora_rank(ba: op.BoundArguments) -> int:
  num_groups = ba.arguments["x"].shape[1] // base.HEADS_PER_GROUP
  return ba.arguments["wo_a"].shape[1] // num_groups


def _vmem_bytes(config: Config, ba: op.BoundArguments) -> int:
  """Estimates the kernel's VMEM use: its blocks and per-sub-chunk temporaries."""
  head_dim = ba.arguments["x"].shape[2]
  reduction = base.HEADS_PER_GROUP * head_dim
  # Double-buffered blocks: bf16 x, fp8 wo_a, f32 cos_sin and bf16 output.
  blocks = 2 * (
      config.tile_t * reduction * 2
      + reduction * config.tile_r
      + config.tile_t * 2 * _LANE * 4
      + config.tile_t * config.tile_r * 2
  )
  # The roped (f32 and bf16) and quantized sub-chunk, and the f32 product.
  temporaries = config.sub_t * (reduction * 7 + config.tile_r * 4)
  return blocks + temporaries


def _aligned(tile: int, total: int) -> bool:
  """Whether Mosaic accepts `tile`-row blocks of a `total`-row array."""
  return tile % _SUBLANE == 0 or tile == total


class PallasTpuOProjection(base.OProjection[Config]):
  """Tokamax operator invoking the Pallas kernels for the `wo_a` projection.

  Shapes the upstream kernels fail to lower raise `NotImplementedError`:

  *   `head_dim == 128`: the kernel concatenates a zero-width NoPE part, which
      Mosaic rejects.
  *   Token tiles that are neither a multiple of 8 nor all the tokens, for
      `gather_cos_sin`'s `tile_n` (the largest divisor of the number of tokens
      up to 128, e.g. 100 for 200 tokens) or for `tile_t` (e.g. upstream's
      default 515 for 1030 tokens).
  """

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      x: Float[Array, "T H head_dim"],
      positions: Shaped[Array, "T"],
      cos_sin_cache: Float[Array, "max_pos rotary_dim"],
      wo_a: Shaped[Array, "D GR"],
      wo_a_scale: Float[Array, "GR"],
      *,
      inverse: bool = True,
      quantize_activations: bool = True,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[jax.Array, None]:
    """Applies the RoPE to `x`, then projects it with `wo_a`.

    Runs `gather_cos_sin` to build each token's expanded `[cos | sin]` row,
    then `wo_a_projection`. See `base.OProjection._fwd` for the arguments.

    The `cos_sin_cache` row gather is not bounds checked, so every position
    must be in `[0, max_pos)`.

    Args:
      x: `(T, G * 8, head_dim)` bf16 attention output.
      positions: `(T,)` int32 RoPE position of each token.
      cos_sin_cache: `(max_pos, rotary_dim)` float32 packed `[cos | sin]` rows.
      wo_a: `(8 * head_dim, G * R)` float8_e4m3fn projection weights.
      wo_a_scale: `(G * R,)` float32 per-column weight scales.
      inverse: Whether to negate `sin`, i.e. apply the transposed rotation.
      quantize_activations: Whether to quantize the activations to fp8 per token
        and group before the projection.
      return_residuals: Unused; the op has no residuals.
      config: The tile sizes to run the kernel with. Defaults to upstream's
        heuristic (see `_get_heuristics_config`).

    Returns:
      `(out, None)`, where `out` is the `(T, G * R)` bf16 projection.

    Raises:
      NotImplementedError: For the shapes listed in the class docstring.
    """
    tile_t = tile_r = sub_t = None
    if config is not None:
      tile_t, tile_r, sub_t = config.tile_t, config.tile_r, config.sub_t
    num_tokens, _, head_dim = x.shape
    if head_dim == _LANE:
      raise NotImplementedError(f"head_dim must be > {_LANE}, got {head_dim}.")
    gather_tile_n = rope_kernel.largest_divisor(
        num_tokens, cap=_GATHER_TILE_N_CAP
    )
    if not _aligned(gather_tile_n, num_tokens):
      raise NotImplementedError(
          f"gather_cos_sin's tile_n ({gather_tile_n}) for {num_tokens} tokens"
          f" is neither a multiple of {_SUBLANE} nor all the tokens."
      )
    if tile_t is None:
      tile_t = rope_kernel.largest_divisor(num_tokens, cap=_TILE_T_CAP)
    if not _aligned(tile_t, num_tokens):
      raise NotImplementedError(
          f"tile_t ({tile_t}) for {num_tokens} tokens is neither a multiple of"
          f" {_SUBLANE} nor all the tokens."
      )
    return (
        pallas_mosaic_tpu_kernel.fused_reverse_rope_wo_a_projection(
            x,
            positions,
            cos_sin_cache,
            wo_a,
            wo_a_scale,
            head_dim=x.shape[2],
            inverse=inverse,
            tile_t=tile_t,
            tile_r=tile_r,
            sub_t=sub_t,
            quantize_activations=quantize_activations,
        ),
        None,
    )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    # Upstream's choice: all of the LoRA rank, the largest divisor of the
    # number of tokens up to 1024, and the largest divisor of that up to 128.
    num_tokens = ba.arguments["x"].shape[0]
    tile_t = rope_kernel.largest_divisor(num_tokens, cap=_TILE_T_CAP)
    return Config(
        tile_t=tile_t,
        tile_r=_lora_rank(ba),
        sub_t=rope_kernel.largest_divisor(tile_t, cap=_SUB_T_CAP),
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    num_tokens = ba.arguments["x"].shape[0]
    lora_rank = _lora_rank(ba)
    tile_ts = {
        t
        for cap in _TILE_T_CAPS
        if _aligned(
            t := rope_kernel.largest_divisor(num_tokens, cap=cap), num_tokens
        )
    }
    tile_rs = {lora_rank}
    if lora_rank % 256 == 0:
      tile_rs.add(lora_rank // 2)
    configs = {
        Config(
            tile_t=t,
            tile_r=r,
            sub_t=rope_kernel.largest_divisor(t, cap=cap),
        )
        for t in tile_ts
        for r in tile_rs
        for cap in _SUB_T_CAPS
    }
    configs = {c for c in configs if _vmem_bytes(c, ba) <= _VMEM_BUDGET_BYTES}
    # Always keep upstream's heuristic.
    return configs | {self._get_heuristics_config(ba)}

  @override
  def supported_on(self, device: jax.Device) -> bool:
    # Upstream only runs (and tests) these kernels on TPU7x.
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 7
