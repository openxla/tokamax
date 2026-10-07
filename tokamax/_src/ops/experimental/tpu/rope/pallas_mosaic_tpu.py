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
"""Pallas/Mosaic operator implementation of the DeepSeek-V4 RoPE on TPU."""

from typing import Annotated, ClassVar, override

import jax
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.rope import base
from tokamax._src.ops.experimental.tpu.rope import pallas_mosaic_tpu_kernel

_LANE = pallas_mosaic_tpu_kernel.LANE
# The autotuner tries `tile_n = largest_divisor(num_tokens, cap)` for each of
# these caps, besides upstream's heuristic (cap 128, or 64 for `qnorm_rope`).
_TILE_N_CAPS = (8, 16, 32, 64, 128, 256)
# Candidates whose estimated VMEM use (`_vmem_bytes`) exceeds this are skipped.
# TPU7x has 64 MiB of VMEM; leave some of it for Mosaic's internal scratch.
_VMEM_BUDGET_BYTES = 56 * 2**20


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning parameters for the DeepSeek-V4 RoPE kernels.

  Attributes:
    tile_n: Tokens per grid step, a divisor of the number of tokens. Larger
      tiles amortize the per-step pipeline overhead (and issue more `cos_sin`
      row gathers per step) but buffer more of `x` in VMEM.
  """

  tile_n: Annotated[int, pydantic.Field(gt=0)]


def _vmem_bytes(config: Config, ba: op.BoundArguments) -> int:
  """Estimates the kernel's VMEM use: its buffers and an f32 temporary."""
  x = ba.arguments["x"]
  mode = ba.arguments["mode"]
  rows = config.tile_n * (x.shape[1] if x.ndim == 3 else 1)
  # `rope` and `rope_quant` only pipeline the last lane block of each row,
  # `qnorm_rope` the whole row.
  width = x.shape[-1] if mode == "qnorm_rope" else _LANE
  block = rows * width
  # Double-buffered input and output blocks (`rope` updates one block in
  # place; `rope_quant` writes 1-byte `q`).
  match mode:
    case "rope":
      buffers = 2 * block * x.dtype.itemsize
    case "qnorm_rope":
      buffers = 4 * block * x.dtype.itemsize
    case _:
      buffers = 2 * block * (x.dtype.itemsize + 1)
  # Plus about one f32 block of temporaries. On TPU7x, `qnorm_rope` with
  # `tile_n=128` and a `(256, 128, 512)` bf16 `x` needs 79.75 MiB; this
  # estimates 96 MiB.
  return buffers + block * 4


def _rank_2_quant(ba: op.BoundArguments) -> bool:
  return ba.arguments["mode"] == "rope_quant" and ba.arguments["x"].ndim == 2


def _default_tile_n(num_tokens: int, mode: base.Mode) -> int:
  # Upstream's choice: the largest divisor of the number of tokens up to 128,
  # or up to 64 for `qnorm_rope`, which buffers whole heads.
  cap = 64 if mode == "qnorm_rope" else 128
  return pallas_mosaic_tpu_kernel.largest_divisor(num_tokens, cap=cap)


class PallasTpuRope(base.Rope[Config]):
  """Tokamax operator invoking the Pallas kernels for the DeepSeek-V4 RoPE.

  As upstream, the `"rope"` and `"qnorm_rope"` kernels donate `x`: they write
  the result into its buffer, so `x` must not be used after the call (under an
  outer `jax.jit` that does not donate `x`, XLA copies it instead).

  Shapes the upstream kernels fail to lower raise `NotImplementedError`:

  *   `"qnorm_rope"` with `head_dim == 128`: the kernel concatenates a
      zero-width NoPE part, which Mosaic rejects.
  *   `"rope_quant"` with a rank 2 `x` and `tile_n` not a multiple of 128: the
      `(tile_n,)` block of the `(num_tokens,)` scales must be lane aligned.
  """

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      x: Float[Array, "N *dims"],
      positions: Int[Array, "N"],
      cos_sin_cache: Float[Array, "max_pos rotary_dim"],
      *,
      mode: base.Mode = "rope",
      inverse: bool = False,
      eps: float = 1e-6,
      quant_dtype: jax.typing.DTypeLike | None = None,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[base.RopeOutput, None]:
    """Applies the DeepSeek-V4 RoPE variant selected by `mode`.

    See `base.Rope` for the modes and `base.Rope._fwd` for the arguments. The
    `"rope"` and `"qnorm_rope"` modes donate `x`.

    The kernels read `positions` and the `cos_sin_cache` rows they point to
    with bounds checks disabled, so every position must be in `[0, max_pos)`.

    Args:
      x: `(num_tokens, head_dim)` or `(num_tokens, num_heads, head_dim)`.
      positions: `(num_tokens,)` int32 RoPE position of each token.
      cos_sin_cache: `(max_pos, rotary_dim)` float32 packed `[cos | sin]` rows.
      mode: The variant, one of `"rope"`, `"qnorm_rope"` and `"rope_quant"`.
      inverse: Whether to negate `sin`, i.e. apply the transposed rotation.
      eps: The RMSNorm epsilon (`"qnorm_rope"` only).
      quant_dtype: The quantized dtype (`"rope_quant"` only), else `None`.
      return_residuals: Unused; the op has no residuals.
      config: The `tile_n` to run the kernel with. Defaults to upstream's
        heuristic (see `_get_heuristics_config`).

    Returns:
      `(out, None)`, see `base.Rope._fwd`.

    Raises:
      NotImplementedError: For the shapes listed in the class docstring.
    """
    tile_n = None if config is None else config.tile_n
    match mode:
      case "rope":
        out = pallas_mosaic_tpu_kernel.rope(
            x, positions, cos_sin_cache, inverse=inverse, tile_n=tile_n
        )
      case "qnorm_rope":
        if x.shape[-1] == _LANE:
          raise NotImplementedError(
              f"qnorm_rope requires head_dim > {_LANE}, got {x.shape[-1]}."
          )
        out = pallas_mosaic_tpu_kernel.qnorm_rope(
            x, positions, cos_sin_cache, eps=eps, inverse=inverse, tile_n=tile_n
        )
      case "rope_quant":
        assert quant_dtype is not None  # Canonicalized by `bind`.
        if tile_n is None:
          tile_n = _default_tile_n(x.shape[0], mode)
        if x.ndim == 2 and tile_n % _LANE:
          raise NotImplementedError(
              "rope_quant with a rank 2 x requires tile_n to be a multiple of"
              f" {_LANE}, got {tile_n}."
          )
        out = pallas_mosaic_tpu_kernel.rope_quant(
            x,
            positions,
            cos_sin_cache,
            inverse=inverse,
            quant_dtype=quant_dtype,
            tile_n=tile_n,
        )
      case _:
        raise ValueError(f"Unknown mode: {mode!r}.")
    return out, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    num_tokens = ba.arguments["x"].shape[0]
    return Config(tile_n=_default_tile_n(num_tokens, ba.arguments["mode"]))

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    num_tokens = ba.arguments["x"].shape[0]
    configs = {
        Config(
            tile_n=pallas_mosaic_tpu_kernel.largest_divisor(num_tokens, cap=cap)
        )
        for cap in _TILE_N_CAPS
    }
    configs = {c for c in configs if _vmem_bytes(c, ba) <= _VMEM_BUDGET_BYTES}
    if _rank_2_quant(ba):
      # Smaller tiles fail to lower (see the class docstring).
      configs = {c for c in configs if c.tile_n % _LANE == 0}
    # Always keep upstream's heuristic.
    return configs | {self._get_heuristics_config(ba)}

  @override
  def supported_on(self, device: jax.Device) -> bool:
    # Upstream only runs (and tests) these kernels on TPU7x.
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 7
