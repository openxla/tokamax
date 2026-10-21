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
"""Pallas/Mosaic TPU VJP of the fused FP8 matmul.

The activation gradient runs on the Pallas dLHS kernel
(`pallas_mosaic_tpu_dlhs_kernel`); the weight gradient is
`base.fused_fp8_drhs_reference`.
"""

import dataclasses
import itertools
from typing import Annotated, Any, ClassVar, override

import jax
import jax.numpy as jnp
import pydantic
from tokamax._src.ops import op
from tokamax._src.ops.experimental.fused_fp8_matmul import base
from tokamax._src.ops.experimental.fused_fp8_matmul import pallas_mosaic_tpu_dlhs_kernel as dlhs_kernel
from tokamax._src.ops.experimental.fused_fp8_matmul import pallas_mosaic_tpu_kernel as fwd_kernel

QArray = base.QArray

_BlockSize = Annotated[int, pydantic.Field(ge=128, multiple_of=128)]

# Budget used when choosing dLHS tiles. Higher than the forward kernel's
# `VMEM_BUDGET_BYTES` because `dlhs_vmem_bytes` is a tighter model: every
# buffer it counts is one the pipeline actually places, and the scale-folded
# cotangent tile is deliberately left out because Mosaic rematerializes it
# rather than holding a full `(bm, bn)` copy. Calibrated against compile
# outcomes on TPU7x: at `bk == k` and `bn == n`, `bm = 1024` models at 58.8 MB
# for (32768, 3072, 4096) and compiles; `bm = 2048` models at 117.5 MB and is
# far outside the 64 MiB of physical VMEM.
VMEM_BUDGET_BYTES = 62 * 1024 * 1024


def dlhs_vmem_bytes(bm: int, bk: int, bn: int, k: int, n: int) -> int:
  """Estimates the dLHS kernel's scoped VMEM footprint in bytes.

  Mirrors `fwd_kernel.forward_vmem_bytes`: the weight block is single buffered
  when `bk == k` and `bn == n` make its index map constant, and the accumulator
  and the cotangent cache are counted only on the paths that read them.

  Args:
    bm: Block size along `m`.
    bk: Block size along `k`.
    bn: Block size along `n`, the contracting dimension.
    k: The output column dimension of `dlhs`.
    n: The contracting dimension.

  Returns:
    The estimated footprint in bytes.
  """
  grid_k = -(-k // bk)
  grid_n = -(-n // bn)
  w_buffers = 1 if (grid_k == 1 and grid_n == 1) else 2
  g_block = bm * bn * 2 * 2  # bf16 cotangent in, double buffered
  o_block = bm * bk * 2 * 2  # bf16 gradient out, double buffered
  gq = bm * bn * 1  # fp8 quantized cotangent tile
  dot_out = bm * bk * 4  # float32 dot_general result
  w_block = bk * bn * 1 * w_buffers  # fp8 weights
  accum = bm * bk * 4 if grid_n > 1 else 0
  gq_cache = grid_n * bm * bn * 1 if grid_k > 1 else 0
  return g_block + o_block + gq + dot_out + w_block + accum + gq_cache


def _tile_candidates(dim: int, candidates: tuple[int, ...]) -> tuple[int, ...]:
  """Returns the candidates that divide `dim`, largest-preferred order kept.

  Falls back to `fwd_kernel.pick_tile`'s power-of-two divisor when none of them
  divides `dim`, so odd shapes still get a valid tile.

  Args:
    dim: The array dimension to tile.
    candidates: Candidate tile sizes, listed largest-preferred.

  Returns:
    A non-empty tuple of tile sizes, each dividing `dim`.
  """
  dividing = tuple(c for c in candidates if c <= dim and dim % c == 0)
  return dividing or (fwd_kernel.pick_tile(dim, candidates),)


def fit_dlhs_tiles(m: int, k: int, n: int) -> tuple[int, int, int]:
  """Picks `(bm, bk, bn)` for the dLHS kernel.

  `bn == n` matters most: it removes the reduction loop and quantizes each
  cotangent tile exactly once. Next comes `bk == k`, so the cotangent is read
  once rather than `k / bk` times, and with `bn == n` as well the weight block
  index is constant across the whole grid, so the weights load once too.
  Measured on (32768, 3072, 4096) at `bm = 512`: 571.4 us with `bk == k`
  against 651.2 us at `bk = 1536` and 666.3 us at `bk = 1024`, i.e. 1.14x over
  the best smaller k block. `bm` carries no traffic penalty, but a larger block
  means fewer grid steps, worth a few percent where it fits: so take the
  largest `bm` the VMEM model allows.

  The search follows that order of preference: for each `bn` (largest first),
  for each `bk` (largest first), the largest `bm` that fits the VMEM budget
  wins. So `bk` shrinks before `bn` does, and `bn` only shrinks once no
  `(bk, bm)` fits at the larger `bn`, which happens when `k * n` alone
  outgrows VMEM.

  Args:
    m: The row dimension.
    k: The output column dimension of `dlhs`.
    n: The contracting dimension.

  Returns:
    `(bm, bk, bn)`, each dividing its dimension.
  """
  bns = _tile_candidates(n, (n, 2048, 1024, 512))
  bks = _tile_candidates(k, (k, 1536, 1024, 512, 256))
  bms = _tile_candidates(m, (1024, 512, 256, 128))
  for bn, bk, bm in itertools.product(bns, bks, bms):
    if dlhs_vmem_bytes(bm, bk, bn, k, n) <= VMEM_BUDGET_BYTES:
      return bm, bk, bn
  # Nothing fits the model's budget; the smallest tiles are the best bet.
  return bms[-1], bks[-1], bns[-1]


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Tiling for the Pallas Mosaic TPU dLHS kernel.

  Attributes:
    block_m: Block size along the row (`m`) axis of the cotangent. Must divide
      `m`.
    block_k: Block size along the column (`k`) axis of `dlhs`. Must divide `k`.
      With `block_k < k` the kernel caches the quantized cotangent across k
      blocks.
    block_n: Block size along the contracting (`n`) axis. Must divide `n`. With
      `block_n == n` each cotangent row is quantized against a single scale and
      no accumulator is needed.
  """

  block_m: _BlockSize
  block_k: _BlockSize
  block_n: _BlockSize


def get_heuristics_config(m: int, k: int, n: int) -> Config:
  """Returns the tiling the measured heuristics pick for `(m, k, n)`."""
  bm, bk, bn = fit_dlhs_tiles(m, k, n)
  return Config(block_m=bm, block_k=bk, block_n=bn)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicTpuFusedFp8MatmulVjp(base.FusedFp8MatmulVjp[Config]):
  """VJP of the fused FP8 matmul using the Pallas dLHS kernel."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  def _fwd(
      self,
      residuals: base.Residuals | None,
      out: jax.Array,
      dout: jax.Array,
      lhs: jax.Array,
      rhs: jax.Array | QArray,
      *,
      return_residuals: bool,
      config: Config,
  ) -> tuple[tuple[jax.Array, Any], None]:
    """Returns `((dlhs, drhs), None)`; see `base.FusedFp8MatmulVjp`."""
    del out  # Unused.
    if residuals is None:
      _, residuals = base.fused_fp8_matmul_reference(lhs, rhs)
    lhs_q, lhs_scale, rhs_q, rhs_scale = residuals
    dlhs = dlhs_kernel.fused_fp8_dlhs_pallas(
        dout,
        rhs_q,
        rhs_scale,
        bm=config.block_m,
        bk=config.block_k,
        bn=config.block_n,
        out_dtype=lhs.dtype,
    )
    if isinstance(rhs, QArray):
      drhs = jax.tree.map(jnp.zeros_like, rhs)
    else:
      drhs = base.fused_fp8_drhs_reference(lhs_q, lhs_scale, dout)
      drhs = drhs.astype(rhs.dtype)
    return (dlhs, drhs), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    m, n = ba.arguments["dout"].shape
    k = ba.arguments["lhs"].shape[1]
    return get_heuristics_config(m, k, n)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    m, n = ba.arguments["dout"].shape
    k = ba.arguments["lhs"].shape[1]
    block_ms = [b for b in (256, 512, 1024, 2048) if m % b == 0]
    block_ks = [b for b in (256, 512, 1024, 1536, 2048, 3072) if k % b == 0]
    block_ns = [b for b in (512, 1024, 2048, 4096) if n % b == 0]
    configs = set()
    for bm, bk, bn in itertools.product(block_ms, block_ks, block_ns):
      if dlhs_vmem_bytes(bm, bk, bn, k, n) <= VMEM_BUDGET_BYTES:
        configs.add(Config(block_m=bm, block_k=bk, block_n=bn))
    configs.add(get_heuristics_config(m, k, n))
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return fwd_kernel.fp8_matmul_supported_on(device)
