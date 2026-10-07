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
"""Pallas/Mosaic TPU implementation of the fused FP8 matmul."""

import dataclasses
import functools
import itertools
from typing import Annotated, ClassVar, override

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import pydantic
from tokamax._src.ops import op
from tokamax._src.ops.experimental.fused_fp8_matmul import base
from tokamax._src.ops.experimental.fused_fp8_matmul import pallas_mosaic_tpu_kernel as kernel

QArray = base.QArray

_BlockSize = Annotated[int, pydantic.Field(ge=128, multiple_of=128)]


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Tiling for the Pallas Mosaic TPU forward kernel.

  Attributes:
    block_m: Block size along the row (`m`) axis of `lhs`. Must divide `m`.
    block_k: Block size along the contracting (`k`) axis. Must divide `k`. When
      `block_k == k` the kernel runs without an accumulator and can emit the
      quantized activations as residuals.
    block_n: Block size along the column (`n`) axis of `rhs`. Must divide `n`.
      `block_n == n` avoids re-fetching each activation tile once per n block.
  """

  block_m: _BlockSize
  block_k: _BlockSize
  block_n: _BlockSize


def get_heuristics_config(m: int, k: int, n: int) -> Config:
  """Returns the tiling the measured heuristics pick for `(m, k, n)`."""
  bk = kernel.default_block_k(k)
  bm, bn = kernel.fit_forward_tiles(m, k, n, bk)
  return Config(block_m=bm, block_k=bk, block_n=bn)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicTpuFusedFp8Matmul(base.FusedFp8Matmul[Config]):
  """Fused FP8 matmul using the Pallas Mosaic TPU forward kernel.

  The backward pass currently uses the XLA reference VJP from the base class.
  """

  config_cls: ClassVar[type[Config]] = Config

  @override
  def _fwd(
      self,
      lhs: jax.Array,
      rhs: jax.Array | QArray,
      *,
      return_residuals: bool,
      config: Config,
  ) -> tuple[jax.Array, base.Residuals | None]:
    rhs_q, rhs_scale = base.rhs_as_fp8(rhs)
    k = lhs.shape[1]
    # The kernel can only emit the quantized activations as residuals when it
    # quantizes each row against a single scale, i.e. with one k block. The
    # heuristics pick `block_k == k` whenever it fits; for a hand-picked or
    # autotuned config with smaller k blocks, keep the configured tiling and
    # recompute the residuals in XLA instead (one extra read of `lhs`).
    run = functools.partial(
        kernel.fused_fp8_matmul_pallas,
        lhs,
        rhs_q,
        rhs_scale,
        bm=config.block_m,
        bk=config.block_k,
        bn=config.block_n,
    )
    if not return_residuals:
      return run(return_xq=False), None
    if config.block_k == k:
      out, lhs_q, lhs_scale = run(return_xq=True)
    else:
      out = run(return_xq=False)
      lhs_q, lhs_scale = base.quantize_rows_fp8(lhs)
    return out, (lhs_q, lhs_scale, rhs_q, rhs_scale)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    m, k = ba.arguments["lhs"].shape
    n = ba.arguments["rhs"].shape[1]
    return get_heuristics_config(m, k, n)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    m, k = ba.arguments["lhs"].shape
    n = ba.arguments["rhs"].shape[1]
    block_ms = [b for b in (256, 512, 1024, 2048) if m % b == 0]
    block_ns = [b for b in (256, 512, 1024, 2048, 4096) if n % b == 0]
    block_ks = [b for b in (256, 512, 1024, 2048, 4096) if k % b == 0]
    configs = set()
    for bm, bk, bn in itertools.product(block_ms, block_ks, block_ns):
      if (
          kernel.forward_vmem_bytes(bm, bk, bn, k, n)
          <= kernel.VMEM_BUDGET_BYTES
      ):
        configs.add(Config(block_m=bm, block_k=bk, block_n=bn))
    configs.add(get_heuristics_config(m, k, n))
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    if device.platform != "tpu":
      return False
    fp8 = jnp.float8_e4m3fn
    return pltpu.get_tpu_info().is_matmul_supported(fp8, fp8)
