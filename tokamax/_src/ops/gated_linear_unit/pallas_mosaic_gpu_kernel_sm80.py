# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Pallas-Mosaic-GPU Gated Linear Unit SM80 kernel."""

import math
from typing import Callable
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-importing-member,g-multiple-import
from tokamax._src import jaxtyping
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src.ops import op
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu_common as common


def _largest_tile_divisor(dim: int, cap: int) -> int:
  """Largest multiple of 64 that is <= `cap` and divides `dim` exactly (>= 64)."""
  tile = (cap // 64) * 64
  while tile > 64 and dim % tile != 0:
    tile -= 64
  return tile


def _best_tile_size(major: int, minor: int, num_sms: int) -> tuple[int, int]:
  """Returns (major, minor) tile sizes based on shape and SM count.

  Picks power-of-two tiles capped at a combined accumulator area of 128x128.
  Tiles are constrained to multiples of 64 that divide dimensions evenly,
  as the kernel does not perform boundary masking.

  Args:
    major: The size of the major (M) dimension.
    minor: The size of the minor (N) dimension.
    num_sms: The number of SMs on the device.
  """
  t_major = min(max(64, pl.next_power_of_2(major)), 128)
  t_minor = min(max(64, pl.next_power_of_2(minor)), 256)
  t_minor = min(t_minor, (128 * 128) // t_major)
  t_major = _largest_tile_divisor(major, t_major)
  t_minor = _largest_tile_divisor(minor, t_minor)
  # Shrink the major tile until there are enough tiles to cover every SM; this
  # also lowers the per-block register/SMEM footprint, raising occupancy.
  while (
      pl.cdiv(major, t_major) * pl.cdiv(minor, t_minor) < num_sms
      and t_major > 64
  ):
    t_major = _largest_tile_divisor(major, t_major // 2)
  return t_major, t_minor


def get_heuristics_config(ba: op.BoundArguments) -> common.Config:
  """Generates a default tile and pipeline configuration for the given shapes."""
  x = ba.arguments["x"]
  weights = ba.arguments["weights"]
  m = math.prod(x.shape[:-1])
  k = x.shape[-1]
  # `weights` is either a fused `[..., K, 2, N]` array or a tuple of two
  # `[..., K, N]` arrays (gate, proj); `N` is the last axis in both cases.
  n = weights[0].shape[-1] if isinstance(weights, tuple) else weights.shape[-1]
  num_sms = backend.get_default_device().core_count
  if n >= m:  # Prefer `tile_n` > `tile_m`.
    tile_m, tile_n = _best_tile_size(m, n, num_sms)
  else:
    tile_n, tile_m = _best_tile_size(n, m, num_sms)
  # Gate and proj maintain separate accumulators; halve `tile_n` to keep the
  # combined accumulator register footprint within budget.
  tile_n = _largest_tile_divisor(n, max(64, tile_n // 2))
  tile_k = math.gcd(k, 32)
  return common.Config(
      tile_m=tile_m,
      tile_n=tile_n,
      tile_k=tile_k,
      num_stages=min(2, k // tile_k),
      grid_minor_dim=common.MatmulDimension.M,
  )


def get_autotuning_configs(ba: op.BoundArguments) -> set[common.Config]:
  """Returns the autotuning configs for the Pallas:MGPU GLU SM80 kernel."""
  m = math.prod(ba.arguments["x"].shape[:-1])
  k = ba.arguments["x"].shape[-1]
  n = ba.arguments["weights"].shape[-1]
  configs = set()
  for tile_m in (64, 128):
    if m % tile_m:
      continue
    for tile_n in (64, 128, 256):
      if n % tile_n:
        continue
      for tile_k in (32, 64, 128):
        if k % tile_k:
          continue
        # No point pipelining deeper than there are k-iterations.
        max_stages = min(5, k // tile_k)
        for num_stages in range(1, max_stages + 1):
          configs.add(
              common.Config(
                  tile_m=tile_m,
                  tile_n=tile_n,
                  tile_k=tile_k,
                  num_stages=num_stages,
              )
          )
  return configs


@jaxtyping.jaxtyped
def gated_linear_unit(
    x: Float[Array, "*B M K"],
    weights: Float[Array, "K 2 N"],
    *,
    activation: Callable[[jax.Array], jax.Array],
    config: common.Config,
) -> Float[Array, "*B M N"]:
  """Gated Linear Unit implementation for SM80."""
  orig_x_shape = x.shape
  x = jax.lax.collapse(x, 0, -1)
  m, k = x.shape
  _, _, n = weights.shape
  dtype = x.dtype
  if x.dtype != weights.dtype:
    raise ValueError(
        f"Matmul LHS and RHS have incompatible dtypes {x.dtype} vs"
        f" {weights.dtype}"
    )

  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  num_stages = config.num_stages
  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")

  m_iters = m // tile_m
  n_iters = n // tile_n
  k_iters = k // tile_k

  num_sms = backend.get_default_device().core_count
  out_swizzle = plgpu.find_swizzle(tile_n * mgpu_lib.num_bits(dtype), "out")

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((m, n), dtype),
      scratch_types=[
          mgpu_lib.tiled_swizzled_smem(
              (tile_m, tile_n), dtype, "out", swizzle=out_swizzle
          ),
      ],
      grid=(num_sms * 4,),
      grid_names=("block",),
      compiler_params=plgpu.CompilerParams(
          # TODO: Migrate to WG semantics once it supports cp_async.
          lowering_semantics=plgpu.LoweringSemantics.Lane,
      ),
  )
  def kernel(a_gmem, weights_gmem, out_gmem, out_smem):

    @plgpu.nd_loop((m_iters * n_iters,), collective_axes="block")
    def mn_loop(loop_info):
      mi, ni = plgpu.planar_snake(
          loop_info.index[0],
          (m_iters, n_iters),
          config.grid_minor_dim,
          config.grid_tile_width,
      )

      acc = plgpu.layout_cast(
          jnp.zeros((tile_m, tile_n), jnp.float32),
          plgpu.Layout.MMA_ACC(dtype),
      )

      def compute(_, a_smem, wg_smem, wp_smem, accs):
        gates, proj = accs
        with jax.named_scope("load"):
          a = plgpu.load(a_smem, layout=plgpu.Layout.MMA_LHS(dtype))
          w = plgpu.load(wg_smem, layout=plgpu.Layout.MMA_RHS(dtype))
          v = plgpu.load(wp_smem, layout=plgpu.Layout.MMA_RHS(dtype))
        with jax.named_scope("mma"):
          gates = plgpu.mma(gates, a, w)
          proj = plgpu.mma(proj, a, v)
        return gates, proj

      gates, proj = plgpu.emit_pipeline(
          compute,
          grid=(k_iters,),
          in_specs=[
              mgpu_lib.tiled_swizzled_block_spec(
                  (tile_m, tile_k), dtype, lambda ki: (mi, ki), "a"
              ),
              mgpu_lib.tiled_swizzled_block_spec(
                  (tile_k, tile_n), dtype, lambda ki: (ki, ni), "wg"
              ),
              mgpu_lib.tiled_swizzled_block_spec(
                  (tile_k, tile_n), dtype, lambda ki: (ki, ni), "wp"
              ),
          ],
          max_concurrent_steps=num_stages,
          init_carry=(acc, acc),
      )(a_gmem, weights_gmem.at[:, :n], weights_gmem.at[:, n:])

      with jax.named_scope("epilogue"):
        out = proj * activation(gates)
        # Relayout through swizzled SMEM so the global store coalesces (MMA_ACC
        # layout emits scattered 4B stores). Ampere lacks TMA, so this is a
        # manual STS/LDS/STG path; the swizzle avoids STS bank conflicts.
        copy_layout = plgpu.Layout.SMEM_GMEM_COPY(
            (tile_m, tile_n), dtype, swizzle=out_swizzle
        )
        m_slice = pl.ds(mi * tile_m, tile_m)
        n_slice = pl.ds(ni * tile_n, tile_n)
        out_smem[...] = out.astype(dtype)
        out_gmem[m_slice, n_slice] = plgpu.layout_cast(
            out_smem[...], copy_layout
        )

  out = kernel(x, weights.reshape(k, 2 * n))
  return jnp.reshape(out, (*orig_x_shape[:-1], n))
