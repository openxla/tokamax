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
from tokamax._src.ops.gated_linear_unit import base
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu_common as common


_SM80_MAX_SMEM_BYTES = 160 * 1024


def _sm80_smem_bytes(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_stages: int,
    in_itemsize: int = 2,
    out_itemsize: int = 2,
) -> int:
  """Returns the shared memory footprint in bytes for an SM80 GLU config."""
  out_bytes = tile_m * tile_n * out_itemsize
  in_bytes = num_stages * (tile_m * tile_k + 2 * tile_k * tile_n) * in_itemsize
  return out_bytes + in_bytes


def _largest_tile_divisor(dim: int, cap: int) -> int:
  """Largest multiple of 64 that is <= `cap` and divides `dim` exactly (>= 64)."""
  tile = (cap // 64) * 64
  while tile > 64 and dim % tile != 0:
    tile -= 64
  return tile


def _best_tile_size(m: int, n: int, num_sms: int) -> tuple[int, int]:
  """Returns `(tile_m, tile_n)` tile sizes based on shape and SM count."""
  tile_n = _largest_tile_divisor(n, 64)
  if m <= 64 or pl.cdiv(m, 128) * max(1, n // tile_n) < 2 * num_sms:
    tile_m = 64
  else:
    tile_m = 128
  return tile_m, tile_n


def get_heuristics_config(ba: op.BoundArguments) -> common.Config:
  """Generates a default tile and pipeline configuration for the given shapes."""
  m, n, k, in_dtype, store_dtype, _ = common.get_problem_spec(ba)
  if m == 0 or n == 0 or k == 0:
    return common.Config(
        tile_m=64,
        tile_n=64,
        tile_k=32,
        num_stages=1,
        grid_minor_dim=common.MatmulDimension.M,
    )
  num_sms = getattr(backend.get_default_device(), "core_count", 1)
  in_itemsize = in_dtype.itemsize
  out_itemsize = store_dtype.itemsize
  tile_m, tile_n = _best_tile_size(m, n, num_sms)
  tile_k = 16 if (k % 32 != 0 and k % 16 == 0) else 32
  # Target 3 stages (fits <= 64 KB SMEM so 2 CTAs co-reside per SM on A100).
  num_stages = max(1, min(3, k // tile_k))
  while (
      num_stages > 1
      and _sm80_smem_bytes(
          tile_m, tile_n, tile_k, num_stages, in_itemsize, out_itemsize
      )
      > _SM80_MAX_SMEM_BYTES
  ):
    num_stages -= 1
  m_iters = pl.cdiv(m, tile_m)
  n_iters = max(1, n // tile_n)
  grid_tile_width = 4 if (m_iters >= 4 and n_iters >= 4) else 1
  grid_minor_dim = (
      common.MatmulDimension.N
      if n_iters >= m_iters
      else common.MatmulDimension.M
  )
  return common.Config(
      tile_m=tile_m,
      tile_n=tile_n,
      tile_k=tile_k,
      num_stages=num_stages,
      grid_minor_dim=grid_minor_dim,
      grid_tile_width=grid_tile_width,
  )


def get_autotuning_configs(ba: op.BoundArguments) -> set[common.Config]:
  """Returns the autotuning configs for the Pallas:MGPU GLU SM80 kernel."""
  m, n, k, in_dtype, store_dtype, _ = common.get_problem_spec(ba)
  if m == 0 or n == 0 or k == 0:
    return set()
  in_itemsize = in_dtype.itemsize
  out_itemsize = store_dtype.itemsize
  configs = set()
  for tile_m in (64, 128):
    for tile_n in (64, 128):
      if n % tile_n or tile_m * tile_n > 128 * 128:
        continue
      for tile_k in (32, 64):
        if k % tile_k:
          continue
        # No point pipelining deeper than there are k-iterations.
        max_stages = min(5, k // tile_k)
        min_stages = 1 if max_stages == 1 else 2
        for num_stages in range(min_stages, max_stages + 1):
          if (
              _sm80_smem_bytes(
                  tile_m, tile_n, tile_k, num_stages, in_itemsize, out_itemsize
              )
              > _SM80_MAX_SMEM_BYTES
          ):
            continue
          for grid_minor_dim in (
              common.MatmulDimension.M,
              common.MatmulDimension.N,
          ):
            for grid_tile_width in (1, 4):
              configs.add(
                  common.Config(
                      tile_m=tile_m,
                      tile_n=tile_n,
                      tile_k=tile_k,
                      num_stages=num_stages,
                      grid_minor_dim=grid_minor_dim,
                      grid_tile_width=grid_tile_width,
                  )
              )
  return configs


@jaxtyping.jaxtyped
def gated_linear_unit(
    x: Float[Array, "*B M K"],
    weights: base.FusedWeights | base.UnfusedWeights,
    *,
    activation: Callable[[jax.Array], jax.Array],
    config: common.Config,
    out_dtype: jax.typing.DTypeLike | None = None,
    return_residuals: bool = False,
    approx_math: bool = True,
) -> (
    Float[Array, "*B M N"]
    | tuple[Float[Array, "*B M N"], Float[Array, "*B M 2 N"]]
):
  """Gated Linear Unit implementation for SM80."""
  is_unfused = isinstance(weights, tuple)
  for w in jax.tree.leaves(weights):
    if x.dtype != w.dtype:
      raise ValueError(
          f"Matmul operands have incompatible dtypes {x.dtype} vs {w.dtype}"
      )

  orig_x_shape = x.shape
  x = jax.lax.collapse(x, 0, -1)
  m, k = x.shape
  n = weights[0].shape[-1] if is_unfused else weights.shape[-1]
  if is_unfused:
    if weights[0].shape != (k, n) or weights[1].shape != (k, n):
      raise ValueError(
          f"Incompatible shapes: x={x.shape},"
          f" weights=({weights[0].shape}, {weights[1].shape})"
      )
  elif weights.shape != (k, 2, n):
    raise ValueError(
        f"Incompatible shapes: x={x.shape}, weights={weights.shape}"
    )
  dtype = x.dtype
  store_dtype = jnp.dtype(dtype if out_dtype is None else out_dtype)

  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k

  # `NotImplementedError`, not `ValueError`: see the note on the equivalent
  # guards in `pallas_mosaic_gpu_kernel_sm90.py`. This is the signal `api.py`
  # catches to fall through to the next implementation.
  if m == 0 or n == 0 or k == 0:
    raise NotImplementedError("Empty dimensions are not supported.")
  if tile_k < 16 or tile_k % 16 != 0:
    raise NotImplementedError(f"{tile_k=} must be a multiple of 16.")
  if n % tile_n != 0:
    raise NotImplementedError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise NotImplementedError(f"{k=} must be divisible by {tile_k=}")

  if m % tile_m != 0:
    x = jnp.pad(x, ((0, -m % tile_m), (0, 0)))
  m_padded = x.shape[0]

  grid = (m_padded // tile_m, n // tile_n)
  out_smem_ty = mgpu_lib.tiled_swizzled_smem((tile_m, tile_n), store_dtype)
  num_blocks = max(
      1, min(math.prod(grid), 4 * backend.get_default_device().core_count)
  )
  out_struct = jax.ShapeDtypeStruct((m_padded, n), store_dtype)
  out_type = (
      (out_struct, jax.ShapeDtypeStruct((m_padded, 2, n), store_dtype))
      if return_residuals
      else out_struct
  )
  eff_stages = min(config.num_stages, k // tile_k)

  def kernel(x_gmem, weights_gmem, *out_refs, out_smem):
    out_gmem = out_refs[0]
    res_gmem = out_refs[1] if return_residuals else None

    @plgpu.nd_loop((math.prod(grid),), collective_axes="block")
    def mn_loop(loop_info: plgpu.NDLoopInfo):
      (lin_idx,) = loop_info.index
      mi, ni = plgpu.planar_snake(
          lin_idx, grid, config.grid_minor_dim, config.grid_tile_width
      )

      def compute(_, x_smem, wg_smem, wp_smem, accs):
        gates, proj = accs
        with jax.named_scope("load"):
          x = x_smem[...]
          w = wg_smem[...]
          v = wp_smem[...]
        with jax.named_scope("mma"):
          gates = plgpu.mma(gates, x, w)
          proj = plgpu.mma(proj, x, v)
        return gates, proj

      spec = mgpu_lib.tiled_swizzled_block_spec
      x_spec = spec((tile_m, tile_k), dtype, lambda ki: (mi, ki), "x")
      w_spec = spec((tile_k, tile_n), dtype, lambda ki: (ki, ni), "w")

      if is_unfused:
        wg_gmem, wp_gmem = weights_gmem
      else:
        wg_gmem, wp_gmem = weights_gmem.at[:, :n], weights_gmem.at[:, n:]

      acc = jnp.zeros(out_smem.shape, jnp.float32)
      gates, proj = plgpu.emit_pipeline(
          compute,
          grid=(k // tile_k,),
          in_specs=(x_spec, w_spec, w_spec),
          max_concurrent_steps=eff_stages,
          init_carry=(acc, acc),
      )(x_gmem, wg_gmem, wp_gmem)

      with jax.named_scope("epilogue"):
        # Relayout through swizzled SMEM so the global store coalesces (MMA_ACC
        # layout emits scattered 4B stores). Ampere lacks TMA, so this is a
        # manual STS/LDS/STG path; the swizzle avoids STS bank conflicts.
        copy_layout = plgpu.Layout.SMEM_GMEM_COPY(
            out_smem.shape,
            store_dtype,
            swizzle=out_smem_ty.transforms[-1].swizzle,  # pyrefly: ignore[missing-attribute]
        )

        def store_tile(gmem_ref, val):
          out_smem[...] = val.astype(store_dtype)
          staged = plgpu.layout_cast(out_smem[...], copy_layout)
          mgpu_lib.store(gmem_ref, staged, optimized=False)

        ms = pl.ds(mi * tile_m, tile_m)
        ns = pl.ds(ni * tile_n, tile_n)
        if return_residuals:
          assert res_gmem is not None
          store_tile(res_gmem.at[ms, 0, ns], gates)
          mgpu_lib.warpgroup_barrier()
          store_tile(res_gmem.at[ms, 1, ns], proj)
          mgpu_lib.warpgroup_barrier()
        store_tile(out_gmem.at[ms, ns], proj * activation(gates))

  f = plgpu.kernel(
      kernel,
      out_type=out_type,
      scratch_types=dict(out_smem=out_smem_ty),
      grid=(num_blocks,),
      grid_names=("block",),
      compiler_params=plgpu.CompilerParams(approx_math=approx_math),
  )
  kernel_weights = weights if is_unfused else weights.reshape(k, 2 * n)
  result = f(x, kernel_weights)
  if return_residuals:
    out, res = result
    if m_padded != m:
      out = out[:m]
      res = res[:m]
    return (
        out.reshape(*orig_x_shape[:-1], n),
        res.reshape(*orig_x_shape[:-1], 2, n),
    )
  if m_padded != m:
    result = result[:m]
  return result.reshape(*orig_x_shape[:-1], n)
