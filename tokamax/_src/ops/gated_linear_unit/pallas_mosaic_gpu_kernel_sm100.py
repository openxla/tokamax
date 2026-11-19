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
"""Pallas-Mosaic-GPU Gated Linear Unit SM100 kernel."""

from typing import Callable

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import, g-importing-member
from tokamax._src import jaxtyping
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src.ops import op
from tokamax._src.ops.gated_linear_unit import base
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu_common as common


ACC_NUM_SLOTS = 2
_SM100_MAX_SMEM_BYTES = 224 * 1024
_SM100_MAX_TMEM_COLS = 512


def _sm100_smem_bytes(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_stages: int,
    epi_tile_n: int,
    cluster_size_m: int,
    k: int,
    in_itemsize: int = 2,
    out_itemsize: int = 2,
) -> int:
  """Returns the shared memory footprint in bytes for an SM100 GLU config."""
  b_tile_n = tile_n // cluster_size_m
  eff_stages = max(1, min(num_stages, k // tile_k))
  in_bytes = (
      eff_stages * (tile_m * tile_k + 2 * tile_k * b_tile_n) * in_itemsize
  )
  out_bytes = ACC_NUM_SLOTS * tile_m * epi_tile_n * out_itemsize
  return in_bytes + out_bytes


def get_heuristics_config(ba: op.BoundArguments) -> common.Config:
  """Returns a shape- and hardware-aware heuristic Config for SM100."""
  m, n, k, in_dtype, store_dtype, _ = common.get_problem_spec(ba)
  if m == 0 or n < 64 or k < 32:
    return common.Config(tile_m=128, tile_n=64, tile_k=64, num_stages=1)
  num_sms = getattr(backend.get_default_device(), "core_count", 1)
  in_itemsize = in_dtype.itemsize
  out_itemsize = store_dtype.itemsize

  tile_m = 128
  # Dual accumulators (acc0_tmem, acc1_tmem) with 2 slots require 4 * tile_n
  # TMEM columns, so tile_n can be 64 or 128.
  if n % 128 == 0 and m >= 256 and pl.cdiv(m, 128) * (n // 128) >= num_sms:
    tile_n = 128
  else:
    tile_n = 64

  # Use 2-CTA collective MMA when there are enough M-tiles and SMs, and RHS
  # weights are large enough to amortize cluster synchronization.
  if (
      m >= 256
      and k >= 128
      and n >= 256
      and pl.cdiv(m, 256) * (n // tile_n) >= num_sms // 2
      and tile_n >= 128
  ):
    cluster_size_m = 2
  else:
    cluster_size_m = 1

  # tile_k=64 enables 4-6 pipeline stages so TMA loads stay ahead of
  # tcgen05_mma; fall back to tile_k=32 when k is not divisible by 64.
  tile_k = 32 if k % 64 != 0 else 64
  epi_tile_n = 32 if (k <= 64 or out_itemsize > 2) else 64

  num_stages = max(1, min(6, k // tile_k))
  while (
      num_stages > 1
      and _sm100_smem_bytes(
          tile_m,
          tile_n,
          tile_k,
          num_stages,
          epi_tile_n,
          cluster_size_m,
          k,
          in_itemsize,
          out_itemsize,
      )
      > _SM100_MAX_SMEM_BYTES
  ):
    num_stages -= 1

  cluster_tile_m = tile_m * cluster_size_m
  grid_tile_width = (
      4 if (pl.cdiv(m, cluster_tile_m) >= 4 and (n // tile_n) >= 4) else 1
  )

  return common.Config(
      tile_m=tile_m,
      tile_n=tile_n,
      tile_k=tile_k,
      num_stages=num_stages,
      epi_tile_n=epi_tile_n,
      grid_minor_dim=common.MatmulDimension.N,
      grid_tile_width=grid_tile_width,
      cluster_size_m=cluster_size_m,
      cluster_size_n=1,
  )


def get_autotuning_configs(ba: op.BoundArguments) -> set[common.Config]:
  """Returns the autotuning configs for the Pallas:MGPU GLU SM100 kernel."""
  m, n, k, in_dtype, store_dtype, _ = common.get_problem_spec(ba)
  if m == 0 or n == 0 or k == 0:
    return set()
  in_itemsize = in_dtype.itemsize
  out_itemsize = store_dtype.itemsize
  configs = set()
  tile_m = 128
  tile_k_candidates = (64, 128) if k % 64 == 0 else (32,)
  for tile_n in (64, 128):
    if n % tile_n != 0 or 2 * ACC_NUM_SLOTS * tile_n > _SM100_MAX_TMEM_COLS:
      continue
    for tile_k in tile_k_candidates:
      if k % tile_k != 0:
        continue
      for cluster_size_m in (1, 2):
        b_tile_n = tile_n // cluster_size_m
        swizzle = min(
            plgpu.find_swizzle(tile_k * in_itemsize * 8),
            plgpu.find_swizzle(b_tile_n * in_itemsize * 8),
        )
        swizzle_elems = swizzle // in_itemsize
        if tile_k % swizzle_elems != 0 or b_tile_n % swizzle_elems != 0:
          continue
        for epi_tile_n in (32, 64):
          if tile_n % epi_tile_n != 0:
            continue
          max_stages = k // tile_k
          stage_candidates = (
              (1,)
              if max_stages == 1
              else tuple(s for s in (2, 4, 6) if s <= max_stages)
          )
          for num_stages in stage_candidates:
            if (
                _sm100_smem_bytes(
                    tile_m,
                    tile_n,
                    tile_k,
                    num_stages,
                    epi_tile_n,
                    cluster_size_m,
                    k,
                    in_itemsize,
                    out_itemsize,
                )
                > _SM100_MAX_SMEM_BYTES
            ):
              continue
            for grid_tile_width in (1, 4, 8):
              configs.add(
                  common.Config(
                      tile_m=tile_m,
                      tile_n=tile_n,
                      tile_k=tile_k,
                      num_stages=num_stages,
                      epi_tile_n=epi_tile_n,
                      grid_minor_dim=common.MatmulDimension.N,
                      grid_tile_width=grid_tile_width,
                      cluster_size_m=cluster_size_m,
                      cluster_size_n=1,
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
  """Gated Linear Unit implementation for SM100.

  Args:
    x: LHS operand.
    weights: RHS operand (fused `(K, 2, N)` or unfused `((K, N), (K, N))`).
    activation: The activation function to use.
    config: Kernel tile and pipeline configuration.
    out_dtype: Output and residual storage dtype (defaults to `x.dtype`).
    return_residuals: Whether to also return `(..., 2, N)` pre-activations.
    approx_math: Whether to allow PTX `approx` math instructions.

  Returns:
    The result of the GLU, or `(out, residuals)` if `return_residuals=True`.
  """
  is_unfused = isinstance(weights, tuple)
  for w in jax.tree.leaves(weights):
    if x.dtype != w.dtype:
      raise ValueError(
          f"Matmul LHS and RHS have incompatible dtypes {x.dtype} vs {w.dtype}"
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

  if m == 0 or n == 0 or k == 0:
    raise NotImplementedError("Empty dimensions are not supported.")

  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  if tile_m != 128:
    raise NotImplementedError("Only tile_m=128 is supported on SM100.")
  if config.cluster_size_n > 1:
    raise NotImplementedError("cluster_size_n > 1 is not supported on SM100.")
  if 2 * ACC_NUM_SLOTS * tile_n > _SM100_MAX_TMEM_COLS:
    raise NotImplementedError(
        f"{tile_n=} exceeds TMEM capacity ({_SM100_MAX_TMEM_COLS} columns)."
    )
  if n % tile_n != 0:
    raise NotImplementedError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise NotImplementedError(f"{k=} must be divisible by {tile_k=}")

  epi_tile_n = config.epi_tile_n or tile_n
  if tile_n % epi_tile_n != 0:
    raise NotImplementedError(f"{tile_n=} must be divisible by {epi_tile_n=}")

  cluster_tile_m = tile_m * config.cluster_size_m
  b_tile_n = tile_n // config.cluster_size_m

  in_itemsize = jnp.dtype(dtype).itemsize
  out_itemsize = store_dtype.itemsize
  swizzle = min(
      plgpu.find_swizzle(tile_k * in_itemsize * 8),
      plgpu.find_swizzle(b_tile_n * in_itemsize * 8),
  )
  swizzle_elems = swizzle // in_itemsize
  if tile_k % swizzle_elems != 0 or b_tile_n % swizzle_elems != 0:
    raise NotImplementedError(
        f"{tile_k=} and {b_tile_n=} must be multiples of {swizzle_elems=}"
    )
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )

  out_swizzle = plgpu.find_swizzle(epi_tile_n * out_itemsize * 8)
  out_swizzle_elems = out_swizzle // out_itemsize
  out_transforms = (
      plgpu.TilingTransform((8, out_swizzle_elems)),
      plgpu.SwizzleTransform(out_swizzle),
  )

  # TilingTransform((8, swizzle_elems)) requires the GMEM M dimension to be a
  # multiple of 8; hardware TMA zero-fills out-of-bounds rows up to
  # cluster_tile_m and ignores out-of-bounds rows on TMA store.
  if m % 8 != 0:
    x = jnp.pad(x, ((0, -m % 8), (0, 0)))
  m_padded = x.shape[0]

  cluster_size = config.cluster_size_m * config.cluster_size_n
  collective = cluster_size > 1
  collective_axis = "cluster" if collective else None
  m_iters = pl.cdiv(m_padded, cluster_tile_m)
  n_iters = n // tile_n
  k_iters = k // tile_k
  max_concurrent_steps = min(config.num_stages, k_iters)

  def kernel(
      a_gmem,
      b_gmem,
      *out_refs,
      a_smem,
      b0_smem,
      b1_smem,
      acc0_tmem,
      acc1_tmem,
      acc_smem,
      ab_tma_barrier,
      store_done_barrier,
      mma_done_barrier,
      consumed_barrier_0,
      consumed_barrier_1,
  ):
    out_gmem = out_refs[0]
    res_gmem = out_refs[1] if return_residuals else None
    wg_idx = lax.axis_index("wg")
    cluster_idx = lax.axis_index("cluster")
    is_lead_block = cluster_idx == 0

    @plgpu.dynamic_scheduling_loop(
        grid_names=("mn_linear",),
        thread_axis="wg",
        cluster_axes=("cluster",) if collective else (),
    )
    def mn_loop(loop_info):
      (lin_idx,) = loop_info.index
      local_index = loop_info.local_index
      m_index, n_index = plgpu.planar_snake(
          lin_idx,
          (m_iters, n_iters),
          config.grid_minor_dim,
          config.grid_tile_width,
      )
      block_m_index, block_n_index = m_index, n_index
      if config.cluster_size_m > 1:
        block_m_index = m_index * config.cluster_size_m + cluster_idx
      acc_slot = mgpu_lib.urem(local_index, jnp.int32(ACC_NUM_SLOTS))

      slice_m = pl.ds(block_m_index * tile_m, tile_m)
      a_slice_m = pl.ds(m_index * cluster_tile_m, cluster_tile_m)

      # Accumulator for X @ W0
      acc_tmem_slice_0 = acc0_tmem.at[:, pl.ds(acc_slot * tile_n, tile_n)]
      # Accumulator for X @ W1
      acc_tmem_slice_1 = acc1_tmem.at[:, pl.ds(acc_slot * tile_n, tile_n)]

      @mgpu_lib.warp_map_when(wg_idx == 0)
      def _compute_wg_warp(warp_id):
        b_smems = (b0_smem, b1_smem)
        consumed_barriers = (consumed_barrier_0, consumed_barrier_1)

        @pl.when(warp_id == 0)
        def _tma_warp():
          def _loop_body(ki, _):
            slice_k = pl.ds(ki * tile_k, tile_k)
            slice_n = pl.ds(block_n_index * tile_n, tile_n)
            slot = mgpu_lib.urem(ki, jnp.int32(max_concurrent_steps))

            def _wait_barrier(barrier):
              @pl.when(
                  jnp.logical_or(ki >= max_concurrent_steps, local_index > 0)
              )
              def _():
                plgpu.barrier_wait(barrier.at[slot])

            for i in range(2):
              _wait_barrier(consumed_barriers[i])
              b_gmem_i = b_gmem[i] if is_unfused else b_gmem.at[:, i]
              plgpu.copy_gmem_to_smem(
                  b_gmem_i.at[slice_k, slice_n],
                  b_smems[i].at[slot],
                  ab_tma_barrier.at[slot],
                  leader_tracked=plgpu.CopyPartition.PARTITIONED(1)
                  if config.cluster_size_m > 1
                  else None,
                  collective_axes=collective_axis,
              )
            plgpu.copy_gmem_to_smem(
                a_gmem.at[a_slice_m, slice_k],
                a_smem.at[slot],
                ab_tma_barrier.at[slot],
                leader_tracked=plgpu.CopyPartition.PARTITIONED(0)
                if config.cluster_size_m > 1
                else None,
                collective_axes=collective_axis,
            )

          lax.fori_loop(jnp.int32(0), jnp.int32(k_iters), _loop_body, None)

        @pl.when(jnp.logical_and(warp_id == 1, local_index > 1))
        def _wait_store():
          plgpu.barrier_wait(store_done_barrier.at[acc_slot])

        @pl.when(jnp.logical_and(warp_id == 1, is_lead_block))
        def _mma_warp():
          def _loop_body(ki, _):
            slot = mgpu_lib.urem(ki, jnp.int32(max_concurrent_steps))
            plgpu.barrier_wait(ab_tma_barrier.at[slot])
            # TODO: When tcgen05 supports reshapes with
            # transposes like wgmma we can avoid the dual
            # accumulators.
            acc_tmem_slices = (acc_tmem_slice_0, acc_tmem_slice_1)
            for i in range(2):
              plgpu.tcgen05_mma(
                  acc_tmem_slices[i],
                  a_smem.at[slot],
                  b_smems[i].at[slot],
                  consumed_barriers[i].at[slot],
                  accumulate=(ki > 0),
                  collective_axis=collective_axis,
              )

            @pl.when(ki >= k_iters - 1)
            def _arrive():
              plgpu.tcgen05_commit_arrive(
                  mma_done_barrier.at[acc_slot],
                  collective_axis=collective_axis,
              )

          lax.fori_loop(jnp.int32(0), jnp.int32(k_iters), _loop_body, None)

      @pl.when(wg_idx == 1)
      def _store_wg():
        plgpu.barrier_wait(mma_done_barrier.at[acc_slot])
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        step_out_gmem = out_gmem.at[
            slice_m, pl.ds(block_n_index * tile_n, tile_n)
        ]
        step_res0_gmem = (
            res_gmem.at[slice_m, 0, pl.ds(block_n_index * tile_n, tile_n)]
            if res_gmem is not None
            else None
        )
        step_res1_gmem = (
            res_gmem.at[slice_m, 1, pl.ds(block_n_index * tile_n, tile_n)]
            if res_gmem is not None
            else None
        )
        store_idx = 0

        def store_epi_tile(val, dst_ref):
          nonlocal store_idx
          acc_smem_slot = acc_smem.at[store_idx % ACC_NUM_SLOTS]
          plgpu.wait_smem_to_gmem(ACC_NUM_SLOTS - 1, wait_read_only=True)
          acc_smem_slot[...] = val.astype(store_dtype)
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(acc_smem_slot, dst_ref)
          store_idx += 1

        for ni in range(tile_n // epi_tile_n):
          ni_col_slice = pl.ds(ni * epi_tile_n, epi_tile_n)
          val0 = plgpu.async_load_tmem(acc_tmem_slice_0.at[:, ni_col_slice])
          val1 = plgpu.async_load_tmem(acc_tmem_slice_1.at[:, ni_col_slice])
          plgpu.wait_load_tmem()
          if return_residuals:
            assert step_res0_gmem is not None and step_res1_gmem is not None
            store_epi_tile(val0, step_res0_gmem.at[:, ni_col_slice])
            store_epi_tile(val1, step_res1_gmem.at[:, ni_col_slice])
          val0 = val0.astype(store_dtype).astype(jnp.float32)
          val1 = val1.astype(store_dtype).astype(jnp.float32)
          store_epi_tile(
              activation(val0) * val1, step_out_gmem.at[:, ni_col_slice]
          )
        plgpu.barrier_arrive(store_done_barrier.at[acc_slot])

    @pl.when(wg_idx == 1)
    def _wait_final_stores():
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  barrier_kwargs = dict(
      num_arrivals=1,
      num_barriers=2,
      orders_tensor_core=True,
  )
  if collective:
    store_done_barrier = plgpu.ClusterBarrier(
        collective_axes=("cluster",), **barrier_kwargs
    )
  else:
    store_done_barrier = plgpu.Barrier(**barrier_kwargs)
  smem_b_shape = (max_concurrent_steps, tile_k, b_tile_n)
  tmem_acc_shape = (tile_m, tile_n * 2)
  consumed_barrier_kwargs = dict(
      num_arrivals=1,
      num_barriers=max_concurrent_steps,
      orders_tensor_core=True,
  )
  out_struct = jax.ShapeDtypeStruct((m_padded, n), store_dtype)
  out_type = (
      (out_struct, jax.ShapeDtypeStruct((m_padded, 2, n), store_dtype))
      if return_residuals
      else out_struct
  )
  f = plgpu.kernel(
      kernel,
      out_type=out_type,
      grid=(m_iters * n_iters,),
      grid_names=("mn_linear",),
      cluster=(cluster_size,),
      cluster_names=("cluster",),
      num_threads=2,
      thread_name="wg",
      scratch_types=dict(
          a_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_m, tile_k),
              dtype,
              transforms=transforms,
          ),
          b0_smem=plgpu.SMEM(smem_b_shape, dtype, transforms=transforms),
          b1_smem=plgpu.SMEM(smem_b_shape, dtype, transforms=transforms),
          acc0_tmem=plgpu.TMEM(
              tmem_acc_shape, jnp.float32, collective=collective
          ),
          acc1_tmem=plgpu.TMEM(
              tmem_acc_shape, jnp.float32, collective=collective
          ),
          acc_smem=plgpu.SMEM(
              (2, tile_m, epi_tile_n), store_dtype, transforms=out_transforms
          ),
          ab_tma_barrier=plgpu.Barrier(
              num_arrivals=3, num_barriers=max_concurrent_steps
          ),
          store_done_barrier=store_done_barrier,
          mma_done_barrier=plgpu.Barrier(**barrier_kwargs),
          consumed_barrier_0=plgpu.Barrier(**consumed_barrier_kwargs),
          consumed_barrier_1=plgpu.Barrier(**consumed_barrier_kwargs),
      ),
      compiler_params=plgpu.CompilerParams(approx_math=approx_math),
  )
  result = f(x, weights)
  if return_residuals:
    out, res = result
    if m_padded != m:
      out = out[:m]
      res = res[:m]
    return (
        jnp.reshape(out, (*orig_x_shape[:-1], n)),
        jnp.reshape(res, (*orig_x_shape[:-1], 2, n)),
    )
  if m_padded != m:
    result = result[:m]
  return jnp.reshape(result, (*orig_x_shape[:-1], n))
