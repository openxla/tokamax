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
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import, g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu_common as common


ACC_NUM_SLOTS = 2


def get_heuristics_config(ba: op.BoundArguments) -> common.Config:
  del ba
  return common.Config(
      tile_m=128,
      tile_n=64,
      tile_k=64,
      num_stages=4,
  )


def get_autotuning_configs(ba: op.BoundArguments) -> set[common.Config]:
  del ba
  return set()


@jaxtyping.jaxtyped
def gated_linear_unit(
    x: Float[Array, "*B M K"],
    weights: Float[Array, "K 2 N"],
    *,
    activation: Callable[[jax.Array], jax.Array],
    config: common.Config,
) -> Float[Array, "*B M N"]:
  """Gated Linear Unit implementation for SM100.

  Args:
    x: LHS operand.
    weights: RHS operand.
    activation: The activation function to use.
    config: Optional config. If `None`, a good default is chosen.

  Returns:
    The result of the GLU.
  """
  orig_x_shape = x.shape
  x = jax.lax.collapse(x, 0, -1)
  m, k = x.shape
  _, _, n = weights.shape
  dtype = x.dtype
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  if tile_m != 128:
    raise NotImplementedError("Only tile_m=128 is supported on SM100.")
  epi_tile_n = config.epi_tile_n or tile_n
  max_concurrent_steps = config.num_stages

  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )

  out_swizzle = plgpu.find_swizzle(epi_tile_n * jnp.dtype(dtype).itemsize * 8)
  out_swizzle_elems = out_swizzle // jnp.dtype(dtype).itemsize
  out_transforms = (
      plgpu.TilingTransform((8, out_swizzle_elems)),
      plgpu.SwizzleTransform(out_swizzle),
  )

  cluster_size = config.cluster_size_m * config.cluster_size_n
  collective = cluster_size > 1
  collective_axes = ("cluster",) if collective else None
  m_iters, n_iters, k_iters = m // tile_m, n // tile_n, k // tile_k

  def kernel(
      a_gmem,
      b_gmem,
      out_gmem,
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
    wg_idx = lax.axis_index("wg")
    cluster_idx = lax.axis_index("cluster")
    is_lead_block = cluster_idx == 0

    @plgpu.dynamic_scheduling_loop(grid_names=("mn_linear",), thread_axis="wg")
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
      elif config.cluster_size_n > 1:
        block_n_index = n_index * config.cluster_size_n + cluster_idx
      acc_slot = lax.rem(local_index, jnp.int32(ACC_NUM_SLOTS))

      slice_m = pl.ds(block_m_index * tile_m, tile_m)

      # Accumulator for X @ W0
      acc_tmem_slice_0 = acc0_tmem.at[:, pl.ds(acc_slot * tile_n, tile_n)]
      # Accumulator for X @ W1
      acc_tmem_slice_1 = acc1_tmem.at[:, pl.ds(acc_slot * tile_n, tile_n)]

      @pl.when(wg_idx == 0)
      def _compute_wg():
        @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
        def _per_warp():
          warp_id = lax.axis_index("warp")
          b_smems = (b0_smem, b1_smem)
          consumed_barriers = (consumed_barrier_0, consumed_barrier_1)

          @pl.when(warp_id == 0)
          def _tma_warp():
            def _loop_body(ki, _):
              slice_k = pl.ds(ki * tile_k, tile_k)
              slice_n = pl.ds(block_n_index * tile_n, tile_n)
              slot = lax.rem(ki, max_concurrent_steps)

              def _wait_barrier(barrier):
                @pl.when(
                    jnp.logical_or(ki >= max_concurrent_steps, local_index > 0)
                )
                def _():
                  plgpu.barrier_wait(barrier.at[slot])

              for i in range(2):
                _wait_barrier(consumed_barriers[i])
                plgpu.copy_gmem_to_smem(
                    b_gmem.at[slice_k, i, slice_n],
                    b_smems[i].at[slot],
                    ab_tma_barrier.at[slot],
                    leader_tracked=plgpu.CopyPartition.PARTITIONED(1)
                    if config.cluster_size_m > 1
                    else None,
                    collective_axes=collective_axes,
                )
              plgpu.copy_gmem_to_smem(
                  a_gmem.at[slice_m, slice_k],
                  a_smem.at[slot],
                  ab_tma_barrier.at[slot],
                  leader_tracked=plgpu.CopyPartition.PARTITIONED(1)
                  if config.cluster_size_n > 1
                  else None,
                  collective_axes="cluster" if collective else None,
              )

            lax.fori_loop(0, k_iters, _loop_body, None)

          @pl.when(jnp.logical_and(warp_id == 1, local_index > 1))
          def _wait_store():
            plgpu.barrier_wait(store_done_barrier.at[acc_slot])

          @pl.when(jnp.logical_and(warp_id == 1, is_lead_block))
          def _mma_warp():
            def _loop_body(ki, _):
              slot = lax.rem(ki, max_concurrent_steps)
              plgpu.barrier_wait(ab_tma_barrier.at[slot])
              # TODO: When tcgen05 supports reshapes with
              # transposes like wgmma we can avoid the dual
              # accumulators.
              #
              acc_tmem_slices = (acc_tmem_slice_0, acc_tmem_slice_1)
              for i in range(2):
                plgpu.tcgen05_mma(
                    acc_tmem_slices[i],
                    a_smem.at[slot],
                    b_smems[i].at[slot],
                    consumed_barriers[i].at[slot],
                    accumulate=(ki > 0),
                    collective_axis=collective_axes,
                )

              @pl.when(ki >= k_iters - 1)
              def _arrive():
                plgpu.tcgen05_commit_arrive(
                    mma_done_barrier.at[acc_slot],
                    collective_axis=collective_axes,
                )

            lax.fori_loop(0, k_iters, _loop_body, None)

      @pl.when(wg_idx == 1)
      def _store_wg():
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        plgpu.barrier_wait(mma_done_barrier.at[acc_slot])
        step_out_gmem = out_gmem.at[
            slice_m, pl.ds(block_n_index * tile_n, tile_n)
        ]
        for ni in range(tile_n // epi_tile_n):
          ni_col_slice = pl.ds(ni * epi_tile_n, epi_tile_n)
          val0 = plgpu.async_load_tmem(acc_tmem_slice_0.at[:, ni_col_slice])
          val1 = plgpu.async_load_tmem(acc_tmem_slice_1.at[:, ni_col_slice])
          plgpu.wait_load_tmem()
          val0 = val0.astype(dtype).astype(jnp.float32)
          val1 = val1.astype(dtype).astype(jnp.float32)
          acc_smem_ni = acc_smem.at[ni % ACC_NUM_SLOTS]
          acc_smem_ni[...] = (activation(val0) * val1).astype(dtype)
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(
              acc_smem_ni, step_out_gmem.at[:, ni_col_slice]
          )
          plgpu.wait_smem_to_gmem(2, wait_read_only=True)
        plgpu.barrier_arrive(store_done_barrier.at[acc_slot])

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
  smem_b_shape = (max_concurrent_steps, tile_k, tile_n)
  tmem_acc_shape = (tile_m, tile_n * 2)
  consumed_barrier_kwargs = dict(
      num_arrivals=1,
      num_barriers=max_concurrent_steps,
      orders_tensor_core=True,
  )
  kernel = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(m_iters * n_iters,),
      grid_names=("mn_linear",),
      cluster=(cluster_size,),
      cluster_names=("cluster",),
      num_threads=2,
      thread_name="wg",
      scratch_shapes=dict(
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
              (2, tile_m, epi_tile_n), dtype, transforms=out_transforms
          ),
          ab_tma_barrier=plgpu.Barrier(
              num_arrivals=3, num_barriers=max_concurrent_steps
          ),
          store_done_barrier=store_done_barrier,
          mma_done_barrier=plgpu.Barrier(**barrier_kwargs),
          consumed_barrier_0=plgpu.Barrier(**consumed_barrier_kwargs),
          consumed_barrier_1=plgpu.Barrier(**consumed_barrier_kwargs),
      ),
  )
  return jnp.reshape(kernel(x, weights), (*orig_x_shape[:-1], n))
