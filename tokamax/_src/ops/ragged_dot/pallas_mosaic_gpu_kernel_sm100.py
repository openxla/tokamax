# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Ragged dot Pallas-Mosaic-GPU Non-Quantized Kernel (Blackwell)."""

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common

_COMPUTE_WG = 0
_MMA_WARP = 0
_TMA_WARP = 1
_EPILOGUE_WG = 1

_TCGEN05 = plgpu.Layout.TCGEN05
_TCGEN05_TRANSPOSED = plgpu.Layout.TCGEN05_TRANSPOSED


@jaxtyping.jaxtyped
def ragged_dot_gpu_non_quant_blackwell_kernel(
    lhs: Float[Array, "M K"],
    rhs: Float[Array, "G K N"],
    group_sizes: Integer[Array, "G"],
    out_dtype: jnp.dtype,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
) -> Float[Array, "M N"]:
  """Pallas kernel for ragged dot with GPU quantization."""
  common.check_bf16xbf16_or_f16xf16(lhs, rhs)

  block_m = config.block_m
  block_n = config.block_n
  block_k = config.block_k
  num_stages = config.num_stages
  collective = config.collective
  cluster_block_m = (block_m * 2) if collective else block_m
  cluster_block_n = (block_n * 2) if collective else block_n

  w, x = (rhs.mT, lhs)

  (num_groups, n, k_w), (m, k_x) = w.shape, x.shape
  if k_w != k_x:
    raise ValueError(
        f"Contraction dim mismatch: weights.shape[1]={k_w}, x.shape[-1]={k_x}"
    )
  if group_sizes.shape != (num_groups,):
    raise ValueError(
        "Expected group_sizes to have shape"
        f" {(num_groups,)} but got {group_sizes.shape}"
    )

  # num_stages must be less than or equal to the number of blocks
  num_stages = min(num_stages, k_w // block_k)

  group_info = common.GroupInfo.create_aligned(
      group_sizes, cluster_block_m, pl.cdiv(m, cluster_block_m) + num_groups - 1
  )
  m_iters = pl.cdiv(m, cluster_block_m) + num_groups - 1
  n_iters = pl.cdiv(n, cluster_block_n)

  def kernel(
      x_gmem,
      w_gmem,
      _,
      group_id_gmem,
      start_within_block_gmem,
      actual_size_gmem,
      block_start_gmem,
      out_gmem,
      x_smem,
      w_smem,
      acc_smem,
      acc_tmem,
      xw_barrier,
      xw_consumed_barrier,
      acc_barrier,
      acc_consumed_barrier,
  ):
    m, k = x_gmem.shape
    ub = pl.cdiv(k, block_k)
    cluster_idx = lax.axis_index("cluster")

    @plgpu.nd_loop(
        (m_iters * n_iters,), collective_axes=("cluster_grid",), init_carry=0
    )
    def mn_loop(loop_info: plgpu.NDLoopInfo, carry):
      (idx,) = loop_info.index
      tid_m, ni = plgpu.planar_snake(
          idx, (m_iters, n_iters), config.grid_minor_dim, config.grid_tile_width
      )

      wg = jax.lax.axis_index("wg")
      group_id = group_id_gmem[tid_m]
      start_within_block = start_within_block_gmem[tid_m]
      actual_size = actual_size_gmem[tid_m]
      block_start = block_start_gmem[tid_m]
      ms = pl.ds(block_start, cluster_block_m)
      ns = pl.ds(ni * cluster_block_n, cluster_block_n)

      @pl.when(actual_size > 0)
      def compute():
        @pl.when(wg == _COMPUTE_WG)
        def compute_wg():
          @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
          def compute_warps():
            warp_id = lax.axis_index("warp")
            cluster_axis = "cluster" if collective else None

            @pl.when(warp_id == _TMA_WARP)
            def tma_warp():
              @pl.loop(0, ub)
              def tma_loop(ki):
                ks = pl.ds(ki * block_k, block_k)
                si = lax.rem(ki, num_stages)

                @pl.when((ki >= num_stages) | (carry > 0))
                def _():
                  plgpu.barrier_wait(xw_consumed_barrier.at[si])

                plgpu.copy_gmem_to_smem(
                    x_gmem.at[ms, ks],
                    x_smem.at[si],
                    xw_barrier.at[si],
                    partitioned_axis=0 if collective else None,
                    collective_axes=cluster_axis,
                )
                plgpu.copy_gmem_to_smem(
                    w_gmem.at[group_id, ns, ks],
                    w_smem.at[si],
                    xw_barrier.at[si],
                    partitioned_axis=0 if collective else None,
                    collective_axes=cluster_axis,
                )

            @pl.when((warp_id == _MMA_WARP) & (cluster_idx == 0))
            def mma_warp():
              si_acc = lax.rem(carry, jnp.int32(2))
              ms_acc = pl.ds(si_acc * cluster_block_m, cluster_block_m)

              @pl.when(carry > 1)
              def _():
                with jax.named_scope("wait for store"):
                  plgpu.barrier_wait(acc_consumed_barrier.at[si_acc])

              @pl.loop(0, ub)
              def mma_loop_body(ki):
                si_tma = lax.rem(ki, num_stages)
                with jax.named_scope("wait for xw"):
                  plgpu.barrier_wait(xw_barrier.at[si_tma])
                with jax.named_scope("issuing mma"):
                  plgpu.tcgen05_mma(
                      acc_tmem.at[:, ms_acc],
                      w_smem.at[si_tma],
                      x_smem.at[si_tma].T,
                      xw_consumed_barrier.at[si_tma],
                      accumulate=(ki > 0),
                      collective_axis=cluster_axis,
                  )

              plgpu.tcgen05_commit_arrive(
                  acc_barrier.at[si_acc], collective_axis=cluster_axis
              )

        @pl.when(wg == _EPILOGUE_WG)
        def epilogue_wg():
          si = lax.rem(carry, jnp.int32(2))
          plgpu.barrier_wait(acc_barrier.at[si])
          with jax.named_scope("tmem -> smem"):
            ms_acc = pl.ds(si * cluster_block_m, cluster_block_m)
            acc = plgpu.async_load_tmem(acc_tmem.at[:, ms_acc], layout=_TCGEN05)
            if activation is not None:
              acc = activation(acc)
            acc = acc.astype(acc_smem.dtype)
            acc_smem.T[...] = plgpu.layout_cast(acc, _TCGEN05_TRANSPOSED)
            plgpu.wait_load_tmem()
            plgpu.barrier_arrive(acc_consumed_barrier.at[si])
            plgpu.commit_smem()

          with jax.named_scope("smem -> gmem"):
            # Write out the largest power of two rows first,
            # then the next largest, etc.
            # This allows us to coalesce writes as much as possible.
            offset = start_within_block
            size = 1 << (min(cluster_block_m, m).bit_length() - 1)
            while size > 0:
              ns = pl.ds(ni * cluster_block_n + cluster_idx * block_n, block_n)

              @pl.when(actual_size & size != 0)
              def _():
                ms = pl.ds(block_start + offset, size)
                out_smem_slice = acc_smem.at[pl.ds(offset, size)]
                plgpu.copy_smem_to_gmem(
                    out_smem_slice, out_gmem.at[ms, ns], commit_group=False
                )

              offset += actual_size & size
              size //= 2
            plgpu.commit_smem_to_gmem_group()
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)

      return carry + (actual_size > 0)

  def tiled_smem(shape, dtype, what=""):
    transforms = common.tile_swizzle_transforms(shape, dtype, what)
    return plgpu.SMEM(shape, dtype, transforms=transforms)

  acc_smem = plgpu.SMEM(
      (cluster_block_m, block_n),
      dtype=out_dtype,
      # workaround for ValueError: Dynamic slice base index (which is a
      # dynamic value) cannot be statically proven to be divisible by
      # the tiling (8)
      transforms=(
          plgpu.TilingTransform((1, 128 // jnp.dtype(out_dtype).itemsize)),
          plgpu.SwizzleTransform(128),
      ),
  )

  if collective:
    acc_consumed_barrier = plgpu.ClusterBarrier(
        collective_axes=("cluster",), num_barriers=2, orders_tensor_core=True
    )
  else:
    acc_consumed_barrier = plgpu.Barrier(
        num_barriers=2, orders_tensor_core=True
    )

  scratch_shapes = dict(
      x_smem=tiled_smem((num_stages, block_m, block_k), x.dtype, "x"),
      w_smem=tiled_smem((num_stages, block_n, block_k), w.dtype, "w"),
      acc_tmem=plgpu.TMEM(
          (block_n, cluster_block_m * 2), jnp.float32, collective=collective
      ),
      acc_smem=acc_smem,
      xw_barrier=plgpu.Barrier(num_arrivals=2, num_barriers=num_stages),
      xw_consumed_barrier=plgpu.Barrier(
          num_barriers=num_stages, orders_tensor_core=True
      ),
      acc_barrier=plgpu.Barrier(num_barriers=2, orders_tensor_core=True),
      acc_consumed_barrier=acc_consumed_barrier,
  )

  num_sms = backend.get_default_device().core_count
  profile = False
  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
      scratch_shapes=scratch_shapes,
      num_threads=2,
      thread_name="wg",
      grid=(num_sms // 2,) if collective else (num_sms,),
      grid_names=("cluster_grid",),
      cluster=(1 + collective,),
      cluster_names=("cluster",),
      compiler_params=plgpu.CompilerParams(
          approx_math=True,
          unsafe_no_auto_barriers=True,
          profile_space=30 if profile else 0,
          profile_dir="sponge" if profile else "",
      ),
  )
  return f(
      x,
      w,
      group_info.block,
      group_info.group_id,
      group_info.start_within_block,
      group_info.actual_size,
      group_info.block_start,
  )
