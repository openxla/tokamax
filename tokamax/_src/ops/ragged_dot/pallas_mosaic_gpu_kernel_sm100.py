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
_STORE_WG = 1


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
  block_m = config.block_m
  block_n = config.block_n
  block_k = config.block_k
  num_stages = config.num_stages
  collective = config.collective
  # `tile` is for each block
  tile_m = block_m
  tile_n = block_n
  if collective:
    block_m *= 2
    block_n *= 2

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
  if (x.dtype, w.dtype) != (jnp.bfloat16, jnp.bfloat16):
    raise ValueError(
        "Only the same precision bfloat16 x bfloat16 supported, got:"
        f" {x.dtype=} {w.dtype=}."
    )

  # num_stages must be less than or equal to the number of blocks
  num_stages = min(num_stages, k_w // block_k)

  group_info = common.GroupInfo.create_aligned(
      group_sizes, block_m, pl.cdiv(m, block_m) + num_groups - 1
  )
  m_iters = pl.cdiv(m, block_m) + num_groups - 1
  n_iters = pl.cdiv(n, block_n)

  def kernel(*refs, scoped):
    (
        x_gmem,
        w_gmem,
        _,
        group_id_gmem,
        start_within_block_gmem,
        actual_size_gmem,
        block_start_gmem,
        out_gmem,
    ) = refs
    scratch_buffers, barriers = scoped
    x_smem, w_smem, acc_smem, acc_tmem = scratch_buffers
    (
        xw_tma_barrier,
        consumed_barrier,
        mma_done_barrier,
        store_done_barrier,
    ) = barriers

    m, k = x_gmem.shape
    num_k_iters = pl.cdiv(k, block_k)
    cluster_idx = lax.axis_index("x")

    @plgpu.nd_loop((m_iters * n_iters,), collective_axes=("sm",), init_carry=0)
    def mn_loop(loop_info: plgpu.NDLoopInfo, carry):
      (lin_idx,) = loop_info.index
      tid_m, ni = plgpu.planar_snake(
          lin_idx,
          (m_iters, n_iters),
          config.grid_minor_dim,
          config.grid_tile_width,
      )

      wg = jax.lax.axis_index("wg")
      group_id = group_id_gmem[tid_m]
      start_within_block = start_within_block_gmem[tid_m]
      actual_size = actual_size_gmem[tid_m]
      block_start = block_start_gmem[tid_m]
      acc_slot = lax.rem(carry, jnp.int32(2))
      slice_m = pl.ds(block_start, block_m)
      slice_n = pl.ds(ni * block_n, block_n)
      slice_acc_m = pl.ds(acc_slot * block_m, block_m)

      is_lead_block = cluster_idx == 0

      @pl.when(actual_size > 0)
      def _body():

        @pl.when(wg == _COMPUTE_WG)
        def _():
          @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
          def _per_warp():
            warp_id = lax.axis_index("warp")

            @pl.when(warp_id == _TMA_WARP)
            def _memory():
              def _loop_body(ki, _):
                slice_k = pl.ds(ki * block_k, block_k)
                slot = lax.rem(ki, num_stages)

                @pl.when((ki >= num_stages) | (carry > 0))
                def _():
                  plgpu.barrier_wait(consumed_barrier.at[slot])

                plgpu.copy_gmem_to_smem(
                    x_gmem.at[slice_m, slice_k],
                    x_smem.at[slot],
                    xw_tma_barrier.at[slot],
                    partitioned_axis=0 if collective else None,
                    collective_axes="x" if collective else None,
                )
                plgpu.copy_gmem_to_smem(
                    w_gmem.at[group_id, slice_n, slice_k],
                    w_smem.at[slot],
                    xw_tma_barrier.at[slot],
                    partitioned_axis=0 if collective else None,
                    collective_axes="x" if collective else None,
                )

              lax.fori_loop(0, num_k_iters, _loop_body, None)

            @pl.when((warp_id == _MMA_WARP) & (carry > 1))
            def _wait_store():
              with jax.named_scope("wait for store"):
                plgpu.barrier_wait(store_done_barrier.at[acc_slot])

            @pl.when((warp_id == _MMA_WARP) & is_lead_block)
            def _mma():
              def _loop_body(ki, _):
                slot = lax.rem(ki, num_stages)
                with jax.named_scope("wait for xw"):
                  plgpu.barrier_wait(xw_tma_barrier.at[slot])
                with jax.named_scope("issuing mma"):
                  plgpu.tcgen05_mma(
                      acc_tmem.at[:, slice_acc_m],
                      w_smem.at[slot],
                      x_smem.at[slot].T,
                      consumed_barrier.at[slot],
                      accumulate=(ki > 0),
                      collective_axis="x" if collective else None,
                  )

                @pl.when(ki >= num_k_iters - 1)
                def _():
                  plgpu.tcgen05_commit_arrive(
                      mma_done_barrier.at[acc_slot],
                      collective_axis="x" if collective else None,
                  )

              lax.fori_loop(0, num_k_iters, _loop_body, None)

        @pl.when(wg == _STORE_WG)
        def _():
          plgpu.wait_smem_to_gmem(0, wait_read_only=True)
          plgpu.barrier_wait(mma_done_barrier.at[acc_slot])
          with jax.named_scope("tmem -> smem"):
            acc_smem_t = plgpu.transpose_ref(acc_smem, (1, 0))
            acc = plgpu.async_load_tmem(acc_tmem.at[:, slice_acc_m])
            plgpu.wait_load_tmem()
            if activation is not None:
              acc = activation(acc)
            acc = acc.astype(acc_smem_t.dtype)
            acc_smem_t[...] = plgpu.layout_cast(
                acc, plgpu.Layout.TCGEN05_TRANSPOSED
            )
            plgpu.commit_smem()
            del acc_smem_t

          with jax.named_scope("smem -> gmem"):
            # Write out the largest power of two rows first,
            # then the next largest, etc.
            # This allows us to coalesce writes as much as possible.
            offset = start_within_block
            size = 1 << (min(block_m, m).bit_length() - 1)
            while size > 0:

              @pl.when(actual_size & size != 0)
              def _():
                out_smem_slice = acc_smem.at[pl.ds(offset, size)]
                o_gref_slice = out_gmem.at[
                    pl.ds(block_start + offset, size),
                    pl.ds(ni * block_n + cluster_idx * tile_n, tile_n),
                ]
                plgpu.copy_smem_to_gmem(out_smem_slice, o_gref_slice)

              offset += actual_size & size
              size //= 2
            plgpu.wait_smem_to_gmem(0)
            plgpu.barrier_arrive(store_done_barrier.at[acc_slot])

      return carry + (actual_size > 0)

  swizzle = plgpu.find_swizzle(block_k * jnp.dtype(x.dtype).itemsize * 8)

  swizzle_elems = swizzle // jnp.dtype(x.dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )

  def kernel_entry(*refs):
    x_smem = plgpu.SMEM(
        (num_stages, tile_m, block_k),
        dtype=x.dtype,
        transforms=transforms,
    )
    w_smem = plgpu.SMEM(
        (num_stages, tile_n, block_k),
        dtype=w.dtype,
        transforms=transforms,
    )
    acc_tmem = plgpu.TMEM(
        (tile_n, block_m * 2), dtype=jnp.float32, collective=collective
    )
    acc_smem = plgpu.SMEM(
        (block_m, tile_n),
        dtype=out_dtype,
        # workaround for ValueError: Dynamic slice base index (which is a
        # dynamic value) cannot be statically proven to be divisible by
        # the tiling (8)
        transforms=(
            plgpu.TilingTransform((1, 128 // jnp.dtype(out_dtype).itemsize)),
            plgpu.SwizzleTransform(128),
        ),
    )
    xw_tma_barrier = plgpu.Barrier(num_arrivals=2, num_barriers=num_stages)
    consumed_barrier = plgpu.Barrier(
        num_barriers=num_stages, orders_tensor_core=True
    )
    mma_done_barrier = plgpu.Barrier(num_barriers=2, orders_tensor_core=True)
    if collective:
      store_done_barrier = plgpu.ClusterBarrier(
          collective_axes=("x",),
          num_barriers=2,
          orders_tensor_core=True,
      )
    else:
      store_done_barrier = plgpu.Barrier(
          num_barriers=2, orders_tensor_core=True
      )
    pl.run_scoped(
        lambda *args: kernel(*refs, scoped=args),
        (x_smem, w_smem, acc_smem, acc_tmem),
        (
            xw_tma_barrier,
            consumed_barrier,
            mma_done_barrier,
            store_done_barrier,
        ),
        collective_axes="wg",
    )

  num_sms = backend.get_default_device().core_count
  profile = False
  f = plgpu.kernel(
      kernel_entry,
      out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
      num_threads=2,
      thread_name="wg",
      grid=(num_sms // 2,) if collective else (num_sms,),
      grid_names=("sm",),
      cluster=(1 + collective,),
      cluster_names=("x",),
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
