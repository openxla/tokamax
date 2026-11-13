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
"""Ragged dot Pallas-Mosaic-GPU Quantized Kernel (Blackwell)."""

from typing import Any
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array  # pylint: disable=g-multiple-import,g-importing-member
from jaxtyping import Float  # pylint: disable=g-multiple-import,g-importing-member
from jaxtyping import Integer  # pylint: disable=g-multiple-import,g-importing-member
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import memref
import qwix
from tokamax._src import jaxtyping
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common


# Warp in main WarpGroup
_MMA_WARP = 0
_W_TMA_WARP = 1
_X_TMA_WARP = 2

_TMEM = plgpu.Layout.TCGEN05_TMEM_NATIVE
_TCGEN05 = plgpu.Layout.TCGEN05
_TCGEN05_TRANSPOSED = plgpu.Layout.TCGEN05_TRANSPOSED
_TCGEN05_ROW = _TCGEN05.reduce(1)


def _make_barrier(
    *,
    num_arrivals: int = 1,
    num_barriers: int = 1,
    collective: bool = False,
    orders_tensor_core: bool = False,
    leader_tracked: bool = False,
):
  """Creates a Barrier or ClusterBarrier depending on `collective`."""
  if collective:
    return plgpu.ClusterBarrier(
        collective_axes=("x",),
        num_arrivals=num_arrivals,
        num_barriers=num_barriers,
        orders_tensor_core=orders_tensor_core,
        leader_tracked=leader_tracked,
    )
  return plgpu.Barrier(
      num_arrivals=num_arrivals,
      num_barriers=num_barriers,
      orders_tensor_core=orders_tensor_core,
  )


def dequant(s_ref, w):
  """Dequantize the array `w` using a 1D ref `s_ref`."""

  @plgpu.inline_mgpu(
      arg_types=(plgpu.RefType(), _TMEM),
      return_type=plgpu.ShapeDtypeStruct(w.shape, s_ref.dtype, _TMEM),
  )
  def scaled_w(_, s_smem, w):
    def scale(w_val, idx):
      assert s_smem.type.shape == [w.shape[0]]
      return arith.mulf(memref.load(s_smem, (idx[0],)), w_val)

    return w.foreach(scale, create_array=True)

  return scaled_w(s_ref, w.astype(s_ref.dtype))


@jaxtyping.jaxtyped
def ragged_dot_gpu_quant_blackwell_kernel(
    lhs: Float[Array, "M K"],
    rhs: Float[qwix.QArray, "G K N"],
    group_sizes: Integer[Array, "G"],
    out_dtype: jax.typing.DTypeLike,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
) -> Float[Array, "M N"]:
  """Pallas kernel for bf16 x int4 ragged dot (in-loop or post-MMA scale)."""
  assert rhs.zero_point is None

  block_m = config.block_m
  block_n = config.block_n
  block_k = config.block_k
  split_m = config.split_m
  num_stages = config.num_stages
  collective = config.collective
  post_scale = config.post_scale
  # `tile` is for each block
  tile_m = block_m
  tile_n = block_n
  if collective:
    block_m *= 2
    block_n *= 2

  w, w_scales, x = (rhs.qvalue.mT, rhs.scale, lhs)
  (num_groups, n, k_w), (m, k_x) = w.shape, x.shape
  tile_k = k_w // w_scales.shape[1]

  if tile_k < block_k:
    raise NotImplementedError(
        f"{tile_k=} must be greater than or equal to {block_k=}."
    )
  if k_w != k_x:
    raise ValueError(
        f"Contraction dim mismatch: weights.shape[1]={k_w}, x.shape[-1]={k_x}"
    )
  if group_sizes.shape != (num_groups,):
    raise ValueError(
        "Expected group_sizes to have shape"
        f" {(num_groups,)} but got {group_sizes.shape}"
    )
  if (x.dtype, w.dtype) != (jnp.bfloat16, jnp.int4):
    raise NotImplementedError(
        "Only mixed precision bfloat16 x int4 supported, got:"
        f" {x.dtype=} {w.dtype=}."
    )
  if post_scale:
    if split_m != 1:
      raise NotImplementedError("Only split_m = 1 is supported.")
  else:
    if w_scales.shape[2] * jnp.dtype(w_scales.dtype).itemsize < 16:
      raise NotImplementedError(
          "Scales array doesn't have stride of at least 16 bytes."
      )
    if block_m % split_m != 0:
      raise NotImplementedError(
          f"Expected block_m ({block_m}) to be divisible by split_m ({split_m})"
      )

  # Validate x and w swizzles before kernel compilation.
  num_stages = min(num_stages, k_w // block_k)
  x_smem_shape = (num_stages, tile_m, block_k)
  w_smem_shape = (num_stages, tile_n, block_k)
  x_smem = common.tiled_swizzled_smem(x_smem_shape, x.dtype, "x")
  w_smem = common.tiled_swizzled_smem(w_smem_shape, w.dtype, "w")

  m_iters = pl.cdiv(m, block_m) + num_groups - 1
  n_iters = pl.cdiv(n, block_n)
  align_tile = 8
  group_info = common.GroupInfo.create_aligned(
      group_sizes, block_m, m_iters, align_tile
  )

  main_wg = 0
  deq_wg = 1
  store_wg = 2

  def kernel(
      x_gmem,
      w_gmem,
      w_scales_gmem,
      group_id_gmem,
      start_within_block_gmem,
      actual_size_gmem,
      block_start_gmem,
      out_gmem,
      *,
      x_smem,
      w_smem,
      w_bf16_tmem,
      out_smem,
      acc_tmem,
      x_tma_barrier,
      w_tma_barrier,
      w_tma_consumed_barrier,
      w_bf16_barrier,
      w_consumed_tcgen05_barrier,
      x_consumed_tcgen05_barrier,
      mma_barrier,
      acc_consumed_barrier,
      ws_smem: Any = None,
  ):
    m, k = x_gmem.shape
    num_k_iters = pl.cdiv(k, block_k)
    cluster_idx = lax.axis_index("x")
    is_lead_block = cluster_idx == 0
    wg = jax.lax.axis_index("wg")

    @plgpu.nd_loop((m_iters * n_iters,), collective_axes=("sm",), init_carry=0)
    def mn_loop(loop_info: plgpu.NDLoopInfo, carry):
      (lin_idx,) = loop_info.index
      tid_m, ni = plgpu.planar_snake(
          lin_idx,
          (m_iters, n_iters),
          config.grid_minor_dim,
          config.grid_tile_width,
      )
      group_id = group_id_gmem[tid_m]
      start_within_block = start_within_block_gmem[tid_m]
      actual_size = actual_size_gmem[tid_m]
      block_start = block_start_gmem[tid_m]
      block_start = pl.multiple_of(block_start, align_tile)
      slice_m = pl.ds(block_start, block_m)
      slice_n = pl.ds(ni * block_n + cluster_idx * tile_n, tile_n)

      @pl.when(actual_size > 0)
      def _body():

        @mgpu_lib.warp_map_when(wg == main_wg)
        def _main_wg_warp(warp_id):

          @pl.when(warp_id == _W_TMA_WARP)
          def w_tma_warp():
            def _iter_w(ki, _):
              slot = lax.rem(ki, num_stages)

              @pl.when((ki >= num_stages) | (carry > 0))
              def _():
                plgpu.barrier_wait(w_tma_consumed_barrier.at[slot])

              plgpu.copy_gmem_to_smem(
                  w_gmem.at[group_id, slice_n, pl.ds(ki * block_k, block_k)],
                  w_smem.at[slot],
                  w_tma_barrier.at[slot],
              )
              if not post_scale:
                plgpu.copy_gmem_to_smem(
                    w_scales_gmem.at[
                        group_id,
                        lax.div((ki * block_k), tile_k),
                        slice_n,
                    ],
                    ws_smem.at[slot],
                    w_tma_barrier.at[slot],
                )

            lax.fori_loop(0, num_k_iters, _iter_w, None)

          @pl.when(warp_id == _X_TMA_WARP)
          def x_tma_warp():
            def _iter_x(ki, _):
              slot = lax.rem(ki, num_stages)

              @pl.when((ki >= num_stages) | (carry > 0))
              def _():
                plgpu.barrier_wait(x_consumed_tcgen05_barrier.at[slot])

              plgpu.copy_gmem_to_smem(
                  x_gmem.at[slice_m, pl.ds(ki * block_k, block_k)],
                  x_smem.at[slot],
                  x_tma_barrier.at[slot],
                  leader_tracked=plgpu.CopyPartition.PARTITIONED(0)
                  if collective
                  else None,
                  collective_axes="x" if collective else None,
              )

            lax.fori_loop(0, num_k_iters, _iter_x, None)

          @pl.when((warp_id == _MMA_WARP) & is_lead_block)
          def mma_warp():
            if not post_scale:

              @pl.when(carry > 0)
              def _():
                plgpu.barrier_wait(acc_consumed_barrier)

            def do_mma(ki, _):
              slot = lax.rem(ki, num_stages)
              if post_scale:

                @pl.when((ki >= num_stages) | (carry > 0))
                def _():
                  plgpu.barrier_wait(acc_consumed_barrier.at[slot])

              plgpu.barrier_wait(w_bf16_barrier.at[slot])
              plgpu.barrier_wait(x_tma_barrier.at[slot])
              acc_slice = (
                  acc_tmem.at[:, pl.ds(slot * block_m, block_m)]
                  if post_scale
                  else acc_tmem
              )
              plgpu.tcgen05_mma(
                  acc_slice,
                  w_bf16_tmem.at[:, pl.ds(slot * block_k, block_k)],
                  x_smem.at[slot].T,
                  x_consumed_tcgen05_barrier.at[slot],
                  accumulate=False if post_scale else (ki > 0),
                  collective_axis="x" if collective else None,
              )
              plgpu.tcgen05_commit_arrive(
                  w_consumed_tcgen05_barrier.at[slot],
                  collective_axis="x" if collective else None,
              )
              plgpu.tcgen05_commit_arrive(
                  mma_barrier.at[slot] if post_scale else mma_barrier,
                  collective_axis="x" if collective else None,
                  predicate=None if post_scale else (ki == num_k_iters - 1),
              )

            lax.fori_loop(0, num_k_iters, do_mma, None)

        @pl.when(wg == deq_wg)
        def _():
          def _deq(ki, _):
            slot = lax.rem(ki, num_stages)
            plgpu.barrier_wait(w_tma_barrier.at[slot])
            w = plgpu.load(w_smem.at[slot], layout=_TMEM(8), optimized=False)
            w_deq = plgpu.layout_cast(
                w.astype(x_smem.dtype if post_scale else ws_smem.dtype), _TMEM
            )
            if not post_scale:
              w_deq = dequant(ws_smem.at[slot], w_deq)

            @pl.when((ki >= num_stages) | (carry > 0))
            def _():
              plgpu.barrier_wait(w_consumed_tcgen05_barrier.at[slot])

            plgpu.async_store_tmem(
                w_bf16_tmem.at[:, pl.ds(slot * block_k, block_k)], w_deq
            )
            plgpu.commit_tmem()
            plgpu.barrier_arrive(w_tma_consumed_barrier.at[slot])
            plgpu.barrier_arrive(w_bf16_barrier.at[slot])

          lax.fori_loop(0, num_k_iters, _deq, None)

        @pl.when(wg == store_wg)
        def _():
          if post_scale:
            acc = plgpu.layout_cast(
                jnp.zeros((tile_n, block_m), dtype=jnp.float32),
                _TCGEN05,
            )

            def _loop_body(ki, acc):
              slot = lax.rem(ki, num_stages)
              scale = plgpu.load(
                  w_scales_gmem.at[
                      group_id,
                      lax.div((ki * block_k), tile_k),
                      slice_n,
                  ],
                  layout=_TCGEN05_ROW,
                  optimized=False,
              ).astype(acc.dtype)
              scale = plgpu.layout_cast(
                  lax.broadcast_in_dim(scale, acc.shape, [0]), _TCGEN05
              )
              plgpu.barrier_wait(mma_barrier.at[slot])
              local_acc = plgpu.async_load_tmem(
                  acc_tmem.at[:, pl.ds(slot * block_m, block_m)]
              )
              acc += local_acc * scale
              plgpu.wait_load_tmem()
              plgpu.barrier_arrive(acc_consumed_barrier.at[slot])
              return acc

            acc = lax.fori_loop(0, num_k_iters, _loop_body, acc)
            if activation is not None:
              acc = activation(acc)
            out_smem.T[...] = plgpu.layout_cast(
                acc.astype(out_smem.dtype), _TCGEN05_TRANSPOSED
            )
            plgpu.commit_smem()
            common.copy_ragged_smem_to_gmem(
                out_smem,
                out_gmem,
                block_start=block_start,
                start_within_block=start_within_block,
                actual_size=actual_size,
                ns=slice_n,
                max_rows=min(block_m, m),
            )
          else:
            plgpu.barrier_wait(mma_barrier)
            split_block_m = block_m // split_m

            def tmem_to_smem_loop_body(mi, _):
              acc = plgpu.async_load_tmem(
                  acc_tmem.at[:, pl.ds(mi * split_block_m, split_block_m)]
              )
              plgpu.wait_load_tmem()
              is_last_iter = mi == split_m - 1
              plgpu.barrier_arrive(acc_consumed_barrier, predicate=is_last_iter)
              if activation is not None:
                acc = activation(acc)
              out_smem.at[mi].T[...] = plgpu.layout_cast(
                  acc.astype(out_smem.dtype), _TCGEN05_TRANSPOSED
              )
              plgpu.commit_smem()

            lax.fori_loop(0, split_m, tmem_to_smem_loop_body, None)

            def store_loop_body(index, _):
              global_start = lax.max(start_within_block, index * split_block_m)
              global_end = lax.min(
                  start_within_block + actual_size,
                  (index + 1) * split_block_m,
              )

              @pl.when(global_end > global_start)
              def _():
                common.copy_ragged_smem_to_gmem(
                    out_smem.at[index],
                    out_gmem,
                    block_start=block_start + index * split_block_m,
                    start_within_block=global_start - index * split_block_m,
                    actual_size=global_end - global_start,
                    ns=slice_n,
                    max_rows=min(split_block_m, m),
                )

            lax.fori_loop(0, split_m, store_loop_body, None)

      return carry + (actual_size > 0)

  out_smem_shape = (
      (block_m, tile_n)
      if post_scale
      else (split_m, block_m // split_m, tile_n)
  )
  scratch_shapes = dict(
      x_smem=x_smem,
      w_smem=w_smem,
      w_bf16_tmem=plgpu.TMEM(
          (tile_n, num_stages * block_k),
          dtype=w_scales.dtype,
          packed=True,
          collective=collective,
      ),
      out_smem=common.tiled_swizzled_smem(
          out_smem_shape, out_dtype, "out", swizzle=128, tiling_prefix=(1,)
      ),
      acc_tmem=plgpu.TMEM(
          (tile_n, (num_stages if post_scale else 1) * block_m),
          dtype=jnp.float32,
          collective=collective,
      ),
      x_tma_barrier=plgpu.Barrier(num_barriers=num_stages),
      w_tma_barrier=plgpu.Barrier(
          num_arrivals=1 if post_scale else 2, num_barriers=num_stages
      ),
      w_tma_consumed_barrier=plgpu.Barrier(num_barriers=num_stages),
      w_bf16_barrier=_make_barrier(
          num_barriers=num_stages,
          collective=collective,
          orders_tensor_core=True,
          leader_tracked=True,
      ),
      w_consumed_tcgen05_barrier=plgpu.Barrier(
          num_barriers=num_stages, orders_tensor_core=True
      ),
      x_consumed_tcgen05_barrier=plgpu.Barrier(
          num_barriers=num_stages, orders_tensor_core=True
      ),
      mma_barrier=plgpu.Barrier(
          num_barriers=num_stages if post_scale else 1, orders_tensor_core=True
      ),
      acc_consumed_barrier=_make_barrier(
          num_barriers=num_stages if post_scale else 1,
          collective=collective,
          orders_tensor_core=True,
          leader_tracked=True,
      ),
  )
  if not post_scale:
    scratch_shapes["ws_smem"] = plgpu.SMEM(
        (num_stages, tile_n), dtype=w_scales.dtype
    )

  num_sms = backend.get_default_device().core_count
  profile = False
  if profile:
    num_sms = 1 + collective

  f = plgpu.kernel(
      kernel,
      out_type=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
      scratch_types=scratch_shapes,
      num_threads=3,
      thread_name="wg",
      grid=(num_sms // (1 + collective),),
      grid_names=("sm",),
      cluster=(1 + collective,),
      cluster_names=("x",),
      kernel_name=(
          "ragged_dot_quant_post_scale_sm100"
          if post_scale
          else "ragged_dot_quant_sm100"
      ),
      compiler_params=plgpu.CompilerParams(
          approx_math=True,
          unsafe_no_auto_barriers=True,
          profile_space=(160 if post_scale else 80) if profile else 0,
          profile_dir="sponge" if profile else "",
          lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
      ),
  )
  return f(
      x,
      w,
      w_scales,
      group_info.group_id,
      group_info.start_within_block,
      group_info.actual_size,
      group_info.block_start,
  )
