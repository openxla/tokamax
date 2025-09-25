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

import functools
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import memref
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common

QuantizedArray = quantization.QuantizedArray

_MMA_WG = 0
_MEMORY_WG = 1
_MMA_WARP = 0
_STORE_WARP = 1
_TMA_WARP = 4
_TMEM = plgpu.Layout.TCGEN05_TMEM_NATIVE


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


def ragged_dot_gpu_quant_blackwell_kernel(
    lhs: jax.Array,
    rhs: QuantizedArray,
    group_sizes: jax.Array,
    out_dtype,
    config: common.Config,
) -> jax.Array:
  """Pallas kernel for ragged dot with GPU quantization."""
  block_m = config.block_m
  block_n = config.block_n
  block_k = config.block_k
  num_stages = config.num_stages
  collective = config.collective
  grid_block_n = config.grid_block_n
  persistent = config.persistent
  # `tile` is for each block
  tile_m = block_m
  tile_n = block_n
  if collective:
    block_m *= 2
    block_n *= 2

  w, w_scales, x = (
      rhs.values.transpose(0, 2, 1),
      rhs.scales,
      lhs,
  )
  (num_groups, n, k_w), (m, k_x) = w.shape, x.shape
  tile_k = k_w // w_scales.shape[1]
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
    raise ValueError(
        "Only mixed precision bfloat16 x int4 supported, got:"
        f" {x.dtype=} {w.dtype=}."
    )
  swizzle = plgpu.find_swizzle(block_k * jnp.dtype(x.dtype).itemsize * 8)

  swizzle_elems = swizzle // jnp.dtype(x.dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )

  w_elem_bits = 4
  w_swizzle = plgpu.find_swizzle(block_k * w_elem_bits)  # n,k
  w_swizzle_elems = (w_swizzle * 8) // w_elem_bits
  # num_stages must be less than or equal to the number of blocks
  num_stages = min(num_stages, k_w // block_k)

  group_info = common.GroupInfo.create(
      group_sizes, block_m, pl.cdiv(m, block_m) + num_groups - 1
  )

  def kernel(*refs, scoped):
    (
        x_gmem,
        w_gmem,
        w_scales_gmem,
        block_gmem,
        group_id_gmem,
        start_within_block_gmem,
        actual_size_gmem,
        block_start_gmem,
        out_gmem,
    ) = refs
    (
        scratch_buffers,
        barriers,
    ) = scoped
    (x_smem, w_smem, w_bf16_tmem, w_scales_smem, out_smem, acc_tmem) = (
        scratch_buffers
    )
    (
        x_tma_barrier,
        w_tma_barrier,
        w_scales_tma_barrier,
        w_bf16_barrier,
        tcgen05_barrier,
        mma_done_barrier,
        cluster_barrier,
    ) = barriers

    m, k = x_gmem.shape
    num_k_iters = pl.cdiv(k, block_k)

    # TODO: use emit_pipeline_warp_specialized, improve it if needed.
    def mn_loop(m_offset, idx, carry):
      if collective:
        block_ni, tid_m, remainder_ni, cluster_idx = idx
      else:
        block_ni, tid_m, remainder_ni = idx
      tid_m += m_offset
      ni = block_ni * pl.cdiv(n, block_n * grid_block_n) + remainder_ni
      mi = block_gmem[tid_m]
      group_id = group_id_gmem[tid_m]
      start_within_block = start_within_block_gmem[tid_m]
      actual_size = actual_size_gmem[tid_m]
      block_start = block_start_gmem[tid_m]
      wg = jax.lax.axis_index("wg")

      is_lead_block = cluster_idx == 0

      def do_tma_w(ki, slot):
        plgpu.copy_gmem_to_smem(  # e,n,k
            w_gmem.at[
                group_id,
                pl.Slice(ni * block_n + cluster_idx * tile_n, tile_n),
                pl.Slice(ki * block_k, block_k),
            ],
            w_smem.at[slot],
            w_tma_barrier.at[slot],
        )
        plgpu.copy_gmem_to_smem(  # e,k//t,n
            w_scales_gmem.at[
                group_id,
                jax.lax.div((ki * block_k), tile_k),
                pl.Slice(ni * block_n + cluster_idx * tile_n, tile_n),
            ],
            w_scales_smem.at[slot],
            w_scales_tma_barrier.at[slot],
        )

      def do_cluster_schedule():
        plgpu.barrier_arrive(cluster_barrier)
        plgpu.barrier_wait(cluster_barrier)

      @pl.when(actual_size > 0)
      def _body():
        def _deq(ki, _):
          slot = lax.rem(ki, num_stages)

          @pl.when((ki >= num_stages) | (carry > 0))
          def _():
            # Wait for the previous mma to complete.
            plgpu.barrier_wait(tcgen05_barrier.at[slot])

          # load x
          @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
          def _per_warp():
            warp_id = lax.axis_index("warp")

            @pl.when(warp_id == _TMA_WARP)
            def _():
              plgpu.copy_gmem_to_smem(  # m,k
                  x_gmem.at[
                      pl.Slice(mi * block_m, block_m),
                      pl.Slice(ki * block_k, block_k),
                  ],
                  x_smem.at[slot],
                  x_tma_barrier.at[slot],
                  partitioned_axis=0 if collective else None,
                  collective_axes="x" if collective else None,
              )

          # dequant w
          plgpu.barrier_wait(w_tma_barrier.at[slot])
          w = plgpu.load(
              w_smem.at[slot],
              (),
              layout=_TMEM(8),
              optimized=False,
          )
          w = w.astype(w_scales_smem.dtype)
          w = plgpu.layout_cast(w, _TMEM)
          plgpu.barrier_wait(w_scales_tma_barrier.at[slot])
          w_deq = dequant(w_scales_smem.at[slot], w)
          plgpu.async_store_tmem(
              w_bf16_tmem.at[:, pl.ds(slot * block_k, block_k)], w_deq
          )
          plgpu.commit_tmem()
          if collective:
            do_cluster_schedule()
          plgpu.barrier_arrive(w_bf16_barrier.at[slot])
          fetch_step = ki + num_stages
          fetch_slot = lax.rem(fetch_step, num_stages)
          lax.cond(
              fetch_step < num_k_iters,
              lambda: do_tma_w(fetch_step, fetch_slot),
              lambda: None,
          )

        def _mma(ki, _):
          slot = lax.rem(ki, num_stages)
          is_last_iter = ki >= num_k_iters - 1
          plgpu.barrier_wait(w_bf16_barrier.at[slot])

          @pl.when(is_lead_block)
          def _():
            plgpu.barrier_wait(x_tma_barrier.at[slot])
            plgpu.tcgen05_mma(
                acc_tmem,
                w_bf16_tmem.at[:, pl.ds(slot * block_k, block_k)],
                plgpu.transpose_ref(x_smem.at[slot], (1, 0)),
                tcgen05_barrier.at[slot],
                accumulate=(ki > 0),
                collective_axis="x" if collective else None,
            )

            @pl.when(is_last_iter)
            def _():
              plgpu.tcgen05_commit_arrive(
                  mma_done_barrier,
                  collective_axis="x" if collective else None,
              )

        @pl.when(wg == _MEMORY_WG)
        def _():
          # prologue
          for ki in range(num_stages):
            slot = jax.lax.rem(ki, num_stages)
            do_tma_w(ki, slot)
          lax.fori_loop(0, num_k_iters, _deq, None)

        @pl.when(wg == _MMA_WG)
        def _():
          @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
          def _per_warp():
            warp_id = lax.axis_index("warp")

            @pl.when(warp_id == _MMA_WARP)
            def _():
              lax.fori_loop(0, num_k_iters, _mma, None)

          plgpu.barrier_wait(mma_done_barrier)
          out_smem_t = plgpu.transpose_ref(out_smem, (1, 0))
          acc = plgpu.async_load_tmem(acc_tmem)
          plgpu.wait_load_tmem()
          acc = acc.astype(out_smem_t.dtype)
          out_smem_t[...] = plgpu.layout_cast(
              acc, plgpu.Layout.TCGEN05_TRANSPOSED
          )
          plgpu.commit_smem()
          del out_smem_t

          @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
          def _():
            warp_id = lax.axis_index("warp")

            @pl.when(warp_id == _STORE_WARP)
            def _():
              # Write out the largest power of two rows first,
              # then the next largest, etc.
              # This allows us to coalesce writes as much as possible.
              offset = start_within_block
              size = 1 << (min(block_m, m).bit_length() - 1)
              while size > 0:

                @pl.when(actual_size & size != 0)
                def _():
                  out_smem_slice = out_smem.at[pl.ds(offset, size)]
                  o_gref_slice = out_gmem.at[
                      pl.ds(block_start + offset, size),
                      pl.ds(ni * block_n + cluster_idx * tile_n, tile_n),
                  ]
                  plgpu.copy_smem_to_gmem(out_smem_slice, o_gref_slice)

                offset += actual_size & size
                size //= 2
              plgpu.wait_smem_to_gmem(0)

      return carry + (actual_size > 0)

    if collective:
      collective_axes = ("sm", "x")
    else:
      collective_axes = "sm"

    if persistent:
      grid = (
          grid_block_n,
          pl.cdiv(m, block_m),
          pl.cdiv(n, grid_block_n * block_n),
      )
      if collective:
        grid = grid + (2,)
      carry = plgpu.nd_loop(
          grid, collective_axes=collective_axes, init_carry=0
      )(functools.partial(mn_loop, 0))

      grid = (
          grid_block_n,
          num_groups - 1,
          pl.cdiv(n, grid_block_n * block_n),
      )
      if collective:
        grid = grid + (2,)
      plgpu.nd_loop(grid, collective_axes=collective_axes, init_carry=carry)(
          functools.partial(mn_loop, pl.cdiv(m, block_m))
      )
    else:
      grid = (
          grid_block_n,
          pl.cdiv(m, block_m) + num_groups - 1,
          pl.cdiv(n, grid_block_n * block_n),
      )
      if collective:
        grid = grid + (2,)
      plgpu.nd_loop(grid, collective_axes=collective_axes, init_carry=0)(
          functools.partial(mn_loop, 0)
      )

  def kernel_entry(*refs):
    x_smem = plgpu.SMEM(
        (num_stages, tile_m, block_k),
        dtype=x.dtype,
        transforms=transforms,
    )
    out_smem = plgpu.SMEM(
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
    w_smem = plgpu.SMEM(
        (num_stages, tile_n, block_k),
        dtype=w.dtype,
        transforms=(
            plgpu.TilingTransform((8, w_swizzle_elems)),
            plgpu.SwizzleTransform(w_swizzle),
        ),
    )
    w_bf16_tmem = plgpu.TMEM(
        (tile_n, num_stages * block_k),
        dtype=w_scales.dtype,
        packed=True,
        collective=collective,
    )
    ws_smem = plgpu.SMEM(
        (num_stages, tile_n),
        dtype=w_scales.dtype,
    )
    acc_tmem = plgpu.TMEM(
        (tile_n, block_m), dtype=jnp.float32, collective=collective
    )
    x_tma_barrier = plgpu.Barrier(num_barriers=num_stages)
    w_tma_barrier = plgpu.Barrier(num_barriers=num_stages)
    w_scales_tma_barrier = plgpu.Barrier(num_barriers=num_stages)
    w_bf16_barrier = plgpu.Barrier(num_barriers=num_stages)
    tcgen05_barrier = plgpu.Barrier(
        num_barriers=num_stages, orders_tensor_core=True
    )
    mma_done_barrier = plgpu.Barrier(orders_tensor_core=True)
    cluster_barrier = (
        plgpu.ClusterBarrier(collective_axes=("x",)) if collective else None
    )
    pl.run_scoped(
        lambda *args: kernel(*refs, scoped=args),
        (
            x_smem,
            w_smem,
            w_bf16_tmem,
            ws_smem,
            out_smem,
            acc_tmem,
        ),
        (
            x_tma_barrier,
            w_tma_barrier,
            w_scales_tma_barrier,
            w_bf16_barrier,
            tcgen05_barrier,
            mma_done_barrier,
            cluster_barrier,
        ),
        collective_axes="wg",
    )

  num_sms = backend.get_default_device().core_count
  profile = False
  f = plgpu.kernel(
      kernel_entry,
      out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
      num_threads=2,
      thread_name="wg",
      grid=(num_sms // 2,) if collective else (num_sms,),
      grid_names=("sm",),
      cluster=(2,) if collective else (),
      cluster_names=("x",) if collective else (),
      compiler_params=plgpu.CompilerParams(
          approx_math=True,
          unsafe_no_auto_barriers=True,
          profile_space=200 if profile else 0,
          profile_dir="sponge" if profile else "",
      ),
  )
  return f(
      x,
      w,
      w_scales,
      group_info.block,
      group_info.group_id,
      group_info.start_within_block,
      group_info.actual_size,
      group_info.block_start,
  )
