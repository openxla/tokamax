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

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer  # pylint: disable=g-multiple-import,g-importing-member
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
import numpy as np
import qwix
from tokamax._src import jaxtyping
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common


# Dequant WarpGroup
_DEQ_WG = 2
_DEQ_REG = 128
_DEQ_REG_ACTION = "increase"
# MMA + TMA
_MAIN_WG = _DEQ_WG
_MAIN_REG = 104
_MAIN_REG_ACTION = "decrease"
# Scale ACC and Store
_STORE_WG = _MAIN_WG + 1
_STORE_REG = 128 + 24
_STORE_REG_ACTION = "increase"
# Warp in main WarpGroup
_MMA_WARP = _MAIN_WG * 4
_W_TMA_WARP = _MAIN_WG * 4 + 1
_X_TMA_WARP = _MAIN_WG * 4 + 2
_SCALE_TMA_WARP = _MAIN_WG * 4 + 3
_TMEM = plgpu.Layout.TCGEN05_TMEM_NATIVE
_TCGEN05 = plgpu.Layout.TCGEN05
_TCGEN05_COL = _TCGEN05.reduce(0)
_TCGEN05_ROW = _TCGEN05.reduce(1)


def rescale_i32_tcgen05_acc(running_acc, acc, row_scale, col_scale):
  """Dequantizes S32 TCGEN05 accumulator and adds it to a running f32 accumulator."""

  @plgpu.inline_mgpu(
      arg_types=(
          _TCGEN05,
          _TCGEN05,
          _TCGEN05_ROW,
          _TCGEN05_COL,
      ),
      return_type=plgpu.ShapeDtypeStruct(acc.shape, jnp.float32, _TCGEN05),
  )
  def rescale(
      _,
      running_acc: mgpu.FragmentedArray,
      a: mgpu.FragmentedArray,
      rs: mgpu.FragmentedArray,
      cs: mgpu.FragmentedArray,
  ):
    # Example of register shapes for (128, 64):
    # TCGEN05     (1, 8, 1, 1, 4, 1, 1, 1, 1)
    # TCGEN05_ROW (1, 1, 4, 1, 1)
    # TCGEN05_COL (8, 1, 1)
    f32 = ir.F32Type.get()
    out_vec_ty = ir.VectorType.get((2,), f32)
    f32_registers = running_acc.registers
    for idx, reg_a in np.ndenumerate(a.registers):
      rs_reg = rs.registers[(0, 0, idx[4], 0, 0)]
      cs_reg = cs.registers[(idx[1], 0, 0)]
      out_reg = arith.sitofp(out_vec_ty, reg_a)
      scale_reg = arith.mulf(
          mgpu.utils.vector_concat([rs_reg, rs_reg]),
          cs_reg,
          fastmath=arith.FastMathFlags.fast,
      )
      out_reg = arith.mulf(
          out_reg,
          scale_reg,
          fastmath=arith.FastMathFlags.fast,
      )
      f32_registers[idx] = arith.addf(
          out_reg,
          f32_registers[idx],
          fastmath=arith.FastMathFlags.fast,
      )
    return mgpu.FragmentedArray(
        _registers=f32_registers,
        _layout=mgpu.TCGEN05_LAYOUT,
        _is_signed=None,
    )

  return rescale(running_acc, acc, row_scale, col_scale)


@jaxtyping.jaxtyped
def ragged_dot_gpu_quant_i8_blackwell_kernel(
    lhs: Float[Array | qwix.QArray, "M K"],
    rhs: Float[qwix.QArray, "G K N"],
    group_sizes: Integer[Array, "G"],
    out_dtype,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
) -> Float[Array, "M N"]:
  """Returns the Pallas kernel for quantized ragged dot by using native int8 MMA.

  There are 4 Warp Group in this kernel:
    DEQUANT_WGS(2): dequant
      | load lhs[:128,:],rhs -> dequant -> to TMEM | ...
      | load lhs[128:,:],rhs -> dequant -> to TMEM | ...
    MEMORY_WG(1): issue TMA for loading lhs, rhs from HBM to SMEM.
      | wait for x, w consumed -> issue TMA | ...
    STORE_WG(1): scale acc and store the result from SMEM to HBM.
      | MMA Ready -> scale acc -> ... -> Reg to SMEM -> SMEM to HBM |
    memory loading and computing.

  Args:
    lhs: The left hand side of the ragged dot. shape: (m, k)
    rhs: The right hand side of the ragged dot. shape: (g, k, n)
    group_sizes: The group sizes of the ragged dot. shape: (g)
    out_dtype: The output dtype of the ragged dot.
    config: The configuration of the ragged dot.
    activation: Optional activation function to apply to the output of the
      ragged dot.

  Returns:
    The output of the ragged dot. shape: (m, n)
  """

  assert rhs.zero_point is None

  block_m = config.block_m
  block_n = config.block_n
  block_k = config.block_k
  collective = config.collective
  num_stages = config.num_stages
  # `tile` is for each block
  tile_m = block_m
  tile_n = block_n
  if collective:
    block_m *= 2
    block_n *= 2

  w, w_scales = (rhs.qvalue.mT, rhs.scale)
  num_groups, n, k_w = w.shape
  m, k_x = lhs.shape
  tile_k = k_w // w_scales.shape[1]

  if k_w != k_x:
    raise NotImplementedError(
        f"Contraction dim mismatch: weights.shape[1]={k_w}, x.shape[-1]={k_x}"
    )

  if group_sizes.shape != (num_groups,):
    raise NotImplementedError(
        "Expected group_sizes to have shape"
        f" {(num_groups,)} but got {group_sizes.shape}"
    )

  if config.collective:
    raise NotImplementedError("collective is not supported.")

  if config.split_m != 1:
    raise NotImplementedError("split_m is not supported.")

  if config.post_scale:
    raise NotImplementedError("post_scale is not supported.")

  if isinstance(lhs, qwix.QArray) and lhs.qtype != jnp.int8:
    raise NotImplementedError(
        f"Only supported lhs to be bfloat16/int8, got: {lhs.dtype=}."
    )

  if not isinstance(rhs, qwix.QArray) or rhs.qtype != jnp.int4:
    raise NotImplementedError(
        f"Only supported rhs to be int4, got: {rhs.dtype=}."
    )

  if tile_k % block_k != 0:
    raise NotImplementedError(
        f"tile_k must be multiple of block_k, got: {tile_k=} {block_k=}."
    )

  x_dtype = jnp.int8
  try:
    swizzle = plgpu.find_swizzle(block_k * jnp.dtype(x_dtype).itemsize * 8)
  except ValueError as e:
    raise NotImplementedError("No possible swizzle.") from e

  swizzle_elems = swizzle // jnp.dtype(x_dtype).itemsize

  try:
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )
  except ValueError as e:
    raise NotImplementedError(
        f"{swizzle=} {swizzle_elems=} unsupported."
    ) from e

  w_elem_bits = 4
  try:
    w_swizzle = plgpu.find_swizzle(block_k * w_elem_bits)  # n,k
  except ValueError as e:
    raise NotImplementedError("No possible swizzle.") from e

  w_swizzle_elems = (w_swizzle * 8) // w_elem_bits

  tile_shape = (1, tile_k, 1)
  tiled_axes = {i: d for i, d in enumerate(tile_shape)}
  # TODO: Use custom quantization kernel instead of XLA.
  with jax.named_scope("quantize"):
    if not isinstance(lhs, qwix.QArray):
      lhs_quant = quantization.AsQArray(
          lhs, x_dtype, tiled_axes=tiled_axes
      ).as_qarray()
    else:
      lhs_quant = lhs
  x, x_scales = lhs_quant.qvalue, lhs_quant.scale.mT
  tile_xk = k_x // x_scales.shape[0]

  if tile_xk != tile_k:
    raise NotImplementedError(
        f"tile must be equal, got: {tile_xk=} {tile_k=}."
    )

  m_iters = pl.cdiv(m, block_m) + num_groups - 1
  n_iters = pl.cdiv(n, block_n * _DEQ_WG)
  group_info = common.GroupInfo.create_aligned(
      group_sizes, block_m, m_iters, noops_at_end=False
  )

  def kernel(*refs, scoped):
    (
        x_gmem,
        x_scales_gmem,
        w_gmem,
        w_scales_gmem,
        _,
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
    (
        x_smem,
        w_smem,
        w_i8_tmem,
        out_smem,
        acc_tmem,
    ) = scratch_buffers
    (
        x_tma_barrier,
        x_consumed_barrier,
        w_tma_barrier,
        w_consumed_barrier,
        w_ready_barrier,
        tcgen05_barrier,
        acc_consumed_barrier,
    ) = barriers

    m, k = x_gmem.shape
    num_k_iters = pl.cdiv(k, block_k)
    cluster_idx = lax.axis_index("x")

    @plgpu.nd_loop((m_iters * n_iters,), collective_axes=("sm",), init_carry=0)
    def mn_loop(loop_info: plgpu.NDLoopInfo, carry):
      (lin_idx,) = loop_info.index
      tid_m, tid_n = plgpu.planar_snake(
          lin_idx,
          (m_iters, n_iters),
          config.grid_minor_dim,
          config.grid_tile_width,
      )
      group_id = group_id_gmem[tid_m]
      start_within_block = start_within_block_gmem[tid_m]
      actual_size = actual_size_gmem[tid_m]
      block_start = block_start_gmem[tid_m]
      slice_m = pl.ds(block_start, block_m)
      slice_n = lambda i: pl.ds(
          tid_n * block_n * _DEQ_WG + i * block_n + cluster_idx * tile_n, tile_n
      )
      wg = jax.lax.axis_index("wg")

      is_lead_block = cluster_idx == 0

      @pl.when(actual_size > 0)
      def _body():
        @pl.when(wg == _MAIN_WG)
        def _():
          plgpu.set_max_registers(_MAIN_REG, action=_MAIN_REG_ACTION)

          @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
          def _per_warp():
            warp_id = lax.axis_index("warp")

            @pl.when(warp_id == _W_TMA_WARP)
            def w_tma_warp():
              def do_tma_w(ki, ni, slot):
                plgpu.copy_gmem_to_smem(  # e,n,k
                    w_gmem.at[
                        group_id,
                        slice_n(ni),
                        pl.ds(ki * block_k, block_k),
                    ],
                    w_smem.at[ni * num_stages + slot],
                    w_tma_barrier.at[ni * num_stages + slot],
                )

              def loop_body(ki, _):
                global_ki = ki + carry * num_k_iters
                slot = lax.rem(global_ki, num_stages)

                for ni in range(_DEQ_WG):

                  @pl.when(global_ki >= num_stages)
                  def _():
                    plgpu.barrier_wait(
                        w_consumed_barrier.at[ni * num_stages + slot]
                    )

                  do_tma_w(ki, ni, slot)

              lax.fori_loop(0, num_k_iters, loop_body, None)

            @pl.when(warp_id == _X_TMA_WARP)
            def x_tma_warp():
              def do_tma_x(ki, slot):
                plgpu.copy_gmem_to_smem(  # m,k
                    x_gmem.at[
                        slice_m,
                        pl.ds(ki * block_k, block_k),
                    ],
                    x_smem.at[slot],
                    x_tma_barrier.at[slot],
                    partitioned_axis=0 if collective else None,
                    collective_axes="x" if collective else None,
                )

              def loop_body(ki, _):
                global_ki = ki + carry * num_k_iters
                slot = lax.rem(global_ki, num_stages)

                @pl.when(global_ki >= num_stages)
                def _():
                  # Wait for the previous mma to complete.
                  plgpu.barrier_wait(x_consumed_barrier.at[slot])

                do_tma_x(ki, slot)

              lax.fori_loop(0, num_k_iters, loop_body, None)

            @pl.when(warp_id == _MMA_WARP)
            def mma_warp():
              def do_mma(ki, _):
                global_ki = ki + carry * num_k_iters
                slot = lax.rem(global_ki, num_stages)

                @pl.when(is_lead_block)
                def _():
                  with jax.named_scope("[x]wait"):
                    plgpu.barrier_wait(x_tma_barrier.at[slot])

                for ni in range(_DEQ_WG):
                  mma_slot = ni * num_stages + slot
                  with jax.named_scope("[acc]wait"):

                    @pl.when(global_ki >= num_stages)
                    def _():
                      # wait for the previous mma to complete
                      # to ensure the acc_tmem is consumed.
                      plgpu.barrier_wait(acc_consumed_barrier.at[mma_slot])

                  with jax.named_scope("[w]wait"):
                    plgpu.barrier_wait(w_ready_barrier.at[ni])

                  @pl.when(is_lead_block)
                  def _():
                    with jax.named_scope("MMA"):
                      plgpu.tcgen05_mma(
                          acc_tmem.at[
                              :,
                              pl.ds(mma_slot * block_m, block_m),
                          ],
                          w_i8_tmem.at[:, pl.ds(ni * block_k, block_k)],
                          x_smem.at[slot].T,
                          tcgen05_barrier.at[mma_slot],
                          accumulate=False,
                          collective_axis="x" if collective else None,
                      )

                plgpu.tcgen05_commit_arrive(
                    x_consumed_barrier.at[slot],
                    collective_axis="x" if collective else None,
                )

              lax.fori_loop(0, num_k_iters, do_mma, None)

        @pl.when(wg < _DEQ_WG)
        def _():
          plgpu.set_max_registers(_DEQ_REG, action=_DEQ_REG_ACTION)

          def _deq(ki, _):
            global_ki = ki + carry * num_k_iters
            slot = lax.rem(global_ki, num_stages)

            with jax.named_scope("MMA"):
              plgpu.barrier_wait(w_tma_barrier.at[wg * num_stages + slot])

              @pl.when(global_ki > 0)
              def _():
                # Wait for the previous mma to complete.
                prev_slot = lax.rem(global_ki - 1, num_stages)
                plgpu.barrier_wait(
                    tcgen05_barrier.at[wg * num_stages + prev_slot]
                )

            # S -> T
            with jax.named_scope("S->R"):
              w = plgpu.load(
                  w_smem.at[wg * num_stages + slot],
                  (),
                  layout=_TMEM(8),
                  optimized=False,
              )

            # dequant
            with jax.named_scope("i4->i8"):
              w = w.astype(jnp.int8)
            with jax.named_scope("lcast"):
              w = plgpu.layout_cast(w, _TMEM(4))

            # R -> T
            plgpu.async_store_tmem(
                w_i8_tmem.at[:, pl.ds(wg * block_k, block_k)], w
            )
            plgpu.commit_tmem()

            plgpu.barrier_arrive(w_consumed_barrier.at[wg * num_stages + slot])
            with jax.named_scope("[w]ready"):
              plgpu.barrier_arrive(w_ready_barrier.at[wg])

          lax.fori_loop(
              0,
              num_k_iters,
              _deq,
              None,
          )

        @pl.when(wg == _STORE_WG)
        def _():
          plgpu.set_max_registers(_STORE_REG, action=_STORE_REG_ACTION)
          acc_dtype = jnp.float32
          acc_carry = [
              plgpu.layout_cast(
                  jnp.zeros((tile_n, block_m), dtype=acc_dtype),
                  _TCGEN05,
              ) for _ in range(_DEQ_WG)
          ]

          def _loop_body(ki, acc_carry):
            global_ki = ki + carry * num_k_iters
            slot = lax.rem(global_ki, num_stages)
            with jax.named_scope("[scale]load"):
              x_scale = plgpu.load(
                  x_scales_gmem,
                  (
                      lax.div((ki * block_k), tile_k),
                      slice_m,
                  ),
                  layout=_TCGEN05_COL,
                  optimized=False,
              ).astype(acc_dtype)

            def load_ws_acc(ni):
              ws = plgpu.load(
                  w_scales_gmem,
                  (
                      group_id,
                      lax.div((ki * block_k), tile_k),
                      slice_n(ni),
                  ),
                  layout=_TCGEN05_ROW,
                  optimized=False,
              ).astype(acc_dtype)
              with jax.named_scope("MMA"):
                plgpu.barrier_wait(tcgen05_barrier.at[ni * num_stages + slot])
              with jax.named_scope("T->R"):
                local_acc = plgpu.async_load_tmem(
                    acc_tmem.at[
                        :, pl.ds((ni * num_stages + slot) * block_m, block_m)
                    ]
                )
                plgpu.wait_load_tmem()
              plgpu.barrier_arrive(
                  acc_consumed_barrier.at[ni * num_stages + slot]
              )
              return ws, local_acc

            for ni in range(_DEQ_WG):
              ws, local_acc = load_ws_acc(ni)
              acc_carry[ni] = rescale_i32_tcgen05_acc(
                  acc_carry[ni], local_acc, ws, x_scale
              )
              del ws
            return acc_carry

          acc_carry = lax.fori_loop(0, num_k_iters, _loop_body, acc_carry)
          # Apply activation function to the output in dtype of acc if
          # provided.
          for ni in range(_DEQ_WG):
            acc_carry[ni] = (
                activation(acc_carry[ni])
                if activation is not None
                else acc_carry[ni]
            )

          # acc -> SMEM
          with jax.named_scope("acc -> SMEM"):
            for ni in range(_DEQ_WG):
              out_smem.at[ni].T[...] = plgpu.layout_cast(
                  acc_carry[ni].astype(out_smem.dtype),
                  plgpu.Layout.TCGEN05_TRANSPOSED,
              )
            plgpu.commit_smem()

          with jax.named_scope("SMEM -> GMEM"):
            # Write out the largest power of two rows first,
            # then the next largest, etc.
            # This allows us to coalesce writes as much as possible.
            offset = start_within_block
            size = 1 << (min(block_m, m).bit_length() - 1)
            while size > 0:

              @pl.when(actual_size & size != 0)
              def _():
                for ni in range(_DEQ_WG):
                  out_smem_slice = out_smem.at[ni, pl.ds(offset, size)]
                  o_gref_slice = out_gmem.at[
                      pl.ds(block_start + offset, size),
                      pl.ds(
                          tid_n * block_n * _DEQ_WG
                          + ni * block_n
                          + cluster_idx * tile_n,
                          tile_n,
                      ),
                  ]
                  plgpu.copy_smem_to_gmem(
                      out_smem_slice, o_gref_slice, commit_group=False
                  )

              offset += actual_size & size
              size //= 2
            plgpu.commit_smem_to_gmem_group()
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)

      return carry + (actual_size > 0)

  def kernel_entry(*refs):
    x_smem = plgpu.SMEM(
        (num_stages, tile_m, block_k),
        dtype=x.dtype,
        transforms=transforms,
    )
    w_smem = plgpu.SMEM(
        (_DEQ_WG * num_stages, tile_n, block_k),
        dtype=w.dtype,
        transforms=(
            plgpu.TilingTransform((8, w_swizzle_elems)),
            plgpu.SwizzleTransform(w_swizzle),
        ),
    )
    w_i8_tmem = plgpu.TMEM(
        (tile_n, _DEQ_WG * block_k),
        dtype=jnp.int8,
        packed=True,
        collective=collective,
    )
    out_smem = plgpu.SMEM(
        (_DEQ_WG, block_m, tile_n),
        dtype=out_dtype,
        # workaround for ValueError: Dynamic slice base index (which is a
        # dynamic value) cannot be statically proven to be divisible by
        # the tiling (8)
        transforms=(
            plgpu.TilingTransform((1, 128 // jnp.dtype(out_dtype).itemsize)),
            plgpu.SwizzleTransform(128),
        ),
    )
    acc_tmem = plgpu.TMEM(
        (tile_n, _DEQ_WG * num_stages * block_m),
        dtype=jnp.int32,
        collective=collective,
    )
    x_tma_barrier = plgpu.Barrier(num_barriers=num_stages)
    x_consumed_barrier = plgpu.Barrier(
        num_barriers=num_stages, orders_tensor_core=True
    )
    w_tma_barrier = plgpu.Barrier(num_barriers=_DEQ_WG * num_stages)
    w_consumed_barrier = plgpu.Barrier(num_barriers=_DEQ_WG * num_stages)
    if collective:
      w_ready_barrier = plgpu.ClusterBarrier(
          num_barriers=_DEQ_WG,
          collective_axes=("x",),
      )
      acc_consumed_barrier = plgpu.ClusterBarrier(
          num_barriers=_DEQ_WG * num_stages,
          collective_axes=("x",),
      )
    else:
      w_ready_barrier = plgpu.Barrier(num_barriers=_DEQ_WG)
      acc_consumed_barrier = plgpu.Barrier(num_barriers=_DEQ_WG * num_stages)

    tcgen05_barrier = plgpu.Barrier(
        num_barriers=_DEQ_WG * num_stages, orders_tensor_core=True
    )
    pl.run_scoped(
        lambda *args: kernel(*refs, scoped=args),
        (
            x_smem,
            w_smem,
            w_i8_tmem,
            out_smem,
            acc_tmem,
        ),
        (
            x_tma_barrier,
            x_consumed_barrier,
            w_tma_barrier,
            w_consumed_barrier,
            w_ready_barrier,
            tcgen05_barrier,
            acc_consumed_barrier,
        ),
        collective_axes="wg",
    )

  num_sms = backend.get_default_device().core_count
  profile = False
  if profile:
    num_sms = 1 + collective
  f = plgpu.kernel(
      kernel_entry,
      out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
      num_threads=_DEQ_WG + 2,
      thread_name="wg",
      grid=(num_sms // (1 + collective),),
      grid_names=("sm",),
      cluster=(1 + collective,),
      cluster_names=("x",),
      compiler_params=plgpu.CompilerParams(
          approx_math=True,
          unsafe_no_auto_barriers=True,
          profile_space=160 if profile else 0,
          profile_dir="sponge" if profile else "",
      ),
  )
  return f(
      x,
      x_scales,
      w,
      w_scales,
      group_info.block,
      group_info.group_id,
      group_info.start_within_block,
      group_info.actual_size,
      group_info.block_start,
  )
