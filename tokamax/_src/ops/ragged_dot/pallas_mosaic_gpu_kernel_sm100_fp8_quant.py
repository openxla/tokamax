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

from absl import logging
import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer  # pylint: disable=g-multiple-import,g-importing-member
from jaxlib.mlir.dialects import arith
import numpy as np
import qwix
from tokamax._src import jaxtyping
from tokamax._src import mosaic_gpu as mgpu_utils
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common


# Dequant WarpGroup
_DEQ_WG = 2
# MMA + TMA
_MAIN_WG = _DEQ_WG
# Scale ACC and Store
_STORE_WG = _MAIN_WG + 1
# Warp in main WarpGroup
_MMA_WARP = _MAIN_WG * 4
_W_TMA_WARP = _MAIN_WG * 4 + 1
_X_TMA_WARP = _MAIN_WG * 4 + 2
_SCALE_TMA_WARP = _MAIN_WG * 4 + 3
_TMEM = plgpu.Layout.TCGEN05_TMEM_NATIVE
_TCGEN05 = plgpu.Layout.TCGEN05
_TCGEN05_TRANSPOSED = plgpu.Layout.TCGEN05_TRANSPOSED
_TCGEN05_COL = _TCGEN05.reduce(0)
_TCGEN05_ROW = _TCGEN05.reduce(1)


def rescale_tcgen05_acc(running_acc, acc, row_scale, col_scale):
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
    f32_registers = running_acc.registers
    for idx, reg_a in np.ndenumerate(a.registers):
      rs_reg = rs.registers[(0, 0, idx[4], 0, 0)]
      cs_reg = cs.registers[(idx[1], 0, 0)]
      out_reg = reg_a
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


def _compute_stages(
    block_m: int,
    block_n: int,
    block_k: int,
    x_dtype: jnp.dtype,
    w_dtype: jnp.dtype,
    ws_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
) -> tuple[int, int, int, int]:
  """Compute the number of stages for each type of data."""

  tmem_max_cols = 512
  tmem_cell_bitwidth = 32
  # 4096 bytes is reserved for barriers.
  smem_capacity = common.get_smem_capacity() - 4096
  w_bf16_tmem_cols = _DEQ_WG * block_k // (
      tmem_cell_bitwidth // common.num_bits(x_dtype)
  )
  acc_tmem_cols = (
      _DEQ_WG * block_m // (tmem_cell_bitwidth // common.num_bits(acc_dtype))
  )
  acc_stages = 1
  deq_stages = (tmem_max_cols - acc_stages * acc_tmem_cols) // w_bf16_tmem_cols
  acc_stages += (
      tmem_max_cols - acc_stages * acc_tmem_cols - deq_stages * w_bf16_tmem_cols
  ) // acc_tmem_cols
  # acc_stages is at most 4 to avoid too much barriers in the kernel.
  acc_stages = min(acc_stages, 4)

  out_smem_bytes = _DEQ_WG * block_m * block_n * common.num_bytes(out_dtype)
  x_smem_bytes = block_m * block_k * common.num_bytes(x_dtype)
  w_smem_bytes = _DEQ_WG * block_n * block_k * common.num_bytes(w_dtype)
  ws_smem_bytes = _DEQ_WG * block_n * common.num_bytes(ws_dtype)
  data_stages = (smem_capacity - out_smem_bytes) // (
      x_smem_bytes + w_smem_bytes
  )
  scale_stages = (
      smem_capacity
      - out_smem_bytes
      - data_stages * (x_smem_bytes + w_smem_bytes)
  ) // ws_smem_bytes
  # scale_stages is at most 4 to avoid too much barriers in the kernel.
  scale_stages = min(scale_stages, 4)
  return int(data_stages), int(scale_stages), int(deq_stages), int(acc_stages)


@jaxtyping.jaxtyped
def ragged_dot_gpu_fp8_quant_blackwell_kernel(
    lhs: Float[qwix.QArray, "M K"],
    rhs: Float[qwix.QArray, "G K N"],
    group_sizes: Integer[Array, "G"],
    out_dtype,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
) -> Float[Array, "M N"]:
  """Returns the Pallas kernel for fp8xint4 ragged dot.
  
  The kernel is using the trick of biased fp8 encoding to avoid explicit
  dequantization from int4 to float8_e4m3fn.

  w_fp32 = 512 * w_biased_fp8 - 8.0
  acc = (w_fp32 * w_scale) @ (x * x_scale)
  acc = (512 * (w_biased_fp8 @ x) - 8.0 * x_row_sum) * w_scale * x_scale

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

  if lhs.qtype != jnp.float8_e4m3fn:
    raise NotImplementedError(
        f"Only supported lhs to be float8_e4m3fn, got: {lhs.dtype=}."
    )

  if rhs.qtype != jnp.int4:
    raise NotImplementedError(
        f"Only supported rhs to be int4, got: {rhs.dtype=}."
    )

  if tile_k % block_k != 0:
    raise NotImplementedError(
        f"tile_k must be multiple of block_k, got: {tile_k=} {block_k=}."
    )
  data_stages, scale_stages, deq_stages, acc_stages = _compute_stages(
      block_m,
      block_n,
      block_k,
      x_dtype=lhs.qtype,
      w_dtype=rhs.qtype,
      ws_dtype=rhs.scale.dtype,
      acc_dtype=jnp.float32,
      out_dtype=out_dtype,
  )
  logging.info(
      "data_stages: %d, scale_stages: %d, deq_stages: %d, acc_stages: %d",
      data_stages,
      scale_stages,
      deq_stages,
      acc_stages,
  )

  x_dtype = lhs.qtype
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

  x, x_scales = lhs.qvalue, lhs.scale.mT
  tile_xk = k_x // x_scales.shape[0]

  if tile_xk != tile_k:
    raise NotImplementedError(f"tile must be equal, got: {tile_xk=} {tile_k=}.")

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
        ws_smem,
        w_tmem,
        out_smem,
        acc_tmem,
    ) = scratch_buffers
    (
        x_tma_barrier,
        x_mma_consumed_barrier,
        x_scale_consumed_barrier,
        w_tma_barrier,
        w_consumed_barrier,
        ws_tma_barrier,
        ws_consumed_barrier,
        w_tmem_ready_barrier,
        w_tmem_consumed_barrier,
        acc_ready_barrier,
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
          tid_n * block_n * _DEQ_WG + i * block_n, block_n
      )
      wg = jax.lax.axis_index("wg")

      @pl.when(actual_size > 0)
      def _body():
        @pl.when(wg == _MAIN_WG)
        def _():
          plgpu.set_max_registers(104, action="decrease")

          @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
          def _per_warp():
            warp_id = lax.axis_index("warp")

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
                )

              def loop_body(ki, _):
                global_ki = ki + carry * num_k_iters
                slot = lax.rem(global_ki, data_stages)

                @pl.when(global_ki >= data_stages)
                def _():
                  plgpu.barrier_wait(x_scale_consumed_barrier.at[slot])
                  plgpu.barrier_wait(x_mma_consumed_barrier.at[slot])

                do_tma_x(ki, slot)

              lax.fori_loop(0, num_k_iters, loop_body, None)

            @pl.when(warp_id == _W_TMA_WARP)
            def w_tma_warp():
              def do_tma_w(ki, ni, slot):
                plgpu.copy_gmem_to_smem(  # e,n,k
                    w_gmem.at[
                        group_id,
                        slice_n(ni),
                        pl.ds(ki * block_k, block_k),
                    ],
                    w_smem.at[slot, ni],
                    w_tma_barrier.at[ni * data_stages + slot],
                )

              def loop_body(ki, _):
                global_ki = ki + carry * num_k_iters
                slot = lax.rem(global_ki, data_stages)

                for ni in range(_DEQ_WG):

                  @pl.when(global_ki >= data_stages)
                  def _():
                    plgpu.barrier_wait(
                        w_consumed_barrier.at[ni * data_stages + slot]
                    )

                  do_tma_w(ki, ni, slot)

              lax.fori_loop(0, num_k_iters, loop_body, None)

            @pl.when(warp_id == _SCALE_TMA_WARP)
            def ws_tma_warp():
              def do_tma_ws(ki, ni, slot):
                plgpu.copy_gmem_to_smem(  # e,m,k
                    w_scales_gmem.at[
                        group_id,
                        lax.div((ki * block_k), tile_k),
                        slice_n(ni),
                    ],
                    ws_smem.at[slot, ni],
                    ws_tma_barrier.at[ni * scale_stages + slot],
                )

              def loop_body(ki, _):
                global_ki = ki + carry * num_k_iters
                slot = lax.rem(global_ki, scale_stages)

                for ni in range(_DEQ_WG):

                  @pl.when(global_ki >= scale_stages)
                  def _():
                    plgpu.barrier_wait(
                        ws_consumed_barrier.at[ni * scale_stages + slot]
                    )

                  do_tma_ws(ki, ni, slot)

              lax.fori_loop(0, num_k_iters, loop_body, None)

            @pl.when(warp_id == _MMA_WARP)
            def mma_warp():
              def do_mma(ki, _):
                global_ki = ki + carry * num_k_iters
                x_slot = lax.rem(global_ki, data_stages)
                w_slot = lax.rem(global_ki, deq_stages)
                acc_slot = lax.rem(global_ki, acc_stages)

                plgpu.barrier_wait(x_tma_barrier.at[x_slot])

                for ni in range(_DEQ_WG):
                  ni_acc_slot = ni * acc_stages + acc_slot
                  ni_w_slot = ni * deq_stages + w_slot

                  @pl.when(global_ki >= acc_stages)
                  def _():
                    # wait for the previous mma to complete
                    # to ensure the acc_tmem is consumed.
                    plgpu.barrier_wait(acc_consumed_barrier.at[ni_acc_slot])

                  plgpu.barrier_wait(w_tmem_ready_barrier.at[ni_w_slot])

                  plgpu.tcgen05_mma(
                      acc_tmem.at[
                          :,
                          pl.ds(ni_acc_slot * block_m, block_m),
                      ],
                      w_tmem.at[:, pl.ds(ni_w_slot * block_k, block_k)],
                      x_smem.at[x_slot].T,
                      w_tmem_consumed_barrier.at[ni_w_slot],
                      accumulate=False,
                      collective_axis="x" if collective else None,
                  )
                  plgpu.tcgen05_commit_arrive(
                      acc_ready_barrier.at[ni_acc_slot],
                  )

                plgpu.tcgen05_commit_arrive(
                    x_mma_consumed_barrier.at[x_slot],
                )

              lax.fori_loop(0, num_k_iters, do_mma, None)

        @pl.when(wg < _DEQ_WG)
        def _():
          plgpu.set_max_registers(128, action="increase")

          def _deq(ki, _):
            global_ki = ki + carry * num_k_iters
            data_slot = lax.rem(global_ki, data_stages)
            wg_data_slot = wg * data_stages + data_slot
            w_slot = lax.rem(global_ki, deq_stages)
            wg_w_slot = wg * deq_stages + w_slot

            plgpu.barrier_wait(w_tma_barrier.at[wg_data_slot])

            @pl.when(global_ki >= deq_stages)
            def _():
              plgpu.barrier_wait(w_tmem_consumed_barrier.at[wg_w_slot])

            # S -> T
            with jax.named_scope("S->R"):
              w = plgpu.load(
                  w_smem.at[data_slot, wg],
                  (),
                  layout=_TMEM(8),
                  optimized=False,
              )

            # dequant
            w = mgpu_utils.int4_as_biased_f8e4m3fn(w, _TMEM(8))
            w = plgpu.layout_cast(w, _TMEM(4))

            # R -> T
            plgpu.async_store_tmem(
                w_tmem.at[:, pl.ds(wg_w_slot * block_k, block_k)], w
            )
            plgpu.commit_tmem()

            plgpu.barrier_arrive(w_tmem_ready_barrier.at[wg_w_slot])
            plgpu.barrier_arrive(w_consumed_barrier.at[wg_data_slot])

          lax.fori_loop(
              0,
              num_k_iters,
              _deq,
              None,
          )

        @pl.when(wg == _STORE_WG)
        def _():
          plgpu.set_max_registers(152, action="increase")
          acc_dtype = jnp.float32
          acc_carry = [
              plgpu.layout_cast(
                  jnp.zeros((block_n, block_m), dtype=acc_dtype),
                  _TCGEN05,
              )
              for _ in range(_DEQ_WG)
          ]

          def _loop_body(ki, acc_carry):
            global_ki = ki + carry * num_k_iters
            data_slot = lax.rem(global_ki, data_stages)
            scale_slot = lax.rem(global_ki, scale_stages)
            acc_slot = lax.rem(global_ki, acc_stages)
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

            plgpu.barrier_wait(x_tma_barrier.at[data_slot])
            act = plgpu.load(
                x_smem.at[data_slot].T,
                (),
                layout=_TCGEN05_TRANSPOSED,
            )
            plgpu.commit_smem()
            plgpu.barrier_arrive(x_scale_consumed_barrier.at[data_slot])
            act = plgpu.layout_cast(
                lax.broadcast_in_dim(
                    act.astype(acc_dtype).sum(0),
                    (block_n, block_m),
                    [1],
                ),
                _TCGEN05_TRANSPOSED,
            )
            act = plgpu.layout_cast(act, _TCGEN05)

            def load_ws_acc(ni):
              ni_scale_slot = ni * scale_stages + scale_slot
              ni_acc_slot = ni * acc_stages + acc_slot
              plgpu.barrier_wait(ws_tma_barrier.at[ni_scale_slot])
              ws = plgpu.load(
                  ws_smem.at[scale_slot, ni],
                  (),
                  layout=_TCGEN05_ROW,
                  optimized=True,
              ).astype(acc_dtype)

              plgpu.barrier_wait(acc_ready_barrier.at[ni_acc_slot])
              with jax.named_scope("T->R"):
                local_acc = plgpu.async_load_tmem(
                    acc_tmem.at[:, pl.ds(ni_acc_slot * block_m, block_m)]
                )
                plgpu.wait_load_tmem()
              plgpu.barrier_arrive(acc_consumed_barrier.at[ni_acc_slot])
              plgpu.barrier_arrive(ws_consumed_barrier.at[ni_scale_slot])
              return ws, local_acc

            for ni in range(_DEQ_WG):
              ws, local_acc = load_ws_acc(ni)
              local_acc = 512 * local_acc - 8 * act
              acc_carry[ni] = rescale_tcgen05_acc(
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
                          + cluster_idx * block_n,
                          block_n,
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
        (data_stages, block_m, block_k),
        dtype=x_dtype,
        transforms=transforms,
    )
    w_smem = plgpu.SMEM(
        (data_stages, _DEQ_WG, block_n, block_k),
        dtype=w.dtype,
        transforms=(
            plgpu.TilingTransform((8, w_swizzle_elems)),
            plgpu.SwizzleTransform(w_swizzle),
        ),
    )
    w_tmem = plgpu.TMEM(
        (block_n, _DEQ_WG * deq_stages * block_k),
        dtype=x_dtype,
        packed=True,
    )
    ws_smem = plgpu.SMEM(
        (scale_stages, _DEQ_WG, block_n),
        dtype=w_scales.dtype,
    )
    out_smem = plgpu.SMEM(
        (_DEQ_WG, block_m, block_n),
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
        (block_n, acc_stages * _DEQ_WG * block_m),
        dtype=jnp.float32,
        collective=collective,
    )
    x_tma_barrier = plgpu.Barrier(num_barriers=data_stages)
    x_mma_consumed_barrier = plgpu.Barrier(
        num_barriers=data_stages, orders_tensor_core=True
    )
    x_scale_consumed_barrier = plgpu.Barrier(num_barriers=data_stages)
    w_tma_barrier = plgpu.Barrier(num_barriers=_DEQ_WG * data_stages)
    w_consumed_barrier = plgpu.Barrier(num_barriers=_DEQ_WG * data_stages)
    ws_tma_barrier = plgpu.Barrier(num_barriers=_DEQ_WG * scale_stages)
    ws_consumed_barrier = plgpu.Barrier(num_barriers=_DEQ_WG * scale_stages)
    w_tmem_ready_barrier = plgpu.Barrier(num_barriers=_DEQ_WG * deq_stages)
    w_tmem_consumed_barrier = plgpu.Barrier(
        num_barriers=_DEQ_WG * deq_stages, orders_tensor_core=True
    )
    acc_ready_barrier = plgpu.Barrier(
        num_barriers=_DEQ_WG * acc_stages, orders_tensor_core=True
    )
    acc_consumed_barrier = plgpu.Barrier(num_barriers=_DEQ_WG * acc_stages)

    pl.run_scoped(
        lambda *args: kernel(*refs, scoped=args),
        (
            x_smem,
            w_smem,
            ws_smem,
            w_tmem,
            out_smem,
            acc_tmem,
        ),
        (
            x_tma_barrier,
            x_mma_consumed_barrier,
            x_scale_consumed_barrier,
            w_tma_barrier,
            w_consumed_barrier,
            ws_tma_barrier,
            ws_consumed_barrier,
            w_tmem_ready_barrier,
            w_tmem_consumed_barrier,
            acc_ready_barrier,
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
