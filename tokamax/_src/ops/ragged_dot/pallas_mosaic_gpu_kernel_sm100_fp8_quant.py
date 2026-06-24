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

import dataclasses
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
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common


# Warp in main WarpGroup
_MMA_WARP = 0
_W_TMA_WARP = 1
_X_TMA_WARP = 2
_SCALE_TMA_WARP = 3
_TMEM = plgpu.Layout.TCGEN05_TMEM_NATIVE
_TCGEN05 = plgpu.Layout.TCGEN05
_TCGEN05_TRANSPOSED = plgpu.Layout.TCGEN05_TRANSPOSED
_TCGEN05_COL = _TCGEN05.reduce(0)
_TCGEN05_ROW = _TCGEN05.reduce(1)


@dataclasses.dataclass(frozen=True)
class KernelConfig:
  """Config to set register count and number of warpgroups for dequant."""

  deq_wg: int
  total_wg: int
  main_wg_regs: int
  deq_wg_regs: int
  store_wg_regs: int

  def __post_init__(self):
    """Validate register count and number of warpgroups."""
    avg_regs = 65536 // (self.total_wg * 128)
    default_regs = (avg_regs // 8) * 8
    max_available = self.total_wg * 128 * default_regs

    total_used = 128 * (
        self.main_wg_regs + self.deq_wg * self.deq_wg_regs + self.store_wg_regs
    )
    if total_used > max_available:
      raise ValueError(
          f"Total register count used ({total_used}) exceeds maximum available"
          f" registers ({max_available}) for {self.total_wg} warpgroups"
          f" (default {default_regs} regs/thread)."
      )

  def set_max_registers(self, reg_count: int):
    """Wraps plgpu.set_max_registers to automatically determine the action."""
    avg_regs = 65536 // (self.total_wg * 128)
    default_regs = (avg_regs // 8) * 8
    if reg_count >= default_regs:
      plgpu.set_max_registers(reg_count, action="increase")
    elif reg_count < default_regs:
      plgpu.set_max_registers(reg_count, action="decrease")


PREFILL_CONFIG = KernelConfig(
    deq_wg=1,
    total_wg=3,
    main_wg_regs=96,
    deq_wg_regs=200,
    store_wg_regs=208,
)

DECODE_CONFIG = KernelConfig(
    deq_wg=2,
    total_wg=4,
    main_wg_regs=104,
    deq_wg_regs=128,
    store_wg_regs=152,
)


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
    cluster_size: int,
    block_k: int,
    x_dtype: jnp.dtype,
    w_dtype: jnp.dtype,
    ws_dtype: jnp.dtype,
    xs_dtype: jnp.dtype,
    xsum_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    deq_wg: int,
) -> tuple[int, int, int, int]:
  """Compute the number of stages for each type of data."""
  # 4096 bytes is reserved for barriers.
  smem_capacity = common.get_smem_capacity() - 4096
  tmem_max_cols = 512
  tmem_bank_bits = 32
  tmem_bank_x_elems = tmem_bank_bits // mgpu_lib.num_bits(x_dtype)
  tmem_bank_acc_elems = tmem_bank_bits // mgpu_lib.num_bits(acc_dtype)
  w_bf16_tmem_cols = block_k // tmem_bank_x_elems
  acc_tmem_cols = block_m * cluster_size // tmem_bank_acc_elems
  acc_stages = 1
  deq_stages = (tmem_max_cols - acc_stages * acc_tmem_cols) // w_bf16_tmem_cols
  deq_stages = (deq_stages // deq_wg) * deq_wg
  acc_stages += (
      tmem_max_cols - acc_stages * acc_tmem_cols - deq_stages * w_bf16_tmem_cols
  ) // acc_tmem_cols
  # acc_stages is at most 4 to avoid too much barriers in the kernel.
  acc_stages = min(acc_stages, 4)

  out_smem_bytes = (
      block_m * cluster_size * block_n * jnp.dtype(out_dtype).itemsize
  )
  smem_capacity -= out_smem_bytes
  x_smem_bytes = block_m * block_k * mgpu_lib.num_bits(x_dtype) // 8
  w_smem_bytes = block_n * block_k * mgpu_lib.num_bits(w_dtype) // 8
  ws_smem_bytes = block_n * mgpu_lib.num_bits(ws_dtype) // 8
  xs_smem_bytes = cluster_size * block_m * mgpu_lib.num_bits(xs_dtype) // 8
  xsum_smem_bytes = cluster_size * block_m * mgpu_lib.num_bits(xsum_dtype) // 8
  xw_stages, smem_capacity = divmod(smem_capacity, x_smem_bytes + w_smem_bytes)
  xw_stages = (int(xw_stages) // deq_wg) * deq_wg
  scale_smem_bytes = ws_smem_bytes + xs_smem_bytes + xsum_smem_bytes
  # scale_stages is at most 4 to avoid too much barriers in the kernel.
  scale_stages = min(smem_capacity // scale_smem_bytes, 4)
  return int(xw_stages), int(scale_stages), int(deq_stages), int(acc_stages)


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
  cluster_block_m = (block_m * 2) if collective else block_m
  cluster_block_n = (block_n * 2) if collective else block_n

  w, w_scales = (rhs.qvalue.mT, rhs.scale)
  num_groups, n, k_w = w.shape
  m, k_x = lhs.shape
  tile_k = k_w // w_scales.shape[1]

  use_prefill_config = m >= 8192
  if use_prefill_config:
    logging.info("Using prefill config for m = %d", m)
    kernel_config = PREFILL_CONFIG
  else:
    logging.info("Using decode config for m = %d", m)
    kernel_config = DECODE_CONFIG
  # Dequant WarpGroup
  _DEQ_WG = kernel_config.deq_wg
  # MMA + TMA
  _MAIN_WG = _DEQ_WG
  # Scale ACC and Store
  _STORE_WG = _MAIN_WG + 1

  if k_w != k_x:
    raise NotImplementedError(
        f"Contraction dim mismatch: weights.shape[1]={k_w}, x.shape[-1]={k_x}"
    )

  if group_sizes.shape != (num_groups,):
    raise NotImplementedError(
        "Expected group_sizes to have shape"
        f" {(num_groups,)} but got {group_sizes.shape}"
    )

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
      2 if collective else 1,
      block_k,
      x_dtype=lhs.qtype,
      w_dtype=rhs.qtype,
      ws_dtype=rhs.scale.dtype,
      xs_dtype=lhs.scale.dtype,
      xsum_dtype=jnp.float32,
      acc_dtype=jnp.float32,
      out_dtype=out_dtype,
      deq_wg=_DEQ_WG,
  )

  # data_stages, scale_stages, deq_stages, acc_stages = 4, 4, 2, 2
  logging.info(
      "data_stages: %d, scale_stages: %d, deq_stages: %d, acc_stages: %d",
      data_stages,
      scale_stages,
      deq_stages,
      acc_stages,
  )
  if deq_stages % _DEQ_WG != 0:
    raise ValueError(
        f"deq_stages ({deq_stages}) must be a multiple of _DEQ_WG ({_DEQ_WG})"
        " to avoid deadlocks."
    )
  if data_stages % _DEQ_WG != 0:
    raise ValueError(
        f"data_stages ({data_stages}) must be a multiple of _DEQ_WG"
        f" ({_DEQ_WG}) to avoid deadlocks."
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

  x, scales = lhs.qvalue, lhs.scale
  expected_d = k_x // tile_k
  if scales.shape[1] == 2 * expected_d:
    scales = scales.mT
    x_scales, x_sum = jnp.split(scales, 2, axis=0)
  else:
    x_scales = scales.mT
    x_sum = (
        x.astype(jnp.float32).reshape(m, -1, block_k).sum(-1).swapaxes(-1, -2)
    )

  tile_xk = k_x // x_scales.shape[0]

  if tile_xk != tile_k:
    raise NotImplementedError(f"tile must be equal, got: {tile_xk=} {tile_k=}.")

  m_iters = pl.cdiv(m, cluster_block_m) + num_groups - 1
  n_iters = pl.cdiv(n, cluster_block_n)
  align_tile = 8
  group_info = common.GroupInfo.create_aligned(
      group_sizes, cluster_block_m, m_iters, align_tile
  )
  def kernel(*refs, scoped):
    (
        x_gmem,
        x_sum_gmem,
        x_scales_gmem,
        w_gmem,
        w_scales_gmem,
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
        xs_smem,
        x_sum_smem,
        w_smem,
        ws_smem,
        w_tmem,
        out_smem,
        acc_tmem,
    ) = scratch_buffers
    (
        x_barrier,
        x_consumed_barrier,
        w_barrier,
        w_consumed_barrier,
        xws_barrier,
        xws_consumed_barrier,
        w_tmem_ready_barrier,
        w_tmem_consumed_barrier,
        acc_ready_barrier,
        acc_consumed_barrier,
    ) = barriers

    m, k = x_gmem.shape
    num_k_iters = pl.cdiv(k, block_k)
    cluster_idx = lax.axis_index("x")
    is_lead_block = cluster_idx == 0

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
      block_start = pl.multiple_of(block_start, align_tile)
      slice_m = pl.ds(block_start, cluster_block_m)
      slice_n = pl.ds(tid_n * cluster_block_n + cluster_idx * block_n, block_n)
      wg = jax.lax.axis_index("wg")

      @pl.when(actual_size > 0)
      def _body():
        @pl.when(wg == _MAIN_WG)
        def _mma_tma_wg():
          kernel_config.set_max_registers(kernel_config.main_wg_regs)

          @plgpu.warp_map
          def _per_warp(warp_id):
            if jax.__version_info__ < (0, 10, 0):
              warp_id = lax.rem(warp_id, 4)

            @pl.when(warp_id == _X_TMA_WARP)
            def x_tma_warp():
              def do_tma_x(ki, slot):
                plgpu.copy_gmem_to_smem(
                    x_gmem.at[
                        slice_m,
                        pl.ds(ki * block_k, block_k),
                    ],
                    x_smem.at[slot],
                    x_barrier.at[slot],
                    leader_tracked=plgpu.CopyPartition.PARTITIONED(0)
                    if collective
                    else None,
                    collective_axes="x" if collective else None,
                )

              @pl.loop(0, num_k_iters)
              def _x_tma_loop(ki):
                global_ki = ki + carry * num_k_iters
                slot = lax.rem(global_ki, data_stages)

                @pl.when(global_ki >= data_stages)
                def _wait_x_consumed():
                  plgpu.barrier_wait(x_consumed_barrier.at[slot])

                do_tma_x(ki, slot)

            @pl.when(warp_id == _W_TMA_WARP)
            def w_tma_warp():

              @pl.loop(0, num_k_iters)
              def loop_body(ki):
                global_ki = ki + carry * num_k_iters
                slot = lax.rem(global_ki, data_stages)

                @pl.when(global_ki >= data_stages)
                def _wait_w_consumed():
                  plgpu.barrier_wait(w_consumed_barrier.at[slot])

                # For 2CTA we still use a non-collective due to dequantization
                plgpu.copy_gmem_to_smem(
                    w_gmem.at[group_id, slice_n, pl.ds(ki * block_k, block_k)],
                    w_smem.at[slot],
                    w_barrier.at[slot],
                )

            @pl.when(warp_id == _SCALE_TMA_WARP)
            def ws_tma_warp():

              def do_tma_ws(ki, slot):
                plgpu.copy_gmem_to_smem(
                    w_scales_gmem.at[
                        group_id,
                        lax.div((ki * block_k), tile_k),
                        slice_n,
                    ],
                    ws_smem.at[slot],
                    xws_barrier.at[slot],
                )

              def do_tma_xs(ki, slot):
                plgpu.copy_gmem_to_smem(
                    x_scales_gmem.at[
                        lax.div(block_k * ki, tile_k),
                        pl.ds(block_start, max(cluster_block_m, 64)),
                    ],
                    xs_smem.at[slot],
                    xws_barrier.at[slot],
                )
                plgpu.copy_gmem_to_smem(
                    x_sum_gmem.at[
                        ki,
                        pl.ds(block_start, max(cluster_block_m, 64)),
                    ],
                    x_sum_smem.at[slot],
                    xws_barrier.at[slot],
                )

              @pl.loop(0, num_k_iters)
              def _tma_ws_loop(ki):
                global_ki = ki + carry * num_k_iters
                slot = lax.rem(global_ki, scale_stages)

                @pl.when(global_ki >= scale_stages)
                def _():
                  plgpu.barrier_wait(xws_consumed_barrier.at[slot])

                do_tma_ws(ki, slot)
                do_tma_xs(ki, slot)

            @pl.when((warp_id == _MMA_WARP) & is_lead_block)
            def mma_warp():

              @pl.loop(0, num_k_iters)
              def do_mma(ki):
                global_ki = ki + carry * num_k_iters
                x_slot = lax.rem(global_ki, data_stages)
                w_slot = lax.rem(global_ki, deq_stages)
                acc_slot = lax.rem(global_ki, acc_stages)

                @pl.when(global_ki >= acc_stages)
                def _():
                  with jax.named_scope("wait_acc_consumed"):
                    plgpu.barrier_wait(acc_consumed_barrier.at[acc_slot])

                with jax.named_scope("wait_x_tma"):
                  plgpu.barrier_wait(x_barrier.at[x_slot])
                with jax.named_scope("wait_w_deq"):
                  plgpu.barrier_wait(w_tmem_ready_barrier.at[w_slot])

                acc_slice = pl.ds(acc_slot * cluster_block_m, cluster_block_m)

                plgpu.tcgen05_mma(
                    acc_tmem.at[:, acc_slice],
                    w_tmem.at[:, pl.ds(w_slot * block_k, block_k)],
                    x_smem.at[x_slot].T,
                    w_tmem_consumed_barrier.at[w_slot],
                    accumulate=False,
                    collective_axis="x" if collective else None,
                )
                plgpu.tcgen05_commit_arrive(
                    acc_ready_barrier.at[acc_slot],
                    collective_axis="x" if collective else None,
                )

                plgpu.tcgen05_commit_arrive(
                    x_consumed_barrier.at[x_slot],
                    collective_axis="x" if collective else None,
                )

        @pl.when(wg < _DEQ_WG)
        def _deq_wg():
          kernel_config.set_max_registers(kernel_config.deq_wg_regs)

          @pl.loop(wg, num_k_iters, step=_DEQ_WG)
          def _dequantize(ki):
            with jax.named_scope("indices"):
              global_ki = ki + carry * num_k_iters
              data_slot = lax.rem(global_ki, data_stages)
              w_slot = lax.rem(global_ki, deq_stages)
            with jax.named_scope("wait_w_tma"):
              plgpu.barrier_wait(w_barrier.at[data_slot])

            tile_d = min(256, block_k)
            for di in range(block_k // tile_d):
              with jax.named_scope("load_w_smem"):
                w = plgpu.load(
                    w_smem.at[data_slot, :, pl.ds(di * tile_d, tile_d)],
                    (),
                    layout=_TMEM(8),
                    optimized=False,
                )
              if di == block_k // tile_d - 1:
                mgpu_lib.fence_async_shared_cta()
                plgpu.barrier_arrive(w_consumed_barrier.at[data_slot])

              with jax.named_scope("dequant"):
                w = mgpu_lib.int4_as_biased_f8e4m3fn(w, _TMEM(8))
              w = plgpu.layout_cast(w, _TMEM(4))
              if di == 0:

                @pl.when((global_ki >= deq_stages))
                def _():
                  plgpu.barrier_wait(w_tmem_consumed_barrier.at[w_slot])

              with jax.named_scope("store"):
                plgpu.async_store_tmem(
                    w_tmem.at[:, pl.ds(w_slot * block_k + di * tile_d, tile_d)],
                    w,
                )
            plgpu.commit_tmem()
            with jax.named_scope("arrive"):
              plgpu.barrier_arrive(w_tmem_ready_barrier.at[w_slot])

        @pl.when(wg == _STORE_WG)
        def _store_wg():
          kernel_config.set_max_registers(kernel_config.store_wg_regs)
          acc_dtype = jnp.float32
          acc_carry = plgpu.layout_cast(
              jnp.zeros((block_n, cluster_block_m), dtype=acc_dtype),
              _TCGEN05,
          )

          def _loop_body(ki, acc_carry):
            global_ki = ki + carry * num_k_iters
            scale_slot = lax.rem(global_ki, scale_stages)
            acc_slot = lax.rem(global_ki, acc_stages)

            with jax.named_scope("wait_scales"):
              plgpu.barrier_wait(xws_barrier.at[scale_slot])
            with jax.named_scope("[scale]load"):
              ws = plgpu.load(
                  ws_smem.at[scale_slot],
                  (),
                  layout=_TCGEN05_ROW,
                  optimized=True,
              ).astype(acc_dtype)
              x_scale = plgpu.load(
                  xs_smem,
                  (scale_slot, pl.ds(0, cluster_block_m)),
                  layout=_TCGEN05_COL,
                  optimized=True,
              ).astype(acc_dtype)

              x_sum = plgpu.load(
                  x_sum_smem,
                  (scale_slot, pl.ds(0, cluster_block_m)),
                  layout=_TCGEN05_COL,
                  optimized=True,
              ).astype(acc_dtype)
            mgpu_lib.fence_async_shared_cta()
            plgpu.barrier_arrive(xws_consumed_barrier.at[scale_slot])
            with jax.named_scope("wait_acc_ready"):
              plgpu.barrier_wait(acc_ready_barrier.at[acc_slot])
            with jax.named_scope("load_acc"):
              acc_slice = pl.ds(acc_slot * cluster_block_m, cluster_block_m)
              acc = plgpu.async_load_tmem(acc_tmem.at[:, acc_slice])
              plgpu.wait_load_tmem()
            plgpu.barrier_arrive(acc_consumed_barrier.at[acc_slot])

            with jax.named_scope("rescale_acc"):
              int_bias = plgpu.layout_cast(
                  lax.broadcast_in_dim(-8 * x_sum, acc.shape, [1]),
                  _TCGEN05,
              )
              acc = acc * 512 + int_bias
              return rescale_tcgen05_acc(
                  acc_carry,
                  acc,
                  ws,
                  x_scale,
              )

          acc_carry = lax.fori_loop(0, num_k_iters, _loop_body, acc_carry)
          if activation is not None:
            acc_carry = activation(acc_carry)

          with jax.named_scope("acc -> SMEM"):
            out_smem.T[...] = plgpu.layout_cast(
                acc_carry.astype(out_smem.dtype),
                plgpu.Layout.TCGEN05_TRANSPOSED,
            )
            plgpu.commit_smem()

          with jax.named_scope("SMEM -> GMEM"):
            # Write out the largest power of two rows first,
            # then the next largest, etc.
            # This allows us to coalesce writes as much as possible.
            offset = start_within_block
            size = 1 << (min(cluster_block_m, m).bit_length() - 1)
            while size > 0:

              @pl.when(actual_size & size != 0)
              def _():
                out_smem_slice = out_smem.at[pl.ds(offset, size)]
                o_gref_slice = out_gmem.at[
                    pl.ds(block_start + offset, size),
                    slice_n,
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
        (data_stages, block_n, block_k),
        dtype=w.dtype,
        transforms=(
            plgpu.TilingTransform((8, w_swizzle_elems)),
            plgpu.SwizzleTransform(w_swizzle),
        ),
    )
    w_tmem = plgpu.TMEM(
        (block_n, deq_stages * block_k),
        dtype=x_dtype,
        packed=True,
        collective=collective,
    )
    ws_smem = plgpu.SMEM(
        (scale_stages, block_n),
        dtype=w_scales.dtype,
    )
    xs_smem = plgpu.SMEM(
        (scale_stages, max(cluster_block_m, 64)),
        dtype=x_scales.dtype,
    )
    x_sum_smem = plgpu.SMEM(
        (scale_stages, max(cluster_block_m, 64)),
        dtype=x_sum.dtype,
    )
    out_smem = plgpu.SMEM(
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
    acc_tmem = plgpu.TMEM(
        (block_n, acc_stages * cluster_block_m),
        dtype=jnp.float32,
        collective=collective,
    )
    x_barrier = plgpu.Barrier(num_barriers=data_stages)
    x_consumed_barrier = plgpu.Barrier(
        num_barriers=data_stages, orders_tensor_core=True
    )
    w_barrier = plgpu.Barrier(num_barriers=data_stages)
    w_consumed_barrier = plgpu.Barrier(num_barriers=data_stages)
    xws_barrier = plgpu.Barrier(num_barriers=scale_stages, num_arrivals=3)
    xws_consumed_barrier = plgpu.Barrier(num_barriers=scale_stages)
    w_tmem_consumed_barrier = plgpu.Barrier(
        num_barriers=deq_stages, orders_tensor_core=True
    )
    acc_ready_barrier = plgpu.Barrier(
        num_barriers=acc_stages, orders_tensor_core=True
    )
    if collective:
      w_tmem_ready_barrier = plgpu.ClusterBarrier(
          num_barriers=deq_stages,
          collective_axes=("x",),
          orders_tensor_core=True,
          leader_tracked=True,
      )
      acc_consumed_barrier = plgpu.ClusterBarrier(
          num_barriers=acc_stages,
          collective_axes=("x",),
          orders_tensor_core=True,
          leader_tracked=True,
      )
    else:
      w_tmem_ready_barrier = plgpu.Barrier(
          num_barriers=deq_stages, orders_tensor_core=True
      )
      acc_consumed_barrier = plgpu.Barrier(
          num_barriers=acc_stages, orders_tensor_core=True
      )

    pl.run_scoped(
        lambda *args: kernel(*refs, scoped=args),
        (
            x_smem,
            xs_smem,
            x_sum_smem,
            w_smem,
            ws_smem,
            w_tmem,
            out_smem,
            acc_tmem,
        ),
        (
            x_barrier,
            x_consumed_barrier,
            w_barrier,
            w_consumed_barrier,
            xws_barrier,
            xws_consumed_barrier,
            w_tmem_ready_barrier,
            w_tmem_consumed_barrier,
            acc_ready_barrier,
            acc_consumed_barrier,
        ),
        collective_axes="wg",
    )

  profile = False
  num_sms = backend.get_default_device().core_count
  num_sms = num_sms // 2 if profile else num_sms
  f = plgpu.kernel(
      kernel_entry,
      out_type=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
      num_threads=_DEQ_WG + 2,
      thread_name="wg",
      grid=(num_sms // 2,) if collective else (num_sms,),
      grid_names=("sm",),
      cluster=(1 + collective,),
      cluster_names=("x",),
      kernel_name="ragged_dot_fp8_quant_sm100",
      compiler_params=plgpu.CompilerParams(
          approx_math=True,
          unsafe_no_auto_barriers=True,
          profile_space=250 if profile else 0,
          profile_dir="sponge" if profile else "",
      ),
  )
  return f(
      x,
      x_sum,
      x_scales,
      w,
      w_scales,
      group_info.group_id,
      group_info.start_within_block,
      group_info.actual_size,
      group_info.block_start,
  )
