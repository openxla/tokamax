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
"""Ragged dot Pallas-Mosaic-GPU Quantized Kernel."""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array  # pylint: disable=g-multiple-import,g-importing-member
from jaxtyping import Float  # pylint: disable=g-multiple-import,g-importing-member
from jaxtyping import Integer  # pylint: disable=g-multiple-import,g-importing-member
import qwix
from tokamax._src import jaxtyping
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common


_COMPUTE_WGS = 2
_EPILOGUE_WG = _COMPUTE_WGS
_MEMORY_WG = _COMPUTE_WGS + 1

_WGMMA_ROW = plgpu.Layout.WGMMA.reduce(1)
_WGMMA_TRANSPOSED = plgpu.Layout.WGMMA_TRANSPOSED


@jaxtyping.jaxtyped
def ragged_dot_quantized_ws_async_store_kernel(
    lhs: Float[Array, "M K"],
    rhs: Float[qwix.QArray, "G K N"],
    group_sizes: Integer[Array, "G"],
    out_dtype: jnp.dtype,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
) -> Float[Array, "M N"]:
  """Returns the Pallas kernel for quantized ragged dot.

  There are 4 Warp Group in this kernel:
    COMPUTE_WGS(2): dequant + MMA
      | load lhs,rhs -> dequant -> MMA | ... | Reg -> SMEM |
      | load lhs,rhs -> dequant -> MMA | ... | Reg -> SMEM |
    MEMORY_WG(1): issue TMA for loading lhs, rhs from HBM to SMEM.
      | wait for x, w consumed -> issue TMA | ...
    STORE_WG(1): store the result from SMEM to HBM. It can be overlapped with
      | wait for SMEM ready -> SMEM -> HBM |
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

  m, k = lhs.shape
  g, _, n = rhs.shape
  w, w_scales, x = (rhs.qvalue.mT, rhs.scale, lhs)
  block_m, block_n, block_k = config.block_m, config.block_n, config.block_k

  if n % (2 * block_n) != 0:
    raise NotImplementedError(f"{n=} must be divisible by {2 * block_n=}")

  w_scales_tile_g, w_scales_tile_k, w_scales_tile_n = rhs.scale_tile_shape
  w_scales_tile_k_rem = w_scales_tile_k % block_k

  if (w_scales_tile_g, w_scales_tile_k_rem, w_scales_tile_n) != (1, 0, 1):
    raise NotImplementedError(
        f"Scales tile is not supported got: {rhs.scale_tile_shape=} {block_k=}."
    )

  group_info = common.GroupInfo.create_aligned(
      group_sizes, config.block_m, pl.cdiv(m, config.block_m) + g - 1
  )
  num_stages = min(config.num_stages, k // block_k)

  def kernel(
      x_gmem,
      w_gmem,
      w_scales_gmem,
      _,
      group_id_gmem,
      start_within_block_gmem,
      actual_size_gmem,
      block_start_gmem,
      out_gmem,
      *,
      x_smem,
      w_smem,
      w_scales_smem,
      o_smem,
      x_barrier,
      w_barrier,
      x_consumed_barrier,
      w_consumed_barrier,
      store_gmem_done_barrier,
      store_smem_done_barrier,
  ):
    k_iters = pl.cdiv(k, block_k)

    def mn_loop(m_offset, m_iters, n_iters, loop_info: plgpu.NDLoopInfo, carry):
      (lin_idx,) = loop_info.index
      mi, ni = plgpu.planar_snake(
          lin_idx,
          (m_iters, n_iters),
          config.grid_minor_dim,
          config.grid_tile_width,
      )
      tid_m = mi + m_offset

      with jax.named_scope("load group_info"):
        gi = group_id_gmem[tid_m]
        start_within_block = start_within_block_gmem[tid_m]
        actual_size = actual_size_gmem[tid_m]
        block_start = block_start_gmem[tid_m]

      wg = jax.lax.axis_index("wg")
      ms = pl.ds(block_start, block_m)

      @pl.when(actual_size > 0)
      def body():
        @pl.when(wg == _MEMORY_WG)
        def memory_wg():
          plgpu.set_max_registers(80, action="decrease")
          ns = pl.ds(ni * (_COMPUTE_WGS * block_n), _COMPUTE_WGS * block_n)

          @pl.loop(0, k_iters)
          def k_loop(ki):
            ks = pl.ds(ki * block_k, block_k)
            si = jax.lax.rem(ki, num_stages)

            @pl.when((ki >= num_stages) | (carry > 0))
            def wait_w_consumed():
              plgpu.barrier_wait(w_consumed_barrier.at[si])

            plgpu.copy_gmem_to_smem(  # e,n,k
                w_gmem.at[gi, ns, ks], w_smem.at[si], w_barrier.at[si]
            )
            ki_ = jax.lax.div(ki, w_scales_tile_k // block_k)
            plgpu.copy_gmem_to_smem(  # e,k//t,n
                w_scales_gmem.at[gi, ki_, ns],
                w_scales_smem.at[si],
                w_barrier.at[si],
            )

            @pl.when((ki >= num_stages) | (carry > 0))
            def wait_x_consumed():
              plgpu.barrier_wait(x_consumed_barrier.at[si])

            plgpu.copy_gmem_to_smem(
                x_gmem.at[ms, ks], x_smem.at[si], x_barrier.at[si]
            )

        @pl.when(wg < _COMPUTE_WGS)
        def compute_wg():
          plgpu.set_max_registers(176, action="increase")

          def compute_acc(acc):
            with jax.named_scope("wait W"):
              plgpu.barrier_wait(w_barrier.at[0])

            @pl.loop(0, k_iters)
            def k_loop(ki):
              si = jax.lax.rem(ki, num_stages)
              with jax.named_scope("dequant"):
                idx = (si, pl.ds(wg * block_n, block_n))
                w = w_smem[idx].astype(w_scales_smem.dtype)
                w_scales = plgpu.load(w_scales_smem, idx, layout=_WGMMA_ROW)
                w *= jax.lax.broadcast_in_dim(w_scales, w.shape, [0])
              with jax.named_scope("wait X"):
                plgpu.barrier_wait(x_barrier.at[si])
              with jax.named_scope("mma"):
                plgpu.wgmma(acc, w, x_smem.at[si].T)
              plgpu.barrier_arrive(w_consumed_barrier.at[si])

              @pl.when(ki + 1 < k_iters)
              def _():
                si_next = jax.lax.rem(ki + 1, num_stages)
                with jax.named_scope("wait W"):
                  plgpu.barrier_wait(w_barrier.at[si_next])

              with jax.named_scope("wait MMA"):
                plgpu.wgmma_wait(0)
              plgpu.barrier_arrive(x_consumed_barrier.at[si])

            return acc[...]

          acc = pl.run_scoped(compute_acc, plgpu.ACC((block_n, block_m)))

          with jax.named_scope("acc -> o_smem"):

            @pl.when(carry > 0)
            def _():
              plgpu.barrier_wait(store_gmem_done_barrier)

            o_smem_ = o_smem.at[wg].reshape(block_m // 8, 1, 8, block_n)
            o_smem_ = plgpu.untile_ref(o_smem_, (8, block_n))
            acc = acc if activation is None else activation(acc)
            acc = acc.astype(o_smem_.dtype)
            o_smem_.T[...] = plgpu.layout_cast(acc, _WGMMA_TRANSPOSED)
            plgpu.commit_smem()
            plgpu.barrier_arrive(store_smem_done_barrier)

        @pl.when(wg == _EPILOGUE_WG)
        def epilogue_wg():
          plgpu.set_max_registers(64, action="decrease")
          plgpu.barrier_wait(store_smem_done_barrier)

          with jax.named_scope("store"):
            # Write out the largest power of two rows first,
            # then the next largest, etc. This allows us to coalesce
            # writes as much as possible.
            offset = start_within_block
            size = 1 << (min(block_m, m).bit_length() - 1)
            while size > 0:

              @pl.when(actual_size & size != 0)
              def _():
                for wg_ in range(_COMPUTE_WGS):
                  ns = pl.ds((ni * _COMPUTE_WGS + wg_) * block_n, block_n)
                  o_smem_ = o_smem.at[wg_, pl.ds(offset, size)]
                  o_gmem_ = out_gmem.at[pl.ds(block_start + offset, size), ns]
                  plgpu.copy_smem_to_gmem(o_smem_, o_gmem_, commit_group=False)

              offset += actual_size & size
              size //= 2
            plgpu.commit_smem_to_gmem_group()
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
            plgpu.barrier_arrive(store_gmem_done_barrier)

      return carry + (actual_size > 0)

    n_iters = pl.cdiv(n, _COMPUTE_WGS * block_n)

    if config.persistent:
      # We stratify the grid: first emit a number of blocks that have definitely
      # work to do. Then schedule blocks that may be no-ops. This way we lower
      # the chances that no-op blocks are scheduled to the same SM.
      m0_iters = pl.cdiv(m, block_m)
      carry = plgpu.nd_loop(
          (m0_iters * n_iters,), collective_axes="sm", init_carry=0
      )(functools.partial(mn_loop, 0, m0_iters, n_iters))
      m1_iters = g - 1
      plgpu.nd_loop(
          (m1_iters * n_iters,), collective_axes="sm", init_carry=carry
      )(
          functools.partial(mn_loop, m0_iters, m1_iters, n_iters),
      )
    else:
      m_iters = pl.cdiv(m, block_m) + g - 1
      plgpu.nd_loop((m_iters * n_iters,), collective_axes="sm", init_carry=0)(
          functools.partial(mn_loop, 0, m_iters, n_iters)
      )

  out_elem_bits = jnp.finfo(out_dtype).bits
  swizzle_out = plgpu.find_swizzle(out_elem_bits * block_n, "out")
  out_swizzle_elems = (swizzle_out * 8) // out_elem_bits
  if out_swizzle_elems != block_n:
    raise NotImplementedError(f"{out_swizzle_elems=} must equal {block_n=}")

  def tiled_smem(*args):
    try:
      return common.tiled_swizzled_smem(*args)
    except ValueError as e:
      raise NotImplementedError from e

  scratch_shapes = dict(
      x_smem=tiled_smem((num_stages, block_m, block_k), x.dtype, "x"),
      w_smem=tiled_smem(
          (num_stages, _COMPUTE_WGS * block_n, block_k), w.dtype, "w"
      ),
      w_scales_smem=plgpu.SMEM(
          (num_stages, _COMPUTE_WGS * block_n), w_scales.dtype
      ),
      o_smem=plgpu.SMEM(
          (_COMPUTE_WGS, block_m, block_n),
          dtype=out_dtype,
          transforms=(plgpu.SwizzleTransform(swizzle_out),),
      ),
      x_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=num_stages),
      w_barrier=plgpu.Barrier(num_arrivals=2, num_barriers=num_stages),
      x_consumed_barrier=plgpu.Barrier(
          num_arrivals=_COMPUTE_WGS, num_barriers=num_stages
      ),
      w_consumed_barrier=plgpu.Barrier(
          num_arrivals=_COMPUTE_WGS, num_barriers=num_stages
      ),
      store_gmem_done_barrier=plgpu.Barrier(num_barriers=_COMPUTE_WGS),
      store_smem_done_barrier=plgpu.Barrier(num_arrivals=_COMPUTE_WGS),
  )

  num_sms = backend.get_default_device().core_count
  profile = False
  if profile:
    num_sms = 1
  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
      scratch_shapes=scratch_shapes,
      num_threads=_COMPUTE_WGS + 2,
      thread_name="wg",
      grid=(num_sms,),
      grid_names=("sm",),
      compiler_params=plgpu.CompilerParams(
          approx_math=True,
          unsafe_no_auto_barriers=True,
          profile_space=20 if profile else 0,
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
