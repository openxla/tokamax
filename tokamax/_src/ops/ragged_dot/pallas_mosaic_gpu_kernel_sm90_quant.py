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
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array  # pylint: disable=g-multiple-import,g-importing-member
from jaxtyping import Float  # pylint: disable=g-multiple-import,g-importing-member
from jaxtyping import Integer  # pylint: disable=g-multiple-import,g-importing-member
import qwix
from tokamax._src import jaxtyping
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common


_COMPUTE_WGS = 2
_TMA_WG = _COMPUTE_WGS

_WGMMA = plgpu.Layout.WGMMA
_WGMMA_ROW = plgpu.Layout.WGMMA.reduce(1)
_WGMMA_TRANSPOSED = plgpu.Layout.WGMMA_TRANSPOSED
_WGMMA_UPCAST_2X = plgpu.Layout.WGMMA_UPCAST_2X
_WGMMA_UPCAST_4X = plgpu.Layout.WGMMA_UPCAST_4X


@jaxtyping.jaxtyped
def ragged_dot_quantized_kernel(
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

  m_iters = pl.cdiv(m, block_m) + g - 1
  n_iters = pl.cdiv(n, _COMPUTE_WGS * block_n)
  k_iters = pl.cdiv(k, block_k)

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

  # Before JAX 0.10, `barrier_arrive` and `commit_smem_to_gmem_group` are not
  # supported in a single-warp context.
  supports_warp_tma = jax.__version_info__ >= (0, 10, 0)

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
      w_scales_smem,
      o_smem,
      x_barrier,
      w_barrier,
      o_barrier,
      x_consumed_barrier,
      w_consumed_barrier,
      o_consumed_barrier,
  ):

    def mn_loop_body(m_offset, m_iters, loop_info: plgpu.NDLoopInfo, carry):
      (mni,) = loop_info.index
      mi, ni = plgpu.planar_snake(
          mni, (m_iters, n_iters), config.grid_minor_dim, config.grid_tile_width
      )
      mi += m_offset

      with jax.named_scope("load group_info"):
        gi = group_id_gmem[mi]
        start_within_block = start_within_block_gmem[mi]
        actual_size = actual_size_gmem[mi]
        block_start = block_start_gmem[mi]

      wg = lax.axis_index("wg")
      ms = pl.ds(block_start, block_m)
      ns = pl.ds(wg * block_n, block_n)

      @pl.when(actual_size > 0)
      def body():

        @pl.when(wg < _COMPUTE_WGS)
        def compute_wg():
          max_registers = 216 if supports_warp_tma else 176
          plgpu.set_max_registers(max_registers, action="increase")

          def compute_acc(acc):
            with jax.named_scope("wait W"):
              plgpu.barrier_wait(w_barrier.at[0])

            def k_loop_body(ki, w_scales):
              si = lax.rem(ki, num_stages)
              with jax.named_scope("dequant"):
                w_scale_bits = mgpu_lib.num_bits(w_scales.dtype)
                match w_scale_bits // mgpu_lib.num_bits(w_smem.dtype):
                  case 2:
                    w_layout = _WGMMA_UPCAST_2X
                  case 4:
                    w_layout = _WGMMA_UPCAST_4X
                  case _:
                    w_layout = _WGMMA
                w = plgpu.load(w_smem, (si, ns), layout=w_layout)
                plgpu.barrier_arrive(w_consumed_barrier.at[si])
                w = plgpu.layout_cast(w, _WGMMA).astype(w_scales.dtype)
                w *= lax.broadcast_in_dim(w_scales, w.shape, [0])
              with jax.named_scope("wait X"):
                plgpu.barrier_wait(x_barrier.at[si])
              with jax.named_scope("mma"):
                plgpu.wgmma(acc, w, x_smem.at[si].T)

              def load_scales():
                si = lax.rem(ki + 1, num_stages)
                with jax.named_scope("wait W"):
                  plgpu.barrier_wait(w_barrier.at[si])
                return plgpu.load(w_scales_smem, (si, ns), layout=_WGMMA_ROW)

              carry = lax.cond(ki + 1 < k_iters, load_scales, lambda: w_scales)

              with jax.named_scope("wait MMA"):
                plgpu.wgmma_wait(0)
              plgpu.barrier_arrive(x_consumed_barrier.at[si])
              return carry

            w_scales0 = plgpu.load(w_scales_smem, (0, ns), layout=_WGMMA_ROW)
            lax.fori_loop(0, k_iters, k_loop_body, init_val=w_scales0)
            return acc[...]

          acc = pl.run_scoped(compute_acc, plgpu.ACC((block_n, block_m)))

          with jax.named_scope("acc -> o_smem"):
            pl.when(carry > 0)(lambda: plgpu.barrier_wait(o_consumed_barrier))
            o_smem_ = o_smem.at[wg].reshape(block_m // 8, 1, 8, block_n)
            o_smem_ = plgpu.untile_ref(o_smem_, (8, block_n))
            acc = acc if activation is None else activation(acc)
            acc = acc.astype(o_smem_.dtype)
            o_smem_.T[...] = plgpu.layout_cast(acc, _WGMMA_TRANSPOSED)
            plgpu.barrier_arrive(o_barrier)

        def issue_o_tma():
          plgpu.barrier_wait(o_barrier)
          mgpu_lib.fence_async_shared_cta()  # Ensure store is complete.

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
                  plgpu.copy_smem_to_gmem(
                      o_smem.at[wg_, pl.ds(offset, size)],
                      out_gmem.at[pl.ds(block_start + offset, size), ns],
                      commit_group=False,
                  )

              offset += actual_size & size
              size //= 2
            plgpu.commit_smem_to_gmem_group()
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
            plgpu.barrier_arrive(o_consumed_barrier)

        if supports_warp_tma:

          @pl.when(wg == _TMA_WG)
          def tma_wg():
            plgpu.set_max_registers(72, action="decrease")

            @plgpu.warp_map
            def tma_warps(warp_id):

              @pl.when(warp_id == 0)
              def w_tma_warp():
                ns = pl.ds(ni * (_COMPUTE_WGS * block_n), _COMPUTE_WGS * block_n)

                @pl.loop(0, k_iters)
                def k_loop(ki):
                  ks = pl.ds(ki * block_k, block_k)
                  si = lax.rem(ki, num_stages)

                  @pl.when((ki >= num_stages) | (carry > 0))
                  def wait_w_consumed():
                    plgpu.barrier_wait(w_consumed_barrier.at[si])
                    mgpu_lib.fence_async_shared_cta()  # Ensure read complete.

                  plgpu.copy_gmem_to_smem(  # e,n,k
                      w_gmem.at[gi, ns, ks], w_smem.at[si], w_barrier.at[si]
                  )
                  ki_ = lax.div(ki, w_scales_tile_k // block_k)
                  plgpu.copy_gmem_to_smem(  # e,k//t,n
                      w_scales_gmem.at[gi, ki_, ns],
                      w_scales_smem.at[si],
                      w_barrier.at[si],
                  )

              @pl.when(warp_id == 1)
              def x_tma_warp():

                @pl.loop(0, k_iters)
                def k_loop(ki):
                  ks = pl.ds(ki * block_k, block_k)
                  si = lax.rem(ki, num_stages)

                  @pl.when((ki >= num_stages) | (carry > 0))
                  def wait_x_consumed():
                    plgpu.barrier_wait(x_consumed_barrier.at[si])

                  end = start_within_block + actual_size

                  # Load the smallest power-of-two size that includes all of the
                  # rows we need.
                  def cp_rec(size):
                    cp = lambda: plgpu.copy_gmem_to_smem(
                        x_gmem.at[pl.ds(block_start, size), ks],
                        x_smem.at[si, :size],
                        x_barrier.at[si],
                    )
                    if size == block_m:
                      return cp()
                    return lax.cond(end <= size, cp, lambda: cp_rec(size * 2))

                  cp_rec(8)

              @pl.when(warp_id == 2)
              def o_tma_warp():
                issue_o_tma()

        else:

          @pl.when(wg == _TMA_WG)
          def xw_tma_wg():
            plgpu.set_max_registers(80, action="decrease")

            ns = pl.ds(ni * (_COMPUTE_WGS * block_n), _COMPUTE_WGS * block_n)

            @pl.loop(0, k_iters)
            def k_loop(ki):
              ks = pl.ds(ki * block_k, block_k)
              si = lax.rem(ki, num_stages)

              @pl.when((ki >= num_stages) | (carry > 0))
              def wait_w_consumed():
                plgpu.barrier_wait(w_consumed_barrier.at[si])
                mgpu_lib.fence_async_shared_cta()  # Ensure read is complete.

              plgpu.copy_gmem_to_smem(  # e,n,k
                  w_gmem.at[gi, ns, ks], w_smem.at[si], w_barrier.at[si]
              )
              ki_ = lax.div(ki, w_scales_tile_k // block_k)
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

          @pl.when(wg == _TMA_WG + 1)
          def o_tma_wg():
            plgpu.set_max_registers(64, action="decrease")
            issue_o_tma()

      return carry + (actual_size > 0)

    if config.persistent:
      # We stratify the grid: first emit a number of blocks that have definitely
      # work to do. Then schedule blocks that may be no-ops. This way we lower
      # the chances that no-op blocks are scheduled to the same SM.
      def run_mn_loop(m_offset, m_iters, carry=0):
        return plgpu.nd_loop(
            (m_iters * n_iters,), collective_axes="sm", init_carry=carry
        )(functools.partial(mn_loop_body, m_offset, m_iters))

      m0_iters = pl.cdiv(m, block_m)
      carry = run_mn_loop(0, m0_iters)
      _ = run_mn_loop(m0_iters, m_iters - m0_iters, carry)
    else:
      mni = lax.axis_index("mn")
      loop_info = plgpu.NDLoopInfo((mni,), local_index=0, num_local_steps=1)
      mn_loop_body(0, m_iters, loop_info, carry=0)

  out_elem_bits = jnp.finfo(out_dtype).bits
  swizzle_out = plgpu.find_swizzle(out_elem_bits * block_n, "out")
  out_swizzle_elems = (swizzle_out * 8) // out_elem_bits
  if out_swizzle_elems != block_n:
    raise NotImplementedError(f"{out_swizzle_elems=} must equal {block_n=}")

  def tiled_smem(*args):
    try:
      return mgpu_lib.tiled_swizzled_smem(*args)
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
      x_barrier=plgpu.Barrier(num_barriers=num_stages),
      w_barrier=plgpu.Barrier(num_barriers=num_stages, num_arrivals=2),
      o_barrier=plgpu.Barrier(num_arrivals=_COMPUTE_WGS),
      x_consumed_barrier=plgpu.Barrier(
          num_arrivals=_COMPUTE_WGS, num_barriers=num_stages
      ),
      w_consumed_barrier=plgpu.Barrier(
          num_arrivals=_COMPUTE_WGS, num_barriers=num_stages
      ),
      o_consumed_barrier=plgpu.Barrier(),
  )

  if config.persistent:
    grid = (backend.get_default_device().core_count,)
    grid_names = ("sm",)
  else:
    grid = (m_iters * n_iters,)
    grid_names = ("mn",)

  profile = False
  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
      scratch_shapes=scratch_shapes,
      num_threads=_COMPUTE_WGS + (1 if supports_warp_tma else 2),
      thread_name="wg",
      grid=grid,
      grid_names=grid_names,
      kernel_name="ragged_dot_quant_sm90",
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
      group_info.group_id,
      group_info.start_within_block,
      group_info.actual_size,
      group_info.block_start,
  )
