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
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common


COMPUTE_WGS = 2
STORE_WG = COMPUTE_WGS
MEMORY_WG = COMPUTE_WGS + 1


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

  _, k = lhs.shape
  g, _, _ = rhs.shape
  w, w_scales, x = (rhs.qvalue.mT, rhs.scale, lhs)
  (_, n, k_w), (m, k_x) = w.shape, x.shape
  group_info = common.GroupInfo.create_aligned(
      group_sizes, config.block_m, pl.cdiv(m, config.block_m) + g - 1
  )
  block_m, block_n, block_k = config.block_m, config.block_n, config.block_k
  tile_k = k_w // w_scales.shape[1]

  if group_sizes.shape != (g,):
    raise ValueError(
        f"Expected group_sizes to have shape {(g,)} but got {group_sizes.shape}"
    )
  assert (
      n % (config.block_n * 2) == 0
  ), "n must be divisible by config.block_n * 2"

  if (
      len(rhs.scale_tile_shape) != 3
      or rhs.scale_tile_shape[0] != 1
      or rhs.scale_tile_shape[2] != 1
      or (rhs.scale_tile_shape[1] % config.block_k != 0)
  ):
    raise NotImplementedError(
        "Scaling tile is not supported got:"
        f" {rhs.scale_tile_shape=} (block_k={config.block_k})."
    )

  out_elem_bits = jnp.finfo(out_dtype).bits
  swizzle_out = plgpu.find_swizzle(out_elem_bits * block_n, "out")
  out_swizzle_elems = (swizzle_out * 8) // out_elem_bits

  if out_swizzle_elems != block_n:
    raise ValueError(
        f"Expected out_swizzle_elems ({out_swizzle_elems}) to equal"
        f" block_n ({block_n})"
    )
  num_stages = min(config.num_stages, k_x // block_k)

  def kernel(*refs, scoped):
    (
        x_gmem,
        w_gmem,
        w_scales_gmem,
        _,
        group_id_gmem,
        start_within_block_gmem,
        actual_size_gmem,
        block_start_gmem,
        out_gmem,
    ) = refs
    (scratch_buffers, barriers) = scoped
    x_smem, w_smem, w_scales_smem, o_smem = scratch_buffers
    (
        x_tma_barrier,
        w_tma_barrier,
        x_consume_barrier,
        w_consume_barrier,
        store_gmem_done_barrier,
        store_smem_done_barrier,
    ) = barriers
    num_k_iters = pl.cdiv(k, block_k)

    def mn_loop(m_offset, m_iters, n_iters, loop_info: plgpu.NDLoopInfo, carry):
      (lin_idx,) = loop_info.index
      m_index, n_index = plgpu.planar_snake(
          lin_idx,  # Linear index.
          (m_iters, n_iters),
          # The 2D iteration space.
          config.grid_minor_dim,
          # 0 or 1, indicates the fastest changing dim.
          config.grid_tile_width,
          # The width of tiles along the fastest changing dim.
      )
      tid_m = m_index + m_offset
      ni = n_index

      with jax.named_scope("load group_info"):
        group_id = group_id_gmem[tid_m]
        start_within_block = start_within_block_gmem[tid_m]
        actual_size = actual_size_gmem[tid_m]
        block_start = block_start_gmem[tid_m]
      wg = jax.lax.axis_index("wg")

      def do_tma_x(ki, slot):
        plgpu.copy_gmem_to_smem(
            x_gmem.at[
                pl.ds(block_start, block_m),
                pl.ds(ki * block_k, block_k),
            ],
            x_smem.at[slot],
            x_tma_barrier.at[slot],
        )

      def do_tma_w(ki, slot):
        plgpu.copy_gmem_to_smem(  # e,n,k
            w_gmem.at[
                group_id,
                pl.ds(ni * block_n * COMPUTE_WGS, block_n * COMPUTE_WGS),
                pl.ds(ki * block_k, block_k),
            ],
            w_smem.at[slot],
            w_tma_barrier.at[slot],
        )
        plgpu.copy_gmem_to_smem(  # e,k//t,n
            w_scales_gmem.at[
                group_id,
                jax.lax.div((ki * block_k), tile_k),
                pl.ds(ni * block_n * COMPUTE_WGS, block_n * COMPUTE_WGS),
            ],
            w_scales_smem.at[slot],
            w_tma_barrier.at[slot],
        )

      @pl.when(actual_size > 0)
      def _body():
        @pl.when(wg == MEMORY_WG)
        def _():
          plgpu.set_max_registers(80, action="decrease")

          def _iter(ki, _):
            stage = jax.lax.rem(ki, num_stages)

            @pl.when(jnp.logical_or(ki >= num_stages, carry > 0))
            def _():
              plgpu.barrier_wait(w_consume_barrier.at[stage])

            do_tma_w(ki, stage)

            @pl.when(jnp.logical_or(ki >= num_stages, carry > 0))
            def _():
              plgpu.barrier_wait(x_consume_barrier.at[stage])

            do_tma_x(ki, stage)

          lax.fori_loop(0, num_k_iters, _iter, None)

        @pl.when(wg < COMPUTE_WGS)
        def _():
          plgpu.set_max_registers(176, action="increase")

          def _func(acc_ref):
            def _iter(ki, acc_ref):
              stage = jax.lax.rem(ki, num_stages)
              with jax.named_scope("dequant"):
                w = common.dequant(
                    w_scales_smem.at[stage, pl.ds(wg * block_n, block_n)],
                    w_smem[stage, pl.ds(wg * block_n, block_n)],
                )
              with jax.named_scope("wait X"):
                plgpu.barrier_wait(x_tma_barrier.at[stage])
              with jax.named_scope("mma"):
                plgpu.wgmma(acc_ref, w, x_smem.at[stage].T)
              plgpu.barrier_arrive(w_consume_barrier.at[stage])

              @pl.when(ki + 1 < num_k_iters)
              def _():
                next_stage = jax.lax.rem(ki + 1, num_stages)
                with jax.named_scope("wait W"):
                  plgpu.barrier_wait(w_tma_barrier.at[next_stage])

              with jax.named_scope("wait MMA"):
                plgpu.wgmma_wait(0)
              plgpu.barrier_arrive(x_consume_barrier.at[stage])
              return acc_ref

            with jax.named_scope("wait W"):
              plgpu.barrier_wait(w_tma_barrier.at[0])
            acc_ref = lax.fori_loop(0, num_k_iters, _iter, acc_ref)
            return acc_ref

          acc = pl.run_scoped(
              lambda acc_ref: _func(acc_ref)[...], plgpu.ACC((block_n, block_m))
          )
          # acc -> o_smem
          with jax.named_scope("acc -> o_smem"):

            @pl.when(carry > 0)
            def _():
              plgpu.barrier_wait(store_gmem_done_barrier)

            wg_o_smem = o_smem.at[wg].reshape(block_m // 8, 1, 8, block_n)
            wg_o_smem = plgpu.untile_ref(wg_o_smem, (8, block_n))
            # Apply activation function to the output in dtype of the acc
            acc = (
                activation(acc) if activation is not None else acc
            )
            acc = plgpu.layout_cast(
                acc.astype(out_gmem.dtype), plgpu.Layout.WGMMA_TRANSPOSED
            )
            wg_o_smem.T[...] = acc
            plgpu.commit_smem()
            plgpu.barrier_arrive(store_smem_done_barrier)

        @pl.when(wg == STORE_WG)
        def _():
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
                for slot in range(COMPUTE_WGS):
                  o_smem_ = o_smem.at[slot, pl.ds(offset, size)]
                  o_gmem_ = out_gmem.at[
                      pl.ds(block_start + offset, size),
                      pl.ds((ni * COMPUTE_WGS + slot) * block_n, block_n),
                  ]
                  plgpu.copy_smem_to_gmem(o_smem_, o_gmem_, commit_group=False)

              offset += actual_size & size
              size //= 2
            plgpu.commit_smem_to_gmem_group()
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
            plgpu.barrier_arrive(store_gmem_done_barrier)

      return carry + (actual_size > 0)

    n_iters = pl.cdiv(n, config.block_n * COMPUTE_WGS)

    if config.persistent:
      # We stratify the grid: first emit a number of blocks that have
      # definitely work to do. Then schedule blocks that may be
      # noops. This way we lower the chances that noop bocks are
      # scheduled to the same SM.
      m0_iters = pl.cdiv(m, config.block_m)
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
      m_iters = pl.cdiv(m, config.block_m) + g - 1
      plgpu.nd_loop((m_iters * n_iters,), collective_axes="sm", init_carry=0)(
          functools.partial(mn_loop, 0, m_iters, n_iters)
      )

  def kernel_entry(*refs):
    swizzle = plgpu.find_swizzle(block_k * jnp.dtype(x.dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(x.dtype).itemsize
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )

    w_elem_bits = jnp.iinfo(w.dtype).bits
    try:
      w_swizzle = plgpu.find_swizzle(block_k * w_elem_bits)  # n,k
    except ValueError as e:
      raise NotImplementedError("No possible swizzle.") from e
    w_swizzle_elems = (w_swizzle * 8) // w_elem_bits
    try:
      quantized_transforms = (
          plgpu.TilingTransform((8, w_swizzle_elems)),
          plgpu.SwizzleTransform(w_swizzle),
      )
    except ValueError as e:
      raise NotImplementedError(
          f"{w_swizzle=} {w_swizzle_elems=} unsupported."
      ) from e

    x_smem = plgpu.SMEM(
        (num_stages, block_m, block_k),
        dtype=x.dtype,
        transforms=transforms,
    )
    w_smem = plgpu.SMEM(
        (num_stages, COMPUTE_WGS * block_n, block_k),
        dtype=w.dtype,
        transforms=quantized_transforms,
    )
    w_scales_smem = plgpu.SMEM(
        (num_stages, COMPUTE_WGS * block_n), dtype=w_scales.dtype
    )
    o_smem = plgpu.SMEM(
        (COMPUTE_WGS, block_m, block_n),
        dtype=out_dtype,
        transforms=(plgpu.SwizzleTransform(swizzle_out),),
    )
    x_tma_barrier = plgpu.Barrier(num_arrivals=1, num_barriers=num_stages)
    # w and scale
    w_tma_barrier = plgpu.Barrier(num_arrivals=2, num_barriers=num_stages)
    w_consume_barrier = plgpu.Barrier(
        num_arrivals=COMPUTE_WGS, num_barriers=num_stages
    )
    x_consume_barrier = plgpu.Barrier(
        num_arrivals=COMPUTE_WGS, num_barriers=num_stages
    )
    store_gmem_done_barrier = plgpu.Barrier(num_barriers=COMPUTE_WGS)
    store_smem_done_barrier = plgpu.Barrier(num_arrivals=COMPUTE_WGS)
    pl.run_scoped(
        lambda *args: kernel(*refs, scoped=args),
        (
            x_smem,
            w_smem,
            w_scales_smem,
            o_smem,
        ),
        (
            x_tma_barrier,
            w_tma_barrier,
            x_consume_barrier,
            w_consume_barrier,
            store_gmem_done_barrier,
            store_smem_done_barrier,
        ),
        collective_axes="wg",
    )

  num_sms = backend.get_default_device().core_count
  profile = False
  if profile:
    num_sms = 1
  f = plgpu.kernel(
      kernel_entry,
      out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
      num_threads=COMPUTE_WGS + 2,
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
