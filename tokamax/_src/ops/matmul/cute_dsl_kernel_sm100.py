# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils

from cutlass.jax import cutlass_call, jax_to_cutlass_dtype
from tokamax._src import jaxtyping
from tokamax._src.ops.matmul import cute_dsl_common as common


@jaxtyping.jaxtyped
def matmul_kernel(
    a: Float[Array, "M K"],
    b: Float[Array, "N K"],
    out_dtype: jnp.dtype,
    config: common.Config,
) -> Float[Array, "M N"]:

  m, k_a = a.shape
  n, k_b = b.shape

  if k_a != k_b:
    raise ValueError(
        f"Contraction dim mismatch: {a.shape[1]=}, {b.shape[1]=}."
    )

  if (a.dtype, b.dtype, out_dtype) != 3 * (jnp.float16,):
    raise ValueError(
        "Only float16 x float16 -> float16 is supported, got:"
        f" {a.dtype=} {b.dtype=} {out_dtype=}."
    )

  io_dtype = jax_to_cutlass_dtype(jnp.float16)
  acc_dtype = jax_to_cutlass_dtype(jnp.float32)
  mma_inst_shape_mnk = (128, config.block_m, 16)
  mma_tiler_mnk = (128, config.block_m, config.block_k)
  threads_per_cta = 128

  # Pipeline stage configuration
  ab_stages = config.num_ab_stages
  acc_stages = config.num_acc_stages


  @cute.struct
  class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stages * 2]
    tmem_holding_buf: cutlass.Int32


  @cute.kernel
  def kernel(
      tiled_mma: cute.TiledMma,
      tma_atom_a: cute.CopyAtom,
      mA_mkl: cute.Tensor,
      tma_atom_b: cute.CopyAtom,
      mB_nkl: cute.Tensor,
      mC_mnl: cute.Tensor,
      a_smem_layout: cute.ComposedLayout,
      b_smem_layout: cute.ComposedLayout,
  ):
    # Current thread/warp/block coordinates
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    #
    # 1. Prepare args
    #

    # Allocate SMEM
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    # Allocate all TMEM columns
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    num_tmem_cols = 512
    tmem.allocate(num_tmem_cols)

    # Prefetch tma descriptor
    if warp_idx == 0:
      cpasync.prefetch_descriptor(tma_atom_a)
      cpasync.prefetch_descriptor(tma_atom_b)

    # Pipeline configuration
    num_tma_copy_bytes = cute.size_in_bytes(
        io_dtype, cute.select(
            a_smem_layout, mode=[0, 1, 2])) + cute.size_in_bytes(
                io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=num_tma_copy_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,
                                                 threads_per_cta),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    # Partition tensors for MMA and make fragments
    # (bM, bK, RestK)
    gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    # (bN, bK, RestK)
    gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    # (bM, bN)
    gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
    thr_mma = tiled_mma.get_slice(0)
    # (MMA, MMA_M, MMA_K, RestK)
    tCgA = thr_mma.partition_A(gA)
    # (MMA, MMA_N, MMA_K, RestK)
    tCgB = thr_mma.partition_B(gB)
    # (MMA, MMA_M, MMA_N)
    tCgC = thr_mma.partition_C(gC)
    # (MMA, MMA_M, MMA_K, STAGE)
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K, STAGE)
    tCrB = tiled_mma.make_fragment_B(sB)
    # (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # (MMA, MMA_M, MMA_N)
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
    # Partition tensors for TMA; This requires the tensors partitioned for MMA
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    # CTA-wide sync before retrieving the pointer to the start of the allocated TMEM
    # Only warp 0 does the allocation so we need to sync before retrieving the TMEM start address
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    # Swap the pointer in tCtAcc
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    subtile_cnt = 4
    # (EpiTile)
    epi_tiler = ((
        cute.size(tCtAcc, mode=[0, 0]),
        cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt,
    ),)
    # (EpiTile, NumTiles)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    # (EpiTile, NumTiles)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    # Every thread loads 32x128 bits
    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        cutlass.Float32,
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

    # (TmemCpy,NumTmemCpy,NumTiles)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    # (TmemCpy,NumTmemCpy,NumTiles)
    tDgC = tmem_thr_copy.partition_D(gC_epi)

    # (TmemCpy,NumTmemCpy)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    # (TmemCpy,NumTmemCpy)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)

    #
    # 2. Main loop
    #
    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
      # Wait for a empty accumulator buffer
      acc_empty = acc_producer.acquire_and_advance()
      for k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 2):
        # Issue TMA loads
        ab_empty = ab_producer.acquire_and_advance()
        cute.copy(
            tma_atom_a,
            tAgA[(None, ab_empty.count)],
            tAsA[(None, ab_empty.index)],
            tma_bar_ptr=ab_empty.barrier,
        )
        cute.copy(
            tma_atom_b,
            tBgB[(None, ab_empty.count)],
            tBsB[(None, ab_empty.index)],
            tma_bar_ptr=ab_empty.barrier,
        )

        # Execute one K-block worth of MMA instructions
        ab_full = ab_consumer.wait_and_advance()
        num_k_blocks = cute.size(tCrA, mode=[2])
        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
          k_block_coord = (None, None, k_block_idx, ab_full.index)
          cute.gemm(
              tiled_mma,
              tCtAcc,
              tCrA[k_block_coord],
              tCrB[k_block_coord],
              tCtAcc,
          )
          tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        # Signal that the A/B buffers have been consumed and are ready for the next load
        ab_full.release()

      # Signal that the accumulator is fully computed
      acc_empty.commit()

    #
    # 3. Epilogue
    #

    # Release TMEM allocation lock
    tmem.relinquish_alloc_permit()

    # Wait for the accumulator buffer to be full
    acc_full = acc_consumer.wait_and_advance()

    # TMEM -> RMEM -> GEMM
    # Sub-tiling for better instruction-level parallelism
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
      cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
      tCrC.store(tCrAcc.load().to(io_dtype))
      cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_full.release()

    # Deallocate TMEM
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


  @cute.jit
  def host_function(
      stream,
      a: cute.Tensor,
      b: cute.Tensor,
      c: cute.Tensor,
  ):
    # Construct tiled MMA
    op = tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)

    # Construct SMEM layouts for A and B
    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        a.element_type,
        ab_stages,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        b.element_type,
        ab_stages,
    )
    a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

    # Construct TMA load atoms
    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op,
        a,
        a_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
    )
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        op,
        b,
        b_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
    )

    # Pretty prints kernel attributes useful for debugging
    # print(f"a            = {cute.pretty_str(a)}")
    # print(f"b            = {cute.pretty_str(b)}")
    # print(f"c            = {cute.pretty_str(c)}")
    # print(f"tiled_mma    = {cute.pretty_str(tiled_mma)}")
    # print(f"a_tma_atom   = {cute.pretty_str(a_tma_atom)}")
    # print(f"b_tma_atom   = {cute.pretty_str(b_tma_atom)}")
    # print(f"a_tma_tensor = {cute.pretty_str(a_tma_tensor)}")
    # print(f"b_tma_tensor = {cute.pretty_str(b_tma_tensor)}")

    # Launch the kernel
    grid_shape = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])
    kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        c,
        a_smem_layout,
        b_smem_layout,
    ).launch(
        stream=stream,
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
    )

  f = cutlass_call(
      host_function,
      input_mode=((0, 1), (0, 1)),
      output_mode=((0, 1),),
      output_shape_dtype=jax.ShapeDtypeStruct((m, n), dtype=out_dtype),
  )
  return f(a, b)
