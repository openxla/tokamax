# Copyright 2026 Google LLC
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

"""Main kernel module for orchestrating DMA and computation pipeline."""

import dataclasses
import functools

from jax import lax
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.gmm import compute
from tokamax._src.ops.experimental.routing_moe.gmm import config
from tokamax._src.ops.experimental.routing_moe.gmm import memory_ref
from tokamax._src.ops.experimental.routing_moe.gmm import metadata


def inner_pipeline(
    in_ref: memory_ref.PipelineInputs,
    out_ref: hijax.TransformedRef,
    *,
    x_bref: memory_ref.RoutingBufferedRef,
    out_bref: memory_ref.RoutingBufferedRef,
    metadata_ref: metadata.MetadataRef,
    cfg: config.RoutingGmmConfig,
) -> None:
  """Orchestrates DMA and computation for a single pipeline step.

  Args:
    in_ref: Current VMEM slot references for the pipelined inputs.
    out_ref: Current VMEM slot reference for the output activation.
    x_bref: BufferedRef for the input activation used for manual DMA scheduling.
    out_bref: BufferedRef for the output activation used for manual DMA
      scheduling.
    metadata_ref: SMEM metadata reference for routing and tile row counts.
    cfg: Static configuration for the routing GMM kernel.
  """

  gm_id = pl.program_id(1)

  x_ref = pltpu.annotate(in_ref.x, no_hazard_no_deps=True)
  out_ref = pltpu.annotate(out_ref, no_hazard_no_deps=True)

  def _bucket_fn(bucket_m: int) -> None:
    # Step 1: Wait DMA for the current step slot.
    x_bref.branchless_wait_in(gm_id)
    out_bref.branchless_wait_out(gm_id - cfg.plan.num_xo_buffers)

    # Step 2: Start DMA for previous step slot.
    # NOTE: While we typically issue DMA for the other slots before waiting for
    # current slot, that will prevent DMA instructions to be pipelined with
    # compute.  Therefore, DMA is issued after the waiting.
    x_bref.branchless_copy_in(gm_id + cfg.plan.num_xo_buffers - 1)
    out_bref.branchless_copy_out(gm_id - 1)

    # Step 3: Load data from VMEM.
    x = layout_util.load_compact_to_large(x_ref.at[:bucket_m], cfg.dim.size_k)
    weight = in_ref.w.get_weight()

    # TODO: Potentially move this branching lgoci into matmul.
    weight_scale = None
    if cfg.w.has_scale:
      weight_scale = in_ref.w.get_scale()

    # Step 4: Perform computation.
    out = compute.matmul(cfg, x, weight, weight_scale, None)
    out = compute.apply_act_fn(out, cfg.fuse_act).astype(out_ref.dtype)

    # Step 5: Store output into VMEM.
    layout_util.store_large_to_compact(out_ref.at[:bucket_m], out)

  # Select which bucket to choose based on number of active rows of given tile.
  num_rows = metadata_ref.get_num_rows(gm_id)
  bucket_idx = pl.cdiv(
      jnp.maximum(num_rows - cfg.plan.bucket_base, 0), cfg.plan.bucket_base
  )

  lax.switch(
      bucket_idx,
      [
          functools.partial(_bucket_fn, bucket_m)
          for bucket_m in cfg.plan.bucket_rows
      ],
  )


def outer_pipeline(
    metadata_ref: metadata.MetadataRef,
    in_hbm_ref: memory_ref.PipelineInputs[hijax.TransformedRef],
    out_hbm_ref: hijax.TransformedRef,
    *,
    cfg: config.RoutingGmmConfig,
) -> None:
  """Allocates scoped buffers and schedules the routing GMM pipeline.

  Args:
    metadata_ref: SMEM metadata reference for routing and tile scheduling.
    in_hbm_ref: HBM references for the pipelined kernel inputs.
    out_hbm_ref: HBM reference for the kernel output.
    cfg: Static configuration for the routing GMM kernel.
  """

  num_gm = metadata_ref.fill_metadata_and_return_num_gm()

  allocs = memory_ref.PipelineAllocs.create(
      in_hbm_ref, out_hbm_ref, metadata_ref, cfg
  )
  in_hbm_ref = in_hbm_ref.transform_for_pipeline(cfg)

  @functools.partial(pl.run_scoped, final_allocs=allocs)
  def _run(final_allocs: memory_ref.PipelineAllocs):

    inputs = final_allocs.inputs
    out = final_allocs.out
    gather_idx = inputs.gather_idx
    scatter_idx = inputs.scatter_idx
    if isinstance(gather_idx, memory_ref.IndexBufferedRef):
      # Prefetch num_xo_buffers - 1 worth of indices into gather SMEM.
      offset = gather_idx.fetch_offset
      for gm_id in range(cfg.plan.num_xo_buffers - 1):
        gather_idx.copy_in(in_hbm_ref.gather_idx, (0, gm_id - offset, 0))
      # NOTE: Even if number of gather idx buffers is smaller than
      # num_xo_buffers - 1, it does not cause any race condition as semaphores
      # merely counts the size of the total data transfer without concerning
      # each individual DMA instructions.
      for gm_id in range(cfg.plan.num_xo_buffers - 1):
        gather_idx.wait_in(in_hbm_ref.gather_idx, (0, gm_id - offset, 0))

      inputs = dataclasses.replace(inputs, x=inputs.x.bind_idx_ref(gather_idx))
    if isinstance(scatter_idx, memory_ref.IndexBufferedRef):
      out = out.bind_idx_ref(scatter_idx)
    final_allocs = dataclasses.replace(final_allocs, inputs=inputs, out=out)

    # Prologue: Prefetch first num_xo_buffers - 1 input tiles.
    for gm_id in range(cfg.plan.num_xo_buffers - 1):
      final_allocs.inputs.x.branchless_copy_in(gm_id)

    # TODO: Add support for N/K dim tiling.
    num_n = num_k = 1
    # Body:
    pltpu.emit_pipeline(
        functools.partial(
            inner_pipeline,
            cfg=cfg,
            metadata_ref=metadata_ref,
            x_bref=final_allocs.inputs.x,
            out_bref=final_allocs.out,
        ),
        in_specs=(allocs.in_specs,),
        out_specs=allocs.out_spec,
        grid=(num_n, num_gm, num_k),
    )(in_hbm_ref, out_hbm_ref, allocations=final_allocs.all_tree)

    if isinstance(scatter_idx, memory_ref.IndexBufferedRef):
      scatter_idx.wait_in(in_hbm_ref.scatter_idx, (0, num_gm, 0))

    # Epilogue: Issue DMA for final compute tile and drain all output buffers.
    # NOTE: If output is written into VMEM at step i, output DMA is performed
    # at step i + 1. Therefore, output from last compute step had no
    # crresponding copy_out dma performed. Below code ensures that the output
    # from the last step is correctly issued.
    final_allocs.out.branchless_copy_out(num_gm - 1)
    for buf_idx in range(cfg.plan.num_xo_buffers):
      final_allocs.out.branchless_wait_out(
          num_gm - cfg.plan.num_xo_buffers + buf_idx
      )
