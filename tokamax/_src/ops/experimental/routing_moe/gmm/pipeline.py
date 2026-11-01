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

import jax
from jax import lax
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from tokamax._src.ops.experimental.routing_moe.gmm import compute
from tokamax._src.ops.experimental.routing_moe.gmm import config
from tokamax._src.ops.experimental.routing_moe.gmm import memory_ref
from tokamax._src.ops.experimental.routing_moe.gmm import metadata


def _inner_pipeline(
    step: pltpu.PipelineStep,
    in_ref: memory_ref.PipelineInputs,
    out_ref: hijax.TransformedRef,
    *,
    x_bref: memory_ref.RoutingBufferedRef,
    out_bref: memory_ref.RoutingBufferedRef,
    x_hbm_ref: hijax.TransformedRef,
    out_hbm_ref: hijax.TransformedRef,
    metadata_ref: metadata.AbstractMetadataRef,
    cfg: config.RoutingGmmConfig,
) -> None:
  """Orchestrates DMA and computation for a single pipeline step.

  Args:
    step: Current pipeline step.
    in_ref: Current VMEM slot references for the pipelined inputs.
    out_ref: Current VMEM slot reference for the output activation.
    x_bref: BufferedRef for the input activation used for manual DMA scheduling.
    out_bref: BufferedRef for the output activation used for manual DMA
      scheduling.
    x_hbm_ref: HBM reference of the input activation.
    out_hbm_ref: HBM reference of the output activation.
    metadata_ref: SMEM metadata reference for routing and tile row counts.
    cfg: Static configuration for the routing GMM kernel.
  """

  n_id, gm_id, k_id = step.index
  num_xo_buffers = cfg.plan.num_xo_buffers

  def _at(gm_offset: int) -> memory_ref.GridIndices:
    return (n_id, gm_id + gm_offset, k_id)

  x_ref = pltpu.annotate(in_ref.x, no_hazard_no_deps=True)
  out_ref = pltpu.annotate(out_ref, no_hazard_no_deps=True)

  def _bucket_fn(bucket_m: int) -> None:
    # Step 1: Wait DMA for the current step slot.
    x_bref.branchless_wait_in(x_hbm_ref, _at(0))
    out_bref.branchless_wait_out(out_hbm_ref, _at(-num_xo_buffers))

    # Step 2: Start DMA for previous step slot.
    # NOTE: While we typically issue DMA for the other slots before waiting for
    # current slot, that will prevent DMA instructions to be pipelined with
    # compute. Therefore, DMA is issued after the waiting.
    x_bref.branchless_copy_in(x_hbm_ref, _at(num_xo_buffers - 1))

    # Step 3: Read input from VMEM.
    act = x_bref.read_vmem(x_ref.at[:bucket_m], _at(0))

    # Step 4: Perform computation.
    weight = in_ref.w.get_weight()
    weight_scale = in_ref.w.get_scale() if cfg.w.has_scale else None
    act = compute.matmul(cfg.compute, act, weight, weight_scale, None)
    act = compute.apply_act_fn(act, cfg.fuse_act)
    if cfg.w2 is not None:
      assert in_ref.w2 is not None and cfg.compute2 is not None
      act = act.astype(x_ref.dtype)
      weight2 = in_ref.w2.get_weight()
      weight2_scale = in_ref.w2.get_scale() if cfg.w2.has_scale else None
      act = compute.matmul(cfg.compute2, act, weight2, weight2_scale, None)

    # Start output DMA for previous step slot after the loads because a remote
    # DMA is a global memory barrier, so issuing it earlier would hold every
    # load and the matmul behind it instead of sharing bundles with the matmul.
    out_bref.branchless_copy_out(out_hbm_ref, _at(-1))

    # Step 5: Write output into VMEM.
    out_bref.write_vmem(out_ref.at[:bucket_m], act, _at(0))

  # Select which bucket to choose based on number of active rows of given tile.
  num_rows = metadata_ref.gm_id_to_num_rows[gm_id]
  bucket_base = cfg.plan.bucket_base
  num_buckets = (num_rows + bucket_base - 1) // bucket_base
  bucket_idx = num_buckets - 1

  lax.switch(
      bucket_idx,
      [
          functools.partial(_bucket_fn, bucket_m)
          for bucket_m in cfg.plan.bucket_rows
      ],
  )


def _prologue(
    final_allocs: memory_ref.PipelineAllocs,
    in_hbm_ref: memory_ref.PipelineInputs[hijax.TransformedRef],
    *,
    cfg: config.RoutingGmmConfig,
) -> None:
  """Prefetches gather indices and initial input activation tiles."""
  gather_idx = final_allocs.inputs.gather_idx
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

  final_allocs.inputs.x.prologue()
  final_allocs.out.prologue()

  # Prefetch first num_xo_buffers - 1 input tiles.
  for gm_id in range(cfg.plan.num_xo_buffers - 1):
    final_allocs.inputs.x.branchless_copy_in(in_hbm_ref.x, (0, gm_id, 0))


def _epilogue(
    final_allocs: memory_ref.PipelineAllocs,
    in_hbm_ref: memory_ref.PipelineInputs[hijax.TransformedRef],
    out_hbm_ref: hijax.TransformedRef,
    *,
    num_gm: jax.Array,
    cfg: config.RoutingGmmConfig,
) -> None:
  """Issues the final output DMA and drains all output buffers."""
  scatter_idx = final_allocs.inputs.scatter_idx
  last_n = cfg.num_n - 1
  if isinstance(scatter_idx, memory_ref.IndexBufferedRef):
    scatter_idx.wait_in(in_hbm_ref.scatter_idx, (last_n, num_gm, 0))

  # Issue DMA for final compute tile and drain all output buffers.
  # NOTE: If output is written into VMEM at step i, output DMA is performed
  # at step i + 1. Therefore, output from last compute step had no
  # crresponding copy_out dma performed. Below code ensures that the output
  # from the last step is correctly issued.
  final_allocs.out.branchless_copy_out(out_hbm_ref, (last_n, num_gm - 1, 0))
  for buf_idx in range(cfg.plan.num_xo_buffers):
    final_allocs.out.branchless_wait_out(
        out_hbm_ref, (last_n, num_gm - cfg.plan.num_xo_buffers + buf_idx, 0)
    )

  final_allocs.inputs.x.epilogue(in_hbm_ref.x)
  final_allocs.out.epilogue(out_hbm_ref)


def outer_pipeline(
    metadata_in_ref: metadata.MetadataInputs[hijax.TransformedRef],
    in_hbm_ref: memory_ref.PipelineInputs[hijax.TransformedRef],
    out_hbm_ref: hijax.TransformedRef,
    metadata_ref: metadata.SortedMetadataRef,
    *,
    cfg: config.RoutingGmmConfig,
) -> None:
  """Allocates scoped buffers and schedules the routing GMM pipeline.

  Args:
    metadata_in_ref: SMEM references for inputs required for metadata compute.
    in_hbm_ref: HBM references for the pipelined kernel inputs.
    out_hbm_ref: HBM reference for the kernel output.
    metadata_ref: SMEM scratch reference for routing and tile scheduling.
    cfg: Static configuration for the routing GMM kernel.
  """

  num_gm = metadata_ref.fill_metadata_and_return_num_gm(metadata_in_ref)

  allocs = memory_ref.PipelineAllocs.create(
      in_hbm_ref, out_hbm_ref, metadata_ref, cfg
  )
  in_hbm_ref = in_hbm_ref.transform_for_pipeline(cfg)

  @functools.partial(pl.run_scoped, final_allocs=allocs)
  def _run(final_allocs: memory_ref.PipelineAllocs) -> None:

    inputs = final_allocs.inputs
    out = final_allocs.out
    gather_idx = inputs.gather_idx
    scatter_idx = inputs.scatter_idx
    if isinstance(gather_idx, memory_ref.IndexBufferedRef):
      inputs = dataclasses.replace(inputs, x=inputs.x.bind_idx_ref(gather_idx))
    if isinstance(scatter_idx, memory_ref.IndexBufferedRef):
      out = out.bind_idx_ref(scatter_idx)
    final_allocs = dataclasses.replace(final_allocs, inputs=inputs, out=out)

    _prologue(final_allocs, in_hbm_ref, cfg=cfg)

    # Body:
    pltpu.emit_pipeline(
        functools.partial(
            _inner_pipeline,
            cfg=cfg,
            metadata_ref=metadata_ref,
            x_bref=final_allocs.inputs.x,
            out_bref=final_allocs.out,
            x_hbm_ref=in_hbm_ref.x,
            out_hbm_ref=out_hbm_ref,
        ),
        in_specs=(allocs.in_specs,),
        out_specs=allocs.out_spec,
        grid=(cfg.num_n, num_gm, cfg.num_k),
        _explicit_indices=True,
    )(in_hbm_ref, out_hbm_ref, allocations=final_allocs.all_tree)

    _epilogue(final_allocs, in_hbm_ref, out_hbm_ref, num_gm=num_gm, cfg=cfg)
