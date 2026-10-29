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

"""Module that handles XLA side preprocessing and invokes routing gmm kernel.

Many of the tensor manipulation logic is intentionally placed here instead of
other modules. This is because we want any XLA side logic that is handled
outside of the kernel to be consolidated here without scavenging other modules.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.gmm import config
from tokamax._src.ops.experimental.routing_moe.gmm import memory_ref
from tokamax._src.ops.experimental.routing_moe.gmm import metadata
from tokamax._src.ops.experimental.routing_moe.gmm import pipeline
from tokamax._src.ops.experimental.routing_moe.gmm import plan


@jax.jit(static_argnames=("fuse_act", "out_dtype", "custom_plan"))
def routing_gmm(
    x: arg.ActivationWrapper[jax.Array],
    w: arg.SingleWeightWrapper[jax.Array],
    group_sizes: jax.Array,
    *,
    w2: arg.SingleWeightWrapper[jax.Array] | None = None,
    row_group: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    gather: arg.RoutingWrapper[jax.Array] | None = None,
    scatter: arg.RoutingWrapper[jax.Array] | None = None,
    fuse_act: str | None = None,
    out_dtype: jax.typing.DTypeLike | None = None,
    custom_plan: plan.RoutingGmmPlan | None = None,
) -> arg.ActivationWrapper[jax.Array]:
  """Computes grouped matrix multiplication with pipelined routing.

  Args:
    x: Input activation in compact layout.
    w: First (or only) expert weight.
    group_sizes: Row count per global expert group [global_groups].
    w2: Optional second expert weight for the down projection.
    row_group: Optional global group ID per sorted row [routed_rows].
    group_offset: Optional starting global group index for this shard [1].
    gather: Optional input row-index map. Reads x contiguously when None.
    scatter: Optional output row-index map. Writes contiguously when None.
    fuse_act: Optional gated activation applied after the w projection.
    out_dtype: Optional output dtype. Defaults to x.spec.dtype when None.
    custom_plan: Optional execution plan overriding heuristic defaults.

  Returns:
    Output activation in compact layout.
  """
  gather_spec = None if gather is None else gather.spec
  scatter_spec = None if scatter is None else scatter.spec
  w2_spec = None if w2 is None else w2.spec

  # Step 1: Derive sizes, resolve the plan, and construct the config.
  tpu_info = pltpu.get_tpu_info()
  size_ep = 1
  if scatter_spec is not None and scatter_spec.remote_axis is not None:
    size_ep = jax.lax.axis_size(scatter_spec.remote_axis)
  dim = config.Dimension.create(
      x=x.spec,
      w=w.spec,
      w2=w2_spec,
      num_global_groups=group_sizes.size,
      gather=gather_spec,
      scatter=scatter_spec,
      fuse_act=fuse_act,
      size_ep=size_ep,
  )
  resolved_plan = plan.ResolvedGmmPlan.create(
      custom_plan,
      x=x.spec,
      w=w.spec,
      w2=w2_spec,
      is_fused_act=fuse_act is not None,
      out_dtype=out_dtype,
      size_out_m=dim.size_out_m,
      size_global_groups=dim.size_global_groups,
      size_k=dim.size_k,
      size_k2=dim.size_k2,
      size_out_n=dim.size_out_n,
      tpu_info=tpu_info,
  )
  if w2 is not None and not resolved_plan.fuse_up_and_down:
    gmm1_res = routing_gmm(
        x=x,
        w=w,
        group_sizes=group_sizes,
        row_group=row_group,
        group_offset=group_offset,
        gather=gather,
        fuse_act=fuse_act,
        custom_plan=custom_plan,
    )
    return routing_gmm(
        x=gmm1_res,
        w=w2,
        group_sizes=group_sizes,
        row_group=row_group,
        group_offset=group_offset,
        scatter=scatter,
        out_dtype=out_dtype,
        custom_plan=custom_plan,
    )

  cfg = config.RoutingGmmConfig.create(
      x=x.spec,
      w=w.spec,
      w2=w2_spec,
      dim=dim,
      plan=resolved_plan,
      gather=gather_spec,
      scatter=scatter_spec,
      fuse_act=fuse_act,
      out_dtype=out_dtype,
      tpu_info=tpu_info,
  )

  # Step 2: Pre-process inputs.
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]
  group_cum_sum = jnp.cumulative_sum(group_sizes, include_initial=True)

  # Pad indices by tile_m_idx to ensure DMA and indexing does not trigger OOB.
  def _padded_indices(
      routing: arg.RoutingWrapper[jax.Array] | None,
  ) -> jax.Array | None:
    if routing is None:
      return None
    return jnp.pad(routing.indices, (0, cfg.plan.tile_m_idx))

  # Step 3: Wrap inputs in the pipeline's parameter object. emit_pipeline needs
  # HBM refs and VMEM allocations to share one pytree structure, so the API
  # wrappers are reduced to their data here.
  def _weight_ref(
      wrapper: arg.SingleWeightWrapper[jax.Array],
  ) -> memory_ref.SingleWeightRef[jax.Array]:
    w_scale = wrapper.scale
    if w_scale is not None:
      w_scale = w_scale.astype(jnp.float32)
    return memory_ref.SingleWeightRef(
        weight=wrapper.weight, scale=w_scale, bias=wrapper.bias
    )

  remote = cfg.out.mode == config.RoutingMode.REMOTE
  inputs = memory_ref.PipelineInputs[jax.Array](
      x=x.data,
      w=_weight_ref(w),
      w2=None if w2 is None else _weight_ref(w2),
      gather_idx=_padded_indices(gather),
      scatter_idx=_padded_indices(scatter),
  )

  row_group_i32 = None if row_group is None else row_group.astype(jnp.int32)
  metadata_in, metadata_scratch = metadata.create_metadata(
      group_cum_sum, group_offset, cfg, row_group=row_group_i32
  )

  # Step 4: Specify memory space.
  inputs = memory_ref.map_memory_constraint(inputs, pltpu.HBM)
  metadata_in = memory_ref.map_memory_constraint(metadata_in, pltpu.SMEM)
  out = memory_ref.map_memory_constraint(
      jnp.empty(cfg.out.spec.compact_shape, cfg.out.spec.dtype), pltpu.HBM
  )

  # Step 5: Call kernel.
  num_cores = cfg.tpu_info.num_cores
  # NOTE: On megacore hardware, this will incur redundant compute. Furture work
  # is needed to enable parallelism in such hardware.
  out_data = pl.kernel(
      functools.partial(pipeline.outer_pipeline, cfg=cfg),
      mesh=pltpu.TensorCoreMesh(axis_name="core", num_cores=num_cores),
      out_type=out,
      scratch_types=(metadata_scratch,),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=cfg.plan.vmem_limit_bytes,
          disable_bounds_checks=True,
          # Remote DMA barriers use the collective barrier semaphore.
          collective_id=0 if remote else None,
          # NOTE: This is critical to ensure DMA instruction are correctly
          # scheduled to avoid resource contention.
          flags={"XLA_TPU_EVENLY_DISTRIBUTE_CRITICAL_NODES": True},
      ),
      name=cfg.kernel_name,
      metadata=cfg.kernel_metadata,
  )(metadata_in, inputs)
  return arg.ActivationWrapper(data=out_data, scale=None, spec=cfg.out.spec)
