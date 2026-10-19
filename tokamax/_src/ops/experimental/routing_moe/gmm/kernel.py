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

Many of the tensor manipulation logics are intentionally placed here instead of
other modules. This is because we want any XLA side logic that is handled
outside of the kernel to be consolidated here without scavanging other modules.
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
    group_sizes: jax.Array,  # [global_groups]
    *,
    group_offset: jax.Array | None = None,  # [1]
    gather: arg.RoutingWrapper[jax.Array] | None = None,
    scatter: arg.RoutingWrapper[jax.Array] | None = None,
    fuse_act: str | None = None,
    out_dtype: jnp.dtype | None = None,
    custom_plan: plan.RoutingGmmPlan | None = None,
) -> arg.ActivationWrapper[jax.Array]:
  """GMM that pipelines routing and compute for both input and output.

  With `scatter.spec.remote_axis` (inside shard_map), a row scattered to global
  row `i` lands on device `i // rows_per_dev` along that axis, and each device
  returns its `m_out // axis_size` rows. Scatter indices across all devices must
  then fill every output row exactly once, or the kernel hangs.
  """

  gather_spec = None if gather is None else gather.spec
  scatter_spec = None if scatter is None else scatter.spec

  # Step 1: Derive sizes, resolve the plan, and construct the config.
  tpu_info = pltpu.get_tpu_info()
  size_ep = 1
  if scatter_spec is not None and scatter_spec.remote_axis is not None:
    size_ep = jax.lax.axis_size(scatter_spec.remote_axis)
  dim = config.Dimension.create(
      x=x.spec,
      w=w.spec,
      num_global_groups=group_sizes.size,
      gather=gather_spec,
      scatter=scatter_spec,
      fuse_act=fuse_act,
      size_ep=size_ep,
  )
  resolved_plan = plan.ResolvedPlan.create(
      custom_plan,
      x=x.spec,
      w=w.spec,
      size_out_m=dim.size_out_m,
      size_global_groups=dim.size_global_groups,
      size_k=dim.size_k,
      size_out_n=dim.size_out_n,
      tpu_info=tpu_info,
  )
  cfg = config.RoutingGmmConfig.create(
      x=x.spec,
      w=w.spec,
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
  w_scale = w.scale
  if w_scale is not None:
    w_scale = w_scale.astype(jnp.float32)
  remote = cfg.out.mode == config.RoutingMode.REMOTE
  inputs = memory_ref.PipelineInputs[jax.Array](
      x=x.data,
      w=memory_ref.SingleWeightRef(weight=w.weight, scale=w_scale, bias=w.bias),
      gather_idx=_padded_indices(gather),
      scatter_idx=_padded_indices(scatter),
  )

  metadata_obj = metadata.MetadataRef.create(group_cum_sum, group_offset, cfg)

  # Step 4: Specify memory space.
  inputs = memory_ref.map_memory_constraint(inputs, pltpu.HBM)
  metadata_obj = memory_ref.map_memory_constraint(metadata_obj, pltpu.SMEM)
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
  )(metadata_obj, inputs)
  return arg.ActivationWrapper(data=out_data, scale=None, spec=cfg.out.spec)
