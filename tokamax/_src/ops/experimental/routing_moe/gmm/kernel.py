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

"""Module that invokes routing gmm pallas kernel."""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.gmm import config
from tokamax._src.ops.experimental.routing_moe.gmm import memory_ref
from tokamax._src.ops.experimental.routing_moe.gmm import metadata
from tokamax._src.ops.experimental.routing_moe.gmm import pipeline


@jax.jit(static_argnames=("tile", "fuse_act", "acc_dtype"))
def routing_gmm(
    x: jax.Array,  # [m_in, k // 128 // packing, packing, lane_size]
    w: jax.Array,  # [local_groups, k, n]
    group_sizes: jax.Array,  # [global_groups]
    w_scale: jax.Array | None = None,  # [local_groups, num_scales, 1, n]
    *,
    acc_dtype: jnp.dtype = jnp.bfloat16.dtype,
    group_offset: jax.Array | None = None,  # [1]
    gather_indices: jax.Array | None = None,  # [m_out]
    scatter_indices: jax.Array | None = None,  # [m_out]
    tile: config.TileConfig | None = None,
    fuse_act: str | None = None,
) -> jax.Array:
  """GMM that can pipelines routing and compute for both input and output."""

  # Step 1: Create config object.
  cfg = config.RoutingGmmConfig.create(
      x=x,
      w=w,
      w_scale=w_scale,
      group_sizes=group_sizes,
      group_offset=group_offset,
      gather_indices=gather_indices,
      scatter_indices=scatter_indices,
      tile=tile,
      fuse_act=fuse_act,
      acc_dtype=acc_dtype,
  )
  cfg.validate(x, w, w_scale, group_sizes)

  # Step 2: Pre-process inputs.
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]
  group_cum_sum = jnp.cumulative_sum(group_sizes, include_initial=True)

  # Step 3: Wrap inputs in parameter object.
  if w_scale is not None:
    w_scale = w_scale.astype(jnp.float32)
  wrapped_w = memory_ref.SingleWeightRef(weight=w, scale=w_scale, bias=None)
  inputs = memory_ref.PipelineInputs(x=x, w=wrapped_w)

  metadata_obj = metadata.MetadataRef.create(
      gather_indices, scatter_indices, group_cum_sum, group_offset, cfg
  )

  # Step 4: Specify memory space.
  inputs = memory_ref.map_memory_constraint(inputs, pltpu.HBM)
  metadata_obj = memory_ref.map_memory_constraint(metadata_obj, pltpu.SMEM)
  out = memory_ref.map_memory_constraint(
      jnp.empty(cfg.out.compact_shape, cfg.out.dtype), pltpu.HBM
  )

  # Step 5: Call kernel.
  return pl.kernel(
      functools.partial(pipeline.outer_pipeline, cfg=cfg),
      mesh=pltpu.TensorCoreMesh(axis_name="core", num_cores=1),
      out_type=out,
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=cfg.tpu_info.vmem_capacity_bytes,
          disable_bounds_checks=True,
          # NOTE: This is critical to ensure DMA instruction are correctly
          # scheduled to avoid resource contention.
          flags={"XLA_TPU_EVENLY_DISTRIBUTE_CRITICAL_NODES": True},
      ),
      name=cfg.kernel_name,
  )(metadata_obj, inputs)
