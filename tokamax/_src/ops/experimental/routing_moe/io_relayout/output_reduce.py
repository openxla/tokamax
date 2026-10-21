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

"""Output relayout and top-k reduction helper from compact to large layout."""

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import arg


@jax.jit(static_argnames=("num_local_experts",))
def output_reduce_unpack(
    rows: arg.ActivationWrapper[jax.Array],
    weights: jax.Array,
    topk_indices: jax.Array,
    group_offset: jax.Array,
    num_local_experts: int,
) -> jax.Array:
  """Placeholder for output_reduce_unpack kernel from compact to large layout.

  If `rows.spec.row_shard_axis` is set, `rows` hold all topk rows of this
  device's token shard only: the output covers those tokens, and every row is
  valid regardless of expert.
  """
  # TODO: Replace with custom Pallas output_reduce_unpack kernel.

  num_tokens, topk = weights.shape
  axis = rows.spec.row_shard_axis
  if axis is not None:
    num_local_tokens = rows.spec.row_size // topk
    if num_local_tokens * topk != rows.spec.row_size:
      raise ValueError(
          f"Rows ({rows.spec.row_size}) must hold whole tokens of topk {topk}."
      )
    if num_local_tokens * jax.lax.axis_size(axis) != num_tokens:
      raise ValueError(
          f"{num_tokens} tokens must divide evenly across axis {axis!r}."
      )
    start = jax.lax.axis_index(axis) * num_local_tokens
    weights = jax.lax.dynamic_slice_in_dim(weights, start, num_local_tokens)
    rows3d = rows.data.reshape(num_local_tokens, topk, -1)
    # Keep the relayout in the rows' dtype. Otherwise XLA folds the f32
    # convert below into it and the reduce moves twice the bytes.
    rows3d = jax.lax.optimization_barrier(rows3d)
    x = rows3d * weights.astype(jnp.float32).reshape(-1, topk, 1)
    return x.sum(axis=1).astype(rows.spec.dtype)

  rows3d = rows.data.reshape(num_tokens, topk, -1)

  group_start = group_offset
  group_end = group_start + num_local_experts
  local = jnp.logical_and(group_start <= topk_indices, topk_indices < group_end)
  rows3d = jnp.where(local.reshape(-1, topk, 1), rows3d, 0)

  x = rows3d * weights.astype(jnp.float32).reshape(-1, topk, 1)
  return x.sum(axis=1).astype(rows.spec.dtype)
