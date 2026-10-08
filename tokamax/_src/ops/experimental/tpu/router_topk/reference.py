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
"""Sort-free top-k for MoE routing, in plain JAX ops.

The same algorithm as the Pallas kernel in `pallas_mosaic_tpu_kernel`: `k`
passes of (row max -> lowest matching column -> mask that column out). Used as
the XLA implementation of the op, and as the numerics oracle the kernel tests
compare against.
"""

import jax
import jax.numpy as jnp

# Clamp floor. Below any real routing score, and strictly above the dtype
# minimum used to mask a selected column -- see `pallas_mosaic_tpu_kernel`.
NEG = -3.0e38


def rowmax_topk(scores: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
  """`jax.lax.top_k(scores, k)` without the sort.

  Returns `(values, indices)`, values descending and indices int32. Ties
  resolve to the lowest expert id.

  Scores at or below `NEG` are clamped up to it, so every returned index is a
  real expert. A row made entirely of such scores keeps NaN weights and still
  names k distinct experts, as a sort does.

  Args:
    scores: `[..., num_experts]` floating-point scores.
    k: The number of experts to select per row.

  Returns:
    `(values, indices)` of shape `[..., k]`: values in `scores.dtype`,
    descending, and int32 expert ids.

  Raises:
    ValueError: If `k` exceeds the number of experts.
  """
  n = scores.shape[-1]
  if k > n:
    raise ValueError(f"topk k={k} exceeds the expert count {n}")
  iota = jnp.arange(n, dtype=jnp.int32)
  masked = jnp.finfo(scores.dtype).min

  cur = jnp.where(scores > NEG, scores, NEG)
  values, indices = [], []
  for _ in range(k):
    m = jnp.max(cur, axis=-1, keepdims=True)
    # Lowest column attaining the max; the rest fill with the last expert so
    # the min never selects a phantom.
    idx = jnp.min(jnp.where(cur == m, iota, n - 1), axis=-1, keepdims=True)
    values.append(m)
    indices.append(idx)
    cur = jnp.where(iota == idx, masked, cur)

  weights = jnp.concatenate(values, axis=-1)
  ids = jnp.concatenate(indices, axis=-1)
  # No score above the sentinel means no real maximum: keep the row poisonous
  # so the caller's renormalization propagates it.
  weights = jnp.where(weights[..., :1] <= NEG, jnp.nan, weights)
  return weights, ids
