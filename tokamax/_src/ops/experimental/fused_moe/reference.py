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
"""Reference JAX implementation of the fused MoE operator."""

import functools
import jax
import jax.numpy as jnp
from jax import lax


@functools.partial(jax.jit, static_argnames=("topk", "renormalize", "act_fn"))
def fused_moe_reference(
    x: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    gating: jax.Array,
    w1_scale: jax.Array | None = None,
    w2_scale: jax.Array | None = None,
    w1_bias: jax.Array | None = None,
    w2_bias: jax.Array | None = None,
    *,
    topk: int = 2,
    renormalize: bool = True,
    act_fn: str = "silu",
) -> jax.Array:
  """Standard JAX TPU reference implementation of fused MoE layer for correctness checks.

  Args:
    x: Input tensor [tokens, hidden].
    w1: Expert weight 1 [experts, hidden, 2 * inter].
    w2: Expert weight 2 [experts, inter, hidden].
    gating: Router gating logits [tokens, experts].
    w1_scale: Optional weight 1 scale [experts, 2 * inter] or [experts, blocks,
      2 * inter].
    w2_scale: Optional weight 2 scale [experts, hidden] or [experts, blocks,
      hidden].
    w1_bias: Optional weight 1 bias [experts, 1, 2 * inter].
    w2_bias: Optional weight 2 bias [experts, 1, hidden].
    topk: Number of experts to select per token.
    renormalize: Whether to renormalize top-k weights to sum to 1.0.
    act_fn: FFN activation function ("silu").

  Returns:
    Output tensor [tokens, hidden].
  """
  scores = jax.nn.softmax(gating.astype(jnp.float32), axis=-1)
  topk_weights, topk_idx = lax.top_k(scores, topk)
  if renormalize:
    topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)

  tokens, hidden = x.shape
  e_total = w1.shape[0]
  inter = w1.shape[2] // 2

  x32 = x.astype(jnp.float32)
  out = jnp.zeros((tokens, hidden), dtype=jnp.float32)

  scaled = w1_scale is not None

  for e in range(e_total):
    w1e = w1[e].astype(jnp.float32)
    w2e = w2[e].astype(jnp.float32)
    if scaled:
      w1e = w1e * w1_scale[e]
      w2e = w2e * w2_scale[e]

    weight = jnp.sum(
        jnp.where(topk_idx == e, topk_weights, 0.0), axis=-1
    )[:, None]

    acc1 = x32 @ w1e
    if w1_bias is not None:
      acc1 = acc1 + w1_bias[e]

    gate = acc1[:, :inter]
    up = acc1[:, inter:]
    if act_fn == "silu":
      act = jax.nn.silu(gate) * up
    else:
      raise NotImplementedError(f"Unsupported act_fn: {act_fn}")

    row = act @ w2e
    if w2_bias is not None:
      row = row + w2_bias[e]

    out = out + weight * row

  return out.astype(x.dtype)
