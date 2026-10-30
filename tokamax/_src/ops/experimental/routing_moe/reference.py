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
"""Reference JAX implementation of the routing MoE operator."""

import functools
import jax
from jax import lax
import jax.numpy as jnp


def _as_block_scale(scale: jax.Array) -> jax.Array:
  """Returns a weight scale as `[E, num_blocks, N]`.

  Quantization blocks run along the contraction dimension `K`, so a
  per-channel `[E, N]` scale is a single block spanning all of `K`.

  Args:
    scale: A weight scale, `[E, N]`, `[E, num_blocks, N]` or
      `[E, num_blocks, 1, N]`.

  Returns:
    The scale as `[E, num_blocks, N]`.
  """
  match scale.ndim:
    case 2:
      return scale[:, None, :]
    case 3:
      return scale
    case 4:
      return scale.squeeze(2)
  raise NotImplementedError(f"Unsupported weight scale shape: {scale.shape}")


def _dequantize(w: jax.Array, scale: jax.Array) -> jax.Array:
  """Dequantizes one expert's `[K, N]` weight with its `[num_blocks, N]` scale."""
  num_blocks = scale.shape[0]
  if w.shape[0] % num_blocks:
    raise ValueError(
        f"{num_blocks} scale blocks do not divide the contraction dimension"
        f" {w.shape[0]}."
    )
  return w * jnp.repeat(scale, w.shape[0] // num_blocks, axis=0)


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "renormalize",
        "act_fn",
        "scoring_fn",
        "n_ranks",
        "scatter_rows",
        "routed_scale",
        "axis_name",
    ),
)
def routing_moe_reference(
    x: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    gating: jax.Array | None = None,
    w1_scale: jax.Array | None = None,
    w2_scale: jax.Array | None = None,
    w1_bias: jax.Array | None = None,
    w2_bias: jax.Array | None = None,
    *,
    topk: int = 2,
    renormalize: bool = True,
    act_fn: str = "silu",
    scoring_fn: str = "softmax",
    topk_weights: jax.Array | None = None,
    topk_ids: jax.Array | None = None,
    shared_w1: jax.Array | None = None,
    shared_w1_scale: jax.Array | None = None,
    shared_w2: jax.Array | None = None,
    shared_w2_scale: jax.Array | None = None,
    shared_gate_weight: jax.Array | None = None,
    rank: jax.Array | None = None,
    n_ranks: int = 1,
    expert_start: jax.Array | int | None = None,
    scatter_rows: bool = False,
    routed_scale: float = 1.0,
    axis_name: str | None = None,
) -> jax.Array:
  """Standard JAX TPU reference implementation of routing MoE layer.

  Args:
    x: Input tensor [tokens, hidden].
    w1: Expert weight 1 [experts, hidden, 2 * inter].
    w2: Expert weight 2 [experts, inter, hidden].
    gating: Optional router gating logits [tokens, experts]. Required when
      `topk_weights` and `topk_ids` are not provided.
    w1_scale: Optional weight 1 scale [experts, 2 * inter], [experts, blocks, 2
      * inter], or [experts, blocks, 1, 2 * inter]. Blocks split `hidden`.
    w2_scale: Optional weight 2 scale [experts, hidden], [experts, blocks,
      hidden], or [experts, blocks, 1, hidden]. Blocks split `inter`.
    w1_bias: Optional weight 1 bias [experts, 1, 2 * inter].
    w2_bias: Optional weight 2 bias [experts, 1, hidden].
    topk: Number of experts to select per token when `gating` is provided.
    renormalize: Whether to renormalize top-k weights to sum to 1.0 when
      `gating` is provided.
    act_fn: FFN activation function ("silu").
    scoring_fn: Router scoring function ("softmax" or "sigmoid").
    topk_weights: Optional precomputed router weights [tokens, k] (used when
      `gating` is None).
    topk_ids: Optional precomputed routed expert indices [tokens, k] (used when
      `gating` is None).
    shared_w1: Optional shared expert gate and up projection [hidden, 2 * i_s].
    shared_w1_scale: Optional shared expert w1 scale [1, 2 * i_s] or [2 * i_s].
    shared_w2: Optional shared expert down projection [i_s, hidden].
    shared_w2_scale: Optional shared expert w2 scale [1, hidden] or [hidden].
    shared_gate_weight: Optional [hidden, 1] sigmoid gate weight for the shared
      expert; None for an ungated shared expert.
    rank: Optional [1] (or scalar) int32 mesh index of this rank when
      `scatter_rows=True`.
    n_ranks: Number of ranks in the mesh for cross-rank reduction.
    expert_start: Optional global id of this rank's first expert under EP.
    scatter_rows: If True, returns only this rank's `tokens // n_ranks` rows.
    routed_scale: Multiplicative scaling factor applied to the routed sum only.
    axis_name: Optional mesh axis name for `lax.psum` when `n_ranks > 1`.

  Returns:
    Output tensor [tokens, hidden] (or [tokens // n_ranks, hidden] when
    `scatter_rows=True`).
  """
  tokens, hidden = x.shape
  if scatter_rows and tokens % n_ranks:
    raise ValueError(f"scatter_rows needs tokens % n_ranks == 0; got {tokens}")

  if gating is not None:
    gating_f32 = gating.astype(jnp.float32)
    if scoring_fn == "softmax":
      scores = jax.nn.softmax(gating_f32, axis=-1)
    elif scoring_fn == "sigmoid":
      scores = jax.nn.sigmoid(gating_f32)
    else:
      raise NotImplementedError(f"Unsupported scoring_fn: {scoring_fn}")
    topk_weights, topk_idx = lax.top_k(scores, topk)
    if renormalize:
      topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
  else:
    if topk_weights is None or topk_ids is None:
      raise ValueError(
          "Either `gating` or both `topk_weights` and `topk_ids` must be"
          " provided."
      )
    topk_weights = topk_weights.astype(x.dtype).astype(jnp.float32)
    topk_idx = topk_ids

  if expert_start is not None:
    topk_idx = topk_idx - expert_start

  e_total = w1.shape[0]
  inter = w1.shape[2] // 2

  x32 = x.astype(jnp.float32)
  out = jnp.zeros((tokens, hidden), dtype=jnp.float32)

  w1_block_scale = w2_block_scale = None
  if w1_scale is not None:
    assert w2_scale is not None
    w1_block_scale = _as_block_scale(w1_scale)
    w2_block_scale = _as_block_scale(w2_scale)

  for e in range(e_total):
    w1e = w1[e].astype(jnp.float32)
    w2e = w2[e].astype(jnp.float32)
    if w1_block_scale is not None and w2_block_scale is not None:
      w1e = _dequantize(w1e, w1_block_scale[e])
      w2e = _dequantize(w2e, w2_block_scale[e])

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

  if routed_scale != 1.0:
    out = jnp.float32(routed_scale) * out

  if shared_w1 is not None and shared_w2 is not None:
    inter_s = shared_w2.shape[0]
    sw1 = shared_w1.astype(jnp.float32)
    sw2 = shared_w2.astype(jnp.float32)
    if shared_w1_scale is not None:
      sw1 = sw1 * shared_w1_scale.reshape(1, -1)
    if shared_w2_scale is not None:
      sw2 = sw2 * shared_w2_scale.reshape(1, -1)
    s_acc1 = x32 @ sw1
    if act_fn == "silu":
      s_act = jax.nn.silu(s_acc1[:, :inter_s]) * s_acc1[:, inter_s:]
    else:
      raise NotImplementedError(f"Unsupported act_fn: {act_fn}")
    shared = s_act @ sw2
    if shared_gate_weight is not None:
      s_gate = jax.nn.sigmoid(
          jnp.dot(
              x,
              shared_gate_weight.reshape(hidden, 1).astype(x.dtype),
              preferred_element_type=jnp.float32,
          )
      )
      shared = s_gate * shared
    out = out + shared

  if n_ranks > 1:
    ax = axis_name
    if ax is None:
      abstract_mesh = jax.sharding.get_abstract_mesh()
      if len(abstract_mesh.axis_names) == 1:
        ax = abstract_mesh.axis_names[0]
      else:
        ax = "d"
    out = lax.psum(out, ax)

  if scatter_rows:
    rows = tokens // n_ranks
    rank_idx = jnp.asarray(
        0 if rank is None else rank, dtype=jnp.int32
    ).reshape(())
    out = lax.dynamic_slice(out, (rank_idx * rows, 0), (rows, hidden))

  return out.astype(x.dtype)
