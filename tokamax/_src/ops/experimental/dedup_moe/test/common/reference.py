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
"""f32 references for the De-Dup MoE layer.

The expert math is `fused_moe.reference.fused_moe_reference` (the shared
Tokamax MoE baseline) wherever it covers the case (the "silu" activation);
this module adds what it lacks: the routing options below (applied first and
passed to it as precomputed top-k weights and ids), the clamped activations,
and HIGHEST matmul precision.

`moe_reference` is the dense single-device reference the CPU tests compare
against: router scoring (softmax by default), top-k (or the caller's ids),
renormalization, then for every expert act(x @ w1_gate, x @ w1_up) @ w2, all
in f32 at HIGHEST precision on the dequantized weights. `ep_reference` is the
same computation sharded over the expert-parallel mesh axis (a scan over each
shard's local experts and a psum_scatter), for on-device verification at
shapes whose weights do not fit one device. Neither models the kernel's
activation quantization or bf16 roundings, so the comparison is
tolerance-based.

Every dedup_ep_moe option is modeled with the kernel's semantics (read off
layer/monolithic.py, fused_moe/kernel.py and io_relayout/input_pack.py):

  * scoring_fn: softmax, sigmoid or sqrt(softplus) of the f32 logits;
  * score_bias [E]: added to the scores for SELECTION only; the weights are
    the unbiased scores at the selected ids;
  * renormalize, then routed_scaling_factor on the weights;
  * act_fn: silu(g) * u; swigluoai (g clipped above at 7, u clipped to
    [-7, 7], (u + 1) * g * sigmoid(1.702 g)); silu_and_mul_with_clamp (the
    same clips at 10, silu(g) * u);
  * w1_bias [E, 1, 2F] lands on the post-scale gate and up halves before the
    activation, w2_bias [E, 1, D] on the post-scale down row, so the down
    bias is weighted by the router weight of its selection.
"""

from typing import Any

import jax
from jax import lax
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_moe import reference as fused_reference

HIGHEST = lax.Precision.HIGHEST
SCORING_FNS = ("softmax", "sigmoid", "sqrtsoftplus")
ACT_FNS = ("silu", "swigluoai", "silu_and_mul_with_clamp")


def score(gating, scoring_fn: str = "softmax"):
  """[T, E] f32 router scores of the logits."""
  logits = gating.astype(jnp.float32)
  if scoring_fn == "softmax":
    return jax.nn.softmax(logits, axis=-1)
  if scoring_fn == "sigmoid":
    return jax.nn.sigmoid(logits)
  if scoring_fn == "sqrtsoftplus":
    return jnp.sqrt(jax.nn.softplus(logits))
  raise ValueError(
      f"scoring_fn must be one of {SCORING_FNS}; got {scoring_fn!r}"
  )


def activation(gate, up, act_fn: str = "silu"):
  """The gated FFN activation on the post-scale (and post-bias) halves."""
  if act_fn == "silu":
    return jax.nn.silu(gate) * up
  if act_fn == "swigluoai":
    gate = jnp.minimum(gate, 7.0)
    up = jnp.clip(up, -7.0, 7.0)
    return (up + 1.0) * (gate * jax.nn.sigmoid(1.702 * gate))
  if act_fn == "silu_and_mul_with_clamp":
    gate = jnp.minimum(gate, 10.0)
    up = jnp.clip(up, -10.0, 10.0)
    return jax.nn.silu(gate) * up
  raise ValueError(f"act_fn must be one of {ACT_FNS}; got {act_fn!r}")


def route(
    gating,
    topk: int,
    *,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    hash_topk_ids=None,
    scoring_fn: str = "softmax",
    score_bias=None,
):
  """Top-k routing: ([T, topk] f32 weights, [T, topk] i32 ids)."""
  scores = score(gating, scoring_fn)
  if hash_topk_ids is not None:
    ids = hash_topk_ids.astype(jnp.int32)
  elif score_bias is None:
    _, ids = lax.top_k(scores, topk)
  else:
    _, ids = lax.top_k(scores + score_bias.astype(jnp.float32)[None, :], topk)
  ids = ids.astype(jnp.int32)
  weights = jnp.take_along_axis(scores, ids, axis=-1)
  if renormalize:
    weights = weights / jnp.maximum(weights.sum(-1, keepdims=True), 1e-20)
  return weights * jnp.float32(routed_scaling_factor), ids


def dequantize(w, scale):
  """[E, K, N] f32 weights; `scale` is per output channel [E, N] or None."""
  w = w.astype(jnp.float32)
  if scale is None:
    return w
  return w * scale.astype(jnp.float32)[:, None, :]


def expert_ffn(x, w1, w2, *, act_fn="silu", b1=None, b2=None):
  """act(x @ w1[:, :F] + b1, x @ w1[:, F:] + b1) @ w2 + b2, f32 at HIGHEST.

  Args:
    x: [m, D] rows.
    w1: [D, 2F] dequantized gate|up weights.
    w2: [F, D] dequantized down weights.
    act_fn: One of ACT_FNS.
    b1: Optional [2F] (or [1, 2F]) gate|up bias.
    b2: Optional [D] (or [1, D]) down bias.

  Returns:
    [m, D] f32.
  """
  inter = w1.shape[-1] // 2
  h = jnp.dot(x.astype(jnp.float32), w1.astype(jnp.float32), precision=HIGHEST)
  if b1 is not None:
    h = h + b1.astype(jnp.float32).reshape(1, 2 * inter)
  a = activation(h[:, :inter], h[:, inter:], act_fn)
  y = jnp.dot(a, w2.astype(jnp.float32), precision=HIGHEST)
  if b2 is not None:
    y = y + b2.astype(jnp.float32).reshape(1, -1)
  return y


_expert_ffn = expert_ffn


def _experts_reference(
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    weights,
    ids,
    *,
    act_fn,
    w1_bias=None,
    w2_bias=None,
    expert_start=0,
):
  """sum_k weights[:, k] * ffn_{ids[:, k]}(x) over the experts of `w1`: f32.

  Uses `fused_moe_reference` for the "silu" activation and a local scan over
  the experts for the clamped activations it does not implement.

  Args:
    x: [T, D] rows.
    w1: [G, D, 2F] expert weights (G local experts).
    w2: [G, F, D] expert weights.
    w1_scale: [G, 2F] per-channel scales, or None.
    w2_scale: [G, D] per-channel scales, or None.
    weights: [T, K] f32 routed weights.
    ids: [T, K] i32 global expert ids.
    act_fn: One of ACT_FNS.
    w1_bias: Optional [G, 1, 2F] bias.
    w2_bias: Optional [G, 1, D] bias.
    expert_start: Global id of the first expert of `w1`.

  Returns:
    [T, D] f32.
  """
  if act_fn == "silu":
    with jax.default_matmul_precision("highest"):
      out = fused_reference.fused_moe_reference(
          x.astype(jnp.float32),
          w1,
          w2,
          None,
          None if w1_scale is None else w1_scale.astype(jnp.float32),
          None if w2_scale is None else w2_scale.astype(jnp.float32),
          w1_bias,
          w2_bias,
          topk_weights=weights,
          topk_ids=ids,
          expert_start=expert_start,
      )
    return out.astype(jnp.float32)
  local_ids = ids - expert_start
  w1f = dequantize(w1, w1_scale)
  w2f = dequantize(w2, w2_scale)
  xf = x.astype(jnp.float32)

  def body(acc, e):
    coef = jnp.sum(jnp.where(local_ids == e, weights, 0.0), axis=-1)
    y = expert_ffn(
        xf,
        w1f[e],
        w2f[e],
        act_fn=act_fn,
        b1=None if w1_bias is None else w1_bias[e],
        b2=None if w2_bias is None else w2_bias[e],
    )
    return acc + coef[:, None] * y, None

  out, _ = lax.scan(
      body, jnp.zeros(xf.shape, jnp.float32), jnp.arange(w1.shape[0])
  )
  return out


def moe_reference(
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    gating,
    *,
    topk: int,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    hash_topk_ids=None,
    scoring_fn: str = "softmax",
    score_bias=None,
    act_fn: str = "silu",
    w1_bias=None,
    w2_bias=None,
):
  """Dense single-device reference: [T, hidden] f32.

  Args:
    x: [T, hidden] activations.
    w1: [E, hidden, 2 * inter] expert weights.
    w2: [E, inter, hidden] expert weights.
    w1_scale: [E, 2 * inter] per-channel scales, or None.
    w2_scale: [E, hidden] per-channel scales, or None.
    gating: [T, E] router logits.
    topk: Experts per token.
    renormalize: Renormalize the top-k weights.
    routed_scaling_factor: Scale on the routed weights.
    hash_topk_ids: Optional [T, topk] caller-chosen expert ids.
    scoring_fn: Router scoring function (SCORING_FNS).
    score_bias: Optional [E] selection bias.
    act_fn: FFN activation (ACT_FNS).
    w1_bias: Optional [E, 1, 2 * inter] gate/up bias.
    w2_bias: Optional [E, 1, hidden] down bias.

  Returns:
    [T, hidden] f32.
  """
  weights, ids = route(
      gating,
      topk,
      renormalize=renormalize,
      routed_scaling_factor=routed_scaling_factor,
      hash_topk_ids=hash_topk_ids,
      scoring_fn=scoring_fn,
      score_bias=score_bias,
  )
  return _experts_reference(
      x,
      w1,
      w2,
      w1_scale,
      w2_scale,
      weights,
      ids,
      act_fn=act_fn,
      w1_bias=w1_bias,
      w2_bias=w2_bias,
  )


def ep_reference(
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    gating,
    *,
    topk: int,
    mesh: Any,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    hash_topk_ids=None,
    scoring_fn: str = "softmax",
    score_bias=None,
    act_fn: str = "silu",
    w1_bias=None,
    w2_bias=None,
):
  """`moe_reference` sharded over the mesh's single (expert) axis.

  x [T, hidden], gating [T, E] and the weights are sharded on the mesh axis
  as dedup_ep_moe takes them. Every shard gathers all tokens, runs its
  local experts densely (a scan, one expert's f32 weights live at a time) and
  the partial sums are reduce-scattered back to token shards.

  Args:
    x: [T, hidden] activations.
    w1: [E, hidden, 2 * inter] expert weights (fp8 or bf16).
    w2: [E, inter, hidden] expert weights.
    w1_scale: [E, 2 * inter] per-channel scales, or None.
    w2_scale: [E, hidden] per-channel scales, or None.
    gating: [T, E] router logits.
    topk: Experts per token.
    mesh: A one-axis mesh.
    renormalize: Renormalize the top-k weights.
    routed_scaling_factor: Scale on the routed weights.
    hash_topk_ids: Optional [T, topk] caller-chosen expert ids.
    scoring_fn: Router scoring function (SCORING_FNS).
    score_bias: Optional [E] selection bias (replicated).
    act_fn: FFN activation (ACT_FNS).
    w1_bias: Optional [E, 1, 2 * inter] gate/up bias (expert-sharded).
    w2_bias: Optional [E, 1, hidden] down bias (expert-sharded).

  Returns:
    [T, hidden] f32, sharded on the mesh axis.
  """
  (ax,) = mesh.axis_names
  spec = jax.sharding.PartitionSpec(ax)
  has_scales = w1_scale is not None
  has_ids = hash_topk_ids is not None
  has_b1 = w1_bias is not None
  has_b2 = w2_bias is not None
  has_sb = score_bias is not None

  def local(x_l, gate_l, w1_l, w2_l, *rest):
    rest = list(rest)
    ids_l = rest.pop(0) if has_ids else None
    b1_l = rest.pop(0) if has_b1 else None
    b2_l = rest.pop(0) if has_b2 else None
    sb = rest.pop(0) if has_sb else None
    scales = rest
    x_all = lax.all_gather(x_l, ax, axis=0, tiled=True).astype(jnp.float32)
    gate_all = lax.all_gather(gate_l, ax, axis=0, tiled=True)
    ids_all = (
        None if ids_l is None else lax.all_gather(ids_l, ax, axis=0, tiled=True)
    )
    weights, ids = route(
        gate_all,
        topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        hash_topk_ids=ids_all,
        scoring_fn=scoring_fn,
        score_bias=sb,
    )
    g_local = w1_l.shape[0]
    acc = _experts_reference(
        x_all,
        w1_l,
        w2_l,
        scales[0] if has_scales else None,
        scales[1] if has_scales else None,
        weights,
        ids,
        act_fn=act_fn,
        w1_bias=b1_l,
        w2_bias=b2_l,
        expert_start=lax.axis_index(ax) * g_local,
    )
    return lax.psum_scatter(acc, ax, scatter_dimension=0, tiled=True)

  sharded = (
      ((hash_topk_ids,) if has_ids else ())
      + ((w1_bias,) if has_b1 else ())
      + ((w2_bias,) if has_b2 else ())
  )
  replicated = (score_bias,) if has_sb else ()
  scale_args = (w1_scale, w2_scale) if has_scales else ()
  in_specs = (
      (spec,) * (4 + len(sharded))
      + (jax.sharding.PartitionSpec(),) * len(replicated)
      + (spec,) * len(scale_args)
  )
  fn = jax.jit(
      jax.shard_map(
          local,
          mesh=mesh,
          in_specs=in_specs,
          out_specs=spec,
          check_vma=False,
      )
  )
  return fn(x, gating, w1, w2, *sharded, *replicated, *scale_args)


def rel_l2(got, want) -> float:
  """||got - want|| / ||want|| in f32 (host arrays or jax arrays)."""
  got = jnp.asarray(got, jnp.float32)
  want = jnp.asarray(want, jnp.float32)
  return float(
      jnp.linalg.norm(got - want) / jnp.maximum(jnp.linalg.norm(want), 1e-30)
  )


def max_abs_rel(got, want) -> float:
  """max |got - want| / max |want|."""
  got = jnp.asarray(got, jnp.float32)
  want = jnp.asarray(want, jnp.float32)
  return float(
      jnp.max(jnp.abs(got - want)) / jnp.maximum(jnp.max(jnp.abs(want)), 1e-30)
  )
