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
"""Synthetic MoE layer operands for the De-Dup MoE layer's tests.

Everything is generated on the devices that hold it (jit with sharded
outputs), so the same code builds a small CPU test and a large layer on a TPU
slice without staging weights through the host. `expert_ids` forces a
routing through `hash_topk_ids`.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
from tokamax._src.ops.experimental.dedup_moe.common import arg as host

FP8 = jnp.float8_e4m3fn


def _expert_ids(kind, tokens, e_total, topk, experts_per_shard, seed):
  """[tokens, topk] expert ids, distinct within a row as a real table is."""
  rng = np.random.default_rng(seed)
  if kind == "uniform":
    rows = [rng.choice(e_total, topk, replace=False) for _ in range(tokens)]
  elif kind == "skewed":
    # 7 of 8 tokens pick only from shard 0's experts.
    hot = np.arange(max(topk, experts_per_shard))
    rows = [
        rng.choice(hot if t % 8 else e_total, topk, replace=False)
        for t in range(tokens)
    ]
  elif kind == "worst":
    rows = [np.arange(topk)] * tokens
  elif kind == "dominant":
    # Token t hits shard t % ep's local expert 0, plus topk - 1 others.
    ep = e_total // experts_per_shard
    rows = []
    for t in range(tokens):
      hot = (t % ep) * experts_per_shard
      others = np.delete(np.arange(e_total), hot)
      rows.append([hot, *rng.choice(others, topk - 1, replace=False)])
  elif kind.startswith("sparse:") or kind in ("idle_device", "idle_board"):
    pool = expert_pool(kind, e_total, experts_per_shard, seed)
    if pool.size < topk:
      raise ValueError(f"{kind}: {pool.size} live experts < topk={topk}")
    rows = [rng.choice(pool, topk, replace=False) for _ in range(tokens)]
  else:
    raise ValueError(kind)
  return np.asarray(rows, np.int32)


def active_expert_mask(
    e_total: int, active_expert_ratio: float, seed: int
) -> np.ndarray:
  """[E] bool: routing_moe moe_test.gen_moe_inputs's active experts.

  The int(E * ratio) experts with the largest N(0, 1) draw are active (the
  rest receive no tokens), as gen_moe_inputs masks the gating logits.

  Args:
    e_total: Number of experts.
    active_expert_ratio: Fraction of experts that receive tokens.
    seed: PRNG seed.

  Returns:
    The mask.
  """
  n_active = max(1, int(e_total * active_expert_ratio))
  draw = np.asarray(
      jax.random.normal(jax.random.key(seed), (e_total,), jnp.float32)
  )
  threshold = np.sort(draw)[::-1][n_active - 1]
  return draw >= threshold


def expert_pool(
    kind: str,
    e_total: int,
    experts_per_shard: int,
    seed: int,
    board_size: int = 8,
) -> np.ndarray:
  """The experts a sparse / idle routing may pick (sorted ids).

    sparse:<r>   routing_moe's active_expert_ratio: only int(E * r) experts
                 receive tokens, so whole experts (and maybe shards) are empty.
    idle_device  no expert of the last device (ep - 1) receives a token
                 (routing_moe gmm_test's skip_last_shard).
    idle_board   no expert of the last board (devices ep - board_size ..
                 ep - 1) receives a token; needs at least two boards.

  Args:
    kind: The routing name.
    e_total: Number of experts.
    experts_per_shard: Experts per device.
    seed: PRNG seed (sparse only).
    board_size: Devices per board (idle_board only).

  Returns:
    [n] int32 expert ids.
  """
  ep = e_total // experts_per_shard
  if kind.startswith("sparse:"):
    mask = active_expert_mask(e_total, float(kind.split(":", 1)[1]), seed)
  elif kind == "idle_device":
    mask = np.arange(e_total) < (ep - 1) * experts_per_shard
  elif kind == "idle_board":
    if ep < 2 * board_size or ep % board_size:
      raise ValueError(
          f"idle_board needs >= 2 boards of {board_size} devices; ep={ep}"
      )
    mask = np.arange(e_total) < (ep - board_size) * experts_per_shard
  else:
    raise ValueError(kind)
  return np.flatnonzero(mask).astype(np.int32)


expert_ids = _expert_ids
ROUTINGS = ("uniform", "skewed", "worst", "dominant")
# routing_moe's edge cases (expert_pool): not part of ROUTINGS, whose
# parameterized tests assume every shard receives rows.
SPARSE_ROUTINGS = ("sparse:0.6", "idle_device", "idle_board")


def _axis(mesh: Any) -> str:
  (ax,) = mesh.axis_names
  return ax


def random_weights(
    mesh: Any,
    *,
    e_total: int,
    hidden: int,
    inter: int,
    weight_format: str | host.WeightFormat,
    seed: int = 0,
):
  """(w1, w2, w1_scale, w2_scale) in dedup_ep_moe's layout, expert-sharded.

  w1 [E, hidden, 2 * inter] and w2 [E, inter, hidden] are N(0, 1/fan_in).
  fp8 weights are e4m3 with one f32 scale per output channel ([E, 2 * inter]
  and [E, hidden], amax / 448); bf16 weights carry no scales (None).

  Args:
    mesh: One-axis mesh; the expert axis is sharded over it.
    e_total: Number of experts.
    hidden: Model width.
    inter: Expert FFN width (w1 holds gate and up, 2 * inter columns).
    weight_format: "fp8" or "bf16".
    seed: PRNG seed.

  Returns:
    The four operands (scales None for bf16).
  """
  fmt = host.WeightFormat(weight_format)
  if fmt not in (host.WeightFormat.FP8, host.WeightFormat.BF16):
    raise ValueError(f"only fp8 and bf16 weights are generated; got {fmt!r}")
  shard = NamedSharding(mesh, P(_axis(mesh)))

  def gen(key):
    k1, k2 = jax.random.split(key)
    w1 = jax.random.normal(k1, (e_total, hidden, 2 * inter), jnp.bfloat16)
    w2 = jax.random.normal(k2, (e_total, inter, hidden), jnp.bfloat16)
    w1 = w1 * jnp.bfloat16(hidden**-0.5)
    w2 = w2 * jnp.bfloat16(inter**-0.5)
    if fmt == host.WeightFormat.BF16:
      return w1, w2

    def quantize(w):
      wf = w.astype(jnp.float32)
      amax = jnp.maximum(jnp.abs(wf).max(axis=1), 1e-12)
      scale = amax / float(jnp.finfo(FP8).max)
      return (wf / scale[:, None, :]).astype(FP8), scale

    q1, s1 = quantize(w1)
    q2, s2 = quantize(w2)
    return q1, q2, s1, s2

  if fmt == host.WeightFormat.BF16:
    w1, w2 = jax.jit(gen, out_shardings=(shard, shard))(jax.random.key(seed))
    return w1, w2, None, None
  return jax.jit(gen, out_shardings=(shard,) * 4)(jax.random.key(seed))


def random_inputs(
    mesh: Any,
    *,
    tokens: int,
    hidden: int,
    e_total: int,
    seed: int = 1,
    active_expert_ratio: float = 1.0,
    x_scale: float = 1.0,
):
  """(x [T, hidden] bf16, gating [T, E] f32), token-sharded, N(0, 1).

  Args:
    mesh: One-axis mesh; the token axis is sharded over it.
    tokens: T.
    hidden: Model width.
    e_total: Number of experts.
    seed: PRNG seed.
    active_expert_ratio: Below 1, routing_moe gen_moe_inputs's sparse
      experts: the gating logits are U[0, 1) and the inactive experts'
        (active_expert_mask with this seed) are 0, the minimum, so top-k never
        selects them while topk <= active experts.
    x_scale: Multiplier on x (larger values exercise the clamped activations).

  Returns:
    (x, gating).
  """
  shard = NamedSharding(mesh, P(_axis(mesh)))
  mask = (
      None
      if active_expert_ratio >= 1.0
      else jnp.asarray(active_expert_mask(e_total, active_expert_ratio, seed))
  )

  def gen(key):
    kx, kg = jax.random.split(key)
    x = jax.random.normal(kx, (tokens, hidden), jnp.float32) * x_scale
    x = x.astype(jnp.bfloat16)
    if mask is None:
      gating = jax.random.normal(kg, (tokens, e_total), jnp.float32)
    else:
      gating = jax.random.uniform(kg, (tokens, e_total), jnp.float32)
      gating = jnp.where(mask[None, :], gating, 0.0)
    return x, gating

  return jax.jit(gen, out_shardings=(shard, shard))(jax.random.key(seed))


def random_biases(
    mesh: Any,
    *,
    e_total: int,
    hidden: int,
    inter: int,
    seed: int = 5,
    scale: float = 0.1,
    dtype: Any = jnp.float32,
):
  """(w1_bias [E, 1, 2 * inter], w2_bias [E, 1, hidden]), expert-sharded.

  routing_moe gen_moe_inputs's biases: N(0, 1) * `scale`.

  Args:
    mesh: One-axis mesh; the expert axis is sharded over it.
    e_total: Number of experts.
    hidden: Model width.
    inter: Expert FFN width.
    seed: PRNG seed.
    scale: Standard deviation.
    dtype: Element type.

  Returns:
    The two bias tables.
  """
  shard = NamedSharding(mesh, P(_axis(mesh)))

  def gen(key):
    k1, k2 = jax.random.split(key)
    b1 = jax.random.normal(k1, (e_total, 1, 2 * inter), jnp.float32) * scale
    b2 = jax.random.normal(k2, (e_total, 1, hidden), jnp.float32) * scale
    return b1.astype(dtype), b2.astype(dtype)

  return jax.jit(gen, out_shardings=(shard, shard))(jax.random.key(seed))


def shard_rows(mesh: Any, array) -> jax.Array:
  """`array` sharded on its leading axis over the mesh axis."""
  return jax.device_put(array, NamedSharding(mesh, P(_axis(mesh))))


@functools.lru_cache(maxsize=8)
def _cached_take_fn(shard: NamedSharding):
  @functools.partial(jax.jit, out_shardings=shard)
  def _take(w, i):
    return jnp.take(w, i, axis=0)

  return _take


def remap_weights(
    mesh: Any,
    weights: tuple[Any, ...],
    slot_to_orig: np.ndarray | None,
) -> tuple[Any, ...]:
  """Permutes/remaps the expert axis (axis 0) of `(w1, w2, w1_scale, w2_scale)`."""
  if slot_to_orig is None:
    return weights
  slot_to_orig = np.asarray(slot_to_orig, dtype=np.int32)
  if np.array_equal(slot_to_orig, np.arange(len(slot_to_orig), dtype=np.int32)):
    return weights
  shard = NamedSharding(mesh, P(_axis(mesh)))
  rep = NamedSharding(mesh, P())
  idx = jax.device_put(slot_to_orig, rep)
  take_fn = _cached_take_fn(shard)

  out = []
  for w in weights:
    out.append(None if w is None else jax.block_until_ready(take_fn(w, idx)))
  return tuple(out)
