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
"""Input generation and routing helpers shared across routing_moe tests."""

import dataclasses
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.test.common import test_cases
from tokamax._src.ops.experimental.routing_moe.test.common import test_shapes

Mesh = jax.sharding.Mesh


def mesh_devices() -> list[jax.Device]:
  """Returns devices ordered so mesh neighbors are physical neighbors."""
  return sorted(
      jax.devices(),
      key=lambda x: (
          x.coords[0],
          (-1 if x.coords[0] % 2 else 1) * x.coords[1],
      ),
  )


def make_mesh(
    sharding: test_cases.MoESharding = test_cases.MoESharding(),
) -> Mesh:
  """Creates a `(dp, ep, tp)` device mesh for `sharding`."""
  devices = mesh_devices()[: sharding.num_cores]
  return Mesh(
      np.array(devices).reshape(sharding),
      axis_names=test_cases.MESH_AXES,
  )


def random_gating(
    key: jax.Array,
    num_tokens: int,
    num_experts: int,
    *,
    active_expert_ratio: float = 1.0,
    num_skipped: int = 0,
    force_expert: int | None = None,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> jax.Array:
  """Generates uniform router logits with optional expert masking."""
  k_logits, k_active = jax.random.split(key)
  gating_output = jax.random.uniform(
      k_logits, (num_tokens, num_experts), dtype=dtype
  )
  if active_expert_ratio < 1.0:
    score = jax.random.normal(k_active, (num_experts,))
    num_active = int(num_experts * active_expert_ratio)
    threshold, _ = jax.lax.top_k(score, k=num_active)
    gating_output = jnp.where(score >= threshold[-1], gating_output, 0.0)
  if num_skipped:
    gating_output = gating_output.at[:, num_experts - num_skipped :].set(-1)
  if force_expert is not None:
    gating_output = gating_output.at[0, force_expert].set(2.0)
  return gating_output


def top_k(
    gating_output: jax.Array, k: int, *, renormalize: bool = True
) -> tuple[jax.Array, jax.Array]:
  """Computes softmax top-k routing weights and expert indices."""
  weights = jax.nn.softmax(gating_output.astype(jnp.float32), axis=-1)
  weights, indices = jax.lax.top_k(weights, k=k)
  if renormalize:
    weights = weights / weights.sum(axis=-1, keepdims=True)
  return weights, indices.astype(jnp.int32)


def balanced_topk_indices(
    key: jax.Array, num_tokens: int, topk: int, num_experts: int, ep_size: int
) -> jax.Array:
  """Generates top-k expert indices balanced evenly across `ep_size` shards."""
  num_local_experts = num_experts // ep_size
  gating_output = jax.random.uniform(
      key, (num_tokens, ep_size, num_local_experts)
  )
  _, local_indices = jax.lax.top_k(gating_output, topk // ep_size)
  shard_offsets = jnp.arange(ep_size)[:, None] * num_local_experts
  return (local_indices + shard_offsets).reshape(num_tokens, topk)


def sub_channel_quantize(
    x: jax.Array,
    quant_dtype: jax.typing.DTypeLike,
    block_size: int | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes `x` along its second-most-minor axis and returns `(q, scale)`."""
  if block_size is None:
    block_size = x.shape[-2]
  if jnp.issubdtype(quant_dtype, jnp.floating):
    dtype_max = float(jnp.finfo(quant_dtype).max)
  else:
    dtype_max = float(jnp.iinfo(quant_dtype).max)
  assert x.ndim >= 2 and x.shape[-2] % block_size == 0
  w_lst, scale_lst = [], []
  for i in range(0, x.shape[-2], block_size):
    y = x[:, i : i + block_size, :]
    abs_max = jnp.abs(y).max(axis=1, keepdims=True)
    scale = (abs_max / dtype_max).astype(jnp.float32)
    w_lst.append((y / scale).astype(quant_dtype))
    scale_lst.append(scale)
  return jnp.concat(w_lst, axis=1), jnp.concat(scale_lst, axis=1)[:, :, None]


def _random_normal(
    key: jax.Array,
    shape: tuple[int, ...],
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> jax.Array:
  return jax.random.normal(key, shape, dtype=dtype) / 10


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MoEInputs:
  """Every array one MoE layer call takes. Optional parts are None.

  Attributes:
    hidden_states: `[num_tokens, hidden]`.
    gating_output: `[num_tokens, num_experts]` router logits.
    w1: `[num_experts, hidden, 2 * intermediate]` gate and up weights.
    w2: `[num_experts, intermediate, hidden]` down weights.
    w1_scale: `[num_experts, num_blocks, 1, 2 * intermediate]` scales of `w1`.
    w2_scale: `[num_experts, num_blocks, 1, hidden]` scales of `w2`.
    w1_bias: `[num_experts, 1, 2 * intermediate]`.
    w2_bias: `[num_experts, 1, hidden]`.
    w1_shared: `[1, hidden, 2 * intermediate]` shared expert gate/up.
    w2_shared: `[1, intermediate, hidden]` shared expert down.
    w1_shared_scale: Scales of `w1_shared`.
    w2_shared_scale: Scales of `w2_shared`.
    w_shared_gate: `[hidden, 1]` shared expert gate.
  """

  hidden_states: jax.Array
  gating_output: jax.Array
  w1: jax.Array
  w2: jax.Array
  w1_scale: jax.Array | None = None
  w2_scale: jax.Array | None = None
  w1_bias: jax.Array | None = None
  w2_bias: jax.Array | None = None
  w1_shared: jax.Array | None = None
  w2_shared: jax.Array | None = None
  w1_shared_scale: jax.Array | None = None
  w2_shared_scale: jax.Array | None = None
  w_shared_gate: jax.Array | None = None

  @classmethod
  def generate(
      cls,
      shape: test_shapes.MoEShape,
      num_tokens: int,
      *,
      seed: int = 0,
      active_expert_ratio: float = 1.0,
      num_skipped: int = 0,
      force_expert: int | None = None,
  ) -> Self:
    """Generates unsharded, unquantized bfloat16 inputs for `shape`."""
    keys = jax.random.split(jax.random.key(seed), 8)
    e, h, i = shape.num_experts, shape.hidden_size, shape.intermediate_size
    dtype = jnp.bfloat16

    inputs = cls(
        hidden_states=jax.random.uniform(keys[0], (num_tokens, h), dtype) / 10,
        gating_output=random_gating(
            keys[1],
            num_tokens,
            e,
            active_expert_ratio=active_expert_ratio,
            num_skipped=num_skipped,
            force_expert=force_expert,
            dtype=dtype,
        ),
        w1=_random_normal(keys[3], (e, h, 2 * i)),
        w2=_random_normal(keys[4], (e, i, h)),
    )
    if shape.has_bias:
      inputs = dataclasses.replace(
          inputs,
          w1_bias=_random_normal(keys[5], (e, 1, 2 * i)),
          w2_bias=_random_normal(keys[6], (e, 1, h)),
      )
    if shape.has_shared_expert:
      k1, k2, k3 = jax.random.split(keys[7], 3)
      inputs = dataclasses.replace(
          inputs,
          w1_shared=_random_normal(k1, (1, h, 2 * i)),
          w2_shared=_random_normal(k2, (1, i, h)),
          w_shared_gate=_random_normal(k3, (h, 1)),
      )
    return inputs

  def fold_shared_expert(self) -> Self:
    """Appends the shared expert as expert `num_experts` of routed weights."""
    assert self.w1_shared is not None and self.w2_shared is not None
    return dataclasses.replace(
        self,
        w1=jnp.concatenate([self.w1, self.w1_shared], axis=0),
        w2=jnp.concatenate([self.w2, self.w2_shared], axis=0),
        w1_shared=None,
        w2_shared=None,
    )

  def quantize(self, w_dtype: jnp.dtype, block_size: int | None) -> Self:
    """Quantizes every weight to `w_dtype` with attached scales."""

    def _quant(w: jax.Array) -> tuple[jax.Array, jax.Array]:
      return sub_channel_quantize(w, w_dtype, block_size)

    def _maybe_quant(
        w: jax.Array | None,
    ) -> tuple[jax.Array | None, jax.Array | None]:
      return (None, None) if w is None else _quant(w)

    w1, w1_scale = _quant(self.w1)
    w2, w2_scale = _quant(self.w2)
    w1_shared, w1_shared_scale = _maybe_quant(self.w1_shared)
    w2_shared, w2_shared_scale = _maybe_quant(self.w2_shared)
    return dataclasses.replace(
        self,
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_shared=w1_shared,
        w2_shared=w2_shared,
        w1_shared_scale=w1_shared_scale,
        w2_shared_scale=w2_shared_scale,
    )


@dataclasses.dataclass(frozen=True)
class Routing:
  """A top-k routing sorted by expert, and the slice one EP shard serves.

  Rows are `(token, pick)` pairs. Sorting them by expert gives the order the
  kernel computes in. The shard serves experts `[group_offset, group_offset +
  num_local_experts)`, which is the sorted-row range `[row_start, row_end)`.

  Attributes:
    topk_indices: `[num_tokens, topk]` expert of each pick.
    group_sizes: `[num_experts]` rows routed to each expert.
    row_group: `[num_rows]` expert of each sorted row.
    sorted_to_flat: `[num_rows]` flat `(token, pick)` slot of each sorted row.
    group_offset: First expert of the shard.
    num_local_experts: Experts per shard.
    row_start: First sorted row of the shard.
    row_end: One past the last sorted row of the shard.
  """

  topk_indices: jax.Array
  group_sizes: jax.Array
  row_group: jax.Array
  sorted_to_flat: jax.Array
  group_offset: int
  num_local_experts: int
  row_start: int
  row_end: int

  @property
  def topk(self) -> int:
    """Number of selected experts per token."""
    return self.topk_indices.shape[1]

  @property
  def gather_indices(self) -> jax.Array:
    """Token index of each sorted row."""
    return self.sorted_to_flat // self.topk

  @property
  def local_group_sizes(self) -> jax.Array:
    """Row counts for the experts on this shard."""
    lo = self.group_offset
    return self.group_sizes[lo : lo + self.num_local_experts]

  @property
  def sorted_rows(self) -> np.ndarray:
    """Output rows written by the shard when output stays in sorted order."""
    return np.arange(self.row_start, self.row_end)

  @property
  def scattered_rows(self) -> jax.Array:
    """Output rows written by the shard when output is scattered by slot."""
    return self.sorted_to_flat[self.row_start : self.row_end]

  @property
  def gather(self) -> arg.RoutingWrapper[jax.Array]:
    """Routing wrapper for gathering token rows into sorted order."""
    return arg.make_routing_wrapper(self.gather_indices)

  @property
  def scatter(self) -> arg.RoutingWrapper[jax.Array]:
    """Routing wrapper for scattering sorted rows back to flat slots."""
    return arg.make_routing_wrapper(self.sorted_to_flat)

  @property
  def kernel_kwargs(self) -> dict[str, jax.Array]:
    """Group size and offset keyword arguments for `routing_gmm`."""
    return dict(
        group_sizes=self.group_sizes,
        group_offset=jnp.array([self.group_offset], jnp.int32),
        row_group=self.row_group,
    )

  @classmethod
  def make_routing(
      cls,
      topk_indices: jax.Array,
      num_experts: int,
      *,
      group_offset: int = 0,
      num_local_experts: int | None = None,
  ) -> Self:
    """Sorts `topk_indices` by expert and locates the shard's row range."""
    if num_local_experts is None:
      num_local_experts = num_experts
    flat = topk_indices.flatten()
    row_group, sorted_to_flat = jax.lax.sort_key_val(
        flat, jnp.arange(flat.size, dtype=jnp.int32), is_stable=True
    )
    group_sizes = jnp.bincount(flat, length=num_experts)
    group_cum_sum = jnp.cumulative_sum(group_sizes, include_initial=True)
    return cls(
        topk_indices=topk_indices,
        group_sizes=group_sizes,
        row_group=row_group,
        sorted_to_flat=sorted_to_flat,
        group_offset=group_offset,
        num_local_experts=num_local_experts,
        row_start=int(group_cum_sum[group_offset]),
        row_end=int(group_cum_sum[group_offset + num_local_experts]),
    )
