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
"""Benchmark and autotuning argument specifications for Routing MoE."""

from __future__ import annotations

import functools
import hashlib
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import numerics
from tokamax._src.autotuning import arg_spec

_AXIS = "d"
_P = jax.sharding.PartitionSpec


@functools.lru_cache(maxsize=128)
def _compute_deterministic_topk_routing(
    num_tokens: int, num_experts: int, k: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
  """Computes deterministic top-k weights and indices from random logits."""
  rng = np.random.default_rng(seed)
  logits = rng.standard_normal((num_tokens, num_experts), dtype=np.float32)
  max_logits = np.max(logits, axis=-1, keepdims=True)
  exp_logits = np.exp(logits - max_logits)
  scores = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

  partitioned_indices = np.argpartition(-scores, k, axis=-1)[:, :k]
  row_indices = np.arange(num_tokens)[:, None]
  partitioned_scores = scores[row_indices, partitioned_indices]
  sort_order = np.argsort(-partitioned_scores, axis=-1)

  topk_indices = np.take_along_axis(
      partitioned_indices, sort_order, axis=-1
  ).astype(np.int32)
  topk_weights = np.take_along_axis(partitioned_scores, sort_order, axis=-1)
  topk_weights = (
      topk_weights / np.sum(topk_weights, axis=-1, keepdims=True)
  ).astype(np.float32)
  return topk_weights, topk_indices


class _TopkRoutingArrayInitializer(
    jax.ShapeDtypeStruct, numerics.ArrayInitializer
):
  """Abstract array initializer for valid top-k routing weights and expert indices."""

  def __init__(
      self,
      *,
      num_tokens: int,
      num_experts: int,
      k: int,
      field: str,
      seed: int,
      sharding: jax.sharding.Sharding | None = None,
  ):
    dtype = jnp.int32 if field == "indices" else jnp.float32
    super().__init__((num_tokens, k), dtype, sharding=sharding)
    self.num_tokens = num_tokens
    self.num_experts = num_experts
    self.k = k
    self.field = field
    self.seed = seed

  def __call__(self, rng: np.random.Generator) -> np.ndarray:
    del rng
    weights, indices = _compute_deterministic_topk_routing(
        self.num_tokens, self.num_experts, self.k, self.seed
    )
    return indices if self.field == "indices" else weights

  def with_sharding(
      self, sharding: jax.sharding.Sharding
  ) -> _TopkRoutingArrayInitializer:
    return _TopkRoutingArrayInitializer(
        num_tokens=self.num_tokens,
        num_experts=self.num_experts,
        k=self.k,
        field=self.field,
        seed=self.seed,
        sharding=sharding,
    )


def _abstract_mesh(ep: int) -> jax.sharding.AbstractMesh:
  return jax.sharding.AbstractMesh(
      (ep,), (_AXIS,), axis_types=(jax.sharding.AxisType.Auto,)
  )


def _sharded_sds(
    shape: tuple[int, ...],
    dtype: jax.typing.DTypeLike,
    sharding: jax.sharding.NamedSharding,
) -> jax.ShapeDtypeStruct:
  return jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)


def _make_spec(
    name: str,
    *,
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    ep: int = 8,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    w_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    project: str,
    tags: tuple[arg_spec.Tag, ...] = ("forward_only",),
) -> arg_spec.ArgSpec:
  """Creates an EP-sharded `ArgSpec` for Routing MoE in modular layout."""
  mesh = _abstract_mesh(ep)
  act_sharding = jax.sharding.NamedSharding(mesh, _P(_AXIS))
  w_sharding = jax.sharding.NamedSharding(mesh, _P(_AXIS, None, None))
  w_scale_sharding = jax.sharding.NamedSharding(
      mesh, _P(_AXIS, None, None, None)
  )

  seed_bytes = hashlib.sha256(
      f"{project}_{name}_{num_tokens}_{num_experts}_{top_k}".encode("utf-8")
  ).digest()[:4]
  routing_seed = int.from_bytes(seed_bytes, "little")

  args = dict(
      hidden_states=_sharded_sds(
          (num_tokens, hidden_size), dtype, act_sharding
      ),
      topk_weights=_TopkRoutingArrayInitializer(
          num_tokens=num_tokens,
          num_experts=num_experts,
          k=top_k,
          field="weights",
          seed=routing_seed,
          sharding=act_sharding,
      ),
      topk_indices=_TopkRoutingArrayInitializer(
          num_tokens=num_tokens,
          num_experts=num_experts,
          k=top_k,
          field="indices",
          seed=routing_seed,
          sharding=act_sharding,
      ),
      w1=_sharded_sds(
          (num_experts, hidden_size, 2 * intermediate_size),
          w_dtype,
          w_sharding,
      ),
      w2=_sharded_sds(
          (num_experts, intermediate_size, hidden_size), w_dtype, w_sharding
      ),
      mesh=mesh,
      activation="silu",
      tp_axis=None,
      ep_axis=_AXIS,
      use_sp=True,
  )
  if jnp.dtype(w_dtype) != jnp.dtype(dtype):
    args["w1_scale"] = _sharded_sds(
        (num_experts, 1, 1, 2 * intermediate_size),
        jnp.float32,
        w_scale_sharding,
    )
    args["w2_scale"] = _sharded_sds(
        (num_experts, 1, 1, hidden_size), jnp.float32, w_scale_sharding
    )
  return arg_spec.ArgSpec(args=args, project=project, name=name, tags=tags)


_INFERENCEX_8K1K_NUM_TOKENS: tuple[tuple[str, int], ...] = (
    ("decode_32", 32),
    ("decode_64", 64),
    ("decode_128", 128),
    ("decode_256", 256),
    ("decode_512", 512),
    ("decode_1024", 1024),
    ("prefill_8192", 8192),
    ("prefill_16384", 16384),
)


_DSV4_FLASH_NUM_TOKENS: tuple[tuple[str, int], ...] = (
    ("decode_64", 64),
    ("decode_256", 256),
    ("prefill_1024", 1024),
    ("prefill_8192", 8192),
)

_DSV4_PRO_NUM_TOKENS: tuple[tuple[str, int], ...] = (
    ("decode_256", 256),
    ("decode_512", 512),
    ("prefill_1024", 1024),
    ("prefill_16384", 16384),
)

_GLM5_2_NUM_TOKENS: tuple[tuple[str, int], ...] = (
    ("decode_16", 16),
    ("decode_256", 256),
    ("prefill_4096", 4096),
    ("prefill_8192", 8192),
)

_KIMI_K3_NUM_TOKENS: tuple[tuple[str, int], ...] = (
    ("decode_16", 16),
    ("decode_256", 256),
    ("prefill_8192", 8192),
    ("prefill_16384", 16384),
)


def _model_specs(
    *,
    project: str,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    tags: tuple[arg_spec.Tag, ...],
    token_counts: tuple[tuple[str, int], ...] = _INFERENCEX_8K1K_NUM_TOKENS,
) -> tuple[arg_spec.ArgSpec, ...]:
  """Creates one model's specs over the given token counts, EP=8."""
  return tuple(
      _make_spec(
          name,
          num_tokens=tokens,
          num_experts=num_experts,
          hidden_size=hidden_size,
          intermediate_size=intermediate_size,
          top_k=top_k,
          ep=8,
          project=project,
          tags=tags,
      )
      for name, tokens in token_counts
  )


def _qwen3_5_397b_specs() -> tuple[arg_spec.ArgSpec, ...]:
  """Qwen3.5-397B MoE layer with FP8 weights and EP=8."""
  return _model_specs(
      project="qwen3.5_397b",
      num_experts=512,
      hidden_size=4096,
      intermediate_size=1024,
      top_k=10,
      tags=("primary", "forward_only"),
  )


def _qwen3_8_specs() -> tuple[arg_spec.ArgSpec, ...]:
  """Qwen3.8-Max MoE layers with FP8 weights and EP=8.

  Qwen3.8-Flash-Next (intermediate_size=640) is omitted until the kernel
  supports intermediate sizes that are not divisible by 256.

  Returns:
    Argument specs for Qwen3.8-Max MoE benchmarks.
  """
  return _model_specs(
      project="qwen3.8_max",
      num_experts=512,
      hidden_size=8192,
      intermediate_size=2048,
      top_k=10,
      tags=("forward_only",),
  )


def _dsv4_flash_specs() -> tuple[arg_spec.ArgSpec, ...]:
  """DeepSeek-V4-Flash MoE layer with FP8 weights and EP=8."""
  return _model_specs(
      project="dsv4_flash",
      num_experts=256,
      hidden_size=4096,
      intermediate_size=2048,
      top_k=6,
      tags=("forward_only",),
      token_counts=_DSV4_FLASH_NUM_TOKENS,
  )


def _dsv4_pro_specs() -> tuple[arg_spec.ArgSpec, ...]:
  """DeepSeek-V4-Pro MoE layer with FP8 weights and EP=8."""
  return _model_specs(
      project="dsv4_pro",
      num_experts=384,
      hidden_size=7168,
      intermediate_size=3072,
      top_k=6,
      tags=("forward_only",),
      token_counts=_DSV4_PRO_NUM_TOKENS,
  )


def _glm5_2_specs() -> tuple[arg_spec.ArgSpec, ...]:
  """GLM-5.2 MoE layer with FP8 weights and EP=8."""
  return _model_specs(
      project="glm5.2",
      num_experts=256,
      hidden_size=6144,
      intermediate_size=2048,
      top_k=8,
      tags=("forward_only",),
      token_counts=_GLM5_2_NUM_TOKENS,
  )


def _kimi_k3_specs() -> tuple[arg_spec.ArgSpec, ...]:
  """Kimi K3 MoE layer with FP8 weights and EP=8."""
  return _model_specs(
      project="kimi_k3",
      num_experts=896,
      hidden_size=3584,
      intermediate_size=3072,
      top_k=16,
      tags=("forward_only",),
      token_counts=_KIMI_K3_NUM_TOKENS,
  )


ARG_SPECS: tuple[arg_spec.ArgSpec, ...] = (
    _qwen3_5_397b_specs()
    + _qwen3_8_specs()
    + _dsv4_flash_specs()
    + _dsv4_pro_specs()
    + _glm5_2_specs()
    + _kimi_k3_specs()
)
