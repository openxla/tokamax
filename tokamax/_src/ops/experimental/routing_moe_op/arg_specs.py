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
import zlib

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
  chosen = np.argsort(-scores, axis=-1)[:, :k]
  weights = np.take_along_axis(scores, chosen, axis=-1)
  weights = (weights / np.sum(weights, axis=-1, keepdims=True)).astype(
      np.float32
  )
  return weights, chosen.astype(np.int32)


class _TopkRoutingArrayInitializer(
    jax.ShapeDtypeStruct, numerics.ArrayInitializer
):
  """Initializer for valid top-k routing weights and expert indices."""

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
  mesh = jax.sharding.AbstractMesh(
      (ep,), (_AXIS,), axis_types=(jax.sharding.AxisType.Auto,)
  )
  act_sharding = jax.sharding.NamedSharding(mesh, _P(_AXIS))
  w_sharding = jax.sharding.NamedSharding(mesh, _P(_AXIS, None, None))
  w_scale_sharding = jax.sharding.NamedSharding(
      mesh, _P(_AXIS, None, None, None)
  )

  routing_seed = zlib.adler32(
      f"{project}_{name}_{num_tokens}_{num_experts}_{top_k}".encode()
  )

  args = dict(
      hidden_states=jax.ShapeDtypeStruct(
          (num_tokens, hidden_size), dtype, sharding=act_sharding
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
      w1=jax.ShapeDtypeStruct(
          (num_experts, hidden_size, 2 * intermediate_size),
          w_dtype,
          sharding=w_sharding,
      ),
      w2=jax.ShapeDtypeStruct(
          (num_experts, intermediate_size, hidden_size),
          w_dtype,
          sharding=w_sharding,
      ),
      mesh=mesh,
      activation="silu",
      tp_axis=None,
      ep_axis=_AXIS,
      use_sp=True,
  )
  if jnp.dtype(w_dtype) != jnp.dtype(dtype):
    args["w1_scale"] = jax.ShapeDtypeStruct(
        (num_experts, 1, 1, 2 * intermediate_size),
        jnp.float32,
        sharding=w_scale_sharding,
    )
    args["w2_scale"] = jax.ShapeDtypeStruct(
        (num_experts, 1, 1, hidden_size),
        jnp.float32,
        sharding=w_scale_sharding,
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


def _qwen3_5_397b_specs() -> tuple[arg_spec.ArgSpec, ...]:
  """Qwen3.5-397B MoE layer with FP8 weights and EP=8."""
  return tuple(
      _make_spec(
          name,
          num_tokens=tokens,
          num_experts=512,
          hidden_size=4096,
          intermediate_size=1024,
          top_k=10,
          ep=8,
          project="qwen3.5_397b",
          tags=("primary", "forward_only"),
      )
      for name, tokens in _INFERENCEX_8K1K_NUM_TOKENS
  )


ARG_SPECS: tuple[arg_spec.ArgSpec, ...] = _qwen3_5_397b_specs()
