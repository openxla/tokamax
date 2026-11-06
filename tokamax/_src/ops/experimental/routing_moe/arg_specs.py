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
"""Benchmark and autotuning argument specifications for Routing MoE.

The Routing MoE kernel is expert-parallel (EP), so its argument specifications
are multi-device: each spec contains an abstract `mesh` and arrays sharded over
it. Benchmark runners must replace the abstract mesh (and the shardings that use
it) with a concrete `jax.sharding.Mesh` before initializing the arrays.
"""

import jax
import jax.numpy as jnp
from tokamax._src.autotuning import arg_spec

_AXIS = "d"
_P = jax.sharding.PartitionSpec


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
    renormalize: bool = True,
    project: str,
    tags: tuple[arg_spec.Tag, ...] = ("forward_only",),
) -> arg_spec.ArgSpec:
  """Creates an EP-sharded `ArgSpec` for Routing MoE.

  Tokens (`x`, `gating`) and experts (`w1`, `w2` and scales) are sharded along
  the leading dimension.

  Args:
    name: The name of the argument specification.
    num_tokens: Global number of tokens.
    num_experts: Global number of experts.
    hidden_size: Model hidden size.
    intermediate_size: Expert intermediate size.
    top_k: Number of experts per token.
    ep: Expert-parallel degree (number of devices).
    dtype: Activation dtype.
    w_dtype: Weight dtype. Per-channel float32 scales are added if quantized.
    renormalize: Whether to renormalize the top-k gating weights.
    project: The project the argument specification comes from.
    tags: Tags for the argument specification.

  Returns:
    The argument specification.
  """
  mesh = _abstract_mesh(ep)
  sharding = jax.sharding.NamedSharding(mesh, _P(_AXIS))
  args = dict(
      x=_sharded_sds((num_tokens, hidden_size), dtype, sharding),
      w1=_sharded_sds(
          (num_experts, hidden_size, 2 * intermediate_size), w_dtype, sharding
      ),
      w2=_sharded_sds(
          (num_experts, intermediate_size, hidden_size), w_dtype, sharding
      ),
      gating=_sharded_sds((num_tokens, num_experts), jnp.float32, sharding),
      topk=top_k,
      renormalize=renormalize,
      act_fn="silu",
      mesh=mesh,
  )
  if jnp.dtype(w_dtype) != jnp.dtype(dtype):
    args["w1_scale"] = _sharded_sds(
        (num_experts, 2 * intermediate_size), jnp.float32, sharding
    )
    args["w2_scale"] = _sharded_sds(
        (num_experts, hidden_size), jnp.float32, sharding
    )
  return arg_spec.ArgSpec(args=args, project=project, name=name, tags=tags)


def _qwen3_5_397b_specs() -> tuple[arg_spec.ArgSpec, ...]:
  """Qwen3.5-397B MoE layer with FP8 weights and EP=8."""
  num_tokens = {
      "decode_32": 32,
      "decode_64": 64,
      "decode_128": 128,
      "decode_256": 256,
      "decode_512": 512,
      "decode_1024": 1024,
      "prefill_8192": 8192,
      "prefill_16384": 16384,
  }
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
      for name, tokens in num_tokens.items()
  )


ARG_SPECS: tuple[arg_spec.ArgSpec, ...] = _qwen3_5_397b_specs()
