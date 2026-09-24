# Copyright 2026 Google LLC. All Rights Reserved.
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
"""Splash Attention argument specifications."""

from typing import Any, Final
import jax
import jax.numpy as jnp
from tokamax._src.autotuning import arg_spec
from tokamax._src.ops.experimental.tpu.splash_attention import base

ShapeDtype = jax.ShapeDtypeStruct
Mask = base.Mask


def _create_spec(
    name: str,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    q_seq_len: int,
    kv_seq_len: int,
    head_dim_qk: int,
    head_dim_v: int,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    is_causal: bool = True,
    is_mqa: bool = False,
    attn_logits_soft_cap: float | None = None,
    project: str = "",
    tags: tuple[arg_spec.Tag, ...] = (),
) -> arg_spec.ArgSpec:
  """Helper to construct an ArgSpec for Splash Attention."""
  q = ShapeDtype((num_q_heads, q_seq_len, head_dim_qk), dtype)
  if is_mqa:
    k = ShapeDtype((kv_seq_len, head_dim_qk), dtype)
    v = ShapeDtype((kv_seq_len, head_dim_v), dtype)
  else:
    k = ShapeDtype((num_kv_heads, kv_seq_len, head_dim_qk), dtype)
    v = ShapeDtype((num_kv_heads, kv_seq_len, head_dim_v), dtype)

  mask = base.CAUSAL_MASK if is_causal else base.FULL_MASK

  args: dict[str, Any] = {
      "q": q,
      "k": k,
      "v": v,
      "mask": mask,
      "is_mqa": is_mqa,
  }

  if attn_logits_soft_cap is not None:
    args["attn_logits_soft_cap"] = attn_logits_soft_cap

  return arg_spec.ArgSpec(
      args=args,
      project=project,
      name=name,
      tags=tags,
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _create_spec(
        name="q64_kv8_s8192_dqk128_dv128",
        num_q_heads=64,
        num_kv_heads=8,
        q_seq_len=8192,
        kv_seq_len=8192,
        head_dim_qk=128,
        head_dim_v=128,
        dtype=jnp.bfloat16,
        is_causal=True,
        is_mqa=False,
        tags=("primary",),
    ),
    _create_spec(
        name="q128_kv8_s8192_dqk128_dv128",
        num_q_heads=128,
        num_kv_heads=8,
        q_seq_len=8192,
        kv_seq_len=8192,
        head_dim_qk=128,
        head_dim_v=128,
        dtype=jnp.bfloat16,
        is_causal=True,
        is_mqa=False,
        tags=("primary",),
    ),
    _create_spec(
        name="q128_kv128_s4096_dqk192_dv128",
        num_q_heads=128,
        num_kv_heads=128,
        q_seq_len=4096,
        kv_seq_len=4096,
        head_dim_qk=192,
        head_dim_v=128,
        dtype=jnp.bfloat16,
        is_causal=True,
        is_mqa=False,
        tags=("primary",),
    ),
)
