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

"""Raven Gated Slot Attention (GSA) autotuning argument specifications."""

from typing import Final
import jax
import jax.numpy as jnp
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct


def _make_raven_spec(
    *,
    name: str,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    key_dim: int,
    num_slots: int,
    val_dim: int | None = 128,
    chunk_size: int = 64,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    tags: tuple[arg_spec.Tag, ...] = (),
) -> arg_spec.ArgSpec:
  """Constructs an autotuning ArgSpec for Raven GSA."""
  args = {
      "q": ShapeDtype((batch_size, seq_len, num_heads, key_dim), dtype),
      "k": ShapeDtype((batch_size, seq_len, num_heads, key_dim), dtype),
      "s": ShapeDtype((batch_size, seq_len, num_heads, num_slots), dtype),
      "g": ShapeDtype((batch_size, seq_len, num_heads, num_slots), jnp.float32),
      "chunk_size": chunk_size,
      "output_final_state": False,
  }
  if val_dim is not None:
    args["v"] = ShapeDtype((batch_size, seq_len, num_heads, val_dim), dtype)
  return arg_spec.ArgSpec(
      args=args,
      project="raven_gsa",
      name=name,
      tags=tags,
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _make_raven_spec(
        name="raven_gsa_b2_t1024_h8_k128_m64_v128_bf16",
        batch_size=2,
        seq_len=1024,
        num_heads=8,
        key_dim=128,
        num_slots=64,
        val_dim=128,
        dtype=jnp.bfloat16,
        tags=("primary",),
    ),
    _make_raven_spec(
        name="raven_gsa_b4_t2048_h8_k128_m64_v128_bf16",
        batch_size=4,
        seq_len=2048,
        num_heads=8,
        key_dim=128,
        num_slots=64,
        val_dim=128,
        dtype=jnp.bfloat16,
    ),
)
