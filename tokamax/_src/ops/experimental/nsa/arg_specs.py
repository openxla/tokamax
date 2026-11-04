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

"""Native Sparse Attention (NSA) benchmark argument specifications."""

from typing import Final
import jax
import jax.numpy as jnp
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct


def _nsa_spec(
    *,
    name: str,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    chunk_size: int = 256,
    topk: int = 4,
    window: int = 128,
    cmp_block_size: int = 64,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    tags: tuple[arg_spec.Tag, ...] = ("primary",),
) -> arg_spec.ArgSpec:
  shape = (batch_size, num_heads, seq_len, head_dim)
  return arg_spec.ArgSpec(
      args={
          "q": ShapeDtype(shape, dtype),
          "k": ShapeDtype(shape, dtype),
          "v": ShapeDtype(shape, dtype),
          "g_cmp": ShapeDtype(shape, dtype),
          "g_slc": ShapeDtype(shape, dtype),
          "g_swa": ShapeDtype(shape, dtype),
          "chunk_size": chunk_size,
          "topk": topk,
          "window": window,
          "cmp_block_size": cmp_block_size,
          "scale": head_dim**-0.5,
      },
      project="nsa",
      name=name,
      tags=tags,
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _nsa_spec(
        name="training_b1_t8192_h16_d128_bf16",
        batch_size=1,
        num_heads=16,
        seq_len=8192,
        head_dim=128,
        tags=("primary", "ci_tests"),
    ),
    _nsa_spec(
        name="training_b2_t4096_h8_d64_bf16",
        batch_size=2,
        num_heads=8,
        seq_len=4096,
        head_dim=64,
        tags=("primary",),
    ),
    _nsa_spec(
        name="long_context_b1_t16384_h16_d128_bf16",
        batch_size=1,
        num_heads=16,
        seq_len=16384,
        head_dim=128,
        tags=("primary",),
    ),
)
