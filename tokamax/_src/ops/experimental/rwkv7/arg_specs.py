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
"""RWKV-7 (DPLR Delta Rule) argument specifications."""

from typing import Any, Final

import jax
import jax.numpy as jnp
from tokamax._src import numerics
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct


def _rwkv7_workload_spec(
    *,
    name: str,
    batch_size: int = 1,
    seq_len: int = 2048,
    num_heads: int = 32,
    key_dim: int = 128,
    val_dim: int = 128,
    dtype: Any = jnp.bfloat16,
) -> arg_spec.ArgSpec:
  """Builds a representative RWKV-7 workload."""
  qk_shape = (batch_size, seq_len, num_heads, key_dim)
  v_shape = (batch_size, seq_len, num_heads, val_dim)

  args = {
      "q": ShapeDtype(qk_shape, dtype),
      "k": ShapeDtype(qk_shape, dtype),
      "v": ShapeDtype(v_shape, dtype),
      "alpha": ShapeDtype(qk_shape, dtype),
      "beta": ShapeDtype(qk_shape, dtype),
      "gk": numerics.RangedArrayInitializer(
          shape=qk_shape, dtype=dtype, minval=-5.0, maxval=0.0
      ),
      "chunk_size": 64,
      "scale": key_dim**-0.5,
      "output_final_state": False,
  }

  return arg_spec.ArgSpec(
      args=args,
      project="rwkv7",
      name=name,
      tags=("primary",),
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _rwkv7_workload_spec(
        name="training_b1_t2048_h32_k128_v128_bf16",
        batch_size=1,
        seq_len=2048,
    ),
    _rwkv7_workload_spec(
        name="training_b1_t8192_h32_k128_v128_bf16",
        batch_size=1,
        seq_len=8192,
    ),
)
