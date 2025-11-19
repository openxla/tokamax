# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Triangle multiplication argument specifications."""

import itertools
from typing import Final

import jax
import jax.numpy as jnp
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct


def _create_arg_spec(seq_len: int, triangle_type: str) -> arg_spec.ArgSpec:
  return arg_spec.ArgSpec(
      dict(
          x=ShapeDtype((seq_len, seq_len, 64), jnp.bfloat16),
          mask=ShapeDtype((seq_len, seq_len), jnp.bool_),
          projection_in_weights=ShapeDtype((64, 2, 64), jnp.bfloat16),
          gate_in_weights=ShapeDtype((64, 2, 64), jnp.bfloat16),
          projection_out_weights=ShapeDtype((64, 64), jnp.bfloat16),
          gate_out_weights=ShapeDtype((64, 64), jnp.bfloat16),
          layernorm_in_scale=ShapeDtype((64,), jnp.bfloat16),
          layernorm_in_offset=ShapeDtype((64,), jnp.bfloat16),
          layernorm_out_scale=ShapeDtype((64,), jnp.bfloat16),
          layernorm_out_offset=ShapeDtype((64,), jnp.bfloat16),
          triangle_type=triangle_type,
      ),
      project='alphafold',
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = tuple(
    _create_arg_spec(n, t)
    for n, t in itertools.product(
        (384, 640, 1024, 2048), ('outgoing', 'incoming')
    )
)
