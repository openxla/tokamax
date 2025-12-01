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
"""Linear softmax cross entropy loss argument specifications."""

import itertools
from typing import Final

import jax
import jax.numpy as jnp
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct


def _create_arg_spec(
    batch_dim: int,
    hidden_dim: int,
    output_dim: int,
    reduction: str,
    x_dtype: jax.typing.DTypeLike,
    w_dtype: jax.typing.DTypeLike,
    project: str = '',
) -> arg_spec.ArgSpec:
  return arg_spec.ArgSpec(
      dict(
          x=ShapeDtype((batch_dim, hidden_dim), x_dtype),
          labels=ShapeDtype((batch_dim,), jnp.int32),
          w=ShapeDtype((hidden_dim, output_dim), w_dtype),
          reduction=reduction,
      ),
      project=project,
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    tuple(
        _create_arg_spec(
            batch_dim=batch_dim,
            hidden_dim=4096,
            output_dim=151936,
            reduction=reduction,
            x_dtype=x_dtype,
            w_dtype=jnp.bfloat16,
            project='qwen3-8b',
        )
        for batch_dim, reduction, x_dtype in itertools.product(
            [1024, 2048, 4096, 16384, 32768, 131072],
            ('mean', 'sum'),
            (jnp.bfloat16, jnp.float32),
        )
    )
    + tuple(
        _create_arg_spec(
            batch_dim=batch_dim,
            hidden_dim=2560,
            output_dim=262144,
            reduction=reduction,
            x_dtype=x_dtype,
            w_dtype=jnp.bfloat16,
            project='gemma3-4b',
        )
        for batch_dim, reduction, x_dtype in itertools.product(
            [1024, 2048, 4096, 16384, 32768, 131072],
            ('mean', 'sum'),
            (jnp.bfloat16, jnp.float32),
        )
    )
    + tuple(
        _create_arg_spec(
            batch_dim=batch_dim,
            hidden_dim=3840,
            output_dim=262144,
            reduction=reduction,
            x_dtype=x_dtype,
            w_dtype=jnp.bfloat16,
            project='gemma3-7b',
        )
        for batch_dim, reduction, x_dtype in itertools.product(
            [1024, 2048, 4096, 16384, 32768, 131072],
            ('mean', 'sum'),
            (jnp.bfloat16, jnp.float32),
        )
    )
    + tuple(
        _create_arg_spec(
            batch_dim=batch_dim,
            hidden_dim=4096,
            output_dim=128256,
            reduction=reduction,
            x_dtype=x_dtype,
            w_dtype=jnp.bfloat16,
            project='llama3.1-8b',
        )
        for batch_dim, reduction, x_dtype in itertools.product(
            [1024, 2048, 4096, 16384, 32768, 131072],
            ('mean', 'sum'),
            (jnp.bfloat16, jnp.float32),
        )
    )
    + tuple(
        _create_arg_spec(
            batch_dim=batch_dim,
            hidden_dim=7168,
            output_dim=128256,
            reduction=reduction,
            x_dtype=x_dtype,
            w_dtype=jnp.bfloat16,
            project='deepseek-v3-671b',
        )
        for batch_dim, reduction, x_dtype in itertools.product(
            (4096, 8192, 16384, 32768),
            ('mean', 'sum'),
            (jnp.bfloat16, jnp.float32),
        )
    )
)
