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
"""Attention argument specifications."""

from typing import Final

import jax
import jax.numpy as jnp
from tokamax._src.autotuning import arg_spec
from tokamax._src.ops.attention import base


ShapeDtype = jax.ShapeDtypeStruct
Mask = base.Mask


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    arg_spec.ArgSpec(
        args={
            'q': ShapeDtype((32, 4096, 32, 128), jnp.bfloat16),
            'k': ShapeDtype((32, 4096, 8, 128), jnp.bfloat16),
            'v': ShapeDtype((32, 4096, 8, 128), jnp.bfloat16),
            'is_causal': True,
        },
        project='mixtral',
        name='8x7b_bf16',
        tags=('primary',),
    ),
    arg_spec.ArgSpec(
        args={
            'q': ShapeDtype((512, 1024, 16, 192), jnp.bfloat16),
            'k': ShapeDtype((512, 1024, 16, 192), jnp.bfloat16),
            'v': ShapeDtype((512, 1024, 16, 128), jnp.bfloat16),
            'is_causal': True,
        },
        project='deepseek2',
        name='16b_bf16',
        tags=('primary',),
    ),
    arg_spec.ArgSpec(
        args={
            'q': ShapeDtype((384, 384, 4, 32), jnp.bfloat16),
            'k': ShapeDtype((384, 384, 4, 32), jnp.bfloat16),
            'v': ShapeDtype((384, 384, 4, 32), jnp.bfloat16),
            'bias': ShapeDtype((1, 4, 384, 384), jnp.bfloat16),
            'mask': Mask(ShapeDtype((384, 1, 1, 384), bool)),
        },
        project='alphafold',
    ),
    arg_spec.ArgSpec(
        args={
            'q': ShapeDtype((384, 384, 4, 64), jnp.bfloat16),
            'k': ShapeDtype((384, 384, 4, 64), jnp.bfloat16),
            'v': ShapeDtype((384, 384, 4, 64), jnp.bfloat16),
            'bias': ShapeDtype((1, 4, 384, 384), jnp.bfloat16),
            'mask': Mask(ShapeDtype((384, 1, 1, 384), bool)),
        },
        project='alphafold',
    ),
    arg_spec.ArgSpec(
        args={
            'q': ShapeDtype((768, 768, 4, 64), jnp.bfloat16),
            'k': ShapeDtype((768, 768, 4, 64), jnp.bfloat16),
            'v': ShapeDtype((768, 768, 4, 64), jnp.bfloat16),
            'bias': ShapeDtype((1, 4, 768, 768), jnp.bfloat16),
            'mask': Mask(ShapeDtype((768, 1, 1, 768), bool)),
        },
        project='alphafold',
    ),
)
