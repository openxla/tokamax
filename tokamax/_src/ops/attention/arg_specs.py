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


def _flash_attention_v3_specs() -> tuple[arg_spec.ArgSpec, ...]:
  """Generates the Flash Attention v3 benchmark specifications.

  Taken from Section 4.1 of https://arxiv.org/abs/2407.08608.

  Returns:
    A tuple of Flash Attention v3 argument specifications.
  """
  num_tokens = 16384
  hidden_dim = 2048
  specs = []
  for seq_len in (512, 1024, 2048, 4096, 8192, 16384):
    for head_dim in (64, 128, 256):
      for causal in (True, False):
        for dtype in ('bfloat16', 'float16'):
          batch_size = num_tokens // seq_len
          num_heads = hidden_dim // head_dim
          shape = ShapeDtype(
              shape=(batch_size, seq_len, num_heads, head_dim), dtype=dtype
          )
          spec = arg_spec.ArgSpec(
              args={
                  'q': shape,
                  'k': shape,
                  'v': shape,
                  'is_causal': causal,
              },
              project='flash_attention_v3',
              name=f'seq_len={seq_len}_head_dim={head_dim}_causal={causal}_dtype={dtype}',
          )
          specs.append(spec)
  return tuple(specs)


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
) + _flash_attention_v3_specs()
