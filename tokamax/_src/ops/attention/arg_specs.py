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
from tokamax._src import batching
from tokamax._src.autotuning import arg_spec
from tokamax._src.ops.attention import base

BatchedShapeDtype = batching.BatchedShapeDtype
ShapeDtype = jax.ShapeDtypeStruct
Mask = base.Mask


def _alphafold_oss_inference_spec(
    batch_size: int, seq_len: int, head_dim: int, vmap: bool = False
):
  vmap_axes = ((0, 1),) if vmap else ()
  vmap_axes_mask = (None,) if vmap else ()
  num_heads = 4
  x_shape = BatchedShapeDtype(
      (batch_size, seq_len, num_heads, head_dim),
      jnp.bfloat16,
      vmap_axes=vmap_axes,
  )
  return arg_spec.ArgSpec(
      args={
          'q': x_shape,
          'k': x_shape,
          'v': x_shape,
          'bias': BatchedShapeDtype(
              (1, num_heads, seq_len, seq_len),
              jnp.bfloat16,
              vmap_axes=vmap_axes,
          ),
          'mask': Mask(
              BatchedShapeDtype(
                  (batch_size, 1, 1, seq_len), bool, vmap_axes=vmap_axes_mask
              )
          ),
          'precision': (
              jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
              jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
          ),
      },
      project='alphafold_3',
      name=f'batch_size={batch_size}_seq_len={seq_len}_head_dim={head_dim}_vmap={vmap}',
      tags=('forward_only',),
  )


# Shapes for https://github.com/google-deepmind/alphafold3.
def _alphafold_oss_inference_specs() -> tuple[arg_spec.ArgSpec, ...]:
  """Generates the AlphaFold OSS inference argument specifications."""
  specs = []

  seq_lens = {
      32: tuple(range(512, 1536 + 1, 256)),
      128: tuple(range(1792, 5120 + 1, 256)),
  }
  for batch_size, seq_lens in seq_lens.items():
    for seq_len in seq_lens:
      for head_dim in (16, 32):
        for vmap in (True, False):
          if head_dim == 16 and vmap:
            continue
          specs.append(
              _alphafold_oss_inference_spec(
                  batch_size=batch_size,
                  seq_len=seq_len,
                  head_dim=head_dim,
                  vmap=vmap,
              )
          )
  return tuple(specs)


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
    (
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
    + _flash_attention_v3_specs()
    + _alphafold_oss_inference_specs()
)
