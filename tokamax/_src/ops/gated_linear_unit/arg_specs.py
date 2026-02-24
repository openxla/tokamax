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
"""Gated linear unit argument specifications."""

import functools
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
    batch_1: int,
    batch_2: int | None,
    n_in: int,
    n_out: int,
    vmap_size: int | None = None,
    activation=jax.nn.silu,
    dtype=jnp.bfloat16,
):
  """Generates an argument specification for AlphaFold OSS inference."""
  x_shape = (batch_1, batch_2, n_in) if batch_2 else (batch_1, n_in)
  vmap_axes = ((0, vmap_size),) if vmap_size else ()
  vmap_axes_weights = (None,) if vmap_size else ()
  return arg_spec.ArgSpec(
      args={
          'x': BatchedShapeDtype(
              shape=x_shape, dtype=dtype, vmap_axes=vmap_axes
          ),
          'weights': BatchedShapeDtype(
              shape=(n_in, 2, n_out), dtype=dtype, vmap_axes=vmap_axes_weights
          ),
          'activation': activation,
          'precision': jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
      },
      project='alphafold_3',
      name=f'batch_1={batch_1}_batch_2={batch_2}_n_in={n_in}_n_out={n_out}_vmap_size={vmap_size}_activation={activation.__name__}_dtype={dtype.__name__}',
      tags=('forward_only',),
  )


# Shapes for https://github.com/google-deepmind/alphafold3.
def _alphafold_oss_inference_specs() -> tuple[arg_spec.ArgSpec, ...]:
  """Generates the AlphaFold OSS inference argument specifications."""
  spec_silu = _alphafold_oss_inference_spec
  spec_sigmoid = functools.partial(
      _alphafold_oss_inference_spec, activation=jax.nn.sigmoid
  )
  s = []
  for batch in tuple(range(512, 5120 + 1, 256)):
    s.append(spec_sigmoid(batch, batch, 128, 256, vmap_size=1))
    s.append(spec_silu(batch, batch, 128, 512, vmap_size=1))
    s.append(spec_sigmoid(batch, batch, n_in=128, n_out=256))
    s.append(spec_silu(batch, batch, 128, 512))
    s.append(spec_silu(batch, batch, 128, 256, dtype=jnp.float32))
    s.append(spec_sigmoid(batch, batch, 64, 128))
    s.append(spec_silu(batch, batch, 64, 128))
    s.append(spec_silu(batch, 32, 128, 256, dtype=jnp.float32, vmap_size=5))
    s.append(spec_silu(batch, 32, 128, 256, dtype=jnp.float32))
    s.append(spec_silu(batch, None, 384, 1536, vmap_size=5))
    s.append(spec_silu(batch, None, 384, 1536))
    s.append(spec_silu(batch, None, 384, 768, dtype=jnp.float32, vmap_size=5))
    s.append(spec_silu(batch, None, 384, 768, dtype=jnp.float32))
    s.append(spec_silu(batch, None, 768, 1536, dtype=jnp.float32, vmap_size=5))
    s.append(spec_silu(batch, None, 768, 1536, dtype=jnp.float32))

  return tuple(s)


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _alphafold_oss_inference_specs()
)
