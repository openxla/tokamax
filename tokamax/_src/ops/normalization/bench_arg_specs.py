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
"""Normalization benchmark argument specifications."""

from typing import Final

import jax
from tokamax._src.autotuning import arg_spec

ShapedArray = jax.ShapeDtypeStruct


def _make_argspec(
    *,
    name,
    project,
    x_shape,
    dtype,
    axis=-1,
    subtract_mean=True,
    has_scale=True,
    has_offset=None,
    param_dtype=None,
    **kwargs,
) -> arg_spec.ArgSpec:
  """Make argspec from shapes and kwargs."""
  if has_offset is None:
    has_offset = subtract_mean
  if param_dtype is None:
    param_dtype = dtype

  return arg_spec.ArgSpec(
      args={
          'x': ShapedArray(x_shape, dtype),
          'scale': (
              ShapedArray((x_shape[axis],), param_dtype) if has_scale else None
          ),
          'offset': (
              ShapedArray((x_shape[axis],), param_dtype) if has_offset else None
          ),
          'axis': axis,
          'subtract_mean': subtract_mean,
          **kwargs,
      },
      project=project,
      name=name,
      tags=('primary',),
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _make_argspec(
        name='alphafold_384res_64chan',
        project='alphafold',
        x_shape=(384, 384, 64),
        dtype='bfloat16',
        param_dtype='float32',
    ),
    _make_argspec(
        name='alphafold_384res_128chan',
        project='alphafold',
        x_shape=(384, 384, 128),
        dtype='bfloat16',
        param_dtype='float32',
    ),
    _make_argspec(
        name='alphafold_768res_128chan',
        project='alphafold',
        x_shape=(768, 768, 128),
        dtype='bfloat16',
        param_dtype='float32',
    ),
    _make_argspec(
        name='alphafold_384res_128chan_axis0',
        project='alphafold',
        x_shape=(128, 384, 384),
        dtype='bfloat16',
        param_dtype='float32',
        axis=0,
    ),
    _make_argspec(
        name='alphafold_768res_128chan_axis0',
        project='alphafold',
        x_shape=(128, 768, 768),
        dtype='bfloat16',
        param_dtype='float32',
        axis=0,
    ),
)
