# Copyright 2026 Google LLC
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
"""Ragged gather reduce benchmark argument specifications."""

from typing import Final

import jax
import jax.numpy as jnp
from tokamax._src import numerics
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct


def _make_argspec(
    *,
    name: str,
    project: str,
    input_size: int,
    hidden_size: int,
    reduce_group_size: int,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    indices_dtype: jax.typing.DTypeLike = jnp.int32,
    tags: tuple[arg_spec.Tag, ...] = ("primary", "ci_tests"),
) -> arg_spec.ArgSpec:
  """Make argspec for ragged gather reduce."""
  return arg_spec.ArgSpec(
      args={
          "x": ShapeDtype((input_size, hidden_size), dtype),
          "indices": numerics.RangedArrayInitializer(
              (input_size,), indices_dtype, 0, input_size
          ),
          "topk_weights": ShapeDtype((input_size,), dtype),
          "valid_rows_mask": ShapeDtype((input_size,), jnp.bool_),
          "reduce_group_size": reduce_group_size,
      },
      project=project,
      name=name,
      tags=tags,
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _make_argspec(
        name="8192x4096_group8_bf16",
        project="inference",
        input_size=8192,
        hidden_size=4096,
        reduce_group_size=8,
    ),
    _make_argspec(
        name="1024x512_group4_bf16",
        project="inference",
        input_size=1024,
        hidden_size=512,
        reduce_group_size=4,
    ),
)
