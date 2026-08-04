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
"""Argument specifications for Ragged Scatter autotuning/benchmarking."""

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
    out_size: int,
    hidden_size: int,
    start: int,
    end: int,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    indices_dtype: jax.typing.DTypeLike = jnp.int32,
    tags: tuple[arg_spec.Tag, ...] = ("primary", "ci_tests"),
) -> arg_spec.ArgSpec:
  """Make argspec for ragged scatter."""
  return arg_spec.ArgSpec(
      args={
          "x": ShapeDtype((out_size, hidden_size), dtype),
          "indices": numerics.RangedArrayInitializer(
              (out_size,), indices_dtype, 0, out_size
          ),
          "start": numerics.RangedArrayInitializer(
              (1,), jnp.int32, start, start + 1
          ),
          "end": numerics.RangedArrayInitializer(
              (1,), jnp.int32, end, end + 1
          ),
      },
      project=project,
      name=name,
      tags=tags,
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _make_argspec(
        name="4096x2048_start13_end500_bf16",
        project="inference",
        out_size=4096,
        hidden_size=2048,
        start=13,
        end=500,
    ),
    _make_argspec(
        name="1024x512_start10_end422_bf16",
        project="inference",
        out_size=1024,
        hidden_size=512,
        start=10,
        end=422,
    ),
)
