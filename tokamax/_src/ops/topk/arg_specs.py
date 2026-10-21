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
"""SparseCore top-k benchmark argument specifications."""

from typing import Final

import jax
import jax.numpy as jnp
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct


def _make_argspec(
    *,
    name: str,
    b: int,
    n: int,
    k: int,
    tags: tuple[arg_spec.Tag, ...] = ("forward_only",),
) -> arg_spec.ArgSpec:
  """Makes an argspec for SparseCore top-k."""
  return arg_spec.ArgSpec(
      args={
          "scores": ShapeDtype((b, n), jnp.float32),
          "k": k,
      },
      project="inference",
      name=name,
      tags=tags,
  )


# Covers single-stage (P=1) and two-stage (P>1) SparseCore top-k regimes across
# decode and prefill batch sizes.
ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _make_argspec(
        name="b16_n10240_k2048",
        b=16,
        n=10240,
        k=2048,
        tags=("primary", "forward_only"),
    ),
    _make_argspec(
        name="b64_n10240_k2048",
        b=64,
        n=10240,
        k=2048,
        tags=("primary", "forward_only"),
    ),
    _make_argspec(
        name="b256_n2304_k1024",
        b=256,
        n=2304,
        k=1024,
        tags=("primary", "forward_only"),
    ),
    _make_argspec(
        name="b1024_n8192_k2048",
        b=1024,
        n=8192,
        k=2048,
        tags=("primary", "forward_only"),
    ),
    _make_argspec(
        name="b16_n133120_k2048",
        b=16,
        n=133120,
        k=2048,
        tags=("primary", "forward_only"),
    ),
)
