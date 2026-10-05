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
"""CSA Gather benchmark argument specifications."""

from typing import Final

import jax
import jax.numpy as jnp
from tokamax._src import numerics
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct


def _make_argspec(
    *,
    name: str,
    num_pages: int,
    page_size: int,
    num_indices: int,
    top_k: int,
    tags: tuple[arg_spec.Tag, ...] = ("primary", "ci_tests", "forward_only"),
) -> arg_spec.ArgSpec:
  """Makes an argspec for the CSA gather."""
  return arg_spec.ArgSpec(
      args={
          "nope_cache": ShapeDtype((num_pages, page_size, 128), jnp.int32),
          "rope_cache": ShapeDtype((num_pages, page_size // 4, 128), jnp.int32),
          "indices": numerics.RangedArrayInitializer(
              (num_indices,), jnp.int32, 0, num_pages * page_size
          ),
          "top_k": top_k,
      },
      project="inference",
      name=name,
      tags=tags,
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _make_argspec(
        name="dsv4_1000x256_n4096_topk512",
        num_pages=1000,
        page_size=256,
        num_indices=4096,
        top_k=512,
    ),
    _make_argspec(
        name="dsv4_1000x256_n131072_topk1024",
        num_pages=1000,
        page_size=256,
        num_indices=128 * 1024,
        top_k=1024,
    ),
    _make_argspec(
        name="dsv4_1000x256_n131072_topk2048",
        num_pages=1000,
        page_size=256,
        num_indices=128 * 1024,
        top_k=2048,
    ),
)
