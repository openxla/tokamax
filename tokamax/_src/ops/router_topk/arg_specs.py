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
"""MoE router top-k benchmark argument specifications."""

from typing import Final

import jax
import jax.numpy as jnp
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct


def _make_argspec(
    *,
    name: str,
    num_tokens: int,
    num_experts: int,
    k: int,
    tags: tuple[arg_spec.Tag, ...] = ("forward_only",),
) -> arg_spec.ArgSpec:
  """Makes an argspec for the MoE router top-k."""
  return arg_spec.ArgSpec(
      args={
          "scores": ShapeDtype((num_tokens, num_experts), jnp.float32),
          "k": k,
      },
      project="inference",
      name=name,
      tags=tags,
  )


# Decode (64 tokens) and prefill (8192 tokens) batches for a 512-expert top-10
# router (the kernel's upstream target) and a 256-expert top-8 router
# (DeepSeek-V3).
ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _make_argspec(
        name="e512_k10_t64",
        num_tokens=64,
        num_experts=512,
        k=10,
        tags=("primary", "ci_tests", "forward_only"),
    ),
    _make_argspec(
        name="e512_k10_t8192",
        num_tokens=8192,
        num_experts=512,
        k=10,
        tags=("primary", "ci_tests", "forward_only"),
    ),
    _make_argspec(
        name="e256_k8_t64",
        num_tokens=64,
        num_experts=256,
        k=8,
        tags=("primary", "ci_tests", "forward_only"),
    ),
    _make_argspec(
        name="e256_k8_t8192",
        num_tokens=8192,
        num_experts=256,
        k=8,
        tags=("primary", "ci_tests", "forward_only"),
    ),
)
