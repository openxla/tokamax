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
"""Autotuning argument specifications for mHC-lite ops."""

import math
from typing import Any, Final, cast
import jax
import jax.numpy as jnp
from tokamax._src.autotuning import arg_spec
from tokamax._src.ops.experimental.mhc_lite import common

ArgSpec = arg_spec.ArgSpec

# Canonical (name, tokens, streams, embedding, tags) shapes for mHC-lite.
_SPECS: Final[
    tuple[tuple[str, int, int, int, tuple[arg_spec.Tag, ...]], ...]
] = (
    ("t512_s4_e256", 512, 4, 256, ("primary", "ci_tests")),
    ("t8192_s4_e512", 8192, 4, 512, ("primary",)),
    ("t196608_s4_e512", 196608, 4, 512, ("primary",)),
)


def _s(shape: tuple[int, ...], dtype=jnp.bfloat16) -> jax.Array:
  return cast(jax.Array, jax.ShapeDtypeStruct(shape, dtype))


def _weights_spec(streams: int, embedding: int) -> common.MhcWeights:
  flat = streams * embedding
  num_perms = math.factorial(streams)
  return cast(
      Any,
      common.MhcWeights(
          norm_scale=_s((flat,)),
          w_pre=_s((flat, streams)),
          b_pre=_s((streams,)),
          alpha_pre=_s((1,)),
          w_post=_s((flat, streams)),
          b_post=_s((streams,)),
          alpha_post=_s((1,)),
          w_res=_s((flat, num_perms)),
          b_res=_s((num_perms,)),
          alpha_res=_s((1,)),
      ),
  )


def pre_arg_specs() -> tuple[ArgSpec, ...]:
  """Autotuning specs for `MhcLitePre`."""
  return tuple(
      ArgSpec(
          project="mhc_lite",
          name=name,
          args=dict(
              x=_s((tokens, streams * embedding)),
              weights=_weights_spec(streams, embedding),
              permutations=_s(
                  (math.factorial(streams), streams, streams), jnp.float32
              ),
              rms_epsilon=1e-6,
              pre_mapping_epsilon=1e-6,
          ),
          tags=tags,
      )
      for name, tokens, streams, embedding, tags in _SPECS
  )


def post_arg_specs() -> tuple[ArgSpec, ...]:
  """Autotuning specs for `MhcLitePost`."""
  return tuple(
      ArgSpec(
          project="mhc_lite",
          name=name,
          args=dict(
              layer_output=_s((tokens, embedding)),
              x=_s((tokens, streams * embedding)),
              h_post=_s((tokens, streams), jnp.float32),
              residual=_s((tokens, streams * streams)),
          ),
          tags=tags,
      )
      for name, tokens, streams, embedding, tags in _SPECS
  )


def post_and_pre_arg_specs() -> tuple[ArgSpec, ...]:
  """Autotuning specs for `MhcLitePostAndPre`."""
  return tuple(
      ArgSpec(
          project="mhc_lite",
          name=name,
          args=dict(
              layer_output=_s((tokens, embedding)),
              x=_s((tokens, streams * embedding)),
              h_post=_s((tokens, streams), jnp.float32),
              residual=_s((tokens, streams * streams)),
              next_weights=_weights_spec(streams, embedding),
              permutations=_s(
                  (math.factorial(streams), streams, streams), jnp.float32
              ),
              rms_epsilon=1e-6,
              pre_mapping_epsilon=1e-6,
          ),
          tags=tags,
      )
      for name, tokens, streams, embedding, tags in _SPECS
  )


PRE_ARG_SPECS: Final[tuple[ArgSpec, ...]] = pre_arg_specs()
POST_ARG_SPECS: Final[tuple[ArgSpec, ...]] = post_arg_specs()
POST_AND_PRE_ARG_SPECS: Final[tuple[ArgSpec, ...]] = post_and_pre_arg_specs()
