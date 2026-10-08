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
"""DeepSeek-V4 `wo_a` projection benchmark argument specifications.

The kernels gather `cos_sin_cache` rows without bounds checks, so positions are
drawn in `[0, max_position)`. Token counts are multiples of 128, so every token
tile is aligned (see `pallas_mosaic_tpu.PallasTpuOProjection`).
"""

from typing import Final

import jax
import jax.numpy as jnp
from tokamax._src import numerics
from tokamax._src.autotuning import arg_spec
from tokamax._src.ops.experimental.tpu.o_projection import base

ShapeDtype = jax.ShapeDtypeStruct

_HEAD_DIM = 512
_ROTARY_DIM = 64
_MAX_POSITION = 65536


def _make_argspec(
    *,
    name: str,
    num_tokens: int,
    num_groups: int,
    lora_rank: int,
    tags: tuple[arg_spec.Tag, ...] = ("primary", "ci_tests", "forward_only"),
) -> arg_spec.ArgSpec:
  """Makes an argspec for the `wo_a` projection."""
  num_heads = num_groups * base.HEADS_PER_GROUP
  out_features = num_groups * lora_rank
  return arg_spec.ArgSpec(
      args={
          "x": ShapeDtype((num_tokens, num_heads, _HEAD_DIM), jnp.bfloat16),
          "positions": numerics.RangedArrayInitializer(
              (num_tokens,), jnp.int32, 0, _MAX_POSITION
          ),
          "cos_sin_cache": ShapeDtype(
              (_MAX_POSITION, _ROTARY_DIM), jnp.float32
          ),
          "wo_a": ShapeDtype(
              (base.HEADS_PER_GROUP * _HEAD_DIM, out_features),
              jnp.float8_e4m3fn,
          ),
          "wo_a_scale": ShapeDtype((out_features,), jnp.float32),
      },
      project="inference",
      name=name,
      tags=tags,
  )


# DeepSeek-V4: 128 heads of 512 in 16 groups of 8, LoRA rank 1024.
ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _make_argspec(
        name="dsv4_t128_g16_r1024",
        num_tokens=128,
        num_groups=16,
        lora_rank=1024,
    ),
    _make_argspec(
        name="dsv4_t1024_g16_r1024",
        num_tokens=1024,
        num_groups=16,
        lora_rank=1024,
    ),
    _make_argspec(
        name="dsv4_t4096_g16_r1024",
        num_tokens=4096,
        num_groups=16,
        lora_rank=1024,
    ),
)
