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
"""DeepSeek-V4 RoPE benchmark argument specifications.

The kernels gather `cos_sin_cache` rows without bounds checks, so positions are
drawn in `[0, max_position)`.
"""

from typing import Final

import jax
import jax.numpy as jnp
from tokamax._src import numerics
from tokamax._src.autotuning import arg_spec
from tokamax._src.ops.experimental.tpu.rope import base

ShapeDtype = jax.ShapeDtypeStruct

_ROTARY_DIM = 64
_MAX_POSITION = 65536


def _make_argspec(
    *,
    name: str,
    x_shape: tuple[int, ...],
    mode: base.Mode,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    tags: tuple[arg_spec.Tag, ...] = ("primary", "ci_tests", "forward_only"),
) -> arg_spec.ArgSpec:
  """Makes an argspec for the DeepSeek-V4 RoPE."""
  return arg_spec.ArgSpec(
      args={
          "x": ShapeDtype(x_shape, dtype),
          "positions": numerics.RangedArrayInitializer(
              (x_shape[0],), jnp.int32, 0, _MAX_POSITION
          ),
          "cos_sin_cache": ShapeDtype(
              (_MAX_POSITION, _ROTARY_DIM), jnp.float32
          ),
          "mode": mode,
      },
      project="inference",
      name=name,
      tags=tags,
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    # The query: per-head RMSNorm and RoPE over 128 heads of 512.
    _make_argspec(
        name="dsv4_qnorm_rope_q_t128_h128_d512",
        x_shape=(128, 128, 512),
        mode="qnorm_rope",
    ),
    _make_argspec(
        name="dsv4_qnorm_rope_q_t4096_h128_d512",
        x_shape=(4096, 128, 512),
        mode="qnorm_rope",
    ),
    # The single-head compressed KV.
    _make_argspec(
        name="dsv4_rope_kv_t4096_d512",
        x_shape=(4096, 512),
        mode="rope",
    ),
    # The indexer query: RoPE and fp8 quantization over 64 heads of 128.
    _make_argspec(
        name="dsv4_rope_quant_indexer_q_t4096_h64_d128",
        x_shape=(4096, 64, 128),
        mode="rope_quant",
    ),
)
