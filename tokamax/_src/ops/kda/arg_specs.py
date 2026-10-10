# Copyright 2026 Ant Group. All Rights Reserved.
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
"""Kimi Delta Attention argument specifications."""

from typing import Any, Final
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import numerics
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct


class _HashableNPArray(np.ndarray):
  """Numpy array that can be retained as a concrete argument in an ArgSpec."""

  def __new__(cls, input_array):
    return np.asarray(input_array).view(cls)

  def __hash__(self):
    return hash((self.tobytes(), self.shape, self.dtype))


class _FloatRangedArrayInitializer(
    jax.ShapeDtypeStruct, numerics.ArrayInitializer
):
  """Abstract float array initialized uniformly in `[minval, maxval]`.

  `numerics.RangedArrayInitializer` only supports integer dtypes.
  """

  def __init__(self, shape, dtype, minval: float, maxval: float):
    jax.ShapeDtypeStruct.__init__(self, shape, dtype)
    self.minval = minval
    self.maxval = maxval

  def __call__(self, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(self.minval, self.maxval, self.shape).astype(self.dtype)


def _make_segment_ids(
    *, batch_size: int, seq_len: int, num_segments: int
) -> _HashableNPArray:
  """Builds balanced, 1-indexed packed-sequence segment IDs."""
  segment_len, remainder = divmod(seq_len, num_segments)
  segment_lens = np.full(num_segments, segment_len, dtype=np.int32)
  segment_lens[:remainder] += 1
  segment_ids = np.repeat(
      np.arange(1, num_segments + 1, dtype=np.int32), segment_lens
  )
  return _HashableNPArray(np.broadcast_to(segment_ids, (batch_size, seq_len)))


# Shapes from https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json.
def _kimi_linear_48b_a3b_spec(
    *,
    name: str,
    seq_len: int,
    num_segments: int | None = None,
) -> arg_spec.ArgSpec:
  """Builds a KDA training workload for Kimi Linear 48B-A3B."""
  batch_size = 1
  num_heads = 32
  head_dim = 128
  dtype = jnp.bfloat16
  qkv_shape = (num_heads, batch_size, seq_len, head_dim)

  args: dict[str, Any] = {
      "query": ShapeDtype(qkv_shape, dtype),
      "key": ShapeDtype(qkv_shape, dtype),
      "value": ShapeDtype(qkv_shape, dtype),
      "gate": ShapeDtype(qkv_shape, dtype),
      "beta": ShapeDtype((num_heads, batch_size, seq_len), dtype),
      "a_log": ShapeDtype((num_heads,), jnp.float32),
      "delta_time_bias": ShapeDtype((num_heads * head_dim,), jnp.float32),
      "output_final_state": False,
      "use_qk_l2norm": True,
      "use_gate_in_kernel": True,
      "lower_bound": -5.0,
  }
  if num_segments is not None:
    args["segment_ids"] = _make_segment_ids(
        batch_size=batch_size,
        seq_len=seq_len,
        num_segments=num_segments,
    )
    args["max_num_segments"] = num_segments

  return arg_spec.ArgSpec(
      args=args,
      project="kimi_linear_48b_a3b",
      name=name,
      tags=("primary",),
  )


# Per-chip Gated DeltaNet shapes for Qwen3.5-397B-A17B training: 32 value
# heads with 128-dimensional key and value states.
def _qwen3_5_397b_a17b_spec(*, name: str, seq_len: int) -> arg_spec.ArgSpec:
  """Builds a scalar-gate (Gated DeltaNet) training workload."""
  batch_size = 1
  num_heads = 32
  head_dim = 128
  dtype = jnp.bfloat16
  qkv_shape = (num_heads, batch_size, seq_len, head_dim)

  args: dict[str, Any] = {
      "query": ShapeDtype(qkv_shape, dtype),
      "key": ShapeDtype(qkv_shape, dtype),
      "value": ShapeDtype(qkv_shape, dtype),
      # One pre-activated log-space decay per head and token.
      "gate": _FloatRangedArrayInitializer(
          (num_heads, batch_size, seq_len, 1), dtype, minval=-5.0, maxval=0.0
      ),
      "beta": _FloatRangedArrayInitializer(
          (num_heads, batch_size, seq_len), dtype, minval=0.0, maxval=1.0
      ),
      "output_final_state": False,
      "use_qk_l2norm": True,
      "use_gate_in_kernel": False,
      "per_channel_gate": False,
  }
  return arg_spec.ArgSpec(
      args=args,
      project="qwen3_5_397b_a17b",
      name=name,
      tags=("primary",),
  )


# Kimi Linear uses 32 KDA heads with 128-dimensional key and value states.
ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _kimi_linear_48b_a3b_spec(
        name="fixed_training_b1_t8192_bf16",
        seq_len=8192,
    ),
    _kimi_linear_48b_a3b_spec(
        name="packed_training_b1_t8192_n25_bf16",
        seq_len=8192,
        num_segments=25,
    ),
    _qwen3_5_397b_a17b_spec(
        name="gdn_fixed_training_b1_t8192_bf16",
        seq_len=8192,
    ),
)
