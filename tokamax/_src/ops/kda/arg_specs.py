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
      "beta": numerics.FloatRangedArrayInitializer(
          (num_heads, batch_size, seq_len), dtype, minval=0.0, maxval=1.0
      ),
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
      "gate": numerics.FloatRangedArrayInitializer(
          (num_heads, batch_size, seq_len, 1), dtype, minval=-5.0, maxval=0.0
      ),
      "beta": numerics.FloatRangedArrayInitializer(
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


# Per-device KDA inference shapes for Kimi K3 (moonshotai/Kimi-K3, 2.8T total /
# 104B active, 69 KDA layers, 96 global heads, head_dim=128, causal conv1d W=4):
# - TP=32 (InferenceX megakernel on gf=2x2x4, 32 TensorCores): 3 heads/core
# - TP=16 (per 2-core chip on gf=2x2x4): 6 heads/chip
# - TP=8 (serving on gf=4x4x4 / gf=4x4x8): 12 heads/device
def _kimi_k3_2p8t_inference_spec(
    *,
    name: str,
    num_heads: int,
    seq_len: int,
    batch_size: int = 1,
    num_segments: int | None = None,
) -> arg_spec.ArgSpec:
  """Builds a Kimi K3 (2.8T) KDA inference workload."""
  head_dim = 128
  conv_width = 4
  dtype = jnp.bfloat16
  qkv_shape = (num_heads, batch_size, seq_len, head_dim)
  num_states = num_segments if num_segments is not None else 1

  args: dict[str, Any] = {
      "query": ShapeDtype(qkv_shape, dtype),
      "key": ShapeDtype(qkv_shape, dtype),
      "value": ShapeDtype(qkv_shape, dtype),
      "gate": ShapeDtype(qkv_shape, dtype),
      "beta": numerics.FloatRangedArrayInitializer(
          (num_heads, batch_size, seq_len), dtype, minval=0.0, maxval=1.0
      ),
      "a_log": ShapeDtype((num_heads,), jnp.float32),
      "delta_time_bias": ShapeDtype((num_heads * head_dim,), jnp.float32),
      "initial_state": ShapeDtype(
          (batch_size, num_states, num_heads, head_dim, head_dim), jnp.float32
      ),
      "output_final_state": True,
      "use_qk_l2norm": True,
      "use_gate_in_kernel": True,
      "lower_bound": -5.0,
      "conv_weight_q": ShapeDtype(
          (num_heads, conv_width, head_dim), jnp.float32
      ),
      "conv_weight_k": ShapeDtype(
          (num_heads, conv_width, head_dim), jnp.float32
      ),
      "conv_weight_v": ShapeDtype(
          (num_heads, conv_width, head_dim), jnp.float32
      ),
      "use_conv1d_in_kernel": True,
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
      project="kimi_k3_2p8t",
      name=name,
      tags=("primary", "forward_only"),
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
    _kimi_k3_2p8t_inference_spec(
        name="tp32_prefill_b1_t128_bf16",
        num_heads=3,
        seq_len=128,
    ),
    _kimi_k3_2p8t_inference_spec(
        name="tp32_prefill_b1_t512_bf16",
        num_heads=3,
        seq_len=512,
    ),
    _kimi_k3_2p8t_inference_spec(
        name="tp32_prefill_b1_t1024_bf16",
        num_heads=3,
        seq_len=1024,
    ),
    _kimi_k3_2p8t_inference_spec(
        name="tp32_verify_b1_t64_n8_bf16",
        num_heads=3,
        seq_len=64,
        num_segments=8,
    ),
    _kimi_k3_2p8t_inference_spec(
        name="tp16_prefill_b1_t1024_bf16",
        num_heads=6,
        seq_len=1024,
    ),
    _kimi_k3_2p8t_inference_spec(
        name="tp8_prefill_b1_t2048_bf16",
        num_heads=12,
        seq_len=2048,
    ),
    # Kimi K3 InferenceX decode shapes (DCP=32 -> 3 heads/core, DP=2 x DCP=16
    # -> 6 heads/core; plain 1-token decode up to 64 seqs/worker and DSpark
    # speculative decode with 1+7=8 tokens at 1 seq or 1+3=4 tokens at 16 seqs):
    _kimi_k3_2p8t_inference_spec(
        name="tp32_decode_b1_t1_n1_bf16",
        num_heads=3,
        seq_len=1,
        num_segments=1,
    ),
    _kimi_k3_2p8t_inference_spec(
        name="tp32_dspark_decode_b1_t8_n1_bf16",
        num_heads=3,
        seq_len=8,
        num_segments=1,
    ),
    _kimi_k3_2p8t_inference_spec(
        name="tp32_dspark_decode_b1_t64_n16_bf16",
        num_heads=3,
        seq_len=64,
        num_segments=16,
    ),
    _kimi_k3_2p8t_inference_spec(
        name="tp32_decode_b1_t64_n64_bf16",
        num_heads=3,
        seq_len=64,
        num_segments=64,
    ),
    _kimi_k3_2p8t_inference_spec(
        name="tp16_dp2_dspark_decode_b1_t64_n16_bf16",
        num_heads=6,
        seq_len=64,
        num_segments=16,
    ),
    _kimi_k3_2p8t_inference_spec(
        name="tp16_dp2_decode_b1_t64_n64_bf16",
        num_heads=6,
        seq_len=64,
        num_segments=64,
    ),
)
