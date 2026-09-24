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
"""Causal Conv1D Gated Delta Rule benchmark argument specifications.

Covers representative model architectures and execution scenarios:
- Qwen 3.5 (397B-A17B, 35B-A3B) and Qwen 3.8-Max configs across TP8,
  ATTN_DP4+EP8, and ATTN_DP8+EP8 sharding strategies.
- Prefill, decode, and mixed continuous batching scenarios.
- Long-context continuation prefill with prior context state and concurrent
  multi-sequence decode batches.
"""

from collections.abc import Sequence
from typing import Any, Final

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.autotuning import arg_spec

ShapeDtype = jax.ShapeDtypeStruct


class _HashableNPArray(np.ndarray):
  """Numpy array that can be retained as a concrete argument in an ArgSpec."""

  def __new__(cls, input_array: Any) -> "_HashableNPArray":
    return np.asarray(input_array).view(cls)

  def __hash__(self) -> int:
    return hash((self.tobytes(), self.shape, self.dtype))

  def __eq__(self, other: Any) -> bool:
    return (
        isinstance(other, np.ndarray)
        and self.shape == other.shape
        and self.dtype == other.dtype
        and bool(np.array_equal(self, other))
    )


def make_gdn_spec(
    *,
    project: str,
    name: str,
    lengths: Sequence[int],
    n_kq: int,
    n_v: int,
    d_k: int = 128,
    d_v: int = 128,
    kernel_size: int = 4,
    context_lens: Sequence[int] | int = 0,
    max_reqs: int | None = None,
    num_decode_seqs: int | None = None,
    distribution: Sequence[int] | None = None,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    state_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    tags: tuple[arg_spec.Tag, ...] = ("forward_only",),
) -> arg_spec.ArgSpec:
  """Constructs an ArgSpec for CausalConv1dGatedDeltaRule."""
  num_seqs = len(lengths)
  if max_reqs is None:
    max_reqs = num_seqs
  if max_reqs < num_seqs:
    raise ValueError(
        f"max_reqs ({max_reqs}) must be >= len(lengths) ({num_seqs})"
    )

  num_tokens = sum(lengths)
  dim_size = 2 * n_kq * d_k + n_v * d_v
  num_blocks = max_reqs + 1  # Slot 0 is reserved for null/padding block.

  # Build cumulative query start locations of shape (max_reqs + 1,).
  q_loc = np.cumsum([0] + list(lengths), dtype=np.int32)
  if len(q_loc) < max_reqs + 1:
    q_loc = np.pad(
        q_loc,
        (0, max_reqs + 1 - len(q_loc)),
        constant_values=q_loc[-1],
    )

  # Build total sequence lengths (query_len + prior context_len) of shape
  # (max_reqs,).
  if isinstance(context_lens, int):
    ctx_list = [context_lens] * num_seqs
  else:
    if len(context_lens) != num_seqs:
      raise ValueError(
          f"len(context_lens) ({len(context_lens)}) must match len(lengths)"
          f" ({num_seqs})"
      )
    ctx_list = list(context_lens)

  s_lens = np.array([l + c for l, c in zip(lengths, ctx_list)], dtype=np.int32)
  if len(s_lens) < max_reqs:
    s_lens = np.pad(s_lens, (0, max_reqs - len(s_lens)), constant_values=0)

  # Build sequence distribution [decode_end, prefill_end, mixed_end].
  if distribution is not None:
    dist_array = np.array(distribution, dtype=np.int32)
  else:
    if num_decode_seqs is None:
      num_decode_seqs = 0
      for l in lengths:
        if l == 1:
          num_decode_seqs += 1
        else:
          break
    dist_array = np.array([num_decode_seqs, num_seqs, num_seqs], dtype=np.int32)

  state_indices = np.arange(1, max_reqs + 1, dtype=np.int32)

  args = {
      "qkv": ShapeDtype((num_tokens, dim_size), dtype),
      "b": ShapeDtype((num_tokens, n_v), dtype),
      "a": ShapeDtype((num_tokens, n_v), dtype),
      "conv_state": ShapeDtype((num_blocks, kernel_size - 1, dim_size), dtype),
      "recurrent_state": ShapeDtype((num_blocks, n_v, d_k, d_v), state_dtype),
      "conv_weight": ShapeDtype((dim_size, 1, kernel_size), dtype),
      "conv_bias": ShapeDtype((dim_size,), dtype),
      "a_log": ShapeDtype((n_v,), jnp.float32),
      "dt_bias": ShapeDtype((n_v,), dtype),
      "query_start_loc": _HashableNPArray(q_loc),
      "state_indices": _HashableNPArray(state_indices),
      "distribution": _HashableNPArray(dist_array),
      "seq_lens": _HashableNPArray(s_lens),
      "n_kq": n_kq,
      "n_v": n_v,
      "d_k": d_k,
      "d_v": d_v,
      "kernel_size": kernel_size,
  }

  return arg_spec.ArgSpec(
      args=args,
      project=project,
      name=name,
      tags=tags,
  )


_make_gdn_spec = make_gdn_spec

ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    # Qwen3.5-397B TP8
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_prefill_1024",
        lengths=[1024],
        n_kq=2,
        n_v=8,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_prefill_2048",
        lengths=[2048],
        n_kq=2,
        n_v=8,
        tags=("primary", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_prefill_4096",
        lengths=[4096],
        n_kq=2,
        n_v=8,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_prefill_8192",
        lengths=[8192],
        n_kq=2,
        n_v=8,
        tags=("primary", "ci_tests", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_decode_64",
        lengths=[1] * 64,
        n_kq=2,
        n_v=8,
        tags=("primary", "ci_tests", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_decode_128",
        lengths=[1] * 128,
        n_kq=2,
        n_v=8,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_decode_256",
        lengths=[1] * 256,
        n_kq=2,
        n_v=8,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_decode_512",
        lengths=[1] * 512,
        n_kq=2,
        n_v=8,
    ),
    # Decode with non-zero prior context (has_initial_state=True)
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_decode_cached_4_ctx8192",
        lengths=[1] * 4,
        context_lens=8192,
        n_kq=2,
        n_v=8,
        tags=("primary", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_decode_cached_8_ctx8192",
        lengths=[1] * 8,
        context_lens=8192,
        n_kq=2,
        n_v=8,
        tags=("primary", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_decode_cached_16_ctx8192",
        lengths=[1] * 16,
        context_lens=8192,
        n_kq=2,
        n_v=8,
        tags=("primary", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_decode_cached_32_ctx8192",
        lengths=[1] * 32,
        context_lens=8192,
        n_kq=2,
        n_v=8,
        tags=("primary", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_decode_cached_64_ctx1024",
        lengths=[1] * 64,
        context_lens=1024,
        n_kq=2,
        n_v=8,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_decode_cached_64_ctx8192",
        lengths=[1] * 64,
        context_lens=8192,
        n_kq=2,
        n_v=8,
        tags=("primary", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_decode_cached_128_ctx8192",
        lengths=[1] * 128,
        context_lens=8192,
        n_kq=2,
        n_v=8,
        tags=("primary", "forward_only"),
    ),
    # Mixed batch: 63 decode requests + 1 prefill request (total 8192 tokens)
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_mixed_64_t8192",
        lengths=[1] * 63 + [8129],
        n_kq=2,
        n_v=8,
        tags=("primary", "forward_only"),
    ),
    # Unit test shapes from base_test.py / pallas_mosaic_tpu_test.py
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_mixed_3x512",
        lengths=[256, 128, 128],
        n_kq=2,
        n_v=8,
        tags=("primary", "ci_tests", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_mixed_prefill_decode_11",
        lengths=[1] * 8 + [128, 128, 256],
        n_kq=2,
        n_v=8,
        tags=("primary", "ci_tests", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_padded_mixed_16",
        lengths=[128, 64, 32, 16, 8],
        max_reqs=16,
        n_kq=2,
        n_v=8,
        tags=("primary", "ci_tests", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_padded_decode_64_max512",
        lengths=[1] * 64,
        max_reqs=512,
        n_kq=2,
        n_v=8,
        tags=("primary", "ci_tests", "forward_only"),
    ),
    # Qwen3.5-397B ATTN_DP4, EP8
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp4_ep8_prefill_1024",
        lengths=[1024],
        n_kq=8,
        n_v=32,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp4_ep8_prefill_2048",
        lengths=[2048],
        n_kq=8,
        n_v=32,
        tags=("primary", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp4_ep8_decode_16_ctx1024",
        lengths=[1] * 16,
        context_lens=1024,
        n_kq=8,
        n_v=32,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp4_ep8_decode_32_ctx1024",
        lengths=[1] * 32,
        context_lens=1024,
        n_kq=8,
        n_v=32,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp4_ep8_decode_64_ctx8192",
        lengths=[1] * 64,
        context_lens=8192,
        n_kq=8,
        n_v=32,
        tags=("primary", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp4_ep8_decode_128_ctx1024",
        lengths=[1] * 128,
        context_lens=1024,
        n_kq=8,
        n_v=32,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp4_ep8_mixed_16_t1024",
        lengths=[1] * 15 + [1009],
        context_lens=[1024] * 15 + [0],
        n_kq=8,
        n_v=32,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp4_ep8_mixed_32_t2048",
        lengths=[1] * 31 + [2017],
        context_lens=[8192] * 31 + [0],
        n_kq=8,
        n_v=32,
    ),
    # Qwen3.5-397B ATTN_DP8, EP8
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp8_ep8_decode_8",
        lengths=[1] * 8,
        n_kq=16,
        n_v=64,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp8_ep8_decode_64",
        lengths=[1] * 64,
        n_kq=16,
        n_v=64,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp8_ep8_mixed_8_t1024",
        lengths=[1] * 7 + [1017],
        n_kq=16,
        n_v=64,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp8_ep8_mixed_64_t1024",
        lengths=[1] * 63 + [961],
        n_kq=16,
        n_v=64,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp8_ep8_prefill_512",
        lengths=[512],
        n_kq=16,
        n_v=64,
        tags=("primary", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp8_ep8_prefill_1024",
        lengths=[1024],
        n_kq=16,
        n_v=64,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp8_ep8_prefill_2048",
        lengths=[2048],
        n_kq=16,
        n_v=64,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp8_ep8_prefill_4096",
        lengths=[4096],
        n_kq=16,
        n_v=64,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp8_ep8_decode_16_ctx8192",
        lengths=[1] * 16,
        context_lens=8192,
        n_kq=16,
        n_v=64,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp8_ep8_decode_32_ctx8192",
        lengths=[1] * 32,
        context_lens=8192,
        n_kq=16,
        n_v=64,
        tags=("primary", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp8_ep8_decode_128_ctx1024",
        lengths=[1] * 128,
        context_lens=1024,
        n_kq=16,
        n_v=64,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp8_ep8_mixed_32_t512",
        lengths=[1] * 31 + [481],
        context_lens=[8192] * 31 + [0],
        n_kq=16,
        n_v=64,
        tags=("primary", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp8_ep8_mixed_32_t2048",
        lengths=[1] * 31 + [2017],
        context_lens=[8192] * 31 + [0],
        n_kq=16,
        n_v=64,
        tags=("primary", "forward_only"),
    ),
    # Long-context continuation and multi-sequence decode
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_delta_prefill_1536_ctx65536",
        lengths=[1536],
        context_lens=65536,
        n_kq=2,
        n_v=8,
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_delta_prefill_2048_ctx131072",
        lengths=[2048],
        context_lens=131072,
        n_kq=2,
        n_v=8,
        tags=("primary", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="dp8_delta_prefill_2048_ctx65536",
        lengths=[2048],
        context_lens=65536,
        n_kq=16,
        n_v=64,
    ),
    # Concurrent decode batches
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_multi_seq_decode_8_ctx32768",
        lengths=[1] * 8,
        context_lens=32768,
        n_kq=2,
        n_v=8,
    ),
    # Mixed decode + prefill batch
    _make_gdn_spec(
        project="qwen3_5_397b",
        name="tp8_mixed_decode_8_prefill_2048",
        lengths=[1] * 7 + [2048],
        context_lens=[32768] * 7 + [65536],
        n_kq=2,
        n_v=8,
    ),
    # Qwen3.5-35B and Qwen3.8-Max
    _make_gdn_spec(
        project="qwen3_5_35b",
        name="tp8_prefill_2048",
        lengths=[2048],
        n_kq=2,
        n_v=4,
        tags=("primary", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_5_35b",
        name="tp8_decode_64_ctx1024",
        lengths=[1] * 64,
        context_lens=1024,
        n_kq=2,
        n_v=4,
        tags=("primary", "forward_only"),
    ),
    _make_gdn_spec(
        project="qwen3_8_max",
        name="tp8_prefill_2048",
        lengths=[2048],
        n_kq=2,
        n_v=16,
    ),
    _make_gdn_spec(
        project="qwen3_8_max",
        name="tp8_decode_64_ctx1024",
        lengths=[1] * 64,
        context_lens=1024,
        n_kq=2,
        n_v=16,
    ),
)
