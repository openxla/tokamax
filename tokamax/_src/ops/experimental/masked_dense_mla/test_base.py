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
"""Shared correctness tests for MaskedDenseMLA implementations."""

import math

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.masked_dense_mla import reference

ATOL = RTOL = 2e-2
ROPE_DIM = 64
HEAD_DIM = reference.NOPE_DIM + ROPE_DIM


def make_inputs(
    *,
    batch_size: int = 4,
    num_heads: int = 16,
    page_size: int = 32,
    pages_per_seq: int = 8,
    topk: int = 64,
    pad_tokens: bool = False,
    seed: int = 0,
) -> tuple[tuple[jax.Array, ...], int]:
  """Generates deterministic MaskedDenseMLA inputs and valid token count."""
  rng = np.random.default_rng(seed)

  num_decode_seqs = batch_size // 2
  prefill_lens = rng.integers(8, 24, size=(batch_size - num_decode_seqs,))
  q_lens = np.concatenate(
      [np.ones((num_decode_seqs,), dtype=np.int32), prefill_lens]
  )
  cu_q_lens = jnp.concatenate([
      jnp.array([0], dtype=jnp.int32),
      jnp.cumulative_sum(jnp.asarray(q_lens, dtype=jnp.int32)),
  ])
  kv_lens_np = q_lens + rng.integers(20, 120, size=(batch_size,))
  kv_lens = jnp.asarray(kv_lens_np, dtype=jnp.int32)

  actual_tokens = int(np.sum(q_lens))
  total_tokens = (
      ((actual_tokens + 16 + 31) // 32) * 32 if pad_tokens else actual_tokens
  )

  q_np = rng.standard_normal((total_tokens, num_heads, HEAD_DIM)).astype(
      np.float32
  )
  q = jnp.asarray(q_np, dtype=jnp.bfloat16)

  max_kv_len = pages_per_seq * page_size
  analytic_mode = max_kv_len <= topk
  topk_rows = np.full((total_tokens, topk), -1, dtype=np.int32)
  tok_idx = 0
  for i in range(batch_size):
    kv_len_i = int(kv_lens_np[i])
    q_len_i = int(q_lens[i])
    base_pos = kv_len_i - q_len_i
    for local_q in range(q_len_i):
      q_pos = base_pos + local_q
      if analytic_mode:
        topk_rows[tok_idx, : q_pos + 1] = np.arange(q_pos + 1, dtype=np.int32)
      else:
        n_sel = min(max(1, (q_pos + 1) // 2), topk)
        sel = rng.choice(q_pos + 1, size=n_sel, replace=False)
        topk_rows[tok_idx, :n_sel] = np.sort(sel)
      tok_idx += 1
  topk_indices = jnp.asarray(topk_rows, dtype=jnp.int32)

  total_pages = batch_size * pages_per_seq
  page_indices = jnp.arange(total_pages, dtype=jnp.int32)

  nope_fp8 = jnp.asarray(
      rng.standard_normal((total_pages, page_size, reference.NOPE_DIM)).astype(
          np.float32
      )
  ).astype(jnp.float8_e4m3fn)
  rope_fp8 = jnp.asarray(
      rng.standard_normal((total_pages, page_size, ROPE_DIM)).astype(np.float32)
  ).astype(jnp.float8_e4m3fn)
  rope_padded = jnp.pad(
      rope_fp8,
      ((0, 0), (0, 0), (0, reference.ROPE_STORAGE_DIM - ROPE_DIM)),
  )

  cache_kv_nope = jax.lax.bitcast_convert_type(nope_fp8, jnp.uint8).reshape(
      total_pages, page_size, 4, 128
  )
  cache_kv_rope = jax.lax.bitcast_convert_type(rope_padded, jnp.uint8).reshape(
      total_pages, page_size // 4, 4, 128
  )

  distribution = jnp.array(
      [num_decode_seqs, num_decode_seqs, batch_size], dtype=jnp.int32
  )

  inputs = (
      q,
      cache_kv_nope,
      cache_kv_rope,
      kv_lens,
      topk_indices,
      page_indices,
      cu_q_lens,
      distribution,
  )
  return inputs, actual_tokens


def assert_matches_reference(
    actual: jax.Array,
    inputs: tuple[jax.Array, ...],
    actual_tokens: int,
    *,
    sm_scale: float = 1.0 / math.sqrt(HEAD_DIM),
    k_scale: float = 1.0,
    mask_value: float | None = None,
    max_kv_len: int | None = None,
    sequence_start: jax.Array | None = None,
) -> None:
  """Asserts that `actual` matches the reference output on valid tokens."""
  expected = reference.masked_dense_ragged_paged_attention(
      *inputs,
      sm_scale=sm_scale,
      k_scale=k_scale,
      mask_value=mask_value,
      max_kv_len=max_kv_len,
      sequence_start=sequence_start,
  )
  assert actual.shape == expected.shape, (
      f"Expected shape {expected.shape}, got {actual.shape}."
  )
  np.testing.assert_allclose(
      actual[:actual_tokens],
      expected[:actual_tokens],
      rtol=RTOL,
      atol=ATOL,
  )


class MaskedDenseMlaTestBase(parameterized.TestCase):
  """Shared correctness suite for MaskedDenseMLA operator implementations."""

  def __init__(self, *args, masked_dense_mla_fn):
    super().__init__(*args)
    self._masked_dense_mla_fn = masked_dense_mla_fn

  @parameterized.named_parameters(
      dict(
          testcase_name="bitmap_unpadded",
          topk=64,
          pad_tokens=False,
          k_scale=1.0,
      ),
      dict(
          testcase_name="bitmap_padded",
          topk=64,
          pad_tokens=True,
          k_scale=1.0,
      ),
      dict(
          testcase_name="bitmap_k_scale_half",
          topk=64,
          pad_tokens=False,
          k_scale=0.5,
      ),
      dict(
          testcase_name="analytic_causal_mask",
          topk=256,
          pad_tokens=False,
          k_scale=1.0,
      ),
      dict(
          testcase_name="custom_mask_value",
          topk=64,
          pad_tokens=False,
          k_scale=1.0,
          mask_value=-1e30,
      ),
      dict(
          testcase_name="sequence_start",
          topk=64,
          pad_tokens=False,
          k_scale=1.0,
          sequence_start=2,
      ),
  )
  def test_correctness(
      self,
      topk: int,
      pad_tokens: bool,
      k_scale: float,
      mask_value: float | None = None,
      sequence_start: int | None = None,
  ):
    inputs, actual_tokens = make_inputs(topk=topk, pad_tokens=pad_tokens)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    seq_start_arr = (
        None
        if sequence_start is None
        else jnp.asarray(sequence_start, dtype=jnp.int32)
    )
    actual = self._masked_dense_mla_fn(
        *inputs,
        sm_scale=sm_scale,
        k_scale=k_scale,
        mask_value=mask_value,
        sequence_start=seq_start_arr,
    )
    assert_matches_reference(
        actual,
        inputs,
        actual_tokens,
        sm_scale=sm_scale,
        k_scale=k_scale,
        mask_value=mask_value,
        sequence_start=seq_start_arr,
    )
