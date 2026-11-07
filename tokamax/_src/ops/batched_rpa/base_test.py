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
"""Tests for Batched RPA base operator."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa import base
from tokamax._src.ops.batched_rpa import reference


def _naive_single_token_attention(
    token_idx: jax.Array,
    seq_idx: jax.Array,
    token_kv_pos: jax.Array,
    scaled_queries: jax.Array,
    all_k: jax.Array,
    all_v: jax.Array,
    kv_lens: jax.Array,
    sm_scale: float,
    use_causal_mask: bool,
    sliding_window: int | None,
    soft_cap: float | None,
    out_dtype: jax.typing.DTypeLike,
) -> tuple[jax.Array, jax.Array]:
  """Per-token attention oracle matching the unchunked reference formulation."""
  seq_id = seq_idx[token_idx]
  q_token = scaled_queries[token_idx]
  seq_k = all_k[seq_id]
  seq_v = all_v[seq_id]
  max_kv_len = seq_k.shape[0]

  scores = (
      jnp.einsum(
          "hd,khd->hk", q_token, seq_k, preferred_element_type=jnp.float32
      )
      * sm_scale
  )
  if soft_cap is not None:
    scores = soft_cap * jnp.tanh(scores / soft_cap)

  k_positions = jnp.arange(max_kv_len)
  mask = k_positions < kv_lens[seq_id]
  if use_causal_mask:
    mask = mask & (k_positions <= token_kv_pos[token_idx])
  if sliding_window is not None:
    mask = mask & (k_positions > token_kv_pos[token_idx] - sliding_window)

  scores = jnp.where(mask[None, :], scores, -1e30)
  lse = jax.nn.logsumexp(scores, axis=-1).astype(jnp.float32)
  weights = jax.nn.softmax(scores, axis=-1).astype(out_dtype)
  token_out = jnp.einsum(
      "hk,khd->hd", weights, seq_v, preferred_element_type=jnp.float32
  ).astype(out_dtype)
  return token_out, lse


def _naive_per_token_reference(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    use_causal_mask: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: jax.typing.DTypeLike | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    skip_kv_update: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Unchunked vmap-over-tokens reference used as a test oracle."""
  out_dtype = queries.dtype if out_dtype is None else out_dtype
  actual_head_dim = queries.shape[-1]
  actual_num_q_heads = queries.shape[1]
  actual_num_kv_heads = keys.shape[1]
  actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
  (
      total_pages,
      page_size,
      num_kv_heads_x2_per_kv_packing,
      kv_packing,
      head_dim_aligned,
  ) = kv_cache.shape
  num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
  max_num_seqs = kv_lens.shape[0]
  pages_per_seq = page_indices.shape[0] // max_num_seqs
  max_kv_len = pages_per_seq * page_size
  total_q_tokens = queries.shape[0]

  token_indices = jnp.arange(total_q_tokens)
  seq_idx = jnp.searchsorted(cu_q_lens, token_indices, side="right") - 1
  seq_idx = jnp.clip(seq_idx, 0, max_num_seqs - 1)
  q_start = cu_q_lens[seq_idx]
  q_end = cu_q_lens[seq_idx + 1]
  q_len = q_end - q_start
  token_kv_pos = kv_lens[seq_idx] - q_len + (token_indices - q_start)

  active_count = distribution[-1]
  valid_token = (token_indices < cu_q_lens[active_count]) & (
      seq_idx < active_count
  )

  if skip_kv_update:
    updated_kv_cache = kv_cache
  else:
    merged_kv = reference._merge_kv(keys, values, kv_packing=kv_packing)
    page_num = token_kv_pos // page_size
    page_idx = page_indices[seq_idx * pages_per_seq + page_num]
    page_offset = token_kv_pos % page_size
    dummy_page = jnp.zeros(
        (
            1,
            page_size,
            num_kv_heads_x2_per_kv_packing,
            kv_packing,
            head_dim_aligned,
        ),
        dtype=kv_cache.dtype,
    )
    padded_cache = jnp.concatenate([kv_cache, dummy_page], axis=0)
    safe_page_idx = jnp.where(valid_token, page_idx, total_pages)
    safe_page_offset = jnp.where(valid_token, page_offset, 0)
    padded_cache = padded_cache.at[safe_page_idx, safe_page_offset].set(
        merged_kv
    )
    updated_kv_cache = padded_cache[:total_pages]

  page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
  seq_kv_pages = updated_kv_cache[page_table]
  flat_kv = seq_kv_pages.reshape(
      max_num_seqs, max_kv_len, num_kv_heads_x2, head_dim_aligned
  )[:, :, : actual_num_kv_heads * 2]
  all_k = jnp.repeat(
      flat_kv[:, :, 0::2, :actual_head_dim],
      actual_num_q_heads_per_kv_head,
      axis=2,
  )
  all_v = jnp.repeat(
      flat_kv[:, :, 1::2, :actual_head_dim],
      actual_num_q_heads_per_kv_head,
      axis=2,
  )
  scaled_queries = queries / q_scale if q_scale is not None else queries
  if k_scale is not None:
    all_k = all_k / k_scale
  if v_scale is not None:
    all_v = all_v / v_scale

  attend_fn = functools.partial(
      _naive_single_token_attention,
      seq_idx=seq_idx,
      token_kv_pos=token_kv_pos,
      scaled_queries=scaled_queries,
      all_k=all_k,
      all_v=all_v,
      kv_lens=kv_lens,
      sm_scale=sm_scale,
      use_causal_mask=use_causal_mask,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      out_dtype=out_dtype,
  )
  token_outputs, token_lses = jax.vmap(attend_fn)(token_indices)
  outputs = jnp.where(
      valid_token[:, None, None], token_outputs, jnp.zeros_like(token_outputs)
  )
  lse = jnp.where(
      valid_token[:, None], token_lses, jnp.full_like(token_lses, -jnp.inf)
  )
  return outputs, updated_kv_cache, lse


def _compile_temp_bytes(fn, total_q_tokens: int) -> int:
  """Returns compiled XLA temporary buffer size in bytes for `fn`."""
  max_num_seqs = 4
  page_size = 64
  pages_per_seq = 8
  total_pages = max_num_seqs * pages_per_seq
  num_q_heads = 8
  num_kv_heads = 2
  head_dim = 128

  queries = jax.ShapeDtypeStruct(
      (total_q_tokens, num_q_heads, head_dim), jnp.bfloat16
  )
  keys = jax.ShapeDtypeStruct(
      (total_q_tokens, num_kv_heads, head_dim), jnp.bfloat16
  )
  values = jax.ShapeDtypeStruct(
      (total_q_tokens, num_kv_heads, head_dim), jnp.bfloat16
  )
  kv_cache = jax.ShapeDtypeStruct(
      (total_pages, page_size, num_kv_heads, 2, head_dim), jnp.bfloat16
  )
  kv_lens = jax.ShapeDtypeStruct((max_num_seqs,), jnp.int32)
  page_indices = jax.ShapeDtypeStruct((total_pages,), jnp.int32)
  cu_q_lens = jax.ShapeDtypeStruct((max_num_seqs + 1,), jnp.int32)
  distribution = jax.ShapeDtypeStruct((3,), jnp.int32)
  compiled = (
      jax.jit(fn)
      .lower(
          queries,
          keys,
          values,
          kv_cache,
          kv_lens,
          page_indices,
          cu_q_lens,
          distribution,
      )
      .compile()
  )
  stats = compiled.memory_analysis()
  assert stats is not None
  return stats.temp_size_in_bytes


class BatchedRpaBaseTest(parameterized.TestCase):

  def test_base_matches_reference(self):
    total_q_tokens = 4
    max_num_seqs = 2
    seq_len = 16
    page_size = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    total_pages = 8
    head_dim_aligned = 128
    pages_per_seq = 2

    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    queries = jax.random.normal(
        k1, (total_q_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16
    )
    keys = jax.random.normal(
        k2, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16
    )
    values = jax.random.normal(
        k3, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16
    )
    kv_cache = jnp.zeros(
        (total_pages, page_size, num_kv_heads, 2, head_dim_aligned),
        dtype=jnp.bfloat16,
    )
    kv_lens = jnp.array([8, 8], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

    ref_out, ref_kv = reference.batched_ragged_paged_attention_reference(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
    )

    op = base.BatchedRpa()
    base_out, base_kv = op(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
    )

    self.assertEqual(ref_out.shape, base_out.shape)
    self.assertEqual(ref_kv.shape, base_kv.shape)
    self.assertTrue(jnp.allclose(ref_out, base_out, atol=1e-3))

  def test_reference_jit_mixed_prefill_decode(self):
    total_q_tokens = 5
    max_num_seqs = 2
    page_size = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    total_pages = 8
    head_dim_aligned = 128

    k1, k2, k3 = jax.random.split(jax.random.key(1), 3)
    queries = jax.random.normal(
        k1, (total_q_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16
    )
    keys = jax.random.normal(
        k2, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16
    )
    values = jax.random.normal(
        k3, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16
    )
    kv_cache = jnp.zeros(
        (total_pages, page_size, num_kv_heads, 2, head_dim_aligned),
        dtype=jnp.bfloat16,
    )
    # Sequence 0: decode (q_len = 1, kv_len = 8)
    # Sequence 1: prefill (q_len = 4, kv_len = 4)
    kv_lens = jnp.array([8, 4], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 1, 5], dtype=jnp.int32)
    # distribution: 1 decode, 1 prefill (cumulative 2), 0 mixed (cumulative 2)
    distribution = jnp.array([1, 2, 2], dtype=jnp.int32)

    @jax.jit
    def run_fn(q, k, v, cache, kl, pi, cu_q, dist):
      return reference.batched_ragged_paged_attention_reference(
          queries=q,
          keys=k,
          values=v,
          kv_cache=cache,
          kv_lens=kl,
          page_indices=pi,
          cu_q_lens=cu_q,
          distribution=dist,
      )

    out, new_cache = run_fn(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )
    self.assertEqual(out.shape, queries.shape)
    self.assertEqual(new_cache.shape, kv_cache.shape)

  def test_high_level_api_with_extended_kwargs(self):
    total_q_tokens = 4
    max_num_seqs = 2
    page_size = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    total_pages = 8
    head_dim_aligned = 128

    k1, k2, k3 = jax.random.split(jax.random.key(2), 3)
    queries = jax.random.normal(
        k1, (total_q_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16
    )
    keys = jax.random.normal(
        k2, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16
    )
    values = jax.random.normal(
        k3, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16
    )
    kv_cache = jnp.zeros(
        (total_pages, page_size, num_kv_heads, 2, head_dim_aligned),
        dtype=jnp.bfloat16,
    )
    kv_lens = jnp.array([8, 8], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

    from tokamax._src.ops.batched_rpa import api

    out, new_cache, lse = (  # pyrefly: ignore[bad-unpacking]
        api.batched_ragged_paged_attention(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            v_scale=1.0,
            decode_query_size=2,
            skip_kv_update=True,
            return_lse=True,
            implementation="reference",
        )
    )
    self.assertEqual(out.shape, queries.shape)
    self.assertEqual(new_cache.shape, kv_cache.shape)
    self.assertEqual(lse.shape, (total_q_tokens, num_q_heads))

    out_std, new_cache_std = (  # pyrefly: ignore[bad-unpacking]
        api.batched_ragged_paged_attention(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            v_scale=1.0,
            decode_query_size=2,
            skip_kv_update=True,
            return_lse=False,
            implementation="reference",
        )
    )
    self.assertEqual(out_std.shape, queries.shape)
    self.assertEqual(new_cache_std.shape, kv_cache.shape)

  @parameterized.named_parameters(
      dict(
          testcase_name="decode_only",
          kv_lens=[12, 7, 15, 4],
          cu_q_lens=[0, 1, 2, 3, 4],
          total_q_tokens=4,
          active_seqs=4,
          use_causal_mask=True,
          sliding_window=None,
          soft_cap=None,
          q_scale=None,
          k_scale=None,
          v_scale=None,
          skip_kv_update=False,
      ),
      dict(
          testcase_name="prefill_only",
          kv_lens=[10, 14],
          cu_q_lens=[0, 10, 24],
          total_q_tokens=24,
          active_seqs=2,
          use_causal_mask=True,
          sliding_window=None,
          soft_cap=None,
          q_scale=None,
          k_scale=None,
          v_scale=None,
          skip_kv_update=False,
      ),
      dict(
          testcase_name="mixed_multi_chunk_and_inactive_tail",
          kv_lens=[16, 80, 12, 72, 16],
          cu_q_lens=[0, 1, 71, 72, 137, 140],
          total_q_tokens=144,
          active_seqs=4,
          use_causal_mask=True,
          sliding_window=None,
          soft_cap=None,
          q_scale=None,
          k_scale=None,
          v_scale=None,
          skip_kv_update=False,
      ),
      dict(
          testcase_name="non_causal_and_scales",
          kv_lens=[12, 9, 15],
          cu_q_lens=[0, 4, 5, 11],
          total_q_tokens=11,
          active_seqs=3,
          use_causal_mask=False,
          sliding_window=None,
          soft_cap=20.0,
          q_scale=0.5,
          k_scale=0.25,
          v_scale=2.0,
          skip_kv_update=False,
      ),
      dict(
          testcase_name="sliding_window_and_skip_kv_update",
          kv_lens=[16, 14],
          cu_q_lens=[0, 6, 14],
          total_q_tokens=14,
          active_seqs=2,
          use_causal_mask=True,
          sliding_window=4,
          soft_cap=15.0,
          q_scale=None,
          k_scale=None,
          v_scale=None,
          skip_kv_update=True,
      ),
  )
  def test_reference_matches_per_token_oracle(
      self,
      kv_lens,
      cu_q_lens,
      total_q_tokens,
      active_seqs,
      use_causal_mask,
      sliding_window,
      soft_cap,
      q_scale,
      k_scale,
      v_scale,
      skip_kv_update,
  ):
    max_num_seqs = len(kv_lens)
    page_size = 8
    pages_per_seq = (max(kv_lens) + page_size - 1) // page_size
    total_pages = max_num_seqs * pages_per_seq
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    head_dim_aligned = 128

    k1, k2, k3, k4 = jax.random.split(jax.random.key(42), 4)
    queries = jax.random.normal(
        k1, (total_q_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16
    )
    keys = jax.random.normal(
        k2, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16
    )
    values = jax.random.normal(
        k3, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16
    )
    kv_cache = jax.random.normal(
        k4,
        (total_pages, page_size, num_kv_heads, 2, head_dim_aligned),
        dtype=jnp.bfloat16,
    )
    kv_lens_arr = jnp.asarray(kv_lens, dtype=jnp.int32)
    page_indices_arr = jnp.arange(total_pages, dtype=jnp.int32)
    cu_q_lens_arr = jnp.asarray(cu_q_lens, dtype=jnp.int32)
    distribution_arr = jnp.asarray([0, 0, active_seqs], dtype=jnp.int32)

    expected_out, expected_kv, expected_lse = _naive_per_token_reference(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens_arr,
        page_indices=page_indices_arr,
        cu_q_lens=cu_q_lens_arr,
        distribution=distribution_arr,
        use_causal_mask=use_causal_mask,
        sm_scale=0.125,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        skip_kv_update=skip_kv_update,
    )
    actual_out, actual_kv, actual_lse = (  # pyrefly: ignore[bad-unpacking]
        reference.batched_ragged_paged_attention_reference(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens_arr,
            page_indices=page_indices_arr,
            cu_q_lens=cu_q_lens_arr,
            distribution=distribution_arr,
            use_causal_mask=use_causal_mask,
            sm_scale=0.125,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            skip_kv_update=skip_kv_update,
            return_lse=True,
        )
    )

    self.assertTrue(jnp.array_equal(actual_kv, expected_kv))
    self.assertTrue(
        jnp.allclose(actual_out, expected_out, atol=1e-2, rtol=1e-2)
    )
    active_limit = int(cu_q_lens[active_seqs])
    self.assertTrue(
        jnp.allclose(
            actual_lse[:active_limit],
            expected_lse[:active_limit],
            atol=1e-4,
            rtol=1e-4,
        )
    )
    if active_limit < total_q_tokens:
      self.assertTrue(jnp.all(jnp.isneginf(actual_lse[active_limit:])))
      self.assertTrue(jnp.all(actual_out[active_limit:] == 0))

  def test_reference_temp_memory_bounded_with_tokens(self):
    chunked_128 = _compile_temp_bytes(
        reference.batched_ragged_paged_attention_reference, 128
    )
    chunked_512 = _compile_temp_bytes(
        reference.batched_ragged_paged_attention_reference, 512
    )
    # Increasing total_q_tokens 4x (128 -> 512) only grows the padded_queries /
    # padded_outputs buffers (~3 MiB), not [total_q_tokens, max_kv_len, ...].
    self.assertLess(chunked_512 - chunked_128, 16 * 1024 * 1024)

  @parameterized.named_parameters(
      dict(
          testcase_name="qwen3_5_397b_isl8k_mixed_c64_dp8tp1",
          total_q_tokens=1031,
          max_num_seqs=8,
          pages_per_seq=36,
          total_pages=284,
      ),
      dict(
          testcase_name="qwen3_8_max_tp2_mixed_b48_t8192",
          total_q_tokens=8192,
          max_num_seqs=49,
          pages_per_seq=32,
          total_pages=1568,
      ),
  )
  def test_large_mixed_benchmark_specs_temp_memory(
      self,
      total_q_tokens: int,
      max_num_seqs: int,
      pages_per_seq: int,
      total_pages: int,
  ):
    num_q_heads = 32
    num_kv_heads = 2
    head_dim = 256
    page_size = 256
    kv_packing = 4

    queries = jax.ShapeDtypeStruct(
        (total_q_tokens, num_q_heads, head_dim), jnp.bfloat16
    )
    keys = jax.ShapeDtypeStruct(
        (total_q_tokens, num_kv_heads, head_dim), jnp.float8_e4m3fn
    )
    values = jax.ShapeDtypeStruct(
        (total_q_tokens, num_kv_heads, head_dim), jnp.float8_e4m3fn
    )
    kv_cache = jax.ShapeDtypeStruct(
        (total_pages, page_size, 1, kv_packing, head_dim), jnp.float8_e4m3fn
    )
    kv_lens = jax.ShapeDtypeStruct((max_num_seqs,), jnp.int32)
    page_indices = jax.ShapeDtypeStruct(
        (max_num_seqs * pages_per_seq,), jnp.int32
    )
    cu_q_lens = jax.ShapeDtypeStruct((max_num_seqs + 1,), jnp.int32)
    distribution = jax.ShapeDtypeStruct((3,), jnp.int32)

    compiled = (
        jax.jit(reference.batched_ragged_paged_attention_reference)
        .lower(
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
        .compile()
    )
    stats = compiled.memory_analysis()
    self.assertIsNotNone(stats)
    self.assertLess(stats.temp_size_in_bytes, 2 * 1024**3)


if __name__ == "__main__":
  absltest.main()
