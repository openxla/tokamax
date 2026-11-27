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
from tokamax._src.ops.batched_rpa import api
from tokamax._src.ops.batched_rpa import base
from tokamax._src.ops.batched_rpa import reference
from tokamax._src.ops.batched_rpa.kernel import configs

jax.config.parse_flags_with_absl()


@jax.jit
def _reference_run_fn(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  return reference.batched_ragged_paged_attention_reference(
      queries=queries,
      keys=keys,
      values=values,
      kv_cache=kv_cache,
      kv_lens=kv_lens,
      page_indices=page_indices,
      cu_q_lens=cu_q_lens,
      distribution=distribution,
  )


def _naive_single_token_attention(
    token_idx: jax.Array,
    seq_idx: jax.Array,
    token_kv_pos: jax.Array,
    scaled_queries: jax.Array,
    all_k: jax.Array,
    all_v: jax.Array,
    kv_lens: jax.Array,
    sm_scale: float,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    use_causal_mask: bool,
    sliding_window: int | None,
    soft_cap: float | None,
    mask_value: float,
    attention_sink: jax.Array | None,
    quantize_p: bool,
    kv_cache_dtype: jax.typing.DTypeLike,
    out_dtype: jax.typing.DTypeLike,
) -> tuple[jax.Array, jax.Array]:
  """Per-token attention oracle matching the unchunked reference formulation."""
  seq_id = seq_idx[token_idx]
  q_token = scaled_queries[token_idx].astype(jnp.float32)
  seq_k = all_k[seq_id]
  seq_v = all_v[seq_id]
  max_kv_len = seq_k.shape[0]

  scores = (
      jnp.einsum(
          "hd,khd->hk", q_token, seq_k, preferred_element_type=jnp.float32
      )
      * sm_scale
  )
  if q_scale is not None:
    scores = scores * q_scale
  if k_scale is not None:
    scores = scores * k_scale
  if soft_cap is not None:
    scores = soft_cap * jnp.tanh(scores / soft_cap)

  k_positions = jnp.arange(max_kv_len)
  mask = k_positions < kv_lens[seq_id]
  if use_causal_mask:
    mask = mask & (k_positions <= token_kv_pos[token_idx])
  if sliding_window is not None:
    mask = mask & (k_positions > token_kv_pos[token_idx] - sliding_window)

  scores = jnp.where(mask[None, :], scores, mask_value)
  if attention_sink is not None:
    sink = attention_sink.astype(scores.dtype)[:, None]
    max_score = jnp.max(
        jnp.concatenate([sink, scores], axis=-1), axis=-1, keepdims=True
    )
    probs = jnp.exp(scores - max_score)
    sum_probs = jnp.sum(probs, axis=-1, keepdims=True) + jnp.exp(
        sink - max_score
    )
  else:
    max_score = jnp.max(scores, axis=-1, keepdims=True)
    probs = jnp.exp(scores - max_score)
    sum_probs = jnp.sum(probs, axis=-1, keepdims=True)
  lse = jnp.squeeze(max_score + jnp.log(sum_probs), axis=-1).astype(out_dtype)

  if quantize_p:
    probs = probs.astype(kv_cache_dtype).astype(jnp.float32)

  token_out = (
      jnp.einsum("hk,khd->hd", probs, seq_v, preferred_element_type=jnp.float32)
      / sum_probs
  )
  if v_scale is not None:
    token_out = token_out * v_scale
  return token_out.astype(out_dtype), lse


def _naive_per_token_reference(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sink: jax.Array | None = None,
    *,
    use_causal_mask: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    out_dtype: jax.typing.DTypeLike | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    skip_kv_update: bool = False,
    quantize_p: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Unchunked vmap-over-tokens reference used as a test oracle."""
  out_dtype = queries.dtype if out_dtype is None else out_dtype
  effective_mask_value = (
      float(jnp.finfo(jnp.dtype(out_dtype)).min)
      if mask_value is None
      else mask_value
  )
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
    num_dummy_pages = (total_q_tokens + page_size - 1) // page_size
    dummy_pages = jnp.zeros(
        (
            num_dummy_pages,
            page_size,
            num_kv_heads_x2_per_kv_packing,
            kv_packing,
            head_dim_aligned,
        ),
        dtype=kv_cache.dtype,
    )
    padded_cache = jnp.concatenate([kv_cache, dummy_pages], axis=0)
    safe_page_idx = jnp.where(
        valid_token, page_idx, total_pages + token_indices // page_size
    )
    safe_page_offset = jnp.where(
        valid_token, page_offset, token_indices % page_size
    )
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
      flat_kv[:, :, 0::2, :actual_head_dim].astype(jnp.float32),
      actual_num_q_heads_per_kv_head,
      axis=2,
  )
  all_v = jnp.repeat(
      flat_kv[:, :, 1::2, :actual_head_dim].astype(jnp.float32),
      actual_num_q_heads_per_kv_head,
      axis=2,
  )
  if q_scale is not None:
    scaled_queries = queries / q_scale
    if jnp.issubdtype(kv_cache.dtype, jnp.floating):
      dtype_info = jnp.finfo(kv_cache.dtype)
      scaled_queries = jnp.clip(
          scaled_queries,
          min=float(dtype_info.min),
          max=float(dtype_info.max),
      )
    scaled_queries = scaled_queries.astype(kv_cache.dtype)
  else:
    scaled_queries = queries

  attend_fn = functools.partial(
      _naive_single_token_attention,
      seq_idx=seq_idx,
      token_kv_pos=token_kv_pos,
      scaled_queries=scaled_queries,
      all_k=all_k,
      all_v=all_v,
      kv_lens=kv_lens,
      sm_scale=sm_scale,
      q_scale=q_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      use_causal_mask=use_causal_mask,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=effective_mask_value,
      attention_sink=attention_sink,
      quantize_p=quantize_p,
      kv_cache_dtype=kv_cache.dtype,
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
    page_size = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    total_pages = 8
    head_dim_aligned = 128

    k1, k2, k3, _ = jax.random.split(jax.random.key(0), 4)
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
    self.assertTrue(jnp.allclose(ref_out, base_out, atol=0.1, rtol=0.1))

  def test_reference_jit_mixed_prefill_decode(self):
    total_q_tokens = 5
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
    kv_lens = jnp.array([8, 4], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 1, 5], dtype=jnp.int32)
    distribution = jnp.array([1, 2, 2], dtype=jnp.int32)

    out, new_cache = _reference_run_fn(
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

  def test_attention_sink_matches_expected(self):
    total_q_tokens = 4
    page_size = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    total_pages = 8
    head_dim_aligned = 128

    k1, k2, k3, k4 = jax.random.split(jax.random.key(3), 4)
    queries = jax.random.normal(
        k1, (total_q_tokens, num_q_heads, head_dim), dtype=jnp.float32
    )
    keys = jax.random.normal(
        k2, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.float32
    )
    values = jax.random.normal(
        k3, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.float32
    )
    kv_cache = jnp.zeros(
        (total_pages, page_size, num_kv_heads, 2, head_dim_aligned),
        dtype=jnp.float32,
    )
    kv_lens = jnp.array([8, 8], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)
    attention_sink = jax.random.normal(k4, (num_q_heads,), dtype=jnp.float32)

    out_no_sink, _, lse_no_sink = (  # pyrefly: ignore[bad-unpacking]
        api.batched_ragged_paged_attention(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            return_lse=True,
            implementation="reference",
        )
    )
    out_sink, _, lse_sink = (  # pyrefly: ignore[bad-unpacking]
        api.batched_ragged_paged_attention(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            attention_sink=attention_sink,
            return_lse=True,
            implementation="reference",
        )
    )

    expected_lse = jnp.logaddexp(lse_no_sink, attention_sink[None, :])
    expected_scale = jnp.exp(lse_no_sink - expected_lse)[:, :, None]
    self.assertTrue(jnp.allclose(lse_sink, expected_lse, atol=0.1, rtol=0.1))
    self.assertTrue(
        jnp.allclose(out_sink, out_no_sink * expected_scale, atol=0.1, rtol=0.1)
    )

  def test_quantize_p_bf16_close_to_unquantized(self):
    total_q_tokens = 4
    page_size = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    total_pages = 8
    head_dim_aligned = 128

    key1, key2, key3 = jax.random.split(jax.random.key(4), 3)
    queries = jax.random.normal(
        key1, (total_q_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16
    )
    keys = jax.random.normal(
        key2, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16
    )
    values = jax.random.normal(
        key3, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16
    )
    kv_cache = jnp.zeros(
        (total_pages, page_size, num_kv_heads, 2, head_dim_aligned),
        dtype=jnp.bfloat16,
    )
    kv_lens = jnp.array([8, 8], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

    out_unquant, _, lse_unquant = (
        reference.batched_ragged_paged_attention_reference(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            quantize_p=False,
            return_lse=True,
        )
    )
    out_quant, _, lse_quant = (
        reference.batched_ragged_paged_attention_reference(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            quantize_p=True,
            return_lse=True,
        )
    )
    self.assertTrue(jnp.allclose(lse_quant, lse_unquant, atol=1e-2, rtol=1e-2))
    self.assertTrue(jnp.allclose(out_quant, out_unquant, atol=0.2, rtol=0.2))

  def test_bind_int8_kv_head_dim_matches(self):
    total_q_tokens = 4
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    op = base.BatchedRpa()
    queries = jnp.zeros(
        (total_q_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16
    )
    keys = jnp.zeros((total_q_tokens, num_kv_heads, head_dim), dtype=jnp.int8)
    values = jnp.zeros((total_q_tokens, num_kv_heads, head_dim), dtype=jnp.int8)
    kv_cache = jnp.zeros((4, 8, num_kv_heads, 2, 128), dtype=jnp.int8)
    kv_lens = jnp.array([8, 8], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

    bound = op.bind(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
    )
    self.assertEqual(bound.arguments["keys"].dtype, jnp.int8)

  def test_bind_packed_uint8_kv_head_dim_halved(self):
    total_q_tokens = 4
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    kv_head_dim = head_dim // 2
    op = base.BatchedRpa()
    queries = jnp.zeros(
        (total_q_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16
    )
    keys = jnp.zeros(
        (total_q_tokens, num_kv_heads, kv_head_dim), dtype=jnp.uint8
    )
    values = jnp.zeros(
        (total_q_tokens, num_kv_heads, kv_head_dim), dtype=jnp.uint8
    )
    kv_cache = jnp.zeros((4, 8, num_kv_heads, 2, 128), dtype=jnp.uint8)
    kv_lens = jnp.array([8, 8], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

    bound = op.bind(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
    )
    self.assertEqual(bound.arguments["keys"].shape[-1], kv_head_dim)

  def test_bind_mismatched_kv_head_dim_raises(self):
    total_q_tokens = 4
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    op = base.BatchedRpa()
    queries = jnp.zeros(
        (total_q_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16
    )
    kv_cache = jnp.zeros((4, 8, num_kv_heads, 2, 128), dtype=jnp.bfloat16)
    kv_lens = jnp.array([8, 8], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

    with self.assertRaises(ValueError):
      op.bind(
          queries=queries,
          keys=jnp.zeros(
              (total_q_tokens, num_kv_heads, 32), dtype=jnp.bfloat16
          ),
          values=jnp.zeros(
              (total_q_tokens, num_kv_heads, 32), dtype=jnp.bfloat16
          ),
          kv_cache=kv_cache,
          kv_lens=kv_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=distribution,
      )

    with self.assertRaises(ValueError):
      op.bind(
          queries=queries,
          keys=jnp.zeros((total_q_tokens, num_kv_heads, 30), dtype=jnp.uint8),
          values=jnp.zeros((total_q_tokens, num_kv_heads, 30), dtype=jnp.uint8),
          kv_cache=kv_cache,
          kv_lens=kv_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=distribution,
      )

  def test_quantize_p_with_v_scale(self):
    total_q_tokens = 4
    page_size = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    total_pages = 8
    head_dim_aligned = 128

    key1, key2, key3 = jax.random.split(jax.random.key(10), 3)
    queries = jax.random.normal(
        key1, (total_q_tokens, num_q_heads, head_dim), dtype=jnp.float32
    )
    keys = jax.random.normal(
        key2, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.float32
    ).astype(jnp.float8_e4m3fn)
    values = jax.random.normal(
        key3, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.float32
    ).astype(jnp.float8_e4m3fn)
    kv_cache = jnp.zeros(
        (total_pages, page_size, num_kv_heads, 2, head_dim_aligned),
        dtype=jnp.float8_e4m3fn,
    )
    kv_lens = jnp.array([8, 8], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

    out_unquant, _ = reference.batched_ragged_paged_attention_reference(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        v_scale=0.5,
        quantize_p=False,
    )
    out_quant, _ = reference.batched_ragged_paged_attention_reference(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        v_scale=0.5,
        quantize_p=True,
    )
    self.assertEqual(out_quant.shape, queries.shape)
    self.assertTrue(jnp.all(jnp.isfinite(out_quant)))
    self.assertTrue(jnp.allclose(out_quant, out_unquant, atol=1.0, rtol=1.0))

  def test_scale_dequantization(self):
    total_q_tokens = 4
    page_size = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    total_pages = 8
    head_dim_aligned = 128

    key1, key2, key3 = jax.random.split(jax.random.key(11), 3)
    queries = jax.random.normal(
        key1, (total_q_tokens, num_q_heads, head_dim), dtype=jnp.float32
    )
    keys = jax.random.normal(
        key2, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.float32
    )
    values = jax.random.normal(
        key3, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.float32
    )
    kv_cache = jnp.zeros(
        (total_pages, page_size, num_kv_heads, 2, head_dim_aligned),
        dtype=jnp.float32,
    )
    kv_lens = jnp.array([8, 8], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

    out_unscaled, _ = reference.batched_ragged_paged_attention_reference(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
    )
    out_id_scales, _ = reference.batched_ragged_paged_attention_reference(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        q_scale=1.0,
        k_scale=1.0,
        v_scale=1.0,
    )
    self.assertTrue(jnp.allclose(out_id_scales, out_unscaled, atol=1e-5))

    out_v_scale, _ = reference.batched_ragged_paged_attention_reference(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        v_scale=2.5,
    )
    self.assertTrue(jnp.allclose(out_v_scale, out_unscaled * 2.5, atol=1e-5))

    out_k_scale, _ = reference.batched_ragged_paged_attention_reference(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=0.5,
        k_scale=2.0,
    )
    out_double_sm, _ = reference.batched_ragged_paged_attention_reference(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=1.0,
    )
    self.assertTrue(jnp.allclose(out_k_scale, out_double_sm, atol=1e-5))

    out_q_scale, _ = reference.batched_ragged_paged_attention_reference(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        q_scale=0.5,
    )
    self.assertTrue(jnp.allclose(out_q_scale, out_unscaled, atol=1e-5))

  def test_scale_q_clipping_to_kv_dtype(self):
    total_q_tokens = 2
    page_size = 8
    num_q_heads = 2
    num_kv_heads = 2
    head_dim = 64
    total_pages = 4
    head_dim_aligned = 128

    queries_large = jnp.full(
        (total_q_tokens, num_q_heads, head_dim), 100000.0, dtype=jnp.float32
    )
    queries_clipped = jnp.full(
        (total_q_tokens, num_q_heads, head_dim), 65504.0, dtype=jnp.float32
    )
    key1, key2 = jax.random.split(jax.random.key(42), 2)
    keys = jax.random.normal(
        key1, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.float16
    )
    values = jax.random.normal(
        key2, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.float16
    )
    kv_cache = jnp.zeros(
        (total_pages, page_size, num_kv_heads, 2, head_dim_aligned),
        dtype=jnp.float16,
    )
    kv_lens = jnp.array([8, 8], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 1, 2], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

    out_large, _ = reference.batched_ragged_paged_attention_reference(
        queries=queries_large,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        q_scale=1.0,
    )
    out_clipped, _ = reference.batched_ragged_paged_attention_reference(
        queries=queries_clipped,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        q_scale=1.0,
    )
    self.assertTrue(jnp.array_equal(out_large, out_clipped))

  def test_custom_mask_value(self):
    total_q_tokens = 4
    page_size = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    total_pages = 8
    head_dim_aligned = 128

    key1, key2, key3 = jax.random.split(jax.random.key(20), 3)
    queries = jax.random.normal(
        key1, (total_q_tokens, num_q_heads, head_dim), dtype=jnp.float32
    )
    keys = jax.random.normal(
        key2, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.float32
    )
    values = jax.random.normal(
        key3, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.float32
    )
    kv_cache = jnp.zeros(
        (total_pages, page_size, num_kv_heads, 2, head_dim_aligned),
        dtype=jnp.float32,
    )
    kv_lens = jnp.array([6, 6], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

    out_default, _, lse_default = (
        reference.batched_ragged_paged_attention_reference(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            mask_value=None,
            return_lse=True,
        )
    )
    out_large_neg, _, lse_large_neg = (
        reference.batched_ragged_paged_attention_reference(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            mask_value=-1e4,
            return_lse=True,
        )
    )
    _, _, lse_zero_mask = reference.batched_ragged_paged_attention_reference(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        mask_value=0.0,
        return_lse=True,
    )
    self.assertTrue(jnp.allclose(out_large_neg, out_default, atol=1e-5))
    self.assertTrue(jnp.allclose(lse_large_neg, lse_default, atol=1e-5))
    self.assertFalse(jnp.allclose(lse_zero_mask, lse_default, atol=1e-2))

  def test_return_lse_dtype(self):
    total_q_tokens = 4
    page_size = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    total_pages = 8
    head_dim_aligned = 128

    key1, key2, key3 = jax.random.split(jax.random.key(30), 3)
    queries_bf16 = jax.random.normal(
        key1, (total_q_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16
    )
    keys_bf16 = jax.random.normal(
        key2, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16
    )
    values_bf16 = jax.random.normal(
        key3, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16
    )
    kv_cache_bf16 = jnp.zeros(
        (total_pages, page_size, num_kv_heads, 2, head_dim_aligned),
        dtype=jnp.bfloat16,
    )
    kv_lens = jnp.array([8, 8], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

    _, _, lse_bf16 = reference.batched_ragged_paged_attention_reference(
        queries=queries_bf16,
        keys=keys_bf16,
        values=values_bf16,
        kv_cache=kv_cache_bf16,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        return_lse=True,
    )
    self.assertEqual(lse_bf16.dtype, jnp.bfloat16)
    self.assertEqual(lse_bf16.shape, (total_q_tokens, num_q_heads))

    _, _, lse_f32 = reference.batched_ragged_paged_attention_reference(
        queries=queries_bf16,
        keys=keys_bf16,
        values=values_bf16,
        kv_cache=kv_cache_bf16,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        out_dtype=jnp.float32,
        return_lse=True,
    )
    self.assertEqual(lse_f32.dtype, jnp.float32)
    self.assertEqual(lse_f32.shape, (total_q_tokens, num_q_heads))

  def test_folded_fp8_kv_cache_head_along_sublane(self):
    total_q_tokens = 4
    page_size = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 256
    total_pages = 4
    kv_packing = 4
    fold = configs.num_lane_folds_in_head_dim(
        head_dim,
        jnp.dtype(jnp.float8_e4m3fn),
        configs.KVLayout.HEAD_ALONG_SUBLANE,
        128,
    )
    self.assertEqual(fold, 2)

    key1, key2, key3, key4 = jax.random.split(jax.random.key(21), 4)
    queries = (
        jax.random.normal(
            key1, (total_q_tokens, num_q_heads, head_dim), dtype=jnp.float32
        )
        * 0.25
    ).astype(jnp.bfloat16)
    keys = (
        jax.random.normal(
            key2, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.float32
        )
        * 0.25
    ).astype(jnp.float8_e4m3fn)
    values = (
        jax.random.normal(
            key3, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.float32
        )
        * 0.25
    ).astype(jnp.float8_e4m3fn)
    kv_cache_unfolded = (
        jax.random.normal(
            key4,
            (
                total_pages,
                page_size,
                (num_kv_heads * 2) // kv_packing,
                kv_packing,
                head_dim,
            ),
            dtype=jnp.float32,
        )
        * 0.25
    ).astype(jnp.float8_e4m3fn)
    kv_cache_folded = (
        kv_cache_unfolded.reshape(
            total_pages, page_size, num_kv_heads * 2, fold, head_dim // fold
        )
        .transpose(0, 1, 3, 2, 4)
        .reshape(
            total_pages,
            page_size,
            (num_kv_heads * 2 * fold) // kv_packing,
            kv_packing,
            head_dim // fold,
        )
    )
    kv_lens = jnp.array([8, 8], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

    op = base.BatchedRpa()
    out_unfolded, cache_unfolded = op(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache_unfolded,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        kv_layout=base.KVLayout.HEAD_ALONG_SUBLANE,
    )
    out_folded, cache_folded = op(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache_folded,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        kv_layout=base.KVLayout.HEAD_ALONG_SUBLANE,
    )

    self.assertEqual(cache_folded.shape, kv_cache_folded.shape)
    self.assertTrue(
        jnp.array_equal(
            out_folded.astype(jnp.float32), out_unfolded.astype(jnp.float32)
        )
    )
    expected_folded = (
        cache_unfolded.reshape(
            total_pages, page_size, num_kv_heads * 2, fold, head_dim // fold
        )
        .transpose(0, 1, 3, 2, 4)
        .reshape(kv_cache_folded.shape)
    )
    self.assertTrue(
        jnp.array_equal(
            cache_folded.astype(jnp.float32),
            expected_folded.astype(jnp.float32),
        )
    )

    out_skip, cache_skip = op(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache_folded,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        skip_kv_update=True,
        kv_layout=base.KVLayout.HEAD_ALONG_SUBLANE,
    )
    self.assertEqual(cache_skip.shape, kv_cache_folded.shape)
    self.assertTrue(
        jnp.array_equal(
            cache_skip.astype(jnp.float32), kv_cache_folded.astype(jnp.float32)
        )
    )
    self.assertTrue(jnp.all(jnp.isfinite(out_skip)))

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
          mask_value=None,
          q_scale=None,
          k_scale=None,
          v_scale=None,
          skip_kv_update=False,
          use_sink=False,
          quantize_p=False,
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
          mask_value=None,
          q_scale=None,
          k_scale=None,
          v_scale=None,
          skip_kv_update=False,
          use_sink=False,
          quantize_p=False,
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
          mask_value=None,
          q_scale=None,
          k_scale=None,
          v_scale=None,
          skip_kv_update=False,
          use_sink=False,
          quantize_p=False,
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
          mask_value=-1e4,
          q_scale=0.5,
          k_scale=0.25,
          v_scale=2.0,
          skip_kv_update=False,
          use_sink=False,
          quantize_p=False,
      ),
      dict(
          testcase_name="sliding_window_sink_quantize_p_and_skip_kv_update",
          kv_lens=[16, 14],
          cu_q_lens=[0, 6, 14],
          total_q_tokens=14,
          active_seqs=2,
          use_causal_mask=True,
          sliding_window=4,
          soft_cap=15.0,
          mask_value=None,
          q_scale=None,
          k_scale=None,
          v_scale=0.5,
          skip_kv_update=True,
          use_sink=True,
          quantize_p=True,
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
      mask_value,
      q_scale,
      k_scale,
      v_scale,
      skip_kv_update,
      use_sink,
      quantize_p,
  ):
    max_num_seqs = len(kv_lens)
    page_size = 8
    pages_per_seq = (max(kv_lens) + page_size - 1) // page_size
    total_pages = max_num_seqs * pages_per_seq
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    head_dim_aligned = 128

    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(42), 5)
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
    attention_sink = (
        jax.random.normal(k5, (num_q_heads,), dtype=jnp.float32)
        if use_sink
        else None
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
        attention_sink=attention_sink,
        use_causal_mask=use_causal_mask,
        sm_scale=0.125,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        skip_kv_update=skip_kv_update,
        quantize_p=quantize_p,
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
            attention_sink=attention_sink,
            use_causal_mask=use_causal_mask,
            sm_scale=0.125,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            mask_value=mask_value,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            skip_kv_update=skip_kv_update,
            quantize_p=quantize_p,
            return_lse=True,
        )
    )

    self.assertTrue(jnp.array_equal(actual_kv, expected_kv))
    self.assertTrue(
        jnp.allclose(actual_out, expected_out, atol=1e-2, rtol=1e-2)
    )
    self.assertEqual(actual_lse.dtype, expected_lse.dtype)
    active_limit = int(cu_q_lens[active_seqs])
    self.assertTrue(
        jnp.allclose(
            actual_lse[:active_limit],
            expected_lse[:active_limit],
            atol=1e-2,
            rtol=1e-2,
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
