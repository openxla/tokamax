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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa import base
from tokamax._src.ops.batched_rpa import reference


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

  def test_scale_dequantization(self):
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
    kv_lens = jnp.array([8, 8], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

    out_default, _ = reference.batched_ragged_paged_attention_reference(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        mask_value=None,
    )
    out_custom, _ = reference.batched_ragged_paged_attention_reference(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        mask_value=-1e4,
    )
    self.assertEqual(out_custom.shape, out_default.shape)
    self.assertTrue(jnp.allclose(out_custom, out_default, atol=1e-3))

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

    queries_f32 = queries_bf16.astype(jnp.float32)
    keys_f32 = keys_bf16.astype(jnp.float32)
    values_f32 = values_bf16.astype(jnp.float32)
    kv_cache_f32 = kv_cache_bf16.astype(jnp.float32)

    _, _, lse_f32 = reference.batched_ragged_paged_attention_reference(
        queries=queries_f32,
        keys=keys_f32,
        values=values_f32,
        kv_cache=kv_cache_f32,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        return_lse=True,
    )
    self.assertEqual(lse_f32.dtype, jnp.float32)
    self.assertEqual(lse_f32.shape, (total_q_tokens, num_q_heads))


if __name__ == "__main__":
  absltest.main()
