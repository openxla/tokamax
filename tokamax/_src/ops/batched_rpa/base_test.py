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
from tokamax._src.ops.batched_rpa import api
from tokamax._src.ops.batched_rpa import base
from tokamax._src.ops.batched_rpa import reference

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


class BatchedRpaBaseTest(parameterized.TestCase):

  def _callTestMethod(self, method):
    try:
      return method()
    except Exception:
      import sys, traceback

      tb = traceback.format_exc().replace("\n", " -- ")
      sys.stdout.write(f"\n::error title={self.id()}::{tb[:1500]}\n")
      sys.stderr.write(f"\n::error title={self.id()}::{tb[:1500]}\n")
      sys.stdout.flush()
      sys.stderr.flush()
      raise

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
    self.assertTrue(jnp.allclose(ref_out, base_out, atol=0.1, rtol=0.1))

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


if __name__ == "__main__":
  absltest.main()
