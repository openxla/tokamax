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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa import api
from tokamax._src.ops.batched_rpa import base
from tokamax._src.ops.batched_rpa import pallas_mosaic_tpu
from tokamax._src.ops.batched_rpa import reference
from tokamax._src.ops.batched_rpa.kernel import configs
from tokamax._src.ops.batched_rpa.kernel import wrapper

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

  def test_decode_only_matches_standard_on_decode_batch(self):
    total_q_tokens = 2
    page_size = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    total_pages = 4
    head_dim_aligned = 128

    k1, k2, k3 = jax.random.split(jax.random.key(31), 3)
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
    kv_lens = jnp.array([6, 8], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 1, 2], dtype=jnp.int32)
    distribution = jnp.array([2, 2, 2], dtype=jnp.int32)

    ref_out, ref_cache = (  # pyrefly: ignore[bad-unpacking]
        api.batched_ragged_paged_attention(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            decode_only=False,
            implementation="reference",
        )
    )
    dec_out, dec_cache = (  # pyrefly: ignore[bad-unpacking]
        api.batched_ragged_paged_attention(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            decode_only=True,
            implementation="reference",
        )
    )
    self.assertTrue(jnp.array_equal(dec_out, ref_out))
    self.assertTrue(jnp.array_equal(dec_cache, ref_cache))

  @parameterized.named_parameters(
      ("default", None, False, [configs.RpaCase.DECODE, configs.RpaCase.MIXED]),
      (
          "standard",
          False,
          False,
          [configs.RpaCase.DECODE, configs.RpaCase.MIXED],
      ),
      ("decode_only", True, False, [configs.RpaCase.DECODE]),
      ("decode_only_with_lse", True, True, [configs.RpaCase.DECODE]),
  )
  def test_decode_only_skips_mixed_kernel(
      self, decode_only, return_lse, expected_modes
  ):
    self.assertIsInstance(
        api.IMPLEMENTATIONS["mosaic_tpu"], pallas_mosaic_tpu.PallasTpuBatchedRpa
    )
    modes = []

    def _fake_schedule(*args, **kwargs):
      del args, kwargs
      return jnp.zeros((1,), dtype=jnp.int32)

    def _fake_kernel(*args, cfgs, **kwargs):
      del kwargs
      modes.append(cfgs.mode)
      return args[1], args[3], args[4]

    queries = jnp.zeros((2, 4, 64), dtype=jnp.bfloat16)
    keys = jnp.zeros((2, 2, 64), dtype=jnp.bfloat16)
    values = jnp.zeros((2, 2, 64), dtype=jnp.bfloat16)
    kv_cache = jnp.zeros((4, 8, 2, 2, 128), dtype=jnp.bfloat16)
    kv_lens = jnp.array([4, 6], dtype=jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 1, 2], dtype=jnp.int32)
    distribution = jnp.array([2, 2, 2], dtype=jnp.int32)

    kwargs = {} if decode_only is None else {"decode_only": decode_only}
    tpu_info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    jax.clear_caches()

    with (
        mock.patch.object(
            pallas_mosaic_tpu.PallasTpuBatchedRpa,
            "supported_on",
            autospec=True,
            return_value=True,
        ),
        mock.patch.object(
            pltpu,
            "get_tpu_info",
            autospec=True,
            return_value=tpu_info,
        ),
        mock.patch.object(
            wrapper.scheduler_lib,
            "generate_rpa_metadata",
            side_effect=_fake_schedule,
        ),
        mock.patch.object(
            wrapper.schedule,
            "smem_prefetch_start",
            return_value=None,
        ),
        mock.patch.object(
            wrapper.kernel, "rpa_kernel", side_effect=_fake_kernel
        ),
    ):
      result = api.batched_ragged_paged_attention(
          queries=queries,
          keys=keys,
          values=values,
          kv_cache=kv_cache,
          kv_lens=kv_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=distribution,
          return_lse=return_lse,
          implementation="mosaic_tpu",
          **kwargs,
      )
    out, new_cache, *extra = result
    self.assertEqual(out.shape, queries.shape)
    self.assertEqual(new_cache.shape, kv_cache.shape)
    if return_lse:
      self.assertLen(extra, 1)
      self.assertEqual(extra[0].shape, (queries.shape[0], queries.shape[1]))
    else:
      self.assertEmpty(extra)
    self.assertEqual(modes, expected_modes)


if __name__ == "__main__":
  absltest.main()
