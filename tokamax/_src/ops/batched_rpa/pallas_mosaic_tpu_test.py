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
"""Tests for Batched RPA Pallas Mosaic TPU operator."""

import os

_LIBTPU_FLAG = "--xla_tpu_use_dynamic_smem_negotiation=true"
if _LIBTPU_FLAG not in os.environ.get("LIBTPU_INIT_ARGS", ""):
  os.environ["LIBTPU_INIT_ARGS"] = (
      os.environ.get("LIBTPU_INIT_ARGS", "") + f" {_LIBTPU_FLAG}"
  ).strip()

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.batched_rpa import base
from tokamax._src.ops.batched_rpa import pallas_mosaic_tpu
from tokamax._src.ops.batched_rpa import reference

jax.config.parse_flags_with_absl()


class PallasMosaicTpuBatchedRpaTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    try:
      if not pltpu.get_tpu_info().generation >= 5:
        self.skipTest("Pallas TPU kernel requires TPU v5 or newer.")
    except Exception:
      self.skipTest("Failed to get TPU info.")

  @parameterized.named_parameters(
      dict(
          testcase_name="decode_only",
          seq_lens=[128, 128, 128, 128],
          q_lens=[1, 1, 1, 1],
          distribution=[4, 4, 4],
          page_size=128,
      ),
      dict(
          testcase_name="prefill_only",
          seq_lens=[128, 128],
          q_lens=[128, 128],
          distribution=[0, 2, 2],
          page_size=128,
      ),
      dict(
          testcase_name="mixed_prefill_decode",
          seq_lens=[128, 128, 128],
          q_lens=[1, 1, 128],
          distribution=[2, 3, 3],
          page_size=128,
      ),
  )
  def test_pallas_matches_reference(
      self, seq_lens, q_lens, distribution, page_size
  ):
    num_seqs = len(seq_lens)
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 128
    head_dim_aligned = 128

    total_q_tokens = sum(q_lens)
    cu_q_lens = np.zeros(num_seqs + 1, dtype=np.int32)
    for i, ql in enumerate(q_lens):
      cu_q_lens[i + 1] = cu_q_lens[i] + ql

    pages_per_seq = max((sl + page_size - 1) // page_size for sl in seq_lens)
    total_pages = num_seqs * pages_per_seq

    page_indices = np.arange(total_pages, dtype=np.int32)
    kv_lens = np.array(seq_lens, dtype=np.int32)
    distribution = np.array(distribution, dtype=np.int32)

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
    kv_cache = jax.random.normal(
        k4,
        (total_pages, page_size, num_kv_heads, 2, head_dim_aligned),
        dtype=jnp.bfloat16,
    )

    cu_q_lens_jax = jnp.array(cu_q_lens)
    kv_lens_jax = jnp.array(kv_lens)
    page_indices_jax = jnp.array(page_indices)
    distribution_jax = jnp.array(distribution)

    op_base = base.BatchedRpa()
    op_pallas = pallas_mosaic_tpu.PallasTpuBatchedRpa()

    ref_out, ref_kv = op_base(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens_jax,
        page_indices=page_indices_jax,
        cu_q_lens=cu_q_lens_jax,
        distribution=distribution_jax,
    )

    pallas_out, pallas_kv = op_pallas(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens_jax,
        page_indices=page_indices_jax,
        cu_q_lens=cu_q_lens_jax,
        distribution=distribution_jax,
    )

    self.assertEqual(pallas_out.shape, ref_out.shape)
    chex.assert_trees_all_close(pallas_out, ref_out, atol=0.15, rtol=0.15)

  def test_pallas_attention_sink_and_decode_query_size(self):
    seq_lens = [128, 128]
    q_lens = [2, 2]
    distribution = [2, 2, 2]
    page_size = 128
    decode_query_size = 2
    num_seqs = len(seq_lens)
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 128
    head_dim_aligned = 128

    total_q_tokens = sum(q_lens)
    cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
    total_pages = num_seqs
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)
    kv_lens = jnp.array(seq_lens, dtype=jnp.int32)
    distribution_jax = jnp.array(distribution, dtype=jnp.int32)

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
    attention_sink = jax.random.normal(k5, (num_q_heads,), dtype=jnp.float32)

    op_base = base.BatchedRpa()
    op_pallas = pallas_mosaic_tpu.PallasTpuBatchedRpa()

    ref_out, _ = op_base(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution_jax,
        attention_sink=attention_sink,
        decode_query_size=decode_query_size,
    )
    pallas_out, _ = op_pallas(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution_jax,
        attention_sink=attention_sink,
        decode_query_size=decode_query_size,
    )

    self.assertEqual(pallas_out.shape, ref_out.shape)
    chex.assert_trees_all_close(pallas_out, ref_out, atol=0.15, rtol=0.15)

  def test_pallas_quantize_p_fp8_matches_reference(self):
    seq_lens = [128, 128]
    q_lens = [1, 128]
    distribution = [1, 2, 2]
    page_size = 128
    num_seqs = len(seq_lens)
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 128
    head_dim_aligned = 128
    kv_packing = 4

    total_q_tokens = sum(q_lens)
    cu_q_lens = jnp.array([0, 1, 129], dtype=jnp.int32)
    pages_per_seq = max(
        (seq_len + page_size - 1) // page_size for seq_len in seq_lens
    )
    total_pages = num_seqs * pages_per_seq
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)
    kv_lens = jnp.array(seq_lens, dtype=jnp.int32)
    distribution_jax = jnp.array(distribution, dtype=jnp.int32)

    key1, key2, key3, key4 = jax.random.split(jax.random.key(77), 4)
    queries = jax.random.normal(
        key1, (total_q_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16
    )
    keys = jax.random.normal(
        key2, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16
    ).astype(jnp.float8_e4m3fn)
    values = jax.random.normal(
        key3, (total_q_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16
    ).astype(jnp.float8_e4m3fn)
    kv_cache = jax.random.normal(
        key4,
        (
            total_pages,
            page_size,
            (num_kv_heads * 2) // kv_packing,
            kv_packing,
            head_dim_aligned,
        ),
        dtype=jnp.bfloat16,
    ).astype(jnp.float8_e4m3fn)

    op_base = base.BatchedRpa()
    op_pallas = pallas_mosaic_tpu.PallasTpuBatchedRpa()

    ref_out, _ = op_base(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution_jax,
        quantize_p=True,
    )
    pallas_out, _ = op_pallas(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution_jax,
        quantize_p=True,
    )

    self.assertEqual(pallas_out.shape, ref_out.shape)
    # Both kernel and reference cast unnormalized p = exp(scores - m) to
    # float8_e4m3fn before the PV matmul and divide by f32 rowsum l; 0.15
    # matches the standard bf16 Pallas-vs-reference tolerance in this suite.
    chex.assert_trees_all_close(pallas_out, ref_out, atol=0.15, rtol=0.15)


class PallasMosaicTpuSchedulerContractTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  def test_cp_group_size_without_cp_rank_raises(self):
    with self.assertRaisesRegex(
        ValueError, "cp_rank must be provided when cp_group_size is set"
    ):
      pallas_mosaic_tpu._build_scheduler_and_extra_scalars(
          cp_group_size=2,
          cp_rank=None,
          attention_scope=base.AttentionScope.CACHE_ONLY,
      )

  def test_cp_rank_without_cp_group_size_raises(self):
    with self.assertRaisesRegex(
        ValueError, "cp_group_size must be set when cp_rank is provided"
    ):
      pallas_mosaic_tpu._build_scheduler_and_extra_scalars(
          cp_group_size=None,
          cp_rank=jnp.int32(0),
          attention_scope=base.AttentionScope.FULL,
      )

  def test_attention_scope_without_cp_group_size_raises(self):
    with self.assertRaisesRegex(ValueError, "requires cp_group_size to be set"):
      pallas_mosaic_tpu._build_scheduler_and_extra_scalars(
          cp_group_size=None,
          cp_rank=None,
          attention_scope=base.AttentionScope.CACHE_ONLY,
      )

  def test_scalar_0d_cp_rank_reshaped_to_1d(self):
    scheduler, extra_scalars = (
        pallas_mosaic_tpu._build_scheduler_and_extra_scalars(
            cp_group_size=2,
            cp_rank=jnp.int32(1),
            attention_scope=base.AttentionScope.CACHE_ONLY,
        )
    )
    self.assertEqual(type(scheduler).__name__, "StripedCacheScheduler")
    self.assertEqual(len(extra_scalars), 1)
    self.assertEqual(extra_scalars[0].shape, (1,))


if __name__ == "__main__":
  absltest.main()
