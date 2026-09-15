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

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.batched_rpa import base
from tokamax._src.ops.experimental.batched_rpa import pallas_mosaic_tpu
from tokamax._src.ops.experimental.batched_rpa import reference


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
        (total_pages, page_size, num_kv_heads * 2, head_dim_aligned),
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
    chex.assert_trees_all_close(pallas_out, ref_out, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
  absltest.main()
