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
"""Tests for Pallas Mosaic TPU Batched RPA."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.batched_rpa import pallas_mosaic_tpu as jax_pallas_mosaic_tpu
from tokamax.experimental.torch_tpu.ops.batched_rpa import torch_base
from tokamax.experimental.torch_tpu.ops.batched_rpa import torch_pallas_mosaic_tpu
import torch
import torch_tpu


class PallasMosaicTpuBatchedRpaTest(parameterized.TestCase):

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
    device = "tpu"
    torch.manual_seed(0)

    num_seqs = len(seq_lens)
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 128
    head_dim_aligned = 128

    total_q_tokens = sum(q_lens)
    cu_q_lens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    for i, ql in enumerate(q_lens):
      cu_q_lens[i + 1] = cu_q_lens[i] + ql

    pages_per_seq = max((sl + page_size - 1) // page_size for sl in seq_lens)
    total_pages = num_seqs * pages_per_seq

    page_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    distribution = torch.tensor(distribution, dtype=torch.int32, device=device)

    queries = torch.randn(
        (total_q_tokens, num_q_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    keys = torch.randn(
        (total_q_tokens, num_kv_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    values = torch.randn(
        (total_q_tokens, num_kv_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = torch.randn(
        (total_pages, page_size, num_kv_heads * 2, head_dim_aligned),
        dtype=torch.bfloat16,
        device=device,
    )

    jax_queries = jnp.array(
        queries.detach().cpu().to(torch.float32).numpy(), dtype=jnp.bfloat16
    )
    jax_keys = jnp.array(
        keys.detach().cpu().to(torch.float32).numpy(), dtype=jnp.bfloat16
    )
    jax_values = jnp.array(
        values.detach().cpu().to(torch.float32).numpy(), dtype=jnp.bfloat16
    )
    jax_kv_cache = jnp.array(
        kv_cache.detach().cpu().to(torch.float32).numpy(), dtype=jnp.bfloat16
    )
    jax_kv_lens = jnp.array(kv_lens.detach().cpu().numpy(), dtype=jnp.int32)
    jax_page_indices = jnp.array(
        page_indices.detach().cpu().numpy(), dtype=jnp.int32
    )
    jax_cu_q_lens = jnp.array(cu_q_lens.detach().cpu().numpy(), dtype=jnp.int32)
    jax_distribution = jnp.array(
        distribution.detach().cpu().numpy(), dtype=jnp.int32
    )

    op_jax_kernel = jax_pallas_mosaic_tpu.PallasTpuBatchedRpa()
    op_pallas = torch_pallas_mosaic_tpu.PallasMosaicTpuBatchedRpa

    ref_out, ref_kv = op_jax_kernel(
        queries=jax_queries,
        keys=jax_keys,
        values=jax_values,
        kv_cache=jax_kv_cache,
        kv_lens=jax_kv_lens,
        page_indices=jax_page_indices,
        cu_q_lens=jax_cu_q_lens,
        distribution=jax_distribution,
    )

    pallas_out, pallas_kv = op_pallas(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
    )

    ref_out_as_torch = torch.as_tensor(
        np.asarray(ref_out, dtype=np.float32),
        device=device,
        dtype=torch.bfloat16,
    )

    self.assertEqual(pallas_out.shape, ref_out.shape)
    self.assertEqual(pallas_kv.shape, ref_kv.shape)
    torch.testing.assert_close(
        pallas_out, ref_out_as_torch, atol=0.15, rtol=0.15
    )


if __name__ == "__main__":
  absltest.main()
