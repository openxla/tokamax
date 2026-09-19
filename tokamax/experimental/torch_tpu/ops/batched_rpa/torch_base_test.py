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
import numpy as np
from tokamax._src.ops.experimental.batched_rpa import base as jax_base
from tokamax.experimental.torch_tpu.ops.batched_rpa import torch_base
import torch
import torch_tpu
import torch_tpu._internal.pallas.pallas


class BaseTest(parameterized.TestCase):

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

    torch.manual_seed(0)
    device = "tpu"

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
    kv_cache = torch.zeros(
        (total_pages, page_size, num_kv_heads * 2, head_dim_aligned),
        dtype=torch.bfloat16,
        device=device,
    )
    kv_lens = torch.tensor([8, 8], dtype=torch.int32, device=device)
    page_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    cu_q_lens = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    distribution = torch.tensor([0, 0, 2], dtype=torch.int32, device=device)

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

    ref_out, ref_kv = jax_base.BatchedRpa()(
        queries=jax_queries,
        keys=jax_keys,
        values=jax_values,
        kv_cache=jax_kv_cache,
        kv_lens=jax_kv_lens,
        page_indices=jax_page_indices,
        cu_q_lens=jax_cu_q_lens,
        distribution=jax_distribution,
    )

    op = torch_base.BatchedRpa
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
    ref_out_as_torch = torch.as_tensor(
        np.asarray(ref_out, dtype=np.float32),
        device="tpu",
        dtype=torch.bfloat16,
    )
    self.assertTrue(torch.allclose(ref_out_as_torch, base_out, atol=1e-3))


if __name__ == "__main__":
  absltest.main()
