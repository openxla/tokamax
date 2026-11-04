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
"""Tests for Multi-Head Latent Attention (MLA) V2 kernel."""

import gc
import sys
from absl import logging
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.mla import test_base
from tokamax._src.ops.experimental.mla.v2 import kernel as kernel_v2


jax.config.parse_flags_with_absl()
jax.config.update("jax_numpy_dtype_promotion", "standard")


class MlaRaggedPagedAttentionKernelV2Test(
    test_base.MlaRaggedPagedAttentionTestBase
):
  mla_module = kernel_v2

  def _test_mla_ragged_paged_attention(
      self,
      seq_lens,
      num_heads,
      lkv_dim,
      r_dim,
      page_size,
      q_dtype,
      kv_dtype,
      num_pages,
      *,
      num_kv_pages_per_block=8,
      num_queries_per_block=8,
      vmem_limit_bytes=100 * 1024 * 1024,
      sm_scale=1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
  ):
    if not jax.devices() or jax.devices()[0].platform != "tpu":
      self.skipTest("Expect TPU")
    rng = np.random.default_rng(1234)

    (
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ) = test_base.generate_mla_inputs(
        seq_lens,
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        q_dtype,
        kv_dtype,
        num_pages,
        rng=rng,
    )

    padded_r_dim = test_base.align_to(r_dim, 128)
    padded_lkv_dim = test_base.align_to(lkv_dim, 128)
    padded_kv_dim = padded_lkv_dim + padded_r_dim
    packing = test_base.get_dtype_packing(kv_dtype)
    total_q_len = sum(s[0] for s in seq_lens)
    kv_lens_list = [s[1] for s in seq_lens]
    max_kv_len = max(kv_lens_list) if kv_lens_list else 0
    total_num_pages = max(
        num_pages,
        sum(test_base.cdiv(kv_len, page_size) for kv_len in kv_lens_list),
    )

    expected_out, expected_updated_kv = (
        kernel_v2.ref_mla_ragged_paged_attention(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv.copy(),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    )

    kernel_out, kernel_updated_kv = kernel_v2.mla_ragged_paged_attention(
        jnp.transpose(ql_nope, (1, 0, 2)),
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        s_dtype=jnp.float32,
        decode_batch_size=4,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=test_base.DEBUG_MODE.value,
    )
    kernel_out = jnp.transpose(kernel_out, (1, 0, 2))
    with np.printoptions(threshold=sys.maxsize):
      logging.vlog(2, "new_kv_c: %s", new_kv_c)
      logging.vlog(2, "new_k_pe: %s", new_k_pe)
      logging.vlog(
          2, "expected_updated_kv.shape: %s", expected_updated_kv.shape
      )
      logging.vlog(
          2, "expected_updated_kv[..., 0]: %s", expected_updated_kv[..., 0]
      )
      logging.vlog(2, "kernel_updated_kv.shape: %s", kernel_updated_kv.shape)
      logging.vlog(
          2, "kernel_updated_kv[..., 0]: %s", kernel_updated_kv[..., 0]
      )

    self.assertEqual(
        expected_out.shape, (total_q_len, num_heads, padded_lkv_dim)
    )
    self.assertEqual(
        expected_updated_kv.shape,
        (total_num_pages, page_size // packing, packing, padded_kv_dim),
    )
    self.assertEqual(expected_out.dtype, q_dtype)
    self.assertEqual(expected_updated_kv.dtype, kv_dtype)

    mask = np.zeros_like(expected_updated_kv, dtype=np.bool_)
    pages_per_seq = test_base.cdiv(max_kv_len, page_size)
    for i, kv_len in enumerate(kv_lens_list):
      start_page_idx_in_pages_list = i * pages_per_seq
      num_pages_for_seq = test_base.cdiv(kv_len, page_size)
      for j in range(num_pages_for_seq):
        page_idx = page_indices[start_page_idx_in_pages_list + j]
        if page_idx == -1:
          logging.warning(
              "Sequence %d page %d has invalid page index -1.", i, j
          )
          continue

        is_last_page = j == num_pages_for_seq - 1
        tokens_on_this_page = (
            kv_len % page_size
            if is_last_page and kv_len % page_size != 0
            else page_size
        )

        for token_idx_in_page in range(tokens_on_this_page):
          row = token_idx_in_page // packing
          col = token_idx_in_page % packing
          mask[page_idx, row, col, :] = True
    true_count = np.sum(mask)
    logging.info("Number of True values in KV checker mask: %d", true_count)
    self.assertEqual(true_count, sum(kv_lens_list) * padded_kv_dim)
    expected_valid = np.array(expected_updated_kv)[mask]
    kernel_valid = np.array(kernel_updated_kv)[mask]
    np.testing.assert_array_equal(
        expected_valid,
        kernel_valid,
        err_msg="Updated KV cache mismatch",
    )

    np.testing.assert_allclose(expected_out, kernel_out, atol=0.1, rtol=0.2)
    gc.collect()

  def test_get_kv_cache_shape(self):
    total_num_pages = 10
    page_size = 16
    lkv_dim = 128
    kv_dtype = jnp.bfloat16
    expected_shape = (10, 8, 2, 128)
    self.assertEqual(
        self.mla_module.get_kv_cache_shape(
            total_num_pages, page_size, lkv_dim, kv_dtype
        ),
        expected_shape,
    )


if __name__ == "__main__":
  absltest.main()
