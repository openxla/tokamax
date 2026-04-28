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
"""Tests for the baseline JAX implementation of Multi-Head Latent Attention."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.mla import base
from tokamax._src.ops.experimental.mla import reference
from tokamax._src.ops.experimental.mla import utils

jax.config.parse_flags_with_absl()


class BaselineMlaTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="decode_only",
          seq_lens_list=[(1, 8192)] * 3,
          num_heads=128,
          lkv_dim=512,
          r_dim=64,
          page_size=256,
          q_dtype=jnp.float8_e4m3fn,
          kv_dtype=jnp.float8_e4m3fn,
          output_lkv_dim=512,
      ),
      dict(
          testcase_name="prefill_only",
          seq_lens_list=[(2048, 2048)] * 2,
          num_heads=128,
          lkv_dim=500,
          r_dim=64,
          page_size=256,
          q_dtype=jnp.bfloat16,
          kv_dtype=jnp.bfloat16,
          output_lkv_dim=512,  # 500 Padded to 512.
      ),
  )
  def test_reference_running_correctly(
      self,
      seq_lens_list,
      num_heads,
      lkv_dim,
      r_dim,
      page_size,
      q_dtype,
      kv_dtype,
      output_lkv_dim,
  ):
    total_q_len = sum(s[0] for s in seq_lens_list)
    total_kv_tokens = sum(s[1] for s in seq_lens_list)
    num_pages = utils.cdiv(total_kv_tokens, page_size) + len(seq_lens_list)
    mask_value = float(jnp.finfo(kv_dtype).min)

    inputs = utils.generate_mla_inputs(
        seq_lens_list,
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        q_dtype,
        kv_dtype,
        num_pages,
        rng=np.random.default_rng(
            int(np.array(jax.random.PRNGKey(0)).flatten()[0])
        ),
    )
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
    ) = inputs
    self.assertIsNotNone(cache_kv)
    baseline_op = base.MultiHeadLatentAttention()
    out, updated_kv = baseline_op(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        mask_value=mask_value,
    )

    self.assertEqual(out.shape, (total_q_len, num_heads, output_lkv_dim))
    self.assertEqual(updated_kv.shape, cache_kv.shape)

    out_ref, kv_ref = reference.mla_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        mask_value=mask_value,
    )

    np.testing.assert_allclose(out, out_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(updated_kv, kv_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
