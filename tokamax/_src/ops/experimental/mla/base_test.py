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

from tokamax._src.mosaic_tpu import pl_align_to, pl_cdiv
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
    num_pages = pl_cdiv(total_kv_tokens, page_size) + len(seq_lens_list)
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

  def test_shape_mismatch_validation(self):
    baseline_op = base.MultiHeadLatentAttention()

    # Create inputs with mismatched shapes for dimension 'actual_num_q_heads'
    # ql_nope: [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
    # q_pe: [max_num_tokens, actual_num_q_heads, actual_r_dim]
    # We use different sizes for the second dimension (5 and 6).
    ql_nope = jnp.zeros((10, 5, 128))
    q_pe = jnp.zeros((10, 6, 64))  # Mismatch here: 6 != 5

    new_kv_c = jnp.zeros((10, 128))
    new_k_pe = jnp.zeros((10, 64))
    cache_kv = jnp.zeros((2, 16, 4, 128))
    kv_lens = jnp.zeros((2,), dtype=jnp.int32)
    page_indices = jnp.zeros((2,), dtype=jnp.int32)
    cu_q_lens = jnp.zeros((3,), dtype=jnp.int32)
    distribution = jnp.zeros((3,), dtype=jnp.int32)

    with self.assertRaises(Exception):
      baseline_op.bind(
          ql_nope=ql_nope,
          q_pe=q_pe,
          new_kv_c=new_kv_c,
          new_k_pe=new_k_pe,
          cache_kv=cache_kv,
          kv_lens=kv_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=distribution,
      )

  def test_page_indices_divisibility_validation(self):
    baseline_op = base.MultiHeadLatentAttention()

    ql_nope = jnp.zeros((10, 8, 128))
    q_pe = jnp.zeros((10, 8, 64))
    new_kv_c = jnp.zeros((10, 128))
    new_k_pe = jnp.zeros((10, 64))
    cache_kv = jnp.zeros((2, 16, 4, 128))

    # Mismatch: page_indices size 5 is not divisible by kv_lens size 2.
    kv_lens = jnp.zeros((2,), dtype=jnp.int32)
    page_indices = jnp.zeros((5,), dtype=jnp.int32)
    cu_q_lens = jnp.zeros((3,), dtype=jnp.int32)
    distribution = jnp.zeros((3,), dtype=jnp.int32)

    with self.assertRaisesRegex(ValueError, "divisible by"):
      baseline_op.bind(
          ql_nope=ql_nope,
          q_pe=q_pe,
          new_kv_c=new_kv_c,
          new_k_pe=new_k_pe,
          cache_kv=cache_kv,
          kv_lens=kv_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=distribution,
      )

  def test_cu_q_lens_shape_validation(self):
    baseline_op = base.MultiHeadLatentAttention()

    ql_nope = jnp.zeros((10, 8, 128))
    q_pe = jnp.zeros((10, 8, 64))
    new_kv_c = jnp.zeros((10, 128))
    new_k_pe = jnp.zeros((10, 64))
    cache_kv = jnp.zeros((2, 16, 4, 128))

    # Mismatch: cu_q_lens shape (4,) is not (3,) for kv_lens shape (2,)
    kv_lens = jnp.zeros((2,), dtype=jnp.int32)
    page_indices = jnp.zeros((4,), dtype=jnp.int32)
    cu_q_lens = jnp.zeros((4,), dtype=jnp.int32)
    distribution = jnp.zeros((3,), dtype=jnp.int32)

    with self.assertRaisesRegex(ValueError, "Expected cu_q_lens.shape"):
      baseline_op.bind(
          ql_nope=ql_nope,
          q_pe=q_pe,
          new_kv_c=new_kv_c,
          new_k_pe=new_k_pe,
          cache_kv=cache_kv,
          kv_lens=kv_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=distribution,
      )


if __name__ == "__main__":
  absltest.main()
