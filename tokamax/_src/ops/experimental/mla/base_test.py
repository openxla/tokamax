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
import jaxtyping
import numpy as np
from tokamax._src.ops.experimental.mla import base
from tokamax._src.ops.experimental.mla import reference
from tokamax._src.ops.experimental.mla import test_base
from tokamax._src.ops.experimental.mla import utils

jax.config.parse_flags_with_absl()


def _per_token_reference(
    ql_nope,
    q_pe,
    new_kv_c,
    new_k_pe,
    cache_kv,
    seq_lens,
    page_indices,
    *,
    sm_scale,
):
  """Causal MLA, one query token at a time, straight from the inputs.

  Sequence i's keys are its `kv_len - q_len` cached tokens, read from its pages,
  followed by its `q_len` new tokens. Rows of padding tokens are zero.

  Args:
    ql_nope: [heads, tokens, lkv_dim].
    q_pe: [tokens, heads, r_dim].
    new_kv_c: [tokens, lkv_dim].
    new_k_pe: [tokens, r_dim].
    cache_kv: [pages, kv_dim, page_size].
    seq_lens: (q_len, kv_len) per sequence.
    page_indices: [num_seqs * pages_per_seq].
    sm_scale: Softmax scale.

  Returns:
    [heads, tokens, align_to(lkv_dim, 128)].
  """
  kv_dim = cache_kv.shape[1]
  cache = cache_kv.transpose(0, 2, 1)  # [pages, page_size, kv_dim]
  lkv_dim, r_dim = new_kv_c.shape[1], new_k_pe.shape[1]
  aligned_lkv = utils.align_to(lkv_dim, 128)
  pages_per_seq = page_indices.shape[0] // len(seq_lens)

  out = np.zeros(ql_nope.shape[:2] + (aligned_lkv,), np.float32)
  q_start = 0
  for i, (q_len, kv_len) in enumerate(seq_lens):
    num_cached = kv_len - q_len
    pages = page_indices[i * pages_per_seq : (i + 1) * pages_per_seq]
    cached = cache[pages[pages >= 0]].reshape(-1, kv_dim)[:num_cached]
    c_kv = np.concatenate(
        [cached[:, :lkv_dim], new_kv_c[q_start : q_start + q_len]]
    )
    k_pe = np.concatenate([
        cached[:, aligned_lkv : aligned_lkv + r_dim],
        new_k_pe[q_start : q_start + q_len],
    ])
    for j in range(q_len):
      t = q_start + j
      visible = num_cached + j + 1  # Causal.
      s = (
          ql_nope[:, t] @ c_kv[:visible].T + q_pe[t] @ k_pe[:visible].T
      ) * sm_scale
      p = np.exp(s - s.max(axis=-1, keepdims=True))
      p /= p.sum(axis=-1, keepdims=True)
      out[:, t, :lkv_dim] = p @ c_kv[:visible]
    q_start += q_len
  return out


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

    inputs = test_base.generate_mla_inputs(
        seq_lens_list,
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        q_dtype,
        kv_dtype,
        num_pages,
        rng=0,
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

    self.assertEqual(out.shape, (num_heads, total_q_len, output_lkv_dim))
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

  def test_token_major_cache_is_rejected(self):
    with self.assertRaises(jaxtyping.TypeCheckError):
      base.MultiHeadLatentAttention().bind(
          ql_nope=jnp.zeros((8, 10, 128)),
          q_pe=jnp.zeros((10, 8, 64)),
          new_kv_c=jnp.zeros((10, 128)),
          new_k_pe=jnp.zeros((10, 64)),
          cache_kv=jnp.zeros((2, 16, 4, 256)),
          kv_lens=jnp.zeros((2,), dtype=jnp.int32),
          page_indices=jnp.zeros((2,), dtype=jnp.int32),
          cu_q_lens=jnp.zeros((3,), dtype=jnp.int32),
          distribution=jnp.zeros((3,), dtype=jnp.int32),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="mixed",
          seq_lens_list=[(1, 300), (7, 513), (1, 18), (40, 40), (3, 1000)],
      ),
      # `q_len` above the 256-row chunk, so a sequence spans several chunks.
      dict(testcase_name="long_chunk", seq_lens_list=[(300, 700), (5, 9)]),
      # A sequence with no new tokens between two that have some.
      dict(testcase_name="empty_seq", seq_lens_list=[(4, 100), (0, 0), (2, 3)]),
  )
  def test_ragged_q_lens(self, seq_lens_list):
    page_size, num_heads, lkv_dim, r_dim = 128, 4, 128, 64
    total_kv_tokens = sum(s[1] for s in seq_lens_list)
    num_pages = utils.cdiv(total_kv_tokens, page_size) + len(seq_lens_list)
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
        seq_lens_list,
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        jnp.float32,
        jnp.float32,
        num_pages,
        rng=0,
    )
    sm_scale = 0.1
    expected = _per_token_reference(
        np.asarray(ql_nope),
        np.asarray(q_pe),
        np.asarray(new_kv_c),
        np.asarray(new_k_pe),
        np.asarray(cache_kv),
        seq_lens_list,
        np.asarray(page_indices),
        sm_scale=sm_scale,
    )

    with jax.default_matmul_precision("highest"):
      out, _ = base.MultiHeadLatentAttention()(
          ql_nope,
          q_pe,
          new_kv_c,
          new_k_pe,
          cache_kv,
          kv_lens,
          page_indices,
          cu_q_lens,
          distribution,
          sm_scale=sm_scale,
          s_dtype=jnp.float32,
      )

    self.assertEqual(out.shape, expected.shape)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

  def test_shape_mismatch_validation(self):
    baseline_op = base.MultiHeadLatentAttention()

    # Mismatched `actual_num_q_heads`:
    # ql_nope: [actual_num_q_heads, max_num_tokens, actual_lkv_dim]
    # q_pe: [max_num_tokens, actual_num_q_heads, actual_r_dim]
    ql_nope = jnp.zeros((5, 10, 128))
    q_pe = jnp.zeros((10, 6, 64))  # Mismatch here: 6 != 5

    new_kv_c = jnp.zeros((10, 128))
    new_k_pe = jnp.zeros((10, 64))
    cache_kv = jnp.zeros((2, 256, 64))
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

    ql_nope = jnp.zeros((8, 10, 128))
    q_pe = jnp.zeros((10, 8, 64))
    new_kv_c = jnp.zeros((10, 128))
    new_k_pe = jnp.zeros((10, 64))
    cache_kv = jnp.zeros((2, 256, 64))

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

    ql_nope = jnp.zeros((8, 10, 128))
    q_pe = jnp.zeros((10, 8, 64))
    new_kv_c = jnp.zeros((10, 128))
    new_k_pe = jnp.zeros((10, 64))
    cache_kv = jnp.zeros((2, 256, 64))

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
