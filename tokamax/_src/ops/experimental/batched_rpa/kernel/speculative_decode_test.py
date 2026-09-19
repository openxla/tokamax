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
"""Widened RPAd buckets must respect runtime lengths and preserve paged KV."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.batched_rpa.kernel import configs
from tokamax._src.ops.experimental.batched_rpa.kernel import wrapper


class SpeculativeDecodeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    try:
      if not pltpu.get_tpu_info().generation >= 5:
        self.skipTest("Pallas TPU kernel requires TPU v5 or newer.")
    except (ValueError, RuntimeError, AttributeError):
      self.skipTest("Failed to get TPU info.")

  @parameterized.named_parameters(
      ("bucket_4_baseline", 4, (1, 3, 3, 16), (1025, 1025, 4, 1030), False),
      ("bucket_4_widened", 4, (1, 3, 3, 16), (1025, 1025, 4, 1030), True),
      (
          "bucket_8_baseline",
          8,
          (1, 5, 7, 5, 16),
          (4097, 4098, 4099, 6, 4102),
          False,
      ),
      (
          "bucket_8_widened",
          8,
          (1, 5, 7, 5, 16),
          (4097, 4098, 4099, 6, 4102),
          True,
      ),
  )
  def test_ragged_verify_matches_reference_and_preserves_cache(
      self, bucket, q_lens, kv_lengths, widened
  ):
    """Catch padded-query leakage, cross-page writeback and mixed-stage damage.

    The last request stays in MIXED. Verification uses MIXED in the baseline
    and shares a wider DECODE bucket with q1 in the candidate. Unused
    requests and physical pages are sentinels. A short-history verification
    makes missing causal masks visible instead of diluted by 4K history.
    """
    rng = np.random.default_rng(994)
    num_q_heads, head_dim, page_size = 16, 256, 256
    max_seqs = 8
    pages_per_seq = (max(kv_lengths) + page_size - 1) // page_size + 1
    num_seqs = len(q_lens)
    total_tokens = sum(q_lens)
    layout = configs.KVLayout.SEQ_ALONG_LANE
    cache_shape = wrapper.get_kv_cache_shape(
        max_seqs * pages_per_seq + 2,
        page_size,
        1,
        head_dim,
        jnp.float8_e4m3fn,
        kv_layout=layout,
    )

    def random_array(shape, dtype):
      return (rng.standard_normal(shape) * 0.5).astype(dtype)

    q = random_array((total_tokens, num_q_heads, head_dim), jnp.bfloat16)
    k = random_array((total_tokens, 1, head_dim), jnp.float8_e4m3fn)
    v = random_array((total_tokens, 1, head_dim), jnp.float8_e4m3fn)
    before = random_array(cache_shape, jnp.float8_e4m3fn)
    expected_cache = before.copy()
    page_table = rng.permutation(cache_shape[0])[: max_seqs * pages_per_seq]
    page_table = page_table.reshape(max_seqs, pages_per_seq).astype(np.int32)
    kv_lens = np.zeros(max_seqs, dtype=np.int32)
    kv_lens[:num_seqs] = kv_lengths
    cu_q_lens = np.full(max_seqs + 1, total_tokens, dtype=np.int32)
    cu_q_lens[: num_seqs + 1] = np.cumsum((0, *q_lens))
    reference = np.empty(q.shape, dtype=np.float32)
    protected = np.ones((cache_shape[0], page_size), dtype=bool)

    for seq, (q_len, kv_len) in enumerate(zip(q_lens, kv_lengths)):
      start, end = cu_q_lens[seq : seq + 2]
      history_len = kv_len - q_len
      # HND writes whole pages. Beyond kv_len, the partially filled final
      # page is scratch, not live KV. Every other location is protected.
      tail = kv_len % page_size
      if tail:
        protected[page_table[seq, kv_len // page_size], tail:] = False
      for token, pos in enumerate(range(history_len, kv_len), start):
        page = page_table[seq, pos // page_size]
        lane = pos % page_size
        expected_cache[page, :, :, :, lane] = np.stack(
            (k[token, 0], v[token, 0])
        ).reshape(2, head_dim // 4, 4)

      # Independently gather logical K/V from the full expected cache.
      logical_kv = np.stack([
          expected_cache[
              page_table[seq, pos // page_size],
              :,
              :,
              :,
              pos % page_size,
          ].reshape(2, head_dim)
          for pos in range(kv_len)
      ]).astype(np.float32)
      scores = (
          np.einsum(
              "qhd,kd->qhk",
              q[start:end].astype(np.float32),
              logical_kv[:, 0],
          )
          * head_dim**-0.5
      )
      visible = (
          np.arange(kv_len)[None, :] <= history_len + np.arange(q_len)[:, None]
      )
      scores = np.where(visible[:, None, :], scores, -np.inf)
      weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
      weights /= weights.sum(axis=-1, keepdims=True)
      reference[start:end] = np.einsum("qhk,kd->qhd", weights, logical_kv[:, 1])

    decode_end = num_seqs - 1 if widened else 1
    output, cache = jax.block_until_ready(
        wrapper.ragged_paged_attention(
            jnp.asarray(q),
            jnp.asarray(k),
            jnp.asarray(v),
            jnp.asarray(before),
            jnp.asarray(kv_lens),
            jnp.asarray(page_table.ravel()),
            jnp.asarray(cu_q_lens),
            jnp.asarray([decode_end, decode_end, num_seqs], jnp.int32),
            sm_scale=head_dim**-0.5,
            kv_layout=layout,
            decode_query_size=bucket if widened else 1,
        )
    )
    actual = np.asarray(output.astype(jnp.float32))
    self.assertTrue(np.isfinite(actual).all())
    np.testing.assert_allclose(actual, reference, atol=3e-2, rtol=2e-2)
    for seq in range(num_seqs):
      start, end = cu_q_lens[seq : seq + 2]
      relative_l2 = np.linalg.norm(
          actual[start:end] - reference[start:end]
      ) / np.linalg.norm(reference[start:end])
      self.assertLess(
          relative_l2, 0.02, f"seq {seq}: relative L2 = {relative_l2}"
      )
    # Protect all history, new destinations (on both sides of a page),
    # inactive requests, untouched pages and free pages. Only tail scratch
    # in each active request's last page is excluded.
    np.testing.assert_array_equal(
        np.asarray(cache).transpose(0, 4, 1, 2, 3)[protected],
        expected_cache.transpose(0, 4, 1, 2, 3)[protected],
    )


if __name__ == "__main__":
  absltest.main()
