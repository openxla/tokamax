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
"""Correctness tests for speculative decoding / MTP in batched ragged paged attention."""

import os

_LIBTPU_FLAG = "--xla_tpu_use_dynamic_smem_negotiation=true"
if _LIBTPU_FLAG not in os.environ.get("LIBTPU_INIT_ARGS", ""):
  os.environ["LIBTPU_INIT_ARGS"] = (
      os.environ.get("LIBTPU_INIT_ARGS", "") + f" {_LIBTPU_FLAG}"
  ).strip()

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import test_utils as jtu
from tokamax._src.ops.batched_rpa.kernel import configs
from tokamax._src.ops.batched_rpa.kernel import wrapper

jax.config.parse_flags_with_absl()


class CorrectnessMtpTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(version=4):
      self.skipTest("Requires TPU v4+")

  def _run_compare(
      self,
      histories: tuple[int, ...],
      bucket: int,
      *,
      page_size: int = 256,
      head_dim: int = 128,
      num_q_heads: int = 8,
      num_kv_heads: int = 1,
      q_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      kv_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      bkv_sz: int | None = None,
      seed: int = 7,
  ):
    max_seqs = 8
    num_seqs = len(histories)
    assert num_seqs <= max_seqs
    q_lens = (bucket,) * num_seqs
    kv_lengths = tuple(h + q for h, q in zip(histories, q_lens))
    total_tokens = sum(q_lens)
    pages_per_seq = (max(kv_lengths) + page_size - 1) // page_size + 1
    cache_shape = wrapper.get_kv_cache_shape(
        max_seqs * pages_per_seq + 2,
        page_size,
        num_kv_heads,
        head_dim,
        kv_dtype,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )

    rng = np.random.default_rng(seed)

    def random_array(shape, dtype):
      arr = (rng.standard_normal(shape) * 0.5).astype(np.float32)
      return jnp.asarray(arr).astype(dtype)

    q = random_array((total_tokens, num_q_heads, head_dim), q_dtype)
    k = random_array((total_tokens, num_kv_heads, head_dim), kv_dtype)
    v = random_array((total_tokens, num_kv_heads, head_dim), kv_dtype)
    before = random_array(cache_shape, kv_dtype)

    page_table = rng.permutation(cache_shape[0])[: max_seqs * pages_per_seq]
    page_table = page_table.reshape(max_seqs, pages_per_seq).astype(np.int32)
    kv_lens = np.zeros(max_seqs, dtype=np.int32)
    kv_lens[:num_seqs] = kv_lengths
    cu_q_lens = np.full(max_seqs + 1, total_tokens, dtype=np.int32)
    cu_q_lens[: num_seqs + 1] = np.cumsum((0, *q_lens))

    decode_block_sizes = None
    if bkv_sz is not None:
      decode_block_sizes = configs.BlockSizes(
          bq_sz=bucket,
          bq_c_sz=bucket,
          bkv_sz=bkv_sz,
          batch_size=1,
      )

    def run_case(distribution, decode_q_sz):
      out, new_cache = wrapper.ragged_paged_attention(
          q,
          k,
          v,
          before.copy(),
          jnp.asarray(kv_lens),
          jnp.asarray(page_table.ravel()),
          jnp.asarray(cu_q_lens),
          jnp.asarray(distribution, jnp.int32),
          sm_scale=head_dim**-0.5,
          kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
          decode_query_size=decode_q_sz,
          decode_block_sizes=decode_block_sizes,
      )
      out, new_cache = jax.block_until_ready((out, new_cache))
      return (
          np.asarray(out.astype(jnp.float32)),
          np.asarray(new_cache.astype(jnp.float32)),
      )

    decode_out, decode_cache = run_case(
        (num_seqs, num_seqs, num_seqs),
        bucket,
    )
    mixed_out, mixed_cache = run_case(
        (0, 0, num_seqs),
        1,
    )

    self.assertTrue(np.isfinite(decode_out).all())
    np.testing.assert_allclose(decode_out, mixed_out, atol=2e-2, rtol=0)

    for seq, (_, kv_len) in enumerate(zip(q_lens, kv_lengths)):
      for pos in range(kv_len):
        p = page_table[seq, pos // page_size]
        off = pos % page_size
        np.testing.assert_array_equal(
            decode_cache[p, :, :, :, off],
            mixed_cache[p, :, :, :, off],
            err_msg=f"seq {seq} pos {pos}: DECODE vs MIXED cache mismatch",
        )

  @parameterized.parameters(
      (1, 1, jnp.bfloat16),
      (2, 1, jnp.bfloat16),
      (4, 1, jnp.bfloat16),
      (8, 1, jnp.bfloat16),
      (16, 1, jnp.bfloat16),
      (64, 1, jnp.bfloat16),
      (128, 1, jnp.bfloat16),
      (1, 2, jnp.bfloat16),
      (4, 2, jnp.bfloat16),
      (1, 1, jnp.float8_e4m3fn),
      (4, 1, jnp.float8_e4m3fn),
      (16, 1, jnp.float8_e4m3fn),
  )
  def test_mtp_matches_mixed_prefill(self, bucket, num_kv_heads, kv_dtype):
    """MTP decode stitch must match MIXED prefill roll across buckets."""
    if kv_dtype == jnp.float8_e4m3fn and not jtu.is_device_tpu_at_least(
        version=5
    ):
      self.skipTest("FP8 requires TPU v5+")
    # Cached-token counts mod 128: interior, near-end, last lane, aligned, first lane.
    histories = (895, 1020, 1023, 1024, 1025, 1151)
    self._run_compare(
        histories,
        bucket,
        num_kv_heads=num_kv_heads,
        kv_dtype=kv_dtype,
    )

  @parameterized.product(
      bucket=[1, 2, 4, 8, 16, 128],
      page_size=[128, 256],
  )
  def test_mtp_boundary_offsets_sweep(self, bucket, page_size):
    """Test exact 128-lane VREG column and page boundaries for MTP decode stitch."""
    base = 512
    histories = (
        base + 0,  # dst_rel == 0 (start of column)
        base + 64,  # dst_rel == 64 (mid column)
        base + max(0, 128 - bucket),  # ends exactly at lane 127 of first col
        base + max(0, 128 - bucket + 1),  # spills 1 token into second col
        base + 127,  # starts at last lane of col 0, spills bucket-1 into col 1
        base + page_size - 1,  # crosses page boundary
    )
    self._run_compare(histories, bucket, page_size=page_size)

  @parameterized.parameters(1, 4, 16)
  def test_mtp_multi_block_context(self, bucket):
    """Test MTP decode stitch when sequences span multiple KV blocks."""
    histories = (1535, 1536, 1537, 2047, 2048, 2175)
    self._run_compare(histories, bucket, page_size=256, bkv_sz=512)

  @parameterized.parameters(4, 16, 128)
  def test_short_run_does_not_pull_padding_into_the_page(self, bucket):
    page_size = 256
    head_dim = 128
    num_q_heads = 8
    num_kv_heads = 1
    max_seqs = 8
    dtype = jnp.bfloat16
    rng = np.random.default_rng(11)
    num_seqs = 2
    q_lens = (bucket, bucket)
    kv_lengths = (1024 + bucket, 1023 + bucket)
    total_tokens = sum(q_lens)
    padded_tokens = total_tokens + page_size
    pages_per_seq = (max(kv_lengths) + page_size - 1) // page_size + 1
    cache_shape = wrapper.get_kv_cache_shape(
        max_seqs * pages_per_seq + 2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )

    def padded(heads):
      out = np.full((padded_tokens, heads, head_dim), np.nan)
      out[:total_tokens] = (
          rng.standard_normal((total_tokens, heads, head_dim)) * 0.5
      )
      return out.astype(dtype)

    q = padded(num_q_heads)
    k = padded(num_kv_heads)
    v = padded(num_kv_heads)
    before = (rng.standard_normal(cache_shape) * 0.5).astype(dtype)
    page_table = rng.permutation(cache_shape[0])[: max_seqs * pages_per_seq]
    page_table = page_table.reshape(max_seqs, pages_per_seq).astype(np.int32)
    kv_lens = np.zeros(max_seqs, dtype=np.int32)
    kv_lens[:num_seqs] = kv_lengths
    cu_q_lens = np.full(max_seqs + 1, total_tokens, dtype=np.int32)
    cu_q_lens[: num_seqs + 1] = np.cumsum((0, *q_lens))

    out, cache = wrapper.ragged_paged_attention(
        jnp.asarray(q),
        jnp.asarray(k),
        jnp.asarray(v),
        jnp.asarray(before),
        jnp.asarray(kv_lens),
        jnp.asarray(page_table.ravel()),
        jnp.asarray(cu_q_lens),
        jnp.asarray((num_seqs, num_seqs, num_seqs), jnp.int32),
        sm_scale=head_dim**-0.5,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        decode_query_size=bucket,
    )
    out, cache = jax.block_until_ready((out, cache))
    self.assertTrue(np.isfinite(np.asarray(out[:total_tokens])).all())

  def test_decode_mismatched_q_lens_allowed(self):
    bucket = 4
    page_size = 256
    head_dim = 128
    num_q_heads = 8
    num_kv_heads = 1
    dtype = jnp.bfloat16
    q_lens = (bucket, 1)
    total_tokens = sum(q_lens)
    cache_shape = wrapper.get_kv_cache_shape(
        16,
        page_size,
        num_kv_heads,
        head_dim,
        dtype,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    q = jnp.zeros((total_tokens, num_q_heads, head_dim), dtype=dtype)
    k = jnp.zeros((total_tokens, num_kv_heads, head_dim), dtype=dtype)
    v = jnp.zeros((total_tokens, num_kv_heads, head_dim), dtype=dtype)
    kv_cache = jnp.zeros(cache_shape, dtype=dtype)
    kv_lens = jnp.array([100 + bucket, 100 + 1] + [0] * 6, dtype=jnp.int32)
    page_indices = jnp.arange(16, dtype=jnp.int32)
    cu_q_lens = jnp.array(
        [0, bucket, total_tokens] + [total_tokens] * 6, dtype=jnp.int32
    )

    out, cache = wrapper.ragged_paged_attention(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        jnp.array([2, 2, 2], dtype=jnp.int32),
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        decode_query_size=bucket,
    )
    out, cache = jax.block_until_ready((out, cache))
    self.assertTrue(np.isfinite(np.asarray(out[:total_tokens])).all())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
