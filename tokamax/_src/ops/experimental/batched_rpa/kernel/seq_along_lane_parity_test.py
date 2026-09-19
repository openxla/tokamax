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
"""Numeric parity for the SEQ_ALONG_LANE KV-cache layout."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.batched_rpa.kernel import configs
from tokamax._src.ops.experimental.batched_rpa.kernel import wrapper

_DECODE_BLOCKS = configs.BlockSizes(
    bq_sz=1,
    bq_c_sz=1,
    bkv_sz=256,
    batch_size=8,
    n_buffer=3,
)
_PREFILL_BLOCKS = configs.BlockSizes(
    bq_sz=1792,
    bq_c_sz=28,
    bkv_sz=256,
    batch_size=2,
    n_buffer=3,
)


def _build_cache(
    kv_layout,
    target_k,
    target_v,
    kv_len,
    total_pages,
    page_size,
    num_q_heads,
    num_kv_heads,
    head_dim,
    dtype,
    page_indices,
    sm_scale,
):
  cache_shape = wrapper.get_kv_cache_shape(
      total_pages,
      page_size,
      num_kv_heads,
      head_dim,
      dtype,
      kv_layout=kv_layout,
  )
  _, cache = wrapper.ragged_paged_attention(
      jnp.zeros((kv_len, num_q_heads, head_dim), dtype),
      jnp.asarray(target_k, dtype),
      jnp.asarray(target_v, dtype),
      jnp.zeros(cache_shape, dtype),
      jnp.array([kv_len], jnp.int32),
      page_indices,
      jnp.array([0, kv_len], jnp.int32),
      jnp.array([0, 0, 1], jnp.int32),
      sm_scale=sm_scale,
      decode_block_sizes=_DECODE_BLOCKS,
      prefill_block_sizes=_PREFILL_BLOCKS,
      kv_layout=kv_layout,
  )
  return cache


class SeqAlongLaneParityTest(parameterized.TestCase):

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
      ("prefill_32_16", 32, 16, (0, 0, 1)),
      ("decode_33_1", 33, 1, (1, 1, 1)),
      ("decode_4096_1", 4096, 1, (1, 1, 1)),
  )
  def test_seq_along_lane_matches_head_along_sublane(
      self, kv_len, q_len, distribution
  ):
    rng = np.random.default_rng(0)
    dtype = jnp.bfloat16
    head_dim = 128
    num_kv_heads, num_q_heads = 2, 4
    page_size, total_pages = 128, max(8, -(-kv_len // 16) + 1)
    pages_per_seq = total_pages
    sm_scale = head_dim**-0.5
    page_indices = jnp.arange(pages_per_seq, dtype=jnp.int32)

    def r(*shape):
      return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    target_k = r(kv_len, num_kv_heads, head_dim)
    target_v = r(kv_len, num_kv_heads, head_dim)
    query = r(q_len, num_q_heads, head_dim)
    new_k = target_k[kv_len - q_len : kv_len]
    new_v = target_v[kv_len - q_len : kv_len]

    kv_lens = jnp.array([kv_len], jnp.int32)
    cu_q_lens = jnp.array([0, q_len], jnp.int32)
    distribution_arr = jnp.asarray(distribution, jnp.int32)

    def run(kv_layout):
      cache = _build_cache(
          kv_layout,
          target_k,
          target_v,
          kv_len,
          total_pages,
          page_size,
          num_q_heads,
          num_kv_heads,
          head_dim,
          dtype,
          page_indices,
          sm_scale,
      )
      out, _ = wrapper.ragged_paged_attention(
          jnp.asarray(query, dtype),
          jnp.asarray(new_k, dtype),
          jnp.asarray(new_v, dtype),
          cache,
          kv_lens,
          page_indices,
          cu_q_lens,
          distribution_arr,
          sm_scale=sm_scale,
          decode_block_sizes=_DECODE_BLOCKS,
          prefill_block_sizes=_PREFILL_BLOCKS,
          kv_layout=kv_layout,
      )
      return np.asarray(out.astype(jnp.float32))

    out_sublane = run(configs.KVLayout.HEAD_ALONG_SUBLANE)
    out_lane = run(configs.KVLayout.SEQ_ALONG_LANE)

    np.testing.assert_allclose(out_lane, out_sublane, atol=2e-2, rtol=0)
    self.assertTrue(np.isfinite(out_lane).all())


if __name__ == "__main__":
  absltest.main()
