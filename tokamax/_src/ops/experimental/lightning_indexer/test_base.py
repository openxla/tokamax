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
"""Shared correctness test base for Lightning Indexer operator."""

from collections.abc import Sequence
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.lightning_indexer import reference
from tokamax._src.ops.experimental.lightning_indexer.kernel import config as kernel_config
from tokamax._src.ops.experimental.lightning_indexer.kernel import streamindex_topk as kernel_lib

KVLayout = kernel_config.KVLayout


def skip_if_unsupported(test_case: absltest.TestCase) -> None:
  """Skips `test_case` unless running on a TPU v7+ with SparseCore."""
  if jax.default_backend() != "tpu":
    test_case.skipTest("Only supported on TPUs.")
  try:
    info = pltpu.get_tpu_info()
    if info.generation < 7 or info.sparse_core is None:
      test_case.skipTest("Requires TPU v7+ with SparseCore.")
  except Exception:  # pylint: disable=broad-except
    test_case.skipTest("Failed to get TPU info.")


def _to_byte_lane(x: jax.Array) -> jax.Array:
  b = jax.lax.bitcast_convert_type(x, jnp.uint8)
  if b.ndim > x.ndim:
    b = b.reshape(*x.shape[:-1], -1)
  return b


def quantize_and_pack_cache(
    keys: jax.Array,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
) -> jax.Array:
  """Quantizes `float32[num_pages, page_size, head_dim]` keys into packed uint8 cache."""
  num_pages, page_size, head_dim = keys.shape
  fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
  amax = jnp.clip(jnp.max(jnp.abs(keys), axis=-1, keepdims=True), 1e-4, None)
  scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
  q_fp8 = (keys / scale).astype(jnp.float8_e4m3fn)
  scale_e8m0 = scale.astype(jnp.float8_e8m0fnu)

  record = jnp.concatenate(
      [_to_byte_lane(q_fp8), _to_byte_lane(scale_e8m0)], axis=-1
  )
  width = ((record.shape[-1] + 127) // 128) * 128
  record = jnp.pad(record, ((0, 0), (0, 0), (0, width - record.shape[-1])))
  has_cache = record.reshape(num_pages, page_size // 4, 4, width)
  if kv_layout == KVLayout.SEQ_ALONG_LANE:
    return kernel_lib.convert_cache_to_seq_along_lane(has_cache, head_dim)
  return has_cache


def make_test_inputs(
    q_lens: Sequence[int],
    seq_lens: Sequence[int],
    *,
    page_size: int = 128,
    pages_per_seq: int = 4,
    num_q_heads: int = 4,
    head_dim: int = 128,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
    seed: int = 0,
) -> dict[str, Any]:
  """Builds deterministic test inputs for `LightningIndexer`."""
  rng = np.random.default_rng(seed)
  num_seqs = len(q_lens)
  num_tokens = int(sum(q_lens))
  num_pages = num_seqs * pages_per_seq

  block_table = (
      rng.permutation(num_pages)
      .astype(np.int32)
      .reshape(num_seqs, pages_per_seq)
  )
  keys = jnp.asarray(
      rng.standard_normal((num_pages, page_size, head_dim), dtype=np.float32)
  )
  cache_kv = quantize_and_pack_cache(keys, kv_layout=kv_layout)

  num_decodes = 0
  while num_decodes < num_seqs and q_lens[num_decodes] == 1:
    num_decodes += 1

  q = jnp.asarray(
      rng.standard_normal((num_tokens, num_q_heads, head_dim), dtype=np.float32)
  )
  indexer_weights = jnp.asarray(
      rng.uniform(0.25, 1.75, size=(num_tokens, num_q_heads)).astype(np.float32)
  )
  return dict(
      q=q,
      indexer_weights=indexer_weights,
      cache_kv=cache_kv,
      seq_lens=jnp.asarray(seq_lens, dtype=jnp.int32),
      page_indices=jnp.asarray(block_table.reshape(-1), dtype=jnp.int32),
      cu_q_lens=jnp.asarray(
          np.concatenate([[0], np.cumsum(q_lens)]), dtype=jnp.int32
      ),
      distribution=jnp.asarray(
          [num_decodes, num_decodes, num_seqs], dtype=jnp.int32
      ),
  )


# pylint: disable=missing-function-docstring
class LightningIndexerTestBase(parameterized.TestCase):
  """Shared correctness suite for `LightningIndexer` implementations."""

  def __init__(self, *args, topk_fn):
    super().__init__(*args)
    self._topk_fn = topk_fn
    self._ref_fn = reference.lightning_indexer

  @parameterized.named_parameters(
      dict(
          testcase_name="decode_only",
          q_lens=[1, 1, 1, 1],
          seq_lens=[256, 384, 128, 512],
          k=16,
          compression_ratio=1,
          kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
      ),
      dict(
          testcase_name="prefill_only",
          q_lens=[16, 8],
          seq_lens=[256, 384],
          k=16,
          compression_ratio=1,
          kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
      ),
      dict(
          testcase_name="mixed_decode_and_prefill",
          q_lens=[1, 1, 12],
          seq_lens=[256, 128, 384],
          k=16,
          compression_ratio=1,
          kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
      ),
      dict(
          testcase_name="compressed_kv_ratio_4",
          q_lens=[1, 8],
          seq_lens=[512, 1024],
          k=16,
          compression_ratio=4,
          kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
      ),
      dict(
          testcase_name="seq_along_lane_layout",
          q_lens=[1, 1, 8],
          seq_lens=[256, 384, 256],
          k=16,
          compression_ratio=1,
          kv_layout=KVLayout.SEQ_ALONG_LANE,
      ),
  )
  def test_matches_reference(
      self,
      q_lens,
      seq_lens,
      k,
      compression_ratio,
      kv_layout,
  ):
    inputs = make_test_inputs(
        q_lens,
        seq_lens,
        kv_layout=kv_layout,
        seed=42,
    )
    kwargs = dict(
        **inputs,
        k=k,
        compression_ratio=compression_ratio,
        kv_layout=kv_layout,
    )
    actual = np.asarray(self._topk_fn(**kwargs))
    expected = np.asarray(self._ref_fn(**kwargs))
    self.assertEqual(actual.shape, expected.shape)
    np.testing.assert_array_equal(
        np.sort(actual, axis=-1),
        np.sort(expected, axis=-1),
    )

  def test_return_scores(self):
    inputs = make_test_inputs([1, 4], [256, 256], seed=7)
    kwargs = dict(**inputs, k=16, compression_ratio=1, return_scores=True)
    actual_idxs, actual_scores_bits = self._topk_fn(**kwargs)
    expected_idxs, expected_scores_bits = self._ref_fn(**kwargs)

    actual_idxs_np = np.asarray(actual_idxs)
    expected_idxs_np = np.asarray(expected_idxs)
    np.testing.assert_array_equal(
        np.sort(actual_idxs_np, axis=-1),
        np.sort(expected_idxs_np, axis=-1),
    )

    actual_scores = np.sort(
        np.asarray(
            jax.lax.bitcast_convert_type(actual_scores_bits, jnp.float32)
        ),
        axis=-1,
    )
    expected_scores = np.sort(
        np.asarray(
            jax.lax.bitcast_convert_type(expected_scores_bits, jnp.float32)
        ),
        axis=-1,
    )
    np.testing.assert_allclose(
        actual_scores, expected_scores, rtol=1e-2, atol=1e-2
    )
