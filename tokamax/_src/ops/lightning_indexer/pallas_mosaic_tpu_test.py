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
"""Tests for Pallas Mosaic TPU Lightning Indexer operator."""

import inspect
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.lightning_indexer import pallas_mosaic_tpu
from tokamax._src.ops.lightning_indexer import test_base
from tokamax._src.ops.lightning_indexer.kernel import streamindex_topk as kernel_lib


class SparseCorePrefillHeuristicsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("long_prefill", 2048, 1, 65536, 1, jnp.float8_e4m3fn, 7, 256),
      ("two_windows", 512, 1, 8192, 1, jnp.float8_e4m3fn, 7, 256),
      ("short_query", 128, 1, 65536, 1, jnp.float8_e4m3fn, 7, None),
      ("one_window", 256, 1, 65536, 1, jnp.float8_e4m3fn, 7, None),
      ("short_context", 2048, 1, 4096, 1, jnp.float8_e4m3fn, 7, None),
      ("ragged_bucket", 513, 1, 65536, 1, jnp.float8_e4m3fn, 7, None),
      ("multiple_requests", 2048, 2, 65536, 1, jnp.float8_e4m3fn, 7, None),
      ("context_parallel", 2048, 1, 65536, 4, jnp.float8_e4m3fn, 7, None),
      ("bf16", 2048, 1, 65536, 1, jnp.bfloat16, 7, None),
      ("future_tpu", 2048, 1, 65536, 1, jnp.float8_e4m3fn, 8, None),
  )
  def test_chunk_policy(
      self, tokens, requests, positions, cp_size, dtype, generation, expected
  ):
    shape = jax.ShapeDtypeStruct
    op = pallas_mosaic_tpu.PallasTpuLightningIndexer()
    for layout in test_base.KVLayout:
      with self.subTest(layout=layout):
        cache_shape = (
            (requests * positions // 256, 33, 4, 256)
            if layout == test_base.KVLayout.SEQ_ALONG_LANE
            else (requests * positions // 256, 64, 4, 256)
        )
        bound = op.bind(
            q=shape((tokens, 32, 128), dtype),
            indexer_weights=shape((tokens, 32), jnp.float32),
            cache_kv=shape(cache_shape, jnp.uint8),
            seq_lens=shape((requests,), jnp.int32),
            page_indices=shape((requests * positions // 256,), jnp.int32),
            cu_q_lens=shape((requests + 1,), jnp.int32),
            distribution=shape((3,), jnp.int32),
            k=512,
            kv_layout=layout,
            cp_size=cp_size,
        )
        with mock.patch.object(
            pltpu,
            "get_tpu_info",
            return_value=types.SimpleNamespace(generation=generation),
        ):
          config = pallas_mosaic_tpu._sc_heuristics_config(bound)
          expected_config = pallas_mosaic_tpu.Config()
          if expected is not None:
            expected_config = pallas_mosaic_tpu.Config(
                num_kv_pages_per_block=(1, 8, 8), chunk_tokens=expected
            )
          self.assertEqual(config, expected_config)


class PallasTpuLightningIndexerTest(test_base.LightningIndexerTestBase):

  def __init__(self, *args):
    super().__init__(
        *args,
        topk_fn=pallas_mosaic_tpu.PallasTpuLightningIndexer(),
    )

  def setUp(self):
    super().setUp()
    test_base.skip_if_unsupported(self)

  def test_custom_config(self):
    cfg = pallas_mosaic_tpu.Config(
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 8, 8),
        decode_req_batch_size=1,
    )
    op = pallas_mosaic_tpu.PallasTpuLightningIndexer(config=cfg)
    inputs = test_base.make_test_inputs([1, 4], [256, 256], seed=13)
    actual = np.sort(np.asarray(op(**inputs, k=16)), axis=-1)
    expected = np.sort(np.asarray(self._ref_fn(**inputs, k=16)), axis=-1)
    np.testing.assert_array_equal(actual, expected)

  def test_autotuning_configs_match_reference(self):
    op = pallas_mosaic_tpu.PallasTpuLightningIndexer()
    inputs = test_base.make_test_inputs(
        [1, 4], [256, 256], pages_per_seq=4, seed=21
    )
    bound = op.bind(**inputs, k=16, compression_ratio=1)
    configs = bound.autotuning_configs
    self.assertIn(bound.default_config, configs)

    expected = np.sort(
        np.asarray(self._ref_fn(**inputs, k=16, compression_ratio=1)),
        axis=-1,
    )
    for cfg in configs:
      with self.subTest(f"{cfg=}"):
        actual = np.sort(
            np.asarray(
                op.replace(config=cfg)(**inputs, k=16, compression_ratio=1)
            ),
            axis=-1,
        )
        np.testing.assert_array_equal(actual, expected)

  @parameterized.product(
      layout=(
          test_base.KVLayout.HEAD_ALONG_SUBLANE,
          test_base.KVLayout.SEQ_ALONG_LANE,
      ),
      live_tokens=(0, 497, 512),
  )
  def test_prefill_windows_preserve_score_bits(self, layout, live_tokens):
    self._check_prefill_score_bits(layout, live_tokens, num_heads=4)

  @parameterized.parameters(
      test_base.KVLayout.HEAD_ALONG_SUBLANE,
      test_base.KVLayout.SEQ_ALONG_LANE,
  )
  def test_prefill_windows_64_heads(self, layout):
    self._check_prefill_score_bits(layout, 512, num_heads=64)

  def _check_prefill_score_bits(self, layout, live_tokens, *, num_heads):
    if pltpu.get_tpu_info().generation != 7:
      self.skipTest("The prefill heuristic is tuned for TPU7x.")
    inputs = test_base.make_test_inputs(
        [512],
        [8192],
        pages_per_seq=64,
        num_q_heads=num_heads,
        kv_layout=layout,
        seed=31,
    )
    inputs["q"] = inputs["q"].astype(jnp.float8_e4m3fn)
    inputs["cu_q_lens"] = jnp.array([0, live_tokens], jnp.int32)
    op = pallas_mosaic_tpu.PallasTpuLightningIndexer()
    config = op.bind(**inputs, k=512, kv_layout=layout).heuristics_config
    self.assertEqual(config.chunk_tokens, 256)

    def audit(config):
      def call(inputs):
        scores = []
        original_topk = kernel_lib.sparsecore_topk

        def record_topk(values, *args, **kwargs):
          scores.append(values)
          return original_topk(values, *args, **kwargs)

        # Expose HBM score windows in this audit only; the public result is
        # checked separately against the uninstrumented operator below.
        with (
            mock.patch.object(
                kernel_lib,
                "streamindex_topk",
                inspect.unwrap(kernel_lib.streamindex_topk),
            ),
            mock.patch.object(kernel_lib, "sparsecore_topk", record_topk),
        ):
          result = op.replace(config=config)(
              **inputs, k=512, kv_layout=layout, return_scores=True
          )
        return result, jnp.concatenate(scores, axis=0)

      return jax.jit(call)(inputs)

    expected, expected_scores = audit(pallas_mosaic_tpu.Config())
    actual, actual_scores = audit(config)
    for got, want in zip(actual, expected):
      np.testing.assert_array_equal(got, want)

    # Padding in the HBM score scratch is deliberately unwritten. TopK reads
    # only the visible prefix, so compare the defined score bits only.
    lengths = np.where(
        np.arange(512) < live_tokens,
        np.minimum(8192, 8192 - live_tokens + np.arange(512) + 1),
        0,
    )
    valid = np.arange(8192)[None, :] < lengths[:, None]
    np.testing.assert_array_equal(
        np.asarray(actual_scores)[valid], np.asarray(expected_scores)[valid]
    )
    result = op(**inputs, k=512, kv_layout=layout, return_scores=True)
    for got, want in zip(result, expected):
      np.testing.assert_array_equal(got, want)

  def test_prefill_windows_match_xla_reference(self):
    if pltpu.get_tpu_info().generation != 7:
      self.skipTest("The prefill heuristic is tuned for TPU7x.")
    inputs = test_base.make_test_inputs(
        [512], [8192], pages_per_seq=64, num_q_heads=4, seed=41
    )
    # Integer dot products and unit weights make both the existing scoring
    # kernel and the plain-XLA reference exact, including tied scores.
    keys = jax.random.randint(jax.random.key(42), (64, 128, 128), -1, 2).astype(
        jnp.float32
    )
    inputs["cache_kv"] = test_base.quantize_and_pack_cache(keys)
    inputs["q"] = jnp.ones((512, 4, 128), jnp.float8_e4m3fn)
    inputs["indexer_weights"] = jnp.ones((512, 4), jnp.float32)
    indices, score_bits = self._topk_fn(**inputs, k=512, return_scores=True)
    _, reference_bits = self._ref_fn(**inputs, k=512, return_scores=True)
    np.testing.assert_array_equal(
        np.sort(np.asarray(score_bits).view(np.float32), axis=-1),
        np.sort(np.asarray(reference_bits).view(np.float32), axis=-1),
    )
    sorted_indices = np.sort(np.asarray(indices), axis=-1)
    self.assertTrue(np.all(np.diff(sorted_indices, axis=-1) > 0))


if __name__ == "__main__":
  absltest.main()
