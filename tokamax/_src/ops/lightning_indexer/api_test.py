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
"""Tests for Lightning Indexer API."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import benchmarking
from tokamax._src import hlo_utils
from tokamax._src.ops.lightning_indexer import api
from tokamax._src.ops.lightning_indexer import arg_specs
from tokamax._src.ops.lightning_indexer import base
from tokamax._src.ops.lightning_indexer import test_base
from tokamax._src.ops.lightning_indexer.kernel import config as kernel_config

KVLayout = kernel_config.KVLayout


class ApiTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    test_base.skip_if_unsupported(self)

  def test_api_xla(self):
    inputs = test_base.make_test_inputs([1, 2], [128, 256], seed=99)

    @jax.jit
    def f(
        q, indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens, dist
    ):
      return api.lightning_indexer(
          q=q,
          indexer_weights=indexer_weights,
          cache_kv=cache_kv,
          seq_lens=seq_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=dist,
          k=16,
          implementation="xla",
      )

    args = (
        inputs["q"],
        inputs["indexer_weights"],
        inputs["cache_kv"],
        inputs["seq_lens"],
        inputs["page_indices"],
        inputs["cu_q_lens"],
        inputs["distribution"],
    )
    out_api = np.asarray(f(*args))
    out_base = np.asarray(base.LightningIndexer()(**inputs, k=16))
    np.testing.assert_array_equal(out_api, out_base)

    opspecs = hlo_utils.get_opspecs(f.lower(*args), include_xla_kernels=False)
    self.assertEmpty(opspecs)

  def test_api_mosaic_tpu(self):
    inputs = test_base.make_test_inputs([1, 4], [256, 256], seed=123)

    @jax.jit
    def f(
        q, indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens, dist
    ):
      return api.lightning_indexer(
          q=q,
          indexer_weights=indexer_weights,
          cache_kv=cache_kv,
          seq_lens=seq_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=dist,
          k=16,
          implementation="mosaic",
      )

    args = (
        inputs["q"],
        inputs["indexer_weights"],
        inputs["cache_kv"],
        inputs["seq_lens"],
        inputs["page_indices"],
        inputs["cu_q_lens"],
        inputs["distribution"],
    )
    out_tpu = np.asarray(f(*args))
    out_ref = np.asarray(
        api.lightning_indexer(**inputs, k=16, implementation="xla")
    )
    np.testing.assert_array_equal(
        np.sort(out_tpu, axis=-1),
        np.sort(out_ref, axis=-1),
    )

    opspecs = hlo_utils.get_opspecs(f.lower(*args), include_xla_kernels=False)
    self.assertNotEmpty(opspecs)
    self.assertIsInstance(opspecs[0].op, type(api.IMPLEMENTATIONS["mosaic"]))

  def test_representative_arg_specs_lower_mosaic_tpu(self):
    specs_by_name = {spec.name: spec for spec in arg_specs.ARG_SPECS}
    representative_names = (
        "ragged_decode_b256_k512_shuffled",
        "decode_b256_q1_kv9216_k512_has",
        "prefill_b1_q1024_kv1024_k512",
        "mixed_b9_d8_p1024_kv8192_k512",
        "spec_verify_b256_q4_kv8192_k512",
        "mixed_spec_verify_b12_q6_kv8192_k512",
        "prefill_b1_q1024_kv32768_k1024_cp4",
        "decode_b64_q1_kv1048576_k1024_cp8",
        "decode_b256_q1_kv9216_k512_h8",
        "decode_b256_q1_kv9216_k512_bf16q",
        "decode_b256_q1_kv9216_k512_p512",
        "decode_b256_q1_kv9216_k512_fragmented_pages",
    )
    for name in representative_names:
      with self.subTest(name):
        spec = specs_by_name[name]
        kwargs = dict(spec.args, implementation="mosaic")
        fn, args = benchmarking.standardize_function(
            api.lightning_indexer, kwargs=kwargs, seed=None
        )
        lowered = jax.jit(fn).lower(args)
        self.assertIsNotNone(lowered)

  @parameterized.named_parameters(
      dict(
          testcase_name="shuffled_ragged_decode",
          spec_kwargs=dict(
              batch_size=8,
              num_q_heads=16,
              k=128,
              q_len=1,
              s_list=[512] * 4 + [1024] * 4,
              shuffle_s_list=True,
              page_size=512,
          ),
      ),
      dict(
          testcase_name="compressed_has_prefill",
          spec_kwargs=dict(
              batch_size=1,
              num_q_heads=16,
              k=128,
              q_len=64,
              kv_len=1024,
              page_size=512,
              kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
          ),
      ),
      dict(
          testcase_name="first_chunk_prefill_kv_lt_k",
          spec_kwargs=dict(
              batch_size=1,
              num_q_heads=16,
              k=256,
              q_len=64,
              kv_len=512,
              page_size=512,
          ),
      ),
      dict(
          testcase_name="spec_verify_three_segment",
          spec_kwargs=dict(
              batch_size=6,
              num_q_heads=16,
              k=128,
              t_list=[1, 1, 4, 4, 16, 16],
              s_list=[512, 1024, 512, 1024, 512, 1024],
              page_size=512,
              uniform_q_len=4,
          ),
      ),
      dict(
          testcase_name="context_parallel_cp8_scores",
          spec_kwargs=dict(
              batch_size=4,
              num_q_heads=16,
              k=128,
              q_len=1,
              kv_len=4096,
              page_size=512,
              cp_size=8,
              cp_rank=3,
              interleave_size=4,
              return_scores=True,
          ),
      ),
      dict(
          testcase_name="head_sharded_bf16_fragmented_pages",
          spec_kwargs=dict(
              batch_size=8,
              num_q_heads=8,
              k=128,
              q_len=1,
              kv_len=1024,
              page_size=512,
              q_dtype=jnp.bfloat16,
              indexer_weights_dtype=jnp.bfloat16,
              shuffle_page_indices=True,
          ),
      ),
  )
  def test_argspec_modes_mosaic_matches_xla(self, spec_kwargs):
    # pylint: disable=protected-access
    spec = arg_specs._make_argspec(name="mode_test", **spec_kwargs)
    # pylint: enable=protected-access
    rng = np.random.default_rng(77)

    q_struct = spec.args["q"]
    w_struct = spec.args["indexer_weights"]
    cache_struct = spec.args["cache_kv"]
    kv_layout = spec.args["kv_layout"]

    num_pages = cache_struct.shape[0]
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
      compressed_page_size = cache_struct.shape[3]
    else:
      compressed_page_size = cache_struct.shape[1] * cache_struct.shape[2]
    head_dim = q_struct.shape[2]

    keys = jnp.asarray(
        rng.standard_normal(
            (num_pages, compressed_page_size, head_dim), dtype=np.float32
        )
    )
    cache_kv = test_base.quantize_and_pack_cache(keys, kv_layout=kv_layout)
    self.assertEqual(cache_kv.shape, cache_struct.shape)

    q = jnp.asarray(
        rng.standard_normal(q_struct.shape, dtype=np.float32)
    ).astype(q_struct.dtype)
    weights = jnp.asarray(
        rng.uniform(0.25, 1.75, size=w_struct.shape).astype(np.float32)
    ).astype(w_struct.dtype)

    call_kwargs: dict[str, Any] = {
        k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
        for k, v in spec.args.items()
    }
    call_kwargs.update(
        q=q,
        indexer_weights=weights,
        cache_kv=cache_kv,
    )
    pages_per_seq = (
        spec.args["page_indices"].shape[0] // spec.args["seq_lens"].shape[0]
    )
    max_kv_len = pages_per_seq * compressed_page_size

    out_tpu = api.lightning_indexer(**call_kwargs, implementation="mosaic")
    out_ref = api.lightning_indexer(**call_kwargs, implementation="xla")

    if spec.args.get("return_scores", False):
      tpu_idxs, tpu_bits = out_tpu
      ref_idxs, ref_bits = out_ref
      self._assert_topk_matches_reference(
          np.asarray(tpu_idxs), np.asarray(ref_idxs), call_kwargs, max_kv_len
      )
      tpu_scores = np.sort(
          np.asarray(jax.lax.bitcast_convert_type(tpu_bits, jnp.float32)),
          axis=-1,
      )
      ref_scores = np.sort(
          np.asarray(jax.lax.bitcast_convert_type(ref_bits, jnp.float32)),
          axis=-1,
      )
      np.testing.assert_allclose(tpu_scores, ref_scores, rtol=2e-2, atol=2e-2)
    else:
      self._assert_topk_matches_reference(
          np.asarray(out_tpu), np.asarray(out_ref), call_kwargs, max_kv_len
      )

  def _assert_topk_matches_reference(
      self,
      tpu_idxs: np.ndarray,
      ref_idxs: np.ndarray,
      call_kwargs: dict[str, Any],
      max_kv_len: int,
  ) -> None:
    self.assertEqual(tpu_idxs.shape, ref_idxs.shape)
    self.assertEqual(tpu_idxs.dtype, np.int32)
    np.testing.assert_array_equal(
        np.sum(tpu_idxs == -1, axis=-1),
        np.sum(ref_idxs == -1, axis=-1),
    )

    sorted_tpu = np.sort(tpu_idxs, axis=-1)
    sorted_ref = np.sort(ref_idxs, axis=-1)
    if call_kwargs["q"].dtype != jnp.float8_e4m3fn or np.array_equal(
        sorted_tpu, sorted_ref
    ):
      np.testing.assert_array_equal(sorted_tpu, sorted_ref)
      return

    # For 8-bit queries on blocked multi-token passes (`bf16_reduce=True`),
    # `relu(q @ k)` and `indexer_weights` are rounded to bfloat16 before head
    # accumulation, which can swap near-tied candidates at the k-th cutoff.
    all_kwargs: dict[str, Any] = dict(
        call_kwargs, k=max_kv_len, return_scores=True
    )
    all_idxs, all_bits = api.lightning_indexer(
        **all_kwargs, implementation="xla"
    )
    all_idxs_np = np.asarray(all_idxs)
    all_scores_np = np.asarray(
        jax.lax.bitcast_convert_type(all_bits, jnp.float32)
    )
    for row_idx in range(tpu_idxs.shape[0]):
      valid_tpu = tpu_idxs[row_idx][tpu_idxs[row_idx] >= 0]
      valid_ref = ref_idxs[row_idx][ref_idxs[row_idx] >= 0]
      self.assertLen(np.unique(valid_tpu), len(valid_tpu))
      if not valid_ref.size:
        continue
      valid_all_mask = all_idxs_np[row_idx] >= 0
      score_by_idx = dict(
          zip(
              all_idxs_np[row_idx][valid_all_mask].tolist(),
              all_scores_np[row_idx][valid_all_mask].tolist(),
          )
      )
      self.assertTrue(set(valid_tpu.tolist()).issubset(score_by_idx.keys()))
      overlap = len(set(valid_tpu.tolist()) & set(valid_ref.tolist()))
      self.assertGreaterEqual(overlap / len(valid_ref), 0.95)
      tpu_ref_scores = np.sort([score_by_idx[int(i)] for i in valid_tpu])
      opt_ref_scores = np.sort([score_by_idx[int(i)] for i in valid_ref])
      np.testing.assert_allclose(
          tpu_ref_scores, opt_ref_scores, rtol=2e-2, atol=2e-2
      )


if __name__ == "__main__":
  absltest.main()
