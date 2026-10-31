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
"""Tests for base Lightning Indexer operator and argument specs."""

import collections
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import immutabledict
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import benchmarking
from tokamax._src.ops.lightning_indexer import api
from tokamax._src.ops.lightning_indexer import arg_specs
from tokamax._src.ops.lightning_indexer import base
from tokamax._src.ops.lightning_indexer import pallas_mosaic_tpu
from tokamax._src.ops.lightning_indexer import test_base
from tokamax._src.ops.lightning_indexer.kernel import config as kernel_config
from tokamax._src.ops.lightning_indexer.kernel import metadata

KVLayout = kernel_config.KVLayout


class LightningIndexerBaseTest(test_base.LightningIndexerTestBase):

  def __init__(self, *args):
    super().__init__(*args, topk_fn=base.LightningIndexer())

  def test_validation_errors(self):
    op = base.LightningIndexer()
    inputs = test_base.make_test_inputs([1, 2], [128, 256], seed=0)

    with self.assertRaisesRegex(ValueError, "k must be positive"):
      op.bind(**inputs, k=0)

    with self.assertRaisesRegex(ValueError, "compression_ratio must be a"):
      op.bind(**inputs, k=16, compression_ratio=3)

    bad_cu_q: dict[str, Any] = {
        **inputs,
        "cu_q_lens": jnp.zeros((5,), dtype=jnp.int32),
    }
    with self.assertRaisesRegex(ValueError, "cu_q_lens length"):
      op.bind(**bad_cu_q, k=16)

    bad_pages: dict[str, Any] = {
        **inputs,
        "page_indices": jnp.zeros((5,), dtype=jnp.int32),
    }
    with self.assertRaisesRegex(ValueError, "page_indices length"):
      op.bind(**bad_pages, k=16)

  @parameterized.parameters(2, 8)
  def test_context_parallel_reconstructs_global_topk(self, cp_size: int):
    rng = np.random.default_rng(123 + cp_size)
    compression_ratio = 4
    interleave_size = 4
    interleave_c = interleave_size // compression_ratio
    seq_len = 1024
    global_c = seq_len // compression_ratio
    local_page_size = global_c // cp_size
    k = 16

    keys_full = jnp.asarray(
        rng.standard_normal((1, global_c, 128), dtype=np.float32)
    )
    q = jnp.asarray(rng.standard_normal((2, 4, 128), dtype=np.float32))
    weights = jnp.asarray(
        rng.uniform(0.25, 1.75, size=(2, 4)).astype(np.float32)
    )
    seq_lens = jnp.asarray([seq_len], dtype=jnp.int32)
    cu_q_lens = jnp.asarray([0, 2], dtype=jnp.int32)
    dist = jnp.asarray([0, 0, 1], dtype=jnp.int32)

    op = base.LightningIndexer()
    global_cache = test_base.quantize_and_pack_cache(keys_full)
    expected_idxs, expected_bits = op(
        q=q,
        indexer_weights=weights,
        cache_kv=global_cache,
        seq_lens=seq_lens,
        page_indices=jnp.asarray([0], dtype=jnp.int32),
        cu_q_lens=cu_q_lens,
        distribution=dist,
        k=k,
        compression_ratio=compression_ratio,
        return_scores=True,
    )

    owner = (np.arange(global_c) // interleave_c) % cp_size
    cand_global_idxs = []
    cand_scores = []
    for r in range(cp_size):
      owned = np.flatnonzero(owner == r)
      padded = np.zeros((1, local_page_size, 128), dtype=np.float32)
      padded[0, : len(owned)] = np.asarray(keys_full[0, owned])
      rank_cache = test_base.quantize_and_pack_cache(jnp.asarray(padded))
      rank_idxs, rank_bits = op(
          q=q,
          indexer_weights=weights,
          cache_kv=rank_cache,
          seq_lens=seq_lens,
          page_indices=jnp.asarray([0], dtype=jnp.int32),
          cu_q_lens=cu_q_lens,
          distribution=dist,
          k=k,
          compression_ratio=compression_ratio,
          cp_size=cp_size,
          cp_rank=r,
          interleave_size=interleave_size,
          return_scores=True,
      )
      rank_scores = np.asarray(
          jax.lax.bitcast_convert_type(rank_bits, jnp.float32)
      )
      rank_global = np.asarray(
          metadata.cp_local_to_global(rank_idxs, r, cp_size, interleave_c)
      )
      rank_global = np.where(np.isfinite(rank_scores), rank_global, -1)
      cand_global_idxs.append(rank_global)
      cand_scores.append(rank_scores)

    all_idxs = np.concatenate(cand_global_idxs, axis=1)
    all_scores = np.concatenate(cand_scores, axis=1)
    top_order = np.argsort(-all_scores, axis=1)[:, :k]
    merged_idxs = np.take_along_axis(all_idxs, top_order, axis=1)
    merged_scores = np.take_along_axis(all_scores, top_order, axis=1)

    np.testing.assert_array_equal(
        np.sort(merged_idxs, axis=-1),
        np.sort(np.asarray(expected_idxs), axis=-1),
    )
    expected_scores = np.asarray(
        jax.lax.bitcast_convert_type(expected_bits, jnp.float32)
    )
    np.testing.assert_allclose(merged_scores, expected_scores, rtol=1e-5)


class LightningIndexerArgSpecsTest(parameterized.TestCase):

  def test_arg_specs_unique_and_hashable(self):
    specs = arg_specs.ARG_SPECS
    self.assertGreaterEqual(len(specs), 51)

    names = [s.name for s in specs]
    duplicates = [k for k, v in collections.Counter(names).items() if v > 1]
    self.assertEmpty(duplicates)

    full_names = [s.full_name for s in specs]
    self.assertLen(set(full_names), len(specs))

    for spec in specs:
      self.assertIn("forward_only", spec.tags, spec.name)

    primary_specs = [s for s in specs if "primary" in s.tags]
    self.assertNotEmpty(primary_specs)
    self.assertLess(len(primary_specs), len(specs))

    args_set = {immutabledict.immutabledict(s.args) for s in specs}
    self.assertLen(args_set, len(specs))

  def test_all_arg_specs_bind_and_standardize(self):
    base_op = base.LightningIndexer()
    tpu_op = pallas_mosaic_tpu.PallasTpuLightningIndexer()
    for spec in arg_specs.ARG_SPECS:
      with self.subTest(spec.name):
        abstract_args: dict[str, Any] = {
            k: (
                jax.ShapeDtypeStruct(v.shape, v.dtype)
                if isinstance(v, np.ndarray)
                else v
            )
            for k, v in spec.args.items()
        }
        base_op.bind(**abstract_args)
        tpu_op.bind(**abstract_args)
        fn, args = benchmarking.standardize_function(
            api.lightning_indexer,
            kwargs=dict(spec.args, implementation="xla"),
            seed=None,
        )
        self.assertIsNotNone(jax.jit(fn).lower(args))

  def test_make_argspec_features_and_validation(self):
    # pylint: disable=protected-access
    make_spec = arg_specs._make_argspec

    default_spec = make_spec(name="default_test", batch_size=2, kv_len=1024)
    for optional_key in (
        "cp_size",
        "cp_rank",
        "interleave_size",
        "return_scores",
        "uniform_q_len",
    ):
      self.assertNotIn(optional_key, default_spec.args)

    s_base = [1024] * 4 + [2048] * 4
    unshuffled = make_spec(name="unshuffled", batch_size=8, s_list=s_base)
    shuffled = make_spec(
        name="shuffled", batch_size=8, s_list=s_base, shuffle_s_list=True
    )
    self.assertFalse(
        np.array_equal(unshuffled.args["seq_lens"], shuffled.args["seq_lens"])
    )
    np.testing.assert_array_equal(
        np.sort(unshuffled.args["seq_lens"]),
        np.sort(shuffled.args["seq_lens"]),
    )

    frag = make_spec(
        name="frag", batch_size=4, kv_len=2048, shuffle_page_indices=True
    )
    self.assertFalse(
        np.array_equal(frag.args["page_indices"], np.arange(8, dtype=np.int32))
    )
    np.testing.assert_array_equal(
        np.sort(frag.args["page_indices"]), np.arange(8, dtype=np.int32)
    )

    explicit = make_spec(
        name="explicit",
        batch_size=2,
        kv_len=1024,
        page_indices=[5, 2],
    )
    np.testing.assert_array_equal(
        explicit.args["page_indices"], np.array([5, 2], dtype=np.int32)
    )
    self.assertEqual(explicit.args["cache_kv"].shape[0], 6)

    spec_verify = make_spec(
        name="spec_verify",
        batch_size=5,
        t_list=[1, 1, 6, 6, 32],
        kv_len=2048,
        uniform_q_len=6,
    )
    np.testing.assert_array_equal(
        spec_verify.args["distribution"], np.array([2, 4, 5], dtype=np.int32)
    )
    self.assertEqual(spec_verify.args["uniform_q_len"], 6)

    cp_spec = make_spec(
        name="cp8",
        batch_size=4,
        kv_len=32768,
        page_size=1024,
        cp_size=8,
        cp_rank=3,
        interleave_size=4,
        return_scores=True,
    )
    self.assertEqual(cp_spec.args["cp_size"], 8)
    self.assertEqual(cp_spec.args["cp_rank"], 3)
    self.assertEqual(cp_spec.args["interleave_size"], 4)
    self.assertTrue(cp_spec.args["return_scores"])
    # 32768 / 4 = 8192 global compressed tokens -> 1024 per rank -> 4 pages/seq.
    self.assertEqual(cp_spec.args["page_indices"].shape, (16,))

    with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
      make_spec(name="bad_b", batch_size=0)
    with self.assertRaisesRegex(ValueError, "must match batch_size"):
      make_spec(name="bad_t", batch_size=2, t_list=[1])
    with self.assertRaisesRegex(ValueError, "power of 2"):
      make_spec(name="bad_cr", compression_ratio=3)
    with self.assertRaisesRegex(ValueError, "cp_rank"):
      make_spec(name="bad_rank", cp_size=4, cp_rank=4, interleave_size=4)
    with self.assertRaisesRegex(ValueError, "multiple of compression_ratio"):
      make_spec(name="bad_il", cp_size=4, interleave_size=2)
    with self.assertRaisesRegex(ValueError, "non-negative"):
      make_spec(name="bad_neg_page", batch_size=1, page_indices=[-1])
    with self.assertRaisesRegex(ValueError, "Cannot specify both"):
      make_spec(
          name="bad_both_pages",
          batch_size=1,
          page_indices=[0],
          shuffle_page_indices=True,
      )
    with self.assertRaisesRegex(ValueError, "uniform_q_len must be positive"):
      make_spec(name="bad_u", uniform_q_len=0)
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      make_spec(
          name="bad_sal_page",
          page_size=256,
          compression_ratio=4,
          kv_layout=KVLayout.SEQ_ALONG_LANE,
      )
    # pylint: enable=protected-access


if __name__ == "__main__":
  absltest.main()
