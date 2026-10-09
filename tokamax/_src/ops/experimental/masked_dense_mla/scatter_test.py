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
"""Correctness tests for the SparseCore Pallas sparse-MLA KV cache scatter."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.masked_dense_mla import kv_cache_utils
from tokamax._src.ops.experimental.masked_dense_mla import scatter
from tokamax._src.ops.experimental.masked_dense_mla import scatter_utils

SKIP_ROW = kv_cache_utils.SKIP_ROW
KVCacheLayout = kv_cache_utils.KVCacheLayout
KVCacheType = kv_cache_utils.KVCacheType
SparseMLAKVCacheSpec = kv_cache_utils.SparseMLAKVCacheSpec

LKV_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 256
KV_PACKING = 4

_LIVE_SPECS = (
    ("nope_tc", KVCacheType.NOPE, KVCacheLayout.TENSORCORE, LKV_DIM),
    ("nope_sc", KVCacheType.NOPE, KVCacheLayout.SPARSECORE, LKV_DIM),
    ("rope_sc", KVCacheType.ROPE, KVCacheLayout.SPARSECORE, ROPE_DIM),
)

_SCENARIOS = (
    ("decode_b16", 16, 1, 1024, 64),
    ("decode_b64", 64, 1, 512, 128),
    ("decode_b256", 256, 1, 256, 256),
    ("prefill_b1_t256", 1, 256, 1024, 64),
    ("prefill_b2_t128", 2, 128, 1024, 64),
    ("prefill_b4_t64", 4, 64, 512, 64),
    ("prefill_b16_t16", 16, 16, 1024, 64),
    ("prefill_b64_t16", 64, 16, 512, 128),
)


def _unpack4(words_u32: np.ndarray) -> np.ndarray:
  """Unpacks uint32 words `[..., n]` into four uint8 bands `[..., 4, n]`."""
  w = np.asarray(words_u32).astype(np.uint32)
  bands = [((w >> (8 * b)) & 0xFF).astype(np.uint8) for b in range(4)]
  return np.stack(bands, axis=-2)


def _decode_rows(cache: jax.Array, spec: SparseMLAKVCacheSpec) -> np.ndarray:
  """Decodes any layout to `[num_pages * page_size, token_bytes]` uint8."""
  rows = np.asarray(cache).reshape(spec.num_pages, spec.page_size, -1)
  if spec.layout is KVCacheLayout.SPARSECORE:
    rows = _unpack4(rows)
  return rows.reshape(spec.num_pages * spec.page_size, spec.token_bytes)


class SparseMLAScatterTest(parameterized.TestCase):
  """Correctness tests for the SparseCore Pallas scatter kernel."""

  def setUp(self):
    super().setUp()
    if backend.get_default_device().device_kind != "TPU7x":
      self.skipTest("Only tested on TPU7x.")

  def _generate_inputs(
      self,
      num_seqs: int,
      tokens_per_seq: int,
      max_seq_len: int,
      num_pages: int,
      page_size: int = PAGE_SIZE,
  ):
    rng = np.random.default_rng(0)
    num_tokens = num_seqs * tokens_per_seq

    seq_lens = rng.integers(
        tokens_per_seq, max_seq_len + 1, size=(num_seqs,), dtype=np.int32
    )
    query_start_loc = np.arange(
        0, num_tokens + 1, tokens_per_seq, dtype=np.int32
    )

    pages_per_seq = (max_seq_len + page_size - 1) // page_size
    num_pages = max(num_pages, num_seqs * pages_per_seq)
    base_pages = np.arange(num_seqs * pages_per_seq, dtype=np.int32).reshape(
        num_seqs, pages_per_seq
    )
    block_tables = np.apply_along_axis(rng.permutation, 1, base_pages)

    key = jax.random.PRNGKey(42)
    k_nope, k_rope = jax.random.split(key, 2)
    kv_c_u8 = jax.random.randint(
        k_nope, (num_tokens, LKV_DIM), 0, 255, jnp.int32
    ).astype(jnp.uint8)
    k_pe_u8 = jax.random.randint(
        k_rope, (num_tokens, ROPE_DIM), 0, 255, jnp.int32
    ).astype(jnp.uint8)

    specs = {
        name: SparseMLAKVCacheSpec.create(
            cache_type, layout, num_pages, page_size, head_dim, KV_PACKING
        )
        for name, cache_type, layout, head_dim in _LIVE_SPECS
    }
    page, slot = kv_cache_utils.get_page_and_slot(
        num_tokens,
        jnp.asarray(seq_lens),
        jnp.asarray(block_tables.reshape(-1)),
        jnp.asarray(query_start_loc),
        page_size,
    )

    return {
        "specs": specs,
        "caches": {
            k: jnp.zeros(s.shape, s.jax_dtype) for k, s in specs.items()
        },
        "kv_c_u8": kv_c_u8,
        "kv_c_normed": jax.lax.bitcast_convert_type(
            kv_c_u8, jnp.float8_e4m3fn
        ),
        "k_pe_u8": k_pe_u8,
        "k_pe": jax.lax.bitcast_convert_type(k_pe_u8, jnp.float8_e4m3fn),
        "seq_lens": jnp.asarray(seq_lens),
        "block_tables": jnp.asarray(block_tables.reshape(-1)),
        "query_start_loc": jnp.asarray(query_start_loc),
        "page": np.asarray(page),
        "slot": np.asarray(slot),
        "num_tokens": num_tokens,
        "num_pages": num_pages,
    }

  def _check_scatter(self, inp, key, values, raw_u8):
    spec, cache = inp["specs"][key], inp["caches"][key]

    ref = kv_cache_utils._scatter_rows(
        cache,
        spec,
        values,
        jnp.asarray(inp["page"]),
        jnp.asarray(inp["slot"]),
    )
    out = scatter.scatter(
        cache,
        values,
        spec,
        inp["seq_lens"],
        inp["block_tables"],
        inp["query_start_loc"],
    )
    np.testing.assert_array_equal(
        np.asarray(out),
        np.asarray(ref),
        err_msg=f"Pallas {key} mismatch vs the jnp reference",
    )

    rows = _decode_rows(out, spec).reshape(
        spec.num_pages, spec.page_size, spec.token_bytes
    )
    written = rows[inp["page"], inp["slot"]]
    np.testing.assert_array_equal(
        written[:, : raw_u8.shape[1]],
        np.asarray(raw_u8),
        err_msg=f"Pallas {key} decoded bytes mismatch vs input",
    )
    self.assertEqual(int(np.count_nonzero(written[:, raw_u8.shape[1] :])), 0)

  @parameterized.named_parameters(*_SCENARIOS)
  def test_scatter_nope_tc(
      self, num_seqs, tokens_per_seq, max_seq_len, num_pages
  ):
    inp = self._generate_inputs(
        num_seqs, tokens_per_seq, max_seq_len, num_pages
    )
    self._check_scatter(inp, "nope_tc", inp["kv_c_normed"], inp["kv_c_u8"])

  @parameterized.named_parameters(*_SCENARIOS)
  def test_scatter_nope_sc(
      self, num_seqs, tokens_per_seq, max_seq_len, num_pages
  ):
    inp = self._generate_inputs(
        num_seqs, tokens_per_seq, max_seq_len, num_pages
    )
    self._check_scatter(inp, "nope_sc", inp["kv_c_normed"], inp["kv_c_u8"])

  @parameterized.named_parameters(*_SCENARIOS)
  def test_scatter_rope_sc(
      self, num_seqs, tokens_per_seq, max_seq_len, num_pages
  ):
    inp = self._generate_inputs(
        num_seqs, tokens_per_seq, max_seq_len, num_pages
    )
    self._check_scatter(inp, "rope_sc", inp["k_pe"], inp["k_pe_u8"])

  def test_nope_sparsecore_is_byte_identical_to_tensorcore(self):
    inp = self._generate_inputs(8, 4, 512, 64)
    out = {}
    for key in ("nope_tc", "nope_sc"):
      out[key] = scatter.scatter(
          inp["caches"][key],
          inp["kv_c_normed"],
          inp["specs"][key],
          inp["seq_lens"],
          inp["block_tables"],
          inp["query_start_loc"],
      )
    np.testing.assert_array_equal(
        _decode_rows(out["nope_tc"], inp["specs"]["nope_tc"]),
        _decode_rows(out["nope_sc"], inp["specs"]["nope_sc"]),
    )

  def test_padded_tokens_are_dropped(self):
    inp = self._generate_inputs(
        num_seqs=4, tokens_per_seq=8, max_seq_len=PAGE_SIZE, num_pages=64
    )
    valid_tokens = inp["num_tokens"]
    num_tokens = 2 * valid_tokens

    def _pad(values):
      raw = jax.lax.bitcast_convert_type(values, jnp.uint8)
      tail = jnp.full(
          (num_tokens - valid_tokens, raw.shape[1]), 0xAB, jnp.uint8
      )
      return jax.lax.bitcast_convert_type(
          jnp.concatenate([raw, tail], axis=0), values.dtype
      )

    dst_rows = kv_cache_utils.get_dst_rows(
        num_tokens=num_tokens,
        seq_lens=inp["seq_lens"],
        block_tables=inp["block_tables"],
        query_start_loc=inp["query_start_loc"],
        page_size=PAGE_SIZE,
    )
    np.testing.assert_array_equal(
        np.asarray(dst_rows[valid_tokens:]),
        np.full(num_tokens - valid_tokens, SKIP_ROW),
    )
    self.assertTrue(bool(jnp.all(dst_rows[:valid_tokens] >= 0)))

    for key, values in (
        ("nope_tc", inp["kv_c_normed"]),
        ("nope_sc", inp["kv_c_normed"]),
        ("rope_sc", inp["k_pe"]),
    ):
      spec = inp["specs"][key]
      out = scatter.scatter(
          inp["caches"][key], _pad(values), spec, dst_rows=dst_rows
      )
      rows = _decode_rows(out, spec)
      written_rows = int(rows.any(axis=1).sum())
      self.assertEqual(
          written_rows,
          valid_tokens,
          f"{key}: {written_rows} rows written, expected {valid_tokens}",
      )
      raw = np.asarray(jax.lax.bitcast_convert_type(values, jnp.uint8))
      written = rows[np.asarray(dst_rows[:valid_tokens])]
      np.testing.assert_array_equal(
          written[:, : raw.shape[1]],
          raw,
          err_msg=f"{key}: valid tokens corrupted by the padded tail",
      )

  @parameterized.named_parameters(
      dict(
          testcase_name=f"nope_{nope.value}_rope_{rope.value}",
          nope_layout=nope,
          rope_layout=rope,
      )
      for nope in KVCacheLayout
      for rope in KVCacheLayout
  )
  def test_update_dispatches_without_changing_the_result(
      self, nope_layout, rope_layout
  ):
    inp = self._generate_inputs(8, 16, 512, 64)
    specs = {
        "nope_spec": SparseMLAKVCacheSpec.create(
            KVCacheType.NOPE,
            nope_layout,
            inp["num_pages"],
            PAGE_SIZE,
            LKV_DIM,
            KV_PACKING,
        ),
        "rope_spec": SparseMLAKVCacheSpec.create(
            KVCacheType.ROPE,
            rope_layout,
            inp["num_pages"],
            PAGE_SIZE,
            ROPE_DIM,
            KV_PACKING,
        ),
    }
    args = (
        jnp.zeros(specs["nope_spec"].shape, specs["nope_spec"].jax_dtype),
        jnp.zeros(specs["rope_spec"].shape, specs["rope_spec"].jax_dtype),
        inp["kv_c_normed"],
        inp["k_pe"],
        inp["seq_lens"],
        inp["block_tables"],
        inp["query_start_loc"],
    )

    got = kv_cache_utils.update_sparse_mla_kv_cache(*args, **specs)
    want = kv_cache_utils.update_sparse_mla_kv_cache_jax(*args, **specs)
    for name, g, w in zip(("nope", "rope"), got, want):
      np.testing.assert_array_equal(
          np.asarray(g),
          np.asarray(w),
          err_msg=f"{name} cache differs from the jnp writer",
      )

  @parameterized.named_parameters(
      dict(
          testcase_name=f"d{dcp_size}_i{interleave_size}_{layout.value}",
          dcp_size=dcp_size,
          interleave_size=interleave_size,
          layout=layout,
      )
      for dcp_size, interleave_size in ((2, 4), (4, 64), (8, 16), (16, 4))
      for layout in KVCacheLayout
  )
  def test_dcp_update_matches_the_jnp_writer(
      self, dcp_size, interleave_size, layout
  ):
    inp = self._generate_inputs(8, 16, 512, 64)
    specs = {
        "nope_spec": SparseMLAKVCacheSpec.create(
            KVCacheType.NOPE,
            layout,
            inp["num_pages"],
            PAGE_SIZE,
            LKV_DIM,
            KV_PACKING,
        ),
        "rope_spec": SparseMLAKVCacheSpec.create(
            KVCacheType.ROPE,
            layout,
            inp["num_pages"],
            PAGE_SIZE,
            ROPE_DIM,
            KV_PACKING,
        ),
    }
    num_pad = 24

    def _pad(values):
      raw = jax.lax.bitcast_convert_type(values, jnp.uint8)
      tail = jnp.full((num_pad, raw.shape[1]), 0xAB, jnp.uint8)
      return jax.lax.bitcast_convert_type(
          jnp.concatenate([raw, tail], axis=0), values.dtype
      )

    args = (
        jnp.zeros(specs["nope_spec"].shape, specs["nope_spec"].jax_dtype),
        jnp.zeros(specs["rope_spec"].shape, specs["rope_spec"].jax_dtype),
        _pad(inp["kv_c_normed"]),
        _pad(inp["k_pe"]),
        inp["seq_lens"],
        inp["block_tables"],
        inp["query_start_loc"],
    )
    written = 0
    for rank in range(dcp_size):
      got = kv_cache_utils.update_sparse_mla_kv_cache_dcp(
          *args,
          jnp.int32(rank),
          **specs,
          dcp_size=dcp_size,
          interleave_size=interleave_size,
      )
      want = kv_cache_utils.update_sparse_mla_kv_cache_dcp_jax(
          *args,
          jnp.int32(rank),
          **specs,
          dcp_size=dcp_size,
          interleave_size=interleave_size,
      )
      for name, g, w in zip(("nope", "rope"), got, want):
        np.testing.assert_array_equal(
            np.asarray(g),
            np.asarray(w),
            err_msg=f"rank {rank}: {name} cache differs from the jnp writer",
        )
      rows = _decode_rows(got[0], specs["nope_spec"])
      written += int(rows.any(axis=1).sum())
    self.assertEqual(written, inp["num_tokens"])


class ScatterContractTest(parameterized.TestCase):
  """Host-side argument checks; no device work."""

  def test_rope_tensorcore_is_rejected(self):
    spec = SparseMLAKVCacheSpec.create(
        KVCacheType.ROPE,
        KVCacheLayout.TENSORCORE,
        16,
        PAGE_SIZE,
        ROPE_DIM,
        KV_PACKING,
    )
    cache = jnp.zeros(spec.shape, spec.jax_dtype)
    src = jnp.zeros((8, spec.token_bytes), spec.jax_dtype)
    with self.assertRaisesRegex(AssertionError, "TensorCore tiling folds"):
      scatter_utils.scatter_rows(cache, src, jnp.zeros((8,), jnp.int32))

  def test_scatter_requires_addressing_arrays_or_dst_rows(self):
    spec = SparseMLAKVCacheSpec.create(
        KVCacheType.NOPE,
        KVCacheLayout.SPARSECORE,
        16,
        PAGE_SIZE,
        LKV_DIM,
        KV_PACKING,
    )
    cache = jnp.zeros(spec.shape, spec.jax_dtype)
    values = jnp.zeros((8, LKV_DIM), jnp.float8_e4m3fn)
    with self.assertRaisesRegex(AssertionError, "pass dst_rows"):
      scatter.scatter(cache, values, spec)

  def test_scatter_rejects_a_cache_that_does_not_match_its_spec(self):
    spec = SparseMLAKVCacheSpec.create(
        KVCacheType.NOPE,
        KVCacheLayout.SPARSECORE,
        16,
        PAGE_SIZE,
        LKV_DIM,
        KV_PACKING,
    )
    values = jnp.zeros((8, LKV_DIM), jnp.float8_e4m3fn)
    with self.assertRaisesRegex(AssertionError, "does not match spec"):
      scatter.scatter(
          jnp.zeros(spec.shape, jnp.uint8),
          values,
          spec,
          dst_rows=jnp.zeros((8,), jnp.int32),
      )


if __name__ == "__main__":
  absltest.main()
