# Copyright 2026 Google LLC. All Rights Reserved.
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
"""Tests for StreamIndex Top-K kernel."""

# pylint: disable=redefined-outer-name

import functools
from typing import NamedTuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.lightning_indexer.kernel import metadata
from tokamax._src.ops.lightning_indexer.kernel import tc_streamindex_topk
from tokamax._src.ops.lightning_indexer.kernel import tc_topk
from tokamax._src.ops.lightning_indexer.kernel.config import KVLayout
from tokamax._src.ops.lightning_indexer.kernel.streamindex_topk import (
    convert_cache_to_seq_along_lane,
    streamindex_topk,
    tensorcore_topk_select,
)

# =====================================================================

_EE_PAGE_SIZE = 128
_EE_PAGES_PER_SEQ = 32
_EE_BKV_P = 1
_EE_BQ_SZ = 4
_EE_H_I = 4
_EE_D = 128


def streamindex_topk_ref(
    q,
    weights,
    kv,
    block_table,
    T_list,
    S_list,
    cu_q_lens,
    k,
    comp_ratio,
    H_I,
    H_KV,
):
  """Naive NumPy reference implementation for StreamIndex Top-K."""
  num_tokens = q.shape[0]
  expected_topk = np.full((num_tokens, k), -1, dtype=np.int32)
  B = len(T_list)

  for b in range(B):
    T_seq = T_list[b]
    S_total = S_list[b]
    q_start = cu_q_lens[b]

    if T_seq == 0:
      continue

    S_valid = S_total // comp_ratio
    seq_blocks = block_table[b]
    seq_kv = np.concatenate([kv[p] for p in seq_blocks], axis=0)

    naive_scores = np.full((T_seq, max(S_valid, k)), -np.inf, dtype=np.float32)

    for t_idx in range(T_seq):
      global_t = q_start + t_idx
      q_abs_pos = (S_total - T_seq) + t_idx

      for s_idx in range(S_valid):
        if s_idx * comp_ratio > q_abs_pos:
          continue

        score = 0.0
        for h in range(H_I):
          h_kv = h // (H_I // H_KV)
          inner = np.dot(q[global_t, h], seq_kv[s_idx, h_kv])
          score += max(0.0, inner) * weights[global_t, h]

        naive_scores[t_idx, s_idx] = score

    seq_expected = np.argsort(-naive_scores, axis=-1)[:, :k]

    for t_idx in range(T_seq):
      for i in range(k):
        idx = seq_expected[t_idx, i]
        if naive_scores[t_idx, idx] == -np.inf:
          expected_topk[q_start + t_idx, i] = -1
        else:
          expected_topk[q_start + t_idx, i] = idx

  return expected_topk


def _to_byte_lane(x: jax.Array) -> jax.Array:
  """Reinterpret each element of ``x``'s trailing dim as raw bytes."""
  b = jax.lax.bitcast_convert_type(x, jnp.uint8)
  if b.ndim > x.ndim:
    b = b.reshape(*x.shape[:-1], -1)
  return b


def quantize_fp8_ue8m0(x: jax.Array, block_size: int):
  """Block fp8 quantization with UE8M0 (power-of-two) block scales."""
  fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
  *lead, dim = x.shape
  blocked = x.reshape(*lead, dim // block_size, block_size)
  amax = jnp.clip(jnp.max(jnp.abs(blocked), axis=-1, keepdims=True), 1e-4, None)
  scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
  q = (blocked * (1.0 / scale)).astype(jnp.float8_e4m3fn).reshape(x.shape)
  scale = jnp.squeeze(scale, -1).astype(jnp.float8_e8m0fnu)
  return q, scale


def _pack_fp8_cache(keys):
  """fp8-quantize ``[num_pages, page_size, D]`` keys into the uint8 cache.

  Same record layout the tests above build row by row -- ``[fp8 x D | one
  e8m0 scale byte | pad]`` rounded up to a 128-lane group, addressed as
  ``[page, slot // 4, slot % 4, width]`` -- just quantized in one shot.

  Args:
    keys: ``[num_pages, page_size, D]`` float keys.

  Returns:
    The packed ``uint8`` cache, ``[num_pages, page_size // 4, 4, width]``.
  """
  num_pages, page_size, head_dim = keys.shape
  fp8, scale = quantize_fp8_ue8m0(jnp.asarray(keys), head_dim)
  record = jnp.concatenate([_to_byte_lane(fp8), _to_byte_lane(scale)], axis=-1)
  width = -(-record.shape[-1] // 128) * 128
  record = jnp.pad(record, ((0, 0), (0, 0), (0, width - record.shape[-1])))
  return np.asarray(record).reshape(num_pages, page_size // 4, 4, width)


# (B, T_list, S_list, page_size, H_I, H_KV, D, k, comp_ratio, bq_sz, bkv_p,
#  block_table_list)
_NUMERICAL_CORRECTNESS_CASES = (
    # 0. Single sequence, single page (page_indices shape (1,))
    (1, [1], [8], 8, 4, 1, 16, 1024, 1, 1, 16, [[0]]),
    # 1. Single sequence, highly fragmented block table
    (1, [4], [16], 8, 4, 1, 16, 1024, 2, 8, 16, [[2, 0]]),
    # 2. Batched Decode (T=1, B=2)
    (2, [1, 1], [12, 16], 8, 4, 1, 16, 1024, 1, 8, 16, [[1, 3], [0, 2]]),
    # 3. High GQA (8 Query Heads, 2 KV Heads)
    (1, [6], [20], 4, 8, 1, 16, 1024, 1, 8, 32, [[4, 1, 3, 0, 2]]),
    # 4. Multi-Batch Prefill with variable query/sequence lengths
    (2, [5, 3], [16, 16], 8, 4, 1, 16, 1024, 1, 8, 16, [[3, 1], [2, 4]]),
    # 5. Dummy sequences / Padding (B=3, but sequence 1 has 0 tokens)
    (
        3,
        [2, 0, 3],
        [16, 0, 8],
        8,
        4,
        1,
        16,
        1024,
        1,
        8,
        16,
        [[1, 2], [0, 0], [4, 3]],
    ),
    # 6. Mixed batch: sequence 0 has 1 token, sequence 1 has 5 tokens
    (2, [1, 5], [12, 16], 8, 4, 1, 16, 1024, 2, 8, 16, [[1, 3], [2, 0]]),
    (
        1,
        [4],
        [384],
        16,
        4,
        1,
        16,
        1024,
        1,
        8,
        8,
        [[
            13,
            2,
            21,
            7,
            0,
            18,
            5,
            11,
            23,
            1,
            9,
            16,
            3,
            20,
            6,
            14,
            22,
            4,
            10,
            17,
            8,
            15,
            19,
            12,
        ]],
    ),
    # 8. Mixed batch with max_num_seqs == max_num_tokens (B=4, T=4):
    # 2 decode sequences (T=1), 1 chunked-prefill sequence (T=2), and
    # 1 padded inactive sequence (T=0). S_list // comp_ratio > k so
    # _common_path is exercised even with enable_early_exit=True.
    (
        4,
        [1, 1, 2, 0],
        [64, 80, 96, 0],
        128,
        4,
        1,
        128,
        16,
        2,
        8,
        2,
        [[1, 3], [0, 2], [4, 5], [0, 0]],
    ),
)


class StreamIndexTopKTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().generation < 7:
      self.skipTest(
          "StreamIndex Top-K Pallas TPU kernel requires TPU v7 or newer."
      )

  def _require_topk_backend(self, use_tc_topk: bool = False):
    if not use_tc_topk and not tc_topk.has_full_sparsecore_topk():
      self.skipTest("SparseCore top-k requires >= 32 SparseCore subcores.")

  def _ee_pack_cache(self, keys):
    return _pack_fp8_cache(keys)

  def _ee_inputs(self, q_lens, seq_lens, seed):
    """One kernel invocation at ``compression_ratio=1``.

    Every sequence gets a shuffled, disjoint set of physical pages so a bug in
    the page walk cannot hide behind sequential layout.
    """
    rng = np.random.default_rng(seed)
    num_seqs = len(q_lens)
    num_tokens = int(sum(q_lens))
    num_pages = num_seqs * _EE_PAGES_PER_SEQ

    block_table = (
        rng.permutation(num_pages)
        .astype(np.int32)
        .reshape(num_seqs, _EE_PAGES_PER_SEQ)
    )
    keys = rng.standard_normal(
        (num_pages, _EE_PAGE_SIZE, _EE_D), dtype=np.float32
    )

    # `distribution` requires the decode-only sequences to lead the batch.
    num_decodes = 0
    while num_decodes < num_seqs and q_lens[num_decodes] == 1:
      num_decodes += 1

    return {
        "q": rng.standard_normal(
            (num_tokens, _EE_H_I, _EE_D), dtype=np.float32
        ),
        "indexer_weights": (
            rng.uniform(0.25, 1.75, (num_tokens, _EE_H_I)).astype(np.float32)
        ),
        "cache_kv": self._ee_pack_cache(keys),
        "seq_lens": np.asarray(seq_lens, np.int32),
        "page_indices": block_table.reshape(-1),
        "cu_q_lens": np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32),
        "distribution": np.array(
            [num_decodes, num_decodes, num_seqs], np.int32
        ),
    }

  def _ee_both_paths(self, q_lens, seq_lens, k, seed=11, *, tc_topk=False):
    """Run the same batch with the flag off and on."""
    self._require_topk_backend(tc_topk)
    inputs = self._ee_inputs(q_lens, seq_lens, seed)

    def run(enable_early_exit):
      return np.asarray(
          streamindex_topk(
              **{name: jnp.asarray(value) for name, value in inputs.items()},
              k=k,
              compression_ratio=1,
              num_kv_pages_per_block=_EE_BKV_P,
              num_queries_per_block=_EE_BQ_SZ,
              enable_early_exit=enable_early_exit,
              tc_topk=tc_topk,
          )
      )

    return run(False), run(True)

  def _ee_visible_counts(self, q_lens, seq_lens):
    """Causally visible compressed positions per query token.

    At ``compression_ratio=1`` the queries are the tail of their sequence, so
    query ``i`` of a sequence of length ``S`` sits at absolute position
    ``S - q_len + i`` and sees every position up to and including it.
    """
    counts = []
    for q_len, seq_len in zip(q_lens, seq_lens):
      counts.extend(seq_len - q_len + i + 1 for i in range(q_len))
    return counts

  def _ee_assert_same_selection(self, early, baseline):
    """Both paths must select the same positions, in whatever order.

    The scoring path emits winners in ``approx_max_k`` order and the early-exit
    path in increasing KV position. Order is deliberately unspecified, so
    comparing the raw rows would assert something the kernel never promised --
    compare the selected sets.
    """
    assert early.shape == baseline.shape, (early.shape, baseline.shape)
    for token, (row_e, row_b) in enumerate(zip(early, baseline)):
      kept_e = np.sort(row_e[row_e >= 0])
      kept_b = np.sort(row_b[row_b >= 0])
      np.testing.assert_array_equal(
          kept_e,
          kept_b,
          f"token {token}: early exit selected a different "
          "set of positions than the scoring path",
      )
      assert np.all(
          row_e[len(kept_e) :] < 0
      ), f"token {token}: -1 padding is not a suffix: {row_e}"

  def _ee_assert_every_visible_position(self, actual, q_lens, seq_lens, k):
    """With the whole batch under k the answer is exact, not a ranking.

    Every visible position wins, so the expected index set is closed-form and
    the check does not depend on score arithmetic at all.
    """
    for token, n_visible in enumerate(
        self._ee_visible_counts(q_lens, seq_lens)
    ):
      assert n_visible <= k, "test setup: sequence is not short"
      row = actual[token]
      np.testing.assert_array_equal(
          np.sort(row[row >= 0]),
          np.arange(n_visible),
          f"token {token}: not every visible position was reported",
      )
      assert np.all(
          row[n_visible:] < 0
      ), f"token {token}: -1 padding is not a suffix: {row}"

  def _ee_assert_full_rows(self, actual, q_lens, seq_lens, k):
    """Cheap sanity for the long-sequence cases.

    Correctness of the scoring path itself is what the tests above cover; here
    the claim under test is that the flag is inert, so this only rules out
    gross breakage.
    """
    counts = self._ee_visible_counts(q_lens, seq_lens)
    limits = []
    for q_len, seq_len in zip(q_lens, seq_lens):
      limits.extend([seq_len] * q_len)
    for token, (n_visible, limit) in enumerate(zip(counts, limits)):
      assert n_visible > k, "test setup: sequence is not long"
      row = actual[token]
      assert np.all(row >= 0), f"token {token}: padded a full-length row"
      assert (
          row.max() < limit
      ), f"token {token}: index {row.max()} past seq_len {limit}"

  @parameterized.parameters(
      [
          # Small standard case
          (2, 6, 16, 8, 4, 1, 64, 512, 2, 32, 8, [4, 6], [0, 3, 6]),
          # # Single token batch (Decode Phase)
          (1, 1, 16, 4, 2, 1, 32, 512, 1, 16, 8, [10], [0, 1]),
          # Large batch, multiple tokens per sequence (Prefill Phase)
          (
              4,
              20,
              32,
              16,
              8,
              1,
              128,
              512,
              4,
              64,
              4,
              [10, 20, 30, 40],
              [0, 5, 10, 15, 20],
          ),
          # Odd chunk sizes
          (2, 5, 8, 6, 4, 1, 16, 512, 1, 16, 16, [8, 12], [0, 2, 5]),
          # Single head for KV but multiple heads for Queries (MQA/GQA pattern)
          (2, 10, 16, 8, 8, 1, 64, 512, 2, 32, 8, [16, 24], [0, 4, 10]),
          # # High D, High K
          (1, 8, 16, 4, 2, 1, 256, 512, 1, 32, 8, [60], [0, 8]),
          # Mixed batch: sequence 0 has 1 token (decode), sequence 1 has 5 tokens
          # (prefill)
          (2, 6, 16, 8, 4, 1, 64, 512, 2, 32, 8, [4, 6], [0, 1, 6]),
      ],
  )
  def test_streamindex_topk_shape(
      self,
      B,
      num_tokens,
      page_size,
      max_blocks,
      H_I,
      H_KV,
      D,
      k,
      compression_ratio,
      bq_sz,
      bkv_p,
      seq_lens_list,
      cu_q_lens_list,
  ):
    """Tests the shape and basic execution bounds of streamindex_topk."""
    _ = H_KV
    print(f"\n{'-' * 60}")
    print(f"SHAPE TEST: B={B}, Tokens={num_tokens}, k={k}")
    print(f"{'-' * 60}")

    query_projection = jnp.zeros((num_tokens, H_I, D), dtype=jnp.float32)
    indexer_weights = jnp.zeros((num_tokens, H_I), dtype=jnp.float32)

    num_pages = max_blocks * B
    q_lkv_dim = ((D + 127) // 128) * 128
    record_width = q_lkv_dim + (q_lkv_dim // 128)
    width = ((record_width + 127) // 128) * 128
    kv_cache = jnp.zeros((num_pages, page_size // 4, 4, width), dtype=jnp.uint8)

    block_table = jnp.zeros((B, max_blocks), dtype=jnp.int32)
    page_indices = block_table.flatten()
    cu_q_lens = jnp.array(cu_q_lens_list, dtype=jnp.int32)

    print("Inputs:")
    print(f"  - query_projection: {query_projection.shape}")
    print(f"  - kv_cache:         {kv_cache.shape}")
    print(f"  - seq_lens:         {seq_lens_list}")
    print(f"  - cu_q_lens:        {cu_q_lens_list}")

    # Count number of decode sequences (T == 1) at the beginning of the batch
    num_decodes = 0
    while (
        num_decodes < B
        and (cu_q_lens_list[num_decodes + 1] - cu_q_lens_list[num_decodes]) == 1
    ):
      num_decodes += 1
    distribution = (num_decodes, num_decodes, B)
    expected_shape = (num_tokens, k)

    seq_lens = jnp.array(seq_lens_list, dtype=jnp.int32)

    for use_tc_topk in (False, True):
      out_shape_idxs = jax.eval_shape(
          functools.partial(
              streamindex_topk,
              k=k,
              compression_ratio=compression_ratio,
              num_kv_pages_per_block=bkv_p,
              num_queries_per_block=bq_sz,
              tc_topk=use_tc_topk,
          ),
          query_projection,
          indexer_weights,
          kv_cache,
          seq_lens,
          page_indices,
          cu_q_lens,
          distribution,
      )
      assert out_shape_idxs.shape == expected_shape
      assert out_shape_idxs.dtype == jnp.int32

    print("\nOutputs:")
    print(f"  - Expected shape: {expected_shape}, dtype: int32")
    print(
        f"  - Actual shape:   {out_shape_idxs.shape}, dtype:"
        f" {out_shape_idxs.dtype}"
    )

    print("Result: PASS")

  @parameterized.parameters(*_NUMERICAL_CORRECTNESS_CASES)
  def test_streamindex_topk_numerical_correctness(self, *case):
    self._check_numerical_correctness(*case, tc_topk=False)

  @parameterized.parameters(*_NUMERICAL_CORRECTNESS_CASES)
  def test_tensorcore_topk_numerical_correctness(self, *case):
    """Runs the numerical test with the SparseCore-free kernels forced.

    TPUs with >= 32 SparseCore subcores would otherwise never exercise the
    fused TensorCore scoring + top-k kernel.
    """
    self._check_numerical_correctness(*case, tc_topk=True)

  def _check_numerical_correctness(
      self,
      B,
      T_list,
      S_list,
      page_size,
      H_I,
      H_KV,
      D,
      k,
      comp_ratio,
      bq_sz,
      bkv_p,
      block_table_list,
      *,
      tc_topk=False,
  ):
    """Executes randomized input data against a naive NumPy ground truth."""
    self._require_topk_backend(tc_topk)
    print(f"\n{'=' * 60}")
    print(f"BATCHED NUMERICAL TEST (B={B})")
    print(f"{'=' * 60}")

    np.random.seed(42)

    # 1. Setup Random Tensors
    num_tokens = sum(T_list)
    q = np.random.randn(num_tokens, H_I, D).astype(np.float32)
    weights = np.random.uniform(-1.5, 1.5, size=(num_tokens, H_I)).astype(
        np.float32
    )

    # Create a unified physical KV Cache pool large enough for all block indices
    max_physical_page = np.max(block_table_list)
    float32_kv = np.random.randn(
        max_physical_page + 1, page_size, H_KV, D
    ).astype(np.float32)

    # Pack cache using compressor's quantize_fp8_ue8m0 and _to_byte_lane
    q_lkv_dim = ((D + 127) // 128) * 128
    record_width = q_lkv_dim + (q_lkv_dim // 128)
    width = ((record_width + 127) // 128) * 128

    cache_kv = np.zeros(
        (max_physical_page + 1, page_size // 4, 4, width), dtype=np.uint8
    )
    dequantized_kv = np.zeros_like(float32_kv)

    for p in range(max_physical_page + 1):
      for s in range(page_size):
        for h in range(H_KV):
          row_kv = float32_kv[p, s, h]
          # Quantize using D as block size (each head query key has dimension D)
          q_jax, scale_jax = quantize_fp8_ue8m0(jnp.array(row_kv), D)
          q_bytes = np.array(_to_byte_lane(q_jax))
          scale_bytes = np.array(_to_byte_lane(scale_jax))
          dq = np.array(q_jax).astype(np.float32)
          dequantized_kv[p, s, h] = dq * float(scale_jax[0])
          record = np.concatenate([q_bytes, scale_bytes], axis=-1)
          record = np.pad(record, (0, width - record.shape[-1]))

          w_idx = s // 4
          lane_idx = s % 4
          cache_kv[p, w_idx, lane_idx] = record

    block_table = np.array(block_table_list, dtype=np.int32)
    page_indices = block_table.flatten()
    seq_lens = np.array(S_list, dtype=np.int32)
    cu_q_lens = np.concatenate([[0], np.cumsum(T_list)]).astype(np.int32)

    print("Configuration:")
    print(f"  - Tokens total: {num_tokens}, Sequences(B): {B}, K: {k}")
    print(f"  - GQA: {H_I} Query Heads -> {H_KV} KV Heads")
    print(f"  - Compression Ratio: {comp_ratio}")

    print("\nInputs Generated:")
    print(f"  - Query Tensor: {q.shape}")
    print(f"  - KV Cache:     {cache_kv.shape}")
    print(f"  - Block Table:  {block_table.tolist()}")

    # =====================================================================
    # 2. NAIVE COMPUTATION (The Ground Truth)
    # =====================================================================
    print("\n[1/3] Computing exact ground-truth using naive NumPy loops...")
    expected_topk = streamindex_topk_ref(
        q=q,
        weights=weights,
        kv=dequantized_kv,
        block_table=block_table,
        T_list=T_list,
        S_list=S_list,
        cu_q_lens=cu_q_lens,
        k=k,
        comp_ratio=comp_ratio,
        H_I=H_I,
        H_KV=H_KV,
    )

    print("\nGROUND TRUTH (Naive NumPy):")
    print(expected_topk)

    # =====================================================================
    # 3. Pallas KERNEL COMPUTATION
    # =====================================================================
    print("\n[2/3] Executing optimized JAX Kernel (streamindex_pallas_topk)...")
    # Count number of decode sequences (T == 1) at the beginning of the batch
    num_decodes = 0
    while num_decodes < B and T_list[num_decodes] == 1:
      num_decodes += 1
    distribution = (num_decodes, num_decodes, B)

    actual_topk = streamindex_topk(
        q=jnp.array(q),
        indexer_weights=jnp.array(weights),
        cache_kv=jnp.array(cache_kv),
        seq_lens=jnp.array(seq_lens),
        page_indices=jnp.array(page_indices),
        cu_q_lens=jnp.array(cu_q_lens),
        distribution=distribution,
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
        tc_topk=tc_topk,
    )

    actual_topk_np = np.array(actual_topk)

    print("\nACTUAL OUTPUT (JAX/XLA Kernel):")
    print(actual_topk_np)

    # =====================================================================
    # 4. VERIFY
    # =====================================================================
    print("\n[3/3] Verifying absolute correctness...")
    np.testing.assert_array_equal(
        np.sort(actual_topk_np, axis=-1),
        np.sort(expected_topk, axis=-1),
        err_msg="JAX Pallas Kernel Top-K math did not match Naive Ground Truth",
    )
    print(
        "MATCH VERIFIED! JAX kernel handles batches, fragmentation, and dummy"
        " padding.\n"
    )

  @parameterized.parameters(False, True)
  def test_streamindex_topk_quantized(self, tc_topk):
    """Verifies correctness of streamindex_topk on FP8 packed cache."""
    self._require_topk_backend(tc_topk)
    np.random.seed(42)

    T_seq = 4
    S_seq = 2048
    page_size = 16
    H_I = 2
    H_KV = 1
    D = 128
    k = 512
    comp_ratio = 4
    bkv_p = 8  # page_size * bkv_p = 128 (TPU DMA contract of the scores kernel)
    bq_sz = 1

    q = np.random.randn(T_seq, H_I, D).astype(np.float32)
    weights = np.random.uniform(0.5, 1.5, size=(T_seq, H_I)).astype(np.float32)

    S_valid = S_seq // comp_ratio
    # We need enough pages to hold S_valid compressed tokens.
    num_pages = (S_valid + page_size - 1) // page_size
    float32_kv = np.random.randn(num_pages, page_size, H_KV, D).astype(
        np.float32
    )

    # Pack cache using compressor's quantize_fp8_ue8m0 and _to_byte_lane
    width = 256
    cache_kv = np.zeros((num_pages, page_size // 4, 4, width), dtype=np.uint8)

    # Dequantized KV for naive ground truth
    dequantized_kv = np.zeros_like(float32_kv)

    for p in range(num_pages):
      for s in range(page_size):
        for h_kv in range(H_KV):
          row_kv = float32_kv[p, s, h_kv]

          # Quantize using compressor helper
          # block_size = 128 since we want 1 scale factor for D=128
          q_jax, scale_jax = quantize_fp8_ue8m0(jnp.array(row_kv), 128)

          # Convert to bytes
          q_bytes = np.array(_to_byte_lane(q_jax))
          scale_bytes = np.array(_to_byte_lane(scale_jax))

          # Store dequantized version exactly as hardware sees it for accurate
          # validation
          dq = np.array(q_jax).astype(np.float32)
          dequantized_kv[p, s, h_kv] = dq * float(scale_jax[0])

          record = np.concatenate([q_bytes, scale_bytes], axis=-1)
          record = np.pad(record, (0, width - record.shape[-1]))

          w_idx = s // 4
          lane_idx = s % 4
          cache_kv[p, w_idx, lane_idx] = record

    # Compute expected top-k using exact dequantized keys
    seq_kv = np.concatenate(
        [dequantized_kv[p] for p in range(num_pages)], axis=0
    )

    naive_scores = np.full((T_seq, max(S_valid, k)), -np.inf, dtype=np.float32)

    for t_idx in range(T_seq):
      for s_idx in range(S_valid):
        score = 0.0
        for h in range(H_I):
          h_kv = h // (H_I // H_KV)
          inner = np.dot(q[t_idx, h], seq_kv[s_idx, h_kv])
          score += max(0.0, inner) * weights[t_idx, h]
        naive_scores[t_idx, s_idx] = score

    expected_topk = np.argsort(-naive_scores, axis=-1)[:, :k]
    for t_idx in range(T_seq):
      for i in range(k):
        idx = expected_topk[t_idx, i]
        if naive_scores[t_idx, idx] == -np.inf:
          expected_topk[t_idx, i] = -1

    # Pallas parameters
    actual_topk = streamindex_topk(
        q=jnp.array(q),
        indexer_weights=jnp.array(weights),
        cache_kv=jnp.array(cache_kv),
        seq_lens=jnp.array([S_seq], dtype=jnp.int32),
        page_indices=jnp.arange(num_pages, dtype=jnp.int32),
        cu_q_lens=jnp.array([0, T_seq], dtype=jnp.int32),
        distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
        tc_topk=tc_topk,
    )

    actual_topk_np = np.array(actual_topk)

    np.testing.assert_array_equal(
        np.sort(actual_topk_np, axis=-1),
        np.sort(expected_topk, axis=-1),
    )

  # =====================================================================
  # Early exit
  # =====================================================================
  # `enable_early_exit=True` must change cost, never the answer. Once a
  # sequence's compressed length is <= k, top-k selects every visible position,
  # so the scores decide nothing and the scoring kernel can be skipped outright.
  # The guard is batch-wide, so the tests below pin both halves of that: the
  # shortcut agrees with the scoring path when the whole batch is short, and is
  # inert the moment any single sequence is not.

  # `page_size * bkv_p` is 128, the TPU DMA contract of the scores kernel.

  @parameterized.parameters(False, True)
  def test_streamindex_topk_early_exit_all_short(self, tc_topk):
    """Whole batch under k: the global fast path answers on its own."""
    q_lens, seq_lens, k = (1, 1, 1), (64, 96, 128), 512
    baseline, early = self._ee_both_paths(q_lens, seq_lens, k, tc_topk=tc_topk)

    self._ee_assert_every_visible_position(early, q_lens, seq_lens, k)
    self._ee_assert_same_selection(early, baseline)

  @parameterized.parameters(False, True)
  def test_streamindex_topk_early_exit_no_short_is_inert(self, tc_topk):
    """Every sequence over k: nothing is skipped, so nothing may change.

    Here the flag must be inert, which is a stronger claim than agreeing as
    sets -- no row took the shortcut, so the rows are bit-identical.
    """
    q_lens, seq_lens, k = (1, 1, 4), (2048, 3072, 4096), 128
    baseline, early = self._ee_both_paths(q_lens, seq_lens, k, tc_topk=tc_topk)

    self._ee_assert_full_rows(early, q_lens, seq_lens, k)
    np.testing.assert_array_equal(early, baseline)

  @parameterized.parameters(False, True)
  def test_streamindex_topk_early_exit_is_batch_wide(self, tc_topk):
    """The guard is batch-wide, not per sequence.

    Two short decodes alongside one long prefill: `jnp.max(seq_lens)` is over
    k, so every token goes back through the scoring kernel including the short
    ones, and the flag must be bit-for-bit inert.
    """
    q_lens, seq_lens, k = (1, 1, 4), (64, 96, 4096), 512
    baseline, early = self._ee_both_paths(q_lens, seq_lens, k, tc_topk=tc_topk)

    np.testing.assert_array_equal(early, baseline)

  @parameterized.parameters(False, True)
  def test_streamindex_topk_early_exit_keeps_causal_mask(self, tc_topk):
    """The shortcut is still causal: a prefill token sees only its past.

    The whole sequence fits in k, so a bug that returned `[0, k)` instead of
    `[0, position]` would still look plausible -- this is what catches it.
    """
    q_lens, seq_lens, k = (4,), (64,), 256
    _, early = self._ee_both_paths(q_lens, seq_lens, k, tc_topk=tc_topk)

    self._ee_assert_every_visible_position(early, q_lens, seq_lens, k)
    # Four query tokens at the tail of a 64-long sequence sit at absolute
    # positions 60..63, so they keep 61..64 positions respectively.
    kept = [int(np.count_nonzero(row >= 0)) for row in early]
    assert kept == [61, 62, 63, 64], kept

  def test_pallas_smem_oom_when_unclamped(self):
    """Verifies that Pallas compilation of _scores_kernel fails with SMEM OOM."""
    orig_compute_metadata = metadata.compute_metadata

    def _mock_compute_metadata(*args, **kwargs):
      res = orig_compute_metadata(*args, **kwargs)
      # Return metadata arrays of length 100,000.
      # In Pallas, 100,000 steps require ~2.4 MB of SMEM.
      return metadata.MetadataRef.create(
          num_steps=res.num_steps,
          batch_tile_idx=jnp.zeros((100000,), dtype=jnp.int32),
          bq_idx=jnp.zeros((100000,), dtype=jnp.int32),
          bkv_idx=jnp.zeros((100000,), dtype=jnp.int32),
      )

    num_seqs = 32
    seq_len = 128
    num_pages_per_seq = seq_len // 128
    total_pages = num_seqs * num_pages_per_seq
    head_dim = 128
    width = 256

    q = jnp.zeros((num_seqs * seq_len, 1, head_dim), dtype=jnp.float32)
    weights = jnp.ones((num_seqs * seq_len, 1), dtype=jnp.float32)
    cache_kv = jnp.zeros((total_pages, 32, 4, width), dtype=np.uint8)
    seq_lens = jnp.full((num_seqs,), seq_len, dtype=jnp.int32)
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)
    cu_q_lens = jnp.arange(
        0, (num_seqs + 1) * seq_len, seq_len, dtype=jnp.int32
    )
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

    with (
        mock.patch.object(metadata, "compute_metadata", _mock_compute_metadata),
        self.assertRaises(Exception) as exc_info,
    ):
      streamindex_topk(
          q=q,
          indexer_weights=weights,
          cache_kv=cache_kv,
          seq_lens=seq_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=distribution,
          k=512,
          compression_ratio=1,
          num_kv_pages_per_block=1,
          num_queries_per_block=1,
      )
    err_str = str(exc_info.exception).lower()
    assert (
        "smem" in err_str
    ), f"Expected SMEM OOM error, got: {exc_info.exception}"

  @parameterized.parameters(False, True)
  def test_streamindex_topk_buffer_count(self, tc_topk):
    """Verifies configurable buffer_count parameter behavior."""
    self._require_topk_backend(tc_topk)
    B = 2
    T = 4
    S = 16
    page_size = 128
    H_I = 4
    D = 16
    k = 16
    comp_ratio = 1
    bq_sz = 8
    bkv_p = 2

    total_tokens = B * T
    max_blocks = 2
    num_pages = max_blocks * B
    q_lkv_dim = ((D + 127) // 128) * 128
    record_width = q_lkv_dim + (q_lkv_dim // 128)
    width = ((record_width + 127) // 128) * 128

    rng = np.random.default_rng(42)
    q = jnp.array(
        rng.standard_normal((total_tokens, H_I, D)), dtype=jnp.float32
    )
    weights = jnp.array(
        rng.standard_normal((total_tokens, H_I)), dtype=jnp.float32
    )
    cache_kv = jnp.zeros((num_pages, page_size // 4, 4, width), dtype=jnp.uint8)
    seq_lens = jnp.array([S] * B, dtype=jnp.int32)
    page_indices = jnp.arange(num_pages, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, T, 2 * T], dtype=jnp.int32)
    distribution = jnp.array([0, 0, B], dtype=jnp.int32)

    # 1. Custom integer buffer_count
    topk_int = streamindex_topk(
        q=q,
        indexer_weights=weights,
        cache_kv=cache_kv,
        seq_lens=seq_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
        buffer_count=2,
        tc_topk=tc_topk,
    )
    assert topk_int.shape == (total_tokens, k)

    # 2. Custom tuple buffer_count
    topk_tuple = streamindex_topk(
        q=q,
        indexer_weights=weights,
        cache_kv=cache_kv,
        seq_lens=seq_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
        buffer_count=(2, 2, 2),
        tc_topk=tc_topk,
    )
    assert topk_tuple.shape == (total_tokens, k)
    np.testing.assert_array_equal(np.array(topk_int), np.array(topk_tuple))

    # 3. Invalid buffer_count length raises ValueError
    with self.assertRaisesRegex(ValueError, "buffer_count must be a 3-tuple"):
      streamindex_topk(
          q=q,
          indexer_weights=weights,
          cache_kv=cache_kv,
          seq_lens=seq_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=distribution,
          k=k,
          compression_ratio=comp_ratio,
          num_kv_pages_per_block=bkv_p,
          num_queries_per_block=bq_sz,
          buffer_count=(2, 2),
          tc_topk=tc_topk,
      )

  @parameterized.parameters(False, True)
  def test_streamindex_topk_return_scores_and_validation(self, tc_topk):
    self._require_topk_backend(tc_topk)
    q_dtype = jnp.float8_e4m3fn
    kv_dtype = jnp.uint8
    b = 2
    t = 4
    s = 128
    total_tokens = b * t
    h_i = 4
    d_i = 128
    k = 16
    comp_ratio = 1
    page_size = 64
    bkv_p = 2
    bq_sz = 4
    width = 256

    num_pages = b * (s // page_size)
    q = jnp.ones((total_tokens, h_i, d_i), dtype=q_dtype)
    weights = jnp.ones((total_tokens, h_i), dtype=jnp.float32)
    cache_kv = jnp.zeros((num_pages, page_size // 4, 4, width), dtype=kv_dtype)
    seq_lens = jnp.array([s] * b, dtype=jnp.int32)
    page_indices = jnp.arange(num_pages, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, t, 2 * t], dtype=jnp.int32)
    distribution = jnp.array([0, 0, b], dtype=jnp.int32)

    # 1. return_scores=True
    idxs, scores = streamindex_topk(
        q=q,
        indexer_weights=weights,
        cache_kv=cache_kv,
        seq_lens=seq_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
        return_scores=True,
        tc_topk=tc_topk,
    )
    assert idxs.shape == (total_tokens, k)
    assert scores.shape == (total_tokens, k)

    # 2. cp_size < 1 raises ValueError
    with self.assertRaisesRegex(ValueError, "cp_size must be >= 1"):
      streamindex_topk(
          q=q,
          indexer_weights=weights,
          cache_kv=cache_kv,
          seq_lens=seq_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=distribution,
          k=k,
          compression_ratio=comp_ratio,
          num_kv_pages_per_block=bkv_p,
          num_queries_per_block=bq_sz,
          cp_size=0,
          tc_topk=tc_topk,
      )

    # 3. interleave_size not multiple of compression_ratio
    with self.assertRaisesRegex(
        ValueError, "must be a multiple of compression_ratio"
    ):
      streamindex_topk(
          q=q,
          indexer_weights=weights,
          cache_kv=cache_kv,
          seq_lens=seq_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=distribution,
          k=k,
          compression_ratio=2,
          num_kv_pages_per_block=bkv_p,
          num_queries_per_block=bq_sz,
          cp_size=2,
          interleave_size=3,
          tc_topk=tc_topk,
      )

    # 4. enable_early_exit with cp_size > 1
    with self.assertRaisesRegex(
        NotImplementedError,
        "enable_early_exit is not supported with cp_size > 1",
    ):
      streamindex_topk(
          q=q,
          indexer_weights=weights,
          cache_kv=cache_kv,
          seq_lens=seq_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=distribution,
          k=k,
          compression_ratio=comp_ratio,
          num_kv_pages_per_block=bkv_p,
          num_queries_per_block=bq_sz,
          enable_early_exit=True,
          cp_size=2,
          tc_topk=tc_topk,
      )

    # 5. return_scores with enable_early_exit
    with self.assertRaisesRegex(
        NotImplementedError,
        "return_scores is not supported with enable_early_exit",
    ):
      streamindex_topk(
          q=q,
          indexer_weights=weights,
          cache_kv=cache_kv,
          seq_lens=seq_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=distribution,
          k=k,
          compression_ratio=comp_ratio,
          num_kv_pages_per_block=bkv_p,
          num_queries_per_block=bq_sz,
          enable_early_exit=True,
          return_scores=True,
          tc_topk=tc_topk,
      )


CHUNK_DECODE_REQ_BATCH_SIZE = 4

_HAS = KVLayout.HEAD_ALONG_SUBLANE
_SAL = KVLayout.SEQ_ALONG_LANE


class _ChunkInputs(NamedTuple):
  q: np.ndarray
  weights: np.ndarray
  cache_kv: np.ndarray
  # None when the page size is not a multiple of the 128-lane register
  # width, where the SEQ_ALONG_LANE layout does not exist.
  cache_kv_lane: np.ndarray | None
  # [pages, page_size, 1, D], exactly the values the kernel reads back out
  # of the fp8 cache, for the naive reference below.
  dequantized_kv: np.ndarray
  block_table: np.ndarray
  page_indices: np.ndarray


def _fragmented_pages(num_seqs, pages_per_seq=4):
  """Shuffled page assignment, so logical order never matches physical."""
  pages = list(range(num_seqs * pages_per_seq))
  np.random.default_rng(0).shuffle(pages)
  return tuple(
      tuple(pages[i * pages_per_seq : (i + 1) * pages_per_seq])
      for i in range(num_seqs)
  )


def _align_to(x: int, a: int) -> int:
  return ((x + a - 1) // a) * a


@functools.cache
def _chunk_inputs(T_list, block_table_list, page_size, H_I, D) -> _ChunkInputs:
  """Randomized fp8-packed cache and queries for one chunked geometry.

  Only MQA (one KV head) is built: chunking indexes the schedule, which does
  not depend on the KV head count, and every case below uses H_KV = 1.

  Cached because several parameterizations share a geometry and the packing
  runs one JAX quantize per physical page.
  """
  np.random.seed(42)
  num_tokens = sum(T_list)
  q = np.random.randn(num_tokens, H_I, D).astype(np.float32)
  weights = np.random.uniform(-1.5, 1.5, size=(num_tokens, H_I)).astype(
      np.float32
  )

  num_pages = max(p for row in block_table_list for p in row) + 1
  float32_kv = np.random.randn(num_pages, page_size, D).astype(np.float32)

  q_lkv_dim = _align_to(D, 128)
  width = _align_to(q_lkv_dim + q_lkv_dim // 128, 128)
  cache_kv = np.zeros((num_pages, page_size // 4, 4, width), dtype=np.uint8)
  dequantized_kv = np.zeros((num_pages, page_size, 1, D), dtype=np.float32)
  for p in range(num_pages):
    quant, scale = quantize_fp8_ue8m0(jnp.array(float32_kv[p]), D)
    dequantized_kv[p, :, 0] = np.array(quant).astype(np.float32) * np.array(
        scale
    ).astype(np.float32)
    record = np.concatenate(
        [
            np.array(_to_byte_lane(quant)),
            np.array(_to_byte_lane(scale)),
        ],
        axis=-1,
    )
    record = np.pad(record, ((0, 0), (0, width - record.shape[-1])))
    # Token `s` of a page lives at [s // 4, s % 4], the same layout the
    # packing loop in the numerical test above builds element by element.
    cache_kv[p] = record.reshape(page_size // 4, 4, width)

  return _ChunkInputs(
      q=q,
      weights=weights,
      cache_kv=cache_kv,
      cache_kv_lane=(
          np.array(convert_cache_to_seq_along_lane(jnp.array(cache_kv), D))
          if page_size % 128 == 0
          else None
      ),
      dequantized_kv=dequantized_kv,
      block_table=np.array(block_table_list, dtype=np.int32),
      page_indices=np.array(
          [p for row in block_table_list for p in row], dtype=np.int32
      ),
  )


class ChunkedStreamIndexTopKTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().generation < 7:
      self.skipTest(
          "StreamIndex Top-K Pallas TPU kernel requires TPU v7 or newer."
      )

  def _require_topk_backend(self, use_tc_topk: bool = False):
    if not use_tc_topk and not tc_topk.has_full_sparsecore_topk():
      self.skipTest("SparseCore top-k requires >= 32 SparseCore subcores.")

  def _verify_padding_at_end(self, actual_topk_np):
    """Verifies that -1 is only at the end of the innermost dimension."""
    is_minus_one = (actual_topk_np == -1).astype(np.int32)
    violations = np.diff(is_minus_one, axis=-1) < 0
    assert not np.any(
        violations
    ), "Padding (-1) is not at the end of the innermost dimension"

  def _chunk_call_kwargs(
      self,
      inputs: _ChunkInputs,
      T_list,
      S_list,
      k,
      comp_ratio,
      bq_sz,
      bkv_p,
      kv_layout,
      chunk_tokens,
      return_scores=False,
      *,
      tc_topk=False,
      require_sparsecore: bool = True,
  ):
    """Call kwargs for one case; unchunked when `chunk_tokens` is None."""
    if require_sparsecore:
      self._require_topk_backend(tc_topk)
    # Leading single-token sequences are decodes, the same rule the numerical
    # test above uses to build its distribution.
    num_decodes = 0
    while num_decodes < len(T_list) and T_list[num_decodes] == 1:
      num_decodes += 1

    cache_kv = inputs.cache_kv_lane if kv_layout == _SAL else inputs.cache_kv
    assert cache_kv is not None, "SEQ_ALONG_LANE needs page_size % 128 == 0"
    return dict(
        q=jnp.array(inputs.q),
        indexer_weights=jnp.array(inputs.weights),
        cache_kv=jnp.array(cache_kv),
        seq_lens=jnp.array(np.array(S_list, dtype=np.int32)),
        page_indices=jnp.array(inputs.page_indices),
        cu_q_lens=jnp.array(
            np.concatenate([[0], np.cumsum(T_list)]).astype(np.int32)
        ),
        distribution=(num_decodes, num_decodes, len(T_list)),
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
        # The early-exit fast path returns `arange` without entering Pallas,
        # which would make every comparison below vacuous.
        enable_early_exit=False,
        return_scores=return_scores,
        kv_layout=kv_layout,
        chunk_tokens=chunk_tokens,
        tc_topk=tc_topk,
    )

  @parameterized.product(
      T_list=[(64, 64), (40, 88)],
      chunk_tokens=[32, 64],
      tc_topk=[False, True],
  )
  def test_chunked_topk_matches_the_naive_reference(  # pylint: disable=invalid-name
      self, T_list, chunk_tokens, tc_topk
  ):
    """The chunked kernel reproduces the ground truth wherever the cuts land.

    `TestChunkedPipelining` below only compares chunked against unchunked, so
    a fault shared by both paths would survive it. This one holds the chunked
    output against absolute truth instead, over the two cut positions that
    matter: 64/64 lands every cut on a sequence boundary, 40/88 lands every
    cut inside a sequence, so the seam blocks are recomputed and each chunk
    has to rebuild the per-sequence position bookkeeping from its own start.
    """
    page_size, H_I, H_KV, D = 128, 4, 1, 128
    comp_ratio, bq_sz, bkv_p = 1, 8, 2
    S_list = (512, 512)
    # k == the number of candidate positions, so the reference selects every
    # visible column and no tie can make two correct answers differ.
    k = 512

    inputs = _chunk_inputs(T_list, _fragmented_pages(2), page_size, H_I, D)
    kwargs = self._chunk_call_kwargs(
        inputs,
        T_list,
        S_list,
        k,
        comp_ratio,
        bq_sz,
        bkv_p,
        _HAS,
        chunk_tokens,
        tc_topk=tc_topk,
    )

    hlo = streamindex_topk.lower(**kwargs).as_text()
    assert "_scheduling_group_id" in hlo, (
        f"chunk_tokens={chunk_tokens} did not chunk: no scheduling group id "
        "reached the HLO, so the kernel took the single-pass fallback and "
        "this test would prove nothing"
    )

    actual = np.array(streamindex_topk(**kwargs))
    expected = streamindex_topk_ref(
        q=inputs.q,
        weights=inputs.weights,
        kv=inputs.dequantized_kv,
        block_table=inputs.block_table,
        T_list=list(T_list),
        S_list=list(S_list),
        cu_q_lens=np.concatenate([[0], np.cumsum(T_list)]).astype(np.int32),
        k=k,
        comp_ratio=comp_ratio,
        H_I=H_I,
        H_KV=H_KV,
    )

    self._verify_padding_at_end(actual)
    np.testing.assert_array_equal(
        np.sort(actual, axis=-1),
        np.sort(expected, axis=-1),
        err_msg=(
            f"chunked (chunk_tokens={chunk_tokens}) top-k did not match "
            "the naive ground truth"
        ),
    )

  _CHUNKING_MATCHES_UNCHUNKED_CASES = (
      # 1. Single sequence, highly fragmented block table. Every chunk
      # holds the same tile, so this is the case chunking is most likely
      # to get right; it guards the simple path against regressions.
      (
          "single_seq_fragmented",
          1,
          (64,),
          (512,),
          128,
          4,
          128,
          128,
          1,
          8,
          2,
          _fragmented_pages(1),
          16,
          _HAS,
      ),
      # 2. Batched decode (T=1 for every sequence).
      (
          "batched_decode",
          8,
          (1,) * 8,
          (512,) * 8,
          128,
          4,
          128,
          128,
          1,
          8,
          2,
          _fragmented_pages(8),
          4,
          _HAS,
      ),
      # 3. High GQA (8 query heads, 1 KV head).
      (
          "high_gqa",
          2,
          (32, 32),
          (512, 512),
          128,
          8,
          128,
          128,
          1,
          8,
          2,
          _fragmented_pages(2),
          16,
          _HAS,
      ),
      # 4. Multi-batch prefill, sequence boundary inside a chunk.
      (
          "boundary_inside_a_chunk",
          2,
          (40, 88),
          (512, 512),
          128,
          4,
          128,
          128,
          1,
          8,
          2,
          _fragmented_pages(2),
          32,
          _HAS,
      ),
      # 5. Multi-batch prefill, sequence boundary exactly on a chunk
      # edge.
      (
          "boundary_on_a_chunk_edge",
          2,
          (64, 64),
          (512, 512),
          128,
          4,
          128,
          128,
          1,
          8,
          2,
          _fragmented_pages(2),
          32,
          _HAS,
      ),
      # 6. Dummy sequence / padding: sequence 1 has no query tokens.
      (
          "empty_sequence_in_the_middle",
          3,
          (32, 0, 32),
          (512, 0, 512),
          128,
          4,
          128,
          128,
          1,
          8,
          2,
          _fragmented_pages(3),
          16,
          _HAS,
      ),
      # 7. Mixed batch: eight decodes then one prefill. This runs three
      # passes, each chunked off its own schedule. Eight decodes (not
      # four) makes the prefill pass's first chunk hold fewer query
      # blocks than the chunks after it, which is what a missing step
      # offset needs in order to show up at all.
      (
          "mixed_decode_and_prefill",
          9,
          (1,) * 8 + (120,),
          (512,) * 9,
          128,
          4,
          128,
          128,
          1,
          8,
          2,
          _fragmented_pages(9),
          32,
          _HAS,
      ),
      # 8. Compression ratio 2, so the candidate count is half the
      # sequence length.
      (
          "compression_ratio_2",
          2,
          (32, 32),
          (512, 512),
          128,
          4,
          128,
          128,
          2,
          8,
          2,
          _fragmented_pages(2),
          16,
          _HAS,
      ),
      # 9. k above the candidate count, so every candidate is selected
      # and the output is insensitive to score values. Weak on its own,
      # but it is the regime the `enable_early_exit` fast path targets.
      (
          "k_above_candidate_count",
          2,
          (32, 32),
          (512, 512),
          128,
          4,
          128,
          1024,
          1,
          8,
          2,
          _fragmented_pages(2),
          16,
          _HAS,
      ),
      # 10. SEQ_ALONG_LANE, boundary inside a chunk.
      (
          "seq_along_lane_boundary_inside",
          2,
          (40, 88),
          (512, 512),
          128,
          4,
          128,
          128,
          1,
          8,
          2,
          _fragmented_pages(2),
          32,
          _SAL,
      ),
      # 11. SEQ_ALONG_LANE with a larger query block and a wider KV
      # block. `num_bq` is the only schedule term that varies per chunk,
      # because `metadata.py` tiles `num_bkv` across chunks, so the
      # query block size sets how many schedule rows each chunk owns.
      (
          "seq_along_lane_bq16_bkv4",
          2,
          (64, 64),
          (512, 512),
          128,
          4,
          128,
          128,
          1,
          16,
          4,
          _fragmented_pages(2),
          32,
          _SAL,
      ),
      # 12. SEQ_ALONG_LANE over 256-token pages, so a page spans two
      # lane windows instead of one.
      (
          "seq_along_lane_page256",
          2,
          (64, 64),
          (512, 512),
          256,
          4,
          128,
          128,
          1,
          8,
          1,
          _fragmented_pages(2, pages_per_seq=2),
          32,
          _SAL,
      ),
      # 13. A chunk edge that cuts a query block in half.
      (
          "partial_query_block",
          2,
          (20, 44),
          (512, 512),
          128,
          4,
          128,
          128,
          1,
          8,
          2,
          _fragmented_pages(2),
          16,
          _HAS,
      ),
      # 14. Ten decodes, so `decode_batch_end` is 10 // 4 * 4 = 8 and
      # the ragged remainder pass (seq_batch_size=1) covers sequences
      # [8, 10).
      (
          "ragged_decode_remainder",
          11,
          (1,) * 10 + (118,),
          (512,) * 11,
          128,
          4,
          128,
          128,
          1,
          8,
          2,
          _fragmented_pages(11),
          32,
          _HAS,
      ),
  )
  # fmt: on

  @parameterized.named_parameters(*_CHUNKING_MATCHES_UNCHUNKED_CASES)
  def test_chunking_matches_unchunked(self, *case):
    """Runs one configuration chunked and unchunked, compares selections."""
    self._check_chunking_matches_unchunked(*case)

  @parameterized.named_parameters(*_CHUNKING_MATCHES_UNCHUNKED_CASES)
  def test_tc_topk_chunking_matches_unchunked(self, *case):
    """Runs one configuration chunked and unchunked with tc_topk=True."""
    self._check_chunking_matches_unchunked(*case, tc_topk=True)

  def _check_chunking_matches_unchunked(  # pylint: disable=invalid-name
      self,
      B,
      T_list,
      S_list,
      page_size,
      H_I,
      D,
      k,
      comp_ratio,
      bq_sz,
      bkv_p,
      block_table_list,
      chunk_tokens,
      kv_layout,
      *,
      tc_topk: bool = False,
  ):
    """Runs one configuration chunked and unchunked, compares selections."""
    assert len(T_list) == B and len(S_list) == B
    assert (page_size * bkv_p) % 256 == 0, (
        f"bkv_sz ({page_size} * {bkv_p}) must be a multiple of 256 for TPU"
        " bfloat16 DMA alignment."
    )
    assert chunk_tokens % CHUNK_DECODE_REQ_BATCH_SIZE == 0, (
        f"chunk_tokens={chunk_tokens} must be a multiple of"
        f" {CHUNK_DECODE_REQ_BATCH_SIZE}, or a cut can land inside a"
        " batched decode group"
    )
    assert sum(T_list) % chunk_tokens == 0, (
        f"chunk_tokens={chunk_tokens} does not divide {sum(T_list)}"
        " tokens; the kernel would silently fall back to a single"
        " unchunked pass"
    )

    inputs = _chunk_inputs(T_list, block_table_list, page_size, H_I, D)
    common = (inputs, T_list, S_list, k, comp_ratio, bq_sz, bkv_p, kv_layout)
    chunked_kwargs = self._chunk_call_kwargs(
        *common, chunk_tokens, tc_topk=tc_topk
    )

    # The per-chunk scheduling annotation is the only externally visible
    # proof that more than one chunk was emitted. Without it this would be
    # comparing the unchunked kernel against itself.
    hlo = streamindex_topk.lower(**chunked_kwargs).as_text()
    assert "_scheduling_group_id" in hlo, (
        "no scheduling group annotation: the kernel ran a single unchunked"
        " pass and this comparison would pass trivially"
    )

    chunked = np.array(streamindex_topk(**chunked_kwargs))
    unchunked = np.array(
        streamindex_topk(
            **self._chunk_call_kwargs(*common, None, tc_topk=tc_topk)
        )
    )

    self._verify_padding_at_end(chunked)
    # Top-k returns its indices in unspecified order, so compare them as
    # sets. The scores are distinct by construction, so the set is exact.
    np.testing.assert_array_equal(
        np.sort(chunked, axis=-1),
        np.sort(unchunked, axis=-1),
        err_msg=(
            f"chunk_tokens={chunk_tokens} changed the selection for"
            f" T_list={T_list}, bq_sz={bq_sz},"
            f" kv_layout={kv_layout}; chunking the token axis must be"
            " exact"
        ),
    )

  @parameterized.parameters(False, True)
  def test_chunking_matches_unchunked_with_return_scores(self, tc_topk):
    """`return_scores` adds a second output concatenated per chunk.

    `need_scores` gates a `topk_scores_list` accumulated once per chunk
    and concatenated at the end, and DCP reaches the same code through
    `cp_size > 1`. Every case in the matrix leaves `return_scores` off, so
    without this the per-chunk concatenation is never exercised; the
    existing unchunked scores test only ever runs with one chunk.
    """
    T_list, S_list, chunk_tokens = (40, 88), (512, 512), 32
    inputs = _chunk_inputs(T_list, _fragmented_pages(2), 128, 4, 128)
    common = (inputs, T_list, S_list, 128, 1, 8, 2, _HAS)

    chunked_kwargs = self._chunk_call_kwargs(
        *common, chunk_tokens, return_scores=True, tc_topk=tc_topk
    )
    hlo = streamindex_topk.lower(**chunked_kwargs).as_text()
    assert "_scheduling_group_id" in hlo, (
        "no scheduling group annotation: the kernel ran a single unchunked"
        " pass and this comparison would pass trivially"
    )

    chunked_idx, chunked_scores = streamindex_topk(**chunked_kwargs)
    unchunked_idx, unchunked_scores = streamindex_topk(
        **self._chunk_call_kwargs(
            *common, None, return_scores=True, tc_topk=tc_topk
        )
    )
    chunked_idx = np.array(chunked_idx)
    chunked_scores = np.array(chunked_scores)
    unchunked_idx = np.array(unchunked_idx)
    unchunked_scores = np.array(unchunked_scores)

    self._verify_padding_at_end(chunked_idx)
    # Order is unspecified, so sort each side by its own indices and
    # compare the pairs. Sorting the scores independently would lose the
    # pairing.
    order_c = np.argsort(chunked_idx, axis=-1)
    order_u = np.argsort(unchunked_idx, axis=-1)
    np.testing.assert_array_equal(
        np.take_along_axis(chunked_idx, order_c, axis=-1),
        np.take_along_axis(unchunked_idx, order_u, axis=-1),
        err_msg="chunking changed the selected indices under return_scores",
    )
    np.testing.assert_array_equal(
        np.take_along_axis(chunked_scores, order_c, axis=-1),
        np.take_along_axis(unchunked_scores, order_u, axis=-1),
        err_msg="chunking changed the scores attached to the same indices",
    )

  @parameterized.parameters(False, True)
  def test_indivisible_chunk_tokens_runs_a_single_pass(self, tc_topk):
    """A `chunk_tokens` that does not divide the bucket disables chunking.

    The kernel falls back to one pass instead of raising, so a case that
    picks such a size stops covering the chunked path with no visible
    signal. That is exactly how an earlier version of this coverage became
    vacuous, so the rule the cases above depend on is pinned here.
    """
    T_list, S_list, chunk_tokens = (64, 64), (512, 512), 48
    assert sum(T_list) % chunk_tokens != 0, "pick an indivisible size"

    inputs = _chunk_inputs(T_list, _fragmented_pages(2), 128, 4, 128)
    hlo = streamindex_topk.lower(
        **self._chunk_call_kwargs(
            inputs,
            T_list,
            S_list,
            128,
            1,
            8,
            2,
            _HAS,
            chunk_tokens,
            tc_topk=tc_topk,
        )
    ).as_text()
    assert "_scheduling_group_id" not in hlo, (
        "an indivisible chunk_tokens emitted more than one chunk; the"
        " fallback that the cases above rely on has changed"
    )

  @parameterized.product(
      kv_layout=[_HAS, _SAL],
      block_sizes=[((1, 128, 128), (32, 32, 16)), (8, 2)],
      verify_q_len=[6, 7],
      tc_topk=[False, True],
  )
  def test_static_verify_segment_matches_mixed_pass(
      self, verify_q_len, block_sizes, kv_layout, tc_topk
  ):
    """Verify windows of a fixed length (1 + K) in the distribution's middle

    segment run the static query-length pass (`uniform_q_len`). They must
    select the same positions, with the same scores, as the same step with the
    windows in the mixed segment, and match the naive reference. Covers the 6-
    and 7-row query blocks (padded to 8 rows per head) that only this pass
    runs, in both cache layouts and with the serving block sizes.
    """
    bq_sz, bkv_p = block_sizes
    page_size, H_I, D, comp_ratio = 128, 4, 128, 1
    T_list = (1, 1, verify_q_len, verify_q_len, verify_q_len, 3, 12)
    B = len(T_list)
    S = 2 * page_size
    S_list = (S,) * B
    # k == the number of candidate positions: every visible column is selected,
    # so no tie at the top-k boundary can make two correct answers differ.
    k = S
    inputs = _chunk_inputs(T_list, _fragmented_pages(B, 2), page_size, H_I, D)
    kwargs = self._chunk_call_kwargs(
        inputs,
        T_list,
        S_list,
        k,
        comp_ratio,
        bq_sz,
        bkv_p,
        kv_layout,
        None,
        return_scores=True,
        tc_topk=tc_topk,
    )
    assert kwargs["distribution"] == (
        2,
        2,
        B,
    ), "windows start on the mixed pass"
    mixed_idx, mixed_scores = (np.array(a) for a in streamindex_topk(**kwargs))
    static_idx, static_scores = (
        np.array(a)
        for a in streamindex_topk(**{
            **kwargs,
            "distribution": (2, 5, B),
            "uniform_q_len": verify_q_len,
        })
    )

    expected = streamindex_topk_ref(
        q=inputs.q,
        weights=inputs.weights,
        kv=inputs.dequantized_kv,
        block_table=inputs.block_table,
        T_list=list(T_list),
        S_list=list(S_list),
        cu_q_lens=np.concatenate([[0], np.cumsum(T_list)]).astype(np.int32),
        k=k,
        comp_ratio=comp_ratio,
        H_I=H_I,
        H_KV=1,
    )
    self._verify_padding_at_end(static_idx)
    np.testing.assert_array_equal(
        np.sort(static_idx, axis=-1),
        np.sort(expected, axis=-1),
        err_msg="static-q verify pass did not match the naive ground truth",
    )

    # Same positions with bit-identical scores as the mixed pass, row by row,
    # and rows outside the segment untouched by the split.
    def by_position(idx, scores):
      rows = []
      for i, s in zip(idx, scores):
        valid = i >= 0
        order = np.argsort(i[valid], kind="stable")
        rows.append((i[valid][order], s[valid][order]))
      return rows

    for row, (mi, ms), (si, ss) in zip(
        range(len(mixed_idx)),
        by_position(mixed_idx, mixed_scores),
        by_position(static_idx, static_scores),
    ):
      np.testing.assert_array_equal(
          si, mi, err_msg=f"row {row}: positions differ"
      )
      np.testing.assert_array_equal(ss, ms, err_msg=f"row {row}: scores differ")

  def _assert_fused_matches_reference(
      self, ids_ref, ids_fzd, sc_ref=None, sc_fzd=None
  ):
    """Checks ascending-id fused output matches reference ids and scores."""
    ids_ref_np = np.asarray(ids_ref)
    ids_fzd_np = np.asarray(ids_fzd)
    self._verify_padding_at_end(ids_ref_np)
    self._verify_padding_at_end(ids_fzd_np)
    sc_ref_np = None if sc_ref is None else np.asarray(sc_ref)
    sc_fzd_np = None if sc_fzd is None else np.asarray(sc_fzd)
    for r in range(ids_ref_np.shape[0]):
      n_v = int(np.sum(ids_ref_np[r] >= 0))
      self.assertEqual(
          int(np.sum(ids_fzd_np[r] >= 0)), n_v, msg=f"count, row {r}"
      )
      if n_v > 0:
        order = np.argsort(ids_ref_np[r, :n_v])
        np.testing.assert_array_equal(
            ids_fzd_np[r, :n_v],
            ids_ref_np[r, :n_v][order],
            err_msg=f"ids, row {r}",
        )
        if sc_ref_np is not None and sc_fzd_np is not None:
          np.testing.assert_array_equal(
              sc_fzd_np[r, :n_v],
              sc_ref_np[r, :n_v][order],
              err_msg=f"scores, row {r}",
          )
      np.testing.assert_array_equal(
          ids_fzd_np[r, n_v:], -1, err_msg=f"padding, row {r}"
      )
      if sc_ref_np is not None and sc_fzd_np is not None:
        np.testing.assert_array_equal(
            sc_fzd_np[r, n_v:],
            sc_ref_np[r, n_v:],
            err_msg=f"pad scores, row {r}",
        )

  def test_tensorcore_topk_select_prefix_bucketing_and_edge_widths(self):
    """Verifies tensorcore_topk_select on prefix buckets and edge widths."""
    rng = np.random.default_rng(77)
    score_f32 = rng.standard_normal((8, 4096), dtype=np.float32)
    score_f32[:, 1500:] = 1e30  # Poison beyond row_lengths.
    score_bits = jax.lax.bitcast_convert_type(jnp.asarray(score_f32), jnp.int32)
    row_lens = jnp.array(
        [0, 1, 17, 512, 1024, 1025, 1499, 1500], dtype=jnp.int32
    )
    ids_full, scores_full = tensorcore_topk_select(
        score_bits,
        512,
        row_lens,
        enable_prefix_bucketing=False,
        return_scores=True,
    )
    ids_bkt, scores_bkt = tensorcore_topk_select(
        score_bits,
        512,
        row_lens,
        enable_prefix_bucketing=True,
        return_scores=True,
    )
    ids_only = tensorcore_topk_select(
        score_bits,
        512,
        row_lens,
        enable_prefix_bucketing=True,
        return_scores=False,
    )
    np.testing.assert_array_equal(np.asarray(ids_bkt), np.asarray(ids_full))
    np.testing.assert_array_equal(
        np.asarray(scores_bkt), np.asarray(scores_full)
    )
    np.testing.assert_array_equal(np.asarray(ids_only), np.asarray(ids_full))

    # Single-bucket width (<= 1024), width < k padding, and k > 512 with pad_w.
    for width, k_val, lens in (
        (1024, 128, [0, 37, 512, 1024]),
        (64, 128, [0, 17, 64, 64]),
        (10240, 640, [0, 500, 4097, 10240]),
    ):
      edge_f32 = rng.standard_normal((4, width), dtype=np.float32)
      edge_bits = jax.lax.bitcast_convert_type(jnp.asarray(edge_f32), jnp.int32)
      edge_lens = jnp.array(lens, dtype=jnp.int32)
      e_full, e_sc_full = tensorcore_topk_select(
          edge_bits,
          k_val,
          edge_lens,
          enable_prefix_bucketing=False,
          return_scores=True,
      )
      e_bkt, e_sc_bkt = tensorcore_topk_select(
          edge_bits,
          k_val,
          edge_lens,
          enable_prefix_bucketing=True,
          return_scores=True,
      )
      np.testing.assert_array_equal(np.asarray(e_bkt), np.asarray(e_full))
      np.testing.assert_array_equal(np.asarray(e_sc_bkt), np.asarray(e_sc_full))

  @parameterized.named_parameters(
      ("continuous_16512", False, 16512),
      ("continuous_65536", False, 65536),
      ("ties_16512", True, 16512),
      ("ties_65536", True, 65536),
  )
  def test_tensorcore_topk_select_wide_exactness(
      self, use_ties: bool, max_row_len: int
  ):
    """Verifies 256K-scale tournament exactness on continuous and tied scores."""
    rng = np.random.default_rng(78)
    if use_ties:
      arr = rng.integers(-6, 7, size=(8, 65536)).astype(np.float32)
      arr[:, ::17] = -np.inf
    else:
      arr = rng.standard_normal((8, 65536), dtype=np.float32)
      arr[:, 1000:1512] += 10.0
      arr[:, 200] = -0.0
      arr[:, 205] = +0.0
    wide_bits = jax.lax.bitcast_convert_type(jnp.asarray(arr), jnp.int32)
    wide_lens = jnp.array(
        [
            0,
            1,
            512,
            4096,
            4097,
            16512,
            min(32769, max_row_len),
            max_row_len,
        ],
        dtype=jnp.int32,
    )
    w_ids_full, w_sc_full = tensorcore_topk_select(
        wide_bits,
        512,
        wide_lens,
        enable_prefix_bucketing=False,
        return_scores=True,
    )
    w_ids_bkt, w_sc_bkt = tensorcore_topk_select(
        wide_bits,
        512,
        wide_lens,
        enable_prefix_bucketing=True,
        return_scores=True,
    )
    np.testing.assert_array_equal(np.asarray(w_ids_bkt), np.asarray(w_ids_full))
    np.testing.assert_array_equal(np.asarray(w_sc_bkt), np.asarray(w_sc_full))

  def test_tensorcore_topk_select_relu_zero_ties_and_subbatch_tail(self):
    """Verifies ReLU-zero ties and VMEM sub-batch scan + tail handling."""
    rng = np.random.default_rng(79)
    # 64-row VMEM-tiled scan (with one active and one empty 32-row sub-batch).
    relu_chunk = np.zeros((64, 32768), dtype=np.float32)
    for r in range(32):
      pos_cols = rng.choice(32768, size=120 + r * 5, replace=False)
      relu_chunk[r, pos_cols] = rng.uniform(
          0.1, 5.0, size=pos_cols.size
      ).astype(np.float32)
      relu_chunk[r, (r * 13) % 512] = -0.0
    relu_bits = jax.lax.bitcast_convert_type(jnp.asarray(relu_chunk), jnp.int32)
    relu_lens = jnp.concatenate([
        jnp.full((32,), 32768, dtype=jnp.int32),
        jnp.zeros((32,), dtype=jnp.int32),
    ])
    r_ids_full, r_sc_full = tensorcore_topk_select(
        relu_bits,
        512,
        relu_lens,
        enable_prefix_bucketing=False,
        return_scores=True,
    )
    r_ids_bkt, r_sc_bkt = tensorcore_topk_select(
        relu_bits,
        512,
        relu_lens,
        enable_prefix_bucketing=True,
        return_scores=True,
    )
    r_ids_only = tensorcore_topk_select(
        relu_bits,
        512,
        relu_lens,
        enable_prefix_bucketing=True,
        return_scores=False,
    )
    np.testing.assert_array_equal(np.asarray(r_ids_bkt), np.asarray(r_ids_full))
    np.testing.assert_array_equal(np.asarray(r_sc_bkt), np.asarray(r_sc_full))
    np.testing.assert_array_equal(
        np.asarray(r_ids_only), np.asarray(r_ids_full)
    )
    np.testing.assert_array_equal(np.asarray(r_ids_bkt[32:]), -1)

    # Non-multiple-of-32 batch_rows (45 = 32 scan + 13 tail).
    r45_ids_bkt, r45_sc_bkt = tensorcore_topk_select(
        relu_bits[:45],
        512,
        relu_lens[:45],
        enable_prefix_bucketing=True,
        return_scores=True,
    )
    r45_ids_only = tensorcore_topk_select(
        relu_bits[:45],
        512,
        relu_lens[:45],
        enable_prefix_bucketing=True,
        return_scores=False,
    )
    np.testing.assert_array_equal(
        np.asarray(r45_ids_bkt), np.asarray(r_ids_full[:45])
    )
    np.testing.assert_array_equal(
        np.asarray(r45_sc_bkt), np.asarray(r_sc_full[:45])
    )
    np.testing.assert_array_equal(
        np.asarray(r45_ids_only), np.asarray(r_ids_full[:45])
    )

  def test_tensorcore_decode_only_and_max_score_sublanes(self):
    """Verifies decode_only schedules and max_score_sublanes across TPUs."""
    page_size, h_i, d_dim, comp_ratio, k = 128, 4, 128, 4, 128
    b_sz = 8
    pages_per_seq = 16
    inputs = _chunk_inputs(
        (1,) * b_sz,
        _fragmented_pages(b_sz, pages_per_seq),
        page_size,
        h_i,
        d_dim,
    )
    s_list = (1024, 1536, 2048, 1024, 1536, 0, 0, 0)
    base_kwargs = self._chunk_call_kwargs(
        inputs,
        (1, 1, 1, 1, 1, 0, 0, 0),
        s_list,
        k,
        comp_ratio,
        (4, 4, 4),
        (1, 1, 1),
        _HAS,
        None,
        require_sparsecore=False,
    )
    base_kwargs["q"] = jnp.asarray(inputs.q)
    base_kwargs["indexer_weights"] = jnp.asarray(inputs.weights)
    base_kwargs["cu_q_lens"] = jnp.array(
        [0, 1, 2, 3, 4, 5, 5, 5, 5], dtype=jnp.int32
    )
    base_kwargs["distribution"] = jnp.array([5, 5, 5], dtype=jnp.int32)

    out_tc_full = np.asarray(
        streamindex_topk(**base_kwargs, topk_mode="tensorcore_full")
    )
    out_tc_bounded = np.asarray(
        streamindex_topk(**(
            base_kwargs
            | dict(
                topk_mode="tensorcore_bounded",
                decode_only=True,
                max_score_sublanes=8,
                enable_early_exit=True,
            )
        ))
    )
    out_tc_fused = np.asarray(
        streamindex_topk(**(
            base_kwargs
            | dict(
                topk_mode="tensorcore_fused",
                decode_only=True,
                max_score_sublanes=8,
                enable_early_exit=True,
            )
        ))
    )
    np.testing.assert_array_equal(out_tc_bounded[:5], out_tc_full[:5])
    self._assert_fused_matches_reference(out_tc_full, out_tc_fused)
    np.testing.assert_array_equal(out_tc_bounded[5:], -1)
    if tc_topk.has_full_sparsecore_topk():
      out_sc = np.asarray(
          streamindex_topk(**base_kwargs, topk_mode="sparsecore")
      )
      for r in range(5):
        np.testing.assert_array_equal(
            np.sort(out_tc_bounded[r]), np.sort(out_sc[r])
        )

    # Remainder decode pass (max_num_seqs=6, decode_req_batch_size=4).
    rem_t = (1,) * 6
    rem_s = (1024, 1536, 2048, 800, 1280, 960)
    rem_inputs = _chunk_inputs(
        rem_t, _fragmented_pages(6, 16), page_size, h_i, d_dim
    )
    rem_kw = self._chunk_call_kwargs(
        rem_inputs,
        rem_t,
        rem_s,
        k,
        comp_ratio,
        (4, 4, 4),
        (1, 1, 1),
        _HAS,
        None,
        require_sparsecore=False,
    )
    rem_ref = streamindex_topk(**rem_kw, topk_mode="tensorcore_full")
    rem_fzd = streamindex_topk(
        **(rem_kw | dict(topk_mode="tensorcore_fused", decode_only=True))
    )
    self._assert_fused_matches_reference(rem_ref, rem_fzd)

    # Single-sequence batch (max_num_seqs < decode_req_batch_size).
    one_inputs = _chunk_inputs(
        (1,), _fragmented_pages(1, 16), page_size, h_i, d_dim
    )
    one_kw = self._chunk_call_kwargs(
        one_inputs,
        (1,),
        (1024,),
        k,
        comp_ratio,
        (4, 4, 4),
        (1, 1, 1),
        _HAS,
        None,
        require_sparsecore=False,
    )
    one_ref = streamindex_topk(**one_kw, topk_mode="tensorcore_full")
    one_fzd = streamindex_topk(
        **(one_kw | dict(topk_mode="tensorcore_fused", decode_only=True))
    )
    self._assert_fused_matches_reference(one_ref, one_fzd)

    # decode_only=True with uniform_q_len (distribution[0] == 0, [1] == [2]).
    uni_t = (4, 4, 4, 4, 4, 0, 0, 0)
    uni_inputs = _chunk_inputs(
        uni_t, _fragmented_pages(b_sz, pages_per_seq), page_size, h_i, d_dim
    )
    uni_kw = self._chunk_call_kwargs(
        uni_inputs,
        uni_t,
        s_list,
        k,
        comp_ratio,
        (4, 4, 4),
        (1, 1, 1),
        _HAS,
        None,
        require_sparsecore=False,
    )
    uni_kw["distribution"] = jnp.array([0, 5, 5], dtype=jnp.int32)
    uni_kw["uniform_q_len"] = 4
    uni_ref = streamindex_topk(**uni_kw, topk_mode="tensorcore_full")
    uni_fzd = streamindex_topk(
        **(uni_kw | dict(topk_mode="tensorcore_fused", decode_only=True))
    )
    self._assert_fused_matches_reference(uni_ref, uni_fzd)

  def test_max_score_sublanes_clamps_schedule(self):
    """Verifies max_score_sublanes clamps runtime seq_lens on cp_size == 1."""
    page_size, h_i, d_dim, comp_ratio, k = 128, 4, 128, 4, 128
    pages_per_seq, num_seqs = 8, 2
    rng = np.random.default_rng(101)
    block_table = np.asarray(_fragmented_pages(num_seqs, pages_per_seq))
    num_pages = int(block_table.max()) + 1
    keys = rng.uniform(0.1, 0.5, (num_pages, page_size, d_dim)).astype(
        np.float32
    )
    # Put dominant keys in seq 0 pages 4..5 (compressed [512, 768), past the
    # cap) and seq 1 page 2 (compressed [256, 384), below the compressed cap
    # 512) so missing runtime clamping or missing compression_ratio scaling
    # diverges from the capped reference.
    for p in block_table[0, 4:6]:
      keys[p] = 10.0
    keys[block_table[1, 2]] = 10.0
    cache_kv = jnp.asarray(_pack_fp8_cache(keys))
    q = jnp.ones((num_seqs, h_i, d_dim), dtype=jnp.bfloat16)
    weights = jnp.ones((num_seqs, h_i), dtype=jnp.float32)
    cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32)
    distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
    common_kw = dict(
        q=q,
        indexer_weights=weights,
        cache_kv=cache_kv,
        page_indices=jnp.asarray(block_table.reshape(-1), dtype=jnp.int32),
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=(1, 1, 1),
        num_queries_per_block=(1, 1, 1),
        decode_req_batch_size=1,
        return_scores=True,
    )

    ref_ids, ref_sc = streamindex_topk(**(
        common_kw
        | dict(
            seq_lens=jnp.array([2048, 1536], dtype=jnp.int32),
            topk_mode="tensorcore_full",
        )
    ))
    over_cap_seq_lens = jnp.array([3072, 1536], dtype=jnp.int32)
    bnd_kw = common_kw | dict(
        seq_lens=over_cap_seq_lens,
        topk_mode="tensorcore_bounded",
        decode_only=True,
        max_score_sublanes=4,
    )
    with mock.patch.object(
        metadata, "compute_metadata", wraps=metadata.compute_metadata
    ) as mock_meta:
      streamindex_topk.lower(**bnd_kw)
    self.assertTrue(mock_meta.called)
    self.assertIsNot(mock_meta.call_args.kwargs["seq_lens"], over_cap_seq_lens)

    bnd_ids, bnd_sc = streamindex_topk(**bnd_kw)
    fzd_ids, fzd_sc = streamindex_topk(
        **(bnd_kw | dict(topk_mode="tensorcore_fused"))
    )
    np.testing.assert_array_equal(np.asarray(bnd_ids), np.asarray(ref_ids))
    np.testing.assert_array_equal(np.asarray(bnd_sc), np.asarray(ref_sc))
    self._assert_fused_matches_reference(ref_ids, fzd_ids, ref_sc, fzd_sc)
    bnd_seq1 = np.asarray(bnd_ids[1])
    self.assertTrue(np.all((bnd_seq1 >= 256) & (bnd_seq1 < 384)))

  @parameterized.named_parameters(
      ("head_along_sublane", _HAS),
      ("seq_along_lane", KVLayout.SEQ_ALONG_LANE),
  )
  def test_tensorcore_fused_mixed_ragged_short_rows_and_cp(self, layout):
    """Verifies fused Top-K on mixed ragged batches, short rows, and CP."""
    page_size, h_i, d_dim, comp_ratio, k = 128, 4, 128, 4, 128
    mix_b = 4
    mix_pages_per_seq = 16
    mix_t_list = (1, 4, 140, 0)
    mix_s_list = (1024, 1536, 2048, 0)
    mix_inputs = _chunk_inputs(
        mix_t_list,
        _fragmented_pages(mix_b, mix_pages_per_seq),
        page_size,
        h_i,
        d_dim,
    )
    mix_kw = self._chunk_call_kwargs(
        mix_inputs,
        mix_t_list,
        mix_s_list,
        k,
        comp_ratio,
        (4, 4, 128),
        (2, 2, 2),
        layout,
        None,
        require_sparsecore=False,
    )
    mix_kw["q"] = jnp.asarray(mix_inputs.q, dtype=jnp.bfloat16)
    mix_kw["indexer_weights"] = jnp.asarray(mix_inputs.weights)
    mix_kw["cu_q_lens"] = jnp.array([0, 1, 5, 145, 145], dtype=jnp.int32)
    mix_kw["distribution"] = jnp.array([1, 2, 3], dtype=jnp.int32)
    mix_kw["uniform_q_len"] = 4

    ids_full, sc_full = streamindex_topk(
        **(mix_kw | dict(topk_mode="tensorcore_full", return_scores=True))
    )
    ids_bnd, sc_bnd = streamindex_topk(
        **(mix_kw | dict(topk_mode="tensorcore_bounded", return_scores=True))
    )
    ids_fzd, sc_fzd = streamindex_topk(
        **(mix_kw | dict(topk_mode="tensorcore_fused", return_scores=True))
    )
    ids_fzd_only = np.asarray(
        streamindex_topk(
            **(mix_kw | dict(topk_mode="tensorcore_fused", return_scores=False))
        )
    )
    np.testing.assert_array_equal(np.asarray(ids_bnd), np.asarray(ids_full))
    np.testing.assert_array_equal(np.asarray(sc_bnd), np.asarray(sc_full))
    np.testing.assert_array_equal(ids_fzd_only, np.asarray(ids_fzd))
    self._assert_fused_matches_reference(ids_full, ids_fzd, sc_full, sc_fzd)

    # Short active rows (0 < eff_row_len < k): seq0=32, seq1=72..75,
    # seq2=41..180.
    short_kw = dict(
        mix_kw,
        seq_lens=jnp.array([128, 300, 720, 0], dtype=jnp.int32),
        return_scores=True,
    )
    s_full, s_sc_full = streamindex_topk(
        **short_kw, topk_mode="tensorcore_full"
    )
    s_fzd, s_sc_fzd = streamindex_topk(**short_kw, topk_mode="tensorcore_fused")
    self._assert_fused_matches_reference(s_full, s_fzd, s_sc_full, s_sc_fzd)

    # All-zero tie breaking on short rows (< k) yields [0..n_v-1, -1..].
    zero_short_kw = dict(
        short_kw,
        q=jnp.zeros_like(short_kw["q"]),
        indexer_weights=jnp.zeros_like(short_kw["indexer_weights"]),
        return_scores=False,
    )
    z_short = np.asarray(
        streamindex_topk(**zero_short_kw, topk_mode="tensorcore_fused")
    )
    expected_row0 = np.concatenate(
        [np.arange(32, dtype=np.int32), np.full((k - 32,), -1, dtype=np.int32)]
    )
    np.testing.assert_array_equal(z_short[0], expected_row0)

    # Context-parallelism (cp_size=2) with non-zero scores across both ranks.
    for rank in (0, 1):
      cp_kw = dict(
          mix_kw,
          cp_size=2,
          cp_rank=rank,
          interleave_size=8,
      )
      cp_full, cp_sc_full = streamindex_topk(
          **(cp_kw | dict(topk_mode="tensorcore_full", return_scores=True))
      )
      cp_bnd, cp_sc_bnd = streamindex_topk(
          **(cp_kw | dict(topk_mode="tensorcore_bounded", return_scores=True))
      )
      cp_fzd, cp_sc_fzd = streamindex_topk(
          **(cp_kw | dict(topk_mode="tensorcore_fused", return_scores=True))
      )
      cp_fzd_noscore = np.asarray(
          streamindex_topk(**(
              cp_kw | dict(topk_mode="tensorcore_fused", return_scores=False)
          ))
      )
      np.testing.assert_array_equal(np.asarray(cp_bnd), np.asarray(cp_full))
      np.testing.assert_array_equal(
          np.asarray(cp_sc_bnd), np.asarray(cp_sc_full)
      )
      np.testing.assert_array_equal(cp_fzd_noscore, np.asarray(cp_fzd))
      self._assert_fused_matches_reference(
          cp_full, cp_fzd, cp_sc_full, cp_sc_fzd
      )

  def test_tensorcore_fused_chunked_early_exit_and_vmem_fallback(self):
    """Verifies chunking, early-exit fast path, bq_sz capping, and fallback."""
    page_size, h_i, d_dim, comp_ratio, k = 128, 4, 128, 4, 128
    # 1. Chunked (num_chunks=2) tensorcore_bounded and tensorcore_fused.
    t_list = (12, 20)
    s_list = (1024, 1536)
    inputs = _chunk_inputs(
        t_list, _fragmented_pages(2, 16), page_size, h_i, d_dim
    )
    chunk_kw = self._chunk_call_kwargs(
        inputs,
        t_list,
        s_list,
        k,
        comp_ratio,
        (4, 8, 8),
        (2, 2, 2),
        _HAS,
        16,
        require_sparsecore=False,
    )
    c_full, c_sc_full = streamindex_topk(
        **(chunk_kw | dict(topk_mode="tensorcore_full", return_scores=True))
    )
    c_bnd, c_sc_bnd = streamindex_topk(
        **(chunk_kw | dict(topk_mode="tensorcore_bounded", return_scores=True))
    )
    c_fzd, c_sc_fzd = streamindex_topk(
        **(chunk_kw | dict(topk_mode="tensorcore_fused", return_scores=True))
    )
    c_fzd_only = np.asarray(
        streamindex_topk(**(
            chunk_kw | dict(topk_mode="tensorcore_fused", return_scores=False)
        ))
    )
    np.testing.assert_array_equal(np.asarray(c_bnd), np.asarray(c_full))
    np.testing.assert_array_equal(np.asarray(c_sc_bnd), np.asarray(c_sc_full))
    np.testing.assert_array_equal(c_fzd_only, np.asarray(c_fzd))
    self._assert_fused_matches_reference(c_full, c_fzd, c_sc_full, c_sc_fzd)

    # 2. enable_early_exit=True hitting _fast_path (max(eff_row_lengths) <= k).
    ee_t = (4, 8)
    ee_s = (160, 480)
    ee_inputs = _chunk_inputs(
        ee_t, _fragmented_pages(2, 4), page_size, h_i, d_dim
    )
    ee_kw = self._chunk_call_kwargs(
        ee_inputs,
        ee_t,
        ee_s,
        k,
        comp_ratio,
        (4, 4, 4),
        (1, 1, 1),
        _HAS,
        None,
        require_sparsecore=False,
    )
    ee_kw["enable_early_exit"] = True
    ee_full = np.asarray(streamindex_topk(**ee_kw, topk_mode="tensorcore_full"))
    ee_bnd = np.asarray(
        streamindex_topk(**ee_kw, topk_mode="tensorcore_bounded")
    )
    ee_fzd = np.asarray(streamindex_topk(**ee_kw, topk_mode="tensorcore_fused"))
    np.testing.assert_array_equal(ee_bnd, ee_full)
    np.testing.assert_array_equal(ee_fzd, ee_full)

    # 3. Non-multiple-of-64 uniform_q_len=80 triggers bq_sz=64 cap (128 pages,
    # stays in tensorcore_fused -> ascending ids) and falls back to
    # tensorcore_bounded (256 pages, > 0.75 * eff_vmem_limit -> score-sorted
    # ids) under an 8 MiB VMEM limit.
    for pps, expect_fused_ascending in ((128, True), (256, False)):
      pages = (tuple(p % 4 for p in range(pps)),)
      cap_inputs = _chunk_inputs((80,), pages, page_size, h_i, d_dim)
      cap_kw = self._chunk_call_kwargs(
          cap_inputs,
          (80,),
          (1024,),
          k,
          comp_ratio,
          (1, 128, 128),
          (2, 2, 2),
          _HAS,
          None,
          require_sparsecore=False,
      )
      cap_kw["distribution"] = jnp.array([0, 1, 1], dtype=jnp.int32)
      cap_kw["uniform_q_len"] = 80
      cap_kw["vmem_limit_bytes"] = 8 << 20
      cap_ref = streamindex_topk(**cap_kw, topk_mode="tensorcore_full")
      cap_fzd = streamindex_topk(**cap_kw, topk_mode="tensorcore_fused")
      if expect_fused_ascending:
        self._assert_fused_matches_reference(cap_ref, cap_fzd)
      else:
        np.testing.assert_array_equal(np.asarray(cap_fzd), np.asarray(cap_ref))

  def test_tensorcore_topk_validation_errors(self):
    """Verifies topk_mode, tc_topk, and max_score_sublanes validation."""
    page_size, h_i, d_dim, comp_ratio, k = 128, 4, 128, 4, 128
    inputs = _chunk_inputs(
        (1, 1), _fragmented_pages(2, 8), page_size, h_i, d_dim
    )
    base_kwargs = self._chunk_call_kwargs(
        inputs,
        (1, 1),
        (512, 512),
        k,
        comp_ratio,
        (4, 4, 4),
        (2, 2, 2),
        _HAS,
        None,
        require_sparsecore=False,
    )
    with self.assertRaisesRegex(ValueError, "Unsupported topk_mode"):
      streamindex_topk(**base_kwargs, topk_mode="invalid_mode")

    for bad_mode in (
        "tensorcore_full",
        "tensorcore_bounded",
        "tensorcore_fused",
    ):
      with self.assertRaisesRegex(ValueError, "Cannot set both tc_topk=True"):
        streamindex_topk(
            **(base_kwargs | dict(tc_topk=True, topk_mode=bad_mode))
        )

    # A cap that ends inside a 2-page KV block, a non-positive cap, or
    # combining max_score_sublanes with cp_size > 1.
    for bad_cap in (3, 0):
      with self.assertRaisesRegex(ValueError, "max_score_sublanes"):
        streamindex_topk(**(
            base_kwargs
            | dict(
                topk_mode="tensorcore_fused",
                num_kv_pages_per_block=(2, 2, 2),
                max_score_sublanes=bad_cap,
            )
        ))
    with self.assertRaisesRegex(
        ValueError, "max_score_sublanes is not supported with cp_size > 1"
    ):
      streamindex_topk(**(
          base_kwargs
          | dict(
              topk_mode="tensorcore_fused",
              max_score_sublanes=4,
              cp_size=2,
              interleave_size=8,
          )
      ))

  @parameterized.named_parameters(
      # (decode_req_batch_size == num_seqs, return_scores): r_blk and blocks.
      ("three_row_blocks", 24, False),  # r_blk = 8, 3 row blocks.
      ("two_row_blocks_with_scores", 32, True),  # r_blk = 16, 2 row blocks.
      ("one_row_block", 32, False),  # r_blk = 32, 1 row block.
  )
  def test_fused_topk_batched_decode_row_blocks(self, num_seqs, return_scores):
    """Each fused row block of a batched decode pass uses its own row lengths.

    A batched decode pass holds one row per sequence, and the fused epilogue
    splits its `seq_batch_size` rows into `r_blk`-row blocks. Lengths grow with
    the sequence index here, so a block that used another block's lengths would
    select from the wrong prefix and disagree with the reference.
    """
    page_size, h_i, d_dim, comp_ratio, k = 128, 4, 128, 4, 128
    pages_per_seq = 8
    t_list = (1,) * num_seqs
    # Row i sees 160 + 24 * i compressed positions: all > k and all distinct.
    s_list = tuple(640 + 96 * i for i in range(num_seqs))
    inputs = _chunk_inputs(
        t_list,
        _fragmented_pages(num_seqs, pages_per_seq),
        page_size,
        h_i,
        d_dim,
    )
    kw = self._chunk_call_kwargs(
        inputs,
        t_list,
        s_list,
        k,
        comp_ratio,
        (4, 4, 4),
        (1, 1, 1),
        _HAS,
        None,
        return_scores=return_scores,
        require_sparsecore=False,
    )
    kw["decode_req_batch_size"] = num_seqs
    ref = streamindex_topk(**kw, topk_mode="tensorcore_full")
    fused = streamindex_topk(**kw, topk_mode="tensorcore_fused")
    if return_scores:
      ref_ids, ref_sc = ref
      fused_ids, fused_sc = fused
      self._assert_fused_matches_reference(ref_ids, fused_ids, ref_sc, fused_sc)
    else:
      self._assert_fused_matches_reference(ref, fused)


_SEL_PAGE_SIZE = 128
_SEL_H_I = 4
_SEL_D = 128
# Score scale of each key class: class `c` keys are `_SEL_CLASS_SCALES[c]`
# times one shared direction that every query head is aligned with, so class
# scores are strictly ordered by scale and identical within a class.
_SEL_CLASS_SCALES = (4.0, 3.0, 2.0, 1.0)


def _sel_key_classes(pattern: str, num_positions: int) -> np.ndarray:
  """Key class of every compressed position for a tie-heavy pattern."""
  j = np.arange(num_positions)
  if pattern == "all_equal":
    return np.zeros_like(j)
  if pattern == "periodic":
    return (j % 3) + 1
  if pattern == "lane_skew":
    # The best class only ever lands in lane 0 of each 128-wide tile, and the
    # next class has far more members than `k`.
    return np.where(j % 128 == 0, 0, 1 + (j // 128) % 3)
  if pattern == "blocks":
    # Long runs of equal keys, alternating between two classes.
    return np.where((j // 200) % 2 == 0, 2, 1)
  raise ValueError(pattern)


class TensorCoreSelectTopKTest(parameterized.TestCase):
  """The threshold-select Top-K must match the streaming Top-K bit for bit."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().generation < 7:
      self.skipTest("Requires TPU v7 or newer.")

  def _inputs(self, q_lens, seq_lens, *, pattern=None, seed=0):
    """Builds one batch at `compression_ratio=1` on shuffled pages."""
    rng = np.random.default_rng(seed)
    num_seqs = len(q_lens)
    num_tokens = int(sum(q_lens))
    pages_per_seq = -(-max(seq_lens) // _SEL_PAGE_SIZE)
    num_pages = num_seqs * pages_per_seq
    block_table = (
        rng.permutation(num_pages)
        .astype(np.int32)
        .reshape(num_seqs, pages_per_seq)
    )
    if pattern is None:
      keys = rng.standard_normal(
          (num_pages, _SEL_PAGE_SIZE, _SEL_D), dtype=np.float32
      )
      q = rng.standard_normal((num_tokens, _SEL_H_I, _SEL_D), dtype=np.float32)
    else:
      direction = np.abs(rng.standard_normal(_SEL_D)).astype(np.float32)
      direction /= np.linalg.norm(direction)
      classes = _sel_key_classes(pattern, pages_per_seq * _SEL_PAGE_SIZE)
      scales = np.asarray(_SEL_CLASS_SCALES, np.float32)[classes]
      logical = scales[:, None] * direction[None, :]
      keys = np.zeros((num_pages, _SEL_PAGE_SIZE, _SEL_D), np.float32)
      for s in range(num_seqs):
        for p in range(pages_per_seq):
          keys[block_table[s, p]] = logical[
              p * _SEL_PAGE_SIZE : (p + 1) * _SEL_PAGE_SIZE
          ]
      q = np.broadcast_to(direction, (num_tokens, _SEL_H_I, _SEL_D)).copy()
    num_decodes = 0
    while num_decodes < num_seqs and q_lens[num_decodes] == 1:
      num_decodes += 1
    cache = _pack_fp8_cache(keys)
    return dict(
        q=jnp.asarray(q),
        indexer_weights=jnp.asarray(
            rng.uniform(0.25, 1.75, (num_tokens, _SEL_H_I)).astype(np.float32)
        ),
        cache_kv=jnp.asarray(cache),
        seq_lens=jnp.asarray(seq_lens, jnp.int32),
        page_indices=jnp.asarray(block_table.reshape(-1)),
        cu_q_lens=jnp.asarray(
            np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
        ),
        distribution=jnp.asarray([num_decodes, num_decodes, num_seqs]),
    )

  def _run(self, inputs, *, k, topk_impl, bkv_p, bq, return_scores=True):
    out = tc_streamindex_topk.tc_streamindex_topk(
        **inputs,
        k=k,
        compression_ratio=1,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq,
        decode_req_batch_size=4,
        return_scores=return_scores,
        topk_impl=topk_impl,
    )
    return jax.tree.map(np.asarray, out)

  def _assert_impls_match(self, inputs, *, k, bkv_p, bq):
    stream_idx, stream_sc = self._run(
        inputs, k=k, topk_impl="stream", bkv_p=bkv_p, bq=bq
    )
    sel_idx, sel_sc = self._run(
        inputs, k=k, topk_impl="select", bkv_p=bkv_p, bq=bq
    )
    np.testing.assert_array_equal(sel_idx, stream_idx, err_msg="indices")
    np.testing.assert_array_equal(sel_sc, stream_sc, err_msg="score bits")
    return sel_idx

  @parameterized.named_parameters(
      # name, q_lens, seq_lens, k, bkv_p, bq
      ("prefill_multi_block", [40], [2048], 256, 2, 16),
      ("prefill_single_block", [24], [1536], 128, 12, 8),
      ("decode_batch", [1] * 8, [1024, 2048, 640, 1800, 128, 2048, 900, 1],
       512, 2, 8),
      ("decode_partitions", [1], [4096], 1024, 4, 8),
      ("decode_partitions_2", [1, 1], [3000, 4096], 128, 1, 8),
      ("k_exceeds_kv_len", [3, 1], [200, 90], 512, 1, 8),
      ("large_k", [1, 1, 1, 1], [8192, 6000, 8192, 300], 2048, 4, 8),
      ("mixed_ragged", [1, 7, 33], [700, 1500, 2600], 256, 2, 16),
      # Enough KV blocks of >= 4 tiles for the two-level (cell) select.
      ("two_level_decode", [1, 1, 1], [8192, 5000, 300], 128, 4, 8),
      ("two_level_prefill", [48], [8192], 256, 8, 16),
  )
  def test_random_scores_match_stream(self, q_lens, seq_lens, k, bkv_p, bq):
    inputs = self._inputs(q_lens, seq_lens, seed=len(q_lens) * 7 + k)
    self._assert_impls_match(inputs, k=k, bkv_p=bkv_p, bq=bq)

  @parameterized.product(
      pattern=("all_equal", "periodic", "lane_skew", "blocks"),
      case=(
          # q_lens, seq_lens, k, bkv_p, bq
          ([1, 1], [4096, 1500], 128, 4, 8),
          ([20], [2048], 256, 2, 16),
          ([1], [3072], 1024, 8, 8),
      ),
  )
  def test_ties_resolve_by_lowest_index(self, pattern, case):
    q_lens, seq_lens, k, bkv_p, bq = case
    inputs = self._inputs(q_lens, seq_lens, pattern=pattern, seed=3)
    sel_idx = self._assert_impls_match(inputs, k=k, bkv_p=bkv_p, bq=bq)

    # Closed form: classes ranked by scale, then positions ascending.
    pages_per_seq = -(-max(seq_lens) // _SEL_PAGE_SIZE)
    classes = _sel_key_classes(pattern, pages_per_seq * _SEL_PAGE_SIZE)
    token = 0
    for q_len, seq_len in zip(q_lens, seq_lens):
      for i in range(q_len):
        n_visible = seq_len - q_len + i + 1
        pos = np.arange(n_visible)
        order = np.lexsort((pos, classes[:n_visible]))[:k]
        expected = np.full((k,), -1, np.int32)
        expected[: len(order)] = pos[order]
        np.testing.assert_array_equal(
            sel_idx[token], expected, err_msg=f"token {token}"
        )
        token += 1


if __name__ == "__main__":
  absltest.main()
