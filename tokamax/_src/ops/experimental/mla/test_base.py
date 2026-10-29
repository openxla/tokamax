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
"""Shared test base and test cases for MLA kernels."""

from collections.abc import Sequence
import gc
from typing import Any
from absl import flags
from absl import logging
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_numpy_dtype_promotion", "standard")

# Shared by the v2 and v3 kernel tests. Defined once here because absl flags are
# process-global and pytest imports every test module into one process.
DEBUG_MODE = flags.DEFINE_bool("debug_mode", False, "Run in debug mode.")


def cdiv(a: int, b: int) -> int:
  assert b != 0
  return (a + b - 1) // b


def align_to(x: int, a: int) -> int:
  return cdiv(x, a) * a


def get_dtype_bitwidth(dtype: Any) -> int:
  return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype: Any) -> int:
  bits = get_dtype_bitwidth(dtype)
  return 32 // bits


def generate_mla_inputs(
    seq_lens: Sequence[tuple[int, int]],  # List[(q_len, kv_len)]
    num_heads: int,
    lkv_dim: int,
    r_dim: int,
    page_size: int,
    q_dtype: Any,
    kv_dtype: Any,
    num_pages: int,
    *,
    rng: np.random.Generator | int | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
  """Generates inputs for the MLA kernel.

  Args:
    seq_lens: List of (q_len, kv_len) for each sequence.
    num_heads: Number of attention heads.
    lkv_dim: Dimension of the linear KV part.
    r_dim: Dimension of the rotary embedding part.
    page_size: Size of each page in the KV cache.
    q_dtype: Data type for queries.
    kv_dtype: Data type for keys and values.
    num_pages: Total number of pages in the cache.
    rng: Optional numpy random number generator or seed.

  Returns:
    A tuple containing:
      - ql_nope: Query linear part without positional encoding.
      - q_pe: Query positional encoding part.
      - new_kv_c: New KV cache data.
      - new_k_pe: New Key positional encoding.
      - cache_kv: The existing KV cache.
      - kv_lens: Array of KV lengths for each sequence.
      - page_indices: Indices mapping sequence pages to cache pages.
      - cu_q_lens: Cumulative query lengths.
      - distribution: Mode distribution (e.g., prefill, decode, mixed).
  """
  if isinstance(rng, np.random.Generator):
    key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
  elif isinstance(rng, int):
    key = jax.random.PRNGKey(rng)
  else:
    key = jax.random.PRNGKey(1234)

  def gen_random(shape, dtype):
    nonlocal key
    key, subkey = jax.random.split(key)
    dtype_canonical = jnp.dtype(dtype)
    if dtype_canonical in (
        jnp.dtype(jnp.float32),
        jnp.dtype(jnp.float16),
        jnp.dtype(jnp.bfloat16),
        jnp.dtype(jnp.float64),
    ):
      gen_dtype = dtype_canonical
    else:
      gen_dtype = jnp.bfloat16
    return jax.random.uniform(
        subkey, shape=shape, minval=0.0, maxval=1.0, dtype=gen_dtype
    ).astype(dtype)

  padded_r_dim = align_to(r_dim, 128)
  padded_lkv_dim = align_to(lkv_dim, 128)
  padded_kv_dim = padded_lkv_dim + padded_r_dim
  packing = get_dtype_packing(kv_dtype)
  q_lens = [s[0] for s in seq_lens]
  kv_lens_list = [s[1] for s in seq_lens]
  total_q_len = sum(q_lens)
  cu_q_lens_list = [0]
  for q_len in q_lens:
    cu_q_lens_list.append(cu_q_lens_list[-1] + q_len)

  max_kv_len = max(kv_lens_list) if kv_lens_list else 0
  pages_per_seq = cdiv(max_kv_len, page_size)

  page_indices_list = []
  page_count = 0
  for kv_len in kv_lens_list:
    num_seq_pages = cdiv(kv_len, page_size)
    indices = list(range(page_count, page_count + num_seq_pages))
    page_indices_list.extend(indices + [-1] * (pages_per_seq - num_seq_pages))
    page_count += num_seq_pages

  total_num_pages = max(num_pages, page_count)

  ql_nope = gen_random((total_q_len, num_heads, lkv_dim), q_dtype)
  q_pe = gen_random((total_q_len, num_heads, r_dim), q_dtype)
  new_kv_c = gen_random((total_q_len, lkv_dim), kv_dtype)
  new_k_pe = gen_random((total_q_len, r_dim), kv_dtype)

  cache_kv = gen_random(
      (total_num_pages, page_size // packing, packing, padded_kv_dim),
      kv_dtype,
  )
  kv_lens = jnp.array(kv_lens_list, dtype=jnp.int32)
  page_indices = jnp.array(page_indices_list, dtype=jnp.int32)
  cu_q_lens = jnp.array(cu_q_lens_list, dtype=jnp.int32)

  # Find the number of decode sequences at the beginning of the batch.
  num_decode_seqs = 0
  for s in seq_lens:
    if s[0] == 1:
      num_decode_seqs += 1
    else:
      break
  distribution = jnp.array(
      [num_decode_seqs, num_decode_seqs, len(seq_lens)], dtype=jnp.int32
  )

  return (
      ql_nope,
      q_pe,
      new_kv_c,
      new_k_pe,
      cache_kv,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
  )


class MlaRaggedPagedAttentionTestBase(parameterized.TestCase):
  """Base test class providing common correctness test cases for MLA kernels."""

  kv_dtype = jnp.float8_e4m3fn

  def tearDown(self):
    super().tearDown()
    jax.clear_caches()
    gc.collect()

  def _test_mla_ragged_paged_attention(
      self,
      seq_lens: Sequence[tuple[int, int]],
      num_heads: int,
      lkv_dim: int,
      r_dim: int,
      page_size: int,
      q_dtype: Any,
      kv_dtype: Any,
      num_pages: int,
      *,
      num_kv_pages_per_block: int = 8,
      num_queries_per_block: int = 8,
      vmem_limit_bytes: int = 100 * 1024 * 1024,
      sm_scale: float = 1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
  ):
    raise NotImplementedError(
        "Subclasses must implement _test_mla_ragged_paged_attention"
    )

  def test_ragged_paged_attention_basic(self):
    dtype = jnp.bfloat16
    seq_lens = [(192, 328), (128, 180), (64, 255)]
    num_heads = 128
    lkv_dim = 512
    r_dim = 64
    page_size = 128
    num_pages = 1024

    self._test_mla_ragged_paged_attention(
        seq_lens,
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        dtype,
        self.kv_dtype,
        num_pages,
    )

  def test_ragged_paged_attention_decode_only(self, dtype=jnp.bfloat16):
    seq_lens = [
        (1, 18),
        (1, 129),
        (1, 597),
        (1, 122),
        (1, 64),
        (1, 322),
        (1, 463),
        (1, 181),
        (1, 1107),
        (1, 123),
        (1, 31),
        (1, 18),
        (1, 1229),
        (1, 229),
        (1, 87),
        (1, 1328),
    ]
    num_heads = 128
    lkv_dim = 512
    r_dim = 64
    page_size = 128
    num_pages = 1024

    self._test_mla_ragged_paged_attention(
        seq_lens,
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        dtype,
        self.kv_dtype,
        num_pages,
    )

  def test_ragged_paged_attention_prefill_only(self, dtype=jnp.bfloat16):
    seq_lens = [
        (5, 18),
        (15, 129),
        (120, 597),
        (100, 122),
        (21, 64),
        (32, 322),
        (251, 463),
        (40, 181),
        (64, 1107),
        (99, 123),
        (10, 31),
        (5, 18),
        (3, 1229),
        (120, 229),
        (9, 87),
        (2, 1328),
    ]
    num_heads = 128
    lkv_dim = 512
    r_dim = 64
    page_size = 128
    num_pages = 1024

    self._test_mla_ragged_paged_attention(
        seq_lens,
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        dtype,
        self.kv_dtype,
        num_pages,
    )

  def test_ragged_paged_attention_mixed(self, dtype=jnp.bfloat16):
    seq_lens = [
        (5, 18),
        (1, 129),
        (120, 597),
        (1, 122),
        (1, 64),
        (32, 322),
        (251, 463),
        (1, 181),
        (1, 1107),
        (99, 123),
        (1, 31),
        (5, 18),
        (3, 1229),
        (117, 229),
        (1, 87),
        (1, 1328),
    ]
    num_heads = 128
    lkv_dim = 512
    r_dim = 64
    page_size = 128
    num_pages = 1024

    self._test_mla_ragged_paged_attention(
        seq_lens,
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        dtype,
        self.kv_dtype,
        num_pages,
    )

  @parameterized.product(
      sliding_window=[None, 5, 128],
  )
  def test_ragged_paged_attention_sliding_window(
      self,
      sliding_window: int | None,
  ):
    num_seqs = 5
    num_heads = 128
    lkv_dim = 512
    r_dim = 64
    dtype = jnp.float32
    rng = np.random.default_rng(1234)
    q_lens = rng.integers(1, 100, num_seqs)
    kv_lens = q_lens + rng.integers(0, 50, num_seqs)
    seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
    page_size = 128
    num_pages = 10240

    self._test_mla_ragged_paged_attention(
        seq_lens,
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        dtype,
        self.kv_dtype,
        num_pages,
        sliding_window=sliding_window,
    )

  @parameterized.product(
      soft_cap=[None, 50.0],
  )
  def test_ragged_paged_attention_logit_soft_capping(
      self,
      soft_cap: float | None,
  ):
    num_heads = 128
    num_seqs = 2
    dtype = jnp.float32
    rng = np.random.default_rng(1234)
    q_lens = rng.integers(1, 100, num_seqs)
    kv_lens = q_lens + rng.integers(0, 50, num_seqs)
    seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
    lkv_dim = 512
    r_dim = 64
    page_size = 128
    num_pages = 10240

    self._test_mla_ragged_paged_attention(
        seq_lens,
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        dtype,
        self.kv_dtype,
        num_pages,
        soft_cap=soft_cap,
    )

  def test_ragged_paged_attention_sliding_window_should_be_positive(self):
    dtype = jnp.float32
    seq_lens = [(192, 328), (128, 180), (64, 255)]
    num_heads = 128
    lkv_dim = 512
    r_dim = 64
    page_size = 128
    num_pages = 1000

    with self.assertRaisesRegex(ValueError, "must be positive"):
      self._test_mla_ragged_paged_attention(
          seq_lens,
          num_heads,
          lkv_dim,
          r_dim,
          page_size,
          dtype,
          self.kv_dtype,
          num_pages,
          sliding_window=0,
      )

    with self.assertRaisesRegex(ValueError, "must be positive"):
      self._test_mla_ragged_paged_attention(
          seq_lens,
          num_heads,
          lkv_dim,
          r_dim,
          page_size,
          dtype,
          self.kv_dtype,
          num_pages,
          sliding_window=-1,
      )

  def test_ragged_paged_attention_with_scales(self):
    num_heads = 128
    num_seqs = 2
    dtype = jnp.float32
    rng = np.random.default_rng(1234)
    q_lens = rng.integers(1, 100, num_seqs)
    kv_lens = q_lens + rng.integers(0, 50, num_seqs)
    seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
    lkv_dim = 512
    r_dim = 64
    page_size = 128
    num_pages = 10240

    self._test_mla_ragged_paged_attention(
        seq_lens,
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        dtype,
        self.kv_dtype,
        num_pages,
        q_scale=0.5,
        k_scale=0.5,
        v_scale=0.7,
    )

  def test_ragged_paged_attention_soft_cap_cannot_be_zero(self):
    dtype = jnp.float32
    seq_lens = [(192, 328), (128, 180), (64, 255)]
    num_heads = 128
    lkv_dim = 512
    r_dim = 64
    page_size = 128
    num_pages = 1000

    with self.assertRaisesRegex(ValueError, "must not be 0.0"):
      self._test_mla_ragged_paged_attention(
          seq_lens,
          num_heads,
          lkv_dim,
          r_dim,
          page_size,
          dtype,
          self.kv_dtype,
          num_pages,
          soft_cap=0.0,
      )

  @parameterized.named_parameters(
      dict(testcase_name="default"),
      dict(testcase_name="batch_size_8", batch_size=8),
      dict(testcase_name="batch_size_16", batch_size=16),
      dict(testcase_name="batch_size_32", batch_size=32),
      dict(testcase_name="kv_len_random", kv_len_range=(0, 5120 + 1)),
      dict(
          testcase_name="seq_len_random",
          kv_len_range=(4096, 4096 + 1),
          seq_len_range=(1, 1024 + 1),
      ),
      dict(
          testcase_name="len_random",
          kv_len_range=(0, 4096 + 1),
          seq_len_range=(1, 1024 + 1),
      ),
      dict(testcase_name="page_size_128", page_size=128),
      dict(
          testcase_name="page_size_256", page_size=256, num_kv_pages_per_block=2
      ),
      dict(
          testcase_name="page_size_512", page_size=512, num_kv_pages_per_block=1
      ),
      dict(testcase_name="num_pages_10240", num_pages=10240),
      dict(
          testcase_name="num_kv_pages_per_block_1",
          num_kv_pages_per_block=1,
      ),
      dict(
          testcase_name="num_kv_pages_per_block_2",
          num_kv_pages_per_block=2,
      ),
      dict(
          testcase_name="num_kv_pages_per_block_4",
          num_kv_pages_per_block=4,
      ),
      dict(
          testcase_name="num_kv_pages_per_block_8",
          num_kv_pages_per_block=8,
      ),
      dict(testcase_name="num_queries_per_block_1", num_queries_per_block=1),
      dict(testcase_name="num_queries_per_block_2", num_queries_per_block=2),
      dict(testcase_name="num_queries_per_block_4", num_queries_per_block=4),
      dict(testcase_name="num_queries_per_block_8", num_queries_per_block=8),
      # Corner cases around page boundaries (page_size=128)
      dict(
          testcase_name="decode_bs1_kv127_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(1, 2),
          kv_len_range=(127, 128),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="decode_bs1_kv128_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(1, 2),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="decode_bs1_kv129_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(1, 2),
          kv_len_range=(129, 130),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      # Corner cases around KV block boundaries
      # (page_size=128, num_kv_pages_per_block=2 -> block capacity=256)
      dict(
          testcase_name="decode_bs1_kv255_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(1, 2),
          kv_len_range=(255, 256),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="decode_bs1_kv256_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(1, 2),
          kv_len_range=(256, 257),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="decode_bs1_kv257_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(1, 2),
          kv_len_range=(257, 258),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      # Corner cases for larger sequences spanning multiple blocks
      dict(
          testcase_name="decode_bs1_kv511_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(1, 2),
          kv_len_range=(511, 512),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="decode_bs1_kv512_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(1, 2),
          kv_len_range=(512, 513),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="decode_bs1_kv513_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(1, 2),
          kv_len_range=(513, 514),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="decode_bs2_kv128_ps128_kvb2_qb16",
          batch_size=2,
          seq_len_range=(1, 2),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=16,
      ),
      dict(
          testcase_name="decode_bs3_kv128_ps128_kvb2_qb4",
          batch_size=3,
          seq_len_range=(1, 2),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="decode_bs4_kv128_ps128_kvb2_qb4",
          batch_size=4,
          seq_len_range=(1, 2),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="decode_bs5_kv128_ps128_kvb2_qb4",
          batch_size=5,
          seq_len_range=(1, 2),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="decode_bs5_kv128_ps128_kvb2_qb16",
          batch_size=5,
          seq_len_range=(1, 2),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=16,
      ),
      dict(
          testcase_name="decode_bs8_kv128_ps128_kvb2_qb4",
          batch_size=8,
          seq_len_range=(1, 2),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="decode_bs13_kv128_ps128_kvb2_qb16",
          batch_size=13,
          seq_len_range=(1, 2),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=16,
      ),
      dict(
          testcase_name="decode_bs16_kv128_ps128_kvb2_qb16",
          batch_size=16,
          seq_len_range=(1, 2),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=16,
      ),
      dict(
          testcase_name="decode_bs4_mixed_kv_lens_ps128",
          batch_size=4,
          seq_len_range=(1, 2),
          kv_len_range=[(100, 101), (128, 129), (50, 51), (300, 301)],
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="decode_bs8_mixed_kv_lens_ps128",
          batch_size=8,
          seq_len_range=(1, 2),
          kv_len_range=[
              (20, 21),
              (120, 121),
              (256, 257),
              (512, 513),
              (100, 101),
              (128, 129),
              (50, 51),
              (300, 301),
          ],
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      # Corner cases around query block boundaries (num_queries_per_block=4)
      dict(
          testcase_name="prefill_bs1_seq_len3_kv128_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(3, 4),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="prefill_bs1_seq_len4_kv128_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(4, 5),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="prefill_bs1_seq_len5_kv128_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(5, 6),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="prefill_bs1_seq_len7_kv128_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(7, 8),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="prefill_bs1_seq_len8_kv128_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(8, 9),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="prefill_bs1_seq_len9_kv128_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(9, 10),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      # Corner cases around query block boundaries (num_queries_per_block=16)
      dict(
          testcase_name="prefill_bs1_seq_len15_kv256_ps128_kvb2_qb16",
          batch_size=1,
          seq_len_range=(15, 16),
          kv_len_range=(256, 257),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=16,
      ),
      dict(
          testcase_name="prefill_bs1_seq_len16_kv256_ps128_kvb2_qb16",
          batch_size=1,
          seq_len_range=(16, 17),
          kv_len_range=(256, 257),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=16,
      ),
      dict(
          testcase_name="prefill_bs1_seq_len17_kv256_ps128_kvb2_qb16",
          batch_size=1,
          seq_len_range=(17, 18),
          kv_len_range=(256, 257),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=16,
      ),
      dict(
          testcase_name="prefill_bs1_seq_len31_kv512_ps128_kvb2_qb16",
          batch_size=1,
          seq_len_range=(31, 32),
          kv_len_range=(512, 513),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=16,
      ),
      dict(
          testcase_name="prefill_bs1_seq_len32_kv512_ps128_kvb2_qb16",
          batch_size=1,
          seq_len_range=(32, 33),
          kv_len_range=(512, 513),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=16,
      ),
      dict(
          testcase_name="prefill_bs1_seq_len33_kv512_ps128_kvb2_qb16",
          batch_size=1,
          seq_len_range=(33, 34),
          kv_len_range=(512, 513),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=16,
      ),
      # Mixed cases around page boundaries (page_size=128)
      dict(
          testcase_name="mixed_bs1_seq_len5_kv127_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(5, 6),
          kv_len_range=(127, 128),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="mixed_bs1_seq_len5_kv129_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(5, 6),
          kv_len_range=(129, 130),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      # Mixed cases around KV block boundaries
      # (page_size=128, num_kv_pages_per_block=2 -> block capacity=256)
      dict(
          testcase_name="mixed_bs1_seq_len5_kv255_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(5, 6),
          kv_len_range=(255, 256),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="mixed_bs1_seq_len5_kv256_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(5, 6),
          kv_len_range=(256, 257),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="mixed_bs1_seq_len5_kv257_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(5, 6),
          kv_len_range=(257, 258),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      # Mixed cases for larger sequences spanning multiple blocks
      dict(
          testcase_name="mixed_bs1_seq_len5_kv511_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(5, 6),
          kv_len_range=(511, 512),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="mixed_bs1_seq_len5_kv512_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(5, 6),
          kv_len_range=(512, 513),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="mixed_bs1_seq_len5_kv513_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(5, 6),
          kv_len_range=(513, 514),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      # Mixed cases around query block boundaries (num_queries_per_block=4)
      dict(
          testcase_name="mixed_bs1_seq_len3_kv128_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(3, 4),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="mixed_bs1_seq_len4_kv128_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(4, 5),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="mixed_bs1_seq_len5_kv128_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(5, 6),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="mixed_bs1_seq_len7_kv128_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(7, 8),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="mixed_bs1_seq_len8_kv128_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(8, 9),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="mixed_bs1_seq_len9_kv128_ps128_kvb2_qb4",
          batch_size=1,
          seq_len_range=(9, 10),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="mixed_bs3_seq_len5_kv128_ps128_kvb2_qb4",
          batch_size=3,
          seq_len_range=(5, 6),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="mixed_bs4_seq_len5_kv128_ps128_kvb2_qb4",
          batch_size=4,
          seq_len_range=(5, 6),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="mixed_bs5_seq_len5_kv128_ps128_kvb2_qb4",
          batch_size=5,
          seq_len_range=(5, 6),
          kv_len_range=(128, 129),
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      # Mixed cases with varying kv lengths
      dict(
          testcase_name="mixed_bs4_mixed_kv_lens_ps128",
          batch_size=4,
          seq_len_range=(5, 6),
          kv_len_range=[(100, 101), (128, 129), (50, 51), (300, 301)],
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="mixed_bs8_mixed_kv_lens_ps128",
          batch_size=8,
          seq_len_range=(5, 6),
          kv_len_range=[
              (20, 21),
              (120, 121),
              (256, 257),
              (512, 513),
              (100, 101),
              (128, 129),
              (50, 51),
              (300, 301),
          ],
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      # Mixed cases with varying seq lengths (decode and prefill combinations)
      dict(
          testcase_name="mixed_batch_bs4_ps128_kvb2_qb4",
          batch_size=4,
          seq_len_range=[(1, 2), (5, 6), (1, 2), (5, 6)],
          kv_len_range=[(127, 128), (255, 256), (256, 257), (513, 514)],
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
      dict(
          testcase_name="mixed_batch_bs8_ps128_kvb2_qb4",
          batch_size=8,
          seq_len_range=[
              (1, 2),
              (1, 2),
              (4, 5),
              (5, 6),
              (1, 2),
              (8, 9),
              (1, 2),
              (2, 3),
          ],
          kv_len_range=[
              (127, 128),
              (129, 130),
              (255, 256),
              (257, 258),
              (511, 512),
              (513, 514),
              (100, 101),
              (50, 51),
          ],
          page_size=128,
          num_kv_pages_per_block=2,
          num_queries_per_block=4,
      ),
  )
  def test_ragged_paged_attention_deepseekv3(
      self,
      batch_size=4,
      seq_len_range=(1, 1 + 1),  # default by decode only
      kv_len_range=(5120, 5120 + 1),
      page_size=128,
      num_pages=1024,
      num_kv_pages_per_block=8,
      num_queries_per_block=16,
  ):
    rng = np.random.default_rng(1234)
    if isinstance(seq_len_range, list):
      assert len(seq_len_range) == batch_size
      q_lens = np.array(
          [rng.integers(low, high) for low, high in seq_len_range]
      )
    else:
      q_lens = rng.integers(seq_len_range[0], seq_len_range[1], batch_size)

    if isinstance(kv_len_range, list):
      assert len(kv_len_range) == batch_size
      kv_lens = q_lens + np.array(
          [rng.integers(low, high) for low, high in kv_len_range]
      )
    else:
      kv_lens = q_lens + rng.integers(
          kv_len_range[0], kv_len_range[1], batch_size
      )
    logging.info("Generated kv_lens: %s", kv_lens)
    seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
    num_heads = 128
    lkv_dim = 512
    r_dim = 64
    dtype = jnp.bfloat16
    self._test_mla_ragged_paged_attention(
        seq_lens,
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        dtype,
        self.kv_dtype,
        num_pages,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
    )

  def test_ragged_paged_attention_deepseekv3_batch_decode(
      self,
      batch_size=128,
      seq_len_range=(1, 1 + 1),  # default by decode only
      kv_len_range=(9216, 9216 + 1),
      page_size=1024,
      num_pages=128,
      num_kv_pages_per_block=3,
      num_queries_per_block=1,
  ):
    rng = np.random.default_rng(1234)
    if isinstance(seq_len_range, list):
      assert len(seq_len_range) == batch_size
      q_lens = np.array(
          [rng.integers(low, high) for low, high in seq_len_range]
      )
    else:
      q_lens = rng.integers(seq_len_range[0], seq_len_range[1], batch_size)

    if isinstance(kv_len_range, list):
      assert len(kv_len_range) == batch_size
      kv_lens = q_lens + np.array(
          [rng.integers(low, high) for low, high in kv_len_range]
      )
    else:
      kv_lens = q_lens + rng.integers(
          kv_len_range[0], kv_len_range[1], batch_size
      )
    logging.info("Generated kv_lens: %s", kv_lens)
    seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
    num_heads = 128
    lkv_dim = 512
    r_dim = 64
    dtype = jnp.bfloat16
    self._test_mla_ragged_paged_attention(
        seq_lens,
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        dtype,
        self.kv_dtype,
        num_pages,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
    )
