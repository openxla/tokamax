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
"""The upstream test suite of the compress-and-store Pallas kernel, ported.

The tests are kept as they are upstream: same oracles, shapes, seeds, named
cases, skips and bit-exact comparisons (up to the sign of zeros), calling the
raw kernel entry point with its default `tile_n`. Upstream's
`compress_store_ref` is `reference` here, and its Kernel 1 reference
(`project_and_save_state_ref.ref_wkv_proj_and_save_state`) is the
`proj_and_save_state` op's `reference.ref_wkv_proj_and_save_state`.
"""

import math

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.dsv4_compressor import config
from tokamax._src.ops.dsv4_compressor import csa_cache_layout
from tokamax._src.ops.dsv4_compressor import pallas_mosaic_tpu_kernel
from tokamax._src.ops.dsv4_compressor import reference
from tokamax._src.ops.proj_and_save_state import reference as proj_and_save_state_ref

jax.config.parse_flags_with_absl()


def _tpu_older_than_7x() -> bool:
  """Whether the default device is not a TPU7x or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 7
  )


def generate_kv_slot_mapping(
    num_boundary_tokens: int,
    num_pages: int,
    page_size: int,
    slots_per_part_out: int,
) -> jax.Array:
  """Generates kv_slot_mapping for boundary tokens."""
  kv_slot_mapping_np = np.full((num_boundary_tokens,), -1, dtype=np.int32)
  boundary_count = 0
  total_slots_needed = num_boundary_tokens * slots_per_part_out
  pages_needed = (total_slots_needed + page_size - 1) // page_size
  start_page = max(0, num_pages - pages_needed)

  for t in range(num_boundary_tokens):
    kv_slot_mapping_np[t] = start_page * page_size + boundary_count
    boundary_count += slots_per_part_out
  return jnp.array(kv_slot_mapping_np, dtype=jnp.int32)


def make_block_table(rows, cols, pages_for_state, num_pages):
  """Builds a block table whose rows differ in the order of their state pages."""
  assert rows <= math.factorial(pages_for_state)
  rng = np.random.default_rng(0)
  orders = [tuple(range(pages_for_state))]
  while len(orders) < rows:
    order = tuple(int(p) for p in rng.permutation(pages_for_state))
    if order not in orders:
      orders.append(order)

  table = np.zeros((rows, max(num_pages, cols or 0)), dtype=np.int32)
  table[:, :pages_for_state] = np.array(orders, dtype=np.int32)
  table[:, pages_for_state:num_pages] = np.arange(
      pages_for_state, num_pages, dtype=np.int32
  )
  return table


def _scatter_sal_rows_reference(
    cache_kv: jax.Array, src: jax.Array, dst_rows: jax.Array
) -> jax.Array:
  """Pure-XLA reference scatter into a `SEQ_ALONG_LANE` cache."""
  num_pages, sublane_groups, kv_packing, page_size = cache_kv.shape
  num_rows = num_pages * page_size
  row_bytes = sublane_groups * kv_packing
  head_dim = row_bytes - kv_packing
  rows = jnp.pad(src[:, : head_dim + 1], ((0, 0), (0, kv_packing - 1)))
  valid = (dst_rows >= 0) & (dst_rows < num_rows)
  page = jnp.where(valid, dst_rows // page_size, num_pages)
  slot = jnp.where(valid, dst_rows % page_size, 0)
  return cache_kv.at[page, :, :, slot].set(
      rows.reshape(-1, sublane_groups, kv_packing), mode="drop"
  )


def _convert_cache_to_seq_along_lane(
    cache_kv: jax.Array, head_dim: int
) -> jax.Array:
  """Repacks a `HEAD_ALONG_SUBLANE` indexer cache into `SEQ_ALONG_LANE`."""
  total_num_pages, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
  page_size = page_size_per_kv_packing * kv_packing
  flat = cache_kv.reshape(total_num_pages, page_size, -1)
  keys = flat[:, :, :head_dim]
  scales = jnp.pad(
      flat[:, :, head_dim : head_dim + 1],
      ((0, 0), (0, 0), (0, kv_packing - 1)),
      constant_values=0,
  )
  rows = jnp.concatenate([keys, scales], axis=-1).swapaxes(1, 2)
  return rows.reshape(
      total_num_pages,
      (head_dim + kv_packing) // kv_packing,
      kv_packing,
      page_size,
  )


def normalize_fp8_zero_sign(arr):
  is_zero = (arr & 0x7F) == 0
  return jnp.where(is_zero, 0, arr)


def normalize_bf16_zero_sign(arr):
  orig_shape = arr.shape
  arr_2d = arr.reshape(-1, 2)
  even = arr_2d[..., 0]
  odd = arr_2d[..., 1]
  is_zero = (even == 0) & ((odd & 0x7F) == 0)
  new_odd = jnp.where(is_zero, odd & 0x7F, odd)
  normalized = jnp.stack([even, new_odd], axis=-1)
  return normalized.reshape(orig_shape)


def normalize_bf16_words_zero_sign(words):
  """`normalize_bf16_zero_sign` for int32 words holding two bf16 each."""
  u = jax.lax.bitcast_convert_type(words, jnp.uint32)
  lo, hi = u & 0xFFFF, u >> 16
  lo = jnp.where((lo & 0x7FFF) == 0, 0, lo)
  hi = jnp.where((hi & 0x7FFF) == 0, 0, hi)
  return jax.lax.bitcast_convert_type(lo | (hi << 16), jnp.int32)


def _is_device_tpu_at_least_7() -> bool:
  """Stands in for upstream's `jtu.is_device_tpu_at_least(version=7)`."""
  return jax.default_backend() == "tpu" and pltpu.get_tpu_info().generation >= 7


class Dsv4CompressorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")

  def run_compress_store_correctness(
      self,
      num_tokens,
      head_dim,
      rope_head_dim,
      compress_ratio,
      overlap,
      physical_page_size,
      state_physical_page_size=None,
      quant_block=64,
      rms_eps=1e-6,
      positions=None,
      token_to_req_indices=None,
      block_table=None,
      kv_slot_mapping=None,
      prefill_len=None,
      interpret=False,
      hidden_size=7168,
  ):
    """Checks the kernel against the reference on one upstream case."""
    # Create config first
    if head_dim == 128:
      mode = config.Mode.CSA_INDEXER
    else:
      mode = config.Mode.CSA if overlap else config.Mode.HCA

    def make_cfgs(state_block_size):
      return config.Configs.make(
          mode,
          size_n=num_tokens,
          physical_page_size=physical_page_size,
          state_physical_page_size=state_physical_page_size,
          rms_eps=rms_eps,
          head_dim=head_dim,
          rope_head_dim=rope_head_dim,
          compress_ratio=compress_ratio,
          quant_block=quant_block,
          state_block_size=state_block_size,
      )

    # `state_block_size` is vLLM's paging granularity, an input rather than
    # a property of the page: `config.state_block_size` floors the indexer
    # to CSA's smaller capacity (always exactly half its own) so the two
    # share a cache group, leaving the tail of every indexer page unused.
    # The capacity itself does not depend on it, so probe for it first.
    state_page_capacity = make_cfgs(1).state_page_capacity
    state_block_size = (
        state_page_capacity // 2
        if mode is config.Mode.CSA_INDEXER
        else state_page_capacity
    )
    cfgs = make_cfgs(state_block_size)
    separate_state = mode is config.Mode.HCA

    state_width = cfgs.state_width
    state_dim = 2 * state_width

    # Setup positions and identify boundary tokens
    if positions is None:
      positions_np = np.arange(num_tokens, dtype=np.int32)
    else:
      positions_np = np.array(positions, dtype=np.int32)

    boundary_mask = ((positions_np + 1) % compress_ratio) == 0
    positions_filtered = positions_np[boundary_mask]
    num_boundary = positions_filtered.shape[0]

    # Calculate num_pages
    run_1_tokens = prefill_len if prefill_len is not None else num_tokens
    tokens_per_page = state_block_size
    pages_for_state = (run_1_tokens + tokens_per_page - 1) // tokens_per_page

    total_bytes_needed = num_boundary * cfgs.record_bytes
    page_bytes = cfgs.physical_page_size * cfgs.row_size_bytes
    pages_for_kv_cache = (total_bytes_needed + page_bytes - 1) // page_bytes
    num_pages = pages_for_state + pages_for_kv_cache

    # 1. Initialize keys
    k = jax.random.key(0)
    k1, k2, k3, k4, k5 = jax.random.split(k, 5)

    # 2. Populate cache
    hidden_states = jax.random.normal(k1, (run_1_tokens, hidden_size))
    wkv_wgate = jax.random.normal(k2, (hidden_size, state_dim))
    ape = jax.random.normal(k3, (compress_ratio, state_width))
    run_1_positions = jnp.arange(run_1_tokens, dtype=jnp.int32)

    slots_per_token = cfgs.state_rows_per_token

    def state_slot(token_index):
      """Physical state row of a token, the way `derive_metadata` maps it.

      The block table below is the identity, so the page number is just
      the block index. Tokens are page-strided, not densely packed: a
      page holds `state_page_capacity` states but is only paged at
      `state_block_size` of them.

      Args:
        token_index: The token position.

      Returns:
        The state row.
      """
      return (
          token_index // state_block_size
      ) * cfgs.state_physical_page_size + (
          token_index % state_block_size
      ) * slots_per_token

    run_1_slot_mapping = state_slot(np.arange(run_1_tokens))
    run_1_slot_mapping = jnp.array(run_1_slot_mapping, dtype=jnp.int32)

    init_cache = jnp.zeros(cfgs.cache_shape(num_pages), dtype=jnp.uint8)
    init_state_cache = jnp.zeros(
        cfgs.state_cache_shape(num_pages), dtype=jnp.uint8
    )

    ref_wkv_proj_and_save_state_jit = jax.jit(
        proj_and_save_state_ref.ref_wkv_proj_and_save_state,
        static_argnums=(6, 7, 8, 9),
    )
    # The state scatter targets the state array, which is `init_cache`
    # itself in the shared-buffer layout.
    populated_state_cache = ref_wkv_proj_and_save_state_jit(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        ape=ape,
        positions=run_1_positions,
        slot_mapping=run_1_slot_mapping,
        cache=init_state_cache if separate_state else init_cache,
        state_block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )
    if separate_state:
      populated_cache = init_cache
    else:
      populated_cache = populated_state_cache
      populated_state_cache = None
    if cfgs.dims.has_rope_cache:
      # CSA's cache is allocated as int32 words (see csa_cache_layout).
      populated_cache = csa_cache_layout.slabs_to_words(populated_cache)

    # 3. Setup Kernel 2 inputs
    if token_to_req_indices is None:
      token_to_req_indices_filtered = np.zeros((num_boundary,), dtype=np.int32)
    else:
      token_to_req_indices_np = np.array(token_to_req_indices)
      token_to_req_indices_filtered = token_to_req_indices_np[boundary_mask]

    if kv_slot_mapping is None:
      kv_slot_mapping_filtered = generate_kv_slot_mapping(
          num_boundary,
          num_pages,
          cfgs.kv_block_size * cfgs.kv_stride,
          cfgs.kv_stride,
      )
    else:
      kv_slot_mapping_np = np.array(kv_slot_mapping)
      kv_slot_mapping_filtered = kv_slot_mapping_np[boundary_mask]

    # Pad to num_tokens
    positions_padded = np.pad(
        positions_filtered,
        (0, num_tokens - num_boundary),
        constant_values=0,
    )
    positions = jnp.array(positions_padded)

    token_to_req_indices_padded = np.pad(
        token_to_req_indices_filtered,
        (0, num_tokens - num_boundary),
        constant_values=0,
    )
    token_to_req_indices = jnp.array(token_to_req_indices_padded)

    kv_slot_mapping_padded = np.pad(
        kv_slot_mapping_filtered,
        (0, num_tokens - num_boundary),
        constant_values=-1,
    )
    kv_slot_mapping = jnp.array(kv_slot_mapping_padded)

    if block_table is None:
      block_table = jnp.array([[i for i in range(num_pages)]], dtype=jnp.int32)
    else:
      block_table = jnp.array(block_table)
    block_table_stride = block_table.shape[1]
    block_table = block_table.reshape(-1)

    rms_weight = jax.random.normal(k4, (head_dim,))

    max_pos = int(jnp.max(positions)) + 1 if len(positions) > 0 else 0
    cos_sin_cache_len = max(max_pos, num_tokens)
    cos_sin_cache = jax.random.normal(k5, (cos_sin_cache_len, rope_head_dim))

    if cfgs.dims.has_rope_cache:
      num_rope_pages, rope_rows, _, lanes = cfgs.rope_cache_shape(num_pages)
      init_rope_cache = jnp.zeros(
          (num_rope_pages, rope_rows, lanes), dtype=jnp.int32
      )
    else:
      init_rope_cache = jnp.zeros((num_pages, 1, 1, 128), dtype=jnp.uint8)
    slot_mapping = jnp.where(positions >= 0, state_slot(positions), -1)

    is_quantized = overlap
    has_rope = rope_head_dim > 0

    # 4. Run Reference
    ref_compress_norm_rope_store_jit = jax.jit(
        reference.ref_compress_norm_rope_store,
        static_argnames=(
            "block_table_stride",
            "state_block_size",
            "head_dim",
            "rope_head_dim",
            "compress_ratio",
            "overlap",
            "rms_eps",
            "quant_block",
            "is_quantized",
            "has_rope",
            "has_rope_cache",
        ),
    )
    ref_out = ref_compress_norm_rope_store_jit(
        cache=populated_cache,
        rope_cache=init_rope_cache,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        block_table_stride=block_table_stride,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=rms_eps,
        quant_block=quant_block,
        is_quantized=is_quantized,
        has_rope=has_rope,
        has_rope_cache=cfgs.dims.has_rope_cache,
        state_cache=populated_state_cache,
    )
    ref_cache_output, ref_rope_output = ref_out

    # 5. Run Pallas

    pallas_out = pallas_mosaic_tpu_kernel.compress_norm_rope_store(
        jnp.copy(populated_cache),
        positions,
        block_table,
        token_to_req_indices,
        kv_slot_mapping,
        rms_weight,
        block_table_stride=block_table_stride,
        state_block_size=state_block_size,
        state_cache=(
            jnp.copy(populated_state_cache) if separate_state else None
        ),
        rope_cache=jnp.copy(init_rope_cache),
        cos_sin_cache=cos_sin_cache,
        compress_ratio=cfgs.dims.compress_ratio,
        overlap=cfgs.dims.overlap,
        quant_block=cfgs.dims.quant_block,
        rms_eps=cfgs.dims.rms_eps,
        interpret=interpret,
    )
    pallas_cache_output, pallas_rope_output = pallas_out
    if cfgs.dims.has_rope_cache:
      self.assertEqual(pallas_cache_output.dtype, jnp.int32)
      self.assertEqual(ref_cache_output.dtype, jnp.int32)
      pallas_cache_output = csa_cache_layout.words_to_slabs(pallas_cache_output)
      ref_cache_output = csa_cache_layout.words_to_slabs(ref_cache_output)

    if is_quantized:
      pallas_cache_normalized = normalize_fp8_zero_sign(pallas_cache_output)
      ref_cache_normalized = normalize_fp8_zero_sign(ref_cache_output)
    else:
      pallas_cache_normalized = normalize_bf16_zero_sign(pallas_cache_output)
      ref_cache_normalized = normalize_bf16_zero_sign(ref_cache_output)

    if cfgs.dims.has_rope_cache:
      self.assertEqual(pallas_rope_output.dtype, jnp.int32)
      pallas_rope_normalized = normalize_bf16_words_zero_sign(
          pallas_rope_output
      )
      ref_rope_normalized = normalize_bf16_words_zero_sign(ref_rope_output)
      np.testing.assert_array_equal(pallas_rope_normalized, ref_rope_normalized)

    np.testing.assert_array_equal(pallas_cache_normalized, ref_cache_normalized)

  @parameterized.named_parameters(
      (
          # DSv4-Flash's real geometry: hidden_size 4096 vs state_dim 2048.
          # `tile_k` must divide hidden_size
          "csa_prod_hidden_4096",
          dict(
              num_tokens=8,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=256,
              hidden_size=4096,
          ),
      ),
      (
          "csa_prefill",
          dict(
              num_tokens=128,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=256,
          ),
      ),
      (
          "hca_prefill",
          dict(
              num_tokens=256,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=128,
              overlap=False,
              physical_page_size=16,
              state_physical_page_size=256,
          ),
      ),
      (
          "hca_prefill_small",
          dict(
              num_tokens=128,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=128,
              overlap=False,
              physical_page_size=16,
              state_physical_page_size=256,
          ),
      ),
      (
          "csa_decode_batch_large",
          dict(
              num_tokens=1024,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=256,
              positions=(np.arange(1024, dtype=np.int32) * 3) % 1024,
              prefill_len=1024,
          ),
      ),
      (
          "csa_decode_batch_seq",
          dict(
              num_tokens=4,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=256,
              positions=np.array([3, 7, 11, 15], dtype=np.int32),
              prefill_len=16,
          ),
      ),
      (
          "hca_decode_batch_seq",
          dict(
              num_tokens=4,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=128,
              overlap=False,
              physical_page_size=16,
              state_physical_page_size=256,
              positions=np.array([127, 255, 383, 511], dtype=np.int32),
              prefill_len=512,
          ),
      ),
      (
          "csa_decode_batch_random",
          dict(
              num_tokens=4,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=256,
              positions=np.array([11, 3, 19, 7], dtype=np.int32),
              prefill_len=32,
          ),
      ),
      (
          "hca_decode_batch_random",
          dict(
              num_tokens=4,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=128,
              overlap=False,
              physical_page_size=16,
              state_physical_page_size=256,
              positions=np.array([383, 127, 511, 255], dtype=np.int32),
              prefill_len=512,
          ),
      ),
      (
          "csa_decode_batch_mixed",
          dict(
              num_tokens=6,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=256,
              positions=np.array([3, 2, 7, 6, 11, 10], dtype=np.int32),
              prefill_len=32,
          ),
      ),
      (
          "hca_decode_batch_mixed",
          dict(
              num_tokens=6,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=128,
              overlap=False,
              physical_page_size=16,
              state_physical_page_size=256,
              positions=np.array(
                  [127, 126, 255, 254, 383, 382], dtype=np.int32
              ),
              prefill_len=384,
          ),
      ),
      (
          "csa_indexer_prefill",
          dict(
              num_tokens=128,
              head_dim=128,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=32,
              quant_block=128,
          ),
      ),
      (
          "csa_indexer_decode",
          dict(
              num_tokens=4,
              head_dim=128,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=32,
              quant_block=128,
              positions=np.array([3, 7, 11, 15], dtype=np.int32),
              prefill_len=16,
          ),
      ),
  )
  def test_compress_store(self, cfg):
    # csa_decode_batch_large failed on v6e but passed on TPU7x,
    # temporarily disable that test case for v6e.
    if (
        self._testMethodName.endswith("csa_decode_batch_large")
        and not _is_device_tpu_at_least_7()
    ):
      self.skipTest("skip csa_decode_batch_large on TPU v6e")
    self.run_compress_store_correctness(**cfg)

  def test_derive_aliases(self):
    # HCA: has_rope=True, has_rope_cache=False, num_scalar_prefetch=5
    self.assertEqual(
        pallas_mosaic_tpu_kernel.derive_aliases(
            has_rope=True, has_rope_cache=False, num_scalar_prefetch=5
        ),
        {7: 0},
    )

    # CSA: has_rope=True, has_rope_cache=True, num_scalar_prefetch=5
    self.assertEqual(
        pallas_mosaic_tpu_kernel.derive_aliases(
            has_rope=True, has_rope_cache=True, num_scalar_prefetch=5
        ),
        {7: 0, 8: 1},
    )

    # CSA_INDEXER: has_rope=True, has_rope_cache=False, num_scalar_prefetch=5
    self.assertEqual(
        pallas_mosaic_tpu_kernel.derive_aliases(
            has_rope=True, has_rope_cache=False, num_scalar_prefetch=5
        ),
        {7: 0},
    )

  def test_seq_along_lane_state_block_size(self):
    # DSv4's 1024-token KV block: 256 compressed indexer tokens per page.
    kv_block = 1024
    self.assertEqual(
        config.physical_page_size(
            config.Mode.CSA_INDEXER, kv_block, 4, seq_along_lane=True
        ),
        config.SAL_SUBLANE_GROUPS,
    )
    # The 33-row page halves the indexer's state capacity to CSA's, so the
    # block size the two share is unchanged.
    for mode in (config.Mode.CSA, config.Mode.CSA_INDEXER):
      self.assertEqual(
          config.state_block_size(mode, kv_block, indexer_seq_along_lane=True),
          config.state_block_size(mode, kv_block),
      )
    with self.assertRaisesRegex(ValueError, "256 compressed tokens"):
      config.physical_page_size(
          config.Mode.CSA_INDEXER, 512, 4, seq_along_lane=True
      )
    with self.assertRaisesRegex(
        ValueError, "only supported for the CSA indexer"
    ):
      config.Configs.make(
          config.Mode.CSA,
          size_n=4,
          physical_page_size=config.SAL_SUBLANE_GROUPS,
          state_block_size=16,
          seq_along_lane=True,
      )

  def run_seq_along_lane_matches_head_along_sublane(
      self,
      num_tokens,
      positions=None,
      prefill_len=None,
      token_to_req_indices=None,
      block_table_rows=1,
      # Only seeds the compressor state through the reference projection;
      # both layouts get the same state, so its width is not under test.
      hidden_size=512,
  ):
    """SAL output == the HAS kernel's records, transposed into a SAL cache."""
    head_dim, rope_head_dim, compress_ratio, quant_block = 128, 64, 4, 128
    mode = config.Mode.CSA_INDEXER
    page_tokens = config.SAL_PAGE_SIZE
    has_rows = page_tokens // config.SLOT_PACK
    sal_rows = config.SAL_SUBLANE_GROUPS
    state_block_size = config.state_block_size(
        mode, page_tokens * compress_ratio, indexer_seq_along_lane=True
    )

    def make_cfgs(rows, seq_along_lane):
      return config.Configs.make(
          mode,
          size_n=num_tokens,
          physical_page_size=rows,
          state_block_size=state_block_size,
          head_dim=head_dim,
          rope_head_dim=rope_head_dim,
          compress_ratio=compress_ratio,
          quant_block=quant_block,
          seq_along_lane=seq_along_lane,
      )

    has_cfgs = make_cfgs(has_rows, False)
    sal_cfgs = make_cfgs(sal_rows, True)
    self.assertEqual(has_cfgs.kv_block_size, page_tokens)
    self.assertEqual(sal_cfgs.kv_block_size, page_tokens)
    self.assertLessEqual(state_block_size, sal_cfgs.state_page_capacity)

    positions_np = (
        np.arange(num_tokens, dtype=np.int32)
        if positions is None
        else np.array(positions, dtype=np.int32)
    )
    boundary_mask = ((positions_np + 1) % compress_ratio) == 0
    num_boundary = int(boundary_mask.sum())
    run_1_tokens = prefill_len if prefill_len is not None else num_tokens
    pages_for_state = math.ceil(run_1_tokens / state_block_size)
    num_pages = pages_for_state + math.ceil(num_boundary / page_tokens)

    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(0), 5)
    state_width = has_cfgs.state_width
    hidden_states = jax.random.normal(k1, (run_1_tokens, hidden_size))
    wkv_wgate = jax.random.normal(k2, (hidden_size, 2 * state_width))
    ape = jax.random.normal(k3, (compress_ratio, state_width))
    rms_weight = jax.random.normal(k4, (head_dim,))

    def populate(cfgs):
      tokens = np.arange(run_1_tokens)
      slots = (
          tokens // state_block_size
      ) * cfgs.state_physical_page_size + (
          tokens % state_block_size
      ) * cfgs.state_rows_per_token
      return jax.jit(
          proj_and_save_state_ref.ref_wkv_proj_and_save_state,
          static_argnums=(6, 7, 8, 9),
      )(
          hidden_states=hidden_states,
          wkv_wgate=wkv_wgate,
          ape=ape,
          positions=jnp.arange(run_1_tokens, dtype=jnp.int32),
          slot_mapping=jnp.array(slots, dtype=jnp.int32),
          cache=jnp.zeros(cfgs.cache_shape(num_pages), dtype=jnp.uint8),
          state_block_size=state_block_size,
          head_dim=head_dim,
          compress_ratio=compress_ratio,
          overlap=True,
      )

    has_cache = populate(has_cfgs)
    sal_cache = populate(sal_cfgs)

    pad = num_tokens - num_boundary
    boundary_positions = jnp.array(
        np.pad(positions_np[boundary_mask], (0, pad), constant_values=0)
    )
    if token_to_req_indices is None:
      req_np = np.zeros((num_boundary,), dtype=np.int32)
    else:
      req_np = np.array(token_to_req_indices, dtype=np.int32)[boundary_mask]
    boundary_req = jnp.array(np.pad(req_np, (0, pad), constant_values=0))
    kv_slots = generate_kv_slot_mapping(num_boundary, num_pages, page_tokens, 1)
    kv_slots = jnp.pad(kv_slots, (0, pad), constant_values=-1)
    table = make_block_table(block_table_rows, None, pages_for_state, num_pages)
    cos_sin_cache = jax.random.normal(
        k5, (max(int(positions_np.max()) + 1, num_tokens), rope_head_dim)
    )

    def run(cache, seq_along_lane):
      out, _ = pallas_mosaic_tpu_kernel.compress_norm_rope_store(
          jnp.copy(cache),
          boundary_positions,
          jnp.array(table).reshape(-1),
          boundary_req,
          kv_slots,
          rms_weight,
          block_table_stride=table.shape[1],
          state_block_size=state_block_size,
          rope_cache=jnp.zeros((num_pages, 1, 1, 128), dtype=jnp.uint8),
          cos_sin_cache=cos_sin_cache,
          compress_ratio=compress_ratio,
          overlap=True,
          quant_block=quant_block,
          seq_along_lane=seq_along_lane,
      )
      return out

    has_out = run(has_cache, False)
    sal_out = run(sal_cache, True)
    self.assertEqual(sal_out.shape, sal_cache.shape)

    # HAS rows are row-major in `page * 256 + slot`.
    has_records = has_out.reshape(num_pages * page_tokens, -1)[
        jnp.maximum(kv_slots, 0)
    ]
    expected = _scatter_sal_rows_reference(sal_cache, has_records, kv_slots)
    np.testing.assert_array_equal(sal_out, expected)
    # Cross-check the record pages with the reader's own HAS -> SAL repack.
    kv_pages = slice(pages_for_state, num_pages)
    converted = _convert_cache_to_seq_along_lane(has_out[kv_pages], head_dim)
    written = np.zeros((num_pages * page_tokens,), dtype=bool)
    written[np.asarray(kv_slots)[np.asarray(kv_slots) >= 0]] = True
    lane_mask = written.reshape(num_pages, 1, 1, page_tokens)[kv_pages]
    np.testing.assert_array_equal(
        jnp.where(lane_mask, sal_out[kv_pages], 0),
        jnp.where(lane_mask, converted, 0),
    )

  @parameterized.named_parameters(
      # 384 boundary rows: three full 128-lane tiles, filling page 0 and
      # continuing into page 1.
      ("prefill_two_pages", dict(num_tokens=1536)),
      # Boundary and non-boundary decode tokens interleaved over prefilled
      # state; the non-boundary ones leave -1 (padding) slots.
      (
          "decode_mixed",
          dict(
              num_tokens=6,
              positions=np.array([3, 2, 7, 6, 11, 10], dtype=np.int32),
              prefill_len=32,
          ),
      ),
      # 32 boundary rows from 4 requests: one partial tile, multi-row table.
      (
          "multi_request_table",
          dict(
              num_tokens=128,
              token_to_req_indices=np.arange(128, dtype=np.int32) % 4,
              block_table_rows=4,
          ),
      ),
  )
  def test_compress_store_seq_along_lane(self, cfg):
    self.run_seq_along_lane_matches_head_along_sublane(**cfg)


if __name__ == "__main__":
  absltest.main()
