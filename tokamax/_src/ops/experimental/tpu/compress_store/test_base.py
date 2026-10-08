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
"""Shared correctness tests for compress-and-store implementations."""

from collections.abc import Sequence
import dataclasses
import functools
from typing import Any

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.compress_store import config
from tokamax._src.ops.experimental.tpu.compress_store import csa_cache_layout
from tokamax._src.ops.experimental.tpu.compress_store import reference
from tokamax._src.ops.experimental.tpu.compress_store import test_utils

_proj_and_save_state = jax.jit(
    test_utils.ref_wkv_proj_and_save_state, static_argnums=(6, 7, 8, 9)
)
_ref_compress_norm_rope_store = jax.jit(
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


def generate_kv_slot_mapping(
    num_boundary_tokens: int,
    num_pages: int,
    page_size: int,
    slots_per_part_out: int,
) -> np.ndarray:
  """Generates kv_slot_mapping for boundary tokens, packed into the last pages."""
  kv_slot_mapping_np = np.full((num_boundary_tokens,), -1, dtype=np.int32)
  boundary_count = 0
  total_slots_needed = num_boundary_tokens * slots_per_part_out
  pages_needed = (total_slots_needed + page_size - 1) // page_size
  start_page = max(0, num_pages - pages_needed)

  for t in range(num_boundary_tokens):
    kv_slot_mapping_np[t] = start_page * page_size + boundary_count
    boundary_count += slots_per_part_out
  return kv_slot_mapping_np


def normalize_fp8_zero_sign(arr: jax.Array) -> jax.Array:
  """Maps fp8 -0 bytes to +0."""
  is_zero = (arr & 0x7F) == 0
  return jnp.where(is_zero, 0, arr)


def normalize_bf16_zero_sign(arr: jax.Array) -> jax.Array:
  """Maps bf16 -0 to +0 in little-endian uint8 bytes."""
  orig_shape = arr.shape
  arr_2d = arr.reshape(-1, 2)
  even = arr_2d[..., 0]
  odd = arr_2d[..., 1]
  is_zero = (even == 0) & ((odd & 0x7F) == 0)
  new_odd = jnp.where(is_zero, odd & 0x7F, odd)
  normalized = jnp.stack([even, new_odd], axis=-1)
  return normalized.reshape(orig_shape)


def normalize_bf16_words_zero_sign(words: jax.Array) -> jax.Array:
  """`normalize_bf16_zero_sign` for int32 words holding two bf16 each."""
  u = jax.lax.bitcast_convert_type(words, jnp.uint32)
  lo, hi = u & 0xFFFF, u >> 16
  lo = jnp.where((lo & 0x7FFF) == 0, 0, lo)
  hi = jnp.where((hi & 0x7FFF) == 0, 0, hi)
  return jax.lax.bitcast_convert_type(lo | (hi << 16), jnp.int32)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompressStoreInputs:
  """The arguments of one compress-and-store call.

  `slot_mapping` is the state-cache slot of each token, which only the
  reference takes.
  """

  cache: jax.Array
  positions: jax.Array
  block_table: jax.Array
  token_to_req_indices: jax.Array
  kv_slot_mapping: jax.Array
  rms_weight: jax.Array
  cos_sin_cache: jax.Array
  state_cache: jax.Array | None
  rope_cache: jax.Array | None
  slot_mapping: jax.Array
  block_table_stride: int
  state_block_size: int
  compress_ratio: int
  overlap: bool
  quant_block: int
  rms_eps: float

  @property
  def args(self) -> tuple[jax.Array, ...]:
    return (
        self.cache,
        self.positions,
        self.block_table,
        self.token_to_req_indices,
        self.kv_slot_mapping,
        self.rms_weight,
    )

  @property
  def kwargs(self) -> dict[str, Any]:
    return dict(
        cos_sin_cache=self.cos_sin_cache,
        block_table_stride=self.block_table_stride,
        state_block_size=self.state_block_size,
        compress_ratio=self.compress_ratio,
        overlap=self.overlap,
        state_cache=self.state_cache,
        rope_cache=self.rope_cache,
        quant_block=self.quant_block,
        rms_eps=self.rms_eps,
    )

  def with_cache_copies(self) -> "CompressStoreInputs":
    """Returns these inputs with fresh copies of the donated caches."""
    return dataclasses.replace(
        self,
        cache=jnp.copy(self.cache),
        rope_cache=None
        if self.rope_cache is None
        else jnp.copy(self.rope_cache),
    )

  def reference(self) -> tuple[jax.Array, jax.Array | None]:
    """Runs `reference.ref_compress_norm_rope_store` the way upstream does."""
    head_dim = self.rms_weight.shape[0]
    rope_head_dim = self.cos_sin_cache.shape[1]
    has_rope_cache = self.rope_cache is not None
    # Upstream passes a dummy RoPE cache outside CSA; it is returned as is.
    rope_cache = (
        self.rope_cache
        if has_rope_cache
        else jnp.zeros((self.cache.shape[0], 1, 1, 128), jnp.uint8)
    )
    cache, rope_cache = _ref_compress_norm_rope_store(
        cache=self.cache,
        rope_cache=rope_cache,
        positions=self.positions,
        slot_mapping=self.slot_mapping,
        block_table=self.block_table,
        token_to_req_indices=self.token_to_req_indices,
        kv_slot_mapping=self.kv_slot_mapping,
        rms_weight=self.rms_weight,
        cos_sin_cache=self.cos_sin_cache,
        block_table_stride=self.block_table_stride,
        state_block_size=self.state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=self.compress_ratio,
        overlap=self.overlap,
        rms_eps=self.rms_eps,
        quant_block=self.quant_block,
        is_quantized=self.overlap,
        has_rope=rope_head_dim > 0,
        has_rope_cache=has_rope_cache,
        state_cache=self.state_cache,
    )
    return cache, rope_cache if has_rope_cache else None


def build_inputs(
    *,
    num_tokens: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    physical_page_size: int,
    state_physical_page_size: int | None = None,
    quant_block: int = 64,
    rms_eps: float = 1e-6,
    positions: Sequence[int] | None = None,
    prefill_len: int | None = None,
    hidden_size: int = 7168,
    seed: int = 0,
    shuffle: bool = False,
) -> CompressStoreInputs:
  """Builds consistent, in-bounds inputs as upstream vllm-torchtpu's test does.

  Runs the Kernel 1 reference over `prefill_len` tokens to populate the state
  cache, keeps the boundary tokens of `positions` (default `range(num_tokens)`)
  and pads them to `num_tokens` with position 0, request 0 and no slot. The
  boundary tokens get consecutive compressed-KV slots in the last pages.

  Args:
    num_tokens: The number of tokens after padding.
    head_dim: Head dimension; with `overlap`, selects the mode.
    rope_head_dim: RoPE dimension.
    compress_ratio: Tokens compressed into one record.
    overlap: Whether windows overlap.
    physical_page_size: Rows per page of the compressed-KV cache.
    state_physical_page_size: Rows per page of the state array (HCA).
    quant_block: FP8 quantization block.
    rms_eps: RMSNorm epsilon.
    positions: The token positions, at most `num_tokens` of them.
    prefill_len: Tokens whose state Kernel 1 saves; defaults to `num_tokens`.
    hidden_size: Hidden size of the Kernel 1 projection.
    seed: The random seed.
    shuffle: Whether to shuffle the state pages in the block table and the
      compressed-KV slots within each aligned group of 4 boundary tokens (the
      tokens that share a CSA RoPE / indexer cache row).

  Returns:
    The inputs.
  """
  mode = config.select_mode(head_dim, overlap)

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

  # `state_block_size` is vLLM's paging granularity, an input rather than a
  # property of the page: `config.state_block_size` floors the indexer
  # to CSA's smaller capacity (always exactly half its own) so the two share a
  # cache group, leaving the tail of every indexer page unused. The capacity
  # itself does not depend on it, so probe for it first.
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

  # Identify the boundary tokens.
  if positions is None:
    positions_np = np.arange(num_tokens, dtype=np.int32)
  else:
    positions_np = np.asarray(positions, dtype=np.int32)
  boundary_mask = ((positions_np + 1) % compress_ratio) == 0
  positions_filtered = positions_np[boundary_mask]
  num_boundary = positions_filtered.shape[0]
  assert num_boundary <= num_tokens

  run_1_tokens = prefill_len if prefill_len is not None else num_tokens
  pages_for_state = -(-run_1_tokens // state_block_size)
  page_bytes = cfgs.physical_page_size * cfgs.row_size_bytes
  pages_for_kv_cache = -(-num_boundary * cfgs.record_bytes // page_bytes)
  num_pages = pages_for_state + pages_for_kv_cache

  rng = np.random.default_rng(seed)
  # The block table maps request 0's state blocks to the first pages, and the
  # remaining pages to themselves; the compressed records land in the latter.
  state_pages = np.arange(pages_for_state, dtype=np.int32)
  if shuffle:
    state_pages = rng.permutation(state_pages)
  block_table_np = np.concatenate(
      [state_pages, np.arange(pages_for_state, num_pages, dtype=np.int32)]
  )

  def state_slot(token_index):
    """Physical state row of a token, the way `derive_metadata` maps it.

    Tokens are page-strided, not densely packed: a page holds
    `state_page_capacity` states but is only paged at `state_block_size` of
    them.

    Args:
      token_index: The token position.

    Returns:
      The state row.
    """
    page = block_table_np[token_index // state_block_size]
    return (
        page * cfgs.state_physical_page_size
        + (token_index % state_block_size) * cfgs.state_rows_per_token
    )

  # Populate the state cache.
  k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(seed), 5)
  hidden_states = jax.random.normal(k1, (run_1_tokens, hidden_size))
  wkv_wgate = jax.random.normal(k2, (hidden_size, state_dim))
  ape = jax.random.normal(k3, (compress_ratio, state_width))
  run_1_positions = jnp.arange(run_1_tokens, dtype=jnp.int32)
  run_1_slot_mapping = jnp.asarray(
      state_slot(np.arange(run_1_tokens)), dtype=jnp.int32
  )

  init_cache = jnp.zeros(cfgs.cache_shape(num_pages), dtype=jnp.uint8)
  init_state_cache = jnp.zeros(
      cfgs.state_cache_shape(num_pages), dtype=jnp.uint8
  )
  # The state scatter targets the state array, which is `init_cache` itself in
  # the shared-buffer layout.
  populated_state_cache = _proj_and_save_state(
      hidden_states,
      wkv_wgate,
      ape,
      run_1_positions,
      run_1_slot_mapping,
      init_state_cache if separate_state else init_cache,
      state_block_size,
      head_dim,
      compress_ratio,
      overlap,
  )
  if separate_state:
    populated_cache = init_cache
  else:
    populated_cache = populated_state_cache
    populated_state_cache = None
  if cfgs.dims.has_rope_cache:
    # CSA's cache is allocated as int32 words (see csa_cache_layout).
    populated_cache = csa_cache_layout.slabs_to_words(populated_cache)

  kv_slot_mapping_filtered = generate_kv_slot_mapping(
      num_boundary,
      num_pages,
      cfgs.kv_block_size * cfgs.kv_stride,
      cfgs.kv_stride,
  )
  if shuffle:
    for start in range(0, num_boundary, 4):
      group = kv_slot_mapping_filtered[start : start + 4]
      kv_slot_mapping_filtered[start : start + 4] = rng.permutation(group)

  # Pad to num_tokens.
  pad = num_tokens - num_boundary
  positions_padded = np.pad(positions_filtered, (0, pad), constant_values=0)
  kv_slot_mapping = np.pad(
      kv_slot_mapping_filtered, (0, pad), constant_values=-1
  )

  rms_weight = jax.random.normal(k4, (head_dim,))
  max_pos = int(np.max(positions_padded)) + 1 if num_tokens > 0 else 0
  cos_sin_cache_len = max(max_pos, num_tokens)
  cos_sin_cache = jax.random.normal(k5, (cos_sin_cache_len, rope_head_dim))

  if cfgs.dims.has_rope_cache:
    num_rope_pages, rope_rows, _, lanes = cfgs.rope_cache_shape(num_pages)
    rope_cache = jnp.zeros((num_rope_pages, rope_rows, lanes), dtype=jnp.int32)
  else:
    rope_cache = None

  return CompressStoreInputs(
      cache=populated_cache,
      positions=jnp.asarray(positions_padded, jnp.int32),
      block_table=jnp.asarray(block_table_np, jnp.int32),
      token_to_req_indices=jnp.zeros((num_tokens,), jnp.int32),
      kv_slot_mapping=jnp.asarray(kv_slot_mapping, jnp.int32),
      rms_weight=rms_weight,
      cos_sin_cache=cos_sin_cache,
      state_cache=populated_state_cache,
      rope_cache=rope_cache,
      slot_mapping=jnp.asarray(state_slot(positions_padded), jnp.int32),
      block_table_stride=num_pages,
      state_block_size=state_block_size,
      compress_ratio=compress_ratio,
      overlap=overlap,
      quant_block=quant_block,
      rms_eps=rms_eps,
  )


_CSA: dict[str, Any] = dict(
    head_dim=512,
    rope_head_dim=64,
    compress_ratio=4,
    overlap=True,
    physical_page_size=256,
)
_HCA: dict[str, Any] = dict(
    head_dim=512,
    rope_head_dim=64,
    compress_ratio=128,
    overlap=False,
    physical_page_size=16,
    state_physical_page_size=256,
)
_CSA_INDEXER: dict[str, Any] = dict(
    head_dim=128,
    rope_head_dim=64,
    compress_ratio=4,
    overlap=True,
    physical_page_size=32,
    quant_block=128,
)

# The named cases of upstream vllm-torchtpu's `compress_store_test.py`.
CASES: dict[str, dict[str, Any]] = {
    # DSv4-Flash's real geometry: hidden_size 4096 vs state_dim 2048.
    "csa_prod_hidden_4096": dict(_CSA, num_tokens=8, hidden_size=4096),
    "csa_prefill": dict(_CSA, num_tokens=128),
    "hca_prefill": dict(_HCA, num_tokens=256),
    "hca_prefill_small": dict(_HCA, num_tokens=128),
    "csa_decode_batch_large": dict(
        _CSA,
        num_tokens=1024,
        positions=tuple((np.arange(1024) * 3) % 1024),
        prefill_len=1024,
    ),
    "csa_decode_batch_seq": dict(
        _CSA, num_tokens=4, positions=(3, 7, 11, 15), prefill_len=16
    ),
    "hca_decode_batch_seq": dict(
        _HCA, num_tokens=4, positions=(127, 255, 383, 511), prefill_len=512
    ),
    "csa_decode_batch_random": dict(
        _CSA, num_tokens=4, positions=(11, 3, 19, 7), prefill_len=32
    ),
    "hca_decode_batch_random": dict(
        _HCA, num_tokens=4, positions=(383, 127, 511, 255), prefill_len=512
    ),
    "csa_decode_batch_mixed": dict(
        _CSA, num_tokens=6, positions=(3, 2, 7, 6, 11, 10), prefill_len=32
    ),
    "hca_decode_batch_mixed": dict(
        _HCA,
        num_tokens=6,
        positions=(127, 126, 255, 254, 383, 382),
        prefill_len=384,
    ),
    "csa_indexer_prefill": dict(_CSA_INDEXER, num_tokens=128),
    "csa_indexer_decode": dict(
        _CSA_INDEXER, num_tokens=4, positions=(3, 7, 11, 15), prefill_len=16
    ),
}


@functools.cache
def make_case_inputs(name: str) -> CompressStoreInputs:
  """Returns the (cached) inputs of the named upstream case."""
  return build_inputs(**CASES[name])


def assert_matches_reference(
    inputs: CompressStoreInputs,
    out: tuple[jax.Array, jax.Array | None],
):
  """Checks the caches are bit-exact against the reference, up to zero signs.

  Mirrors upstream: fp8 (CSA / indexer) and bf16 (HCA, RoPE) -0 are mapped to
  +0 before comparing, since the kernel and the reference may round a tiny
  value to zeros of either sign.

  Args:
    inputs: The inputs.
    out: The `(cache, rope_cache)` returned by the implementation under test.
  """
  cache_out, rope_out = out
  cache_ref, rope_ref = inputs.reference()
  np.testing.assert_equal(cache_out.shape, inputs.cache.shape)
  np.testing.assert_equal(cache_out.dtype, inputs.cache.dtype)
  if csa_cache_layout.is_word_array(cache_out.dtype):
    cache_out = csa_cache_layout.words_to_slabs(cache_out)
    cache_ref = csa_cache_layout.words_to_slabs(cache_ref)
  if inputs.overlap:
    normalize = normalize_fp8_zero_sign
  else:
    normalize = normalize_bf16_zero_sign
  np.testing.assert_array_equal(normalize(cache_out), normalize(cache_ref))

  if inputs.rope_cache is None:
    assert rope_out is None, "Expected no RoPE cache outside CSA."
  else:
    assert rope_out is not None, "Expected a RoPE cache for CSA."
    assert rope_ref is not None
    np.testing.assert_equal(rope_out.dtype, jnp.int32)
    np.testing.assert_array_equal(
        normalize_bf16_words_zero_sign(rope_out),
        normalize_bf16_words_zero_sign(rope_ref),
    )


class CompressStoreTestBase(parameterized.TestCase):
  """Correctness suite shared by all compress-and-store implementations.

  Subclasses pass the implementation under test as `compress_store_fn`. Its
  updated caches must match `reference.ref_compress_norm_rope_store`
  bit-exactly, up to the sign of zeros.
  """

  def __init__(self, *args, compress_store_fn):
    super().__init__(*args)
    self._compress_store_fn = compress_store_fn

  @parameterized.named_parameters((name, name) for name in CASES)
  def test_compress_store(self, case_name):
    inputs = make_case_inputs(case_name)
    # Implementations may donate the caches, as upstream's kernel does.
    call_inputs = inputs.with_cache_copies()
    out = self._compress_store_fn(*call_inputs.args, **call_inputs.kwargs)
    assert_matches_reference(inputs, out)
