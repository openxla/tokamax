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
"""Compress-and-store benchmark argument specifications.

The kernel runs with bounds checks disabled, so the index arrays are concrete
and mutually consistent, laid out as upstream vllm-torchtpu's serving path and
`test_base.build_inputs` lay them out:

*   Only boundary tokens (`(position + 1) % compress_ratio == 0`) are passed,
    compacted, and every one of them has a compressed-KV slot.
*   Each request's state blocks that a token's window covers (and the one
    before, which the kernel also fetches) map to distinct state pages. Unused
    block table entries point at page 0.
*   The boundary tokens get consecutive compressed-KV slots in the pages after
    the state pages, starting at a page boundary. So the tokens sharing a CSA
    RoPE / indexer cache row are consecutive and fall in the same token tile,
    and no record overwrites a state another token still reads.
"""

from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import numerics
from tokamax._src.autotuning import arg_spec
from tokamax._src.ops.experimental.tpu.compress_store import config

ShapeDtype = jax.ShapeDtypeStruct

_ROPE_HEAD_DIM = 64
# The f32 states the kernel softmax-pools must be finite. These are the bit
# patterns of f32 values in [2**-7, 2): CSA's int32 words hold one f32 each,
# and a uint8 slab row keeps the four bytes of an f32 in its four sub-rows, so
# every byte in [0x3C, 0x40) also gives f32 values in [2**-7, 2).
_STATE_WORD_RANGE = (0x3C000000, 0x40000000)
_STATE_BYTE_RANGE = (0x3C, 0x40)


class _HashableNPArray(np.ndarray):
  """Hashable numpy array for use as an ArgSpec argument."""

  def __new__(cls, input_array):
    return np.asarray(input_array).view(cls)

  def __hash__(self):
    return hash((self.tobytes(), self.shape, self.dtype))


def _int32_array(x) -> _HashableNPArray:
  return _HashableNPArray(np.asarray(x, dtype=np.int32))


def _make_argspec(
    *,
    name: str,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
    physical_page_size: int,
    positions: np.ndarray,
    token_to_req_indices: np.ndarray,
    state_physical_page_size: int | None = None,
    quant_block: int = 64,
    tags: tuple[arg_spec.Tag, ...] = ("primary", "ci_tests", "forward_only"),
) -> arg_spec.ArgSpec:
  """Makes an argspec for the compress-and-store with consistent metadata."""
  positions = positions.astype(np.int64)
  token_to_req_indices = token_to_req_indices.astype(np.int64)
  num_tokens = positions.shape[0]
  if token_to_req_indices.shape != (num_tokens,):
    raise ValueError("positions and token_to_req_indices must match.")
  if np.any((positions + 1) % compress_ratio):
    raise ValueError("Every token must be a boundary token.")
  mode = config.select_mode(head_dim, overlap)

  def make_cfgs(state_block_size: int) -> config.Configs:
    return config.Configs.make(
        mode,
        size_n=num_tokens,
        physical_page_size=physical_page_size,
        state_physical_page_size=state_physical_page_size,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=_ROPE_HEAD_DIM,
        compress_ratio=compress_ratio,
        quant_block=quant_block,
    )

  # As `test_base.build_inputs`: the indexer pages its state at half its page
  # capacity, to share CSA's cache group.
  capacity = make_cfgs(1).state_page_capacity
  if mode is config.Mode.CSA_INDEXER:
    state_block_size = capacity // 2
  else:
    state_block_size = capacity
  cfgs = make_cfgs(state_block_size)

  # The kernel fetches state blocks `[last - (pages_to_buffer - 1), last]` of
  # each token (clamped at 0), a superset of the blocks its window covers.
  last_blocks = positions // state_block_size
  lookback = cfgs.pages_to_buffer_per_token - 1
  block_table_stride = int(last_blocks.max()) + 1
  num_reqs = int(token_to_req_indices.max()) + 1
  block_table = np.zeros((num_reqs * block_table_stride,), dtype=np.int64)
  state_pages = {}
  for req, last in zip(token_to_req_indices, last_blocks, strict=True):
    for block in range(max(int(last) - lookback, 0), int(last) + 1):
      entry = int(req) * block_table_stride + block
      if entry not in state_pages:
        state_pages[entry] = len(state_pages)
        block_table[entry] = state_pages[entry]
  num_state_pages = len(state_pages)

  # Consecutive compressed-KV slots from the first page after the state pages.
  slots_per_page = cfgs.kv_block_size * cfgs.kv_stride
  num_kv_pages = -(-num_tokens * cfgs.kv_stride // slots_per_page)
  num_pages = num_state_pages + num_kv_pages
  kv_slot_mapping = (
      num_state_pages * slots_per_page
      + np.arange(num_tokens, dtype=np.int64) * cfgs.kv_stride
  )

  state_cache = rope_cache = None
  if mode is config.Mode.CSA:
    # Allocated as int32 words (see `csa_cache_layout`); it hosts the state.
    cache = numerics.RangedArrayInitializer(
        (num_pages, physical_page_size, config.LANE),
        jnp.int32,
        *_STATE_WORD_RANGE,
    )
    rope_cache = ShapeDtype(
        (num_pages, cfgs.rope_page_size, config.LANE), jnp.int32
    )
  elif mode is config.Mode.CSA_INDEXER:
    # The uint8 cache hosts the state.
    cache = numerics.RangedArrayInitializer(
        cfgs.cache_shape(num_pages), jnp.uint8, *_STATE_BYTE_RANGE
    )
  else:
    # HCA keeps its state in a separate array.
    cache = ShapeDtype(cfgs.cache_shape(num_pages), jnp.uint8)
    state_cache = numerics.RangedArrayInitializer(
        cfgs.state_cache_shape(num_pages), jnp.uint8, *_STATE_BYTE_RANGE
    )

  return arg_spec.ArgSpec(
      args={
          "cache": cache,
          "positions": _int32_array(positions),
          "block_table": _int32_array(block_table),
          "token_to_req_indices": _int32_array(token_to_req_indices),
          "kv_slot_mapping": _int32_array(kv_slot_mapping),
          "rms_weight": ShapeDtype((head_dim,), jnp.float32),
          "cos_sin_cache": ShapeDtype(
              (int(positions.max()) + 1, _ROPE_HEAD_DIM), jnp.float32
          ),
          "block_table_stride": block_table_stride,
          "state_block_size": state_block_size,
          "compress_ratio": compress_ratio,
          "overlap": overlap,
          "state_cache": state_cache,
          "rope_cache": rope_cache,
          "quant_block": quant_block,
      },
      project="inference",
      name=name,
      tags=tags,
  )


# DeepSeek-V4's three compressors, with the page sizes upstream allocates for a
# 1024-token KV cache block (see `test_base`).
_CSA = dict(
    head_dim=512, compress_ratio=4, overlap=True, physical_page_size=256
)
_HCA = dict(
    head_dim=512,
    compress_ratio=128,
    overlap=False,
    physical_page_size=16,
    state_physical_page_size=256,
)
_CSA_INDEXER = dict(
    head_dim=128,
    compress_ratio=4,
    overlap=True,
    physical_page_size=32,
    quant_block=128,
)

_DECODE_BATCH = 128
_PREFILL_LEN = 8192


def _decode_positions(compress_ratio: int) -> np.ndarray:
  """One boundary token per request, at context lengths from 4096 to 8128."""
  context_lens = 4096 + 64 * ((37 * np.arange(_DECODE_BATCH)) % 64)
  assert not np.any(context_lens % compress_ratio)
  return context_lens - 1


def _prefill_positions(compress_ratio: int) -> np.ndarray:
  """The boundary tokens of one request's prefill."""
  return np.arange(compress_ratio - 1, _PREFILL_LEN, compress_ratio)


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _make_argspec(
        name=f"dsv4_csa_decode_b{_DECODE_BATCH}",
        positions=_decode_positions(4),
        token_to_req_indices=np.arange(_DECODE_BATCH),
        **_CSA,
    ),
    _make_argspec(
        name=f"dsv4_csa_prefill_t{_PREFILL_LEN}",
        positions=_prefill_positions(4),
        token_to_req_indices=np.zeros(_PREFILL_LEN // 4, dtype=np.int32),
        **_CSA,
    ),
    _make_argspec(
        name=f"dsv4_hca_prefill_t{_PREFILL_LEN}",
        positions=_prefill_positions(128),
        token_to_req_indices=np.zeros(_PREFILL_LEN // 128, dtype=np.int32),
        **_HCA,
    ),
    _make_argspec(
        name=f"dsv4_csa_indexer_prefill_t{_PREFILL_LEN}",
        positions=_prefill_positions(4),
        token_to_req_indices=np.zeros(_PREFILL_LEN // 4, dtype=np.int32),
        **_CSA_INDEXER,
    ),
)
