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
"""Hypothesis tests for the compress-and-store Pallas kernel on TPU.

These call the raw kernel entry point, `compress_norm_rope_store`, and check it
bit-exactly (up to the sign of zeros) against `reference`. They sweep the mode,
`tile_n`, the number of requests and their block tables, the boundary tokens,
the slot-less tokens around them, the page geometry and the initial cache
contents.
"""

from collections.abc import Sequence
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
import hypothesis.strategies as hps
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.compress_store import config
from tokamax._src.ops.experimental.tpu.compress_store import csa_cache_layout
from tokamax._src.ops.experimental.tpu.compress_store import pallas_mosaic_tpu_kernel
from tokamax._src.ops.experimental.tpu.compress_store import test_base
from tokamax._src.ops.experimental.tpu.proj_and_save_state import reference as proj_and_save_state_ref

jax.config.parse_flags_with_absl()


def _tpu_older_than_7x() -> bool:
  """Whether the default device is not a TPU7x or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 7
  )


hp.settings.register_profile(
    name="deterministic",
    database=None,
    derandomize=True,
    deadline=None,
    max_examples=10,
    print_blob=True,
    verbosity=hp.Verbosity.verbose,
)
hp.settings.load_profile(name="deterministic")

_proj_and_save_state = jax.jit(
    proj_and_save_state_ref.ref_wkv_proj_and_save_state,
    static_argnums=(6, 7, 8, 9),
)

_MODES = ("csa", "hca", "csa_indexer")

# Per mode: the case geometry, the number of compressed windows per request
# whose state Kernel 1 populates, and the maximum number of boundary tokens per
# request.
_CASES: dict[str, tuple[dict[str, Any], int, int]] = {
    "csa": (test_base.CASES["csa_prefill"], 32, 12),
    "hca": (test_base.CASES["hca_prefill_small"], 8, 8),
    "csa_indexer": (test_base.CASES["csa_indexer_prefill"], 32, 12),
}

# Page geometries to sweep, as overrides of the case geometry. CSA and the
# indexer host the state in the compressed-KV cache, so their page size also
# sets the state page capacity (2-16 and 8-32 tokens), and with it how many
# pages one window spans. HCA's state array pages separately.
_PAGE_GEOMETRIES: dict[str, list[dict[str, int]]] = {
    "csa": [dict(physical_page_size=p) for p in (32, 64, 128, 256)],
    "hca": [
        dict(physical_page_size=p, state_physical_page_size=s)
        for p in (16, 32)
        for s in (128, 256)
    ],
    "csa_indexer": [dict(physical_page_size=p) for p in (16, 32, 64)],
}


def _tile_ns(overlap: bool) -> list[int]:
  """The `tile_n` values the kernel supports in a mode.

  CSA and the indexer need a multiple of 4 (4 tokens share a cache row). HCA
  buffers larger state pages and runs out of scoped VMEM at `tile_n >= 16`.

  Args:
    overlap: Whether windows overlap (CSA and the indexer).

  Returns:
    The `tile_n` values to draw from.
  """
  return [4, 8, 12, 16, 32] if overlap else [4, 8, 12]


def _build_inputs(
    *,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    physical_page_size: int,
    state_physical_page_size: int | None = None,
    quant_block: int = 64,
    rms_eps: float = 1e-6,
    num_windows: int,
    windows: Sequence[Sequence[int]],
    tile_n: int,
    boundary_per_tile: int | None = None,
    num_empty_tiles: int = 0,
    mixed_fillers: bool = False,
    shuffle_tiles: bool = False,
    extra_stride: int = 0,
    random_init: bool = False,
    hidden_size: int = 1024,
    seed: int = 0,
) -> test_base.CompressStoreInputs:
  """Builds consistent, in-bounds kernel inputs for several requests.

  Generalizes `test_base.build_inputs`: each request `r` gets its own block
  table row (shuffled state pages) and Kernel 1 populates the state of its
  first `num_windows * compress_ratio` positions. Request `r` stores windows
  `windows[r]`, i.e. boundary tokens at positions `(w + 1) * compress_ratio -
  1`, in a random interleaving, with consecutive compressed-KV slots
  (shuffled within each aligned group of 4) in the last pages.

  The tokens are laid out tile by tile: each tile holds up to
  `boundary_per_tile` boundary tokens, then slot-less filler tokens up to
  `tile_n`. As upstream's caller does, the 4 tokens that share a CSA RoPE /
  indexer cache row stay in one tile: the kernel read-modify-writes that row
  once per tile, so two tiles writing the same row would race.

  Args:
    head_dim: Head dimension; with `overlap`, selects the mode.
    rope_head_dim: RoPE dimension.
    compress_ratio: Tokens compressed into one record.
    overlap: Whether windows overlap.
    physical_page_size: Rows per page of the compressed-KV cache.
    state_physical_page_size: Rows per page of the state array (HCA).
    quant_block: FP8 quantization block.
    rms_eps: RMSNorm epsilon.
    num_windows: Windows per request whose state Kernel 1 populates.
    windows: The windows each request stores, each in `[0, num_windows)`.
    tile_n: Tokens per grid step.
    boundary_per_tile: Boundary tokens per tile, a multiple of 4 up to `tile_n`;
      defaults to `tile_n`.
    num_empty_tiles: Tiles of filler tokens only, inserted between the others.
    mixed_fillers: Whether filler tokens also include non-boundary tokens and
      boundary tokens without a slot, of random requests and positions, rather
      than only upstream's padding (position 0, request 0, no slot).
    shuffle_tiles: Whether to shuffle the tokens within each tile.
    extra_stride: Unused block table columns per request.
    random_init: Whether the compressed-KV (and RoPE) cache starts with random
      bytes rather than zeros, so that the bytes the kernel must not touch are
      checked as well.
    hidden_size: Hidden size of the Kernel 1 projection.
    seed: The random seed.

  Returns:
    The inputs.
  """
  if boundary_per_tile is None:
    boundary_per_tile = tile_n
  assert boundary_per_tile % 4 == 0 and 0 < boundary_per_tile <= tile_n
  rng = np.random.default_rng(seed)
  mode = config.select_mode(head_dim, overlap)

  def make_cfgs(state_block_size):
    return config.Configs.make(
        mode,
        size_n=tile_n,
        physical_page_size=physical_page_size,
        state_physical_page_size=state_physical_page_size,
        rms_eps=rms_eps,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        quant_block=quant_block,
        state_block_size=state_block_size,
    )

  # As `test_base.build_inputs`: the indexer pages at half its capacity.
  state_page_capacity = make_cfgs(1).state_page_capacity
  state_block_size = (
      state_page_capacity // 2
      if mode is config.Mode.CSA_INDEXER
      else state_page_capacity
  )
  cfgs = make_cfgs(state_block_size)
  separate_state = mode is config.Mode.HCA
  state_width = cfgs.state_width

  num_reqs = len(windows)
  run_tokens = num_windows * compress_ratio
  pages_per_req = -(-run_tokens // state_block_size)
  num_state_pages = num_reqs * pages_per_req

  boundary = [
      (r, (w + 1) * compress_ratio - 1)
      for r, req_windows in enumerate(windows)
      for w in req_windows
  ]
  boundary = [boundary[i] for i in rng.permutation(len(boundary))]
  num_boundary = len(boundary)
  page_bytes = cfgs.physical_page_size * cfgs.row_size_bytes
  pages_for_kv_cache = -(-num_boundary * cfgs.record_bytes // page_bytes)
  num_pages = num_state_pages + pages_for_kv_cache

  # Request r's state blocks map to a random set of the first pages; the
  # compressed records land in the remaining ones. Neither the kernel nor the
  # reference reads past a token's own block, so the extra columns hold
  # arbitrary pages.
  block_table_stride = pages_per_req + extra_stride
  block_table = rng.integers(
      0, num_pages, (num_reqs, block_table_stride), dtype=np.int32
  )
  block_table[:, :pages_per_req] = rng.permutation(num_state_pages).reshape(
      num_reqs, pages_per_req
  )

  def state_slot(req, pos):
    req, pos = np.asarray(req), np.asarray(pos)
    page = block_table[req, pos // state_block_size]
    return (
        page * cfgs.state_physical_page_size
        + (pos % state_block_size) * cfgs.state_rows_per_token
    )

  # Populate the state cache of every request.
  k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(seed), 5)
  hidden_states = jax.random.normal(k1, (num_reqs * run_tokens, hidden_size))
  wkv_wgate = jax.random.normal(k2, (hidden_size, 2 * state_width))
  ape = jax.random.normal(k3, (compress_ratio, state_width))
  run_reqs = np.repeat(np.arange(num_reqs), run_tokens)
  run_positions = np.tile(np.arange(run_tokens), num_reqs)

  init_cache = jnp.zeros(cfgs.cache_shape(num_pages), dtype=jnp.uint8)
  init_state_cache = jnp.zeros(
      cfgs.state_cache_shape(num_pages), dtype=jnp.uint8
  )
  populated_state_cache = _proj_and_save_state(
      hidden_states,
      wkv_wgate,
      ape,
      jnp.asarray(run_positions, jnp.int32),
      jnp.asarray(state_slot(run_reqs, run_positions), jnp.int32),
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
  if random_init:
    # Only the pages without state: the windows read the state pages.
    populated_cache = populated_cache.at[num_state_pages:].set(
        rng.integers(
            0, 256, populated_cache[num_state_pages:].shape, dtype=np.uint8
        )
    )
  if cfgs.dims.has_rope_cache:
    # CSA's cache is allocated as int32 words (see csa_cache_layout).
    populated_cache = csa_cache_layout.slabs_to_words(populated_cache)

  kv_slots = test_base.generate_kv_slot_mapping(
      num_boundary,
      num_pages,
      cfgs.kv_block_size * cfgs.kv_stride,
      cfgs.kv_stride,
  )
  for start in range(0, num_boundary, 4):
    kv_slots[start : start + 4] = rng.permutation(kv_slots[start : start + 4])

  def filler() -> tuple[int, int, int]:
    """A slot-less `(request, position, kv_slot)` token."""
    kind = rng.integers(3) if mixed_fillers else 0
    if kind == 0:  # Upstream's padding.
      return 0, 0, -1
    req = int(rng.integers(num_reqs))
    window = int(rng.integers(num_windows))
    if kind == 1:  # A non-boundary token.
      offset = int(rng.integers(compress_ratio - 1))
      return req, window * compress_ratio + offset, -1
    # A boundary token without a slot.
    return req, (window + 1) * compress_ratio - 1, -1

  tiles = [
      [
          (*boundary[b], int(kv_slots[b]))
          for b in range(start, min(start + boundary_per_tile, num_boundary))
      ]
      for start in range(0, num_boundary, boundary_per_tile)
  ]
  for _ in range(num_empty_tiles):
    tiles.insert(int(rng.integers(len(tiles) + 1)), [])
  tokens = []
  for tile in tiles:
    tile = tile + [filler() for _ in range(tile_n - len(tile))]
    if shuffle_tiles:
      tile = [tile[i] for i in rng.permutation(tile_n)]
    tokens += tile
  reqs, positions, kv_slot_mapping = (np.asarray(x) for x in zip(*tokens))
  num_tokens = len(tokens)

  rms_weight = jax.random.normal(k4, (head_dim,))
  cos_sin_cache_len = max(int(np.max(positions)) + 1, num_tokens)
  cos_sin_cache = jax.random.normal(k5, (cos_sin_cache_len, rope_head_dim))

  if cfgs.dims.has_rope_cache:
    num_rope_pages, rope_rows, _, lanes = cfgs.rope_cache_shape(num_pages)
    rope_shape = (num_rope_pages, rope_rows, lanes)
    if random_init:
      rope_cache = jnp.asarray(
          rng.integers(-(2**31), 2**31, rope_shape, dtype=np.int32)
      )
    else:
      rope_cache = jnp.zeros(rope_shape, dtype=jnp.int32)
  else:
    rope_cache = None

  return test_base.CompressStoreInputs(
      cache=populated_cache,
      positions=jnp.asarray(positions, jnp.int32),
      block_table=jnp.asarray(block_table.reshape(-1), jnp.int32),
      token_to_req_indices=jnp.asarray(reqs, jnp.int32),
      kv_slot_mapping=jnp.asarray(kv_slot_mapping, jnp.int32),
      rms_weight=rms_weight,
      cos_sin_cache=cos_sin_cache,
      state_cache=populated_state_cache,
      rope_cache=rope_cache,
      slot_mapping=jnp.asarray(state_slot(reqs, positions), jnp.int32),
      block_table_stride=block_table_stride,
      state_block_size=state_block_size,
      compress_ratio=compress_ratio,
      overlap=overlap,
      quant_block=quant_block,
      rms_eps=rms_eps,
  )


def _case_kwargs(case: dict[str, Any]) -> dict[str, Any]:
  """The geometry of a `test_base.CASES` entry, without its token layout."""
  keys = (
      "head_dim",
      "rope_head_dim",
      "compress_ratio",
      "overlap",
      "physical_page_size",
      "state_physical_page_size",
      "quant_block",
  )
  return {k: case[k] for k in keys if k in case}


def _draw_windows(data, num_reqs, num_windows, max_boundary, min_size=1):
  return [
      data.draw(
          hps.lists(
              hps.integers(0, num_windows - 1),
              min_size=min_size,
              max_size=max_boundary,
              unique=True,
          ),
          label=f"windows[{r}]",
      )
      for r in range(num_reqs)
  ]


def _run_kernel(inputs: test_base.CompressStoreInputs, tile_n: int):
  # The kernel donates the caches, which `assert_matches_reference` reads.
  call_inputs = inputs.with_cache_copies()
  return pallas_mosaic_tpu_kernel.compress_norm_rope_store(
      *call_inputs.args, **call_inputs.kwargs, tile_n=tile_n
  )


def _hca_bf16_bits(cache: jax.Array) -> np.ndarray:
  """The bf16 bit patterns of an HCA cache's `[..., 4, 128]` uint8 rows.

  Sub-rows `2 k` and `2 k + 1` of a row hold the low and high bytes of 128 bf16
  values (see `reference`).

  Args:
    cache: The uint8 HCA cache.

  Returns:
    The uint16 bit patterns, `[..., 2, 128]`.
  """
  b = np.asarray(cache).astype(np.uint16)
  b = b.reshape(*b.shape[:-2], 2, 2, b.shape[-1])
  return b[..., 0, :] | (b[..., 1, :] << 8)


def _assert_matches_reference(
    inputs: test_base.CompressStoreInputs,
    out: tuple[jax.Array, jax.Array | None],
):
  """`test_base.assert_matches_reference`, up to 1 bf16 ulp in HCA.

  The kernel and the reference sum the window in different orders, so their
  f32 records can differ in the last bits. For the FP8 modes (CSA, the
  indexer) that has not been seen to change a stored byte, but HCA stores bf16
  records of 128-token windows and rarely lands on the other side of a bf16
  rounding boundary (seen once in ~60 random draws). Upstream's fixed-seed
  cases do not hit it and compare bit-exactly.

  Args:
    inputs: The inputs.
    out: The `(cache, rope_cache)` returned by the kernel.
  """
  if inputs.overlap:
    test_base.assert_matches_reference(inputs, out)
    return
  cache_out, rope_out = out
  cache_ref, _ = inputs.reference()
  assert rope_out is None, "Expected no RoPE cache outside CSA."
  np.testing.assert_equal(cache_out.shape, inputs.cache.shape)
  np.testing.assert_equal(cache_out.dtype, inputs.cache.dtype)
  got, want = _hca_bf16_bits(cache_out), _hca_bf16_bits(cache_ref)

  def to_line(bits):
    # Sign-magnitude bits to a monotonic integer line; maps -0 to +0.
    magnitude = (bits & 0x7FFF).astype(np.int32)
    return np.where(bits & 0x8000, -magnitude, magnitude)

  ulps = np.abs(to_line(got) - to_line(want))
  np.testing.assert_array_less(ulps, 2, err_msg="bf16 ulps from reference")


class PallasMosaicTpuKernelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")

  @hp.given(hps.data())
  def test_kernel_matches_reference(self, data):
    """Sweeps the requests, the token layout and the initial cache contents."""
    mode = data.draw(hps.sampled_from(_MODES), label="mode")
    case, num_windows, max_boundary = _CASES[mode]
    tile_n = data.draw(hps.sampled_from(_tile_ns(case["overlap"])), label="tn")
    num_reqs = data.draw(hps.integers(1, 3), label="num_reqs")
    windows = _draw_windows(data, num_reqs, num_windows, max_boundary)
    boundary_per_tile = 4 * data.draw(
        hps.integers(1, tile_n // 4), label="boundary_per_tile // 4"
    )
    inputs = _build_inputs(
        **_case_kwargs(case),
        rms_eps=data.draw(hps.sampled_from([1e-6, 1e-5]), label="rms_eps"),
        num_windows=num_windows,
        windows=windows,
        tile_n=tile_n,
        boundary_per_tile=boundary_per_tile,
        num_empty_tiles=data.draw(hps.integers(0, 2), label="empty_tiles"),
        mixed_fillers=data.draw(hps.booleans(), label="mixed_fillers"),
        shuffle_tiles=data.draw(hps.booleans(), label="shuffle_tiles"),
        extra_stride=data.draw(hps.integers(0, 2), label="extra_stride"),
        random_init=data.draw(hps.booleans(), label="random_init"),
        seed=data.draw(hps.integers(0, 2**31 - 1), label="seed"),
    )
    _assert_matches_reference(inputs, _run_kernel(inputs, tile_n))

  @hp.given(hps.data())
  def test_kernel_page_geometry(self, data):
    """Sweeps the page sizes, and with them the pages a window spans."""
    mode = data.draw(hps.sampled_from(_MODES), label="mode")
    case, num_windows, max_boundary = _CASES[mode]
    geometry = data.draw(hps.sampled_from(_PAGE_GEOMETRIES[mode]), label="geo")
    tile_n = data.draw(hps.sampled_from(_tile_ns(case["overlap"])), label="tn")
    inputs = _build_inputs(
        **(_case_kwargs(case) | geometry),
        num_windows=num_windows,
        windows=_draw_windows(data, 1, num_windows, max_boundary),
        tile_n=tile_n,
        random_init=True,
        seed=data.draw(hps.integers(0, 2**31 - 1), label="seed"),
    )
    _assert_matches_reference(inputs, _run_kernel(inputs, tile_n))

  @parameterized.parameters(*_MODES)
  def test_kernel_without_slots_keeps_caches(self, mode):
    """Checks a batch without any compressed-KV slot leaves the caches as is."""
    case, num_windows, _ = _CASES[mode]
    inputs = _build_inputs(
        **_case_kwargs(case),
        num_windows=num_windows,
        windows=[[]],
        tile_n=4,
        num_empty_tiles=2,
        mixed_fillers=True,
        random_init=True,
    )
    cache, rope_cache = _run_kernel(inputs, tile_n=4)
    np.testing.assert_array_equal(cache, inputs.cache)
    if inputs.rope_cache is None:
      self.assertIsNone(rope_cache)
    else:
      np.testing.assert_array_equal(rope_cache, inputs.rope_cache)

  def test_kernel_rejects_bad_tile_n(self):
    inputs = test_base.make_case_inputs("csa_decode_batch_seq")
    with self.assertRaisesRegex(
        AssertionError, "tile_n must be a multiple of 4"
    ):
      pallas_mosaic_tpu_kernel.compress_norm_rope_store(
          *inputs.args, **inputs.kwargs, tile_n=6
      )


class KernelHelpersTest(parameterized.TestCase):
  """The kernel's host-side helpers; these run on any backend."""

  @hp.given(hps.data())
  def test_compute_is_first_mask(self, data):
    tile_n = data.draw(hps.integers(1, 8), label="tile_n")
    pack_factor = data.draw(hps.sampled_from([1, 2, 4]), label="pack_factor")
    kv_slots = data.draw(
        hps.lists(hps.integers(-1, 31), min_size=1, max_size=40), label="slots"
    )
    got = pallas_mosaic_tpu_kernel.compute_is_first_mask(
        jnp.asarray(kv_slots, jnp.int32), tile_n, pack_factor=pack_factor
    )
    # A token with a slot is first unless an earlier token of its tile has a
    # slot in the same row.
    want = [
        slot >= 0
        and not any(
            other >= 0 and other // pack_factor == slot // pack_factor
            for other in kv_slots[i - i % tile_n : i]
        )
        for i, slot in enumerate(kv_slots)
    ]
    np.testing.assert_array_equal(got, want)

  def test_derive_aliases(self):
    # HCA and the CSA indexer: cos_sin before the cache, no RoPE cache.
    self.assertEqual(
        pallas_mosaic_tpu_kernel.derive_aliases(
            has_rope=True, has_rope_cache=False, num_scalar_prefetch=5
        ),
        {7: 0},
    )
    # CSA: the RoPE cache follows the cache.
    self.assertEqual(
        pallas_mosaic_tpu_kernel.derive_aliases(
            has_rope=True, has_rope_cache=True, num_scalar_prefetch=5
        ),
        {7: 0, 8: 1},
    )
    # No RoPE: no cos_sin operand.
    self.assertEqual(
        pallas_mosaic_tpu_kernel.derive_aliases(
            has_rope=False, has_rope_cache=False, num_scalar_prefetch=7
        ),
        {8: 0},
    )


if __name__ == "__main__":
  absltest.main()
