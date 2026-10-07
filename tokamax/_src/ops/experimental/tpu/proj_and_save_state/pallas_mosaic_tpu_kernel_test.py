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
"""Hypothesis tests for the compressor projection Pallas kernel on TPU.

These call the raw kernel entry point `pallas_mosaic_tpu_kernel.
proj_and_save_state` directly, sweeping tile sizes, compressor geometries,
positions and slot mappings, and compare it with `reference`.
"""

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
import hypothesis.strategies as hps
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.proj_and_save_state import pallas_mosaic_tpu_kernel
from tokamax._src.ops.experimental.tpu.proj_and_save_state import reference
from tokamax._src.ops.experimental.tpu.proj_and_save_state import test_base

jax.config.parse_flags_with_absl()

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

Geometry = test_base.Geometry


def _tpu_older_than_v6e() -> bool:
  """Whether the default device is not a TPU v6e or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 6
  )


def _upstream_slot_mapping(
    geometry: Geometry, token_index: np.ndarray, state_block_size: int
) -> np.ndarray:
  """Returns upstream's page-strided state slots of `token_index`.

  This is `state_slot` of vLLM-torchtpu's `compress_store_test.py`, with an
  identity block table: a page holds `geometry.tokens_per_page` states but vLLM
  pages it at `state_block_size` of them, so the tail of every page may be left
  unused (as for the CSA indexer, paged at half its capacity).

  Args:
    geometry: The compressor geometry.
    token_index: The tokens' indices in the sequence.
    state_block_size: Token states per page that vLLM pages the cache at.
  """
  page = token_index // state_block_size
  row = (token_index % state_block_size) * geometry.rows_per_token
  return (page * geometry.page_size + row).astype(np.int32)


def _with_positions_and_slots(
    inputs: tuple[jax.Array, ...], positions, slot_mapping
) -> tuple[jax.Array, ...]:
  """Replaces the positions and slot mapping of `make_inputs` inputs."""
  hidden_states, wkv_wgate, ape, _, _, cache = inputs
  return (
      hidden_states,
      wkv_wgate,
      ape,
      jnp.asarray(positions, jnp.int32),
      jnp.asarray(slot_mapping, jnp.int32),
      cache,
  )


class PallasMosaicTpuKernelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if _tpu_older_than_v6e():
      self.skipTest("Only tested on TPU v6e and newer.")

  def _check(
      self,
      geometry: Geometry,
      inputs: tuple[jax.Array, ...],
      *,
      tile_k: int | None = None,
      tile_n: int | None = None,
      atol: float = test_base.ATOL,
      rtol: float = test_base.RTOL,
  ):
    """Checks the raw kernel against the reference on `inputs`."""
    compress_ratio = inputs[2].shape[0]
    actual = pallas_mosaic_tpu_kernel.proj_and_save_state(
        *inputs, compress_ratio, tile_k=tile_k, tile_n=tile_n
    )
    expected = reference.proj_and_save_state(
        *inputs, compress_ratio=compress_ratio
    )
    # On TPU v6e, XLA computes the reference projection of a single token at
    # full f32 precision. The kernel, and XLA for more tokens, use default
    # precision, so the two differ by up to about 1e-2.
    num_tokens = inputs[0].shape[0]
    if num_tokens == 1 and pltpu.get_tpu_info().generation == 6:
      atol = max(atol, 2e-2)
    test_base.assert_caches_match(
        actual,
        expected,
        inputs[4],
        geometry.rows_per_token,
        atol=atol,
        rtol=rtol,
    )

  @hp.given(hps.data())
  def test_tile_sizes(self, data):
    """Checks random tile sizes, token counts, padding and page counts."""
    geometry = test_base.GEOMETRIES[
        data.draw(hps.sampled_from(sorted(test_base.GEOMETRIES)), label="geom")
    ]
    hidden_size = data.draw(
        hps.sampled_from([256, 640, 1024, 4096]), label="hidden_size"
    )
    # The `(tile_k, state_width)` weight block must tile `hidden_size` exactly
    # and be 128-lane aligned.
    tile_k = data.draw(
        hps.sampled_from(
            [k for k in range(128, 2049, 128) if hidden_size % k == 0]
        ),
        label="tile_k",
    )
    num_tokens = data.draw(hps.integers(1, 600), label="num_tokens")
    # Mosaic requires a `(tile_n,)` positions block that does not cover all the
    # (padded) tokens to be 128-aligned; a single block may be any multiple of
    # 8.
    tile_n = data.draw(
        hps.sampled_from([
            n
            for n in (8, 16, 64, 128, 256, 512)
            if n % 128 == 0 or n >= num_tokens
        ]),
        label="tile_n",
    )
    num_padding = data.draw(hps.integers(0, num_tokens), label="num_padding")
    num_pages = -(-num_tokens // geometry.tokens_per_page)
    num_pages += data.draw(hps.integers(0, 2), label="spare_pages")
    seed = data.draw(hps.integers(0, 2**31 - 1), label="seed")

    inputs = test_base.make_inputs(
        geometry,
        num_tokens,
        hidden_size=hidden_size,
        num_pages=num_pages,
        num_padding=num_padding,
        seed=seed,
    )
    self._check(geometry, inputs, tile_k=tile_k, tile_n=tile_n)

  @hp.given(hps.data())
  def test_geometries(self, data):
    """Checks compressor geometries beyond DeepSeek-V4's three.

    Sweeps the head dimension and window overlap (which set `state_width`), the
    compress ratio (APE rows), cache lanes and page size, and the uint8 slab
    versus int32 word cache declaration.
    """
    head_dim = data.draw(hps.sampled_from([128, 512]), label="head_dim")
    overlap = data.draw(hps.booleans(), label="overlap")
    state_width = head_dim * (1 + int(overlap))
    compress_ratio = data.draw(
        hps.sampled_from([4, 128, 1, 8, 32]), label="compress_ratio"
    )
    # A cache row must be 128-lane aligned and a token's `kv` and `score` must
    # each fill whole rows.
    lanes = data.draw(
        hps.sampled_from([l for l in (128, 256) if state_width % l == 0]),
        label="lanes",
    )
    rows_per_token = 2 * state_width // lanes
    tokens_per_page = data.draw(
        hps.sampled_from([1, 8, 16, 32]), label="tokens_per_page"
    )
    # Only 128-lane caches are drawn as int32 word arrays, as upstream deploys
    # (CSA's NoPE array): for a 256-lane int32 cache, Mosaic fails to compile
    # the kernel's int32-to-uint8 slab view (`as_u8_slabs`), rejecting its
    # `memref_reshape` as not implemented.
    words = lanes == 128 and data.draw(hps.booleans(), label="words")
    geometry = Geometry(
        state_width=state_width,
        compress_ratio=compress_ratio,
        lanes=lanes,
        page_size=tokens_per_page * rows_per_token,
        words=words,
    )
    num_tokens = data.draw(hps.integers(1, 300), label="num_tokens")
    num_padding = data.draw(hps.integers(0, num_tokens), label="num_padding")
    seed = data.draw(hps.integers(0, 2**31 - 1), label="seed")

    inputs = test_base.make_inputs(
        geometry,
        num_tokens,
        hidden_size=512,
        num_padding=num_padding,
        seed=seed,
    )
    self._check(geometry, inputs)

  @hp.given(hps.data())
  def test_positions_and_slots(self, data):
    """Checks serving-like positions and slot mappings, in f32 and bf16."""
    geometry = test_base.GEOMETRIES[
        data.draw(hps.sampled_from(sorted(test_base.GEOMETRIES)), label="geom")
    ]
    ratio = geometry.compress_ratio
    num_tokens = data.draw(hps.integers(1, 400), label="num_tokens")
    seed = data.draw(hps.integers(0, 2**31 - 1), label="seed")
    rng = np.random.default_rng(seed)

    # Token positions: a prefill chunk, the compression-boundary tokens of
    # consecutive windows (which upstream compresses), or arbitrary decode
    # positions anywhere in the int32 range.
    position_kind = data.draw(
        hps.sampled_from(["prefill", "boundary", "random"]), label="positions"
    )
    start = data.draw(hps.integers(0, 1 << 16), label="start")
    if position_kind == "prefill":
      positions = start + np.arange(num_tokens)
    elif position_kind == "boundary":
      positions = (start + np.arange(num_tokens)) * ratio + ratio - 1
    else:
      positions = rng.integers(0, 2**31 - 1, num_tokens)

    # Slots: upstream's page-strided layout (in order, or for shuffled
    # tokens), possibly paged below a page's capacity, or random distinct
    # slots anywhere in the cache.
    slot_kind = data.draw(
        hps.sampled_from(["upstream", "upstream_shuffled", "random"]),
        label="slots",
    )
    if slot_kind == "random":
      num_pages = -(-num_tokens // geometry.tokens_per_page)
      num_pages += data.draw(hps.integers(0, 2), label="spare_pages")
      token_slots = rng.permutation(num_pages * geometry.tokens_per_page)
      slot_mapping = token_slots[:num_tokens] * geometry.rows_per_token
    else:
      state_block_size = data.draw(
          hps.sampled_from(
              sorted({geometry.tokens_per_page, geometry.tokens_per_page // 2})
          ),
          label="state_block_size",
      )
      num_pages = -(-num_tokens // state_block_size) + 1
      token_index = np.arange(num_tokens)
      if slot_kind == "upstream_shuffled":
        token_index = rng.permutation(token_index)
      slot_mapping = _upstream_slot_mapping(
          geometry, token_index, state_block_size
      )
    # Skipped tokens anywhere in the batch, not only trailing padding.
    skip_fraction = data.draw(hps.sampled_from([0.0, 0.25, 1.0]), label="skip")
    slot_mapping = np.where(
        rng.random(num_tokens) < skip_fraction, -1, slot_mapping
    )

    dtype = data.draw(
        hps.sampled_from([jnp.float32, jnp.bfloat16]), label="dtype"
    )
    inputs = test_base.make_inputs(
        geometry,
        num_tokens,
        hidden_size=1024,
        num_pages=num_pages,
        dtype=dtype,
        seed=seed,
    )
    inputs = _with_positions_and_slots(inputs, positions, slot_mapping)
    bf16 = dtype == jnp.bfloat16
    self._check(
        geometry,
        inputs,
        atol=test_base.BF16_ATOL if bf16 else test_base.ATOL,
        rtol=test_base.BF16_RTOL if bf16 else test_base.RTOL,
    )

  @parameterized.named_parameters(
      # The projection inputs of vLLM-torchtpu's `compress_store_test.py`
      # cases: the first `prefill_len` (or `num_tokens`) tokens of a sequence,
      # at positions `0, 1, ...`, written to upstream's page-strided slots,
      # with the heuristic tile sizes. Upstream's CSA cache there is uint8
      # slabs; `csa_words_*` also check the int32 array CSA deploys with.
      # (name, geometry, num_tokens, hidden_size)
      ("csa_prod_hidden_4096", "csa_u8", 8, 4096),
      ("csa_prefill", "csa_u8", 128, 7168),
      ("hca_prefill", "hca", 256, 7168),
      ("hca_prefill_small", "hca", 128, 7168),
      ("csa_decode_batch_large", "csa_u8", 1024, 7168),
      ("csa_decode_batch_seq", "csa_u8", 16, 7168),
      ("hca_decode_batch_seq", "hca", 512, 7168),
      ("csa_decode_batch_random", "csa_u8", 32, 7168),
      ("hca_decode_batch_mixed", "hca", 384, 7168),
      ("csa_indexer_prefill", "indexer", 128, 7168),
      ("csa_indexer_decode", "indexer", 16, 7168),
      ("csa_words_hidden_4096", "csa", 128, 4096),
      ("csa_words_hidden_7168", "csa", 128, 7168),
  )
  def test_upstream_cases(self, geometry, num_tokens, hidden_size):
    """Checks the DeepSeek-V4 projections upstream's tests exercise."""
    geometry = test_base.GEOMETRIES[geometry]
    # Upstream pages the indexer at CSA's capacity, half its own.
    if geometry is test_base.INDEXER:
      state_block_size = geometry.tokens_per_page // 2
    else:
      state_block_size = geometry.tokens_per_page
    inputs = test_base.make_inputs(
        geometry,
        num_tokens,
        hidden_size=hidden_size,
        num_pages=-(-num_tokens // state_block_size) + 1,
    )
    token_index = np.arange(num_tokens)
    slot_mapping = _upstream_slot_mapping(
        geometry, token_index, state_block_size
    )
    inputs = _with_positions_and_slots(inputs, token_index, slot_mapping)
    self._check(geometry, inputs)


if __name__ == "__main__":
  absltest.main()
