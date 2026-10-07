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
"""Tests for the baseline JAX implementation of the compressor projection."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.proj_and_save_state import base
from tokamax._src.ops.experimental.tpu.proj_and_save_state import test_base

jax.config.parse_flags_with_absl()


class BaseProjAndSaveStateTest(test_base.ProjAndSaveStateTestBase):

  def __init__(self, *args):
    super().__init__(*args, proj_fn=base.ProjAndSaveState())

  @parameterized.parameters(*test_base.GEOMETRIES)
  def test_matches_layout_spec(self, geometry):
    """Checks the state layout against a NumPy model of the cache."""
    geometry = test_base.GEOMETRIES[geometry]
    num_tokens, hidden_size, num_pages = 24, 256, 3
    hidden_states, wkv_wgate, ape, positions, slot_mapping, cache = (
        test_base.make_inputs(
            geometry,
            num_tokens,
            hidden_size=hidden_size,
            num_pages=num_pages,
            num_padding=3,
        )
    )
    # Small integers keep every product and sum exact in f32.
    hidden_states, wkv_wgate, ape = (
        jnp.round(4 * x) for x in (hidden_states, wkv_wgate * 16, ape)
    )

    out = base.ProjAndSaveState()(
        hidden_states,
        wkv_wgate,
        ape,
        positions,
        slot_mapping,
        cache,
        compress_ratio=geometry.compress_ratio,
    )

    hs, w, ape = (
        np.asarray(x, np.float64) for x in (hidden_states, wkv_wgate, ape)
    )
    state = hs @ w
    state[:, geometry.state_width :] += ape[
        np.asarray(positions) % geometry.compress_ratio
    ]
    # Row `r` of a token's state holds f32 values `[r * lanes, (r + 1) * lanes)`
    # as int32 words, or as the 4 bytes of each word down a uint8 slab.
    words = state.astype(np.float32).view(np.int32)
    words = words.reshape(num_tokens, geometry.rows_per_token, geometry.lanes)
    expected = np.asarray(cache).copy()
    if not geometry.words:
      words = words.view(np.uint8).reshape(*words.shape, 4).swapaxes(-1, -2)
    expected = expected.reshape(-1, *expected.shape[2:])
    for t, slot in enumerate(np.asarray(slot_mapping)):
      if slot >= 0:
        expected[slot : slot + geometry.rows_per_token] = words[t]
    np.testing.assert_array_equal(
        np.asarray(out), expected.reshape(cache.shape)
    )

  def test_rejects_bad_ape_shape(self):
    inputs = list(test_base.make_inputs(test_base.HCA, 8, hidden_size=256))
    inputs[2] = inputs[2][:, :-128]
    with self.assertRaisesRegex(ValueError, "ape must have shape"):
      base.ProjAndSaveState()(*inputs, compress_ratio=128)

  def test_rejects_bad_cache_dtype(self):
    inputs = list(test_base.make_inputs(test_base.CSA, 8, hidden_size=256))
    inputs[5] = inputs[5].astype(jnp.int16)
    with self.assertRaisesRegex(ValueError, "uint8 or"):
      base.ProjAndSaveState()(*inputs, compress_ratio=4)

  def test_rejects_ragged_page(self):
    inputs = list(test_base.make_inputs(test_base.CSA, 8, hidden_size=256))
    inputs[5] = inputs[5][:, :-8]
    with self.assertRaisesRegex(ValueError, "page_size"):
      base.ProjAndSaveState()(*inputs, compress_ratio=4)


if __name__ == "__main__":
  absltest.main()
