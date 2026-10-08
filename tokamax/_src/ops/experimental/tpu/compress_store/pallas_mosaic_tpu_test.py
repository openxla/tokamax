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
"""Tests for the Pallas/Mosaic compress-and-store operator on TPU.

The raw kernel's tests are in `pallas_mosaic_tpu_kernel_test.py`.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
from tokamax._src.ops.experimental.tpu.compress_store import pallas_mosaic_tpu
from tokamax._src.ops.experimental.tpu.compress_store import test_base

jax.config.parse_flags_with_absl()


def _tpu_older_than_7x() -> bool:
  """Whether the default device is not a TPU7x or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 7
  )


class PallasTpuCompressStoreTest(test_base.CompressStoreTestBase):

  def __init__(self, *args):
    super().__init__(
        *args, compress_store_fn=pallas_mosaic_tpu.PallasTpuCompressStore()
    )

  def setUp(self):
    super().setUp()
    if _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")

  @parameterized.parameters(
      # Token counts that aren't a multiple of every `tile_n`.
      "csa_decode_batch_mixed",
      "hca_decode_batch_mixed",
      "csa_decode_batch_random",
      "csa_prefill",
      "hca_prefill_small",
      "csa_indexer_prefill",
  )
  def test_autotuning_configs(self, case_name):
    """Checks every autotuning config is bit-exact against the reference."""
    inputs = test_base.make_case_inputs(case_name)
    op = pallas_mosaic_tpu.PallasTpuCompressStore()
    ba = op.bind(*inputs.args, **inputs.kwargs)
    configs = ba.autotuning_configs
    self.assertIn(pallas_mosaic_tpu.Config(), configs)
    num_tokens = inputs.positions.shape[0]
    for config in configs - {pallas_mosaic_tpu.Config()}:
      self.assertEqual(num_tokens % config.tile_n, 0)

    for config in configs:
      with self.subTest(str(config)):
        call_inputs = inputs.with_cache_copies()
        out = op.replace(config=config)(*call_inputs.args, **call_inputs.kwargs)
        test_base.assert_matches_reference(inputs, out)

  @parameterized.parameters(
      # 128 tokens.
      ("csa_prefill", {4, 8, 16, 32}),
      ("hca_prefill_small", {4, 8}),
      ("csa_indexer_prefill", {4, 8, 16, 32}),
      # 6 and 4 tokens: only upstream's `tile_n`.
      ("csa_decode_batch_mixed", {4}),
      ("hca_decode_batch_mixed", {4}),
      ("csa_decode_batch_random", {4}),
  )
  def test_autotuning_tile_ns_divide_num_tokens(self, case_name, tile_ns):
    inputs = test_base.make_case_inputs(case_name)
    op = pallas_mosaic_tpu.PallasTpuCompressStore()
    configs = op.bind(*inputs.args, **inputs.kwargs).autotuning_configs
    self.assertEqual({c.tile_n for c in configs}, tile_ns)

  @parameterized.parameters(
      "csa_decode_batch_seq", "hca_decode_batch_seq", "csa_indexer_decode"
  )
  def test_donates_caches(self, case_name):
    """Checks the op donates `cache` and `rope_cache`, as upstream does."""
    inputs = test_base.make_case_inputs(case_name).with_cache_copies()
    out = pallas_mosaic_tpu.PallasTpuCompressStore()(
        *inputs.args, **inputs.kwargs
    )
    jax.block_until_ready(out)
    self.assertTrue(inputs.cache.is_deleted())
    if inputs.rope_cache is not None:
      self.assertTrue(inputs.rope_cache.is_deleted())


if __name__ == "__main__":
  absltest.main()
