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
"""Tests for the Pallas/Mosaic MoE router top-k operator and kernel on TPU."""

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
import hypothesis.strategies as hps
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.router_topk import pallas_mosaic_tpu
from tokamax._src.ops.experimental.tpu.router_topk import pallas_mosaic_tpu_kernel
from tokamax._src.ops.experimental.tpu.router_topk import test_base

jax.config.parse_flags_with_absl()

hp.settings.register_profile(
    name="deterministic",
    database=None,
    derandomize=True,
    deadline=None,
    max_examples=25,
    print_blob=True,
    verbosity=hp.Verbosity.verbose,
)
hp.settings.load_profile(name="deterministic")


def _tpu_older_than_v6e() -> bool:
  """Whether the default device is not a TPU v6e or newer chip."""
  return (
      backend.get_default_device().platform != "tpu"
      or pltpu.get_tpu_info().generation < 6
  )


class PallasTpuRouterTopKTest(test_base.RouterTopKTestBase):

  def __init__(self, *args):
    super().__init__(*args, topk_fn=pallas_mosaic_tpu.PallasTpuRouterTopK())

  def setUp(self):
    super().setUp()
    if _tpu_older_than_v6e():
      self.skipTest("Only tested on TPU v6e and newer.")

  @parameterized.parameters(
      (8, 512, 10),
      (250, 512, 10),
      (1000, 512, 10),
      (4384, 512, 10),
      (2048, 256, 8),
      (2048, 2048, 8),
  )
  def test_autotuning_configs(self, rows, experts, k):
    """Checks every autotuning config against the reference."""
    scores = jnp.asarray(test_base.make_scores(rows, experts))
    op = pallas_mosaic_tpu.PallasTpuRouterTopK()
    ba = op.bind(scores, k)
    configs = ba.autotuning_configs
    self.assertIn(op._get_heuristics_config(ba), configs)  # pylint: disable=protected-access
    if rows > 64:
      self.assertGreater(len(configs), 1)

    rw, ri = test_base.reference_topk(scores, k)
    for config in configs:
      with self.subTest(str(config)):
        kw, ki = op.replace(config=config)(scores, k)
        np.testing.assert_array_equal(kw, rw)
        np.testing.assert_array_equal(ki, ri)

  @parameterized.parameters(
      (16384, 512, (64, 128, 256, 512, 1024)),
      (1000, 512, (64, 128, 256, 512, 1000)),
      (100, 512, (64, 100)),
      (40, 512, (40,)),
      (4096, 2048, (64, 128, 256, 512, 1024)),
      (4096, 4096, (64, 128, 256, 512)),
      (4096, 8192, (64, 128, 256, 512)),
  )
  def test_autotuning_candidates(self, rows, experts, expected):
    """Checks the candidates, which skip blocks above 8 MiB of scores."""
    op = pallas_mosaic_tpu.PallasTpuRouterTopK()
    ba = op.bind(jax.ShapeDtypeStruct((rows, experts), jnp.float32), 8)
    self.assertEqual(
        sorted(c.block_rows for c in ba.autotuning_configs), list(expected)
    )

  # Ported from upstream's `test_grid_rounds_up_over_the_tuned_block`.
  @parameterized.parameters(
      (4, 1),
      (250, 1),
      (512, 1),
      (1000, 2),
      (4384, 9),
      (8191, 16),
      (16384, 32),
  )
  def test_heuristic_grid_rounds_up_over_the_tuned_block(self, rows, grid):
    """Checks the heuristic reproduces upstream's block and `cdiv` grid."""
    op = pallas_mosaic_tpu.PallasTpuRouterTopK()
    ba = op.bind(
        jax.ShapeDtypeStruct((rows, test_base.EXPERTS), jnp.float32), 10
    )
    config = op._get_heuristics_config(ba)  # pylint: disable=protected-access
    block = min(pallas_mosaic_tpu_kernel.MAX_BLOCK_ROWS, rows)
    self.assertEqual(config, pallas_mosaic_tpu.Config(block_rows=block))
    self.assertEqual(pl.cdiv(rows, config.block_rows), grid)

  def test_rejects_misaligned_block_rows(self):
    op = pallas_mosaic_tpu.PallasTpuRouterTopK(
        config=pallas_mosaic_tpu.Config(block_rows=100)
    )
    with self.assertRaisesRegex(ValueError, "multiple of 8"):
      op(jnp.asarray(test_base.make_scores(256)), 10)

  @parameterized.parameters(
      dict(rows=600, experts=8192, block_rows=None),
      dict(rows=4096, experts=2048, block_rows=2048),
  )
  def test_oversized_block_not_implemented(self, rows, experts, block_rows):
    op = pallas_mosaic_tpu.PallasTpuRouterTopK()
    if block_rows is not None:
      op = op.replace(config=pallas_mosaic_tpu.Config(block_rows=block_rows))
    scores = jnp.asarray(test_base.make_scores(rows, experts))
    with self.assertRaisesRegex(NotImplementedError, "VMEM budget"):
      op(scores, 8)

  def test_unaligned_block_rows_at_least_num_tokens(self):
    # A block covering all the tokens need not be a multiple of 8.
    scores = jnp.asarray(test_base.make_scores(100))
    op = pallas_mosaic_tpu.PallasTpuRouterTopK(
        config=pallas_mosaic_tpu.Config(block_rows=100)
    )
    kw, ki = op(scores, 10)
    rw, ri = test_base.reference_topk(scores)
    np.testing.assert_array_equal(kw, rw)
    np.testing.assert_array_equal(ki, ri)


class PallasMosaicTpuKernelTest(parameterized.TestCase):
  """Tests of the raw kernel entry point, `pallas_mosaic_tpu_kernel.select`."""

  def setUp(self):
    super().setUp()
    if _tpu_older_than_v6e():
      self.skipTest("Only tested on TPU v6e and newer.")

  @parameterized.parameters(64, 1000)
  def test_interpret_matches_compiled(self, rows):
    """Upstream gates the kernel in interpret mode; check it agrees."""
    scores = jnp.asarray(test_base.make_scores(rows))
    iw, ii = pallas_mosaic_tpu_kernel.select(scores, 10, interpret=True)
    kw, ki = pallas_mosaic_tpu_kernel.select(scores, 10)
    rw, ri = test_base.reference_topk(scores)
    np.testing.assert_array_equal(iw, rw)
    np.testing.assert_array_equal(ii, ri)
    np.testing.assert_array_equal(kw, rw)
    np.testing.assert_array_equal(ki, ri)

  @hp.given(hps.data())
  def test_select(self, data):
    """Sweeps rows, experts, `topk`, `block_rows` and poisoned scores."""
    rows = data.draw(hps.integers(1, 3000), label="rows")
    experts = data.draw(
        hps.one_of(
            hps.sampled_from([64, 128, 256, 384, 512]), hps.integers(1, 1024)
        ),
        label="experts",
    )
    topk = data.draw(hps.integers(1, min(experts, 16)), label="topk")
    block_rows = data.draw(
        hps.sampled_from([None, 8, 64, 128, 256, 512, 1024]), label="block_rows"
    )
    seed = data.draw(hps.integers(0, 2**31 - 1), label="seed")
    rng = np.random.default_rng(seed)
    scores = rng.random((rows, experts), dtype=np.float32)
    # Poison a few entries with each spelling of the sentinel hazard.
    num_bad = data.draw(hps.integers(0, 8), label="num_bad")
    for _ in range(num_bad):
      bad = rng.choice(
          np.array([np.nan, -np.inf, np.finfo(np.float32).min], np.float32)
      )
      scores[rng.integers(rows), rng.integers(experts)] = bad
    if data.draw(hps.booleans(), label="poison_row"):
      scores[rng.integers(rows)] = np.nan

    kwargs = {} if block_rows is None else dict(block_rows=block_rows)
    kw, ki = pallas_mosaic_tpu_kernel.select(
        jnp.asarray(scores), topk, **kwargs
    )
    rw, ri = test_base.reference_topk(scores, topk)
    np.testing.assert_array_equal(kw, rw)
    np.testing.assert_array_equal(ki, ri)


if __name__ == "__main__":
  absltest.main()
