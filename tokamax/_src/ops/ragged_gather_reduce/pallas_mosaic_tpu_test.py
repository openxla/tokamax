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
"""Tests for the Pallas/Mosaic Ragged Gather Reduce operator and kernel."""

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
import hypothesis.strategies as hps
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.ragged_gather_reduce import pallas_mosaic_tpu
from tokamax._src.ops.ragged_gather_reduce import pallas_mosaic_tpu_kernel
from tokamax._src.ops.ragged_gather_reduce import reference
from tokamax._src.ops.ragged_gather_reduce import test_base

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


def _assert_close(out: jax.Array, expected: jax.Array):
  np.testing.assert_allclose(
      np.asarray(out, np.float32),
      np.asarray(expected, np.float32),
      atol=test_base.ATOL,
      rtol=test_base.RTOL,
  )


class PallasTpuRaggedGatherReduceTest(test_base.RaggedGatherReduceTestBase):

  def __init__(self, *args):
    super().__init__(
        *args, gather_fn=pallas_mosaic_tpu.PallasTpuRaggedGatherReduce()
    )

  def setUp(self):
    super().setUp()
    if _tpu_older_than_7x():
      self.skipTest("Only tested on TPU7x and newer.")

  # 7168 is left out: only one column split fits the autotuning column cap.
  @parameterized.parameters(1024, 2048, 4096)
  def test_autotuning_configs(self, hidden_size):
    """Checks every autotuning config matches the reference."""
    reduce_group_size = 8
    # 64 MiB of `x`, so the op runs the kernel rather than its XLA fallback.
    x, indices, topk_weights, valid_rows_mask = test_base.make_inputs(
        (1 << 22) // hidden_size, reduce_group_size, hidden_size
    )
    op = pallas_mosaic_tpu.PallasTpuRaggedGatherReduce()
    ba = op.bind(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
    )
    configs = ba.autotuning_configs
    self.assertIn(ba.heuristics_config, configs)
    self.assertGreater(len(configs), 1)

    expected = reference.ragged_gather_reduce(
        x, indices, topk_weights, valid_rows_mask, reduce_group_size
    )
    for config in configs:
      with self.subTest(str(config)):
        out = op.replace(config=config)(
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            reduce_group_size=reduce_group_size,
        )
        _assert_close(out, expected)

  @hp.given(hps.data())
  def test_kernel_matches_reference(self, data):
    """Checks the raw kernel, which has no small-input fallback."""
    tpu_info = pltpu.get_tpu_info()
    num_tokens = data.draw(hps.integers(1, 300))
    reduce_group_size = data.draw(hps.sampled_from([1, 2, 4, 8]))
    hidden_size = data.draw(hps.sampled_from([128, 256, 1024, 2048, 7168]))
    valid_modes = ["all", "none", "random"]
    if reduce_group_size >= 2:
      valid_modes.append("two_per_token")
    if reduce_group_size >= 3:
      valid_modes.append("uneven_pairs")
    valid_mode = data.draw(hps.sampled_from(valid_modes))
    num_rows = data.draw(hps.integers(1, 2 * num_tokens * reduce_group_size))
    num_column_partitions = data.draw(
        hps.sampled_from([
            n
            for n in (None, 1, 2, 4, 8, 16, 32)
            if n is None
            or pallas_mosaic_tpu_kernel.is_valid_num_column_partitions(
                n, hidden_size, tpu_info
            )
        ])
    )
    seed = data.draw(hps.integers(0, 2**31 - 1))
    self._check_kernel_matches_reference(
        num_tokens,
        reduce_group_size,
        hidden_size,
        valid_mode,
        num_rows,
        num_column_partitions,
        seed,
    )

  # Examples from the hypothesis test above. Column slices wider than fit the
  # SparseCore subcore VMEM (e.g. all of a 2048 or 7168 wide `x`) used to fail
  # to compile with a SparseCore allocation failure.
  @parameterized.parameters(
      (1, 1, 2048, "all", 1, 1, 0),
      (298, 1, 2048, "none", 139, 1, 2842),
      (178, 8, 2048, "all", 1, 1, 172592),
      (280, 4, 2048, "none", 2, 1, 48),
      (57, 8, 2048, "random", 1, 1, 10738),
      (1, 1, 7168, "all", 1, 1, 0),
      (185, 2, 7168, "none", 234, 2, 226),
      (100, 8, 7168, "two_per_token", 700, 4, 7),
      (1, 1, 128, "all", 1, None, 0),
      (225, 1, 128, "all", 1, None, 0),
      (225, 1, 2048, "random", 142, 8, 10204415),
      (281, 2, 1024, "none", 52, 2, 275434),
      (61, 1, 2048, "random", 89, 16, 1565),
      (244, 8, 128, "all", 106, 1, 4739),
      # 57 * 128 wide: the default single column slice is reduced in 19
      # chunks.
      (70, 2, 7296, "random", 100, None, 3),
      # Fewer than 16 rows of `x` with 128-column slices failed to compile
      # until `x` was padded to a whole tile.
      (1, 1, 256, "all", 1, 2, 0),
      (1, 1, 256, "random", 8, 2, 5),
      (64, 1, 1024, "all", 1, 8, 0),
  )
  def test_kernel_matches_reference_examples(self, *args):
    self._check_kernel_matches_reference(*args)

  @parameterized.product(
      col_size=(128, 896, 1024, 2048, 7168, 7296), reduce_group_size=(1, 8, 64)
  )
  def test_col_chunk_fits_vmem(self, col_size, reduce_group_size):
    tpu_info = pltpu.get_tpu_info()
    chunk_size = pallas_mosaic_tpu_kernel.select_col_chunk_size(
        col_size, reduce_group_size, tpu_info
    )
    self.assertIsNotNone(chunk_size)
    self.assertEqual(col_size % chunk_size, 0)
    self.assertEqual(chunk_size % tpu_info.num_lanes, 0)
    self.assertLessEqual(
        pallas_mosaic_tpu_kernel.subcore_vmem_bytes(
            chunk_size, reduce_group_size, tpu_info
        ),
        tpu_info.sparse_core.vmem_capacity_bytes,
    )

  def test_kernel_rejects_too_large_reduce_group_size(self):
    reduce_group_size = 1024
    x = jnp.zeros((64, 128), jnp.bfloat16)
    routes = jnp.zeros((reduce_group_size,), jnp.int32)
    with self.assertRaisesRegex(ValueError, "reduce_group_size"):
      pallas_mosaic_tpu_kernel.ragged_gather_reduce(
          x, routes, routes.astype(x.dtype), routes > 0, reduce_group_size
      )
    with self.assertRaises(NotImplementedError):
      pallas_mosaic_tpu.PallasTpuRaggedGatherReduce()(
          x,
          routes,
          routes.astype(x.dtype),
          routes > 0,
          reduce_group_size=reduce_group_size,
      )

  def _check_kernel_matches_reference(
      self,
      num_tokens,
      reduce_group_size,
      hidden_size,
      valid_mode,
      num_rows,
      num_column_partitions,
      seed,
  ):
    x, indices, topk_weights, valid_rows_mask = test_base.make_inputs(
        num_tokens, reduce_group_size, hidden_size, valid_mode, num_rows, seed
    )

    out = pallas_mosaic_tpu_kernel.ragged_gather_reduce(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size,
        num_column_partitions=num_column_partitions,
    )
    expected = reference.ragged_gather_reduce(
        x, indices, topk_weights, valid_rows_mask, reduce_group_size
    )

    self.assertEqual(out.shape, (num_tokens, hidden_size))
    _assert_close(out, expected)

  def test_kernel_rejects_float32(self):
    x = jnp.zeros((64, 128), jnp.float32)
    routes = jnp.zeros((64,), jnp.int32)
    with self.assertRaisesRegex(ValueError, "bfloat16"):
      pallas_mosaic_tpu_kernel.ragged_gather_reduce(
          x, routes, routes.astype(x.dtype), routes > 0, 8
      )

  def test_kernel_rejects_invalid_num_column_partitions(self):
    x = jnp.zeros((64, 128), jnp.bfloat16)
    routes = jnp.zeros((64,), jnp.int32)
    with self.assertRaisesRegex(ValueError, "num_column_partitions"):
      pallas_mosaic_tpu_kernel.ragged_gather_reduce(
          x,
          routes,
          routes.astype(x.dtype),
          routes > 0,
          8,
          num_column_partitions=2,
      )

  def test_op_rejects_float32(self):
    x = jnp.zeros((64, 128), jnp.float32)
    routes = jnp.zeros((64,), jnp.int32)
    with self.assertRaises(NotImplementedError):
      pallas_mosaic_tpu.PallasTpuRaggedGatherReduce()(
          x, routes, routes.astype(x.dtype), routes > 0, reduce_group_size=8
      )


if __name__ == "__main__":
  absltest.main()
