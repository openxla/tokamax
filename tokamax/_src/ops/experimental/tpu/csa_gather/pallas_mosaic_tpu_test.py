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
"""Tests for the Pallas/Mosaic CSA Gather operator and kernel on TPU."""

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
import hypothesis.strategies as hps
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.csa_gather import pallas_mosaic_tpu
from tokamax._src.ops.experimental.tpu.csa_gather import pallas_mosaic_tpu_kernel
from tokamax._src.ops.experimental.tpu.csa_gather import reference
from tokamax._src.ops.experimental.tpu.csa_gather import test_base

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


class PallasTpuCsaGatherTest(test_base.CsaGatherTestBase):

  def __init__(self, *args):
    super().__init__(*args, gather_fn=pallas_mosaic_tpu.PallasTpuCsaGather())

  def setUp(self):
    super().setUp()
    if backend.get_default_device().device_kind != "TPU7x":
      self.skipTest("Only tested on TPU7x.")

  @parameterized.parameters(128, 256, 1024, 2048)
  def test_autotuning_configs(self, top_k):
    """Checks every autotuning config is bit-exact against the reference."""
    num_indices = 8 * top_k
    nope_cache, rope_cache, indices = test_base.make_inputs(num_indices)
    op = pallas_mosaic_tpu.PallasTpuCsaGather()
    ba = op.bind(nope_cache, rope_cache, indices, top_k=top_k)
    configs = ba.autotuning_configs
    self.assertIn(pallas_mosaic_tpu.Config(), configs)
    self.assertGreater(len(configs), 1)

    nope_ref, rope_ref = reference.csa_gather(
        nope_cache, rope_cache, indices, top_k=top_k
    )
    for config in configs:
      with self.subTest(str(config)):
        nope_out, rope_out = op.replace(config=config)(
            nope_cache, rope_cache, indices, top_k=top_k
        )
        np.testing.assert_array_equal(nope_out, nope_ref)
        np.testing.assert_array_equal(rope_out, rope_ref)

  @hp.given(hps.data())
  def test_kernel_matches_reference(self, data):
    """Checks the raw kernel is bit-exact against the reference."""
    num_lanes = pltpu.get_tpu_info().sparse_core.num_lanes
    top_k = data.draw(hps.sampled_from([128, 256, 512, 1024, 2048]))
    num_streams = data.draw(
        hps.sampled_from(
            [n for n in (1, 2, 4) if (top_k // 2) % (n * num_lanes) == 0]
        )
    )
    num_periods = data.draw(hps.integers(1, 40))
    num_indices = num_periods * top_k
    num_valid = data.draw(hps.integers(1, num_periods)) * top_k
    pass_num_valid = data.draw(hps.booleans())
    num_pages = data.draw(hps.integers(1, 64))
    page_size = data.draw(hps.sampled_from([4, 64, 256]))
    seed = data.draw(hps.integers(0, 2**31 - 1))

    nope_key, rope_key, idx_key = jax.random.split(jax.random.key(seed), 3)
    nope_cache = test_base.random_words(nope_key, (num_pages, page_size, 128))
    rope_cache = test_base.random_words(
        rope_key, (num_pages, page_size // 4, 128)
    )
    indices = jax.random.randint(
        idx_key, (num_indices,), 0, num_pages * page_size, jnp.int32
    )

    nope_out, rope_out = pallas_mosaic_tpu_kernel.csa_gather(
        nope_cache,
        rope_cache,
        indices,
        jnp.array([num_valid], jnp.int32) if pass_num_valid else None,
        top_k=top_k,
        num_streams=num_streams,
    )
    nope_ref, rope_ref = reference.csa_gather(
        nope_cache, rope_cache, indices, top_k=top_k
    )

    self.assertEqual(nope_out.shape, (num_indices, 128))
    self.assertEqual(rope_out.shape, (num_indices // 4, 128))
    n = num_valid if pass_num_valid else num_indices
    np.testing.assert_array_equal(nope_out[:n], nope_ref[:n])
    np.testing.assert_array_equal(rope_out[: n // 4], rope_ref[: n // 4])

  def test_kernel_rejects_invalid_num_streams(self):
    cache = jnp.zeros((2, 64, 128), jnp.int32)
    rope = jnp.zeros((2, 16, 128), jnp.int32)
    with self.assertRaisesRegex(ValueError, "num_streams"):
      pallas_mosaic_tpu_kernel.csa_gather(
          cache, rope, jnp.zeros((128,), jnp.int32), top_k=1024, num_streams=8
      )


if __name__ == "__main__":
  absltest.main()
