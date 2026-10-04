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
"""Tests for the CSA Gather API."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src import hlo_utils
from tokamax._src.ops.experimental.tpu.csa_gather import api
from tokamax._src.ops.experimental.tpu.csa_gather import reference
from tokamax._src.ops.experimental.tpu.csa_gather import test_base

jax.config.parse_flags_with_absl()


class ApiTest(parameterized.TestCase):

  @parameterized.product(
      num_indices_top_k=[(4096, 512), (8192, 1024)],
      num_valid_indices=[None, 2048],
      impl=["xla", "mosaic", "mosaic_tpu"],
  )
  def test_basic_api(self, num_indices_top_k, num_valid_indices, impl):
    if "mosaic" in impl and backend.get_default_device().device_kind != "TPU7x":
      self.skipTest("Only tested on TPU7x.")

    num_indices, top_k = num_indices_top_k
    num_pages, page_size = 64, 256
    nope_key, rope_key, idx_key = jax.random.split(jax.random.key(0), 3)
    nope_cache = test_base.random_words(nope_key, (num_pages, page_size, 128))
    rope_cache = test_base.random_words(
        rope_key, (num_pages, page_size // 4, 128)
    )
    indices = jax.random.randint(
        idx_key, (num_indices,), 0, num_pages * page_size, jnp.int32
    )

    @jax.jit
    def f(nope_cache, rope_cache, indices):
      return api.csa_gather(
          nope_cache,
          rope_cache,
          indices,
          num_valid_indices,
          top_k=top_k,
          implementation=impl,
      )

    nope_out, rope_out = f(nope_cache, rope_cache, indices)
    nope_ref, rope_ref = reference.csa_gather(
        nope_cache, rope_cache, indices, top_k=top_k
    )
    num_valid = num_indices if num_valid_indices is None else num_valid_indices

    with self.subTest("value"):
      np.testing.assert_array_equal(nope_out[:num_valid], nope_ref[:num_valid])
      np.testing.assert_array_equal(
          rope_out[: num_valid // 4], rope_ref[: num_valid // 4]
      )

    with self.subTest("correct_implementation_used"):
      # Check the lowered HLO for the kernel that was really used.
      opspecs = hlo_utils.get_opspecs(
          f.lower(nope_cache, rope_cache, indices), include_xla_kernels=False
      )
      if impl == "xla":
        self.assertEmpty(opspecs)
      else:
        self.assertNotEmpty(opspecs)
        self.assertIsInstance(
            opspecs[0].op, type(api.IMPLEMENTATIONS["mosaic_tpu"])
        )


if __name__ == "__main__":
  absltest.main()
