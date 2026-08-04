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
"""Tests for Pallas/Mosaic Ragged Scatter operator on TPU."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.ragged_scatter import base
from tokamax._src.ops.ragged_scatter import pallas_mosaic_tpu

jax.config.parse_flags_with_absl()


class PallasTpuRaggedScatterTest(parameterized.TestCase):

  @parameterized.product(
      out_size=[400, 1024],
      hidden_size=[128, 512],
      start_end=[(3, 338), (10, 422)],
      dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_sc_scatter(self, out_size, hidden_size, start_end, dtype):
    if backend.get_default_device().platform != "tpu":
      self.skipTest("Only tested on TPU.")

    start, end = start_end
    start = min(start, out_size)
    end = min(end, out_size)

    key = jax.random.key(0)
    x = jax.random.normal(key, (out_size, hidden_size), jnp.float32).astype(
        dtype
    )
    indices = jax.random.permutation(key, out_size)
    start_arr = jnp.array([start], jnp.int32)
    end_arr = jnp.array([end], jnp.int32)

    op = pallas_mosaic_tpu.PallasTpuRaggedScatter()
    actual = op(x, indices, start_arr, end_arr)

    base_op = base.RaggedScatter()
    desired = base_op(x, indices, start_arr, end_arr)

    # Since Pallas ragged_scatter is undefined outside [start, end), mask both before asserting.
    mask = (indices >= start) & (indices < end)
    actual = jnp.where(mask[:, None], actual, 0)
    desired = jnp.where(mask[:, None], desired, 0)

    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        desired.astype(jnp.float32),
        rtol=1e-2,
        atol=1e-2,
    )


if __name__ == "__main__":
  absltest.main()
