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
"""Tests for Pallas/Mosaic Ragged Gather V2 operator on TPU."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.ragged_gather import base
from tokamax._src.ops.ragged_gather import pallas_mosaic_v2_tpu

jax.config.parse_flags_with_absl()


class PallasTpuRaggedGatherV2Test(parameterized.TestCase):

  @parameterized.product(
      in_out_size=[(512, 400), (512, 1024)],
      start_end=[(3, 338), (10, 422)],
      hidden_size=[128, 512, 8192],
      dtype=[jnp.int4, jnp.int8, jnp.bfloat16, jnp.float32],
  )
  def test_sc_gather(self, in_out_size, hidden_size, start_end, dtype):
    if backend.get_default_device().device_kind != "TPU7x":
      self.skipTest("Only tested on TPU7x.")

    in_size, out_size = in_out_size
    start, end = start_end
    start = min(start, out_size)
    end = min(end, out_size)
    key = jax.random.key(0)
    x = jax.random.normal(key, (in_size, hidden_size), jnp.float32)
    x = x.astype(dtype)
    indices = jax.random.randint(key, (out_size,), 0, in_size, jnp.int32)

    start_arr = jnp.array([start], jnp.int32)
    end_arr = jnp.array([end], jnp.int32)

    op = pallas_mosaic_v2_tpu.PallasV2TpuRaggedGather()
    actual = op(x, indices, start_arr, end_arr)

    base_op = base.RaggedGather()
    desired = base_op(x, indices, start_arr, end_arr)

    np.testing.assert_allclose(
        actual[start:end], desired[start:end], rtol=1e-2, atol=1e-2
    )


if __name__ == "__main__":
  absltest.main()
