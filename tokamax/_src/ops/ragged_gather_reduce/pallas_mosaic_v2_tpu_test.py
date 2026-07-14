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
"""Tests for Pallas/Mosaic Ragged Gather Reduce operator v2 on TPU."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.ragged_gather_reduce import base
from tokamax._src.ops.ragged_gather_reduce import pallas_mosaic_v2_tpu

jax.config.parse_flags_with_absl()


class PallasV2TpuRaggedGatherReduceTest(parameterized.TestCase):

  @parameterized.product(
      input_size=[512, 1024],
      hidden_size=[128, 512],
      reduce_group_size=[4, 8],
      dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_sc_gather_reduce_v2(
      self, input_size, hidden_size, reduce_group_size, dtype
  ):
    if backend.get_default_device().platform != "tpu":
      self.skipTest("Only tested on TPU.")

    key = jax.random.key(0)
    x = jax.random.normal(key, (input_size, hidden_size), jnp.float32).astype(
        dtype
    )
    indices = jax.random.randint(key, (input_size,), 0, input_size, jnp.int32)
    topk_weights = jax.random.normal(key, (input_size,), jnp.float32).astype(
        dtype
    )
    valid_rows_mask = jnp.ones((input_size,), jnp.bool_)

    op = pallas_mosaic_v2_tpu.PallasV2TpuRaggedGatherReduce()
    actual = op(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
    )

    base_op = base.RaggedGatherReduce()
    desired = base_op(
        x,
        indices,
        topk_weights,
        valid_rows_mask,
        reduce_group_size=reduce_group_size,
    )

    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        desired.astype(jnp.float32),
        rtol=1e-2,
        atol=1e-2,
    )


if __name__ == "__main__":
  absltest.main()
