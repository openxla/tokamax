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
"""Shared correctness tests for Ragged Gather implementations."""

from absl.testing import parameterized
import jax
from jax.extend import backend
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.ragged_gather import base


class RaggedGatherTestBase(parameterized.TestCase):
  """Correctness suite shared by all Ragged Gather implementations.

  Subclasses pass the implementation under test as `gather_fn`. The results are
  compared against `base.RaggedGather` over the valid `[start, end)` range;
  values outside that range are undefined for the Pallas implementations.
  """

  def __init__(self, *args, gather_fn):
    super().__init__(*args)
    self._gather_fn = gather_fn

  def check_sc_gather(self, in_out_size, hidden_size, start_end, dtype):
    """Checks the gathered rows in `[start, end)` match `base.RaggedGather`."""
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

    actual = self._gather_fn(x, indices, start_arr, end_arr)

    base_op = base.RaggedGather()
    desired = base_op(x, indices, start_arr, end_arr)

    np.testing.assert_allclose(
        actual[start:end], desired[start:end], rtol=1e-2, atol=1e-2
    )

  @parameterized.product(
      in_out_size=[(512, 32), (512, 400), (512, 1024)],
      start_end=[(3, 28), (3, 338), (10, 422)],
      hidden_size=[128, 512, 8192],
      dtype=[jnp.int8, jnp.bfloat16, jnp.float32],
  )
  def test_sc_gather(self, in_out_size, hidden_size, start_end, dtype):
    """Checks the gathered rows in `[start, end)` match `base.RaggedGather`."""
    self.check_sc_gather(in_out_size, hidden_size, start_end, dtype)

