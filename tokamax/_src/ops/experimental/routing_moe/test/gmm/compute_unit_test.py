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
"""Unit tests for routing_gmm compute helpers that run on CPU."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe.gmm import compute
from tokamax._src.ops.experimental.routing_moe.gmm import config

_QUANT = config.DynamicQuantConfig(
    dtype=jnp.dtype(jnp.float8_e4m3fn), block_size=4, stride=128
)


def _expected_scale(acc: np.ndarray) -> np.ndarray:
  """Returns the per-lane scales of `acc` computed with NumPy."""
  groups = np.abs(acc).reshape(*acc.shape[:-1], _QUANT.block_size, -1)
  absmax = groups.max(axis=-2)
  dtype_max = float(jnp.finfo(_QUANT.dtype).max)
  return np.where(absmax == 0, 1.0, absmax / dtype_max)


class RowScaleUnitTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="2d", batch_shape=(8,)),
      dict(testcase_name="3d", batch_shape=(2, 8)),
  )
  def test_matches_numpy(self, batch_shape: tuple[int, ...]):
    shape = (*batch_shape, _QUANT.block_size * _QUANT.stride)
    acc = 1000.0 * jax.random.normal(jax.random.key(0), shape, jnp.float32)
    # Lane 3 is all zeros and must get a unit scale.
    acc = acc.at[..., 3 :: _QUANT.stride].set(0.0)

    scale, inv_scale = compute.row_scale(acc, _QUANT)

    expected = _expected_scale(np.asarray(acc))
    self.assertEqual(scale.shape, (*batch_shape, _QUANT.stride))
    np.testing.assert_allclose(scale, expected, rtol=1e-6)
    np.testing.assert_allclose(inv_scale, 1 / expected, rtol=1e-6)
    np.testing.assert_array_equal(scale[..., 3], 1.0)


if __name__ == "__main__":
  absltest.main()
