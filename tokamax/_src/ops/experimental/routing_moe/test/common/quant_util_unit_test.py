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
"""Unit tests for the shared dynamic quantization helpers, run on CPU."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe.common import quant_util
from tokamax._src.ops.experimental.routing_moe.common import spec

_FP8 = jnp.dtype(jnp.float8_e4m3fn)


def _expected_scale(
    x: np.ndarray, dtype: jax.typing.DTypeLike, quant: spec.QuantSpec
) -> np.ndarray:
  """Returns the scales of one quantization block computed with NumPy."""
  stride = 128 if isinstance(quant, spec.LaneStridedQuantSpec) else 1
  groups = np.abs(x).reshape(*x.shape[:-1], -1, stride)
  return groups.max(axis=-2) / quant_util.dtype_max(dtype)


class ComputeScaleTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="strided_2d",
          batch_shape=(8,),
          quant=spec.LaneStridedQuantSpec(block_col_size=512),
          cols=512,
      ),
      dict(
          testcase_name="strided_3d",
          batch_shape=(2, 8),
          quant=spec.LaneStridedQuantSpec(block_col_size=512),
          cols=512,
      ),
      dict(
          testcase_name="subchannel",
          batch_shape=(8,),
          quant=spec.SubchannelQuantSpec(block_size=512),
          cols=512,
      ),
      dict(
          testcase_name="subchannel_short",
          batch_shape=(8,),
          quant=spec.SubchannelQuantSpec(block_size=512),
          cols=256,
      ),
  )
  def test_matches_numpy(
      self,
      batch_shape: tuple[int, ...],
      quant: spec.QuantSpec,
      cols: int,
  ):
    stride = 128 if isinstance(quant, spec.LaneStridedQuantSpec) else 1
    x = 1000.0 * jax.random.normal(
        jax.random.key(0), (*batch_shape, cols), jnp.float32
    )

    scale, inv_scale = quant_util.compute_scale(x, _FP8, quant)

    expected = _expected_scale(np.asarray(x), _FP8, quant)
    self.assertEqual(scale.shape, (*batch_shape, stride))
    np.testing.assert_allclose(scale, expected, rtol=1e-6)
    np.testing.assert_allclose(inv_scale, 1 / expected, rtol=1e-6)

  @parameterized.parameters(
      spec.SubchannelQuantSpec(block_size=4),
      spec.LaneStridedQuantSpec(block_col_size=4 * 128),
  )
  def test_zero_group_has_zero_scale(self, quant: spec.QuantSpec):
    stride = 128 if isinstance(quant, spec.LaneStridedQuantSpec) else 1
    x = jnp.ones((8, 4 * stride), jnp.float32).at[0].set(0.0)

    scale, inv_scale = quant_util.compute_scale(x, _FP8, quant)

    np.testing.assert_array_equal(scale[0], 0.0)
    np.testing.assert_array_equal(inv_scale[0], 0.0)
    q = quant_util.quantize(x, jnp.tile(inv_scale, (1, 4)), _FP8)
    np.testing.assert_array_equal(q[0].astype(jnp.float32), 0.0)

  @parameterized.parameters(jnp.float8_e4m3fn, jnp.int8)
  def test_quantize_round_trips(self, dtype: jax.typing.DTypeLike):
    target_dtype = jnp.dtype(dtype)
    quant = spec.SubchannelQuantSpec(block_size=64)
    x = jax.random.normal(jax.random.key(1), (8, 64), jnp.float32)

    scale, inv_scale = quant_util.compute_scale(x, target_dtype, quant)
    q = quant_util.quantize(x, inv_scale, target_dtype)

    self.assertEqual(q.dtype, target_dtype)
    # Integer casts truncate, so the largest value may land one step below.
    np.testing.assert_allclose(
        jnp.abs(q.astype(jnp.float32)).max(axis=-1),
        quant_util.dtype_max(dtype),
        rtol=0.01,
    )
    np.testing.assert_allclose(
        q.astype(jnp.float32) * scale, x, atol=0.1 * float(jnp.abs(x).max())
    )


class DtypeMaxTest(absltest.TestCase):

  def test_float_and_int(self):
    self.assertEqual(quant_util.dtype_max(jnp.float8_e4m3fn), 448.0)
    self.assertEqual(quant_util.dtype_max(jnp.int8), 127.0)


if __name__ == "__main__":
  absltest.main()
