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
"""Unit tests for VMEM compact/large layout conversions in layout_util."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_util

jax.config.parse_flags_with_absl()

_LANES = 128


def _random_bits(shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
  """Returns an array of `dtype` with random bit patterns."""
  uint_dtype = layout_util.get_uint_dtype(dtype.itemsize)
  bits = np.random.default_rng(0).integers(
      0, np.iinfo(uint_dtype).max, size=shape, dtype=uint_dtype, endpoint=True
  )
  return jax.lax.bitcast_convert_type(jnp.asarray(bits), dtype)


def _as_bits(x: jax.Array) -> np.ndarray:
  uint_dtype = layout_util.get_uint_dtype(x.dtype.itemsize)
  return np.asarray(jax.lax.bitcast_convert_type(x, uint_dtype))


def _store_and_load(x: jax.Array, stride: int) -> tuple[jax.Array, jax.Array]:
  """Stores `x` in compact layout with `stride` chunks, then loads it back."""
  rows, cols = x.shape
  packing = layout_util.get_packing(x.dtype)

  def _kernel(
      x_ref: hijax.TransformedRef,
      compact_ref: hijax.TransformedRef,
      large_ref: hijax.TransformedRef,
  ) -> None:
    layout_util.store_large_to_compact(compact_ref, x_ref[...])
    large_ref[...] = layout_util.load_compact_to_large(compact_ref, cols)

  return pl.kernel(
      _kernel,
      mesh=pltpu.TensorCoreMesh(axis_name="core", num_cores=1),
      out_type=(
          pltpu.VMEM((rows, stride, packing, _LANES), x.dtype),
          pltpu.VMEM((rows, cols), x.dtype),
      ),
  )(pltpu.with_memory_space_constraint(x, pltpu.VMEM))


class CompactLayoutTest(parameterized.TestCase):

  def setUp(self) -> None:
    helper_util.skip_unless_supported(self)
    super().setUp()

  @parameterized.product(
      dtype=[jnp.float32, jnp.bfloat16, jnp.float8_e4m3fn],
      rows=[8, 16],
      extra_stride=[0, 1],
  )
  def test_store_matches_layout_and_load_round_trips(
      self, dtype: jax.typing.DTypeLike, rows: int, extra_stride: int
  ):
    dtype = jnp.dtype(dtype)
    packing = layout_util.get_packing(dtype)
    blocks = 2
    x = _random_bits((rows, blocks * packing * _LANES), dtype)

    compact, large = jax.jit(_store_and_load, static_argnums=1)(
        x, blocks + extra_stride
    )

    # Element (r, (i * packing + p) * lanes + l) lives at compact[r, i, p, l].
    expected = _as_bits(x).reshape(rows, blocks, packing, _LANES)
    np.testing.assert_array_equal(_as_bits(compact)[:, :blocks], expected)
    np.testing.assert_array_equal(_as_bits(large), _as_bits(x))

  @parameterized.product(
      dtype=[jnp.bfloat16, jnp.float8_e4m3fn],
      cols=[128, 384, 640],
      extra_stride=[0, 1],
  )
  def test_partial_chunk_store_matches_layout_and_load_round_trips(
      self, dtype: jax.typing.DTypeLike, cols: int, extra_stride: int
  ):
    dtype = jnp.dtype(dtype)
    rows = 8
    packing = layout_util.get_packing(dtype)
    _, num_chunks, _, _ = layout_util.to_compact_shape((rows, cols), dtype)
    x = _random_bits((rows, cols), dtype)

    compact, large = jax.jit(_store_and_load, static_argnums=1)(
        x, num_chunks + extra_stride
    )

    num_lane_blocks = cols // _LANES
    expected_blocks = _as_bits(x).reshape(rows, num_lane_blocks, _LANES)
    compact_blocks = _as_bits(compact)[:, :num_chunks].reshape(
        rows, num_chunks * packing, _LANES
    )
    np.testing.assert_array_equal(
        compact_blocks[:, :num_lane_blocks], expected_blocks
    )
    np.testing.assert_array_equal(compact_blocks[:, num_lane_blocks:], 0)
    np.testing.assert_array_equal(_as_bits(large), _as_bits(x))


if __name__ == "__main__":
  absltest.main()
