# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for the `block` library."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from tokamax._src.pallas import block

_SKIP_TPU_TEST_REASON = "Test broken on TPU due to blockspec shape TPU constraints."


class BlockTest(parameterized.TestCase):

  @parameterized.parameters(
      ((16, 16, 16), (16, 32, 48)),
      ((16, 16, None), (16, 32)),
      ((None, 16, None), (32,)),
      ((16, 16, 16), (32, 48), (3,)),
      ((16, 16, None), (16,), (slice(None), 3)),
  )
  def test_block_ref_bounds(self, block_shape, expected_bounds, at=None):
    if jax.default_backend() == "tpu":
      self.skipTest(_SKIP_TPU_TEST_REASON)

    def kernel(x_ref, _):
      if at is not None:
        x_ref = x_ref.at[at]
      self.assertEqual(x_ref.bounds, expected_bounds)

    x = jnp.ones((16, 32, 48))
    x_spec = pl.BlockSpec(block_shape, lambda *_: (0, 0, 0))
    _ = block.pallas_call(kernel, out_shape=x, in_specs=(x_spec,))(x)

  @parameterized.parameters(
      ((16, 16, 16), (False, False, False)),
      ((16, 16, None), (False, False)),
      ((None, 64, None), (True,)),
      ((16, 16, 16), (False, False), (3,)),
      ((32, 16, None), (True,), (slice(None), 3)),
  )
  def test_block_ref_bounds_checked(self, block_shape, expect_checked, at=None):
    if jax.default_backend() == "tpu":
      self.skipTest(_SKIP_TPU_TEST_REASON)
    def kernel(x_ref, _):
      if at is not None:
        x_ref = x_ref.at[at]
      self.assertEqual(x_ref.bounds_checked, expect_checked)

    x = jnp.ones((16, 32, 48))
    x_spec = pl.BlockSpec(block_shape, lambda *_: (0, 0, 0))
    _ = block.pallas_call(kernel, out_shape=x, in_specs=(x_spec,))(x)

  @parameterized.parameters(
      ((16, 16, 16),),
      ((16, 32, None),),
      ((None, 64, None),),
      ((None, 64, None), True),
  )
  def test_block_ref_inbounds_mask(self, block_shape, use_grid_spec=False):
    if jax.default_backend() == "tpu":
      self.skipTest(_SKIP_TPU_TEST_REASON)

    x = jnp.ones((99, 144, 222))
    expected_mask_shape = tuple(
        1 if dim % block_dim == 0 else block_dim
        for block_dim, dim in zip(block_shape, x.shape)
        if block_dim is not None
    )

    def kernel(x_ref, mask_ref):
      mask = x_ref.inbounds_mask()
      self.assertEqual(mask.shape, expected_mask_shape)
      pl.store(mask_ref, (), jnp.broadcast_to(mask, mask_ref.shape))

    grid = []
    out_shape = []
    for dim, block_dim in zip(x.shape, block_shape):
      if block_dim is None:
        block_dim = 1
      grid_dim = pl.cdiv(dim, block_dim)
      grid.append(grid_dim)
      out_shape.append(grid_dim * block_dim)

    grid = tuple(grid)
    idx = jnp.indices(out_shape, sparse=True)
    out_shape = jax.ShapeDtypeStruct(out_shape, dtype=bool)
    spec = pl.BlockSpec(block_shape, lambda i, j, k: (i, j, k))
    if use_grid_spec:
      grid_spec = pl.GridSpec(grid=grid, in_specs=(spec,), out_specs=spec)
      call = block.pallas_call(kernel, out_shape=out_shape, grid_spec=grid_spec)
    else:
      call = block.pallas_call(
          kernel,
          out_shape=out_shape,
          grid=grid,
          in_specs=(spec,),
          out_specs=spec,
      )
    expected = (
        (idx[0] < x.shape[0]) & (idx[1] < x.shape[1]) & (idx[2] < x.shape[2])
    )
    chex.assert_trees_all_equal(call(x), expected)

  def test_block_ref_at_masks(self):
    if jax.default_backend() == "tpu":
      self.skipTest(_SKIP_TPU_TEST_REASON)

    def kernel(x_ref, mask0_ref, mask1_ref, mask_ref):
      x_ref = x_ref.at[-32:, 1, pl.dslice(100, 16)]
      mask0, mask1 = x_ref.inbounds_masks
      pl.store(mask0_ref, (), mask0)
      pl.store(mask1_ref, (), mask1)
      mask0 = x_ref.inbounds_mask(bounds_check=(True, False))
      mask1 = x_ref.inbounds_mask(bounds_check=(False, True))
      self.assertEqual(mask0.shape, (32, 1))
      self.assertEqual(mask1.shape, (1, 16))
      pl.store(mask_ref, (), x_ref.inbounds_mask())

    x = jnp.ones((99, 3, 111))
    mask0, mask1, mask = block.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((32, 1), dtype=bool),
            jax.ShapeDtypeStruct((1, 16), dtype=bool),
            jax.ShapeDtypeStruct((32, 16), dtype=bool),
        ),
        in_specs=(pl.BlockSpec((128, 3, 128), lambda: (0, 0, 0)),),
        out_specs=pl.no_block_spec,
    )(x)
    idx = jnp.indices((32, 16), sparse=True)
    expected0 = (idx[0] + 96) < x.shape[0]
    expected1 = (idx[1] + 100) < x.shape[2]
    chex.assert_trees_all_equal(mask0, expected0)
    chex.assert_trees_all_equal(mask1, expected1)
    chex.assert_trees_all_equal(mask, expected0 & expected1)


if __name__ == "__main__":
  absltest.main()
