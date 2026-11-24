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

import math

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
import qwix
from tokamax._src import gpu_utils
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common
from tokamax._src.ops.ragged_dot import test_base


# A probably suboptimal global config that should work for most shapes. We use
# this just to check that there exists a config that yields correct results.
_CONFIG = pallas_mosaic_gpu.Config(
    block_m=128,
    block_n=64,
    block_k=256,
    num_stages=2,
    split_k=1,
)

# Per-batch partitions of 2048 rows into 8 groups. Every partition is distinct,
# is not uniform and has at least one empty group.
_VMAP_GROUP_SIZES = (
    (0, 640, 17, 301, 0, 512, 400, 178),
    (333, 0, 1024, 64, 5, 0, 250, 372),
    (1000, 48, 0, 8, 700, 120, 0, 172),
    (64, 800, 0, 33, 400, 0, 600, 151),
)


class PallasMosaicGpuRaggedDotTest(test_base.RaggedDotTestBase):

  def __init__(self, *args):
    op = pallas_mosaic_gpu.PallasMosaicGpuRaggedDot()

    def fn(lhs, rhs, *, config=None, **kwargs):
      expect_supported = (lhs.dtype == rhs.dtype == jnp.bfloat16) and (
          lhs.shape[-1] % (128 // jnp.dtype(lhs.dtype).itemsize) == 0
      )

      lhs_ = jax.eval_shape(quantization.as_array_or_qarray, lhs)
      rhs_ = jax.eval_shape(quantization.as_array_or_qarray, rhs)

      device_kind = jax.devices()[0].device_kind.lower()
      if gpu_utils.is_sm100():
        if isinstance(rhs_, qwix.QArray):
          config = pallas_mosaic_gpu.Config(
              block_m=64,
              block_n=128,
              block_k=128,
              num_stages=2,
              split_k=1,
          )
          if (
              rhs_.qtype != jnp.int4
              or rhs_.scale_tile_shape[0] != 1
              or rhs_.scale_tile_shape[1] < config.block_k
              or rhs_.scale_tile_shape[2] != 1
              or rhs_.scale.shape[2] * rhs_.scale.dtype.itemsize <= 16
              or (
                  isinstance(lhs_, qwix.QArray)
                  and lhs_.qtype == jnp.int8
                  and not gpu_utils.has_tcgen05_int8_support()
              )
          ):
            expect_supported = False
        else:
          if kwargs.get("preferred_element_type") not in (None, jnp.bfloat16):
            # MGPU throws a `ValueError` so we skip the test.
            self.skipTest("f32 output not supported on B200.")
          config = pallas_mosaic_gpu.Config(
              block_m=128,
              block_n=128,
              block_k=128,
              num_stages=2,
              split_k=1,
          )
      elif "a100" in device_kind:
        has_custom_dims = "ragged_dot_dimension_numbers" in kwargs
        if has_custom_dims:
          expect_supported = False
        else:
          config = pallas_mosaic_gpu.Config(
              block_m=64,
              block_n=128,
              block_k=64,
              num_stages=2,
              split_k=1,
          )
      elif isinstance(rhs_, qwix.QArray):
        if (
            rhs_.scale_tile_shape[0] != 1
            or rhs_.scale_tile_shape[1] % _CONFIG.block_k != 0
            or rhs_.scale_tile_shape[2] != 1
            or kwargs.get("preferred_element_type") is not None
        ):
          expect_supported = False

      if expect_supported:
        return op.replace(config=config or _CONFIG)(lhs, rhs, **kwargs)

      with self.assertRaises(NotImplementedError) as e:
        _ = op.replace(config=config or _CONFIG)(lhs, rhs, **kwargs)
      self.skipTest(f"Test not supported: {e.msg}")

    super().__init__(*args, dot_fn=fn)

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    device_kind = jax.devices()[0].device_kind.lower()
    if "a100" in device_kind:
      test_name = self._testMethodName
      if "vjp" in test_name:
        self.skipTest("No contracting dim kernel for A100 (drhs gradient).")
    super().setUp()

  def test_get_autotuning_configs(self):
    op = pallas_mosaic_gpu.PallasMosaicGpuRaggedDot()
    lhs = jax.ShapeDtypeStruct((128, 512), dtype=jnp.bfloat16)
    rhs = jax.ShapeDtypeStruct((4, 512, 512), dtype=jnp.bfloat16)
    group_sizes = (32, 32, 32, 32)
    ba = op.bind(lhs, rhs, group_sizes=group_sizes)

    configs = op._get_autotuning_configs(ba)
    self.assertIsNotNone(configs)
    self.assertNotEmpty(configs)

  def _check_vmap(self, batch_shape, in_axes, collective):
    num_groups, m, k, n = 8, 2048, 768, 2048
    unbatched_shapes = ((m, k), (num_groups, k, n), (num_groups,))
    shapes = [
        (*batch_shape, *shape) if axis == 0 else shape
        for shape, axis in zip(unbatched_shapes, in_axes, strict=True)
    ]
    kx, kw = jax.random.split(jax.random.key(0))
    lhs = jax.random.normal(kx, shapes[0], dtype=jnp.bfloat16)
    rhs = jax.random.normal(kw, shapes[1], dtype=jnp.bfloat16)
    partitions = jnp.array(_VMAP_GROUP_SIZES, dtype=jnp.int32)
    if in_axes[2] is None:
      group_sizes = partitions[0]
    else:
      group_sizes = partitions[: math.prod(batch_shape)].reshape(shapes[2])

    config = None
    if gpu_utils.is_sm100():
      config = pallas_mosaic_gpu.Config(
          block_m=64,
          block_n=128,
          block_k=256,
          num_stages=2,
          split_k=1,
          persistent=False,
          collective=collective,
          grid_minor_dim=common.MatmulDimension.M,
          grid_tile_width=4,
      )
      # The grid has more tiles than the SMs can hold, so that CTAs run several
      # tiles. `k // block_k` is 3, so the K-loop pipeline slot of a tile
      # depends on the number of tiles the CTA ran before.
      cluster_size = 2 if collective else 1
      grid_tiles = (
          math.ceil(m / (config.block_m * cluster_size)) + num_groups - 1
      ) * math.ceil(n / (config.block_n * cluster_size))
      self.assertGreater(
          grid_tiles, jax.devices()[0].core_count // cluster_size
      )
    elif not collective:
      self.skipTest("collective=False only differs on SM100.")
    op = pallas_mosaic_gpu.PallasMosaicGpuRaggedDot(config=config)

    expected = jnp.stack([
        test_base.ref(*(
            x if axis is None else x[idx]
            for x, axis in zip((lhs, rhs, group_sizes), in_axes, strict=True)
        ))
        for idx in np.ndindex(*batch_shape)
    ]).reshape(*batch_shape, m, n)
    expected.block_until_ready()
    # Output elements that the kernel does not write keep the previous content
    # of the device memory. A freed NaN buffer makes that content differ from
    # `expected` when the allocator reuses the buffer for the output.
    poison = jnp.full(expected.shape, jnp.nan, dtype=jnp.bfloat16)
    poison.block_until_ready()
    poison.delete()

    fn = lambda a, b, gs: op(a, b, group_sizes=gs)
    for _ in batch_shape:
      fn = jax.vmap(fn, in_axes=in_axes)
    actual = jax.jit(fn)(lhs, rhs, group_sizes).block_until_ready()
    chex.assert_trees_all_close(actual, expected, atol=0.05, rtol=0.01)

  @parameterized.product(
      in_axes=[(0, None, 0), (None, 0, None), (0, 0, None), (0, 0, 0)],
      collective=[True, False],
  )
  def test_vmap(self, in_axes, collective):
    self._check_vmap((2,), in_axes, collective)

  @parameterized.parameters(True, False)
  def test_nested_vmap(self, collective):
    self._check_vmap((2, 2), (0, 0, 0), collective)


if __name__ == "__main__":
  absltest.main()
