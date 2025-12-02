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

from absl.testing import absltest
import jax
import jax.numpy as jnp
import qwix
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu
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


class PallasMosaicGpuRaggedDotTest(test_base.RaggedDotTestBase):

  def __init__(self, *args):
    op = pallas_mosaic_gpu.PallasMosaicGpuRaggedDot()

    def fn(lhs, rhs, *, config=None, **kwargs):
      expect_supported = (lhs.dtype == rhs.dtype == jnp.bfloat16) and (
          lhs.shape[-1] % (128 // jnp.dtype(lhs.dtype).itemsize) == 0
      )

      rhs_ = jax.eval_shape(quantization.as_array_or_qarray, rhs)

      device_kind = jax.devices()[0].device_kind.lower()
      if "b200" in device_kind:
        config = pallas_mosaic_gpu.Config(
            block_m=128,
            block_n=128,
            block_k=256,
            num_stages=2,
            split_k=1,
            collective=True,
            persistent=True,
        )
        if (
            not isinstance(rhs_, qwix.QArray)
            or (rhs_.qtype != jnp.int4)
            or (rhs_.scale_tile_shape[0] != 1)
            or (rhs_.scale_tile_shape[1] < _CONFIG.block_k)
            or (rhs_.scale_tile_shape[2] != 1)
        ):
          expect_supported = False
      elif isinstance(rhs_, qwix.QArray):
        if (
            rhs_.scale_tile_shape != (1, _CONFIG.block_k, 1)
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
    super().setUp()


if __name__ == "__main__":
  absltest.main()
