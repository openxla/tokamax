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
import immutabledict
import jax
import jax.numpy as jnp
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu
from tokamax._src.ops.ragged_dot import test_base


QuantizedArray = quantization.QuantizedArray

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
      if not isinstance(lhs, jax.Array):
        self.skipTest(
            f"MGPU Kernel only supports jax.Arrays for lhs, got: {lhs}."
        )

      if lhs.dtype != jnp.bfloat16:
        self.skipTest(
            f"Non-bfloat16 not supported by mgpu kernel ({lhs.dtype=})."
        )

      if lhs.dtype != rhs.dtype:
        self.skipTest(
            f"lhs.dtype={lhs.dtype} must be equal to rhs.dtype={rhs.dtype}"
        )

      if kwargs.get("preferred_element_type") is not None:
        self.skipTest("TODO: Support preferred_element_type.")

      if lhs.shape[-1] % (128 // jnp.dtype(lhs.dtype).itemsize):
        self.skipTest("TODO: Support tile aligned K dimension.")

      if isinstance(rhs, QuantizedArray) and rhs.tile_shape != (1, 256, 1):
        self.skipTest(
            "TODO: Only scaling tile supported is (1, 256, 1) got:"
            f" {rhs.tile_shape}."
        )

      return op.with_config(config or _CONFIG)(lhs, rhs, **kwargs)

    super().__init__(*args, dot_fn=fn)


if __name__ == "__main__":
  absltest.main()
