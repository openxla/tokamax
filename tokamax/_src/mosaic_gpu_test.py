# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np
from tokamax._src import gpu_utils
from tokamax._src import mosaic_gpu as common


class PallasMosaicGpuConversionUtilsTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    if not (gpu_utils.is_sm100() or gpu_utils.is_sm90()):
      self.skipTest("The test is only for Hopper and Blackwell gen gpus")
    super().setUp()

  def test_encode_i4_as_biased_f8e4m3fn(self,):

    shape = (128, 128)

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float8_e4m3fn),
        in_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),),
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        compiler_params=plgpu.CompilerParams(),
    )
    def kernel(x_ref, o_ref):
      layout = plgpu.Layout.WGMMA_UPCAST_4X
      x = plgpu.load(x_ref, (), layout=layout, optimized=False)
      x = common.int4_as_biased_f8e4m3fn(x, layout)
      o_ref[...] = x

    x = jnp.arange(-8, 8, dtype=jnp.int8)
    x = jnp.tile(x, shape[0] * shape[1] // 16).astype(jnp.int4)
    x = jax.random.permutation(jax.random.key(1), x, axis=0).reshape(shape)

    actual = kernel(x).astype(jnp.float32) * 512 - 8
    np.testing.assert_array_equal(actual, x.astype(jnp.float32))

  @parameterized.parameters(
      ((128, 128, 256), plgpu.Layout.WGMMA_UPCAST_4X),
      ((128, 128, 256), plgpu.Layout.TCGEN05_TMEM_NATIVE(8)),
  )
  def test_tcgen05_biased_i4_f8e4m3fn(self, shape, layout):
    if not gpu_utils.is_sm100():
      self.skipTest("The test is only for Blackwell")
    m, n, k = shape
    w_dtype = jnp.int4
    fp8_dtype = jnp.float8_e4m3fn

    i4_transforms = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(32))
    f8_transforms = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(64))
    o_transforms = (plgpu.TilingTransform((8, 32)), plgpu.SwizzleTransform(128))

    scratch_shapes = [
        plgpu.SMEM((n, k), fp8_dtype, transforms=f8_transforms),
        plgpu.Barrier(orders_tensor_core=True),
        plgpu.TMEM((m, n), jnp.float32),
    ]

    @functools.partial(
        pl.pallas_call,
        in_specs=(
            plgpu.BlockSpec(memory_space=plgpu.SMEM, transforms=f8_transforms),
            plgpu.BlockSpec(memory_space=plgpu.SMEM, transforms=i4_transforms),
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        out_specs=plgpu.BlockSpec(transforms=o_transforms),
        scratch_shapes=scratch_shapes,
        compiler_params=plgpu.CompilerParams(),
    )
    def kernel(
        a_smem,
        b_smem,
        out_ref,
        b_up_smem,
        barrier_ref,
        acc_tmem,
    ):
      b = plgpu.load(b_smem, (), layout=layout, optimized=False)

      # Note, this approach for row sum calculation is slow due to upcasting
      a_row_sum = plgpu.load(
          a_smem,
          (),
          layout=plgpu.Layout.TCGEN05,
          optimized=True,
      ).astype(jnp.float32).sum(1)

      b_up_smem[...] = common.int4_as_biased_f8e4m3fn(b, layout)
      plgpu.commit_smem()

      plgpu.tcgen05_mma(
          acc_tmem,
          a_smem,
          plgpu.transpose_ref(b_up_smem, (1, 0)),
          accumulate=False,
          barrier=barrier_ref
      )

      plgpu.barrier_wait(barrier_ref)
      # Following the scheme D = A@B with B as biased f8e4m3fn
      # the accumulator should be descaled & debiased
      # `S * D - Z * row_sum(A)` where S is scale=512 and Z is bias=8
      acc = 512 * plgpu.async_load_tmem(acc_tmem, layout=plgpu.Layout.TCGEN05)
      acc -= 8 * lax.broadcast_in_dim(a_row_sum, acc.shape, [0])
      out_ref[...] = acc
    a = jax.random.uniform(jax.random.key(1), shape=(m, k), dtype=jnp.float32)
    a_f8 = a.astype(fp8_dtype)
    b = jax.random.randint(
        jax.random.key(2), shape=(n, k), minval=-8, maxval=7, dtype=w_dtype
    )
    actual = kernel(a_f8, b)
    ref = a_f8.astype(jnp.float32) @ b.astype(jnp.float32).T
    np.testing.assert_array_equal(actual, ref)


if __name__ == "__main__":
  absltest.main()
