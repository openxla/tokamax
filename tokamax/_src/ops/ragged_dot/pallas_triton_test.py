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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import gpu_utils
from tokamax._src.ops.ragged_dot import pallas_triton
from tokamax._src.ops.ragged_dot import test_base
from typing_extensions import override


class PallasTritonRaggedDotTest(test_base.RaggedDotTestBase):

  def __init__(self, *args):
    super().__init__(*args, dot_fn=pallas_triton.PallasTritonRaggedDot())

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    super().setUp()

  @parameterized.parameters(2, 4)
  def test_split_k(self, split_k):
    config = pallas_triton.Config(
        block_m=128,
        block_n=128,
        block_k=32,
        split_k=split_k,
        num_warps=4,
        num_stages=4,
    )
    split_k_dot = pallas_triton.PallasTritonRaggedDot(config=config)

    with mock.patch.object(self, "_dot_fn", split_k_dot):
      with test_base.override_chex_args(atol=2e-5):
        self.test_simple1()  # pytype: disable=attribute-error

  def test_split_k_quantized(self):
    config = pallas_triton.Config(
        block_m=128,
        block_n=128,
        block_k=32,
        split_k=4,
        num_warps=4,
        num_stages=4,
    )
    split_k_dot = pallas_triton.PallasTritonRaggedDot(
        config=config, split_k_intermediate_dtype=jnp.float32
    )

    with mock.patch.object(self, "_dot_fn", split_k_dot):
      self.test_quantized0()  # pytype: disable=attribute-error

  def test_vjp_empty_groups(self):
    lhs = jax.random.normal(
        jax.random.key(0), (16, 64), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    rhs = jax.random.normal(
        jax.random.key(1), (4, 64, 32), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    group_sizes = jnp.array([8, 0, 8, 0], jnp.int32)

    def loss(lhs_arg, rhs_arg):
      out = self._dot_fn_f32(
          lhs_arg,
          rhs_arg,
          group_sizes=group_sizes,
          precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
      )
      return jnp.sum(out.astype(jnp.float32))

    _, (_, drhs) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(lhs, rhs)

    self.assertFalse(np.asarray(~jnp.isfinite(drhs)).any())
    np.testing.assert_array_equal(
        np.asarray(drhs)[np.asarray(group_sizes) == 0],
        np.zeros_like(np.asarray(drhs)[np.asarray(group_sizes) == 0]),
    )

  @override
  def _test_simple(self, dtype):
    with test_base.override_chex_args(atol=1e-6):
      super()._test_simple(dtype)

  @override
  def _test_bench(self, spec):
    xs = jax.tree.leaves((spec["lhs"], spec["rhs"]))
    if any(x.dtype == jnp.float8_e4m3fn for x in xs) and gpu_utils.is_sm80():
      with self.assertRaises(NotImplementedError):
        super()._test_bench(spec)
    else:
      super()._test_bench(spec)


if __name__ == "__main__":
  absltest.main()
