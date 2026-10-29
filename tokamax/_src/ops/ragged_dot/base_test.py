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
import functools
from typing import override

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import test_base


# TODO: Make ragged dot base test work for both GPU and TPU.
class RaggedDotTest(test_base.RaggedDotTestBase):

  def __init__(self, *args):
    super().__init__(*args, dot_fn=base.RaggedDot())

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Disabled for now due to numeric issues.")
    super().setUp()


class RaggedDotWithExplicitVjpTest(test_base.RaggedDotTestBase):

  def __init__(self, *args):
    super().__init__(*args, dot_fn=base.RaggedDot(vjp=base.vjp))

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Disabled for now due to numeric issues.")
    super().setUp()

  @override
  def _test_bench(self, spec):
    if "128_m16384" in self._testMethodName and jax.default_backend() == "gpu":
      self.skipTest("OOMs on MiG slice.")
    super()._test_bench(spec)

  @parameterized.product(
      rhs_dtype=(jnp.bfloat16, jnp.float32),
      preferred_element_type=(None, jnp.bfloat16, jnp.float32),
  )
  def test_vjp_preferred_element_type(self, rhs_dtype, preferred_element_type):
    num_groups, m, k, n = 8, 128, 64, 128
    lhs, rhs, group_sizes = self._create_inputs(
        num_groups, m, k, n, jnp.bfloat16
    )
    rhs = rhs.astype(rhs_dtype)
    f = functools.partial(
        self._dot_fn,
        group_sizes=group_sizes,
        preferred_element_type=preferred_element_type,
    )
    out, f_vjp = jax.vjp(f, lhs, rhs)

    if preferred_element_type:
      expected_dtype = jnp.dtype(preferred_element_type)
    else:
      expected_dtype = jnp.result_type(lhs, rhs)

    self.assertEqual(out.dtype, expected_dtype)

    dlhs, drhs = f_vjp(jnp.ones_like(out))
    # `dlhs` always follow `lhs`(primal)'s data type
    self.assertEqual(dlhs.dtype, lhs.dtype)

    # `drhs` follows user-stet `preferred_element_type` if set
    self.assertEqual(drhs.dtype, expected_dtype)


class GenerateGroupSizesTest(parameterized.TestCase):

  @parameterized.parameters((256, 4), (16, 64), (256, 4, (0.1, 0.2, 0.3, 0.4)))
  def test_generate_group_sizes(self, m, num_groups, p=None):
    group_sizes = base.generate_group_sizes(m=m, num_groups=num_groups, p=p)
    self.assertLen(group_sizes, num_groups)
    self.assertEqual(sum(group_sizes), m)


if __name__ == "__main__":
  absltest.main()
