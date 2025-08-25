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
from absl.testing import parameterized
import jax
from tokamax._src.ops.flex_attention import pallas_triton
from tokamax._src.ops.flex_attention import test_base
from tokamax._src.ops.flex_attention import wrapper_test_base


class PallasTritonFlexAttentionTest(test_base.FlexAttentionTestBase):

  def __init__(self, *args):
    super().__init__(*args, flex_attn=pallas_triton.PallasTritonFlexAttention())

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    super().setUp()


class WrappedPallasTritonFlexAttentionTest(
    wrapper_test_base.WrappedFlexAttentionTestBase
):

  def __init__(self, *args):
    super().__init__(
        *args,
        flex_attn=pallas_triton.PallasTritonFlexAttention(),
        supports_vjp=False,  # TODO: Support VJP.
    )

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    super().setUp()

  @parameterized.parameters(
      *wrapper_test_base.base_names_and_params("test_vmap")
  )
  def test_vmap(self, *_):
    self.skipTest("TODO: Fix `vmap` support.")


if __name__ == "__main__":
  absltest.main()
