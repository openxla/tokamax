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


if __name__ == "__main__":
  absltest.main()
