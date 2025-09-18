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
from tokamax._src.ops.flex_attention import base
from tokamax._src.ops.flex_attention import test_base
from tokamax._src.ops.flex_attention import wrapper_test_base


class FlexAttentionTest(test_base.FlexAttentionTestBase):

  def __init__(self, *args):
    super().__init__(*args, flex_attn=base.FlexAttention())


class WrappedFlexAttentionTest(wrapper_test_base.WrappedFlexAttentionTestBase):

  def __init__(self, *args):
    super().__init__(*args, flex_attn=base.FlexAttention())

  # Allow for non-deterministic XLA reference in OSS (b/445821600).
  def test_normalize_output(self):
    with wrapper_test_base.test_base.override_test_args(atol=5e-5):
      super().test_normalize_output()


if __name__ == "__main__":
  absltest.main()
