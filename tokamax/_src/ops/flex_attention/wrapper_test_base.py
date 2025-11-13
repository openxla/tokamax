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
"""Test base for wrapped FlexAttention ops."""
import jax
from tokamax._src.ops.attention import test_base
from tokamax._src.ops.flex_attention import base
from tokamax._src.ops.flex_attention import wrapper


class WrappedFlexAttentionTestBase(test_base.AttentionTestBase):
  """Test base for wrapped FlexAttention ops."""

  def __init__(self, *args, flex_attn: base.FlexAttention, **kwargs):
    wrapped = wrapper.WrappedFlexAttention(impl=flex_attn)
    super().__init__(*args, attention_fn=wrapped, **kwargs)

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    super().setUp()
