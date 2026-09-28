# Copyright 2026 Google LLC
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
"""Tests for Pallas Mosaic TPU GDN attention."""

from absl.testing import absltest
from tokamax._src.ops.causal_conv1d_gated_delta_rule import pallas_mosaic_tpu
from tokamax._src.ops.causal_conv1d_gated_delta_rule import test_base


_OP = pallas_mosaic_tpu.PallasMosaicTpuCausalConv1dGatedDeltaRule


class GDNAttentionTest(test_base.CausalConv1dGatedDeltaRuleTestBase):

  def __init__(self, *args):
    super().__init__(
        *args,
        gdn_fn=_OP(),
        static_argnames=test_base.OP_STATIC_ARGNAMES,
    )


class GDNSecurityTest(test_base.CausalConv1dGatedDeltaRuleSecurityTestBase):

  def __init__(self, *args):
    super().__init__(*args, gdn_fn=_OP())


if __name__ == "__main__":
  absltest.main()
