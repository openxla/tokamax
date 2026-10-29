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
"""Tests for the baseline JAX implementation of MLA-SWA."""

from absl.testing import absltest
import jax.numpy as jnp
from tokamax._src.ops.experimental.mla_swa import base
from tokamax._src.ops.experimental.mla_swa import test_base


class BaseMlaSwaTest(test_base.MlaSwaTestBase):

  def __init__(self, *args):
    super().__init__(*args, mla_swa_fn=base.MlaSwa())

  def test_rejects_bad_sliding_window(self):
    inputs, _ = test_base.make_inputs(batch_size=2)
    with self.assertRaisesRegex(ValueError, "sliding_window"):
      base.MlaSwa().bind(
          *inputs,
          sliding_window=0,
          logical_page_size=test_base.PAGE_SIZE,
      )

  def test_rejects_bad_logical_page_size(self):
    inputs, _ = test_base.make_inputs(batch_size=2)
    with self.assertRaisesRegex(ValueError, "logical_page_size"):
      base.MlaSwa().bind(
          *inputs,
          sliding_window=test_base.SLIDING_WINDOW,
          logical_page_size=1000,
      )

  def test_rejects_mismatched_kv_lens_length(self):
    inputs, _ = test_base.make_inputs(batch_size=2)
    bad_kv_lens = jnp.zeros((5,), dtype=jnp.int32)
    with self.assertRaisesRegex(ValueError, "kv_lens"):
      base.MlaSwa().bind(
          inputs[0],
          inputs[1],
          inputs[2],
          bad_kv_lens,
          *inputs[4:],
          sliding_window=test_base.SLIDING_WINDOW,
          logical_page_size=test_base.PAGE_SIZE,
      )


if __name__ == "__main__":
  absltest.main()
