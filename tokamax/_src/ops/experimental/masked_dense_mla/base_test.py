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
"""Tests for the baseline JAX implementation of MaskedDenseMLA."""

from absl.testing import absltest
import jax.numpy as jnp
from tokamax._src.ops.experimental.masked_dense_mla import base
from tokamax._src.ops.experimental.masked_dense_mla import test_base


class BaseMaskedDenseMlaTest(test_base.MaskedDenseMlaTestBase):

  def __init__(self, *args):
    super().__init__(*args, masked_dense_mla_fn=base.MaskedDenseMla())

  def test_rejects_mismatched_rope_rows(self):
    inputs, _ = test_base.make_inputs(batch_size=2, topk=64)
    bad_rope = jnp.zeros((inputs[2].shape[0], 2, 4, 128), dtype=jnp.uint8)
    with self.assertRaisesRegex(ValueError, "page_size"):
      base.MaskedDenseMla().bind(inputs[0], inputs[1], bad_rope, *inputs[3:])

  def test_rejects_mismatched_kv_lens_length(self):
    inputs, _ = test_base.make_inputs(batch_size=2, topk=64)
    bad_kv_lens = jnp.zeros((5,), dtype=jnp.int32)
    with self.assertRaisesRegex(ValueError, "kv_lens"):
      base.MaskedDenseMla().bind(
          inputs[0], inputs[1], inputs[2], bad_kv_lens, *inputs[4:]
      )

  def test_rejects_out_of_bounds_max_kv_len(self):
    inputs, _ = test_base.make_inputs(batch_size=2, topk=64)
    with self.assertRaisesRegex(ValueError, "max_kv_len"):
      base.MaskedDenseMla().bind(*inputs, max_kv_len=100000)


if __name__ == "__main__":
  absltest.main()
