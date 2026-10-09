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
"""Tests for the baseline JAX implementation of SparseMLA."""

from absl.testing import absltest
import jax.numpy as jnp
from tokamax._src.ops.experimental.sparse_mla import base
from tokamax._src.ops.experimental.sparse_mla import test_base


class BaseSparseMlaTest(test_base.SparseMlaTestBase):

  def __init__(self, *args):
    super().__init__(*args, sparse_mla_fn=base.SparseMla())

  def test_rejects_non_int32_caches(self):
    inputs, _ = test_base.make_inputs(batch_size=2, topk=128)
    bad_nope = inputs[1].astype(jnp.int16)
    with self.assertRaisesRegex(ValueError, "int32"):
      base.SparseMla().bind(inputs[0], bad_nope, *inputs[2:])

  def test_rejects_mismatched_rope_rows(self):
    inputs, _ = test_base.make_inputs(batch_size=2, topk=128)
    bad_rope = jnp.zeros((inputs[2].shape[0], 2, 128), dtype=jnp.int32)
    with self.assertRaisesRegex(ValueError, "page_size"):
      base.SparseMla().bind(inputs[0], inputs[1], bad_rope, *inputs[3:])

  def test_rejects_bad_topk(self):
    inputs, _ = test_base.make_inputs(batch_size=2, topk=128)
    bad_topk = jnp.zeros((inputs[0].shape[0], 192), dtype=jnp.int32)
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      base.SparseMla().bind(
          inputs[0], inputs[1], inputs[2], bad_topk, *inputs[4:]
      )


if __name__ == "__main__":
  absltest.main()
