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
"""Tests for the baseline JAX implementation of compress-and-store."""

import dataclasses

from absl.testing import absltest
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.tpu.compress_store import base
from tokamax._src.ops.experimental.tpu.compress_store import test_base

jax.config.parse_flags_with_absl()


def _call(inputs: test_base.CompressStoreInputs, **overrides):
  inputs = dataclasses.replace(inputs, **overrides)
  return base.CompressStore()(*inputs.args, **inputs.kwargs)


class BaseCompressStoreTest(test_base.CompressStoreTestBase):

  def __init__(self, *args):
    super().__init__(*args, compress_store_fn=base.CompressStore())

  def test_rejects_non_int32_positions(self):
    inputs = test_base.make_case_inputs("csa_decode_batch_seq")
    with self.assertRaisesRegex(ValueError, "positions must be int32"):
      _call(inputs, positions=inputs.positions.astype(jnp.int16))

  def test_rejects_bad_rope_head_dim(self):
    inputs = test_base.make_case_inputs("csa_decode_batch_seq")
    with self.assertRaisesRegex(ValueError, "rope_head_dim must be 64"):
      _call(inputs, cos_sin_cache=inputs.cos_sin_cache[:, :32])

  def test_rejects_csa_without_rope_cache(self):
    inputs = test_base.make_case_inputs("csa_decode_batch_seq")
    with self.assertRaisesRegex(ValueError, "needs rope_cache"):
      _call(inputs, rope_cache=None)

  def test_rejects_csa_bad_quant_block(self):
    inputs = test_base.make_case_inputs("csa_decode_batch_seq")
    with self.assertRaisesRegex(ValueError, "quant_block must be 64"):
      _call(inputs, quant_block=128)

  def test_rejects_indexer_without_overlap(self):
    inputs = test_base.make_case_inputs("csa_indexer_decode")
    with self.assertRaisesRegex(ValueError, "requires overlap"):
      _call(inputs, overlap=False)

  def test_rejects_wrong_hca_cache_dtype(self):
    inputs = test_base.make_case_inputs("hca_decode_batch_seq")
    with self.assertRaisesRegex(ValueError, "HCA cache must be uint8"):
      _call(inputs, cache=inputs.cache.astype(jnp.int8))


if __name__ == "__main__":
  absltest.main()
