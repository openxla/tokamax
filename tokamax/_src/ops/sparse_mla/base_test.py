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

import copy
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import numerics
from tokamax._src.ops.sparse_mla import arg_specs
from tokamax._src.ops.sparse_mla import base
from tokamax._src.ops.sparse_mla import reference
from tokamax._src.ops.sparse_mla import test_base


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

  def test_arg_specs_packed_cache_initializers(self):
    # Verify host (< threshold) and device (>= threshold) initialization paths
    # produce finite, zero-mean sub-words across every packed byte / bf16 half.
    for num_pages, expected_device_puts in ((1, 2), (32, 0)):
      nope_spec = arg_specs._PackedCacheInitializer(
          (num_pages, 256, 128), "nope"
      )
      rope_spec = arg_specs._PackedCacheInitializer(
          (num_pages, 64 * (1 if num_pages == 1 else 4), 128), "rope"
      )
      self.assertEqual(copy.deepcopy(nope_spec), nope_spec)
      self.assertNotEqual(
          arg_specs._PackedCacheInitializer((1, 8, 128), "nope"),
          arg_specs._PackedCacheInitializer((1, 8, 128), "rope"),
      )
      with mock.patch.object(
          jax, "device_put", wraps=jax.device_put
      ) as mock_device_put:
        nope, rope = numerics.random_initialize((nope_spec, rope_spec), seed=0)
      self.assertEqual(mock_device_put.call_count, expected_device_puts)
      self.assertEqual(nope.dtype, jnp.int32)
      self.assertEqual(rope.dtype, jnp.int32)

      nope_u32 = np.asarray(nope).view(np.uint32)
      for shift in (0, 8, 16, 24):
        plane = (nope_u32 >> shift) & 0xFF
        self.assertGreater(len(np.unique(plane[:, :, :64])), 16)
        if shift < 24:
          self.assertGreater(len(np.unique(plane[:, :, 64:])), 16)
        else:
          self.assertEqual(
              set(np.unique(plane[:, :, 64:])), {0x78, 0x79, 0x7A, 0x7B}
          )

      rope_u32 = np.asarray(rope).view(np.uint32)
      self.assertGreater(len(np.unique(rope_u32 & 0xFFFF)), 64)
      self.assertGreater(len(np.unique(rope_u32 >> 16)), 64)

    # Check reference dequantization of an actual ArgSpec produces finite values
    # and no constant planes.
    small_spec = next(
        s
        for s in arg_specs.ARG_SPECS
        if s.name == "prefill_b1_q1024_kv1024_p256_topk2048_h64"
    )
    args = numerics.random_initialize(small_spec.args, seed=0)
    kv = reference.dequantize_kv_cache(
        args["cache_kv_nope"], args["cache_kv_rope"]
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(kv))))
    self.assertLess(abs(float(jnp.mean(kv))), 0.05)
    self.assertGreater(float(jnp.std(kv[..., :448])), 0.1)
    self.assertGreater(float(jnp.std(kv[..., 448:])), 0.1)


if __name__ == "__main__":
  absltest.main()
