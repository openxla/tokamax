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

from absl.testing import absltest
from absl.testing import parameterized
from tokamax._src.ops.experimental.mla.v2.tuned_params import TunableParams


class TestMlaTunedParams(parameterized.TestCase):

  @parameterized.named_parameters(
      # identical params → both comparisons are True
      (
          "identical_params",
          TunableParams(
              decode_batch_size=8,
              num_kv_pages_per_block=3,
              num_queries_per_block=1,
              vmem_limit_bytes=64 * 1024 * 1024,
          ),
          TunableParams(
              decode_batch_size=8,
              num_kv_pages_per_block=3,
              num_queries_per_block=1,
              vmem_limit_bytes=64 * 1024 * 1024,
          ),
          True,
          True,
      ),
      # lo is strictly smaller in the resource-demand dimensions,
      # while also having a larger vmem limit.
      (
          "lo_strictly_smaller_and_larger_vmem",
          TunableParams(
              decode_batch_size=4,
              num_kv_pages_per_block=2,
              num_queries_per_block=1,
              vmem_limit_bytes=128 * 1024 * 1024,
          ),
          TunableParams(
              decode_batch_size=8,
              num_kv_pages_per_block=3,
              num_queries_per_block=2,
              vmem_limit_bytes=64 * 1024 * 1024,
          ),
          True,
          False,
      ),
      # lo is strictly larger in the resource-demand dimensions,
      # while also having a smaller vmem limit.
      (
          "lo_strictly_larger_and_smaller_vmem",
          TunableParams(
              decode_batch_size=16,
              num_kv_pages_per_block=4,
              num_queries_per_block=2,
              vmem_limit_bytes=32 * 1024 * 1024,
          ),
          TunableParams(
              decode_batch_size=8,
              num_kv_pages_per_block=3,
              num_queries_per_block=1,
              vmem_limit_bytes=64 * 1024 * 1024,
          ),
          False,
          True,
      ),
      # mixed dimensions should not satisfy either ordering relation
      (
          "mixed_dimensions",
          TunableParams(
              decode_batch_size=8,
              num_kv_pages_per_block=4,
              num_queries_per_block=1,
              vmem_limit_bytes=64 * 1024 * 1024,
          ),
          TunableParams(
              decode_batch_size=16,
              num_kv_pages_per_block=2,
              num_queries_per_block=2,
              vmem_limit_bytes=64 * 1024 * 1024,
          ),
          False,
          False,
      ),
  )
  def test_tunable_params_ge_le(self, lo, hi, expect_le, expect_ge):
    self.assertEqual(lo <= hi, expect_le, f"Expected lo<=hi to be {expect_le}")
    self.assertEqual(lo >= hi, expect_ge, f"Expected lo>=hi to be {expect_ge}")

  def test_tunable_params_ge_le_single_dim_difference(self):
    """A single larger dimension should make the larger object >= and the

    smaller object <=, because both operators require all dimensions to match
    the ordering relation.
    """

    base = TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=2,
        num_queries_per_block=1,
        vmem_limit_bytes=64 * 1024 * 1024,
    )
    larger_decode_batch = TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=2,
        num_queries_per_block=1,
        vmem_limit_bytes=64 * 1024 * 1024,
    )

    self.assertTrue(base <= larger_decode_batch)
    self.assertFalse(base >= larger_decode_batch)
    self.assertTrue(larger_decode_batch >= base)
    self.assertFalse(larger_decode_batch <= base)


if __name__ == "__main__":
  absltest.main()
