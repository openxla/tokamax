# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax.numpy as jnp

from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common


class GroupInfoTest(parameterized.TestCase):

  @parameterized.parameters(3, 5, 100)
  def test_create_aligned(self, tid_size):
    group_sizes = jnp.array([17, 31, 24])
    actual = common.GroupInfo.create_aligned(
        group_sizes=group_sizes, tile=32, tid_size=tid_size, align_tile=8
    )
    block_start_tail = [48 + 32 * (i + 1) for i in range(tid_size - 3)]
    expected = dict(
        group_id=jnp.array([0, 1, 2] + [2] * (tid_size - 3)),
        block_start=jnp.array([0, 16, 48] + block_start_tail),
        actual_start=jnp.array([0, 17, 48] + block_start_tail),
        actual_end=jnp.array([17, 48, 72] + [72] * (tid_size - 3)),
        start_within_block=jnp.array([0, 1, 0] + [0] * (tid_size - 3)),
        actual_size=jnp.array([17, 31, 24] + [0] * (tid_size - 3)),
    )
    chex.assert_trees_all_equal(dataclasses.asdict(actual), expected)

  def test_create_aligned_empty_groups(self):
    actual = common.GroupInfo.create_aligned(
        group_sizes=jnp.array([17, 0, 31]), tile=32, tid_size=5, align_tile=8
    )
    expected = dict(
        group_id=jnp.array([0, 2, 2, 2, 2]),
        block_start=jnp.array([0, 16, 48, 80, 112]),
        actual_start=jnp.array([0, 17, 48, 80, 112]),
        actual_end=jnp.array([17, 48, 48, 48, 48]),
        start_within_block=jnp.array([0, 1, 0, 0, 0]),
        actual_size=jnp.array([17, 31, 0, 0, 0]),
    )
    chex.assert_trees_all_equal(dataclasses.asdict(actual), expected)

  def test_create_aligned_multiple_blocks_per_group(self):
    actual = common.GroupInfo.create_aligned(
        group_sizes=jnp.array([41, 32, 33]), tile=32, tid_size=6, align_tile=8
    )
    expected = dict(
        group_id=jnp.array([0, 0, 1, 1, 2, 2]),
        block_start=jnp.array([0, 32, 40, 72, 72, 104]),
        actual_start=jnp.array([0, 32, 41, 72, 73, 104]),
        actual_end=jnp.array([32, 41, 72, 73, 104, 106]),
        start_within_block=jnp.array([0, 0, 1, 0, 1, 0]),
        actual_size=jnp.array([32, 9, 31, 1, 31, 2]),
    )
    chex.assert_trees_all_equal(dataclasses.asdict(actual), expected)

  def test_create_aligned_multiple_blocks_per_group_aligned(self):
    actual = common.GroupInfo.create_aligned(
        group_sizes=jnp.array([40, 32, 33]), tile=32, tid_size=6, align_tile=8
    )
    expected = dict(
        group_id=jnp.array([0, 0, 1, 2, 2, 2]),
        block_start=jnp.array([0, 32, 40, 72, 104, 136]),
        actual_start=jnp.array([0, 32, 40, 72, 104, 136]),
        actual_end=jnp.array([32, 40, 72, 104, 105, 105]),
        start_within_block=jnp.array([0, 0, 0, 0, 0, 0]),
        actual_size=jnp.array([32, 8, 32, 32, 1, 0]),
    )
    chex.assert_trees_all_equal(dataclasses.asdict(actual), expected)


if __name__ == "__main__":
  absltest.main()
