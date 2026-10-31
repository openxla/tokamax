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
"""Tests for the router logit generator in test_inputs."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe.test.common import test_inputs


def _rows_per_expert(gating: jax.Array, topk: int) -> np.ndarray:
  """Returns the rows each expert receives from top-k routing of `gating`."""
  _, indices = test_inputs.top_k(gating, topk)
  routing = test_inputs.Routing.make_routing(indices, gating.shape[1])
  return np.asarray(routing.group_sizes)


class RandomGatingTest(parameterized.TestCase):

  def test_default_is_uniform(self):
    key = jax.random.key(0)
    k_logits, _ = jax.random.split(key)
    np.testing.assert_array_equal(
        test_inputs.random_gating(key, 16, 8),
        jax.random.uniform(k_logits, (16, 8), dtype=jnp.bfloat16),
    )

  @parameterized.product(num_experts=[256, 512], max_to_mean_load=[2.0, 4.0])
  def test_skewed_load_follows_target(
      self, num_experts: int, max_to_mean_load: float
  ):
    gating = test_inputs.random_gating(
        jax.random.key(0),
        8192,
        num_experts,
        max_to_mean_load=max_to_mean_load,
    )
    rows = _rows_per_expert(gating, topk=8)
    self.assertBetween(
        rows.max() / rows.mean(),
        0.8 * max_to_mean_load,
        1.2 * max_to_mean_load,
    )
    # The lognormal tail also leaves some experts well below the mean.
    self.assertLess(rows.min(), 0.8 * rows.mean())

  def test_skewed_keeps_skipped_and_forced_experts(self):
    num_experts, num_skipped, forced = 64, 4, 10
    gating = test_inputs.random_gating(
        jax.random.key(0),
        1024,
        num_experts,
        max_to_mean_load=4.0,
        num_skipped=num_skipped,
        force_expert=forced,
    )
    _, indices = test_inputs.top_k(gating, 8)
    self.assertLess(int(indices.max()), num_experts - num_skipped)
    self.assertEqual(int(indices[0, 0]), forced)

  @parameterized.named_parameters(
      dict(testcase_name="_below_one", max_to_mean_load=0.5),
      dict(testcase_name="_unreachable", max_to_mean_load=1e6),
  )
  def test_invalid_max_to_mean_load_raises(self, max_to_mean_load: float):
    with self.assertRaisesRegex(ValueError, "max_to_mean_load must be in"):
      test_inputs.random_gating(
          jax.random.key(0), 16, 8, max_to_mean_load=max_to_mean_load
      )

  def test_skew_with_active_expert_ratio_raises(self):
    with self.assertRaisesRegex(ValueError, "cannot both be set"):
      test_inputs.random_gating(
          jax.random.key(0),
          16,
          8,
          active_expert_ratio=0.5,
          max_to_mean_load=2.0,
      )


if __name__ == "__main__":
  absltest.main()
