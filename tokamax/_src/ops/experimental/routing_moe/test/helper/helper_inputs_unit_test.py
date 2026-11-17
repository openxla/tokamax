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
"""Tests for helper_inputs and helper_reference."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import block_scaled
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_inputs
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_reference


def _rows_per_expert(gating: jax.Array, topk: int) -> np.ndarray:
  """Returns the rows each expert receives from top-k routing of `gating`."""
  _, indices = helper_inputs.top_k(gating, topk)
  routing = helper_inputs.Routing.make_routing(indices, gating.shape[1])
  return np.asarray(routing.group_sizes)


class RandomGatingTest(parameterized.TestCase):

  def test_default_is_uniform(self):
    key = jax.random.key(0)
    k_logits, _ = jax.random.split(key)
    np.testing.assert_array_equal(
        helper_inputs.random_gating(key, 16, 8),
        jax.random.uniform(k_logits, (16, 8), dtype=jnp.bfloat16),
    )

  @parameterized.product(
      num_experts=[256, 512], max_to_mean_load=[1.67, 2.0, 4.0]
  )
  def test_skewed_load_follows_target(
      self, num_experts: int, max_to_mean_load: float
  ):
    gating = helper_inputs.random_gating(
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
    gating = helper_inputs.random_gating(
        jax.random.key(0),
        1024,
        num_experts,
        max_to_mean_load=4.0,
        num_skipped=num_skipped,
        force_expert=forced,
    )
    _, indices = helper_inputs.top_k(gating, 8)
    self.assertLess(int(indices.max()), num_experts - num_skipped)
    self.assertEqual(int(indices[0, 0]), forced)

  @parameterized.named_parameters(
      dict(testcase_name="_below_one", max_to_mean_load=0.5),
      dict(testcase_name="_unreachable", max_to_mean_load=1e6),
  )
  def test_invalid_max_to_mean_load_raises(self, max_to_mean_load: float):
    with self.assertRaisesRegex(ValueError, "max_to_mean_load must be in"):
      helper_inputs.random_gating(
          jax.random.key(0), 16, 8, max_to_mean_load=max_to_mean_load
      )


class HelperReferenceBlockScaledTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if not block_scaled.is_available():
      self.skipTest("Block-scaled matmul is not available in this build.")

  def test_reference_gmm_block_scaled_compact_and_prepared_scales_match(self):
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(0), 5)
    x_sorted = jax.random.normal(k1, (12, 512), dtype=jnp.bfloat16)
    group_sizes = jnp.array([5, 7], dtype=jnp.int32)
    w1_f32 = jax.random.normal(k2, (2, 512, 256), dtype=jnp.float32) / 10
    w2_f32 = jax.random.normal(k3, (2, 128, 512), dtype=jnp.float32) / 10
    b1 = jax.random.normal(k4, (2, 1, 256), dtype=jnp.bfloat16) / 10
    b2 = jax.random.normal(k5, (2, 1, 512), dtype=jnp.bfloat16) / 10

    w1_q, s1 = block_scaled.quantize(w1_f32, jnp.float8_e4m3fn)
    w2_q, s2 = block_scaled.quantize(w2_f32, jnp.float8_e5m2)
    s1_pad = block_scaled.prepare_scales(s1, 512)
    s2_pad = block_scaled.prepare_scales(s2, 128)

    out_compact = helper_reference.reference_gmm(
        x_sorted,
        arg.SingleWeightWrapper.create(w1_q, s1, b1, fused_gate_up=True),
        group_sizes,
        w2=arg.SingleWeightWrapper.create(w2_q, s2, b2),
        fuse_act="silu",
    )
    out_padded = helper_reference.reference_gmm(
        x_sorted,
        arg.SingleWeightWrapper.create(w1_q, s1_pad, b1, fused_gate_up=True),
        group_sizes,
        w2=arg.SingleWeightWrapper.create(w2_q, s2_pad, b2),
        fuse_act="silu",
    )
    out_unquant = helper_reference.reference_gmm(
        x_sorted,
        arg.SingleWeightWrapper.create(
            w1_f32.astype(jnp.bfloat16), bias=b1, fused_gate_up=True
        ),
        group_sizes,
        w2=arg.SingleWeightWrapper.create(w2_f32.astype(jnp.bfloat16), bias=b2),
        fuse_act="silu",
    )
    np.testing.assert_array_equal(out_compact, out_padded)
    np.testing.assert_allclose(
        out_compact,
        out_unquant,
        rtol=0.0,
        atol=0.12 * float(np.abs(out_unquant).max()),
    )

  def test_shared_expert_block_scaled_matches_dequantized(self):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(1), 4)
    x = jax.random.normal(k1, (8, 512), dtype=jnp.bfloat16)
    w1_f32 = jax.random.normal(k2, (1, 512, 256), dtype=jnp.float32) / 10
    w2_f32 = jax.random.normal(k3, (1, 128, 512), dtype=jnp.float32) / 10
    w_gate = jax.random.normal(k4, (512, 1), dtype=jnp.bfloat16) / 10

    w1_q, s1 = block_scaled.quantize(w1_f32, jnp.float8_e4m3fn)
    w2_q, s2 = block_scaled.quantize(w2_f32, jnp.float8_e4m3fn)
    s1_pad = block_scaled.prepare_scales(s1, 512)
    s2_pad = block_scaled.prepare_scales(s2, 128)

    out_compact = helper_reference.shared_expert(
        x, w1_q, w2_q, s1, s2, w_gate, activation="silu"
    )
    out_padded = helper_reference.shared_expert(
        x, w1_q, w2_q, s1_pad, s2_pad, w_gate, activation="silu"
    )
    out_unquant = helper_reference.shared_expert(
        x,
        w1_f32.astype(jnp.bfloat16),
        w2_f32.astype(jnp.bfloat16),
        None,
        None,
        w_gate,
        activation="silu",
    )
    np.testing.assert_array_equal(out_compact, out_padded)
    np.testing.assert_allclose(
        out_compact,
        out_unquant,
        rtol=0.0,
        atol=0.06 * float(np.abs(out_unquant).max()),
    )


if __name__ == "__main__":
  absltest.main()
