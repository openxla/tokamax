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
"""Unit tests for RoutingMoEConfig."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import sharding
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.layer import config as moe_config
from tokamax._src.ops.experimental.routing_moe.layer import plan as moe_plan

P = sharding.PartitionSpec


def _make_config(
    *,
    ep_axis: str | None = "ep",
    tp_axis: str | None = "tp",
    use_sp: bool = False,
    reduce_output: bool = True,
    remote_scatter: bool = False,
    w1_scale_shape: tuple[int, ...] | None = None,
    w2_scale_shape: tuple[int, ...] | None = None,
    axis_names: tuple[str, ...] = ("ep", "tp"),
) -> moe_config.RoutingMoEConfig:
  mesh = sharding.AbstractMesh((2,) * len(axis_names), axis_names)
  resolved = moe_plan.ResolvedMoEPlan.create(
      moe_plan.RoutingMoEPlan(remote_scatter=remote_scatter),
      x_dtype=jnp.bfloat16,
  )
  return moe_config.RoutingMoEConfig.create(
      tp_axis=tp_axis,
      ep_axis=ep_axis,
      use_sp=use_sp,
      reduce_output=reduce_output,
      activation="silu",
      plan=resolved,
      mesh=mesh,
      w1_scale_shape=w1_scale_shape,
      w2_scale_shape=w2_scale_shape,
  )


class RoutingMoEConfigTest(parameterized.TestCase):

  def test_weight_and_bias_pspecs(self):
    cfg = _make_config()
    self.assertEqual(cfg.w1_pspec, P("ep", None, "tp"))
    self.assertEqual(cfg.w2_pspec, P("ep", "tp", None))
    self.assertEqual(cfg.w1_bias_pspec, P("ep", None, "tp"))
    self.assertEqual(cfg.w2_bias_pspec, P("ep", None, None))
    self.assertEqual(cfg.ep_shard_idx_pspec, P("ep"))

  @parameterized.named_parameters(
      dict(
          testcase_name="unquantized",
          w1_scale_shape=None,
          w2_scale_shape=None,
          expected_w1=P("ep", None, None, "tp"),
          expected_w2=P("ep", None, None, None),
      ),
      dict(
          testcase_name="per_channel_4d",
          w1_scale_shape=(4, 1, 1, 256),
          w2_scale_shape=(4, 1, 1, 512),
          expected_w1=P("ep", None, None, "tp"),
          expected_w2=P("ep", None, None, None),
      ),
      dict(
          testcase_name="subchannel_4d",
          w1_scale_shape=(4, 2, 1, 256),
          w2_scale_shape=(4, 2, 1, 512),
          expected_w1=P("ep", None, None, "tp"),
          expected_w2=P("ep", "tp", None, None),
      ),
      dict(
          testcase_name="block_scaled_3d",
          w1_scale_shape=(4, 16, 256),
          w2_scale_shape=(4, 16, 512),
          expected_w1=P("ep", None, "tp"),
          expected_w2=P("ep", "tp", None),
      ),
  )
  def test_scale_pspecs(
      self,
      *,
      w1_scale_shape: tuple[int, ...] | None,
      w2_scale_shape: tuple[int, ...] | None,
      expected_w1: sharding.PartitionSpec,
      expected_w2: sharding.PartitionSpec,
  ):
    cfg = _make_config(
        w1_scale_shape=w1_scale_shape, w2_scale_shape=w2_scale_shape
    )
    self.assertEqual(cfg.w1_scale_pspec, expected_w1)
    self.assertEqual(cfg.w2_scale_pspec, expected_w2)

  @parameterized.named_parameters(
      dict(
          testcase_name="bad_w1_scale_rank",
          w1_scale_shape=(4, 256),
          w2_scale_shape=None,
          expected_regex="Expected w1_scale to have rank 3 or 4",
      ),
      dict(
          testcase_name="bad_w2_scale_rank",
          w1_scale_shape=None,
          w2_scale_shape=(4, 1, 1, 1, 512),
          expected_regex="Expected w2_scale to have rank 3 or 4",
      ),
  )
  def test_invalid_scale_rank_raises_value_error(
      self,
      *,
      w1_scale_shape: tuple[int, ...] | None,
      w2_scale_shape: tuple[int, ...] | None,
      expected_regex: str,
  ):
    with self.assertRaisesRegex(ValueError, expected_regex):
      _make_config(
          w1_scale_shape=w1_scale_shape, w2_scale_shape=w2_scale_shape
      )

  def test_output_collective_modes(self):
    self.assertEqual(
        _make_config(use_sp=False, reduce_output=True).output_collective,
        moe_config.OutputCollective.ALL_REDUCE,
    )
    self.assertEqual(
        _make_config(use_sp=True, reduce_output=True).output_collective,
        moe_config.OutputCollective.REDUCE_SCATTER,
    )
    self.assertEqual(
        _make_config(use_sp=False, reduce_output=False).output_collective,
        moe_config.OutputCollective.NONE,
    )
    self.assertEqual(
        _make_config(
            tp_axis=None, use_sp=False, remote_scatter=True
        ).output_collective,
        moe_config.OutputCollective.ALL_GATHER,
    )
    self.assertEqual(
        _make_config(
            tp_axis=None, use_sp=True, remote_scatter=True
        ).output_collective,
        moe_config.OutputCollective.NONE,
    )

  def test_tp_before_ep_in_hybrid_mesh_raises_not_implemented(self):
    with self.assertRaisesRegex(
        NotImplementedError, "ep_axis comes before tp_axis"
    ):
      _make_config(axis_names=("tp", "ep"))


if __name__ == "__main__":
  absltest.main()
