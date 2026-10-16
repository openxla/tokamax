# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops import op
from tokamax._src.ops.linear_softmax_cross_entropy_loss import base
from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_tpu
from tokamax._src.ops.linear_softmax_cross_entropy_loss import test_utils

generate_random_data = test_utils.generate_random_data
PallasMosaicTpuLinearSoftmaxCrossEntropyLoss = (
    pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLoss
)
LinearSoftmaxCrossEntropyLoss = base.LinearSoftmaxCrossEntropyLoss


def setUpModule() -> None:
  # Initialize the default backend so that every test shard touches the
  # accelerator, including shards that only run the host-side
  # TpuLaneAndSublaneAlignmentTest cases.
  jax.devices()


class FlashLcePallasMosaicTpuTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  @parameterized.named_parameters(
      dict(
          testcase_name="op_wrapper_fwd_bwd_small_size_sum_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=2048,
          reduction="sum",
      ),
      dict(
          testcase_name="op_wrapper_fwd_bwd_medium_size_sum_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="sum",
      ),
      dict(
          testcase_name="op_wrapper_fwd_bwd_large_size_sum_reduction_test",
          b_dim=16384,
          h_dim=4096,
          v_dim=16384,
          reduction="sum",
      ),
      dict(
          testcase_name="op_wrapper_fwd_bwd_small_size_mean_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=2048,
          reduction="mean",
      ),
      dict(
          testcase_name="op_wrapper_fwd_bwd_medium_size_mean_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="mean",
      ),
      dict(
          testcase_name="op_wrapper_fwd_bwd_large_size_mean_reduction_test",
          b_dim=16384,
          h_dim=4096,
          v_dim=16384,
          reduction="mean",
      ),
  )
  def test_op_wrapper_fwd_bwd(self, b_dim, h_dim, v_dim, reduction):
    x, labels, w = generate_random_data(jax.random.key(42), b_dim, h_dim, v_dim)

    fn_pallas = PallasMosaicTpuLinearSoftmaxCrossEntropyLoss()
    fn_reference = LinearSoftmaxCrossEntropyLoss()

    kernel_loss, (kernel_grad_x, kernel_grad_w) = jax.value_and_grad(
        fn_pallas, argnums=(0, 2)
    )(x, labels, w, reduction=reduction)
    ref_loss, (ref_grad_x, ref_grad_w) = jax.value_and_grad(
        fn_reference, argnums=(0, 2)
    )(x, labels, w, reduction=reduction)

    self.assertTrue(jnp.allclose(ref_loss, kernel_loss, atol=3e-2, rtol=3e-2))
    self.assertTrue(
        jnp.allclose(ref_grad_x, kernel_grad_x, atol=3e-2, rtol=3e-2)
    )
    self.assertTrue(
        jnp.allclose(ref_grad_w, kernel_grad_w, atol=3e-2, rtol=3e-2)
    )


def _mock_tpu_info(**overrides: int) -> pallas_mosaic_tpu.pltpu.TpuInfo:
  """Returns a hermetic TpuInfo based on TPU v6e with the given overrides."""
  base_info = pallas_mosaic_tpu.pltpu.get_tpu_info_for_chip(
      pallas_mosaic_tpu.pltpu.ChipVersion.TPU_V6E, 1
  )
  overrides.setdefault("vmem_capacity_bytes", 128 * 1024 * 1024)
  return dataclasses.replace(base_info, **overrides)


class TpuLaneAndSublaneAlignmentTest(parameterized.TestCase):
  """Tests op-level config generation on TPUs with non-default lane counts."""

  def _bind_ops(self, b_dim: int, h_dim: int, v_dim: int) -> tuple[
      PallasMosaicTpuLinearSoftmaxCrossEntropyLoss,
      op.BoundArguments,
      pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp,
      op.BoundArguments,
  ]:
    x = jax.ShapeDtypeStruct((b_dim, h_dim), jnp.float32)
    labels = jax.ShapeDtypeStruct((b_dim,), jnp.int32)
    w = jax.ShapeDtypeStruct((h_dim, v_dim), jnp.float32)
    lse = jax.ShapeDtypeStruct((b_dim,), jnp.float32)
    scalar = jax.ShapeDtypeStruct((), jnp.float32)

    fwd_op = PallasMosaicTpuLinearSoftmaxCrossEntropyLoss()
    bwd_op = pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp()
    fwd_ba = fwd_op.bind(x, labels, w)
    bwd_ba = bwd_op.bind((lse,), scalar, scalar, x, labels, w)
    return fwd_op, fwd_ba, bwd_op, bwd_ba

  def _assert_lane_aligned(
      self,
      cfg: pallas_mosaic_tpu.Config,
      tpu_info: pallas_mosaic_tpu.pltpu.TpuInfo,
      name: str,
  ) -> None:
    num_lanes = tpu_info.num_lanes
    self.assertGreaterEqual(
        cfg.b_block_size, tpu_info.num_sublanes * num_lanes, f"{name}: {cfg}"
    )
    for field, value in (
        ("b_block_size", cfg.b_block_size),
        ("h_block_size", cfg.h_block_size),
        ("v_block_size", cfg.v_block_size),
    ):
      self.assertGreater(value, 0, f"{name} {field}: {cfg}")
      self.assertEqual(
          value % num_lanes,
          0,
          f"{name} {field} {value} not multiple of {num_lanes}",
      )

  @parameterized.named_parameters(
      dict(testcase_name="normal", b_dim=8192, h_dim=512, v_dim=2048),
      dict(testcase_name="small", b_dim=512, h_dim=128, v_dim=128),
      dict(testcase_name="small_h_only", b_dim=8192, h_dim=128, v_dim=2048),
  )
  def test_autotuning_configs_respect_custom_tpu_lane_count(
      self, b_dim: int, h_dim: int, v_dim: int
  ) -> None:
    """Autotuning configs are non-empty and lane-aligned for 256 lanes."""
    mock_tpu_info = _mock_tpu_info(num_lanes=256, num_sublanes=16)
    fwd_op, fwd_ba, bwd_op, bwd_ba = self._bind_ops(b_dim, h_dim, v_dim)

    with mock.patch.object(
        pallas_mosaic_tpu.pltpu, "get_tpu_info", return_value=mock_tpu_info
    ):
      b_range, h_range, v_range = (
          pallas_mosaic_tpu._get_autotuning_block_size_ranges(
              b_dim, h_dim, v_dim
          )
      )
      fwd_configs = fwd_op._get_autotuning_configs(fwd_ba)
      bwd_configs = bwd_op._get_autotuning_configs(bwd_ba)

    self.assertNotEmpty(b_range)
    self.assertNotEmpty(h_range)
    self.assertNotEmpty(v_range)
    self.assertNotEmpty(fwd_configs)
    self.assertNotEmpty(bwd_configs)
    min_b_block_size = mock_tpu_info.num_sublanes * mock_tpu_info.num_lanes
    if b_dim > min_b_block_size and v_dim > mock_tpu_info.num_lanes:
      self.assertGreater(len(fwd_configs), 1)
      self.assertGreater(len(bwd_configs), 1)
    for cfg in fwd_configs:
      self._assert_lane_aligned(cfg, mock_tpu_info, "fwd autotuning")
    for cfg in bwd_configs:
      self._assert_lane_aligned(cfg, mock_tpu_info, "bwd autotuning")

  @parameterized.named_parameters(
      dict(testcase_name="normal", b_dim=8192, h_dim=512, v_dim=2048),
      dict(testcase_name="small", b_dim=512, h_dim=128, v_dim=128),
  )
  def test_heuristics_config_respects_custom_tpu_lane_count(
      self, b_dim: int, h_dim: int, v_dim: int
  ) -> None:
    """Op-level heuristic configs are lane-aligned for 256 lanes."""
    mock_tpu_info = _mock_tpu_info(num_lanes=256, num_sublanes=16)
    fwd_op, fwd_ba, bwd_op, bwd_ba = self._bind_ops(b_dim, h_dim, v_dim)

    with mock.patch.object(
        pallas_mosaic_tpu.pltpu, "get_tpu_info", return_value=mock_tpu_info
    ):
      fwd_cfg = fwd_op._get_heuristics_config(fwd_ba)
      bwd_cfg = bwd_op._get_heuristics_config(bwd_ba)

    self._assert_lane_aligned(fwd_cfg, mock_tpu_info, "fwd heuristic")
    self._assert_lane_aligned(bwd_cfg, mock_tpu_info, "bwd heuristic")


if __name__ == "__main__":
  absltest.main()
