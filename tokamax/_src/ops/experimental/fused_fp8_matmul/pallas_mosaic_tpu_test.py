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
"""Tests for the Pallas/Mosaic TPU fused FP8 matmul."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import backend
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_fp8_matmul import base
from tokamax._src.ops.experimental.fused_fp8_matmul import pallas_mosaic_tpu
from tokamax._src.ops.experimental.fused_fp8_matmul import pallas_mosaic_tpu_kernel as kernel
from tokamax._src.ops.experimental.fused_fp8_matmul import test_base

jax.config.parse_flags_with_absl()


def _skip_unless_fp8_tpu(test: absltest.TestCase):
  op = pallas_mosaic_tpu.PallasMosaicTpuFusedFp8Matmul()
  device = backend.get_default_device()
  if not op.supported_on(device):
    test.skipTest(f"FP8 matmul is not supported on {device.device_kind}.")


class PallasMosaicTpuFusedFp8MatmulTest(test_base.FusedFp8MatmulTestBase):

  def __init__(self, *args):
    super().__init__(
        *args, matmul_fn=pallas_mosaic_tpu.PallasMosaicTpuFusedFp8Matmul()
    )

  def setUp(self):
    super().setUp()
    _skip_unless_fp8_tpu(self)

  @parameterized.parameters(
      # (m, k, n, block_m, block_k, block_n)
      (1024, 3072, 4096, 512, 3072, 4096),  # single k block, bn == n
      (1024, 3072, 4096, 256, 1024, 1024),  # multi-k with an n grid
      (1024, 3072, 4096, 1024, 3072, 512),  # bm == m
  )
  def test_explicit_config(self, m, k, n, block_m, block_k, block_n):
    config = pallas_mosaic_tpu.Config(
        block_m=block_m, block_k=block_k, block_n=block_n
    )
    op = pallas_mosaic_tpu.PallasMosaicTpuFusedFp8Matmul(config=config)
    lhs, rhs = test_base.random_inputs(m, k, n, jnp.bfloat16)
    actual = jax.jit(op)(lhs, rhs)
    expected = base.fused_fp8_matmul_reference(lhs, rhs)[0]
    if block_k == k:
      tol = test_base.SAME_QUANTIZATION_TOL
    else:
      tol = test_base.FP8_NOISE_TOL
    self.assertLess(test_base.relative_error(actual, expected), tol)

  def test_grad_with_multi_k_config(self):
    # With more than one k block the kernel cannot emit the per-row residuals,
    # so the op recomputes them in XLA; the gradient must still match.
    m, k, n = 512, 2048, 1024
    config = pallas_mosaic_tpu.Config(block_m=256, block_k=512, block_n=1024)
    op = pallas_mosaic_tpu.PallasMosaicTpuFusedFp8Matmul(config=config)
    lhs, rhs = test_base.random_inputs(m, k, n, jnp.bfloat16)
    ref = base.FusedFp8Matmul()

    def loss(a, b):
      return jnp.sum(op(a, b).astype(jnp.float32))

    def loss_ref(a, b):
      return jnp.sum(ref(a, b).astype(jnp.float32))

    dlhs, drhs = jax.jit(jax.grad(loss, (0, 1)))(lhs, rhs)
    dlhs_ref, drhs_ref = jax.jit(jax.grad(loss_ref, (0, 1)))(lhs, rhs)
    self.assert_close_to_reference(dlhs, dlhs_ref, k=k)
    self.assert_close_to_reference(drhs, drhs_ref, k=k)


class HeuristicsTest(parameterized.TestCase):
  """Tile heuristics are pure Python and run anywhere."""

  @parameterized.parameters(*test_base.SHAPES, (32768, 3072, 4096))
  def test_heuristics_config_divides_shape_and_fits(self, m, k, n):
    config = pallas_mosaic_tpu.get_heuristics_config(m, k, n)
    self.assertEqual(m % config.block_m, 0)
    self.assertEqual(k % config.block_k, 0)
    self.assertEqual(n % config.block_n, 0)
    self.assertLessEqual(
        kernel.forward_vmem_bytes(
            config.block_m, config.block_k, config.block_n, k, n
        ),
        kernel.VMEM_BUDGET_BYTES,
    )

  def test_heuristics_prefer_single_k_and_full_n(self):
    # The production shape: the measured best tiling is bk == k and bn == n.
    config = pallas_mosaic_tpu.get_heuristics_config(32768, 3072, 4096)
    self.assertEqual(config.block_k, 3072)
    self.assertEqual(config.block_n, 4096)

  def test_autotuning_configs_are_valid(self):
    op = pallas_mosaic_tpu.PallasMosaicTpuFusedFp8Matmul()
    m, k, n = 2048, 3072, 4096
    lhs = jax.ShapeDtypeStruct((m, k), jnp.bfloat16)
    rhs = jax.ShapeDtypeStruct((k, n), jnp.bfloat16)
    ba = op.bind(lhs, rhs)
    configs = op._get_autotuning_configs(ba)  # pylint: disable=protected-access
    self.assertNotEmpty(configs)
    self.assertIn(pallas_mosaic_tpu.get_heuristics_config(m, k, n), configs)
    for c in configs:
      self.assertEqual((m % c.block_m, k % c.block_k, n % c.block_n), (0, 0, 0))

  def test_config_rejects_unaligned_blocks(self):
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      pallas_mosaic_tpu.Config(block_m=100, block_k=256, block_n=256)


if __name__ == "__main__":
  absltest.main()
