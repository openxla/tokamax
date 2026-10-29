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
"""Tests for the Pallas/Mosaic TPU VJP of the fused FP8 matmul."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import backend
import jax.numpy as jnp
import qwix
from tokamax._src.ops.experimental.fused_fp8_matmul import base
from tokamax._src.ops.experimental.fused_fp8_matmul import pallas_mosaic_tpu
from tokamax._src.ops.experimental.fused_fp8_matmul import pallas_mosaic_tpu_vjp
from tokamax._src.ops.experimental.fused_fp8_matmul import test_base

jax.config.parse_flags_with_absl()

# The production shape the heuristics were tuned on.
_WI_1 = (32768, 3072, 4096)


def _grads(matmul_fn, lhs, rhs, dout):
  """Returns `(dlhs, drhs)` of `sum(matmul_fn(lhs, rhs) * dout)`."""

  def loss(a, b):
    return jnp.sum(matmul_fn(a, b).astype(jnp.float32) * dout)

  return jax.jit(jax.grad(loss, (0, 1)))(lhs, rhs)


class PallasMosaicTpuVjpTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    device = backend.get_default_device()
    if not pallas_mosaic_tpu.PallasMosaicTpuFusedFp8Matmul().supported_on(
        device
    ):
      self.skipTest(f"FP8 matmul is not supported on {device.device_kind}.")

  def test_pallas_op_installs_pallas_vjp(self):
    op = pallas_mosaic_tpu.PallasMosaicTpuFusedFp8Matmul()
    self.assertIsInstance(
        op.vjp, pallas_mosaic_tpu_vjp.PallasMosaicTpuFusedFp8MatmulVjp
    )

  @parameterized.parameters((1024, 3072, 4096), (512, 1024, 2048))
  def test_grad_matches_reference_op(self, m, k, n):
    lhs, rhs = test_base.random_inputs(m, k, n, jnp.bfloat16)
    dout = jax.random.normal(jax.random.PRNGKey(3), (m, n), jnp.float32)
    op = pallas_mosaic_tpu.PallasMosaicTpuFusedFp8Matmul()
    dlhs, drhs = _grads(op, lhs, rhs, dout)
    dlhs_ref, drhs_ref = _grads(base.FusedFp8Matmul(), lhs, rhs, dout)
    self.assertEqual((dlhs.shape, dlhs.dtype), (lhs.shape, lhs.dtype))
    self.assertEqual((drhs.shape, drhs.dtype), (rhs.shape, rhs.dtype))
    # Same numerics on both sides (the heuristics pick `block_n == n`), so
    # the gap is compiler rounding of the quantization only.
    tol = test_base.SAME_QUANTIZATION_TOL
    self.assertLess(test_base.relative_error(dlhs, dlhs_ref), tol)
    self.assertLess(test_base.relative_error(drhs, drhs_ref), tol)

  @parameterized.parameters(
      # (block_m, block_k, block_n) at (m, k, n) = (512, 2048, 2048)
      (256, 512, 2048),  # k blocks: cached cotangent
      (256, 2048, 512),  # n blocks: accumulator
      (512, 512, 512),  # both
  )
  def test_explicit_vjp_config(self, block_m, block_k, block_n):
    m, k, n = 512, 2048, 2048
    lhs, rhs = test_base.random_inputs(m, k, n, jnp.bfloat16)
    dout = jax.random.normal(jax.random.PRNGKey(3), (m, n), jnp.float32)
    config = pallas_mosaic_tpu_vjp.Config(
        block_m=block_m, block_k=block_k, block_n=block_n
    )
    vjp = pallas_mosaic_tpu_vjp.PallasMosaicTpuFusedFp8MatmulVjp(config=config)
    op = pallas_mosaic_tpu.PallasMosaicTpuFusedFp8Matmul(vjp=vjp)
    dlhs, _ = _grads(op, lhs, rhs, dout)
    dlhs_ref, _ = _grads(base.FusedFp8Matmul(), lhs, rhs, dout)
    # With `block_n < n` each n block is quantized against its own scale, so
    # the result only agrees with the reference up to FP8 noise.
    tol = (
        test_base.SAME_QUANTIZATION_TOL
        if block_n == n
        else test_base.FP8_NOISE_TOL
    )
    self.assertLess(test_base.relative_error(dlhs, dlhs_ref), tol)

  def test_grad_with_qarray_rhs(self):
    m, k, n = 512, 1024, 1024
    lhs, rhs = test_base.random_inputs(m, k, n, jnp.bfloat16)
    rhs_q = qwix.quantize(
        rhs, jnp.float8_e4m3fn, channelwise_axes=(1,), scale_dtype=jnp.float32
    )
    dout = jax.random.normal(jax.random.PRNGKey(3), (m, n), jnp.float32)
    op = pallas_mosaic_tpu.PallasMosaicTpuFusedFp8Matmul()
    dlhs, drhs = _grads(op, lhs, rhs_q, dout)
    dlhs_plain, _ = _grads(op, lhs, rhs, dout)
    # `rhs` was quantized by qwix in one case and by the op in the other.
    self.assertLess(
        test_base.relative_error(dlhs, dlhs_plain), test_base.FP8_NOISE_TOL
    )
    # A pre-quantized weight gets a zero gradient with the QArray's structure.
    self.assertEqual(drhs.qvalue.shape, rhs_q.qvalue.shape)
    self.assertEqual(float(jnp.abs(drhs.scale).max()), 0.0)


class HeuristicsTest(parameterized.TestCase):
  """Tile heuristics are pure Python and run anywhere."""

  @parameterized.parameters(*test_base.SHAPES, _WI_1)
  def test_heuristics_config_divides_shape_and_fits(self, m, k, n):
    config = pallas_mosaic_tpu_vjp.get_heuristics_config(m, k, n)
    self.assertEqual(m % config.block_m, 0)
    self.assertEqual(k % config.block_k, 0)
    self.assertEqual(n % config.block_n, 0)
    self.assertLessEqual(
        pallas_mosaic_tpu_vjp.dlhs_vmem_bytes(
            config.block_m, config.block_k, config.block_n, k, n
        ),
        pallas_mosaic_tpu_vjp.VMEM_BUDGET_BYTES,
    )

  def test_heuristics_pick_the_measured_production_tiling(self):
    # Full-k, full-n, and the largest m block that fits: the tiling behind the
    # number reported for this shape.
    config = pallas_mosaic_tpu_vjp.get_heuristics_config(*_WI_1)
    self.assertEqual(
        (config.block_m, config.block_k, config.block_n), (1024, 3072, 4096)
    )

  def test_heuristics_shrink_k_before_n_when_full_k_does_not_fit(self):
    # A `(k, n)` weight block of 64 MiB cannot fit at any `bm`, so `bk` must
    # shrink; `bn == n` is kept because it is the stronger preference.
    m, k, n = 2048, 4096, 16384
    config = pallas_mosaic_tpu_vjp.get_heuristics_config(m, k, n)
    self.assertEqual(config.block_n, n)
    self.assertLess(config.block_k, k)
    self.assertEqual((m % config.block_m, k % config.block_k), (0, 0))
    self.assertLessEqual(
        pallas_mosaic_tpu_vjp.dlhs_vmem_bytes(
            config.block_m, config.block_k, config.block_n, k, n
        ),
        pallas_mosaic_tpu_vjp.VMEM_BUDGET_BYTES,
    )

  def test_autotuning_configs_are_valid(self):
    m, k, n = 2048, 3072, 4096
    vjp = pallas_mosaic_tpu_vjp.PallasMosaicTpuFusedFp8MatmulVjp()
    lhs = jax.ShapeDtypeStruct((m, k), jnp.bfloat16)
    rhs = jax.ShapeDtypeStruct((k, n), jnp.bfloat16)
    dout = jax.ShapeDtypeStruct((m, n), jnp.bfloat16)
    ba = vjp.bind(None, dout, dout, lhs, rhs)
    configs = vjp._get_autotuning_configs(ba)  # pylint: disable=protected-access
    self.assertNotEmpty(configs)
    self.assertIn(pallas_mosaic_tpu_vjp.get_heuristics_config(m, k, n), configs)
    for c in configs:
      self.assertEqual((m % c.block_m, k % c.block_k, n % c.block_n), (0, 0, 0))
      self.assertLessEqual(
          pallas_mosaic_tpu_vjp.dlhs_vmem_bytes(
              c.block_m, c.block_k, c.block_n, k, n
          ),
          pallas_mosaic_tpu_vjp.VMEM_BUDGET_BYTES,
      )

  def test_config_rejects_unaligned_blocks(self):
    with self.assertRaisesRegex(ValueError, "multiple of 128"):
      pallas_mosaic_tpu_vjp.Config(block_m=100, block_k=256, block_n=256)


if __name__ == "__main__":
  absltest.main()
