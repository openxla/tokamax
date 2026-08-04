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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src import numerics
import tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_tpu as pallas_mosaic_tpu
import tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_tpu_kernel as kernel
import tokamax._src.ops.linear_softmax_cross_entropy_loss.reference as reference


class FlashLcePallasMosaicTpuKernelTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  @parameterized.named_parameters(
      dict(
          testcase_name="fwd_small_size_sum_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=2048,
          reduction="sum",
      ),
      dict(
          testcase_name="fwd_medium_size_sum_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="sum",
      ),
      dict(
          testcase_name="fwd_large_size_sum_reduction_test",
          b_dim=16384,
          h_dim=4096,
          v_dim=16384,
          reduction="sum",
      ),
      dict(
          testcase_name="fwd_small_size_mean_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=2048,
          reduction="sum",
      ),
      dict(
          testcase_name="fwd_medium_size_mean_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="mean",
      ),
      dict(
          testcase_name="fwd_large_size_mean_reduction_test",
          b_dim=16384,
          h_dim=4096,
          v_dim=16384,
          reduction="mean",
      ),
      dict(
          testcase_name="fwd_small_size_none_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=2048,
          reduction="none",
      ),
      dict(
          testcase_name="fwd_medium_size_none_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="none",
      ),
      dict(
          testcase_name="fwd_large_size_none_reduction_test",
          b_dim=16384,
          h_dim=4096,
          v_dim=16384,
          reduction="none",
      ),
      dict(
          testcase_name="fwd_v_non_aligned_block_size_sum_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=2560,
          reduction="sum",
      ),
      dict(
          testcase_name="fwd_v_non_aligned_block_size_mean_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=2560,
          reduction="mean",
      ),
      dict(
          testcase_name="fwd_v_non_aligned_block_size_none_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=2560,
          reduction="none",
      ),
      dict(
          testcase_name="fwd_v_non_aligned_multiple_of_128_sum_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=2664,
          reduction="sum",
      ),
      dict(
          testcase_name="fwd_v_non_aligned_multiple_of_128_mean_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=2664,
          reduction="mean",
      ),
      dict(
          testcase_name="fwd_v_non_aligned_multiple_of_128_none_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=2664,
          reduction="none",
      ),
      dict(
          testcase_name="fwd_h_non_aligned_block_size_sum_reduction_test",
          b_dim=4096,
          h_dim=1152,
          v_dim=2048,
          reduction="sum",
      ),
      dict(
          testcase_name="fwd_h_non_aligned_block_size_mean_reduction_test",
          b_dim=4096,
          h_dim=1152,
          v_dim=2048,
          reduction="mean",
      ),
      dict(
          testcase_name="fwd_h_non_aligned_block_size_none_reduction_test",
          b_dim=4096,
          h_dim=1152,
          v_dim=2048,
          reduction="none",
      ),
      dict(
          testcase_name="fwd_h_non_aligned_multiple_of_128_sum_reduction_test",
          b_dim=4096,
          h_dim=1288,
          v_dim=2048,
          reduction="sum",
      ),
      dict(
          testcase_name="fwd_h_non_aligned_multiple_of_128_mean_reduction_test",
          b_dim=4096,
          h_dim=1288,
          v_dim=2048,
          reduction="mean",
      ),
      dict(
          testcase_name="fwd_h_non_aligned_multiple_of_128_none_reduction_test",
          b_dim=4096,
          h_dim=1288,
          v_dim=2048,
          reduction="none",
      ),
      dict(
          testcase_name="fwd_bfloat16_sum_reduction_test",
          b_dim=4096,
          h_dim=512,
          v_dim=2048,
          reduction="sum",
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="fwd_float16_mean_reduction_test",
          b_dim=4096,
          h_dim=512,
          v_dim=2048,
          reduction="mean",
          dtype=jnp.float16,
      ),
      dict(
          testcase_name="fwd_float16_none_reduction_test",
          b_dim=4096,
          h_dim=512,
          v_dim=2048,
          reduction="none",
          dtype=jnp.float16,
      ),
  )
  def test_kernel_forward_matches_reference(
      self, b_dim, h_dim, v_dim, reduction, dtype=jnp.float32
  ):
    x_shape = jax.ShapeDtypeStruct((b_dim, h_dim), dtype)
    labels_shape = numerics.RangedArrayInitializer(
        (b_dim,), jnp.int32, 0, v_dim
    )
    w_shape = jax.ShapeDtypeStruct((h_dim, v_dim), dtype)
    x, labels, w = numerics.random_initialize(
        (x_shape, labels_shape, w_shape), seed=42
    )
    config = kernel.get_heuristic_fwd_config(b_dim, h_dim, v_dim)

    ref_loss, ref_lse = (
        reference.linear_softmax_cross_entropy_loss_fwd_reference(
            x, labels, w, reduction=reduction
        )
    )
    kernel_loss, kernel_lse = (
        kernel.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
            x,
            labels,
            w,
            reduction=reduction,
            b_block_size=config.b_block_size,
            h_block_size=config.h_block_size,
            v_block_size=config.v_block_size,
        )
    )

    atol = 1e-4 if dtype == jnp.float32 else 5e-2
    rtol = 1e-4 if dtype == jnp.float32 else 5e-2
    self.assertTrue(jnp.allclose(ref_loss, kernel_loss, atol=atol, rtol=rtol))
    self.assertTrue(jnp.allclose(ref_lse, kernel_lse, atol=atol, rtol=rtol))

  @parameterized.named_parameters(
      dict(
          testcase_name="bwd_small_size_sum_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=2048,
          reduction="sum",
      ),
      dict(
          testcase_name="bwd_medium_size_sum_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="sum",
      ),
      dict(
          testcase_name="bwd_large_size_sum_reduction_test",
          b_dim=16384,
          h_dim=4096,
          v_dim=16384,
          reduction="sum",
      ),
      dict(
          testcase_name="bwd_small_size_mean_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=2048,
          reduction="mean",
      ),
      dict(
          testcase_name="bwd_medium_size_mean_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="mean",
      ),
      dict(
          testcase_name="bwd_large_size_mean_reduction_test",
          b_dim=16384,
          h_dim=4096,
          v_dim=16384,
          reduction="mean",
      ),
      dict(
          testcase_name="bwd_v_non_aligned_block_size_sum_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=2560,
          reduction="sum",
      ),
      dict(
          testcase_name="bwd_v_non_aligned_block_size_mean_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=2560,
          reduction="mean",
      ),
      dict(
          testcase_name="bwd_v_non_aligned_multiple_of_128_sum_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=2664,
          reduction="sum",
      ),
      dict(
          testcase_name="bwd_v_non_aligned_multiple_of_128_mean_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=2664,
          reduction="mean",
      ),
      dict(
          testcase_name="bwd_h_non_aligned_block_size_sum_reduction_test",
          b_dim=4096,
          h_dim=1152,
          v_dim=2048,
          reduction="sum",
      ),
      dict(
          testcase_name="bwd_h_non_aligned_block_size_mean_reduction_test",
          b_dim=4096,
          h_dim=1152,
          v_dim=2048,
          reduction="mean",
      ),
      dict(
          testcase_name="bwd_h_non_aligned_multiple_of_128_sum_reduction_test",
          b_dim=4096,
          h_dim=1288,
          v_dim=2048,
          reduction="sum",
      ),
      dict(
          testcase_name="bwd_h_non_aligned_multiple_of_128_mean_reduction_test",
          b_dim=4096,
          h_dim=1288,
          v_dim=2048,
          reduction="mean",
      ),
  )
  def test_kernel_bwd_matches_reference(self, b_dim, h_dim, v_dim, reduction):
    config = kernel.get_heuristic_bwd_config(b_dim, h_dim, v_dim)
    x_shape = jax.ShapeDtypeStruct((b_dim, h_dim), jnp.float32)
    labels_shape = numerics.RangedArrayInitializer(
        (b_dim,), jnp.int32, 0, v_dim
    )
    w_shape = jax.ShapeDtypeStruct((h_dim, v_dim), jnp.float32)
    x, labels, w = numerics.random_initialize(
        (x_shape, labels_shape, w_shape), seed=42
    )
    lse = jax.nn.logsumexp(x @ w, axis=-1)

    dout = 1.0
    kernel_grad_x, kernel_grad_w = (
        kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
            dout,
            lse,
            x,
            labels,
            w,
            reduction=reduction,
            b_block_size=config.b_block_size,
            h_block_size=config.h_block_size,
            v_block_size=config.v_block_size,
        )
    )

    ref_grad_x, ref_grad_w = (
        reference.linear_softmax_cross_entropy_loss_bwd_reference(
            dout, lse, x, labels, w, reduction=reduction
        )
    )

    self.assertTrue(
        jnp.allclose(ref_grad_x, kernel_grad_x, atol=3e-2, rtol=3e-2),
        f"ref_grad_x: {ref_grad_x}, kernel_grad_x: {kernel_grad_x}",
    )
    self.assertTrue(
        jnp.allclose(ref_grad_w, kernel_grad_w, atol=3e-2, rtol=3e-2),
        f"ref_grad_w: {ref_grad_w}, kernel_grad_w: {kernel_grad_w}",
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="b_dimension_not_multiple_of_b_block_size",
          b_dim=1500,
          h_dim=512,
          v_dim=1024,
      ),  # B dimension is not a multiple b_block_size
      dict(
          testcase_name="h_dimension_not_multiple_of_8",
          b_dim=1024,
          h_dim=513,
          v_dim=1024,
      ),  # H dimension is not a multiple of 8
  )
  def test_validation_errors(self, b_dim, h_dim, v_dim):
    config = kernel.get_heuristic_fwd_config(b_dim, h_dim, v_dim)
    x_shape = jax.ShapeDtypeStruct((b_dim, h_dim), jnp.float32)
    labels_shape = numerics.RangedArrayInitializer(
        (b_dim,), jnp.int32, 0, v_dim
    )
    w_shape = jax.ShapeDtypeStruct((h_dim, v_dim), jnp.float32)
    x, labels, w = numerics.random_initialize(
        (x_shape, labels_shape, w_shape), seed=42
    )
    lse = jax.nn.logsumexp(x @ w, axis=-1)

    with self.assertRaises(ValueError):
      kernel.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
          x,
          labels,
          w,
          b_block_size=config.b_block_size,
          h_block_size=config.h_block_size,
          v_block_size=config.v_block_size,
      )

    with self.assertRaises(ValueError):
      kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
          1.0,
          lse,
          x,
          labels,
          w,
          b_block_size=config.b_block_size,
          h_block_size=config.h_block_size,
          v_block_size=config.v_block_size,
      )


class HeuristicConfigTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  @parameterized.named_parameters(
      dict(
          testcase_name="vmem_16mb",
          b_dim=4096,
          h_dim=512,
          v_dim=32768,
          vmem_limit_bytes=16 * 1024 * 1024,
          expected_config=kernel.Config(
              b_block_size=1024, h_block_size=512, v_block_size=512
          ),
      ),
      dict(
          testcase_name="vmem_32mb",
          b_dim=4096,
          h_dim=512,
          v_dim=32768,
          vmem_limit_bytes=32 * 1024 * 1024,
          expected_config=kernel.Config(
              b_block_size=1024, h_block_size=512, v_block_size=2048
          ),
      ),
      dict(
          testcase_name="vmem_57mb",
          b_dim=4096,
          h_dim=512,
          v_dim=32768,
          vmem_limit_bytes=57 * 1024 * 1024,
          expected_config=kernel.Config(
              b_block_size=1024, h_block_size=512, v_block_size=4096
          ),
      ),
  )
  def test_get_heuristic_fwd_config(
      self,
      b_dim,
      h_dim,
      v_dim,
      vmem_limit_bytes,
      expected_config,
      dtype=jnp.float32,
  ):
    config = kernel.get_heuristic_fwd_config(
        b_dim=b_dim,
        h_dim=h_dim,
        v_dim=v_dim,
        dtype=dtype,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    self.assertEqual(config, expected_config)

    op_config = pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLoss.get_heuristic_fwd_config(
        b_dim=b_dim,
        h_dim=h_dim,
        v_dim=v_dim,
        dtype=dtype,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    self.assertEqual(op_config, expected_config)

    self.assertEqual(b_dim % config.b_block_size, 0)
    self.assertEqual(h_dim % config.h_block_size, 0)
    self.assertEqual(v_dim % config.v_block_size, 0)

    vmem_used = kernel._calculate_fwd_vmem_bytes(
        config.b_block_size,
        config.h_block_size,
        config.v_block_size,
        dtype=dtype,
    )
    self.assertLessEqual(vmem_used, vmem_limit_bytes)

  @parameterized.named_parameters(
      dict(
          testcase_name="vmem_16mb",
          b_dim=4096,
          h_dim=512,
          v_dim=32768,
          vmem_limit_bytes=16 * 1024 * 1024,
          expected_config=kernel.Config(
              b_block_size=1024, h_block_size=512, v_block_size=512
          ),
      ),
      dict(
          testcase_name="vmem_32mb",
          b_dim=4096,
          h_dim=512,
          v_dim=32768,
          vmem_limit_bytes=32 * 1024 * 1024,
          expected_config=kernel.Config(
              b_block_size=1024, h_block_size=512, v_block_size=1024
          ),
      ),
      dict(
          testcase_name="vmem_57mb",
          b_dim=4096,
          h_dim=512,
          v_dim=32768,
          vmem_limit_bytes=57 * 1024 * 1024,
          expected_config=kernel.Config(
              b_block_size=1024, h_block_size=512, v_block_size=2048
          ),
      ),
  )
  def test_get_heuristic_bwd_config(
      self,
      b_dim,
      h_dim,
      v_dim,
      vmem_limit_bytes,
      expected_config,
      dtype=jnp.float32,
  ):
    config = kernel.get_heuristic_bwd_config(
        b_dim=b_dim,
        h_dim=h_dim,
        v_dim=v_dim,
        dtype=dtype,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    self.assertEqual(config, expected_config)

    op_config = pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp.get_heuristic_bwd_config(
        b_dim=b_dim,
        h_dim=h_dim,
        v_dim=v_dim,
        dtype=dtype,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    self.assertEqual(op_config, expected_config)

    self.assertEqual(b_dim % config.b_block_size, 0)
    self.assertEqual(h_dim % config.h_block_size, 0)
    self.assertEqual(v_dim % config.v_block_size, 0)

    vmem_used = kernel._calculate_bwd_vmem_bytes(
        config.b_block_size,
        config.h_block_size,
        config.v_block_size,
        dtype=dtype,
    )
    self.assertLessEqual(vmem_used, vmem_limit_bytes)


if __name__ == "__main__":
  absltest.main()
