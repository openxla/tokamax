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
from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_tpu
from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_tpu_kernel as kernel
from tokamax._src.ops.linear_softmax_cross_entropy_loss import reference


class FlashLcePallasMosaicTpuKernelTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  def _assert_allclose(self, actual, expected, atol=1e-4, rtol=1e-4, name=""):
    """Asserts that two arrays are close, printing detailed diagnostic info on failure."""
    abs_err = jnp.abs(actual - expected)
    max_abs_err = float(jnp.max(abs_err))
    max_rel_err = float(jnp.max(abs_err / (jnp.abs(expected) + 1e-7)))
    mismatched = abs_err > (atol + rtol * jnp.abs(expected))
    mismatch_count = int(jnp.sum(mismatched))
    total_elements = actual.size

    diag = ""
    if mismatch_count > 0:
      if actual.ndim == 0:
        first_idx = ()
      else:
        first_idx = tuple(int(x[0]) for x in jnp.where(mismatched))
      diag = (
          f"\n[{name}] NUMERICAL MISMATCH:\n"
          f"  Shape: {actual.shape}, Dtype: {actual.dtype}\n"
          f"  Max Absolute Error: {max_abs_err:.6e}  (tolerance atol={atol})\n"
          f"  Max Relative Error: {max_rel_err:.6e}  (tolerance rtol={rtol})\n"
          f"  Mismatched Elements: {mismatch_count} / {total_elements} "
          f"({100.0 * mismatch_count / total_elements:.2f}%)\n"
          f"  First mismatch at index {first_idx}:\n"
          f"    Actual (Kernel):   {actual[first_idx]}\n"
          f"    Expected (Ref):    {expected[first_idx]}\n"
          f"    Absolute Diff:     {abs_err[first_idx]}\n"
      )

    self.assertEqual(
        mismatch_count,
        0,
        msg=diag
        or f"[{name}] arrays are not close within atol={atol}, rtol={rtol}",
    )

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
          reduction="mean",
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
          testcase_name="fwd_b_non_aligned_block_size_sum_reduction_test",
          b_dim=4352,
          h_dim=1024,
          v_dim=2048,
          reduction="sum",
      ),
      dict(
          testcase_name="fwd_b_non_aligned_block_size_mean_reduction_test",
          b_dim=4352,
          h_dim=1024,
          v_dim=2048,
          reduction="mean",
      ),
      dict(
          testcase_name="fwd_b_non_aligned_block_size_none_reduction_test",
          b_dim=4352,
          h_dim=1024,
          v_dim=2048,
          reduction="none",
      ),
      dict(
          testcase_name="fwd_b_non_aligned_multiple_of_128_sum_reduction_test",
          b_dim=5136,
          h_dim=1024,
          v_dim=2048,
          reduction="sum",
      ),
      dict(
          testcase_name="fwd_b_non_aligned_multiple_of_128_mean_reduction_test",
          b_dim=5136,
          h_dim=1024,
          v_dim=2048,
          reduction="mean",
      ),
      dict(
          testcase_name="fwd_b_non_aligned_multiple_of_128_none_reduction_test",
          b_dim=5136,
          h_dim=1024,
          v_dim=2048,
          reduction="none",
      ),
      dict(
          testcase_name="fwd_all_non_aligned_sum_reduction_test",
          b_dim=5136,
          h_dim=1288,
          v_dim=2664,
          reduction="sum",
      ),
      dict(
          testcase_name="fwd_all_non_aligned_mean_reduction_test",
          b_dim=5136,
          h_dim=1288,
          v_dim=2664,
          reduction="mean",
      ),
      dict(
          testcase_name="fwd_all_non_aligned_none_reduction_test",
          b_dim=5136,
          h_dim=1288,
          v_dim=2664,
          reduction="none",
      ),
      dict(
          testcase_name="fwd_all_non_aligned_large_sum_reduction_test",
          b_dim=3600,
          h_dim=1200,
          v_dim=10000,
          reduction="sum",
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
    self._assert_allclose(
        kernel_loss, ref_loss, atol=atol, rtol=rtol, name="loss"
    )
    self._assert_allclose(kernel_lse, ref_lse, atol=atol, rtol=rtol, name="lse")

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
          testcase_name="bwd_small_size_none_reduction_test",
          b_dim=1024,
          h_dim=512,
          v_dim=2048,
          reduction="none",
      ),
      dict(
          testcase_name="bwd_medium_size_none_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=4096,
          reduction="none",
      ),
      dict(
          testcase_name="bwd_large_size_none_reduction_test",
          b_dim=16384,
          h_dim=4096,
          v_dim=16384,
          reduction="none",
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
          testcase_name="bwd_v_non_aligned_block_size_none_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=2560,
          reduction="none",
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
          testcase_name="bwd_v_non_aligned_multiple_of_128_none_reduction_test",
          b_dim=4096,
          h_dim=1024,
          v_dim=2664,
          reduction="none",
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
          testcase_name="bwd_h_non_aligned_block_size_none_reduction_test",
          b_dim=4096,
          h_dim=1152,
          v_dim=2048,
          reduction="none",
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
      dict(
          testcase_name="bwd_h_non_aligned_multiple_of_128_none_reduction_test",
          b_dim=4096,
          h_dim=1288,
          v_dim=2048,
          reduction="none",
      ),
      dict(
          testcase_name="bwd_b_non_aligned_block_size_sum_reduction_test",
          b_dim=4352,
          h_dim=1024,
          v_dim=2048,
          reduction="sum",
      ),
      dict(
          testcase_name="bwd_b_non_aligned_block_size_mean_reduction_test",
          b_dim=4352,
          h_dim=1024,
          v_dim=2048,
          reduction="mean",
      ),
      dict(
          testcase_name="bwd_b_non_aligned_block_size_none_reduction_test",
          b_dim=4352,
          h_dim=1024,
          v_dim=2048,
          reduction="none",
      ),
      dict(
          testcase_name="bwd_b_non_aligned_multiple_of_128_sum_reduction_test",
          b_dim=5136,
          h_dim=1024,
          v_dim=2048,
          reduction="sum",
      ),
      dict(
          testcase_name="bwd_b_non_aligned_multiple_of_128_mean_reduction_test",
          b_dim=5136,
          h_dim=1024,
          v_dim=2048,
          reduction="mean",
      ),
      dict(
          testcase_name="bwd_b_non_aligned_multiple_of_128_none_reduction_test",
          b_dim=5136,
          h_dim=1024,
          v_dim=2048,
          reduction="none",
      ),
      dict(
          testcase_name="bwd_all_non_aligned_sum_reduction_test",
          b_dim=5136,
          h_dim=1288,
          v_dim=2664,
          reduction="sum",
      ),
      dict(
          testcase_name="bwd_all_non_aligned_mean_reduction_test",
          b_dim=5136,
          h_dim=1288,
          v_dim=2664,
          reduction="mean",
      ),
      dict(
          testcase_name="bwd_all_non_aligned_none_reduction_test",
          b_dim=5136,
          h_dim=1288,
          v_dim=2664,
          reduction="none",
      ),
      dict(
          testcase_name="bwd_all_non_aligned_large_sum_reduction_test",
          b_dim=3600,
          h_dim=1200,
          v_dim=10000,
          reduction="sum",
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

    dout_shape = (b_dim,) if reduction == "none" else ()
    dout = numerics.random_initialize(
        (jax.ShapeDtypeStruct(dout_shape, jnp.float32),), seed=42
    )[0]
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

    self._assert_allclose(
        kernel_grad_x, ref_grad_x, atol=5e-2, rtol=5e-2, name="grad_x"
    )
    self._assert_allclose(
        kernel_grad_w, ref_grad_w, atol=5e-2, rtol=5e-2, name="grad_w"
    )

  @parameterized.named_parameters(
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
              b_block_size=1024, h_block_size=512, v_block_size=256
          ),
      ),
      dict(
          testcase_name="vmem_32mb",
          b_dim=4096,
          h_dim=512,
          v_dim=32768,
          vmem_limit_bytes=32 * 1024 * 1024,
          expected_config=kernel.Config(
              b_block_size=1024, h_block_size=512, v_block_size=512
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


class CostEstimateTest(parameterized.TestCase):

  def test_fwd_cost_estimate(self):
    b, h, v = 1024, 512, 2048
    x = jax.ShapeDtypeStruct(shape=(b, h), dtype=jnp.bfloat16)
    labels = jax.ShapeDtypeStruct(shape=(b,), dtype=jnp.int32)
    w = jax.ShapeDtypeStruct(shape=(h, v), dtype=jnp.bfloat16)
    out_type = [
        jax.ShapeDtypeStruct(shape=(b,), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(b,), dtype=jnp.float32),
    ]

    cost = kernel.linear_softmax_cross_entropy_loss_fwd_cost_estimate(
        x=x, labels=labels, w=w, out_type=out_type
    )

    expected_matmul_flops = 2 * b * h * v
    expected_reduction_flops = 2 * b * v + b
    expected_flops = expected_matmul_flops + expected_reduction_flops
    expected_transcendentals = b * v + b
    expected_bytes = b * h * 2 + b * 4 + h * v * 2 + b * 4 + b * 4

    self.assertEqual(cost.flops, expected_flops)
    self.assertEqual(cost.transcendentals, expected_transcendentals)
    self.assertEqual(cost.bytes_accessed, expected_bytes)

  def test_bwd_cost_estimate(self):
    b, h, v = 1024, 512, 2048
    dout = jax.ShapeDtypeStruct(shape=(b,), dtype=jnp.float32)
    x = jax.ShapeDtypeStruct(shape=(b, h), dtype=jnp.bfloat16)
    labels = jax.ShapeDtypeStruct(shape=(b,), dtype=jnp.int32)
    w = jax.ShapeDtypeStruct(shape=(h, v), dtype=jnp.bfloat16)
    lse = jax.ShapeDtypeStruct(shape=(b,), dtype=jnp.float32)
    out_type = [
        jax.ShapeDtypeStruct(shape=(1, b, h), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(h, v), dtype=jnp.float32),
    ]

    cost = kernel.linear_softmax_cross_entropy_loss_bwd_cost_estimate(
        dout=dout, x=x, labels=labels, w=w, lse=lse, out_type=out_type
    )

    expected_matmul_flops = 3 * (2 * b * h * v)
    expected_softmax_flops = 3 * b * v
    expected_flops = expected_matmul_flops + expected_softmax_flops
    expected_transcendentals = b * v
    expected_bytes = (
        b * 4 + b * h * 2 + b * 4 + h * v * 2 + b * 4 + b * h * 4 + h * v * 4
    )

    self.assertEqual(cost.flops, expected_flops)
    self.assertEqual(cost.transcendentals, expected_transcendentals)
    self.assertEqual(cost.bytes_accessed, expected_bytes)


if __name__ == "__main__":
  absltest.main()
