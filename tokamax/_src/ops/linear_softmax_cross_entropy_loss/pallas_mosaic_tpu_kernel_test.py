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
import math
from typing import Any
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src import numerics
from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_tpu
from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_tpu_kernel as kernel
from tokamax._src.ops.linear_softmax_cross_entropy_loss import reference


def setUpModule() -> None:
  # Initialize the default backend so that every test shard touches the
  # accelerator, including shards that only run host-side heuristic test cases.
  jax.devices()


def _skip_if_partial_blocks_on_old_jax(
    test_case: parameterized.TestCase,
    *,
    b_dim: int,
    h_dim: int,
    v_dim: int,
    b_block_size: int,
    h_block_size: int,
    v_block_size: int,
) -> None:
  """Skips test on JAX < 0.11.1 if any dimension has a partial final block.

  Before jax-ml/jax@4700d0c (released in JAX 0.11.1), emit_pipeline did not
  round in-bounds BoundedSlice sizes up to the tiling, so the kernel's clamped
  partial final blocks produce non-tile-aligned DMAs on those versions.
  """
  if jax.__version_info__ < (0, 11, 1):
    if (
        b_dim % b_block_size != 0
        or h_dim % h_block_size != 0
        or v_dim % v_block_size != 0
    ):
      test_case.skipTest(
          "Partial final blocks fail in JAX 0.11.0; fixed in JAX 0.11.1."
      )


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
          # h_dim is a multiple of num_sublanes (8) but not of the bf16 sublane
          # tiling (16).
          testcase_name="fwd_bfloat16_h_not_multiple_of_sublane_tiling_test",
          b_dim=4096,
          h_dim=1288,
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
    if jax.__version_info__ < (0, 11, 1):
      self.skipTest("Test fails in JAX 0.11.0; fixed in JAX 0.11.1.")

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
      dict(
          testcase_name=(
              "bwd_all_non_aligned_b4096_h1288_v2664_sum_reduction_test"
          ),
          b_dim=4096,
          h_dim=1288,
          v_dim=2664,
          reduction="sum",
      ),
  )
  def test_kernel_bwd_matches_reference(self, b_dim, h_dim, v_dim, reduction):
    config = kernel.get_heuristic_bwd_config(b_dim, h_dim, v_dim)
    _skip_if_partial_blocks_on_old_jax(
        self,
        b_dim=b_dim,
        h_dim=h_dim,
        v_dim=v_dim,
        b_block_size=config.b_block_size,
        h_block_size=config.h_block_size,
        v_block_size=config.v_block_size,
    )
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

  def test_bwd_odd_num_v_blocks_multi_core(self):
    b_dim, h_dim, v_dim = 1024, 512, 2560
    b_block_size, h_block_size, v_block_size = 1024, 512, 1024
    _skip_if_partial_blocks_on_old_jax(
        self,
        b_dim=b_dim,
        h_dim=h_dim,
        v_dim=v_dim,
        b_block_size=b_block_size,
        h_block_size=h_block_size,
        v_block_size=v_block_size,
    )
    x_shape = jax.ShapeDtypeStruct((b_dim, h_dim), jnp.float32)
    labels_shape = numerics.RangedArrayInitializer(
        (b_dim,), jnp.int32, 0, v_dim
    )
    w_shape = jax.ShapeDtypeStruct((h_dim, v_dim), jnp.float32)
    x, labels, w = numerics.random_initialize(
        (x_shape, labels_shape, w_shape), seed=42
    )
    lse = jax.nn.logsumexp(x @ w, axis=-1)
    dout = numerics.random_initialize(
        (jax.ShapeDtypeStruct((), jnp.float32),), seed=42
    )[0]

    kernel_grad_x, kernel_grad_w = (
        kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
            dout,
            lse,
            x,
            labels,
            w,
            reduction="sum",
            b_block_size=b_block_size,
            h_block_size=h_block_size,
            v_block_size=v_block_size,
        )
    )
    self.assertEqual(kernel_grad_x.shape, (b_dim, h_dim))
    self.assertEqual(kernel_grad_w.shape, (h_dim, v_dim))

    ref_grad_x, ref_grad_w = (
        reference.linear_softmax_cross_entropy_loss_bwd_reference(
            dout, lse, x, labels, w, reduction="sum"
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
          testcase_name="b1024_h520_v2048",
          b_dim=1024,
          h_dim=520,
          v_dim=2048,
          b_block_size=1024,
          h_block_size=512,
          v_block_size=1024,
      ),
      dict(
          testcase_name="b1024_h640_v2048",
          b_dim=1024,
          h_dim=640,
          v_dim=2048,
          b_block_size=1024,
          h_block_size=512,
          v_block_size=1024,
      ),
      dict(
          testcase_name="b2048_h520_v2048",
          b_dim=2048,
          h_dim=520,
          v_dim=2048,
          b_block_size=1024,
          h_block_size=512,
          v_block_size=1024,
      ),
      dict(
          testcase_name="b1024_h1024_v2048",
          b_dim=1024,
          h_dim=1024,
          v_dim=2048,
          b_block_size=1024,
          h_block_size=512,
          v_block_size=1024,
      ),
      dict(
          testcase_name="b1024_h520_v2049",
          b_dim=1024,
          h_dim=520,
          v_dim=2049,
          b_block_size=1024,
          h_block_size=512,
          v_block_size=1024,
      ),
  )
  def test_kernel_bwd_explicit_block_sizes_matches_reference(
      self,
      b_dim: int,
      h_dim: int,
      v_dim: int,
      b_block_size: int,
      h_block_size: int,
      v_block_size: int,
  ):
    _skip_if_partial_blocks_on_old_jax(
        self,
        b_dim=b_dim,
        h_dim=h_dim,
        v_dim=v_dim,
        b_block_size=b_block_size,
        h_block_size=h_block_size,
        v_block_size=v_block_size,
    )
    x_shape = jax.ShapeDtypeStruct((b_dim, h_dim), jnp.float32)
    labels_shape = numerics.RangedArrayInitializer(
        (b_dim,), jnp.int32, 0, v_dim
    )
    w_shape = jax.ShapeDtypeStruct((h_dim, v_dim), jnp.float32)
    x, labels, w = numerics.random_initialize(
        (x_shape, labels_shape, w_shape), seed=42
    )
    lse = jax.nn.logsumexp(x @ w, axis=-1)
    dout = numerics.random_initialize(
        (jax.ShapeDtypeStruct((), jnp.float32),), seed=42
    )[0]

    ref_grad_x, ref_grad_w = (
        reference.linear_softmax_cross_entropy_loss_bwd_reference(
            dout, lse, x, labels, w, reduction="sum"
        )
    )

    for i in range(3):
      kernel_grad_x, kernel_grad_w = (
          kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
              dout,
              lse,
              x,
              labels,
              w,
              reduction="sum",
              b_block_size=b_block_size,
              h_block_size=h_block_size,
              v_block_size=v_block_size,
          )
      )
      self._assert_allclose(
          kernel_grad_x,
          ref_grad_x,
          atol=5e-2,
          rtol=5e-2,
          name=f"grad_x_run_{i}",
      )
      self._assert_allclose(
          kernel_grad_w,
          ref_grad_w,
          atol=5e-2,
          rtol=5e-2,
          name=f"grad_w_run_{i}",
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="both_b_and_h_revisits",
          b_dim=2048,
          h_dim=1024,
          v_dim=2048,
          b_block_size=1024,
          h_block_size=512,
          v_block_size=1024,
      ),
      dict(
          testcase_name="non_aligned_heuristic_blocks",
          b_dim=5136,
          h_dim=1288,
          v_dim=2664,
          b_block_size=None,
          h_block_size=None,
          v_block_size=None,
      ),
      dict(
          testcase_name="odd_num_v_blocks_multi_core",
          b_dim=1024,
          h_dim=512,
          v_dim=2560,
          b_block_size=1024,
          h_block_size=512,
          v_block_size=1024,
      ),
  )
  def test_kernel_bwd_single_buffered_matches_reference(
      self,
      b_dim: int,
      h_dim: int,
      v_dim: int,
      b_block_size: int | None,
      h_block_size: int | None,
      v_block_size: int | None,
  ) -> None:
    if b_block_size is None or h_block_size is None or v_block_size is None:
      config = kernel.get_heuristic_bwd_config(b_dim, h_dim, v_dim)
      b_block_size = config.b_block_size
      h_block_size = config.h_block_size
      v_block_size = config.v_block_size

    _skip_if_partial_blocks_on_old_jax(
        self,
        b_dim=b_dim,
        h_dim=h_dim,
        v_dim=v_dim,
        b_block_size=b_block_size,
        h_block_size=h_block_size,
        v_block_size=v_block_size,
    )

    x_shape = jax.ShapeDtypeStruct((b_dim, h_dim), jnp.float32)
    labels_shape = numerics.RangedArrayInitializer(
        (b_dim,), jnp.int32, 0, v_dim
    )
    w_shape = jax.ShapeDtypeStruct((h_dim, v_dim), jnp.float32)
    x, labels, w = numerics.random_initialize(
        (x_shape, labels_shape, w_shape), seed=42
    )
    lse = jax.nn.logsumexp(x @ w, axis=-1)
    dout = numerics.random_initialize(
        (jax.ShapeDtypeStruct((), jnp.float32),), seed=42
    )[0]

    ref_grad_x, ref_grad_w = (
        reference.linear_softmax_cross_entropy_loss_bwd_reference(
            dout, lse, x, labels, w, reduction="sum"
        )
    )

    kernel_grad_x, kernel_grad_w = (
        kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
            dout,
            lse,
            x,
            labels,
            w,
            reduction="sum",
            b_block_size=b_block_size,
            h_block_size=h_block_size,
            v_block_size=v_block_size,
            buffer_count=1,
        )
    )

    self._assert_allclose(
        kernel_grad_x, ref_grad_x, atol=5e-2, rtol=5e-2, name="grad_x"
    )
    self._assert_allclose(
        kernel_grad_w, ref_grad_w, atol=5e-2, rtol=5e-2, name="grad_w"
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
        dtype=dtype,  # pyrefly: ignore[bad-argument-type]
        vmem_limit_bytes=vmem_limit_bytes,
    )
    self.assertEqual(config, expected_config)

    op_config = pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLoss.get_heuristic_fwd_config(
        b_dim=b_dim,
        h_dim=h_dim,
        v_dim=v_dim,
        dtype=dtype,  # pyrefly: ignore[bad-argument-type]
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
        dtype=dtype,  # pyrefly: ignore[bad-argument-type]
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
              b_block_size=1024, h_block_size=512, v_block_size=128
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
              b_block_size=1024, h_block_size=512, v_block_size=1024
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
        dtype=dtype,  # pyrefly: ignore[bad-argument-type]
        vmem_limit_bytes=vmem_limit_bytes,
    )
    self.assertEqual(config, expected_config)

    op_config = pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp.get_heuristic_bwd_config(
        b_dim=b_dim,
        h_dim=h_dim,
        v_dim=v_dim,
        dtype=dtype,  # pyrefly: ignore[bad-argument-type]
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
        dtype=dtype,  # pyrefly: ignore[bad-argument-type]
    )
    self.assertLessEqual(vmem_used, vmem_limit_bytes)

  @parameterized.named_parameters(
      dict(testcase_name="h1024", h_dim=1024),
      dict(testcase_name="h640", h_dim=640),
      dict(testcase_name="h520", h_dim=520),
      dict(testcase_name="h768", h_dim=768),
      dict(testcase_name="h256", h_dim=256),
      dict(testcase_name="h200", h_dim=200),
      dict(testcase_name="h2048", h_dim=2048),
      dict(testcase_name="h4096", h_dim=4096),
  )
  def test_heuristic_bwd_never_two_h_blocks(self, h_dim: int):
    config = kernel.get_heuristic_bwd_config(
        b_dim=4096,
        h_dim=h_dim,
        v_dim=119548,
        dtype=jnp.dtype(jnp.float32),
    )
    num_h_blocks = math.ceil(h_dim / config.h_block_size)
    self.assertNotEqual(
        num_h_blocks,
        2,
        f"h_dim={h_dim} yielded exactly 2 H blocks ({config.h_block_size=}), "
        "triggering the slow single-buffered x_grad fallback.",
    )
    for vmem_mb in (16, 32):
      cfg_constrained = kernel.get_heuristic_bwd_config(
          b_dim=4096,
          h_dim=h_dim,
          v_dim=32768,
          dtype=jnp.dtype(jnp.float32),
          vmem_limit_bytes=vmem_mb * 1024 * 1024,
      )
      self.assertNotEqual(
          math.ceil(h_dim / cfg_constrained.h_block_size),
          2,
          f"h_dim={h_dim} at {vmem_mb}MB VMEM yielded exactly 2 H blocks.",
      )


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


class SafeInputOutputBufferCountTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="out_1_passthrough",
          requested=(2, 1),
          revisit_distance=2,
          expected=(2, 1),
      ),
      dict(
          testcase_name="none_revisit_passthrough",
          requested=(2, 2),
          revisit_distance=None,
          expected=(2, 2),
      ),
      dict(
          testcase_name="distance_1_passthrough",
          requested=(2, 2),
          revisit_distance=1,
          expected=(2, 2),
      ),
      dict(
          testcase_name="distance_2_fallback",
          requested=(2, 2),
          revisit_distance=2,
          expected=(2, 1),
      ),
      dict(
          testcase_name="distance_3_safe",
          requested=(2, 2),
          revisit_distance=3,
          expected=(2, 2),
      ),
      dict(
          testcase_name="in_1_out_2_distance_2_safe",
          requested=(1, 2),
          revisit_distance=2,
          expected=(1, 2),
      ),
      dict(
          testcase_name="in_3_out_2_distance_2_fallback",
          requested=(3, 2),
          revisit_distance=2,
          expected=(2, 1),
      ),
      dict(
          testcase_name="in_3_out_1_distance_2_fallback",
          requested=(3, 1),
          revisit_distance=2,
          expected=(2, 1),
      ),
      dict(
          testcase_name="int_input_safe",
          requested=2,
          revisit_distance=2,
          expected=2,
      ),
      dict(
          testcase_name="int_input_fallback",
          requested=3,
          revisit_distance=2,
          expected=(2, 1),
      ),
  )
  def test_safe_input_output_buffer_count(
      self,
      requested: tuple[int, int] | int,
      revisit_distance: int | None,
      expected: tuple[int, int] | int,
  ):
    actual = kernel._safe_input_output_buffer_count(requested, revisit_distance)
    self.assertEqual(actual, expected)


def _mock_tpu_info(**overrides: int) -> kernel.pltpu.TpuInfo:
  """Returns a hermetic TpuInfo based on TPU v6e with the given overrides.

  Uses `get_tpu_info_for_chip` rather than `get_tpu_info` so that tests do not
  depend on (or fail without) the physical device the test runs on.

  Args:
    **overrides: TpuInfo fields to override (e.g. `num_lanes=256`).
  """
  base_info = kernel.pltpu.get_tpu_info_for_chip(
      kernel.pltpu.ChipVersion.TPU_V6E, 1
  )
  overrides.setdefault("vmem_capacity_bytes", 128 * 1024 * 1024)
  return dataclasses.replace(base_info, **overrides)


class TpuLaneAndSublaneAlignmentTest(parameterized.TestCase):
  """Tests heuristics and validation on TPUs with non-default lane counts."""

  @parameterized.named_parameters(
      dict(
          testcase_name="default_mxu",
          b_dim=2048,
          h_dim=384,
          v_dim=384,
          tpu_overrides={},
          vmem_limit_bytes=None,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="raw_h_dim_max_below_num_lanes",
          b_dim=2048,
          h_dim=384,
          v_dim=384,
          tpu_overrides=dict(mxu_column_size=128, num_mxus=1),
          vmem_limit_bytes=None,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="raw_h_dim_max_not_multiple_of_num_lanes",
          b_dim=2048,
          h_dim=1008,
          v_dim=1000,
          tpu_overrides=dict(mxu_column_size=128, num_mxus=3),
          vmem_limit_bytes=None,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="h_dim_768",
          b_dim=2048,
          h_dim=768,
          v_dim=4096,
          tpu_overrides={},
          vmem_limit_bytes=None,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="two_h_block_avoidance",
          b_dim=2048,
          h_dim=1024,
          v_dim=4096,
          tpu_overrides={},
          vmem_limit_bytes=None,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="small_dims",
          b_dim=512,
          h_dim=128,
          v_dim=128,
          tpu_overrides={},
          vmem_limit_bytes=None,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="non_power_of_two_batch",
          b_dim=1280,
          h_dim=1280,
          v_dim=2816,
          tpu_overrides={},
          vmem_limit_bytes=None,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="tight_vmem_fallback",
          b_dim=2048,
          h_dim=1024,
          v_dim=4096,
          tpu_overrides={},
          vmem_limit_bytes=1024,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="num_lanes_384",
          b_dim=4608,
          h_dim=768,
          v_dim=1536,
          tpu_overrides=dict(num_lanes=384),
          vmem_limit_bytes=None,
          dtype=jnp.float32,
      ),
  )
  def test_heuristic_respects_custom_tpu_lane_count(
      self,
      b_dim: int,
      h_dim: int,
      v_dim: int,
      tpu_overrides: dict[str, int],
      vmem_limit_bytes: int | None,
      dtype: jnp.dtype,
  ) -> None:
    """Heuristic block sizes are positive, lane-aligned, and pass validation."""
    overrides = dict(num_lanes=256, num_sublanes=16)
    overrides.update(tpu_overrides)
    mock_tpu_info = _mock_tpu_info(**overrides)
    num_lanes = mock_tpu_info.num_lanes
    x = jnp.zeros((b_dim, h_dim), dtype=dtype)
    labels = jnp.zeros((b_dim,), dtype=jnp.int32)
    w = jnp.zeros((h_dim, v_dim), dtype=dtype)

    with mock.patch.object(
        kernel.pltpu, "get_tpu_info", return_value=mock_tpu_info
    ):
      fwd_cfg = kernel.get_heuristic_fwd_config(
          b_dim, h_dim, v_dim, dtype=dtype, vmem_limit_bytes=vmem_limit_bytes
      )
      bwd_cfg = kernel.get_heuristic_bwd_config(
          b_dim, h_dim, v_dim, dtype=dtype, vmem_limit_bytes=vmem_limit_bytes
      )
      for name, cfg in (("fwd", fwd_cfg), ("bwd", bwd_cfg)):
        with self.subTest(name):
          self.assertGreaterEqual(
              cfg.b_block_size, mock_tpu_info.num_sublanes * num_lanes
          )
          self.assertGreater(cfg.h_block_size, 0)
          self.assertGreater(cfg.v_block_size, 0)
          self.assertEqual(cfg.b_block_size % num_lanes, 0)
          self.assertEqual(cfg.h_block_size % num_lanes, 0)
          self.assertEqual(cfg.v_block_size % num_lanes, 0)
          # Must not raise.
          kernel.validate_inputs(
              x,
              labels,
              w,
              b_block_size=cfg.b_block_size,
              h_block_size=cfg.h_block_size,
              v_block_size=cfg.v_block_size,
          )

  def test_heuristic_avoids_two_h_blocks_in_bwd(self) -> None:
    """The 2-H-block bwd avoidance never yields 2 lane-aligned blocks."""
    mock_tpu_info = _mock_tpu_info(num_lanes=256, num_sublanes=16)
    with mock.patch.object(
        kernel.pltpu, "get_tpu_info", return_value=mock_tpu_info
    ):
      fwd_cfg = kernel.get_heuristic_fwd_config(
          2048, 1024, 4096, dtype=jnp.dtype(jnp.bfloat16)
      )
      bwd_cfg = kernel.get_heuristic_bwd_config(
          2048, 1024, 4096, dtype=jnp.dtype(jnp.bfloat16)
      )
    # Precondition: the forward heuristic picks 2 H blocks for this shape.
    self.assertEqual(math.ceil(1024 / fwd_cfg.h_block_size), 2)
    self.assertNotEqual(math.ceil(1024 / bwd_cfg.h_block_size), 2)
    self.assertEqual(bwd_cfg.h_block_size % 256, 0)

  def test_validate_inputs_enforces_lane_alignment(self) -> None:
    """validate_inputs enforces custom lane and sublane alignment."""
    mock_tpu_info = _mock_tpu_info(num_lanes=256, num_sublanes=16)
    x = jnp.zeros((1024, 256), dtype=jnp.float32)
    labels = jnp.zeros((1024,), dtype=jnp.int32)
    w = jnp.zeros((256, 1024), dtype=jnp.float32)

    with mock.patch.object(
        kernel.pltpu, "get_tpu_info", return_value=mock_tpu_info
    ):
      # 256-lane / 16-sublane aligned inputs should pass without raising,
      # including b_block_size < 1024.
      cfg_small_b = kernel.Config(
          b_block_size=256, h_block_size=256, v_block_size=256
      )
      kernel.validate_inputs(
          x,
          labels,
          w,
          b_block_size=cfg_small_b.b_block_size,
          h_block_size=cfg_small_b.h_block_size,
          v_block_size=cfg_small_b.v_block_size,
      )
      kernel.validate_inputs(
          x, labels, w, b_block_size=1280, h_block_size=512, v_block_size=768
      )
      # Non-positive or 128-aligned (not 256-aligned) should raise ValueError.
      for bad_b in (0, -256, 1152):
        with self.assertRaisesRegex(
            ValueError, r"b_block_size must be a positive multiple of 256"
        ):
          kernel.validate_inputs(
              x,
              labels,
              w,
              b_block_size=bad_b,
              h_block_size=256,
              v_block_size=256,
          )
      for bad_h in (0, -256, 128):
        with self.assertRaisesRegex(
            ValueError, r"h_block_size must be a positive multiple of 256"
        ):
          kernel.validate_inputs(
              x,
              labels,
              w,
              b_block_size=1024,
              h_block_size=bad_h,
              v_block_size=256,
          )
      for bad_v in (0, -256, 128):
        with self.assertRaisesRegex(
            ValueError, r"v_block_size must be a positive multiple of 256"
        ):
          kernel.validate_inputs(
              x,
              labels,
              w,
              b_block_size=1024,
              h_block_size=256,
              v_block_size=bad_v,
          )
      # 8-aligned but not 16-aligned sublane dimension should raise ValueError
      w_bad_h = jnp.zeros((248, 1024), dtype=jnp.float32)
      x_bad_h = jnp.zeros((1024, 248), dtype=jnp.float32)
      with self.assertRaisesRegex(
          ValueError, r"hidden dimension of w must be a multiple of 16, got 248"
      ):
        kernel.validate_inputs(
            x_bad_h,
            labels,
            w_bad_h,
            b_block_size=1024,
            h_block_size=256,
            v_block_size=256,
        )


class InputOutputBufferCountTest(parameterized.TestCase):

  def test_has_separate_input_output_buffering_flag_on_head(self) -> None:
    if jax.__version_info__ < (0, 12, 0):
      self.skipTest(
          "Separate input_output buffering is only expected on JAX >= 0.12.0."
      )
    self.assertTrue(
        kernel._HAS_SEPARATE_INPUT_OUTPUT_BUFFERING,
        msg=(
            "kernel._HAS_SEPARATE_INPUT_OUTPUT_BUFFERING is unexpectedly False"
            f" on JAX {jax.__version__}. Check if BufferedRef.in_buffer_count"
            " was renamed."
        ),
    )

  @parameterized.product(
      requested=[(2, 2), (3, 2), (2, 1), 2, 1],
      revisit_distance=[None, 1, 2, 8],
  )
  def test_fallback_returns_one_when_buffering_unsupported(
      self,
      requested: tuple[int, int] | int,
      revisit_distance: int | None,
  ) -> None:
    with mock.patch.object(
        kernel, "_HAS_SEPARATE_INPUT_OUTPUT_BUFFERING", False
    ):
      self.assertEqual(
          kernel._input_output_buffer_count(requested, revisit_distance), 1
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="tuple_2_2_none",
          requested=(2, 2),
          revisit_distance=None,
      ),
      dict(
          testcase_name="tuple_2_2_dist_1",
          requested=(2, 2),
          revisit_distance=1,
      ),
      dict(
          testcase_name="tuple_2_2_dist_2",
          requested=(2, 2),
          revisit_distance=2,
      ),
      dict(
          testcase_name="tuple_2_2_dist_8",
          requested=(2, 2),
          revisit_distance=8,
      ),
      dict(
          testcase_name="tuple_3_2_dist_2",
          requested=(3, 2),
          revisit_distance=2,
      ),
      dict(
          testcase_name="tuple_2_1_dist_2",
          requested=(2, 1),
          revisit_distance=2,
      ),
      dict(testcase_name="int_2_dist_2", requested=2, revisit_distance=2),
      dict(testcase_name="int_1_dist_2", requested=1, revisit_distance=2),
  )
  def test_delegates_to_safe_helper_when_buffering_supported(
      self,
      requested: tuple[int, int] | int,
      revisit_distance: int | None,
  ) -> None:
    with mock.patch.object(
        kernel, "_HAS_SEPARATE_INPUT_OUTPUT_BUFFERING", True
    ):
      expected = kernel._safe_input_output_buffer_count(
          requested, revisit_distance
      )
      self.assertEqual(
          kernel._input_output_buffer_count(requested, revisit_distance),
          expected,
      )


class AutotuningCandidateConfigsTest(parameterized.TestCase):
  """Tests that `_get_autotuning_configs` generates valid candidate search spaces.

  Note: This does NOT test or assert an autotuned "best" config (which is an
  empirical result that can vary across tuning runs, JAX/Pallas releases, or
  hardware platforms). Instead, it unit tests the candidate generation and
  pruning logic to ensure that every candidate configuration admitted into the
  autotuner search space satisfies hardware feasibility invariants:
    1. VMEM capacity: Candidate block sizes fit within hardware VMEM limits
       (calculated via `_calculate_{fwd,bwd}_vmem_bytes`), preventing compiler
       VMEM OOM crashes during fleet-scale autotuning sweeps.
    2. Multi-core partitioning: The number of vocabulary blocks divides evenly
       across cores when >= max_cores, avoiding idle cores or sequencer hangs.
    3. Pipeline safety (bwd): Excludes `ceil(h_dim / h_block_size) == 2` to
       avoid the synchronous `(in, 1)` output-buffer fallback in
       `emit_pipeline`.
  """

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  def test_get_autotuning_configs_fwd(self):
    op = pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLoss()
    x = jnp.zeros((4096, 4096), dtype=jnp.bfloat16)
    labels = jnp.zeros((4096,), dtype=jnp.int32)
    w = jnp.zeros((4096, 119548), dtype=jnp.bfloat16)
    ba = op.bind(x=x, labels=labels, w=w, reduction="sum")
    configs = op._get_autotuning_configs(ba)
    self.assertNotEmpty(configs)

    vmem_limit_bytes = kernel._get_vmem_limit_bytes()
    max_cores = pltpu.get_tpu_info().num_cores

    for config in configs:
      # 1. VMEM capacity check
      vmem_used = kernel._calculate_fwd_vmem_bytes(
          config.b_block_size,
          config.h_block_size,
          config.v_block_size,
          dtype=x.dtype,
      )
      self.assertLessEqual(vmem_used, vmem_limit_bytes)

      # 2. Multi-core partitioning compatibility check
      num_v_blocks = math.ceil(119548 / config.v_block_size)
      if max_cores > 1 and num_v_blocks >= max_cores:
        self.assertEqual(num_v_blocks % max_cores, 0)

  @parameterized.named_parameters(
      ("h_dim_512", 512),
      ("h_dim_1024", 1024),
      ("h_dim_4096", 4096),
  )
  def test_get_autotuning_configs_bwd(self, h_dim: int):
    op_vjp = pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp()
    x = jnp.zeros((4096, h_dim), dtype=jnp.bfloat16)
    labels = jnp.zeros((4096,), dtype=jnp.int32)
    w = jnp.zeros((h_dim, 119548), dtype=jnp.bfloat16)
    lse = jnp.zeros((4096,), dtype=jnp.float32)
    dout = jnp.zeros((), dtype=jnp.float32)
    out = jnp.zeros((), dtype=jnp.float32)
    ba = op_vjp.bind(
        residuals=(lse,),
        out=out,
        dout=dout,
        x=x,
        labels=labels,
        w=w,
        reduction="sum",
        return_residuals=False,
    )
    configs = op_vjp._get_autotuning_configs(ba)
    self.assertNotEmpty(configs)

    vmem_limit_bytes = kernel._get_vmem_limit_bytes()
    max_cores = pltpu.get_tpu_info().num_cores

    for config in configs:
      # 1. VMEM capacity check
      vmem_used = kernel._calculate_bwd_vmem_bytes(
          b_block_size=config.b_block_size,
          h_block_size=config.h_block_size,
          v_block_size=config.v_block_size,
          dtype=x.dtype,
          buffer_count=config.buffer_count,
      )
      self.assertLessEqual(vmem_used, vmem_limit_bytes)

      # 2. Multi-core partitioning compatibility check
      num_v_blocks = math.ceil(119548 / config.v_block_size)
      if max_cores > 1 and num_v_blocks >= max_cores:
        self.assertEqual(num_v_blocks % max_cores, 0)

      # 3. Synchronous (2, 1) output fallback avoidance check
      self.assertNotEqual(math.ceil(h_dim / config.h_block_size), 2)


_NON_ALIGNED_VOCAB_CONFIGS = (
    ("heuristic", (2, 2)),
    (5504, (2, 2)),
    (6144, (2, 2)),
    (4096, (2, 2)),
    (4096, (3, 3)),
)

_NON_ALIGNED_VOCAB_DTYPES = (
    ("bf16_bf16", jnp.bfloat16, jnp.bfloat16),
    ("f32_bf16", jnp.float32, jnp.bfloat16),
)

_NON_ALIGNED_VOCAB_SCENARIOS = (
    "random_normal",
    "padded_batch_realistic",
    "all_masked",
    "large_magnitude_logits",
    "nan_in_x",
)

_NON_ALIGNED_VOCAB_B4096_CONFIGS = (
    (
        "f32_f32_candidate_config_b2048_v2048_buf3_2",
        jnp.float32,
        jnp.float32,
        2048,
        256,
        2048,
        (3, 2),
    ),
    (
        "f32_f32_heuristic",
        jnp.float32,
        jnp.float32,
        None,
        None,
        "heuristic",
        (2, 2),
    ),
    (
        "f32_f32_b1024_v5504_buf2_2",
        jnp.float32,
        jnp.float32,
        1024,
        256,
        5504,
        (2, 2),
    ),
    (
        "f32_bf16_candidate_config_b1024_v5504_buf2_2",
        jnp.float32,
        jnp.bfloat16,
        1024,
        256,
        5504,
        (2, 2),
    ),
    (
        "bf16_bf16_candidate_config_b1024_v6144_buf2_2",
        jnp.bfloat16,
        jnp.bfloat16,
        1024,
        256,
        6144,
        (2, 2),
    ),
    (
        "bf16_f32_candidate_config_b2048_v2048_buf2_3",
        jnp.bfloat16,
        jnp.float32,
        2048,
        256,
        2048,
        (2, 3),
    ),
)


def _generate_non_aligned_vocab_test_cases() -> list[dict[str, Any]]:
  cases = []
  for dtype_name, x_dtype, w_dtype in _NON_ALIGNED_VOCAB_DTYPES:
    for v_block_size, buffer_count in _NON_ALIGNED_VOCAB_CONFIGS:
      name = (
          f"{dtype_name}_v_{v_block_size}_buf_"
          f"{buffer_count[0]}_{buffer_count[1]}"
      )
      cases.append(
          dict(
              testcase_name=name,
              x_dtype=x_dtype,
              w_dtype=w_dtype,
              v_block_size=v_block_size,
              buffer_count=buffer_count,
          )
      )
  return cases


def _generate_non_aligned_vocab_b4096_test_cases() -> list[dict[str, Any]]:
  cases = []
  for (
      name_suffix,
      x_dtype,
      w_dtype,
      b_block_size,
      h_block_size,
      v_block_size,
      buffer_count,
  ) in _NON_ALIGNED_VOCAB_B4096_CONFIGS:
    cases.append(
        dict(
            testcase_name=name_suffix,
            x_dtype=x_dtype,
            w_dtype=w_dtype,
            b_block_size=b_block_size,
            h_block_size=h_block_size,
            v_block_size=v_block_size,
            buffer_count=buffer_count,
        )
    )
  return cases


class NonAlignedVocabBwdTest(parameterized.TestCase):
  """Tests the backward kernel at large non-aligned vocabulary shapes.

  Uses V=119548, which is not divisible by the tested `v_block_size` values or
  by 128, so the final vocabulary block is partial and must be tail-masked.
  """

  def setUp(self) -> None:
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  def _assert_allclose(
      self,
      actual: jax.Array,
      expected: jax.Array,
      atol: float = 1e-4,
      rtol: float = 1e-4,
      name: str = "",
  ) -> None:
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
        first_idx: tuple[int, ...] = ()
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

  def _generate_inputs(
      self,
      scenario: str,
      x_dtype: jnp.dtype,
      w_dtype: jnp.dtype,
      rng_key: jax.Array,
      b: int = 1024,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    h, v = 256, 119548
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    x = jax.random.normal(k1, (b, h), dtype=x_dtype)
    w = jax.random.normal(k2, (h, v), dtype=w_dtype) / math.sqrt(h)
    labels = jax.random.randint(k3, (b,), minval=0, maxval=v, dtype=jnp.int32)
    dout = jax.random.normal(k4, (b,), dtype=jnp.float32)

    if scenario == "random_normal":
      pass
    elif scenario == "padded_batch_realistic":
      # ~50% masked rows simulating padded sequence batches:
      valid_mask = jnp.arange(b) % 2 == 0
      x = jnp.where(valid_mask[:, None], x, jnp.zeros_like(x))
      tail_labels = jnp.array([v - 1, v - 2, v - 100, v - 500], dtype=jnp.int32)
      valid_indices = jnp.array([0, 2, 4, 6], dtype=jnp.int32)
      labels = labels.at[valid_indices].set(tail_labels)
      labels = jnp.where(valid_mask, labels, 0)
      dout = jnp.where(valid_mask, dout, 0.0)
    elif scenario == "all_masked":
      x = jnp.zeros((b, h), dtype=x_dtype)
      labels = jnp.zeros((b,), dtype=jnp.int32)
      dout = jnp.zeros((b,), dtype=jnp.float32)
    elif scenario == "large_magnitude_logits":
      x = x * 1000.0
    elif scenario == "nan_in_x":
      nan_indices = jnp.array([0, 5, 42])
      x = x.at[nan_indices].set(jnp.nan)
    else:
      raise ValueError(f"Unknown scenario: {scenario}")

    return x, labels, w, dout

  def test_heuristic_bwd_config_non_aligned_vocab(self) -> None:
    cfg_bf16 = kernel.get_heuristic_bwd_config(
        1024,
        256,
        119548,
        dtype=jnp.dtype(jnp.bfloat16),  # pyrefly: ignore[bad-argument-type]
    )
    cfg_f32 = kernel.get_heuristic_bwd_config(
        1024,
        256,
        119548,
        dtype=jnp.dtype(jnp.float32),  # pyrefly: ignore[bad-argument-type]
    )
    self.assertEqual(cfg_bf16.b_block_size, 1024)
    self.assertEqual(cfg_bf16.h_block_size, 256)
    self.assertEqual(cfg_f32.b_block_size, 1024)
    self.assertEqual(cfg_f32.h_block_size, 256)

    cfg_b4096_f32 = kernel.get_heuristic_bwd_config(
        4096,
        256,
        119548,
        dtype=jnp.dtype(jnp.float32),  # pyrefly: ignore[bad-argument-type]
    )
    self.assertEqual(cfg_b4096_f32.b_block_size, 1024)
    self.assertEqual(cfg_b4096_f32.h_block_size, 256)

  def _run_non_aligned_vocab_bwd_test(
      self,
      b_dim: int,
      x_dtype: jnp.dtype,
      w_dtype: jnp.dtype,
      resolved_config: kernel.Config,
  ) -> None:
    _skip_if_partial_blocks_on_old_jax(
        self,
        b_dim=b_dim,
        h_dim=256,
        v_dim=119548,
        b_block_size=resolved_config.b_block_size,
        h_block_size=resolved_config.h_block_size,
        v_block_size=resolved_config.v_block_size,
    )
    rng_key = jax.random.key(42)
    op = pallas_mosaic_tpu.PallasMosaicTpuLinearSoftmaxCrossEntropyLoss(
        config=resolved_config
    )

    for scenario in _NON_ALIGNED_VOCAB_SCENARIOS:
      x, labels, w, dout = self._generate_inputs(
          scenario, x_dtype, w_dtype, rng_key, b=b_dim
      )

      # Reference in f32
      x_f32 = x.astype(jnp.float32)
      w_f32 = w.astype(jnp.float32)
      dout_f32 = dout.astype(jnp.float32)

      ref_loss, ref_lse = (
          reference.linear_softmax_cross_entropy_loss_fwd_reference(
              x_f32, labels, w_f32, reduction="none"
          )
      )
      ref_dx, ref_dw = (
          reference.linear_softmax_cross_entropy_loss_bwd_reference(
              dout_f32, ref_lse, x_f32, labels, w_f32, reduction="none"
          )
      )

      try:
        # Kernel forward
        k_loss, k_lse = (
            kernel.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
                x,
                labels,
                w,
                b_block_size=resolved_config.b_block_size,
                h_block_size=resolved_config.h_block_size,
                v_block_size=resolved_config.v_block_size,
                reduction="none",
            )
        )

        # Kernel direct backward
        k_dx, k_dw = (
            kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
                dout,
                k_lse,
                x,
                labels,
                w,
                reduction="none",
                b_block_size=resolved_config.b_block_size,
                h_block_size=resolved_config.h_block_size,
                v_block_size=resolved_config.v_block_size,
                buffer_count=resolved_config.buffer_count,
            )
        )

        loss_vjp = dx_vjp = dw_vjp = None
        if scenario == "random_normal":
          # Kernel via Op VJP (verified once per config to avoid duplicate
          # XLA compilations across input scenarios)
          def fwd_fn(_x: jax.Array, _w: jax.Array) -> jax.Array:
            return op(_x, labels, _w, reduction="none")

          loss_vjp, vjp_fn = jax.vjp(fwd_fn, x, w)
          dx_vjp, dw_vjp = vjp_fn(dout)
      except Exception as e:
        if "Ran out of memory in memory space vmem" in str(e):
          self.skipTest(
              f"Config {resolved_config} exceeds hardware VMEM on this"
              f" platform: {e}"
          )
        raise

      with self.subTest(scenario=scenario):
        # Consistency check between VJP and direct kernel call
        if scenario == "random_normal":
          assert (
              loss_vjp is not None and dx_vjp is not None and dw_vjp is not None
          )
          self._assert_allclose(
              loss_vjp, k_loss, atol=1e-4, rtol=1e-4, name="loss_vjp_vs_direct"
          )
          self._assert_allclose(
              dx_vjp, k_dx, atol=1e-4, rtol=1e-4, name="dx_vjp_vs_direct"
          )
          self._assert_allclose(
              dw_vjp, k_dw, atol=1e-4, rtol=1e-4, name="dw_vjp_vs_direct"
          )

        if scenario == "nan_in_x":
          nan_indices = jnp.array([0, 5, 42])
          non_nan_mask = jnp.ones(b_dim, dtype=bool).at[nan_indices].set(False)

          # Check finiteness on non-NaN rows
          self.assertTrue(
              bool(jnp.all(jnp.isfinite(k_loss[non_nan_mask]))),
              msg="Kernel loss has NaNs in non-NaN rows",
          )
          self.assertTrue(
              bool(jnp.all(jnp.isfinite(k_lse[non_nan_mask]))),
              msg="Kernel lse has NaNs in non-NaN rows",
          )
          self.assertTrue(
              bool(jnp.all(jnp.isfinite(k_dx[non_nan_mask]))),
              msg="Kernel dx has NaNs in non-NaN rows",
          )

          # Assert non-NaN rows of dx match reference
          self._assert_allclose(
              k_dx[non_nan_mask],
              ref_dx[non_nan_mask],
              atol=1e-1,
              rtol=1e-1,
              name="nan_in_x_dx_valid_rows",
          )
        else:
          # Assert finiteness where reference is finite
          self.assertTrue(
              bool(jnp.all(jnp.isfinite(k_loss))),
              msg=f"Non-finite values in kernel loss for {scenario}",
          )
          self.assertTrue(
              bool(jnp.all(jnp.isfinite(k_lse))),
              msg=f"Non-finite values in kernel lse for {scenario}",
          )
          self.assertTrue(
              bool(jnp.all(jnp.isfinite(k_dx))),
              msg=f"Non-finite values in kernel dx for {scenario}",
          )
          self.assertTrue(
              bool(jnp.all(jnp.isfinite(k_dw))),
              msg=f"Non-finite values in kernel dw for {scenario}",
          )

          if scenario == "all_masked":
            tol = 1e-4
            dw_tol = 1e-4
          elif scenario == "large_magnitude_logits":
            tol = 2e-1
            dw_tol = 2e-1
          elif x_dtype == jnp.float32:
            tol = 1e-4
            dw_tol = 2e-4  # Headroom for B=1024/4096 dot_general reduction
          else:
            tol = 5e-2
            dw_tol = 5e-2

          self._assert_allclose(
              k_loss, ref_loss, atol=tol, rtol=tol, name=f"{scenario}_loss"
          )
          self._assert_allclose(
              k_dx, ref_dx, atol=tol, rtol=tol, name=f"{scenario}_dx"
          )
          self._assert_allclose(
              k_dw, ref_dw, atol=dw_tol, rtol=dw_tol, name=f"{scenario}_dw"
          )

  @parameterized.named_parameters(*_generate_non_aligned_vocab_test_cases())
  def test_non_aligned_vocab_bwd(
      self,
      x_dtype: jnp.dtype,
      w_dtype: jnp.dtype,
      v_block_size: int | str,
      buffer_count: tuple[int, int],
  ) -> None:
    b_dim, h_dim, v_dim = 1024, 256, 119548
    if v_block_size == "heuristic":
      resolved_config = kernel.get_heuristic_bwd_config(
          b_dim=b_dim,
          h_dim=h_dim,
          v_dim=v_dim,
          dtype=jnp.dtype(x_dtype),  # pyrefly: ignore[bad-argument-type]
          buffer_count=buffer_count,
      )
    else:
      resolved_config = kernel.Config(
          b_block_size=1024,
          h_block_size=256,
          v_block_size=int(v_block_size),
          buffer_count=buffer_count,
      )
    self._run_non_aligned_vocab_bwd_test(
        b_dim=b_dim,
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        resolved_config=resolved_config,
    )

  @parameterized.named_parameters(
      *_generate_non_aligned_vocab_b4096_test_cases()
  )
  def test_non_aligned_vocab_bwd_b4096(
      self,
      x_dtype: jnp.dtype,
      w_dtype: jnp.dtype,
      b_block_size: int | None,
      h_block_size: int | None,
      v_block_size: int | str,
      buffer_count: tuple[int, int],
  ) -> None:
    b_dim, h_dim, v_dim = 4096, 256, 119548
    if v_block_size == "heuristic":
      resolved_config = kernel.get_heuristic_bwd_config(
          b_dim=b_dim,
          h_dim=h_dim,
          v_dim=v_dim,
          dtype=jnp.dtype(x_dtype),  # pyrefly: ignore[bad-argument-type]
          buffer_count=buffer_count,
      )
    else:
      assert b_block_size is not None and h_block_size is not None
      resolved_config = kernel.Config(
          b_block_size=b_block_size,
          h_block_size=h_block_size,
          v_block_size=int(v_block_size),
          buffer_count=buffer_count,
      )
    self._run_non_aligned_vocab_bwd_test(
        b_dim=b_dim,
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        resolved_config=resolved_config,
    )


if __name__ == "__main__":
  absltest.main()
