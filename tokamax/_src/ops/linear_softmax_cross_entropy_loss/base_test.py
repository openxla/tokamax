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

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src.ops.linear_softmax_cross_entropy_loss import base
from tokamax._src.ops.linear_softmax_cross_entropy_loss import reference
from tokamax._src.ops.linear_softmax_cross_entropy_loss import test_utils


def _make_inputs(
    seed: int,
    b_dim: int,
    h_dim: int,
    v_dim: int,
    x_dtype: jax.typing.DTypeLike = jnp.float32,
    w_dtype: jax.typing.DTypeLike = jnp.float32,
):
  x, labels, w = test_utils.generate_random_data(
      jax.random.key(seed), b_dim, h_dim, v_dim, dtype=jnp.float32
  )
  return x.astype(x_dtype), labels, (w / jnp.sqrt(h_dim)).astype(w_dtype)


class LinearSoftmaxCrossEntropyLossBaseTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("small_sum_f32", 1024, 512, 2048, "sum", jnp.float32, jnp.float32),
      ("medium_sum_f32", 4096, 1024, 4096, "sum", jnp.float32, jnp.float32),
      (
          "unaligned_none_bf16",
          1100,
          64,
          2100,
          "none",
          jnp.bfloat16,
          jnp.bfloat16,
      ),
      (
          "unaligned_mean_f32_bf16",
          1100,
          64,
          2100,
          "mean",
          jnp.float32,
          jnp.bfloat16,
      ),
  )
  def test_op_fwd_bwd_matches_reference(
      self, b_dim, h_dim, v_dim, reduction, x_dtype, w_dtype
  ):
    x, labels, w = _make_inputs(42, b_dim, h_dim, v_dim, x_dtype, w_dtype)
    fn = base.LinearSoftmaxCrossEntropyLoss()

    with jax.default_matmul_precision("highest"):
      op_loss, vjp_fn = jax.vjp(
          lambda x_in, w_in: fn(x_in, labels, w_in, reduction=reduction), x, w
      )
      dout = jnp.ones_like(op_loss)
      op_grad_x, op_grad_w = vjp_fn(dout)

      x_f32, w_f32 = x.astype(jnp.float32), w.astype(jnp.float32)
      ref_loss, ref_lse = (
          reference.linear_softmax_cross_entropy_loss_fwd_reference(
              x_f32, labels, w_f32, reduction=reduction
          )
      )
      ref_grad_x, ref_grad_w = (
          reference.linear_softmax_cross_entropy_loss_bwd_reference(
              dout, ref_lse, x_f32, labels, w_f32, reduction=reduction
          )
      )

    tol = 1e-3 if x_dtype == w_dtype == jnp.float32 else 5e-2
    self.assertEqual(op_loss.dtype, jnp.float32)
    self.assertEqual(op_grad_x.dtype, x_dtype)
    self.assertEqual(op_grad_w.dtype, w_dtype)
    chex.assert_trees_all_close(
        (op_loss, op_grad_x, op_grad_w),
        (ref_loss, ref_grad_x, ref_grad_w),
        atol=tol,
        rtol=tol,
    )

  @parameterized.named_parameters(
      ("aligned_sum_f32", 128, 512, "sum", jnp.float32, jnp.float32),
      ("unaligned_v_mean_f32", 128, 300, "mean", jnp.float32, jnp.float32),
      ("unaligned_bv_sum_f32", 200, 300, "sum", jnp.float32, jnp.float32),
      ("unaligned_bv_none_f32", 200, 300, "none", jnp.float32, jnp.float32),
      (
          "smaller_than_block_none_f32",
          100,
          150,
          "none",
          jnp.float32,
          jnp.float32,
      ),
      ("one_block_plus_one_sum_f32", 129, 257, "sum", jnp.float32, jnp.float32),
      ("single_element_none_f32", 1, 1, "none", jnp.float32, jnp.float32),
      ("unaligned_bv_mean_bf16", 200, 300, "mean", jnp.bfloat16, jnp.bfloat16),
      (
          "one_block_plus_one_none_bf16",
          129,
          257,
          "none",
          jnp.bfloat16,
          jnp.bfloat16,
      ),
      ("unaligned_bv_sum_f16", 200, 300, "sum", jnp.float16, jnp.float16),
      (
          "unaligned_bv_none_f32_bf16",
          200,
          300,
          "none",
          jnp.float32,
          jnp.bfloat16,
      ),
      ("unaligned_bv_sum_bf16_f32", 200, 300, "sum", jnp.bfloat16, jnp.float32),
  )
  def test_chunked_matches_reference(
      self, b_dim, v_dim, reduction, x_dtype, w_dtype
  ):
    h_dim, b_block_size, v_block_size = 64, 128, 256
    x, labels, w = _make_inputs(42, b_dim, h_dim, v_dim, x_dtype, w_dtype)
    if reduction == "none":
      dout = jax.random.normal(jax.random.key(123), (b_dim,), dtype=jnp.float32)
    else:
      dout = jnp.array(1.0, dtype=jnp.float32)

    with jax.default_matmul_precision("highest"):
      loss, lse = base.linear_softmax_cross_entropy_loss_fwd_xla(
          x,
          labels,
          w,
          reduction=reduction,
          b_block_size=b_block_size,
          v_block_size=v_block_size,
      )
      dx, dw = base.linear_softmax_cross_entropy_loss_bwd_xla(
          dout,
          lse,
          x,
          labels,
          w,
          reduction=reduction,
          b_block_size=b_block_size,
          v_block_size=v_block_size,
      )
      x_f32, w_f32 = x.astype(jnp.float32), w.astype(jnp.float32)
      ref_loss, ref_lse = (
          reference.linear_softmax_cross_entropy_loss_fwd_reference(
              x_f32, labels, w_f32, reduction=reduction
          )
      )
      ref_dx, ref_dw = (
          reference.linear_softmax_cross_entropy_loss_bwd_reference(
              dout, ref_lse, x_f32, labels, w_f32, reduction=reduction
          )
      )

    tol = 1e-3 if x_dtype == w_dtype == jnp.float32 else 5e-2
    self.assertEqual(loss.dtype, jnp.float32)
    self.assertEqual(lse.dtype, jnp.float32)
    self.assertEqual(dx.dtype, x_dtype)
    self.assertEqual(dw.dtype, w_dtype)
    chex.assert_trees_all_close(
        (loss, lse, dx, dw),
        (ref_loss, ref_lse, ref_dx, ref_dw),
        atol=tol,
        rtol=tol,
    )

  def test_scalar_dout_with_none_reduction(self):
    b_dim, h_dim, v_dim = 200, 64, 300
    x, labels, w = _make_inputs(11, b_dim, h_dim, v_dim)
    _, lse = base.linear_softmax_cross_entropy_loss_fwd_xla(
        x, labels, w, reduction="none", b_block_size=128, v_block_size=256
    )
    dx_scalar, dw_scalar = base.linear_softmax_cross_entropy_loss_bwd_xla(
        2.5,
        lse,
        x,
        labels,
        w,
        reduction="none",
        b_block_size=128,
        v_block_size=256,
    )
    dx_vec, dw_vec = base.linear_softmax_cross_entropy_loss_bwd_xla(
        jnp.full((b_dim,), 2.5, dtype=jnp.float32),
        lse,
        x,
        labels,
        w,
        reduction="none",
        b_block_size=128,
        v_block_size=256,
    )
    chex.assert_trees_all_close(
        (dx_scalar, dw_scalar), (dx_vec, dw_vec), atol=1e-5, rtol=1e-5
    )

  @parameterized.named_parameters(
      ("f32", jnp.float32),
      ("bf16", jnp.bfloat16),
  )
  def test_very_negative_logits_no_nan_grads(self, dtype):
    b_dim, h_dim, v_dim = 128, 64, 300
    x, labels, w = test_utils.generate_random_data(
        jax.random.key(123), b_dim, h_dim, v_dim, dtype=dtype
    )
    w = w - (1000.0 / jnp.sqrt(h_dim)).astype(dtype)

    loss, lse = base.linear_softmax_cross_entropy_loss_fwd_xla(
        x, labels, w, b_block_size=128, v_block_size=256
    )
    dx, dw = base.linear_softmax_cross_entropy_loss_bwd_xla(
        jnp.ones_like(loss),
        lse,
        x,
        labels,
        w,
        b_block_size=128,
        v_block_size=256,
    )

    self.assertTrue(jnp.all(jnp.isfinite(loss)))
    self.assertTrue(jnp.all(jnp.isfinite(dx)))
    self.assertTrue(jnp.all(jnp.isfinite(dw)))

  def test_nan_row_isolated(self):
    b_dim, h_dim, v_dim = 128, 64, 300
    x, labels, w = _make_inputs(7, b_dim, h_dim, v_dim)
    nan_rows = jnp.array([0, 5, 42])
    x = x.at[nan_rows].set(jnp.nan)
    dout = jax.random.normal(jax.random.key(8), (b_dim,))

    loss, lse = base.linear_softmax_cross_entropy_loss_fwd_xla(
        x, labels, w, reduction="none", b_block_size=64, v_block_size=128
    )
    dx, _ = base.linear_softmax_cross_entropy_loss_bwd_xla(
        dout,
        lse,
        x,
        labels,
        w,
        reduction="none",
        b_block_size=64,
        v_block_size=128,
    )

    valid_rows = jnp.ones((b_dim,), dtype=bool).at[nan_rows].set(False)
    self.assertTrue(jnp.all(jnp.isnan(loss[nan_rows])))
    self.assertTrue(jnp.all(jnp.isfinite(loss[valid_rows])))
    self.assertTrue(jnp.all(jnp.isfinite(lse[valid_rows])))
    self.assertTrue(jnp.all(jnp.isfinite(dx[valid_rows])))

  def test_unsigned_labels_match_signed(self):
    x, labels, w = _make_inputs(9, 64, 32, 200)
    labels_u8 = labels.astype(jnp.uint8)

    loss_i32, lse_i32 = base.linear_softmax_cross_entropy_loss_fwd_xla(
        x, labels, w, b_block_size=32, v_block_size=64
    )
    loss_u8, lse_u8 = base.linear_softmax_cross_entropy_loss_fwd_xla(
        x, labels_u8, w, b_block_size=32, v_block_size=64
    )
    dx_i32, dw_i32 = base.linear_softmax_cross_entropy_loss_bwd_xla(
        1.0, lse_i32, x, labels, w, b_block_size=32, v_block_size=64
    )
    dx_u8, dw_u8 = base.linear_softmax_cross_entropy_loss_bwd_xla(
        1.0, lse_u8, x, labels_u8, w, b_block_size=32, v_block_size=64
    )
    chex.assert_trees_all_close(
        (loss_u8, lse_u8, dx_u8, dw_u8),
        (loss_i32, lse_i32, dx_i32, dw_i32),
        atol=1e-5,
        rtol=1e-5,
    )

  def test_unsupported_reduction_raises_value_error(self):
    x, labels, w = _make_inputs(0, 16, 8, 32)
    lse = jnp.zeros((16,), dtype=jnp.float32)
    bad_reduction: Any = "invalid"
    with self.assertRaisesRegex(ValueError, "Unsupported reduction"):
      base.linear_softmax_cross_entropy_loss_fwd_xla(
          x, labels, w, reduction=bad_reduction
      )
    with self.assertRaisesRegex(ValueError, "Unsupported reduction"):
      base.linear_softmax_cross_entropy_loss_bwd_xla(
          1.0, lse, x, labels, w, reduction=bad_reduction
      )

  def test_vmap_value_and_grad(self):
    fn = base.LinearSoftmaxCrossEntropyLoss()
    x0, labels0, w = _make_inputs(1, 64, 32, 128)
    x1, labels1, _ = _make_inputs(2, 64, 32, 128)
    vg = jax.value_and_grad(fn, argnums=(0, 2))

    with jax.default_matmul_precision("highest"):
      losses, (dxs, dws) = jax.vmap(vg, in_axes=(0, 0, None))(
          jnp.stack([x0, x1]), jnp.stack([labels0, labels1]), w
      )
      for i, (xi, li) in enumerate([(x0, labels0), (x1, labels1)]):
        ref_loss, (ref_dx, ref_dw) = vg(xi, li, w)
        chex.assert_trees_all_close(
            (losses[i], dxs[i], dws[i]),
            (ref_loss, ref_dx, ref_dw),
            atol=1e-5,
            rtol=1e-5,
        )

  def test_symbolic_shapes_raise_not_implemented(self):
    fn = base.LinearSoftmaxCrossEntropyLoss()
    (b_sym,) = jax.export.symbolic_shape("B")
    x = jax.ShapeDtypeStruct((b_sym, 32), jnp.float32)
    labels = jax.ShapeDtypeStruct((b_sym,), jnp.int32)
    w = jax.ShapeDtypeStruct((32, 64), jnp.float32)
    with self.assertRaisesRegex(NotImplementedError, "symbolic shapes"):
      jax.eval_shape(fn, x, labels, w)


if __name__ == "__main__":
  absltest.main()
