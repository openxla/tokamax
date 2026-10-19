# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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

"""Tests for Splash Attention quantization utilities across tensor shapes."""

from typing import Any, cast
from unittest import mock

from absl.testing import absltest
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.splash_attention import splash_attention_quant_util as quant_util


def _make_nonuniform_tensor(
    shape: tuple[int, ...],
    contract_axis: int = -1,
    seed: int = 0,
    dtype: Any = jnp.bfloat16,
) -> jax.Array:
  """Creates a non-uniform tensor with distinct slice scales along each axis."""
  # Along `contract_axis`, every slice has a maximum magnitude of `7 * 2^k` so
  # that `FP8_MAX / amax = 448 / (7 * 2^k) = 2^(6-k)` is an exact power of 2,
  # while entries within each slice have varied positive and negative values and
  # every non-contracting slice has a distinct power-of-2 scale.
  rng = np.random.default_rng(seed)
  ndim = len(shape)
  c_axis = contract_axis % ndim

  # Choose non-zero signed mantissas in {-6, ..., -1, 1, ..., 6}.
  mags = rng.integers(1, 7, size=shape)
  signs = rng.choice([-1.0, 1.0], size=shape)
  base = (mags * signs).astype(np.float32)

  # Ensure every slice along `c_axis` attains magnitude 7.0 at a varied index.
  idx_grid = np.indices(shape)
  non_c_sum = sum(idx_grid[a] for a in range(ndim) if a != c_axis)
  peak_pos = non_c_sum % shape[c_axis]
  peak_mask = idx_grid[c_axis] == peak_pos
  peak_signs = np.where(non_c_sum % 2 == 0, -7.0, 7.0).astype(np.float32)
  base = np.where(peak_mask, peak_signs, base)

  # Scale each non-contracting axis by distinct powers of 2 so that wrong-axis
  # reductions or transposed scale broadcasts produce large errors.
  for axis, dim in enumerate(shape):
    if axis == c_axis:
      continue
    exponents = np.arange(dim, dtype=np.float32) - (dim // 2)
    axis_scales = (2.0**exponents).astype(np.float32)
    bcast_shape = [1] * ndim
    bcast_shape[axis] = dim
    base = base * axis_scales.reshape(bcast_shape)

  return jnp.array(base, dtype=dtype)


class QuantizationUtilityTest(absltest.TestCase):

  def test_resolve_dtype(self):
    for alias in ("float8_e4m3fn", "fp8_e4m3fn", "e4m3fn", jnp.float8_e4m3fn):
      self.assertEqual(quant_util._resolve_dtype(alias), jnp.float8_e4m3fn)
    for alias in ("float8_e5m2", "fp8_e5m2", "e5m2", jnp.float8_e5m2):
      self.assertEqual(quant_util._resolve_dtype(alias), jnp.float8_e5m2)
    self.assertEqual(quant_util._resolve_dtype("bfloat16"), jnp.bfloat16)
    self.assertEqual(quant_util._resolve_dtype(jnp.float32), jnp.float32)

  def test_apply_quantization_unquantized_and_cast_numerical(self):
    # Include out-of-range values (< -448, > 448), negative, zero, positive,
    # and a value (1.015625) that rounds in E4M3 (to 1.0) but not in BF16.
    x = jnp.array(
        [
            -1024.0,
            -448.0,
            -224.0,
            -3.5,
            0.0,
            1.015625,
            112.0,
            448.0,
            896.0,
        ],
        dtype=jnp.bfloat16,
    )
    for contract_axis in (-1, 0):
      q_unq, scale_unq = quant_util.apply_quantization(
          x, quant_util.QuantizationStrategy.UNQUANTIZED, contract_axis
      )
      np.testing.assert_array_equal(q_unq, x)
      self.assertIsNone(scale_unq)

      q_cast, scale_cast = quant_util.apply_quantization(
          x, quant_util.QuantizationStrategy.CAST, contract_axis
      )
      self.assertEqual(q_cast.dtype, jnp.float8_e4m3fn)
      self.assertIsNone(scale_cast)
      expected_cast = np.array(
          [-448.0, -448.0, -224.0, -3.5, 0.0, 1.0, 112.0, 448.0, 448.0],
          dtype=np.float32,
      )
      np.testing.assert_array_equal(
          np.asarray(q_cast.astype(jnp.float32)), expected_cast
      )

  def test_apply_quantization_per_tile_numerical(self):
    # Case 1: amax = 224.0 (< FP8_MAX), scale = 0.5, multiplier = 2.0.
    x1 = jnp.array([-224.0, -56.0, 0.0, 14.0, 112.0], dtype=jnp.bfloat16)
    q1, s1 = quant_util.apply_quantization(
        x1, quant_util.QuantizationStrategy.PER_TILE, contract_axis=-1
    )
    self.assertEqual(q1.dtype, jnp.float8_e4m3fn)
    self.assertIsNotNone(s1)
    self.assertEqual(s1.shape, ())
    np.testing.assert_allclose(np.asarray(s1), 0.5, rtol=1e-6)
    np.testing.assert_array_equal(
        np.asarray(q1.astype(jnp.float32)),
        np.array([-448.0, -112.0, 0.0, 28.0, 224.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(q1.astype(jnp.float32) * s1),
        np.asarray(x1.astype(jnp.float32)),
        rtol=1e-6,
    )

    # Case 2: 2D array with amax = 896.0 (> FP8_MAX), scale = 2.0.
    x2 = jnp.array([[-896.0, 224.0], [-112.0, 448.0]], dtype=jnp.bfloat16)
    q2, s2 = quant_util.apply_quantization(
        x2, quant_util.QuantizationStrategy.PER_TILE, contract_axis=0
    )
    self.assertIsNotNone(s2)
    np.testing.assert_allclose(np.asarray(s2), 2.0, rtol=1e-6)
    np.testing.assert_array_equal(
        np.asarray(q2.astype(jnp.float32)),
        np.array([[-448.0, 112.0], [-56.0, 224.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(q2.astype(jnp.float32) * s2),
        np.asarray(x2.astype(jnp.float32)),
        rtol=1e-6,
    )

    # Case 3: All-zero array clamps amax to 1e-12 without NaNs.
    x_zero = jnp.zeros((2, 4), dtype=jnp.bfloat16)
    q_zero, s_zero = quant_util.apply_quantization(
        x_zero, quant_util.QuantizationStrategy.PER_TILE, contract_axis=-1
    )
    self.assertFalse(bool(jnp.any(jnp.isnan(q_zero))))
    np.testing.assert_array_equal(np.asarray(q_zero.astype(jnp.float32)), 0.0)
    np.testing.assert_allclose(np.asarray(s_zero), 1e-12 / 448.0, rtol=1e-5)

    # Case 4: Zero slice in PER_AXIS clamps amax to 1e-12 without NaNs.
    x_zero_row = jnp.array(
        [[0.0, 0.0, 0.0, 0.0], [-224.0, 112.0, -56.0, 28.0]],
        dtype=jnp.bfloat16,
    )
    q_zr, s_zr = quant_util.apply_quantization(
        x_zero_row, quant_util.QuantizationStrategy.PER_AXIS, contract_axis=-1
    )
    self.assertIsNotNone(s_zr)
    self.assertFalse(bool(jnp.any(jnp.isnan(q_zr))))
    np.testing.assert_array_equal(
        np.asarray(q_zr[0].astype(jnp.float32)),
        np.zeros((4,), dtype=np.float32),
    )
    np.testing.assert_allclose(float(s_zr[0, 0]), 1e-12 / 448.0, rtol=1e-2)
    np.testing.assert_allclose(float(s_zr[1, 0]), 0.5, rtol=1e-5)

  def test_apply_quantization_1d(self):
    x = jnp.array([-112.0, -28.0, 0.0, 14.0, 56.0], dtype=jnp.bfloat16)
    for contract_axis in (-1, 0):
      q_x, scale = quant_util.apply_quantization(
          x, quant_util.QuantizationStrategy.PER_AXIS, contract_axis
      )
      self.assertEqual(q_x.dtype, jnp.float8_e4m3fn)
      self.assertIsNotNone(scale)
      self.assertEqual(scale.shape, (1,))
      self.assertEqual(scale.dtype, jnp.bfloat16)
      np.testing.assert_allclose(
          np.asarray(scale.astype(jnp.float32)), [0.25], rtol=1e-5
      )
      np.testing.assert_array_equal(
          np.asarray(q_x.astype(jnp.float32)),
          np.array([-448.0, -112.0, 0.0, 56.0, 224.0], dtype=np.float32),
      )

  def test_apply_quantization_2d(self):
    # Exact powers of 2 with distinct row and column maxima and signs.
    x = jnp.array(
        [
            [-896.0, 224.0, -112.0, 56.0],
            [28.0, -112.0, 56.0, -14.0],
            [-7.0, 14.0, -56.0, 28.0],
        ],
        dtype=jnp.bfloat16,
    )
    # contract_axis = -1 (row-wise scaling): row amax = [896, 112, 56],
    # expected row scales = [2.0, 0.25, 0.125].
    q_row, scale_row = quant_util.apply_quantization(
        x, quant_util.QuantizationStrategy.PER_AXIS, contract_axis=-1
    )
    self.assertEqual(q_row.shape, (3, 4))
    self.assertIsNotNone(scale_row)
    self.assertEqual(scale_row.shape, (3, 1))
    np.testing.assert_allclose(
        np.asarray(scale_row.astype(jnp.float32)),
        np.array([[2.0], [0.25], [0.125]], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.max(np.abs(np.asarray(q_row.astype(jnp.float32))), axis=-1),
        np.array([448.0, 448.0, 448.0], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(q_row.astype(jnp.float32) * scale_row.astype(jnp.float32)),
        np.asarray(x.astype(jnp.float32)),
        rtol=1e-5,
    )

    # contract_axis = 0 (column-wise scaling): col amax = [896, 224, 112, 56],
    # expected col scales = [2.0, 0.5, 0.25, 0.125].
    q_col, scale_col = quant_util.apply_quantization(
        x, quant_util.QuantizationStrategy.PER_AXIS, contract_axis=0
    )
    self.assertEqual(q_col.shape, (3, 4))
    self.assertIsNotNone(scale_col)
    self.assertEqual(scale_col.shape, (1, 4))
    np.testing.assert_allclose(
        np.asarray(scale_col.astype(jnp.float32)),
        np.array([[2.0, 0.5, 0.25, 0.125]], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.max(np.abs(np.asarray(q_col.astype(jnp.float32))), axis=0),
        np.array([448.0, 448.0, 448.0, 448.0], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(q_col.astype(jnp.float32) * scale_col.astype(jnp.float32)),
        np.asarray(x.astype(jnp.float32)),
        rtol=1e-5,
    )

  def test_apply_quantization_3d(self):
    for contract_axis, expected_scale_shape in (
        (-1, (2, 4, 1)),
        (1, (2, 1, 16)),
        (0, (1, 4, 16)),
    ):
      x = _make_nonuniform_tensor(
          (2, 4, 16), contract_axis=contract_axis, seed=11
      )
      q_x, scale = quant_util.apply_quantization(
          x, quant_util.QuantizationStrategy.PER_AXIS, contract_axis
      )
      self.assertEqual(q_x.shape, (2, 4, 16))
      self.assertEqual(q_x.dtype, jnp.float8_e4m3fn)
      self.assertIsNotNone(scale)
      self.assertEqual(scale.shape, expected_scale_shape)
      expected_scale = (
          jnp.max(jnp.abs(x), axis=contract_axis, keepdims=True)
          / quant_util.FP8_MAX
      )
      np.testing.assert_allclose(
          np.asarray(scale.astype(jnp.float32)),
          np.asarray(expected_scale.astype(jnp.float32)),
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          np.max(
              np.abs(np.asarray(q_x.astype(jnp.float32))), axis=contract_axis
          ),
          quant_util.FP8_MAX,
          rtol=1e-5,
      )
      deq = q_x.astype(jnp.float32) * scale.astype(jnp.float32)
      np.testing.assert_allclose(
          np.asarray(deq),
          np.asarray(x.astype(jnp.float32)),
          rtol=1e-5,
      )

  def test_apply_quantization_4d(self):
    for contract_axis, expected_scale_shape in (
        (-1, (2, 3, 4, 1)),
        (2, (2, 3, 1, 16)),
        (1, (2, 1, 4, 16)),
    ):
      x = _make_nonuniform_tensor(
          (2, 3, 4, 16), contract_axis=contract_axis, seed=23
      )
      q_x, scale = quant_util.apply_quantization(
          x, quant_util.QuantizationStrategy.PER_AXIS, contract_axis
      )
      self.assertEqual(q_x.shape, (2, 3, 4, 16))
      self.assertEqual(q_x.dtype, jnp.float8_e4m3fn)
      self.assertIsNotNone(scale)
      self.assertEqual(scale.shape, expected_scale_shape)
      deq = q_x.astype(jnp.float32) * scale.astype(jnp.float32)
      np.testing.assert_allclose(
          np.asarray(deq),
          np.asarray(x.astype(jnp.float32)),
          rtol=1e-5,
      )

  def test_fake_micro_dispatch_and_invalid_strategy(self):
    x = _make_nonuniform_tensor((4, 16), contract_axis=1, seed=7)
    mock_fn = mock.MagicMock(
        return_value=(
            x.astype(jnp.float8_e5m2),
            jnp.ones((4, 1), dtype=jnp.bfloat16),
        )
    )
    self.assertIsNone(quant_util._get_fast_fake_micro_quantize_no_reshape())

    with mock.patch.object(
        quant_util,
        "_get_fast_fake_micro_quantize_no_reshape",
        return_value=mock_fn,
    ):
      q_x, scale = quant_util.apply_quantization(
          x,
          quant_util.QuantizationStrategy.FAKE_MICRO,
          contract_axis=1,
          ablate_scale=True,
          use_ceil_scale=True,
          dtype="e5m2",
      )
      mock_fn.assert_called_once_with(
          x,
          contract_axis=1,
          dtype=jnp.float8_e5m2,
          ablate_scale=True,
          use_ceil_scale=True,
      )
      self.assertEqual(q_x.dtype, jnp.float8_e5m2)
      self.assertIsNotNone(scale)
      self.assertEqual(scale.shape, (4, 1))

    with mock.patch.object(
        quant_util,
        "_get_fast_fake_micro_quantize_no_reshape",
        return_value=None,
    ):
      with self.assertRaises(NotImplementedError):
        quant_util.fast_fake_micro_quantize_no_reshape(x, contract_axis=-1)

    with self.assertRaises(ValueError):
      quant_util.apply_quantization(x, cast(Any, 999), contract_axis=-1)

  def _assert_dot_general_quantized_fidelity(
      self,
      lhs_shape: tuple[int, ...],
      rhs_shape: tuple[int, ...],
      dimension_numbers: Any,
      expected_shape: tuple[int, ...],
  ):
    lhs_c = dimension_numbers[0][0][0]
    rhs_c = dimension_numbers[0][1][0]
    lhs = _make_nonuniform_tensor(lhs_shape, contract_axis=lhs_c, seed=101)
    rhs = _make_nonuniform_tensor(rhs_shape, contract_axis=rhs_c, seed=202)
    ref = lax.dot_general(
        lhs.astype(jnp.float32),
        rhs.astype(jnp.float32),
        dimension_numbers,
    )

    for use_fp8 in (False, True):
      out = quant_util.dot_general_quantized(
          lhs,
          rhs,
          dimension_numbers,
          quant_util.QuantizationStrategy.PER_AXIS,
          quant_util.QuantizationStrategy.PER_AXIS,
          use_fp8=use_fp8,
      )
      self.assertEqual(out.shape, expected_shape)
      np.testing.assert_allclose(
          np.asarray(out), np.asarray(ref), rtol=1e-4, atol=1e-3
      )

  def test_dot_general_quantized_2d(self):
    self._assert_dot_general_quantized_fidelity(
        (4, 16), (16, 8), (((1,), (0,)), ((), ())), (4, 8)
    )

  def test_dot_general_quantized_2d_transposed(self):
    self._assert_dot_general_quantized_fidelity(
        (16, 4), (16, 8), (((0,), (0,)), ((), ())), (4, 8)
    )
    self._assert_dot_general_quantized_fidelity(
        (4, 16), (8, 16), (((1,), (1,)), ((), ())), (4, 8)
    )

  def test_dot_general_quantized_3d_batched(self):
    self._assert_dot_general_quantized_fidelity(
        (2, 4, 16), (2, 16, 8), (((2,), (1,)), ((0,), (0,))), (2, 4, 8)
    )
    self._assert_dot_general_quantized_fidelity(
        (2, 16, 4), (2, 8, 16), (((1,), (2,)), ((0,), (0,))), (2, 4, 8)
    )

  def test_dot_general_quantized_4d_multi_head_qk(self):
    # Shape: (B, H, S_q, D) @ (B, H, D, S_kv) -> (B, H, S_q, S_kv)
    self._assert_dot_general_quantized_fidelity(
        (2, 3, 4, 16),
        (2, 3, 16, 8),
        (((3,), (2,)), ((0, 1), (0, 1))),
        (2, 3, 4, 8),
    )
    # Shape: (B, H, S_q, D) @ (B, H, S_kv, D) -> (B, H, S_q, S_kv)
    self._assert_dot_general_quantized_fidelity(
        (2, 3, 4, 16),
        (2, 3, 8, 16),
        (((3,), (3,)), ((0, 1), (0, 1))),
        (2, 3, 4, 8),
    )

  def test_dot_general_quantized_4d_multi_head_pv(self):
    # Shape: (B, H, S_q, S_kv) @ (B, H, S_kv, D_v) -> (B, H, S_q, D_v)
    self._assert_dot_general_quantized_fidelity(
        (2, 3, 4, 8),
        (2, 3, 8, 16),
        (((3,), (2,)), ((0, 1), (0, 1))),
        (2, 3, 4, 16),
    )

  def test_dot_general_quantized_strategy_combinations(self):
    lhs = _make_nonuniform_tensor((2, 3, 8), contract_axis=2, seed=303) * 2.0
    rhs = _make_nonuniform_tensor((2, 8, 5), contract_axis=1, seed=404) * 4.0
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    ref = lax.dot_general(
        lhs.astype(jnp.float32),
        rhs.astype(jnp.float32),
        dimension_numbers,
    )

    strategy_pairs = (
        (
            quant_util.QuantizationStrategy.UNQUANTIZED,
            quant_util.QuantizationStrategy.UNQUANTIZED,
        ),
        (
            quant_util.QuantizationStrategy.CAST,
            quant_util.QuantizationStrategy.CAST,
        ),
        (
            quant_util.QuantizationStrategy.PER_TILE,
            quant_util.QuantizationStrategy.PER_TILE,
        ),
        (
            quant_util.QuantizationStrategy.PER_TILE,
            quant_util.QuantizationStrategy.PER_AXIS,
        ),
        (
            quant_util.QuantizationStrategy.PER_AXIS,
            quant_util.QuantizationStrategy.PER_TILE,
        ),
        (
            quant_util.QuantizationStrategy.PER_AXIS,
            quant_util.QuantizationStrategy.UNQUANTIZED,
        ),
        (
            quant_util.QuantizationStrategy.UNQUANTIZED,
            quant_util.QuantizationStrategy.PER_AXIS,
        ),
    )
    for lhs_strat, rhs_strat in strategy_pairs:
      out = quant_util.dot_general_quantized(
          lhs,
          rhs,
          dimension_numbers,
          lhs_strat,
          rhs_strat,
          use_fp8=False,
      )
      self.assertEqual(out.shape, (2, 3, 5))
      np.testing.assert_allclose(
          np.asarray(out), np.asarray(ref), rtol=1e-4, atol=1e-3
      )

    # Verify use_fp8=True raises ValueError when either lhs or rhs is not e4m3.
    with self.assertRaises(ValueError):
      quant_util.dot_general_quantized(
          lhs,
          rhs,
          dimension_numbers,
          quant_util.QuantizationStrategy.UNQUANTIZED,
          quant_util.QuantizationStrategy.PER_AXIS,
          use_fp8=True,
      )
    with self.assertRaises(ValueError):
      quant_util.dot_general_quantized(
          lhs,
          rhs,
          dimension_numbers,
          quant_util.QuantizationStrategy.PER_AXIS,
          quant_util.QuantizationStrategy.UNQUANTIZED,
          use_fp8=True,
      )


if __name__ == "__main__":
  absltest.main()
