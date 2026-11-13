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
"""Unit and TPU tests for standalone Pallas ragged_causal_conv1d."""

# pylint: disable=invalid-name

import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src import jaxtyping as tokamax_jaxtyping
from tokamax._src import mosaic_tpu
from tokamax._src.ops.experimental import ragged_causal_conv1d
from tokamax._src.ops.experimental.ragged_causal_conv1d import causal_conv1d
from tokamax._src.ops.experimental.ragged_causal_conv1d import strided_ldst


def _silu_f64(x: np.ndarray) -> np.ndarray:
  return x / (1.0 + np.exp(-x))


def _ragged_causal_conv1d_reference(
    x: jax.Array,
    conv_state: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None = None,
    query_start_loc: jax.Array | None = None,
    state_indices: jax.Array | None = None,
    has_initial_state: jax.Array | None = None,
    distribution: jax.Array | None = None,
    *,
    activation: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Independent sequential float64 NumPy oracle for ragged_causal_conv1d."""
  x_f64 = np.asarray(x.astype(jnp.float32), dtype=np.float64)
  B, D = x_f64.shape
  _, _, K = weight.shape
  K_prev = K - 1

  orig_state_shape = conv_state.shape
  state_3d = conv_state[:, :, 0, :] if conv_state.ndim == 4 else conv_state
  expected_state = np.array(
      state_3d.astype(jnp.float32), dtype=np.float64, copy=True
  )
  S, _, _ = expected_state.shape
  w_f64 = np.asarray(weight[:, 0, :].astype(jnp.float32), dtype=np.float64)
  b_f64 = (
      np.asarray(bias.astype(jnp.float32), dtype=np.float64)
      if bias is not None
      else None
  )

  if query_start_loc is None:
    if state_indices is None or state_indices.shape == (1,):
      qsl = np.array([0, B], dtype=np.int32)
    else:
      qsl = np.arange(B + 1, dtype=np.int32)
  else:
    qsl = np.clip(np.asarray(query_start_loc, dtype=np.int32), 0, B)
    qsl[0] = 0
    qsl = np.maximum.accumulate(qsl)

  M = len(qsl) - 1
  slots = (
      np.arange(M, dtype=np.int32)
      if state_indices is None
      else np.asarray(state_indices, dtype=np.int32)
  )
  has_init = (
      np.zeros((M,), dtype=bool)
      if has_initial_state is None
      else np.asarray(has_initial_state, dtype=bool)
  )
  raw_active = (
      M
      if distribution is None
      else int(np.asarray(distribution, dtype=np.int32)[2])
  )
  num_active_seqs = int(np.clip(raw_active, 0, M))

  out_f64 = np.zeros_like(x_f64)
  for s_idx in range(num_active_seqs):
    q_start = int(qsl[s_idx])
    q_end = int(qsl[s_idx + 1])
    q_len = q_end - q_start
    if q_len <= 0:
      continue

    slot = int(slots[s_idx])
    valid_slot = 0 <= slot < S
    use_state = valid_slot and bool(has_init[s_idx])
    prev_rows = (
        expected_state[slot].copy()
        if use_state
        else np.zeros((K_prev, D), dtype=np.float64)
    )
    window = np.concatenate([prev_rows, x_f64[q_start:q_end]], axis=0)
    for t_idx in range(q_len):
      acc = np.zeros((D,), dtype=np.float64)
      for j in range(K):
        acc += window[t_idx + j] * w_f64[:, j]
      if b_f64 is not None:
        acc += b_f64
      if activation in ("silu", "swish"):
        acc = _silu_f64(acc)
      out_f64[q_start + t_idx] = acc

    if valid_slot:
      expected_state[slot] = window[q_len : q_len + K_prev]

  out_cast = np.asarray(
      jnp.asarray(out_f64, dtype=jnp.float32)
      .astype(x.dtype)
      .astype(jnp.float32)
  )
  state_cast = np.asarray(expected_state, dtype=np.float32).reshape(
      orig_state_shape
  )
  return out_cast, state_cast


class _FakeVmemRef:
  """Minimal 2D VMEM ref test double tracking `bitcast` invocations."""

  def __init__(self, arr: np.ndarray):
    self.arr = arr
    self.bitcast_calls = 0

  @property
  def ndim(self) -> int:
    return self.arr.ndim

  @property
  def shape(self) -> tuple[int, ...]:
    return self.arr.shape

  @property
  def dtype(self):
    return jnp.dtype(self.arr.dtype)

  def bitcast(self, target_dtype):
    del target_dtype
    self.bitcast_calls += 1
    return self

  def __getitem__(self, idx):
    return jnp.asarray(self.arr[idx])

  def __setitem__(self, idx, val):
    self.arr[idx] = np.asarray(val)


class RaggedCausalConv1dTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() == "gpu":
      self.skipTest("Pallas Mosaic TPU kernels are not supported on GPU.")

  def test_preprocess_conv1d_metadata_and_oob_clamping(self):
    dim = 4
    kernel_size = 4
    cfgs = causal_conv1d.ConvConfigs(
        batch_size=32, dim_size=dim, kernel_size=kernel_size, tile_size=16
    )
    with self.assertRaisesRegex(ValueError, "divisible by tile_size"):
      causal_conv1d.ConvConfigs(
          batch_size=30, dim_size=dim, kernel_size=kernel_size, tile_size=16
      )

    # 4 active sequences: seq 1 has `PAD_SLOT_ID=-1`, seq 3 has positive OOB
    # `state_indices=5 >= num_slots=5`, plus 1 inactive slot beyond `num_seqs`.
    query_start_loc = jnp.array([0, 1, 4, 5, 19, 24], dtype=jnp.int32)
    state_indices = jnp.array([2, -1, 0, 5, -1], dtype=jnp.int32)
    has_initial_state = jnp.array([True, True, False, True, True], dtype=bool)

    metadata = ragged_causal_conv1d.preprocess_conv1d_metadata(
        cfgs=cfgs,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        num_seqs=jnp.array(4, dtype=jnp.int32),
        num_slots=5,
    )
    self.assertLen(metadata, 6)
    self.assertEqual(int(metadata.num_tiles), 2)
    np.testing.assert_array_equal(
        np.asarray(metadata.s_idx_to_state_idx), np.array([2, 0, 0, 0, 0])
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.s_idx_has_initial_state),
        np.array([True, False, False, False, False]),
    )
    # Both seq 1 (`-1`) and seq 3 (`5 >= num_slots=5`) must suppress state
    # writes so slot 0 (owned by seq 2 at row 4) is never overwritten!
    write_rows = np.flatnonzero(np.asarray(metadata.b_idx_should_write))
    np.testing.assert_array_equal(write_rows, np.array([0, 4]))
    np.testing.assert_array_equal(
        np.asarray(metadata.b_idx_to_sz_from_old[19:]), 0
    )

    # OOB `num_seqs > max_seqs` and `query_start_loc > batch_size` clamp safely
    # into `[0, max_seqs]` and `[0, cfgs.batch_size]`.
    cfgs_clamp = causal_conv1d.ConvConfigs(
        batch_size=16, dim_size=dim, kernel_size=kernel_size, tile_size=16
    )
    meta_clamp = ragged_causal_conv1d.preprocess_conv1d_metadata(
        cfgs=cfgs_clamp,
        query_start_loc=jnp.array([0, 10, 50], dtype=jnp.int32),
        state_indices=jnp.array([1, 2], dtype=jnp.int32),
        has_initial_state=jnp.array([True, True], dtype=bool),
        num_seqs=jnp.array(99, dtype=jnp.int32),
        num_slots=4,
    )
    self.assertEqual(int(meta_clamp.num_tiles), 1)
    np.testing.assert_array_equal(
        np.flatnonzero(np.asarray(meta_clamp.b_idx_should_write)),
        np.array([9, 15]),
    )

    # Negative `num_seqs < 0` clamps to `0` (`num_tiles == 0`, zero writes).
    meta_neg_seqs = ragged_causal_conv1d.preprocess_conv1d_metadata(
        cfgs=cfgs_clamp,
        query_start_loc=jnp.array([0, 8, 16], dtype=jnp.int32),
        state_indices=jnp.array([0, 1], dtype=jnp.int32),
        has_initial_state=jnp.array([True, True], dtype=bool),
        num_seqs=jnp.array(-3, dtype=jnp.int32),
        num_slots=4,
    )
    self.assertEqual(int(meta_neg_seqs.num_tiles), 0)
    self.assertEqual(
        int(np.sum(np.asarray(meta_neg_seqs.b_idx_should_write))), 0
    )

    # Inter-tile boundary at row 15 -> 16 (`tile_size=16`).
    cfgs_boundary = causal_conv1d.ConvConfigs(
        batch_size=48, dim_size=dim, kernel_size=kernel_size, tile_size=16
    )
    meta_boundary = ragged_causal_conv1d.preprocess_conv1d_metadata(
        cfgs=cfgs_boundary,
        query_start_loc=jnp.array([0, 15, 40], dtype=jnp.int32),
        state_indices=jnp.array([1, 3], dtype=jnp.int32),
        has_initial_state=jnp.array([True, True], dtype=bool),
        num_seqs=jnp.array(2, dtype=jnp.int32),
        num_slots=4,
    )
    self.assertEqual(int(meta_boundary.num_tiles), 3)
    np.testing.assert_array_equal(
        np.asarray(meta_boundary.b_idx_to_sz_from_old[14:20]),
        np.array([0, 3, 2, 1, 0, 0]),
    )

  def test_ragged_causal_conv1d_jax_against_float64_oracle(self):
    dim = 128
    kernel_size = 4
    rng = jax.random.key(303)
    k1, k2, k3, k4 = jax.random.split(rng, 4)

    x = jax.random.normal(k1, (19, dim), dtype=jnp.float32)
    conv_state = jax.random.normal(
        k2, (6, kernel_size - 1, dim), dtype=jnp.float32
    )
    weight = jax.random.normal(k3, (dim, 1, kernel_size), dtype=jnp.float32)
    bias = jax.random.normal(k4, (dim,), dtype=jnp.float32)
    qsl = jnp.array([0, 1, 5, 19], dtype=jnp.int32)
    slots = jnp.array([3, -1, 1], dtype=jnp.int32)
    has_init = jnp.array([True, True, True], dtype=bool)
    dist = jnp.array([1, 3, 3], dtype=jnp.int32)

    out_f32, state_f32 = ragged_causal_conv1d.ragged_causal_conv1d_jax(
        x,
        conv_state,
        weight,
        bias,
        qsl,
        slots,
        has_init,
        dist,
        activation="silu",
    )
    with jax.enable_x64():
      out_f64, state_f64 = ragged_causal_conv1d.ragged_causal_conv1d_jax(
          x.astype(jnp.float64),
          conv_state.astype(jnp.float64),
          weight.astype(jnp.float64),
          bias.astype(jnp.float64),
          qsl,
          slots,
          has_init,
          dist,
          activation="silu",
      )
    out_seq, state_seq = _ragged_causal_conv1d_reference(
        x,
        conv_state,
        weight,
        bias,
        qsl,
        slots,
        has_init,
        dist,
        activation="silu",
    )
    np.testing.assert_allclose(out_f32, out_f64, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_f32, state_f64, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_f32, out_seq, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_f32, state_seq, rtol=1e-5, atol=1e-5)

  @parameterized.product(
      sublane_style=[(16, 2), (8, 4)],
      vmem_mib=[16, 8],
  )
  def test_select_conv1d_tile_size_and_vmem_limit(
      self, sublane_style: tuple[int, int], vmem_mib: int
  ):
    self.assertEqual(
        causal_conv1d._estimate_conv1d_vmem_bytes(
            tile_size=16,
            dim=129,
            kernel_size=4,
            x_itemsize=2,
            has_bias=True,
            num_lanes=128,
        ),
        158720,
    )
    self.assertEqual(
        causal_conv1d._estimate_conv1d_vmem_bytes(
            tile_size=8,
            dim=128,
            kernel_size=4,
            x_itemsize=4,
            has_bias=False,
            num_lanes=128,
        ),
        46080,
    )
    sublane_tiling, x_itemsize = sublane_style
    vmem_capacity_bytes = vmem_mib * 1024 * 1024
    tile_size, vmem_limit_bytes = (
        causal_conv1d._select_conv1d_tile_size_and_vmem_limit(
            batch_size=128,
            dim=8448,
            kernel_size=4,
            requested_tile_size=64,
            sublane_tiling=sublane_tiling,
            x_itemsize=x_itemsize,
            has_bias=True,
            vmem_capacity_bytes=vmem_capacity_bytes,
            num_lanes=128,
        )
    )
    self.assertEqual(vmem_limit_bytes, int(0.90 * vmem_capacity_bytes))
    self.assertLess(tile_size, 64)
    self.assertGreaterEqual(tile_size, sublane_tiling)
    self.assertEqual(tile_size % sublane_tiling, 0)
    if tile_size > sublane_tiling:
      est = causal_conv1d._estimate_conv1d_vmem_bytes(
          tile_size=tile_size,
          dim=8448,
          kernel_size=4,
          x_itemsize=x_itemsize,
          has_bias=True,
          num_lanes=128,
      )
      self.assertLessEqual(est, int(0.80 * vmem_capacity_bytes))

  def test_strided_ldst_default_dst_dtype_and_hoisted_bitcast(self):
    # 1. Default `dst_dtype=jnp.float32` on float32 ref (zero bitcasts).
    src_f32 = np.arange(32, dtype=np.float32).reshape(4, 8)
    src_ref_f32 = _FakeVmemRef(src_f32.copy())
    compact = strided_ldst.load_large_to_compact(src_ref_f32)
    self.assertEqual(compact.shape, (4, 1, 8))
    self.assertEqual(compact.dtype, jnp.float32)
    self.assertEqual(src_ref_f32.bitcast_calls, 0)

    dst_ref_f32 = _FakeVmemRef(np.zeros_like(src_f32))
    strided_ldst.store_compact_to_large(dst_ref_f32, compact * 2.0)
    self.assertEqual(dst_ref_f32.bitcast_calls, 0)
    np.testing.assert_allclose(dst_ref_f32.arr, src_f32 * 2.0)

    # 2. Default `dst_dtype=jnp.float32` on bfloat16 ref: verify `bitcast` is
    # hoisted outside the row loop (`bitcast_calls == 1` even for 8 rows!).
    bf16_ref = _FakeVmemRef(np.zeros((8, 16), dtype=jnp.bfloat16))

    def _fake_unpack(packed, index, packed_dtype, unpacked_dtype):
      del packed, packed_dtype
      return jnp.full((1, 16), float(index + 1), dtype=unpacked_dtype)

    with mock.patch.object(
        pltpu, "unpack_elementwise", side_effect=_fake_unpack
    ):
      unpacked = strided_ldst.load_large_to_compact(bf16_ref)
    self.assertEqual(unpacked.shape, (8, 1, 16))
    self.assertEqual(unpacked.dtype, jnp.float32)
    self.assertEqual(bf16_ref.bitcast_calls, 1)

    bf16_out_ref = _FakeVmemRef(np.zeros((8, 16), dtype=jnp.bfloat16))
    with mock.patch.object(
        pltpu,
        "pack_elementwise",
        side_effect=lambda xs, packed_dtype: xs[0].astype(packed_dtype),
    ):
      strided_ldst.store_compact_to_large(bf16_out_ref, unpacked)
    self.assertEqual(bf16_out_ref.bitcast_calls, 1)

    # 3. Symmetric itemsize / shape validation errors.
    int8_ref = _FakeVmemRef(np.zeros((8, 16), dtype=np.int8))
    with self.assertRaisesRegex(ValueError, "Sublane unpacking requires"):
      strided_ldst.load_large_to_compact(int8_ref, jnp.float32)
    with self.assertRaisesRegex(ValueError, "Sublane packing requires"):
      strided_ldst.store_compact_to_large(int8_ref, unpacked)
    with self.assertRaisesRegex(ValueError, "Unpacked compact layout requires"):
      strided_ldst.load_large_to_compact(bf16_ref, jnp.bfloat16)

  def test_input_validation_raises_value_and_not_implemented_errors(self):
    dim = 128
    kernel_size = 4
    x = jnp.zeros((16, dim), dtype=jnp.float32)
    conv_state = jnp.zeros((4, kernel_size - 1, dim), dtype=jnp.float32)
    weight = jnp.zeros((dim, 1, kernel_size), dtype=jnp.float32)
    bias = jnp.zeros((dim,), dtype=jnp.float32)
    qsl = jnp.array([0, 8, 16], dtype=jnp.int32)
    state_indices = jnp.array([0, 1], dtype=jnp.int32)
    has_init = jnp.array([False, True], dtype=bool)
    dist = jnp.array([0, 2, 2], dtype=jnp.int32)

    # 1. Unsupported floating-point dtypes raise `NotImplementedError` (§3).
    with self.assertRaisesRegex(NotImplementedError, r"supports x\.dtype"):
      ragged_causal_conv1d.ragged_causal_conv1d(
          x.astype(jnp.float16), conv_state, weight, interpret=True
      )
    with self.assertRaisesRegex(
        NotImplementedError, r"requires conv_state\.dtype == float32"
    ):
      ragged_causal_conv1d.ragged_causal_conv1d(
          x, conv_state.astype(jnp.bfloat16), weight, interpret=True
      )

    # 2. `@jaxtyping.jaxtyped` catches non-float / wrong-rank tensors at API
    # boundary (`jaxtyping.TypeCheckError` subclasses `TypeError`).
    with self.assertRaises(TypeError):
      ragged_causal_conv1d.ragged_causal_conv1d(
          jnp.zeros((2, 8, dim), dtype=jnp.float32),
          conv_state,
          weight,
          interpret=True,
      )
    with self.assertRaises(TypeError):
      ragged_causal_conv1d.ragged_causal_conv1d(
          jnp.zeros((16, dim), dtype=jnp.int32),
          conv_state,
          weight,
          interpret=True,
      )

    # 3. Explicit in-body `ValueError` validation (with jaxtyping disabled).
    with tokamax_jaxtyping.disable_jaxtyping():
      with self.assertRaisesRegex(ValueError, "Expected positive 2D x"):
        ragged_causal_conv1d.ragged_causal_conv1d(
            jnp.zeros((2, 8, dim), dtype=jnp.float32),
            conv_state,
            weight,
            interpret=True,
        )
      with self.assertRaisesRegex(ValueError, "Expected activation in"):
        ragged_causal_conv1d.ragged_causal_conv1d(
            x, conv_state, weight, activation="relu", interpret=True
        )
      with self.assertRaisesRegex(ValueError, "Expected weight shape"):
        ragged_causal_conv1d.ragged_causal_conv1d(
            x,
            conv_state,
            jnp.zeros((dim, kernel_size), dtype=jnp.float32),
            interpret=True,
        )
      with self.assertRaisesRegex(ValueError, "Expected kernel_size >= 2"):
        ragged_causal_conv1d.ragged_causal_conv1d(
            x,
            conv_state,
            jnp.zeros((dim, 1, 1), dtype=jnp.float32),
            interpret=True,
        )
      with self.assertRaisesRegex(ValueError, "Expected conv_state shape"):
        ragged_causal_conv1d.ragged_causal_conv1d(
            x,
            jnp.zeros((4, dim, kernel_size - 1), dtype=jnp.float32),
            weight,
            interpret=True,
        )
      with self.assertRaisesRegex(ValueError, "Expected bias shape"):
        ragged_causal_conv1d.ragged_causal_conv1d(
            x,
            conv_state,
            weight,
            jnp.zeros((dim + 1,), dtype=jnp.float32),
            interpret=True,
        )
      with self.assertRaisesRegex(ValueError, "tile_size > 0"):
        ragged_causal_conv1d.ragged_causal_conv1d(
            x, conv_state, weight, tile_size=0, interpret=True
        )
      with self.assertRaisesRegex(ValueError, "divisible by sublane_tiling"):
        ragged_causal_conv1d.ragged_causal_conv1d(
            x, conv_state, weight, tile_size=12, interpret=True
        )
      with self.assertRaisesRegex(ValueError, "divisible by sublane_tiling"):
        ragged_causal_conv1d.ragged_causal_conv1d(
            x.astype(jnp.bfloat16),
            conv_state,
            weight,
            tile_size=24,
            interpret=True,
        )
      with self.assertRaisesRegex(
          ValueError, "When query_start_loc is None, state_indices must be"
      ):
        ragged_causal_conv1d.ragged_causal_conv1d(
            x,
            conv_state,
            weight,
            bias,
            None,
            state_indices,
            interpret=True,
        )
      with self.assertRaisesRegex(ValueError, "Expected state_indices shape"):
        ragged_causal_conv1d.ragged_causal_conv1d(
            x,
            conv_state,
            weight,
            bias,
            qsl,
            jnp.array([0, 1, 2], dtype=jnp.int32),
            has_init,
            dist,
            interpret=True,
        )
      with self.assertRaisesRegex(
          ValueError, "Expected has_initial_state shape"
      ):
        ragged_causal_conv1d.ragged_causal_conv1d(
            x,
            conv_state,
            weight,
            bias,
            qsl,
            state_indices,
            jnp.zeros((3,), dtype=bool),
            dist,
            interpret=True,
        )
      with self.assertRaisesRegex(ValueError, "Expected distribution shape"):
        ragged_causal_conv1d.ragged_causal_conv1d(
            x,
            conv_state,
            weight,
            bias,
            qsl,
            state_indices,
            has_init,
            jnp.zeros((2,), dtype=jnp.int32),
            interpret=True,
        )

  @parameterized.product(
      dtype=[jnp.float32, jnp.bfloat16],
      activation=[None, "silu", "swish"],
  )
  def test_decode_and_oob_slot_safety_interpret(
      self, dtype, activation: str | None
  ):
    dim = 128
    kernel_size = 4
    batch_size = 19  # Non-multiple of tile_size=16 (padded to 32).
    num_slots = 8
    rng = jax.random.key(101 + (0 if activation is None else len(activation)))
    k1, k2, k3, k4 = jax.random.split(rng, 4)

    x = jax.random.normal(k1, (batch_size, dim), dtype=dtype)
    conv_state = jax.random.normal(
        k2, (num_slots, kernel_size - 1, dim), dtype=jnp.float32
    )
    weight = jax.random.normal(k3, (dim, 1, kernel_size), dtype=dtype)
    bias = jax.random.normal(k4, (dim,), dtype=dtype)

    # 5 active sequences: seq 2 has `PAD_SLOT_ID=-1` and seq 4 has positive OOB
    # `state_indices=99 >= num_slots`, both with `has_initial_state=True` while
    # slot `0` is untouched (`untouched` includes slot `0`!).
    query_start_loc = jnp.array([0, 1, 2, 5, 6, 7, 7], dtype=jnp.int32)
    state_indices = jnp.array([5, 1, -1, 7, 99, 2], dtype=jnp.int32)
    has_initial_state = jnp.array(
        [True, False, True, True, True, True], dtype=bool
    )
    distribution = jnp.array([2, 3, 5], dtype=jnp.int32)

    expected_out, expected_state = (
        ragged_causal_conv1d.ragged_causal_conv1d_jax(
            x,
            conv_state,
            weight,
            bias,
            query_start_loc,
            state_indices,
            has_initial_state,
            distribution,
            activation=activation,
        )
    )
    conv_state_orig = np.array(conv_state, copy=True)

    actual_out, actual_state = ragged_causal_conv1d.ragged_causal_conv1d(
        x,
        conv_state,
        weight,
        bias,
        query_start_loc,
        state_indices,
        has_initial_state,
        distribution,
        activation=activation,
        tile_size=16,
        interpret=True,
    )

    self.assertEqual(actual_out.shape, (batch_size, dim))
    self.assertEqual(actual_out.dtype, jnp.dtype(dtype))
    self.assertEqual(actual_state.dtype, jnp.float32)

    tol = 2e-2 if dtype == jnp.bfloat16 else 1e-5
    np.testing.assert_allclose(
        np.asarray(actual_out[:7].astype(jnp.float32)),
        np.asarray(expected_out[:7].astype(jnp.float32)),
        rtol=tol,
        atol=tol,
    )
    for slot in (5, 1, 7):
      np.testing.assert_allclose(
          np.asarray(actual_state[slot]),
          np.asarray(expected_state[slot]),
          rtol=tol,
          atol=tol,
      )
    # Slot 0 MUST remain untouched despite both `-1` and `99` OOB slots!
    for untouched in (0, 2, 3, 4, 6):
      np.testing.assert_array_equal(
          np.asarray(actual_state[untouched]), conv_state_orig[untouched]
      )

  @parameterized.named_parameters(
      ("fp32_bias_none_3d_noinit", jnp.float32, True, None, False, False),
      ("fp32_nobias_silu_3d_init", jnp.float32, False, "silu", False, True),
      ("fp32_bias_silu_4d_init", jnp.float32, True, "silu", True, True),
      ("bf16_bias_none_3d_init", jnp.bfloat16, True, None, False, True),
      ("bf16_nobias_silu_3d_noinit", jnp.bfloat16, False, "silu", False, False),
      ("bf16_bias_silu_4d_init", jnp.bfloat16, True, "silu", True, True),
  )
  def test_multi_tile_prefill_3d_4d_state_and_activation_interpret(
      self,
      dtype,
      use_bias: bool,
      activation: str | None,
      use_4d_state: bool,
      seq1_has_init: bool,
  ):
    dim = 128
    kernel_size = 4
    tile_size = 16
    batch_size = 40
    num_slots = 4
    rng = jax.random.key(
        202 + int(use_bias) + 2 * int(use_4d_state) + 4 * int(seq1_has_init)
    )
    k1, k2, k3, k4 = jax.random.split(rng, 4)

    x = jax.random.normal(k1, (batch_size, dim), dtype=dtype)
    conv_state_shape = (
        (num_slots, kernel_size - 1, 1, dim)
        if use_4d_state
        else (num_slots, kernel_size - 1, dim)
    )
    conv_state = jax.random.normal(k2, conv_state_shape, dtype=jnp.float32)
    weight = jax.random.normal(k3, (dim, 1, kernel_size), dtype=dtype)
    bias = jax.random.normal(k4, (dim,), dtype=dtype) if use_bias else None

    # Sequence 1 starts at row 15 (last row of tile 0) and spans into tiles 1-2.
    query_start_loc = jnp.array([0, 15, 40], dtype=jnp.int32)
    state_indices = jnp.array([1, 3], dtype=jnp.int32)
    has_initial_state = jnp.array([True, seq1_has_init], dtype=bool)
    distribution = jnp.array([0, 2, 2], dtype=jnp.int32)

    expected_out, expected_state = (
        ragged_causal_conv1d.ragged_causal_conv1d_jax(
            x,
            conv_state,
            weight,
            bias,
            query_start_loc,
            state_indices,
            has_initial_state,
            distribution,
            activation=activation,
        )
    )
    conv_state_orig = np.array(conv_state, copy=True)

    actual_out, actual_state = ragged_causal_conv1d.ragged_causal_conv1d(
        x,
        conv_state,
        weight,
        bias,
        query_start_loc,
        state_indices,
        has_initial_state,
        distribution,
        activation=activation,
        tile_size=tile_size,
        interpret=True,
    )

    self.assertEqual(actual_state.shape, conv_state_shape)
    tol = 2e-2 if dtype == jnp.bfloat16 else 1e-5
    np.testing.assert_allclose(
        np.asarray(actual_out.astype(jnp.float32)),
        np.asarray(expected_out.astype(jnp.float32)),
        rtol=tol,
        atol=tol,
    )
    for slot in (1, 3):
      np.testing.assert_allclose(
          np.asarray(actual_state[slot]),
          np.asarray(expected_state[slot]),
          rtol=tol,
          atol=tol,
      )
    for untouched in (0, 2):
      np.testing.assert_array_equal(
          np.asarray(actual_state[untouched]), conv_state_orig[untouched]
      )

  def test_default_optional_arguments_jit_and_zero_active_seqs(self):
    dim = 128
    kernel_size = 4
    rng = jax.random.key(505)
    k1, k2, k3, k4 = jax.random.split(rng, 4)

    x = jax.random.normal(k1, (20, dim), dtype=jnp.bfloat16)
    conv_state = jax.random.normal(
        k2, (4, kernel_size - 1, dim), dtype=jnp.float32
    )
    weight = jax.random.normal(k3, (dim, 1, kernel_size), dtype=jnp.bfloat16)
    bias = jax.random.normal(k4, (dim,), dtype=jnp.bfloat16)

    # Eager call with default `interpret=None` via `causal_conv1d` alias.
    expected_out, expected_state = (
        ragged_causal_conv1d.ragged_causal_conv1d_jax(
            x, conv_state, weight, activation="silu"
        )
    )
    actual_out, actual_state = causal_conv1d.causal_conv1d(
        x, conv_state, weight, activation="silu", tile_size=16
    )
    np.testing.assert_allclose(
        np.asarray(actual_out.astype(jnp.float32)),
        np.asarray(expected_out.astype(jnp.float32)),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        actual_state, expected_state, rtol=2e-2, atol=2e-2
    )

    # `jax.jit` call with dynamic metadata tensors and `distribution[-1] > M`.
    qsl = jnp.array([0, 1, 14, 14], dtype=jnp.int32)
    state_indices = jnp.array([2, -1, 1], dtype=jnp.int32)
    has_init = jnp.array([True, True, False], dtype=bool)
    dist = jnp.array([1, 2, 7], dtype=jnp.int32)
    jitted_fn = jax.jit(
        functools.partial(
            ragged_causal_conv1d.ragged_causal_conv1d,
            activation="silu",
            tile_size=16,
        )
    )
    jit_out, jit_state = jitted_fn(
        x, conv_state, weight, bias, qsl, state_indices, has_init, dist
    )
    ref_jit_out, ref_jit_state = ragged_causal_conv1d.ragged_causal_conv1d_jax(
        x,
        conv_state,
        weight,
        bias,
        qsl,
        state_indices,
        has_init,
        dist,
        activation="silu",
    )
    np.testing.assert_allclose(
        np.asarray(jit_out[:14].astype(jnp.float32)),
        np.asarray(ref_jit_out[:14].astype(jnp.float32)),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(jit_state), np.asarray(ref_jit_state), rtol=2e-2, atol=2e-2
    )

    # Zero active sequences (`distribution=[0, 0, 0]`) must leave `conv_state`
    # completely unchanged.
    conv_state_orig = np.array(conv_state, copy=True)
    _, zero_state = jitted_fn(
        x,
        conv_state,
        weight,
        bias,
        jnp.array([0, 0, 0, 0], dtype=jnp.int32),
        state_indices,
        has_init,
        jnp.array([0, 0, 0], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(np.asarray(zero_state), conv_state_orig)

    # `query_start_loc=None` with 1D `state_indices` of length `batch_size`
    # defaults to 1 token per sequence (`jnp.arange(B + 1)`).
    decode_slots = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    decode_out, decode_state = ragged_causal_conv1d.ragged_causal_conv1d(
        x[:4],
        conv_state,
        weight,
        bias,
        None,
        decode_slots,
        activation="silu",
        tile_size=16,
        interpret=True,
    )
    ref_dec_out, ref_dec_state = ragged_causal_conv1d.ragged_causal_conv1d_jax(
        x[:4],
        conv_state,
        weight,
        bias,
        None,
        decode_slots,
        activation="silu",
    )
    np.testing.assert_allclose(
        np.asarray(decode_out.astype(jnp.float32)),
        np.asarray(ref_dec_out.astype(jnp.float32)),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(decode_state),
        np.asarray(ref_dec_state),
        rtol=2e-2,
        atol=2e-2,
    )

  @parameterized.named_parameters(
      ("small_batch4_bf16_default_tile", 4, 64, jnp.bfloat16, "silu", False),
      ("small_batch4_fp32_default_tile", 4, 64, jnp.float32, None, True),
      ("multitile_fp32_none_3d", 37, 16, jnp.float32, None, False),
      ("multitile_fp32_silu_4d", 37, 16, jnp.float32, "silu", True),
      ("multitile_bf16_none_4d", 37, 16, jnp.bfloat16, None, True),
      ("multitile_bf16_silu_3d", 37, 16, jnp.bfloat16, "silu", False),
  )
  def test_ragged_causal_conv1d_tpu(
      self,
      batch_size: int,
      tile_size: int,
      dtype,
      activation: str | None,
      use_4d_state: bool,
  ):
    if jax.default_backend() != "tpu":
      self.skipTest("TPU hardware execution requires TPU backend.")

    dim = 256
    kernel_size = 4
    num_slots = 6
    rng = jax.random.key(606 + batch_size + int(use_4d_state))
    k1, k2, k3, k4 = jax.random.split(rng, 4)

    x = jax.random.normal(k1, (batch_size, dim), dtype=dtype)
    state_shape = (
        (num_slots, kernel_size - 1, 1, dim)
        if use_4d_state
        else (num_slots, kernel_size - 1, dim)
    )
    conv_state = jax.random.normal(k2, state_shape, dtype=jnp.float32)
    weight = jax.random.normal(k3, (dim, 1, kernel_size), dtype=dtype)
    bias = jax.random.normal(k4, (dim,), dtype=dtype)

    if batch_size == 4:
      # Small-batch (`batch_size <= 8`) decode bucket with `tile_size=64`
      # exercises `num_sublanes * packing` (`16` for bf16 on v6e) alignment.
      query_start_loc = jnp.array([0, 1, 2, 3, 3], dtype=jnp.int32)
      state_indices = jnp.array([3, -1, 5, 99], dtype=jnp.int32)
      has_initial_state = jnp.array([True, True, False, True], dtype=bool)
      distribution = jnp.array([3, 3, 3], dtype=jnp.int32)
      active_tokens = 3
      active_slots = (3, 5)
      untouched_slots = (0, 1, 2, 4)
    else:
      query_start_loc = jnp.array([0, 1, 2, 18, 31, 37], dtype=jnp.int32)
      state_indices = jnp.array([4, -1, 2, 5, 99], dtype=jnp.int32)
      has_initial_state = jnp.array([True, True, True, False, True], dtype=bool)
      distribution = jnp.array([2, 2, 4], dtype=jnp.int32)
      active_tokens = 31
      active_slots = (4, 2, 5)
      untouched_slots = (0, 1, 3)

    expected_out, expected_state = (
        ragged_causal_conv1d.ragged_causal_conv1d_jax(
            x,
            conv_state,
            weight,
            bias,
            query_start_loc,
            state_indices,
            has_initial_state,
            distribution,
            activation=activation,
        )
    )
    conv_state_orig = np.array(conv_state, copy=True)

    actual_out, actual_state = ragged_causal_conv1d.ragged_causal_conv1d(
        x,
        conv_state,
        weight,
        bias,
        query_start_loc,
        state_indices,
        has_initial_state,
        distribution,
        activation=activation,
        tile_size=tile_size,
    )

    tol = 2e-2 if dtype == jnp.bfloat16 else 1e-5
    np.testing.assert_allclose(
        np.asarray(actual_out[:active_tokens].astype(jnp.float32)),
        np.asarray(expected_out[:active_tokens].astype(jnp.float32)),
        rtol=tol,
        atol=tol,
    )
    for slot in active_slots:
      np.testing.assert_allclose(
          np.asarray(actual_state[slot]),
          np.asarray(expected_state[slot]),
          rtol=tol,
          atol=tol,
      )
    for untouched in untouched_slots:
      np.testing.assert_array_equal(
          np.asarray(actual_state[untouched]), conv_state_orig[untouched]
      )

  def test_uninitialized_vmem_nan_robustness_and_lane_check_tpu(self):
    if jax.default_backend() != "tpu":
      self.skipTest("TPU hardware execution requires TPU backend.")

    # Non-lane-aligned `dim` on TPU must raise `NotImplementedError`.
    with self.assertRaisesRegex(NotImplementedError, "divisible by"):
      ragged_causal_conv1d.ragged_causal_conv1d(
          jnp.zeros((16, 64), dtype=jnp.bfloat16),
          jnp.zeros((2, 3, 64), dtype=jnp.float32),
          jnp.zeros((64, 1, 4), dtype=jnp.bfloat16),
      )

    # Case 1: `num_tokens=1 < K_prev=3` inside `batch_size=16` after poisoning
    # VMEM/SMEM with NaNs must produce zero NaNs across ALL `batch_size` rows.
    dim = 128
    kernel_size = 4
    rng = jax.random.key(909)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    x = jax.random.normal(k1, (16, dim), dtype=jnp.bfloat16)
    conv_state = jax.random.normal(
        k2, (4, kernel_size - 1, dim), dtype=jnp.float32
    )
    weight = jax.random.normal(k3, (dim, 1, kernel_size), dtype=jnp.bfloat16)
    bias = jax.random.normal(k4, (dim,), dtype=jnp.bfloat16)

    mosaic_tpu.poison_tpu_memory()
    qsl_1 = jnp.array([0, 1, 1], dtype=jnp.int32)
    slots_1 = jnp.array([2, 99], dtype=jnp.int32)
    has_init_1 = jnp.array([False, True], dtype=bool)
    dist_1 = jnp.array([1, 1, 5], dtype=jnp.int32)
    out_1, state_1 = ragged_causal_conv1d.ragged_causal_conv1d(
        x,
        conv_state,
        weight,
        bias,
        qsl_1,
        slots_1,
        has_init_1,
        dist_1,
        activation="silu",
        tile_size=16,
    )
    self.assertFalse(bool(jnp.any(jnp.isnan(out_1))))
    self.assertFalse(bool(jnp.any(jnp.isnan(state_1))))
    ref_out_1, ref_state_1 = ragged_causal_conv1d.ragged_causal_conv1d_jax(
        x,
        conv_state,
        weight,
        bias,
        qsl_1,
        slots_1,
        has_init_1,
        dist_1,
        activation="silu",
    )
    np.testing.assert_allclose(
        np.asarray(out_1[:1].astype(jnp.float32)),
        np.asarray(ref_out_1[:1].astype(jnp.float32)),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(state_1), np.asarray(ref_state_1), rtol=2e-2, atol=2e-2
    )

    # Case 2: `num_tokens=2` (`0 < sz_from_old < K_prev` on row 1) with
    # `has_initial_state=True` while TPU memory remains poisoned.
    qsl_2 = jnp.array([0, 2], dtype=jnp.int32)
    slots_2 = jnp.array([1], dtype=jnp.int32)
    has_init_2 = jnp.array([True], dtype=bool)
    dist_2 = jnp.array([0, 1, 1], dtype=jnp.int32)
    out_2, state_2 = ragged_causal_conv1d.ragged_causal_conv1d(
        x,
        conv_state,
        weight,
        bias,
        qsl_2,
        slots_2,
        has_init_2,
        dist_2,
        activation="silu",
        tile_size=16,
    )
    self.assertFalse(bool(jnp.any(jnp.isnan(out_2))))
    self.assertFalse(bool(jnp.any(jnp.isnan(state_2))))
    ref_out_2, ref_state_2 = ragged_causal_conv1d.ragged_causal_conv1d_jax(
        x,
        conv_state,
        weight,
        bias,
        qsl_2,
        slots_2,
        has_init_2,
        dist_2,
        activation="silu",
    )
    np.testing.assert_allclose(
        np.asarray(out_2[:2].astype(jnp.float32)),
        np.asarray(ref_out_2[:2].astype(jnp.float32)),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(state_2), np.asarray(ref_state_2), rtol=2e-2, atol=2e-2
    )


if __name__ == "__main__":
  absltest.main()
