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
"""Unit tests and performance benchmarks for input_pack relayout kernel."""

import functools
from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import benchmarking
from tokamax._src.ops.experimental.routing_moe.common import arg as arg_lib
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.common import out_format as out_format_lib
from tokamax._src.ops.experimental.routing_moe.common import spec as spec_lib
from tokamax._src.ops.experimental.routing_moe.pack import kernel
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_util

jax.config.parse_flags_with_absl()


_XPROF = flags.DEFINE_bool(
    "xprof",
    False,
    "Also capture an XProf session per benchmarked function and log its XProf"
    " and Percale links.",
)


def _measure_us(fn, arg, num_evals: int = 10) -> float:
  """Measures average device execution time in microseconds via Xprof."""
  compiled = jax.jit(fn).lower(arg).compile()
  jax.block_until_ready(compiled(arg))
  with benchmarking.XprofProfileSession(hermetic=True) as profile:
    for _ in range(num_evals):
      jax.block_until_ready(compiled(arg))
  if _XPROF.value:
    with helper_util.XprofTrace():
      for _ in range(num_evals):
        jax.block_until_ready(compiled(arg))
  return (profile.total_op_time.total_seconds() * 1e6) / num_evals


def _random_input(
    key: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype
) -> jax.Array:
  """Returns a random array of `dtype`, including integer and fp8 dtypes."""
  if jnp.issubdtype(dtype, jnp.integer):
    info = jnp.iinfo(dtype)
    return jax.random.randint(key, shape, info.min, info.max, dtype=dtype)
  return jax.random.normal(key, shape, jnp.float32).astype(dtype)


def _assert_bitwise_equal(actual: jax.Array, expected: jax.Array) -> None:
  actual_np, expected_np = np.asarray(actual), np.asarray(expected)
  assert actual_np.shape == expected_np.shape, (
      actual_np.shape,
      expected_np.shape,
  )
  uint_dtype = np.dtype(f"uint{actual_np.dtype.itemsize * 8}")
  np.testing.assert_array_equal(
      actual_np.view(uint_dtype), expected_np.view(uint_dtype)
  )


def _xla_reshape(arr: jax.Array, out_shape: tuple[int, ...]) -> jax.Array:
  return arr.reshape(out_shape)


_QUANT_OUT_FORMAT = out_format_lib.ActivationFormat(
    dtype=jnp.float8_e4m3fn,
    quant=out_format_lib.Subchannel(block_size=out_format_lib.tile_size),
)


def _split_data_and_scales(
    actual: arg_lib.ActivationWrapper[jax.Array],
) -> tuple[np.ndarray, np.ndarray]:
  """Splits compact `actual` into 2D data and 2D f32 block scales."""
  # Scales are read on the host because a device-side bitcast of the fp8 array
  # rewrites f32 scale bytes that match the fp8 NaN bit pattern.
  actual_np = np.asarray(actual.data)
  num_logical_chunks = actual.spec.num_logical_chunks
  payload = actual_np[:, :num_logical_chunks]
  data = payload.reshape(layout_util.to_large_shape(payload.shape))
  scale_chunks = actual_np[:, num_logical_chunks:]
  # Each lane across the trailing scale chunks stores one block's f32 scale,
  # packed along the `packing` axis.
  scales = np.ascontiguousarray(scale_chunks.transpose(0, 1, 3, 2)).view(
      np.float32
  )
  return data, scales.reshape(data.shape[0], -1)


def _assert_uniform_lane_scales(scale: np.ndarray) -> None:
  """Asserts all 128 lane scales in each row are bitwise identical."""
  np.testing.assert_equal(scale.ndim, 2)
  np.testing.assert_equal(scale.shape[1], 128)
  expected = np.broadcast_to(scale[:, :1], scale.shape)
  np.testing.assert_array_equal(scale, expected)
  scale_u32 = np.ascontiguousarray(scale).view(np.uint32)
  np.testing.assert_array_equal(
      scale_u32, np.broadcast_to(scale_u32[:, :1], scale_u32.shape)
  )


def _assert_dequantizes_to(
    data: np.ndarray,
    scales: np.ndarray,
    x: jax.Array,
    quant: spec_lib.SubchannelQuantSpec,
) -> None:
  """Asserts quantized `data` and `scales` dequantize to `[rows, cols]` `x`."""
  _, cols = x.shape
  block_size = quant.block_size
  num_blocks = (cols + block_size - 1) // block_size
  col_scales = np.repeat(scales[:, :num_blocks], block_size, axis=1)[:, :cols]
  dequant = data.astype(np.float32) * col_scales
  np.testing.assert_allclose(
      dequant, np.asarray(x.astype(jnp.float32)), rtol=0.07, atol=1e-4
  )


class InputPackUnitTest(parameterized.TestCase):

  def setUp(self):
    helper_util.skip_unless_supported(self)
    super().setUp()

  @parameterized.product(
      batch_size=[16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096],
      dim_size=[1024, 4096],
      dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_input_pack_sweep_batch_size_and_dtype(
      self,
      batch_size: int,
      dim_size: int,
      dtype: jnp.dtype,
  ):
    x = jax.random.normal(
        jax.random.key(0), (batch_size, dim_size), dtype=dtype
    )
    expected_shape = layout_util.to_compact_shape(
        (batch_size, dim_size), x.dtype
    )
    expected = x.reshape(expected_shape)

    actual = kernel.input_pack(x)

    self.assertEqual(actual.data.shape, expected_shape)
    self.assertEqual(actual.data.dtype, dtype)
    self.assertIsNone(actual.scale)
    self.assertEqual(actual.spec.row_size, batch_size)
    self.assertEqual(actual.spec.col_size, dim_size)
    np.testing.assert_array_equal(np.asarray(actual.data), np.asarray(expected))

  @parameterized.named_parameters(
      dict(
          testcase_name="custom_tile_size_128_bf16",
          batch_size=512,
          dim_size=2048,
          tile_size=128,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="custom_tile_size_128_f32",
          batch_size=512,
          dim_size=2048,
          tile_size=128,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="large_dim_12288_bf16",
          batch_size=512,
          dim_size=12288,
          tile_size=256,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="large_dim_12288_f32",
          batch_size=512,
          dim_size=12288,
          tile_size=256,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="large_batch_8192_bf16",
          batch_size=8192,
          dim_size=4096,
          tile_size=None,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="large_batch_8192_f32",
          batch_size=8192,
          dim_size=4096,
          tile_size=None,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="non_padded_vmem_stride_dim_256_bf16",
          batch_size=256,
          dim_size=256,
          tile_size=128,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="non_padded_vmem_stride_dim_256_f32",
          batch_size=256,
          dim_size=256,
          tile_size=128,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="remainder_tile_batch_768_tile_512_bf16",
          batch_size=768,
          dim_size=2048,
          tile_size=512,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="remainder_tile_batch_768_tile_512_f32",
          batch_size=768,
          dim_size=2048,
          tile_size=512,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="prime_sublanes_batch_496_auto_bf16",
          batch_size=496,
          dim_size=2048,
          tile_size=None,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="prime_sublanes_batch_496_auto_f32",
          batch_size=496,
          dim_size=2048,
          tile_size=None,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="non_sublane_aligned_batch_10_bf16",
          batch_size=10,
          dim_size=1024,
          tile_size=None,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="non_sublane_aligned_batch_10_f32",
          batch_size=10,
          dim_size=1024,
          tile_size=None,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="non_sublane_aligned_batch_500_bf16",
          batch_size=500,
          dim_size=2048,
          tile_size=256,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="non_sublane_aligned_batch_500_f32",
          batch_size=500,
          dim_size=2048,
          tile_size=256,
          dtype=jnp.float32,
      ),
      # Batches that are not a multiple of `tile_size`: the remaining rows go
      # through the in-kernel tail (no HBM pad/slice), including tails that
      # are not sublane-aligned and tail-only batches.
      dict(
          testcase_name="tail_aligned_1_tile_bf16",
          batch_size=272,
          dim_size=2048,
          tile_size=256,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="tail_aligned_1_tile_f32",
          batch_size=264,
          dim_size=2048,
          tile_size=256,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="tail_aligned_2_tiles_bf16",
          batch_size=528,
          dim_size=2048,
          tile_size=256,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="tail_aligned_4_tiles_f32",
          batch_size=1000,
          dim_size=1024,
          tile_size=224,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="tail_aligned_large_remainder_bf16",
          batch_size=1008,
          dim_size=4096,
          tile_size=256,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="tail_unaligned_3_tiles_bf16",
          batch_size=777,
          dim_size=2048,
          tile_size=256,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="tail_unaligned_3_tiles_f32",
          batch_size=777,
          dim_size=2048,
          tile_size=256,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="tail_unaligned_1_row_bf16",
          batch_size=257,
          dim_size=1024,
          tile_size=128,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="tail_only_batch_3_bf16",
          batch_size=3,
          dim_size=1024,
          tile_size=None,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="tail_only_batch_5_f32",
          batch_size=5,
          dim_size=1024,
          tile_size=None,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name="non_pow2_dim_768_bf16",
          batch_size=333,
          dim_size=768,
          tile_size=None,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="non_pow2_dim_1536_f32",
          batch_size=257,
          dim_size=1536,
          tile_size=128,
          dtype=jnp.float32,
      ),
  )
  def test_input_pack_edge_shapes(
      self,
      batch_size: int,
      dim_size: int,
      tile_size: int | None,
      dtype: jnp.dtype,
  ):
    x = jax.random.normal(
        jax.random.key(1), (batch_size, dim_size), dtype=dtype
    )
    expected_shape = layout_util.to_compact_shape(
        (batch_size, dim_size), x.dtype
    )
    expected = x.reshape(expected_shape)

    actual = kernel.input_pack(
        x, plan=kernel.InputPackPlan(tile_rows=tile_size)
    )

    self.assertEqual(actual.data.shape, expected_shape)
    self.assertEqual(actual.data.dtype, dtype)
    np.testing.assert_array_equal(np.asarray(actual.data), np.asarray(expected))

  @parameterized.product(
      batch_size=[128, 768],
      dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_input_pack_repeated_calls_leave_semaphores_clean(
      self, batch_size: int, dtype: jnp.dtype
  ):
    dim_size = 2048
    x = jax.random.normal(
        jax.random.key(7), (batch_size, dim_size), dtype=dtype
    )
    expected = np.asarray(
        x.reshape(layout_util.to_compact_shape((batch_size, dim_size), x.dtype))
    )

    for _ in range(2):
      actual = kernel.input_pack(
          x, plan=kernel.InputPackPlan(tile_rows=256)
      )
      np.testing.assert_array_equal(np.asarray(actual.data), expected)

  @parameterized.product(
      batch_size=[100, 257, 768, 1000, 1024],
      dtype=[jnp.bfloat16, jnp.float32, jnp.int8, jnp.float8_e4m3fn],
  )
  def test_input_pack_has_no_extra_hbm_copies(
      self, batch_size: int, dtype: jnp.dtype
  ):
    # With the input in XLA's default layout (e.g. a jit parameter), the
    # compiled module should only contain the Pallas kernel, without
    # pad/slice/copy ops that would move the input or output through HBM an
    # extra time.
    dim_size = 1024
    shape = (batch_size, dim_size)
    compiled = (
        jax.jit(lambda x: kernel.input_pack(x).data)
        .lower(jax.ShapeDtypeStruct(shape, dtype))
        .compile()
    )
    hlo = compiled.as_text()
    assert hlo is not None
    for op in ("pad(", "slice(", "copy(", "copy-start("):
      self.assertNotIn(f" {op}", hlo, msg=f"Unexpected `{op}` in:\n{hlo}")

    x = _random_input(jax.random.key(5), shape, dtype)
    expected = x.reshape(layout_util.to_compact_shape(shape, x.dtype))
    _assert_bitwise_equal(compiled(x), expected)

  @parameterized.product(
      batch_size=[1, 3, 5, 7, 17, 32, 100, 257, 512, 1000, 1024, 4128],
      dim_size=[512, 2048],
      tile_size=[None, 128],
      dtype=[jnp.int8, jnp.float8_e4m3fn],
  )
  def test_input_pack_8bit_shapes(
      self, batch_size: int, dim_size: int, tile_size: int | None, dtype
  ):
    # 8-bit inputs may get a 32-row large 2nd-minor HBM tiling when the batch
    # is a multiple of 32, and an 8-row (or smaller) tiling otherwise.
    shape = (batch_size, dim_size)
    x = _random_input(jax.random.key(11), shape, dtype)
    actual = kernel.input_pack(
        x, plan=kernel.InputPackPlan(tile_rows=tile_size)
    )
    expected_shape = layout_util.to_compact_shape(shape, x.dtype)
    self.assertEqual(actual.data.shape, expected_shape)
    self.assertEqual(actual.data.dtype, dtype)

  @parameterized.product(
      batch_size=[8, 16, 32, 64],
      dim_size=[1536, 2048],
      block_size=[128, 256, 512, 1024, 2048],
  )
  def test_input_pack_quantized_block_out(
      self, batch_size: int, dim_size: int, block_size: int
  ):
    shape = (batch_size, dim_size)
    x = _random_input(jax.random.key(11), shape, jnp.bfloat16.dtype)
    out_format = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn,
        quant=out_format_lib.Subchannel(block_size=block_size),
    )

    actual = kernel.input_pack(x, out_format)

    num_blocks = (dim_size + block_size - 1) // block_size
    expected_scale_chunks = (num_blocks + 127) // 128
    self.assertEqual(actual.spec.num_scale_chunks, expected_scale_chunks)
    assert isinstance(actual.spec.quant, spec_lib.SubchannelQuantSpec)
    self.assertEqual(
        actual.spec.quant.is_replicated_scale, block_size == dim_size
    )
    data, scales = _split_data_and_scales(actual)
    if actual.spec.quant.is_replicated_scale:
      _assert_uniform_lane_scales(scales)
    _assert_dequantizes_to(data, scales, x, actual.spec.quant)

  def test_input_pack_quantized_multi_scale_chunks(self):
    # 16896 / 128 = 132 blocks, spanning 2 trailing scale chunks.
    x = _random_input(jax.random.key(13), (32, 16896), jnp.bfloat16.dtype)
    out_format = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn,
        quant=out_format_lib.Subchannel(block_size=128),
    )

    actual = kernel.input_pack(x, out_format)

    self.assertEqual(actual.spec.num_scale_chunks, 2)
    assert isinstance(actual.spec.quant, spec_lib.SubchannelQuantSpec)
    self.assertFalse(actual.spec.quant.is_replicated_scale)
    data, scales = _split_data_and_scales(actual)
    _assert_dequantizes_to(data, scales, x, actual.spec.quant)

  def test_input_pack_rejects_unaligned_quant_block_size(self):
    x = jnp.zeros((32, 1024), dtype=jnp.bfloat16)
    out_format = out_format_lib.ActivationFormat(
        dtype=jnp.float8_e4m3fn,
        quant=out_format_lib.Subchannel(block_size=64),
    )
    with self.assertRaisesRegex(NotImplementedError, "multiple of num_lanes"):
      kernel.input_pack(x, out_format)

  def test_activation_spec_sub_row_quant_col_size(self):
    packing = layout_util.get_packing(jnp.dtype(jnp.float8_e4m3fn))
    # Subchannel allows non-dividing block sizes and packs 128 scales per chunk.
    single_chunk = spec_lib.ActivationSpec(
        row_size=16,
        col_size=2048,
        dtype=jnp.dtype(jnp.float8_e4m3fn),
        quant=spec_lib.SubchannelQuantSpec(
            block_size=300, is_replicated_scale=False
        ),
    )
    self.assertEqual(single_chunk.quant_col_size, 128 * packing)

    two_chunks = spec_lib.ActivationSpec(
        row_size=16,
        col_size=2048,
        dtype=jnp.dtype(jnp.float8_e4m3fn),
        quant=spec_lib.SubchannelQuantSpec(
            block_size=12, is_replicated_scale=False
        ),
    )
    self.assertEqual(two_chunks.quant_col_size, 2 * 128 * packing)

    # Lane-strided sub-row quantization remains unsupported.
    lane_strided = spec_lib.ActivationSpec(
        row_size=16,
        col_size=2048,
        dtype=jnp.dtype(jnp.float8_e4m3fn),
        quant=spec_lib.LaneStridedQuantSpec(block_col_size=512),
    )
    with self.assertRaisesRegex(
        NotImplementedError, "Sub-row LaneStridedQuantSpec"
    ):
      _ = lane_strided.quant_col_size

  @parameterized.product(
      batch_size=[8, 16, 32, 64],
      dim_size=[512, 2048],
  )
  def test_input_pack_quantized_out(self, batch_size: int, dim_size: int):
    shape = (batch_size, dim_size)
    x = _random_input(jax.random.key(11), shape, jnp.bfloat16.dtype)

    actual = kernel.input_pack(x, _QUANT_OUT_FORMAT)

    assert isinstance(actual.spec.quant, spec_lib.SubchannelQuantSpec)
    self.assertTrue(actual.spec.quant.is_replicated_scale)
    data, scales = _split_data_and_scales(actual)
    _assert_uniform_lane_scales(scales)
    _assert_dequantizes_to(data, scales, x, actual.spec.quant)

  @parameterized.product(
      batch_size=[5, 100, 257, 1000],
      tile_size=[None, 64],
      dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_input_pack_quantized_out_edge_shapes(
      self, batch_size: int, tile_size: int | None, dtype: jnp.dtype
  ):
    # Ragged and multi-tile batches, plus an all-zero row whose scale is 0.
    shape = (batch_size, 1024)
    x = _random_input(jax.random.key(11), shape, jnp.dtype(dtype))
    x = x.at[batch_size // 2].set(0)
    # A float32 row maximum of 3.49 gives a scale with a 0xFF byte, which is
    # the fp8 NaN bit pattern.
    x = x.at[1].multiply(0.5).at[1, 0].set(3.49)

    actual = kernel.input_pack(
        x, _QUANT_OUT_FORMAT, plan=kernel.InputPackPlan(tile_rows=tile_size)
    )

    self.assertIsNone(actual.scale)
    self.assertEqual(actual.spec.row_size, batch_size)
    self.assertEqual(actual.spec.col_size, 1024)
    assert isinstance(actual.spec.quant, spec_lib.SubchannelQuantSpec)
    self.assertTrue(actual.spec.quant.is_replicated_scale)
    data, scales = _split_data_and_scales(actual)
    _assert_uniform_lane_scales(scales)
    _assert_dequantizes_to(data, scales, x, actual.spec.quant)

  def test_input_pack_invalid_shapes_raise_value_error(self):
    with self.assertRaisesRegex(ValueError, "Expected 2D input"):
      kernel.input_pack(jnp.zeros((16, 16, 256), dtype=jnp.bfloat16))
    with self.assertRaisesRegex(
        ValueError, "must be divisible by packing \\* num_lanes"
    ):
      kernel.input_pack(jnp.zeros((32, 128), dtype=jnp.bfloat16))
    with self.assertRaisesRegex(
        ValueError, "must be a positive multiple of sublane_size"
    ):
      kernel.input_pack(
          jnp.zeros((64, 1024), dtype=jnp.bfloat16),
          plan=kernel.InputPackPlan(tile_rows=10),
      )
    with self.assertRaisesRegex(ValueError, "num_buffers.*must be positive"):
      kernel.input_pack(
          jnp.zeros((64, 1024), dtype=jnp.bfloat16),
          plan=kernel.InputPackPlan(num_buffers=0),
      )

  @parameterized.product(
      batch_size=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
      dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_benchmark_input_pack_sweep(self, batch_size: int, dtype: jnp.dtype):
    dim_size = 4096
    tile_sizes = [None, 64, 128, 256, 512]
    dtype_name = jnp.dtype(dtype).name
    itemsize = jnp.dtype(dtype).itemsize

    # Input in XLA's default layout, as for a jit parameter.
    x = jax.random.normal(
        jax.random.key(0), (batch_size, dim_size), dtype=dtype
    )

    out_shape = layout_util.to_compact_shape((batch_size, dim_size), x.dtype)
    total_bytes = 2 * batch_size * dim_size * itemsize

    xla_us = _measure_us(
        functools.partial(_xla_reshape, out_shape=out_shape), x
    )
    xla_gbps = (total_bytes / (xla_us * 1e-6)) / 1e9

    tile_results = {}
    for ts in tile_sizes:
      fn_pack = functools.partial(
          kernel.input_pack, plan=kernel.InputPackPlan(tile_rows=ts)
      )
      pack_us = _measure_us(fn_pack, x)
      pack_gbps = (total_bytes / (pack_us * 1e-6)) / 1e9
      tile_results[ts] = (pack_us, pack_gbps)

    logging.info(
        "BENCHMARK dtype=%s batch=%4d dim=%4d | xla_reshape=%.2f us"
        " (%.1f GB/s) | auto=%.2f us (%.1f GB/s) | ts64=%.2f us"
        " (%.1f GB/s) | ts128=%.2f us (%.1f GB/s) | ts256=%.2f us"
        " (%.1f GB/s) | ts512=%.2f us (%.1f GB/s)",
        dtype_name,
        batch_size,
        dim_size,
        xla_us,
        xla_gbps,
        tile_results[None][0],
        tile_results[None][1],
        tile_results[64][0],
        tile_results[64][1],
        tile_results[128][0],
        tile_results[128][1],
        tile_results[256][0],
        tile_results[256][1],
        tile_results[512][0],
        tile_results[512][1],
    )

  @parameterized.product(
      batch_size=[100, 257, 500, 777, 1000, 1500, 2050, 4100, 8200],
      dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_benchmark_input_pack_ragged(self, batch_size: int, dtype: jnp.dtype):
    # Batches that are not a multiple of the tile size. The measured time is
    # the total device time of the jitted function, so it includes any XLA
    # pad/slice/relayout ops around the kernel. The input is in XLA's default
    # layout.
    dim_size = 4096
    dtype_name = jnp.dtype(dtype).name
    itemsize = jnp.dtype(dtype).itemsize
    x = jax.random.normal(
        jax.random.key(0), (batch_size, dim_size), dtype=dtype
    )

    out_shape = layout_util.to_compact_shape((batch_size, dim_size), x.dtype)
    total_bytes = 2 * batch_size * dim_size * itemsize

    xla_us = _measure_us(
        functools.partial(_xla_reshape, out_shape=out_shape), x
    )
    pack_us = _measure_us(lambda x: kernel.input_pack(x).data, x)
    logging.info(
        "BENCHMARK_RAGGED dtype=%s batch=%4d dim=%4d | xla_reshape=%.2f us"
        " (%.1f GB/s) | input_pack=%.2f us (%.1f GB/s)",
        dtype_name,
        batch_size,
        dim_size,
        xla_us,
        (total_bytes / (xla_us * 1e-6)) / 1e9,
        pack_us,
        (total_bytes / (pack_us * 1e-6)) / 1e9,
    )
    # The XLA reshape pads/relayouts ragged batches, so `input_pack` is >= 1.5x
    # faster across these sizes.
    self.assertLess(pack_us, xla_us)

  @parameterized.product(
      batch_size=[
          *[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
          *[100, 257, 500, 777, 1000, 1500, 2050, 4100, 8200],
      ],
      dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_benchmark_input_pack_quantized(
      self, batch_size: int, dtype: jnp.dtype
  ):
    # Same aligned and ragged batches as the unquantized benchmarks. The
    # unquantized `input_pack` of the same input is the baseline.
    dim_size = 4096
    block_sizes = (256, 512, 1024, 4096)
    x = jax.random.normal(
        jax.random.key(0), (batch_size, dim_size), dtype=dtype
    )

    pack_us = _measure_us(lambda x: kernel.input_pack(x).data, x)
    quant_results = {}
    for bs in block_sizes:
      out_fmt = out_format_lib.ActivationFormat(
          dtype=jnp.float8_e4m3fn,
          quant=out_format_lib.Subchannel(block_size=bs),
      )
      fn_quant = functools.partial(
          lambda x, fmt: kernel.input_pack(x, fmt).data, fmt=out_fmt
      )
      quant_results[bs] = _measure_us(fn_quant, x)
    logging.info(
        "BENCHMARK_QUANT dtype=%s batch=%4d dim=%4d | input_pack=%.2f us"
        " | bs256=%.2f us (%.2fx) | bs512=%.2f us (%.2fx)"
        " | bs1024=%.2f us (%.2fx) | bs4096=%.2f us (%.2fx)",
        jnp.dtype(dtype).name,
        batch_size,
        dim_size,
        pack_us,
        quant_results[256],
        quant_results[256] / pack_us,
        quant_results[512],
        quant_results[512] / pack_us,
        quant_results[1024],
        quant_results[1024] / pack_us,
        quant_results[4096],
        quant_results[4096] / pack_us,
    )


if __name__ == "__main__":
  absltest.main()
