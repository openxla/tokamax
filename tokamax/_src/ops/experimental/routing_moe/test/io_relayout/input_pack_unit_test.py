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
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src import benchmarking
from tokamax._src.ops.experimental.routing_moe.common import layout_util
from tokamax._src.ops.experimental.routing_moe.io_relayout import input_pack
from tokamax._src.ops.experimental.routing_moe.test.helper import helper_util

jax.config.parse_flags_with_absl()


def _unpack_kernel(
    in_hbm_ref, out_hbm_ref, in_vmem_ref, out_vmem_ref, sem_ref
):
  """Pallas kernel that loads compact HBM data and unpacks to large layout."""
  num_packed_chunks = in_hbm_ref.shape[1]
  dim_size = out_hbm_ref.shape[1]
  in_vmem_slice = in_vmem_ref.at[:, :num_packed_chunks]
  pltpu.make_async_copy(
      in_hbm_ref,
      in_vmem_slice,
      sem_ref.at[0],
  ).start()
  pltpu.make_async_copy(
      in_vmem_slice,
      in_vmem_slice,
      sem_ref.at[0],
  ).wait()
  out_vmem_ref[...] = layout_util.load_compact_to_large(in_vmem_ref, dim_size)
  pltpu.make_async_copy(
      out_vmem_ref,
      out_hbm_ref,
      sem_ref.at[1],
  ).start()
  pltpu.make_async_copy(
      out_vmem_ref,
      out_vmem_ref,
      sem_ref.at[1],
  ).wait()


def _measure_us(fn, arg, num_evals: int = 10) -> float:
  """Measures average device execution time in microseconds via Xprof."""
  compiled = jax.jit(fn).lower(arg).compile()
  jax.block_until_ready(compiled(arg))
  with benchmarking.XprofProfileSession(hermetic=True) as profile:
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

    actual = input_pack.input_pack(x)

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

    actual = input_pack.input_pack(x, tile_size=tile_size)

    self.assertEqual(actual.data.shape, expected_shape)
    self.assertEqual(actual.data.dtype, dtype)
    np.testing.assert_array_equal(np.asarray(actual.data), np.asarray(expected))

  @parameterized.product(
      batch_size=[1, 7, 17, 33, 100, 255, 257, 333, 1000],
      dim_size=[768, 1536, 2816],
      tile_size=[None, 128],
      dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_input_pack_irregular_shapes(
      self,
      batch_size: int,
      dim_size: int,
      tile_size: int | None,
      dtype: jnp.dtype,
  ):
    x = jax.random.normal(
        jax.random.key(3), (batch_size, dim_size), dtype=dtype
    )
    expected_shape = layout_util.to_compact_shape(
        (batch_size, dim_size), x.dtype
    )
    expected = x.reshape(expected_shape)

    actual = input_pack.input_pack(x, tile_size=tile_size)

    self.assertEqual(actual.data.shape, expected_shape)
    self.assertEqual(actual.data.dtype, dtype)
    np.testing.assert_array_equal(np.asarray(actual.data), np.asarray(expected))

  @parameterized.product(
      dim_size=[1024, 4096],
      dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_input_pack_roundtrips_with_load_compact_to_large(
      self, dim_size: int, dtype: jnp.dtype
  ):
    batch_size = 256
    x = jax.random.normal(
        jax.random.key(42), (batch_size, dim_size), dtype=dtype
    )
    packed = input_pack.input_pack(x, tile_size=128)
    num_packed_chunks = packed.data.shape[1]
    vmem_chunks = layout_util.to_bank_conflict_aware_stride(num_packed_chunks)

    unpacked = pl.kernel(
        _unpack_kernel,
        out_type=jax.ShapeDtypeStruct((batch_size, dim_size), dtype),
        mesh=pltpu.TensorCoreMesh(axis_name="core", num_cores=1),
        scratch_types=(
            pltpu.VMEM(
                (batch_size, vmem_chunks, *packed.data.shape[2:]), dtype
            ),
            pltpu.VMEM((batch_size, dim_size), dtype),
            pltpu.SemaphoreType.DMA((2,)),
        ),
    )(packed.data)

    np.testing.assert_array_equal(np.asarray(unpacked), np.asarray(x))

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
      actual = input_pack.input_pack(x, tile_size=256)
      np.testing.assert_array_equal(np.asarray(actual.data), expected)

  @parameterized.product(
      batch_size=[7, 100, 257, 768, 1000, 1024],
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
        jax.jit(lambda x: input_pack.input_pack(x).data)
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
    actual = input_pack.input_pack(x, tile_size=tile_size)
    expected_shape = layout_util.to_compact_shape(shape, x.dtype)
    self.assertEqual(actual.data.shape, expected_shape)
    self.assertEqual(actual.data.dtype, dtype)
    _assert_bitwise_equal(actual.data, x.reshape(expected_shape))

  def test_input_pack_invalid_shapes_raise_value_error(self):
    with self.assertRaisesRegex(ValueError, "Expected 2D input"):
      input_pack.input_pack(jnp.zeros((16, 16, 256), dtype=jnp.bfloat16))
    with self.assertRaisesRegex(
        ValueError, "must be divisible by packing \\* num_lanes"
    ):
      input_pack.input_pack(jnp.zeros((32, 128), dtype=jnp.bfloat16))
    with self.assertRaisesRegex(
        ValueError, "must be a positive multiple of sublane_size"
    ):
      input_pack.input_pack(
          jnp.zeros((64, 1024), dtype=jnp.bfloat16), tile_size=10
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
      fn_pack = functools.partial(input_pack.input_pack, tile_size=ts)
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
    pack_us = _measure_us(lambda x: input_pack.input_pack(x).data, x)
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


if __name__ == "__main__":
  absltest.main()
