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

from typing import override

from absl.testing import absltest
import jax
import jax.numpy as jnp
import qwix
from tokamax._src import gpu_utils
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu
from tokamax._src.ops.ragged_dot import test_base


# A probably suboptimal global config that should work for most shapes. We use
# this just to check that there exists a config that yields correct results.
_CONFIG = pallas_mosaic_gpu.Config(
    block_m=128,
    block_n=64,
    block_k=256,
    num_stages=2,
    split_k=1,
)


def _abstract_quantized(shape, dtype, qtype, tile_shape):
  if tile_shape is None:
    return jax.ShapeDtypeStruct(shape, dtype)
  scale_shape = tuple((d + t - 1) // t for d, t in zip(shape, tile_shape))
  return qwix.QArray(
      qvalue=jax.ShapeDtypeStruct(shape, qtype),  # pyrefly: ignore[bad-argument-type]
      scale=jax.ShapeDtypeStruct(scale_shape, dtype),  # pyrefly: ignore[bad-argument-type]
  )


def _resolve_gpu_support(lhs, rhs, *, config=None, **kwargs):
  """Returns `(expect_supported, skip_reason, config)` from shape/dtype metadata."""
  expect_supported = (lhs.dtype == rhs.dtype == jnp.bfloat16) and (
      lhs.shape[-1] % (128 // jnp.dtype(lhs.dtype).itemsize) == 0
  )

  lhs_ = jax.eval_shape(quantization.as_array_or_qarray, lhs)
  rhs_ = jax.eval_shape(quantization.as_array_or_qarray, rhs)

  device_kind = jax.devices()[0].device_kind.lower()
  if gpu_utils.is_sm100():
    if isinstance(rhs_, qwix.QArray):
      config = pallas_mosaic_gpu.Config(
          block_m=64,
          block_n=128,
          block_k=128,
          num_stages=2,
          split_k=1,
      )
      if (
          rhs_.qtype != jnp.int4
          or rhs_.scale_tile_shape[0] != 1
          or rhs_.scale_tile_shape[1] < config.block_k
          or rhs_.scale_tile_shape[2] != 1
          or rhs_.scale.shape[2] * rhs_.scale.dtype.itemsize <= 16
          or (
              isinstance(lhs_, qwix.QArray)
              and lhs_.qtype == jnp.int8
              and not gpu_utils.has_tcgen05_int8_support()
          )
      ):
        expect_supported = False
    else:
      if kwargs.get("preferred_element_type") not in (None, jnp.bfloat16):
        return False, "f32 output not supported on B200.", config or _CONFIG
      config = pallas_mosaic_gpu.Config(
          block_m=128,
          block_n=128,
          block_k=128,
          num_stages=2,
          split_k=1,
      )
  elif "a100" in device_kind:
    has_custom_dims = "ragged_dot_dimension_numbers" in kwargs
    if has_custom_dims:
      expect_supported = False
    else:
      config = pallas_mosaic_gpu.Config(
          block_m=64,
          block_n=128,
          block_k=64,
          num_stages=2,
          split_k=1,
      )
  elif isinstance(rhs_, qwix.QArray):
    if (
        rhs_.scale_tile_shape[0] != 1
        or rhs_.scale_tile_shape[1] % _CONFIG.block_k != 0
        or rhs_.scale_tile_shape[2] != 1
        or kwargs.get("preferred_element_type") is not None
    ):
      expect_supported = False

  return expect_supported, None, config or _CONFIG


class PallasMosaicGpuRaggedDotTest(test_base.RaggedDotTestBase):

  def __init__(self, *args):
    op = pallas_mosaic_gpu.PallasMosaicGpuRaggedDot()

    def fn(lhs, rhs, *, config=None, **kwargs):
      expect_supported, skip_reason, resolved_config = _resolve_gpu_support(
          lhs, rhs, config=config, **kwargs
      )
      if skip_reason is not None:
        self.skipTest(skip_reason)
      if expect_supported:
        return op.replace(config=resolved_config)(lhs, rhs, **kwargs)

      with self.assertRaises(NotImplementedError) as e:
        _ = op.replace(config=resolved_config)(lhs, rhs, **kwargs)
      self.skipTest(f"Test not supported: {e.msg}")

    super().__init__(*args, dot_fn=fn)

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    device_kind = jax.devices()[0].device_kind.lower()
    test_name = self._testMethodName
    if "a100" in device_kind and "vjp" in test_name:
      self.skipTest("No contracting dim kernel for A100 (drhs gradient).")
    if gpu_utils.is_sm100() and test_name.startswith((
        "test_simple",
        "test_padded",
        "test_zero_group_sizes",
        "test_group_sizes",
        "test_vjp",
    )):
      self.skipTest("f32 output not supported on B200.")
    super().setUp()

  @override
  def _test_simple(self, dtype):
    lhs = jax.ShapeDtypeStruct((1024, 128), dtype)
    rhs = jax.ShapeDtypeStruct((8, 128, 256), dtype)
    supported, reason, _ = _resolve_gpu_support(
        lhs, rhs, preferred_element_type=jnp.float32
    )
    if not supported:
      self.skipTest(reason or "Unsupported dtype.")
    super()._test_simple(dtype)

  @override
  def _test_preferred_element_type(self, out_type):
    lhs = jax.ShapeDtypeStruct((1024, 128), jnp.bfloat16)
    rhs = jax.ShapeDtypeStruct((8, 128, 256), jnp.bfloat16)
    supported, reason, _ = _resolve_gpu_support(
        lhs, rhs, preferred_element_type=out_type
    )
    if not supported:
      self.skipTest(reason or "Unsupported preferred_element_type.")
    super()._test_preferred_element_type(out_type)

  @override
  def _test_quantized(
      self,
      a_dtype,
      b_dtype,
      a_tile_shape,
      b_tile_shape,
      use_as_qarray,
      activation=None,
      task=(8, 512, 256, 512),
  ):
    num_groups, m, k, n = task
    lhs = _abstract_quantized(
        (m, k), jnp.bfloat16, jnp.dtype(a_dtype), a_tile_shape
    )
    rhs = _abstract_quantized(
        (num_groups, k, n), jnp.bfloat16, jnp.dtype(b_dtype), b_tile_shape
    )
    supported, reason, _ = _resolve_gpu_support(lhs, rhs, activation=activation)
    if not supported:
      self.skipTest(reason or "Unsupported quantized configuration.")
    super()._test_quantized(
        a_dtype,
        b_dtype,
        a_tile_shape,
        b_tile_shape,
        use_as_qarray,
        activation,
        task,
    )

  @override
  def _test_bench(self, spec):
    supported, reason, _ = _resolve_gpu_support(**spec)
    if not supported:
      self.skipTest(reason or "Unsupported benchmark spec.")
    super()._test_bench(spec)

  def test_unsupported_config_raises(self):
    op = pallas_mosaic_gpu.PallasMosaicGpuRaggedDot(config=_CONFIG)
    lhs = jnp.ones((64, 64), dtype=jnp.float32)
    rhs = jnp.ones((2, 64, 64), dtype=jnp.float32)
    group_sizes = jnp.array([32, 32], dtype=jnp.int32)
    with self.assertRaises(NotImplementedError):
      _ = op(lhs, rhs, group_sizes=group_sizes)

  def test_get_autotuning_configs(self):
    op = pallas_mosaic_gpu.PallasMosaicGpuRaggedDot()
    lhs = jax.ShapeDtypeStruct((128, 512), dtype=jnp.bfloat16)
    rhs = jax.ShapeDtypeStruct((4, 512, 512), dtype=jnp.bfloat16)
    group_sizes = (32, 32, 32, 32)
    ba = op.bind(lhs, rhs, group_sizes=group_sizes)

    configs = op._get_autotuning_configs(ba)
    self.assertIsNotNone(configs)
    self.assertNotEmpty(configs)


if __name__ == "__main__":
  absltest.main()
