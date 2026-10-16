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
from typing import cast, override
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import qwix
from tokamax._src import gpu_utils
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu
from tokamax._src.ops.ragged_dot import (
    pallas_mosaic_gpu_kernel_sm100_fp8_quant as sm100_fp8_quant,
)
from tokamax._src.ops.ragged_dot import test_base


def silu(x):
  return x * jax.nn.sigmoid(x)


_CONFIG = pallas_mosaic_gpu.Config(
    block_m=16,
    block_n=128,
    block_k=128,
    num_stages=2,
    split_k=1,
    persistent=True,
    collective=False,
    grid_minor_dim=pallas_mosaic_gpu.common.MatmulDimension.M,
    grid_tile_width=1,
)


class _Sm100QuantTestBase(test_base.RaggedDotTestBase):
  """Shared base class for SM100 8-bit x int4 quantized ragged dot tests."""

  _quant_a_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn
  _default_block_k: int = 128

  def __init__(self, *args):
    op = pallas_mosaic_gpu.PallasMosaicGpuRaggedDot()

    def fn(lhs, rhs, *, config=None, **kwargs):
      if config is None:
        config = test_base.test_config.get()
      if config is None:
        config = dataclasses.replace(_CONFIG, block_k=self._default_block_k)

      expect_supported = gpu_utils.is_sm100()
      lhs_ = jax.eval_shape(quantization.as_array_or_qarray, lhs)
      rhs_ = jax.eval_shape(quantization.as_array_or_qarray, rhs)

      tile_k = rhs_.scale_tile_shape[1] if isinstance(rhs_, qwix.QArray) else 0
      config = dataclasses.replace(
          config,
          block_k=min(config.block_k, max(tile_k, 32)),
      )
      if (
          not isinstance(rhs_, qwix.QArray)
          or (rhs_.qtype != jnp.int4)
          or (rhs_.scale_tile_shape[0] != 1)
          or (rhs_.scale_tile_shape[1] % config.block_k != 0)
          or (rhs_.scale_tile_shape[2] != 1)
      ):
        expect_supported = False

      if isinstance(lhs_, qwix.QArray) and lhs_.qtype != self._quant_a_dtype:
        expect_supported = False
      if (
          self._quant_a_dtype == jnp.int8
          and not gpu_utils.has_tcgen05_int8_support()
      ):
        expect_supported = False

      if expect_supported:
        return op.replace(config=config)(lhs, rhs, **kwargs)

      with self.assertRaises(NotImplementedError) as e:
        _ = op.replace(config=config)(lhs, rhs, **kwargs)
      self.skipTest(f"Test not supported: {e.exception}")

    super().__init__(*args, dot_fn=fn)

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    if not gpu_utils.is_sm100():
      self.skipTest("Not supported on non-sm100 GPUs.")
    super().setUp()

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
    self.skipTest("Not supported.")

  @override
  def _test_preferred_element_type(self, out_type):
    self.skipTest("Not supported.")

  @override
  def _test_vjp(self, num_groups, m, k, n, activation=None):
    self.skipTest("Not supported.")

  @override
  def _test_bench(self, spec):
    self.skipTest("Not supported.")

  @override
  def _test_simple(self, dtype):
    self.skipTest("Not supported.")

  @override
  def test_padded(self):
    num_groups, m, k, n = 8, 1024, 128, 256
    a, b, group_sizes = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        random_groups=True,
        use_as_qarray=True,
        quant_a_dtype=self._quant_a_dtype,
        a_tile_shape=(1, 128),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, 128, 1),
    )
    expected = test_base.ref(a, b, group_sizes)
    actual = self._dot_fn(a, b, group_sizes=group_sizes, activation=None)
    count = sum(group_sizes)
    chex.assert_trees_all_close(
        actual[:count], expected[:count], atol=0.01, rtol=0.005
    )

  @override
  def test_group_sizes(self):
    num_groups, m, k, n = 8, 1024, 128, 256
    a, b, group_sizes = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        random_groups=False,
        use_as_qarray=True,
        quant_a_dtype=self._quant_a_dtype,
        a_tile_shape=(1, 128),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, 128, 1),
    )
    expected = test_base.ref(a, b, group_sizes=group_sizes)
    group_sizes = base.GroupSizes(group_sizes, (1,) * num_groups)
    actual = self._dot_fn(a, b, group_sizes=group_sizes, activation=None)
    chex.assert_trees_all_close(actual, expected, atol=0.01, rtol=0.005)

  @override
  def test_zero_group_sizes(self):
    num_groups, m, k, n = 8, 1024, 512, 256
    a, b, group_sizes = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        random_groups=True,
        use_as_qarray=True,
        quant_a_dtype=self._quant_a_dtype,
        a_tile_shape=(1, 256),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, 256, 1),
    )

    for i in range(2**num_groups):
      group_sizes_ = jnp.where(jnp.unpackbits(jnp.uint8(i)), group_sizes, 0)
      with self.subTest(f"group_sizes={group_sizes_.tolist()}"):
        expected = test_base.ref(a, b, group_sizes_)
        actual = self._dot_fn(a, b, group_sizes=group_sizes_, activation=None)
        count = sum(group_sizes_)
        chex.assert_trees_all_close(
            actual[:count], expected[:count], atol=0.01, rtol=0.005
        )


class PallasMosaicGpuKernelSm100FP8QuantTest(_Sm100QuantTestBase):
  """Tests for Pallas Mosaic GPU kernel with fp8xi4 quantization."""

  _quant_a_dtype = jnp.float8_e4m3fn
  _default_block_k = 128

  @parameterized.product(
      subchannels=(512, 256, 128),
      use_as_qarray=(True, False),
      activation=(None, test_base.relu, jax.nn.tanh),
      task=(
          (8, 512, 512, 512),
          (16, 1024, 1024, 1024),
          (16, 2048, 1024, 1024),
      ),
      block_m=(16, 32),
      block_k=(512, 256, 128),
  )
  def test_wi4_afp8_quantized(
      self, subchannels, use_as_qarray, activation, task, block_m, block_k
  ):
    if subchannels < block_k:
      self.skipTest("subchannels < block_k")
    config = dataclasses.replace(_CONFIG, block_m=block_m, block_k=block_k)
    with test_base.ConfigManager(config):
      test_base.RaggedDotTestBase._test_quantized(
          self,
          "float8_e4m3fn",
          "int4",
          (1, subchannels),
          (1, subchannels, 1),
          use_as_qarray,
          activation,
          task,
      )

  def _dequant(self, qarray: qwix.QArray, subchannel: int) -> jax.Array:
    q = qarray.qvalue.astype(jnp.float32)
    s = jnp.repeat(qarray.scale.astype(jnp.float32), subchannel, axis=1)
    return q * s

  @parameterized.product(
      block_k=(128,), activation=(None, test_base.relu, silu)
  )
  def test_epilogue_quant(self, block_k, activation):
    num_groups, m, k, n = 8, 512, 256, 512
    sub = 128
    a, b, group_sizes = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, sub),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, sub, 1),
    )
    config = dataclasses.replace(
        _CONFIG,
        block_m=16,
        block_n=128,
        block_k=block_k,
        epilogue_quant_qtype=jnp.float8_e4m3fn,
        epilogue_quant_subchannel_size=sub,
    )
    out = sm100_fp8_quant.ragged_dot_gpu_fp8_quant_blackwell_kernel(
        a, b, jnp.asarray(group_sizes), jnp.bfloat16, config, activation
    )
    self.assertIsInstance(out, qwix.QArray)
    self.assertEqual(out.qvalue.dtype, jnp.float8_e4m3fn)
    actual = self._dequant(out, sub)
    expected = test_base.ref(a, b, group_sizes, activation)
    count = int(sum(group_sizes))
    chex.assert_trees_all_close(
        actual[:count], expected[:count], atol=0.06, rtol=0.1
    )
    out2 = sm100_fp8_quant.ragged_dot_gpu_fp8_quant_blackwell_kernel(
        a, b, jnp.asarray(group_sizes), jnp.bfloat16, config, activation
    )
    self.assertIsInstance(out2, qwix.QArray)
    chex.assert_trees_all_equal(out.qvalue, out2.qvalue)
    chex.assert_trees_all_equal(out.scale, out2.scale)

  @parameterized.product(block_k=(128,), activation=(None, test_base.relu))
  def test_relaxed_activation_subchannel(self, block_k, activation):
    num_groups, m, k, n = 8, 512, 512, 512
    a, b, group_sizes = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, 128),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, 512, 1),
    )
    config = dataclasses.replace(
        _CONFIG, block_m=16, block_n=128, block_k=block_k
    )
    out = sm100_fp8_quant.ragged_dot_gpu_fp8_quant_blackwell_kernel(
        a, b, jnp.asarray(group_sizes), jnp.bfloat16, config, activation
    )
    expected = test_base.ref(a, b, group_sizes, activation)
    count = int(sum(group_sizes))
    chex.assert_trees_all_close(
        out[:count], expected[:count], atol=0.01, rtol=0.005
    )

  def _offline_permute_inputs(self, m=512):
    num_groups, k, n = 8, 1024, 512
    a, b, group_sizes = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, 512),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, 512, 1),
    )
    b_permuted = dataclasses.replace(
        b, qvalue=mgpu_lib.apply_int4_to_fp8_offline_permute(b.qvalue, axis=1)
    )
    return a, b, b_permuted, jnp.asarray(group_sizes)

  @parameterized.named_parameters(
      dict(testcase_name="decode"),
      dict(testcase_name="decode_silu", activation=silu),
      # Uneven and empty groups, so tiles straddle group boundaries.
      dict(testcase_name="ragged", group_sizes=(0, 130, 0, 200, 0, 98, 84, 0)),
      # m >= 8192 selects the prefill warpgroup config.
      dict(testcase_name="prefill_collective", m=8192, collective=True),
  )
  def test_int4_to_fp8_offline_permute_kernel(
      self, m=512, group_sizes=None, activation=None, collective=False
  ):
    a, b, b_permuted, default_group_sizes = self._offline_permute_inputs(m)
    if group_sizes is None:
      group_sizes = default_group_sizes
    else:
      group_sizes = jnp.array(group_sizes, jnp.int32)
    config = dataclasses.replace(
        _CONFIG, block_m=32 if collective else 16, collective=collective
    )
    kernel = sm100_fp8_quant.ragged_dot_gpu_fp8_quant_blackwell_kernel
    expected = kernel(a, b, group_sizes, jnp.bfloat16, config, activation)
    actual = kernel(
        a,
        b_permuted,
        group_sizes,
        jnp.bfloat16,
        config,
        activation,
        int4_to_fp8_offline_permute=True,
    )
    chex.assert_trees_all_equal(actual, expected)
    # Negative control: permuted weights without the flag must differ.
    wrong = kernel(a, b_permuted, group_sizes, jnp.bfloat16, config, activation)
    self.assertFalse(
        bool(jnp.array_equal(cast(jax.Array, wrong), cast(jax.Array, expected)))
    )

  def test_int4_to_fp8_offline_permute_op(self):
    a, b, b_permuted, group_sizes = self._offline_permute_inputs()
    expected = pallas_mosaic_gpu.PallasMosaicGpuRaggedDot()(
        a, b, group_sizes=group_sizes
    )
    actual = pallas_mosaic_gpu.PallasMosaicGpuRaggedDot(
        int4_to_fp8_offline_permute=True
    )(a, b_permuted, group_sizes=group_sizes)
    chex.assert_trees_all_equal(actual, expected)

  def test_int4_to_fp8_offline_permute_unsupported_raises(self):
    _, b, _, group_sizes = self._offline_permute_inputs()
    lhs = jnp.ones((512, 1024), jnp.bfloat16)  # Non-fp8 lhs -> other kernel.
    with self.assertRaisesRegex(
        NotImplementedError, "int4_to_fp8_offline_permute"
    ):
      pallas_mosaic_gpu.PallasMosaicGpuRaggedDot(
          int4_to_fp8_offline_permute=True
      )(lhs, b, group_sizes=group_sizes)

  @parameterized.product(
      block_k=(128,),
      activation=(None, test_base.relu),
      block_m=(16, 64),
      group_sizes_pat=(
          (33, 67, 50, 80, 70, 60, 90, 62),
          (1, 7, 199, 1, 100, 50, 151, 3),
          (0, 130, 0, 200, 0, 98, 84, 0),
      ),
  )
  def test_epilogue_quant_ragged(
      self, block_k, activation, block_m, group_sizes_pat
  ):
    num_groups, m, k, n = 8, 512, 256, 512
    sub = 128
    a, b, _ = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, sub),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, sub, 1),
    )
    group_sizes = jnp.array(group_sizes_pat, jnp.int32)
    assert int(group_sizes.sum()) == m, group_sizes_pat
    config = dataclasses.replace(
        _CONFIG,
        block_m=block_m,
        block_n=128,
        block_k=block_k,
        epilogue_quant_qtype=jnp.float8_e4m3fn,
        epilogue_quant_subchannel_size=sub,
    )
    out = sm100_fp8_quant.ragged_dot_gpu_fp8_quant_blackwell_kernel(
        a, b, group_sizes, jnp.bfloat16, config, activation
    )
    self.assertIsInstance(out, qwix.QArray)
    actual = self._dequant(out, sub)
    expected = test_base.ref(a, b, group_sizes, activation)
    count = int(group_sizes.sum())
    chex.assert_trees_all_close(
        actual[:count], expected[:count], atol=0.06, rtol=0.1
    )

  @parameterized.product(block_m=(128, 256))
  def test_epilogue_quant_ragged_large_block_m(self, block_m):
    num_groups, m, k, n = 8, 1024, 256, 512
    sub = 128
    a, b, _ = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, sub),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, sub, 1),
    )
    group_sizes = jnp.array([130, 1, 200, 99, 150, 7, 250, 187], jnp.int32)
    assert int(group_sizes.sum()) == m
    config = dataclasses.replace(
        _CONFIG,
        block_m=block_m,
        block_n=128,
        block_k=128,
        epilogue_quant_qtype=jnp.float8_e4m3fn,
        epilogue_quant_subchannel_size=sub,
    )
    out = sm100_fp8_quant.ragged_dot_gpu_fp8_quant_blackwell_kernel(
        a, b, group_sizes, jnp.bfloat16, config, None
    )
    self.assertIsInstance(out, qwix.QArray)
    actual = self._dequant(out, sub)
    expected = test_base.ref(a, b, group_sizes, None)
    count = int(group_sizes.sum())
    chex.assert_trees_all_close(
        actual[:count], expected[:count], atol=0.06, rtol=0.1
    )

  @parameterized.product(activation=(None, test_base.relu, silu))
  def test_epilogue_quant_prod_config(self, activation):
    num_groups, m, k, n = 8, 512, 1024, 512
    sub = 512
    a, b, _ = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, sub),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, sub, 1),
    )
    group_sizes = jnp.array([33, 67, 50, 80, 70, 60, 90, 62], jnp.int32)
    assert int(group_sizes.sum()) == m
    config = dataclasses.replace(
        _CONFIG,
        block_m=32,
        block_n=128,
        block_k=256,
        epilogue_quant_qtype=jnp.float8_e4m3fn,
        epilogue_quant_subchannel_size=128,
    )
    out = sm100_fp8_quant.ragged_dot_gpu_fp8_quant_blackwell_kernel(
        a, b, group_sizes, jnp.bfloat16, config, activation
    )
    self.assertIsInstance(out, qwix.QArray)
    actual = self._dequant(out, 128)
    expected = test_base.ref(a, b, group_sizes, activation)
    count = int(group_sizes.sum())
    chex.assert_trees_all_close(
        actual[:count], expected[:count], atol=0.06, rtol=0.1
    )

  @parameterized.product(activation=(None, test_base.relu))
  def test_epilogue_quant_target_config(self, activation):
    num_groups, m, k, n = 8, 512, 1024, 512
    sub = 512
    a, b, _ = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, sub),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, sub, 1),
    )
    group_sizes = jnp.array([33, 67, 50, 80, 70, 60, 90, 62], jnp.int32)
    assert int(group_sizes.sum()) == m
    config = dataclasses.replace(
        _CONFIG,
        block_m=16,
        block_n=128,
        block_k=512,
        epilogue_quant_qtype=jnp.float8_e4m3fn,
        epilogue_quant_subchannel_size=128,
    )
    out = sm100_fp8_quant.ragged_dot_gpu_fp8_quant_blackwell_kernel(
        a, b, group_sizes, jnp.bfloat16, config, activation
    )
    self.assertIsInstance(out, qwix.QArray)
    self.assertEqual(out.qvalue.dtype, jnp.float8_e4m3fn)
    actual = self._dequant(out, 128)
    expected = test_base.ref(a, b, group_sizes, activation)
    count = int(group_sizes.sum())
    chex.assert_trees_all_close(
        actual[:count], expected[:count], atol=0.06, rtol=0.1
    )

  @parameterized.product(activation=(None, test_base.relu, silu))
  def test_chained_two_ragged_dots(self, activation):
    num_groups, m, k, n = 8, 512, 256, 512
    sub = 128
    a, w1, group_sizes = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, sub),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, sub, 1),
    )
    _, w2, _ = self._create_inputs(
        num_groups,
        m,
        n,
        n,
        jnp.bfloat16,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, sub),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, sub, 1),
    )
    gs = jnp.asarray(group_sizes)
    cfg_fused = dataclasses.replace(
        _CONFIG,
        block_m=16,
        block_n=128,
        block_k=128,
        epilogue_quant_qtype=jnp.float8_e4m3fn,
        epilogue_quant_subchannel_size=sub,
    )
    cfg_plain = dataclasses.replace(
        _CONFIG, block_m=16, block_n=128, block_k=128
    )
    kernel = sm100_fp8_quant.ragged_dot_gpu_fp8_quant_blackwell_kernel

    h_fused = kernel(a, w1, gs, jnp.bfloat16, cfg_fused, activation)
    self.assertIsInstance(h_fused, qwix.QArray)
    self.assertEqual(h_fused.qvalue.dtype, jnp.float8_e4m3fn)
    y_fused = kernel(h_fused, w2, gs, jnp.bfloat16, cfg_plain, None)

    h_bf16 = kernel(a, w1, gs, jnp.bfloat16, cfg_plain, activation)
    self.assertIsInstance(h_bf16, jax.Array)
    h_base = qwix.quantize(h_bf16, jnp.float8_e4m3fn, tiled_axes={0: 1, 1: sub})
    y_base = kernel(h_base, w2, gs, jnp.bfloat16, cfg_plain, None)

    count = int(sum(group_sizes))
    fused = jnp.asarray(y_fused[:count], jnp.float32)
    base_out = jnp.asarray(y_base[:count], jnp.float32)
    rel_mae = jnp.abs(fused - base_out).mean() / jnp.maximum(
        jnp.abs(base_out).mean(), 1e-6
    )
    self.assertLess(float(rel_mae), 0.1)

  def test_epilogue_quant_routing_gating(self):
    g, m, k, n = 8, 512, 256, 512
    x = jnp.zeros((m, k), jnp.bfloat16)
    w = jnp.zeros((g, k, n), jnp.bfloat16)
    x_fp8 = quantization.AsQArray(
        x, jnp.float8_e4m3fn, tiled_axes={0: 1, 1: 128}
    )
    w_i4 = quantization.AsQArray(w, jnp.int4, tiled_axes={0: 1, 1: 128, 2: 1})

    op = pallas_mosaic_gpu.PallasMosaicGpuRaggedDot(
        enable_fused_epilogue_quant=True
    )

    def epi_qtype(lhs, rhs, pet):
      ba = pallas_mosaic_gpu.op.BoundArguments(
          op, {"lhs": lhs, "rhs": rhs, "preferred_element_type": pet}
      )
      return op._get_heuristics_config(ba).epilogue_quant_qtype

    self.assertEqual(
        jnp.dtype(epi_qtype(x_fp8, w_i4, jnp.float8_e4m3fn)), jnp.float8_e4m3fn
    )
    self.assertIsNone(epi_qtype(x_fp8, w_i4, jnp.bfloat16))
    self.assertIsNone(epi_qtype(x_fp8, w_i4, None))
    self.assertIsNone(epi_qtype(x, w, jnp.float8_e4m3fn))

    op_disabled = pallas_mosaic_gpu.PallasMosaicGpuRaggedDot()

    def epi_qtype_disabled(lhs, rhs, pet):
      ba = pallas_mosaic_gpu.op.BoundArguments(
          op_disabled, {"lhs": lhs, "rhs": rhs, "preferred_element_type": pet}
      )
      return op_disabled._get_heuristics_config(ba).epilogue_quant_qtype

    self.assertIsNone(epi_qtype_disabled(x_fp8, w_i4, jnp.float8_e4m3fn))

  @parameterized.product(
      use_as_qarray=(True, False),
      task=(
          (16, 2048, 1024, 2048),
          (16, 1024, 1024, 2048),
          (16, 2048, 2048, 1024),
          (16, 1024, 2048, 1024),
      ),
  )
  def test_heuristic_quality_eval_shapes_op(self, use_as_qarray, task):
    a_dtype = jnp.dtype("float8_e4m3fn")
    b_dtype = jnp.dtype("int4")
    num_groups, m, k, n = task
    a, b, group_sizes = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        random_groups=True,
        use_as_qarray=use_as_qarray,
        quant_a_dtype=a_dtype,
        a_tile_shape=(1, 512),
        quant_b_dtype=b_dtype,
        b_tile_shape=(1, 512, 1),
    )
    config = dataclasses.replace(
        _CONFIG,
        block_m=16,
        block_n=128,
        block_k=256,
    )
    with test_base.ConfigManager(config):
      actual = self._dot_fn(a, b, group_sizes=group_sizes, activation=None)
    expected = test_base.ref(a, b, group_sizes, None)
    count = sum(group_sizes)
    chex.assert_trees_all_close(
        actual[:count], expected[:count], atol=0.015, rtol=0.01
    )

  @parameterized.product(
      task=(
          (16, 2048, 1024, 2048),
          (16, 1024, 1024, 2048),
          (16, 2048, 2048, 1024),
          (16, 1024, 2048, 1024),
      ),
  )
  def test_heuristic_quality_eval_shapes_kernel(self, task):
    num_groups, m, k, n = task
    sub = 512
    a, b, group_sizes = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, sub),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, sub, 1),
    )
    config = dataclasses.replace(
        _CONFIG,
        block_m=16,
        block_n=128,
        block_k=256,
        epilogue_quant_qtype=None,
        epilogue_quant_subchannel_size=None,
    )
    out = sm100_fp8_quant.ragged_dot_gpu_fp8_quant_blackwell_kernel(
        a, b, jnp.asarray(group_sizes), jnp.bfloat16, config, None
    )
    self.assertNotIsInstance(out, qwix.QArray)
    expected = test_base.ref(a, b, group_sizes, None)
    count = int(sum(group_sizes))
    chex.assert_trees_all_close(
        out[:count], expected[:count], atol=0.015, rtol=0.01
    )

  @parameterized.product(
      task=(
          (16, 2048, 1024, 2048),
          (16, 1024, 1024, 2048),
          (16, 2048, 2048, 1024),
          (16, 1024, 2048, 1024),
      ),
      block_k=(256, 128),
  )
  def test_heuristic_quality_eval_shapes_bug_reproduction(self, task, block_k):
    num_groups, m, k, n = task
    sub = 512
    a, b, group_sizes = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, sub),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, sub, 1),
    )
    expected = test_base.ref(a, b, group_sizes, None)

    qvalue_f32 = a.qvalue.astype(jnp.float32)
    rowsum = qvalue_f32.reshape(m, -1, sub).sum(-1)
    a_scales = jnp.concatenate([a.scale.astype(jnp.float32), rowsum], axis=1)
    a_fused = qwix.QArray(a.qvalue, a_scales, None, jnp.float8_e4m3fn)

    config = dataclasses.replace(
        _CONFIG,
        block_m=16,
        block_n=128,
        block_k=block_k,
        epilogue_quant_qtype=None,
        epilogue_quant_subchannel_size=None,
    )
    out = sm100_fp8_quant.ragged_dot_gpu_fp8_quant_blackwell_kernel(
        a_fused, b, jnp.asarray(group_sizes), jnp.bfloat16, config, None
    )
    self.assertNotIsInstance(out, qwix.QArray)
    count = int(sum(group_sizes))
    chex.assert_trees_all_close(
        out[:count], expected[:count], atol=0.015, rtol=0.01
    )


class PallasMosaicGpuKernelSm100I8QuantTest(_Sm100QuantTestBase):
  """Tests for Pallas Mosaic GPU kernel with i8xi4 quantization."""

  _quant_a_dtype = jnp.int8
  _default_block_k = 512

  @parameterized.product(
      tile_shape=(
          (1, 512, 1),
          (1, 256, 1),
      ),
      use_as_qarray=(True, False),
      activation=(None, test_base.relu, jax.nn.tanh),
      task=((8, 512, 512, 512),),
  )
  def test_wi4_ai8_quantized(self, tile_shape, use_as_qarray, activation, task):
    test_base.RaggedDotTestBase._test_quantized(
        self,
        "int8",
        "int4",
        tile_shape,
        tile_shape,
        use_as_qarray,
        activation,
        task,
    )


if __name__ == "__main__":
  absltest.main()
