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
"""Tokamax Megablox TPU tests for core functionality."""

from absl.testing import absltest
import chex
import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import qwix
from tokamax._src import mosaic_tpu as common
from tokamax._src import quantization
from tokamax._src.ops import op as op_lib
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu
from tokamax._src.ops.ragged_dot import test_base
from typing_extensions import override


AsQArray = quantization.AsQArray


def _is_scale_tiling_supported(x: qwix.QArray, axis: int) -> bool:
  min_addressable_sizes = (
      [1] * x.ndim
      + [common._adaptive_sublane_size(), pltpu.get_tpu_info().num_lanes]
  )[-x.ndim :]
  cdiv = lambda x, y: (x + y - 1) // y
  eps_list = [cdiv(x, y) for x, y in zip(x.qvalue.shape, x.scale.shape)]
  for ax, (mas, eps) in enumerate(zip(min_addressable_sizes, eps_list)):
    if eps != 1 and eps % mas != 0:
      return False
    if ax != axis and not (eps == 1 or eps == x.qvalue.shape[ax]):
      return False
  return True


def _is_config_supported(
    lhs: jax.Array | qwix.QArray | AsQArray,
    rhs: jax.Array | qwix.QArray | AsQArray,
    config: pallas_mosaic_tpu.Config,
) -> bool:
  (m, k), (_, _, n) = lhs.shape, rhs.shape
  if m < config.tile_m or k < config.tile_k or n < config.tile_n:
    return False

  lhs_ = jax.eval_shape(quantization.as_array_or_qarray, lhs)
  rhs_ = jax.eval_shape(quantization.as_array_or_qarray, rhs)

  if isinstance(lhs_, qwix.QArray) and not _is_scale_tiling_supported(lhs_, 1):
    return False
  if isinstance(rhs_, qwix.QArray) and not _is_scale_tiling_supported(rhs_, 1):
    return False
  return True


# TODO : Add QWIX tests for ragged dot once QWIX is in Ragged Dot.
# TODO: Merge QWIX quantization tests into ragged dot API tests.
# also add shapes which tile sizes do not cleanly divide to test masking.
class PallasMosaicTpuRaggedDotTest(test_base.RaggedDotTestBase):
  """Pallas Mosaic TPU Ragged Dot tests."""

  def __init__(self, *args):

    def fn(lhs, rhs, *, config=None, **kwargs):
      config = config or pallas_mosaic_tpu.Config()
      op = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(config=config)

      # skip unsupported tiling and quantization
      if _is_config_supported(lhs, rhs, config):
        return op(lhs, rhs, **kwargs)

      with self.assertRaises(NotImplementedError) as e:
        _ = op(lhs, rhs, **kwargs)
      self.skipTest(f"Test not supported: {e.msg}")

    super().__init__(*args, dot_fn=fn)

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  def test_vjp0(self):
    with test_base.override_chex_args(atol=0.2, rtol=0.01):
      super().test_vjp0()  # pytype: disable=attribute-error

  @override
  def _test_quantized(
      self,
      a_dtype,
      b_dtype,
      a_tile_shape,
      b_tile_shape,
      use_as_qarray,
      activation=None,
      # (num_groups, m, k, n)
      task=(8, 512, 256, 512),
  ):
    with test_base.override_chex_args(atol=0.4, rtol=0.1):
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
    if "i8xi8" in self._testMethodName:
      kwargs = dict(atol=2.0, rtol=0.5)  # This is really bad!
    elif "i4" in self._testMethodName:
      kwargs = dict(atol=0.7, rtol=0.1)
    else:
      kwargs = {}
    with test_base.override_chex_args(**kwargs):
      super()._test_bench(spec)

  def test_autotuning_configs(self):
    tpu_ragged_dot = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot()
    ba = op_lib.BoundArguments(
        op=tpu_ragged_dot,
        arguments={
            "lhs": jax.ShapeDtypeStruct((262144, 7168), dtype=jnp.bfloat16),
            "rhs": jax.ShapeDtypeStruct((256, 7168, 2048), dtype=jnp.bfloat16),
        },
    )
    autotuning_configs = ba.autotuning_configs
    self.assertGreaterEqual(len(autotuning_configs), 3 * 3 * 3)


class FP8RaggedDotTest(absltest.TestCase):
  """Tests for FP8 block-wise quantization in ragged dot."""

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  def _run_fp8_forward(self, block_size, use_as_qarray):
    num_groups, m, k, n = 8, 512, 256, 512
    rng = jax.random.PRNGKey(42)
    a = jax.random.normal(rng, (m, k), dtype=jnp.bfloat16) * 0.1
    b = jax.random.normal(jax.random.split(rng)[0], (num_groups, k, n), dtype=jnp.bfloat16) * 0.1
    group_sizes = jnp.array([m // num_groups] * num_groups, dtype=jnp.int32)

    # Quantize inputs
    a_tile = {0: 1, 1: block_size}
    b_tile = {0: 1, 1: block_size, 2: 1}
    if use_as_qarray:
      a_q = quantization.AsQArray(a, jnp.float8_e4m3fn, tiled_axes=a_tile)
      b_q = quantization.AsQArray(b, jnp.float8_e4m3fn, tiled_axes=b_tile)
    else:
      a_q = qwix.quantize(a, jnp.float8_e4m3fn, tiled_axes=a_tile)
      b_q = qwix.quantize(b, jnp.float8_e4m3fn, tiled_axes=b_tile)

    # Reference: dequantize and compute
    ref_result = test_base.ref(a_q, b_q, group_sizes)

    # Actual: use FP8 ragged dot
    op = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot()
    actual = op(a_q, b_q, group_sizes=group_sizes, preferred_element_type=jnp.float32)

    count = sum(group_sizes)
    chex.assert_trees_all_close(actual[:count], ref_result[:count], atol=0.5, rtol=0.1)

  def test_fp8_forward_block128(self):
    self._run_fp8_forward(128, use_as_qarray=False)

  def test_fp8_forward_block256(self):
    self._run_fp8_forward(256, use_as_qarray=False)

  def test_fp8_forward_block128_as_qarray(self):
    self._run_fp8_forward(128, use_as_qarray=True)

  def test_fp8_forward_block256_as_qarray(self):
    self._run_fp8_forward(256, use_as_qarray=True)

  def _run_fp8_vjp(self, block_size):
    """Test FP8 block-wise forward + backward with forward-save optimization."""
    num_groups, m, k, n = 4, 512, 256, 256
    rng = jax.random.PRNGKey(42)
    a = jax.random.normal(rng, (m, k), dtype=jnp.bfloat16) * 0.1
    b = jax.random.normal(jax.random.split(rng)[0], (num_groups, k, n), dtype=jnp.bfloat16) * 0.1
    group_sizes = jnp.array([m // num_groups] * num_groups, dtype=jnp.int32)

    # FP8 quantized op
    a_tile = {0: 1, 1: block_size}
    b_tile = {0: 1, 1: block_size, 2: 1}
    a_q = quantization.AsQArray(a, jnp.float8_e4m3fn, tiled_axes=a_tile)
    b_q = quantization.AsQArray(b, jnp.float8_e4m3fn, tiled_axes=b_tile)

    op = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot()
    f = lambda a, b: op(a, b, group_sizes=group_sizes, preferred_element_type=jnp.float32)
    f_ref = lambda a, b: test_base.ref(a, b, group_sizes)

    # Forward
    actual = f(a_q, b_q)
    expected = f_ref(a, b)
    count = sum(group_sizes)
    chex.assert_trees_all_close(actual[:count], expected[:count], atol=1.0, rtol=0.2)

    # Backward (sum to get scalar loss for grad)
    actual_grads = jax.grad(lambda a, b: jnp.sum(f(a, b)), argnums=(0, 1))(a_q, b_q)
    expected_grads = jax.grad(lambda a, b: jnp.sum(f_ref(a, b)), argnums=(0, 1))(a, b)

    chex.assert_trees_all_close(actual_grads[0], expected_grads[0], atol=2.0, rtol=0.3)
    chex.assert_trees_all_close(actual_grads[1], expected_grads[1], atol=2.0, rtol=0.3)

  def test_fp8_vjp_block128(self):
    self._run_fp8_vjp(128)

  def test_fp8_vjp_block256(self):
    self._run_fp8_vjp(256)


if __name__ == "__main__":
  absltest.main()
