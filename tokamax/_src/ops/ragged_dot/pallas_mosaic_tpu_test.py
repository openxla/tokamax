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
"""Tests for Pallas Mosaic TPU Ragged Dot."""

import functools
from typing import Any
from absl.testing import absltest
from jax import shard_map
from jax.sharding import PartitionSpec as P
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

# TODO: Directly import ManualAxisType JAX is upgraded.
try:
  from jax.sharding import ManualAxisType
except ImportError:
  ManualAxisType = Any

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

  def test_shard_map_manual_axis_type_varying(self):
    # TODO: Remove the skipTest once JAX is upgraded.
    if ManualAxisType is Any:
      self.skipTest("ManualAxisType is not supported.")
    if jax.devices()[0].device_kind.startswith(
        "TPU7x"
    ):
      self.skipTest("Skip the test on TPU v7.")
    num_devices = jax.device_count()
    mesh = jax.make_mesh((num_devices, 1), ("x", "y"))
    # NB: Must not directly run jax.set_mesh in test method.
    # jax.set_mesh(mesh) sets a global mesh for the entire process but never
    # clears it. If this test runs before other tests, the global mesh remains
    # active and causes other tests to fail, while if you only run that failing
    # test, it passes.
    with jax.set_mesh(mesh):
      m, k, n = 256, 128, 128
      num_groups = 1

      lhs = jnp.ones((m, k))
      lhs = jax.device_put(lhs, P("x",))
      rhs = jnp.ones((num_groups, k, n))
      rhs = jax.device_put(rhs, P(None, None, "y"))
      group_sizes = jnp.array([m], jnp.uint32)

      config = pallas_mosaic_tpu.Config(tile_m=128, tile_k=128, tile_n=128)
      op = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(config=config)

      @functools.partial(
          shard_map,
          mesh=mesh,
          in_specs=(P("x", None), P(None, None, "y"), P(None)),
          out_specs=P("x", "y"),
          check_vma=True,
      )
      def f(lhs, rhs, sizes):
        manual_axis_type = ManualAxisType(
            varying={"x", "y"},
        )
        res = op(lhs, rhs, group_sizes=sizes, manual_axis_type=manual_axis_type)
        self.assertEqual(jax.typeof(res).manual_axis_type, manual_axis_type)
        return res

      _, vjpfunc = jax.vjp(
          f,
          lhs,
          rhs,
          group_sizes,
      )
      cotangent = jax.random.normal(
          jax.random.key(0), (m, n), dtype=jnp.float32
      )
      cotangent = jax.device_put(cotangent, P("x", "y"))
      vjpfunc(cotangent)[0].block_until_ready()

  def test_shard_map_manual_axis_type_unreduced(self):
    # TODO: Remove the skipTest once JAX is upgraded.
    if ManualAxisType is Any:
      self.skipTest("ManualAxisType is not supported.")
    num_devices = jax.device_count()
    mesh = jax.make_mesh((num_devices, 1), ("x", "y"))
    with jax.set_mesh(mesh):
      m, k, n = 256, 256, 256
      num_groups = 1

      lhs = jnp.ones((m, k))
      lhs = jax.device_put(lhs, P(None, "x"))
      rhs = jnp.ones((num_groups, k, n))
      rhs = jax.device_put(rhs, P(None, "x", None))
      group_sizes = jnp.array([m], jnp.uint32)

      config = pallas_mosaic_tpu.Config(tile_m=128, tile_k=128, tile_n=128)
      op = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(config=config)

      @jax.jit
      @functools.partial(
          shard_map,
          mesh=mesh,
          in_specs=(P(None, "x"), P(None, "x", None), P(None)),
          out_specs=P(None, None, unreduced={"x"}),
          check_vma=True,
      )
      def f(lhs, rhs, sizes):
        manual_axis_type = ManualAxisType(
            unreduced={"x"},
        )
        res = op(lhs, rhs, group_sizes=sizes, manual_axis_type=manual_axis_type)
        self.assertEqual(jax.typeof(res).manual_axis_type, manual_axis_type)
        return res

      _, vjpfunc = jax.vjp(
          f,
          lhs,
          rhs,
          group_sizes,
      )
      cotangent = jax.random.normal(
          jax.random.key(0), (m, n), dtype=jnp.float32
      )
      cotangent = jax.device_put(cotangent, P(None, None, reduced={"x"}))
      vjpfunc(cotangent)[0].block_until_ready()

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


if __name__ == "__main__":
  absltest.main()
