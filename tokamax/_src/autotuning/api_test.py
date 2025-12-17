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
import functools
import types
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.experimental
from jax.extend import backend
import jax.numpy as jnp
from tokamax._src import batching
from tokamax._src import benchmarking
from tokamax._src.autotuning import api
from tokamax._src.autotuning import autotuner
from tokamax._src.ops import op as op_lib
from tokamax._src.ops.attention import api as attn_api
from tokamax._src.ops.gated_linear_unit import api as glu_api
from tokamax._src.ops.gated_linear_unit import base as glu_base
from tokamax._src.ops.gated_linear_unit import pallas_triton as pl_glu
from tokamax._src.ops.normalization import api as norm_api
from tokamax._src.ops.normalization import pallas_triton as pl_norm
from tokamax._src.ops.ragged_dot import api as ragged_dot_api
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu as pl_ragged_dot_mosaic_tpu
from tokamax._src.ops.ragged_dot import pallas_triton as pl_ragged_dot

from tensorflow.compiler.xla.service import hlo_pb2  # pylint: disable=g-direct-tensorflow-import


@dataclasses.dataclass(frozen=True)
class _FakeOpConfig:
  foo: int


_HEURISTICS_CONFIG = _FakeOpConfig(42)


class _FakeOp(op_lib.Op[Any, jax.Array, types.NoneType, _FakeOpConfig, Any]):

  def _fwd(self, x: jax.Array, y: jax.Array, *, return_residuals: bool, config):
    return x + y, None

  def _get_heuristics_config(self, ba: op_lib.BoundArguments) -> _FakeOpConfig:
    return _HEURISTICS_CONFIG


def get_fn_and_args_and_expected_bound_args(x_shape, vmap=False):
  norm = pl_norm.PallasTritonNormalization()
  glu = pl_glu.PallasTritonGatedLinearUnit()
  eps = 0.32
  act = jax.nn.swish

  def f(x, scale, offset, weights):
    return glu(norm(x, scale, offset, epsilon=eps), weights, activation=act)

  d = x_shape[-1]
  x = jax.ShapeDtypeStruct(x_shape, jnp.bfloat16)
  scale = jax.ShapeDtypeStruct((d,), jnp.bfloat16)
  offset = jax.ShapeDtypeStruct((d,), jnp.bfloat16)
  weights = jax.ShapeDtypeStruct((d, 2, d), jnp.bfloat16)
  args = (x, scale, offset, weights)

  if vmap:
    ax = (0, None, None, None)
    f = jax.vmap(f, in_axes=ax)
    def as_batched(x, a):
      shape = list(x.shape)
      vmap_axis = None if a is None else (a, shape.pop(a))
      return batching.BatchedShapeDtype(shape, x.dtype, (vmap_axis,))
    x, scale, offset, weights = map(as_batched, args, ax)
  expected_bound_args = (
      norm.bind(x, scale, offset, epsilon=eps),  # pytype: disable=wrong-arg-types
      glu.bind(x, weights, activation=act),  # pytype: disable=wrong-arg-types
  )
  return f, args, expected_bound_args


class AutotuningTest(parameterized.TestCase):

  def test_get_op_implementations(self):
    self.assertDictEqual(
        api.get_op_implementations(pl_norm.PallasTritonNormalization()),
        dict(norm_api.IMPLEMENTATIONS),
    )
    self.assertDictEqual(
        api.get_op_implementations(pl_glu.PallasTritonGatedLinearUnit()),
        dict(glu_api.IMPLEMENTATIONS),
    )

    with self.subTest("current_device_only"):
      if jax.default_backend() == "tpu":
        tpu_norm_impls = api.get_op_implementations(
            pl_norm.PallasTritonNormalization(),
            device=backend.get_default_device(),
        )
        self.assertNotIn(pl_norm.PallasTritonNormalization(), tpu_norm_impls)
      elif jax.default_backend() == "gpu":
        ragged_dot_impls = api.get_op_implementations(
            pl_ragged_dot.PallasTritonRaggedDot(),
            device=backend.get_default_device(),
        )
        self.assertNotIn(
            pl_ragged_dot_mosaic_tpu.PallasMosaicTpuRaggedDot(),
            ragged_dot_impls,
        )

  def test_get_bound_args_from_callable(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Currently only supported on GPU.")

    f, args, expected = get_fn_and_args_and_expected_bound_args((64, 128))
    self.assertEqual(api.get_bound_args(f, *args), expected)

  @parameterized.parameters(False, True)
  def test_get_bound_args_from_lowered(self, vmap):
    if jax.default_backend() == "tpu":
      self.skipTest("Currently only supported on GPU.")

    x_shape = (2, 64, 128) if vmap else (64, 128)
    f, args, expected = get_fn_and_args_and_expected_bound_args(x_shape, vmap)
    f_lowered = jax.jit(f).lower(*args)
    actual = api.get_bound_args(f_lowered)
    self.assertEqual(actual, expected)

  def test_get_bound_args_from_hlo(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Currently only supported on GPU.")

    f, args, expected = get_fn_and_args_and_expected_bound_args((64, 128))
    f_lowered = jax.jit(f).lower(*args)
    hlo_modules = f_lowered.compile().runtime_executable().hlo_modules()
    hlo_modules = [
        hlo_pb2.HloModuleProto.FromString(hlo.as_serialized_hlo_module_proto())
        for hlo in hlo_modules
    ]
    # Replicate the HLO modules multiple times to ensure that the bound args are
    # unique.
    self.assertEqual(api.get_bound_args(hlo_modules * 10), expected)

  def test_get_bound_args_unique(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Currently only supported on GPU.")

    def f(x, weights):
      x = glu_api.gated_linear_unit(x, weights, implementation="triton")
      x = glu_api.gated_linear_unit(x, weights, implementation="triton")
      x = glu_api.gated_linear_unit(x, weights, implementation="xla")
      return x

    shapes = dict(
        x=jax.ShapeDtypeStruct((64, 128), dtype=jnp.bfloat16),
        weights=jax.ShapeDtypeStruct((128, 2, 128), dtype=jnp.bfloat16),
    )
    bound_arg0 = pl_glu.PallasTritonGatedLinearUnit().bind(**shapes)  # pytype: disable=wrong-arg-types
    bound_arg1 = glu_base.GatedLinearUnit().bind(**shapes)  # pytype: disable=wrong-arg-types
    assert bound_arg0.autotuning_cache_key == bound_arg1.autotuning_cache_key
    expected = (bound_arg0, bound_arg1)
    f_lowered = jax.jit(f).lower(**shapes)
    self.assertCountEqual(api.get_bound_args(f_lowered), expected)

  def test_get_bound_args_vjp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Currently only supported on GPU.")

    glu = pl_glu.PallasTritonGatedLinearUnit()
    act = jax.nn.swish
    f = functools.partial(glu, activation=act)
    g = jax.value_and_grad(lambda x, weights: f(x, weights).sum())  # pytype: disable=attribute-error

    x_shape = (64, 128)
    d = x_shape[-1]
    x = jax.ShapeDtypeStruct(x_shape, dtype=jnp.bfloat16)
    weights = jax.ShapeDtypeStruct((d, 2, d), dtype=jnp.bfloat16)
    actual = api.get_bound_args(jax.jit(g).lower(x, weights))
    bound_arg = glu.bind(x, weights, activation=act, return_residuals=True)  # pytype: disable=wrong-arg-types
    vjp_bound_arg = glu.vjp.bind(**bound_arg.vjp_arg_spec)  # pytype: disable=attribute-error
    self.assertCountEqual(actual, (bound_arg, vjp_bound_arg))

  def test_autotune(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Currently only supported on GPU.")

    f, args, expected = get_fn_and_args_and_expected_bound_args((64, 128))
    result = api.autotune(f, *args, all_implementations=False)
    self.assertEqual(result.device_kind, jax.devices()[0].device_kind)
    self.assertEqual(tuple(x[0] for x in result.data), expected)

    f_lowered = jax.jit(f).lower(*args)
    result = api.autotune(f_lowered, all_implementations=False)
    self.assertEqual(result.device_kind, jax.devices()[0].device_kind)
    self.assertEqual(tuple(x[0] for x in result.data), expected)

    result = api.autotune(f_lowered)
    self.assertEqual(result.device_kind, jax.devices()[0].device_kind)
    # TODO: Test that we autotune against all implementations.
    self.assertContainsSubset(expected, tuple(x[0] for x in result.data))

    res_round_trip = api.AutotuningResult.loads(result.dumps())
    self.assertEqual(result, res_round_trip)

    tempfile = self.create_tempfile("autotuning_results.json")
    with open(tempfile.full_path, "w") as f:
      result.dump(f)
    with open(tempfile.full_path, "r") as f:
      self.assertEqual(result, api.AutotuningResult.load(f))

  def test_autotuning_result_context(self):
    op = _FakeOp()
    ba = op.bind(jnp.zeros((1, 2)), jnp.zeros((3,)))
    ba2 = op.bind(jnp.zeros((4, 5)), jnp.zeros((6,)))
    device_kind = jax.devices()[0].device_kind
    bmark_data = benchmarking.BenchmarkData(
        compile_time_ms=0.0,
        lower_time_ms=0.0,
        evaluation_times_ms=(0.0,),
        metadata={},
    )
    config0, config1, config2, config3 = map(_FakeOpConfig, range(4))
    data0 = autotuner.AutotuningData({config0: bmark_data})
    data1 = autotuner.AutotuningData({config1: bmark_data})
    data2 = autotuner.AutotuningData({config2: bmark_data})
    data3 = autotuner.AutotuningData({config3: bmark_data})
    op.get_autotuning_cache(device_kind)[ba2.autotuning_cache_key] = data3
    result0 = api.AutotuningResult(device_kind, ((ba, data0), (ba2, data2)))
    result1 = api.AutotuningResult(device_kind, ((ba, data1),))

    orig_data = ba.cached_autotuning_data
    orig_data2 = ba2.cached_autotuning_data
    self.assertEqual(ba.default_config, _HEURISTICS_CONFIG)
    self.assertEqual(ba2.default_config, config3)
    with result0:
      self.assertEqual(ba.cached_autotuning_data, data0)
      self.assertEqual(ba.default_config, config0)
      self.assertEqual(ba2.cached_autotuning_data, data2)
      self.assertEqual(ba2.default_config, config2)
      with result1:
        self.assertEqual(ba.cached_autotuning_data, data1)
        self.assertEqual(ba.default_config, config1)
        self.assertEqual(ba2.cached_autotuning_data, data2)
        self.assertEqual(ba2.default_config, config2)
      self.assertEqual(ba.cached_autotuning_data, data0)
      self.assertEqual(ba.default_config, config0)
      self.assertEqual(ba2.cached_autotuning_data, data2)
      self.assertEqual(ba2.default_config, config2)
    self.assertEqual(ba.cached_autotuning_data, orig_data)
    self.assertEqual(ba.default_config, _HEURISTICS_CONFIG)
    self.assertEqual(ba2.cached_autotuning_data, orig_data2)
    self.assertEqual(ba2.default_config, config3)

    # Test that the two results can be combined.
    combined_result = result0 | result1
    self.assertEqual(combined_result.device_kind, result1.device_kind)
    expected_ba_cache_keys = [
        b.autotuning_cache_key for b, _ in combined_result.data
    ]
    self.assertIn(ba.autotuning_cache_key, expected_ba_cache_keys)
    self.assertIn(ba2.autotuning_cache_key, expected_ba_cache_keys)

    expected_data = ((ba, data0 | data1), (ba2, data2))
    combined_data_1, combined_data_2 = combined_result.data
    self.assertEqual(expected_data[0][1], combined_data_1[1])
    self.assertEqual(expected_data[1][1], combined_data_2[1])

  # TODO: Figure out why this test fails on TPU.
  def test_autotuning_result_context_retraced(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Currently only supported on GPU.")

    op = _FakeOp()
    orig_fwd = op._fwd
    traced_configs = []
    called_configs = []

    def my_fwd(x: jax.Array, y: jax.Array, *, return_residuals: bool, config):
      traced_configs.append(config)
      jax.experimental.io_callback(lambda: called_configs.append(config), None)
      return orig_fwd(x, y, return_residuals=return_residuals, config=config)

    op._fwd = my_fwd
    x = jnp.zeros((1, 2))
    y = jnp.zeros((2,))
    ba = op.bind(x, y)
    device_kind = jax.devices()[0].device_kind
    bmark_data = benchmarking.BenchmarkData(
        compile_time_ms=0.0,
        lower_time_ms=0.0,
        evaluation_times_ms=(0.0,),
        metadata={},
    )
    config0 = _FakeOpConfig(0)
    config1 = _FakeOpConfig(1)
    data0 = autotuner.AutotuningData({config0: bmark_data})
    data1 = autotuner.AutotuningData({config1: bmark_data})
    result0 = api.AutotuningResult(device_kind, ((ba, data0),))
    result1 = api.AutotuningResult(device_kind, ((ba, data1),))
    f = jax.jit(op)

    # Register context hook before first call, so first and last are identical.
    _ = op_lib.get_autotuning_cache_overlay_state()
    _ = f(x, y)
    with result0:
      _ = f(x, y)
      with result1:
        _ = f(x, y)
      _ = f(x, y)
    _ = f(x, y)
    self.assertEqual(traced_configs, [_HEURISTICS_CONFIG, config0, config1])
    self.assertEqual(
        called_configs,
        [_HEURISTICS_CONFIG, config0, config1, config0, _HEURISTICS_CONFIG],
    )

  def test_ragged_dot_autotuning(self):
    """Tests for autotuning of ragged dot. Works on both TPU and GPU."""

    def _get_ragged_dot_input_data(num_experts, m, k, n, dtype=jnp.bfloat16):
      rng0, rng1 = jax.random.split(jax.random.PRNGKey(0))
      lhs = jax.random.normal(rng0, (m, k), dtype=dtype)
      rhs = jax.random.normal(rng1, (num_experts, k, n), dtype=dtype)
      group_sizes = jnp.array([m // num_experts] * num_experts, jnp.uint32)
      return (lhs, rhs, group_sizes)

    @jax.jit
    def f(lhs, rhs, group_sizes):
      return ragged_dot_api.ragged_dot(
          lhs,
          rhs,
          group_sizes,
          implementation="mosaic",
          preferred_element_type=jnp.bfloat16,
      )

    lhs, rhs, group_sizes = _get_ragged_dot_input_data(
        num_experts=8, m=128, k=128, n=128
    )
    ragged_dot_lowered = f.lower(lhs, rhs, group_sizes=group_sizes)
    result = api.autotune(ragged_dot_lowered, all_implementations=False)
    self.assertNotEmpty(result.data)
    for _, data in result.data:
      for _, benchmark in data.items():
        self.assertGreater(benchmark.median_evaluation_time_ms, 0.0)


if __name__ == "__main__":
  absltest.main()
