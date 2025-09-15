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
import jax.numpy as jnp
import tokamax
from tokamax._src import batching
from tokamax._src import benchmarking
from tokamax._src.autotuning import api
from tokamax._src.autotuning import autotuner
from tokamax._src.ops import op as op_base
from tokamax._src.ops.attention import api as attention_api
from tokamax._src.ops.gated_linear_unit import api as glu_api
from tokamax._src.ops.gated_linear_unit import base as glu_base
from tokamax._src.ops.gated_linear_unit import pallas_triton as pl_glu
from tokamax._src.ops.normalization import api as norm_api
from tokamax._src.ops.normalization import pallas_triton as pl_norm

from tensorflow.compiler.xla.service import hlo_pb2  # pylint: disable=g-direct-tensorflow-import


@dataclasses.dataclass(frozen=True)
class _FakeOpConfig:
  foo: int


_HEURISTICS_CONFIG = _FakeOpConfig(42)


class _FakeOp(op_base.Op[Any, jax.Array, types.NoneType, _FakeOpConfig, Any]):

  def _fwd(self, x: jax.Array, y: jax.Array, *, return_residuals: bool, config):
    return x + y, None

  def _get_heuristics_config(self, ba: op_base.BoundArguments) -> _FakeOpConfig:
    return _HEURISTICS_CONFIG


def get_lowered_fn_and_expected_bound_args(x_shape, vmap=False):
  eps = 0.32
  activation = jax.nn.swish

  norm = pl_norm.PallasTritonNormalization()
  norm_f = functools.partial(norm, epsilon=eps)
  glu = pl_glu.PallasTritonGatedLinearUnit()
  glu_f = functools.partial(glu, activation=activation)

  if vmap:
    norm_f = jax.vmap(norm, in_axes=(0, None, None))
    glu_f = jax.vmap(glu_f, in_axes=(0, None))

  def f(x, scale, offset, weights):
    return glu_f(norm_f(x, scale, offset), weights)

  d = x_shape[-1]
  array_type = batching.BatchedShapeDtype if vmap else jax.ShapeDtypeStruct
  x_batch = {"vmap_axes": (0,)} if vmap else {}
  other_batch = {"vmap_axes": (None,)} if vmap else {}

  x = array_type(x_shape, dtype=jnp.bfloat16, **x_batch)
  scale = array_type((d,), dtype=jnp.bfloat16, **other_batch)
  offset = array_type((d,), dtype=jnp.bfloat16, **other_batch)
  weights = array_type((d, 2, d), dtype=jnp.bfloat16, **x_batch)

  f_lowered = jax.jit(f).lower(x, scale, offset, weights)

  expected_bound_args = (
      norm.bind(x, scale, offset, epsilon=eps),  # pytype: disable=wrong-arg-types
      glu.bind(x, weights, activation=jax.nn.swish),  # pytype: disable=wrong-arg-types
  )
  return f_lowered, expected_bound_args


# TODO: Make autotuning work for both GPU and TPU.
class AutotuningTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Currently only supported on GPU.")
    super().setUp()

  def test_get_op_implementations(self):
    self.assertDictEqual(
        api.get_op_implementations(pl_norm.PallasTritonNormalization()),
        dict(norm_api.IMPLEMENTATIONS),
    )
    self.assertDictEqual(
        api.get_op_implementations(pl_glu.PallasTritonGatedLinearUnit()),
        dict(glu_api.IMPLEMENTATIONS),
    )

  @parameterized.parameters(False, True)
  def test_get_bound_args_from_lowered(self, vmap):

    # TODO: re-enable once the serialization bug is fixed.
    if vmap:
      self.skipTest("Vmap serialization is currently broken.")

    x_shape = (64, 128)
    batch_size = 2
    if vmap:
      x_shape = (batch_size,) + x_shape

    f_lowered, expected = get_lowered_fn_and_expected_bound_args(
        x_shape, vmap=vmap
    )
    actual = api.get_bound_args(f_lowered)
    self.assertEqual(actual, expected)

  def test_get_bound_args_from_hlo(self):
    f_lowered, expected = get_lowered_fn_and_expected_bound_args((64, 128))
    hlo_modules = f_lowered.compile().runtime_executable().hlo_modules()
    hlo_modules = [
        hlo_pb2.HloModuleProto.FromString(hlo.as_serialized_hlo_module_proto())
        for hlo in hlo_modules
    ]
    # Replicate the HLO modules multiple times to ensure that the bound args are
    # unique.
    self.assertEqual(api.get_bound_args(hlo_modules * 10), expected)

  def test_get_bound_args_unique(self):
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

  def test_autotune(self):
    f_lowered, expected = get_lowered_fn_and_expected_bound_args((64, 128))
    result = api.autotune(f_lowered, all_implementations=False)
    self.assertEqual(result.device_kind, jax.devices()[0].device_kind)
    self.assertEqual(tuple(x[0] for x in result.data), expected)

    result = api.autotune(f_lowered)
    self.assertEqual(result.device_kind, jax.devices()[0].device_kind)
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

  def test_autotuning_result_context_retraced(self):
    if jax.version.__version_info__ < (0, 7, 2):
      self.skipTest("Requires JAX >= 0.7.2.")

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
    _ = op_base.get_autotuning_cache_overlay_state()
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


if __name__ == "__main__":
  absltest.main()
