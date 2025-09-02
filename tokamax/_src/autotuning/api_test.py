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
import types
from typing import Any

from absl.testing import absltest
import jax
import jax.numpy as jnp
from tokamax._src import benchmarking
from tokamax._src.autotuning import api
from tokamax._src.autotuning import autotuner
from tokamax._src.ops import op as op_base
from tokamax._src.ops.gated_linear_unit import api as glu_api
from tokamax._src.ops.gated_linear_unit import pallas_triton as pl_glu
from tokamax._src.ops.normalization import api as norm_api
from tokamax._src.ops.normalization import pallas_triton as pl_norm

from tensorflow.compiler.xla.service import hlo_pb2


_HEURISTICS_CONFIG = object()


class _FakeOp(op_base.Op[Any, jax.Array, types.NoneType, object, Any]):

  def _fwd(self, x, y, *, return_residuals, config):
    ...

  def _get_heuristics_config(self, ba: op_base.BoundArguments) -> object:
    return _HEURISTICS_CONFIG


@jax.jit
def tokamax_norm_and_glu(x, scale, offset, weights):
  norm_x = pl_norm.PallasTritonNormalization()(x, scale, offset)
  glu_x = pl_glu.PallasTritonGatedLinearUnit()(
      x=norm_x, weights=weights, activation=jax.nn.swish
  )
  return jnp.sum(glu_x)


def get_lowered_norm_and_glu(x_shape):
  (key1, key2, key3, key4) = jax.random.split(jax.random.PRNGKey(0), 4)
  param_shape = (x_shape[-1],)
  x = jax.random.normal(key=key1, shape=x_shape, dtype=jnp.bfloat16)
  scale = jax.random.normal(key=key2, shape=param_shape, dtype=jnp.bfloat16)
  offset = jax.random.normal(key=key3, shape=param_shape, dtype=jnp.bfloat16)
  weights = jax.random.normal(
      key=key4, shape=(x_shape[-1], 2, x_shape[-1]), dtype=jnp.bfloat16
  )

  f_lowered = tokamax_norm_and_glu.lower(x, scale, offset, weights)
  return f_lowered


def get_expected_bound_args(x_shape):
  return (
      pl_norm.PallasTritonNormalization().bind(  # pytype: disable=wrong-arg-types
          x=jax.ShapeDtypeStruct(x_shape, dtype=jnp.bfloat16),
          scale=jax.ShapeDtypeStruct((x_shape[-1],), dtype=jnp.bfloat16),
          offset=jax.ShapeDtypeStruct((x_shape[-1],), dtype=jnp.bfloat16),
      ),
      pl_glu.PallasTritonGatedLinearUnit().bind(  # pytype: disable=wrong-arg-types
          x=jax.ShapeDtypeStruct(x_shape, dtype=jnp.bfloat16),
          weights=jax.ShapeDtypeStruct(
              (x_shape[-1], 2, x_shape[-1]), dtype=jnp.bfloat16
          ),
          activation=jax.nn.swish,
      ),
  )


# TODO: Make autotuning work for both GPU and TPU.
class AutotuningTest(absltest.TestCase):

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

  def test_get_bound_args_from_lowered(self):
    x_shape = (64, 128)
    expected = get_expected_bound_args(x_shape)
    f_lowered = get_lowered_norm_and_glu(x_shape)
    self.assertEqual(api.get_bound_args(f_lowered), expected)

  def test_get_bound_args_from_hlo(self):
    x_shape = (64, 128)
    expected = get_expected_bound_args(x_shape)
    f_lowered = get_lowered_norm_and_glu(x_shape)
    hlo_modules = f_lowered.compile().runtime_executable().hlo_modules()
    hlo_modules = [
        hlo_pb2.HloModuleProto.FromString(hlo.as_serialized_hlo_module_proto())
        for hlo in hlo_modules
    ]

    # Replicate the HLO modules multiple times to ensure that the bound args are
    # unique.
    hlo_modules = hlo_modules * 10
    self.assertEqual(api.get_bound_args(hlo_modules), expected)

  def test_autotune(self):
    x_shape = (64, 128)
    expected_bound_args = get_expected_bound_args(x_shape)
    f_lowered = get_lowered_norm_and_glu(x_shape)
    bound_args = api.get_bound_args(f_lowered)
    result = api.autotune(bound_args, all_implementations=False)
    self.assertEqual(result.device_kind, jax.devices()[0].device_kind)
    self.assertEqual(tuple(x[0] for x in result.data), expected_bound_args)

    result_all_impls = api.autotune(api.get_bound_args(f_lowered))
    self.assertGreaterEqual(len(result_all_impls.data), len(result.data))

    tempfile = self.create_tempfile("autotuning_results.json")
    with open(tempfile.full_path, "w") as f:
      result_all_impls.dump(f)
    with open(tempfile.full_path, "r") as f:
      self.assertEqual(result_all_impls, api.AutotuningResult.load(f))

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
    config0 = object()
    config1 = object()
    config2 = object()
    config3 = object()
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


if __name__ == "__main__":
  absltest.main()
