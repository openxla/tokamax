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
import functools
import json

from absl import flags
from absl import logging
from absl.testing import absltest
import jax
import jax.numpy as jnp
from tokamax._src import serialization
from tokamax._src.autotuning import api as autotuning_api
from tokamax._src.ops.gated_linear_unit import pallas_triton as pl_glu
from tokamax._src.ops.normalization import pallas_triton as pl_norm

from tensorflow.compiler.xla.service import hlo_pb2


@jax.jit
def tokamax_norm_and_glu(x, scale, offset, weights):
  pt_normalization = functools.partial(
      pl_norm.PallasTritonNormalization(), axis=(-1)
  )
  normalized_x = pt_normalization(x, scale, offset)
  glu_x = pl_glu.PallasTritonGatedLinearUnit()(
      x=normalized_x, weights=weights, activation=jax.nn.swish
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


class AutotuningTest(absltest.TestCase):

  def test_get_bound_args_from_lowered(self):
    x_shape = (64, 128)
    f_lowered = get_lowered_norm_and_glu(x_shape)

    bound_args = autotuning_api.get_bound_args(f_lowered)
    self.assertLen(bound_args, 2)
    self.assertIsInstance(bound_args[0].op, pl_norm.PallasTritonNormalization)
    self.assertIsInstance(bound_args[1].op, pl_glu.PallasTritonGatedLinearUnit)

    # Test Serialization/Deserialization round trip.
    for bound_arg in bound_args:
      # TODO For GLU, we need to remove the activation argument from the
      # bound args because jitted JAX functions cannot be serialized.
      if "activation" in bound_arg.arguments:
        del bound_arg.arguments["activation"]
      dump_str = json.dumps(bound_arg, cls=serialization.JsonEncoder)
      loaded_bound_arg = json.loads(dump_str, cls=serialization.JsonDecoder)
      self.maxDiff = None
      self.assertEqual(bound_arg.op, loaded_bound_arg.op)
      self.assertEqual(bound_arg.arguments, loaded_bound_arg.arguments)

  def test_get_bound_args(self):
    x_shape = (64, 128)
    f_lowered = get_lowered_norm_and_glu(x_shape)

    hlo_modules = f_lowered.compile().runtime_executable().hlo_modules()
    hlo_modules = [
        hlo_pb2.HloModuleProto.FromString(hlo.as_serialized_hlo_module_proto())
        for hlo in hlo_modules
    ]

    # Replicate the HLO modules multiple times to ensure that the bound args are
    # unique.
    hlo_modules = hlo_modules * 10
    bound_args = autotuning_api.get_bound_args(hlo_modules)
    self.assertLen(bound_args, 2)
    self.assertIsInstance(bound_args[0].op, pl_norm.PallasTritonNormalization)
    self.assertIsInstance(bound_args[1].op, pl_glu.PallasTritonGatedLinearUnit)

  def test_autotune(self):
    x_shape = (64, 128)
    f_lowered = get_lowered_norm_and_glu(x_shape)

    autotuned_results = autotuning_api.autotune(
        autotuning_api.get_bound_args(f_lowered),
        all_implementations=False,
    )
    self.assertLen(autotuned_results.result, 2)
    self.assertIsInstance(
        autotuned_results.result[0][0].op, pl_norm.PallasTritonNormalization
    )
    self.assertIsInstance(
        autotuned_results.result[1][0].op, pl_glu.PallasTritonGatedLinearUnit
    )

    # TODO For GLU, we need to remove the activation argument from the
    # bound args because jitted JAX functions cannot be serialized.
    if "activation" in autotuned_results.result[1][0].arguments:
      del autotuned_results.result[1][0].arguments["activation"]

    all_api_autotuned_results = autotuning_api.autotune(
        autotuning_api.get_bound_args(f_lowered),
        all_implementations=True,
    )
    self.assertGreaterEqual(
        len(all_api_autotuned_results.result), len(autotuned_results.result)
    )


if __name__ == "__main__":
  absltest.main()
