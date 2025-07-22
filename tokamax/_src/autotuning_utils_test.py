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

import collections
import functools

from absl import logging
from absl.testing import absltest
import jax
import jax.numpy as jnp
from tokamax._src import autotuning_utils
from tokamax._src import hlo_utils
from tokamax._src.ops.gated_linear_unit import base as glu_base
from tokamax._src.ops.gated_linear_unit import pallas_triton as pl_glu
from tokamax._src.ops.normalization import base as normalization_base
from tokamax._src.ops.normalization import pallas_triton as pl_norm
from tokamax._src.ops.ragged_dot import pallas_triton as pl_ragged_dot

deque = collections.deque


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


def get_norm_glu_opspecs(x_shape: tuple[int, ...]):
  (key1, key2, key3, key4) = jax.random.split(jax.random.PRNGKey(0), 4)
  param_shape = (x_shape[-1],)
  x = jax.random.normal(key=key1, shape=x_shape, dtype=jnp.bfloat16)
  scale = jax.random.normal(key=key2, shape=param_shape, dtype=jnp.bfloat16)
  offset = jax.random.normal(key=key3, shape=param_shape, dtype=jnp.bfloat16)
  weights = jax.random.normal(
      key=key4, shape=(x_shape[-1], 2, x_shape[-1]), dtype=jnp.bfloat16
  )

  f_lowered = tokamax_norm_and_glu.lower(x, scale, offset, weights)

  return hlo_utils.get_opspecs(f_lowered)


@jax.jit
def ragged_dot_simple(a, b, group_sizes):
  dot_fn = pl_ragged_dot.PallasTritonRaggedDot()
  return dot_fn(a, b, group_sizes=group_sizes)


def get_ragged_dot_opspecs(
    num_groups=8, m=1024, k=128, n=256, dtype=jnp.bfloat16
):
  rng0, rng1 = jax.random.split(jax.random.PRNGKey(0))
  a = jax.random.normal(rng0, (m, k), dtype=dtype)
  b = jax.random.normal(rng1, (num_groups, k, n), dtype=dtype)
  group_sizes = jnp.array([m // num_groups] * num_groups, jnp.uint32)

  f_lowered = ragged_dot_simple.lower(a, b, group_sizes=group_sizes)
  return hlo_utils.get_opspecs(f_lowered)


class AutotuningUtilsTest(absltest.TestCase):

  def test_get_all_api_implementations_with_specs(self):
    # Lower a JAX function to get the opspec, and then fetch all available APIs
    # from the op using the opspec.
    (key1, key2, key3, key4) = jax.random.split(jax.random.PRNGKey(0), 4)
    x_shape = (64, 128)
    param_shape = (x_shape[-1],)
    x = jax.random.normal(key=key1, shape=x_shape, dtype=jnp.bfloat16)
    scale = jax.random.normal(key=key2, shape=param_shape, dtype=jnp.bfloat16)
    offset = jax.random.normal(key=key3, shape=param_shape, dtype=jnp.bfloat16)
    weights = jax.random.normal(
        key=key4, shape=(128, 2, 128), dtype=jnp.bfloat16
    )

    f_lowered = tokamax_norm_and_glu.lower(x, scale, offset, weights)

    op_specs = hlo_utils.get_opspecs(f_lowered)
    norm_api_spec_tuple = (
        autotuning_utils.get_all_api_implementations_with_specs(op_specs[0])
    )

    self.assertLen(norm_api_spec_tuple, 2)
    self.assertIsInstance(
        norm_api_spec_tuple[0].op, normalization_base.Normalization
    )
    self.assertDictEqual(
        norm_api_spec_tuple[0].arguments,
        op_specs[0].arguments,
    )
    self.assertIsInstance(
        norm_api_spec_tuple[1].op, pl_norm.PallasTritonNormalization
    )
    self.assertDictEqual(
        norm_api_spec_tuple[1].arguments,
        op_specs[0].arguments,
    )

    glu_api_spec_tuple = (
        autotuning_utils.get_all_api_implementations_with_specs(op_specs[1])
    )
    self.assertIsInstance(glu_api_spec_tuple[0].op, glu_base.GatedLinearUnit)
    self.assertDictEqual(
        glu_api_spec_tuple[0].arguments,
        op_specs[1].arguments,
    )
    self.assertIsInstance(
        glu_api_spec_tuple[1].op, pl_glu.PallasTritonGatedLinearUnit
    )
    self.assertDictEqual(
        glu_api_spec_tuple[1].arguments,
        op_specs[1].arguments,
    )

  def test_autotuning_ragged_dot(self):
    """Tests that autotuning works for a simple ragged dot case."""
    # TODO: Re-enable this test once the group_sizes can be hashed.
    self.skipTest('group_sizes not hashedable')
    op_specs = get_ragged_dot_opspecs()
    logging.info('op_specs: %s', op_specs)
    _, best_config = autotuning_utils.launch_op_autotuning(op_specs[0])
    logging.info('best_config: %s', best_config)
    self.assertNotEmpty(best_config)

  def test_autotuning_all_captured_ops(self):
    op_specs = get_norm_glu_opspecs(x_shape=(64, 128))
    op_specs_2 = get_norm_glu_opspecs(x_shape=(128, 256))
    # TODO: Revise this test when more Tokamax ops support autotuning.
    norm_api_spec_tuple = (
        autotuning_utils.get_all_api_implementations_with_specs(op_specs[0])
        + autotuning_utils.get_all_api_implementations_with_specs(op_specs_2[0])
    )

    tuned_ops = autotuning_utils.autotune_all_captured_ops(norm_api_spec_tuple)
    self.assertLen(tuned_ops, 4)
    # # From within all_data, there are two normalization API ops.
    self.assertIsInstance(tuned_ops[0][0].op, normalization_base.Normalization)
    self.assertIsInstance(tuned_ops[1][0].op, pl_norm.PallasTritonNormalization)

    # Check that all specs have autotuning data and that the fastest config is
    # greater than 0.
    expected_shapes = deque([
        (64, 128),
        (128, 256),
    ])
    spec, autotuning_data = tuned_ops[1]
    self.assertEqual(spec.arguments['x'].shape, expected_shapes.popleft())
    self.assertNotEmpty(autotuning_data.items())
    fastest_config = autotuning_data.fastest_config
    self.assertGreater(
        autotuning_data[fastest_config].median_evaluation_time_ms, 0
    )

    spec, autotuning_data = tuned_ops[3]
    self.assertEqual(spec.arguments['x'].shape, expected_shapes.popleft())
    self.assertNotEmpty(autotuning_data.items())
    fastest_config = autotuning_data.fastest_config
    self.assertGreater(
        autotuning_data[fastest_config].median_evaluation_time_ms, 0
    )


if __name__ == '__main__':
  absltest.main()
