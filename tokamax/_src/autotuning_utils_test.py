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


if __name__ == '__main__':
  absltest.main()
