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
from absl.testing import absltest
import jax
import jax.numpy as jnp
from tokamax._src.ops.gated_linear_unit import pallas_triton as pl_glu
from tokamax._src.ops.gated_linear_unit import test_base


class PallasTritonGatedLinearUnitTest(test_base.GatedLinearUnitTestBase):

  def __init__(self, *args):
    super().__init__(*args, glu_fn=pl_glu.PallasTritonGatedLinearUnit())

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    super().setUp()

  def test_autotuning_search_space(self):
    m = 256
    n = 256
    k = 64
    glu = pl_glu.PallasTritonGatedLinearUnit()
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(0), 2)
    x = jax.random.normal(rng0, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(rng1, (k, 2, n), dtype=jnp.bfloat16)

    glu_bound_args = glu.bind(x, w)
    autotuning_configs = glu_bound_args.autotuning_configs
    self.assertNotEmpty(autotuning_configs)


if __name__ == "__main__":
  absltest.main()
