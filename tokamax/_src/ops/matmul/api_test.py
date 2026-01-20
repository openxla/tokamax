# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.random as jr
from tokamax._src import gpu_utils
from tokamax._src.autotuning.api import autotune
from tokamax._src.ops.matmul import api
from tokamax._src.ops.matmul import test_base


class MatmulXlaTest(test_base.MatmulTestBase):

  def __init__(self, *args, implementation=None):
    matmul_fn = functools.partial(api.matmul, implementation="xla")
    super().__init__(*args, matmul_fn=matmul_fn)

  def setUp(self):
    # Strange numerical issues on TPU.
    if jax.default_backend() == "tpu":
      self.skipTest("Test disabled on TPU.")
    super().setUp()


class MatmulCuteDslTest(test_base.MatmulTestBase):

  def __init__(self, *args, implementation=None):
    matmul_fn = functools.partial(api.matmul, implementation="cute_dsl")
    super().__init__(*args, matmul_fn=matmul_fn)

  def setUp(self):
    if not (jax.default_backend() == "gpu" and gpu_utils.is_sm100()):
      self.skipTest("Only SM100 GPU is supported.")
    super().setUp()


class MatmulAutotuneTest(parameterized.TestCase):
  def test_autotune(self):
    dtype = jnp.float16
    key0, key1 = jr.split(jr.key(0), 2)
    m, n, k = 2048, 2048, 2048
    a = jr.normal(key0, shape=(m, k), dtype=dtype)
    b = jr.normal(key1, shape=(n, k), dtype=dtype)
    autotune_result = autotune(api.matmul, a, b)
    self.assertNotEmpty(autotune_result.data)

  def setUp(self):
    if not (jax.default_backend() == "gpu" and gpu_utils.is_sm100()):
      self.skipTest("Only SM100 GPU is supported.")
    super().setUp()


if __name__ == "__main__":
  absltest.main()
