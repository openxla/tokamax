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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.ragged_dot import pallas_triton
from tokamax._src.ops.ragged_dot import test_base


class PallasTritonRaggedDotTest(test_base.RaggedDotTestBase):

  def __init__(self, *args):
    super().__init__(*args, dot_fn=pallas_triton.PallasTritonRaggedDot())

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    super().setUp()

  @parameterized.parameters(2, 4)
  def test_split_k(self, split_k):
    config = pallas_triton.Config(
        block_m=128,
        block_n=128,
        block_k=32,
        split_k=split_k,
        num_warps=4,
        num_stages=4,
    )
    split_k_dot = pallas_triton.PallasTritonRaggedDot(config=config)

    with mock.patch.object(self, "_dot_fn", split_k_dot):
      self.test_simple1()  # pytype: disable=attribute-error

  def test_split_k_quantized(self):
    config = pallas_triton.Config(
        block_m=128,
        block_n=128,
        block_k=32,
        split_k=4,
        num_warps=4,
        num_stages=4,
    )
    split_k_dot = pallas_triton.PallasTritonRaggedDot(
        config=config, split_k_intermediate_dtype=jnp.float32
    )

    with mock.patch.object(self, "_dot_fn", split_k_dot):
      self.test_quantized0()  # pytype: disable=attribute-error

  @parameterized.named_parameters(test_base.NAMED_ARG_SPECS.items())
  def test_bench(self, _):
    # TODO: Fix tolerance and enable tests.
    self.skipTest(
        "Accuracy for triton pallas is slightly less than mgpu. We need to"
        " figure out how to fix it or to increase the tolerance."
    )


if __name__ == "__main__":
  absltest.main()
