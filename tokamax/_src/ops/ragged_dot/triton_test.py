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
from typing import Any, override
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src.ops.ragged_dot import test_base

try:
  import jax_triton as jt
except ImportError:
  triton = None  # pyrefly: ignore[assignment]
  _JAX_TRITON_AVAILABLE = False
  _TestTritonRaggedDot: Any = object
else:
  from tokamax._src.ops.ragged_dot import triton
  _JAX_TRITON_AVAILABLE = jt.__version_info__ >= (0, 4, 1)
  del jt

  class _TestTritonRaggedDot(triton.TritonRaggedDot):

    @override
    @functools.wraps(triton.TritonRaggedDot._fwd)
    def _fwd(self, *args, activation=None, **kwargs):
      if activation is test_base.relu:
        activation = jax.nn.relu
      return super()._fwd(*args, activation=activation, **kwargs)


@absltest.skipIf(not _JAX_TRITON_AVAILABLE, "Requires jax_triton >=0.4.1.")
class TritonRaggedDotTest(test_base.RaggedDotTestBase):

  def __init__(self, *args):
    super().__init__(*args, dot_fn=_TestTritonRaggedDot())

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    super().setUp()

  @parameterized.parameters(2, 4)
  def test_split_k(self, split_k):
    assert triton is not None
    config = triton.Config(
        block_m=128,
        block_n=128,
        block_k=32,
        split_k=split_k,
        num_warps=4,
        num_stages=4,
    )
    split_k_dot = _TestTritonRaggedDot(config=config)

    with mock.patch.object(self, "_dot_fn", split_k_dot):
      with test_base.override_chex_args(atol=2e-5):
        self.test_simple1()  # pyrefly: ignore[missing-attribute]

  def test_split_k_quantized(self):
    assert triton is not None
    config = triton.Config(
        block_m=128,
        block_n=128,
        block_k=32,
        split_k=4,
        num_warps=4,
        num_stages=4,
    )
    split_k_dot = _TestTritonRaggedDot(
        config=config, split_k_intermediate_dtype=jnp.float32
    )

    with mock.patch.object(self, "_dot_fn", split_k_dot):
      self.test_quantized0()  # pyrefly: ignore[missing-attribute]

  @override
  def _test_simple(self, dtype):
    with test_base.override_chex_args(atol=1e-6):
      super()._test_simple(dtype)

  @override
  def _test_bench(self, spec):
    xs = jax.tree.leaves((spec["lhs"], spec["rhs"]))
    if any(x.dtype == jnp.float8_e4m3fn for x in xs) and gpu_utils.is_sm80():
      with self.assertRaises(NotImplementedError):
        super()._test_bench(spec)
    else:
      super()._test_bench(spec)


if __name__ == "__main__":
  absltest.main()
