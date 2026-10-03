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
from unittest import mock

from absl.testing import absltest
import jax
from jax.extend import backend
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src.ops.gated_linear_unit import test_base

try:
  from tokamax._src.ops.gated_linear_unit import triton as triton_glu

  _JAX_TRITON_AVAILABLE = True
except ImportError:
  triton_glu = None  # pyrefly: ignore[assignment]
  _JAX_TRITON_AVAILABLE = False


@absltest.skipIf(not _JAX_TRITON_AVAILABLE, "Requires jax_triton.")
class TritonGatedLinearUnitTest(test_base.GatedLinearUnitTestBase):

  def __init__(self, *args):
    glu_fn = (
        triton_glu.TritonGatedLinearUnit() if triton_glu is not None else None
    )
    super().__init__(*args, glu_fn=glu_fn)

  def setUp(self):
    if not _JAX_TRITON_AVAILABLE:
      self.skipTest("Requires jax_triton.")
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    super().setUp()

  def test_autotuning_search_space(self):
    assert triton_glu is not None
    m = 256
    n = 256
    k = 64
    glu = triton_glu.TritonGatedLinearUnit()
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(0), 2)
    x = jax.random.normal(rng0, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(rng1, (k, 2, n), dtype=jnp.bfloat16)

    glu_bound_args = glu.bind(x, w)
    autotuning_configs = glu_bound_args.autotuning_configs
    self.assertNotEmpty(autotuning_configs)

  def test_configs_fit_shared_memory(self):
    assert triton_glu is not None
    # RTX PRO 4500 Blackwell: 82 SMs, 101376 bytes of shared memory per block.
    device = mock.MagicMock(platform="gpu", compute_capability="12.0")
    device.core_count = 82
    limit = gpu_utils.SMEM_CAPACITY_BYTES["sm_120"]
    glu = triton_glu.TritonGatedLinearUnit()
    x = jnp.zeros((64, 384), jnp.float32)
    w = jnp.zeros((384, 2, 4096), jnp.float32)

    devices = mock.patch.object(jax, "devices", return_value=[device])
    default_device = mock.patch.object(
        backend, "get_default_device", return_value=device
    )
    with devices, default_device:
      bound_args = glu.bind(x, w)
      heuristics_config = bound_args.heuristics_config
      configs = bound_args.autotuning_configs

    self.assertEqual(
        (
            heuristics_config.block_m,
            heuristics_config.block_n,
            heuristics_config.block_k,
            heuristics_config.num_stages,
        ),
        (32, 64, 32, 4),
    )
    self.assertNotEmpty(configs)
    for config in configs:
      self.assertLessEqual(triton_glu._smem_bytes(config, 4), limit)


if __name__ == "__main__":
  absltest.main()
