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
"""Tests for Pallas Mosaic TPU v2 Ragged Dot (GMM v2)."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import qwix
from tokamax._src import quantization
from tokamax._src.ops import op as op_lib
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2
from tokamax._src.ops.ragged_dot import test_base
from typing_extensions import override

AsQArray = quantization.AsQArray


def _is_qarray(x) -> bool:
  return isinstance(x, (qwix.QArray, AsQArray))


def _is_config_supported(
    lhs: jax.Array,
    rhs: jax.Array,
    config: pallas_mosaic_tpu_v2.Config,
) -> bool:
  """Returns whether the v2 heuristic tiling can serve the given shapes."""
  # v2 tiles cannot exceed the corresponding input dimension.
  (m, k), (_, _, n) = lhs.shape, rhs.shape
  return m >= config.tile_m and k >= config.tile_k and n >= config.tile_n


class PallasMosaicTpuV2RaggedDotTest(test_base.RaggedDotTestBase):
  """Standard correctness for the GMM v2 wrapper via `RaggedDotTestBase`."""

  def __init__(self, *args):

    def fn(lhs, rhs, *, config=None, **kwargs):
      # The v2 wrapper accepts only raw arrays; quantization flows through the
      # `rhs_scale`/`rhs_bias` API kwargs instead.
      if _is_qarray(lhs) or _is_qarray(rhs):
        self.skipTest("v2 wrapper does not accept QArray.")

      config = config or pallas_mosaic_tpu_v2.Config()
      if not _is_config_supported(lhs, rhs, config):
        self.skipTest("Tile sizes larger than input dims are not supported.")

      op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot(config=config)
      return op(lhs, rhs, **kwargs)

    super().__init__(*args, dot_fn=fn)

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  @override
  def _test_quantized(self, *args, **kwargs):
    # v2 surfaces quantization through the API `rhs_scale`/`rhs_bias` kwargs
    # rather than QArray inputs, so the QArray-based base subtests do not apply.
    self.skipTest("v2 wrapper does not accept QArray.")

  def test_vjp0(self):
    # Both the DRHS (`tgmm_v2`) and DLHS (explicit `rhs.swapaxes` then
    # `gmm_v2`) backward paths are supported; loosen tolerances to match v1.
    with test_base.override_chex_args(atol=0.2, rtol=0.01):
      super().test_vjp0()  # pytype: disable=attribute-error

  def test_heuristics_config_returns_valid_tiling(self):
    ragged_dot = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot()
    ba = op_lib.BoundArguments(
        op=ragged_dot,
        arguments={
            "lhs": jax.ShapeDtypeStruct((262144, 7168), dtype=jnp.bfloat16),
            "rhs": jax.ShapeDtypeStruct((256, 7168, 2048), dtype=jnp.bfloat16),
            "group_sizes": base.generate_group_sizes(m=262144, num_groups=256),
        },
    )
    heuristics_config = ba.heuristics_config
    self.assertIsInstance(heuristics_config, pallas_mosaic_tpu_v2.Config)
    self.assertGreater(heuristics_config.tile_m, 0)
    self.assertGreater(heuristics_config.tile_k, 0)
    self.assertGreater(heuristics_config.tile_n, 0)


if __name__ == "__main__":
  absltest.main()
