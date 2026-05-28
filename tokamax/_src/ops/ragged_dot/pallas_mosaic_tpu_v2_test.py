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
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import qwix
from tokamax._src import quantization
from tokamax._src.ops import op as op_lib
from tokamax._src.ops.ragged_dot import api
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2_gmm_kernel as gmm_backend
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_backend
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
    del args, kwargs
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


# Small deterministic shapes used by the parameter-piping tests. `n` is a
# multiple of `2 * num_lanes` (256) so the `fuse_act` path, which splits `n`
# into gate/up halves, validates.
_NUM_GROUPS, _M, _K, _N = 2, 256, 128, 256
# Identity scale (1.0) and zero bias: with these, providing `rhs_scale` /
# `rhs_bias` is a no-op numerically, so the only thing under test is that the
# kwarg threads through the wrapper to the kernel unchanged.
_SCALE = jnp.ones((_NUM_GROUPS, 1, 1, _N), jnp.bfloat16)
_BIAS = jnp.zeros((_NUM_GROUPS, 1, _N), jnp.bfloat16)


class PallasMosaicTpuV2ParameterPipingTest(parameterized.TestCase):
  """Verifies each public `ragged_dot` kwarg reaches the v2 kernel unchanged.

  The mapping (`activation -> fuse_act`, passthrough cases) is the contract
  under test: each row spells out the API-side and kernel-side kwargs
  side-by-side, and the test asserts that routing through
  `api.ragged_dot(..., implementation="mosaic_tpu_v2", **api_kwargs)` produces
  the same result as a direct kernel call with `**kernel_kwargs`.
  """

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  def _make_gmm_inputs(self):
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(0))
    lhs = jax.random.normal(rng0, (_M, _K), jnp.bfloat16)
    rhs = jax.random.normal(rng1, (_NUM_GROUPS, _K, _N), jnp.bfloat16)
    group_sizes = jnp.array([_M // _NUM_GROUPS] * _NUM_GROUPS, jnp.int32)
    return lhs, rhs, group_sizes

  def _make_drhs_inputs(self):
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(1))
    lhs = jax.random.normal(rng0, (_M, _K), jnp.bfloat16)
    rhs = jax.random.normal(rng1, (_M, _N), jnp.bfloat16)
    group_sizes = jnp.array([_M // _NUM_GROUPS] * _NUM_GROUPS, jnp.int32)
    return lhs, rhs, group_sizes

  @parameterized.named_parameters(
      # (name, api_kwargs, kernel_kwargs)
      ("default", {}, {}),
      ("rhs_scale", {"rhs_scale": _SCALE}, {"rhs_scale": _SCALE}),
      ("rhs_bias", {"rhs_bias": _BIAS}, {"rhs_bias": _BIAS}),
      ("maybe_quantize_lhs",
       {"maybe_quantize_lhs": True}, {"maybe_quantize_lhs": True}),
      ("activation_silu", {"activation": jax.nn.silu}, {"fuse_act": "silu"}),
      ("activation_gelu", {"activation": jax.nn.gelu}, {"fuse_act": "gelu"}),
      ("preferred_element_type_f32",
       {"preferred_element_type": jnp.float32},
       {"preferred_element_type": jnp.float32}),
      ("precision_default",
       {"precision": jax.lax.Precision.DEFAULT},
       {"precision": jax.lax.Precision.DEFAULT}),
  )
  def test_api_pipes_kwarg_to_kernel(self, api_kwargs, kernel_kwargs):
    lhs, rhs, group_sizes = self._make_gmm_inputs()
    via_api = api.ragged_dot(
        lhs, rhs, group_sizes, implementation="mosaic_tpu_v2", **api_kwargs
    )
    via_kernel = gmm_backend.gmm_v2(
        lhs, rhs, group_sizes=group_sizes, **kernel_kwargs
    )
    # Tolerances are loose: the wrapper and the direct call may pick different
    # tilings (the heuristic is fuse_act-agnostic), which changes f32
    # accumulation order but not the mathematical result.
    chex.assert_trees_all_close(via_api, via_kernel, atol=1e-2, rtol=1e-2)

  @parameterized.named_parameters(
      # (name, api_kwargs, kernel_kwargs)
      ("default", {}, {}),
      ("precision_default",
       {"precision": jax.lax.Precision.DEFAULT},
       {"precision": jax.lax.Precision.DEFAULT}),
      ("preferred_element_type_f32",
       {"preferred_element_type": jnp.float32},
       {"preferred_element_type": jnp.float32}),
  )
  def test_drhs_api_pipes_kwarg_to_kernel(self, api_kwargs, kernel_kwargs):
    lhs, rhs, group_sizes = self._make_drhs_inputs()
    via_api = api.ragged_dot_general(
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers=pallas_mosaic_tpu_v2.DRHS_RAGGED_DOT_DIM_NUMS,
        implementation="mosaic_tpu_v2",
        **api_kwargs,
    )
    via_kernel = tgmm_backend.tgmm_v2(
        lhs,
        rhs,
        group_sizes=group_sizes,
        num_actual_groups=int(group_sizes.shape[0]),
        **kernel_kwargs,
    )
    chex.assert_trees_all_close(via_api, via_kernel, atol=1e-2, rtol=1e-2)

  def test_drhs_config_pipes_acc_dtype_to_kernel(self):
    # `acc_dtype` is a wrapper `Config` field rather than an `api.ragged_dot`
    # kwarg, so it is piped by constructing the op with an explicit `Config`.
    # Matching tiles are passed on both sides so the comparison is exact.
    lhs, rhs, group_sizes = self._make_drhs_inputs()
    tiles = gmm_backend.TileSizes(_K, _K, _N)
    op = pallas_mosaic_tpu_v2.PallasMosaicTpuV2RaggedDot(
        config=pallas_mosaic_tpu_v2.Config(
            tile_m=_K, tile_k=_K, tile_n=_N, acc_dtype="float32"
        )
    )
    via_op = op(
        lhs,
        rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=pallas_mosaic_tpu_v2.DRHS_RAGGED_DOT_DIM_NUMS,
    )
    via_kernel = tgmm_backend.tgmm_v2(
        lhs,
        rhs,
        group_sizes=group_sizes,
        num_actual_groups=int(group_sizes.shape[0]),
        tile_info=tiles,
        acc_dtype=jnp.float32,
    )
    chex.assert_trees_all_close(via_op, via_kernel, atol=1e-2, rtol=1e-2)

  def _assert_drhs_rejects(self, **api_kwargs):
    lhs, rhs, group_sizes = self._make_drhs_inputs()
    with self.assertRaises(NotImplementedError):
      api.ragged_dot_general(
          lhs,
          rhs,
          group_sizes,
          ragged_dot_dimension_numbers=pallas_mosaic_tpu_v2.DRHS_RAGGED_DOT_DIM_NUMS,
          implementation="mosaic_tpu_v2",
          **api_kwargs,
      )

  def test_drhs_rejects_rhs_scale(self):
    self._assert_drhs_rejects(
        rhs_scale=jnp.ones((_NUM_GROUPS, 1, 1, _N), jnp.bfloat16)
    )

  def test_drhs_rejects_rhs_bias(self):
    self._assert_drhs_rejects(
        rhs_bias=jnp.zeros((_NUM_GROUPS, 1, _N), jnp.bfloat16)
    )

  def test_drhs_rejects_maybe_quantize_lhs(self):
    self._assert_drhs_rejects(maybe_quantize_lhs=True)


if __name__ == "__main__":
  absltest.main()
