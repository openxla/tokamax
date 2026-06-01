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
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2_kernel_test as kernel_test
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

  def _assert_drhs_rejects(self, **api_kwargs):
    # These tests belong to this file instead of api_test.py.
    # If we put them to api_test.py then api_test.py needs to import
    # pallas_mosaic_tpu_v2 which will then be inconsistent with other tests in
    # api_test.py.
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

  # ---------------------------------------------------------------------------
  # Per-scenario GMM piping tests.
  #
  # Each test below mirrors one GMM case from `GmmTest` in
  # `pallas_mosaic_tpu_v2_kernel_test.py` (which compares the kernel against a
  # numpy-style reference). Here we instead route the *same* inputs through
  # `api.ragged_dot(..., implementation="mosaic_tpu_v2")` and a direct
  # `gmm_backend.gmm_v2(...)` call and assert they agree. Because both paths end
  # up in the same kernel with the same default tiling, any disagreement means
  # the API failed to thread a kwarg through to the kernel unchanged.
  #
  # The kernel-only cases that are not reachable from the public API are
  # intentionally skipped here: `test_gmm_*_block_larger_than_tile_k` depend on
  # an explicit `tile_info`, which is not an `api.ragged_dot` argument, and the
  # `security`/`uninitialized_memory` cases test kernel internals, not piping.
  # ---------------------------------------------------------------------------

  def _assert_gmm_api_matches_kernel(
      self, lhs, rhs, group_sizes, *, kwargs, atol=2e-2, rtol=2e-2
  ):
    """Asserts the API and direct-kernel GMM calls agree for `kwargs`.

    `kwargs` are passed verbatim to both entry points, so every key must be a
    name shared by `api.ragged_dot` and `gmm_backend.gmm_v2` (e.g. `rhs_scale`,
    `rhs_bias`, `group_offset`, `maybe_quantize_lhs`, `fuse_act`).
    """
    via_api = api.ragged_dot(
        lhs, rhs, group_sizes, implementation="mosaic_tpu_v2", **kwargs
    )
    via_kernel = gmm_backend.gmm_v2(lhs, rhs, group_sizes=group_sizes, **kwargs)
    chex.assert_trees_all_close(via_api, via_kernel, atol=atol, rtol=rtol)

  def test_gmm_basic_pipes(self):
    # Mirrors test_gmm_basic: exercises `rhs_bias` + `group_offset`.
    batch_size, in_size, out_size = 256, 256, 256
    num_groups, group_offset = 4, 1
    num_local_groups = num_groups - group_offset
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)

    lhs = jax.random.normal(k0, (batch_size, in_size), jnp.bfloat16)
    rhs = jax.random.normal(
        k1, (num_local_groups, in_size, out_size), jnp.bfloat16
    )
    rhs_bias = jax.random.normal(k2, (num_local_groups, 1, out_size), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    self._assert_gmm_api_matches_kernel(
        lhs,
        rhs,
        group_sizes,
        kwargs=dict(
            rhs_bias=rhs_bias,
            group_offset=jnp.array([group_offset], jnp.int32),
        ),
    )

  def test_gmm_weight_quantized_pipes(self):
    # Mirrors test_gmm_weight_quantized: int8 `rhs` + `rhs_scale` + `rhs_bias` +
    # `group_offset`, with `maybe_quantize_lhs=False`.
    batch_size, in_size, out_size = 128, 512, 512
    num_groups, group_offset, block_size = 4, 1, 256
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_q, rhs_scale = kernel_test.quantize_tensor(
        rhs, jnp.int8, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    rhs_bias = jax.random.normal(key, (num_local_groups, 1, out_size), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    self._assert_gmm_api_matches_kernel(
        lhs,
        rhs_q,
        group_sizes,
        kwargs=dict(
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            group_offset=jnp.array([group_offset], jnp.int32),
            maybe_quantize_lhs=False,
        ),
    )

  def test_gmm_activation_weight_quantized_pipes(self):
    # Mirrors test_gmm_activation_weight_quantized: int8 `rhs` + `rhs_scale` with
    # `maybe_quantize_lhs=True` (the lhs-quantization path).
    batch_size, in_size, out_size = 128, 512, 512
    num_groups, block_size = 4, 512
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_q, rhs_scale = kernel_test.quantize_tensor(
        rhs, jnp.int8, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    self._assert_gmm_api_matches_kernel(
        lhs,
        rhs_q,
        group_sizes,
        kwargs=dict(rhs_scale=rhs_scale, maybe_quantize_lhs=True),
    )

  def test_gmm_implicit_padding_pipes(self):
    # Mirrors test_gmm_implicit_padding: non-tile-aligned `in_size`/`out_size`
    # with `rhs_bias`.
    batch_size, in_size, out_size = 128, 255, 255
    num_groups = 4
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)

    lhs = jax.random.normal(k0, (batch_size, in_size), jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_groups, in_size, out_size), jnp.bfloat16)
    rhs_bias = jax.random.normal(k2, (num_groups, 1, out_size), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    self._assert_gmm_api_matches_kernel(
        lhs, rhs, group_sizes, kwargs=dict(rhs_bias=rhs_bias)
    )

  def test_gmm_weight_quantized_padding_pipes(self):
    # Mirrors test_gmm_weight_quantized_padding: int8 `rhs` + `rhs_scale` +
    # `rhs_bias` with a non-tile-aligned `out_size`.
    batch_size, in_size, out_size = 128, 512, 500
    num_groups, block_size = 4, 512
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), jnp.bfloat16)
    rhs = jax.random.normal(key, (num_groups, in_size, out_size), jnp.bfloat16)
    rhs_q, rhs_scale = kernel_test.quantize_tensor(
        rhs, jnp.int8, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    rhs_bias = jax.random.normal(key, (num_groups, 1, out_size), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    self._assert_gmm_api_matches_kernel(
        lhs,
        rhs_q,
        group_sizes,
        kwargs=dict(
            rhs_scale=rhs_scale, rhs_bias=rhs_bias, maybe_quantize_lhs=False
        ),
    )

  def test_gmm_nonlocal_groups_produce_zeros_pipes(self):
    # Mirrors test_gmm_nonlocal_groups_produce_zeros: a `group_offset` that makes
    # some global groups non-local (group < 0 or group >= num_local_groups), so
    # the kernel zero-fills their rows. `group_sizes` has `num_groups` entries
    # while `rhs` carries only `num_local_groups` weights.
    batch_size, in_size, out_size = 128, 256, 256
    num_groups, group_offset, num_local_groups = 16, 2, 4
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16
    )
    rhs_bias = jax.random.normal(key, (num_local_groups, 1, out_size), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    self._assert_gmm_api_matches_kernel(
        lhs,
        rhs,
        group_sizes,
        kwargs=dict(
            rhs_bias=rhs_bias,
            group_offset=jnp.array([group_offset], jnp.int32),
        ),
    )

  def test_gmm_fused_activation_pipes(self):
    # Mirrors test_gmm_fused_activation: `fuse_act` (GLU-style, halves `n`) plus
    # `rhs_bias`. `out_size` is a multiple of 2 * num_lanes (256) so the gate/up
    # split validates.
    batch_size, in_size, out_size = 128, 512, 512
    num_groups = 4
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)

    lhs = jax.random.uniform(k0, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        k1, (num_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_bias = jax.random.normal(k2, (num_groups, 1, out_size), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    self._assert_gmm_api_matches_kernel(
        lhs,
        rhs,
        group_sizes,
        kwargs=dict(rhs_bias=rhs_bias, fuse_act="silu"),
    )


if __name__ == "__main__":
  absltest.main()
