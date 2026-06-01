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
from tokamax._src.ops.ragged_dot import api
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2_gmm_kernel as gmm_backend
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2_kernel_test as kernel_test


class PallasMosaicTpuV2ParameterPipingTest(parameterized.TestCase):
  """Verifies the "mosaic_tpu_v2" `ragged_dot` API pipes kwargs to the kernel.

  Each test below mirrors one GMM case from `GmmTest` in
  `pallas_mosaic_tpu_v2_kernel_test.py` (which compares the kernel against a
  numpy-style reference). Here we instead route the *same* inputs through
  `api.ragged_dot(..., implementation="mosaic_tpu_v2")` and a direct
  `gmm_backend.gmm_v2(...)` call and assert they agree. Because both paths end
  up in the same kernel with the same default tiling, any disagreement means
  the API failed to thread a kwarg through to the kernel unchanged.

  These tests belong to this file instead of `api_test.py`. If we put them in
  `api_test.py` then `api_test.py` would need to import the GMM v2 impl, which
  would be inconsistent with the status quo of `api_test.py`.
  """

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

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

  def test_gmm_preferred_element_type_pipes(self):
    """`preferred_element_type` must reach the kernel and set the output dtype.

    With a non-default output dtype (f32), a wrapper that dropped the kwarg
    would return bf16 (`lhs.dtype`); the explicit dtype checks below catch that
    even though the values themselves would still be numerically close.
    """
    batch_size, in_size, out_size = 256, 256, 256
    num_groups = 4
    k0, k1 = jax.random.split(jax.random.key(0), 2)

    lhs = jax.random.normal(k0, (batch_size, in_size), jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_groups, in_size, out_size), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    via_api = api.ragged_dot(
        lhs,
        rhs,
        group_sizes,
        implementation="mosaic_tpu_v2",
        preferred_element_type=jnp.float32,
    )
    via_kernel = gmm_backend.gmm_v2(
        lhs, rhs, group_sizes=group_sizes, preferred_element_type=jnp.float32
    )
    self.assertEqual(via_api.dtype, jnp.float32)
    self.assertEqual(via_kernel.dtype, jnp.float32)
    chex.assert_trees_all_close(via_api, via_kernel, atol=2e-2, rtol=2e-2)

  def test_gmm_precision_pipes(self):
    """`precision` must thread through the API to the kernel without error.

    The v2 kernel ignores `precision` (it exists only for API compatibility and
    is `del`-eted inside `gmm_v2`), so this is an accepts-and-forwards check
    rather than a numeric one: the API must accept a non-default precision and
    still produce the same result as the direct kernel call.
    """
    batch_size, in_size, out_size = 256, 256, 256
    num_groups = 4
    k0, k1 = jax.random.split(jax.random.key(0), 2)

    lhs = jax.random.normal(k0, (batch_size, in_size), jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_groups, in_size, out_size), jnp.bfloat16)
    group_sizes = kernel_test.get_group_sizes(batch_size, num_groups)

    self._assert_gmm_api_matches_kernel(
        lhs,
        rhs,
        group_sizes,
        kwargs=dict(precision=jax.lax.Precision.HIGHEST),
    )


if __name__ == "__main__":
  absltest.main()
