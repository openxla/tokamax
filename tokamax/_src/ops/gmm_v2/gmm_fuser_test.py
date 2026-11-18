# Copyright 2026 Google LLC
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
from absl.testing import parameterized
import chex
import jax
from jax.experimental.pallas import fuser
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src import config
from tokamax._src.ops.gmm_v2 import gmm_fuser
from tokamax._src.ops.gmm_v2 import gmm_v2
from tokamax._src.ops.gmm_v2 import tgmm_v2
from tokamax._src.ops.gmm_v2 import util
from tokamax.ops import op_fusion


def _rms_norm(
    x: jax.Array, weight: jax.Array, eps: float = 1e-6
) -> jax.Array:
  ms = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
  return (x.astype(jnp.float32) * jax.lax.rsqrt(ms + eps)).astype(
      x.dtype
  ) * weight


class GmmFuserTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    if pltpu.get_tpu_info().generation < 5:
      self.skipTest("Only supported on TPU gen 5+.")
    super().setUp()

  @parameterized.product(
      fuse_decorator=[fuser.fuse, op_fusion.fuse],
      disable_multi_core_mode=[False, True],
      skewed_groups=[False, True],
  )
  def test_gmm_v2_fusible_prologue_and_epilogue(
      self, fuse_decorator, disable_multi_core_mode, skewed_groups
  ):
    m, k, n, num_groups = 256, 256, 256, 4
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)
    rms_weight = jax.random.normal(k1, (1, k), jnp.bfloat16)
    rhs = jax.random.normal(k2, (num_groups, k, n), jnp.bfloat16)
    out_bias = jax.random.normal(k3, (1, n), jnp.bfloat16)

    @jax.jit
    @fuse_decorator
    def fused_gmm(x, w, r, b, gs):
      x_norm = _rms_norm(x, w)
      out = gmm_fuser.gmm_v2_fusible(x_norm, r, gs)
      return jax.nn.silu(out + b)

    gs = (
        jnp.array([3, 2, 17, m - 22], dtype=jnp.int32)
        if skewed_groups
        else util.get_group_sizes(m, num_groups)
    )
    with config.disable_multi_core_mode(disable_multi_core_mode):
      actual = fused_gmm(lhs, rms_weight, rhs, out_bias, gs)
      expected = jax.nn.silu(
          gmm_v2.gmm_v2(_rms_norm(lhs, rms_weight), rhs, gs) + out_bias
      )
      chex.assert_trees_all_close(actual, expected, atol=2e-2, rtol=2e-2)

  def test_gmm_v2_fusible_rhs_prologue(self):
    m, k, n, num_groups = 256, 256, 256, 4
    k0, k1, k2 = jax.random.split(jax.random.key(1), 3)
    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_groups, k, n), jnp.bfloat16)
    rhs_scale_factor = jax.random.uniform(
        k2, (1, 1, n), jnp.bfloat16, minval=0.5, maxval=1.5
    )
    group_sizes = util.get_group_sizes(m, num_groups)

    @jax.jit
    @fuser.fuse
    def fused_gmm(x, r, s, gs):
      return gmm_fuser.gmm_v2_fusible(x, r * s, gs)

    actual = fused_gmm(lhs, rhs, rhs_scale_factor, group_sizes)
    expected = gmm_v2.gmm_v2(lhs, rhs * rhs_scale_factor, group_sizes)
    chex.assert_trees_all_close(actual, expected, atol=2e-2, rtol=2e-2)

  def test_gmm_v2_fusible_quantized_and_fuse_act(self):
    m, k, n, num_groups = 256, 256, 512, 4
    k0, k1, k2 = jax.random.split(jax.random.key(2), 3)
    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)
    rms_weight = jax.random.normal(k1, (1, k), jnp.bfloat16)
    rhs = jax.random.normal(k2, (num_groups, k, n), jnp.bfloat16)
    rhs_q, rhs_scale = util.quantize_tensor(
        rhs, jnp.float8_e4m3fn, axis=1, block_size=256  # pyrefly: ignore[bad-argument-type]
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    group_sizes = util.get_group_sizes(m, num_groups)

    @jax.jit
    @fuser.fuse
    def fused_gmm(x, w, rq, rs, gs):
      return gmm_fuser.gmm_v2_fusible(
          _rms_norm(x, w),
          rq,
          gs,
          rhs_scale=rs,
          fuse_act="silu",
          preferred_element_type=jnp.bfloat16,
      )

    actual = fused_gmm(lhs, rms_weight, rhs_q, rhs_scale, group_sizes)
    expected = gmm_v2.gmm_v2(
        _rms_norm(lhs, rms_weight),
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        fuse_act="silu",
        preferred_element_type=jnp.bfloat16,
    )
    chex.assert_trees_all_close(actual, expected, atol=2e-2, rtol=2e-2)

  @parameterized.product(
      fuse_decorator=[fuser.fuse, op_fusion.fuse],
      disable_multi_core_mode=[False, True],
  )
  def test_tgmm_v2_fusible_prologue_and_epilogue(
      self, fuse_decorator, disable_multi_core_mode
  ):
    m, k, n, num_groups = 256, 256, 256, 4
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.key(3), 5)
    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)
    rms_weight = jax.random.normal(k1, (1, k), jnp.bfloat16)
    rhs = jax.random.normal(k2, (m, n), jnp.bfloat16)
    rhs_gain = jax.random.uniform(
        k3, (1, n), jnp.bfloat16, minval=0.5, maxval=1.5
    )
    out_bias = jax.random.normal(k4, (num_groups, 1, n), jnp.bfloat16)
    group_sizes = util.get_group_sizes(m, num_groups)

    @jax.jit
    @fuse_decorator
    def fused_tgmm(x, w, dy, gain, b, gs):
      out = gmm_fuser.tgmm_v2_fusible(
          _rms_norm(x, w),
          dy * gain,
          gs,
          num_groups,
          preferred_element_type=jnp.bfloat16,
      )
      return out + b

    with config.disable_multi_core_mode(disable_multi_core_mode):
      actual = fused_tgmm(lhs, rms_weight, rhs, rhs_gain, out_bias, group_sizes)
      expected = (
          tgmm_v2.tgmm_v2(
              _rms_norm(lhs, rms_weight),
              rhs * rhs_gain,
              group_sizes,
              num_groups,
              preferred_element_type=jnp.bfloat16,
          )
          + out_bias
      )
      chex.assert_trees_all_close(actual, expected, atol=2e-2, rtol=2e-2)

  @parameterized.product(
      fuse_decorator=[None, fuser.fuse, op_fusion.fuse],
      disable_multi_core_mode=[False, True],
      skewed_groups=[False, True],
  )
  def test_gmm_v2_fusible_no_fusion(
      self, fuse_decorator, disable_multi_core_mode, skewed_groups
  ):
    m, k, n, num_groups = 200, 256, 256, 4
    k0, k1 = jax.random.split(jax.random.key(4), 2)
    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_groups, k, n), jnp.bfloat16)
    gs = (
        jnp.array([3, 2, 17, m - 22], dtype=jnp.int32)
        if skewed_groups
        else util.get_group_sizes(m, num_groups)
    )

    def run_gmm(x, r, group_sizes):
      return gmm_fuser.gmm_v2_fusible(x, r, group_sizes)

    if fuse_decorator is not None:
      run_gmm = fuse_decorator(run_gmm)
    run_gmm = jax.jit(run_gmm)

    with config.disable_multi_core_mode(disable_multi_core_mode):
      actual = run_gmm(lhs, rhs, gs)
      expected = gmm_v2.gmm_v2(lhs, rhs, gs)
      chex.assert_trees_all_equal(actual, expected)

  @parameterized.product(
      fuse_decorator=[None, fuser.fuse, op_fusion.fuse],
      disable_multi_core_mode=[False, True],
      skewed_groups=[False, True],
  )
  def test_tgmm_v2_fusible_no_fusion(
      self, fuse_decorator, disable_multi_core_mode, skewed_groups
  ):
    m, k, n, num_groups = 256, 256, 256, 4
    k0, k1 = jax.random.split(jax.random.key(5), 2)
    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)
    dout = jax.random.normal(k1, (m, n), jnp.bfloat16)
    gs = (
        jnp.array([3, 2, 17, m - 22], dtype=jnp.int32)
        if skewed_groups
        else util.get_group_sizes(m, num_groups)
    )

    def run_tgmm(x, dy, group_sizes):
      return gmm_fuser.tgmm_v2_fusible(
          x, dy, group_sizes, num_groups, preferred_element_type=jnp.bfloat16
      )

    if fuse_decorator is not None:
      run_tgmm = fuse_decorator(run_tgmm)
    run_tgmm = jax.jit(run_tgmm)

    with config.disable_multi_core_mode(disable_multi_core_mode):
      actual = run_tgmm(lhs, dout, gs)
      expected = tgmm_v2.tgmm_v2(
          lhs, dout, gs, num_groups, preferred_element_type=jnp.bfloat16
      )
      chex.assert_trees_all_equal(actual, expected)

  def test_direct_unfused_calls(self):
    m, k, n, num_groups = 256, 256, 256, 4
    k0, k1, k2 = jax.random.split(jax.random.key(6), 3)
    lhs = jax.random.normal(k0, (m, k), jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_groups, k, n), jnp.bfloat16)
    dout = jax.random.normal(k2, (m, n), jnp.bfloat16)
    group_sizes = util.get_group_sizes(m, num_groups)

    gmm_actual = gmm_fuser.gmm_v2_fusible(lhs, rhs, group_sizes)
    gmm_expected = gmm_v2.gmm_v2(lhs, rhs, group_sizes)
    chex.assert_trees_all_equal(gmm_actual, gmm_expected)

    tgmm_actual = gmm_fuser.tgmm_v2_fusible(
        lhs, dout, group_sizes, num_groups, preferred_element_type=jnp.bfloat16
    )
    tgmm_expected = tgmm_v2.tgmm_v2(
        lhs, dout, group_sizes, num_groups, preferred_element_type=jnp.bfloat16
    )
    chex.assert_trees_all_equal(tgmm_actual, tgmm_expected)

    gmm_with_psum = gmm_v2.gmm_v2(
        lhs, rhs, group_sizes, partial_sum=dout
    )
    chex.assert_trees_all_close(
        gmm_with_psum, gmm_expected + dout, atol=2e-2, rtol=2e-2
    )
    tgmm_psum_init = jnp.ones_like(tgmm_expected)
    tgmm_with_psum = tgmm_v2.tgmm_v2(
        lhs,
        dout,
        group_sizes,
        num_groups,
        partial_sum=tgmm_psum_init,
        preferred_element_type=jnp.bfloat16,
    )
    chex.assert_trees_all_close(
        tgmm_with_psum, tgmm_expected + tgmm_psum_init, atol=2e-2, rtol=2e-2
    )


if __name__ == "__main__":
  absltest.main()
