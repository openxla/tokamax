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
"""Tests for Triton Triangle Multiplication."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src import hlo_utils
from tokamax._src import numerics
from tokamax._src.ops.triangle_multiplication import api
from tokamax._src.ops.triangle_multiplication import base

try:
  from tokamax._src.ops.triangle_multiplication import triton as triton_tri_mul  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]
except ImportError:
  triton_tri_mul = None


def _get_params(n: int, c: int, h: int, d: int, dtype: jax.typing.DTypeLike):
  params = dict(
      x=jax.ShapeDtypeStruct((n, n, c), dtype=dtype),
      mask=jax.ShapeDtypeStruct((n, n), dtype=jnp.bool_),
      projection_in_weights=jax.ShapeDtypeStruct((c, 2, h), dtype=dtype),
      gate_in_weights=jax.ShapeDtypeStruct((c, 2, h), dtype=dtype),
      projection_out_weights=jax.ShapeDtypeStruct((h, d), dtype=dtype),
      gate_out_weights=jax.ShapeDtypeStruct((c, d), dtype=dtype),
      layernorm_in_scale=jax.ShapeDtypeStruct((c,), dtype=dtype),
      layernorm_in_offset=jax.ShapeDtypeStruct((c,), dtype=dtype),
      layernorm_out_scale=jax.ShapeDtypeStruct((h,), dtype=dtype),
      layernorm_out_offset=jax.ShapeDtypeStruct((h,), dtype=dtype),
  )
  return numerics.random_initialize(params)


@absltest.skipIf(triton_tri_mul is None, "Requires jax_triton.")
class TritonTriangleMultiplicationTest(parameterized.TestCase):

  def setUp(self):
    if not gpu_utils.has_triton_support():
      self.skipTest("Triton not supported on this platform.")
    super().setUp()
    assert triton_tri_mul is not None
    self._op = triton_tri_mul.TritonTriangleMultiplication()
    self._ref_op = base.TriangleMultiplication()

  def _assert_matches_reference(self, out, params, triangle_type):
    # Compare against the XLA reference in the same dtype, which rounds the
    # intermediate projections like the kernels do. A few isolated elements
    # differ by up to ~0.25 due to LayerNorm_out amplifying bf16 rounding of
    # the contraction output.
    expected = jax.jit(
        lambda p: self._ref_op(**p, triangle_type=triangle_type)
    )(params)
    chex.assert_trees_all_close(out, expected, atol=0.3, rtol=0.05)

  @parameterized.product(
      n=[100, 128, 384],
      triangle_type=["incoming", "outgoing"],
      dtype=[jnp.bfloat16, jnp.float16],
  )
  def test_triangle_multiplication(self, n, triangle_type, dtype):
    params = _get_params(n=n, c=64, h=64, d=64, dtype=dtype)
    params_f32 = {
        k: v.astype(jnp.float32) if k != "mask" else v
        for k, v in params.items()
    }

    out = jax.jit(lambda p: self._op(**p, triangle_type=triangle_type))(params)
    expected = jax.jit(
        lambda p: self._ref_op(
            **p,
            triangle_type=triangle_type,
            precision=jax.lax.Precision.HIGHEST,
        )
    )(params_f32).astype(dtype)

    self.assertEqual(out.shape, (n, n, 64))
    self.assertEqual(out.dtype, dtype)
    atol = (0.15 if n < 128 else 0.45) if dtype == jnp.bfloat16 else 0.05
    rtol = 0.05 if dtype == jnp.bfloat16 else 0.02
    chex.assert_trees_all_close(out, expected, atol=atol, rtol=rtol)

  @parameterized.parameters("incoming", "outgoing")
  def test_non_uniform_channel_dims(self, triangle_type):
    params = _get_params(n=64, c=16, h=32, d=64, dtype=jnp.bfloat16)
    params_f32 = {
        k: v.astype(jnp.float32) if k != "mask" else v
        for k, v in params.items()
    }

    out = jax.jit(lambda p: self._op(**p, triangle_type=triangle_type))(params)
    expected = jax.jit(
        lambda p: self._ref_op(
            **p,
            triangle_type=triangle_type,
            precision=jax.lax.Precision.HIGHEST,
        )
    )(params_f32).astype(jnp.bfloat16)

    self.assertEqual(out.shape, (64, 64, 64))
    self.assertEqual(out.dtype, jnp.bfloat16)
    chex.assert_trees_all_close(out, expected, atol=0.15, rtol=0.05)

  @parameterized.parameters("incoming", "outgoing")
  def test_api_dispatch(self, triangle_type):
    assert triton_tri_mul is not None
    params = _get_params(n=128, c=64, h=64, d=64, dtype=jnp.bfloat16)

    @jax.jit
    def f_triton(p):
      return api.triangle_multiplication(
          **p, triangle_type=triangle_type, implementation="triton"
      )

    @jax.jit
    def f_xla(p):
      return api.triangle_multiplication(
          **p, triangle_type=triangle_type, implementation="xla"
      )

    out_triton = f_triton(params)
    out_xla = f_xla(params)

    chex.assert_trees_all_close(out_triton, out_xla, atol=0.15, rtol=0.05)

    bound_args = hlo_utils.get_bound_args(f_triton.lower(params))
    self.assertLen(bound_args, 1)
    self.assertIsInstance(
        bound_args[0].op, triton_tri_mul.TritonTriangleMultiplication
    )

  def test_autotuning_configs(self):
    assert triton_tri_mul is not None
    n, c, h, d = 128, 64, 64, 64
    dtype = jnp.bfloat16
    params = _get_params(n=n, c=c, h=h, d=d, dtype=dtype)
    ba = self._op.bind(**params, triangle_type="outgoing")
    configs = self._op._get_autotuning_configs(ba)
    self.assertNotEmpty(configs)
    expected = jax.jit(lambda p: self._ref_op(**p, triangle_type="outgoing"))(
        params
    )
    for cfg in configs:
      op_cfg = triton_tri_mul.TritonTriangleMultiplication(config=cfg)
      out = jax.jit(lambda p, op=op_cfg: op(**p, triangle_type="outgoing"))(
          params
      )
      chex.assert_trees_all_close(out, expected, atol=0.15, rtol=0.05)

  @parameterized.product(
      prologue_persistent=[True, False],
      epilogue_persistent=[True, False],
      n=[64, 100],
  )
  def test_persistent_and_pointer_kernels_match(
      self, prologue_persistent, epilogue_persistent, n
  ):
    assert triton_tri_mul is not None
    # On SM90+, `persistent=True` selects the TMA kernels when M is a multiple
    # of `block_m` (n=64); otherwise (n=100, or pre-SM90) the pointer kernels
    # are used.
    params = _get_params(n=n, c=64, h=64, d=64, dtype=jnp.bfloat16)
    config = triton_tri_mul.Config(
        block_m=64,
        prologue_persistent=prologue_persistent,
        epilogue_persistent=epilogue_persistent,
    )
    op_cfg = triton_tri_mul.TritonTriangleMultiplication(config=config)
    out = jax.jit(lambda p: op_cfg(**p, triangle_type="outgoing"))(params)
    expected = jax.jit(lambda p: self._ref_op(**p, triangle_type="outgoing"))(
        params
    )
    chex.assert_trees_all_close(out, expected, atol=0.15, rtol=0.05)

  @parameterized.product(
      (
          dict(c=128, block_m=64),
          dict(c=128, block_m=128),
          dict(c=64, block_m=256),
      ),
      persistent=[True, False],
  )
  def test_channel_and_block_sizes(self, c, block_m, persistent):
    assert triton_tri_mul is not None
    # n=64 gives M=4096, a multiple of every `block_m`, so `persistent=True`
    # selects the TMA kernels on SM90+.
    params = _get_params(n=64, c=c, h=c, d=c, dtype=jnp.bfloat16)
    config = triton_tri_mul.Config(
        block_m=block_m,
        prologue_persistent=persistent,
        epilogue_persistent=persistent,
    )
    op_cfg = triton_tri_mul.TritonTriangleMultiplication(config=config)
    out = jax.jit(lambda p: op_cfg(**p, triangle_type="outgoing"))(params)
    self._assert_matches_reference(out, params, "outgoing")

  @parameterized.parameters(True, False)
  def test_epilogue_config_overrides(self, persistent):
    assert triton_tri_mul is not None
    params = _get_params(n=64, c=64, h=64, d=64, dtype=jnp.bfloat16)
    config = triton_tri_mul.Config(
        block_m=128,
        num_warps=4,
        num_stages=3,
        epilogue_block_m=64,
        epilogue_num_warps=8,
        epilogue_num_stages=4,
        prologue_persistent=persistent,
        epilogue_persistent=persistent,
    )
    op_cfg = triton_tri_mul.TritonTriangleMultiplication(config=config)
    out = jax.jit(lambda p: op_cfg(**p, triangle_type="incoming"))(params)
    self._assert_matches_reference(out, params, "incoming")

  @parameterized.product(
      triangle_type=["incoming", "outgoing"],
      n=[64, 100],
  )
  def test_sparse_mask(self, triangle_type, n):
    params = _get_params(n=n, c=64, h=64, d=64, dtype=jnp.bfloat16)
    # Zero out whole rows as well as ~90% of the remaining entries.
    mask = jax.random.uniform(jax.random.key(1), (n, n)) < 0.1
    params["mask"] = mask.at[: n // 2].set(False)

    out = jax.jit(lambda p: self._op(**p, triangle_type=triangle_type))(params)
    self._assert_matches_reference(out, params, triangle_type)

  def test_config_validation(self):
    assert triton_tri_mul is not None
    triton_tri_mul.Config(block_m=256, epilogue_block_m=32, num_warps=16)
    for kwargs in (
        dict(block_m=96),  # Not a power of two.
        dict(block_m=16),  # Too small.
        dict(block_m=512),  # Larger than the TMA block limit.
        dict(epilogue_block_m=192),
        dict(num_warps=3),
        dict(epilogue_num_warps=32),
    ):
      with self.subTest(str(kwargs)), self.assertRaises(ValueError):
        triton_tri_mul.Config(**kwargs)

  @parameterized.product(
      triangle_type=["incoming", "outgoing"],
      use_api=[False, True],
  )
  def test_gradients(self, triangle_type, use_api):
    params = _get_params(n=64, c=64, h=64, d=64, dtype=jnp.bfloat16)
    mask = params.pop("mask")
    dout = jax.random.normal(
        jax.random.key(0), (64, 64, 64), dtype=jnp.bfloat16
    )

    if use_api:
      # The default implementation tries Triton first.
      fn = api.triangle_multiplication
    else:
      fn = self._op

    def loss(p, fn, **kwargs):
      out = fn(**p, mask=mask, triangle_type=triangle_type, **kwargs)
      return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32))

    grads = jax.jit(jax.grad(lambda p: loss(p, fn)))(params)
    expected = jax.jit(
        jax.grad(lambda p: loss(p, self._ref_op, implementation="xla"))
    )(params)

    self.assertEqual(jax.tree.structure(grads), jax.tree.structure(expected))
    for k in params:
      with self.subTest(k):
        chex.assert_trees_all_close(
            grads[k].astype(jnp.float32),
            expected[k].astype(jnp.float32),
            atol=0.5,
            rtol=0.05,
        )

  def test_return_residuals(self):
    params = _get_params(n=64, c=64, h=64, d=64, dtype=jnp.bfloat16)
    out, residuals = jax.jit(
        lambda p: self._op(**p, triangle_type="outgoing", return_residuals=True)
    )(params)
    self.assertIsNone(residuals)
    self.assertEqual(out.shape, (64, 64, 64))

  def test_int32_offset_overflow_raises(self):
    def get_shapes(n, c):
      # Use abstract shapes, so that no memory is allocated.
      dtype = jnp.bfloat16
      return dict(
          x=jax.ShapeDtypeStruct((n, n, c), dtype),
          mask=jax.ShapeDtypeStruct((n, n), jnp.bool_),
          projection_in_weights=jax.ShapeDtypeStruct((c, 2, c), dtype),
          gate_in_weights=jax.ShapeDtypeStruct((c, 2, c), dtype),
          projection_out_weights=jax.ShapeDtypeStruct((c, c), dtype),
          gate_out_weights=jax.ShapeDtypeStruct((c, c), dtype),
          layernorm_in_scale=jax.ShapeDtypeStruct((c,), dtype),
          layernorm_in_offset=jax.ShapeDtypeStruct((c,), dtype),
          layernorm_out_scale=jax.ShapeDtypeStruct((c,), dtype),
          layernorm_out_offset=jax.ShapeDtypeStruct((c,), dtype),
      )

    f = lambda p: self._op(**p, triangle_type="outgoing")
    # 4096 * 4096 * 128 == 2**31, so the largest offset is 2**31 - 1.
    out = jax.eval_shape(f, get_shapes(n=4096, c=128))
    self.assertEqual(out.shape, (4096, 4096, 128))
    with self.assertRaisesRegex(NotImplementedError, "int32"):
      jax.eval_shape(f, get_shapes(n=4097, c=128))

  def test_unsupported_raises_not_implemented(self):
    with self.subTest("unsupported_dtype"):
      params = _get_params(n=128, c=64, h=64, d=64, dtype=jnp.float32)
      with self.assertRaises(NotImplementedError):
        self._op(**params, triangle_type="outgoing")

    with self.subTest("unsupported_channel_dims"):
      params = _get_params(n=128, c=48, h=48, d=64, dtype=jnp.bfloat16)
      with self.assertRaises(NotImplementedError):
        self._op(**params, triangle_type="outgoing")

    with self.subTest("unsupported_precision"):
      params = _get_params(n=128, c=64, h=64, d=64, dtype=jnp.bfloat16)
      with self.assertRaises(NotImplementedError):
        self._op(
            **params,
            triangle_type="outgoing",
            precision=jax.lax.DotAlgorithmPreset.F32_F32_F32,
        )

    with self.subTest("mismatched_weight_dtype"):
      params = _get_params(n=128, c=64, h=64, d=64, dtype=jnp.bfloat16)
      params["projection_in_weights"] = params["projection_in_weights"].astype(
          jnp.float16
      )
      with self.assertRaises(NotImplementedError):
        self._op(**params, triangle_type="outgoing")


if __name__ == "__main__":
  absltest.main()
