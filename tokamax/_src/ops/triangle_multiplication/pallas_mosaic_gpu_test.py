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
"""Tests for Pallas-Mosaic-GPU Triangle Multiplication."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from tokamax._src import hlo_utils
from tokamax._src import numerics
from tokamax._src.ops.triangle_multiplication import api
from tokamax._src.ops.triangle_multiplication import base
from tokamax._src.ops.triangle_multiplication import pallas_mosaic_gpu as mgpu_tri_mul
from tokamax._src.ops.triangle_multiplication import pallas_mosaic_gpu_vjp as mgpu_tri_mul_vjp


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


class PallasMosaicGpuTriangleMultiplicationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._op = mgpu_tri_mul.PallasMosaicGpuTriangleMultiplication()
    if not self._op.supported_on(jax.devices()[0]):
      self.skipTest("Mosaic GPU not supported on this platform.")
    self._ref_op = base.TriangleMultiplication()

  def _assert_matches_reference(self, out, params, triangle_type):
    params_f32 = {
        k: v.astype(jnp.float32) if k != "mask" else v
        for k, v in params.items()
    }
    expected = jax.jit(
        lambda p: self._ref_op(
            **p,
            triangle_type=triangle_type,
            precision=jax.lax.Precision.HIGHEST,
        )
    )(params_f32).astype(out.dtype)
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

  @parameterized.product(
      triangle_type=["incoming", "outgoing"],
      chd=[(16, 32, 64), (64, 32, 16), (32, 128, 32)],
  )
  def test_non_uniform_channel_dims(self, triangle_type, chd):
    c, h, d = chd
    params = _get_params(n=64, c=c, h=h, d=d, dtype=jnp.bfloat16)
    mask = params.pop("mask")
    params_f32 = {k: v.astype(jnp.float32) for k, v in params.items()}
    dout = (
        jax.random.normal(jax.random.key(7), (64, 64, d), dtype=jnp.bfloat16)
        / 64
    )

    out, vjp_fn = jax.jit(
        lambda p: jax.vjp(
            lambda q: self._op(**q, mask=mask, triangle_type=triangle_type), p
        )
    )(params)
    expected, ref_vjp = jax.jit(
        lambda p: jax.vjp(
            lambda q: self._ref_op(
                **q,
                mask=mask,
                triangle_type=triangle_type,
                precision=jax.lax.Precision.HIGHEST,
            ),
            p,
        )
    )(params_f32)
    (grads,) = jax.jit(vjp_fn)(dout)
    (ref_grads,) = jax.jit(ref_vjp)(dout.astype(jnp.float32))
    expected_grads = {
        k: v.astype(jnp.bfloat16) for k, v in ref_grads.items()
    }

    self.assertEqual(out.shape, (64, 64, d))
    self.assertEqual(out.dtype, jnp.bfloat16)
    chex.assert_trees_all_close(
        out, expected.astype(jnp.bfloat16), atol=0.2, rtol=0.05
    )
    chex.assert_trees_all_close(grads, expected_grads, atol=0.5, rtol=0.05)

  @parameterized.parameters("incoming", "outgoing")
  def test_api_dispatch(self, triangle_type):
    params = _get_params(n=128, c=64, h=64, d=64, dtype=jnp.bfloat16)

    @jax.jit
    def f_mosaic(p):
      return api.triangle_multiplication(
          **p, triangle_type=triangle_type, implementation="mosaic"
      )

    @jax.jit
    def f_xla(p):
      return api.triangle_multiplication(
          **p, triangle_type=triangle_type, implementation="xla"
      )

    out_mosaic = f_mosaic(params)
    out_xla = f_xla(params)

    chex.assert_trees_all_close(out_mosaic, out_xla, atol=0.15, rtol=0.05)

    bound_args = hlo_utils.get_bound_args(f_mosaic.lower(params))
    self.assertLen(bound_args, 1)
    self.assertIsInstance(
        bound_args[0].op, mgpu_tri_mul.PallasMosaicGpuTriangleMultiplication
    )

  def test_autotuning_configs(self):
    n, c, h, d = 128, 64, 64, 64
    dtype = jnp.bfloat16
    params = _get_params(n=n, c=c, h=h, d=d, dtype=dtype)
    ba = self._op.bind(**params, triangle_type="outgoing")
    configs = self._op._get_autotuning_configs(ba)
    self.assertNotEmpty(configs)
    self.assertIn(self._op._get_heuristics_config(ba), configs)
    expected = jax.jit(lambda p: self._ref_op(**p, triangle_type="outgoing"))(
        params
    )
    for cfg in configs:
      op_cfg = mgpu_tri_mul.PallasMosaicGpuTriangleMultiplication(config=cfg)
      out = jax.jit(lambda p, op=op_cfg: op(**p, triangle_type="outgoing"))(
          params
      )
      chex.assert_trees_all_close(out, expected, atol=0.15, rtol=0.05)

    vjp_op = mgpu_tri_mul_vjp.PallasMosaicGpuTriangleMultiplicationVjp()
    vjp_ba = vjp_op.bind(**ba.vjp_arg_spec)
    vjp_configs = vjp_op._get_autotuning_configs(vjp_ba)
    self.assertNotEmpty(vjp_configs)
    self.assertIn(vjp_op._get_heuristics_config(vjp_ba), vjp_configs)
    dout = jax.random.normal(jax.random.key(0), (n, n, d), dtype=dtype)
    _, vjp_ref = jax.jit(
        lambda p: jax.vjp(
            lambda q: self._ref_op(
                **q, triangle_type="outgoing", implementation="xla"
            ),
            p,
        )
    )(params)
    (expected_grads,) = jax.jit(vjp_ref)(dout)
    for vjp_cfg in vjp_configs:
      vjp_op_cfg = mgpu_tri_mul_vjp.PallasMosaicGpuTriangleMultiplicationVjp(
          config=vjp_cfg
      )
      op_with_vjp = mgpu_tri_mul.PallasMosaicGpuTriangleMultiplication(
          vjp=vjp_op_cfg
      )
      _, vjp_fn = jax.jit(
          lambda p, op=op_with_vjp: jax.vjp(
              lambda q: op(**q, triangle_type="outgoing"), p
          )
      )(params)
      (grads,) = jax.jit(vjp_fn)(dout)
      for k in params:
        if k == "mask":
          continue
        chex.assert_trees_all_close(
            grads[k].astype(jnp.float32),
            expected_grads[k].astype(jnp.float32),
            atol=0.35 * n,
            rtol=0.03,
        )

  @parameterized.product(
      (
          dict(c=128, block_m=64),
          dict(c=128, block_m=128),
          dict(c=64, block_m=256),
      ),
      persistent=[True, False],
  )
  def test_channel_and_block_sizes(self, c, block_m, persistent):
    params = _get_params(n=64, c=c, h=c, d=c, dtype=jnp.bfloat16)
    config = mgpu_tri_mul.Config(
        block_m=block_m,
        prologue_persistent=persistent,
        epilogue_persistent=persistent,
    )
    op_cfg = mgpu_tri_mul.PallasMosaicGpuTriangleMultiplication(config=config)
    out = jax.jit(lambda p: op_cfg(**p, triangle_type="outgoing"))(params)
    self._assert_matches_reference(out, params, "outgoing")

  @parameterized.parameters(True, False)
  def test_epilogue_config_overrides(self, persistent):
    params = _get_params(n=64, c=64, h=64, d=64, dtype=jnp.bfloat16)
    config = mgpu_tri_mul.Config(
        block_m=128,
        num_warps=4,
        num_stages=3,
        epilogue_block_m=64,
        epilogue_num_warps=8,
        epilogue_num_stages=4,
        prologue_persistent=persistent,
        epilogue_persistent=persistent,
    )
    op_cfg = mgpu_tri_mul.PallasMosaicGpuTriangleMultiplication(config=config)
    out = jax.jit(lambda p: op_cfg(**p, triangle_type="incoming"))(params)
    self._assert_matches_reference(out, params, "incoming")

  @parameterized.product(
      triangle_type=["incoming", "outgoing"],
      n=[64, 100],
  )
  def test_sparse_mask(self, triangle_type, n):
    params = _get_params(n=n, c=64, h=64, d=64, dtype=jnp.bfloat16)
    mask = jax.random.uniform(jax.random.key(1), (n, n)) < 0.1
    params["mask"] = mask.at[: n // 2].set(False)

    out = jax.jit(lambda p: self._op(**p, triangle_type=triangle_type))(params)
    self._assert_matches_reference(out, params, triangle_type)

  def test_config_validation(self):
    mgpu_tri_mul.Config(
        block_m=256,
        epilogue_block_m=32,
        num_warps=16,
    )
    mgpu_tri_mul_vjp.Config(
        block_m=64,
        epilogue_block_m=32,
        num_warps=8,
        epilogue_num_warps=4,
    )
    for kwargs in (
        dict(block_m=96),
        dict(block_m=16),
        dict(block_m=512),
        dict(epilogue_block_m=192),
        dict(num_warps=3),
        dict(epilogue_num_warps=32),
    ):
      with self.subTest(str(kwargs)), self.assertRaises(ValueError):
        mgpu_tri_mul.Config(**kwargs)
      with self.subTest(f"vjp_{kwargs}"), self.assertRaises(ValueError):
        mgpu_tri_mul_vjp.Config(**kwargs)

  @parameterized.product(
      triangle_type=["incoming", "outgoing"],
      use_api=[False, True],
      n=[64, 100],
      dtype=[jnp.bfloat16, jnp.float16],
  )
  def test_gradients(self, triangle_type, use_api, n, dtype):
    params = _get_params(n=n, c=64, h=64, d=64, dtype=dtype)
    mask = params.pop("mask")
    dout = jax.random.normal(jax.random.key(0), (n, n, 64), dtype=dtype)

    if use_api:
      fn = api.triangle_multiplication
    else:
      fn = self._op

    def loss(p, fn, **kwargs):
      out = fn(**p, mask=mask, triangle_type=triangle_type, **kwargs)
      return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32)) / n

    grad_fn = jax.jit(jax.grad(lambda p: loss(p, fn)))
    grads = grad_fn(params)
    expected = jax.jit(
        jax.grad(lambda p: loss(p, self._ref_op, implementation="xla"))
    )(params)

    bound_args = hlo_utils.get_bound_args(grad_fn.lower(params))
    abstract_params, abstract_mask = jax.eval_shape(
        lambda p, m: (p, m), params, mask
    )
    expected_fwd_ba = self._op.bind(
        **abstract_params,
        mask=abstract_mask,
        triangle_type=triangle_type,
        return_residuals=True,
    )
    expected_vjp_ba = self._op.vjp.bind(**expected_fwd_ba.vjp_arg_spec)  # pyrefly: ignore[missing-attribute]
    self.assertCountEqual(bound_args, (expected_fwd_ba, expected_vjp_ba))

    self.assertEqual(jax.tree.structure(grads), jax.tree.structure(expected))
    for k in params:
      with self.subTest(k):
        chex.assert_trees_all_close(
            grads[k].astype(jnp.float32),
            expected[k].astype(jnp.float32),
            atol=0.35 if dtype == jnp.bfloat16 else 0.05,
            rtol=0.03,
        )

  @parameterized.parameters(64, 128)
  def test_nonzero_mean_cotangent_layernorm_grads(self, c):
    n = 64
    params = _get_params(n=n, c=c, h=c, d=c, dtype=jnp.bfloat16)
    mask = params.pop("mask")
    chan_bias = jnp.linspace(-2.0, 3.0, c, dtype=jnp.float32)[None, None, :]
    dout = (
        jax.random.normal(jax.random.key(7), (n, n, c), dtype=jnp.float32)
        + chan_bias
    ).astype(jnp.bfloat16)

    def loss(p, fn, **kwargs):
      out = fn(**p, mask=mask, triangle_type="outgoing", **kwargs)
      return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32)) / n

    grads = jax.jit(jax.grad(lambda p: loss(p, self._op)))(params)
    expected = jax.jit(
        jax.grad(lambda p: loss(p, self._ref_op, implementation="xla"))
    )(params)
    for k in (
        "layernorm_in_scale",
        "layernorm_in_offset",
        "layernorm_out_scale",
        "layernorm_out_offset",
    ):
      with self.subTest(k):
        chex.assert_trees_all_close(
            grads[k].astype(jnp.float32),
            expected[k].astype(jnp.float32),
            atol=1.5,
            rtol=0.03,
        )

  @parameterized.product(
      triangle_type=["incoming", "outgoing"],
      dims=[
          dict(n=64, c=16, h=32, d=64),
          dict(n=64, c=64, h=64, d=64),
          dict(n=32, c=128, h=64, d=64),
          dict(n=32, c=64, h=128, d=64),
          dict(n=32, c=128, h=64, d=128),
          dict(n=64, c=128, h=128, d=128),
      ],
  )
  def test_gradients_channel_dims_and_sparse_mask(self, triangle_type, dims):
    n, c, h, d = dims["n"], dims["c"], dims["h"], dims["d"]
    params = _get_params(n=n, c=c, h=h, d=d, dtype=jnp.bfloat16)
    mask = jax.random.uniform(jax.random.key(1), (n, n)) < 0.25
    mask = mask.at[: n // 4].set(False)
    params.pop("mask")
    dout = jax.random.normal(jax.random.key(0), (n, n, d), dtype=jnp.bfloat16)

    def loss(p, fn, **kwargs):
      out = fn(**p, mask=mask, triangle_type=triangle_type, **kwargs)
      return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32)) / n

    grads = jax.jit(jax.grad(lambda p: loss(p, self._op)))(params)
    expected = jax.jit(
        jax.grad(lambda p: loss(p, self._ref_op, implementation="xla"))
    )(params)

    atol = 0.5 if max(c, h, d) > 64 else 0.35
    for k in params:
      with self.subTest(k):
        chex.assert_trees_all_close(
            grads[k].astype(jnp.float32),
            expected[k].astype(jnp.float32),
            atol=atol,
            rtol=0.03,
        )

  @parameterized.product(
      bwd_persistent=[True, False],
      triangle_type=["incoming", "outgoing"],
  )
  def test_gradients_bwd_config_overrides(self, bwd_persistent, triangle_type):
    n = 64
    params = _get_params(n=n, c=64, h=64, d=64, dtype=jnp.bfloat16)
    dout = jax.random.normal(jax.random.key(0), (n, n, 64), dtype=jnp.bfloat16)
    config = mgpu_tri_mul.Config(block_m=64)
    vjp_config = mgpu_tri_mul_vjp.Config(
        block_m=64,
        num_warps=8,
        num_stages=2,
        epilogue_block_m=32,
        epilogue_num_warps=4,
        epilogue_num_stages=3,
        prologue_persistent=bwd_persistent,
        epilogue_persistent=bwd_persistent,
    )
    vjp_op = mgpu_tri_mul_vjp.PallasMosaicGpuTriangleMultiplicationVjp(
        config=vjp_config
    )
    op_cfg = mgpu_tri_mul.PallasMosaicGpuTriangleMultiplication(
        config=config, vjp=vjp_op
    )

    def f_mosaic(p):
      return op_cfg(**p, triangle_type=triangle_type)

    def f_ref(p):
      return self._ref_op(
          **p, triangle_type=triangle_type, implementation="xla"
      )

    out, vjp_fn = jax.jit(lambda p: jax.vjp(f_mosaic, p))(params)
    (grads,) = jax.jit(vjp_fn)(dout)
    out_ref, vjp_ref = jax.jit(lambda p: jax.vjp(f_ref, p))(params)
    (expected,) = jax.jit(vjp_ref)(dout)

    chex.assert_trees_all_close(out, out_ref, atol=0.15, rtol=0.05)
    for k in params:
      if k == "mask":
        continue
      with self.subTest(k):
        chex.assert_trees_all_close(
            grads[k].astype(jnp.float32),
            expected[k].astype(jnp.float32),
            atol=0.35 * n,
            rtol=0.03,
        )

  def test_zero_epsilon_padded_rows_no_nan(self):
    n = 20
    params = _get_params(n=n, c=64, h=64, d=64, dtype=jnp.bfloat16)
    mask = params.pop("mask")
    dout = jax.random.normal(jax.random.key(0), (n, n, 64), dtype=jnp.bfloat16)
    out, vjp_fn = jax.jit(
        lambda p: jax.vjp(
            lambda q: self._op(
                **q, mask=mask, triangle_type="outgoing", epsilon=0.0
            ),
            p,
        )
    )(params)
    (grads,) = jax.jit(vjp_fn)(dout)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    for g in jax.tree.leaves(grads):
      self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

  def test_return_residuals(self):
    params = _get_params(n=64, c=64, h=64, d=64, dtype=jnp.bfloat16)
    out, residuals = jax.jit(
        lambda p: self._op(**p, triangle_type="outgoing", return_residuals=True)
    )(params)
    expected_out, expected_residuals = jax.jit(
        lambda p: self._ref_op(
            **p, triangle_type="outgoing", return_residuals=True
        )
    )(params)
    self.assertIsInstance(residuals, tuple)
    self.assertLen(residuals, 3)
    for res in residuals:
      self.assertEqual(res.shape, (64, 64, 64))
      self.assertEqual(res.dtype, jnp.bfloat16)
    self.assertEqual(out.shape, (64, 64, 64))
    chex.assert_trees_all_close(
        residuals[0], expected_residuals[0], atol=0.15, rtol=0.05
    )
    chex.assert_trees_all_close(
        residuals[1], expected_residuals[1], atol=0.15, rtol=0.05
    )
    chex.assert_trees_all_close(
        residuals[2],
        jnp.moveaxis(expected_residuals[2], -1, 0),
        atol=0.2 * 64,
        rtol=0.05,
    )
    chex.assert_trees_all_close(out, expected_out, atol=0.3, rtol=0.05)

  @parameterized.parameters("incoming", "outgoing")
  def test_small_n_boundary(self, triangle_type):
    params = _get_params(n=1, c=64, h=64, d=64, dtype=jnp.bfloat16)
    mask = params.pop("mask")
    dout = jax.random.normal(jax.random.key(0), (1, 1, 64), dtype=jnp.bfloat16)

    def f_mosaic(p):
      return self._op(**p, mask=mask, triangle_type=triangle_type)

    def f_ref(p):
      return self._ref_op(
          **p, mask=mask, triangle_type=triangle_type, implementation="xla"
      )

    out, vjp_fn = jax.jit(lambda p: jax.vjp(f_mosaic, p))(params)
    (grads,) = jax.jit(vjp_fn)(dout)
    out_ref, vjp_ref = jax.jit(lambda p: jax.vjp(f_ref, p))(params)
    (expected,) = jax.jit(vjp_ref)(dout)

    chex.assert_trees_all_close(out, out_ref, atol=0.15, rtol=0.05)
    for k in params:
      with self.subTest(k):
        chex.assert_trees_all_close(
            grads[k].astype(jnp.float32),
            expected[k].astype(jnp.float32),
            atol=0.5,
            rtol=0.05,
        )

  def test_int32_offset_overflow_raises(self):
    vjp_op = mgpu_tri_mul_vjp.PallasMosaicGpuTriangleMultiplicationVjp()

    def get_shapes(n, c):
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
    out = jax.eval_shape(f, get_shapes(n=4096, c=128))
    self.assertEqual(out.shape, (4096, 4096, 128))
    with self.assertRaisesRegex(NotImplementedError, "int32"):
      jax.eval_shape(f, get_shapes(n=4097, c=128))

    def f_vjp(p, res):
      return vjp_op(
          res,
          p["x"],
          p["x"],
          **p,
          triangle_type="outgoing",
          precision=(jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
          epsilon=1e-6,
      )

    res_4097 = (jax.ShapeDtypeStruct((128, 4097, 4097), jnp.bfloat16),) * 3
    with self.assertRaisesRegex(NotImplementedError, "int32"):
      jax.eval_shape(f_vjp, get_shapes(n=4097, c=128), res_4097)

  def test_unsupported_raises_not_implemented(self):
    vjp_op = mgpu_tri_mul_vjp.PallasMosaicGpuTriangleMultiplicationVjp()
    default_prec = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)

    with self.subTest("empty_inputs"):
      params = dict(
          x=jnp.zeros((0, 0, 64), dtype=jnp.bfloat16),
          mask=jnp.zeros((0, 0), dtype=jnp.bool_),
          projection_in_weights=jnp.zeros((64, 2, 64), dtype=jnp.bfloat16),
          gate_in_weights=jnp.zeros((64, 2, 64), dtype=jnp.bfloat16),
          projection_out_weights=jnp.zeros((64, 64), dtype=jnp.bfloat16),
          gate_out_weights=jnp.zeros((64, 64), dtype=jnp.bfloat16),
          layernorm_in_scale=jnp.zeros((64,), dtype=jnp.bfloat16),
          layernorm_in_offset=jnp.zeros((64,), dtype=jnp.bfloat16),
          layernorm_out_scale=jnp.zeros((64,), dtype=jnp.bfloat16),
          layernorm_out_offset=jnp.zeros((64,), dtype=jnp.bfloat16),
      )
      res = (jnp.zeros((64, 0, 0), dtype=jnp.bfloat16),) * 3
      with self.assertRaises(NotImplementedError):
        self._op(**params, triangle_type="outgoing")
      with self.assertRaises(NotImplementedError):
        vjp_op(
            res,
            params["x"],
            params["x"],
            **params,
            triangle_type="outgoing",
            precision=default_prec,
            epsilon=1e-6,
        )

    with self.subTest("unsupported_dtype"):
      params = _get_params(n=128, c=64, h=64, d=64, dtype=jnp.float32)
      res = (jnp.zeros((64, 128, 128), dtype=jnp.float32),) * 3
      with self.assertRaises(NotImplementedError):
        self._op(**params, triangle_type="outgoing")
      with self.assertRaises(NotImplementedError):
        vjp_op(
            res,
            params["x"],
            params["x"],
            **params,
            triangle_type="outgoing",
            precision=default_prec,
            epsilon=1e-6,
        )

    with self.subTest("unsupported_channel_dims"):
      params = _get_params(n=128, c=48, h=48, d=64, dtype=jnp.bfloat16)
      res = (jnp.zeros((48, 128, 128), dtype=jnp.bfloat16),) * 3
      with self.assertRaises(NotImplementedError):
        self._op(**params, triangle_type="outgoing")
      out = jnp.zeros((128, 128, 64), dtype=jnp.bfloat16)
      with self.assertRaises(NotImplementedError):
        vjp_op(
            res,
            out,
            out,
            **params,
            triangle_type="outgoing",
            precision=default_prec,
            epsilon=1e-6,
        )

    with self.subTest("unsupported_precision"):
      params = _get_params(n=128, c=64, h=64, d=64, dtype=jnp.bfloat16)
      res = (jnp.zeros((64, 128, 128), dtype=jnp.bfloat16),) * 3
      with self.assertRaises(NotImplementedError):
        self._op(
            **params,
            triangle_type="outgoing",
            precision=jax.lax.DotAlgorithmPreset.F32_F32_F32,
        )
      with self.assertRaises(NotImplementedError):
        vjp_op(
            res,
            params["x"],
            params["x"],
            **params,
            triangle_type="outgoing",
            precision=jax.lax.DotAlgorithmPreset.F32_F32_F32,
            epsilon=1e-6,
        )

    with self.subTest("mismatched_weight_dtype"):
      params = _get_params(n=128, c=64, h=64, d=64, dtype=jnp.bfloat16)
      res = (jnp.zeros((64, 128, 128), dtype=jnp.bfloat16),) * 3
      params["projection_in_weights"] = params["projection_in_weights"].astype(
          jnp.float16
      )
      with self.assertRaises(NotImplementedError):
        self._op(**params, triangle_type="outgoing")
      with self.assertRaises(NotImplementedError):
        vjp_op(
            res,
            params["x"],
            params["x"],
            **params,
            triangle_type="outgoing",
            precision=default_prec,
            epsilon=1e-6,
        )

    with self.subTest("mismatched_residual_dtype"):
      params = _get_params(n=128, c=64, h=64, d=64, dtype=jnp.bfloat16)
      res = (jnp.zeros((64, 128, 128), dtype=jnp.float16),) * 3
      with self.assertRaises(NotImplementedError):
        vjp_op(
            res,
            params["x"],
            params["x"],
            **params,
            triangle_type="outgoing",
            precision=default_prec,
            epsilon=1e-6,
        )

    with self.subTest("mismatched_residual_shape"):
      params = _get_params(n=128, c=64, h=64, d=64, dtype=jnp.bfloat16)
      res = (
          jnp.zeros((64, 128, 128), dtype=jnp.bfloat16),
          jnp.zeros((64, 128, 128), dtype=jnp.bfloat16),
          jnp.zeros((128, 128, 64), dtype=jnp.bfloat16),
      )
      with self.assertRaises(NotImplementedError):
        vjp_op(
            res,
            params["x"],
            params["x"],
            **params,
            triangle_type="outgoing",
            precision=default_prec,
            epsilon=1e-6,
        )

    with self.subTest("vjp_return_residuals"):
      params = _get_params(n=64, c=64, h=64, d=64, dtype=jnp.bfloat16)
      res = (jnp.zeros((64, 64, 64), dtype=jnp.bfloat16),) * 3
      with self.assertRaises(NotImplementedError):
        vjp_op(
            res,
            params["x"],
            params["x"],
            **params,
            triangle_type="outgoing",
            precision=default_prec,
            epsilon=1e-6,
            return_residuals=True,
        )

  @parameterized.parameters(64, 128)
  def test_mosaic_gpu_race_detection(self, c):
    params = _get_params(n=16, c=c, h=64, d=64, dtype=jnp.bfloat16)
    interpret_params = plgpu.InterpretGPUParams(detect_races=True)
    with plgpu.force_gpu_interpret_mode(interpret_params):
      out, vjp_fn = jax.vjp(
          lambda p: self._op(**p, triangle_type="outgoing"), params
      )
      grads = vjp_fn(jnp.ones_like(out))
      jax.block_until_ready((out, grads))

  @parameterized.parameters(
      (100, jnp.float16, "outgoing"),
      (100, jnp.bfloat16, "outgoing"),
      (160, jnp.float16, "outgoing"),
      (160, jnp.bfloat16, "outgoing"),
  )
  def test_deterministic_repeated_execution(self, n, dtype, triangle_type):
    params = _get_params(n=n, c=64, h=64, d=64, dtype=dtype)
    mask = params.pop("mask")
    dout = jax.random.normal(jax.random.key(3), (n, n, 64), dtype=dtype)

    @jax.jit
    def fwd_with_res(p):
      return self._op(
          **p, mask=mask, triangle_type=triangle_type, return_residuals=True
      )

    vjp_op = mgpu_tri_mul_vjp.PallasMosaicGpuTriangleMultiplicationVjp()

    @jax.jit
    def only_bwd(res, out, p):
      return vjp_op(
          res,
          out,
          dout,
          **p,
          mask=mask,
          triangle_type=triangle_type,
          precision=(jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
          epsilon=1e-6,
      )

    @jax.jit
    def fwd_and_bwd(p):
      out, vjp_fn = jax.vjp(
          lambda q: self._op(**q, mask=mask, triangle_type=triangle_type), p
      )
      (grads,) = vjp_fn(dout)
      return out, grads

    out0, res0 = fwd_with_res(params)
    for step_idx in range(4):
      out_i, res_i = fwd_with_res(params)
      with self.subTest(f"fwd_{step_idx}"):
        chex.assert_trees_all_equal(res_i, res0)
        chex.assert_trees_all_equal(out_i, out0)

    grads0 = only_bwd(res0, out0, params)
    leaves0 = dict(jax.tree.leaves_with_path(grads0))
    for step_idx in range(4):
      grads_i = only_bwd(res0, out0, params)
      for path, leaf_i in jax.tree.leaves_with_path(grads_i):
        with self.subTest(f"bwd_{step_idx}_{path}"):
          chex.assert_trees_all_equal(leaf_i, leaves0[path])

    fb_out0, fb_grads0 = fwd_and_bwd(params)
    for step_idx in range(4):
      fb_out_i, fb_grads_i = fwd_and_bwd(params)
      with self.subTest(f"fb_out_{step_idx}"):
        chex.assert_trees_all_equal(fb_out_i, fb_out0)
      for k in fb_grads0:
        with self.subTest(f"fb_{step_idx}_{k}"):
          chex.assert_trees_all_equal(fb_grads_i[k], fb_grads0[k])


if __name__ == "__main__":
  absltest.main()
