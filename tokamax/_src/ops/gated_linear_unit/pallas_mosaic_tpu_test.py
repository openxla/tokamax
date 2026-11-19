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
"""Tests for Pallas Mosaic TPU GatedLinearUnit."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src import batching
from tokamax._src.ops.gated_linear_unit import base
from tokamax._src.ops.gated_linear_unit import pallas_mosaic_tpu
from tokamax._src.ops.gated_linear_unit import test_base


class PallasMosaicTpuGatedLinearUnitTest(test_base.GatedLinearUnitTestBase):

  def __init__(self, *args):
    super().__init__(
        *args, glu_fn=pallas_mosaic_tpu.PallasMosaicTpuGatedLinearUnit()
    )

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPU.")
    super().setUp()

  @parameterized.parameters(
      dict(
          m_dim=128,
          k=128,
          n_dim=128,
          dtype=jnp.bfloat16,
          activation=jax.nn.silu,
      ),
      dict(
          m_dim=256,
          k=512,
          n_dim=256,
          dtype=jnp.bfloat16,
          activation=jax.nn.gelu,
      ),
      dict(
          m_dim=128, k=256, n_dim=128, dtype=jnp.float32, activation=jax.nn.relu
      ),
      dict(
          m_dim=128, k=128, n_dim=128, dtype=jnp.float16, activation=jax.nn.silu
      ),
      dict(
          m_dim=100,
          k=100,
          n_dim=150,
          dtype=jnp.bfloat16,
          activation=jax.nn.silu,
      ),
      dict(
          m_dim=256,
          k=256,
          n_dim=256,
          dtype=jnp.bfloat16,
          activation=jax.nn.silu,
          use_tuple_weights=True,
      ),
      dict(
          m_dim=128,
          k=256,
          n_dim=128,
          dtype=jnp.float32,
          activation=jax.nn.silu,
          precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
      ),
      dict(m_dim=128, k=128, n_dim=128, dtype=jnp.bfloat16, activation=None),
      dict(m_dim=64, k=64, n_dim=128, dtype=jnp.bfloat16, activation=None),
  )
  def test_shapes_and_dtypes_with_vjp(
      self,
      *,
      m_dim: int,
      k: int,
      n_dim: int,
      dtype: jax.typing.DTypeLike,
      activation,
      use_tuple_weights: bool = False,
      precision: jax.lax.PrecisionLike = None,
  ):
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(42), 3)
    lhs = jax.random.normal(rng0, (m_dim, k), dtype=dtype)
    weights = jax.random.normal(rng1, (k, 2, n_dim), dtype=dtype) / jnp.sqrt(
        jnp.array(k, dtype=jnp.float32)
    ).astype(dtype)
    dout = jax.random.normal(rng2, (m_dim, n_dim), dtype=dtype)

    if use_tuple_weights:
      weights = jnp.unstack(weights, axis=1)

    glu_op = pallas_mosaic_tpu.PallasMosaicTpuGatedLinearUnit()
    ref_op = base.GatedLinearUnit()

    fn = functools.partial(glu_op, activation=activation, precision=precision)
    ref_fn = functools.partial(
        ref_op, activation=activation, precision=precision
    )

    out, vjp_fn = jax.vjp(fn, lhs, weights)
    out_ref, vjp_ref_fn = jax.vjp(ref_fn, lhs, weights)

    atol = 5e-2 if dtype != jnp.float32 or precision is not None else 1e-2
    rtol = 2e-2 if dtype != jnp.float32 or precision is not None else 1e-2
    chex.assert_trees_all_close(out, out_ref, atol=atol, rtol=rtol)
    grads = vjp_fn(dout)
    grads_ref = vjp_ref_fn(dout)
    chex.assert_trees_all_close(grads, grads_ref, atol=0.2, rtol=0.05)

  def test_default_vjp_is_lane_dense(self):
    glu_op = pallas_mosaic_tpu.PallasMosaicTpuGatedLinearUnit()
    self.assertIsInstance(
        glu_op.vjp, pallas_mosaic_tpu.PallasMosaicTpuGatedLinearUnitVjp
    )

  @parameterized.product(
      use_tuple_weights=(False, True),
      # (batch, K, N, dtype): the first three train with the Pallas backward
      # kernel (K=64 is padded), the fourth with the lane-dense VJP, the last
      # with the base VJP (or the lane-dense one on TPU v5).
      shape=(
          ((4, 1024), 128, 256, jnp.bfloat16),
          ((2, 2048), 64, 128, jnp.bfloat16),
          ((32768,), 128, 256, jnp.float32),
          ((5, 4096), 128, 256, jnp.float32),
          ((3, 64), 128, 256, jnp.bfloat16),
      ),
  )
  def test_jitted_grads(self, use_tuple_weights, shape):
    rows, k, n, dtype = shape
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(7), 3)
    lhs = jax.random.normal(rng0, (*rows, k), dtype=dtype)
    weights = jax.random.normal(rng1, (k, 2, n), dtype=dtype) / 10.0
    dout = jax.random.normal(rng2, (*rows, n), dtype=dtype)
    if use_tuple_weights:
      weights = jnp.unstack(weights, axis=1)

    def grads(op, lhs, weights, dout, **kwargs):
      fn = functools.partial(op, activation=jax.nn.swish, **kwargs)
      return jax.jit(lambda x, w: jax.vjp(fn, x, w)[1](dout))(lhs, weights)

    actual = grads(
        pallas_mosaic_tpu.PallasMosaicTpuGatedLinearUnit(), lhs, weights, dout
    )
    xla = grads(base.GatedLinearUnit(), lhs, weights, dout)
    chex.assert_trees_all_equal_shapes_and_dtypes(actual, xla)
    # Both are compared with an f32 reference: the kernel's error must be in
    # line with XLA's, which rounds the same bf16 intermediates differently.
    to_f32 = lambda t: jax.tree.map(lambda a: a.astype(jnp.float32), t)
    expected = grads(
        base.GatedLinearUnit(),
        *to_f32((lhs, weights, dout)),
        precision=jax.lax.Precision.HIGHEST,
    )

    def rel_err(got, want):
      got, want = got.astype(jnp.float32), want.astype(jnp.float32)
      return float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))

    for got, ref, want in zip(*map(jax.tree.leaves, (actual, xla, expected))):
      self.assertLessEqual(rel_err(got, want), 1.5 * rel_err(ref, want) + 1e-3)

  def test_vmapped_grads(self):
    # Vmapped shapes outside the fused kernel use the lane-dense VJP.
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(5), 3)
    lhs = jax.random.normal(rng0, (8, 240, 32, 128), dtype=jnp.float32)
    weights = jax.random.normal(rng1, (128, 2, 256), dtype=jnp.float32) / 10.0
    dout = jax.random.normal(rng2, (8, 240, 32, 256), dtype=jnp.float32)

    def grads(op, **kwargs):
      fn = jax.vmap(
          functools.partial(op, activation=jax.nn.silu, **kwargs),
          in_axes=(0, None),
      )
      return jax.jit(lambda x, w: jax.vjp(fn, x, w)[1](dout))(lhs, weights)

    actual = grads(pallas_mosaic_tpu.PallasMosaicTpuGatedLinearUnit())
    expected = grads(
        base.GatedLinearUnit(), precision=jax.lax.Precision.HIGHEST
    )
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
      rel_err = jnp.linalg.norm(got - want) / jnp.linalg.norm(want)
      self.assertLess(float(rel_err), 1e-2)

  def test_training_strategy(self):
    batched = batching.BatchedShapeDtype(
        (768, 768), jnp.float32, vmap_axes=((0, 8),)
    )
    self.assertEqual(pallas_mosaic_tpu._training_strategy(batched, 1536),
                     "lane_dense")
    unbatched = jax.ShapeDtypeStruct((768, 768), jnp.float32)
    self.assertEqual(pallas_mosaic_tpu._training_strategy(unbatched, 1536),
                     "xla")
    pair = jax.ShapeDtypeStruct((256, 256, 128), jnp.bfloat16)
    self.assertEqual(pallas_mosaic_tpu._training_strategy(pair, 256), "fused")
    # Too few rows per weight element for the fused kernel.
    small = jax.ShapeDtypeStruct((2304, 384), jnp.bfloat16)
    expected = "lane_dense" if pltpu.get_tpu_info().generation == 5 else "xla"
    self.assertEqual(pallas_mosaic_tpu._training_strategy(small, 768),
                     expected)

  @parameterized.parameters((128, 512), (64, 256))
  def test_forward_more_accurate_than_xla(self, k, n):
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(3))
    lhs = jax.random.normal(rng0, (8, 1024, k), dtype=jnp.bfloat16)
    weights = jax.random.normal(rng1, (k, 2, n), dtype=jnp.bfloat16)

    def fwd(op, lhs, weights, precision):
      fn = functools.partial(op, activation=jax.nn.silu, precision=precision)
      return jax.jit(fn)(lhs, weights).astype(jnp.float32)

    bf16 = jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    actual = fwd(
        pallas_mosaic_tpu.PallasMosaicTpuGatedLinearUnit(), lhs, weights, bf16
    )
    xla = fwd(base.GatedLinearUnit(), lhs, weights, bf16)
    expected = fwd(
        base.GatedLinearUnit(),
        lhs.astype(jnp.float32),
        weights.astype(jnp.float32),
        jax.lax.Precision.HIGHEST,
    )
    rel_err = lambda got: float(
        jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
    )
    # XLA rounds the projections to bf16 before the activation; the kernel
    # keeps them in f32.
    self.assertLess(rel_err(actual), rel_err(xla))

  def test_vmap_execution(self):
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(7))
    lhs = jax.random.normal(rng0, (4, 128, 256), dtype=jnp.bfloat16)
    weights = jax.random.normal(rng1, (256, 2, 256), dtype=jnp.bfloat16) / 16.0

    glu_op = pallas_mosaic_tpu.PallasMosaicTpuGatedLinearUnit()
    ref_op = base.GatedLinearUnit()

    fn = jax.vmap(
        functools.partial(glu_op, activation=jax.nn.silu), in_axes=(0, None)
    )
    ref_fn = jax.vmap(
        functools.partial(ref_op, activation=jax.nn.silu), in_axes=(0, None)
    )

    out = jax.jit(fn)(lhs, weights)
    out_ref = jax.jit(ref_fn)(lhs, weights)
    chex.assert_trees_all_close(out, out_ref, atol=5e-2, rtol=2e-2)

  @parameterized.parameters(
      # (M, N, generation, tile_m) for f32 with K=768. Mid-size M uses tiles
      # dividing M: 256 rows on v5, the largest fitting 16 MiB of VMEM on v6+.
      # Large M keeps 512-row tiles.
      (768, 1536, 5, 256),
      (1280, 1536, 7, 256),
      (3840, 1536, 7, 384),
      (4096, 1536, 5, 256),
      (16384, 1536, 5, 512),
  )
  def test_heuristic_tile_m(self, m, n, generation, tile_m):
    config = pallas_mosaic_tpu._select_heuristic_config(
        m, 768, n, 4, return_residuals=False, num_cores=1, generation=generation
    )
    self.assertEqual(config.tile_m, tile_m)

  def test_autotuning_configs_non_empty(self):
    glu_op = pallas_mosaic_tpu.PallasMosaicTpuGatedLinearUnit()
    ba = glu_op.bind(
        jax.ShapeDtypeStruct((1024, 1024), jnp.bfloat16),
        jax.ShapeDtypeStruct((1024, 2, 2048), jnp.bfloat16),
        activation=jax.nn.silu,
    )
    configs = glu_op._get_autotuning_configs(ba)
    self.assertNotEmpty(configs)
    self.assertIn(glu_op._get_heuristics_config(ba), configs)

  def test_return_residuals_unaligned_with_batch(self):
    rng0, rng1 = jax.random.split(jax.random.PRNGKey(123))
    lhs = jax.random.normal(rng0, (2, 100, 70), dtype=jnp.bfloat16)
    weights = jax.random.normal(rng1, (70, 2, 150), dtype=jnp.bfloat16) / 10.0

    glu_op = pallas_mosaic_tpu.PallasMosaicTpuGatedLinearUnit()
    ref_op = base.GatedLinearUnit()

    out, res = glu_op(lhs, weights, activation=None, return_residuals=True)
    out_ref, res_ref = ref_op(
        lhs, weights, activation=None, return_residuals=True
    )

    self.assertEqual(out.shape, (2, 100, 150))
    self.assertEqual(res.shape, (2, 100, 2, 150))
    chex.assert_trees_all_close(out, out_ref, atol=5e-2, rtol=2e-2)
    chex.assert_trees_all_close(res, res_ref, atol=5e-2, rtol=2e-2)


if __name__ == "__main__":
  absltest.main()
