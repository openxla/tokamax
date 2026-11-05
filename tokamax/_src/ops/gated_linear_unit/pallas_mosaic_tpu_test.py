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
import jax.numpy as jnp
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

  @parameterized.product(
      shape=((8192, 4096, 4096), (16384, 512, 256)),
      dtype=(jnp.bfloat16, jnp.float32),
      return_residuals=(False, True),
  )
  def test_largest_configs_fit_in_vmem(self, shape, dtype, return_residuals):
    # Configs are picked against the full VMEM capacity, so the largest ones
    # must compile without a scoped VMEM OOM.
    m_dim, k_dim, n_dim = shape
    x = jax.ShapeDtypeStruct((m_dim, k_dim), dtype)
    weights = jax.ShapeDtypeStruct((k_dim, 2, n_dim), dtype)
    glu_op = pallas_mosaic_tpu.PallasMosaicTpuGatedLinearUnit()
    ba = glu_op.bind(
        x, weights, activation=jax.nn.silu, return_residuals=return_residuals
    )
    itemsize = jnp.dtype(dtype).itemsize

    def estimate(config):
      return pallas_mosaic_tpu._estimate_vmem_bytes(
          config.tile_m,
          config.tile_k,
          config.tile_n,
          config.input_buffer_count,
          itemsize,
          single_k_step=config.tile_k >= k_dim,
          return_residuals=return_residuals,
      )

    largest = max(glu_op._get_autotuning_configs(ba), key=estimate)
    for config in {glu_op._get_heuristics_config(ba), largest}:
      with self.subTest(config=config):
        fn = functools.partial(
            pallas_mosaic_tpu.PallasMosaicTpuGatedLinearUnit(config=config),
            activation=jax.nn.silu,
            return_residuals=return_residuals,
        )
        jax.jit(fn).lower(x, weights).compile()

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
