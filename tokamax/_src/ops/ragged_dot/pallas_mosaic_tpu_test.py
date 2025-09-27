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
"""Tokamax Megablox TPU tests for core functionality."""

from functools import partial  # pylint: disable=g-importing-member
import itertools
import math
from absl.testing import absltest, parameterized  # pylint: disable=g-importing-member
import chex
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from tokamax._src import quantization
from tokamax._src.ops import op
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu
from tokamax._src.ops.ragged_dot import test_base

QuantizedArray = quantization.QuantizedArray

TESTCASE_SAMPLES = 16


def _relative_err_fn(x, x_ref, axis=-1):
  num = jnp.linalg.norm(x - x_ref, axis=axis)
  denom = jnp.maximum(jnp.linalg.norm(x_ref, axis=axis), 1e-6)
  return num / denom


def _generate_group_sizes(key, *, target_m: int, g: int) -> jax.Array:
  """Generate group sizes for a given target m."""
  assert target_m >= 0
  if target_m == 0:
    return jnp.zeros(g, dtype=jnp.int32)
  gs = jnp.round(
      target_m * jax.nn.softmax(3e-1 * random.normal(key, (g,)))
  ).astype(jnp.int32)
  while jnp.sum(gs) != target_m:
    idx = jnp.argmax(gs)
    gs = gs.at[idx].set(jnp.maximum(target_m - jnp.sum(gs) + gs[idx], 0))
  return gs


def _generate_gmm_inputs(
    key, *, m: int, n: int, k: int, g: int, random_values: bool = True,
    dtype: jnp.dtype = jnp.bfloat16
):
  """Generate group sizes for a given target m."""
  keys = iter(random.split(key, 1024))

  def init_fn(shape):
    if random_values:
      return random.normal(next(keys), shape, dtype=dtype) / (shape[-1] ** 0.5)
    else:
      return jnp.ones(shape, dtype=dtype) / (shape[-1] ** 0.5)

  lhs, rhs, dout = init_fn((m, k)), init_fn((g, k, n)), init_fn((m, n))
  return lhs, rhs, dout


def sampled_product(**kws):
  keys, values = list(kws.keys()), list(kws.values())
  all_combinations = list(itertools.product(*values))
  combinations = np.random.default_rng(0).choice(
      all_combinations,
      size=min(TESTCASE_SAMPLES, len(all_combinations)),
      replace=False
  )
  return parameterized.parameters(
      *[dict(zip(keys, values)) for values in combinations]
  )

_round_to_LANES = lambda x: math.ceil(x / 128) * 128


# TODO : Add QWIX tests for ragged dot once QWIX is in Ragged Dot.
# TODO: Merge QWIX quantization tests into ragged dot API tests.
# also add shapes which tile sizes do not cleanly divide to test masking.
class PallasMosaicTpuRaggedDotTest(test_base.RaggedDotTestBase):
  """Pallas Mosaic TPU Ragged Dot tests."""

  def __init__(self, *args):

    def fn(lhs, rhs, *, config=None, **kwargs):
      lhs = lhs.recompose() if isinstance(lhs, QuantizedArray) else lhs
      rhs = rhs.recompose() if isinstance(rhs, QuantizedArray) else rhs
      if any(s < 128 for s in (tuple(lhs.shape) + tuple(rhs.shape))):
        self.skipTest(f"Skipping ragged dot inputs, {lhs.shape=} {rhs.shape=},"
                      " that are too small for TPU.")
      return pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(config=config)(
          lhs, rhs, **kwargs
      )

    super().__init__(*args, dot_fn=fn)
    self.tol = dict(atol=1e-2, rtol=0)

    def assert_close(a, b, **tol):
      l2_diff = jnp.linalg.norm(a - b, axis=-1)
      l2_norm = jnp.maximum(jnp.linalg.norm(b, axis=-1), 1e-6)
      l2_rel = l2_diff / l2_norm
      chex.assert_trees_all_close(
          l2_rel, jnp.zeros_like(l2_rel), **dict(self.tol, **tol)
      )

    self.assert_close = assert_close

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  @sampled_product(
      m=[128, 256],
      n=[1024, 2048],
      k=[7168, 1536],
      g=[32, 256],
      full_groups=[True, False],
      dtype=[jnp.bfloat16, jnp.float32],
      dividing_tile_k=[True, False],
  )
  def test_gmm_fwd(self, m, n, k, g, full_groups, dtype, dividing_tile_k):
    if g * k * n > 3 * (1024 ** 3):
      self.skipTest("Skipping very large experts case.")
    keys = iter(random.split(random.key(0), 1024))
    target_m = m if full_groups else (m // 2)
    lhs, rhs, _ = _generate_gmm_inputs(
        next(keys), m=m, n=n, k=k, g=g, dtype=dtype
    )
    gs = _generate_group_sizes(next(keys), target_m=target_m, g=g)
    tile_k = 128 if dividing_tile_k else _round_to_LANES(k * 3 / 4)
    config = pallas_mosaic_tpu.Config(
        gmm_tiling=(128, tile_k, 128),
        gmm_rhs_transpose_tiling=(128, 128, 128),
        tgmm_tiling=(128, 128, 128),
    )

    @jax.jit
    def run_gmm(lhs, rhs, group_sizes):
      return pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(config=config)(
          lhs, rhs, group_sizes=group_sizes
      )

    o = run_gmm(lhs, rhs, gs)
    o_ref = jax.lax.ragged_dot(lhs, rhs, gs)
    assert dtype == o.dtype
    mask = jnp.arange(o.shape[0]) < jnp.sum(gs)
    norm_error = _relative_err_fn(o, o_ref, axis=-1)
    ratio = jnp.where(mask, norm_error, 0.0)

    chex.assert_trees_all_close(ratio, jnp.zeros_like(ratio), atol=1e-3)
    # TODO: Add numerics test for backwards pass.

  @sampled_product(
      m=[128, 256],
      n=[1024, 2048],
      k=[7168, 1536],
      g=[32, 64],
      full_groups=[True, False],
      dtype=[jnp.bfloat16, jnp.float32],
      dividing_tile_k=[True, False],
  )
  def test_gmm_bwd(self, m, n, k, g, full_groups, dtype, dividing_tile_k):
    if g * k * n > 3 * (1024 ** 3):
      self.skipTest("Skipping very large experts case.")
    keys = iter(random.split(random.key(0), 1024))
    target_m = m if full_groups else (m // 2)
    lhs, rhs, dout = _generate_gmm_inputs(
        next(keys), m=m, n=n, k=k, g=g, dtype=dtype
    )
    gs = _generate_group_sizes(next(keys), target_m=target_m, g=g)
    tile_k = 128 if dividing_tile_k else _round_to_LANES(k * 3 / 4)
    config = pallas_mosaic_tpu.Config(
        gmm_tiling=(128, tile_k, 128),
        gmm_rhs_transpose_tiling=(128, tile_k, 128),
        tgmm_tiling=(128, tile_k, 128),
    )

    @jax.jit
    def gmm_vjp(lhs, rhs, group_sizes):
      fwd_fn = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(config=config)
      o, vjp_fn = jax.vjp(partial(fwd_fn, group_sizes=group_sizes), lhs, rhs)
      return o, vjp_fn(dout)

    @jax.jit
    def gmm_vjp_ref(lhs, rhs, group_sizes):
      o, vjp_fn = jax.vjp(partial(jax.lax.ragged_dot, group_sizes=group_sizes),
                          lhs, rhs)
      return o, vjp_fn(dout)

    o, (dlhs, drhs) = gmm_vjp(lhs, rhs, gs)
    o_ref, (dlhs_ref, drhs_ref) = gmm_vjp_ref(lhs, rhs, gs)
    assert dtype == o.dtype
    assert dtype == dlhs.dtype
    assert dtype == drhs.dtype

    mask = jnp.arange(o.shape[0]) < jnp.sum(gs)
    o_err = jnp.where(mask, _relative_err_fn(o, o_ref, axis=-1), 0.0)
    chex.assert_trees_all_close(o_err, jnp.zeros_like(o_err), atol=1e-3)

    dlhs_err = jnp.where(mask, _relative_err_fn(dlhs, dlhs_ref, axis=-1), 0.0)
    chex.assert_trees_all_close(dlhs_err, jnp.zeros_like(dlhs_err), atol=1e-3)

    drhs_err = _relative_err_fn(drhs, drhs_ref, axis=-1)
    chex.assert_trees_all_close(drhs_err, jnp.zeros_like(drhs_err), atol=1e-3)

  def test_gmm_quantized(self):
    # TODO: Add QWIX tests for ragged dot once QWIX is in Ragged
    # Dot.
    self.skipTest("Quantized gmm coming soon.")
    keys = iter(random.split(random.key(0), 1024))
    m, n, k, g = 128, 2048, 7168, 256
    dtype = jnp.bfloat16
    lhs1 = random.normal(next(keys), (m, k), dtype=dtype)
    rhs1 = random.normal(next(keys), (g, k, n), dtype=dtype)
    gs = _generate_group_sizes(next(keys), target_m=m, g=g)
    mask = jnp.arange(m) < jnp.sum(gs)

    @partial(jax.jit, static_argnames=("qtype",))
    def run_gmm(lhs, rhs, group_sizes, qtype):
      def fwd(lhs, rhs):
        out = _gmm(
            lhs,
            rhs,
            group_sizes,
            tiling=(128, 128, 128, 128, 128, 128),
            lhs_quantize_dtype=qtype,
            rhs_quantize_dtype=qtype,
        )
        out = jnp.where(mask[:, None], out, 0.0)
        return jnp.mean(out), out

      (_, out), (dlhs, drhs) = jax.value_and_grad(
          fwd, has_aux=True, argnums=(0, 1)
      )(lhs, rhs)
      dlhs = jnp.where(mask[:, None], dlhs, 0.0)
      return out, dlhs, drhs

    non_quantized = run_gmm(lhs1, rhs1, gs, None)
    int8_quantized = run_gmm(lhs1, rhs1, gs, jnp.int8)

    @jax.jit
    def rel_mae(x, y):
      return jnp.mean(jnp.abs(x - y) / jnp.maximum(jnp.abs(x), 1e-8))

    print(jax.tree.map(rel_mae, non_quantized, int8_quantized))
    # TODO: Test output vector norm errors.
    self.assertLess(rel_mae(non_quantized[0], int8_quantized[0]), 0.12)
    self.assertLess(rel_mae(non_quantized[1], int8_quantized[1]), 0.09)
    self.assertLess(rel_mae(non_quantized[2], int8_quantized[2]), 0.02)

  def test_bench_memory_bound(self):
    self.skipTest("GPU Only test.")

  def test_maxtext_config(self):
    # Test to ensure that we can get the correct config for a specific model.
    # For this test we are using jax.ShapeDtypeStruct instead of jax.Array
    # because jax.Array would trigger OOM for our tests.
    tpu_ragged_dot = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot()
    maxtext_config = tpu_ragged_dot._get_heuristics_config(
        op.BoundArguments(
            op=tpu_ragged_dot,
            arguments={
                "lhs": jax.ShapeDtypeStruct((262144, 7168), dtype=jnp.bfloat16),
                "rhs": jax.ShapeDtypeStruct(
                    (256, 7168, 2048), dtype=jnp.bfloat16
                ),
            },
        )
    )
    self.assertEqual(maxtext_config.gmm_tiling, (256, 7168, 512))


if __name__ == "__main__":
  absltest.main()
