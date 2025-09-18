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

import functools
from functools import partial
from typing import Literal
from absl.testing import absltest
import chex
import jax
from jax import random
import jax.numpy as jnp
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_kernel as ops
from tokamax._src.ops.ragged_dot import test_base


# TODO : Add QWIX tests for ragged dot once QWIX is in Ragged Dot.
# TODO: Merge QWIX quantization tests into ragged dot API tests.
# also add shapes which tile sizes do not cleanly divide to test masking.
# Also enable test_base.RaggedDotTestBase with this test.
class PallasMosaicTpuRaggedDotTest(absltest.TestCase):
  """Pallas Mosaic TPU Ragged Dot tests."""

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  def test_gmm_fwd(self):
    keys = iter(random.split(random.key(0), 1024))
    m, n, k, g = 128, 2048, 7168, 256
    dtype = jnp.bfloat16
    lhs1 = random.normal(next(keys), (m, k), dtype=dtype)
    rhs1 = random.normal(next(keys), (g, k, n), dtype=dtype)
    gs = jnp.round(
        (m - 2)
        * jax.nn.softmax(jnp.exp(1e-1 * random.normal(next(keys), (g,))))
    ).astype(jnp.int32)
    while jnp.sum(gs) > m:
      gs = jnp.round(
          (m - 2)
          * jax.nn.softmax(jnp.exp(1e-1 * random.normal(next(keys), (g,))))
      ).astype(jnp.int32)

    @jax.jit
    def run_gmm(lhs, rhs, group_sizes):
      return pallas_mosaic_tpu.PallasMosaicTpuRaggedDot()(
          lhs, rhs, group_sizes=group_sizes
      )

    output = run_gmm(lhs1, rhs1, gs)

    rd_output = jax.lax.ragged_dot(
        lhs1, rhs1, gs, precision=jax.lax.Precision.DEFAULT
    )

    mask = jnp.where(jnp.arange(output.shape[0]) < jnp.sum(gs), 1.0, 0.0)

    ratio = jnp.linalg.norm(output - rd_output, axis=-1) / jnp.linalg.norm(
        rd_output, axis=-1
    )
    ratio = ratio * mask
    chex.assert_trees_all_close(ratio, jnp.zeros_like(ratio), atol=2e-3)
    # TODO: Add numerics test for backwards pass.

  def test_gmm_quantized(self):
    # TODO: Add QWIX tests for ragged dot once QWIX is in Ragged Dot.
    self.skipTest("Quantized gmm coming soon.")
    keys = iter(random.split(random.key(0), 1024))
    m, n, k, g = 128, 2048, 7168, 256
    dtype = jnp.bfloat16
    lhs1 = random.normal(next(keys), (m, k), dtype=dtype)
    rhs1 = random.normal(next(keys), (g, k, n), dtype=dtype)
    gs = jnp.round(
        (m - 2)
        * jax.nn.softmax(jnp.exp(1e-1 * random.normal(next(keys), (g,))))
    ).astype(jnp.int32)
    while jnp.sum(gs) > m:
      gs = jnp.round(
          (m - 2)
          * jax.nn.softmax(jnp.exp(1e-1 * random.normal(next(keys), (g,))))
      ).astype(jnp.int32)
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


if __name__ == "__main__":
  absltest.main()
