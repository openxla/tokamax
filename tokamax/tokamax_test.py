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
"""A test of the Tokamax public API."""

from absl.testing import absltest
import chex
import jax
from jax import export
import jax.numpy as jnp
import tokamax
from tokamax._src.autotuning import api as autotuning
from tokamax._src.ops.attention import api as attention_api
from tokamax._src.ops.attention import pallas_mosaic_gpu_vjp
from tokamax._src.ops.attention import pallas_triton_vjp as pl_triton_attn_vjp
from tokamax._src.ops.normalization import api as norm_api
from tokamax._src.ops.normalization import pallas_triton_vjp as pl_norm_vjp


class TokamaxTest(absltest.TestCase):

  # TODO: Add a test for TPU.
  def test_full_example_gpu(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Current test only runs on GPU.")

    def loss(x, scale):
      x = tokamax.layer_norm(
          x, scale=scale, offset=None, implementation="triton"
      )
      x = tokamax.dot_product_attention(x, x, x, implementation="triton")
      x = tokamax.layer_norm(x, scale=scale, offset=None, implementation=None)
      x = tokamax.dot_product_attention(x, x, x, implementation="mosaic")
      return jnp.sum(x)

    channels = 64
    seq_len = 2048
    batch_size = 32
    num_heads = 16

    scale = jax.random.normal(jax.random.key(0), (channels,), dtype=jnp.float32)
    x = jax.random.normal(
        jax.random.key(1),
        (batch_size, seq_len, num_heads, channels),
        dtype=jnp.bfloat16,
    )

    f_grad = jax.jit(jax.grad(loss))
    out = f_grad(x, scale)

    with self.subTest("DISABLE_JAX_EXPORT_CHECKS"):
      exported = export.export(
          f_grad,
          disabled_checks=tokamax.DISABLE_JAX_EXPORT_CHECKS,
      )(
          jax.ShapeDtypeStruct(x.shape, x.dtype),
          jax.ShapeDtypeStruct(scale.shape, scale.dtype),
      )
      serialized = exported.serialize()
      f_grad_roundtrip = export.deserialize(serialized)
      out_roundtrip = jax.jit(f_grad_roundtrip.call)(x, scale)
      chex.assert_trees_all_close(out, out_roundtrip)

    with self.subTest("has_correct_kernels"):
      arg_specs = autotuning.get_bound_args(f_grad, x, scale)
      ops = set(a.op.__class__ for a in arg_specs)
      ops_expected = set([
          attention_api.IMPLEMENTATIONS["triton"].__class__,
          attention_api.IMPLEMENTATIONS["mosaic"].__class__,
          norm_api.IMPLEMENTATIONS["triton"].__class__,
          pl_triton_attn_vjp.PallasTritonFlashAttentionVjp,
          pallas_mosaic_gpu_vjp.PallasMosaicGpuFlashAttentionVjp,
          pl_norm_vjp.PallasTritonNormalizationVjp,
      ])
      self.assertContainsSubset(ops_expected, ops)

    with self.subTest("Autotune"):
      autotune_res = tokamax.autotune(f_grad, x, scale)
      self.assertIsInstance(autotune_res, tokamax.AutotuningResult)
      with autotune_res:
        out_autotuned = f_grad(x, scale)

        # TODO: Reduce tolerance once mgpu attention supports higher precision.
        def l2_rel(a, b):
          l2_diff = jnp.linalg.norm(a - b, axis=-1)
          l2_norm = jnp.maximum(jnp.linalg.norm(b, axis=-1), 1e-6)
          return l2_diff / l2_norm

        l2_rel_diff = l2_rel(out, out_autotuned)
        chex.assert_trees_all_close(
            jnp.zeros_like(l2_rel_diff), l2_rel_diff, atol=5e-2
        )

    with self.subTest("Benchmark"):
      f_std, args = tokamax.benchmarking.standardize_function(f_grad, x, scale)
      bench = tokamax.benchmarking.compile_benchmark(f_std, args)(args)
      self.assertGreater(bench.median_evaluation_time_ms, 0.0)


if __name__ == "__main__":
  absltest.main()
