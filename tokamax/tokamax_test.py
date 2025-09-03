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
from tokamax import autotuning


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

    f_vjp = jax.jit(jax.grad(loss))
    f_vjp_lowered = f_vjp.lower(x, scale)

    out = f_vjp(x, scale)

    with self.subTest("DISABLE_JAX_EXPORT_CHECKS"):
      exported = export.export(
          f_vjp,
          disabled_checks=tokamax.DISABLE_JAX_EXPORT_CHECKS,
      )(
          jax.ShapeDtypeStruct(x.shape, x.dtype),
          jax.ShapeDtypeStruct(scale.shape, scale.dtype),
      )
      serialized = exported.serialize()
      f_vjp_roundtrip = export.deserialize(serialized)
      out_roundtrip = jax.jit(f_vjp_roundtrip.call)(x, scale)
      chex.assert_trees_all_close(out, out_roundtrip)

    with self.subTest("Autotune"):
      autotune_res = autotuning.autotune(f_vjp_lowered)
      with autotune_res:
        out_autotuned = f_vjp(x, scale)
        chex.assert_trees_all_close(out, out_autotuned)


if __name__ == "__main__":
  absltest.main()
