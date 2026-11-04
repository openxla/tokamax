# Copyright 2026 Google LLC. All Rights Reserved.
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

"""Unit tests for MoBA Pallas TPU operator."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.moba import base
from tokamax._src.ops.experimental.moba import pallas_mosaic_tpu


class PallasMosaicTpuMixtureOfBlockAttentionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() == "gpu":
      self.skipTest("Pallas Mosaic TPU kernel is not supported on GPU.")

  def test_pallas_fwd_matches_reference(self):
    key = jax.random.PRNGKey(123)
    batch_size, num_heads, seq_len, head_dim = 2, 4, 256, 16
    chunk_size = 64
    topk = 2

    k0, k1, k2 = jax.random.split(key, 3)
    q = jax.random.normal(
        k0, (batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32
    )
    k = jax.random.normal(
        k1, (batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32
    )
    v = jax.random.normal(
        k2, (batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32
    )

    op_instance = pallas_mosaic_tpu.PallasMosaicTpuMixtureOfBlockAttention(
        bypass_device_check=True
    )
    actual_out = op_instance(
        q, k, v, topk=topk, chunk_size=chunk_size
    )
    expected_out, _ = base.moba_reference(
        q, k, v, topk=topk, chunk_size=chunk_size
    )

    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-5, atol=1e-5)

  def test_invalid_seq_len_raises(self):
    q = jnp.zeros((1, 2, 300, 16))
    k = jnp.zeros((1, 2, 300, 16))
    v = jnp.zeros((1, 2, 300, 16))
    op_instance = pallas_mosaic_tpu.PallasMosaicTpuMixtureOfBlockAttention()
    with self.assertRaisesRegex(ValueError, "must be divisible by chunk_size"):
      op_instance._fwd(q, k, v, chunk_size=256)

  def test_invalid_config_raises(self):
    q = jnp.zeros((1, 2, 256, 16))
    k = jnp.zeros((1, 2, 256, 16))
    v = jnp.zeros((1, 2, 256, 16))
    op_instance = pallas_mosaic_tpu.PallasMosaicTpuMixtureOfBlockAttention()
    with self.assertRaises(NotImplementedError):
      op_instance._fwd(
          q,
          k,
          v,
          chunk_size=256,
          topk=-1,
      )

  def test_tpu_export_lowering(self):
    shape = jax.ShapeDtypeStruct((2, 2, 256, 16), jnp.float32)
    exported = jax.export.export(
        jax.jit(
            lambda q, k, v: pallas_mosaic_tpu.pallas_mosaic_tpu_kernel.moba_pallas_fwd(
                q, k, v, topk=2, chunk_size=64, interpret=False
            )[0]
        ),
        platforms=("tpu",),
    )(shape, shape, shape)
    self.assertIsNotNone(exported.mlir_module())


if __name__ == "__main__":
  absltest.main()
