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

"""Unit tests for Pallas TPU Native Sparse Attention (NSA) operator."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.nsa import pallas_mosaic_tpu
from tokamax._src.ops.experimental.nsa import reference


class PallasTpuNativeSparseAttentionTest(absltest.TestCase):

  def test_pallas_fwd_matches_reference(self):
    key = jax.random.PRNGKey(42)
    b, h, t, d = 2, 4, 256, 16
    chunk_size = 64
    topk = 2
    window = 32
    cmp_size = 16

    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    q = jax.random.normal(k0, (b, h, t, d), dtype=jnp.float32)
    k = jax.random.normal(k1, (b, h, t, d), dtype=jnp.float32)
    v = jax.random.normal(k2, (b, h, t, d), dtype=jnp.float32)
    gc = jax.nn.sigmoid(jax.random.normal(k3, (b, h, t, d), dtype=jnp.float32))
    gs = jax.nn.sigmoid(jax.random.normal(k4, (b, h, t, d), dtype=jnp.float32))
    gw = jax.nn.sigmoid(jax.random.normal(k5, (b, h, t, d), dtype=jnp.float32))

    op_instance = pallas_mosaic_tpu.PallasTpuNativeSparseAttention()
    config = pallas_mosaic_tpu.Config(
        chunk_size=chunk_size,
        topk=topk,
        window=window,
        cmp_block_size=cmp_size,
    )
    (actual_out, _), _ = op_instance._fwd(
        q, k, v, gc, gs, gw, config=config
    )
    expected_out, _ = reference.nsa_reference(
        q,
        k,
        v,
        gc,
        gs,
        gw,
        topk=topk,
        chunk_size=chunk_size,
        window=window,
        cmp_block_size=cmp_size,
    )

    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
