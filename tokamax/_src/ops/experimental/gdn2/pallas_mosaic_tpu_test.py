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

"""Unit tests for GDN-2 Pallas TPU operator."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from tokamax._src.ops.experimental.gdn2 import base, pallas_mosaic_tpu


class PallasTpuGatedDeltaNet2Test(absltest.TestCase):

  def test_pallas_fwd_matches_reference(self):
    key = jax.random.PRNGKey(123)
    batch_size, seq_len, num_heads, key_dim, val_dim = 2, 128, 4, 16, 16
    chunk_size = 64

    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    q = jax.random.normal(k0, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32) * (key_dim**-0.5)
    k = jax.random.normal(k1, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32) * (key_dim**-0.5)
    v = jax.random.normal(k2, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32)
    g = -jax.nn.sigmoid(jax.random.normal(k3, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32))
    b_gate = jax.nn.sigmoid(jax.random.normal(k4, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32))
    w_gate = jax.nn.sigmoid(jax.random.normal(k5, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32))

    op_instance = pallas_mosaic_tpu.PallasTpuGatedDeltaNet2()
    (actual_out, actual_s), _ = op_instance._fwd(
        q, k, v, g, b_gate, w_gate, chunk_size=chunk_size, scale=1.0, output_final_state=True
    )
    expected_out, expected_s = base.gdn2_reference(
        q, k, v, g, b_gate, w_gate, chunk_size=chunk_size, scale=1.0, output_final_state=True
    )

    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_s, expected_s, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
