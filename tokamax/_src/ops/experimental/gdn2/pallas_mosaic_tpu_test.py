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

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.gdn2 import pallas_mosaic_tpu
from tokamax._src.ops.experimental.gdn2 import reference


class PallasTpuGatedDeltaNet2Test(absltest.TestCase):

  def test_pallas_fwd_matches_reference(self):
    key = jax.random.PRNGKey(123)
    b, t, h, k_dim, v_dim = 2, 128, 4, 16, 16
    chunk_size = 64

    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    q = jax.random.normal(k0, (b, t, h, k_dim), dtype=jnp.float32) * (k_dim**-0.5)
    k = jax.random.normal(k1, (b, t, h, k_dim), dtype=jnp.float32) * (k_dim**-0.5)
    v = jax.random.normal(k2, (b, t, h, v_dim), dtype=jnp.float32)
    g = -jax.nn.sigmoid(jax.random.normal(k3, (b, t, h, k_dim), dtype=jnp.float32))
    b_gate = jax.nn.sigmoid(jax.random.normal(k4, (b, t, h, k_dim), dtype=jnp.float32))
    w_gate = jax.nn.sigmoid(jax.random.normal(k5, (b, t, h, v_dim), dtype=jnp.float32))

    op_instance = pallas_mosaic_tpu.PallasTpuGatedDeltaNet2()
    (actual_out, _), _ = op_instance._fwd(
        q, k, v, g, b_gate, w_gate, chunk_size=chunk_size, scale=1.0
    )
    expected_out, _ = reference.gdn2_reference(
        q, k, v, g, b_gate, w_gate, chunk_size=chunk_size, scale=1.0
    )

    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
