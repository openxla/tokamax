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

"""Unit tests for Raven GSA Pallas TPU operator."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.raven import pallas_mosaic_tpu
from tokamax._src.ops.experimental.raven import reference


class PallasTpuRavenGSATest(absltest.TestCase):

  def test_pallas_fwd_matches_reference(self):
    key = jax.random.PRNGKey(123)
    b, t, h, k_dim, num_slots = 2, 128, 4, 16, 8
    chunk_size = 64

    k0, k1, k2, k3 = jax.random.split(key, 4)
    q = jax.random.normal(k0, (b, t, h, k_dim), dtype=jnp.float32)
    k = jax.random.normal(k1, (b, t, h, k_dim), dtype=jnp.float32)
    s = jax.random.normal(k2, (b, t, h, num_slots), dtype=jnp.float32)
    g = -jax.nn.sigmoid(jax.random.normal(k3, (b, t, h, num_slots), dtype=jnp.float32))

    op_instance = pallas_mosaic_tpu.PallasTpuRavenGSA()
    (actual_out, _), _ = op_instance._fwd(
        q, k, s, g, chunk_size=chunk_size
    )
    expected_out, _ = reference.raven_stage1_reference(
        q, k, s, g, chunk_size=chunk_size
    )

    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
