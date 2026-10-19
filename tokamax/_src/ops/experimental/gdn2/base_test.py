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

"""Unit tests for GDN-2 base operator and API."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from tokamax._src.ops.experimental.gdn2 import api, base


class GatedDeltaNet2BaseTest(absltest.TestCase):

  def test_base_and_api(self):
    key = jax.random.PRNGKey(42)
    batch_size, seq_len, num_heads, key_dim, val_dim = 2, 64, 4, 16, 16

    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    q = jax.random.normal(k0, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32) * (key_dim**-0.5)
    k = jax.random.normal(k1, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32) * (key_dim**-0.5)
    v = jax.random.normal(k2, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32)
    g = -jax.nn.sigmoid(jax.random.normal(k3, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32))
    b_gate = jax.nn.sigmoid(jax.random.normal(k4, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32))
    w_gate = jax.nn.sigmoid(jax.random.normal(k5, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32))

    op_instance = base.GatedDeltaNet2()
    (base_out, base_s), _ = op_instance._fwd(
        q, k, v, g, b_gate, w_gate, scale=1.0, output_final_state=True
    )
    ref_out, ref_s = base.gdn2_reference(
        q, k, v, g, b_gate, w_gate, scale=1.0, output_final_state=True
    )
    api_out, api_s = api.gated_delta_net_2(
        q, k, v, g, b_gate, w_gate, scale=1.0, output_final_state=True, implementation="xla"
    )

    np.testing.assert_allclose(base_out, ref_out, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(base_s, ref_s, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(api_out, ref_out, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(api_s, ref_s, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
