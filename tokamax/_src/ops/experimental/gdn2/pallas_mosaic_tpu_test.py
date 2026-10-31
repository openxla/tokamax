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
from tokamax._src.ops.experimental.gdn2 import base, pallas_mosaic_tpu


class PallasMosaicTpuGatedDeltaNet2Test(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() == "gpu":
      self.skipTest("Pallas Mosaic TPU kernel is not supported on GPU.")

  def test_pallas_fwd_matches_reference(self):
    key = jax.random.PRNGKey(123)
    batch_size, seq_len, num_heads, key_dim, val_dim = 2, 128, 4, 16, 16
    chunk_size = 64

    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    q = jax.random.normal(
        k0, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    ) * (key_dim**-0.5)
    k = jax.random.normal(
        k1, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    ) * (key_dim**-0.5)
    v = jax.random.normal(
        k2, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32
    )
    g = -jax.nn.sigmoid(
        jax.random.normal(
            k3, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
        )
    )
    b = jax.nn.sigmoid(
        jax.random.normal(
            k4, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
        )
    )
    w = jax.nn.sigmoid(
        jax.random.normal(
            k5, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32
        )
    )

    op_instance = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet2()
    (actual_out, actual_s), _ = op_instance._fwd(
        q,
        k,
        v,
        g,
        b,
        w,
        chunk_size=chunk_size,
        scale=1.0,
        output_final_state=True,
    )
    expected_out, expected_s = base.gdn2_reference(
        q,
        k,
        v,
        g,
        b,
        w,
        chunk_size=chunk_size,
        scale=1.0,
        output_final_state=True,
    )

    np.testing.assert_allclose(actual_out, expected_out, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(actual_s, expected_s, rtol=5e-4, atol=5e-4)

  def test_pallas_bwd_matches_autodiff_reference(self):
    key = jax.random.PRNGKey(456)
    batch_size, seq_len, num_heads, key_dim, val_dim = 2, 128, 4, 16, 16
    chunk_size = 64

    k0, k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 9)
    q = jax.random.normal(
        k0, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    ) * (key_dim**-0.5)
    k = jax.random.normal(
        k1, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    )
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    v = jax.random.normal(
        k2, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32
    )
    g = -jax.nn.sigmoid(
        jax.random.normal(
            k3, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
        )
    )
    b = jax.nn.sigmoid(
        jax.random.normal(
            k4, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
        )
    )
    w = jax.nn.sigmoid(
        jax.random.normal(
            k5, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32
        )
    )
    grad_out = jax.random.normal(
        k6, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32
    )
    s0 = jax.random.normal(
        k7, (batch_size, num_heads, key_dim, val_dim), dtype=jnp.float32
    )
    grad_final_state = jax.random.normal(
        k8, (batch_size, num_heads, key_dim, val_dim), dtype=jnp.float32
    )

    op_instance = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet2()
    (actual_out, _), residuals = op_instance._fwd(
        q,
        k,
        v,
        g,
        b,
        w,
        chunk_size=chunk_size,
        scale=1.0,
        initial_state=s0,
        output_final_state=True,
        return_residuals=True,
    )

    vjp_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet2Vjp()
    actual_grads, _ = vjp_op._fwd(
        residuals,
        actual_out,
        (grad_out, grad_final_state),
        q,
        k,
        v,
        g,
        b,
        w,
        chunk_size=chunk_size,
        scale=1.0,
        initial_state=s0,
        output_final_state=True,
    )

    def loss_fn(q_, k_, v_, g_, b_, w_, s0_):
      out_ref, s_final_ref = base.gdn2_reference(
          q_,
          k_,
          v_,
          g_,
          b_,
          w_,
          chunk_size=chunk_size,
          scale=1.0,
          initial_state=s0_,
          output_final_state=True,
      )
      return jnp.sum(out_ref * grad_out) + jnp.sum(
          s_final_ref * grad_final_state
      )

    (
        expected_dq,
        expected_dk,
        expected_dv,
        expected_dg,
        expected_db,
        expected_dw,
        expected_ds0,
    ) = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))(
        q, k, v, g, b, w, s0
    )

    np.testing.assert_allclose(
        actual_grads["q"], expected_dq, rtol=5e-4, atol=5e-4
    )
    np.testing.assert_allclose(
        actual_grads["k"], expected_dk, rtol=5e-4, atol=5e-4
    )
    np.testing.assert_allclose(
        actual_grads["v"], expected_dv, rtol=5e-4, atol=5e-4
    )
    np.testing.assert_allclose(
        actual_grads["g"], expected_dg, rtol=5e-4, atol=5e-4
    )
    np.testing.assert_allclose(
        actual_grads["b"], expected_db, rtol=5e-4, atol=5e-4
    )
    np.testing.assert_allclose(
        actual_grads["w"], expected_dw, rtol=5e-4, atol=5e-4
    )
    np.testing.assert_allclose(
        actual_grads["initial_state"], expected_ds0, rtol=5e-4, atol=5e-4
    )

  def test_invalid_chunk_size_raises(self):
    q = jnp.ones((1, 100, 2, 16), dtype=jnp.float32)
    op_instance = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet2()
    with self.assertRaises(ValueError):
      op_instance.bind(q, q, q, q, q, q, chunk_size=64)
    with self.assertRaises(NotImplementedError):
      q128 = jnp.ones((1, 128, 2, 16), dtype=jnp.float32)
      op_instance._fwd(q128, q128, q128, q128, q128, q128, chunk_size=128)


if __name__ == "__main__":
  absltest.main()
