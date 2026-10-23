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

"""Unit tests for RWKV-7 Pallas TPU operator."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.rwkv7 import pallas_mosaic_tpu
from tokamax._src.ops.experimental.rwkv7 import reference


class PallasTpuRWKV7Test(absltest.TestCase):

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
    alpha = jax.random.normal(
        k3, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    ) * (key_dim**-0.5)
    beta = jax.random.normal(
        k4, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    ) * (key_dim**-0.5)
    gk = -jax.nn.sigmoid(
        jax.random.normal(
            k5, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
        )
    )

    op_instance = pallas_mosaic_tpu.PallasTpuRWKV7()
    (actual_out, actual_s), _ = op_instance._fwd(
        q,
        k,
        v,
        alpha,
        beta,
        gk,
        chunk_size=chunk_size,
        scale=1.0,
        output_final_state=True,
    )
    expected_out, expected_s = reference.rwkv7_reference(
        q,
        k,
        v,
        alpha,
        beta,
        gk,
        chunk_size=chunk_size,
        scale=1.0,
        output_final_state=True,
    )

    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual_s, expected_s, rtol=1e-4, atol=1e-4)

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
    alpha = jax.random.normal(
        k3, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    ) * (key_dim**-0.5)
    beta = jax.random.normal(
        k4, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    ) * (key_dim**-0.5)
    gk = -jax.nn.sigmoid(
        jax.random.normal(
            k5, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
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

    op_instance = pallas_mosaic_tpu.PallasTpuRWKV7()
    (actual_out, _), residuals = op_instance._fwd(
        q,
        k,
        v,
        alpha,
        beta,
        gk,
        chunk_size=chunk_size,
        scale=1.0,
        initial_state=s0,
        output_final_state=True,
        return_residuals=True,
    )

    vjp_op = pallas_mosaic_tpu.PallasTpuRWKV7Vjp()
    actual_grads, _ = vjp_op._fwd(
        residuals,
        actual_out,
        (grad_out, grad_final_state),
        q,
        k,
        v,
        alpha,
        beta,
        gk,
        chunk_size=chunk_size,
        scale=1.0,
        initial_state=s0,
        output_final_state=True,
    )

    def loss_fn(q_, k_, v_, a_, b_, gk_, s0_):
      out_ref, s_final_ref = reference.rwkv7_reference(
          q_,
          k_,
          v_,
          a_,
          b_,
          gk_,
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
        expected_da,
        expected_db,
        expected_dgk,
        expected_ds0,
    ) = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))(
        q, k, v, alpha, beta, gk, s0
    )

    np.testing.assert_allclose(
        actual_grads["q"], expected_dq, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        actual_grads["k"], expected_dk, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        actual_grads["v"], expected_dv, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        actual_grads["alpha"], expected_da, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        actual_grads["beta"], expected_db, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        actual_grads["gk"], expected_dgk, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        actual_grads["initial_state"], expected_ds0, rtol=1e-4, atol=1e-4
    )


if __name__ == "__main__":
  absltest.main()
