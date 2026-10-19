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

import os
from absl.testing import absltest
import jax
import jax.export
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.raven import base
from tokamax._src.ops.experimental.raven import pallas_mosaic_tpu
from tokamax._src.ops.experimental.raven import pallas_mosaic_tpu_bwd_kernel
from tokamax._src.ops.experimental.raven import pallas_mosaic_tpu_kernel


class PallasTpuRavenGSATest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() == "gpu":
      self.skipTest("Pallas Mosaic TPU kernel is not supported on GPU.")

  def test_pallas_fwd_matches_reference(self):
    key = jax.random.PRNGKey(123)
    batch_size, seq_len, num_heads, key_dim, num_slots = 2, 128, 4, 16, 8
    chunk_size = 64

    key_q, key_k, key_s, key_g = jax.random.split(key, 4)
    q = jax.random.normal(
        key_q, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    )
    k = jax.random.normal(
        key_k, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    )
    s = jax.random.normal(
        key_s, (batch_size, seq_len, num_heads, num_slots), dtype=jnp.float32
    )
    g = -jax.nn.sigmoid(
        jax.random.normal(
            key_g,
            (batch_size, seq_len, num_heads, num_slots),
            dtype=jnp.float32,
        )
    )

    op_instance = pallas_mosaic_tpu.PallasTpuRavenGSA()
    (actual_out, _), residuals = op_instance._fwd(
        q, k, s, g, chunk_size=chunk_size, return_residuals=False
    )
    self.assertIsNone(residuals)
    expected_out, _ = base.raven_stage1_reference(
        q, k, s, g, chunk_size=chunk_size
    )

    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-4, atol=1e-4)

  def test_pallas_bwd_matches_reference(self):
    key = jax.random.PRNGKey(456)
    batch_size, seq_len, num_heads, key_dim, num_slots = 2, 128, 4, 16, 8
    chunk_size = 64

    key_q, key_k, key_s, key_g, key_do = jax.random.split(key, 5)
    q = jax.random.normal(
        key_q, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    )
    k = jax.random.normal(
        key_k, (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    )
    s = jax.random.normal(
        key_s, (batch_size, seq_len, num_heads, num_slots), dtype=jnp.float32
    )
    g = -jax.nn.sigmoid(
        jax.random.normal(
            key_g,
            (batch_size, seq_len, num_heads, num_slots),
            dtype=jnp.float32,
        )
    )
    do = jax.random.normal(
        key_do, (batch_size, seq_len, num_heads, num_slots), dtype=jnp.float32
    )

    def ref_loss(q_, k_, s_, g_):
      out, _ = base.raven_stage1_reference(q_, k_, s_, g_, chunk_size=chunk_size)
      return jnp.sum(out * do)

    dq_ref, dk_ref, ds_ref, dg_ref = jax.grad(ref_loss, argnums=(0, 1, 2, 3))(
        q, k, s, g
    )
    dq, dk, ds, dg, _ = pallas_mosaic_tpu_bwd_kernel.raven_pallas_stage1_bwd(
        q, k, s, g, do, chunk_size=chunk_size
    )

    np.testing.assert_allclose(dq, dq_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dk, dk_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ds, ds_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dg, dg_ref, rtol=1e-4, atol=1e-4)

  def test_pallas_stage2_fwd_and_bwd_matches_reference(self):
    key = jax.random.PRNGKey(789)
    batch_size, seq_len, num_heads, num_slots, val_dim = 2, 128, 4, 8, 16
    chunk_size = 64

    key_qv, key_s, key_v, key_g, key_s0, key_do, key_dsf = jax.random.split(
        key, 7
    )
    qv = jax.nn.softmax(
        jax.random.normal(
            key_qv,
            (batch_size, seq_len, num_heads, num_slots),
            dtype=jnp.float32,
        ),
        axis=-1,
    )
    s = jax.nn.sigmoid(
        jax.random.normal(
            key_s,
            (batch_size, seq_len, num_heads, num_slots),
            dtype=jnp.float32,
        )
    )
    v = jax.random.normal(
        key_v, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32
    )
    g = (
        -jnp.abs(
            jax.random.normal(
                key_g,
                (batch_size, seq_len, num_heads, num_slots),
                dtype=jnp.float32,
            )
        )
        * 0.1
    )
    s0 = jax.random.normal(
        key_s0, (batch_size, num_heads, num_slots, val_dim), dtype=jnp.float32
    )
    do = jax.random.normal(
        key_do, (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32
    )
    d_s2_final = jax.random.normal(
        key_dsf, (batch_size, num_heads, num_slots, val_dim), dtype=jnp.float32
    )

    expected_out, expected_s2 = base.raven_stage2_reference(
        qv,
        s,
        v,
        g,
        chunk_size=chunk_size,
        initial_state=s0,
        output_final_state=True,
    )
    (actual_out, actual_s2), residuals = (
        pallas_mosaic_tpu_kernel.raven_pallas_stage2_fwd(
            qv,
            s,
            v,
            g,
            chunk_size=chunk_size,
            initial_state=s0,
            output_final_state=True,
            return_residuals=True,
        )
    )
    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual_s2, expected_s2, rtol=1e-4, atol=1e-4)

    def ref_loss(qv_, s_, v_, g_, s0_):
      out, s_final = base.raven_stage2_reference(
          qv_,
          s_,
          v_,
          g_,
          chunk_size=chunk_size,
          initial_state=s0_,
          output_final_state=True,
      )
      return jnp.sum(out * do) + jnp.sum(s_final * d_s2_final)

    dqv_ref, ds_ref, dv_ref, dg_ref, ds0_ref = jax.grad(
        ref_loss, argnums=(0, 1, 2, 3, 4)
    )(qv, s, v, g, s0)
    dqv, ds, dv, dg, ds0 = pallas_mosaic_tpu_bwd_kernel.raven_pallas_stage2_bwd(
        qv,
        s,
        v,
        g,
        do,
        grad_final_state=d_s2_final,
        chunk_size=chunk_size,
        initial_state=s0,
        s2_in_c=residuals,
    )
    np.testing.assert_allclose(dqv, dqv_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ds, ds_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dv, dv_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dg, dg_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ds0, ds0_ref, rtol=1e-4, atol=1e-4)

  def test_full_two_stage_op_fwd_and_vjp_matches_reference(self):
    key = jax.random.PRNGKey(999)
    batch_size, seq_len, num_heads, key_dim, num_slots, val_dim = (
        2,
        128,
        4,
        16,
        8,
        16,
    )
    chunk_size = 64
    scale = key_dim**-0.5

    keys = jax.random.split(key, 10)
    q = jax.random.normal(
        keys[0], (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    )
    k = jax.random.normal(
        keys[1], (batch_size, seq_len, num_heads, key_dim), dtype=jnp.float32
    )
    s = jax.nn.sigmoid(
        jax.random.normal(
            keys[2],
            (batch_size, seq_len, num_heads, num_slots),
            dtype=jnp.float32,
        )
    )
    g = (
        -jnp.abs(
            jax.random.normal(
                keys[3],
                (batch_size, seq_len, num_heads, num_slots),
                dtype=jnp.float32,
            )
        )
        * 0.1
    )
    v = jax.random.normal(
        keys[4], (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32
    )
    s1_0 = jax.random.normal(
        keys[5], (batch_size, num_heads, key_dim, num_slots), dtype=jnp.float32
    )
    s2_0 = jax.random.normal(
        keys[6], (batch_size, num_heads, num_slots, val_dim), dtype=jnp.float32
    )
    do = jax.random.normal(
        keys[7], (batch_size, seq_len, num_heads, val_dim), dtype=jnp.float32
    )
    ds1_final = jax.random.normal(
        keys[8], (batch_size, num_heads, key_dim, num_slots), dtype=jnp.float32
    )
    ds2_final = jax.random.normal(
        keys[9], (batch_size, num_heads, num_slots, val_dim), dtype=jnp.float32
    )

    op_pallas = pallas_mosaic_tpu.PallasTpuRavenGSA(bypass_device_check=True)
    (actual_out, (actual_s1, actual_s2)), residuals = op_pallas._fwd(
        q,
        k,
        s,
        g,
        v,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=(s1_0, s2_0),
        output_final_state=True,
        return_residuals=True,
    )
    expected_out, (expected_s1, expected_s2) = base.raven_gsa_reference(
        q,
        k,
        s,
        g,
        v,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=(s1_0, s2_0),
        output_final_state=True,
    )
    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual_s1, expected_s1, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual_s2, expected_s2, rtol=1e-4, atol=1e-4)

    vjp_op = pallas_mosaic_tpu.PallasTpuRavenGSAVjp()
    grads, _ = vjp_op._fwd(
        residuals,
        (actual_out, (actual_s1, actual_s2)),
        (do, (ds1_final, ds2_final)),
        q,
        k,
        s,
        g,
        v,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=(s1_0, s2_0),
        output_final_state=True,
    )

    def ref_loss(q_, k_, s_, g_, v_, s1_0_, s2_0_):
      out, (s1_f, s2_f) = base.raven_gsa_reference(
          q_,
          k_,
          s_,
          g_,
          v_,
          chunk_size=chunk_size,
          scale=scale,
          initial_state=(s1_0_, s2_0_),
          output_final_state=True,
      )
      return (
          jnp.sum(out * do)
          + jnp.sum(s1_f * ds1_final)
          + jnp.sum(s2_f * ds2_final)
      )

    dq_ref, dk_ref, ds_ref, dg_ref, dv_ref, ds1_0_ref, ds2_0_ref = jax.grad(
        ref_loss, argnums=(0, 1, 2, 3, 4, 5, 6)
    )(q, k, s, g, v, s1_0, s2_0)

    np.testing.assert_allclose(grads["q"], dq_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(grads["k"], dk_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(grads["s"], ds_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(grads["g"], dg_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(grads["v"], dv_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        grads["initial_state"][0], ds1_0_ref, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        grads["initial_state"][1], ds2_0_ref, rtol=1e-4, atol=1e-4
    )

  def test_heuristics_config_and_validation(self):
    op_pallas = pallas_mosaic_tpu.PallasTpuRavenGSA()
    q = jnp.ones((1, 64, 2, 16), dtype=jnp.float32)
    k = jnp.ones((1, 64, 2, 16), dtype=jnp.float32)
    s = jnp.ones((1, 64, 2, 8), dtype=jnp.float32)
    g = -0.1 * jnp.ones((1, 64, 2, 8), dtype=jnp.float32)
    ba = op_pallas.bind(q, k, s, g, chunk_size=32)
    cfg = op_pallas._get_heuristics_config(ba)
    self.assertEqual(cfg.chunk_size, 32)

    cpu_dev = jax.devices("cpu")[0]
    self.assertFalse(op_pallas.supported_on(cpu_dev))

    q_bad = jnp.ones((1, 100, 2, 16), dtype=jnp.float32)
    with self.assertRaises(ValueError):
      op_pallas._fwd(q_bad, k, s, g, chunk_size=64)

  def test_tpu_export_fwd_and_bwd(self):
    batch_size, seq_len, num_heads, key_dim, num_slots, val_dim = (
        2,
        128,
        4,
        128,
        64,
        128,
    )
    q_spec = jax.ShapeDtypeStruct(
        (batch_size, seq_len, num_heads, key_dim), jnp.bfloat16
    )
    k_spec = jax.ShapeDtypeStruct(
        (batch_size, seq_len, num_heads, key_dim), jnp.bfloat16
    )
    s_spec = jax.ShapeDtypeStruct(
        (batch_size, seq_len, num_heads, num_slots), jnp.bfloat16
    )
    g_spec = jax.ShapeDtypeStruct(
        (batch_size, seq_len, num_heads, num_slots), jnp.float32
    )
    v_spec = jax.ShapeDtypeStruct(
        (batch_size, seq_len, num_heads, val_dim), jnp.bfloat16
    )
    do_spec = jax.ShapeDtypeStruct(
        (batch_size, seq_len, num_heads, val_dim), jnp.bfloat16
    )

    prev_env = os.environ.get("PALLAS_INTERPRET")
    os.environ["PALLAS_INTERPRET"] = "0"
    try:
      exp_fwd = jax.export.export(
          jax.jit(
              lambda q, k, s, g, v: pallas_mosaic_tpu_kernel.raven_pallas_fwd(
                  q, k, s, g, v, chunk_size=64
              )
          ),
          platforms=("tpu",),
      )(q_spec, k_spec, s_spec, g_spec, v_spec)
      self.assertIn("tpu_custom_call", exp_fwd.mlir_module())

      exp_bwd = jax.export.export(
          jax.jit(
              lambda q, k, s, g, v, do: (
                  pallas_mosaic_tpu_bwd_kernel.raven_pallas_bwd(
                      q, k, s, g, v, do, chunk_size=64
                  )[:5]
              )
          ),
          platforms=("tpu",),
      )(q_spec, k_spec, s_spec, g_spec, v_spec, do_spec)
      self.assertIn("tpu_custom_call", exp_bwd.mlir_module())
    finally:
      if prev_env is None:
        os.environ.pop("PALLAS_INTERPRET", None)
      else:
        os.environ["PALLAS_INTERPRET"] = prev_env


if __name__ == "__main__":
  absltest.main()
