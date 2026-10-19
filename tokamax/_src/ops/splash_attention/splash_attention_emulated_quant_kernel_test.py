# Copyright 2026 The Tokamax Authors.
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

"""Tests for Splash Attention emulated quantization kernel."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.splash_attention import base
from tokamax._src.ops.splash_attention import splash_attention_emulated_quant_kernel as splash
from tokamax._src.ops.splash_attention import splash_attention_mask as mask_lib
from tokamax._src.ops.splash_attention import splash_attention_test_utils as test_utils


def _make_multiscale_attention_inputs(
    key: jax.Array,
    num_heads: int,
    seq_len: int,
    head_dim_qk: int,
    head_dim_v: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Generates non-uniform Q, K, V, dO with distinct head and token scales."""
  k1, k2, k3, k4 = random.split(key, 4)
  qk_head_scales = jnp.linspace(0.35, 0.65, num_heads, dtype=jnp.float32)[
      :, None, None
  ]
  qk_token_scales = jnp.linspace(0.6, 1.1, seq_len, dtype=jnp.float32)[
      None, :, None
  ]
  qk_scale = qk_head_scales * qk_token_scales
  v_head_scales = jnp.linspace(0.5, 2.5, num_heads, dtype=jnp.float32)[
      :, None, None
  ]
  v_token_scales = jnp.linspace(0.5, 2.0, seq_len, dtype=jnp.float32)[
      None, :, None
  ]
  v_scale = v_head_scales * v_token_scales
  q = (random.normal(k1, (num_heads, seq_len, head_dim_qk)) * qk_scale).astype(
      jnp.bfloat16
  )
  k = (random.normal(k2, (num_heads, seq_len, head_dim_qk)) * qk_scale).astype(
      jnp.bfloat16
  )
  v = (random.normal(k3, (num_heads, seq_len, head_dim_v)) * v_scale).astype(
      jnp.bfloat16
  )
  do = (random.normal(k4, (num_heads, seq_len, head_dim_v)) * v_scale).astype(
      jnp.bfloat16
  )
  return q, k, v, do


class SplashAttentionEmulatedQuantKernelTest(
    test_utils.SplashAttentionTestCase
):

  def setUp(self):
    super().setUp()
    if not test_utils.test_device_matches(["tpu"]):
      self.skipTest("Test requires TPU.")

  @parameterized.parameters("CAST", "PER_AXIS", "PER_TILE")
  def test_attention_quantization(self, quant_strategy):
    """Verifies that quantized attention runs and is distinct from BF16."""
    seq_len, num_heads, head_dim_qk, head_dim_v = 256, 2, 64, 64
    block = 128

    strategy_enum = getattr(splash.QuantizationStrategy, quant_strategy)
    q, k, v, do = _make_multiscale_attention_inputs(
        random.key(42), num_heads, seq_len, head_dim_qk, head_dim_v
    )

    mask = mask_lib.CausalMask(shape=(seq_len, seq_len))

    def run_attn(strat):
      config = splash.SplashConfig(
          block_q=block,
          block_kv=seq_len,
          block_kv_compute=block,
          block_q_dkv=block,
          block_kv_dkv=seq_len,
          block_kv_dkv_compute=block,
          use_fused_bwd_kernel=True,
          residual_checkpoint_name="context",
          fwd_qk_lhs_quant_strategy=strat,
          fwd_qk_rhs_quant_strategy=strat,
          fwd_pv_lhs_quant_strategy=strat,
          fwd_pv_rhs_quant_strategy=strat,
          bwd_dp_lhs_quant_strategy=strat,
          bwd_dp_rhs_quant_strategy=strat,
          bwd_dv_lhs_quant_strategy=strat,
          bwd_dv_rhs_quant_strategy=strat,
          bwd_dq_lhs_quant_strategy=strat,
          bwd_dq_rhs_quant_strategy=strat,
          bwd_dk_lhs_quant_strategy=strat,
          bwd_dk_rhs_quant_strategy=strat,
      )
      attn = splash.make_splash_mha_single_device(
          mask, config=config, save_residuals=True
      )

      def fwd_bwd(q, k, v, do):
        (o, stats), vjp_fn = jax.vjp(
            functools.partial(attn, max_logit_value=None), q, k, v, None, None
        )
        cotangents = (do, jax.tree.map(jnp.zeros_like, stats))
        dq, dk, dv, _, _ = vjp_fn(cotangents)
        return o, dq, dk, dv

      return jax.jit(fwd_bwd)(q, k, v, do)

    mask_array = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    o_ref, stats_ref = base.attention_reference(
        q, k, v, mask_array, is_mqa=False, save_residuals=True
    )
    dq_ref, dk_ref, dv_ref, _ = base.attention_reference_vjp(
        do,
        q,
        k,
        v,
        mask_array,
        None,
        None,
        o_ref,
        stats_ref["logsumexp"],
        is_mqa=False,
    )
    o_unq, dq_unq, dk_unq, dv_unq = run_attn(
        splash.QuantizationStrategy.UNQUANTIZED
    )
    o_q, dq_q, dk_q, dv_q = run_attn(strategy_enum)

    for name, q_arr, unq_arr in (
        ("O", o_q, o_unq),
        ("dQ", dq_q, dq_unq),
        ("dK", dk_q, dk_unq),
        ("dV", dv_q, dv_unq),
    ):
      self.assertFalse(
          np.allclose(q_arr, unq_arr, atol=1e-6, rtol=1e-6),
          f"Quantized {name} is identically equal to BF16.",
      )

    def compute_nmse(actual, desired):
      actual_f32 = actual.astype(jnp.float32)
      desired_f32 = desired.astype(jnp.float32)
      error_energy = jnp.sum((actual_f32 - desired_f32) ** 2)
      ref_energy = jnp.sum(desired_f32**2)
      return error_energy / (ref_energy + 1e-12)

    for unq_arr, ref_arr in (
        (o_unq, o_ref),
        (dq_unq, dq_ref),
        (dk_unq, dk_ref),
        (dv_unq, dv_ref),
    ):
      self.assertLess(compute_nmse(unq_arr, ref_arr), 1e-3)

    o_nmse = compute_nmse(o_q, o_ref)
    dq_nmse = compute_nmse(dq_q, dq_ref)
    dk_nmse = compute_nmse(dk_q, dk_ref)
    dv_nmse = compute_nmse(dv_q, dv_ref)

    self.assertLess(
        o_nmse,
        0.01,
        f"Forward output O NMSE {o_nmse:.6f} exceeded threshold 0.01",
    )
    self.assertLess(
        dq_nmse, 0.05, f"Backward dQ NMSE {dq_nmse:.6f} exceeded threshold 0.05"
    )
    self.assertLess(
        dk_nmse, 0.05, f"Backward dK NMSE {dk_nmse:.6f} exceeded threshold 0.05"
    )
    self.assertLess(
        dv_nmse, 0.05, f"Backward dV NMSE {dv_nmse:.6f} exceeded threshold 0.05"
    )

  def test_numpy_mask_natural_exp_and_max_logit_const(self):
    """Exercises block_mask_ref, use_base2_exp=False, and max_logit_const."""
    seq_len, num_heads, head_dim_qk, head_dim_v = 256, 2, 64, 64
    block = 128
    q, k, v, _ = _make_multiscale_attention_inputs(
        random.key(77), num_heads, seq_len, head_dim_qk, head_dim_v
    )
    mask_array = np.ones((seq_len, seq_len), dtype=np.bool_)
    mask = mask_lib.NumpyMask(mask_array)
    strat = splash.QuantizationStrategy.PER_AXIS
    config = splash.SplashConfig(
        block_q=block,
        block_kv=block,
        block_kv_compute=block,
        use_base2_exp=False,
        max_logit_const=10.0,
        fwd_qk_lhs_quant_strategy=strat,
        fwd_qk_rhs_quant_strategy=strat,
        fwd_pv_lhs_quant_strategy=strat,
        fwd_pv_rhs_quant_strategy=strat,
    )
    attn = splash.make_splash_mha_single_device(mask, config=config)
    o = jax.jit(attn)(q, k, v)
    o_ref = base.attention_reference(
        q, k, v, jnp.asarray(mask_array), is_mqa=False, save_residuals=False
    )
    nmse = jnp.sum((o.astype(jnp.float32) - o_ref.astype(jnp.float32)) ** 2) / (
        jnp.sum(o_ref.astype(jnp.float32) ** 2) + 1e-12
    )
    self.assertLess(float(nmse), 0.01)

  def test_fp8_error_on_bf16_inputs(self):
    seq_len, num_heads, head_dim = 128, 2, 64
    block = 128
    k1, k2, k3 = random.split(random.key(0), 3)
    q = random.normal(k1, (num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
    k = random.normal(k2, (num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
    v = random.normal(k3, (num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

    mask = mask_lib.CausalMask(shape=(seq_len, seq_len))
    config = splash.SplashConfig(
        block_q=block,
        block_kv=block,
        block_kv_compute=block,
        fwd_qk_use_fp8=True,
    )
    attn = splash.make_splash_mha_single_device(mask, config=config)

    with self.assertRaisesRegex(ValueError, "TRUE fp8 matmul requires"):
      _ = jax.jit(attn)(q, k, v)

  def test_helper_primitives_and_dtypes(self):
    """Verifies dtype helpers and unvmap primitive rules."""
    fp8_arr = jnp.zeros((2, 2), dtype=jnp.float8_e4m3fn)
    bf16_arr = jnp.zeros((2, 2), dtype=jnp.bfloat16)
    f32_arr = jnp.zeros((2, 2), dtype=jnp.float32)
    self.assertTrue(splash._is_fp8(fp8_arr))
    self.assertFalse(splash._is_fp8(bf16_arr))
    self.assertEqual(splash._unquantized_dtype(fp8_arr), jnp.bfloat16)
    self.assertEqual(splash._unquantized_dtype(f32_arr), jnp.float32)

    bool_batch = jnp.array([[True, False], [False, False]])
    np.testing.assert_array_equal(
        splash._unvmap_any_p.bind(bool_batch), bool_batch
    )
    np.testing.assert_array_equal(
        jax.jit(splash._unvmap_any_p.bind)(bool_batch), bool_batch
    )
    np.testing.assert_array_equal(
        jax.vmap(splash._unvmap_any_p.bind, out_axes=None)(bool_batch),
        jnp.array([True, False]),
    )


if __name__ == "__main__":
  absltest.main()
