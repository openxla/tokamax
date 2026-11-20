# Copyright 2026 Ant Group. All Rights Reserved.
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
"""Tests for the Pallas/Mosaic TPU KDA adapter and public support checks."""

import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import hlo_utils
from tokamax._src import jaxtyping
from tokamax._src.ops.kda import api
from tokamax._src.ops.kda import common
from tokamax._src.ops.kda import pallas_mosaic_tpu
from tokamax._src.ops.kda import utils
from tokamax._src.ops.kda.cp_utils import ContextParallelMetadata


def _call_attention(implementation, q, v, **kwargs):
  kwargs.setdefault("use_gate_in_kernel", True)
  kwargs.setdefault("a_log", jnp.zeros((q.shape[0],), dtype=jnp.float32))
  kwargs.setdefault("lower_bound", -5.0)

  def call():
    return api.kimi_delta_attention(
        q,
        jnp.ones_like(q),
        v,
        jnp.zeros_like(q),
        jnp.ones(q.shape[:-1], dtype=q.dtype),
        scale=1.0,
        implementation=implementation,
        **kwargs,
    )

  uses_mosaic = implementation == "mosaic" or (
      not isinstance(implementation, str) and "mosaic" in implementation
  )
  if not uses_mosaic:
    return call()
  # Exercise Mosaic's public API validation on every test platform without
  # tracing a kernel: every case below is rejected at the start of `_fwd`.
  with mock.patch.object(
      pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention,
      "supported_on",
      return_value=True,
  ):
    return call()


class PallasMosaicTpuKimiDeltaAttentionTest(parameterized.TestCase):

  def test_tpu_limits_use_pallas_hardware_info(self):
    tpu_info = types.SimpleNamespace(
        vmem_capacity_bytes=64 * 1024 * 1024,
        num_sublanes=16,
        num_lanes=128,
    )

    with mock.patch.object(
        common.pltpu, "get_tpu_info", return_value=tpu_info
    ) as get_tpu_info:
      limits = common.get_tpu_limits()

    get_tpu_info.assert_called_once_with()
    self.assertEqual(
        limits.vmem_limit_bytes, int(tpu_info.vmem_capacity_bytes * 0.9)
    )
    self.assertEqual(limits.block_align_minor, tpu_info.num_sublanes)
    self.assertEqual(limits.block_align_major, tpu_info.num_lanes)

  def test_default_execution_config(self):
    attention = pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention()
    vjp = pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttentionVjp()
    expected = pallas_mosaic_tpu.Config(chunk_size=64)

    self.assertEqual(attention._get_heuristics_config(None), expected)
    self.assertEqual(attention._get_autotuning_configs(None), {expected})
    self.assertEqual(vjp._get_heuristics_config(None), expected)
    self.assertEqual(vjp._get_autotuning_configs(None), {expected})
    self.assertIsNone(expected.safe_gate)
    self.assertFalse(expected.rematerialize_for_backward)

  @parameterized.named_parameters(
      ("preactivated_gate", False, None, True),
      ("softplus_gate", True, None, False),
      ("bounded_gate", True, -5.0, True),
  )
  def test_safe_gate_is_selected_internally(
      self, use_gate_in_kernel, lower_bound, expected
  ):
    config = pallas_mosaic_tpu.Config()

    self.assertEqual(
        pallas_mosaic_tpu._resolve_safe_gate(
            config,
            use_gate_in_kernel=use_gate_in_kernel,
            lower_bound=lower_bound,
        ),
        expected,
    )

  def test_safe_gate_config_override(self):
    config = pallas_mosaic_tpu.Config(safe_gate=False)

    self.assertFalse(
        pallas_mosaic_tpu._resolve_safe_gate(
            config,
            use_gate_in_kernel=False,
            lower_bound=None,
        )
    )

  @parameterized.named_parameters(
      ("missing_lower_bound", True, None, False, "lower_bound.*set"),
      ("unsafe_gate", True, -5.0, False, "safe_gate=True"),
  )
  def test_rejects_unsupported_gate_config(
      self, use_gate_in_kernel, lower_bound, safe_gate, expected_error
  ):
    with self.assertRaisesRegex(NotImplementedError, expected_error):
      pallas_mosaic_tpu._check_gate_support(  # pylint: disable=protected-access
          use_gate_in_kernel=use_gate_in_kernel,
          lower_bound=lower_bound,
          safe_gate=safe_gate,
      )

  @parameterized.named_parameters(
      ("default_safe_gate", None, True),
      ("explicit_unsafe_gate", -5.0, False),
  )
  def test_allows_preactivated_gate_config(self, lower_bound, safe_gate):
    pallas_mosaic_tpu._check_gate_support(  # pylint: disable=protected-access
        use_gate_in_kernel=False,
        lower_bound=lower_bound,
        safe_gate=safe_gate,
    )

  def test_rejects_missing_lower_bound_before_kernel(self):
    q = jnp.ones((1, 1, 64, 1), dtype=jnp.float32)
    v = jnp.ones_like(q)

    with self.assertRaisesRegex(NotImplementedError, "lower_bound.*set"):
      _call_attention(
          "mosaic",
          q,
          v,
          use_gate_in_kernel=True,
          lower_bound=None,
      )

  def test_unsupported_gate_config_falls_back_to_xla(self):
    q = jnp.ones((1, 1, 64, 1), dtype=jnp.float32)
    v = jnp.ones_like(q)
    expected = _call_attention(
        "xla", q, v, use_gate_in_kernel=True, lower_bound=None
    )
    actual = _call_attention(
        ("mosaic", "xla"),
        q,
        v,
        use_gate_in_kernel=True,
        lower_bound=None,
    )

    self.assertTrue(jnp.array_equal(actual[0], expected[0]))
    self.assertEqual(actual[1], expected[1])

  def test_large_key_dimension_is_mosaic_specific(self):
    q = jnp.ones((1, 1, 1, 257), dtype=jnp.float32)
    v = jnp.ones((1, 1, 1, 1), dtype=jnp.float32)

    output, final_state = _call_attention("xla", q, v)
    self.assertEqual(output.shape, v.shape)
    self.assertIsNone(final_state)

    with self.assertRaisesRegex(NotImplementedError, "up to 256"):
      _call_attention("mosaic", q, v)

  @parameterized.parameters((0, 1), (1, 0))
  def test_rejects_empty_kv_dimension_before_kernel(self, key_dim, value_dim):
    q = jnp.ones((1, 1, 64, key_dim), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, value_dim), dtype=jnp.float32)

    with self.assertRaisesRegex(NotImplementedError, "positive key and value"):
      _call_attention("mosaic", q, v)

  @parameterized.parameters((0, 1, 64), (1, 0, 64), (1, 1, 0))
  def test_rejects_empty_grid_dimension_before_kernel(
      self, heads, batch, seq_len
  ):
    q = jnp.ones((heads, batch, seq_len, 1), dtype=jnp.float32)
    v = jnp.ones((heads, batch, seq_len, 1), dtype=jnp.float32)

    with self.assertRaisesRegex(NotImplementedError, "positive head, batch"):
      _call_attention("mosaic", q, v)

  def test_rejects_multiple_fixed_states_before_kernel(self):
    q = jnp.ones((1, 1, 64, 1), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, 1), dtype=jnp.float32)
    initial_state = jnp.zeros((1, 2, 1, 1, 1), dtype=jnp.float32)

    with self.assertRaisesRegex(NotImplementedError, "exactly one"):
      _call_attention("mosaic", q, v, initial_state=initial_state)

  @parameterized.parameters((64, 128), (128, 64))
  def test_rejects_unaligned_cp_dimensions_before_kernel(
      self, key_dim, value_dim
  ):
    q = jnp.ones((1, 1, 64, key_dim), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, value_dim), dtype=jnp.float32)
    segment_ids = jnp.ones((1, 64), dtype=jnp.int32)
    context_parallel_metadata = ContextParallelMetadata(
        mesh=types.SimpleNamespace(shape={"context": 2})
    )

    with self.assertRaisesRegex(NotImplementedError, "multiples of 128"):
      _call_attention(
          "mosaic",
          q,
          v,
          segment_ids=segment_ids,
          context_parallel_metadata=context_parallel_metadata,
          max_num_segments=1,
      )

  def test_rejects_cp_contract_gaps_before_kernel(self):
    q = jnp.ones((1, 1, 64, 128), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, 128), dtype=jnp.float32)
    segment_ids = jnp.ones((1, 64), dtype=jnp.int32)
    initial_state = jnp.zeros((1, 1, 1, 128, 128), dtype=jnp.float32)
    context_parallel_metadata = ContextParallelMetadata(
        mesh=types.SimpleNamespace(shape={"context": 2})
    )
    cases = (
        (
            "initial_state",
            dict(
                initial_state=initial_state,
                segment_ids=segment_ids,
                max_num_segments=1,
            ),
        ),
        (
            "output_final_state",
            dict(
                output_final_state=True,
                segment_ids=segment_ids,
                max_num_segments=1,
            ),
        ),
        ("segment_ids", dict(max_num_segments=1)),
    )

    for error_fragment, overrides in cases:
      with self.subTest(error_fragment=error_fragment):
        with self.assertRaisesRegex(NotImplementedError, error_fragment):
          _call_attention(
              "mosaic",
              q,
              v,
              context_parallel_metadata=context_parallel_metadata,
              **overrides,
          )


def _count_pallas_calls(jaxpr_or_closed) -> int:
  jaxpr = getattr(jaxpr_or_closed, "jaxpr", jaxpr_or_closed)
  count = 0
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "pallas_call":
      count += 1
    for param_val in eqn.params.values():
      if hasattr(param_val, "jaxpr") or hasattr(param_val, "eqns"):
        count += _count_pallas_calls(param_val)
  return count


class GatedDeltaNetMosaicTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() == "gpu":
      self.skipTest("Pallas Mosaic TPU kernel is not supported on GPU.")
    if jax.default_backend() == "cpu":
      self.enterContext(
          mock.patch.object(
              pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet,
              "supported_on",
              return_value=True,
          )
      )

  @parameterized.parameters((True,), (False,))
  def test_gdn_mosaic_fwd_bwd_matches_xla(self, fused_qkv: bool):
    b, t, h_qk, h_v, k_dim, v_dim = 1, 128, 2, 8, 128, 128
    c_qkv = 2 * h_qk * k_dim + h_v * v_dim
    keys = jax.random.split(jax.random.PRNGKey(42), 9)
    qkv = jax.random.normal(keys[0], (b, t, c_qkv), jnp.float32) * 0.5
    conv_w = jax.random.normal(keys[1], (4, c_qkv), jnp.float32) * 0.2
    conv_b = jax.random.normal(keys[2], (c_qkv,), jnp.float32) * 0.1
    gate = jax.random.normal(keys[3], (b, t, h_v), jnp.float32) * 0.5
    beta = jax.random.normal(keys[4], (b, t, h_v), jnp.float32)
    a_log = jnp.log(jax.random.uniform(keys[5], (h_v,), minval=0.1, maxval=1.0))
    dt_bias = jax.random.normal(keys[6], (h_v,), jnp.float32) * 0.1
    h0 = jax.random.normal(keys[7], (b, h_v, k_dim, v_dim), jnp.float32) * 0.1
    do = jax.random.normal(keys[8], (b, t, h_v, v_dim), jnp.float32)

    def run(impl: str):
      if fused_qkv:
        def fn(qkv_, gate_, beta_, conv_w_, conv_b_, a_log_, dt_bias_, h0_):
          out, ht = api.gated_delta_net(
              qkv=qkv_,
              gate=gate_,
              beta=beta_,
              num_k_heads=h_qk,
              num_v_heads=h_v,
              head_k_dim=k_dim,
              head_v_dim=v_dim,
              conv_weight=conv_w_,
              conv_bias=conv_b_,
              a_log=a_log_,
              delta_time_bias=dt_bias_,
              initial_state=h0_,
              output_final_state=True,
              use_qk_l2norm=True,
              use_gate_in_kernel=True,
              use_beta_sigmoid_in_kernel=True,
              chunk_size=64,
              implementation=impl,
          )
          return jnp.sum(out * do) + 0.1 * jnp.sum(ht)

        return jax.value_and_grad(fn, argnums=tuple(range(8)))(
            qkv, gate, beta, conv_w, conv_b, a_log, dt_bias, h0
        )
      else:
        q_s = qkv[:, :, : h_qk * k_dim].reshape(b, t, h_qk, k_dim)
        k_s = qkv[:, :, h_qk * k_dim : 2 * h_qk * k_dim].reshape(
            b, t, h_qk, k_dim
        )
        v_s = qkv[:, :, 2 * h_qk * k_dim :].reshape(b, t, h_v, v_dim)

        def fn_split(q_, k_, v_, gate_, beta_, a_log_, dt_bias_, h0_):
          out, ht = api.gated_delta_net(
              q_,
              k_,
              v_,
              gate_,
              beta_,
              a_log=a_log_,
              delta_time_bias=dt_bias_,
              initial_state=h0_,
              output_final_state=True,
              use_qk_l2norm=True,
              use_gate_in_kernel=True,
              use_beta_sigmoid_in_kernel=True,
              chunk_size=64,
              implementation=impl,
          )
          return jnp.sum(out * do) + 0.1 * jnp.sum(ht)

        return jax.value_and_grad(fn_split, argnums=tuple(range(8)))(
            q_s, k_s, v_s, gate, beta, a_log, dt_bias, h0
        )

    loss_ref, grads_ref = run("xla")
    loss_mos, grads_mos = run("mosaic")
    self.assertAlmostEqual(float(loss_mos), float(loss_ref), places=2)
    for g_mos, g_ref in zip(grads_mos, grads_ref):
      max_diff = float(jnp.max(jnp.abs(g_mos - g_ref)))
      self.assertLess(max_diff, 2e-3)

  def test_gdn_mosaic_zero_norm_backward_guard(self):
    b, t, h_qk, h_v, d = 1, 64, 2, 4, 128
    keys = jax.random.split(jax.random.PRNGKey(99), 5)
    q_nz = jax.random.normal(keys[0], (b, t, h_qk, d), jnp.float32) * 0.5
    k_nz = jax.random.normal(keys[1], (b, t, h_qk, d), jnp.float32) * 0.5
    # Zero out disjoint token halves so upstream cotangents are non-zero at
    # the zero-norm rows (t < 32 for q, t >= 32 for k).
    q = jnp.concatenate([jnp.zeros_like(q_nz[:, :32]), q_nz[:, 32:]], axis=1)
    k = jnp.concatenate([k_nz[:, :32], jnp.zeros_like(k_nz[:, 32:])], axis=1)
    v = jax.random.normal(keys[2], (b, t, h_v, d), jnp.float32)
    h0 = jax.random.normal(keys[3], (b, h_v, d, d), jnp.float32) * 0.2
    do = jax.random.normal(keys[4], (b, t, h_v, d), jnp.float32)
    gate = jnp.full((b, t, h_v), -0.1, jnp.float32)
    beta = jnp.full((b, t, h_v), 0.5, jnp.float32)

    def loss_fn(q_in, k_in):
      out, _ = api.gated_delta_net(
          q_in,
          k_in,
          v,
          gate,
          beta,
          initial_state=h0,
          use_qk_l2norm=True,
          chunk_size=64,
          implementation="mosaic",
      )
      return jnp.sum(out * do)

    dq, dk = jax.grad(loss_fn, argnums=(0, 1))(q, k)
    self.assertEqual(float(jnp.max(jnp.abs(dq[:, :32]))), 0.0)
    self.assertEqual(float(jnp.max(jnp.abs(dk[:, 32:]))), 0.0)
    self.assertTrue(bool(jnp.all(jnp.isfinite(dq))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(dk))))
    self.assertGreater(float(jnp.max(jnp.abs(dq[:, 32:]))), 0.0)
    self.assertGreater(float(jnp.max(jnp.abs(dk[:, :32]))), 0.0)

  def test_gdn_rematerialize_for_backward_matches_xla(self):
    b, t, h_qk, h_v, d = 1, 128, 2, 4, 128
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(jax.random.PRNGKey(107), 7)
    qkv = jax.random.normal(keys[0], (b, t, c_qkv), jnp.float32) * 0.4
    conv_w = jax.random.normal(keys[1], (4, c_qkv), jnp.float32) * 0.15
    conv_b = jax.random.normal(keys[2], (c_qkv,), jnp.float32) * 0.05
    gate = jax.random.normal(keys[3], (b, t, h_v), jnp.float32) * 0.4
    beta = jax.random.normal(keys[4], (b, t, h_v), jnp.float32)
    a_log = jnp.log(jax.random.uniform(keys[5], (h_v,), minval=0.1, maxval=0.8))
    do = jax.random.normal(keys[6], (b, t, h_v, d), jnp.float32)

    remat_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet(
        config=pallas_mosaic_tpu.Config(
            chunk_size=64,
            rematerialize_for_backward=True,
            residual_checkpoint_name="gdn_remat_states",
        )
    )
    policy = jax.checkpoint_policies.save_only_these_names("gdn_remat_states")

    def loss_mos(qkv_, gate_, beta_, conv_w_, conv_b_, a_log_):
      def step(qkv_in, gate_in, beta_in, conv_w_in, conv_b_in, a_log_in):
        out, _ = remat_op(
            qkv=qkv_in,
            gate=gate_in,
            beta=beta_in,
            num_k_heads=h_qk,
            num_v_heads=h_v,
            head_k_dim=d,
            head_v_dim=d,
            conv_weight=conv_w_in,
            conv_bias=conv_b_in,
            a_log=a_log_in,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
        )
        return out

      ckpt_step = jax.checkpoint(step, policy=policy)
      return jnp.sum(
          ckpt_step(qkv_, gate_, beta_, conv_w_, conv_b_, a_log_) * do
      )

    def loss_xla(qkv_, gate_, beta_, conv_w_, conv_b_, a_log_):
      out, _ = api.gated_delta_net(
          qkv=qkv_,
          gate=gate_,
          beta=beta_,
          num_k_heads=h_qk,
          num_v_heads=h_v,
          head_k_dim=d,
          head_v_dim=d,
          conv_weight=conv_w_,
          conv_bias=conv_b_,
          a_log=a_log_,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          implementation="xla",
      )
      return jnp.sum(out * do)

    val_mos, grads_mos = jax.value_and_grad(loss_mos, argnums=tuple(range(6)))(
        qkv, gate, beta, conv_w, conv_b, a_log
    )
    val_xla, grads_xla = jax.value_and_grad(loss_xla, argnums=tuple(range(6)))(
        qkv, gate, beta, conv_w, conv_b, a_log
    )
    self.assertAlmostEqual(float(val_mos), float(val_xla), places=2)
    for g_mos, g_xla in zip(grads_mos, grads_xla):
      self.assertLess(float(jnp.max(jnp.abs(g_mos - g_xla))), 2e-3)

  def test_kda_residual_checkpoint_name_prevents_fwd_reexecution(self):
    h, b, t, d = 2, 1, 64, 128
    keys = jax.random.split(jax.random.PRNGKey(27), 6)
    q = jax.random.normal(keys[0], (h, b, t, d), jnp.bfloat16) * 0.4
    k = jax.random.normal(keys[1], (h, b, t, d), jnp.bfloat16) * 0.4
    v = jax.random.normal(keys[2], (h, b, t, d), jnp.bfloat16) * 0.4
    g = jax.random.normal(keys[3], (h, b, t, d), jnp.bfloat16) * 0.4
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (h, b, t), jnp.bfloat16))
    a_log = jnp.zeros((h,), jnp.float32)

    policy = jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=("kda_states",),
        names_which_can_be_offloaded=(),
        offload_src="device",
        offload_dst="pinned_host",
    )

    with mock.patch.object(
        pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention,
        "supported_on",
        return_value=True,
    ):
      def step(q_, k_, v_, g_, beta_, a_log_):
        out, _ = api.kimi_delta_attention(
            q_,
            k_,
            v_,
            g_,
            beta_,
            a_log=a_log_,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            lower_bound=-5.0,
            chunk_size=64,
            residual_checkpoint_name="kda_states",
            implementation="mosaic",
        )
        return out

      ckpt_step = jax.checkpoint(step, policy=policy)

      def loss_ckpt(q_, k_, v_, g_, beta_, a_log_):
        return jnp.sum(ckpt_step(q_, k_, v_, g_, beta_, a_log_).astype(jnp.float32))

      def loss_plain(q_, k_, v_, g_, beta_, a_log_):
        return jnp.sum(step(q_, k_, v_, g_, beta_, a_log_).astype(jnp.float32))

      def loss_none(q_, k_, v_, g_, beta_, a_log_):
        return jnp.sum(
            jax.checkpoint(
                step, policy=jax.checkpoint_policies.nothing_saveable
            )(q_, k_, v_, g_, beta_, a_log_).astype(jnp.float32)
        )

      jaxpr_ckpt = jax.make_jaxpr(jax.value_and_grad(loss_ckpt))(
          q, k, v, g, beta, a_log
      )
      jaxpr_plain = jax.make_jaxpr(jax.value_and_grad(loss_plain))(
          q, k, v, g, beta, a_log
      )
      jaxpr_none = jax.make_jaxpr(jax.value_and_grad(loss_none))(
          q, k, v, g, beta, a_log
      )
      self.assertEqual(
          _count_pallas_calls(jaxpr_ckpt), _count_pallas_calls(jaxpr_plain)
      )
      self.assertLess(
          _count_pallas_calls(jaxpr_ckpt), _count_pallas_calls(jaxpr_none)
      )

      val_ckpt, grad_ckpt = jax.value_and_grad(loss_ckpt)(
          q, k, v, g, beta, a_log
      )
      val_plain, grad_plain = jax.value_and_grad(loss_plain)(
          q, k, v, g, beta, a_log
      )
    self.assertAlmostEqual(float(val_ckpt), float(val_plain), places=3)
    self.assertLess(float(jnp.max(jnp.abs(grad_ckpt - grad_plain))), 1e-3)

  def test_residual_checkpoint_name_prevents_fwd_reexecution(self):
    b, t, h_qk, h_v, d = 1, 64, 2, 4, 128
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(jax.random.PRNGKey(7), 5)
    qkv = jax.random.normal(keys[0], (b, t, c_qkv), jnp.bfloat16)
    conv_w = jax.random.normal(keys[1], (4, c_qkv), jnp.bfloat16) * 0.1
    gate = jax.random.normal(keys[2], (b, t, h_v), jnp.bfloat16)
    beta = jax.random.normal(keys[3], (b, t, h_v), jnp.bfloat16)
    a_log = jnp.zeros((h_v,), jnp.float32)

    policy = jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=("gdn_states",),
        names_which_can_be_offloaded=(),
        offload_src="device",
        offload_dst="pinned_host",
    )

    def step(qkv_, gate_, beta_, conv_w_, a_log_):
      out, _ = api.gated_delta_net(
          qkv=qkv_,
          gate=gate_,
          beta=beta_,
          num_k_heads=h_qk,
          num_v_heads=h_v,
          head_k_dim=d,
          head_v_dim=d,
          conv_weight=conv_w_,
          a_log=a_log_,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          chunk_size=64,
          residual_checkpoint_name="gdn_states",
          implementation="mosaic",
      )
      return out

    ckpt_step = jax.checkpoint(step, policy=policy)

    def loss_ckpt(qkv_, gate_, beta_, conv_w_, a_log_):
      return jnp.sum(ckpt_step(qkv_, gate_, beta_, conv_w_, a_log_).astype(jnp.float32))

    def loss_plain(qkv_, gate_, beta_, conv_w_, a_log_):
      return jnp.sum(step(qkv_, gate_, beta_, conv_w_, a_log_).astype(jnp.float32))

    jaxpr_ckpt = jax.make_jaxpr(jax.value_and_grad(loss_ckpt))(
        qkv, gate, beta, conv_w, a_log
    )
    jaxpr_plain = jax.make_jaxpr(jax.value_and_grad(loss_plain))(
        qkv, gate, beta, conv_w, a_log
    )
    # 2 forward Pallas calls (intra + h_o) + 1 backward Pallas call = 3 total.
    self.assertEqual(_count_pallas_calls(jaxpr_plain), 3)
    self.assertEqual(_count_pallas_calls(jaxpr_ckpt), 3)

    val_ckpt, grad_ckpt = jax.value_and_grad(loss_ckpt)(
        qkv, gate, beta, conv_w, a_log
    )
    val_plain, grad_plain = jax.value_and_grad(loss_plain)(
        qkv, gate, beta, conv_w, a_log
    )
    self.assertAlmostEqual(float(val_ckpt), float(val_plain), places=3)
    self.assertLess(float(jnp.max(jnp.abs(grad_ckpt - grad_plain))), 1e-3)

  def test_residual_checkpoint_name_supports_numpy_args_and_bound_args(self):
    b, t, h_qk, h_v, d = 1, 64, 2, 4, 128
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(jax.random.PRNGKey(17), 4)
    qkv = jax.random.normal(keys[0], (b, t, c_qkv), jnp.float32) * 0.3
    gate = jax.random.normal(keys[1], (b, t, h_v), jnp.float32) * 0.3
    beta = jax.random.normal(keys[2], (b, t, h_v), jnp.float32)
    a_log_np = np.zeros((h_v,), dtype=np.float32)

    def fn(qkv_, gate_, beta_):
      out, _ = api.gated_delta_net(
          qkv=qkv_,
          gate=gate_,
          beta=beta_,
          num_k_heads=h_qk,
          num_v_heads=h_v,
          head_k_dim=d,
          head_v_dim=d,
          a_log=a_log_np,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          chunk_size=64,
          residual_checkpoint_name="gdn_np_ckpt",
          implementation="mosaic",
      )
      return jnp.sum(out)

    with jaxtyping.disable_jaxtyping():
      val, grads = jax.value_and_grad(fn, argnums=(0, 1, 2))(qkv, gate, beta)
      self.assertTrue(bool(jnp.isfinite(val)))
      self.assertLen(grads, 3)

      bound_args = hlo_utils.get_bound_args(
          jax.jit(fn).lower(qkv, gate, beta)
      )
    self.assertTrue(
        any(
            isinstance(ba.op, pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet)
            for ba in bound_args
        )
    )

  def test_kda_scalar_gate_with_dt_bias_mosaic_fwd_bwd_matches_xla(self):
    h, b, t, d = 4, 1, 64, 128
    keys = jax.random.split(jax.random.PRNGKey(31), 8)
    q = jax.random.normal(keys[0], (h, b, t, d), jnp.float32) * 0.4
    k = jax.random.normal(keys[1], (h, b, t, d), jnp.float32) * 0.4
    v = jax.random.normal(keys[2], (h, b, t, d), jnp.float32) * 0.4
    g = jax.random.normal(keys[3], (h, b, t, 1), jnp.float32) * 0.5
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (h, b, t), jnp.float32))
    a_log = jnp.log(jax.random.uniform(keys[5], (h,), minval=0.1, maxval=1.0))
    dt_bias = jax.random.normal(keys[6], (h,), jnp.float32) * 0.1
    do = jax.random.normal(keys[7], (h, b, t, d), jnp.float32)

    with mock.patch.object(
        pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention,
        "supported_on",
        return_value=True,
    ):
      def run(impl: str):
        def loss_fn(q_, k_, v_, g_, beta_, a_log_, dt_bias_):
          out, _ = api.kimi_delta_attention(
              q_,
              k_,
              v_,
              g_,
              beta_,
              a_log=a_log_,
              delta_time_bias=dt_bias_,
              use_qk_l2norm=True,
              use_gate_in_kernel=True,
              per_channel_gate=False,
              chunk_size=64,
              implementation=impl,
          )
          return jnp.sum(out * do)

        return jax.value_and_grad(loss_fn, argnums=tuple(range(7)))(
            q, k, v, g, beta, a_log, dt_bias
        )

      loss_ref, grads_ref = run("xla")
      loss_mos, grads_mos = run("mosaic")

    self.assertAlmostEqual(float(loss_mos), float(loss_ref), places=2)
    for g_mos, g_ref in zip(grads_mos, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_mos - g_ref))), 1e-3)

  def test_off_tpu_vmem_fallback_and_oversized_gqa_rejection(self):
    with mock.patch.object(
        common.pltpu, "get_tpu_info", side_effect=RuntimeError("no TPU")
    ), mock.patch.object(common, "get_interpret", return_value=False):
      limits = common.get_tpu_limits()
    self.assertEqual(limits.vmem_limit_bytes, int(64 * 1024 * 1024 * 0.9))

    b, t, h_qk, h_v, d = 1, 16, 1, 16, 128
    q = jnp.zeros((b, t, h_qk, d), jnp.float32)
    v = jnp.zeros((b, t, h_v, d), jnp.float32)
    gate = jnp.zeros((b, t, h_v), jnp.float32)
    beta = jnp.full((b, t, h_v), 0.5, jnp.float32)
    with mock.patch.object(common, "get_vmem_limit_bytes", return_value=65536):
      with self.assertRaisesRegex(NotImplementedError, "GQA group size"):
        api.gated_delta_net(
            q, q, v, gate, beta, chunk_size=16, implementation="mosaic"
        )
      out_fb, _ = api.gated_delta_net(
          q, q, v, gate, beta, chunk_size=16, implementation=("mosaic", "xla")
      )
    self.assertEqual(out_fb.shape, (b, t, h_v, d))


if __name__ == "__main__":
  absltest.main()
