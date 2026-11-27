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

  def test_forward_megakernel_options_are_opt_in(self):
    config = pallas_mosaic_tpu.Config()
    self.assertFalse(config.fuse_forward)
    self.assertFalse(config.packed_forward)
    self.assertFalse(config.packed_output)

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

    with self.assertRaisesRegex(ValueError, "max_num_segments"):
      _call_attention(
          "mosaic",
          q,
          v,
          context_parallel_metadata=context_parallel_metadata,
          segment_ids=segment_ids,
      )

    q_unaligned = jnp.ones((1, 1, 80, 128), dtype=jnp.float32)
    v_unaligned = jnp.ones((1, 1, 80, 128), dtype=jnp.float32)
    with self.assertRaisesRegex(NotImplementedError, "chunk_size"):
      _call_attention(
          "mosaic",
          q_unaligned,
          v_unaligned,
          context_parallel_metadata=context_parallel_metadata,
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

  @parameterized.parameters((16,), (32,), (64,), (128,))
  def test_tri_inv_xor_dnc_matches_exact_inverse(self, bt: int):
    rng = jax.random.PRNGKey(100 + bt)
    raw = jax.random.normal(rng, (4, bt, bt), jnp.float32) * 0.12
    idx = jnp.arange(bt)
    strict = idx[:, None] > idx[None, :]
    l_mat = jnp.where(strict[None, :, :], raw, 0.0)
    eye = jnp.eye(bt, dtype=jnp.float32)[None, :, :]
    exact = jnp.linalg.solve(eye + l_mat, jnp.broadcast_to(eye, l_mat.shape))

    inv_fp32 = common.tri_inv_xor_dnc(l_mat, use_fp32=True)
    self.assertLess(float(jnp.max(jnp.abs(inv_fp32 - exact))), 4e-5)

    inv_1pass = common.tri_inv_xor_dnc(l_mat, use_fp32=False, two_pass_bf16=False)
    inv_2pass = common.tri_inv_xor_dnc(l_mat, use_fp32=False, two_pass_bf16=True)
    err_1pass = float(jnp.max(jnp.abs(inv_1pass - exact)))
    err_2pass = float(jnp.max(jnp.abs(inv_2pass - exact)))
    self.assertLessEqual(err_2pass, err_1pass + 1e-6)
    self.assertLess(err_2pass, 4e-5)

  def test_gdn_chunk128_omits_akk_residual_and_matches_xla(self):
    b, t, h_qk, h_v, d = 1, 256, 2, 4, 128
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(jax.random.PRNGKey(99), 6)
    qkv = jax.random.normal(keys[0], (b, t, c_qkv), jnp.float32) * 0.4
    conv_w = jax.random.normal(keys[1], (4, c_qkv), jnp.float32) * 0.15
    gate = jax.random.normal(keys[2], (b, t, h_v), jnp.float32) * 0.5
    beta = jax.random.normal(keys[3], (b, t, h_v), jnp.float32)
    a_log = jnp.zeros((h_v,), jnp.float32)
    do = jax.random.normal(keys[4], (b, t, h_v, d), jnp.float32)

    gdn_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet()
    _, res64 = gdn_op(
        qkv=qkv,
        gate=gate,
        beta=beta,
        num_k_heads=h_qk,
        num_v_heads=h_v,
        head_k_dim=d,
        head_v_dim=d,
        conv_weight=conv_w,
        a_log=a_log,
        use_qk_l2norm=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        chunk_size=64,
        return_residuals=True,
    )
    _, res128 = gdn_op(
        qkv=qkv,
        gate=gate,
        beta=beta,
        num_k_heads=h_qk,
        num_v_heads=h_v,
        head_k_dim=d,
        head_v_dim=d,
        conv_weight=conv_w,
        a_log=a_log,
        use_qk_l2norm=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        chunk_size=128,
        return_residuals=True,
    )
    self.assertIsNotNone(res64.akk)
    self.assertIsNone(res128.akk)

    def run(impl: str):
      def loss_fn(qkv_, gate_, beta_, conv_w_, a_log_):
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
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            chunk_size=128,
            implementation=impl,
        )
        return jnp.sum(out * do)

      return jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(
          qkv, gate, beta, conv_w, a_log
      )

    loss_ref, grads_ref = run("xla")
    loss_mos, grads_mos = run("mosaic")
    self.assertAlmostEqual(float(loss_mos), float(loss_ref), places=2)
    for g_mos, g_ref in zip(grads_mos, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_mos - g_ref))), 1e-3)

  def test_kda_chunk128_per_channel_gate_fwd_bwd_matches_xla(self):
    h, b, t, d = 2, 1, 256, 128
    keys = jax.random.split(jax.random.PRNGKey(123), 7)
    q = jax.random.normal(keys[0], (h, b, t, d), jnp.float32) * 0.4
    k = jax.random.normal(keys[1], (h, b, t, d), jnp.float32) * 0.4
    v = jax.random.normal(keys[2], (h, b, t, d), jnp.float32) * 0.4
    g = jax.random.normal(keys[3], (h, b, t, d), jnp.float32) * 0.4
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (h, b, t), jnp.float32))
    a_log = jnp.zeros((h,), jnp.float32)
    do = jax.random.normal(keys[5], (h, b, t, d), jnp.float32)

    with mock.patch.object(
        pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention,
        "supported_on",
        return_value=True,
    ):
      def run(impl: str):
        def loss_fn(q_, k_, v_, g_, beta_, a_log_):
          out, _ = api.kimi_delta_attention(
              q_,
              k_,
              v_,
              g_,
              beta_,
              a_log=a_log_,
              use_qk_l2norm=True,
              use_gate_in_kernel=True,
              per_channel_gate=True,
              lower_bound=-5.0,
              chunk_size=128,
              implementation=impl,
          )
          return jnp.sum(out * do)

        return jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(
            q, k, v, g, beta, a_log
        )

      loss_ref, grads_ref = run("xla")
      loss_mos, grads_mos = run("mosaic")

    self.assertAlmostEqual(float(loss_mos), float(loss_ref), places=2)
    for g_mos, g_ref in zip(grads_mos, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_mos - g_ref))), 2e-3)

  def test_gdn_two_pass_bf16_improves_precision(self):
    b, t, h_qk, h_v, d = 1, 128, 2, 4, 128
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(jax.random.PRNGKey(2026), 6)
    qkv_f32 = jax.random.normal(keys[0], (b, t, c_qkv), jnp.float32) * 0.6
    gate_f32 = jax.random.normal(keys[1], (b, t, h_v), jnp.float32) * 0.4
    beta_f32 = jax.random.normal(keys[2], (b, t, h_v), jnp.float32) + 0.5
    a_log = jnp.log(jnp.full((h_v,), 0.25, jnp.float32))
    do_f32 = jax.random.normal(keys[3], (b, t, h_v, d), jnp.float32) * 0.2

    qkv_bf16 = qkv_f32.astype(jnp.bfloat16)
    gate_bf16 = gate_f32.astype(jnp.bfloat16)
    beta_bf16 = beta_f32.astype(jnp.bfloat16)
    do_bf16 = do_f32.astype(jnp.bfloat16)

    def ref_fn(qkv_, gate_, beta_):
      out, _ = api.gated_delta_net(
          qkv=qkv_,
          gate=gate_,
          beta=beta_,
          num_k_heads=h_qk,
          num_v_heads=h_v,
          head_k_dim=d,
          head_v_dim=d,
          a_log=a_log,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          implementation="xla",
      )
      return jnp.sum(out * do_bf16.astype(jnp.float32))

    loss_ref, (dqkv_ref, dg_ref, db_ref) = jax.value_and_grad(
        ref_fn, argnums=(0, 1, 2)
    )(
        qkv_bf16.astype(jnp.float32),
        gate_bf16.astype(jnp.float32),
        beta_bf16.astype(jnp.float32),
    )

    for two_pass, expected_h_dtype in (
        (False, jnp.bfloat16),
        (True, jnp.float32),
    ):
      gdn_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet(
          config=pallas_mosaic_tpu.Config(
              chunk_size=128, two_pass_bf16=two_pass
          )
      )
      (out_chk, _), res_chk = gdn_op(
          qkv=qkv_bf16,
          gate=gate_bf16,
          beta=beta_bf16,
          num_k_heads=h_qk,
          num_v_heads=h_v,
          head_k_dim=d,
          head_v_dim=d,
          a_log=a_log,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          return_residuals=True,
      )
      self.assertEqual(out_chk.dtype, jnp.bfloat16)
      self.assertEqual(res_chk.h.dtype, expected_h_dtype)

    def mos_fn(two_pass: bool):
      gdn_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet(
          config=pallas_mosaic_tpu.Config(
              chunk_size=128,
              two_pass_bf16=two_pass,
          )
      )

      def loss(qkv_, gate_, beta_):
        out, _ = gdn_op(
            qkv=qkv_,
            gate=gate_,
            beta=beta_,
            num_k_heads=h_qk,
            num_v_heads=h_v,
            head_k_dim=d,
            head_v_dim=d,
            a_log=a_log,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
        )
        return jnp.sum(out.astype(jnp.float32) * do_bf16.astype(jnp.float32))

      return jax.value_and_grad(loss, argnums=(0, 1, 2))(
          qkv_bf16, gate_bf16, beta_bf16
      )

    loss_1p, (dqkv_1p, dg_1p, db_1p) = mos_fn(False)
    loss_2p, (dqkv_2p, dg_2p, db_2p) = mos_fn(True)
    err_loss_1p = abs(float(loss_1p) - float(loss_ref))
    err_loss_2p = abs(float(loss_2p) - float(loss_ref))
    self.assertLessEqual(err_loss_2p, err_loss_1p + 1e-5)
    err_dqkv_1p = float(jnp.max(jnp.abs(dqkv_1p.astype(jnp.float32) - dqkv_ref)))
    err_dqkv_2p = float(jnp.max(jnp.abs(dqkv_2p.astype(jnp.float32) - dqkv_ref)))
    err_dg_1p = float(jnp.max(jnp.abs(dg_1p.astype(jnp.float32) - dg_ref)))
    err_dg_2p = float(jnp.max(jnp.abs(dg_2p.astype(jnp.float32) - dg_ref)))
    err_db_1p = float(jnp.max(jnp.abs(db_1p.astype(jnp.float32) - db_ref)))
    err_db_2p = float(jnp.max(jnp.abs(db_2p.astype(jnp.float32) - db_ref)))
    self.assertLessEqual(err_dg_2p, err_dg_1p)
    self.assertLess(err_dg_2p, 2e-2)
    if jax.default_backend() == "tpu":
      self.assertLess(err_dqkv_2p, err_dqkv_1p)
      self.assertLess(err_dqkv_2p, 2e-4)
      self.assertLessEqual(err_dg_2p, err_dg_1p)
      self.assertLess(err_dg_2p, 2e-4)
      self.assertLessEqual(err_db_2p, err_db_1p)
      self.assertLess(err_db_2p, 5e-4)

  @parameterized.parameters((False,), (True,))
  def test_gdn_fwd_chunk128_with_vjp_config_none(self, remat: bool):
    b, t, h_qk, h_v, d = 1, 128, 4, 16, 128
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(jax.random.PRNGKey(301 + int(remat)), 5)
    qkv = jax.random.normal(keys[0], (b, t, c_qkv), jnp.bfloat16) * 0.4
    gate = jax.random.normal(keys[1], (b, t, h_v), jnp.bfloat16) * 0.3
    beta = jax.random.normal(keys[2], (b, t, h_v), jnp.bfloat16)
    a_log = jnp.zeros((h_v,), jnp.float32)
    do = jax.random.normal(keys[3], (b, t, h_v, d), jnp.bfloat16) * 0.2

    gdn_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet()
    self.assertIsNone(gdn_op.config)
    self.assertIsNone(gdn_op.vjp.config)
    fwd_cfg = pallas_mosaic_tpu.Config(
        chunk_size=128,
        two_pass_bf16=True,
        rematerialize_for_backward=remat,
    )
    tpu7x_limits = common._TpuLimits(  # pylint: disable=protected-access
        vmem_limit_bytes=int(64 * 1024 * 1024 * 0.9),
        block_align_minor=8,
        block_align_major=128,
    )
    with mock.patch.object(
        pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet,
        "_get_heuristics_config",
        return_value=fwd_cfg,
    ), mock.patch.object(
        pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet,
        "supported_on",
        return_value=True,
    ), mock.patch.object(
        common,
        "get_tpu_limits",
        return_value=tpu7x_limits,
    ):
      ba = gdn_op.bind(
          qkv=qkv,
          gate=gate,
          beta=beta,
          num_k_heads=h_qk,
          num_v_heads=h_v,
          head_k_dim=d,
          head_v_dim=d,
          a_log=a_log,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
      )
      vjp_ba = gdn_op.vjp.bind(**ba.vjp_arg_spec)
      vjp_cfgs = gdn_op.vjp._get_autotuning_configs(vjp_ba)  # pylint: disable=protected-access
      self.assertTrue(all(c.chunk_size == 128 for c in vjp_cfgs))
      self.assertTrue(all(c.two_pass_bf16 for c in vjp_cfgs))

      def loss_mos(qkv_, gate_, beta_):
        out, _ = gdn_op(
            qkv=qkv_,
            gate=gate_,
            beta=beta_,
            num_k_heads=h_qk,
            num_v_heads=h_v,
            head_k_dim=d,
            head_v_dim=d,
            a_log=a_log,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
        )
        return jnp.sum(out.astype(jnp.float32) * do.astype(jnp.float32))

      loss_m, grads_m = jax.value_and_grad(loss_mos, argnums=(0, 1, 2))(
          qkv, gate, beta
      )

    def loss_ref(qkv_, gate_, beta_):
      out, _ = api.gated_delta_net(
          qkv=qkv_,
          gate=gate_,
          beta=beta_,
          num_k_heads=h_qk,
          num_v_heads=h_v,
          head_k_dim=d,
          head_v_dim=d,
          a_log=a_log,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          implementation="xla",
      )
      return jnp.sum(out * do.astype(jnp.float32))

    loss_r, grads_r = jax.value_and_grad(loss_ref, argnums=(0, 1, 2))(
        qkv.astype(jnp.float32),
        gate.astype(jnp.float32),
        beta.astype(jnp.float32),
    )
    self.assertAlmostEqual(float(loss_m), float(loss_r), delta=0.05)
    for gm, gr in zip(grads_m, grads_r):
      self.assertEqual(gm.dtype, jnp.bfloat16)
      self.assertLess(float(jnp.max(jnp.abs(gm.astype(jnp.float32) - gr))), 3e-2)

  @parameterized.parameters((64,), (128,))
  def test_gdn_bf16_large_beta_logits_fwd_bwd_matches_xla(self, bt: int):
    b, t, h_qk, h_v, d = 1, 128, 2, 4, 128
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(jax.random.PRNGKey(400 + bt), 5)
    qkv = (jax.random.normal(keys[0], (b, t, c_qkv), jnp.float32) * 0.5).astype(
        jnp.bfloat16
    )
    gate = (jax.random.normal(keys[1], (b, t, h_v), jnp.float32) * 0.3).astype(
        jnp.bfloat16
    )
    beta = jax.random.uniform(
        keys[2], (b, t, h_v), jnp.float32, minval=-8.0, maxval=8.0
    ).astype(jnp.bfloat16)
    beta = beta.at[0, 0, 0].set(jnp.bfloat16(7.5))
    beta = beta.at[0, 1, 0].set(jnp.bfloat16(-7.5))
    a_log = jnp.zeros((h_v,), jnp.float32)
    do = (jax.random.normal(keys[3], (b, t, h_v, d), jnp.float32) * 0.5).astype(
        jnp.bfloat16
    )

    gdn_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet(
        config=pallas_mosaic_tpu.Config(chunk_size=bt, two_pass_bf16=True)
    )

    def loss_mos(qkv_, gate_, beta_):
      out, _ = gdn_op(
          qkv=qkv_,
          gate=gate_,
          beta=beta_,
          num_k_heads=h_qk,
          num_v_heads=h_v,
          head_k_dim=d,
          head_v_dim=d,
          a_log=a_log,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
      )
      return jnp.sum(out.astype(jnp.float32) * do.astype(jnp.float32))

    def loss_ref(qkv_, gate_, beta_):
      out, _ = api.gated_delta_net(
          qkv=qkv_,
          gate=gate_,
          beta=beta_,
          num_k_heads=h_qk,
          num_v_heads=h_v,
          head_k_dim=d,
          head_v_dim=d,
          a_log=a_log,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          implementation="xla",
      )
      return jnp.sum(out * do.astype(jnp.float32))

    with mock.patch.object(
        pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet,
        "supported_on",
        return_value=True,
    ):
      loss_m, (_, _, db_m) = jax.value_and_grad(loss_mos, argnums=(0, 1, 2))(
          qkv, gate, beta
      )
    loss_r, (_, _, db_r) = jax.value_and_grad(loss_ref, argnums=(0, 1, 2))(
        qkv.astype(jnp.float32),
        gate.astype(jnp.float32),
        beta.astype(jnp.float32),
    )
    self.assertEqual(db_m.dtype, jnp.bfloat16)
    self.assertGreater(abs(float(db_r[0, 0, 0])), 0.0)
    self.assertGreater(abs(float(db_m[0, 0, 0])), 0.0)
    self.assertAlmostEqual(float(loss_m), float(loss_r), delta=0.05)
    self.assertLess(
        float(jnp.max(jnp.abs(db_m.astype(jnp.float32) - db_r))), 2e-2
    )

  def test_chunk128_vmem_mini_batch_sizing_for_hv128(self):
    from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_kernel as fwd_mod  # pylint: disable=g-import-not-at-top

    tpu_info = types.SimpleNamespace(
        vmem_capacity_bytes=64 * 1024 * 1024,
        num_sublanes=16,
        num_lanes=128,
    )
    with mock.patch.object(common.pltpu, "get_tpu_info", return_value=tpu_info):
      mb_gdn = fwd_mod._compute_intra_fused_mini_batch(  # pylint: disable=protected-access
          128,
          128,
          128,
          128,
          jnp.bfloat16,
          per_channel_gate=False,
          store_akk=False,
      )
      mb_kda = fwd_mod._compute_intra_fused_mini_batch(  # pylint: disable=protected-access
          128,
          128,
          128,
          128,
          jnp.bfloat16,
          per_channel_gate=True,
          store_akk=False,
      )
      gdn_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet()
      gdn_ba = gdn_op.bind(
          qkv=jnp.zeros((1, 128, 2 * 16 * 128 + 64 * 128), jnp.bfloat16),
          gate=jnp.zeros((1, 128, 64), jnp.bfloat16),
          beta=jnp.zeros((1, 128, 64), jnp.bfloat16),
          num_k_heads=16,
          num_v_heads=64,
          head_k_dim=128,
          head_v_dim=128,
      )
      gdn_cfgs = gdn_op._get_autotuning_configs(gdn_ba)  # pylint: disable=protected-access
      kda_op = pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention()
      kda_ba = kda_op.bind(
          jnp.zeros((32, 1, 128, 128), jnp.bfloat16),
          jnp.zeros((32, 1, 128, 128), jnp.bfloat16),
          jnp.zeros((32, 1, 128, 128), jnp.bfloat16),
          jnp.zeros((32, 1, 128, 128), jnp.bfloat16),
          jnp.zeros((32, 1, 128), jnp.bfloat16),
          per_channel_gate=True,
      )
      kda_cfgs = kda_op._get_autotuning_configs(kda_ba)  # pylint: disable=protected-access
      gdn_ba_256 = gdn_op.bind(
          qkv=jnp.zeros((1, 128, 2 * 16 * 256 + 64 * 256), jnp.bfloat16),
          gate=jnp.zeros((1, 128, 64), jnp.bfloat16),
          beta=jnp.zeros((1, 128, 64), jnp.bfloat16),
          num_k_heads=16,
          num_v_heads=64,
          head_k_dim=256,
          head_v_dim=256,
      )
      gdn_cfgs_256 = gdn_op._get_autotuning_configs(gdn_ba_256)  # pylint: disable=protected-access
      gdn_ba_t64 = gdn_op.bind(
          qkv=jnp.zeros((1, 64, 2 * 16 * 128 + 64 * 128), jnp.bfloat16),
          gate=jnp.zeros((1, 64, 64), jnp.bfloat16),
          beta=jnp.zeros((1, 64, 64), jnp.bfloat16),
          num_k_heads=16,
          num_v_heads=64,
          head_k_dim=128,
          head_v_dim=128,
      )
      gdn_cfgs_t64 = gdn_op._get_autotuning_configs(gdn_ba_t64)  # pylint: disable=protected-access
    self.assertEqual(mb_gdn, 16)
    self.assertEqual(mb_kda, 8)
    self.assertEqual(128 % mb_gdn, 0)
    self.assertEqual(128 % mb_kda, 0)
    for cfg in gdn_cfgs:
      if cfg.chunk_size > 64 and cfg.mini_batch is not None:
        self.assertLessEqual(cfg.mini_batch, 16)
    for cfg in kda_cfgs:
      if cfg.chunk_size > 64 and cfg.mini_batch is not None:
        self.assertLessEqual(cfg.mini_batch, 8)
    for cfg in gdn_cfgs_256:
      if cfg.chunk_size > 64 and cfg.mini_batch is not None:
        self.assertLessEqual(cfg.mini_batch, 8)
    self.assertTrue(all(cfg.chunk_size <= 64 for cfg in gdn_cfgs_t64))

  @parameterized.parameters(
      (64, False),
      (64, True),
      (128, False),
      (128, True),
  )
  def test_kda_two_pass_bf16_matches_xla(
      self, chunk_size: int, remat: bool
  ):
    h, b, t, d = 2, 1, 256, 128
    keys = jax.random.split(jax.random.PRNGKey(2027 + chunk_size + int(remat)), 7)
    q_bf16 = (jax.random.normal(keys[0], (h, b, t, d), jnp.float32) * 0.4).astype(
        jnp.bfloat16
    )
    k_bf16 = (jax.random.normal(keys[1], (h, b, t, d), jnp.float32) * 0.4).astype(
        jnp.bfloat16
    )
    v_bf16 = (jax.random.normal(keys[2], (h, b, t, d), jnp.float32) * 0.4).astype(
        jnp.bfloat16
    )
    g_bf16 = (jax.random.normal(keys[3], (h, b, t, d), jnp.float32) * 0.4).astype(
        jnp.bfloat16
    )
    beta_bf16 = jax.nn.sigmoid(
        jax.random.normal(keys[4], (h, b, t), jnp.float32)
    ).astype(jnp.bfloat16)
    a_log = jnp.zeros((h,), jnp.float32)
    do_bf16 = (
        jax.random.normal(keys[5], (h, b, t, d), jnp.float32) * 0.2
    ).astype(jnp.bfloat16)

    def ref_fn(q_, k_, v_, g_, beta_, a_log_):
      out, _ = api.kimi_delta_attention(
          q_,
          k_,
          v_,
          g_,
          beta_,
          a_log=a_log_,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          per_channel_gate=True,
          lower_bound=-5.0,
          implementation="xla",
      )
      return jnp.sum(out * do_bf16.astype(jnp.float32)), out

    (loss_ref, out_ref), grads_ref = jax.value_and_grad(
        ref_fn, argnums=tuple(range(6)), has_aux=True
    )(
        q_bf16.astype(jnp.float32),
        k_bf16.astype(jnp.float32),
        v_bf16.astype(jnp.float32),
        g_bf16.astype(jnp.float32),
        beta_bf16.astype(jnp.float32),
        a_log,
    )

    kda_op = pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention(
        config=pallas_mosaic_tpu.Config(
            chunk_size=chunk_size,
            two_pass_bf16=True,
            rematerialize_for_backward=remat,
        )
    )
    with mock.patch.object(
        pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention,
        "supported_on",
        return_value=True,
    ):
      def mos_fn(q_, k_, v_, g_, beta_, a_log_):
        out, _ = kda_op(
            q_,
            k_,
            v_,
            g_,
            beta_,
            a_log=a_log_,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            per_channel_gate=True,
            lower_bound=-5.0,
        )
        return jnp.sum(out.astype(jnp.float32) * do_bf16.astype(jnp.float32)), out

      (loss_mos, out_mos), grads_mos = jax.value_and_grad(
          mos_fn, argnums=tuple(range(6)), has_aux=True
      )(q_bf16, k_bf16, v_bf16, g_bf16, beta_bf16, a_log)

    self.assertEqual(out_mos.dtype, jnp.bfloat16)
    self.assertAlmostEqual(float(loss_mos), float(loss_ref), places=2)
    self.assertLess(
        float(jnp.max(jnp.abs(out_mos.astype(jnp.float32) - out_ref))), 3e-4
    )
    # (dq, dk, dv, dg, dbeta, da_log)
    tols = (2e-4, 2e-4, 2e-4, 2e-4, 5e-4, 5e-4)
    for g_mos, g_ref, tol in zip(grads_mos, grads_ref, tols):
      self.assertLess(
          float(jnp.max(jnp.abs(g_mos.astype(jnp.float32) - g_ref))), tol
      )

  @parameterized.parameters((16,), (32,), (64,), (128,))
  def test_gdn_inplace_packed_fwd_bwd_matches_xla(self, chunk_size: int):
    b, t, h_qk, h_v, d = 2, 256, 1, 2, 128
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(jax.random.PRNGKey(500 + chunk_size), 9)
    qkv = jax.random.normal(keys[0], (b, t, c_qkv), jnp.float32) * 0.4
    conv_w = jax.random.normal(keys[1], (4, c_qkv), jnp.float32) * 0.15
    conv_b = jax.random.normal(keys[2], (c_qkv,), jnp.float32) * 0.05
    gate = jax.random.normal(keys[3], (b, t, h_v), jnp.float32) * 0.4
    beta = jax.random.normal(keys[4], (b, t, h_v), jnp.float32)
    a_log = jnp.log(jax.random.uniform(keys[5], (h_v,), minval=0.1, maxval=0.8))
    dt_bias = jax.random.normal(keys[6], (h_v,), jnp.float32) * 0.1
    h0 = jax.random.normal(keys[7], (b, h_v, d, d), jnp.float32) * 0.1
    do = jax.random.normal(keys[8], (b, t, h_v, d), jnp.float32)

    # Row 0: leading 0-pad, [5, 0, 5] continuity across chunk 0->1, 4 docs in
    # chunk 1, cross-chunk doc 9, and trailing 0-pad.
    seg_row0 = jnp.concatenate([
        jnp.zeros((5,), jnp.int32),
        jnp.full((20,), 5, jnp.int32),
        jnp.zeros((7,), jnp.int32),
        jnp.full((48,), 5, jnp.int32),
        jnp.full((15,), 2, jnp.int32),
        jnp.full((15,), 7, jnp.int32),
        jnp.full((25,), 1, jnp.int32),
        jnp.full((65,), 9, jnp.int32),
        jnp.zeros((56,), jnp.int32),
    ])
    # Row 1: reused non-adjacent ID [1, 2, 1] (must reset at 1->2 and 2->1)
    # plus a full 128-token 0-padding span.
    seg_row1 = jnp.concatenate([
        jnp.full((20,), 1, jnp.int32),
        jnp.full((20,), 2, jnp.int32),
        jnp.full((30,), 1, jnp.int32),
        jnp.zeros((128,), jnp.int32),
        jnp.full((58,), 4, jnp.int32),
    ])
    segment_ids = jnp.stack([seg_row0, seg_row1], axis=0)

    def run(impl: str):
      def fn(qkv_, gate_, beta_, conv_w_, conv_b_, a_log_, dt_bias_, h0_):
        out, ht = api.gated_delta_net(
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
            delta_time_bias=dt_bias_,
            initial_state=h0_,
            output_final_state=True,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            chunk_size=chunk_size,
            segment_ids=segment_ids,
            implementation=impl,
        )
        loss = jnp.sum(out * do) + 0.1 * jnp.sum(ht)
        return loss, (out, ht)

      return jax.value_and_grad(fn, argnums=tuple(range(8)), has_aux=True)(
          qkv, gate, beta, conv_w, conv_b, a_log, dt_bias, h0
      )

    (loss_ref, (out_ref, ht_ref)), grads_ref = run("xla")
    (loss_mos, (out_mos, ht_mos)), grads_mos = run("mosaic")

    pad_mask = segment_ids == 0
    self.assertEqual(
        float(jnp.max(jnp.abs(jnp.where(pad_mask[:, :, None, None], out_mos, 0.0)))),
        0.0,
    )
    self.assertEqual(
        float(jnp.max(jnp.abs(jnp.where(pad_mask[:, :, None], grads_mos[0], 0.0)))),
        0.0,
    )
    out_atol = 1e-3 if jax.default_backend() == "cpu" else 3e-3
    grad_atol = 2e-3 if jax.default_backend() == "cpu" else 5e-3
    self.assertLess(float(jnp.max(jnp.abs(out_mos - out_ref))), out_atol)
    self.assertLess(float(jnp.max(jnp.abs(ht_mos - ht_ref))), out_atol)
    self.assertAlmostEqual(float(loss_mos), float(loss_ref), places=2)
    for g_mos, g_ref in zip(grads_mos, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_mos - g_ref))), grad_atol)

  @parameterized.parameters((64, False), (128, False), (64, True))
  def test_kda_inplace_packed_fwd_bwd_matches_xla(
      self, chunk_size: int, remat_bwd: bool
  ):
    h, b, t, d, w_conv = 2, 2, 256, 128, 4
    keys = jax.random.split(
        jax.random.PRNGKey(700 + chunk_size + int(remat_bwd)), 15
    )
    q = jax.random.normal(keys[0], (h, b, t, d), jnp.float32) * 0.4
    k = jax.random.normal(keys[1], (h, b, t, d), jnp.float32) * 0.4
    v = jax.random.normal(keys[2], (h, b, t, d), jnp.float32) * 0.4
    g = jax.random.normal(keys[3], (h, b, t, d), jnp.float32) * 0.4
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (h, b, t), jnp.float32))
    a_log = jnp.zeros((h,), jnp.float32)
    dt_bias = jax.random.normal(keys[5], (h * d,), jnp.float32) * 0.05
    cw_q = jax.random.normal(keys[6], (h, w_conv, d), jnp.float32) * 0.15
    cw_k = jax.random.normal(keys[7], (h, w_conv, d), jnp.float32) * 0.15
    cw_v = jax.random.normal(keys[8], (h, w_conv, d), jnp.float32) * 0.15
    cb_q = jax.random.normal(keys[9], (h, d), jnp.float32) * 0.05
    cb_k = jax.random.normal(keys[10], (h, d), jnp.float32) * 0.05
    cb_v = jax.random.normal(keys[11], (h, d), jnp.float32) * 0.05
    do = jax.random.normal(keys[12], (h, b, t, d), jnp.float32)

    seg_row0 = jnp.concatenate([
        jnp.zeros((6,), jnp.int32),
        jnp.full((22,), 1, jnp.int32),
        jnp.zeros((6,), jnp.int32),
        jnp.full((46,), 1, jnp.int32),
        jnp.full((18,), 2, jnp.int32),
        jnp.full((20,), 3, jnp.int32),
        jnp.full((30,), 4, jnp.int32),
        jnp.full((52,), 5, jnp.int32),
        jnp.zeros((56,), jnp.int32),
    ])
    seg_row1 = jnp.concatenate([
        jnp.full((40,), 1, jnp.int32),
        jnp.zeros((128,), jnp.int32),
        jnp.full((60,), 2, jnp.int32),
        jnp.full((28,), 99, jnp.int32),  # > max_num_segments=5 -> padding
    ])
    segment_ids = jnp.stack([seg_row0, seg_row1], axis=0)

    kda_op = pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention(
        config=pallas_mosaic_tpu.Config(
            chunk_size=chunk_size, rematerialize_for_backward=remat_bwd
        )
    )
    with mock.patch.object(
        pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention,
        "supported_on",
        return_value=True,
    ):
      _, res = kda_op(
          q,
          k,
          v,
          g,
          beta,
          a_log=a_log,
          delta_time_bias=dt_bias,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          per_channel_gate=True,
          lower_bound=-5.0,
          chunk_size=chunk_size,
          segment_ids=segment_ids,
          max_num_segments=5,
          conv_weight_q=cw_q,
          conv_weight_k=cw_k,
          conv_weight_v=cw_v,
          conv_bias_q=cb_q,
          conv_bias_k=cb_k,
          conv_bias_v=cb_v,
          use_conv1d_in_kernel=True,
          return_residuals=True,
      )
      self.assertIsNone(res.aligned_cu_seqlens)

      def run_mos(q_, k_, v_, g_, beta_, a_log_, dt_bias_, cw_q_, cw_k_, cw_v_):
        out, _ = kda_op(
            q_,
            k_,
            v_,
            g_,
            beta_,
            a_log=a_log_,
            delta_time_bias=dt_bias_,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            per_channel_gate=True,
            lower_bound=-5.0,
            chunk_size=chunk_size,
            segment_ids=segment_ids,
            max_num_segments=5,
            conv_weight_q=cw_q_,
            conv_weight_k=cw_k_,
            conv_weight_v=cw_v_,
            conv_bias_q=cb_q,
            conv_bias_k=cb_k,
            conv_bias_v=cb_v,
            use_conv1d_in_kernel=True,
        )
        return jnp.sum(out * do), out

      def run_ref(q_, k_, v_, g_, beta_, a_log_, dt_bias_, cw_q_, cw_k_, cw_v_):
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
            per_channel_gate=True,
            lower_bound=-5.0,
            chunk_size=chunk_size,
            segment_ids=segment_ids,
            max_num_segments=5,
            conv_weight_q=cw_q_,
            conv_weight_k=cw_k_,
            conv_weight_v=cw_v_,
            conv_bias_q=cb_q,
            conv_bias_k=cb_k,
            conv_bias_v=cb_v,
            use_conv1d_in_kernel=True,
            implementation="xla",
        )
        return jnp.sum(out * do), out

      args_all = (q, k, v, g, beta, a_log, dt_bias, cw_q, cw_k, cw_v)
      (loss_ref, out_ref), grads_ref = jax.value_and_grad(
          run_ref, argnums=tuple(range(10)), has_aux=True
      )(*args_all)
      (loss_mos, out_mos), grads_mos = jax.value_and_grad(
          run_mos, argnums=tuple(range(10)), has_aux=True
      )(*args_all)

    out_atol = 1e-3 if jax.default_backend() == "cpu" else 3e-3
    grad_atol = 2e-3 if jax.default_backend() == "cpu" else 5e-3
    loss_places = 2 if jax.default_backend() == "cpu" else 1
    self.assertLess(float(jnp.max(jnp.abs(out_mos - out_ref))), out_atol)
    self.assertAlmostEqual(
        float(loss_mos), float(loss_ref), places=loss_places
    )
    for g_mos, g_ref in zip(grads_mos, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_mos - g_ref))), grad_atol)

  @parameterized.parameters((False,), (True,))
  def test_kda_packed_reused_ids_and_unaligned_seq_len_matches_xla(
      self, remat_bwd: bool
  ):
    h, h_q, b, t, d, w_conv = 4, 2, 2, 90, 128, 4  # T=90 % 64 != 0, H_Q < H
    keys = jax.random.split(jax.random.PRNGKey(810 + int(remat_bwd)), 13)
    q_gqa = jax.random.normal(keys[0], (h_q, b, t, d), jnp.float32) * 0.4
    k_gqa = jax.random.normal(keys[1], (h_q, b, t, d), jnp.float32) * 0.4
    v = jax.random.normal(keys[2], (h, b, t, d), jnp.float32) * 0.4
    g = jax.random.normal(keys[3], (h, b, t, d), jnp.float32) * 0.4
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (h, b, t), jnp.float32))
    a_log = jnp.zeros((h,), jnp.float32)
    dt_bias = jax.random.normal(keys[5], (h * d,), jnp.float32) * 0.05
    cw_q = jax.random.normal(keys[6], (h_q, w_conv, d), jnp.float32) * 0.15
    cw_k = jax.random.normal(keys[7], (h_q, w_conv, d), jnp.float32) * 0.15
    cw_v = jax.random.normal(keys[8], (h, w_conv, d), jnp.float32) * 0.15
    cb_q = jax.random.normal(keys[9], (h_q, d), jnp.float32) * 0.05
    cb_k = jax.random.normal(keys[10], (h_q, d), jnp.float32) * 0.05
    cb_v = jax.random.normal(keys[11], (h, d), jnp.float32) * 0.05
    do = jax.random.normal(keys[12], (h, b, t, d), jnp.float32)

    # Row 0: leading 0-pad, [1, 2, 1] reused IDs (3 spans with max_num_segments=2),
    # mid 0-pad continuing span 1, and > max_num_segments clipped to padding.
    seg_row0 = jnp.concatenate([
        jnp.zeros((5,), jnp.int32),
        jnp.full((20,), 1, jnp.int32),
        jnp.full((25,), 2, jnp.int32),
        jnp.full((15,), 1, jnp.int32),
        jnp.zeros((6,), jnp.int32),
        jnp.full((14,), 1, jnp.int32),
        jnp.full((5,), 9, jnp.int32),
    ])
    # Row 1: mid 0-pad + [2, 1, 2] reused IDs + trailing 0-pad.
    seg_row1 = jnp.concatenate([
        jnp.full((22,), 2, jnp.int32),
        jnp.zeros((8,), jnp.int32),
        jnp.full((24,), 1, jnp.int32),
        jnp.full((26,), 2, jnp.int32),
        jnp.zeros((10,), jnp.int32),
    ])
    segment_ids = jnp.stack([seg_row0, seg_row1], axis=0)

    q_rep = jnp.repeat(q_gqa, h // h_q, axis=0)
    k_rep = jnp.repeat(k_gqa, h // h_q, axis=0)
    kda_default = pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention()
    ba_short = kda_default.bind(
        q_rep[:, :, :48, :],
        k_rep[:, :, :48, :],
        v[:, :, :48, :],
        g[:, :, :48, :],
        beta[:, :, :48],
        segment_ids=segment_ids[:, :48],
        max_num_segments=2,
    )
    cfgs_short = kda_default._get_autotuning_configs(ba_short)  # pylint: disable=protected-access
    self.assertEqual({c.chunk_size for c in cfgs_short}, {64})
    ba_long = kda_default.bind(
        q_rep, k_rep, v, g, beta, segment_ids=segment_ids, max_num_segments=2
    )
    cfgs_long = kda_default._get_autotuning_configs(ba_long)  # pylint: disable=protected-access
    self.assertEqual({c.chunk_size for c in cfgs_long}, {64, 128})

    kda_op = pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention(
        config=pallas_mosaic_tpu.Config(
            chunk_size=64,
            rematerialize_for_backward=remat_bwd,
            residual_checkpoint_name="kda_unaligned_ckpt",
        )
    )
    with mock.patch.object(
        pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention,
        "supported_on",
        return_value=True,
    ):
      def run_mos(q_, k_, v_, g_, beta_, a_log_, dt_bias_, cw_q_, cw_k_, cw_v_):
        out, _ = kda_op(
            jnp.repeat(q_, h // h_q, axis=0),
            jnp.repeat(k_, h // h_q, axis=0),
            v_,
            g_,
            beta_,
            a_log=a_log_,
            delta_time_bias=dt_bias_,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            per_channel_gate=True,
            lower_bound=-5.0,
            chunk_size=64,
            segment_ids=segment_ids,
            max_num_segments=2,
            conv_weight_q=cw_q_,
            conv_weight_k=cw_k_,
            conv_weight_v=cw_v_,
            conv_bias_q=cb_q,
            conv_bias_k=cb_k,
            conv_bias_v=cb_v,
            use_conv1d_in_kernel=True,
        )
        return jnp.sum(out * do), out

      def run_ref(q_, k_, v_, g_, beta_, a_log_, dt_bias_, cw_q_, cw_k_, cw_v_):
        out, _ = api.kimi_delta_attention(
            jnp.repeat(q_, h // h_q, axis=0),
            jnp.repeat(k_, h // h_q, axis=0),
            v_,
            g_,
            beta_,
            a_log=a_log_,
            delta_time_bias=dt_bias_,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            per_channel_gate=True,
            lower_bound=-5.0,
            chunk_size=64,
            segment_ids=segment_ids,
            max_num_segments=2,
            conv_weight_q=cw_q_,
            conv_weight_k=cw_k_,
            conv_weight_v=cw_v_,
            conv_bias_q=cb_q,
            conv_bias_k=cb_k,
            conv_bias_v=cb_v,
            use_conv1d_in_kernel=True,
            implementation="xla",
        )
        return jnp.sum(out * do), out

      args_all = (q_gqa, k_gqa, v, g, beta, a_log, dt_bias, cw_q, cw_k, cw_v)
      (loss_ref, out_ref), grads_ref = jax.value_and_grad(
          run_ref, argnums=tuple(range(10)), has_aux=True
      )(*args_all)
      ckpt_mos = jax.checkpoint(
          run_mos,
          policy=jax.checkpoint_policies.save_only_these_names(
              "kda_unaligned_ckpt"
          ),
      )
      (loss_mos, out_mos), grads_mos = jax.value_and_grad(
          ckpt_mos, argnums=tuple(range(10)), has_aux=True
      )(*args_all)

    self.assertEqual(out_mos.shape, (h, b, t, d))
    self.assertLess(float(jnp.max(jnp.abs(out_mos - out_ref))), 1e-3)
    self.assertAlmostEqual(float(loss_mos), float(loss_ref), places=2)
    for g_mos, g_ref in zip(grads_mos, grads_ref):
      self.assertEqual(g_mos.shape, g_ref.shape)
      self.assertLess(float(jnp.max(jnp.abs(g_mos - g_ref))), 2e-3)

  @parameterized.parameters((64,), (128,))
  def test_kda_scalar_gate_packed_fwd_bwd_matches_xla(self, chunk_size: int):
    h, b, t, d = 2, 2, 128, 128
    keys = jax.random.split(jax.random.PRNGKey(850 + chunk_size), 8)
    q = jax.random.normal(keys[0], (h, b, t, d), jnp.float32) * 0.4
    k = jax.random.normal(keys[1], (h, b, t, d), jnp.float32) * 0.4
    v = jax.random.normal(keys[2], (h, b, t, d), jnp.float32) * 0.4
    g = jax.random.normal(keys[3], (h, b, t, 1), jnp.float32) * 0.4
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (h, b, t), jnp.float32))
    a_log = jnp.zeros((h,), jnp.float32)
    dt_bias = jax.random.normal(keys[5], (h,), jnp.float32) * 0.1
    do = jax.random.normal(keys[6], (h, b, t, d), jnp.float32)
    segment_ids = jnp.asarray(
        [
            [0] * 4 + [1] * 28 + [0] * 8 + [1] * 24 + [2] * 36 + [1] * 28,
            [2] * 32 + [1] * 32 + [2] * 40 + [0] * 24,
        ],
        dtype=jnp.int32,
    )

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
              chunk_size=chunk_size,
              segment_ids=segment_ids,
              max_num_segments=2,
              implementation=impl,
          )
          return jnp.sum(out * do), out

        return jax.value_and_grad(
            loss_fn, argnums=tuple(range(7)), has_aux=True
        )(q, k, v, g, beta, a_log, dt_bias)

      (loss_ref, out_ref), grads_ref = run("xla")
      (loss_mos, out_mos), grads_mos = run("mosaic")

    out_atol = 1e-3 if jax.default_backend() == "cpu" else 3e-3
    grad_atol = 2e-3 if jax.default_backend() == "cpu" else 5e-3
    loss_places = 2 if jax.default_backend() == "cpu" else 1
    self.assertLess(float(jnp.max(jnp.abs(out_mos - out_ref))), out_atol)
    self.assertAlmostEqual(
        float(loss_mos), float(loss_ref), places=loss_places
    )
    for g_mos, g_ref in zip(grads_mos, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_mos - g_ref))), grad_atol)

  def test_gdn_packed_all_zero_and_negative_segments_with_h0_matches_xla(self):
    b, t, h_qk, h_v, d = 2, 128, 1, 2, 128
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(jax.random.PRNGKey(860), 8)
    qkv = jax.random.normal(keys[0], (b, t, c_qkv), jnp.float32) * 0.4
    conv_w = jax.random.normal(keys[1], (4, c_qkv), jnp.float32) * 0.15
    gate = jax.random.normal(keys[2], (b, t, h_v), jnp.float32) * 0.4
    beta = jax.random.normal(keys[3], (b, t, h_v), jnp.float32)
    a_log = jnp.zeros((h_v,), jnp.float32)
    h0 = jax.random.normal(keys[4], (b, h_v, d, d), jnp.float32) * 0.2
    do = jax.random.normal(keys[5], (b, t, h_v, d), jnp.float32)
    dht = jax.random.normal(keys[6], (b, h_v, d, d), jnp.float32)

    # Row 0: all-padding (0 and negative -1) -> ht[0] == h0[0], dh0[0] == dht[0].
    # Row 1: all-zero chunk 0 ([0]*64) then active doc 5 with -1 trailing pad.
    seg_row0 = jnp.where(jnp.arange(t) % 2 == 0, 0, -1).astype(jnp.int32)
    seg_row1 = jnp.concatenate([
        jnp.zeros((64,), jnp.int32),
        jnp.full((48,), 5, jnp.int32),
        jnp.full((16,), -1, jnp.int32),
    ])
    segment_ids = jnp.stack([seg_row0, seg_row1], axis=0)

    def run(impl: str):
      def fn(qkv_, gate_, beta_, conv_w_, a_log_, h0_):
        out, ht = api.gated_delta_net(
            qkv=qkv_,
            gate=gate_,
            beta=beta_,
            num_k_heads=h_qk,
            num_v_heads=h_v,
            head_k_dim=d,
            head_v_dim=d,
            conv_weight=conv_w_,
            a_log=a_log_,
            initial_state=h0_,
            output_final_state=True,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            chunk_size=64,
            segment_ids=segment_ids,
            implementation=impl,
        )
        loss = jnp.sum(out * do) + jnp.sum(ht * dht)
        return loss, (out, ht)

      return jax.value_and_grad(fn, argnums=tuple(range(6)), has_aux=True)(
          qkv, gate, beta, conv_w, a_log, h0
      )

    (loss_ref, (out_ref, ht_ref)), grads_ref = run("xla")
    (loss_mos, (out_mos, ht_mos)), grads_mos = run("mosaic")

    self.assertTrue(bool(jnp.all(jnp.isfinite(out_mos))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(ht_mos))))
    for g in grads_mos:
      self.assertTrue(bool(jnp.all(jnp.isfinite(g))))
    # Row 0 is entirely padding (<= 0): m_keep preserves h0[0] -> ht[0] and dh0[0] == dht[0].
    self.assertLess(float(jnp.max(jnp.abs(ht_mos[0] - h0[0]))), 1e-6)
    self.assertLess(float(jnp.max(jnp.abs(grads_mos[5][0] - dht[0]))), 1e-6)
    self.assertEqual(float(jnp.max(jnp.abs(out_mos[0]))), 0.0)
    self.assertEqual(float(jnp.max(jnp.abs(out_mos[1, -16:]))), 0.0)
    out_atol = 1e-3 if jax.default_backend() == "cpu" else 3e-3
    grad_atol = 2e-3 if jax.default_backend() == "cpu" else 5e-3
    self.assertLess(float(jnp.max(jnp.abs(out_mos - out_ref))), out_atol)
    self.assertLess(float(jnp.max(jnp.abs(ht_mos - ht_ref))), out_atol)
    self.assertAlmostEqual(float(loss_mos), float(loss_ref), places=2)
    for g_mos, g_ref in zip(grads_mos, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_mos - g_ref))), grad_atol)

  def test_gdn_packed_split_remat_no_conv_no_l2norm_matches_xla(self):
    b, t, h_qk, h_v, d = 2, 128, 1, 2, 128
    keys = jax.random.split(jax.random.PRNGKey(870), 7)
    q = jax.random.normal(keys[0], (b, t, h_qk, d), jnp.float32) * 0.4
    k = jax.random.normal(keys[1], (b, t, h_qk, d), jnp.float32) * 0.4
    v = jax.random.normal(keys[2], (b, t, h_v, d), jnp.float32) * 0.4
    gate = jax.random.normal(keys[3], (b, t, h_v), jnp.float32) * 0.4
    beta = jax.random.normal(keys[4], (b, t, h_v), jnp.float32)
    a_log = jnp.zeros((h_v,), jnp.float32)
    do = jax.random.normal(keys[5], (b, t, h_v, d), jnp.float32)
    segment_ids = jnp.asarray(
        [
            [1] * 30 + [0] * 10 + [1] * 40 + [2] * 48,
            [2] * 50 + [-1] * 14 + [3] * 64,
        ],
        dtype=jnp.int32,
    )

    gdn_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet(
        config=pallas_mosaic_tpu.Config(
            chunk_size=64, rematerialize_for_backward=True
        )
    )

    def run_mos(q_, k_, v_, gate_, beta_, a_log_):
      out, _ = gdn_op(
          q_,
          k_,
          v_,
          gate_,
          beta_,
          a_log=a_log_,
          use_qk_l2norm=False,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          segment_ids=segment_ids,
      )
      return jnp.sum(out * do), out

    def run_ref(q_, k_, v_, gate_, beta_, a_log_):
      out, _ = api.gated_delta_net(
          q_,
          k_,
          v_,
          gate_,
          beta_,
          a_log=a_log_,
          use_qk_l2norm=False,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          segment_ids=segment_ids,
          implementation="xla",
      )
      return jnp.sum(out * do), out

    args_all = (q, k, v, gate, beta, a_log)
    (loss_ref, out_ref), grads_ref = jax.value_and_grad(
        run_ref, argnums=tuple(range(6)), has_aux=True
    )(*args_all)
    (loss_mos, out_mos), grads_mos = jax.value_and_grad(
        run_mos, argnums=tuple(range(6)), has_aux=True
    )(*args_all)
    out_atol = 1e-3 if jax.default_backend() == "cpu" else 3e-3
    grad_atol = 2e-3 if jax.default_backend() == "cpu" else 2e-2
    loss_places = 2 if jax.default_backend() == "cpu" else 1
    self.assertLess(float(jnp.max(jnp.abs(out_mos - out_ref))), out_atol)
    self.assertAlmostEqual(
        float(loss_mos), float(loss_ref), places=loss_places
    )
    for g_mos, g_ref in zip(grads_mos, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_mos - g_ref))), grad_atol)

  def test_packed_bf16_two_pass_matches_xla(self):
    h, b, t, d = 2, 2, 128, 128
    keys = jax.random.split(jax.random.PRNGKey(880), 8)
    q = (jax.random.normal(keys[0], (h, b, t, d), jnp.float32) * 0.4).astype(
        jnp.bfloat16
    )
    k = (jax.random.normal(keys[1], (h, b, t, d), jnp.float32) * 0.4).astype(
        jnp.bfloat16
    )
    v = (jax.random.normal(keys[2], (h, b, t, d), jnp.float32) * 0.4).astype(
        jnp.bfloat16
    )
    g = (jax.random.normal(keys[3], (h, b, t, d), jnp.float32) * 0.4).astype(
        jnp.bfloat16
    )
    beta = jax.nn.sigmoid(
        jax.random.normal(keys[4], (h, b, t), jnp.float32)
    ).astype(jnp.bfloat16)
    a_log = jnp.zeros((h,), jnp.float32)
    do = (jax.random.normal(keys[5], (h, b, t, d), jnp.float32) * 0.2).astype(
        jnp.bfloat16
    )
    segment_ids = jnp.asarray(
        [
            [0] * 8 + [1] * 40 + [2] * 32 + [1] * 48,
            [1] * 60 + [-1] * 4 + [2] * 64,
        ],
        dtype=jnp.int32,
    )

    kda_op = pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention(
        config=pallas_mosaic_tpu.Config(
            chunk_size=64, two_pass_bf16=True, rematerialize_for_backward=True
        )
    )
    with mock.patch.object(
        pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention,
        "supported_on",
        return_value=True,
    ):
      def run_mos(q_, k_, v_, g_, beta_, a_log_):
        out, _ = kda_op(
            q_,
            k_,
            v_,
            g_,
            beta_,
            a_log=a_log_,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            per_channel_gate=True,
            lower_bound=-5.0,
            segment_ids=segment_ids,
            max_num_segments=2,
        )
        return jnp.sum(out.astype(jnp.float32) * do.astype(jnp.float32)), out

      def run_ref(q_, k_, v_, g_, beta_, a_log_):
        out, _ = api.kimi_delta_attention(
            q_,
            k_,
            v_,
            g_,
            beta_,
            a_log=a_log_,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            per_channel_gate=True,
            lower_bound=-5.0,
            segment_ids=segment_ids,
            max_num_segments=2,
            implementation="xla",
        )
        return jnp.sum(out * do.astype(jnp.float32)), out

      (loss_ref, out_ref), grads_ref = jax.value_and_grad(
          run_ref, argnums=tuple(range(6)), has_aux=True
      )(
          q.astype(jnp.float32),
          k.astype(jnp.float32),
          v.astype(jnp.float32),
          g.astype(jnp.float32),
          beta.astype(jnp.float32),
          a_log,
      )
      (loss_mos, out_mos), grads_mos = jax.value_and_grad(
          run_mos, argnums=tuple(range(6)), has_aux=True
      )(q, k, v, g, beta, a_log)

    self.assertEqual(out_mos.dtype, jnp.bfloat16)
    self.assertLess(
        float(jnp.max(jnp.abs(out_mos.astype(jnp.float32) - out_ref))), 5e-4
    )
    self.assertAlmostEqual(float(loss_mos), float(loss_ref), places=2)
    for g_mos, g_ref in zip(grads_mos, grads_ref):
      self.assertLess(
          float(jnp.max(jnp.abs(g_mos.astype(jnp.float32) - g_ref))), 1e-3
      )

    # Also verify GDN packed bf16 + two_pass_bf16=True.
    gdn_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet(
        config=pallas_mosaic_tpu.Config(chunk_size=64, two_pass_bf16=True)
    )
    q_g = q.transpose(1, 2, 0, 3)
    k_g = k.transpose(1, 2, 0, 3)
    v_g = v.transpose(1, 2, 0, 3)
    g_g = g[:, :, :, 0].transpose(1, 2, 0)
    b_g = beta.transpose(1, 2, 0)
    do_g = do.transpose(1, 2, 0, 3)

    def gdn_mos(q_, k_, v_, g_, b_):
      out, _ = gdn_op(
          q_,
          k_,
          v_,
          g_,
          b_,
          a_log=a_log,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          segment_ids=segment_ids,
      )
      return jnp.sum(out.astype(jnp.float32) * do_g.astype(jnp.float32)), out

    def gdn_ref(q_, k_, v_, g_, b_):
      out, _ = api.gated_delta_net(
          q_,
          k_,
          v_,
          g_,
          b_,
          a_log=a_log,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          segment_ids=segment_ids,
          implementation="xla",
      )
      return jnp.sum(out * do_g.astype(jnp.float32)), out

    (gl_ref, go_ref), gg_ref = jax.value_and_grad(
        gdn_ref, argnums=tuple(range(5)), has_aux=True
    )(
        q_g.astype(jnp.float32),
        k_g.astype(jnp.float32),
        v_g.astype(jnp.float32),
        g_g.astype(jnp.float32),
        b_g.astype(jnp.float32),
    )
    (gl_mos, go_mos), gg_mos = jax.value_and_grad(
        gdn_mos, argnums=tuple(range(5)), has_aux=True
    )(q_g, k_g, v_g, g_g, b_g)
    self.assertEqual(go_mos.dtype, jnp.bfloat16)
    self.assertLess(
        float(jnp.max(jnp.abs(go_mos.astype(jnp.float32) - go_ref))), 5e-4
    )
    self.assertAlmostEqual(float(gl_mos), float(gl_ref), places=2)
    for gm, gr in zip(gg_mos, gg_ref):
      self.assertLess(
          float(jnp.max(jnp.abs(gm.astype(jnp.float32) - gr))), 1e-3
      )


if __name__ == "__main__":
  absltest.main()
