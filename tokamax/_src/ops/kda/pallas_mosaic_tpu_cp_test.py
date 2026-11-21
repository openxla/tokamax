# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Multi-device sequence-CP tests for KDA and GDN Pallas/Mosaic TPU kernels."""

import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
from tokamax._src import config
from tokamax._src.ops.kda import api
from tokamax._src.ops.kda import cp_utils
from tokamax._src.ops.kda import pallas_mosaic_tpu
from tokamax._src.ops.kda import pallas_mosaic_tpu_bwd_kernel
from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_kernel


def _count_primitive(jaxpr_or_closed, prim_name: str) -> int:
  jaxpr = getattr(jaxpr_or_closed, "jaxpr", jaxpr_or_closed)
  count = 0
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == prim_name:
      count += 1
    for param_val in eqn.params.values():
      if hasattr(param_val, "jaxpr") or hasattr(param_val, "eqns"):
        count += _count_primitive(param_val, prim_name)
  return count


class SequenceContextParallelMosaicTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() == "gpu":
      self.skipTest("Pallas Mosaic TPU kernel is not supported on GPU.")
    if len(jax.devices()) < 4:
      self.skipTest("Requires at least 4 JAX devices for cp_size=4 tests.")
    if jax.default_backend() == "cpu":
      self.enterContext(
          mock.patch.object(
              pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet,
              "supported_on",
              return_value=True,
          )
      )
      self.enterContext(
          mock.patch.object(
              pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention,
              "supported_on",
              return_value=True,
          )
      )
      self.enterContext(
          mock.patch(
              "jax._src.tpu_info.get_tpu_info",
              return_value=pltpu.get_tpu_info_for_chip(
                  pltpu.ChipVersion.TPU_7X, 1
              ),
          )
      )
    self.mesh = Mesh(np.asarray(jax.devices()[:4]), ("context",))

  @parameterized.parameters((64,), (128,))
  def test_gdn_seq_cp_unpacked_matches_xla(self, chunk_size: int):
    b, t_global, h_qk, h_v, d = 2, 512, 1, 2, 128
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(jax.random.PRNGKey(1000 + chunk_size), 8)
    qkv = jax.random.normal(keys[0], (b, t_global, c_qkv), jnp.float32) * 0.35
    conv_w = jax.random.normal(keys[1], (4, c_qkv), jnp.float32) * 0.15
    conv_b = jax.random.normal(keys[2], (c_qkv,), jnp.float32) * 0.05
    gate = jax.random.normal(keys[3], (b, t_global, h_v), jnp.float32) * 0.4
    beta = jax.random.normal(keys[4], (b, t_global, h_v), jnp.float32)
    a_log = jnp.log(jax.random.uniform(keys[5], (h_v,), minval=0.1, maxval=0.7))
    dt_bias = jax.random.normal(keys[6], (h_v,), jnp.float32) * 0.1
    do = jax.random.normal(keys[7], (b, t_global, h_v, d), jnp.float32)
    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=self.mesh, axis_name="context"
    )

    def ref_loss(qkv_, gate_, beta_, conv_w_, conv_b_, a_log_, dt_bias_):
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
          delta_time_bias=dt_bias_,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          chunk_size=chunk_size,
          implementation="xla",
      )
      return jnp.sum(out * do), out

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=(
            P(None, "context", None),
            P(None, "context", None),
            P(None, "context", None),
            P(),
            P(),
            P(),
            P(),
        ),
        out_specs=P(None, "context", None, None),
        check_vma=False,
    )
    def cp_fwd(qkv_, gate_, beta_, conv_w_, conv_b_, a_log_, dt_bias_):
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
          delta_time_bias=dt_bias_,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          chunk_size=chunk_size,
          context_parallel_metadata=cp_meta,
          implementation="mosaic",
      )
      return out

    def cp_loss(qkv_, gate_, beta_, conv_w_, conv_b_, a_log_, dt_bias_):
      out = cp_fwd(qkv_, gate_, beta_, conv_w_, conv_b_, a_log_, dt_bias_)
      return jnp.sum(out * do), out

    args = (qkv, gate, beta, conv_w, conv_b, a_log, dt_bias)
    (loss_ref, out_ref), grads_ref = jax.value_and_grad(
        ref_loss, argnums=tuple(range(7)), has_aux=True
    )(*args)
    with jax.set_mesh(self.mesh):
      (loss_cp, out_cp), grads_cp = jax.jit(
          jax.value_and_grad(cp_loss, argnums=tuple(range(7)), has_aux=True)
      )(*args)

    out_atol = 1e-3 if jax.default_backend() == "cpu" else 1e-2
    grad_atol = 2e-3 if jax.default_backend() == "cpu" else 2e-2
    loss_places = 2 if jax.default_backend() == "cpu" else 1
    self.assertLess(float(jnp.max(jnp.abs(out_cp - out_ref))), out_atol)
    self.assertAlmostEqual(float(loss_cp), float(loss_ref), places=loss_places)
    for g_cp, g_ref in zip(grads_cp, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_cp - g_ref))), grad_atol)

  @parameterized.parameters((64,), (128,))
  def test_gdn_seq_cp_packed_matches_xla(self, chunk_size: int):
    b, t_global, h_qk, h_v, d = 2, 512, 1, 2, 128
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(jax.random.PRNGKey(2000 + chunk_size), 8)
    qkv = jax.random.normal(keys[0], (b, t_global, c_qkv), jnp.float32) * 0.35
    conv_w = jax.random.normal(keys[1], (4, c_qkv), jnp.float32) * 0.15
    conv_b = jax.random.normal(keys[2], (c_qkv,), jnp.float32) * 0.05
    gate = jax.random.normal(keys[3], (b, t_global, h_v), jnp.float32) * 0.4
    beta = jax.random.normal(keys[4], (b, t_global, h_v), jnp.float32)
    a_log = jnp.log(jax.random.uniform(keys[5], (h_v,), minval=0.1, maxval=0.7))
    dt_bias = jax.random.normal(keys[6], (h_v,), jnp.float32) * 0.1
    do = jax.random.normal(keys[7], (b, t_global, h_v, d), jnp.float32)

    # 4 shards of 128 tokens each:
    # Row 0: [0..10 pad, 10..128 doc 5] | [all 128 zero pad!] |
    #        [0..50 doc 5 resumed across rank 1, 50..90 doc 2, 90..128 doc 7] |
    #        [0..70 doc 7 continued, 70..128 zero pad].
    seg_row0 = jnp.concatenate([
        jnp.zeros((10,), jnp.int32),
        jnp.full((118,), 5, jnp.int32),
        jnp.zeros((128,), jnp.int32),
        jnp.full((50,), 5, jnp.int32),
        jnp.full((40,), 2, jnp.int32),
        jnp.full((38,), 7, jnp.int32),
        jnp.full((70,), 7, jnp.int32),
        jnp.zeros((58,), jnp.int32),
    ])
    # Row 1: [0..80 doc 1, 80..128 doc 2] | [0..60 doc 1 reused across rank
    #        boundary -> MUST reset state & conv1d!, 60..128 doc 3] |
    #        [all 128 doc 3 spanning rank 2] | [0..45 doc 3, 45..128 doc 8].
    seg_row1 = jnp.concatenate([
        jnp.full((80,), 1, jnp.int32),
        jnp.full((48,), 2, jnp.int32),
        jnp.full((60,), 1, jnp.int32),
        jnp.full((68,), 3, jnp.int32),
        jnp.full((128,), 3, jnp.int32),
        jnp.full((45,), 3, jnp.int32),
        jnp.full((83,), 8, jnp.int32),
    ])
    segment_ids = jnp.stack([seg_row0, seg_row1], axis=0)
    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=self.mesh, axis_name="context"
    )

    def ref_loss(qkv_, gate_, beta_, conv_w_, conv_b_, a_log_, dt_bias_):
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
          delta_time_bias=dt_bias_,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          chunk_size=chunk_size,
          segment_ids=segment_ids,
          implementation="xla",
      )
      return jnp.sum(out * do), out

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=(
            P(None, "context", None),
            P(None, "context", None),
            P(None, "context", None),
            P(None, "context"),
            P(),
            P(),
            P(),
            P(),
        ),
        out_specs=P(None, "context", None, None),
        check_vma=False,
    )
    def cp_fwd(qkv_, gate_, beta_, seg_, conv_w_, conv_b_, a_log_, dt_bias_):
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
          delta_time_bias=dt_bias_,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          chunk_size=chunk_size,
          segment_ids=seg_,
          context_parallel_metadata=cp_meta,
          implementation="mosaic",
      )
      return out

    def cp_loss(qkv_, gate_, beta_, conv_w_, conv_b_, a_log_, dt_bias_):
      out = cp_fwd(
          qkv_, gate_, beta_, segment_ids, conv_w_, conv_b_, a_log_, dt_bias_
      )
      return jnp.sum(out * do), out

    args = (qkv, gate, beta, conv_w, conv_b, a_log, dt_bias)
    (loss_ref, out_ref), grads_ref = jax.value_and_grad(
        ref_loss, argnums=tuple(range(7)), has_aux=True
    )(*args)
    with jax.set_mesh(self.mesh):
      (loss_cp, out_cp), grads_cp = jax.jit(
          jax.value_and_grad(cp_loss, argnums=tuple(range(7)), has_aux=True)
      )(*args)

    pad_mask = segment_ids == 0
    self.assertEqual(
        float(
            jnp.max(
                jnp.abs(jnp.where(pad_mask[:, :, None, None], out_cp, 0.0))
            )
        ),
        0.0,
    )
    self.assertEqual(
        float(
            jnp.max(jnp.abs(jnp.where(pad_mask[:, :, None], grads_cp[0], 0.0)))
        ),
        0.0,
    )
    out_atol = 1e-3 if jax.default_backend() == "cpu" else 1e-2
    grad_atol = 2e-3 if jax.default_backend() == "cpu" else 2e-2
    loss_places = 2 if jax.default_backend() == "cpu" else 1
    self.assertLess(float(jnp.max(jnp.abs(out_cp - out_ref))), out_atol)
    self.assertAlmostEqual(float(loss_cp), float(loss_ref), places=loss_places)
    for g_cp, g_ref in zip(grads_cp, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_cp - g_ref))), grad_atol)

  @parameterized.parameters((64, False), (128, False), (64, True), (128, True))
  def test_kda_seq_cp_unpacked_and_packed_matches_xla(
      self, chunk_size: int, packed: bool
  ):
    h, b, t_global, d, w_conv = 2, 2, 512, 128, 4
    keys = jax.random.split(
        jax.random.PRNGKey(3000 + chunk_size + int(packed)), 15
    )
    q = jax.random.normal(keys[0], (h, b, t_global, d), jnp.float32) * 0.35
    k = jax.random.normal(keys[1], (h, b, t_global, d), jnp.float32) * 0.35
    v = jax.random.normal(keys[2], (h, b, t_global, d), jnp.float32) * 0.35
    g = jax.random.normal(keys[3], (h, b, t_global, d), jnp.float32) * 0.35
    beta = jax.nn.sigmoid(
        jax.random.normal(keys[4], (h, b, t_global), jnp.float32)
    )
    a_log = jnp.zeros((h,), jnp.float32)
    dt_bias = jax.random.normal(keys[5], (h * d,), jnp.float32) * 0.05
    cw_q = jax.random.normal(keys[6], (h, w_conv, d), jnp.float32) * 0.15
    cw_k = jax.random.normal(keys[7], (h, w_conv, d), jnp.float32) * 0.15
    cw_v = jax.random.normal(keys[8], (h, w_conv, d), jnp.float32) * 0.15
    cb_q = jax.random.normal(keys[9], (h, d), jnp.float32) * 0.05
    cb_k = jax.random.normal(keys[10], (h, d), jnp.float32) * 0.05
    cb_v = jax.random.normal(keys[11], (h, d), jnp.float32) * 0.05
    do = jax.random.normal(keys[12], (h, b, t_global, d), jnp.float32)

    if packed:
      seg_row0 = jnp.concatenate([
          jnp.zeros((8,), jnp.int32),
          jnp.full((120,), 1, jnp.int32),
          jnp.zeros((128,), jnp.int32),
          jnp.full((60,), 1, jnp.int32),
          jnp.full((68,), 2, jnp.int32),
          jnp.full((80,), 2, jnp.int32),
          jnp.full((48,), 3, jnp.int32),
      ])
      seg_row1 = jnp.concatenate([
          jnp.full((90,), 1, jnp.int32),
          jnp.full((166,), 2, jnp.int32),
          jnp.full((200,), 3, jnp.int32),
          jnp.zeros((56,), jnp.int32),
      ])
      segment_ids = jnp.stack([seg_row0, seg_row1], axis=0)
      max_num_segments = 4
    else:
      segment_ids = None
      max_num_segments = None

    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=self.mesh, axis_name="context"
    )

    def ref_loss(q_, k_, v_, g_, beta_, a_log_, dt_bias_, cw_q_, cw_k_, cw_v_):
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
          max_num_segments=max_num_segments,
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

    def cp_loss(q_, k_, v_, g_, beta_, a_log_, dt_bias_, cw_q_, cw_k_, cw_v_):
      if packed:
        @functools.partial(
            jax.shard_map,
            mesh=self.mesh,
            in_specs=(
                P(None, None, "context", None),
                P(None, None, "context", None),
                P(None, None, "context", None),
                P(None, None, "context", None),
                P(None, None, "context"),
                P(None, "context"),
                P(),
                P(),
                P(),
                P(),
                P(),
            ),
            out_specs=P(None, None, "context", None),
            check_vma=False,
        )
        def run_sharded(
            ql, kl, vl, gl, bl, segl, al, dtl, cwql, cwkl, cwvl
        ):
          out, _ = api.kimi_delta_attention(
              ql,
              kl,
              vl,
              gl,
              bl,
              a_log=al,
              delta_time_bias=dtl,
              use_qk_l2norm=True,
              use_gate_in_kernel=True,
              per_channel_gate=True,
              lower_bound=-5.0,
              chunk_size=chunk_size,
              segment_ids=segl,
              max_num_segments=max_num_segments,
              context_parallel_metadata=cp_meta,
              conv_weight_q=cwql,
              conv_weight_k=cwkl,
              conv_weight_v=cwvl,
              conv_bias_q=cb_q,
              conv_bias_k=cb_k,
              conv_bias_v=cb_v,
              use_conv1d_in_kernel=True,
              implementation="mosaic",
          )
          return out

        out = run_sharded(
            q_,
            k_,
            v_,
            g_,
            beta_,
            segment_ids,
            a_log_,
            dt_bias_,
            cw_q_,
            cw_k_,
            cw_v_,
        )
      else:
        @functools.partial(
            jax.shard_map,
            mesh=self.mesh,
            in_specs=(
                P(None, None, "context", None),
                P(None, None, "context", None),
                P(None, None, "context", None),
                P(None, None, "context", None),
                P(None, None, "context"),
                P(),
                P(),
                P(),
                P(),
                P(),
            ),
            out_specs=P(None, None, "context", None),
            check_vma=False,
        )
        def run_sharded_unpacked(ql, kl, vl, gl, bl, al, dtl, cwql, cwkl, cwvl):
          out, _ = api.kimi_delta_attention(
              ql,
              kl,
              vl,
              gl,
              bl,
              a_log=al,
              delta_time_bias=dtl,
              use_qk_l2norm=True,
              use_gate_in_kernel=True,
              per_channel_gate=True,
              lower_bound=-5.0,
              chunk_size=chunk_size,
              context_parallel_metadata=cp_meta,
              conv_weight_q=cwql,
              conv_weight_k=cwkl,
              conv_weight_v=cwvl,
              conv_bias_q=cb_q,
              conv_bias_k=cb_k,
              conv_bias_v=cb_v,
              use_conv1d_in_kernel=True,
              implementation="mosaic",
          )
          return out

        out = run_sharded_unpacked(
            q_, k_, v_, g_, beta_, a_log_, dt_bias_, cw_q_, cw_k_, cw_v_
        )
      return jnp.sum(out * do), out

    args = (q, k, v, g, beta, a_log, dt_bias, cw_q, cw_k, cw_v)
    (loss_ref, out_ref), grads_ref = jax.value_and_grad(
        ref_loss, argnums=tuple(range(10)), has_aux=True
    )(*args)
    with jax.set_mesh(self.mesh):
      (loss_cp, out_cp), grads_cp = jax.jit(
          jax.value_and_grad(cp_loss, argnums=tuple(range(10)), has_aux=True)
      )(*args)

    out_atol = 1e-3 if jax.default_backend() == "cpu" else 1e-2
    grad_atol = 2e-3 if jax.default_backend() == "cpu" else 2e-2
    loss_places = 2 if jax.default_backend() == "cpu" else 1
    self.assertLess(float(jnp.max(jnp.abs(out_cp - out_ref))), out_atol)
    self.assertAlmostEqual(float(loss_cp), float(loss_ref), places=loss_places)
    for g_cp, g_ref in zip(grads_cp, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_cp - g_ref))), grad_atol)

  def test_cp_residual_checkpoint_name_saves_boundary_collectives(self):
    b, t_global, h_qk, h_v, d = 1, 256, 1, 2, 128
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(jax.random.PRNGKey(4000), 5)
    qkv = jax.random.normal(keys[0], (b, t_global, c_qkv), jnp.float32) * 0.3
    conv_w = jax.random.normal(keys[1], (4, c_qkv), jnp.float32) * 0.15
    gate = jax.random.normal(keys[2], (b, t_global, h_v), jnp.float32) * 0.3
    beta = jax.random.normal(keys[3], (b, t_global, h_v), jnp.float32)
    a_log = jnp.zeros((h_v,), jnp.float32)

    cp_meta_tagged = cp_utils.ContextParallelMetadata(
        mesh=self.mesh,
        axis_name="context",
        cp_residual_checkpoint_name="cp_boundary",
    )

    def make_loss(policy):
      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=(
              P(None, "context", None),
              P(None, "context", None),
              P(None, "context", None),
              P(),
              P(),
          ),
          out_specs=P(None, "context", None, None),
          check_vma=False,
      )
      def sharded_fn(qkv_, gate_, beta_, conv_w_, a_log_):
        def step(x, g, bt, cw, al):
          out, _ = api.gated_delta_net(
              qkv=x,
              gate=g,
              beta=bt,
              num_k_heads=h_qk,
              num_v_heads=h_v,
              head_k_dim=d,
              head_v_dim=d,
              conv_weight=cw,
              a_log=al,
              use_qk_l2norm=True,
              use_gate_in_kernel=True,
              use_beta_sigmoid_in_kernel=True,
              chunk_size=64,
              context_parallel_metadata=cp_meta_tagged,
              implementation="mosaic",
          )
          return out

        return jax.checkpoint(step, policy=policy)(
            qkv_, gate_, beta_, conv_w_, a_log_
        )

      def loss(qkv_, gate_, beta_, conv_w_, a_log_):
        return jnp.sum(sharded_fn(qkv_, gate_, beta_, conv_w_, a_log_))

      return loss

    loss_save_cp = make_loss(
        jax.checkpoint_policies.save_only_these_names("cp_boundary")
    )
    loss_save_none = make_loss(jax.checkpoint_policies.nothing_saveable)

    with jax.set_mesh(self.mesh):
      jaxpr_save_cp = jax.make_jaxpr(jax.value_and_grad(loss_save_cp))(
          qkv, gate, beta, conv_w, a_log
      )
      jaxpr_save_none = jax.make_jaxpr(jax.value_and_grad(loss_save_none))(
          qkv, gate, beta, conv_w, a_log
      )
      val_cp, grad_cp = jax.jit(jax.value_and_grad(loss_save_cp))(
          qkv, gate, beta, conv_w, a_log
      )
      val_none, grad_none = jax.jit(jax.value_and_grad(loss_save_none))(
          qkv, gate, beta, conv_w, a_log
      )

    pperm_save_cp = _count_primitive(jaxpr_save_cp, "ppermute")
    pperm_save_none = _count_primitive(jaxpr_save_none, "ppermute")
    self.assertLess(pperm_save_cp, pperm_save_none)
    self.assertAlmostEqual(
        float(val_cp),
        float(val_none),
        places=3 if jax.default_backend() == "cpu" else 1,
    )
    grad_tol = 1e-4 if jax.default_backend() == "cpu" else 1e-2
    self.assertLess(float(jnp.max(jnp.abs(grad_cp - grad_none))), grad_tol)

  def test_gdn_cp_rematerialize_for_backward_avoids_extra_ppermutes(self):
    b, t_global, h_qk, h_v, d = 1, 256, 1, 2, 128
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(jax.random.PRNGKey(4500), 6)
    qkv = jax.random.normal(keys[0], (b, t_global, c_qkv), jnp.float32) * 0.3
    conv_w = jax.random.normal(keys[1], (4, c_qkv), jnp.float32) * 0.15
    gate = jax.random.normal(keys[2], (b, t_global, h_v), jnp.float32) * 0.3
    beta = jax.random.normal(keys[3], (b, t_global, h_v), jnp.float32)
    a_log = jnp.log(jnp.full((h_v,), 0.3, jnp.float32))
    do = jax.random.normal(keys[4], (b, t_global, h_v, d), jnp.float32)
    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=self.mesh, axis_name="context"
    )

    def make_loss(remat_bwd: bool):
      gdn_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet(
          config=pallas_mosaic_tpu.Config(
              chunk_size=64, rematerialize_for_backward=remat_bwd
          )
      )

      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=(
              P(None, "context", None),
              P(None, "context", None),
              P(None, "context", None),
              P(),
              P(),
          ),
          out_specs=P(None, "context", None, None),
          check_vma=False,
      )
      def sharded_fn(qkv_, gate_, beta_, conv_w_, a_log_):
        out, _ = gdn_op(
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
            context_parallel_metadata=cp_meta,
        )
        return out

      def loss(qkv_, gate_, beta_, conv_w_, a_log_):
        return jnp.sum(sharded_fn(qkv_, gate_, beta_, conv_w_, a_log_) * do)

      return loss

    loss_saved = make_loss(False)
    loss_remat = make_loss(True)
    with jax.set_mesh(self.mesh):
      jaxpr_saved = jax.make_jaxpr(jax.value_and_grad(loss_saved))(
          qkv, gate, beta, conv_w, a_log
      )
      jaxpr_remat = jax.make_jaxpr(jax.value_and_grad(loss_remat))(
          qkv, gate, beta, conv_w, a_log
      )
      val_saved, grad_saved = jax.jit(jax.value_and_grad(loss_saved))(
          qkv, gate, beta, conv_w, a_log
      )
      val_remat, grad_remat = jax.jit(jax.value_and_grad(loss_remat))(
          qkv, gate, beta, conv_w, a_log
      )

    self.assertEqual(
        _count_primitive(jaxpr_remat, "ppermute"),
        _count_primitive(jaxpr_saved, "ppermute"),
    )
    self.assertAlmostEqual(
        float(val_remat),
        float(val_saved),
        places=3 if jax.default_backend() == "cpu" else 1,
    )
    grad_tol = 1e-4 if jax.default_backend() == "cpu" else 1e-2
    self.assertLess(float(jnp.max(jnp.abs(grad_remat - grad_saved))), grad_tol)

  def test_gdn_cp_rejects_initial_state_and_output_final_state(self):
    b, t_local, h_qk, h_v, d = 1, 64, 1, 2, 128
    q = jnp.ones((b, t_local, h_qk, d), jnp.float32)
    k = jnp.ones((b, t_local, h_qk, d), jnp.float32)
    v = jnp.ones((b, t_local, h_v, d), jnp.float32)
    gate = jnp.zeros((b, t_local, h_v), jnp.float32)
    beta = jnp.zeros((b, t_local, h_v), jnp.float32)
    h0 = jnp.zeros((b, h_v, d, d), jnp.float32)
    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=self.mesh, axis_name="context"
    )
    with self.assertRaisesRegex(NotImplementedError, "initial_state"):
      api.gated_delta_net(
          q,
          k,
          v,
          gate,
          beta,
          initial_state=h0,
          context_parallel_metadata=cp_meta,
          implementation="mosaic",
      )
    with self.assertRaisesRegex(NotImplementedError, "output_final_state"):
      api.gated_delta_net(
          q,
          k,
          v,
          gate,
          beta,
          output_final_state=True,
          context_parallel_metadata=cp_meta,
          implementation="mosaic",
      )

  @parameterized.parameters((2, False), (2, True), (3, False), (3, True))
  def test_gdn_seq_cp_submesh_sizes_match_xla(
      self, cp_size: int, packed: bool
  ):
    submesh = Mesh(np.asarray(jax.devices()[:cp_size]), ("context",))
    b, t_local, h_qk, h_v, d = 2, 128, 1, 2, 128
    t_global = cp_size * t_local
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(
        jax.random.PRNGKey(5000 + 10 * cp_size + int(packed)), 8
    )
    qkv = jax.random.normal(keys[0], (b, t_global, c_qkv), jnp.float32) * 0.35
    conv_w = jax.random.normal(keys[1], (4, c_qkv), jnp.float32) * 0.15
    conv_b = jax.random.normal(keys[2], (c_qkv,), jnp.float32) * 0.05
    gate = jax.random.normal(keys[3], (b, t_global, h_v), jnp.float32) * 0.4
    beta = jax.random.normal(keys[4], (b, t_global, h_v), jnp.float32)
    a_log = jnp.log(jax.random.uniform(keys[5], (h_v,), minval=0.1, maxval=0.7))
    dt_bias = jax.random.normal(keys[6], (h_v,), jnp.float32) * 0.1
    do = jax.random.normal(keys[7], (b, t_global, h_v, d), jnp.float32)
    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=submesh, axis_name="context"
    )

    if packed:
      if cp_size == 2:
        seg_row0 = jnp.concatenate([
            jnp.zeros((16,), jnp.int32),
            jnp.full((84,), 1, jnp.int32),
            jnp.full((60,), 2, jnp.int32),
            jnp.full((60,), 1, jnp.int32),
            jnp.zeros((36,), jnp.int32),
        ])
        seg_row1 = jnp.concatenate([
            jnp.full((128,), 3, jnp.int32),
            jnp.full((80,), 3, jnp.int32),
            jnp.full((48,), 1, jnp.int32),
        ])
      else:
        seg_row0 = jnp.concatenate([
            jnp.zeros((20,), jnp.int32),
            jnp.full((108,), 1, jnp.int32),
            jnp.zeros((128,), jnp.int32),
            jnp.full((50,), 1, jnp.int32),
            jnp.full((40,), 2, jnp.int32),
            jnp.full((38,), 1, jnp.int32),
        ])
        seg_row1 = jnp.concatenate([
            jnp.full((70,), 2, jnp.int32),
            jnp.full((110,), 1, jnp.int32),
            jnp.full((130,), 2, jnp.int32),
            jnp.zeros((74,), jnp.int32),
        ])
      segment_ids = jnp.stack([seg_row0, seg_row1], axis=0)
    else:
      segment_ids = None

    def ref_loss(qkv_, gate_, beta_, conv_w_, conv_b_, a_log_, dt_bias_):
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
          delta_time_bias=dt_bias_,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          chunk_size=64,
          segment_ids=segment_ids,
          implementation="xla",
      )
      return jnp.sum(out * do), out

    def cp_loss(qkv_, gate_, beta_, conv_w_, conv_b_, a_log_, dt_bias_):
      if packed:
        @functools.partial(
            jax.shard_map,
            mesh=submesh,
            in_specs=(
                P(None, "context", None),
                P(None, "context", None),
                P(None, "context", None),
                P(None, "context"),
                P(),
                P(),
                P(),
                P(),
            ),
            out_specs=P(None, "context", None, None),
            check_vma=False,
        )
        def run_packed(qkv_l, gate_l, beta_l, seg_l, cw_l, cb_l, al_l, dt_l):
          out, _ = api.gated_delta_net(
              qkv=qkv_l,
              gate=gate_l,
              beta=beta_l,
              num_k_heads=h_qk,
              num_v_heads=h_v,
              head_k_dim=d,
              head_v_dim=d,
              conv_weight=cw_l,
              conv_bias=cb_l,
              a_log=al_l,
              delta_time_bias=dt_l,
              use_qk_l2norm=True,
              use_gate_in_kernel=True,
              use_beta_sigmoid_in_kernel=True,
              chunk_size=64,
              segment_ids=seg_l,
              context_parallel_metadata=cp_meta,
              implementation="mosaic",
          )
          return out

        out = run_packed(
            qkv_, gate_, beta_, segment_ids, conv_w_, conv_b_, a_log_, dt_bias_
        )
      else:
        @functools.partial(
            jax.shard_map,
            mesh=submesh,
            in_specs=(
                P(None, "context", None),
                P(None, "context", None),
                P(None, "context", None),
                P(),
                P(),
                P(),
                P(),
            ),
            out_specs=P(None, "context", None, None),
            check_vma=False,
        )
        def run_unpacked(qkv_l, gate_l, beta_l, cw_l, cb_l, al_l, dt_l):
          out, _ = api.gated_delta_net(
              qkv=qkv_l,
              gate=gate_l,
              beta=beta_l,
              num_k_heads=h_qk,
              num_v_heads=h_v,
              head_k_dim=d,
              head_v_dim=d,
              conv_weight=cw_l,
              conv_bias=cb_l,
              a_log=al_l,
              delta_time_bias=dt_l,
              use_qk_l2norm=True,
              use_gate_in_kernel=True,
              use_beta_sigmoid_in_kernel=True,
              chunk_size=64,
              context_parallel_metadata=cp_meta,
              implementation="mosaic",
          )
          return out

        out = run_unpacked(
            qkv_, gate_, beta_, conv_w_, conv_b_, a_log_, dt_bias_
        )
      return jnp.sum(out * do), out

    args = (qkv, gate, beta, conv_w, conv_b, a_log, dt_bias)
    (loss_ref, out_ref), grads_ref = jax.value_and_grad(
        ref_loss, argnums=tuple(range(7)), has_aux=True
    )(*args)
    with jax.set_mesh(submesh):
      (loss_cp, out_cp), grads_cp = jax.jit(
          jax.value_and_grad(cp_loss, argnums=tuple(range(7)), has_aux=True)
      )(*args)

    out_atol = 1e-3 if jax.default_backend() == "cpu" else 1e-2
    grad_atol = 2e-3 if jax.default_backend() == "cpu" else 2e-2
    loss_places = 2 if jax.default_backend() == "cpu" else 1
    self.assertLess(float(jnp.max(jnp.abs(out_cp - out_ref))), out_atol)
    self.assertAlmostEqual(float(loss_cp), float(loss_ref), places=loss_places)
    for g_cp, g_ref in zip(grads_cp, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_cp - g_ref))), grad_atol)

  @parameterized.parameters((64,), (128,))
  def test_kda_seq_cp_packed_reused_ids_and_gqa_conv_matches_xla(
      self, chunk_size: int
  ):
    h, h_qk, b, t_global, d, w_conv = 4, 2, 2, 512, 128, 4
    repeats = h // h_qk
    keys = jax.random.split(jax.random.PRNGKey(6000 + chunk_size), 14)
    q_gqa = (
        jax.random.normal(keys[0], (h_qk, b, t_global, d), jnp.float32) * 0.35
    )
    k_gqa = (
        jax.random.normal(keys[1], (h_qk, b, t_global, d), jnp.float32) * 0.35
    )
    v = jax.random.normal(keys[2], (h, b, t_global, d), jnp.float32) * 0.35
    g = jax.random.normal(keys[3], (h, b, t_global, d), jnp.float32) * 0.35
    beta = jax.nn.sigmoid(
        jax.random.normal(keys[4], (h, b, t_global), jnp.float32)
    )
    a_log = jnp.zeros((h,), jnp.float32)
    dt_bias = jax.random.normal(keys[5], (h * d,), jnp.float32) * 0.05
    cw_q = jax.random.normal(keys[6], (h_qk, w_conv, d), jnp.float32) * 0.15
    cw_k = jax.random.normal(keys[7], (h_qk, w_conv, d), jnp.float32) * 0.15
    cw_v = jax.random.normal(keys[8], (h, w_conv, d), jnp.float32) * 0.15
    cb_q = jax.random.normal(keys[9], (h_qk, d), jnp.float32) * 0.05
    cb_k = jax.random.normal(keys[10], (h_qk, d), jnp.float32) * 0.05
    cb_v = jax.random.normal(keys[11], (h, d), jnp.float32) * 0.05
    do = jax.random.normal(keys[12], (h, b, t_global, d), jnp.float32)

    # `[1, 2, 1]` reused document IDs with `max_num_segments=2` (canonical
    # spans = 3..4 > max_num_segments) plus an all-zero rank 1 on row 0.
    seg_row0 = jnp.concatenate([
        jnp.zeros((10,), jnp.int32),
        jnp.full((118,), 1, jnp.int32),
        jnp.zeros((128,), jnp.int32),
        jnp.full((50,), 1, jnp.int32),
        jnp.full((40,), 2, jnp.int32),
        jnp.full((38,), 1, jnp.int32),
        jnp.full((60,), 1, jnp.int32),
        jnp.zeros((68,), jnp.int32),
    ])
    seg_row1 = jnp.concatenate([
        jnp.full((80,), 1, jnp.int32),
        jnp.full((48,), 2, jnp.int32),
        jnp.full((70,), 1, jnp.int32),
        jnp.full((58,), 2, jnp.int32),
        jnp.full((128,), 2, jnp.int32),
        jnp.full((50,), 2, jnp.int32),
        jnp.full((78,), 1, jnp.int32),
    ])
    segment_ids = jnp.stack([seg_row0, seg_row1], axis=0)
    max_num_segments = 2
    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=self.mesh, axis_name="context"
    )

    def ref_loss(
        qg_, kg_, v_, g_, beta_, a_log_, dt_bias_, cw_q_, cw_k_, cw_v_
    ):
      q_ = jnp.repeat(qg_, repeats, axis=0)
      k_ = jnp.repeat(kg_, repeats, axis=0)
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
          max_num_segments=max_num_segments,
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

    def cp_loss(qg_, kg_, v_, g_, beta_, a_log_, dt_bias_, cw_q_, cw_k_, cw_v_):
      q_ = jnp.repeat(qg_, repeats, axis=0)
      k_ = jnp.repeat(kg_, repeats, axis=0)

      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=(
              P(None, None, "context", None),
              P(None, None, "context", None),
              P(None, None, "context", None),
              P(None, None, "context", None),
              P(None, None, "context"),
              P(None, "context"),
              P(),
              P(),
              P(),
              P(),
              P(),
          ),
          out_specs=P(None, None, "context", None),
          check_vma=False,
      )
      def run_sharded(ql, kl, vl, gl, bl, segl, al, dtl, cwql, cwkl, cwvl):
        out, _ = api.kimi_delta_attention(
            ql,
            kl,
            vl,
            gl,
            bl,
            a_log=al,
            delta_time_bias=dtl,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            per_channel_gate=True,
            lower_bound=-5.0,
            chunk_size=chunk_size,
            segment_ids=segl,
            max_num_segments=max_num_segments,
            context_parallel_metadata=cp_meta,
            conv_weight_q=cwql,
            conv_weight_k=cwkl,
            conv_weight_v=cwvl,
            conv_bias_q=cb_q,
            conv_bias_k=cb_k,
            conv_bias_v=cb_v,
            use_conv1d_in_kernel=True,
            implementation="mosaic",
        )
        return out

      out = run_sharded(
          q_,
          k_,
          v_,
          g_,
          beta_,
          segment_ids,
          a_log_,
          dt_bias_,
          cw_q_,
          cw_k_,
          cw_v_,
      )
      return jnp.sum(out * do), out

    args = (q_gqa, k_gqa, v, g, beta, a_log, dt_bias, cw_q, cw_k, cw_v)
    (loss_ref, out_ref), grads_ref = jax.value_and_grad(
        ref_loss, argnums=tuple(range(10)), has_aux=True
    )(*args)
    with jax.set_mesh(self.mesh):
      (loss_cp, out_cp), grads_cp = jax.jit(
          jax.value_and_grad(cp_loss, argnums=tuple(range(10)), has_aux=True)
      )(*args)

    out_atol = 1e-3 if jax.default_backend() == "cpu" else 1e-2
    grad_atol = 2e-3 if jax.default_backend() == "cpu" else 2e-2
    loss_places = 2 if jax.default_backend() == "cpu" else 1
    self.assertLess(float(jnp.max(jnp.abs(out_cp - out_ref))), out_atol)
    self.assertAlmostEqual(float(loss_cp), float(loss_ref), places=loss_places)
    for g_cp, g_ref in zip(grads_cp, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_cp - g_ref))), grad_atol)

  def test_packed_cp_remat_and_checkpoint_name_avoid_extra_ppermutes(self):
    b, t_global, d = 1, 256, 128
    segment_ids = jnp.concatenate([
        jnp.zeros((8,), jnp.int32),
        jnp.full((60,), 1, jnp.int32),
        jnp.full((60,), 2, jnp.int32),
        jnp.full((68,), 1, jnp.int32),
        jnp.zeros((60,), jnp.int32),
    ])[None, :]

    # 1. GDN packed CP: remat_bwd invariance + cp_residual_checkpoint_name.
    h_qk, h_v = 1, 2
    c_qkv = 2 * h_qk * d + h_v * d
    gdn_keys = jax.random.split(jax.random.PRNGKey(6500), 6)
    qkv = jax.random.normal(gdn_keys[0], (b, t_global, c_qkv), jnp.float32) * 0.3
    conv_w = jax.random.normal(gdn_keys[1], (4, c_qkv), jnp.float32) * 0.15
    gate = jax.random.normal(gdn_keys[2], (b, t_global, h_v), jnp.float32) * 0.3
    beta = jax.random.normal(gdn_keys[3], (b, t_global, h_v), jnp.float32)
    a_log = jnp.log(jnp.full((h_v,), 0.3, jnp.float32))
    do_gdn = jax.random.normal(gdn_keys[4], (b, t_global, h_v, d), jnp.float32)

    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=self.mesh, axis_name="context"
    )
    cp_meta_tagged = cp_utils.ContextParallelMetadata(
        mesh=self.mesh,
        axis_name="context",
        cp_residual_checkpoint_name="cp_boundary",
    )

    def make_gdn_remat_loss(remat_bwd: bool):
      gdn_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet(
          config=pallas_mosaic_tpu.Config(
              chunk_size=64, rematerialize_for_backward=remat_bwd
          )
      )

      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=(
              P(None, "context", None),
              P(None, "context", None),
              P(None, "context", None),
              P(None, "context"),
              P(),
              P(),
          ),
          out_specs=P(None, "context", None, None),
          check_vma=False,
      )
      def sharded_fn(qkv_, gate_, beta_, seg_, conv_w_, a_log_):
        out, _ = gdn_op(
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
            segment_ids=seg_,
            context_parallel_metadata=cp_meta,
        )
        return out

      return lambda qkv_, gate_, beta_, conv_w_, a_log_: jnp.sum(
          sharded_fn(qkv_, gate_, beta_, segment_ids, conv_w_, a_log_) * do_gdn
      )

    def make_gdn_ckpt_loss(
        policy,
        *,
        cp_meta_arg=cp_meta_tagged,
        res_ckpt_name: str | None = None,
    ):
      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=(
              P(None, "context", None),
              P(None, "context", None),
              P(None, "context", None),
              P(None, "context"),
              P(),
              P(),
          ),
          out_specs=P(None, "context", None, None),
          check_vma=False,
      )
      def sharded_fn(qkv_, gate_, beta_, seg_, conv_w_, a_log_):
        def step(x, g, bt, cw, al):
          out, _ = api.gated_delta_net(
              qkv=x,
              gate=g,
              beta=bt,
              num_k_heads=h_qk,
              num_v_heads=h_v,
              head_k_dim=d,
              head_v_dim=d,
              conv_weight=cw,
              a_log=al,
              use_qk_l2norm=True,
              use_gate_in_kernel=True,
              use_beta_sigmoid_in_kernel=True,
              chunk_size=64,
              segment_ids=seg_,
              context_parallel_metadata=cp_meta_arg,
              residual_checkpoint_name=res_ckpt_name,
              implementation="mosaic",
          )
          return out

        return jax.checkpoint(step, policy=policy)(
            qkv_, gate_, beta_, conv_w_, a_log_
        )

      return lambda qkv_, gate_, beta_, conv_w_, a_log_: jnp.sum(
          sharded_fn(qkv_, gate_, beta_, segment_ids, conv_w_, a_log_) * do_gdn
      )

    with jax.set_mesh(self.mesh):
      gdn_saved = make_gdn_remat_loss(False)
      gdn_remat = make_gdn_remat_loss(True)
      jaxpr_gdn_saved = jax.make_jaxpr(jax.value_and_grad(gdn_saved))(
          qkv, gate, beta, conv_w, a_log
      )
      jaxpr_gdn_remat = jax.make_jaxpr(jax.value_and_grad(gdn_remat))(
          qkv, gate, beta, conv_w, a_log
      )
      val_gdn_s, grad_gdn_s = jax.jit(jax.value_and_grad(gdn_saved))(
          qkv, gate, beta, conv_w, a_log
      )
      val_gdn_r, grad_gdn_r = jax.jit(jax.value_and_grad(gdn_remat))(
          qkv, gate, beta, conv_w, a_log
      )
      self.assertEqual(
          _count_primitive(jaxpr_gdn_remat, "ppermute"),
          _count_primitive(jaxpr_gdn_saved, "ppermute"),
      )
      self.assertAlmostEqual(float(val_gdn_r), float(val_gdn_s), places=4)
      self.assertLess(float(jnp.max(jnp.abs(grad_gdn_r - grad_gdn_s))), 1e-4)

      gdn_ckpt_cp = make_gdn_ckpt_loss(
          jax.checkpoint_policies.save_only_these_names("cp_boundary")
      )
      gdn_ckpt_none = make_gdn_ckpt_loss(
          jax.checkpoint_policies.nothing_saveable
      )
      gdn_ckpt_res_only = make_gdn_ckpt_loss(
          jax.checkpoint_policies.save_only_these_names("gdn_res_only"),
          cp_meta_arg=cp_meta,
          res_ckpt_name="gdn_res_only",
      )
      jaxpr_gdn_cp = jax.make_jaxpr(jax.value_and_grad(gdn_ckpt_cp))(
          qkv, gate, beta, conv_w, a_log
      )
      jaxpr_gdn_none = jax.make_jaxpr(jax.value_and_grad(gdn_ckpt_none))(
          qkv, gate, beta, conv_w, a_log
      )
      jaxpr_gdn_res_only = jax.make_jaxpr(
          jax.value_and_grad(gdn_ckpt_res_only)
      )(qkv, gate, beta, conv_w, a_log)
      self.assertLess(
          _count_primitive(jaxpr_gdn_cp, "ppermute"),
          _count_primitive(jaxpr_gdn_none, "ppermute"),
      )
      self.assertLess(
          _count_primitive(jaxpr_gdn_res_only, "ppermute"),
          _count_primitive(jaxpr_gdn_none, "ppermute"),
      )
      val_gdn_cp, grad_gdn_cp = jax.jit(jax.value_and_grad(gdn_ckpt_cp))(
          qkv, gate, beta, conv_w, a_log
      )
      val_gdn_none, grad_gdn_none = jax.jit(jax.value_and_grad(gdn_ckpt_none))(
          qkv, gate, beta, conv_w, a_log
      )
      val_gdn_ro, grad_gdn_ro = jax.jit(
          jax.value_and_grad(gdn_ckpt_res_only)
      )(qkv, gate, beta, conv_w, a_log)
      self.assertAlmostEqual(float(val_gdn_cp), float(val_gdn_s), places=4)
      self.assertAlmostEqual(float(val_gdn_none), float(val_gdn_s), places=4)
      self.assertAlmostEqual(float(val_gdn_ro), float(val_gdn_s), places=4)
      self.assertLess(float(jnp.max(jnp.abs(grad_gdn_cp - grad_gdn_s))), 1e-4)
      self.assertLess(
          float(jnp.max(jnp.abs(grad_gdn_none - grad_gdn_s))), 1e-4
      )
      self.assertLess(float(jnp.max(jnp.abs(grad_gdn_ro - grad_gdn_s))), 1e-4)

    # 2. KDA packed CP: remat_bwd invariance + cp_residual_checkpoint_name.
    h = 2
    kda_keys = jax.random.split(jax.random.PRNGKey(6600), 10)
    q_k = jax.random.normal(kda_keys[0], (h, b, t_global, d), jnp.float32) * 0.3
    k_k = jax.random.normal(kda_keys[1], (h, b, t_global, d), jnp.float32) * 0.3
    v_k = jax.random.normal(kda_keys[2], (h, b, t_global, d), jnp.float32) * 0.3
    g_k = jax.random.normal(kda_keys[3], (h, b, t_global, d), jnp.float32) * 0.3
    beta_k = jax.nn.sigmoid(
        jax.random.normal(kda_keys[4], (h, b, t_global), jnp.float32)
    )
    a_log_k = jnp.zeros((h,), jnp.float32)
    cw_q = jax.random.normal(kda_keys[5], (h, 4, d), jnp.float32) * 0.15
    cw_k = jax.random.normal(kda_keys[6], (h, 4, d), jnp.float32) * 0.15
    cw_v = jax.random.normal(kda_keys[7], (h, 4, d), jnp.float32) * 0.15
    do_kda = jax.random.normal(kda_keys[8], (h, b, t_global, d), jnp.float32)

    def make_kda_remat_loss(remat_bwd: bool):
      kda_op = pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention(
          config=pallas_mosaic_tpu.Config(
              chunk_size=64, rematerialize_for_backward=remat_bwd
          )
      )

      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=(
              P(None, None, "context", None),
              P(None, None, "context", None),
              P(None, None, "context", None),
              P(None, None, "context", None),
              P(None, None, "context"),
              P(None, "context"),
              P(),
              P(),
              P(),
              P(),
          ),
          out_specs=P(None, None, "context", None),
          check_vma=False,
      )
      def sharded_kda(ql, kl, vl, gl, bl, segl, al, cwql, cwkl, cwvl):
        out, _ = kda_op(
            ql,
            kl,
            vl,
            gl,
            bl,
            a_log=al,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            per_channel_gate=True,
            lower_bound=-5.0,
            segment_ids=segl,
            max_num_segments=2,
            context_parallel_metadata=cp_meta,
            conv_weight_q=cwql,
            conv_weight_k=cwkl,
            conv_weight_v=cwvl,
            use_conv1d_in_kernel=True,
        )
        return out

      return lambda ql, kl, vl, gl, bl: jnp.sum(
          sharded_kda(ql, kl, vl, gl, bl, segment_ids, a_log_k, cw_q, cw_k, cw_v)
          * do_kda
      )

    def make_kda_ckpt_loss(policy):
      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=(
              P(None, None, "context", None),
              P(None, None, "context", None),
              P(None, None, "context", None),
              P(None, None, "context", None),
              P(None, None, "context"),
              P(None, "context"),
              P(),
              P(),
              P(),
              P(),
          ),
          out_specs=P(None, None, "context", None),
          check_vma=False,
      )
      def sharded_kda(ql, kl, vl, gl, bl, segl, al, cwql, cwkl, cwvl):
        def step(qx, kx, vx, gx, bx):
          out, _ = api.kimi_delta_attention(
              qx,
              kx,
              vx,
              gx,
              bx,
              a_log=al,
              use_qk_l2norm=True,
              use_gate_in_kernel=True,
              per_channel_gate=True,
              lower_bound=-5.0,
              chunk_size=64,
              segment_ids=segl,
              max_num_segments=2,
              context_parallel_metadata=cp_meta_tagged,
              conv_weight_q=cwql,
              conv_weight_k=cwkl,
              conv_weight_v=cwvl,
              use_conv1d_in_kernel=True,
              implementation="mosaic",
          )
          return out

        return jax.checkpoint(step, policy=policy)(ql, kl, vl, gl, bl)

      return lambda ql, kl, vl, gl, bl: jnp.sum(
          sharded_kda(ql, kl, vl, gl, bl, segment_ids, a_log_k, cw_q, cw_k, cw_v)
          * do_kda
      )

    with jax.set_mesh(self.mesh):
      kda_saved = make_kda_remat_loss(False)
      kda_remat = make_kda_remat_loss(True)
      jaxpr_kda_s = jax.make_jaxpr(jax.value_and_grad(kda_saved))(
          q_k, k_k, v_k, g_k, beta_k
      )
      jaxpr_kda_r = jax.make_jaxpr(jax.value_and_grad(kda_remat))(
          q_k, k_k, v_k, g_k, beta_k
      )
      val_kda_s, grad_kda_s = jax.jit(jax.value_and_grad(kda_saved))(
          q_k, k_k, v_k, g_k, beta_k
      )
      val_kda_r, grad_kda_r = jax.jit(jax.value_and_grad(kda_remat))(
          q_k, k_k, v_k, g_k, beta_k
      )
      self.assertEqual(
          _count_primitive(jaxpr_kda_r, "ppermute"),
          _count_primitive(jaxpr_kda_s, "ppermute"),
      )

      kda_ckpt_cp = make_kda_ckpt_loss(
          jax.checkpoint_policies.save_only_these_names("cp_boundary")
      )
      kda_ckpt_none = make_kda_ckpt_loss(
          jax.checkpoint_policies.nothing_saveable
      )
      jaxpr_kda_cp = jax.make_jaxpr(jax.value_and_grad(kda_ckpt_cp))(
          q_k, k_k, v_k, g_k, beta_k
      )
      jaxpr_kda_none = jax.make_jaxpr(jax.value_and_grad(kda_ckpt_none))(
          q_k, k_k, v_k, g_k, beta_k
      )
      val_kda_cp, grad_kda_cp = jax.jit(jax.value_and_grad(kda_ckpt_cp))(
          q_k, k_k, v_k, g_k, beta_k
      )
      val_kda_none, grad_kda_none = jax.jit(jax.value_and_grad(kda_ckpt_none))(
          q_k, k_k, v_k, g_k, beta_k
      )
      self.assertLess(
          _count_primitive(jaxpr_kda_cp, "ppermute"),
          _count_primitive(jaxpr_kda_none, "ppermute"),
      )

    grad_tol = 1e-4 if jax.default_backend() == "cpu" else 1e-2
    places = 3 if jax.default_backend() == "cpu" else 1
    self.assertAlmostEqual(float(val_gdn_r), float(val_gdn_s), places=places)
    self.assertLess(
        float(jnp.max(jnp.abs(grad_gdn_r - grad_gdn_s))), grad_tol
    )
    self.assertAlmostEqual(float(val_kda_r), float(val_kda_s), places=places)
    self.assertLess(
        float(jnp.max(jnp.abs(grad_kda_r - grad_kda_s))), grad_tol
    )
    self.assertAlmostEqual(
        float(val_kda_cp), float(val_kda_none), places=places
    )
    self.assertLess(
        float(jnp.max(jnp.abs(grad_kda_cp - grad_kda_none))), grad_tol
    )

  def test_kda_cp_conv_width1_skips_halo_ppermutes_and_matches_xla(self):
    h, b, t_global, d = 2, 1, 256, 128
    keys = jax.random.split(jax.random.PRNGKey(6700), 10)
    q = jax.random.normal(keys[0], (h, b, t_global, d), jnp.float32) * 0.3
    k = jax.random.normal(keys[1], (h, b, t_global, d), jnp.float32) * 0.3
    v = jax.random.normal(keys[2], (h, b, t_global, d), jnp.float32) * 0.3
    g = jax.random.normal(keys[3], (h, b, t_global, d), jnp.float32) * 0.3
    beta = jax.nn.sigmoid(
        jax.random.normal(keys[4], (h, b, t_global), jnp.float32)
    )
    a_log = jnp.zeros((h,), jnp.float32)
    do = jax.random.normal(keys[5], (h, b, t_global, d), jnp.float32)
    segment_ids = jnp.ones((b, t_global), jnp.int32)
    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=self.mesh, axis_name="context"
    )

    def make_cp_loss(w_conv: int):
      cw_q = jax.random.normal(keys[6], (h, w_conv, d), jnp.float32) * 0.15
      cw_k = jax.random.normal(keys[7], (h, w_conv, d), jnp.float32) * 0.15
      cw_v = jax.random.normal(keys[8], (h, w_conv, d), jnp.float32) * 0.15

      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=(
              P(None, None, "context", None),
              P(None, None, "context", None),
              P(None, None, "context", None),
              P(None, None, "context", None),
              P(None, None, "context"),
              P(None, "context"),
          ),
          out_specs=P(None, None, "context", None),
          check_vma=False,
      )
      def run_cp(ql, kl, vl, gl, bl, segl):
        out, _ = api.kimi_delta_attention(
            ql,
            kl,
            vl,
            gl,
            bl,
            a_log=a_log,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            per_channel_gate=True,
            lower_bound=-5.0,
            chunk_size=64,
            segment_ids=segl,
            max_num_segments=2,
            context_parallel_metadata=cp_meta,
            conv_weight_q=cw_q,
            conv_weight_k=cw_k,
            conv_weight_v=cw_v,
            use_conv1d_in_kernel=True,
            implementation="mosaic",
        )
        return out

      def loss(ql, kl, vl, gl, bl):
        return jnp.sum(run_cp(ql, kl, vl, gl, bl, segment_ids) * do)

      return loss, (cw_q, cw_k, cw_v)

    loss_w1, (cw_q1, cw_k1, cw_v1) = make_cp_loss(1)
    loss_w4, _ = make_cp_loss(4)
    with jax.set_mesh(self.mesh):
      jaxpr_w1 = jax.make_jaxpr(jax.value_and_grad(loss_w1))(q, k, v, g, beta)
      jaxpr_w4 = jax.make_jaxpr(jax.value_and_grad(loss_w4))(q, k, v, g, beta)
      val_w1, grad_w1 = jax.jit(jax.value_and_grad(loss_w1))(q, k, v, g, beta)

    # W=1 skips 3 fwd conv halo + 3 bwd conv halo + 1 packed seg halo = 7 ppermutes.
    self.assertLessEqual(
        _count_primitive(jaxpr_w1, "ppermute"),
        _count_primitive(jaxpr_w4, "ppermute") - 7,
    )

    def ref_loss_w1(ql, kl, vl, gl, bl):
      out, _ = api.kimi_delta_attention(
          ql,
          kl,
          vl,
          gl,
          bl,
          a_log=a_log,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          per_channel_gate=True,
          lower_bound=-5.0,
          chunk_size=64,
          segment_ids=segment_ids,
          max_num_segments=2,
          conv_weight_q=cw_q1,
          conv_weight_k=cw_k1,
          conv_weight_v=cw_v1,
          use_conv1d_in_kernel=True,
          implementation="xla",
      )
      return jnp.sum(out * do)

    val_ref, grad_ref = jax.value_and_grad(ref_loss_w1)(q, k, v, g, beta)
    grad_atol = 2e-3 if jax.default_backend() == "cpu" else 2e-2
    self.assertAlmostEqual(
        float(val_w1),
        float(val_ref),
        places=2 if jax.default_backend() == "cpu" else 1,
    )
    self.assertLess(float(jnp.max(jnp.abs(grad_w1 - grad_ref))), grad_atol)

  def test_gdn_cp_split_qkv_bf16_and_hv16_chunk128_matches_xla(self):
    b, t_global, h_qk, h_v, d, chunk_size = 1, 512, 4, 16, 128, 128
    keys = jax.random.split(jax.random.PRNGKey(6800), 7)
    q = (
        jax.random.normal(keys[0], (b, t_global, h_qk, d), jnp.float32) * 0.25
        + 0.5
    ).astype(jnp.bfloat16)
    k = (
        jax.random.normal(keys[1], (b, t_global, h_qk, d), jnp.float32) * 0.25
        + 0.5
    ).astype(jnp.bfloat16)
    v = (
        jax.random.normal(keys[2], (b, t_global, h_v, d), jnp.float32) * 0.5
        + 0.5
    ).astype(jnp.bfloat16)
    gate = (
        jax.random.normal(keys[3], (b, t_global, h_v), jnp.float32) * 0.1
    ).astype(jnp.bfloat16)
    beta = jax.random.uniform(
        keys[4], (b, t_global, h_v), minval=1.0, maxval=6.0
    ).astype(jnp.bfloat16)
    a_log = jnp.log(jnp.full((h_v,), 0.02, jnp.float32))
    do = (
        jax.random.normal(keys[5], (b, t_global, h_v, d), jnp.float32) * 0.05
        + 0.03
    ).astype(jnp.bfloat16)
    segment_ids = jnp.concatenate([
        jnp.full((180,), 1, jnp.int32),
        jnp.full((180,), 2, jnp.int32),
        jnp.full((120,), 1, jnp.int32),
        jnp.zeros((32,), jnp.int32),
    ])[None, :]
    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=self.mesh, axis_name="context"
    )

    def ref_loss(qf, kf, vf, gf, bf):
      out, _ = api.gated_delta_net(
          qf,
          kf,
          vf,
          gf,
          bf,
          a_log=a_log,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          chunk_size=chunk_size,
          segment_ids=segment_ids,
          implementation="xla",
      )
      return jnp.sum(out * do.astype(jnp.float32)), out

    (loss_ref, out_ref), grads_ref = jax.value_and_grad(
        ref_loss, argnums=(0, 1, 2, 3, 4), has_aux=True
    )(
        q.astype(jnp.float32),
        k.astype(jnp.float32),
        v.astype(jnp.float32),
        gate.astype(jnp.float32),
        beta.astype(jnp.float32),
    )

    def run_bf16_cp(two_pass: bool):
      gdn_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet(
          config=pallas_mosaic_tpu.Config(
              chunk_size=chunk_size, two_pass_bf16=two_pass
          )
      )

      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=(
              P(None, "context", None, None),
              P(None, "context", None, None),
              P(None, "context", None, None),
              P(None, "context", None),
              P(None, "context", None),
              P(None, "context"),
          ),
          out_specs=P(None, "context", None, None),
          check_vma=False,
      )
      def sharded_fn(ql, kl, vl, gl, bl, segl):
        out, _ = gdn_op(
            ql,
            kl,
            vl,
            gl,
            bl,
            a_log=a_log,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            segment_ids=segl,
            context_parallel_metadata=cp_meta,
        )
        return out

      def loss(ql, kl, vl, gl, bl):
        out = sharded_fn(ql, kl, vl, gl, bl, segment_ids)
        return jnp.sum(out.astype(jnp.float32) * do.astype(jnp.float32)), out

      with jax.set_mesh(self.mesh):
        (loss_cp, out_cp), grads_cp = jax.jit(
            jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4), has_aux=True)
        )(q, k, v, gate, beta)
      return loss_cp, out_cp, grads_cp

    loss_1p, out_1p, grads_1p = run_bf16_cp(False)
    loss_2p, out_2p, grads_2p = run_bf16_cp(True)
    self.assertEqual(out_2p.dtype, jnp.bfloat16)
    self.assertEqual(grads_2p[4].dtype, jnp.bfloat16)

    err_loss_1p = abs(float(loss_1p) - float(loss_ref))
    err_loss_2p = abs(float(loss_2p) - float(loss_ref))
    err_out_1p = float(jnp.max(jnp.abs(out_1p.astype(jnp.float32) - out_ref)))
    err_out_2p = float(jnp.max(jnp.abs(out_2p.astype(jnp.float32) - out_ref)))
    err_grad_1p = max(
        float(jnp.max(jnp.abs(g_b.astype(jnp.float32) - g_r)))
        for g_b, g_r in zip(grads_1p, grads_ref)
    )
    err_grad_2p = max(
        float(jnp.max(jnp.abs(g_b.astype(jnp.float32) - g_r)))
        for g_b, g_r in zip(grads_2p, grads_ref)
    )
    self.assertLess(err_out_1p, 5e-2)
    self.assertLess(err_out_2p, 2e-2)
    self.assertLess(err_grad_1p, 1.5e-1)
    self.assertLess(err_grad_2p, 6e-2)
    if jax.default_backend() == "tpu":
      self.assertLess(err_out_2p, err_out_1p)
      self.assertLess(err_loss_2p, 0.5 * err_loss_1p)
      self.assertLess(err_grad_2p, 0.5 * err_grad_1p)

  def test_kda_cp_packed_rejects_unaligned_local_seq_len_and_autotune_cs(self):
    h, b, t_local, d = 2, 1, 96, 128
    q = jnp.ones((h, b, t_local, d), jnp.float32)
    k = jnp.ones((h, b, t_local, d), jnp.float32)
    v = jnp.ones((h, b, t_local, d), jnp.float32)
    g = jnp.zeros((h, b, t_local, d), jnp.float32)
    beta = jnp.zeros((h, b, t_local), jnp.float32)
    seg = jnp.ones((b, t_local), jnp.int32)
    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=self.mesh, axis_name="context"
    )
    with self.assertRaisesRegex(NotImplementedError, "divisible by"):
      api.kimi_delta_attention(
          q,
          k,
          v,
          g,
          beta,
          segment_ids=seg,
          max_num_segments=2,
          chunk_size=64,
          context_parallel_metadata=cp_meta,
          implementation="mosaic",
      )

    # Verify autotuning restricts chunk_size to (64, 128) under CP or packing.
    kda_op = pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention()
    ba_kda = kda_op.bind(
        jnp.ones((8, 1, 256, 128), jnp.bfloat16),
        jnp.ones((8, 1, 256, 128), jnp.bfloat16),
        jnp.ones((8, 1, 256, 128), jnp.bfloat16),
        jnp.zeros((8, 1, 256, 1), jnp.bfloat16),
        jnp.zeros((8, 1, 256), jnp.bfloat16),
        per_channel_gate=False,
        context_parallel_metadata=cp_meta,
    )
    kda_cs = {
        cfg.chunk_size
        for cfg in kda_op._get_autotuning_configs(ba_kda)  # pylint: disable=protected-access
    }
    self.assertEqual(kda_cs, {64, 128})

    gdn_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet()
    ba_gdn = gdn_op.bind(
        jnp.ones((1, 256, 4, 128), jnp.bfloat16),
        jnp.ones((1, 256, 4, 128), jnp.bfloat16),
        jnp.ones((1, 256, 8, 128), jnp.bfloat16),
        jnp.zeros((1, 256, 8), jnp.bfloat16),
        jnp.zeros((1, 256, 8), jnp.bfloat16),
        context_parallel_metadata=cp_meta,
    )
    gdn_cs = {
        cfg.chunk_size
        for cfg in gdn_op._get_autotuning_configs(ba_gdn)  # pylint: disable=protected-access
    }
    self.assertEqual(gdn_cs, {64, 128})

  @parameterized.parameters(
      (64, False, True),
      (128, False, True),
      (64, True, True),
      (128, True, True),
      (64, True, False),
  )
  def test_kda_seq_cp_scalar_gate_matches_xla(
      self, chunk_size: int, packed: bool, use_gate_in_kernel: bool
  ):
    h, b, t_global, d, w_conv = 2, 2, 512, 128, 4
    keys = jax.random.split(
        jax.random.PRNGKey(
            3500 + chunk_size + int(packed) + int(use_gate_in_kernel)
        ),
        15,
    )
    q = jax.random.normal(keys[0], (h, b, t_global, d), jnp.float32) * 0.35
    k = jax.random.normal(keys[1], (h, b, t_global, d), jnp.float32) * 0.35
    v = jax.random.normal(keys[2], (h, b, t_global, d), jnp.float32) * 0.35
    if use_gate_in_kernel:
      g = jax.random.normal(keys[3], (h, b, t_global, 1), jnp.float32) * 0.35
      a_log = jnp.log(
          jax.random.uniform(
              keys[5], (h,), jnp.float32, minval=1.0, maxval=16.0
          )
      )
      dt_bias = jax.random.normal(keys[6], (h,), jnp.float32) * 0.05
    else:
      g = jnp.full((h, b, t_global, 1), -77.0, jnp.float32)
      a_log = None
      dt_bias = None
    beta = jax.nn.sigmoid(
        jax.random.normal(keys[4], (h, b, t_global), jnp.float32)
    )
    cw_q = jax.random.normal(keys[7], (h, w_conv, d), jnp.float32) * 0.15
    cw_k = jax.random.normal(keys[8], (h, w_conv, d), jnp.float32) * 0.15
    cw_v = jax.random.normal(keys[9], (h, w_conv, d), jnp.float32) * 0.15
    cb_q = jax.random.normal(keys[10], (h, d), jnp.float32) * 0.05
    cb_k = jax.random.normal(keys[11], (h, d), jnp.float32) * 0.05
    cb_v = jax.random.normal(keys[12], (h, d), jnp.float32) * 0.05
    do = jax.random.normal(keys[13], (h, b, t_global, d), jnp.float32)

    if packed:
      seg_row0 = jnp.concatenate([
          jnp.zeros((8,), jnp.int32),
          jnp.full((120,), 1, jnp.int32),
          jnp.zeros((128,), jnp.int32),
          jnp.full((60,), 1, jnp.int32),
          jnp.full((68,), 2, jnp.int32),
          jnp.full((80,), 2, jnp.int32),
          jnp.full((48,), 3, jnp.int32),
      ])
      seg_row1 = jnp.concatenate([
          jnp.full((90,), 1, jnp.int32),
          jnp.full((166,), 2, jnp.int32),
          jnp.full((200,), 3, jnp.int32),
          jnp.zeros((56,), jnp.int32),
      ])
      segment_ids = jnp.stack([seg_row0, seg_row1], axis=0)
      max_num_segments = 4
    else:
      segment_ids = None
      max_num_segments = None

    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=self.mesh, axis_name="context"
    )

    def _call_kda(
        q_, k_, v_, g_, beta_, cw_q_, cw_k_, cw_v_, al_, dtl_, seg_, impl, cp
    ):
      kw = dict(
          a_log=al_,
          delta_time_bias=dtl_,
          use_qk_l2norm=True,
          use_gate_in_kernel=use_gate_in_kernel,
          per_channel_gate=False,
          lower_bound=None,
          chunk_size=chunk_size,
          segment_ids=seg_,
          max_num_segments=max_num_segments,
          conv_weight_q=cw_q_,
          conv_weight_k=cw_k_,
          conv_weight_v=cw_v_,
          conv_bias_q=cb_q,
          conv_bias_k=cb_k,
          conv_bias_v=cb_v,
          use_conv1d_in_kernel=True,
          implementation=impl,
      )
      if cp:
        kw["context_parallel_metadata"] = cp_meta
      out, _ = api.kimi_delta_attention(q_, k_, v_, g_, beta_, **kw)
      return out

    def ref_loss(q_, k_, v_, g_, beta_, cw_q_, cw_k_, cw_v_, *extra):
      al_ = extra[0] if use_gate_in_kernel else None
      dtl_ = extra[1] if use_gate_in_kernel else None
      out = _call_kda(
          q_,
          k_,
          v_,
          g_,
          beta_,
          cw_q_,
          cw_k_,
          cw_v_,
          al_,
          dtl_,
          segment_ids,
          "xla",
          False,
      )
      return jnp.sum(out * do), out

    def cp_loss(*args):
      in_specs = [
          P(None, None, "context", None),
          P(None, None, "context", None),
          P(None, None, "context", None),
          P(None, None, "context", None),
          P(None, None, "context"),
          P(),
          P(),
          P(),
      ]
      if use_gate_in_kernel:
        in_specs.extend([P(), P()])
      if packed:
        in_specs.append(P(None, "context"))

      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=tuple(in_specs),
          out_specs=P(None, None, "context", None),
          check_vma=False,
      )
      def run_sharded(ql, kl, vl, gl, bl, cwql, cwkl, cwvl, *extra):
        idx = 0
        al = extra[idx] if use_gate_in_kernel else None
        dtl = extra[idx + 1] if use_gate_in_kernel else None
        if use_gate_in_kernel:
          idx += 2
        segl = extra[idx] if packed else None
        return _call_kda(
            ql, kl, vl, gl, bl, cwql, cwkl, cwvl, al, dtl, segl, "mosaic", True
        )

      sharded_inputs = list(args) + ([segment_ids] if packed else [])
      out = run_sharded(*sharded_inputs)
      return jnp.sum(out * do), out

    diff_args = [q, k, v, g, beta, cw_q, cw_k, cw_v]
    if use_gate_in_kernel:
      assert a_log is not None and dt_bias is not None
      diff_args.extend([a_log, dt_bias])
    argnums = tuple(range(len(diff_args)))

    (loss_ref, out_ref), grads_ref = jax.value_and_grad(
        ref_loss, argnums=argnums, has_aux=True
    )(*diff_args)
    with jax.set_mesh(self.mesh):
      (loss_cp, out_cp), grads_cp = jax.jit(
          jax.value_and_grad(cp_loss, argnums=argnums, has_aux=True)
      )(*diff_args)

    for leaf in jax.tree.leaves((out_cp, grads_cp)):
      self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))
    out_atol = 1e-3 if jax.default_backend() == "cpu" else 1e-2
    grad_atol = 2e-3 if jax.default_backend() == "cpu" else 2e-2
    loss_places = 2 if jax.default_backend() == "cpu" else 1
    self.assertLess(float(jnp.max(jnp.abs(out_cp - out_ref))), out_atol)
    self.assertAlmostEqual(float(loss_cp), float(loss_ref), places=loss_places)
    for g_cp, g_ref in zip(grads_cp, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_cp - g_ref))), grad_atol)

  def test_cp_chain_precision_context_retraces(self):
    b, t_global, h_qk, h_v, d, chunk_size = 1, 512, 2, 2, 128, 64
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(jax.random.PRNGKey(7200), 8)
    qkv = (jax.random.normal(keys[0], (b, t_global, c_qkv)) * 0.35).astype(
        jnp.bfloat16
    )
    conv_w = (jax.random.normal(keys[1], (4, c_qkv)) * 0.15).astype(
        jnp.bfloat16
    )
    conv_b = (jax.random.normal(keys[2], (c_qkv,)) * 0.05).astype(jnp.bfloat16)
    gate = (jax.random.normal(keys[3], (b, t_global, h_v)) * 0.4).astype(
        jnp.bfloat16
    )
    beta = jax.random.normal(keys[4], (b, t_global, h_v)).astype(jnp.bfloat16)
    a_log = jnp.log(
        jax.random.uniform(keys[5], (h_v,), jnp.float32, minval=0.1, maxval=0.7)
    )
    dt_bias = jax.random.normal(keys[6], (h_v,), jnp.float32) * 0.1
    do = jax.random.normal(keys[7], (b, t_global, h_v, d), jnp.float32)
    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=self.mesh, axis_name="context"
    )
    gdn_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet(
        config=pallas_mosaic_tpu.Config(chunk_size=chunk_size)
    )

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=tuple([P(None, "context", None)] * 3 + [P()] * 4),
        out_specs=P(None, "context", None, None),
        check_vma=False,
    )
    def cp_fwd(qkv_, gate_, beta_, cw_, cb_, al_, dtb_):
      out, _ = gdn_op(
          qkv=qkv_,
          gate=gate_,
          beta=beta_,
          num_k_heads=h_qk,
          num_v_heads=h_v,
          head_k_dim=d,
          head_v_dim=d,
          conv_weight=cw_,
          conv_bias=cb_,
          a_log=al_,
          delta_time_bias=dtb_,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          context_parallel_metadata=cp_meta,
      )
      return out

    def cp_loss(*args):
      out = cp_fwd(*args)
      return jnp.sum(out.astype(jnp.float32) * do), out

    f_cp = jax.jit(
        jax.value_and_grad(cp_loss, argnums=tuple(range(7)), has_aux=True)
    )
    args = (qkv, gate, beta, conv_w, conv_b, a_log, dt_bias)

    seen = []

    def _wrap_chain_precision(mod):
      orig = mod.get_chain_precision

      def _spy():
        prec = orig()
        seen.append((mod.__name__.rsplit(".", 1)[-1], prec))
        return prec

      return mock.patch.object(mod, "get_chain_precision", _spy)

    with (
        _wrap_chain_precision(pallas_mosaic_tpu_fwd_kernel),
        _wrap_chain_precision(pallas_mosaic_tpu_bwd_kernel),
        jax.set_mesh(self.mesh),
    ):
      (_, out_a), g_a = f_cp(*args)
      n0 = len(seen)
      (_, _), _ = f_cp(*args)
      n1 = len(seen)
      with config.kda_chain_precision("default"):
        (_, out_d), g_d = f_cp(*args)
      n2 = len(seen)
      (_, out_b), g_b = f_cp(*args)
      n3 = len(seen)

    self.assertGreater(n0, 0)
    self.assertEqual(n1 - n0, 0)
    self.assertGreater(n2 - n1, 0)
    self.assertEqual(n3 - n2, 0)
    self.assertSetEqual(
        {mod_name for mod_name, _ in seen[:n0]},
        {"pallas_mosaic_tpu_fwd_kernel", "pallas_mosaic_tpu_bwd_kernel"},
    )
    self.assertSetEqual(
        {mod_name for mod_name, _ in seen[n1:n2]},
        {"pallas_mosaic_tpu_fwd_kernel", "pallas_mosaic_tpu_bwd_kernel"},
    )
    self.assertSetEqual(
        {prec for _, prec in seen[n1:n2]},
        {jax.lax.Precision.DEFAULT},
    )
    for x in jax.tree.leaves((out_d, g_d)):
      self.assertTrue(bool(jnp.all(jnp.isfinite(x.astype(jnp.float32)))))
    for a, b_ in zip(
        jax.tree.leaves((out_b, g_b)), jax.tree.leaves((out_a, g_a))
    ):
      self.assertEqual(float(jnp.max(jnp.abs(a - b_))), 0.0)

  @parameterized.parameters((64, False), (64, True), (128, False), (128, True))
  def test_gdn_bf16_cp_fused_qkv_matches_xla(
      self, chunk_size: int, two_pass_bf16: bool
  ):
    b, t_global, h_qk, h_v, d = 1, 512, 2, 4, 128
    c_qkv = 2 * h_qk * d + h_v * d
    keys = jax.random.split(
        jax.random.PRNGKey(7000 + chunk_size + int(two_pass_bf16)), 8
    )
    qkv = (jax.random.normal(keys[0], (b, t_global, c_qkv)) * 0.35).astype(
        jnp.bfloat16
    )
    conv_w = (jax.random.normal(keys[1], (4, c_qkv)) * 0.15).astype(
        jnp.bfloat16
    )
    conv_b = (jax.random.normal(keys[2], (c_qkv,)) * 0.05).astype(jnp.bfloat16)
    gate = (jax.random.normal(keys[3], (b, t_global, h_v)) * 0.4).astype(
        jnp.bfloat16
    )
    beta = jax.random.normal(keys[4], (b, t_global, h_v)).astype(jnp.bfloat16)
    a_log = jnp.log(
        jax.random.uniform(keys[5], (h_v,), jnp.float32, minval=0.1, maxval=0.7)
    )
    dt_bias = jax.random.normal(keys[6], (h_v,), jnp.float32) * 0.1
    do = jax.random.normal(keys[7], (b, t_global, h_v, d), jnp.float32)
    segment_ids = jnp.concatenate([
        jnp.full((10,), 0, jnp.int32),
        jnp.full((118,), 5, jnp.int32),
        jnp.full((128,), 0, jnp.int32),
        jnp.full((50,), 5, jnp.int32),
        jnp.full((40,), 2, jnp.int32),
        jnp.full((108,), 7, jnp.int32),
        jnp.full((58,), 0, jnp.int32),
    ])[None, :]
    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=self.mesh, axis_name="context"
    )
    gdn_op = pallas_mosaic_tpu.PallasMosaicTpuGatedDeltaNet(
        config=pallas_mosaic_tpu.Config(
            chunk_size=chunk_size, two_pass_bf16=two_pass_bf16
        )
    )

    def _ref_loss(qkv_, gate_, beta_, cw_, cb_, al_, dtb_):
      out, _ = api.gated_delta_net(
          qkv=qkv_,
          gate=gate_,
          beta=beta_,
          num_k_heads=h_qk,
          num_v_heads=h_v,
          head_k_dim=d,
          head_v_dim=d,
          conv_weight=cw_,
          conv_bias=cb_,
          a_log=al_,
          delta_time_bias=dtb_,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          chunk_size=chunk_size,
          segment_ids=segment_ids,
          implementation="xla",
      )
      return jnp.sum(out * do), out

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=tuple(
            [P(None, "context", None)] * 3 + [P()] * 4 + [P(None, "context")]
        ),
        out_specs=P(None, "context", None, None),
        check_vma=False,
    )
    def _cp_fwd(qkv_, gate_, beta_, cw_, cb_, al_, dtb_, segl_):
      out, _ = gdn_op(
          qkv=qkv_,
          gate=gate_,
          beta=beta_,
          num_k_heads=h_qk,
          num_v_heads=h_v,
          head_k_dim=d,
          head_v_dim=d,
          conv_weight=cw_,
          conv_bias=cb_,
          a_log=al_,
          delta_time_bias=dtb_,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True,
          segment_ids=segl_,
          context_parallel_metadata=cp_meta,
      )
      return out

    def _cp_loss(*args):
      out = _cp_fwd(*args, segment_ids)
      return jnp.sum(out.astype(jnp.float32) * do), out

    args = (qkv, gate, beta, conv_w, conv_b, a_log, dt_bias)
    ref_args = tuple(x.astype(jnp.float32) for x in args)
    (_, out_ref), grads_ref = jax.value_and_grad(
        _ref_loss, argnums=tuple(range(7)), has_aux=True
    )(*ref_args)
    with jax.set_mesh(self.mesh):
      (_, out_cp), grads_cp = jax.jit(
          jax.value_and_grad(_cp_loss, argnums=tuple(range(7)), has_aux=True)
      )(*args)

    def _rel_err(a, b_):
      a_f = jnp.asarray(a, jnp.float32)
      b_f = jnp.asarray(b_, jnp.float32)
      return float(jnp.linalg.norm(a_f - b_f) / (jnp.linalg.norm(b_f) + 1e-30))

    for leaf in jax.tree.leaves((out_cp, grads_cp)):
      self.assertTrue(bool(jnp.all(jnp.isfinite(leaf.astype(jnp.float32)))))
    self.assertLess(_rel_err(out_cp, out_ref), 5e-2)
    for g_cp, g_ref in zip(grads_cp, grads_ref):
      self.assertLess(_rel_err(g_cp, g_ref), 1e-1)

  @parameterized.parameters(
      (128, 128, 4, 16, "bfloat16", False, 16, 16),
      (128, 256, 4, 16, "bfloat16", False, 16, 8),
      (64, 128, 8, 32, "bfloat16", False, 32, 16),
      (128, 256, 4, 16, "float32", False, 8, 4),
  )
  def test_gdn_local_cp_bwd_summary_clamps_oversized_mini_batch(
      self,
      chunk_size: int,
      head_dim: int,
      num_k_heads: int,
      num_v_heads: int,
      dtype_name: str,
      two_pass_bf16: bool,
      requested_mb: int,
      expected_mb: int,
  ):
    dt = jnp.dtype(dtype_name)
    b, t_local = 1, chunk_size
    q_bt = jnp.zeros((b, t_local, num_k_heads, head_dim), dt)
    k_bt = jnp.zeros((b, t_local, num_k_heads, head_dim), dt)
    beta_hbt = jnp.zeros((num_v_heads, b, t_local), jnp.float32)
    g_hbt = jnp.zeros((num_v_heads, b, t_local), jnp.float32)
    do_bt = jnp.zeros((b, t_local, num_v_heads, head_dim), dt)
    captured_grids = []

    def _fake_pallas_call(unused_kernel, *, grid_spec, out_shape, **_):
      captured_grids.append(grid_spec.grid)
      return lambda *_args: jax.tree.map(
          lambda s: jnp.zeros(s.shape, s.dtype), out_shape
      )

    tpu7x_vmem_limit = int(64 * 1024 * 1024 * 0.9)
    with (
        mock.patch.object(
            pallas_mosaic_tpu_bwd_kernel,
            "get_vmem_limit_bytes",
            return_value=tpu7x_vmem_limit,
        ),
        mock.patch.object(
            pallas_mosaic_tpu_bwd_kernel,
            "get_max_mini_batch",
            side_effect=lambda chunk_size=None: 16 if chunk_size is None else 1,
        ),
        mock.patch.object(
            pallas_mosaic_tpu_bwd_kernel.pl, "pallas_call", _fake_pallas_call
        ),
    ):
      pallas_mosaic_tpu_bwd_kernel.compute_gdn_local_cp_bwd_summary(
          qkv_conv=None,
          q_bt=q_bt,
          k_bt=k_bt,
          beta_hbt=beta_hbt,
          g_hbt=g_hbt,
          do_bt=do_bt,
          akk=None,
          num_k_heads=num_k_heads,
          num_v_heads=num_v_heads,
          head_k_dim=head_dim,
          head_v_dim=head_dim,
          scale=1.0 / (head_dim**0.5),
          chunk_size=chunk_size,
          use_qk_l2norm=True,
          two_pass_bf16=two_pass_bf16,
          mini_batch=requested_mb,
      )

    self.assertLen(captured_grids, 1)
    self.assertEqual(captured_grids[0][0], num_v_heads // expected_mb)

  def test_kda_cp_k256_chunk64_f32_matches_xla(self):
    # On 4-chip TPU v7x, H=16 K=V=256 cs=64 f32 exceeds the 32 MiB default
    # scoped VMEM limit unless summary pallas_call sets vmem_limit_bytes.
    h = 2 if jax.default_backend() == "cpu" else 16
    b, t_global, d, cs = (
        1,
        (256 if jax.default_backend() == "cpu" else 512),
        256,
        64,
    )
    keys = jax.random.split(jax.random.PRNGKey(7500), 6)
    q = jax.random.normal(keys[0], (h, b, t_global, d), jnp.float32) * 0.35
    k = jax.random.normal(keys[1], (h, b, t_global, d), jnp.float32) * 0.35
    v = jax.random.normal(keys[2], (h, b, t_global, d), jnp.float32) * 0.35
    g = jax.random.normal(keys[3], (h, b, t_global, d), jnp.float32) * 0.35
    beta = jax.nn.sigmoid(
        jax.random.normal(keys[4], (h, b, t_global), jnp.float32)
    )
    a_log = jnp.zeros((h,), jnp.float32)
    do = jax.random.normal(keys[5], (h, b, t_global, d), jnp.float32)
    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=self.mesh, axis_name="context"
    )

    def _ref_loss(q_, k_, v_, g_, beta_):
      out, _ = api.kimi_delta_attention(
          q_,
          k_,
          v_,
          g_,
          beta_,
          a_log=a_log,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          per_channel_gate=True,
          lower_bound=-5.0,
          chunk_size=cs,
          implementation="xla",
      )
      return jnp.sum(out * do), out

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=tuple(
            [P(None, None, "context", None)] * 4 + [P(None, None, "context")]
        ),
        out_specs=P(None, None, "context", None),
        check_vma=False,
    )
    def _cp_fwd(ql, kl, vl, gl, bl):
      out, _ = api.kimi_delta_attention(
          ql,
          kl,
          vl,
          gl,
          bl,
          a_log=a_log,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          per_channel_gate=True,
          lower_bound=-5.0,
          chunk_size=cs,
          context_parallel_metadata=cp_meta,
          implementation="mosaic",
      )
      return out

    def _cp_loss(q_, k_, v_, g_, beta_):
      out = _cp_fwd(q_, k_, v_, g_, beta_)
      return jnp.sum(out * do), out

    args = (q, k, v, g, beta)
    (loss_ref, out_ref), grads_ref = jax.value_and_grad(
        _ref_loss, argnums=tuple(range(5)), has_aux=True
    )(*args)
    with jax.set_mesh(self.mesh):
      (loss_cp, out_cp), grads_cp = jax.jit(
          jax.value_and_grad(_cp_loss, argnums=tuple(range(5)), has_aux=True)
      )(*args)

    for leaf in jax.tree.leaves((out_cp, grads_cp)):
      self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))
    out_atol = 1e-3 if jax.default_backend() == "cpu" else 1e-2
    grad_atol = 2e-3 if jax.default_backend() == "cpu" else 2e-2
    loss_places = 2 if jax.default_backend() == "cpu" else 1
    self.assertLess(float(jnp.max(jnp.abs(out_cp - out_ref))), out_atol)
    self.assertAlmostEqual(float(loss_cp), float(loss_ref), places=loss_places)
    for g_cp, g_ref in zip(grads_cp, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_cp - g_ref))), grad_atol)

  def test_reference_kda_cp_conv1d_matches_unsharded(self):
    h, b, t_local, d, w_conv = 2, 2, 16, 32, 4
    t_global = t_local * self.mesh.shape["context"]
    keys = jax.random.split(jax.random.PRNGKey(7400), 10)
    q = jax.random.normal(keys[0], (h, b, t_global, d), jnp.float32) * 0.35
    k = jax.random.normal(keys[1], (h, b, t_global, d), jnp.float32) * 0.35
    v = jax.random.normal(keys[2], (h, b, t_global, d), jnp.float32) * 0.35
    g = jax.random.normal(keys[3], (h, b, t_global, d), jnp.float32) * 0.35
    beta = jax.nn.sigmoid(
        jax.random.normal(keys[4], (h, b, t_global), jnp.float32)
    )
    a_log = jnp.zeros((h,), jnp.float32)
    cw_q = jax.random.normal(keys[5], (h, w_conv, d), jnp.float32) * 0.15
    cw_k = jax.random.normal(keys[6], (h, w_conv, d), jnp.float32) * 0.15
    cw_v = jax.random.normal(keys[7], (h, w_conv, d), jnp.float32) * 0.15
    do = jax.random.normal(keys[8], (h, b, t_global, d), jnp.float32)
    # Arbitrary packed layout crossing rank boundaries with padding and reused IDs.
    seg_row0 = jnp.concatenate([
        jnp.full((12,), 5, jnp.int32),
        jnp.zeros((4,), jnp.int32),
        jnp.full((t_global - 16,), 5, jnp.int32),
    ])
    seg_row1 = jnp.concatenate([
        jnp.full((10,), 1, jnp.int32),
        jnp.full((10,), 2, jnp.int32),
        jnp.full((t_global - 20,), 1, jnp.int32),
    ])
    segment_ids = jnp.stack([seg_row0, seg_row1], axis=0)
    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=self.mesh, axis_name="context"
    )

    def ref_loss(q_, k_, v_, g_, beta_, cw_q_, cw_k_, cw_v_):
      out, _ = api.kimi_delta_attention(
          q_,
          k_,
          v_,
          g_,
          beta_,
          a_log=a_log,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          per_channel_gate=True,
          lower_bound=-5.0,
          segment_ids=segment_ids,
          max_num_segments=2,
          conv_weight_q=cw_q_,
          conv_weight_k=cw_k_,
          conv_weight_v=cw_v_,
          use_conv1d_in_kernel=True,
          implementation="xla",
      )
      return jnp.sum(out * do), out

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=tuple(
            [P(None, None, "context", None)] * 4
            + [P(None, None, "context"), P(None, "context")]
            + [P()] * 3
        ),
        out_specs=P(None, None, "context", None),
        check_vma=False,
    )
    def cp_fwd(ql, kl, vl, gl, bl, segl, cw_q_, cw_k_, cw_v_):
      out, _ = api.kimi_delta_attention(
          ql,
          kl,
          vl,
          gl,
          bl,
          a_log=a_log,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          per_channel_gate=True,
          lower_bound=-5.0,
          segment_ids=segl,
          max_num_segments=2,
          conv_weight_q=cw_q_,
          conv_weight_k=cw_k_,
          conv_weight_v=cw_v_,
          use_conv1d_in_kernel=True,
          context_parallel_metadata=cp_meta,
          implementation="xla",
      )
      return out

    def cp_loss(q_, k_, v_, g_, beta_, cw_q_, cw_k_, cw_v_):
      out = cp_fwd(q_, k_, v_, g_, beta_, segment_ids, cw_q_, cw_k_, cw_v_)
      return jnp.sum(out * do), out

    args = (q, k, v, g, beta, cw_q, cw_k, cw_v)
    (loss_ref, out_ref), grads_ref = jax.value_and_grad(
        ref_loss, argnums=tuple(range(8)), has_aux=True
    )(*args)
    with jax.set_mesh(self.mesh):
      (loss_cp, out_cp), grads_cp = jax.jit(
          jax.value_and_grad(cp_loss, argnums=tuple(range(8)), has_aux=True)
      )(*args)

    self.assertLess(float(jnp.max(jnp.abs(out_cp - out_ref))), 1e-5)
    self.assertAlmostEqual(float(loss_cp), float(loss_ref), places=4)
    for g_cp, g_ref in zip(grads_cp, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_cp - g_ref))), 1e-4)

  def test_kda_unpacked_cp_remat_matches_xla(self):
    h, b, t_local, d = 2, 2, 128, 128
    t_global = t_local * self.mesh.shape["context"]
    keys = jax.random.split(jax.random.PRNGKey(7500), 6)
    q = jax.random.normal(keys[0], (h, b, t_global, d), jnp.float32) * 0.35
    k = jax.random.normal(keys[1], (h, b, t_global, d), jnp.float32) * 0.35
    v = jax.random.normal(keys[2], (h, b, t_global, d), jnp.float32) * 0.35
    g = jax.random.normal(keys[3], (h, b, t_global, d), jnp.float32) * 0.35
    beta = jax.nn.sigmoid(
        jax.random.normal(keys[4], (h, b, t_global), jnp.float32)
    )
    a_log = jnp.zeros((h,), jnp.float32)
    do = jax.random.normal(keys[5], (h, b, t_global, d), jnp.float32)
    cp_meta = cp_utils.ContextParallelMetadata(
        mesh=self.mesh, axis_name="context"
    )
    kda_op = pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention(
        config=pallas_mosaic_tpu.Config(
            chunk_size=64, rematerialize_for_backward=True
        )
    )

    def ref_loss(q_, k_, v_, g_, beta_):
      out, _ = api.kimi_delta_attention(
          q_,
          k_,
          v_,
          g_,
          beta_,
          a_log=a_log,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          per_channel_gate=True,
          lower_bound=-5.0,
          chunk_size=64,
          implementation="xla",
      )
      return jnp.sum(out * do), out

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=tuple(
            [P(None, None, "context", None)] * 4 + [P(None, None, "context")]
        ),
        out_specs=P(None, None, "context", None),
        check_vma=False,
    )
    def cp_fwd(ql, kl, vl, gl, bl):
      out, _ = kda_op(
          ql,
          kl,
          vl,
          gl,
          bl,
          a_log=a_log,
          use_qk_l2norm=True,
          use_gate_in_kernel=True,
          per_channel_gate=True,
          lower_bound=-5.0,
          context_parallel_metadata=cp_meta,
      )
      return out

    def cp_loss(q_, k_, v_, g_, beta_):
      out = cp_fwd(q_, k_, v_, g_, beta_)
      return jnp.sum(out * do), out

    args = (q, k, v, g, beta)
    (loss_ref, out_ref), grads_ref = jax.value_and_grad(
        ref_loss, argnums=tuple(range(5)), has_aux=True
    )(*args)
    with jax.set_mesh(self.mesh):
      (loss_cp, out_cp), grads_cp = jax.jit(
          jax.value_and_grad(cp_loss, argnums=tuple(range(5)), has_aux=True)
      )(*args)

    out_atol = 1e-3 if jax.default_backend() == "cpu" else 1e-2
    grad_atol = 2e-3 if jax.default_backend() == "cpu" else 2e-2
    loss_places = 2 if jax.default_backend() == "cpu" else 1
    self.assertLess(float(jnp.max(jnp.abs(out_cp - out_ref))), out_atol)
    self.assertAlmostEqual(float(loss_cp), float(loss_ref), places=loss_places)
    for g_cp, g_ref in zip(grads_cp, grads_ref):
      self.assertLess(float(jnp.max(jnp.abs(g_cp - g_ref))), grad_atol)


if __name__ == "__main__":
  absltest.main()
