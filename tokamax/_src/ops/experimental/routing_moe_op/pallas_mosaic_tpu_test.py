# Copyright 2026 Google LLC
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
"""Tests for the Pallas Mosaic TPU implementation of the Routing MoE op."""

import inspect

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
from tokamax._src.ops import op
from tokamax._src.ops.experimental.routing_moe_op import api
from tokamax._src.ops.experimental.routing_moe_op import arg_specs
from tokamax._src.ops.experimental.routing_moe_op import base
from tokamax._src.ops.experimental.routing_moe_op import pallas_mosaic_tpu
from tokamax._src.ops.experimental.routing_moe_op import reference

_EP = 8
_AXIS = "d"
_FP8 = jnp.float8_e4m3fn


def _quantize_per_channel(
    weights: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes `[E, K, N]` weights to FP8 with one `[E, 1, 1, N]` scale."""
  amax = jnp.max(jnp.abs(weights), axis=1)
  scale = (amax / float(jnp.finfo(_FP8).max)).astype(jnp.float32)
  return (weights / scale[:, None, :]).astype(_FP8), scale[:, None, None, :]


def _interleave_gate_up(weights: jax.Array, n_shards: int) -> jax.Array:
  """Reorders `[.., gate | up]` columns so each TP shard gets both halves."""
  gate, up = jnp.split(weights, 2, axis=-1)
  shards = weights.shape[:-1] + (n_shards, -1)
  interleaved = jnp.concat([gate.reshape(shards), up.reshape(shards)], axis=-1)
  return interleaved.reshape(weights.shape)


def _deinterleave_gate_up(w_interleaved: jax.Array, n_shards: int) -> jax.Array:
  """Restores global `[.., gate | up]` layout from TP-interleaved weights."""
  shards = w_interleaved.reshape(w_interleaved.shape[:-1] + (n_shards, 2, -1))
  gate = shards[..., 0, :].reshape(w_interleaved.shape[:-1] + (-1,))
  up = shards[..., 1, :].reshape(w_interleaved.shape[:-1] + (-1,))
  return jnp.concat([gate, up], axis=-1)


def _skip_unless_supported(
    test_case: absltest.TestCase,
    *,
    min_devices: int = 1,
    require_fp8: bool = False,
) -> None:
  """Skips `test_case` unless TPU backend, arch, and device count match."""
  if jax.default_backend() != "tpu":
    test_case.skipTest("Routing MoE requires TPU.")
  if require_fp8 and pltpu.get_tpu_info().generation < 7:
    test_case.skipTest(
        "FP8 weights require TPU generation >= 7 (kernel limitation)."
    )
  if jax.device_count() < min_devices:
    test_case.skipTest(
        f"Requires at least {min_devices} devices for the expert-parallel"
        f" mesh (found {jax.device_count()})."
    )


class PallasMosaicTpuRoutingMoeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    _skip_unless_supported(self)

  def _make_mesh(self, num_devices: int) -> Mesh:
    _skip_unless_supported(self, min_devices=num_devices)
    return Mesh(np.asarray(jax.devices()[:num_devices]), axis_names=(_AXIS,))

  def _inputs(self, *, mesh, tokens, hidden, inter, experts, quantized):
    k_x, k_w1, k_w2, k_logits = jax.random.split(jax.random.PRNGKey(42), 4)
    inputs = jax.random.normal(k_x, (tokens, hidden), jnp.float32) / 10
    w1 = jax.random.normal(k_w1, (experts, hidden, 2 * inter), jnp.float32)
    w2 = jax.random.normal(k_w2, (experts, inter, hidden), jnp.float32)
    w1, w2 = w1 / 10, w2 / 10
    router_logits = jax.random.normal(k_logits, (tokens, experts), jnp.float32)

    scores = jax.nn.softmax(router_logits, axis=-1)
    topk_weights, topk_indices = lax.top_k(scores, k=4)
    topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)

    args = dict(
        hidden_states=inputs.astype(jnp.bfloat16),
        topk_weights=topk_weights.astype(jnp.float32),
        topk_indices=topk_indices.astype(jnp.int32),
    )
    if quantized:
      args["w1"], args["w1_scale"] = _quantize_per_channel(w1)
      args["w2"], args["w2_scale"] = _quantize_per_channel(w2)
    else:
      args["w1"], args["w2"] = (
          w1.astype(jnp.bfloat16),
          w2.astype(jnp.bfloat16),
      )

    shard_act = NamedSharding(mesh, P(_AXIS))
    shard_w = NamedSharding(mesh, P(_AXIS, None, None))
    shard_scale = NamedSharding(mesh, P(_AXIS, None, None, None))
    sharded = {}
    for key, val in args.items():
      if key in ("hidden_states", "topk_weights", "topk_indices"):
        sharded[key] = jax.device_put(val, shard_act)
      elif key in ("w1", "w2"):
        sharded[key] = jax.device_put(val, shard_w)
      elif key in ("w1_scale", "w2_scale"):
        sharded[key] = jax.device_put(val, shard_scale)
      else:
        sharded[key] = val
    return sharded

  @parameterized.product(
      ep=(1, _EP),
      quantized=(False, True),
      remote_scatter=(False, True),
      use_sp=(True, False),
  )
  def test_matches_reference(self, ep, quantized, remote_scatter, use_sp):
    if ep == 1 and remote_scatter:
      self.skipTest("remote_scatter requires ep > 1.")
    _skip_unless_supported(self, min_devices=ep, require_fp8=quantized)
    mesh = Mesh(np.asarray(jax.devices()[:ep]), axis_names=(_AXIS,))
    args = self._inputs(
        mesh=mesh,
        tokens=256,
        hidden=512,
        inter=256,
        experts=64,
        quantized=quantized,
    )
    if not use_sp:
      shard_repl = NamedSharding(mesh, P(None))
      for key in ("hidden_states", "topk_weights", "topk_indices"):
        args[key] = jax.device_put(args[key], shard_repl)
    config = pallas_mosaic_tpu.Config(remote_scatter=remote_scatter)
    moe_op = pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe(config=config)
    out = moe_op(
        **args,
        activation="silu",
        mesh=mesh,
        ep_axis=_AXIS,
        use_sp=use_sp,
    )
    ref = reference.routing_moe_reference(
        **args,
        activation="silu",
    )

    self.assertEqual(out.shape, ref.shape)
    np_out = np.asarray(out, dtype=np.float64)
    np_ref = np.asarray(ref, dtype=np.float64)
    rel_l2 = float(np.linalg.norm(np_out - np_ref) / np.linalg.norm(np_ref))
    self.assertLess(rel_l2, 0.06 if quantized else 0.02)

  def test_bias_rejects_mosaic_and_falls_back_to_xla(self):
    mesh = self._make_mesh(1)
    args = self._inputs(
        mesh=mesh,
        tokens=64,
        hidden=256,
        inter=256,
        experts=16,
        quantized=False,
    )
    k_b1, k_b2 = jax.random.split(jax.random.PRNGKey(99), 2)
    w1_bias = jax.random.normal(k_b1, (16, 1, 512), jnp.bfloat16) / 10
    w2_bias = jax.random.normal(k_b2, (16, 1, 256), jnp.bfloat16) / 10
    moe_op = pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe()
    with self.assertRaisesRegex(NotImplementedError, "Bias"):
      moe_op(
          **args,
          w1_bias=w1_bias,
          mesh=mesh,
          ep_axis=_AXIS,
      )
    with self.assertRaisesRegex(NotImplementedError, "Bias"):
      api.routing_moe(
          **args,
          w1_bias=w1_bias,
          mesh=mesh,
          ep_axis=_AXIS,
          implementation="mosaic_tpu",
      )

    out = api.routing_moe(
        **args,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        mesh=mesh,
        ep_axis=_AXIS,
        use_sp=True,
        implementation=("mosaic_tpu", "xla"),
    )
    ref = reference.routing_moe_reference(
        **args,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )
    np_out = np.asarray(out, dtype=np.float64)
    np_ref = np.asarray(ref, dtype=np.float64)
    rel_l2 = float(np.linalg.norm(np_out - np_ref) / np.linalg.norm(np_ref))
    self.assertLess(rel_l2, 0.02)

  def test_fp8_dispatch_by_tpu_generation(self):
    mesh = self._make_mesh(1)
    args = self._inputs(
        mesh=mesh,
        tokens=128,
        hidden=512,
        inter=256,
        experts=16,
        quantized=True,
    )
    ref = reference.routing_moe_reference(**args, activation="silu")
    np_ref = np.asarray(ref, dtype=np.float64)

    if pltpu.get_tpu_info().generation < 7:
      with self.assertRaises(NotImplementedError):
        api.routing_moe(
            **args,
            mesh=mesh,
            ep_axis=_AXIS,
            use_sp=True,
            implementation="mosaic_tpu",
        )
      out = api.routing_moe(
          **args,
          mesh=mesh,
          ep_axis=_AXIS,
          use_sp=True,
          implementation=None,
      )
      np_out = np.asarray(out, dtype=np.float64)
      rel_l2 = float(np.linalg.norm(np_out - np_ref) / np.linalg.norm(np_ref))
      self.assertLess(rel_l2, 0.02)
    else:
      out_mosaic = api.routing_moe(
          **args,
          mesh=mesh,
          ep_axis=_AXIS,
          use_sp=True,
          implementation="mosaic_tpu",
      )
      out_default = api.routing_moe(
          **args,
          mesh=mesh,
          ep_axis=_AXIS,
          use_sp=True,
          implementation=None,
      )
      np.testing.assert_allclose(
          np.asarray(out_mosaic, dtype=np.float32),
          np.asarray(out_default, dtype=np.float32),
          rtol=1e-4,
          atol=1e-4,
      )
      np_out = np.asarray(out_default, dtype=np.float64)
      rel_l2 = float(np.linalg.norm(np_out - np_ref) / np.linalg.norm(np_ref))
      self.assertLess(rel_l2, 0.06)

  @parameterized.parameters((2, 2), (4, 2))
  def test_2d_mesh_ep_tp(self, ep_size, tp_size):
    num_devices = ep_size * tp_size
    _skip_unless_supported(self, min_devices=num_devices)
    self.assertGreaterEqual(jax.device_count(), num_devices)
    devices_2d = np.asarray(jax.devices()[:num_devices]).reshape(
        ep_size, tp_size
    )
    mesh_2d = Mesh(devices_2d, axis_names=("ep", "tp"))
    # Global inter=512 with tp=2 gives local inter=256 (kernel requires % 256).
    tokens, hidden, inter, experts = 256, 512, 512, 64

    k_x, k_w1, k_w2, k_logits = jax.random.split(jax.random.PRNGKey(42), 4)
    inputs = jax.random.normal(k_x, (tokens, hidden), jnp.float32) / 10
    w1 = jax.random.normal(k_w1, (experts, hidden, 2 * inter), jnp.float32) / 10
    w2 = jax.random.normal(k_w2, (experts, inter, hidden), jnp.float32) / 10
    router_logits = jax.random.normal(k_logits, (tokens, experts), jnp.float32)

    scores = jax.nn.softmax(router_logits, axis=-1)
    topk_weights, topk_indices = lax.top_k(scores, k=4)
    topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)

    w1_interleaved = _interleave_gate_up(w1.astype(jnp.bfloat16), tp_size)
    args = dict(
        hidden_states=jax.device_put(
            inputs.astype(jnp.bfloat16),
            NamedSharding(mesh_2d, P(("ep", "tp"))),
        ),
        topk_weights=jax.device_put(
            topk_weights.astype(jnp.float32),
            NamedSharding(mesh_2d, P(("ep", "tp"))),
        ),
        topk_indices=jax.device_put(
            topk_indices.astype(jnp.int32),
            NamedSharding(mesh_2d, P(("ep", "tp"))),
        ),
        w1=jax.device_put(
            w1_interleaved,
            NamedSharding(mesh_2d, P("ep", None, "tp")),
        ),
        w2=jax.device_put(
            w2.astype(jnp.bfloat16),
            NamedSharding(mesh_2d, P("ep", "tp", None)),
        ),
    )

    config = pallas_mosaic_tpu.Config(remote_scatter=False)
    moe_op = pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe(config=config)
    lowered = jax.jit(
        lambda kwargs: moe_op(
            **kwargs,
            mesh=mesh_2d,
            activation="silu",
            ep_axis="ep",
            tp_axis="tp",
            use_sp=True,
        )
    ).lower(args)
    self.assertIn("tpu_custom_call", lowered.as_text())

    out = api.routing_moe(
        **args,
        mesh=mesh_2d,
        activation="silu",
        ep_axis="ep",
        tp_axis="tp",
        use_sp=True,
        implementation="mosaic_tpu",
    )
    with self.assertRaises(NotImplementedError):
      api.routing_moe(
          **args,
          mesh=mesh_2d,
          activation="silu",
          ep_axis="ep",
          tp_axis="tp",
          use_sp=True,
          implementation="xla",
      )

    w1_ref = _deinterleave_gate_up(w1_interleaved, tp_size)
    ref = reference.routing_moe_reference(
        hidden_states=inputs.astype(jnp.bfloat16),
        topk_weights=topk_weights.astype(jnp.float32),
        topk_indices=topk_indices.astype(jnp.int32),
        w1=w1_ref,
        w2=w2.astype(jnp.bfloat16),
        activation="silu",
    )

    self.assertEqual(out.shape, ref.shape)
    np_out = np.asarray(out, dtype=np.float64)
    np_ref = np.asarray(ref, dtype=np.float64)
    rel_l2 = float(np.linalg.norm(np_out - np_ref) / np.linalg.norm(np_ref))
    self.assertLess(rel_l2, 0.02)


class HostTest(parameterized.TestCase):
  """Host-only tests for the routing MoE op."""

  def test_rejects_missing_mesh(self):
    tokens, hidden, inter, experts = 8, 16, 8, 4
    moe_op = pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe(
        bypass_device_check=True
    )
    with self.assertRaisesRegex(NotImplementedError, "mesh"):
      moe_op(
          hidden_states=jnp.zeros((tokens, hidden), jnp.bfloat16),
          topk_weights=jnp.zeros((tokens, 2), jnp.float32),
          topk_indices=jnp.zeros((tokens, 2), jnp.int32),
          w1=jnp.zeros((experts, hidden, 2 * inter), jnp.bfloat16),
          w2=jnp.zeros((experts, inter, hidden), jnp.bfloat16),
      )

  def test_not_supported_off_tpu(self):
    device = jax.devices()[0]
    if device.platform == "tpu":
      self.skipTest("Checks non-TPU devices.")
    self.assertFalse(
        pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe().supported_on(device)
    )


class AutotuningConfigsTest(absltest.TestCase):
  """Tests for autotuning config binding."""

  def test_all_specs_bind_and_produce_autotuning_configs(self):
    base_op = base.RoutingMoe()
    mosaic_op = pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe()
    bind_signature = inspect.signature(base_op.bind)
    self.assertNotEmpty(arg_specs.ARG_SPECS)
    for spec in arg_specs.ARG_SPECS:
      with self.subTest(spec.full_name):
        bind_signature.bind(**spec.args)
        bound = op.BoundArguments(op=mosaic_op, arguments=spec.args)
        configs = mosaic_op._get_autotuning_configs(bound)
        self.assertNotEmpty(configs)
        self.assertLessEqual(len(configs), 32)
        self.assertEqual(
            {config.remote_scatter for config in configs}, {False, True}
        )

  def test_autotuning_configs_filter_remote_scatter(self):
    mosaic_op = pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe()
    tokens, hidden, inter, experts = 8, 16, 8, 4
    hidden_states = jnp.zeros((tokens, hidden), jnp.bfloat16)
    topk_weights = jnp.zeros((tokens, 2), jnp.float32)
    topk_indices = jnp.zeros((tokens, 2), jnp.int32)
    w1 = jnp.zeros((experts, hidden, 2 * inter), jnp.bfloat16)
    w2 = jnp.zeros((experts, inter, hidden), jnp.bfloat16)

    cases = (
        (
            "missing_ep_axis",
            mosaic_op.bind(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_indices=topk_indices,
                w1=w1,
                w2=w2,
                ep_axis=None,
                tp_axis=None,
                reduce_output=True,
            ),
        ),
        (
            "with_tp_axis",
            mosaic_op.bind(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_indices=topk_indices,
                w1=w1,
                w2=w2,
                ep_axis="ep",
                tp_axis="tp",
                reduce_output=True,
            ),
        ),
        (
            "reduce_output_false",
            mosaic_op.bind(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_indices=topk_indices,
                w1=w1,
                w2=w2,
                ep_axis="ep",
                tp_axis=None,
                reduce_output=False,
            ),
        ),
    )

    for case_name, bound in cases:
      with self.subTest(case_name):
        configs = mosaic_op._get_autotuning_configs(bound)
        self.assertNotEmpty(configs)
        self.assertEqual({config.remote_scatter for config in configs}, {False})

    bound_allowed = mosaic_op.bind(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_indices=topk_indices,
        w1=w1,
        w2=w2,
        ep_axis="ep",
        tp_axis=None,
        reduce_output=True,
    )
    allowed_configs = mosaic_op._get_autotuning_configs(bound_allowed)
    self.assertEqual(
        {config.remote_scatter for config in allowed_configs}, {False, True}
    )


if __name__ == "__main__":
  absltest.main()
