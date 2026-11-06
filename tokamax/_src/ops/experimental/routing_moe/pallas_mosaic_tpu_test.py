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
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
from tokamax._src.ops import op
from tokamax._src.ops.experimental.routing_moe import api
from tokamax._src.ops.experimental.routing_moe import arg_specs
from tokamax._src.ops.experimental.routing_moe import base
from tokamax._src.ops.experimental.routing_moe import pallas_mosaic_tpu
from tokamax._src.ops.experimental.routing_moe import reference

_EP = 8
_AXIS = "d"
_FP8 = jnp.float8_e4m3fn


def _quantize_per_channel(weights: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Quantizes `[E, K, N]` weights to FP8 with one `[E, N]` scale per column."""
  amax = jnp.max(jnp.abs(weights), axis=1)
  scale = (amax / float(jnp.finfo(_FP8).max)).astype(jnp.float32)
  return (weights / scale[:, None, :]).astype(_FP8), scale


class PallasMosaicTpuRoutingMoeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Routing MoE requires TPU.")
    if jax.device_count() < _EP:
      self.skipTest(f"Requires {_EP} devices for the expert-parallel mesh.")
    self.mesh = Mesh(np.asarray(jax.devices()[:_EP]), axis_names=(_AXIS,))

  def _inputs(self, *, tokens, hidden, inter, experts, quantized):
    k_x, k_w1, k_w2, k_g = jax.random.split(jax.random.PRNGKey(42), 4)
    inputs = jax.random.normal(k_x, (tokens, hidden), jnp.float32) / 10
    w1 = jax.random.normal(k_w1, (experts, hidden, 2 * inter), jnp.float32)
    w2 = jax.random.normal(k_w2, (experts, inter, hidden), jnp.float32)
    w1, w2 = w1 / 10, w2 / 10
    gating = jax.random.normal(k_g, (tokens, experts), jnp.float32)
    args = dict(x=inputs.astype(jnp.bfloat16), gating=gating)
    if quantized:
      args["w1"], args["w1_scale"] = _quantize_per_channel(w1)
      args["w2"], args["w2_scale"] = _quantize_per_channel(w2)
    else:
      args["w1"], args["w2"] = w1.astype(jnp.bfloat16), w2.astype(jnp.bfloat16)
    shard = NamedSharding(self.mesh, P(_AXIS))
    return {k: jax.device_put(val, shard) for k, val in args.items()}

  @parameterized.product(
      quantized=(False, True),
      remote_scatter=(False, True),
      use_sp=(True, False),
  )
  def test_matches_reference(self, quantized, remote_scatter, use_sp):
    args = self._inputs(
        tokens=256, hidden=512, inter=256, experts=64, quantized=quantized
    )
    config = pallas_mosaic_tpu.Config(
        use_sp=use_sp, remote_scatter=remote_scatter
    )
    moe_op = pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe()
    out, _ = moe_op._fwd(
        **args,
        topk=4,
        renormalize=True,
        act_fn="silu",
        mesh=self.mesh,
        config=config,
    )
    ref = reference.routing_moe_reference(
        **args, topk=4, renormalize=True, act_fn="silu"
    )

    self.assertEqual(out.shape, ref.shape)
    np_out = np.asarray(out, dtype=np.float64)
    np_ref = np.asarray(ref, dtype=np.float64)
    rel_l2 = float(np.linalg.norm(np_out - np_ref) / np.linalg.norm(np_ref))
    # With FP8 weights routing gmm feeds the MXU FP8 x FP8 (its `matmul_dtype`
    # defaults to the weight dtype), quantizing activations on the fly, while
    # the reference dequantizes the weights and multiplies in float32. That
    # costs ~0.047 relative L2 here, independent of `use_sp` and
    # `remote_scatter`; routing MoE's own layer tests loosen to 5e-2 likewise.
    self.assertLess(rel_l2, 0.06 if quantized else 0.02)

  def test_rejects_bias(self):
    args = self._inputs(
        tokens=64, hidden=256, inter=128, experts=16, quantized=False
    )
    moe_op = pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe()
    with self.assertRaisesRegex(NotImplementedError, "biases"):
      moe_op._fwd(
          **args,
          w1_bias=jnp.zeros((16, 1, 256), jnp.bfloat16),
          topk=2,
          mesh=self.mesh,
      )

  def test_rejects_non_softmax_scoring(self):
    args = self._inputs(
        tokens=64, hidden=256, inter=128, experts=16, quantized=False
    )
    moe_op = pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe()
    with self.assertRaisesRegex(NotImplementedError, "softmax"):
      moe_op._fwd(
          **args,
          topk=2,
          scoring_fn="sigmoid",
          mesh=self.mesh,
      )

  def test_rejects_multi_axis_mesh(self):
    mesh = Mesh(
        np.asarray(jax.devices()[:_EP]).reshape(2, _EP // 2), ("ep", "tp")
    )
    args = self._inputs(
        tokens=64, hidden=256, inter=128, experts=16, quantized=False
    )
    moe_op = pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe()
    with self.assertRaisesRegex(NotImplementedError, "1-D"):
      moe_op._fwd(**args, topk=2, mesh=mesh)

  def test_api_matches_reference(self):
    args = self._inputs(
        tokens=256, hidden=512, inter=256, experts=64, quantized=False
    )
    out = api.routing_moe(
        **args, topk=4, mesh=self.mesh, implementation="mosaic_tpu"
    )
    ref = reference.routing_moe_reference(**args, topk=4)

    np_out = np.asarray(out, dtype=np.float64)
    np_ref = np.asarray(ref, dtype=np.float64)
    rel_l2 = float(np.linalg.norm(np_out - np_ref) / np.linalg.norm(np_ref))
    self.assertLess(rel_l2, 0.02)


class HostTest(parameterized.TestCase):
  """Checks that don't need a TPU, so they run on every backend."""

  @parameterized.parameters(
      ((4, 6), (4, 1, 1, 6)),
      ((4, 2, 6), (4, 2, 1, 6)),
      ((4, 2, 1, 6), (4, 2, 1, 6)),
  )
  def test_as_block_scale(self, shape, expected_shape):
    scale = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    block_scale = pallas_mosaic_tpu._as_block_scale(scale)
    assert block_scale is not None
    self.assertEqual(block_scale.shape, expected_shape)
    np.testing.assert_array_equal(
        np.asarray(block_scale).ravel(), np.asarray(scale).ravel()
    )

  def test_as_block_scale_passes_none(self):
    self.assertIsNone(pallas_mosaic_tpu._as_block_scale(None))

  def test_as_block_scale_rejects_other_ranks(self):
    with self.assertRaisesRegex(NotImplementedError, "weight scales"):
      pallas_mosaic_tpu._as_block_scale(jnp.zeros((4,), jnp.float32))

  def test_rejects_missing_mesh(self):
    tokens, hidden, inter, experts = 8, 16, 8, 4
    moe_op = pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe()
    with self.assertRaisesRegex(NotImplementedError, "expert-parallel mesh"):
      moe_op._fwd(
          x=jnp.zeros((tokens, hidden), jnp.bfloat16),
          w1=jnp.zeros((experts, hidden, 2 * inter), jnp.bfloat16),
          w2=jnp.zeros((experts, inter, hidden), jnp.bfloat16),
          gating=jnp.zeros((tokens, experts), jnp.float32),
          topk=2,
      )

  def test_not_supported_off_tpu(self):
    device = jax.devices()[0]
    if device.platform == "tpu":
      self.skipTest("Checks non-TPU devices.")
    self.assertFalse(
        pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe().supported_on(device)
    )


class AutotuningConfigsTest(absltest.TestCase):
  """Autotuning configs don't depend on the device, so these run anywhere."""

  def test_all_specs_bind_and_produce_autotuning_configs(self):
    base_op = base.RoutingMoe()
    mosaic_op = pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe()
    bind_signature = inspect.signature(base_op.bind)
    self.assertNotEmpty(arg_specs.ARG_SPECS)
    for spec in arg_specs.ARG_SPECS:
      with self.subTest(spec.full_name):
        # Raises if a spec argument isn't part of the `RoutingMoe` contract.
        bind_signature.bind(**spec.args)
        bound = op.BoundArguments(op=mosaic_op, arguments=spec.args)
        configs = mosaic_op._get_autotuning_configs(bound)
        self.assertNotEmpty(configs)
        self.assertLessEqual(len(configs), 32)
        self.assertTrue(
            all(isinstance(c, pallas_mosaic_tpu.Config) for c in configs)
        )
        self.assertEqual({c.remote_scatter for c in configs}, {False, True})
        self.assertEqual(configs, mosaic_op._get_autotuning_configs(bound))


if __name__ == "__main__":
  absltest.main()
