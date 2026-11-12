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
"""Unit tests for PallasMosaicTpuFusedMoeDecode operator."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import backend
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from tokamax._src.ops.experimental.fused_moe import reference
from tokamax._src.ops.experimental.fused_moe_decode import pallas_mosaic_tpu

F8 = jnp.float8_e4m3fn


def _make_single_rank_inputs(
    tokens: int = 16,
    h: int = 512,
    inter: int = 256,
    inter_s: int = 128,
    experts: int = 8,
    topk: int = 2,
    shared_gated: bool = True,
    seed: int = 42,
):
  rng = np.random.default_rng(seed)

  def quant(w, axis):
    s = np.max(np.abs(w), axis=axis, keepdims=True) / 448.0
    return (
        jnp.asarray(w / s, dtype=F8),
        jnp.asarray(s, dtype=jnp.float32),
    )

  hidden = jnp.asarray(rng.normal(size=(tokens, h)), dtype=jnp.bfloat16)
  w1, w1_scale = quant(rng.normal(size=(experts, h, 2 * inter)) * 0.02, 1)
  w2, w2_scale = quant(rng.normal(size=(experts, inter, h)) * 0.02, 1)
  shared_w1, shared_w1_scale = quant(
      rng.normal(size=(h, 2 * inter_s)) * 0.02, 0
  )
  shared_w2, shared_w2_scale = quant(rng.normal(size=(inter_s, h)) * 0.02, 0)
  shared_gate_weight = (
      jnp.asarray(rng.normal(size=(h, 1)) * 0.02, dtype=jnp.bfloat16)
      if shared_gated
      else None
  )
  topk_weights = jnp.asarray(
      rng.dirichlet(np.ones(topk), size=tokens), dtype=jnp.bfloat16
  )
  topk_ids = jnp.asarray(
      np.stack(
          [rng.choice(experts, topk, replace=False) for _ in range(tokens)]
      ),
      dtype=jnp.int32,
  )
  rank = jnp.array([0], dtype=jnp.int32)
  return (
      rank,
      hidden,
      w1,
      w2,
      w1_scale[:, :, None, :],
      w2_scale[:, :, None, :],
      shared_w1,
      shared_w1_scale,
      shared_w2,
      shared_w2_scale,
      shared_gate_weight,
      topk_weights,
      topk_ids,
  )


def _call_ref(
    rank,
    hidden,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    topk_weights,
    topk_ids,
    **kwargs,
):
  return reference.fused_moe_reference(
      x=hidden,
      w1=w1,
      w2=w2,
      w1_scale=w1_scale,
      w2_scale=w2_scale,
      topk_weights=topk_weights,
      topk_ids=topk_ids,
      shared_w1=shared_w1,
      shared_w1_scale=shared_w1_scale,
      shared_w2=shared_w2,
      shared_w2_scale=shared_w2_scale,
      shared_gate_weight=shared_gate_weight,
      rank=rank,
      **kwargs,
  )


def _max_rel(a: np.ndarray, ref: np.ndarray) -> float:
  return float(
      (
          np.linalg.norm(a - ref, axis=1)
          / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
      ).max()
  )


class PallasMosaicTpuFusedMoeDecodeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    op = pallas_mosaic_tpu.PallasMosaicTpuFusedMoeDecode()
    device = backend.get_default_device()
    if not op.supported_on(device):
      self.skipTest(
          "PallasMosaicTpuFusedMoeDecode not supported on"
          f" {device.device_kind}."
      )

  @parameterized.product(
      shared_gated=[True, False],
      routed_scale=[1.0, 2.5],
      i_block=[None, 128],
  )
  def test_single_rank_matches_reference(
      self, shared_gated, routed_scale, i_block
  ):
    inputs = _make_single_rank_inputs(shared_gated=shared_gated)
    config = pallas_mosaic_tpu.Config(i_block=i_block)
    op = pallas_mosaic_tpu.PallasMosaicTpuFusedMoeDecode(config=config)
    got = np.asarray(
        op(*inputs, n_ranks=1, routed_scale=routed_scale), dtype=np.float32
    )
    want = np.asarray(
        _call_ref(*inputs, n_ranks=1, routed_scale=routed_scale),
        dtype=np.float32,
    )
    self.assertTrue(np.isfinite(got).all())
    self.assertLess(_max_rel(got, want), 0.08)

  def test_single_rank_custom_config_matches_reference(self):
    inputs = _make_single_rank_inputs(shared_gated=True)
    config = pallas_mosaic_tpu.Config(n_chunk=256, num_buffers=4, q_block=256)
    op = pallas_mosaic_tpu.PallasMosaicTpuFusedMoeDecode(config=config)
    got = np.asarray(op(*inputs, n_ranks=1), dtype=np.float32)
    want = np.asarray(
        _call_ref(*inputs, n_ranks=1),
        dtype=np.float32,
    )
    self.assertTrue(np.isfinite(got).all())
    self.assertLess(_max_rel(got, want), 0.08)

  def test_eight_rank_shard_map_matches_reference(self):
    if jax.device_count() < 8:
      self.skipTest("Requires 8 TPU devices for 8-rank shard_map test.")
    n_ranks, m, h, inter, inter_s, experts, topk = 8, 16, 1024, 256, 128, 16, 4
    mesh = Mesh(np.asarray(jax.devices()[:n_ranks]), ("d",))
    sh = NamedSharding(mesh, P("d"))
    rng = np.random.default_rng(7)

    def quant(w, axis):
      s = np.max(np.abs(w), axis=axis, keepdims=True) / 448.0
      return (
          jnp.asarray(w / s, dtype=F8),
          jnp.asarray(s, dtype=jnp.float32),
      )

    w1, w1_scale = quant(
        rng.normal(size=(n_ranks * experts, h, 2 * inter)) * 0.02, 1
    )
    w2, w2_scale = quant(
        rng.normal(size=(n_ranks * experts, inter, h)) * 0.02, 1
    )
    shared_w1, shared_w1_scale = quant(
        rng.normal(size=(n_ranks * h, 2 * inter_s)) * 0.02, 0
    )
    shared_w2, shared_w2_scale = quant(
        rng.normal(size=(n_ranks * inter_s, h)) * 0.02, 0
    )
    shared_gate_weight = jax.device_put(
        jnp.asarray(rng.normal(size=(h, 1)) * 0.02, dtype=jnp.bfloat16),
        NamedSharding(mesh, P()),
    )
    x = jnp.asarray(rng.normal(size=(m, h)), dtype=jnp.bfloat16)
    tw = jnp.asarray(rng.dirichlet(np.ones(topk), size=m), dtype=jnp.bfloat16)
    ti = jnp.asarray(
        np.stack([rng.choice(experts, topk, replace=False) for _ in range(m)]),
        dtype=jnp.int32,
    )
    rank = jax.device_put(
        np.arange(n_ranks, dtype=np.int32).reshape(n_ranks, 1), sh
    )
    args = (
        rank,
        jax.device_put(jnp.tile(x, (n_ranks, 1)), sh),
        jax.device_put(w1, sh),
        jax.device_put(w2, sh),
        jax.device_put(w1_scale[:, :, None, :], sh),
        jax.device_put(w2_scale[:, :, None, :], sh),
        jax.device_put(shared_w1, sh),
        jax.device_put(jnp.tile(shared_w1_scale, (n_ranks, 1)), sh),
        jax.device_put(shared_w2, sh),
        jax.device_put(jnp.tile(shared_w2_scale, (n_ranks, 1)), sh),
        shared_gate_weight,
        jax.device_put(jnp.tile(tw, (n_ranks, 1)), sh),
        jax.device_put(jnp.tile(ti, (n_ranks, 1)), sh),
    )
    specs = (P("d"),) * 10 + (P(), P("d"), P("d"))
    op = pallas_mosaic_tpu.PallasMosaicTpuFusedMoeDecode()

    got = jax.jit(
        jax.shard_map(
            lambda r, *rest: op(r.reshape(-1), *rest, n_ranks=n_ranks),
            mesh=mesh,
            in_specs=specs,
            out_specs=P("d"),
            check_vma=False,
        )
    )(*args)
    want = jax.jit(
        jax.shard_map(
            lambda r, *rest: _call_ref(r.reshape(-1), *rest, n_ranks=n_ranks),
            mesh=mesh,
            in_specs=specs,
            out_specs=P("d"),
            check_vma=False,
        )
    )(*args)
    got_np = np.asarray(got, dtype=np.float32).reshape(n_ranks, m, h)
    want_np = np.asarray(want, dtype=np.float32).reshape(n_ranks, m, h)
    self.assertLess(_max_rel(got_np[0], want_np[0]), 0.08)


class ConfigAndHeuristicsTest(parameterized.TestCase):

  def test_autotuning_configs(self):
    op = pallas_mosaic_tpu.PallasMosaicTpuFusedMoeDecode()
    ba_256 = op.bind(*_make_single_rank_inputs(inter=256))
    configs_256 = op._get_autotuning_configs(ba_256)  # pylint: disable=protected-access
    self.assertEqual(
        configs_256,
        {
            pallas_mosaic_tpu.Config(i_block=None),
            pallas_mosaic_tpu.Config(i_block=128),
        },
    )
    ba_512 = op.bind(*_make_single_rank_inputs(inter=512))
    configs_512 = op._get_autotuning_configs(ba_512)  # pylint: disable=protected-access
    self.assertEqual(
        configs_512,
        {
            pallas_mosaic_tpu.Config(i_block=None),
            pallas_mosaic_tpu.Config(i_block=128),
            pallas_mosaic_tpu.Config(i_block=256),
        },
    )

  def test_config_defaults_and_validation(self):
    cfg = pallas_mosaic_tpu.Config()
    self.assertIsNone(cfg.i_block)
    self.assertEqual(cfg.n_chunk, 512)
    self.assertEqual(cfg.num_buffers, 6)
    self.assertEqual(cfg.q_block, 512)
    with self.assertRaises(ValueError):
      pallas_mosaic_tpu.Config(i_block=64)
    with self.assertRaises(ValueError):
      pallas_mosaic_tpu.Config(n_chunk=64)
    with self.assertRaises(ValueError):
      pallas_mosaic_tpu.Config(num_buffers=2)
    with self.assertRaises(ValueError):
      pallas_mosaic_tpu.Config(q_block=64)


if __name__ == "__main__":
  absltest.main()
