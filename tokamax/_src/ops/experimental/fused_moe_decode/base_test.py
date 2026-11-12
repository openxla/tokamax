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
"""Correctness and validation unit tests for FusedMoeDecode base operator."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.fused_moe import reference
from tokamax._src.ops.experimental.fused_moe_decode import base

F8 = jnp.float8_e4m3fn


def _make_inputs(
    tokens: int = 8,
    h: int = 512,
    inter: int = 128,
    inter_s: int = 128,
    experts: int = 4,
    topk: int = 2,
    shared_gated: bool = True,
    seed: int = 0,
):
  key = jax.random.PRNGKey(seed)
  ks = jax.random.split(key, 8)

  def quant(w, axis):
    s = jnp.max(jnp.abs(w), axis=axis, keepdims=True) / 448.0
    return (w / s).astype(F8), s.astype(jnp.float32)

  hidden = jax.random.normal(ks[0], (tokens, h), dtype=jnp.bfloat16)
  w1, w1_scale = quant(
      jax.random.normal(ks[1], (experts, h, 2 * inter)) * 0.02, 1
  )
  w2, w2_scale = quant(jax.random.normal(ks[2], (experts, inter, h)) * 0.02, 1)
  shared_w1, shared_w1_scale = quant(
      jax.random.normal(ks[3], (h, 2 * inter_s)) * 0.02, 0
  )
  shared_w2, shared_w2_scale = quant(
      jax.random.normal(ks[4], (inter_s, h)) * 0.02, 0
  )
  shared_gate_weight = (
      jax.random.normal(ks[5], (h, 1), dtype=jnp.bfloat16) * 0.02
      if shared_gated
      else None
  )
  topk_weights = jax.nn.softmax(
      jax.random.normal(ks[6], (tokens, topk), dtype=jnp.float32), axis=-1
  ).astype(jnp.bfloat16)
  topk_ids = jax.random.randint(
      ks[7], (tokens, topk), 0, experts, dtype=jnp.int32
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


class FusedMoeDecodeBaseTest(parameterized.TestCase):

  @parameterized.product(
      shared_gated=[True, False],
      routed_scale=[1.0, 2.5],
      ep=[False, True],
  )
  def test_base_op_matches_reference(self, shared_gated, routed_scale, ep):
    inputs = _make_inputs(shared_gated=shared_gated)
    expert_start = jnp.array(0, dtype=jnp.int32) if ep else None
    op = base.FusedMoeDecode()
    got = op(
        *inputs,
        n_ranks=1,
        expert_start=expert_start,
        routed_scale=routed_scale,
    )
    (
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
    ) = inputs
    want = reference.fused_moe_reference(
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
        n_ranks=1,
        expert_start=expert_start,
        routed_scale=routed_scale,
    )
    self.assertEqual(got.shape, inputs[1].shape)
    self.assertEqual(got.dtype, jnp.bfloat16)
    np.testing.assert_allclose(
        np.asarray(got, dtype=np.float32),
        np.asarray(want, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )

  def test_validation_rejects_unaligned_hidden(self):
    inputs = _make_inputs(h=256)
    op = base.FusedMoeDecode()
    with self.assertRaisesRegex(ValueError, "multiple of 512"):
      op.bind(*inputs)

  def test_validation_rejects_scatter_rows_unaligned_tokens(self):
    inputs = _make_inputs(tokens=7, h=1024)
    op = base.FusedMoeDecode()
    with self.assertRaisesRegex(ValueError, "tokens % n_ranks == 0"):
      op.bind(*inputs, n_ranks=8, scatter_rows=True)

  def test_validation_rejects_mismatched_expert_shapes(self):
    inputs = list(_make_inputs())
    inputs[2] = jnp.zeros((4, 512, 512), dtype=F8)
    inputs[4] = jnp.zeros((4, 1, 1, 512), dtype=jnp.float32)
    op = base.FusedMoeDecode()
    with self.assertRaisesRegex(ValueError, "Incompatible w1"):
      op.bind(*inputs)

  def test_allows_larger_shared_intermediate_and_squeezed_scales(self):
    inputs = list(_make_inputs(inter=128, inter_s=256))
    # Pass 3D expert scales and 1D shared scales / gate weight.
    inputs[4] = inputs[4].squeeze(2)
    inputs[5] = inputs[5].squeeze(2)
    inputs[7] = inputs[7].squeeze(0)
    inputs[9] = inputs[9].squeeze(0)
    inputs[10] = inputs[10].squeeze(-1)
    op = base.FusedMoeDecode()
    ba = op.bind(*inputs)
    self.assertEqual(ba.arguments["w1_scale"].shape, (4, 1, 1, 256))
    self.assertEqual(ba.arguments["w2_scale"].shape, (4, 1, 1, 512))
    self.assertEqual(ba.arguments["shared_w1_scale"].shape, (1, 512))
    self.assertEqual(ba.arguments["shared_w2_scale"].shape, (1, 512))
    self.assertEqual(ba.arguments["shared_gate_weight"].shape, (512, 1))
    got = op(*inputs, n_ranks=1)
    self.assertEqual(got.shape, inputs[1].shape)


if __name__ == "__main__":
  absltest.main()
