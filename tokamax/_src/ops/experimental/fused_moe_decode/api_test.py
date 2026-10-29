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
"""API tests for Fused MoE Decode."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src import hlo_utils
from tokamax._src.ops.experimental.fused_moe import reference
from tokamax._src.ops.experimental.fused_moe_decode import api

F8 = jnp.float8_e4m3fn


def _make_inputs(
    tokens: int = 16,
    h: int = 512,
    inter: int = 256,
    inter_s: int = 128,
    experts: int = 4,
    topk: int = 2,
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


class FusedMoeDecodeApiTest(parameterized.TestCase):

  @parameterized.parameters(None, "xla", "mosaic_tpu")
  def test_api_dispatch(self, implementation):
    mosaic_supported = (
        "mosaic_tpu" in api.IMPLEMENTATIONS
        and api.IMPLEMENTATIONS["mosaic_tpu"].supported_on(jax.devices()[0])
    )
    if implementation == "mosaic_tpu" and not mosaic_supported:
      self.skipTest("mosaic_tpu not supported on this device.")
    inputs = _make_inputs()

    @jax.jit
    def run_fn(*args):
      return api.fused_moe_decode(
          *args, n_ranks=1, implementation=implementation
      )

    got = np.asarray(run_fn(*inputs), dtype=np.float32)
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
    want = np.asarray(
        reference.fused_moe_reference(
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
        ),
        dtype=np.float32,
    )
    rel = (
        np.linalg.norm(got - want, axis=1)
        / np.maximum(np.linalg.norm(want, axis=1), 1e-30)
    ).max()
    self.assertLess(rel, 0.08)

    opspecs = hlo_utils.get_opspecs(
        run_fn.lower(*inputs), include_xla_kernels=False
    )
    if implementation == "xla" or not mosaic_supported:
      self.assertEmpty(opspecs)
    else:
      self.assertIsInstance(
          opspecs[0].op, type(api.IMPLEMENTATIONS["mosaic_tpu"])
      )

  def test_invalid_implementation_raises(self):
    inputs = _make_inputs()
    with self.assertRaisesRegex(ValueError, "Unknown implementation"):
      api.fused_moe_decode(
          *inputs,
          implementation="unknown_backend",  # pyrefly: ignore[bad-argument-type]
      )
    with self.assertRaisesRegex(ValueError, "empty sequence"):
      api.fused_moe_decode(*inputs, implementation=())


if __name__ == "__main__":
  absltest.main()
