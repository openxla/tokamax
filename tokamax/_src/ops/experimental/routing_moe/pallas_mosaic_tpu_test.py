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
"""Tests for Pallas Mosaic TPU RoutingMoe operator."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe import base
from tokamax._src.ops.experimental.routing_moe import pallas_mosaic_tpu


def _quantize_subchannel(
    w: jax.Array, dtype: jax.typing.DTypeLike, block_size: int
) -> tuple[jax.Array, jax.Array]:
  g, k, n = w.shape
  max_val = float(jnp.finfo(dtype).max)
  blocks = w.astype(jnp.float32).reshape(g, k // block_size, block_size, n)
  scale = jnp.max(jnp.abs(blocks), axis=2, keepdims=True) / max_val
  inv_scale = jnp.where(scale == 0, 0.0, 1.0 / scale)
  q = (blocks * inv_scale).astype(dtype).reshape(g, k, n)
  return q, scale.astype(jnp.float32)


class PallasMosaicTpuRoutingMoeTest(parameterized.TestCase):

  def setUp(self) -> None:
    super().setUp()
    op = pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe()
    device = jax.devices()[0]
    if not op.supported_on(device):
      self.skipTest(
          f"PallasMosaicTpuRoutingMoe not supported on {device.device_kind}."
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="bf16_single_shard_inter640",
          tokens=16,
          hidden=512,
          inter=640,
          local_experts=8,
          global_experts=8,
          topk=4,
          w_dtype=jnp.bfloat16,
          block_size=None,
      ),
      dict(
          testcase_name="fp8_ep_shard_inter640",
          tokens=32,
          hidden=512,
          inter=640,
          local_experts=8,
          global_experts=16,
          topk=4,
          w_dtype=jnp.float8_e4m3fn,
          block_size=128,
      ),
      dict(
          testcase_name="fp4_w4a8_ep_shard_qwen38_widths",
          tokens=16,
          hidden=2560,
          inter=640,
          local_experts=8,
          global_experts=32,
          topk=10,
          w_dtype=jnp.float4_e2m1fn,
          block_size=128,
      ),
  )
  def test_matches_reference(
      self,
      tokens: int,
      hidden: int,
      inter: int,
      local_experts: int,
      global_experts: int,
      topk: int,
      w_dtype: jax.typing.DTypeLike,
      block_size: int | None,
  ) -> None:
    k_x, k_w1, k_w2, k_g = jax.random.split(jax.random.key(42), 4)
    x = jax.random.normal(k_x, (tokens, hidden), dtype=jnp.bfloat16) * 0.2
    w1_f = (
        jax.random.normal(
            k_w1, (local_experts, hidden, 2 * inter), dtype=jnp.float32
        )
        * 0.1
    )
    w2_f = (
        jax.random.normal(
            k_w2, (local_experts, inter, hidden), dtype=jnp.float32
        )
        * 0.1
    )
    if block_size is None:
      w1, w1_scale = w1_f.astype(w_dtype), None
      w2, w2_scale = w2_f.astype(w_dtype), None
    else:
      w1, w1_scale = _quantize_subchannel(w1_f, w_dtype, block_size)
      w2, w2_scale = _quantize_subchannel(w2_f, w_dtype, block_size)

    logits = jax.random.normal(k_g, (tokens, global_experts), dtype=jnp.float32)
    vals, ids = jax.lax.top_k(logits, topk)
    weights = jax.nn.softmax(vals, axis=-1)
    # Zero out trailing padded rows to exercise skip_padded_tokens.
    weights = weights.at[-4:].set(0.0)
    ids = ids.at[-4:].set(jnp.arange(topk, dtype=jnp.int32))

    experts_start = (
        None
        if global_experts == local_experts
        else jnp.array(local_experts, dtype=jnp.int32)
    )
    ref_op = base.RoutingMoe()
    tpu_op = pallas_mosaic_tpu.PallasMosaicTpuRoutingMoe()

    ref_out = ref_op(
        x,
        weights,
        ids,
        w1,
        w2,
        w1_scale,
        w2_scale,
        experts_start=experts_start,
        global_num_experts=global_experts,
        skip_padded_tokens=True,
    )
    tpu_out = tpu_op(
        x,
        weights,
        ids,
        w1,
        w2,
        w1_scale,
        w2_scale,
        experts_start=experts_start,
        global_num_experts=global_experts,
        skip_padded_tokens=True,
    )
    np_out = np.asarray(tpu_out, dtype=np.float64)
    np_ref = np.asarray(ref_out, dtype=np.float64)
    np.testing.assert_array_equal(np_out[-4:], np.zeros((4, hidden)))
    denom = max(float(np.linalg.norm(np_ref)), 1e-8)
    rel_l2 = float(np.linalg.norm(np_out - np_ref) / denom)
    tol = 0.02 if block_size is None else 0.06
    self.assertLess(rel_l2, tol)


if __name__ == "__main__":
  absltest.main()
