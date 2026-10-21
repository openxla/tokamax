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
"""Unit tests for PallasMosaicTpuFusedMoe operator."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
from tokamax._src.ops.experimental.fused_moe import pallas_mosaic_tpu
from tokamax._src.ops.experimental.fused_moe import reference
from tokamax._src.ops.experimental.fused_moe.kernel import AXIS


class PallasMosaicTpuTest(parameterized.TestCase):

  def test_op_with_mesh(self):
    if jax.device_count() < 8:
      self.skipTest("Requires 8 JAX devices for EP mesh test")

    mesh = Mesh(np.asarray(jax.devices()[:8]), axis_names=(AXIS,))
    tokens, hidden, inter, experts = 256, 512, 256, 32
    key = jax.random.PRNGKey(42)
    k_x, k_w1, k_w2, k_g = jax.random.split(key, 4)

    x = jax.random.normal(k_x, (tokens, hidden), dtype=jnp.bfloat16)
    w1 = jax.random.normal(
        k_w1, (experts, hidden, 2 * inter), dtype=jnp.bfloat16
    )
    w2 = jax.random.normal(k_w2, (experts, inter, hidden), dtype=jnp.bfloat16)
    gating = jax.random.normal(k_g, (tokens, experts), dtype=jnp.float32)

    op_wrapper = pallas_mosaic_tpu.PallasMosaicTpuFusedMoe()
    out, _ = op_wrapper._fwd(
        x=x,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=4,
        renormalize=True,
        act_fn="silu",
        mesh=mesh,
    )

    ref_out = reference.fused_moe_reference(
        x=x,
        w1=w1,
        w2=w2,
        gating=gating,
        topk=4,
        renormalize=True,
        act_fn="silu",
    )

    np_out = np.asarray(out, dtype=np.float64)
    np_ref = np.asarray(ref_out, dtype=np.float64)
    rel_l2 = float(np.linalg.norm(np_out - np_ref) / np.linalg.norm(np_ref))
    self.assertLess(rel_l2, 0.01)


if __name__ == "__main__":
  absltest.main()
