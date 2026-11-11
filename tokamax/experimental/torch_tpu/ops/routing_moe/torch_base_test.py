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
"""Tests for PyTorch base interface to Tokamax Routing MoE."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe import base as jax_base
from tokamax.experimental.torch_tpu.ops.routing_moe import torch_base
import torch
import torch_tpu  # pylint: disable=unused-import


class TorchRoutingMoeBaseTest(parameterized.TestCase):

  def test_base_matches_jax_reference(self):
    if jax.default_backend() != "tpu":
      self.skipTest("TPU required.")
    device = "tpu"
    torch.manual_seed(0)
    x = torch.randn((16, 256), dtype=torch.bfloat16, device=device)
    w1 = torch.randn((8, 256, 256), dtype=torch.bfloat16, device=device)
    w2 = torch.randn((8, 128, 256), dtype=torch.bfloat16, device=device)
    logits = torch.randn((16, 8), dtype=torch.float32, device=device)
    vals, ids = torch.topk(logits, k=2, dim=-1)
    weights = torch.softmax(vals, dim=-1)
    ids = ids.to(torch.int32)

    out = torch_base.RoutingMoe(x, weights, ids, w1, w2)

    jax_out = jax_base.RoutingMoe()(
        jnp.array(x.cpu().float().numpy(), dtype=jnp.bfloat16),
        jnp.array(weights.cpu().numpy(), dtype=jnp.float32),
        jnp.array(ids.cpu().numpy(), dtype=jnp.int32),
        jnp.array(w1.cpu().float().numpy(), dtype=jnp.bfloat16),
        jnp.array(w2.cpu().float().numpy(), dtype=jnp.bfloat16),
    )
    jax_out_t = torch.as_tensor(
        np.asarray(jax_out, dtype=np.float32),
        device=device,
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(out, jax_out_t, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
  absltest.main()
