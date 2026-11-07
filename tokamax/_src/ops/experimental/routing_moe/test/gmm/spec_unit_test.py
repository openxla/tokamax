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
"""Unit tests for RoutingGmmSpec."""

import dataclasses

from absl.testing import absltest
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import spec
from tokamax._src.ops.experimental.routing_moe.gmm import spec as gmm_spec_lib


def _make_spec(
    num_tokens: int,
    *,
    with_down: bool = False,
) -> gmm_spec_lib.RoutingGmmSpec:
  """Builds a RoutingGmmSpec for a 512-expert, top-10 MoE layer."""
  num_experts, ep_size, top_k, size_k = 512, 4, 10, 2560
  size_n = 1536 if with_down else 1280
  x = spec.ActivationSpec(
      row_size=num_tokens,
      col_size=size_k,
      dtype=jnp.dtype(jnp.bfloat16),
      quant=None,
  )

  def weight_spec(
      in_size: int, out_size: int, *, fused_gate_up: bool = False
  ) -> spec.WeightSpec:
    return spec.WeightSpec(
        group_size=num_experts // ep_size,
        in_size=in_size,
        out_size=out_size,
        dtype=jnp.dtype(jnp.float8_e4m3fn),
        quant=spec.SubchannelQuantSpec(block_size=in_size),
        has_bias=False,
        fused_gate_up=fused_gate_up,
    )

  w = weight_spec(size_k, size_n, fused_gate_up=with_down)
  w2 = weight_spec(size_n // 2, size_k) if with_down else None
  routed_rows = num_tokens * top_k
  return gmm_spec_lib.RoutingGmmSpec(
      x=x,
      w=w,
      w2=w2,
      gather=spec.RoutingSpec(num_indices=routed_rows, remote_axis=None),
      scatter=spec.RoutingSpec(num_indices=routed_rows, remote_axis=None),
      num_global_groups=num_experts,
      fuse_act="silu" if with_down else None,
      size_ep=1,
  )


class RoutingGmmSpecTest(absltest.TestCase):

  def test_derived_sizes_single_and_two_stage(self):
    single = _make_spec(8, with_down=False)
    self.assertEqual(single.size_k, 2560)
    self.assertIsNone(single.size_k2)
    self.assertEqual(single.size_out_m, 80)
    self.assertEqual(single.size_out_n, 1280)
    self.assertEqual(single.size_local_groups, 128)
    self.assertFalse(single.is_fused_act)

    fused = _make_spec(8, with_down=True)
    self.assertEqual(fused.size_k, 2560)
    self.assertEqual(fused.size_k2, 768)
    self.assertEqual(fused.size_out_m, 80)
    self.assertEqual(fused.size_out_n, 2560)
    self.assertTrue(fused.is_fused_act)

  def test_validation_rejects_mismatched_specs(self):
    base = _make_spec(8, with_down=True)
    bad_x = dataclasses.replace(base.x, col_size=512)
    with self.assertRaisesRegex(ValueError, "must match w in_size"):
      dataclasses.replace(base, x=bad_x)

    with self.assertRaisesRegex(ValueError, "requires a remote scatter"):
      dataclasses.replace(base, size_ep=2)

    bad_scatter = spec.RoutingSpec(num_indices=79, remote_axis=None)
    with self.assertRaisesRegex(ValueError, "scatter must map every computed"):
      dataclasses.replace(base, scatter=bad_scatter)


if __name__ == "__main__":
  absltest.main()
