# Copyright 2026 Rabdos AI
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
"""Tests for fused GDN tile and memory planning."""

from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import tiling as fused_tiling


class GDNReducedDecodePlanningTest(parameterized.TestCase):
  """Wide FP32 calls can trade decode concurrency for VMEM capacity."""

  def _select(self, dtype=jnp.float32, decode_hint=None, budget=None):
    """Return (eligible, decode_tile, prefill_tile) for a wide-layout fixture.

    Mock VMEM/SMEM capacity; budget overrides VMEM bytes when provided.
    """
    desc = jax.ShapeDtypeStruct
    tokens, requests, slots = 1032, 10, 17
    n_kq, n_v = 16, 64
    d_k, d_v = 128, 128
    kernel_size = 4
    width = 2 * n_kq * d_k + n_v * d_v
    args = (
        desc((tokens, width), dtype),
        desc((tokens, n_v), dtype),
        desc((tokens, n_v), dtype),
        desc((slots, kernel_size - 1, width), jnp.float32),
        desc((slots, n_v, d_k, d_v), jnp.float32),
        desc((width, 1, kernel_size), jnp.bfloat16),
        desc((width,), jnp.float32),
        desc((n_v,), jnp.float32),
        desc((n_v,), jnp.float32),
        desc((requests + 1,), jnp.int32),
        desc((requests,), jnp.int32),
        desc((3,), jnp.int32),
        desc((requests,), jnp.int32),
    )
    with (
        mock.patch.object(
            fused_tiling.config,
            "get_vmem_limit_bytes",
            return_value=int(0.8 * 128 * 1024 * 1024)
            if budget is None
            else budget,
        ),
        mock.patch.object(
            fused_tiling.pltpu,
            "get_tpu_info",
            return_value=SimpleNamespace(smem_capacity_bytes=1024 * 1024),
        ),
    ):
      return fused_tiling.select_tiles(
          args,
          None,
          n_kq=n_kq,
          n_v=n_v,
          d_k=d_k,
          d_v=d_v,
          kernel_size=kernel_size,
          compute_precision=jnp.float32,
          decode_tile_size=decode_hint,
          zero_initialize_out=True,
      )

  def test_automatic_group_shrinks_when_prefill_tiles_do_not_fit(self):
    """Check automatic wide FP32 planning reduces the decode group and uses a
    smaller prefill tile.
    """
    self.assertEqual(self._select(), (True, 3, 64))

  def test_existing_bf16_activation_plan_is_preserved(self):
    """Check the existing BF16 activation plan is retained."""
    self.assertEqual(self._select(dtype=jnp.bfloat16), (True, None, 64))

  @parameterized.parameters((2, True, 64), (4, False, None))
  def test_explicit_decode_hint_is_preserved(self, group, eligible, tile):
    """Keep an explicit decode group when it fits; reject it otherwise."""
    self.assertEqual(self._select(decode_hint=group), (eligible, group, tile))

  def test_no_group_fits_an_exhausted_budget(self):
    """Check an exhausted memory budget rejects every group."""
    self.assertEqual(self._select(budget=0), (False, None, None))


if __name__ == "__main__":
  absltest.main()
