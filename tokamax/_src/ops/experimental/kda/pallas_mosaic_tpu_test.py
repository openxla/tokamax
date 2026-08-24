# Copyright 2026 Ant Group. All Rights Reserved.
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
"""Tests for the Pallas/Mosaic TPU KDA adapter and public support checks."""

import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from tokamax._src.ops.experimental.kda import api
from tokamax._src.ops.experimental.kda import common
from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu
from tokamax._src.ops.experimental.kda import utils
from tokamax._src.ops.experimental.kda.cp_utils import ContextParallelMetadata


def _call_attention(implementation, q, v, **kwargs):
  def call():
    return api.kimi_delta_attention(
        q,
        jnp.ones_like(q),
        v,
        jnp.zeros_like(q),
        jnp.ones(q.shape[:-1], dtype=q.dtype),
        scale=1.0,
        implementation=implementation,
        **kwargs,
    )

  if implementation != "mosaic":
    return call()
  # Exercise Mosaic's public API validation on every test platform without
  # tracing a kernel: every case below is rejected at the start of `_fwd`.
  with mock.patch.object(
      pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention,
      "supported_on",
      return_value=True,
  ):
    return call()


class PallasMosaicTpuKimiDeltaAttentionTest(parameterized.TestCase):

  def test_tpu_limits_use_pallas_hardware_info(self):
    tpu_info = types.SimpleNamespace(
        vmem_capacity_bytes=64 * 1024 * 1024,
        num_sublanes=16,
        num_lanes=128,
    )

    with mock.patch.object(
        common.pltpu, "get_tpu_info", return_value=tpu_info
    ) as get_tpu_info:
      limits = common.get_tpu_limits()

    get_tpu_info.assert_called_once_with()
    self.assertEqual(
        limits.vmem_limit_bytes, int(tpu_info.vmem_capacity_bytes * 0.9)
    )
    self.assertEqual(limits.block_align_minor, tpu_info.num_sublanes)
    self.assertEqual(limits.block_align_major, tpu_info.num_lanes)

  def test_default_execution_config(self):
    attention = pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttention()
    vjp = pallas_mosaic_tpu.PallasMosaicTpuKimiDeltaAttentionVjp()
    expected = pallas_mosaic_tpu.Config(chunk_size=64)

    self.assertEqual(attention._get_heuristics_config(None), expected)
    self.assertEqual(attention._get_autotuning_configs(None), {expected})
    self.assertEqual(vjp._get_heuristics_config(None), expected)
    self.assertEqual(vjp._get_autotuning_configs(None), {expected})
    self.assertIsNone(expected.safe_gate)
    self.assertFalse(expected.rematerialize_for_backward)

  @parameterized.named_parameters(
      ("preactivated_gate", False, None, True),
      ("softplus_gate", True, None, False),
      ("bounded_gate", True, -5.0, True),
  )
  def test_safe_gate_is_selected_internally(
      self, use_gate_in_kernel, lower_bound, expected
  ):
    config = pallas_mosaic_tpu.Config()

    self.assertEqual(
        pallas_mosaic_tpu._resolve_safe_gate(
            config,
            use_gate_in_kernel=use_gate_in_kernel,
            lower_bound=lower_bound,
        ),
        expected,
    )

  def test_safe_gate_config_override(self):
    config = pallas_mosaic_tpu.Config(safe_gate=False)

    self.assertFalse(
        pallas_mosaic_tpu._resolve_safe_gate(
            config,
            use_gate_in_kernel=False,
            lower_bound=None,
        )
    )

  def test_large_key_dimension_is_mosaic_specific(self):
    q = jnp.ones((1, 1, 1, 257), dtype=jnp.float32)
    v = jnp.ones((1, 1, 1, 1), dtype=jnp.float32)

    output, final_state = _call_attention("xla", q, v)
    self.assertEqual(output.shape, v.shape)
    self.assertIsNone(final_state)

    with self.assertRaisesRegex(NotImplementedError, "up to 256"):
      _call_attention("mosaic", q, v)

  @parameterized.parameters((0, 1), (1, 0))
  def test_rejects_empty_kv_dimension_before_kernel(self, key_dim, value_dim):
    q = jnp.ones((1, 1, 64, key_dim), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, value_dim), dtype=jnp.float32)

    with self.assertRaisesRegex(NotImplementedError, "positive key and value"):
      _call_attention("mosaic", q, v)

  @parameterized.parameters((0, 1, 64), (1, 0, 64), (1, 1, 0))
  def test_rejects_empty_grid_dimension_before_kernel(
      self, heads, batch, seq_len
  ):
    q = jnp.ones((heads, batch, seq_len, 1), dtype=jnp.float32)
    v = jnp.ones((heads, batch, seq_len, 1), dtype=jnp.float32)

    with self.assertRaisesRegex(NotImplementedError, "positive head, batch"):
      _call_attention("mosaic", q, v)

  def test_rejects_multiple_fixed_states_before_kernel(self):
    q = jnp.ones((1, 1, 64, 1), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, 1), dtype=jnp.float32)
    initial_state = jnp.zeros((1, 2, 1, 1, 1), dtype=jnp.float32)

    with self.assertRaisesRegex(NotImplementedError, "exactly one"):
      _call_attention("mosaic", q, v, initial_state=initial_state)

  @parameterized.parameters((64, 128), (128, 64))
  def test_rejects_unaligned_cp_dimensions_before_kernel(
      self, key_dim, value_dim
  ):
    q = jnp.ones((1, 1, 64, key_dim), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, value_dim), dtype=jnp.float32)
    segment_ids = jnp.ones((1, 64), dtype=jnp.int32)
    context_parallel_metadata = ContextParallelMetadata(
        mesh=types.SimpleNamespace(shape={"context": 2})
    )

    with self.assertRaisesRegex(NotImplementedError, "multiples of 128"):
      _call_attention(
          "mosaic",
          q,
          v,
          segment_ids=segment_ids,
          context_parallel_metadata=context_parallel_metadata,
          max_num_segments=1,
      )

  def test_rejects_cp_contract_gaps_before_kernel(self):
    q = jnp.ones((1, 1, 64, 128), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, 128), dtype=jnp.float32)
    segment_ids = jnp.ones((1, 64), dtype=jnp.int32)
    initial_state = jnp.zeros((1, 1, 1, 128, 128), dtype=jnp.float32)
    context_parallel_metadata = ContextParallelMetadata(
        mesh=types.SimpleNamespace(shape={"context": 2})
    )
    cases = (
        (
            "initial_state",
            dict(
                initial_state=initial_state,
                segment_ids=segment_ids,
                max_num_segments=1,
            ),
        ),
        (
            "output_final_state",
            dict(
                output_final_state=True,
                segment_ids=segment_ids,
                max_num_segments=1,
            ),
        ),
        ("segment_ids", dict(max_num_segments=1)),
    )

    for error_fragment, overrides in cases:
      with self.subTest(error_fragment=error_fragment):
        with self.assertRaisesRegex(NotImplementedError, error_fragment):
          _call_attention(
              "mosaic",
              q,
              v,
              context_parallel_metadata=context_parallel_metadata,
              **overrides,
          )


if __name__ == "__main__":
  absltest.main()
