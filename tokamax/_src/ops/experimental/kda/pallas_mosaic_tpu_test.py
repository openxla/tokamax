# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for the Pallas/Mosaic TPU KDA adapter and input heuristics."""

import types

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu
from tokamax._src.ops.experimental.kda.cp_utils import CPContext


def _check_inputs_support(q, v, **overrides):
  kwargs = dict(
      initial_state=None,
      output_final_state=False,
      segment_ids=None,
      cp_context=None,
      chunk_size=64,
      N_max=None,
  )
  kwargs.update(overrides)
  return pallas_mosaic_tpu.check_inputs_support(q, v, **kwargs)


class PallasMosaicTpuKimiDeltaAttentionTest(parameterized.TestCase):

  def test_rejects_large_key_dimension_before_kernel(self):
    q = jnp.ones((1, 1, 64, 257), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, 1), dtype=jnp.float32)

    with self.assertRaisesRegex(NotImplementedError, "up to 256"):
      _check_inputs_support(q, v)

  @parameterized.parameters((0, 1), (1, 0))
  def test_rejects_empty_kv_dimension_before_kernel(self, key_dim, value_dim):
    q = jnp.ones((1, 1, 64, key_dim), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, value_dim), dtype=jnp.float32)

    with self.assertRaisesRegex(NotImplementedError, "positive key and value"):
      _check_inputs_support(q, v)

  @parameterized.parameters((0, 1, 64), (1, 0, 64), (1, 1, 0))
  def test_rejects_empty_grid_dimension_before_kernel(
      self, heads, batch, seq_len
  ):
    q = jnp.ones((heads, batch, seq_len, 1), dtype=jnp.float32)
    v = jnp.ones((heads, batch, seq_len, 1), dtype=jnp.float32)

    with self.assertRaisesRegex(NotImplementedError, "positive head, batch"):
      _check_inputs_support(q, v)

  def test_rejects_multiple_fixed_states_before_kernel(self):
    q = jnp.ones((1, 1, 64, 1), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, 1), dtype=jnp.float32)
    initial_state = jnp.zeros((1, 2, 1, 1, 1), dtype=jnp.float32)

    with self.assertRaisesRegex(NotImplementedError, "exactly one"):
      _check_inputs_support(q, v, initial_state=initial_state)

  @parameterized.parameters((64, 128), (128, 64))
  def test_rejects_unaligned_cp_dimensions_before_kernel(
      self, key_dim, value_dim
  ):
    q = jnp.ones((1, 1, 64, key_dim), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, value_dim), dtype=jnp.float32)
    segment_ids = jnp.ones((1, 64), dtype=jnp.int32)
    cp_context = CPContext(mesh=types.SimpleNamespace(shape={"context": 2}))

    with self.assertRaisesRegex(NotImplementedError, "multiples of 128"):
      _check_inputs_support(
          q,
          v,
          segment_ids=segment_ids,
          cp_context=cp_context,
          N_max=1,
      )

  def test_rejects_cp_contract_gaps_before_kernel(self):
    q = jnp.ones((1, 1, 64, 128), dtype=jnp.float32)
    v = jnp.ones((1, 1, 64, 128), dtype=jnp.float32)
    segment_ids = jnp.ones((1, 64), dtype=jnp.int32)
    initial_state = jnp.zeros((1, 1, 1, 128, 128), dtype=jnp.float32)
    cp_context = CPContext(mesh=types.SimpleNamespace(shape={"context": 2}))
    cases = (
        (
            "initial_state",
            dict(
                initial_state=initial_state,
                segment_ids=segment_ids,
                N_max=1,
            ),
        ),
        (
            "output_final_state",
            dict(
                output_final_state=True,
                segment_ids=segment_ids,
                N_max=1,
            ),
        ),
        ("segment_ids", dict(N_max=1)),
        ("N_max", dict(segment_ids=segment_ids)),
    )

    for error_fragment, overrides in cases:
      with self.subTest(error_fragment=error_fragment):
        with self.assertRaisesRegex(NotImplementedError, error_fragment):
          _check_inputs_support(
              q,
              v,
              cp_context=cp_context,
              **overrides,
          )


if __name__ == "__main__":
  absltest.main()
