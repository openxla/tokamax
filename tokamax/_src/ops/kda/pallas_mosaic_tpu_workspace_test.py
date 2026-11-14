# Copyright 2026 Ant Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scheduling-only CPU tests; native KDA comparisons are separate."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokamax._src.ops.kda import pallas_mosaic_tpu_workspace as workspace


@pytest.mark.parametrize("outer_remat", [False, True])
@pytest.mark.parametrize("groups", [0, 1, 2, 4])
@pytest.mark.parametrize("over", [False, True])
def test_branch_and_head_gradients(monkeypatch, outer_remat, groups, over):
  capacities = []

  def call(
      q, k, v, g, b, *, a_log, delta_time_bias, max_num_segments, **kwargs
  ):
    del kwargs
    capacities.append(max_num_segments)
    return (
        q * k
        + v
        + g * a_log[:, None, None, None]
        + b[..., None]
        * delta_time_bias.reshape(q.shape[0], 2)[:, None, None, :]
    ), None

  monkeypatch.setattr(
      workspace.mosaic,
      "PallasMosaicTpuKimiDeltaAttention",
      lambda **kwargs: call,
  )
  # Binary-exact operands isolate head/branch routing from fusion rounding.
  q = jnp.arange(4 * 2 * 5 * 2, dtype=jnp.float32).reshape(4, 2, 5, 2) * 0.125
  xs = (
      q,
      q * 2,
      q * 3,
      q * 4,
      q[..., 0],
      jnp.arange(4, dtype=jnp.float32),
      jnp.arange(8, dtype=jnp.float32),
  )
  ids = jnp.array([[1, 1, 2, 2, 0], [1, 2, 3 if over else 2, 0, 0]], jnp.int32)
  meta = types.SimpleNamespace(is_cp_enabled=True, axis_name=None)

  def scheduled(*xs):
    return workspace.kimi_delta_attention_cp(
        *xs[:5],
        a_log=xs[5],
        delta_time_bias=xs[6],
        segment_ids=ids,
        context_parallel_metadata=meta,
        max_num_segments=262144,
        fast_segment_capacity=2,
        head_group_size=groups,
    )[0]

  def expected(*xs):
    return call(
        *xs[:5], a_log=xs[5], delta_time_bias=xs[6], max_num_segments=5
    )[0]

  if outer_remat:
    scheduled = jax.checkpoint(scheduled)
  actual, back = jax.vjp(scheduled, *xs)
  reference, reference_back = jax.vjp(expected, *xs)
  np.testing.assert_array_equal(actual, reference)
  for a, e in zip(
      back(jnp.ones_like(actual)),
      reference_back(jnp.ones_like(reference)),
      strict=True,
  ):
    np.testing.assert_allclose(a, e, atol=1e-6, rtol=1e-6)
  assert set(capacities) <= {2, 5}


@pytest.mark.parametrize(
    "ids,expected",
    [
        ([[7, 0, 7]], 2),
        ([[0, 0, 0]], 0),
        ([[1, 1, 1], [1, 2, 3]], 3),
    ],
)
def test_count_nonpadding_runs(ids, expected):
  assert (
      int(workspace._uniform_segment_count(jnp.asarray(ids), None)) == expected
  )


def test_local_state_indices_do_not_replace_cross_rank_labels(monkeypatch):
  mosaic = workspace.mosaic
  ids = jnp.array([[15] * 65 + [17] * 63], jnp.int32)
  original = np.asarray(ids).copy()
  metadata = object()
  monkeypatch.setattr(
      mosaic,
      "derive_context_parallel_metadata",
      lambda **kwargs: (metadata, jnp.array([[0, 65, 128]], jnp.int32)),
  )
  q = jnp.ones((2, 1, 128, 128), jnp.bfloat16)
  prepared = mosaic.PallasMosaicTpuKimiDeltaAttention._preprocess_inputs(
      q,
      q,
      q,
      -q * 0.01,
      q[..., 0] * 0.5,
      initial_state=None,
      output_final_state=False,
      use_qk_l2norm=False,
      use_gate_in_kernel=False,
      segment_ids=ids,
      context_parallel_metadata=None,
      chunk_size=64,
      max_num_segments=2,
      local_state_indices=True,
  )
  np.testing.assert_array_equal(
      prepared.aligned_segment_ids[:, ::64], [[1, 1, 2, 0]]
  )
  np.testing.assert_array_equal(ids, original)
  assert prepared.context_parallel_metadata is metadata


def test_local_state_indices_reject_non_cp_entry():
  q = jnp.ones((1, 1, 64, 128), jnp.bfloat16)
  attention = workspace.mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=workspace.mosaic.Config(cp_local_state_indices=True),
      bypass_device_check=True,
  )
  with pytest.raises(NotImplementedError, match="Local state indices require"):
    attention(q, q, q, -q * 0.01, q[..., 0] * 0.5)


@pytest.mark.parametrize(
    "options",
    [
        dict(max_num_segments=True),
        dict(fast_segment_capacity=True),
        dict(head_group_size=True),
        dict(return_residuals=True),
    ],
)
def test_invalid_schedule_rejects_before_dispatch(options):
  q = jnp.ones((4, 1, 64, 128), jnp.bfloat16)
  kwargs = dict(
      max_num_segments=8,
      segment_ids=jnp.ones((1, 64), jnp.int32),
      context_parallel_metadata=types.SimpleNamespace(
          is_cp_enabled=True, axis_name=None
      ),
  )
  kwargs.update(options)
  with pytest.raises(ValueError):
    workspace.kimi_delta_attention_cp(
        q, q, q, -q * 0.01, q[..., 0] * 0.5, **kwargs
    )
