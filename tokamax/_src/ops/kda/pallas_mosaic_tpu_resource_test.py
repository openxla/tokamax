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
"""Large-capacity metadata checks; floating-point kernels are not mocked."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_kernel as kernels
from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_fused_test as forward_tests
from tokamax._src.ops.kda import pallas_mosaic_tpu_output as output
from tokamax._src.ops.kda import utils

interpret_on_cpu = forward_tests.interpret_on_cpu


def test_large_forward_table_matches_tight_table():
  q = jnp.arange(2 * 128 * 128, dtype=jnp.float32).reshape(2, 1, 128, 128)
  q = (jnp.sin(q) * 0.05).astype(jnp.bfloat16)
  mapping = jnp.array([[[0, 0], [1, 0]]], jnp.int32)
  tight = jnp.array([[0, 64, 128]], jnp.int32)
  large = jnp.pad(tight, ((0, 0), (0, 262144 - 2)), constant_values=128)

  def call(boundaries):
    return kernels.chunk_kda_fwd_fused(
        q,
        q,
        q,
        jnp.full_like(q, -0.01),
        jnp.full(q.shape[:-1], 0.5, q.dtype),
        boundaries,
        mapping,
        None,
        None,
        None,
        scale=128**-0.5,
        chunk_size=64,
        safe_gate=True,
        use_gate_in_kernel=False,
        lower_bound=None,
        store_h=False,
        store_residuals=False,
        store_gate=False,
        output_final_state=False,
    )[0]

  expected = jax.jit(call)(tight)
  actual = jax.jit(call)(large)
  assert np.isfinite(np.asarray(actual)).all()
  np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("capacity", [8192, 262144])
def test_chunk_windows_preserve_boundaries(batch, capacity):
  cu = jnp.broadcast_to(jnp.array([0, 45, 128]), (batch, 3))
  cu = jnp.pad(cu, ((0, 0), (0, capacity - 2)), constant_values=128)
  mapping = jnp.broadcast_to(jnp.array([0, 1, 1]), (batch, 3))
  windows = kernels._bounded_chunk_boundaries(cu, mapping)
  if cu.size <= 8193:
    np.testing.assert_array_equal(windows, cu)
  else:
    expected = np.array([0, 45, 45, 128, 45, 128] * batch)
    np.testing.assert_array_equal(windows, expected)
    assert windows.ndim == 1
    assert windows.size == batch * 3 * 2


@pytest.mark.parametrize("capacity", [8193, 262144])
@pytest.mark.parametrize("lengths", [(3, 5), (1,) * 8, ()])
def test_large_align_round_trip(capacity, lengths):
  cu = np.full(capacity + 1, sum(lengths), np.int32)
  cu[: len(lengths) + 1] = np.cumsum((0, *lengths))
  x = jnp.arange(2 * 8 * 3, dtype=jnp.float32).reshape(2, 1, 8, 3)
  aligned, _, padded, _ = utils._align_seqs([x], [], jnp.asarray(cu), 64)
  assert aligned[0].shape[2] == 8 * 64 + 64
  actual = utils._unalign_output(aligned[0], jnp.asarray(cu), padded, 8)
  expected = jnp.where(jnp.arange(8)[None, None, :, None] < sum(lengths), x, 0)
  np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("batch", [1, 2])
def test_large_compactor_never_prefetches_full_table(batch, monkeypatch):
  capacity, tokens = 262144, 8
  cu = np.full((batch, capacity + 1), tokens, np.int32)
  cu[:, 0] = 0
  aligned = np.full_like(cu, 64)
  aligned[:, 0] = 0
  x = jnp.arange(8 * batch * 128 * 128, dtype=jnp.float32).reshape(
      8, batch, 128, 128
  )

  def forbidden(*args, **kwargs):
    raise AssertionError("Large tables must not be scalar-prefetched")

  monkeypatch.setattr(output.pl, "pallas_call", forbidden)
  # Disable the jit cache: otherwise a previous trace could bypass the guard.
  with jax.disable_jit():
    actual = output.compact_output(
        x, jnp.asarray(cu), jnp.asarray(aligned), tokens
    )
  np.testing.assert_array_equal(actual, x[:, :, :tokens])
