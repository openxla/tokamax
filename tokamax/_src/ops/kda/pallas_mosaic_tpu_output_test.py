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
"""Packed output compaction boundaries and forward compatibility."""

import functools
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_fused_test as forward_tests
from tokamax._src.ops.kda import pallas_mosaic_tpu_output as output_kernel

interpret_on_cpu = forward_tests.interpret_on_cpu


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("fallback", [False, True])
def test_compaction_poisoned_tails_and_empty_batch(
    dtype, fallback, monkeypatch
):
  # Multiple head groups; unequal boundaries; a middle empty sequence;
  # no trailing guard at the maximum aligned end; non-eight-aligned output.
  original = jnp.array(
      [[0, 1, 1, 4, 19], [0, 8, 9, 17, 19], [0, 0, 0, 0, 0]], jnp.int32
  )
  aligned = jnp.array(
      [[0, 64, 64, 128, 192], [0, 64, 128, 192, 256], [0, 0, 0, 0, 0]],
      jnp.int32,
  )
  source = np.full((16, 3, 256, 128), np.nan, np.float32)
  expected = np.zeros((16, 3, 23, 128), np.float32)
  for head in range(16):
    for batch in range(3):
      for seq in range(4):
        start, end = map(int, original[batch, seq : seq + 2])
        aligned_start = int(aligned[batch, seq])
        values = (
            head * 0.125
            + batch * 0.25
            + np.arange(start, end)[:, None] * 0.0625
        )
        source[head, batch, aligned_start : aligned_start + end - start] = (
            values
        )
        expected[head, batch, start:end] = values
  if fallback:
    monkeypatch.setattr(
        output_kernel.common,
        "get_tpu_limits",
        lambda: types.SimpleNamespace(vmem_limit_bytes=1, block_align_minor=8),
    )
  calls = []
  kernel = output_kernel._compact_output_kernel

  def tracked_kernel(*args, **kwargs):
    calls.append(True)
    return kernel(*args, **kwargs)

  monkeypatch.setattr(output_kernel, "_compact_output_kernel", tracked_kernel)
  # Hardware limits are read at trace time; recompile after changing them.
  output_kernel.compact_output.clear_cache()
  actual = output_kernel.compact_output(
      jnp.asarray(source, dtype), original, aligned, 23
  )
  assert bool(calls) == (not fallback)
  forward_tests._assert_close(actual, jnp.asarray(expected, dtype), tolerance=0)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_compaction_forward(dtype):
  args, kwargs = forward_tests._inputs(dtype, True)
  args = (
      *[jnp.repeat(x, 8, axis=0) for x in args[:5]],
      jnp.repeat(args[5], 8, axis=2),
      jnp.repeat(args[6], 8),
      jnp.tile(args[7], 8),
  )
  results = []
  for compact in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(
            fuse_forward=True,
            fuse_backward=True,
            packed_output=compact,
            packed_forward=True,
        )
    )
    call = functools.partial(forward_tests._call, op, kwargs)
    results.append(jax.jit(call)(*args))
  forward_tests._assert_close(
      results[1], results[0], tolerance=0.002 if dtype == jnp.bfloat16 else 1e-5
  )


@pytest.mark.parametrize("heads,width", [(1, 128), (8, 64)])
def test_unsupported_shapes_keep_gather(heads, width, monkeypatch):
  def unexpected(*args, **kwargs):
    raise AssertionError("Unsupported compaction shape must retain gather")

  monkeypatch.setattr(output_kernel, "_compact_output_kernel", unexpected)
  source = jnp.ones((heads, 1, 128, width), jnp.float32)
  actual = output_kernel.compact_output(
      source, jnp.array([0, 3]), jnp.array([0, 64]), 7
  )
  expected = jnp.concatenate(
      [jnp.ones((heads, 1, 3, width)), jnp.zeros((heads, 1, 4, width))], axis=2
  )
  forward_tests._assert_close(actual, expected, tolerance=0)
