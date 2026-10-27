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
"""Packed input windows preserve the aligned forward contract."""

import functools

import jax
import jax.numpy as jnp
import pytest
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_fused_test as forward_tests

interpret_on_cpu = forward_tests.interpret_on_cpu


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_packed_forward(dtype):
  args, kwargs = forward_tests._inputs(dtype, True)
  results = []
  for packed in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(
            fuse_forward=True, fuse_backward=True, packed_forward=packed
        )
    )
    call = functools.partial(forward_tests._call, op, kwargs)
    results.append(jax.jit(call)(*args))
  forward_tests._assert_close(
      results[1], results[0], tolerance=0.002 if dtype == jnp.bfloat16 else 1e-5
  )


@pytest.mark.parametrize("raw_gate", [False, True])
def test_all_window_offsets_without_initial_state(raw_gate):
  args, kwargs = forward_tests._inputs(jnp.float32, True)
  # Cumulative lengths visit every offset modulo eight; the final sequence
  # also crosses a chunk boundary and ends next to the allocation guard.
  labels = [label for label in range(1, 9) for _ in range(label)] + [9] * 79
  kwargs.update(
      segment_ids=jnp.array([labels], jnp.int32),
      max_num_segments=10,
      output_final_state=False,
      use_gate_in_kernel=raw_gate,
      lower_bound=-1.0 if raw_gate else None,
  )
  args = (
      *args[:3],
      args[3] if raw_gate else -jnp.ones_like(args[3]) * 0.01,
      args[4],
      None,
      args[6] if raw_gate else None,
      args[7] if raw_gate else None,
  )
  args = tuple(x[:, :, :115] if i < 5 else x for i, x in enumerate(args))
  outputs = []
  for packed in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(
            fuse_forward=True, fuse_backward=True, packed_forward=packed
        )
    )
    outputs.append(
        jax.jit(functools.partial(forward_tests._call, op, kwargs))(*args)
    )
  forward_tests._assert_close(outputs[1], outputs[0], tolerance=1e-5)


def test_fixed_length_falls_back(monkeypatch):
  from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_kernel as kernels

  def unexpected_packed(*args, **kwargs):
    raise AssertionError("Fixed-length inputs must retain aligned dispatch")

  monkeypatch.setattr(
      kernels, "_packed_fused_forward_kernel", unexpected_packed
  )
  args, kwargs = forward_tests._inputs(jnp.float32, False)
  op = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=mosaic.Config(
          fuse_forward=True, fuse_backward=True, packed_forward=True
      )
  )
  actual = forward_tests._call(op, kwargs, *args)
  expected = forward_tests._call(
      mosaic.PallasMosaicTpuKimiDeltaAttention(), kwargs, *args
  )
  forward_tests._assert_close(actual, expected, tolerance=1e-5)


def test_multiple_head_groups_and_empty_batch(monkeypatch):
  from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_kernel as kernels

  monkeypatch.setattr(kernels, "estimate_mini_batch", lambda *args, **kwargs: 1)
  args, kwargs = forward_tests._inputs(jnp.float32, True)
  expanded = []
  for x in args[:5]:
    x = jnp.concatenate([x, x * 0.75], axis=0)
    expanded.append(jnp.repeat(x, 2, axis=1))
  state = jnp.repeat(jnp.repeat(args[5], 2, axis=0), 2, axis=2)
  args = (*expanded, state, jnp.repeat(args[6], 2), jnp.tile(args[7], 2))
  kwargs["segment_ids"] = jnp.concatenate(
      [kwargs["segment_ids"], jnp.zeros_like(kwargs["segment_ids"])]
  )
  outputs = []
  for packed in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(
            fuse_forward=True, fuse_backward=True, packed_forward=packed
        )
    )
    outputs.append(
        jax.jit(functools.partial(forward_tests._call, op, kwargs))(*args)
    )
  forward_tests._assert_close(outputs[1], outputs[0], tolerance=1e-5)
