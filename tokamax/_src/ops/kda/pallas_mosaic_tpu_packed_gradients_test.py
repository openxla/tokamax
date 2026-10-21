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
"""Packed gradient compaction preserves the complete custom VJP."""

import functools

import jax
import jax.numpy as jnp
import pytest
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
from tokamax._src.ops.kda import pallas_mosaic_tpu_bwd_kernel as kernels
from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_fused_test as forward_tests

interpret_on_cpu = forward_tests.interpret_on_cpu


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("rematerialize", [False, True])
def test_full_gradient_compaction(dtype, rematerialize):
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
        config=mosaic.Config(fuse_forward=True, fuse_backward=True, 
            packed_forward=True,
            packed_output=True,
            packed_backward=True,
            packed_gradients=compact,
            rematerialize_for_backward=rematerialize,
            fuse_rematerialization=rematerialize,
        )
    )
    output, pullback = jax.vjp(
        functools.partial(forward_tests._call, op, kwargs), *args
    )
    results.append(
        pullback(jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, output))
    )
  # Compaction changes token placement only. Separate XLA compilations can
  # still schedule the preceding FP32 reductions differently by a few ULPs.
  forward_tests._assert_close(
      results[1], results[0], tolerance=0 if dtype == jnp.bfloat16 else 1e-7
  )


def test_fixed_length_retains_existing_path(monkeypatch):
  def unexpected(*args, **kwargs):
    raise AssertionError("Fixed-length backward must not compact gradients")

  monkeypatch.setattr(kernels, "compact_output", unexpected)
  args, kwargs = forward_tests._inputs(jnp.float32, False)
  op = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=mosaic.Config(fuse_forward=True, fuse_backward=True, packed_gradients=True)
  )
  output, pullback = jax.vjp(
      functools.partial(forward_tests._call, op, kwargs), *args
  )
  grads = pullback(jax.tree.map(jnp.ones_like, output))
  forward_tests._assert_close(grads, grads, tolerance=0)


@pytest.mark.parametrize("key_dim", [128, 256])
def test_preactivated_gate_without_normalization_or_state(key_dim):
  args, kwargs = forward_tests._inputs(jnp.bfloat16, True, key_dim=key_dim)
  args = (
      *[jnp.repeat(x, 8, axis=0) for x in args[:3]],
      -jnp.ones((8, 1, 128, key_dim), jnp.bfloat16) * 0.01,
      jnp.repeat(args[4], 8, axis=0),
      None,
      None,
      None,
  )
  kwargs.update(
      use_gate_in_kernel=False,
      lower_bound=None,
      use_qk_l2norm=False,
      output_final_state=False,
      max_num_segments=3,
  )
  results = []
  for compact in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(fuse_forward=True, fuse_backward=True, packed_gradients=compact)
    )
    output, pullback = jax.vjp(
        functools.partial(forward_tests._call, op, kwargs), *args
    )
    results.append(pullback(jax.tree.map(jnp.ones_like, output)))
  forward_tests._assert_close(results[1], results[0], tolerance=0)
