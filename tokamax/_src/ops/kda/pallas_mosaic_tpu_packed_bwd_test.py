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
"""Packed input windows for the fused reverse KDA traversal."""

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
def test_packed_backward(dtype, rematerialize):
  args, kwargs = forward_tests._inputs(dtype, True)
  results = []
  for packed in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(fuse_forward=True, fuse_backward=True, 
            packed_forward=True,
            packed_backward=packed,
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
  forward_tests._assert_close(
      results[1], results[0], tolerance=0.002 if dtype == jnp.bfloat16 else 1e-5
  )


def test_every_offset_multiple_groups_and_empty_batch(monkeypatch):
  original_launcher = kernels._fused_dhu_wy_intra_cumsum_pallas_jit
  monkeypatch.setattr(
      kernels,
      "_fused_dhu_wy_intra_cumsum_pallas_jit",
      functools.partial(original_launcher, mini_batch=1),
  )
  args, kwargs = forward_tests._inputs(jnp.float32, True)
  labels = [label for label in range(1, 9) for _ in range(label)] + [9] * 79
  kwargs.update(
      segment_ids=jnp.array([labels, [0] * 115], jnp.int32),
      max_num_segments=10,
      output_final_state=False,
  )
  expanded = []
  for x in args[:5]:
    x = jnp.concatenate([x, x * 0.75], axis=0)
    expanded.append(jnp.repeat(x[:, :, :115], 2, axis=1))
  args = (*expanded, None, jnp.repeat(args[6], 2), jnp.tile(args[7], 2))
  results = []
  for packed in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(fuse_forward=True, fuse_backward=True, packed_backward=packed)
    )
    output, pullback = jax.vjp(
        functools.partial(forward_tests._call, op, kwargs), *args
    )
    results.append(pullback(jax.tree.map(jnp.ones_like, output)))
  forward_tests._assert_close(results[1], results[0], tolerance=1e-5)


@pytest.mark.parametrize("mode", ["fixed", "no_fusion", "staged_remat"])
def test_fallback_dispatch(mode, monkeypatch):
  def unexpected(*args, **kwargs):
    raise AssertionError("Packed backward must retain the selected fallback")

  monkeypatch.setattr(
      kernels, "_packed_saved_state_backward_kernel", unexpected
  )
  args, kwargs = forward_tests._inputs(jnp.float32, mode != "fixed")
  op = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=mosaic.Config(fuse_forward=True, 
          packed_backward=True,
          fuse_backward=mode != "no_fusion",
          rematerialize_for_backward=mode == "staged_remat",
      )
  )
  output, pullback = jax.vjp(
      functools.partial(forward_tests._call, op, kwargs), *args
  )
  grads = pullback(jax.tree.map(jnp.ones_like, output))
  forward_tests._assert_close(grads, grads, tolerance=0)


@pytest.mark.parametrize("key_dim,value_dim", [(128, 256), (256, 128)])
def test_wide_values_and_keys(key_dim, value_dim):
  keys = jax.random.split(jax.random.key(94), 3)
  q = (jax.random.normal(keys[0], (2, 2, 128, key_dim)) * 0.05).astype(
      jnp.bfloat16
  )
  k = (jax.random.normal(keys[1], q.shape) * 0.05).astype(q.dtype)
  v = jax.random.normal(keys[2], (2, 2, 128, value_dim)).astype(q.dtype)
  args = (
      q,
      k,
      v,
      jnp.full_like(q, -0.01),
      jnp.full(q.shape[:-1], 0.5, q.dtype),
  )
  segments = jnp.array([[1] * 65 + [2] * 63, [1] * 17 + [2] * 111], jnp.int32)
  results = []
  for packed in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(fuse_forward=True, fuse_backward=True, packed_backward=packed)
    )
    call = functools.partial(op, segment_ids=segments, max_num_segments=2)
    output, pullback = jax.vjp(call, *args)
    results.append(pullback(jax.tree.map(jnp.ones_like, output)))
  forward_tests._assert_close(results[1], results[0], tolerance=0.002)
