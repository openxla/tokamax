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
"""Fused state rematerialization preserves the low-memory backward path."""

import functools

import jax
import jax.numpy as jnp
import pytest
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
from tokamax._src.ops.kda import pallas_mosaic_tpu_bwd_kernel as kernels
from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_fused_test as forward_tests

interpret_on_cpu = forward_tests.interpret_on_cpu


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
def test_rematerialized_backward(dtype, packed):
  args, kwargs = forward_tests._inputs(dtype, packed)
  results = []
  for fused in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(fuse_forward=True, fuse_backward=True, 
            rematerialize_for_backward=True,
            fuse_rematerialization=fused,
        )
    )
    output, pullback = jax.vjp(
        functools.partial(forward_tests._call, op, kwargs), *args
    )
    results.append(
        (
            output,
            pullback(jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, output)),
        )
    )
  forward_tests._assert_close(
      results[1], results[0], tolerance=0.002 if dtype == jnp.bfloat16 else 1e-5
  )


@pytest.mark.parametrize("save_state", [False, True])
def test_dispatch_requires_rematerialization_and_backward_fusion(
    monkeypatch, save_state
):
  def unexpected(*args, **kwargs):
    raise AssertionError("Fused rematerialization should not be selected")

  monkeypatch.setattr(kernels, "_rematerialize_states_pallas", unexpected)
  args, kwargs = forward_tests._inputs(jnp.float32, False)
  op = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=mosaic.Config(fuse_forward=True, 
          rematerialize_for_backward=not save_state,
          fuse_backward=save_state,
          fuse_rematerialization=True,
      )
  )
  output, pullback = jax.vjp(
      functools.partial(forward_tests._call, op, kwargs), *args
  )
  grads = pullback(jax.tree.map(jnp.ones_like, output))
  forward_tests._assert_close(grads, grads, tolerance=0)


def test_fp32_value_correction_preserves_running_state():
  args, kwargs = forward_tests._inputs(jnp.bfloat16, False)
  op = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=mosaic.Config(fuse_forward=True, fuse_backward=True, rematerialize_for_backward=True)
  )
  _, residuals = forward_tests._call(
      op, dict(kwargs, return_residuals=True), *args
  )
  r = residuals

  @jax.jit
  def staged_reference(r):
    w, u, _, kg = kernels._recompute_w_u_fwd(
        r.q, r.k, r.v, r.beta, r.akk, r.g_cumsum, 64
    )
    h, v_new, _ = kernels.chunk_gated_delta_rule_fwd_h(
        kg, w, u, gk=r.g_cumsum, initial_state=r.initial_state, chunk_size=64
    )
    return h, v_new

  expected_h, expected_v = staged_reference(r)
  actual_h, actual_v = kernels._rematerialize_states_pallas(
      r.q, r.k, r.v, r.beta, r.akk, r.g_cumsum, r.initial_state, None
  )
  assert actual_v.dtype == jnp.float32
  forward_tests._assert_close(
      (actual_h, actual_v), (expected_h, expected_v), tolerance=1e-6
  )


@pytest.mark.parametrize("key_dim,value_dim", [(128, 256), (256, 128)])
def test_multiple_heads_batches_without_state(key_dim, value_dim):
  keys = jax.random.split(jax.random.key(92), 3)
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
  results = []
  for fused in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(fuse_forward=True, fuse_backward=True, 
            rematerialize_for_backward=True, fuse_rematerialization=fused
        )
    )
    output, pullback = jax.vjp(op, *args)
    results.append(pullback(jax.tree.map(jnp.ones_like, output)))
  forward_tests._assert_close(results[1], results[0], tolerance=0.002)
