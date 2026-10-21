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
"""Saved-state backward fusion and staged fallback comparisons."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokamax._src.ops.kda import api
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
from tokamax._src.ops.kda import pallas_mosaic_tpu_bwd_kernel as kernels
from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_fused_test as forward_tests

interpret_on_cpu = forward_tests.interpret_on_cpu


@pytest.mark.parametrize("fuse_backward", [False, True])
@pytest.mark.parametrize("rematerialize", [False, True])
@pytest.mark.parametrize("lower_bound", [None, -1.0])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_packed_padding_and_empty_state_gradients(
    fuse_backward, rematerialize, lower_bound, output_final_state
):
  args, kwargs = forward_tests._inputs(jnp.float32, True)
  kwargs.update(lower_bound=lower_bound, output_final_state=output_final_state)
  op = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=mosaic.Config(fuse_forward=True, 
          fuse_backward=fuse_backward,
          rematerialize_for_backward=rematerialize,
      )
  )
  results = []
  for implementation in (
      op,
      functools.partial(api.kimi_delta_attention, implementation="xla"),
  ):
    output, pullback = jax.vjp(
        functools.partial(forward_tests._call, implementation, kwargs), *args
    )
    results.append(
        pullback(jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, output))
    )
  forward_tests._assert_close(results[0], results[1], tolerance=0.002)
  for gradient in results[0][:5]:
    np.testing.assert_array_equal(np.asarray(gradient[:, :, 96:]), 0)
  np.testing.assert_allclose(
      np.asarray(results[0][5][:, 2]), 0.1 if output_final_state else 0
  )


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("fuse_forward", [False, True])
def test_saved_state_backward(dtype, packed, fuse_forward):
  args, kwargs = forward_tests._inputs(dtype, packed)
  # Cover pre-activated gates separately from the raw-gate regression above.
  args = (*args[:3], -jnp.ones_like(args[3]) * 0.01, *args[4:6], None, None)
  kwargs.update(use_gate_in_kernel=False, lower_bound=None)
  results = []
  for fuse_backward in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(
            fuse_forward=fuse_forward, fuse_backward=fuse_backward
        ),
    )
    call = functools.partial(forward_tests._call, op, kwargs)
    output, pullback = jax.vjp(call, *args)
    grads = pullback(jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, output))
    results.append((output, grads))
  forward_tests._assert_close(results[1][0], results[0][0], tolerance=1e-5)
  for actual, expected in zip(
      results[1][1][:6], results[0][1][:6], strict=True
  ):
    forward_tests._assert_close(
        actual, expected, tolerance=0.002 if dtype == jnp.bfloat16 else 1e-5
    )


@pytest.mark.parametrize("rematerialize", [False, True])
def test_backward_gate_gradients_against_autodiff(rematerialize):
  args, kwargs = forward_tests._inputs(jnp.bfloat16, True)
  args = (*args[:5], args[5][:, :1], *args[6:])
  kwargs["segment_ids"] = None
  gradients = []
  for implementation in ("staged", "fused", "xla"):
    op = (
        functools.partial(api.kimi_delta_attention, implementation="xla")
        if implementation == "xla"
        else mosaic.PallasMosaicTpuKimiDeltaAttention(
            config=mosaic.Config(fuse_forward=True, 
                fuse_backward=implementation == "fused",
                rematerialize_for_backward=rematerialize,
            )
        )
    )
    output, pullback = jax.vjp(
        functools.partial(forward_tests._call, op, kwargs), *args
    )
    gradients.append(
        pullback(jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, output))
    )
  forward_tests._assert_close(gradients[1], gradients[0], tolerance=0.002)
  forward_tests._assert_close(gradients[1][3], gradients[2][3], tolerance=0.05)
  forward_tests._assert_close(
      gradients[1][6:], gradients[2][6:], tolerance=0.05
  )


def test_rematerialization_keeps_staged_path(monkeypatch):
  args, kwargs = forward_tests._inputs(jnp.float32, False)

  def unexpected_fusion(*args, **kwargs):
    raise AssertionError("State-rematerialization must retain its staged path")

  monkeypatch.setattr(
      kernels, "_saved_state_backward_kernel", unexpected_fusion
  )
  op = mosaic.PallasMosaicTpuKimiDeltaAttention(
      config=mosaic.Config(fuse_forward=True, fuse_backward=True, rematerialize_for_backward=True),
  )
  output, pullback = jax.vjp(
      functools.partial(forward_tests._call, op, kwargs), *args
  )
  grads = pullback(jax.tree.map(jnp.ones_like, output))
  forward_tests._assert_close(grads, grads, tolerance=0.0)


def test_backward_config_is_forwarded_to_vjp():
  config = mosaic.Config(fuse_forward=True, fuse_backward=False)
  attention = mosaic.PallasMosaicTpuKimiDeltaAttention(config=config)
  assert attention.vjp.config == config


@pytest.mark.parametrize("key_dim,value_dim", [(128, 256), (256, 128)])
def test_multiple_heads_batches_without_state(key_dim, value_dim):
  keys = jax.random.split(jax.random.key(91), 3)
  q = (jax.random.normal(keys[0], (2, 2, 64, key_dim)) * 0.05).astype(
      jnp.bfloat16
  )
  k = (jax.random.normal(keys[1], q.shape) * 0.05).astype(q.dtype)
  v = jax.random.normal(keys[2], (2, 2, 64, value_dim)).astype(q.dtype)
  inputs = (
      q,
      k,
      v,
      jnp.full_like(q, -0.01),
      jnp.full(q.shape[:-1], 0.5, q.dtype),
  )
  results = []
  for fuse in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(fuse_forward=True, fuse_backward=fuse)
    )
    output, pullback = jax.vjp(op, *inputs)
    results.append(pullback(jax.tree.map(jnp.ones_like, output)))
  forward_tests._assert_close(results[1], results[0], tolerance=0.002)
