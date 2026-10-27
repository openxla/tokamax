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
"""Fused forward equivalence and compatibility with the existing backward."""

import functools
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokamax._src.ops.kda import api
from tokamax._src.ops.kda import common
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
from tokamax._src.ops.kda import pallas_mosaic_tpu_fwd_kernel as kernels


@pytest.fixture(autouse=True)
def interpret_on_cpu(monkeypatch):
  if jax.default_backend() == "tpu":
    yield
    return
  monkeypatch.setenv("PALLAS_INTERPRET", "1")
  monkeypatch.setattr(
      common.pltpu,
      "get_tpu_info",
      lambda: types.SimpleNamespace(
          vmem_capacity_bytes=32 * 1024**2,
          num_sublanes=8,
          num_lanes=128,
      ),
  )
  monkeypatch.setattr(
      mosaic.PallasMosaicTpuKimiDeltaAttention,
      "supported_on",
      lambda self, device: True,
  )
  # CUDA interpretation executes the kernels' FP32 contractions on the GPU.
  # Its default TF32 precision can change across the fused/staged JIT
  # boundaries, even though the native TPU schedules agree. Keep the
  # interpreter comparison in FP32; explicit BF16 casts and the numerical
  # tolerances stay unchanged. Native TPU execution keeps its own precision.
  with jax.default_matmul_precision("highest"):
    yield


def _inputs(dtype, packed, *, key_dim=128, heads=1, batch=1):
  heads, batch, tokens, value_dim = heads, batch, 128, 128
  keys = jax.random.split(jax.random.key(31), 8)

  def normal(key, shape):
    return jax.random.normal(key, shape).astype(dtype)

  shape = (heads, batch, tokens, key_dim)
  q = normal(keys[0], shape) * 0.1
  k = normal(keys[1], shape) * 0.1
  v = normal(keys[2], (heads, batch, tokens, value_dim))
  g = normal(keys[3], shape) * 0.1 if packed else -jnp.ones(shape, dtype) * 0.01
  beta = jax.nn.sigmoid(normal(keys[4], shape[:-1]))
  # Include a non-chunk-aligned boundary, padding, and an empty state slot.
  segment_ids = (
      jnp.broadcast_to(
          jnp.array([[1] * 17 + [2] * 79 + [0] * 32], jnp.int32),
          (batch, tokens),
      )
      if packed
      else None
  )
  states = 3 if packed else 1
  h0 = 0.1 * normal(keys[5], (batch, states, heads, key_dim, value_dim))
  a_log = normal(keys[6], (heads,)) * 0.1 if packed else None
  bias = normal(keys[7], (heads * key_dim,)) * 0.1 if packed else None
  args = (q, k, v, g, beta, h0, a_log, bias)
  kwargs = dict(
      segment_ids=segment_ids,
      output_final_state=True,
      use_gate_in_kernel=packed,
      lower_bound=-1.0 if packed else None,
      use_qk_l2norm=packed,
  )
  return args, kwargs


def _call(op, kwargs, q, k, v, g, beta, h0, a_log, bias):
  return op(
      q,
      k,
      v,
      g,
      beta,
      initial_state=h0,
      a_log=a_log,
      delta_time_bias=bias,
      **kwargs,
  )


def _assert_close(actual, expected, *, tolerance):
  assert jax.tree.structure(actual) == jax.tree.structure(expected)
  for (path, a), (_, b) in zip(
      jax.tree_util.tree_flatten_with_path(actual)[0],
      jax.tree_util.tree_flatten_with_path(expected)[0],
      strict=True,
  ):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    assert np.isfinite(a).all()
    assert np.isfinite(b).all()
    np.testing.assert_allclose(
        a, b, rtol=tolerance, atol=tolerance,
        err_msg=f"KDA result leaf {jax.tree_util.keystr(path)}",
    )


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("rematerialize", [False, True])
def test_forward_and_existing_vjp(dtype, packed, rematerialize):
  args, kwargs = _inputs(dtype, packed)
  results = []
  for fuse in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(
            fuse_forward=fuse, rematerialize_for_backward=rematerialize
        ),
    )
    call = functools.partial(_call, op, kwargs)
    # Forward-only tracing can drop residual outputs; VJP tracing retains them.
    forward = jax.jit(call)(*args)
    output, pullback = jax.vjp(call, *args)
    cotangents = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, output)
    gradients = pullback(cotangents)
    results.append((forward, output, gradients))
  tolerance = 0.002 if dtype == jnp.bfloat16 else 1e-5
  _assert_close(results[1][:2], results[0][:2], tolerance=tolerance)

  reference = functools.partial(api.kimi_delta_attention, implementation="xla")
  expected = _call(reference, kwargs, *args)
  _assert_close(
      results[1][0],
      expected,
      tolerance=0.05 if dtype == jnp.bfloat16 else 0.002,
  )

  _assert_close(results[1][2], results[0][2], tolerance=tolerance)
  if packed:
    for gradients in (results[0][2], results[1][2]):
      for gradient in gradients[:5]:
        np.testing.assert_array_equal(np.asarray(gradient[:, :, 96:]), 0)
      np.testing.assert_allclose(np.asarray(gradients[5][:, 2]), 0.1)


@pytest.mark.parametrize("rematerialize", [False, True])
def test_residual_contract(rematerialize):
  args, kwargs = _inputs(jnp.bfloat16, True)
  q, k, v, g, beta, h0, a_log, bias = args
  prepared = mosaic.PallasMosaicTpuKimiDeltaAttention._preprocess_inputs(
      q,
      k,
      v,
      g,
      beta,
      initial_state=h0,
      output_final_state=True,
      use_qk_l2norm=True,
      use_gate_in_kernel=True,
      segment_ids=kwargs["segment_ids"],
      context_parallel_metadata=None,
      chunk_size=64,
      max_num_segments=3,
  )
  results = []
  for fuse in (False, True):
    results.append(
        kernels.chunk_kda_fwd_custom(
            prepared.q,
            prepared.k,
            prepared.v,
            prepared.g,
            prepared.beta,
            a_log=a_log,
            delta_time_bias=bias,
            scale=128**-0.5,
            initial_state=prepared.initial_state,
            output_final_state=True,
            use_gate_in_kernel=True,
            segment_ids=kwargs["segment_ids"],
            lower_bound=-1.0,
            disable_recompute=not rematerialize,
            return_residuals=True,
            cu_seqlens=prepared.cu_seqlens,
            aligned_cu_seqlens=prepared.aligned_cu_seqlens,
            chunk_indices=prepared.chunk_indices,
            aligned_segment_ids=prepared.aligned_segment_ids,
            q_rstd=prepared.q_rstd,
            k_rstd=prepared.k_rstd,
            fuse_forward=fuse,
        )
    )
  _assert_close(results[1], results[0], tolerance=0.002)
  residuals = results[1][1]
  assert (residuals.h is None) == rematerialize
  assert (residuals.g_cumsum is None) == rematerialize
  assert residuals.aqk.shape[-1] == 64


def test_non_aligned_width_uses_staged_forward(monkeypatch):
  args, kwargs = _inputs(jnp.float32, False, key_dim=64)

  def unexpected_fusion(*args, **kwargs):
    raise AssertionError("Non-aligned widths must use the staged kernel")

  monkeypatch.setattr(kernels, "chunk_kda_fwd_fused", unexpected_fusion)
  actual = _call(mosaic.PallasMosaicTpuKimiDeltaAttention(), kwargs, *args)
  reference = functools.partial(api.kimi_delta_attention, implementation="xla")
  _assert_close(actual, _call(reference, kwargs, *args), tolerance=0.002)


@pytest.mark.parametrize("packed", [False, True])
def test_forward_without_state(packed):
  args, kwargs = _inputs(jnp.bfloat16, packed)
  args = (*args[:5], None, *args[6:])
  kwargs["output_final_state"] = False
  if packed:
    kwargs["max_num_segments"] = 3
  results = []
  for fuse in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(fuse_forward=fuse),
    )
    results.append(jax.jit(functools.partial(_call, op, kwargs))(*args))
  _assert_close(results[1], results[0], tolerance=0.002)
  assert results[1][1] is None


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("packed", [False, True])
def test_multi_head_forward_and_vjp(heads, packed):
  """Fused and staged kernels must agree with XLA for multiple heads.

  Regression test: the suite only exercised ``heads=1``, where Mosaic's
  (8, 128) tiling pads the singleton head dimension to eight tiles and
  masked the multi-head code path entirely.  Uses a nontrivial mini batch
  to keep the head and batch axes distinct.
  """
  args, kwargs = _inputs(jnp.bfloat16, packed, heads=heads, batch=2)
  results = []
  for fuse in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(fuse_forward=fuse, rematerialize_for_backward=True)
    )
    call = functools.partial(_call, op, kwargs)
    output, pullback = jax.vjp(call, *args)
    cotangents = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, output)
    gradients = pullback(cotangents)
    results.append((output, gradients))
  _assert_close(results[1], results[0], tolerance=0.002)

  reference = functools.partial(api.kimi_delta_attention, implementation="xla")
  expected_output, _ = jax.vjp(
      functools.partial(_call, reference, kwargs), *args
  )
  _assert_close(
      results[1][0],
      expected_output,
      tolerance=0.05,
  )


@pytest.mark.parametrize("rematerialize", [False, True])
def test_fixed_raw_gate_parameter_gradients(rematerialize):
  args, kwargs = _inputs(jnp.bfloat16, True)
  args = (*args[:5], args[5][:, :1], *args[6:])
  kwargs["segment_ids"] = None
  results = []
  for fuse in (False, True):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config(
            fuse_forward=fuse,
            rematerialize_for_backward=rematerialize,
        ),
    )
    output, pullback = jax.vjp(functools.partial(_call, op, kwargs), *args)
    results.append(
        pullback(jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, output))
    )
  _assert_close(results[1], results[0], tolerance=0.002)

  reference = functools.partial(api.kimi_delta_attention, implementation="xla")
  output, pullback = jax.vjp(functools.partial(_call, reference, kwargs), *args)
  expected = pullback(jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, output))
  # Compare gate and gate-parameter gradients against autodiff as well as
  # against the unchanged staged custom VJP.
  _assert_close(results[1][3], expected[3], tolerance=0.05)
  _assert_close(results[1][6:], expected[6:], tolerance=0.05)
