# Copyright 2026 Google LLC
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
"""Microbenchmark comparing static HC, mHC-lite XLA reference, and Pallas TPU kernel."""

import functools
import itertools
import time
from typing import Literal

from absl.testing import absltest
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.mhc_lite import api as mhc_api
from tokamax._src.ops.experimental.mhc_lite import common as mhc_common
from tokamax._src.ops.experimental.mhc_lite import mhc_xla

xprof_session = None

jax.config.parse_flags_with_absl()

# Default benchmark dimensions.
BATCH, SEQ, STREAMS, EMB = 128, 1536, 4, 512
EPS = 1e-6
DTYPE = jnp.bfloat16


def rms_norm(x: jax.Array) -> jax.Array:
  """Pre-norm of the wrapped (identity) branch, with unit scale."""
  with jax.named_scope("rms_norm"):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), -1, keepdims=True) + EPS)


@jax.jit
def rel_err(reference: jax.Array, candidate: jax.Array) -> jax.Array:
  """Max |reference - candidate| relative to max |reference|."""
  reference = reference.astype(jnp.float32)
  candidate = candidate.astype(jnp.float32)
  return jnp.max(jnp.abs(reference - candidate)) / (
      jnp.max(jnp.abs(reference)) + 1e-6
  )


def time_ms(func, *args, num_iterations: int = 20) -> float:
  """Measures mean execution time in milliseconds over `num_iterations` runs."""
  output = None
  start_time = time.perf_counter()
  for _ in range(num_iterations):
    output = func(*args)
  jax.block_until_ready(output)
  return (time.perf_counter() - start_time) * 1e3 / num_iterations


def _variant_fwd(
    name: str,
    implementation: mhc_api.Implementation | Literal["hc"],
    permutations: jax.Array | None,
    config: mhc_common.MhcKernelConfig,
    inputs: jax.Array,
    variant_weights: mhc_common.MhcWeights | mhc_common.HcWeights,
) -> jax.Array:
  """Runs the forward pass for a benchmark variant."""
  if implementation == "hc":
    assert isinstance(variant_weights, mhc_common.HcWeights)
    with jax.named_scope(f"{name}.pre"):
      layer_input, context = mhc_api.hc_pre(inputs, variant_weights)
    layer_output = rms_norm(layer_input)
    with jax.named_scope(f"{name}.post"):
      return mhc_api.hc_post(layer_output, context)
  assert isinstance(variant_weights, mhc_common.MhcWeights)
  assert permutations is not None
  with jax.named_scope(f"{name}.pre"):
    layer_input, context = mhc_api.mhc_lite_pre(
        inputs,
        variant_weights,
        permutations,
        config=config,
        implementation=implementation,
    )
  layer_output = rms_norm(layer_input)
  with jax.named_scope(f"{name}.post"):
    return mhc_api.mhc_lite_post(layer_output, context, config=config)


def _fwd_bwd(name: str, fwd_fn, inputs, weights, grad):
  """Runs the forward and backward VJP pass for `fwd_fn`."""
  output, vjp_fn = jax.vjp(fwd_fn, inputs, weights)
  with jax.named_scope(f"{name}.bwd"):
    return output, vjp_fn(grad)


def _compile_variant(name: str, fwd_fn, weights):
  """Builds JIT-compiled forward and forward+backward functions for `fwd_fn`."""

  def _named_fwd(inputs, variant_weights):
    return fwd_fn(inputs, variant_weights)

  _named_fwd.__name__ = f"{name}_fwd"
  _named_fwd.__qualname__ = f"{name}_fwd"

  def _named_fwd_bwd(inputs, variant_weights, grad):
    return _fwd_bwd(name, fwd_fn, inputs, variant_weights, grad)

  _named_fwd_bwd.__name__ = f"{name}_fwd_bwd"
  _named_fwd_bwd.__qualname__ = f"{name}_fwd_bwd"

  return (
      f"mhc_lite_{name}",
      jax.jit(_named_fwd),
      jax.jit(_named_fwd_bwd),
      weights,
  )


def _fused_boundary_fwd(
    name: str,
    implementation: mhc_api.Implementation,
    permutations: jax.Array,
    config: mhc_common.MhcKernelConfig,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    variant_weights: mhc_common.MhcWeights,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Runs one boundary-fused sublayer step (`rms_norm` + `mhc_lite_post_and_pre`)."""
  layer_input, x_prev, h_post_prev, residual_prev = inputs
  layer_output = rms_norm(layer_input)
  context = mhc_common.MHCContext(
      x=x_prev,
      h_post=h_post_prev,
      residual=residual_prev,
      implementation=implementation,
  )
  with jax.named_scope(f"{name}.post_and_pre"):
    next_layer_input, next_context = mhc_api.mhc_lite_post_and_pre(
        layer_output,
        context,
        variant_weights,
        permutations,
        config=config,
        implementation=implementation,
    )
  return (
      next_layer_input,
      next_context.x.reshape(x_prev.shape),
      next_context.h_post,
      next_context.residual.astype(residual_prev.dtype).reshape(
          residual_prev.shape
      ),
  )


class MhcBenchmarkTest(absltest.TestCase):
  """Microbenchmark suite comparing static HC, mHC-lite XLA, and Pallas TPU."""

  def test_benchmark(self):
    """Benchmarks forward and forward+backward passes across variants."""
    streams, embedding = STREAMS, EMB
    tokens = BATCH * SEQ
    key_x, key_dy, key_pre, key_post, key_res = jax.random.split(
        jax.random.key(0), 5
    )
    x = jax.random.normal(key_x, (BATCH, SEQ, streams, embedding), DTYPE)
    dy = jax.random.normal(key_dy, (BATCH, SEQ, streams, embedding), DTYPE)
    x_2d = x.reshape(tokens, streams * embedding)
    dy_2d = dy.reshape(tokens, streams * embedding)
    permutations = jnp.eye(streams, dtype=DTYPE)[
        jnp.array(list(itertools.permutations(range(streams))))
    ]
    num_permutations = permutations.shape[0]
    initializer = jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal")
    mhc_weights = mhc_common.MhcWeights(
        norm_scale=jnp.ones((streams * embedding,), DTYPE),
        w_pre=initializer(key_pre, (streams * embedding, streams), DTYPE),
        b_pre=jnp.zeros((streams,), DTYPE),
        alpha_pre=jnp.full((1,), 0.01, DTYPE),
        w_post=initializer(key_post, (streams * embedding, streams), DTYPE),
        b_post=jnp.zeros((streams,), DTYPE),
        alpha_post=jnp.full((1,), 0.01, DTYPE),
        w_res=initializer(
            key_res, (streams * embedding, num_permutations), DTYPE
        ),
        b_res=jnp.zeros((num_permutations,), DTYPE),
        alpha_res=jnp.full((1,), 0.01, DTYPE),
    )
    hc_weights = mhc_common.HcWeights(
        w_pre=(
            initializer(key_pre, (streams, 1), DTYPE).squeeze(-1)
            / float(streams)
        ),
        w_post=initializer(key_post, (streams, 1), DTYPE),
        w_res=jnp.eye(streams, dtype=DTYPE),
    )
    config = mhc_common.MhcKernelConfig(
        block_size=256,
        fwd_pipeline_blocks=2,
        bwd_block_size=256,
        rms_epsilon=EPS,
    )

    hc_variant = _compile_variant(
        "hc_xla",
        functools.partial(_variant_fwd, "hc_xla", "hc", None, config),
        hc_weights,
    )
    xla_variant = _compile_variant(
        "xla_ref",
        functools.partial(_variant_fwd, "xla_ref", "xla", permutations, config),
        mhc_weights,
    )
    pallas_2d_variant = _compile_variant(
        "pallas_tpu_2d",
        functools.partial(
            _variant_fwd, "pallas_tpu_2d", "mosaic_tpu", permutations, config
        ),
        mhc_weights,
    )

    # Prepare boundary-fused inputs and cotangents from Stage-1 pre_xla.
    layer_input_0, ctx_0 = mhc_xla.pre_xla(
        x, mhc_weights, permutations, config=config
    )
    boundary_inputs = (
        layer_input_0,
        x_2d,
        ctx_0.h_post,
        ctx_0.residual.astype(DTYPE).reshape(tokens, streams * streams),
    )
    _, xla_boundary_fwd, xla_boundary_fwd_bwd, _ = _compile_variant(
        "xla_post_and_pre",
        functools.partial(
            _fused_boundary_fwd,
            "xla_post_and_pre",
            "xla",
            permutations,
            config,
        ),
        mhc_weights,
    )
    next_li_0, next_x_0, next_hp_0, next_res_0 = xla_boundary_fwd(
        boundary_inputs, mhc_weights
    )
    _, tail_vjp = jax.vjp(
        lambda li, xm, hp, res: mhc_xla.post_xla(
            rms_norm(li),
            mhc_common.MHCContext(
                x=xm, h_post=hp, residual=res, implementation="xla"
            ),
        ),
        next_li_0,
        next_x_0,
        next_hp_0,
        next_res_0,
    )
    boundary_grad = tail_vjp(dy)
    del next_li_0, next_x_0, next_hp_0, next_res_0, tail_vjp
    fused_variant = _compile_variant(
        "pallas_post_and_pre",
        functools.partial(
            _fused_boundary_fwd,
            "pallas_post_and_pre",
            "mosaic_tpu",
            permutations,
            config,
        ),
        mhc_weights,
    )

    benchmarks = [
        (name, fwd_fn, fwd_bwd_fn, inp, weights, grad)
        for (name, fwd_fn, fwd_bwd_fn, weights), inp, grad in [
            (hc_variant, x, dy),
            (xla_variant, x, dy),
            (pallas_2d_variant, x_2d, dy_2d),
            (fused_variant, boundary_inputs, boundary_grad),
        ]
    ]

    print("Warming up...")
    for _, fwd_fn, fwd_bwd_fn, inp, weights, grad in benchmarks:
      for _ in range(3):
        jax.block_until_ready((
            fwd_fn(inp, weights),
            fwd_bwd_fn(inp, weights, grad),
        ))

    # Verify numerical parity between mHC-lite JAX reference and Pallas TPU.
    _, _, xla_fwd_bwd, _ = xla_variant
    _, _, fwd_bwd_2d_fn, _ = pallas_2d_variant
    _, _, fused_fwd_bwd_fn, _ = fused_variant
    reference_leaves = jax.tree.leaves(xla_fwd_bwd(x, mhc_weights, dy))
    candidate_2d_leaves = jax.tree.leaves(
        fwd_bwd_2d_fn(x_2d, mhc_weights, dy_2d)
    )
    max_error_2d = max(
        float(rel_err(ref_leaf.reshape(new_leaf.shape), new_leaf))
        for ref_leaf, new_leaf in zip(
            reference_leaves, candidate_2d_leaves, strict=True
        )
    )
    print(f"Max rel. error pallas_tpu_2d vs xla_ref: {max_error_2d:.3e}")
    self.assertLess(max_error_2d, 1e-2)
    del reference_leaves, candidate_2d_leaves

    ref_boundary_leaves = jax.tree.leaves(
        xla_boundary_fwd_bwd(boundary_inputs, mhc_weights, boundary_grad)
    )
    cand_boundary_leaves = jax.tree.leaves(
        fused_fwd_bwd_fn(boundary_inputs, mhc_weights, boundary_grad)
    )
    boundary_errors = [
        float(rel_err(ref_leaf, new_leaf))
        for ref_leaf, new_leaf in zip(
            ref_boundary_leaves, cand_boundary_leaves, strict=True
        )
    ]
    max_boundary_error = max(boundary_errors)
    print(
        "Max rel. error pallas_post_and_pre vs xla_post_and_pre:"
        f" {max_boundary_error:.3e} (per-leaf:"
        f" {[f'{e:.3e}' for e in boundary_errors]})"
    )
    del ref_boundary_leaves, cand_boundary_leaves

    # Timed outside the XProf session so tracing overhead is excluded.
    times = {}
    for name, fwd_fn, fwd_bwd_fn, inp, weights, grad in benchmarks:
      fwd_ms = time_ms(fwd_fn, inp, weights)
      fwd_bwd_ms = time_ms(fwd_bwd_fn, inp, weights, grad)
      times[name] = (fwd_ms, fwd_bwd_ms)
      print(
          f"[{name}] Fwd: {fwd_ms:.3f} ms ({tokens / fwd_ms / 1e3:.2f}M tok/s)"
          f" | Fwd+Bwd: {fwd_bwd_ms:.3f} ms (Bwd: {fwd_bwd_ms - fwd_ms:.3f} ms,"
          f" {tokens / fwd_bwd_ms / 1e3:.2f}M tok/s)"
      )

    hc_fwd, hc_fwd_bwd = times["mhc_lite_hc_xla"]
    ref_fwd, ref_fwd_bwd = times["mhc_lite_xla_ref"]
    fwd_2d_ms, fwd_bwd_2d_ms = times["mhc_lite_pallas_tpu_2d"]
    fused_fwd_ms, fused_fwd_bwd_ms = times["mhc_lite_pallas_post_and_pre"]
    hc_bwd = hc_fwd_bwd - hc_fwd
    ref_bwd = ref_fwd_bwd - ref_fwd
    bwd_2d_ms = fwd_bwd_2d_ms - fwd_2d_ms
    fused_bwd_ms = fused_fwd_bwd_ms - fused_fwd_ms
    print(
        f"Speedup pallas_tpu_2d vs xla_ref: Fwd {ref_fwd / fwd_2d_ms:.3f}x |"
        f" Bwd {ref_bwd / bwd_2d_ms:.3f}x |"
        f" Fwd+Bwd {ref_fwd_bwd / fwd_bwd_2d_ms:.3f}x"
    )
    print(
        "Speedup pallas_post_and_pre vs pallas_tpu_2d: Fwd"
        f" {fwd_2d_ms / fused_fwd_ms:.3f}x | Bwd"
        f" {bwd_2d_ms / fused_bwd_ms:.3f}x | Fwd+Bwd"
        f" {fwd_bwd_2d_ms / fused_fwd_bwd_ms:.3f}x"
    )
    print(
        "Ratio pallas_post_and_pre vs hc_xla: Fwd"
        f" {fused_fwd_ms / hc_fwd:.3f}x | Bwd {fused_bwd_ms / hc_bwd:.3f}x |"
        f" Fwd+Bwd {fused_fwd_bwd_ms / hc_fwd_bwd:.3f}x"
    )

    if xprof_session is not None:
      print("Running benchmark with xprof...")
      session = xprof_session.XprofSession()
      session.start_session(
          trace_mode="TRACE_COMPUTE_AND_DMA",
          host_trace_level=1,
          enable_python_tracer=True,
          enable_kernel_profiling=True,
      )
      for name, fwd_fn, fwd_bwd_fn, inp, weights, grad in benchmarks:
        with jax.profiler.TraceAnnotation(name):
          jax.block_until_ready((
              fwd_fn(inp, weights),
              fwd_bwd_fn(inp, weights, grad),
          ))
      session_id = session.end_session_and_get_session_id()
      print("XProf URL:", f"http://xprof/trace_viewer/{session_id}")

    self.assertLess(max_boundary_error, 2e-2)


if __name__ == "__main__":
  absltest.main()
