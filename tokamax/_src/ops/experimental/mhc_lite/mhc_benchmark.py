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

xprof_session = None

# GUM 1.4 configuration.
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


def _variant_fwd_bwd(
    name: str,
    implementation: mhc_api.Implementation | Literal["hc"],
    permutations: jax.Array | None,
    config: mhc_common.MhcKernelConfig,
    inputs: jax.Array,
    variant_weights: mhc_common.MhcWeights | mhc_common.HcWeights,
    grad: jax.Array,
):
  """Runs the forward and backward VJP pass for a benchmark variant."""
  fwd_fn = functools.partial(
      _variant_fwd, name, implementation, permutations, config
  )
  output, vjp_fn = jax.vjp(fwd_fn, inputs, variant_weights)
  return output, vjp_fn(grad)


def _make_variant(
    name: str,
    implementation: mhc_api.Implementation | Literal["hc"],
    weights: mhc_common.MhcWeights | mhc_common.HcWeights,
    permutations: jax.Array | None,
    config: mhc_common.MhcKernelConfig,
):
  """Builds JIT-compiled forward and forward+backward functions for a variant."""
  fwd_fn = functools.partial(
      _variant_fwd, name, implementation, permutations, config
  )
  fwd_bwd_fn = functools.partial(
      _variant_fwd_bwd, name, implementation, permutations, config
  )
  return f"mhc_lite_{name}", jax.jit(fwd_fn), jax.jit(fwd_bwd_fn), weights


class MhcBenchmarkTest(absltest.TestCase):
  """Microbenchmark suite comparing static HC, mHC-lite XLA, and Pallas TPU."""

  def test_benchmark(self):
    """Benchmarks forward and forward+backward passes on GUM 1.4 dimensions."""
    streams, embedding = STREAMS, EMB
    key_x, key_dy, key_pre, key_post, key_res = jax.random.split(
        jax.random.key(0), 5
    )
    x = jax.random.normal(key_x, (BATCH, SEQ, streams, embedding), DTYPE)
    dy = jax.random.normal(key_dy, (BATCH, SEQ, streams, embedding), DTYPE)
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

    variants = [
        _make_variant("hc_xla", "hc", hc_weights, None, config),
        _make_variant("xla_ref", "xla", mhc_weights, permutations, config),
        _make_variant(
            "pallas_tpu", "mosaic_tpu", mhc_weights, permutations, config
        ),
    ]

    print("Warming up...")
    for _, fwd_fn, fwd_bwd_fn, variant_weights in variants:
      for _ in range(3):
        jax.block_until_ready((
            fwd_fn(x, variant_weights),
            fwd_bwd_fn(x, variant_weights, dy),
        ))

    # Verify numerical parity between mHC-lite JAX reference and Pallas TPU.
    _, _, xla_fwd_bwd, xla_weights = variants[1]
    _, _, pallas_fwd_bwd, pallas_weights = variants[2]
    reference_leaves = jax.tree.leaves(xla_fwd_bwd(x, xla_weights, dy))
    candidate_leaves = jax.tree.leaves(pallas_fwd_bwd(x, pallas_weights, dy))
    max_error = max(
        float(rel_err(ref_leaf, new_leaf))
        for ref_leaf, new_leaf in zip(
            reference_leaves, candidate_leaves, strict=True
        )
    )
    print(f"Max rel. error vs xla_ref (out, dx, dweights): {max_error:.3e}")
    self.assertLess(max_error, 1e-2)
    del reference_leaves, candidate_leaves

    # Timed outside the XProf session so tracing overhead is excluded.
    tokens = BATCH * SEQ
    times = {}
    for name, fwd_fn, fwd_bwd_fn, variant_weights in variants:
      fwd_ms = time_ms(fwd_fn, x, variant_weights)
      fwd_bwd_ms = time_ms(fwd_bwd_fn, x, variant_weights, dy)
      times[name] = (fwd_ms, fwd_bwd_ms)
      print(
          f"[{name}] Fwd: {fwd_ms:.3f} ms ({tokens / fwd_ms / 1e3:.2f}M tok/s)"
          f" | Fwd+Bwd: {fwd_bwd_ms:.3f} ms (Bwd: {fwd_bwd_ms - fwd_ms:.3f} ms,"
          f" {tokens / fwd_bwd_ms / 1e3:.2f}M tok/s)"
      )

    hc_fwd, hc_fwd_bwd = times["mhc_lite_hc_xla"]
    ref_fwd, ref_fwd_bwd = times["mhc_lite_xla_ref"]
    new_fwd, new_fwd_bwd = times["mhc_lite_pallas_tpu"]
    hc_bwd = hc_fwd_bwd - hc_fwd
    ref_bwd = ref_fwd_bwd - ref_fwd
    new_bwd = new_fwd_bwd - new_fwd
    print(
        f"Speedup pallas_tpu vs xla_ref: Fwd {ref_fwd / new_fwd:.3f}x |"
        f" Bwd {ref_bwd / new_bwd:.3f}x |"
        f" Fwd+Bwd {ref_fwd_bwd / new_fwd_bwd:.3f}x"
    )
    print(
        f"Ratio pallas_tpu vs hc_xla: Fwd {new_fwd / hc_fwd:.3f}x |"
        f" Bwd {new_bwd / hc_bwd:.3f}x |"
        f" Fwd+Bwd {new_fwd_bwd / hc_fwd_bwd:.3f}x"
    )

    if xprof_session is not None:
      print("Running benchmark with xprof...")
      session = xprof_session.XprofSession()
      session.start_session(
          trace_mode="TRACE_COMPUTE_AND_DMA",
          host_trace_level=1,
          enable_python_tracer=False,
      )
      for name, fwd_fn, fwd_bwd_fn, variant_weights in variants:
        with jax.profiler.TraceAnnotation(name):
          jax.block_until_ready((
              fwd_fn(x, variant_weights),
              fwd_bwd_fn(x, variant_weights, dy),
          ))
      session_id = session.end_session_and_get_session_id()
      print("XProf URL:", f"http://xprof/trace_viewer/{session_id}")


if __name__ == "__main__":
  absltest.main()
