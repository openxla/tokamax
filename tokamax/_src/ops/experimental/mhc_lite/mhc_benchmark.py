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
"""Benchmarks mHC-lite Pallas kernels: MaxText vs. our copy (GUM 1.4 sizes)."""

import importlib
import itertools
import time

from absl.testing import absltest
import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.mhc_lite import common as songbaie_common
from tokamax._src.ops.experimental.mhc_lite import mhc_kernels_fwd as songbaie_fwd

xprof_session = None

# The MaxText kernels target is only visible inside //third_party/py/maxtext, so
# it is reached through //third_party/py/maxtext and imported dynamically.
_MAXTEXT_MHC = "maxtext.src.maxtext.kernels.mhc."

# GUM 1.4 configuration.
BATCH, SEQ, STREAMS, EMB = 128, 1536, 4, 512
EPS = 1e-6
DTYPE = jnp.bfloat16
COMPARE_MAXTEXT = False


def rms_norm(x):
  """Pre-norm of the wrapped (identity) branch, with unit scale."""
  with jax.named_scope("rms_norm"):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), -1, keepdims=True) + EPS)


@jax.jit
def rel_err(ref, new):
  """Max |ref - new| relative to max |ref|."""
  ref, new = ref.astype(jnp.float32), new.astype(jnp.float32)
  return jnp.max(jnp.abs(ref - new)) / (jnp.max(jnp.abs(ref)) + 1e-6)


def time_ms(fn, *args, iters=20):
  """Measures mean execution time in milliseconds over `iters` runs."""
  out = None
  t0 = time.perf_counter()
  for _ in range(iters):
    out = fn(*args)
  jax.block_until_ready(out)
  return (time.perf_counter() - t0) * 1e3 / iters


class MhcBenchmarkTest(absltest.TestCase):
  """Benchmark suite for mHC-lite Pallas TPU kernels."""

  def test_benchmark(self):
    """Benchmarks forward and forward+backward passes on GUM 1.4 dimensions."""
    k, d = STREAMS, EMB
    key_x, key_dy, key_pre, key_post, key_res = jax.random.split(
        jax.random.key(0), 5
    )
    x = jax.random.normal(key_x, (BATCH, SEQ, k, d), DTYPE)
    dy = jax.random.normal(key_dy, (BATCH, SEQ, k, d), DTYPE)
    perms = jnp.eye(k, dtype=DTYPE)[
        jnp.array(list(itertools.permutations(range(k))))
    ]
    n = perms.shape[0]  # k! permutation matrices.
    init = jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal")
    weight_fields = dict(
        norm_scale=jnp.ones((k * d,), DTYPE),
        w_pre=init(key_pre, (k * d, k), DTYPE),
        b_pre=jnp.zeros((k,), DTYPE),
        alpha_pre=jnp.full((1,), 0.01, DTYPE),
        w_post=init(key_post, (k * d, k), DTYPE),
        b_post=jnp.zeros((k,), DTYPE),
        alpha_post=jnp.full((1,), 0.01, DTYPE),
        w_res=init(key_res, (k * d, n), DTYPE),
        b_res=jnp.zeros((n,), DTYPE),
        alpha_res=jnp.full((1,), 0.01, DTYPE),
    )

    def make_variant(name, common, kernels):
      if name == "maxtext":
        maxtext_fields = dict(
            norm_scale=weight_fields["norm_scale"],
            pre_alpha=weight_fields["w_pre"],
            pre_bias=weight_fields["b_pre"],
            pre_scale=weight_fields["alpha_pre"],
            post_alpha=weight_fields["w_post"],
            post_bias=weight_fields["b_post"],
            post_scale=weight_fields["alpha_post"],
            res_alpha=weight_fields["w_res"],
            res_bias=weight_fields["b_res"],
            res_scale=weight_fields["alpha_res"],
        )
        weights = common.MhcWeights(**maxtext_fields)
      else:
        weights = common.MhcWeights(**weight_fields)
      cfg = common.MhcKernelConfig(
          block_size=256, bwd_block_size=256, rms_epsilon=EPS
      )

      # Calls mhc_kernels_fwd directly: api.pre wraps the permutations in
      # stop_gradient, which yields a tracer under jit and fails the
      # nondiff_argnums check of mhc_kernels_fwd._pre_op.
      def fwd(inputs, w):
        with jax.named_scope(f"{name}.pre"):
          layer_in, ctx = kernels.pre(inputs, w, perms, config=cfg)
        layer_out = rms_norm(layer_in)  # Identity branch after the pre-norm.
        with jax.named_scope(f"{name}.post"):
          return kernels.post(layer_out, ctx, config=cfg)

      def fwd_bwd(inputs, w, g):
        out, vjp_fn = jax.vjp(fwd, inputs, w)
        return out, vjp_fn(g)

      # Named so XProf shows jit_mhc_lite_<name>_fwd[_bwd].
      fwd.__name__ = f"mhc_lite_{name}_fwd"
      fwd_bwd.__name__ = f"mhc_lite_{name}_fwd_bwd"
      return f"mhc_lite_{name}", jax.jit(fwd), jax.jit(fwd_bwd), weights

    variants = [make_variant("songbaie", songbaie_common, songbaie_fwd)]
    if COMPARE_MAXTEXT:
      maxtext_common = importlib.import_module(_MAXTEXT_MHC + "common")
      maxtext_fwd = importlib.import_module(_MAXTEXT_MHC + "mhc_kernels_fwd")
      variants.insert(0, make_variant("maxtext", maxtext_common, maxtext_fwd))

    print("Warming up...")
    for _, fwd_fn, fwd_bwd_fn, w in variants:
      for _ in range(3):
        jax.block_until_ready((fwd_fn(x, w), fwd_bwd_fn(x, w, dy)))

    if COMPARE_MAXTEXT:
      # Our copy must match MaxText on the output, dx and all weight gradients.
      ref, new = (jax.tree.leaves(fb(x, w, dy)) for _, _, fb, w in variants)
      err = max(float(rel_err(a, b)) for a, b in zip(ref, new, strict=True))
      print(f"Max rel. error vs maxtext (out, dx, dweights): {err:.3e}")
      self.assertLess(err, 1e-2)
      del ref, new

    # Timed outside the XProf session so tracing overhead is excluded.
    tokens = BATCH * SEQ
    times = []
    for name, fwd_fn, fwd_bwd_fn, w in variants:
      fwd_ms, fb_ms = time_ms(fwd_fn, x, w), time_ms(fwd_bwd_fn, x, w, dy)
      times.append((fwd_ms, fb_ms))
      print(
          f"[{name}] Fwd: {fwd_ms:.3f} ms ({tokens / fwd_ms / 1e3:.2f}M tok/s)"
          f" | Fwd+Bwd: {fb_ms:.3f} ms (Bwd: {fb_ms - fwd_ms:.3f} ms,"
          f" {tokens / fb_ms / 1e3:.2f}M tok/s)"
      )
    if COMPARE_MAXTEXT:
      ref_fwd, ref_fb = times[0]
      new_fwd, new_fb = times[1]
      print(
          f"Speedup songbaie vs maxtext: Fwd {ref_fwd / new_fwd:.3f}x |"
          f" Fwd+Bwd {ref_fb / new_fb:.3f}x"
      )

    if xprof_session is not None:
      print("Running benchmark with xprof...")
      session = xprof_session.XprofSession()
      session.start_session(
          trace_mode="TRACE_COMPUTE_AND_DMA",
          host_trace_level=1,
          enable_python_tracer=False,
      )
      for name, fwd_fn, fwd_bwd_fn, w in variants:
        with jax.profiler.TraceAnnotation(name):
          jax.block_until_ready((fwd_fn(x, w), fwd_bwd_fn(x, w, dy)))
      session_id = session.end_session_and_get_session_id()
      print("XProf URL:", f"http://xprof/trace_viewer/{session_id}")


if __name__ == "__main__":
  absltest.main()
