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
"""Small, reproducible staged-vs-megakernel TPU latency comparisons.

Run from the repository root, for example:
  python -m tokamax.benchmarks.kda_megakernel_compare --suite forward

The four suites correspond to the forward, backward, CP backward and inference
PRs. Later suites are only available on branches containing those features.
To compare with openxla/tokamax PR #1103, check out its pinned head in a
separate tree, copy this script into that tree, and pass ``--baseline``.
"""

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import tokamax
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic


def _report(name, fn, args, iterations):
  result = tokamax.benchmark(
      fn, args, iterations=iterations, method="wallclock"
  )
  print(json.dumps({
      "case": name,
      "device": jax.devices()[0].device_kind,
      "jax": jax.__version__,
      "median_ms": result.median_evaluation_time_ms,
      "samples_ms": result.evaluation_times_ms,
      "compile_ms": result.compile_time_ms,
      "peak_memory_mb": result.peak_memory_mb,
  }), flush=True)


def _training(suite, iterations, baseline, heads, tokens, num_segments):
  # Non-64-aligned boundaries exercise native packed I/O.
  batch, dim = 1, 128
  key = jax.random.key(19)
  q, k, v = [
      jax.random.normal(x, (heads, batch, tokens, dim), jnp.bfloat16)
      for x in jax.random.split(key, 3)
  ]
  example = dict(
      query=q,
      key=k,
      value=v,
      gate=jnp.full_like(q, -0.01),
      beta=jnp.full((heads, batch, tokens), 0.5, jnp.bfloat16),
      a_log=jnp.zeros((heads,), jnp.float32),
      delta_time_bias=jnp.full((heads * dim,), -2.0, jnp.float32),
      use_qk_l2norm=True,
      use_gate_in_kernel=True,
      lower_bound=-5.0,
      output_final_state=False,
  )
  if suite in ("forward-packed", "backward-packed"):
    if (tokens, num_segments) == (1024, 4):
      boundaries = (239, 496, 737, 1024)
      lengths = np.diff((0, *boundaries))
    else:
      segment_len, remainder = divmod(tokens, num_segments)
      lengths = np.full(num_segments, segment_len, dtype=np.int32)
      lengths[:remainder] += 1
    segment_ids = np.repeat(
        np.arange(1, num_segments + 1, dtype=np.int32), lengths
    )
    example["segment_ids"] = jnp.asarray(segment_ids[None, :])
    example["max_num_segments"] = num_segments

  if baseline:
    configs = [("upstream-1103", mosaic.Config())]
    mode = "forward" if suite.startswith("forward") else "forward_and_vjp"
  elif suite == "forward":
    configs = [("staged", mosaic.Config(fuse_forward=False)),
               ("fused", mosaic.Config(fuse_forward=True))]
    mode = "forward"
  elif suite == "forward-packed":
    configs = [("aligned", mosaic.Config(fuse_forward=True)),
               ("native-packed", mosaic.Config(
                   fuse_forward=True, packed_forward=True,
                   packed_output=True))]
    mode = "forward"
  elif suite == "backward":
    configs = [("staged", mosaic.Config(fuse_forward=True, fuse_backward=False)),
               ("fused", mosaic.Config(fuse_forward=True, fuse_backward=True))]
    mode = "forward_and_vjp"
  else:
    configs = [("aligned", mosaic.Config(fuse_forward=True, fuse_backward=True)),
               ("native-packed", mosaic.Config(
                   fuse_forward=True, fuse_backward=True, packed_forward=True,
                   packed_output=True, packed_backward=True,
                   packed_gradients=True))]
    mode = "forward_and_vjp"
  for name, config in configs:
    fn, args = tokamax.standardize_function(
        mosaic.PallasMosaicTpuKimiDeltaAttention(config=config),
        kwargs=example, mode=mode)
    _report(f"{suite}/{name}", fn, args, iterations)


def _cp(iterations, baseline, tokens):
  from jax.sharding import Mesh, PartitionSpec as P
  from tokamax._src import jaxtyping
  from tokamax._src.ops.kda.cp_utils import ContextParallelMetadata

  cp_size = 4
  if jax.device_count() < cp_size:
    raise RuntimeError("CP benchmark requires four TPU devices")
  mesh = Mesh(np.array(jax.devices()[:cp_size]), ("context",))
  meta = ContextParallelMetadata(mesh=mesh, axis_name="context")
  heads, batch, dim = 8, 2, 128
  keys = jax.random.split(jax.random.key(97), 3)
  q, k, v = [
      jax.random.normal(x, (heads, batch, tokens, dim), jnp.bfloat16)
      for x in keys
  ]
  g = jnp.full_like(q, -0.01)
  beta = jnp.full(q.shape[:-1], 0.5)
  segments = jnp.ones((batch, tokens), jnp.int32)
  spec = P(None, None, "context", None)
  for fused in ((False,) if baseline else (False, True)):
    op = mosaic.PallasMosaicTpuKimiDeltaAttention(
        config=mosaic.Config() if baseline else
        mosaic.Config(cp_megakernel=fused))

    def local(q, k, v, g, beta, segments):
      def forward(q, k, v, g, beta):
        return op(q, k, v, g, beta, segment_ids=segments,
                  max_num_segments=1, context_parallel_metadata=meta)[0]
      out, back = jax.vjp(forward, q, k, v, g, beta)
      return back(jnp.ones_like(out))

    with jaxtyping.disable_jaxtyping(), jax.set_mesh(mesh):
      mapped = jax.shard_map(
          local, mesh=mesh,
          in_specs=(spec,) * 4 + (P(None, None, "context"),
                                   P(None, "context")),
          out_specs=(spec,) * 4 + (P(None, None, "context"),),
          check_vma=False)
      _report(f"cp-t{tokens}/" + ("upstream-1103" if baseline else
              ("megakernel" if fused else "staged")),
              lambda x: mapped(*x), (q, k, v, g, beta, segments),
              iterations)


def _inference(iterations, tokens, baseline):
  from tokamax._src.ops.kda import api
  if not baseline:
    from tokamax._src.ops.kda import inference

  heads, dim = 8, 128
  q, k, v, g = [
      jax.random.normal(x, (1, tokens, heads, dim), jnp.bfloat16)
      for x in jax.random.split(jax.random.key(19), 4)
  ]
  beta = jnp.full((1, tokens, heads), 0.5, jnp.bfloat16)
  ids = jnp.where(jnp.arange(tokens)[None, :] < tokens // 2 - 17,
                  1, 2).astype(jnp.int32)
  kwargs = dict(segment_ids=ids, a_log=jnp.zeros((heads,)),
                delta_time_bias=jnp.full((heads * dim,), -2.0),
                use_gate_in_kernel=True, lower_bound=-5.0,
                max_num_segments=2)

  def existing_mosaic_forward(x):
    q, k, v, g, beta = x
    return api.kimi_delta_attention(
        *(z.transpose(2, 0, 1, 3) for z in (q, k, v, g)),
        beta.transpose(2, 0, 1), **kwargs, use_qk_l2norm=True,
        implementation="mosaic")

  def native(x):
    return inference.kimi_delta_attention_inference(
        *x, **kwargs, use_qk_l2norm_in_kernel=True)

  args = (q, k, v, g, beta)
  _report(f"inference-t{tokens}/" + (
      "upstream-1103" if baseline else "existing-mosaic-forward"),
          existing_mosaic_forward, args, iterations)
  if not baseline:
    _report(f"inference-t{tokens}/native", native, args, iterations)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--suite", required=True, choices=(
      "forward", "forward-packed", "backward", "backward-packed", "cp",
      "inference"))
  parser.add_argument("--iterations", type=int, default=10)
  parser.add_argument("--tokens", type=int, default=512,
                      help="Inference sequence length (default: 512)")
  parser.add_argument("--training-tokens", type=int, default=1024,
                      help="Training sequence length (default: 1024)")
  parser.add_argument("--heads", type=int, default=8,
                      help="Training head count (default: 8)")
  parser.add_argument("--num-segments", type=int, default=4,
                      help="Packed training segment count (default: 4)")
  parser.add_argument("--cp-tokens", type=int, default=512,
                      help="Global CP sequence length (default: 512)")
  parser.add_argument("--baseline", action="store_true",
                      help="Run only the default KDA path, for PR #1103")
  args = parser.parse_args()
  if jax.default_backend() != "tpu":
    raise RuntimeError("KDA megakernel benchmarks require a TPU")
  if args.suite == "cp":
    _cp(args.iterations, args.baseline, args.cp_tokens)
  elif args.suite == "inference":
    _inference(args.iterations, args.tokens, args.baseline)
  else:
    if args.training_tokens <= 0 or args.heads <= 0:
      parser.error("--training-tokens and --heads must be positive")
    if not 0 < args.num_segments <= args.training_tokens:
      parser.error("--num-segments must be in [1, --training-tokens]")
    _training(args.suite, args.iterations, args.baseline, args.heads,
              args.training_tokens, args.num_segments)


if __name__ == "__main__":
  main()
