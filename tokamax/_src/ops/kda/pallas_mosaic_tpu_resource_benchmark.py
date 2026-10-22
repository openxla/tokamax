# Copyright 2026 Ant Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Same-input native TPU public forward/VJP resource and latency evidence."""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from tokamax._src import jaxtyping
from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
from tokamax._src.ops.kda.cp_utils import ContextParallelMetadata


def digest(x):
  return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True)
  parser.add_argument("--batch", type=int, default=2)
  parser.add_argument("--heads", type=int, default=8)
  parser.add_argument("--local-tokens", type=int, default=128)
  parser.add_argument("--cp", type=int, default=8)
  parser.add_argument("--capacity", type=int, default=32)
  parser.add_argument(
      "--route", choices=("plain", "bucket", "groups"), default="plain"
  )
  parser.add_argument("--group", type=int, default=4)
  parser.add_argument("--remat", action="store_true")
  parser.add_argument("--outer-remat", action="store_true")
  parser.add_argument("--over", action="store_true")
  parser.add_argument("--fixed", action="store_true")
  parser.add_argument("--repeat", type=int, default=20)
  parser.add_argument("--warmup", type=int, default=5)
  parser.add_argument("--save-arrays", action="store_true")
  parser.add_argument("--profile", action="store_true")
  parser.add_argument("--profile-repeat", type=int, default=20)
  parser.add_argument("--packed-backward", action="store_true")
  args = parser.parse_args()
  if jax.default_backend() != "tpu" or os.environ.get("PALLAS_INTERPRET"):
    raise RuntimeError("Native TPU execution is required")
  if args.cp > jax.device_count():
    raise RuntimeError("Insufficient devices; do not silently skip")
  root = Path(args.output)
  root.mkdir(parents=True, exist_ok=True)
  h, b, t, d = args.heads, args.batch, args.local_tokens * args.cp, 128
  rng = np.random.default_rng(619)

  def normal(shape, scale):
    return jnp.asarray(
        (rng.standard_normal(shape) * scale).astype(np.float32), jnp.bfloat16
    )

  q, k = (normal((h, b, t, d), 0.05) for _ in range(2))
  v, g = normal(q.shape, 0.05), normal(q.shape, 0.1)
  beta = jax.nn.sigmoid(normal(q.shape[:-1], 0.1))
  alog = jnp.asarray(rng.uniform(-1, 0, h), jnp.float32)
  bias = jnp.asarray(rng.uniform(-4, -3, h * d), jnp.float32)
  dout = normal(q.shape, 0.05)
  ids = np.zeros((b, t), np.int32)
  for rank in range(args.cp):
    start = rank * args.local_tokens
    # Last segment continues into the next rank's first segment.
    split = args.local_tokens // 2 + 1
    ids[:, start : start + split] = 2 * rank + 1
    ids[:, start + split : start + args.local_tokens] = 2 * rank + 3
  if args.over:
    start = (args.cp - 1) * args.local_tokens
    ids[
        -1, start + args.local_tokens // 4 : start + args.local_tokens // 2 + 1
    ] = (2 * args.cp)
  ids = jnp.asarray(ids)
  inputs = (q, k, v, g, beta, alog, bias, dout, ids)
  result = dict(
      arguments=vars(args),
      jax=jax.__version__,
      devices=list(map(str, jax.devices())),
      environment={
          k: os.environ.get(k)
          for k in ("LIBTPU_INIT_ARGS", "XLA_FLAGS", "PALLAS_INTERPRET")
      },
      input_hashes=[digest(x) for x in inputs],
  )
  (root / "inputs.json").write_text(json.dumps(result, indent=2))
  config = mosaic.Config(
      fuse_forward=True,
      fuse_backward=True,
      fuse_cp_backward=True,
      fuse_rematerialization=True,
      rematerialize_for_backward=args.remat,
      packed_forward=True,
      packed_backward=args.packed_backward,
      packed_output=True,
      packed_gradients=True,
  )
  if not config.fuse_backward or (
      args.cp > 1 and not getattr(config, "fuse_cp_backward", False)
  ):
    raise AssertionError("Requested backward schedule was not selected")
  result["config"] = dataclasses.asdict(config)
  attention = mosaic.PallasMosaicTpuKimiDeltaAttention(config=config)
  mesh = Mesh(np.array(jax.devices()[: args.cp]), ("context",))
  meta = (
      ContextParallelMetadata(mesh=mesh, axis_name="context")
      if args.cp > 1
      else None
  )

  def local(q, k, v, g, beta, alog, bias, dout, ids):
    def call(q, k, v, g, beta, alog, bias):
      kwargs = dict(
          a_log=alog,
          delta_time_bias=bias,
          use_gate_in_kernel=True,
          lower_bound=-1.0,
          segment_ids=None if args.fixed else ids,
          max_num_segments=args.capacity,
          context_parallel_metadata=meta,
      )
      if args.route != "plain":
        from tokamax._src.ops.kda import pallas_mosaic_tpu_workspace as workspace

        return workspace.kimi_delta_attention_cp(
            q,
            k,
            v,
            g,
            beta,
            config=config,
            head_group_size=args.group,
            fast_segment_capacity=2 if args.route == "bucket" else 0,
            **kwargs,
        )[0]
      return attention(q, k, v, g, beta, **kwargs)[0]

    if args.outer_remat:
      call = jax.checkpoint(call)
    output, pullback = jax.vjp(call, q, k, v, g, beta, alog, bias)
    gradients = pullback(dout)
    if args.cp > 1:
      gradients = (
          *gradients[:5],
          jax.lax.psum(gradients[5], "context"),
          jax.lax.psum(gradients[6], "context"),
      )
    return output, gradients

  spec = P(None, None, "context", None)
  outspec = (spec, (spec, spec, spec, spec, P(None, None, "context"), P(), P()))
  with jaxtyping.disable_jaxtyping(), jax.set_mesh(mesh):
    fn = (
        jax.jit(
            jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(spec,) * 4
                + (
                    P(None, None, "context"),
                    P(),
                    P(),
                    spec,
                    P(None, "context"),
                ),
                out_specs=outspec,
                check_vma=False,
            )
        )
        if args.cp > 1
        else jax.jit(local)
    )
    lowered = fn.lower(*inputs)
    compiled = lowered.compile()
    memory = compiled.memory_analysis()
    result["compiled_memory"] = {
        name: int(getattr(memory, name))
        for name in dir(memory)
        if name.endswith("_size_in_bytes")
    }
    (root / "compiled.txt").write_text(compiled.as_text())
    actual = jax.block_until_ready(compiled(*inputs))
    leaves = jax.tree.leaves(actual)
    result["finite"] = [bool(np.isfinite(np.asarray(x)).all()) for x in leaves]
    result["output_hashes"] = [digest(x) for x in leaves]
    if not all(result["finite"]):
      raise AssertionError("Nonfinite public output or gradient")
    if args.save_arrays:
      np.savez_compressed(
          root / "outputs.npz",
          **{
              f"x{i}": np.asarray(x).astype(np.float32)
              for i, x in enumerate(leaves)
          },
      )
      np.savez_compressed(
          root / "inputs.npz",
          **{
              f"x{i}": np.asarray(x).astype(np.float64)
              for i, x in enumerate(inputs)
          },
      )
    for _ in range(args.warmup):
      jax.block_until_ready(compiled(*inputs))
    samples = []
    for _ in range(args.repeat):
      start = time.perf_counter_ns()
      jax.block_until_ready(compiled(*inputs))
      samples.append((time.perf_counter_ns() - start) / 1e6)
    result.update(
        samples_ms=samples,
        median_ms=statistics.median(samples),
        device_memory=[x.memory_stats() for x in jax.devices()],
    )
    if args.profile:
      with jax.profiler.trace(str(root / "profile")):
        for step in range(args.profile_repeat):
          with jax.profiler.StepTraceAnnotation(
              "public_forward_vjp", step_num=step
          ):
            jax.block_until_ready(compiled(*inputs))
  (root / "result.json").write_text(json.dumps(result, indent=2))
  print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
  main()
