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
"""Compare context-parallel backward schedules."""

import argparse
from tokamax.benchmarks.kda_megakernel_compare import _report

import jax
import jax.numpy as jnp
import numpy as np

try:
  from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
except ModuleNotFoundError as error:
  if error.name != "tokamax._src.ops.kda":
    raise
  from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu as mosaic


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
        config=mosaic.Config()
        if baseline
        else mosaic.Config(cp_megakernel=fused)
    )

    def local(q, k, v, g, beta, segments):
      def forward(q, k, v, g, beta):
        return op(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=segments,
            max_num_segments=1,
            context_parallel_metadata=meta,
        )[0]

      out, back = jax.vjp(forward, q, k, v, g, beta)
      return back(jnp.ones_like(out))

    with jaxtyping.disable_jaxtyping(), jax.set_mesh(mesh):
      mapped = jax.shard_map(
          local,
          mesh=mesh,
          in_specs=(spec,) * 4 + (P(None, None, "context"), P(None, "context")),
          out_specs=(spec,) * 4 + (P(None, None, "context"),),
          check_vma=False,
      )
      _report(
          f"cp-t{tokens}/"
          + (
              "upstream-1103"
              if baseline
              else ("megakernel" if fused else "staged")
          ),
          lambda x: mapped(*x),
          (q, k, v, g, beta, segments),
          iterations,
      )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--suite", required=True, choices=("cp",))
  parser.add_argument("--iterations", type=int, default=10)
  parser.add_argument("--baseline", action="store_true")
  parser.add_argument("--cp-tokens", type=int, default=512)
  args = parser.parse_args()
  if jax.default_backend() != "tpu":
    raise RuntimeError("This comparison requires a native TPU")
  if args.cp_tokens <= 0 or args.cp_tokens % 256:
    parser.error("--cp-tokens must be positive and divisible by 4 * 64")
  _cp(args.iterations, args.baseline, args.cp_tokens)


if __name__ == "__main__":
  main()
