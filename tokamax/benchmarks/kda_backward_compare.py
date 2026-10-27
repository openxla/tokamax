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
"""Compare non-CP backward schedules with fixed forward configuration."""

import argparse
from tokamax.benchmarks.kda_megakernel_compare import _report

import jax
import tokamax
import jax.numpy as jnp
import numpy as np

try:
  from tokamax._src.ops.kda import pallas_mosaic_tpu as mosaic
except ModuleNotFoundError as error:
  if error.name != "tokamax._src.ops.kda":
    raise
  from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu as mosaic


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
  if suite in ("backward-packed",):
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
    mode = "forward_and_vjp"
  elif suite == "backward":
    configs = [
        ("staged", mosaic.Config(fuse_forward=True, fuse_backward=False)),
        ("fused", mosaic.Config(fuse_forward=True, fuse_backward=True)),
    ]
    mode = "forward_and_vjp"
  else:
    configs = [
        (
            "aligned",
            mosaic.Config(
                fuse_forward=True,
                packed_forward=True,
                packed_output=True,
                fuse_backward=True,
            ),
        ),
        (
            "native-packed",
            mosaic.Config(
                fuse_forward=True,
                fuse_backward=True,
                packed_forward=True,
                packed_output=True,
                packed_backward=True,
                packed_gradients=True,
            ),
        ),
    ]
    mode = "forward_and_vjp"
  for name, config in configs:
    fn, args = tokamax.standardize_function(
        mosaic.PallasMosaicTpuKimiDeltaAttention(config=config),
        kwargs=example,
        mode=mode,
    )
    _report(f"{suite}/{name}", fn, args, iterations)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--suite", required=True, choices=("backward", "backward-packed")
  )
  parser.add_argument("--iterations", type=int, default=10)
  parser.add_argument("--baseline", action="store_true")
  parser.add_argument("--training-tokens", type=int, default=1024)
  parser.add_argument("--heads", type=int, default=8)
  parser.add_argument("--num-segments", type=int, default=4)
  args = parser.parse_args()
  if jax.default_backend() != "tpu":
    raise RuntimeError("This comparison requires a native TPU")
  if args.training_tokens <= 0 or args.heads <= 0:
    parser.error("--training-tokens and --heads must be positive")
  if not 0 < args.num_segments <= args.training_tokens:
    parser.error("--num-segments must be in [1, --training-tokens]")
  _training(
      args.suite,
      args.iterations,
      args.baseline,
      args.heads,
      args.training_tokens,
      args.num_segments,
  )


if __name__ == "__main__":
  main()
