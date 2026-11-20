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
"""Compare standalone inference with the existing Mosaic forward."""

import argparse
from tokamax.benchmarks.kda_megakernel_compare import _report

import jax


def _inference(iterations, tokens, baseline):
  from tokamax.benchmarks.kda_inference import make_example

  for native in ((False,) if baseline else (False, True)):
    fn, args = make_example(tokens, native)
    label = (
        "upstream-1103"
        if baseline
        else ("native" if native else "existing-mosaic-forward")
    )
    _report(f"inference-t{tokens}/{label}", fn, args, iterations)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--suite", required=True, choices=("inference",))
  parser.add_argument("--iterations", type=int, default=10)
  parser.add_argument("--baseline", action="store_true")
  parser.add_argument("--tokens", type=int, default=512)
  args = parser.parse_args()
  if jax.default_backend() != "tpu":
    raise RuntimeError("This comparison requires a native TPU")
  if args.tokens <= 0 or args.tokens % 64:
    parser.error("--tokens must be positive and divisible by 64")
  _inference(args.iterations, args.tokens, args.baseline)


if __name__ == "__main__":
  main()
