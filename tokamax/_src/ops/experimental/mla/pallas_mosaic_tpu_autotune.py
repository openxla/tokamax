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
"""Regenerates the MLA op's autotuning cache. Not a regular test.

Autotunes every `arg_specs.ARG_SPECS` entry on the current TPU, one test case
(and so roughly one shard) per spec. The op's search space varies the decode and
the prefill/mixed knobs in two separate sweeps; this script then merges the two
sweeps' winners into one config and times it against them and the heuristic.
The passes share no knobs, so the merge is the best of both. It writes each
spec's cache entry to `$TEST_UNDECLARED_OUTPUTS_DIR/<spec name>.json`. Merge
those JSON objects into
`tokamax/data/autotuning/<device kind>/pallas_tpu_multi_head_latent_attention.json`.

Run the `pallas_mosaic_tpu_autotune` test target for the TPU generation to
tune. The target is tagged `manual`: it regenerates the cache rather than
testing anything, and ties up a TPU shard per spec for about 15-20 minutes.
"""

import dataclasses
import os

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from tokamax._src.autotuning import api as autotuning_api
from tokamax._src.autotuning import autotuner
from tokamax._src.ops.experimental.mla import arg_specs
from tokamax._src.ops.experimental.mla import pallas_mosaic_tpu

# The fastest config must beat the heuristic by this fraction to replace it.
_MIN_GAIN = 0.04

_Config = pallas_mosaic_tpu.Config
_FIELDS = [f.name for f in dataclasses.fields(_Config)]
_DECODE_FIELDS = [f for f in _FIELDS if f.startswith("decode_")]
_PREFILL_FIELDS = [f for f in _FIELDS if f.startswith("prefill_")]


def _knobs(config: _Config, fields: list[str]) -> tuple[int, ...]:
  return tuple(getattr(config, f) for f in fields)


def _pick(timings, heuristic: _Config, candidates) -> _Config:
  """The fastest candidate, unless the heuristic is within `_MIN_GAIN` of it.

  Smaller margins are within the noise of the end-to-end A/B runs that
  validated the heuristics.
  """
  ms = lambda c: timings[c].median_evaluation_time_ms
  candidates = [c for c in candidates if c in timings]
  if not candidates:
    return heuristic
  best = min(candidates, key=ms)
  if heuristic in timings and ms(heuristic) < (1 + _MIN_GAIN) * ms(best):
    return heuristic
  return best


class RegenerateAutotuningCache(parameterized.TestCase):

  @parameterized.named_parameters(
      (spec.name, spec) for spec in arg_specs.ARG_SPECS
  )
  def test_autotune(self, spec):
    if jax.default_backend() != "tpu":
      self.skipTest("Requires a TPU.")
    ba = pallas_mosaic_tpu.PallasTpuMultiHeadLatentAttention().bind(**spec.args)
    logging.info(
        "Autotuning %s over %d configs", spec.name, len(ba.autotuning_configs)
    )
    # No timeout: the autotuner's applies to all configs' compiles together.
    result = autotuning_api.autotune(
        [ba], ignore_cache=True, progress_bar=False, timeout=None
    )
    self.assertLen(result.data, 1)
    data = result.data[0][1]
    logging.info(
        "%s: fastest %s (heuristic %s)",
        spec.name,
        data.fastest_config,
        ba.heuristics_config,
    )
    for config, timing in sorted(
        data.prune_errors().items(),
        key=lambda kv: kv[1].median_evaluation_time_ms,
    ):
      logging.info(
          "%s: %.4f ms  %s", spec.name, timing.median_evaluation_time_ms, config
      )

    # Each sweep's winner, kept at the heuristic unless it beats it by
    # `_MIN_GAIN`; otherwise noise would pick e.g. the prefill knobs of a
    # decode-only shape.
    heuristic = ba.heuristics_config
    timings = data.prune_errors()
    decode_sweep = [
        c
        for c in timings
        if _knobs(c, _PREFILL_FIELDS) == _knobs(heuristic, _PREFILL_FIELDS)
    ]
    prefill_sweep = [
        c
        for c in timings
        if _knobs(c, _DECODE_FIELDS) == _knobs(heuristic, _DECODE_FIELDS)
    ]
    best_decode = _pick(timings, heuristic, decode_sweep)
    best_prefill = _pick(timings, heuristic, prefill_sweep)
    merged = dataclasses.replace(
        heuristic,
        **{f: getattr(best_decode, f) for f in _DECODE_FIELDS},
        **{f: getattr(best_prefill, f) for f in _PREFILL_FIELDS},
    )
    finalists = {heuristic, best_decode, best_prefill, merged}
    logging.info("%s: finalists %s", spec.name, finalists)

    # Re-time the finalists together: timings from separate runs are not
    # comparable to within `_MIN_GAIN`.
    if len(finalists) > 1:
      final = ba.autotune(
          finalists,
          autotuner=autotuner.Autotuner(),
          cache_results=False,
          timeout=None,
      )
    else:
      final = autotuner.AutotuningData({heuristic: data[heuristic]})
    final_timings = final.prune_errors()
    for config, timing in final_timings.items():
      logging.info(
          "%s: final %.4f ms  %s",
          spec.name,
          timing.median_evaluation_time_ms,
          config,
      )
    chosen = _pick(final_timings, heuristic, final_timings)
    logging.info("%s: chose %s", spec.name, chosen)

    out_dir = os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]
    pruned = dataclasses.replace(
        result,
        data=((ba, autotuner.AutotuningData({chosen: final[chosen]})),),
    )
    with open(os.path.join(out_dir, f"{spec.name}.json"), "w") as f:
      f.write(pruned.dump_cache_str())


if __name__ == "__main__":
  absltest.main()
