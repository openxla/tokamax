# Compact KDA migration: validation and performance

Source: primatrix/pallas-kernel `dca6a09feec2659f2777a6db02f2ee55560f991e`.
Forward increment: `c7736c0e3b285f5b91a79fb3b75ffe8d74c24214` (#1688).
Backward: companion #1689, native M5 consuming the same original-token tape.

## Scope

Native segment IDs; full-chunk, two-fragment and arbitrary-fragment paths;
compact saved states, original-token M5 gradients and eligible fused gate VJP.
Opt in with `Config(compact_forward=True, compact_sorted_segments=True)` for
grouped nondecreasing positive IDs with trailing zero padding. The general
configuration preserves interleaved-label semantics through a stable permutation,
but reserves extra fallback buffers. CP, rematerialization, QK normalization and
user initial/final states retain existing paths; this is the saved-state non-CP
training profile, not an expansion of those other profiles.

## Validation

- Final merged stack: 102 native TPU tests passed, including 50 compact tests and
  existing forward/fused-backward regression. Legacy tolerances unchanged.
- Forward increment: 41 CPU interpretation tests passed.
- Public compact benchmark: forward and forward+VJP both passed on TPU.
- CI shard/dependency/summary checks: 161 tests and 15 subtests passed.
- Ten small source/port configurations and all five benchmark shapes: output and
  enabled input gradients bitwise equal on TPU. Same input arrays and cotangents.
- An independent recurrence covers metadata and gradient correctness. Raw gate
  parameter gradients retain the source's 5% relative-RMS budget: observed source
  errors around 2–4% are reproduced exactly, not claimed to be fixed by migration.

## Synchronized host timing

TPU7x device 0, JAX/JAXLIB 0.11.2, libtpu 0.0.48. K=V=128, BF16,
bounded raw gate (-5), no QK normalization, no initial/final state. Each executable
is warmed up 5 times; ABBA order with 20 blocking calls per round yields 40
samples per implementation. Compilation and input generation are excluded.
Input seed is 619; report.json records input/cotangent hashes, all raw samples,
environment flags and memory analysis. Both public layout conversions are included.

| B / T / H / segments | Source forward ms | Port forward ms | Source forward+VJP ms | Port forward+VJP ms |
|---|---:|---:|---:|---:|
| 1 / 8192 / 32 / 1 | 0.9158 | 0.9716 | 3.3131 | 3.3656 |
| 1 / 8192 / 32 / 25 | 0.9680 | 1.0221 | 3.3603 | 3.4193 |
| 2 / 8192 / 64 / 25 | 3.2657 | 3.3332 | 13.1967 | 13.3197 |
| 1 / 2048 / 8 / 512 | 0.2870 | 0.3554 | 0.5963 | 0.6700 |
| 1 / 65536 / 8 / 64 | 2.6354 | 2.6879 | 7.0419 | 7.1266 |

These are medians from the final merged stack. Host wall time remains modestly
higher for the port; no host-latency speedup is claimed. Pure-forward and complete
forward+VJP compiler temporary-memory sizes match the source for every shape.
Training-forward tape structures differ slightly; their memory is reported
separately in the raw report, not conflated with complete training.

## Device trace cross-check

XProf 2.23.1; two independent captures per implementation, five invocations each.
Raw `/device:TPU:0` XLA Modules events were extracted with the official
`list_xplane_events` CLI. Source/port packed-8K forward capture medians are
769.496/769.351 us and 769.435/769.245 us. Full forward+VJP capture medians are
3506.871/3506.578 us and 3507.222/3505.625 us. This supports device execution
parity; it does not establish the cause of host-wall differences.

For fragmented-2K forward the two source medians are 100.632/100.371 us,
versus port 100.461/100.631 us. Full forward+VJP is 330.194/329.904 us versus
330.006/337.181 us: one port capture has ~2.2% variation. Do not describe every
capture as identical. Profiling enables custom-call tracing/debug flags, so
trace durations should not be directly subtracted from non-profiled wall time.

The official LLO analysis and debug-string endpoints returned explicit extraction
errors despite debug registration. Raw XPlane, HLO and the error payloads are
retained; no instruction-level or per-Pallas-kernel attribution is claimed.

## Reproduction

The evidence bundle contains the pinned internal source, exact port snapshot,
`benchmark.py`, parity/test logs, all samples, HLO and raw XPlane captures.
Run `benchmark.py` with `/workspace/internal` and `/workspace/port` on its Python
path; `--cases` and `--modes` select the matrix, `--profile` selects trace capture.
The repository benchmark adds `mosaic_compact` and a separate packed workload
without in-kernel QK normalization, preventing accidental fallback measurements.
