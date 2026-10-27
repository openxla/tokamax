# Compact M5 backward: migration validation

This record covers the backward increment in #1689. It consumes the compact
residual contract defined by #1688. Source: primatrix/pallas-kernel
`dca6a09feec2659f2777a6db02f2ee55560f991e`.

## Correctness

- 28 compact backward cases passed on native TPU. Coverage includes bounded raw
  and preactivated gates, padding, interleaved labels, dynamic metadata replay,
  multiple batches/heads and non-square logical widths.
- Ten small source/port configurations and five benchmark shapes reproduce all
  enabled input and parameter gradients bitwise on identical inputs/cotangents.
- Independent recurrence comparisons retain the existing 2% input-gradient and
  5% gate-parameter relative-RMS budgets. The source's observed 2-4% gate-parameter
  errors are reproduced; migration is not a gate-precision improvement.
- Existing aligned backward thresholds are unchanged. CP and rematerialized
  compact execution are not introduced by this saved-state compact path.

## Historical full-call migration timing

The following measures output plus VJP, including the forward call. It checks
migration parity with the internal source, not isolated backward speedup over
public Tokamax. TPU7x device 0, JAX/JAXLIB 0.11.2, libtpu 0.0.48, K=V=128,
BF16 bounded raw gates (-5), no Q/K normalization or user state. Grouped sorted
segments use `compact_forward=True, compact_sorted_segments=True`.

| B / T / H / segments | Source operator + VJP ms | Port operator + VJP ms |
|---|---:|---:|
| 1 / 8192 / 32 / 1 | 3.3131 | 3.3656 |
| 1 / 8192 / 32 / 25 | 3.3603 | 3.4193 |
| 2 / 8192 / 64 / 25 | 13.1967 | 13.3197 |
| 1 / 2048 / 8 / 512 | 0.5963 | 0.6700 |
| 1 / 65536 / 8 / 64 | 7.0419 | 7.1266 |

Five warmups; 20 synchronized calls in source/port/port/source order, 40 samples
per implementation. Medians include public layout conversions; compilation and
input creation are excluded. Seed 619; input/cotangent hashes and raw samples are
preserved. Complete operator+VJP compiler temporary bytes match the source on all
five shapes. No host-latency speedup is claimed from this migration A/B.

## Device cross-check

XProf 2.23.1, raw TPU:0 XLA Modules events, two independent captures with five
calls each. Packed-8K full-call medians are source 3506.871/3507.222 us versus
port 3506.578/3505.625 us. Fragmented-2K is source 330.194/329.904 us versus
port 330.006/337.181 us; the second port capture has about 2.2% variation.
Capture flags differ from ordinary wall timing, so these clocks are not subtracted.
Official LLO analysis/debug extraction returned errors; raw errors, XPlane and HLO
are retained. No instruction-level attribution is claimed.

## Reproduce

The evidence bundle contains the exact source/port snapshots, `benchmark.py`,
inputs, cotangents, samples, gradient comparisons and captures:
`gs://data_ant/diagnostics/kda-compact-training-20261026/kda-compact-training-evidence.tgz#1793020431222266`.
SHA256 `2324a9c5c56b408efc3294761d7e80ac170a184991e7e088e8933d204d851792`;
CRC32C `3MtjPQ==`; archived size/checksum verified.

For a backward-configuration comparison with fixed forward flags, use
`python -m tokamax.benchmarks.kda_backward_compare --suite backward` or
`--suite backward-packed`. Those timers also cover the complete operator/VJP.
