# Non-CP backward: latest-main benchmark (2026-10-27)

The first table measures the complete operator plus VJP. The second times only
the pullback, with precomputed residuals passed as runtime pytree arguments.
No large residual arrays are baked into HLO constants. This keeps VJP-only
measurements distinct from forward work and from compiler constant folding.

## Complete operator + VJP

| Case | Main ms | Aligned optimized ms | Compact ms | Selected vs main |
|---|---|---|---|---|
| arg_fixed | 4.8571 | 4.4382 | - | 1.09x |
| arg_packed | 8.7702 | 9.0839 | - | 0.97x |
| compact_packed | 7.9797 | 7.6058 | 3.4823 | 2.29x |
| fragmented | 20.4451 | 16.5081 | 0.6113 | 33.45x |
| remat_packed | 10.1210 | 10.6745 | - | 0.95x (precision rejected) |

## VJP only

| Case | Main ms | Aligned optimized ms | Compact ms | Selected vs main |
|---|---|---|---|---|
| arg_fixed | 3.2933 | 2.9382 | - | 1.12x |
| arg_packed | 6.1065 | 5.8280 | - | 1.05x |
| compact_packed | 5.2644 | 5.0301 | 2.0202 | 2.61x |
| fragmented | 8.5246 | 8.9015 | 0.4282 | 19.91x |
| remat_packed | 6.9792 | 6.5154 | - | 1.07x (precision rejected) |

## Accuracy and regressions

- Compact packed: complete-call 2.29x and pure VJP 2.61x. Its gate gradients differ
  from main beyond the pairwise diagnostic limit, but the independent FP32
  recurrence shows improvement: A_log RMS **6.43% -> 3.15%**, bias **5.10% -> 2.91%**.
  All compact gradients pass the existing 2% input / 5% gate-parameter budgets
  against that independent reference. Do not label this an accuracy regression.
- Saved packed with Q/K normalization: pure VJP improves about 4.8%, but the full
  call is about 3.6% slower. The default full call is about 4.1% slower than main;
  no universal/default-path non-regression claim is made.
- Fused rematerialization is **not accepted**. FP32 recurrence dQ RMS rises from
  about 0.41% to 3.31%, dgate from 1.64% to 4.59%, and A_log from 4.98% to 10.10%.
  Its repeat hashes also differ. Reported timing is diagnostic, not accepted gain.
- Main itself has about 5.2% bias error in the normalized random-cotangent case;
  the equivalent saved-state candidate retains that existing limitation. Reference
  failures are preserved rather than widening a tolerance.
- The independent reference uses the unchanged token recurrence in FP32 with
  highest matmul precision; only per-head checkpointing is added to bound memory.
  Full-call and pure-VJP arrays are both checked. No reference arrays or test
  thresholds were edited to make a comparison pass.

The `forward` control in JSON fixes forward switches while leaving backward
fusion/packed-gradient flags off. A compiler can still alter materialization when
the backward residual requirements change; full-call and VJP-only deltas are both
reported. Compact additionally changes the tape contract and requires #1688.

## Protocol

Frozen main: `4eeb302c1fe118c66274039ba4ac6b5b6753d0bd`. Candidate: `e333c798cfbbc1a645515e01edc9dddb71ef6961`. TPU7x, dedicated 4-chip/8-device
Pod; this suite uses device 0. JAX/JAXLIB 0.11.2, libtpu 0.0.48, Python 3.12.15,
Pydantic 2.14.1, NumPy 2.5.4, Flax 0.12.10, Qwix 0.1.8, XProf 2.23.1. Large second-minor layout enabled; interpretation
disabled. No donation. Each executable warms up 5 times, then records 20 blocking
calls in each of two fresh-process sweeps with reversed variant order. All tables
use the median of 40 raw host-wall samples; compilation/input creation excluded.
Selected three-call device captures are repeated independently on device 0.

NumPy seed 619; BF16 q/k/v/dO Gaussian std .05, raw gate std .1; beta is sigmoid
of FP32 Gaussian std .1 then BF16. FP32 A_log uniform [-1,0], bias uniform [-4,-3].
K=V=128, chunk size 64, scale=128**-.5, bounded raw gate lower=-5. Inputs/cotangents are identical and hash-checked.
No user state unless stated. All raw timings and correctness deltas are in the
raw timing/diagnostic CSV. A main fixed-length baseline case was replayed at an idle boundary
to exclude possible host-analysis overlap; the rejected samples are preserved.

The fixed and packed training rows reproduce both declared KDA ArgSpec shapes
and switches. The pinned main's obsolete float-initializer call is bypassed by
constructing explicit deterministic arrays in valid ranges; operator code is
unmodified. Additional compact and rematerialized cases extend that shape coverage.

Compiler temporary memory for `compact_packed`: **2,897,866,592 -> 808,693,344 bytes**. This excludes arguments/output buffers and is not a measured allocator or model live-HBM peak. Per-case argument/output sizes are in the raw data.

## Device evidence

| Capture | Case | Mode | Max-rank module median us |
|---|---|---|---|
| backward-compact-0 | compact_packed | full | 3174.717 |
| backward-compact-0 | compact_packed | vjp | 1758.762 |
| backward-compact-1 | compact_packed | full | 3174.348 |
| backward-compact-1 | compact_packed | vjp | 1758.938 |
| backward-main-0 | compact_packed | full | 7681.822 |
| backward-main-0 | compact_packed | vjp | 4916.776 |
| backward-main-1 | compact_packed | full | 7682.472 |
| backward-main-1 | compact_packed | vjp | 4918.838 |

## Evidence and raw samples

[Raw timing/diagnostic CSV](benchmarks/backward_latest_main_benchmark.csv) includes every configuration, all 40 timings and the unfavorable rows.

Immutable raw source/input/HLO/XPlane archive (internal GCS access): `gs://data_ant/diagnostics/kda-latest-main-bench-20261027/latest-main-matrix-4eeb302c.tgz#1793086505059538`. Size 14,027,576,540 bytes; SHA256 `521c1577d5161b3e608b1bfb47b93695e7527f09ce71a522aea4b2c475367074`; CRC32C `7KbiBw==`. Streamed upload size and server checksum verified; the dedicated TPU Pod was then deleted.
