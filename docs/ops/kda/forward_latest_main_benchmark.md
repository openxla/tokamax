# Forward: latest-main benchmark (2026-10-27)

Formal ArgSpecs use in-kernel Q/K normalization; their compact-config calls select
the aligned fallback. Native compact numbers below use the explicitly equivalent
no-normalization workload on both main and candidate. Ratios below 1 mean slower.

| Case B/T/H/N | QK norm | Main ms | Fused packed ms | Compact-config ms | Selected vs main |
|---|---|---|---|---|---|
| arg_fixed 1/8192/32/1 | True | 1.9510 | 1.9015 | 1.8813 (aligned fallback) | 1.03x |
| arg_packed 1/8192/32/25 | True | 4.0142 | 2.8996 | 2.8871 (aligned fallback) | 1.38x |
| compact_packed 1/8192/32/25 | False | 3.5562 | 2.7388 | 1.3600 | 2.61x |
| fragmented 1/2048/8/512 | False | 14.6190 | 4.6694 | 0.3521 | 41.52x |
| long64k 1/65536/8/64 | False | 8.4364 | 5.8961 | 3.2673 | 2.58x |
| batch2 2/8192/64/25 | False | 15.8032 | 11.2215 | 4.5148 | 3.50x |

All outputs pass the 2% relative-RMS comparison to main. The two formal ArgSpecs
show about 1.03x (fixed) and 1.38x (packed) forward throughput, not the much larger
compact-only gains. The 41.52x case contains 512 four-token sequences in 2K tokens;
it removes disproportionate sequence-alignment overhead. It is not a model-step
or universally representative speedup. Default-config forward stays within about
1.5% of main in this matrix. Raw repeats are bitwise stable.

## Protocol

Frozen main: `4eeb302c1fe118c66274039ba4ac6b5b6753d0bd`. Candidate: `c2e35dba528c6cf136df73f666dc861cd105e01c`. TPU7x, dedicated 4-chip/8-device
Pod; this suite uses device 0. JAX/JAXLIB 0.11.2, libtpu 0.0.48, Python 3.12.15,
Pydantic 2.14.1, NumPy 2.5.4, Flax 0.12.10, Qwix 0.1.8, XProf 2.23.1. Large second-minor layout enabled; interpretation
disabled. No donation. Each executable warms up 5 times, then records 20 blocking
calls in each of two fresh-process sweeps with reversed variant order. All tables
use the median of 40 raw host-wall samples; compilation/input creation excluded.
Selected three-call device captures are repeated independently on device 0.

NumPy seed 619; BF16 q/k/v Gaussian std .05, raw gate std .1; beta is sigmoid
of FP32 Gaussian std .1 then BF16. FP32 A_log uniform [-1,0], bias uniform [-4,-3].
K=V=128, chunk size 64, scale=128**-.5, bounded raw gate lower=-5. Inputs are identical and hash-checked.
No user state unless stated. All raw timings and correctness deltas are in the
raw timing/diagnostic CSV.

The fixed and packed training rows reproduce both declared KDA ArgSpec shapes
and switches. The pinned main's obsolete float-initializer call is bypassed by
constructing explicit deterministic arrays in valid ranges; operator code is
unmodified. Additional compact cases extend that shape coverage.

Compiler temporary memory for `compact_packed`: **1,101,359,552 -> 338,246,752 bytes**. This excludes arguments/output buffers and is not a measured allocator or model live-HBM peak. Per-case argument/output sizes are in the raw data.

## Device evidence

| Capture | Case | Mode | Max-rank module median us |
|---|---|---|---|
| forward-aligned-0 | arg_packed | forward | 2702.015 |
| forward-aligned-1 | arg_packed | forward | 2702.018 |
| forward-compact-0 | compact_packed | forward | 1093.262 |
| forward-compact-1 | compact_packed | forward | 1093.475 |
| forward-main-0 | arg_packed | forward | 3815.501 |
| forward-main-0 | compact_packed | forward | 3355.139 |
| forward-main-1 | arg_packed | forward | 3815.295 |
| forward-main-1 | compact_packed | forward | 3357.482 |

## Evidence and raw samples

[Raw timing/diagnostic CSV](benchmarks/forward_latest_main_benchmark.csv) includes every configuration, all 40 timings and the unfavorable rows.

Immutable raw source/input/HLO/XPlane archive (internal GCS access): `gs://data_ant/diagnostics/kda-latest-main-bench-20261027/latest-main-matrix-4eeb302c.tgz#1793086505059538`. Size 14,027,576,540 bytes; SHA256 `521c1577d5161b3e608b1bfb47b93695e7527f09ce71a522aea4b2c475367074`; CRC32C `7KbiBw==`. Streamed upload size and server checksum verified; the dedicated TPU Pod was then deleted.
