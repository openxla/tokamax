# Inference: latest-main benchmark (2026-10-27)

Both implementations receive the same batch-first BTHD arrays and return BTHD
output plus optional BNHKV state. The existing main API's layout conversions are
inside the timed wrapper. Original head-first integration measurements are kept
separately and are not substituted into this public batch-first comparison.

| Case B/T/H/N | Main ms | Native ms | Native vs main |
|---|---|---|---|
| tokens512 1/512/8/2 | 0.2952 | 0.1672 | 1.77x |
| tokens8192 1/8192/8/2 | 1.2725 | 0.5069 | 2.51x |
| arg_packed 1/8192/32/25 | 3.9100 | 4.5757 | 0.85x |
| fragmented 1/2048/8/512 | 15.1705 | 1.7119 | 8.86x |
| state512 1/512/8/2 | 0.3760 | 0.2505 | 1.50x |

All numerical comparisons pass the 2% relative-RMS limit; repeated outputs are
bitwise stable. H8 two-segment prefill gains 1.77x at T512 and 2.51x at T8192;
the optional-state T512 case gains 1.50x. The boundary-dense T2048/N512 case gains
8.86x. **H32/T8192/N25 is about 17% slower** despite passing correctness; two
independent device captures of that regression are retained. Boundary recurrence
cost must be addressed before claiming universal inference gains. These are
prefill/operator measurements, not single-token decode or full-model step times.

## Protocol

Frozen main: `4eeb302c1fe118c66274039ba4ac6b5b6753d0bd`. Candidate: `3d9257c2fdd858e1a1b31e60289760537eb657cd`. TPU7x, dedicated 4-chip/8-device
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

Compiler temporary memory for `tokens8192`: **223,836,256 -> 2,190,816 bytes**. This excludes arguments/output buffers and is not a measured allocator or model live-HBM peak. Per-case argument/output sizes are in the raw data.

## Device evidence

| Capture | Case | Mode | Max-rank module median us |
|---|---|---|---|
| inference-main-0 | arg_packed | forward | 3708.284 |
| inference-main-0 | tokens8192 | forward | 1078.412 |
| inference-main-1 | arg_packed | forward | 3711.980 |
| inference-main-1 | tokens8192 | forward | 1078.199 |
| inference-native-0 | arg_packed | forward | 4381.496 |
| inference-native-0 | tokens8192 | forward | 313.355 |
| inference-native-1 | arg_packed | forward | 4381.726 |
| inference-native-1 | tokens8192 | forward | 313.324 |

## Evidence and raw samples

[Raw timing/diagnostic CSV](benchmarks/inference_latest_main_benchmark.csv) includes every configuration, all 40 timings and the unfavorable rows.

Immutable raw source/input/HLO/XPlane archive (internal GCS access): `gs://data_ant/diagnostics/kda-latest-main-bench-20261027/latest-main-matrix-4eeb302c.tgz#1793086505059538`. Size 14,027,576,540 bytes; SHA256 `521c1577d5161b3e608b1bfb47b93695e7527f09ce71a522aea4b2c475367074`; CRC32C `7KbiBw==`. Streamed upload size and server checksum verified; the dedicated TPU Pod was then deleted.
