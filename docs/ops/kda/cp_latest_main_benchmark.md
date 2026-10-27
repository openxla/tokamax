# CP: latest-main benchmark (2026-10-27)

All timings cover the complete public CP operator plus VJP, including input
reshards. Parent control is PR #1689 (`e333c798`) with no CP-specific fusion or
workspace adapter; this distinguishes the CP increment from inherited changes.

| Case | Main ms | Parent ms | Local fused ms | Workspace ms | Remote-DMA ms | Workspace vs main |
|---|---|---|---|---|---|---|
| cp4_normal | 3.3166 | 3.3024 | 3.5994 | 2.3747 | - | 1.40x |
| cp8_normal | 5.9344 | 5.8939 | 6.1327 | 3.9351 | - | 1.51x |
| cp4_remat | 3.4555 | 3.4580 | 3.4958 | 2.4364 | - | 1.42x |
| cp8_capacity | 741.8791 | 746.0007 | 717.9214 | 3.9029 | - | 190.09x |
| cp8_overflow | 741.7926 | 746.0351 | 717.8778 | 26.6587 | - | 27.83x |
| cp4_mega | 2.4076 | 2.3614 | 2.6536 | 3.9110 | 2.5045 | 0.62x |

Workspace uses all-rank-uniform fast capacity 2 and head groups of 2. Overflow
falls back to the logical capacity with bounded stateless local capacity. Normal
cases use capacity 32; the large-capacity cases use logical 8192 with only two
local sequences (three on one rank for overflow). Global T is 512 for CP4 and
1024 for CP8; B/H are 1/8 and 2/8 normally, 2/4 for the capacity stress cases.
The remote-DMA case is CP4, B2/H8, preactivated gate, capacity 2.

All relative output/gradient comparisons pass. Normal workspace cases gain
1.40-1.51x. The 190.09x and 27.83x gains depend on sparse occupancy of a very
large capacity bound; they are not general CP or 256K-training throughput gains.
The parent control remains about 746 ms in those stress cases, so the incremental
workspace gains are also about 191.1x / 28.0x versus the parent.

Local fusion alone is not uniformly faster. The sampled remote-DMA complete call
is about 4% slower than main (though about 6% faster than local fusion). Grouped
workspace on the already-small capacity-2 case is also slower. These regressions
are retained, and the large workspace win is not attributed to remote DMA.

## Protocol

Frozen main: `4eeb302c1fe118c66274039ba4ac6b5b6753d0bd`. Candidate: `2379bd5ba29d3eafbaf478e8ea9e6839bea6c058`. TPU7x, dedicated 4-chip/8-device
Pod; these cases use CP4 or CP8. JAX/JAXLIB 0.11.2, libtpu 0.0.48, Python 3.12.15,
Pydantic 2.14.1, NumPy 2.5.4, Flax 0.12.10, Qwix 0.1.8, XProf 2.23.1. Large second-minor layout enabled; interpretation
disabled. No donation. Each main/candidate executable warms up 5 times, then records 20 blocking
calls in each of two fresh-process sweeps with reversed variant order. All tables
use the median of 40 raw host-wall samples; compilation/input creation excluded.
Selected three-call device captures are repeated independently; CP reports the
maximum rank duration per invocation, never a sum of parallel ranks.

NumPy seed 619; BF16 q/k/v/dO Gaussian std .05, raw gate std .1; beta is sigmoid
of FP32 Gaussian std .1 then BF16. FP32 A_log uniform [-1,0], bias uniform [-4,-3].
K=V=128, chunk size 64, scale=128**-.5, bounded raw gate lower=-5 except the
explicit preactivated CP case. Inputs/cotangents are identical and hash-checked.
No user state unless stated. All raw timings and correctness deltas are in the
raw timing/diagnostic CSV.

The parent-PR control uses two supplementary fresh-process repeats after the main/candidate sweeps, rather than the reversed variant ordering.

Compiler temporary memory for `cp8_capacity`: **53,752,904,288 -> 362,021,856 bytes**. This excludes arguments/output buffers and is not a measured allocator or model live-HBM peak. Per-case argument/output sizes are in the raw data.

## Device evidence

| Capture | Case | Mode | Max-rank module median us |
|---|---|---|---|
| cp-fused-0 | cp8_normal | full | 2883.792 |
| cp-fused-1 | cp8_normal | full | 2922.226 |
| cp-main-0 | cp4_mega | full | 586.651 |
| cp-main-0 | cp8_capacity | full | 737811.657 |
| cp-main-0 | cp8_normal | full | 2860.529 |
| cp-main-1 | cp4_mega | full | 597.212 |
| cp-main-1 | cp8_capacity | full | 737676.615 |
| cp-main-1 | cp8_normal | full | 2857.838 |
| cp-mega-0 | cp4_mega | full | 545.300 |
| cp-mega-1 | cp4_mega | full | 544.404 |
| cp-parent-0 | cp8_capacity | full | 741963.225 |
| cp-parent-0 | cp8_normal | full | 2870.689 |
| cp-parent-1 | cp8_capacity | full | 741857.299 |
| cp-parent-1 | cp8_normal | full | 2929.902 |
| cp-workspace-0 | cp8_capacity | full | 810.330 |
| cp-workspace-0 | cp8_normal | full | 837.808 |
| cp-workspace-1 | cp8_capacity | full | 744.932 |
| cp-workspace-1 | cp8_normal | full | 899.061 |

## Evidence and raw samples

[Raw timing/diagnostic CSV](benchmarks/cp_latest_main_benchmark.csv) includes every configuration, all 40 timings and the unfavorable rows.

Immutable raw source/input/HLO/XPlane archive (internal GCS access): `gs://data_ant/diagnostics/kda-latest-main-bench-20261027/latest-main-matrix-4eeb302c.tgz#1793086505059538`. Size 14,027,576,540 bytes; SHA256 `521c1577d5161b3e608b1bfb47b93695e7527f09ce71a522aea4b2c475367074`; CRC32C `7KbiBw==`. Streamed upload size and server checksum verified; the dedicated TPU Pod was then deleted.
