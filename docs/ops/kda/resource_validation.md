# CP workspace: native TPU validation (2026-10-21)

This is the CP-only subset of the archived resource-validation matrix for #1690.
It compares the complete public CP operator and VJP, not a local backward kernel.

## Environment and source

Independent TPU7x, 4 chips / 8 devices, topology 2x2x1; Python 3.12.14,
JAX/JAXLIB 0.11.2, libtpu 0.0.48, Qwix 0.1.8, NumPy 2.5.4, Flax 0.12.10.
XProf 2.24.0. `JAX_PLATFORMS=tpu,cpu`; interpretation and XLA_FLAGS unset.
LIBTPU_INIT_ARGS enables large second-minor layout and enhanced launch barrier.
Baseline: `17ce59f4f20dd58366b37649865937a2604b1f98`, with the identical harness.
The workspace schedules are opt-in; `cp_megakernel=False` in this matrix.

## Inputs and protocol

CP8, B=2, global T=1024, H=4, K=V=128, chunk size 64, logical capacity 8192.
BF16 q/k/v/raw gate/beta/dO; FP32 A_log and flattened bias. No initial/final
state, no Q/K normalization, bounded raw gates (-1), saved-state backward.
NumPy seed 619: q/k/v/dO Gaussian std .05, raw gate std .1, beta sigmoid of
BF16 Gaussian std .1, A_log uniform [-1,0], bias uniform [-4,-3].

Each rank has 128 local tokens, split 65/63 with a cross-rank continuation.
The overflow case splits only the final batch row on the last rank 32/33/63.
Fast capacity is 2. N_max is segment capacity, not sequence length.

Five warmups, 20 blocking calls per old/new/new/old round, 40 samples/revision.
Compilation, data creation and diagnostics are excluded. Wall time includes
input reshards from device 0. Replicated parameter gradients are summed over
context ranks; rank device durations are not summed.

| CP capacity route | Before ms | After ms |
|---|---:|---:|
| Every rank fits fast capacity | 712.485593 | 3.990666 |
| One rank exceeds fast capacity | 712.342825 | 26.501276 |

The gain is specific to sparse occupancy of an overprovisioned capacity bound.
It measures workspace/capacity scheduling, not faster matrix multiplication,
remote-DMA megakernel speedup or model training throughput.

Compiler temporary bytes: 47,254,802,176 -> 362,021,856.
Runtime per-device allocator peak_reserved: 40,057,143,296 -> 290,390,016.
These are distinct counters; neither is a measured model live-HBM peak.

## Correctness and device evidence

- Seven native workspace cases pass: CP4/8, B1/2, capacities 8192/262144,
  saved/rematerialized states, fast/overflow/group-only routes, suffix padding,
  an empty rank, all padding and outer rematerialization. Every public output
  and all seven gradients match full-capacity calls at relative RMS <=1e-5.
- Historical A/B token outputs/gradients are bitwise equal. Parameter-gradient
  relative RMS is <=3.8e-7. Numerical thresholds are unchanged.
- Independent complete `jit_local` captures agree in direction:
  fast 708.245948 -> .874476 ms; overflow 708.146928 -> 22.501086 ms.
  Replacement captures have two complete calls per rank and two independent
  rounds. Original partial captures hit the collection window and are excluded.
- CPU scheduling/entry checks: 25 passed. The adapter preserves global labels
  for CP chains and uses local ordinal state slots; labels are not clipped.

This scheduling work does not establish full-model AOT/CP16-256K acceptance or
repair unrelated repeated-label CP-chain semantics. Unsupported state/residual
requests reject before the workspace adapter dispatches.

## Reproduce and evidence

`pallas_mosaic_tpu_resource_benchmark.py` now accepts CP2/4/8 only; use `--route
plain` versus `--route bucket`, with identical seeds, to compare capacity routes.
`--over` introduces the single-rank overflow; `--group` controls head grouping.

The complete historical source, raw arrays/hashes/samples, HLO and accepted/rejected
captures remain in the immutable internal archive:
`gs://data_ant/diagnostics/tokamax-kda-resources-20261021/evidence-803d05d5.tar.gz#1792589118155652`.
Size 2,400,815,867 bytes; SHA256
`1e9a24a14db36043b4bae2911491aa625bb847ff3bca3efa33de234a773473a2`.
Upload checksum and object metadata were verified.
