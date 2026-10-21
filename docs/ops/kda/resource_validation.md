# Resource safeguards: native TPU validation (2026-10-21)

## Scope and environment

Independent TPU7x Pod, 4 chips / 8 TensorCore devices, topology 2x2x1.
No production Pod or MaxText training was modified. No-CP uses one device;
CP8 uses all eight. Python 3.12.14; JAX/JAXLIB 0.11.2; libtpu 0.0.48;
Qwix 0.1.8; NumPy 2.5.4; Flax 0.12.10; XProf 2.24.0. Full pip freeze and
immutable container image ID are retained with the evidence.

Effective LIBTPU_INIT_ARGS:
`--xla_tpu_impure_enable_large_2nd_minor_layout=true
--xla_tpu_use_enhanced_launch_barrier=true`.
JAX_PLATFORMS=tpu,cpu; PALLAS_INTERPRET unset; XLA_FLAGS unset. These are native
device executions, not CPU Pallas interpretation or abstract export.

Unmodified comparison sources: CP cumulative branch `17ce59f4f20dd58366b37649865937a2604b1f98`;
native-packed backward branch `8e2fc0fd672b49a376478f01c00954e94db7030f`.
Only the identical benchmark harness was copied to each baseline.

## Inputs and measured interface

The benchmark calls the full Mosaic operator and its VJP, returning the public
output and all seven gradients: query, key, value, raw gate, beta, a_log and
delta_time_bias. It does not time only an internal Pallas call or a scalar loss.
For CP, replicated parameter gradients are summed across ranks. Short-conv,
output gated norm, projections, optimizer updates and MaxText step time are not
part of this operator benchmark.

Layout: q/k/g [H,B,T,K], v [H,B,T,V], beta [H,B,T]. K=V=128; chunk=64;
scale=128**-0.5. BF16 token inputs and dO; FP32 a_log and flattened [H*K] bias.
Initial state=None; output_final_state=False; use_qk_l2norm=False;
use_gate_in_kernel=True; lower_bound=-1; saved-h mode unless stated otherwise.
Beta is already sigmoid-activated; the kernel must not apply sigmoid again.

NumPy default_rng seed=619 generates q/k/v/dO as Gaussian std=0.05 and raw
gate as Gaussian std=0.1, rounded through FP32 to BF16. Beta is sigmoid of a
BF16 Gaussian std=0.1. a_log is U[-1,0], bias U[-4,-3], both FP32.
All input hashes and output hashes are recorded per invocation group.

No-CP fixed is one 8192-token sequence. Packed no-CP has lengths 4097/4095,
labels 1/3, capacity 32. For CP, each rank has T_local=128 and two runs of
65/63 tokens, labels (2*r+1)/(2*r+3); adjacent ranks share a continued segment.
The overflow case splits only the last batch row on the final rank into
32/33/63 tokens, using the intervening label 2*CP. Thus the fast-capacity=2
test must select the same fallback on every rank. The logical maximum remains
8192; physical local capacity is bounded by local T; head group size=2 at H=4.
N_max here is segment capacity, not context length.

Flags: fuse_forward/fuse_backward/fuse_cp_backward/fuse_rematerialization=True;
packed_forward/packed_output/packed_gradients=True. CP retains its staged
forward and fused local backward; cp_megakernel=False in these timings.
Packed backward=False except the separately labeled native-packed row.
Original operator defaults are unchanged. Workspace dispatch is opt-in.

## Full-interface latency and memory

Warmup=5, repeat=20, two rounds per revision, order old/new/new/old. Reported
wall time is the median of all 40 synchronized calls, not an average of round
medians. Compilation, input creation, hashes, NPZ writes and reference checks
are outside timing. CP inputs are initially created on device 0, so wall time
also includes compiled-call input reshards. Device-module timings below omit
those external input reshards and include the full in-shard forward/VJP.

| Case (B/T/H, D=128) | Before ms | After ms | Wall change |
| --- | ---: | ---: | ---: |
| No-CP fixed, 2/8192/32 | 7.822664 | 7.803448 | -0.25% |
| No-CP packed, 2/8192/8 | 9.441532 | 9.355743 | -0.91% |
| No-CP packed, 2/8192/64 | 38.022879 | 37.857858 | -0.43% |
| Native-packed backward, 2/8192/8 | 7.783400 | 7.696932 | -1.11% |
| CP8, B2/global T1024/H4/N_max8192, fast | 712.485593 | 3.990666 | -99.44% |
| Same CP shape, one rank over fast capacity | 712.342825 | 26.501276 | -96.28% |

The large CP gain is specific to a low-occupancy, highly overprovisioned
capacity bound. It removes capacity-driven alignment/phantom work and unused
state buffers; it is not a faster-matmul claim or a training speedup.

CompiledMemoryStats.temp_size_in_bytes: native-packed 1,353,104,512 ->
1,252,502,752; both CP rows 47,254,802,176 -> 362,021,856. Runtime per-device
allocator peak_reserved: native-packed 1,172,357,120 -> 1,088,421,888;
CP 40,057,143,296 -> 290,390,016. These are different accounting measures;
neither is a measured MaxText training live-HBM peak. The fixed H32 control's
compiled temp increases by 8 MiB, while runtime peak_reserved is unchanged;
other no-CP controls' compiled temp is unchanged. Logical buffer removal alone
must not be advertised as an equal reduction of physical peak allocation.

Independent XPlane captures agree in direction for the complete jit_local
device module. No-CP captures have 20 calls per round; the long CP capture
hit the collection window and its trace-viewer JSON is unusable. Those original
files are retained but excluded from acceptance. Replacement CP captures have
two complete calls on every rank per round, in two independent rounds. Native
XPlanes are traversed with XProf's xplane_tools; parallel rank times are not
summed. No jax.debug.callback or io_callback is inserted into the graph.

## Correctness and resource evidence

- Forward branch alone: 32 native tests passed (forward fusion plus 13 resource
  tests); inference branch alone: 22 native tests passed.
- Cumulative integration: 66 native tests passed, including optional-state,
  packed backward/gradient compaction and resource tests. Earlier 78 native
  backward/CP tests also passed; these sets overlap and must not be summed.
- The final workspace matrix has seven native cases: CP4/8, B1/2,
  saved/rematerialized h, fast/overflow/group-only, suffix padding, a fully
  padded rank, all padding and outer remat. Every output and seven gradients
  compare to the unchanged full-capacity public operator at relative RMS<=1e-5.
- Normal no-CP A/B outputs and all gradients are bitwise equal. CP token outputs
  and token gradients are bitwise equal; parameter-gradient differences are
  checked separately and are below the same scheduling-equivalence bound.
- An isolated dynamic forward boundary table reproduces the old error:
  requested SMEM 1,049,088 > 1,048,576 bytes at N_max=262144. The guarded source
  compiles and its output is bitwise equal to the tight-table reference.
- A no-allocation JAXPR audit at B2/N8192/H64/K=V128 confirms a 64 GiB logical
  absent-dht tensor in the baseline and no GiB-sized corresponding aval after
  the fix. This is symbolic evidence, not a measured 64 GiB runtime saving.
- Full native public forward/VJP at N_max=262144, B2/T128/H2/K=V128 is compared
  to tight capacity with the same inputs, including native packed backward.

## Limitations retained explicitly

The independent unchanged XLA FP64 screen still finds an existing dt_bias error:
about 0.969% relative RMS versus the 0.5% diagnostic limit, in both baseline and
candidate. The threshold and gold are not changed; its failing logs are retained.
Scheduling-equivalence success is not a claim of complete FLA/GPU precision or
training-loss alignment. A pre-existing CPU-interpretation multi-head packed VJP
NaN is also reproduced at the original forward head; native tests above pass.

This does not port token-compact training (#586), fix all repeated-label CP
chain semantics, adopt the separate NaN/numerical changes, or validate arbitrary
interior padding holes. Metadata and aligned workspaces still depend on local
token/chunk counts. State-output ABI, dtype, gate math and existing tolerances
are unchanged. No full-model AOT, CP16, 256K training, universal memory-fit or
universal no-regression claim is made. Keep the PRs Draft pending review.

## Reproduction

Use the recorded dependency freeze and leave PALLAS_INTERPRET unset:

```bash
export JAX_PLATFORMS=tpu,cpu
export LIBTPU_INIT_ARGS=--xla_tpu_impure_enable_large_2nd_minor_layout=true
python -m pytest tokamax/_src/ops/kda/pallas_mosaic_tpu_workspace_tpu_test.py -q
python -m tokamax._src.ops.kda.pallas_mosaic_tpu_resource_benchmark \
  --cp 8 --heads 4 --group 2 --local-tokens 128 --capacity 8192 \
  --route bucket --warmup 5 --repeat 20 --output /tmp/kda-fast
# Add --over for the uniform fallback; --remat for internal state recomputation.
```

Run identical commands/harness on the unmodified baseline with --route plain,
alternating revisions across two rounds. The evidence archive retains raw
samples, input/output hashes, saved correctness arrays, XML, HLOs, XPlanes,
source archives, the failed prototypes and unchanged failing FP64 screens.
