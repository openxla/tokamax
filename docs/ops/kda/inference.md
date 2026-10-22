# Standalone native KDA inference

The experimental `inference.kimi_delta_attention_inference` entry point ports
`tops/ops/kda/forward_inference{,_kernel}.py` from primatrix/pallas-kernel
`c82625a2bdadd5881b12f1ab37ec3fa2004c9780`. Inputs and output are batch-first
[B,T,H,D], as in that standalone source API. Existing training APIs and VJPs
are unchanged; no inference derivative is registered.

This is one native segment-ID Pallas call, with in-kernel Q/K normalization,
safe gate activation, intra-chunk solve, recurrence and output. It creates no
aligned token buffers and stores no Aqk/Akk/gate-prefix/hidden-state residual
tape. Logical K/V dimensions may be padded for TPU alignment. Final states
are optional FP32 [B,N,H,K,V]; empty slots retain h0 or zero.

The port retains the BF16-only contract, T divisible by 64, positive ordered
contiguous segment IDs with trailing zero padding, explicit max_num_segments,
safe raw gates with lower_bound in [-5,0), and in-kernel Q/K normalization.
Beta is clipped to [0,1] and a_log capped at 80, matching the inference source.
It does not support CP. Tuning uses Tokamax hardware limits rather than source
environment overrides; source training/aligned alternatives are not enabled.

The source boundary solve assumed at most two segments in a token block. A
three-segment test exposed omitted outputs and corrupted carried states.
Boundary blocks now use a token recurrence for arbitrary segment counts;
full and single-segment partial blocks retain the source fused fast paths.
This correctness change can affect boundary-heavy workloads and must be timed.

Run inference_test.py for comparison against Tokamax's independent XLA
recurrence, including in-block boundaries, padding, initial/final states and
empty slots. CPU interpretation is not TPU compilation or performance proof.
Run `python -m tokamax.benchmarks.kda_inference` on TPU to compare the native
entry with existing Mosaic forward using the device timer. Keep this PR draft
until TPU lowering, memory feasibility, accuracy and timings are recorded.

Local results: 16 inference checks passed (including relative L2 error <=2%
against XLA and race detection), 33 CI shard checks passed, shard consistency
passed, and four TPU benchmark cases collected without timing. Interpreter
execution completes before the CPU reference starts to avoid callback dispatch
contention. Chunk size and head grouping are not public API tuning arguments.

Token selection in the boundary recurrence uses mask/reduce, and output
updates use select. JAX 0.11.1 TPU lowering does not support dynamic_slice or
dynamic_update_slice even when the interpreter accepts them. The loop remains
rolled. Four inference_lowering_test.py cases export to an abstract TPU7x
target with interpretation disabled, covering both initial/final-state options.
These check Pallas lowering, not device execution or performance. The updated
kernel passes the 16 numerical tests and four export tests locally.
