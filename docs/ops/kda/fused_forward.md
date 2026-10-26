# Non-CP KDA forward fusion

The existing Mosaic forward writes `w`, `u`, `kg`, and `g_cumsum` between
the intra-chunk solve and the state/output kernel. The fused path executes
both stages in one Pallas call, passing these values through VMEM scratch
refs. It uses the existing compute bodies, including the dtype conversion
at the former HBM boundary, to isolate fusion from changes to the algorithm
or its precision policy.

This follows the persistent forward execution in
[`primatrix/pallas-kernel` at c82625a2bdadd5881b12f1ab37ec3fa2004c9780](https://github.com/primatrix/pallas-kernel/blob/c82625a2bdadd5881b12f1ab37ec3fa2004c9780/tops/ops/kda/chunk_fwd_mega.py).
The Tokamax adaptation deliberately reuses the already-landed intra and
state/output arithmetic. It does not import the source repository's packed
DMA, output workspace API, hardware tables, backward fusion, or CP code.

## Execution and support

The grid is `(heads // mini_batch, batch, chunks)` with an ordered final
axis. Each head group keeps its FP32 recurrent state in VMEM across chunks.
At each segment boundary it loads the corresponding initial state or zeros.
It writes the final state when requested; empty segments retain their initial
state, or zero when no initial state was provided.

```text
existing aligned inputs and segment metadata
    -> gate activation / prefix sum / intra-chunk solve
    -> VMEM: w, u, kg, gate prefix, Aqk, Akk
    -> state recurrence and output
    -> existing output unalignment and public layout
```

`Config.fuse_forward` selects the fused path for non-CP calls with K/V
multiples of 128. CP and other widths continue to use the existing staged
implementation. `Config(fuse_forward=False)` retains the staged path for
comparisons. The public API and implementation selector are unchanged.
Chunk size remains 64. Both BF16 and FP32 use their existing solve and
matrix-multiplication precision policies.

Mini-batch sizing includes the persistent state, bridge buffers, input/output
buffers, and solve temporaries, using Tokamax's hardware-derived VMEM budget.
Compiler allocation and performance must still be checked on TPU; CPU
interpretation does not validate Mosaic lowering or memory capacity.

## Backward contract

There is no change to `KdaResiduals`, preprocessing, or backward code.

| Forward mode | HBM residual outputs from the fused call |
| --- | --- |
| Forward only | None |
| Saved-state VJP | Aqk, Akk, gate prefix, chunk-start h |
| Rematerialized VJP, activated gate | Aqk, Akk |
| Rematerialized VJP, pre-activated gate | Aqk, Akk, gate prefix |

Prepared q/k/v/beta and the existing optional gate and normalization inputs
remain on the residual tape. Aqk/Akk retain their logical last dimension of
64. Fusion removes the stage-to-stage materialization of `w/u/kg` and avoids
materializing unused `qg`; it does not eliminate the residuals needed by the
existing backward. The saved `h` has the same dtype as the staged path.

## Validation and measurement

The focused tests compare fused and staged forward-only execution, VJP
execution, gradients, and the residual tree. Cases include BF16/FP32,
fixed/packed inputs, initial/final state, empty state slots, normalization,
gate activation, and both rematerialization settings. The existing KDA
kernel suite continues to compare Mosaic with XLA, including CP fallbacks.

Packed backward masks unwritten padding lanes before gate-parameter
reductions and restores the final-state cotangent for empty initial-state
slots. Focused tests compare the complete gradient tree, explicitly require
zero padding gradients and the empty-slot identity derivative, and retain
the original tolerances without xfails for these cases.

Non-TPU interpretation uses highest matrix-multiply precision to avoid a
backend-selected reduced-precision FP32 reference. Interpretation also keeps
optimization barriers at the rounded inter-phase tensors: the staged path
materializes these tensors between calls, whereas the fused interpreter
otherwise compiles both phases together. Native Mosaic keeps the VMEM
boundaries and its existing precision configuration. CUDA validation of the
interpreter comparison is tracked separately from native TPU correctness.

```bash
python -m pytest tokamax/_src/ops/kda/pallas_mosaic_tpu_fwd_fused_test.py -v
python -m pytest tokamax/_src/ops/kda -v
python -m tokamax.benchmarks.kda --skip_implementations=xla
```

The benchmark compares staged and fused Mosaic on the same primary fixed
and packed N=25 BF16 workloads (B=1, T=8192, H=32, K=V=128). It measures
forward and forward+VJP using `tokamax.benchmark`'s default device timer,
including preprocessing and output unalignment. Random gate parameters
are retained. Report hardware, JAX/JAXLIB/libtpu versions and measurements
together; performance is not inferred from the eliminated HBM traffic.

### Initial validation (2026-09-07)

Windows CPU, Python 3.12.14, JAX/JAXLIB 0.11.1:

- Focused interpret cases: 9 passed, 4 xfailed for the staged packed
  raw-gate parameter-gradient limitation described above.
- Additional fixed raw-gate gradient cases: 2 passed.
- KDA API/adapter tests: 36 passed, 9 subtests passed.
- CI shard consistency and its 53 unit tests passed.

TPU lowering, device correctness, VMEM
allocation, and staged/fused device-time measurements remain pending.
No speedup is claimed from the CPU tests. Keep the PR in draft until these
checks have completed; adjust the fused dispatch heuristic if measured
performance or compiler resource limits require a narrower support set.
