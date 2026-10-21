# Fused non-CP KDA state rematerialization

This increment extends the backward fusion work to calls with
`rematerialize_for_backward=True`. It uses the two-pass reconstruction and
reverse traversal structure of
[`primatrix/pallas-kernel` c82625a2bdadd5881b12f1ab37ec3fa2004c9780](https://github.com/primatrix/pallas-kernel/blob/c82625a2bdadd5881b12f1ab37ec3fa2004c9780/tops/ops/kda/chunk_bwd_mega.py),
while retaining Tokamax's aligned metadata, compute formulas, and residual
contract. It does not import the source's native packed or manual DMA paths.

## Execution

Enable with `Config(rematerialize_for_backward=True,
fuse_backward=True, fuse_rematerialization=True)`. The new option defaults
to false pending TPU compilation and device-time validation. CP, non-aligned
K/V, disabled backward fusion and saved-state calls keep their prior dispatch.

After the existing gate-prefix recomputation, the first Pallas call combines
WY recomputation and the forward state recurrence. Per-chunk w/u/gated-key
intermediates remain inside the kernel. State recurrence uses FP32 and
resets at each sequence start. Inactive chunks are initialized to zero.
The call returns the existing rounded h snapshots and FP32 v_new.

The second Pallas call traverses chunks in reverse. It recomputes w/qg/kg
in VMEM, consumes rematerialized h/v_new, and fuses dAqk/dv0 with the existing
local backward stages. It does not repeat the u matmul or reconstruct v_new
from a rounded state snapshot. Gate-parameter reduction, normalization
backward and unalignment retain their existing implementations.

Keeping FP32 v_new is intentional: the state recurrence uses the full FP32
running state, but h snapshots use the input dtype (including BF16). Using
those rounded snapshots to reconstruct v_new would change the established
low-memory path's numerics. Direct tests compare both h and v_new with the
jitted staged reconstruction, in addition to end-to-end gradient tests.

The forward residual tree is unchanged and still omits h in this mode.
During backward, h and FP32 v_new are still materialized in HBM between the
two calls. This change removes the w/u/qg/kg and dAqk/dv0 HBM boundaries; it
does not claim a single-call backward, removal of all temporary state, or
measured peak-memory savings. Hardware-derived head-group sizing includes
state and recomputation temporaries.

## Validation and benchmark

CPU interpretation tests cover fixed-length BF16/FP32 gradients, raw gate
parameters, initial/final states and padding/empty-state semantics. Further
tests compare reconstructed h/FP32 v_new directly, exercise two heads and
two batches with K/V=256, and check saved-state/disabled-fusion fallbacks.

```bash
python -m pytest tokamax/_src/ops/kda/pallas_mosaic_tpu_remat_test.py -v
python -m tokamax.benchmarks.kda --skip_implementations=xla,mosaic_staged,mosaic_fwd_fused,mosaic_fused
```

Compare `mosaic_remat` against `mosaic_remat_fused`, both with identical
forward fusion and `rematerialize_for_backward=True`. Record forward+VJP
using the default device timer, hardware and JAX/JAXLIB/libtpu versions.
Local checks (Python 3.12.14, JAX/JAXLIB 0.11.1): 9 focused tests passed,
66 existing backward/API/adapter tests and 9 subtests passed, and 33 CI shard
tests passed. Shard consistency and all 28 benchmark collections passed.

CPU interpretation does not establish TPU lowering, allocation, accuracy or
performance. Those device checks remain pending; keep the option disabled
by default and the PR in draft until they are available.

Packed gradient coverage is enabled by the separate padding-correctness PR; native packed DMA coverage is added by the packed I/O PR.
