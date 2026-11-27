# Non-CP KDA backward with state rematerialization

Enable with `Config(rematerialize_for_backward=True, fuse_backward=True,
fuse_rematerialization=True)`. The options remain opt-in. CP, unsupported
K/V, and explicit mini-batch schedules retain their established paths.

The reconstruction uses a parallel Pallas WY pass that writes rounded W/U
and gated keys, followed by the canonical staged state recurrence. The local
reverse pass then shares the staged backward body, keeping its recomputation
bridges, dAqk and dv0 in VMEM. This replaces the earlier producer-inside-state
loop: that loop increased long-sequence dQ and gate-parameter errors even
when short interpreted tests passed. No numerical budget was enlarged.

The shared state launcher preserves FP32 h snapshots for varlen execution,
FP32 value corrections, initial states, and the fixed-length snapshot dtype.
The forward tape still omits h. Reconstruction temporarily materializes
W/U/gated keys and h/value corrections in HBM; this is not a single-call
backward or removal of all state workspace.

The reverse body applies Q/K normalization before storing activation-dtype
q/k/v gradients. Scalar beta/db use token-minor tiles. Cotangents stay in the
activation dtype in HBM and promote inside the kernels. Inactive chunks skip
matmuls and write zero gradient tiles; gate-parameter reductions retain their
per-token padding masks. Head-group caps follow the hardware VMEM policy.

Tests compare live reconstructed states and FP32 corrections with the shared
staged launcher, including multi-head packed cases. Inactive intermediate
state slots are undefined in that launcher; a separate poisoned-state test
requires finite, exactly zero inactive gradient outputs. Public output and
gradient checks continue to include padding and empty-state semantics.

See the [November 15 performance and numerical follow-up](https://github.com/antgroup/tokamax/blob/9b67d57c482c0cdb621b54d9162c82de5daf62f8/docs/ops/kda/backward_performance_followup_20261115.md)
for frozen revisions, raw samples, independent FP32 reference errors and
unfavorable historical controls. CPU interpretation is not a native-device
performance result. Review readiness follows the internal numerical contracts
and final-head checks in that report; bespoke parameter-RMS diagnostics are
not additional acceptance gates.
