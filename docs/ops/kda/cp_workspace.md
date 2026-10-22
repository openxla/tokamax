# Opt-in CP resource scheduling

`pallas_mosaic_tpu_workspace.kimi_delta_attention_cp` wraps the existing public
Mosaic operator without changing its defaults or gate arithmetic. Inputs remain
head-first. A small static capacity is selected only after a collective confirms
that every rank and every batch row fits; the fallback preserves the original
maximum. Forward saves public operands and the predicate, then backward rebuilds
only the selected branch's residuals. Ordinary cond AD must not join residuals
from both capacity branches.

The stateless CP adapter bounds its physical local capacity by local T: more
than T nonempty runs cannot occur in T tokens. The caller's logical maximum
and the collective fast-branch test are unchanged. This bound is not applied
to stateful calls with a capacity-sized state input/output ABI.

Global segment labels can exceed a small local capacity. The adapter opts into
local ordinal state-slot indices for aligned backward metadata. CP chain
derivation still uses the original public labels, and forward's chunk-to-sequence
mapping already uses local ordinals. This separation is required before a small
capacity is safe; clipping labels would merge distinct segments and is forbidden.

Sequential head groups split query/key/value/gate/beta and their gate parameters
in the same order. Each group is checkpointed. This trades extra computation and
collective launches for bounded workspace liveness and is deliberately opt-in.
Initial/final states are not supported by this training adapter; the unchanged
operator remains available for those contracts. A head map is bypassed when its
group size is zero or equals H.

Acceptance requires real multi-device TPU comparisons of the complete public
output and VJP, including exact-capacity and one-rank-overflow layouts, B=1/2,
padding, gate-parameter gradients, and saved/rematerialized execution. Measure
complete forward plus backward and compiled/device memory with identical input
and cotangent hashes. No performance claim follows from an internal kernel timer
or a baseline that does not compile. Resource scheduling does not repair unrelated
CP repeated-label chain semantics or change numerical thresholds.
