# Opt-in compact all-gather context parallel KDA

This follow-up composes the existing aligned CP state exchange with the native
original-token forward/backward. It does not change the public operator default.
The implementation depends on the shared compact core in #1688/#1689 and CP
state-slot/workspace support in #1690. While those dependencies are unmerged,
this branch carries their shared code; review the new `cp_compact.py`,
`cp_compact_test.py` and this document as the feature-specific change.

## Algorithm

Within the staged run cap (default 64), select the aligned result for every token.
Above the cap, only the first/last positive label can continue across a rank
boundary: stage those labels in bounded aligned workspace and select the native
result for interior labels. The native pass always runs so a conditional does not
retain two differently sized reverse tapes. The local native capacity is bounded
by token count, not global label magnitude. Source is internal #687 at `18ea6891`.

Tokamax preserves per-label recurrence. Globally grouped positive labels are
required; arbitrary interleaved labels spanning ranks are outside this Draft.
Zero padding is permitted. Native grouped records retain physical positions;
a label resuming after padding uses the stable grouping fallback. Initial/final
states, convolution and Q/K normalization are unsupported by this training entry.
Inputs are BF16 [H,1,T_local,D], with local T divisible by 64. Gate parameters may
be FP32. The opt-in entry is:

```
from tokamax._src.ops.kda.cp_compact import kimi_delta_attention_cp_compact
# Call inside shard_map defining context_parallel_metadata.axis_name.
output, _ = kimi_delta_attention_cp_compact(
    q, k, v, g, beta, segment_ids=ids,
    context_parallel_metadata=context, a_log=a_log,
    delta_time_bias=bias, use_gate_in_kernel=True, lower_bound=-1.0,
    staged_run_cap=64,
)
```

## Correctness evidence

Dedicated TPU7x, four chips/eight logical devices; JAX/jaxlib 0.12.0,
libtpu 0.0.50.1, large second-minor layout enabled, no interpretation.

```
python -m pytest -q -x tokamax/_src/ops/kda/cp_compact_test.py
```

7 passed (65.89 s) on the final shared native carry fix. CP4/CP8; H4/B1;
T_local128 (global512/1024), K=V128; BF16 tokens and raw gates, FP32 gate
parameters; seed619. Global positive IDs are shifted by 262100, beyond the local
staged capacity. Tests cover within-cap, split cap2, interior/tail padding,
all-padding mesh and empty ranks. Compare output and all seven gradients against
the existing aligned CP route; parameter gradients are psummed across ranks.
The new feature reuses the existing compact/reference 2% relative-RMS bound and
finite checks; no existing CP tests or numerical bounds were loosened.

The shared forward/backward carry and padding fixes have independent native
reference and scratch-poison checks in #1688/#1689. The full internal-main
migration ledger inventories 24 KDA source files and all three post-port commits.

## Final review validation

Native CP4/CP8 tests now include H64, long local token windows, nested scan and
rematerialization. Two-process CP16 validation covers H64/global256K, randomized
packing and a highest-precision independent recurrence. Default cap64 and forced
split cap2 whole-call wall time, full-device traces and compiler temporary bytes
are recorded on the final M5 parameter-reduction fix. All original numerical
bounds remain unchanged. See [full final-source evidence](https://github.com/antgroup/tokamax/blob/9b67d57c482c0cdb621b54d9162c82de5daf62f8/docs/ops/kda/review_validation_20261116.md).

Cross-rank continuation follows the aligned parent's contiguous owning-rank
chain contract. A sequence resuming across an entirely empty rank and arbitrary
cross-rank interleaving are outside that contract. Existing empty-rank fixtures
place padding between unrelated runs. Intra-rank padding/resumed-label grouping
is retained. Full-model loss, peak training memory and whole-model throughput
remain separate integrations; the report measures the operator and its VJP.
