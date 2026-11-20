# Capacity-dependent resource protection

Small metadata keeps its existing schedule. Large sequence tables must not be
scalar-prefetched in full into SMEM. Forward launchers instead gather the two
aligned boundaries for each physical chunk before entering Pallas; packed
input launchers do the same for the original boundaries. The per-chunk table
and sequence mapping are shared by all head groups.

Output compaction declines the full-VMEM compactor when the boundary tables
exceed its scalar-prefetch budget. Its gather fallback uses a binary search
only for large tables, preserving the existing small-table path. Empty and
trailing sequence slots must not change token routing.

For capacities above 8192, alignment uses binary-search routing instead of
an O(capacity * aligned_tokens) scan or validity matrix. Its static token bound
uses at most min(capacity, original_tokens) nonempty runs, plus one zero guard
chunk. Every nonempty run consumes at least one token; empty capacity slots add
no aligned tokens. The same bound is used for token tensors and segment IDs.
Public boundary/state capacities remain unchanged, including empty state slots.

This change does not introduce token-compact training, renumber public segment
IDs, change recurrent-state shapes, or change matmul precision. Resource claims
require compiled HLO/allocation evidence, and performance acceptance covers the
entire public forward plus VJP rather than only a Pallas call.
