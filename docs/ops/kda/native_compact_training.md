# Native segment-ID compact KDA

Source baseline: `primatrix/pallas-kernel` commit
`dca6a09feec2659f2777a6db02f2ee55560f991e`.

This port moves the accepted original-token forward and matching M5 backward
into Tokamax. The original-token grid contains `T / 64` chunks instead of
rounding every sequence independently. Per-token segment IDs determine gate
scan resets, same-segment causal masks, incoming-state ownership, and the
outgoing recurrent state. Full chunks, padded chunks, two-fragment boundaries,
and arbitrary-fragment boundaries retain the source's distinct fast paths.

The initial training dispatch matches the source production profile: non-CP,
BF16, BT=64, complete token chunks, bounded raw or preactivated gates, no user initial/final
state, and saved chunk states. Other legal calls retain the existing aligned
implementation. The opt-in configuration must be explicit in benchmarks.

Tokamax's reference maintains state per label, including interleaved labels.
Already grouped labels keep the direct path. Other inputs are grouped by a
stable token permutation, then outputs and gradients are restored to original
order. This retains per-label token order and handles interior padding without
expanding the token grid. Canonical ordinal IDs bound metadata independently
of the magnitude of public labels. Original input arrays are retained outside
the conditional, so the normal training path does not need sorted input copies.

For the same grouped-label contract as the source production path, set
`Config(compact_forward=True, compact_sorted_segments=True)`. This static
contract omits the reorder branches and their worst-case buffers entirely.
It requires nondecreasing positive labels followed by zero padding; the
constraint is checked eagerly and exposed to `checkify` under JIT. Without
this declaration, interleaved labels remain supported through the general
path. Benchmarks state which contract is selected.

## Residual contract

Q/K/V/beta and gate inputs retain original packed order. Aqk/Akk and gate
prefixes use the compact token domain, and saved h has exactly T/64 entries.
The backward must consume this same domain; aligned backward must never be
given compact residuals. Both public head-first layout and internal batch-first
layout conversions are included in end-to-end measurement.

The forward increment defines the compact tape; the dependent backward
increment supplies its native M5 VJP. Compact differentiation requires both
increments. Existing aligned configurations retain their original VJP. The
completed compact stack avoids aligned input/residual materialization in both
directions.

## Numerical and resource contracts

Preserve the source gate scan/scale order, BF16 boundaries, FP32 recurrent
state, boundary masks, and fused-gate backward eligibility. Keep Tokamax's
public beta/gate validation and its established unsupported-shape fallback.
No private `tops` package is a runtime dependency. Hardware limits are derived
from Tokamax/JAX device information rather than import-time device probing.
Unused initial/final-state capacity buffers must not be reintroduced.

## Validation and performance protocol

Compare fixed input arrays and cotangents against the pinned source and an
independent recurrence. Cover multiple batches/heads, full/partial chunks,
empty batches, padding, two and many fragments per chunk, large logical
segment capacity, normalization, gate parameters and all enabled gradients.
Run existing aligned-path regression tests without changing tolerances.

On an independent TPU7x Pod, freeze dependencies and effective flags. Compare
the pinned source and the completed port on the same data. Report
forward-only and complete forward+VJP separately. Warm up 5 times and measure
20 synchronized calls in each source/port/port/source order (40 samples per
implementation). Exclude compilation and input creation. Save compiler memory
analysis, HLO and XProf captures; report full device-module time separately
from host wall time and individual Pallas events. Repeat any measured gain
with a second independent capture before making a performance claim.

Results and exact measurement contracts: [compact training results](compact_training_results.md).
