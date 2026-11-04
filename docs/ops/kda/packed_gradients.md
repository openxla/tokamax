# Packed KDA backward padding and empty states

This correctness increment follows the saved-state backward fusion PR and
precedes native packed layout optimizations. It keeps aligned preprocessing
and the existing residual contract.

The reverse recurrence does not write every padding or empty-state output.
CPU Pallas interpretation exposed non-finite token gradients, empty initial
state gradients, and gate-parameter reductions in both staged and fused
paths. A regression against XLA autodiff failed before this change.

Before gate differentiation and Q/K normalization backward, zero gradients
outside aligned sequences. Replace the raw-gate padding sentinel with a
finite value before reducing gate-parameter derivatives, avoiding both
unwritten scratch values and zero times negative infinity. Empty sequences
have final state equal to initial state, so their initial-state cotangent
is the incoming final-state cotangent, or zero when no final state is used.

Regression coverage includes both gate formulas, saved/recomputed chunk
states, staged/fused backward, and final-state output enabled/disabled.
Tests compare the complete gradient tree with XLA autodiff, require finite
values, and explicitly check zero padding gradients and empty-state identity.
Existing BF16/FP32 forward and backward equivalence checks now also compare
padding and empty states. No CPU xfail remains in these focused tests.

Local validation (Python 3.12.14, JAX/JAXLIB 0.11.1, CPU interpretation):
45 focused tests passed with no xfails; 36 API/adapter tests and 9 subtests
passed. Formatting and `git diff --check` passed.

TPU lowering, device correctness, and performance remain pending. This is
not a native packed implementation or a measured performance improvement.
