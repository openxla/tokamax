# How to contribute

We'd love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

### Feature additions, bug fixes, and optimizations

For PRs modifying existing code, follow these guidelines:

*   If this is a new feature in a specific kernel implementation, ensure that
    equivalent functionality is also introduced in the `base` plain JAX
    implementation of the operator. Add correctness tests that validate
    equivalent behavior between the two.

*   For any potentially performance-impacting changes, run a full benchmarking
    sweep across *all* defined `ArgSpecs` for the operator, and include the
    results in the PR, to avoid any shape-specific regressions. If the op does
    not have pre-defined ArgSpecs, create at least one that covers the common
    use-case, and include it in the benchmark results.

*   If your feature is under an opt-in flag (strongly preferred during
    development), be especially careful to ensure there is no accidental
    performance regression in the default path.

### Introducing new operators

If your PR introduces a new op, follow these guidelines:

*   Provide an explanation (preferably in separate markdown documentation) of
    the origins of the operator (e.g., a published paper), what the operator
    does, where it is used, how generalizable it is (or if its very model- or
    use-case specific), and so on. This is useful to understand its value in a
    general-purpose kernel library. At a minimum, update the "supported ops"
    documentation page with a description of the new op.

*   Begin by implementing the op in `_src/ops/experimental/`. These ops are
    intended for rapid iteration and exploration, and may change frequently and
    significantly, with generally lower expectations around ongoing maintenance
    and support. Once the implementation is complete, well-tested, and validated
    in end-to-end settings, it can be moved to `_src/ops/`, where there is a
    heavier expectation of maintenance burdens.

*   Follow the existing
    [repository structure](https://openxla.org/tokamax/library_design) and
    [coding style](https://openxla.org/tokamax/code_style).

*   Ensure that the operator has a plain JAX implementation, used both to
    validate the kernel's correctness and as a fallback on unsupported devices.

*   Define `ArgSpecs` for the operator that cover input definitions for all
    expected use-cases, and run a full benchmark sweep across them, to establish
    a baseline performance metric in comparison to the plain JAX implementation.
    If there is negligible speedup compared to the baseline, clearly explain why
    a kernel implementation is justified.
