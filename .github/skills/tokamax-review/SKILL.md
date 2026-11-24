---
name: tokamax-review
description: >-
  Review guidelines for Tokamax changes. Use when reviewing a Tokamax pull
  request or changelist.
---

# Tokamax Code Review Guidelines

<!--
Shared by the GitHub pull request review and the Google-internal review. Keep
this file platform-neutral.
-->

In these guidelines, "the change" means the pull request or changelist under
review, and "the description" means its title and description.

Review each change once, where it was written. Do not apply these guidelines to
a changelist that was imported from a GitHub pull request (its description has
a `COPYBARA_INTEGRATE_REVIEW=` line). Pull requests are reviewed on GitHub.

## Reviewer Judgment

Apply these principles to every finding before posting it:

-   **Enforce hardware invariants over legacy workarounds:** When a change
    enforces a hardware requirement (such as MXU/VPU tile dimensions,
    lane/sublane alignment, or memory-space bounds) or fixes buggy behavior, do
    not ask for special-case dtype or caller workarounds just to keep legacy
    callers or stale tests passing; ask to update the stale callers or tests
    instead.
-   **Make every concern concrete:** Before raising a regression or numerical
    risk, name the exact configuration that triggers it (shapes, dtypes,
    hardware generation, caller) and verify that it can occur. If you cannot
    make it concrete, drop the comment or ask it as a question.
-   **Stay in scope:** Ask only for what the change needs. Do not request
    unsolicited refactors or unrelated test coverage. Flag files outside the
    change's stated scope and suggest splitting them into a separate change.
    Suggest small, well-scoped PRs or CLs that make one logical,
    self-contained change as much as possible.
-   **Ask when intent is unclear; do not over-prescribe:** Author agents often
    apply suggested fixes verbatim. Reserve concrete `suggestion` blocks for
    unambiguous bugs or style rules; for unexplained design choices, ask the
    author to justify them and leave real tradeoffs to human reviewers.
-   **Trust the diff, not the reply:** On follow-up review rounds, verify every
    "Done" or "Fixed" reply against the updated diff before resolving it.

## Canonical Guidelines

Read [docs/contributing.md](../../../docs/contributing.md) and
[docs/code_style.md](../../../docs/code_style.md) in full, and review every
change against them.

## Severity Markers & Comment Style

-   `🔴` **Critical:** Correctness, crash, numerical, or security bug.
-   `🟠` **High:** A requirement in `docs/contributing.md` that the change does
    not meet.
-   `🟡` **Medium:** A rule in `docs/code_style.md` that the change does not
    follow.

Keep each inline comment to 1–3 clear sentences, use a `suggestion` block
only for unambiguous local fixes, and avoid repeating the same style nit
across multiple lines (flag the pattern once and note other affected lines).
