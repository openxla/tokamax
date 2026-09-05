# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test shard definitions for `.github/workflows/ci-build.yml`.

The test suite is too slow to run as a single job, so it is split into shards
that run in parallel, one GitHub Actions job per (shard, runner). `SHARDS`
below is the only place a shard is declared; the job matrix and the catch-all
shard are derived from it. Stdlib only, so CI can run this before installing
anything.

Every test file is named individually, and paths are written out in full, with
no directory targets, shared prefix constants, or interpolation. It is
repetitive, and that is the point: a shard's targets read the same way they
would on a pytest command line, every path can be found by grepping for it,
and a file that is deleted, renamed, or newly added shows up as a diff here
instead of silently changing what some directory expands to.

A file too slow to run in one job can be split across shards by pytest node
ID -- `path/to/test.py::SomeTest` -- at the class boundary. `check_consistency`
parses any file split this way and fails if it declares a class no shard names,
since such a class stops running without any path here going stale.

Usage:
  python3 .github/ci/shards.py matrix  # emit GitHub Actions outputs
  python3 .github/ci/shards.py check   # every test file is claimed by a shard
  python3 .github/ci/shards.py list    # human-readable summary
"""

from __future__ import annotations

import argparse
import ast
import collections
from collections.abc import Callable, Collection, Iterable, Sequence
import fnmatch
import json
import os
import shlex
import sys
from typing import TypedDict

# Runner label -> (device extra it installs and tests, short name used in the
# job title). One table rather than two keyed by the same labels, so a runner
# cannot be listed in one and missing from the other.
#
# Short display name only; runs-on still uses the full runner label above.
#
# Every shard runs on every runner. Which runners a test actually needs is a
# per-file fact -- it wants a TPU or it does not -- and saying so is a
# follow-up, deliberately not part of this change: running everything
# everywhere is the behaviour the suite has today, and it is the baseline the
# new shard set has to be compared against before anything is pinned.
RUNNERS = {
    'linux-x86-a3-8g-h100-1gpu': ('cuda', 'h100'),
    'linux-x86-ct6e-44-1tpu': ('tpu', 'tpu6e'),
    'linux-x86-tpu7x-56-1tpu': ('tpu', 'tpu7x'),
}

# pytest's default `python_files`. This repo almost always uses the suffix
# form, but pytest collects both, and `all_test_files` has to enumerate the
# same universe pytest does: a file this misses is one no check here can
# reason about at all -- not covered, not unclaimed, just invisible.
PYTEST_FILE_PATTERNS = ('test_*.py', '*_test.py')

# Flags that decide *which* tests run. Emitted once, as the `pytest_flags` job
# output, so every shard is selected the same way; the reporting-only flags
# (`-s`, `--durations`) are written in the workflow instead.
PYTEST_SELECT_FLAGS = (
    # Skip the tests marked `long`, which the weekly job runs instead of
    # presubmit. The marker is declared in pyproject.toml, under
    # [tool.pytest.ini_options]. Two argv entries because pytest takes the
    # marker expression as a separate argument; one logical flag.
    '-m',
    'not long',
    # `test_base.py` holds the abstract `*TestBase` classes. Each requires the
    # implementation under test as a keyword-only argument (`attention_fn`,
    # `dot_fn`, `norm_fn`, `glu_fn`, `flex_attn`) that a subclass in some real
    # test file supplies. Collected on its own there is nothing to supply it,
    # so pytest instantiating one raises TypeError.
    '--ignore-glob=*/test_base.py',
    # `test_utils.py` is helper functions: no test classes, no test functions.
    # Collecting it finds nothing, but it still has to be named here or the
    # coverage checks below would report it as running in no shard.
    '--ignore-glob=*/test_utils.py',
)

# The `--ignore-glob` patterns above, which the coverage checks need too: a
# file pytest never collects must not be reported as running in no shard.
# Read back off the flags rather than written out a second time, so the model
# cannot end up disagreeing with the command line the test jobs run.
IGNORED_GLOBS = tuple(
    flag.removeprefix('--ignore-glob=')
    for flag in PYTEST_SELECT_FLAGS
    if flag.startswith('--ignore-glob=')
)

# The themes shards group into, in the order the job list should read. A
# shard's name must be a theme or start with `<theme>-`; `check_consistency`
# rejects one that matches none, so a new shard cannot land ungrouped.
#
# Grouping is carried by the name rather than by a separate `theme` field so
# that the name in a job title, in `--durations` output, and in this table are
# the same string, and there is nothing to keep in agreement. Within a theme,
# `build_matrix` orders by `minutes`, longest first: the top of each group is
# the job that sets that theme's wall clock. Shards with no timing sort last,
# because an absent number is an unknown, not a zero.
THEMES = (
    'attention',
    'splash',
    'ragged-dot',
    'gmm',
    'ops',
    'experimental',
    'core',
    'catch-all',
)

# Test files intentionally run by no shard.
EXCLUDED_TESTS = (
)


class _RequiredSpec(TypedDict):
  """The part of a shard spec every shard has."""

  # `None` only on the catch-all shard, and only until `resolve_shards` fills
  # it in. Every other shard names its targets here.
  paths: tuple[str, ...] | None


class Spec(_RequiredSpec, total=False):
  """One shard's entry in `SHARDS`. See the table below for what each means.

  Split across two classes because `paths` is required and the rest are not,
  which is the only way `TypedDict` spells that.
  """

  minutes: float


# Shard name -> spec, as `SHARDS` and as `resolve_shards` returns it.
ShardMap = dict[str, Spec]

# One `strategy.matrix.include` entry: a (shard, runner) job. Every value is a
# string because that is what the workflow interpolates.
MatrixEntry = dict[str, str]

# This script describes what tests are in a given shard.
# All tests are automatically assigned a shard (oftentimes catchall).
# When specifying a test to a shard, ignore that test the shard by default.
# These shards are created first to distribute the load by timing,
# then by test area.
#
# Carried over from the SHARD_TEST_PATHS table in ci-build.yml, which this
# replaces. Two of those lines no longer describe the mechanism, and are kept
# because the intent behind them still holds:
#
#   "ignore that test the shard by default" was the `--ignore=` flag a
#   directory-targeted shard needed to carve out the files another shard had
#   claimed. No shard here targets a directory, so there is nothing to carve
#   out and no `--ignore=` anywhere in the table; see the note below.
#
#   "by timing, then by test area" is now the other way round. `shard_order`
#   groups by theme first and sorts longest-first inside each, so the job list
#   reads by area; the load balancing happens within a theme, not across the
#   whole table.
#
# One entry per shard.
#
#   paths    a tuple of the test files this shard runs, each optionally
#            narrowed to one class by a `::ClassName` node ID. Required,
#            except on the derived catch-all shard, which `resolve_shards`
#            fills in. Test files only: `check_consistency` rejects a
#            directory or a flag, both of which would break the coverage
#            model below.
#   minutes  measured wall clock, worst case across the runners the shard is
#            scheduled on. Set it only from a real run: omitting it means no
#            number yet, which is not the same as fast.
#
# There is no key for which runners a shard belongs on, and `Spec` has no
# field for one: every shard runs on every runner in `RUNNERS`. Restricting a
# shard is a follow-up, and when it comes it should be stated per test file
# rather than here -- a shard is a scheduling bucket, sized for wall clock and
# rebalanced whenever the timings move, so a per-file fact stated at shard
# granularity gets forced onto every file that happens to be grouped with it.
#
#            Every shard below is measured, from run 33423920015
#            (2026-08-31), a 72-job all-green run on all three runners. That
#            run's shards were coarser than these, so the numbers are not
#            copied across by name: they are per-file times recovered from the
#            job logs and re-summed against the `paths` here. pytest flushes a
#            progress line when the file finishes, so the timestamp on
#            `foo_test.py ....` is that file's end, and the gap back to the
#            previous one is its duration. The recovered per-file times account
#            for all but 0.5-0.9m of each job's wall clock, which is the
#            fixed cost either side of the pytest session.
#
#            Two traps, if this is ever redone.
#
#            The gap-back arithmetic needs every progress line, and a file
#            whose tests print to stdout does not produce a bare one: under
#            `-s`, `mla/pallas_mosaic_tpu_kernel_test.py` emits
#            `<path> Test case: ...`. Matching only `<path> <progress chars>`
#            drops it silently, and dropping it charges its time to whichever
#            file ran before it. Match on the path and take the rest as it
#            comes.
#
#            `attention/base_test.py` is split across two shards by node ID,
#            so its time is per class group, not per file: 28.9m for the two
#            classes in `attention-base` and 36.1m for the one in
#            `attention-base-vjp` on tpu6e. Summing the file to 65.0m and
#            giving that to both is the mistake to avoid. An earlier pass took
#            these from run 32866355103, whose shards also overlapped on
#            gmm_v2 -- `linear_softmax_and_experimental` ran it a second time
#            because of a stale `--ignore` path, since fixed in e6142dc.
#            Summing that overlap doubled `gmm-v2-perf` to 11m and
#            `gmm-v2-kernel` to 8m, which read as plausible drift and were
#            not. This run is post-fix, so base_test.py is the only overlap
#            left.
#
#            Rounded up to whole minutes, floored at 1, because the field must
#            be a positive number and 12 shards here finish in under a minute.
#            A 1 means "too fast to matter", not "measured at 60s".
#
# Every file is named. No shard targets a directory, which is why there are no
# `--ignore=` flags either: a directory target and the file shards carved out
# of it have to be kept in agreement, and naming files directly removes the
# thing they could disagree about. It costs a line per test file, and buys two
# checks a directory target cannot support -- a deleted or renamed file fails
# `check` by name rather than quietly shrinking a shard, and a *new* file is
# unclaimed rather than silently absorbed by whichever directory contained it.
# `check_consistency` enforces this: a directory target is an error, not a
# style preference.
#
# So adding a test file does require an edit here. Forgetting is not fatal:
# `resolve_shards` sweeps unclaimed files into the catch-all shard, where they
# run, but visibly, and their `--durations` tell you which shard has room for
# them. That is the catch-all's whole job, and it has now done it once: it
# measured 16m, all but a minute of which was
# `experimental/kda/pallas_mosaic_tpu_kernel_test.py`, a file nobody had ever
# scheduled and the fifth-longest job in CI. That is the mechanism working, so
# the fix was `experimental-kda-kernel` rather than a bigger catch-all, and
# this shard is back to reading as empty.
#
# Ordered by theme, longest first within each -- the same order `shard_order`
# produces, so this table, `python3 shards.py list`, and the job list all read
# the same way, and the first entry under a divider is the job that sets that
# theme's wall clock. It used to be measured-shards-first-then-the-rest, which
# stopped meaning anything once every shard had a number.
#
# Nothing enforces the order: `build_matrix` sorts by `shard_order` regardless,
# so an entry in the wrong place is a readability bug and not a CI one. If a
# `minutes` changes enough to move a shard, move it.
#
# Paths run past 80 columns rather than wrapping. This is the style guide's
# own exception -- "long string module-level constants not containing
# whitespace that would be inconvenient to split across lines such as URLs or
# pathnames" (Google Python style, 3.2) -- and not a limit worth working
# around here. The longest path is 90 characters, so it does not fit at any
# indentation, and the two ways to make it fit both cost more than they save:
# a wrapped path cannot be grepped for, which is half the reason paths are
# written out in full, and implicit concatenation inside a tuple is the exact
# shape of the bug this table exists to prevent -- drop a comma and two paths
# silently become one, which is how `ragged_dot_misc` lost an `--ignore`.
# pylint: disable=line-too-long
SHARDS: ShardMap = {
    # -- attention ---------------------------------------------------------
    # `base_test.py` was one ~65m job, the longest in CI, because
    # `AttentionTestBase` contributes its 32 methods to both
    # `DotProductAttentionTest` and the explicit-VJP subclass below: the file
    # is two full passes over the attention suite plus `MaskTest`'s 7, and
    # every one of the 32 is a `parameterized.product` or `.parameters` grid.
    #
    # The file declares exactly three classes and the two shards name all
    # three, so the split covers it completely. That is checked, not asserted
    # -- see `declared_test_classes` and the orphan error in
    # `check_consistency`.
    #
    # Split here on the class boundary, by node ID. The 29m/36m below are the
    # two halves of the old workflow's `-k` split on tpu6e, and are an upper
    # bound now, since each half paid its own startup there too. That puts
    # both under `splash-kernel`, which is why the split stops at
    # two: a third shard would not move the suite.
    #
    # `check_consistency` parses this file and fails if it grows a class that
    # neither shard names -- the failure the node IDs introduce, and the only
    # one no path here going stale would catch.
    'attention-base-vjp': Spec(
        paths=(
            'tokamax/_src/ops/attention/base_test.py::DotProductAttentionWithExplicitVjpTest',
        ),
        minutes=37,
    ),
    'attention-base': Spec(
        paths=(
            'tokamax/_src/ops/attention/base_test.py::MaskTest',
            'tokamax/_src/ops/attention/base_test.py::DotProductAttentionTest',
        ),
        minutes=29,
    ),
    'attention-triton': Spec(
        paths=('tokamax/_src/ops/attention/pallas_triton_test.py',),
        minutes=12,
    ),
    'attention-xla-chunked': Spec(
        paths=('tokamax/_src/ops/attention/xla_chunked_test.py',),
        minutes=11,
    ),
    # CUDA-pinned on measurement, not on inspection: 0 of its 257 cases run on
    # either TPU runner, and `jax_nn_test.py` says why -- two
    # `if jax.default_backend() == "tpu": self.skipTest("Not supported on
    # TPUs.")` guards, one per class, covering the file.
    'attention-jax-nn': Spec(
        paths=('tokamax/_src/ops/attention/jax_nn_test.py',),
        minutes=4,
    ),
    'attention-api': Spec(
        paths=('tokamax/_src/ops/attention/api_test.py',),
        minutes=2,
    ),
    'attention-api-sharding': Spec(
        paths=('tokamax/_src/ops/attention/api_sharding_test.py',),
        minutes=1,
    ),
    'attention-mosaic-gpu': Spec(
        paths=('tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py',),
        minutes=1,
    ),
    'attention-mosaic-tpu': Spec(
        paths=('tokamax/_src/ops/attention/pallas_mosaic_tpu_test.py',),
        minutes=1,
    ),
    # -- splash ------------------------------------------------------------
    # The longest TPU job, and the only shard whose runtime is set by something
    # other than the hardware: 337 tests, stable across runs, 37.5m on tpu6e
    # and 34.9m on tpu7x. It is no longer the longest job in CI -- that is
    # `ragged-dot-misc` at 43m, on h100 -- but it still sets the TPU wall clock,
    # so splitting it is the only thing that shortens a TPU run.
    #
    # It cannot be split the way `attention-base` below is. The file declares
    # one test class, so there is no class boundary to cut on, and its five
    # methods are `parameterized.product` grids -- `test_splash_attention_fwd`
    # alone is 256 of the 337 cases. absl names product cases by index
    # (`test_splash_attention_fwd0`, `...1`), not by argument, so neither a
    # node ID nor `-k` can address a subset of one grid by what it varies.
    # Splitting the remaining 81 cases off would leave a 30m shard and gain
    # ~10m; going further needs pytest-split, or a smaller grid upstream.
    'splash-kernel': Spec(
        paths=(
            'tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_test.py',
        ),
        minutes=38,
    ),
    'splash-misc': Spec(
        paths=(
            'tokamax/_src/ops/experimental/tpu/splash_attention/ring_attention_kernel_test.py',
            'tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel_sharded_test.py',
            'tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask_test.py',
        ),
        minutes=1,
    ),
    # -- ragged-dot --------------------------------------------------------
    # The longest job in CI, and the whole of it is one file: `api_test.py` is
    # 35.7m of the 42.8m `ragged-dot-misc` measured on h100 before this split,
    # and nothing else scheduled on h100 is over 13m. So the GPU wall clock was
    # this one file plus 7m of other people's tests waiting behind it.
    #
    # Split out rather than split up: 925 of its 1117 cases run on h100, all in
    # one class, so there is no class boundary to cut on and node IDs would not
    # help. On its own it sets the h100 wall clock at ~36m, which is under
    # `splash-kernel`, so cutting it further would not shorten a CI run --
    # something else has to get faster first.
    #
    # Not CUDA-pinned despite 7/1117 running on TPU: those 7 are real coverage,
    # and 6 seconds of TPU is not worth the chance of losing them silently.
    'ragged-dot-api': Spec(
        paths=('tokamax/_src/ops/ragged_dot/api_test.py',),
        minutes=33,
    ),
    'ragged-dot-mosaic-gpu': Spec(
        paths=('tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_test.py',),
        minutes=13,
    ),
    'ragged-dot-triton': Spec(
        paths=('tokamax/_src/ops/ragged_dot/pallas_triton_test.py',),
        minutes=8,
    ),
    # The ragged_dot tests with no shard of their own.
    'ragged-dot-misc': Spec(
        paths=(
            'tokamax/_src/ops/ragged_dot/base_test.py',
            'tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common_test.py',
            'tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py',
            'tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py',
        ),
        minutes=6,
    ),
    # GPU-only and additionally gated on sm100/B200, so both of these skip
    # every test on the H100 runner. They are shards rather than entries in
    # EXCLUDED_TESTS because the files are real tests that pass on the right
    # hardware; there is just no runner for them in RUNNERS.
    'ragged-dot-sm100-fp8': Spec(
        paths=(
            'tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py',
        ),
        minutes=1,
    ),
    'ragged-dot-sm100-i8': Spec(
        paths=(
            'tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py',
        ),
        minutes=1,
    ),
    # -- gmm ---------------------------------------------------------------
    'gmm-v2-perf': Spec(
        paths=('tokamax/_src/ops/experimental/gmm_v2/gmm_v2_perf_test.py',),
        minutes=6,
    ),
    'gmm-v2-kernel': Spec(
        paths=('tokamax/_src/ops/experimental/gmm_v2/gmm_v2_test.py',),
        minutes=5,
    ),
    # -- ops ---------------------------------------------------------------
    # These were three mixed bundles -- `experimental_misc`, `flex_and_scatter`,
    # `glu_and_norm`, `triangle_and_gathers` -- each holding two or three
    # unrelated ops because none was long enough to fill a job on its own.
    # Packing them saved runner slots and cost the shard name its meaning:
    # `flex_and_scatter` told you what was in it only if you already knew.
    #
    # Split one-op-per-shard. It costs 5 shards and 15 jobs at worst, and buys
    # two things. A job title now names the op it is testing, so a red X in the
    # PR checks says what broke without opening it. And it is the
    # granularity dependency-based selection will need: once a change can pick
    # shards, an edit to one of these ops should not drag the op it was packed
    # with onto three runners. Every shard runs on every PR today, so that
    # half is not paid back yet, and the 15 jobs are a real cost until it is.
    #
    # Not symmetric, and worth knowing before splitting more: every op reaches
    # `tokamax/__init__.py`, so a change to a *base* module still fans out
    # widely regardless of how these are packed. Splitting helps at the leaves,
    # not at the roots.
    'ops-flex-attention': Spec(
        paths=('tokamax/_src/ops/flex_attention/base_test.py',),
        minutes=13,
    ),
    'ops-linear-softmax-xent': Spec(
        paths=(
            'tokamax/_src/ops/linear_softmax_cross_entropy_loss/api_test.py',
            'tokamax/_src/ops/linear_softmax_cross_entropy_loss/base_test.py',
            'tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla_test.py',
            'tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel_test.py',
            'tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_test.py',
            'tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference_test.py',
        ),
        minutes=13,
    ),
    'ops-ragged-gather': Spec(
        paths=(
            'tokamax/_src/ops/ragged_gather/api_test.py',
            'tokamax/_src/ops/ragged_gather/base_test.py',
            'tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu_test.py',
            'tokamax/_src/ops/ragged_gather/pallas_mosaic_v2_tpu_test.py',
        ),
        minutes=8,
    ),
    'ops-causal-conv1d': Spec(
        paths=(
            'tokamax/_src/ops/causal_conv1d_gated_delta_rule/base_test.py',
            'tokamax/_src/ops/causal_conv1d_gated_delta_rule/pallas_mosaic_tpu_test.py',
        ),
        minutes=5,
    ),
    'ops-normalization': Spec(
        paths=(
            'tokamax/_src/ops/normalization/api_test.py',
            'tokamax/_src/ops/normalization/base_test.py',
            'tokamax/_src/ops/normalization/pallas_triton_test.py',
        ),
        minutes=3,
    ),
    'ops-ragged-scatter': Spec(
        paths=(
            'tokamax/_src/ops/ragged_scatter/base_test.py',
            'tokamax/_src/ops/ragged_scatter/pallas_mosaic_tpu_test.py',
        ),
        minutes=3,
    ),
    'ops-gated-linear-unit': Spec(
        paths=(
            'tokamax/_src/ops/gated_linear_unit/api_test.py',
            'tokamax/_src/ops/gated_linear_unit/base_test.py',
            'tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_test.py',
            'tokamax/_src/ops/gated_linear_unit/pallas_triton_test.py',
        ),
        minutes=1,
    ),
    'ops-ragged-gather-reduce': Spec(
        paths=(
            'tokamax/_src/ops/ragged_gather_reduce/base_test.py',
            'tokamax/_src/ops/ragged_gather_reduce/pallas_mosaic_tpu_test.py',
        ),
        minutes=1,
    ),
    'ops-triangle-multiplication': Spec(
        paths=(
            'tokamax/_src/ops/triangle_multiplication/api_test.py',
            'tokamax/_src/ops/triangle_multiplication/base_test.py',
        ),
        minutes=1,
    ),
    # -- experimental ------------------------------------------------------
    # kda was in the catch-all, which is where an unscheduled file is supposed
    # to end up and be noticed: at 15.4m on tpu6e it was the entire catch-all's
    # runtime, and the fifth-longest job in CI, under a name that said nothing
    # about what was slow. Split on the same line as topk above, and for the
    # same reason -- the kernel test runs 0 of its 217 cases on h100, while
    # `base_test.py` runs all 20 everywhere, so pinning them together would
    # either waste a GPU job or drop the CUDA coverage.
    'experimental-kda-kernel': Spec(
        paths=(
            'tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_kernel_test.py',
        ),
        minutes=16,
    ),
    'experimental-mla': Spec(
        paths=(
            'tokamax/_src/ops/experimental/mla/base_test.py',
            'tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel_test.py',
            'tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_test.py',
        ),
        minutes=2,
    ),
    # The kda tests that are not the kernel; see `experimental-kda-kernel`.
    'experimental-kda': Spec(
        paths=(
            'tokamax/_src/ops/experimental/kda/base_test.py',
            'tokamax/_src/ops/experimental/kda/pallas_mosaic_tpu_test.py',
        ),
        minutes=1,
    ),
    # The topk tests that are not the kernel. Separate from
    # `experimental-topk-kernel` because that one is TPU-pinned and these two
    # run everywhere; merging them would drop the CUDA coverage.
    'experimental-topk': Spec(
        paths=(
            'tokamax/_src/ops/experimental/tpu/topk/api_test.py',
            'tokamax/_src/ops/experimental/tpu/topk/base_test.py',
        ),
        minutes=1,
    ),
    'experimental-topk-kernel': Spec(
        paths=(
            'tokamax/_src/ops/experimental/tpu/topk/pallas_mosaic_tpu_test.py',
        ),
        minutes=1,
    ),
    # -- core --------------------------------------------------------------
    # Public-API smoke test: the canary for an import or packaging break.
    'core-api': Spec(
        paths=('tokamax/tokamax_test.py',),
        minutes=3,
    ),
    'core-autotuning': Spec(
        paths=(
            'tokamax/_src/autotuning/api_test.py',
            'tokamax/_src/autotuning/cache_test.py',
        ),
        minutes=3,
    ),
    'core-op': Spec(
        paths=('tokamax/_src/ops/op_test.py',),
        minutes=2,
    ),
    'core-pallas-block': Spec(
        paths=('tokamax/_src/pallas/block_test.py',),
        minutes=1,
    ),
    # The library internals under `tokamax/_src/`, none of which is slow enough
    # to deserve a shard. Enumerated file by file rather than given as the
    # directory `tokamax/_src/`, which would swallow every op subdirectory, and
    # rather than left to the catch-all shard, which is reserved for files
    # nobody has scheduled yet. `benchmarking_test.py` is absent on purpose;
    # see EXCLUDED_TESTS.
    'core-utils': Spec(
        paths=(
            'tokamax/_src/ad_test.py',
            'tokamax/_src/batching_test.py',
            'tokamax/_src/config_test.py',
            'tokamax/_src/gpu_utils_test.py',
            'tokamax/_src/hlo_utils_common_test.py',
            'tokamax/_src/hlo_utils_test.py',
            'tokamax/_src/jaxtyping_test.py',
            'tokamax/_src/mosaic_gpu_test.py',
            'tokamax/_src/numerics_test.py',
            'tokamax/_src/precision_test.py',
            'tokamax/_src/pydantic_test.py',
            'tokamax/_src/shape_test.py',
            'tokamax/_src/test_utils_test.py',
        ),
        minutes=1,
    ),
    # -- catch-all ---------------------------------------------------------
    # Temporary shard that ideally should be empty. It's filled in by
    # `resolve_shards`: every test file that no shard above collects, and
    # should be redistributed to other shards after some time.
    'catch-all': Spec(
        paths=None,
        minutes=1,
    ),
}
# pylint: enable=line-too-long

_CATCH_ALL_SHARD = 'catch-all'


def _is_test_filename(name: str) -> bool:
  """Returns whether pytest would collect a file with this base name.

  Args:
    name: A file's base name, without any directory part.

  Returns:
    True if the name matches one of `PYTEST_FILE_PATTERNS`.
  """
  return any(fnmatch.fnmatch(name, p) for p in PYTEST_FILE_PATTERNS)


def all_test_files(root: str = 'tokamax') -> list[str]:
  """Returns every file pytest would collect, repo-relative.

  Matched against `PYTEST_FILE_PATTERNS` rather than the `_test.py` suffix
  this repo mostly uses, so the set is the one pytest starts from. Files that
  match but hold no tests are subtracted later, via `IGNORED_GLOBS`.

  Args:
    root: Directory to walk, relative to the repository root.

  Returns:
    Sorted repo-relative, slash-separated paths.
  """
  return sorted(
      os.path.join(dirpath, name).replace(os.sep, '/')
      for dirpath, _, filenames in os.walk(root)
      for name in filenames
      if _is_test_filename(name) and '__pycache__' not in dirpath
  )


def split_node_id(path: str) -> tuple[str, str | None]:
  """Splits a `paths` entry into its file and class parts.

  A path is at most one `::` deep: pytest would accept a method or a parameter
  case too, but `declared_test_classes` sees only classes, so a finer split
  would leave the checks unable to tell whether the file is still fully
  covered.

  Args:
    path: A `paths` entry, either `some_test.py` or `some_test.py::SomeTest`.

  Returns:
    `(test file, class name)`, where the class name is None for a whole-file
    entry, which is the usual case.
  """
  file, sep, selector = path.partition('::')
  return file, (selector if sep else None)


def declared_test_classes(path: str) -> list[str]:
  """Returns class names a shard must name if the file is split by node ID.

  Only this file's syntax is available -- `ast` does not resolve imports, so
  whether a base class is ultimately a `TestCase` is not decidable here. It
  does not need to be. This is used only on files split by node ID, where the
  question is not "is this a test?" but "did a class appear that no shard
  names?", and the safe answer is every class the module defines.

  The one exemption is a private class, since a leading underscore is how this
  repo already spells "not part of the interface" and pytest's default
  `python_classes` prefix will not collect it. A private class that inherits
  from something test-shaped is *not* exempt: pytest collects `TestCase`
  subclasses whatever they are called, so that one would really run.

  Args:
    path: The test file to parse.

  Returns:
    Every non-private class the module defines, in source order.

  Raises:
    OSError: If `path` cannot be read.
    SyntaxError: If `path` does not parse.
  """
  with open(path, encoding='utf-8') as f:
    tree = ast.parse(f.read(), filename=path)

  # Module level means "becomes a module attribute", which is what pytest
  # collects -- not "appears at indent zero". A class guarded by a backend
  # check is still a module attribute, so descend through the statements that
  # can hold one, and no further: a class nested in a function or in another
  # class is not collected, and flagging it would be a false positive nothing
  # short of renaming it could silence.
  body = list(tree.body)
  names = []
  while body:
    node = body.pop(0)
    if isinstance(node, (ast.If, ast.Try)):
      body = [
          *node.body,
          *node.orelse,
          *getattr(node, 'finalbody', []),
          *[s for h in getattr(node, 'handlers', []) for s in h.body],
          *body,
      ]
      continue
    if not isinstance(node, ast.ClassDef):
      continue
    bases = [
        b.attr if isinstance(b, ast.Attribute) else getattr(b, 'id', '')
        for b in node.bases
    ]
    private = node.name.startswith('_') and not any(
        b.endswith(('Test', 'TestCase', 'TestBase')) for b in bases
    )
    if not private:
      names.append(node.name)
  return names


def _matches_glob(path: str, globs: Iterable[str]) -> bool:
  """Returns whether `path` matches any of `globs`.

  Args:
    path: A repo-relative path.
    globs: `fnmatch` patterns.

  Returns:
    True if any pattern matches.
  """
  return any(fnmatch.fnmatch(path, g) for g in globs)


def collected_by(paths: Iterable[str], test_files: Collection[str]) -> set[str]:
  """Returns which of `test_files` pytest collects, given a shard's `paths`.

  The one definition of what a shard runs, used on both sides so they cannot
  disagree: `resolve_shards` sums it over the shards to find the files nothing
  claims, and `check_consistency` reads it per shard to catch one that runs
  nothing and files that run twice.

  It is set membership rather than a guess only because `paths` is a sequence
  of test files, optionally narrowed by a node ID, and nothing else.
  `check_consistency` rejects the two shapes that would make it a guess -- see
  the errors it raises for a directory target and for a selection flag.

  A node ID contributes its file, so a shard that runs one class of a file
  counts as collecting that file. Which classes of it run is a finer question
  than coverage, and `check_consistency` answers it separately.

  Args:
    paths: A shard's `paths`, each a test file or a `file::Class` node ID.
    test_files: The universe to select from, as from `all_test_files`.

  Returns:
    The subset of `test_files` these paths name, minus anything matching
    `IGNORED_GLOBS`.
  """
  named = {split_node_id(p)[0] for p in paths}
  return {
      f
      for f in test_files
      if f in named and not _matches_glob(f, IGNORED_GLOBS)
  }


def _matches(test_files: Iterable[str], globs: Iterable[str]) -> set[str]:
  """Returns the files matching any glob.

  Args:
    test_files: Repo-relative paths.
    globs: `fnmatch` patterns.

  Returns:
    Every path in `test_files` that matches at least one pattern.
  """
  return {f for f in test_files if _matches_glob(f, globs)}


def resolve_shards(
    test_files: Collection[str] | None = None,
) -> tuple[ShardMap, list[str], set[str], set[str]]:
  """Fills in the catch-all shard and partitions the tree it sharded against.

  The catch-all shard collects every test file no other shard names, so a test
  added without a `SHARDS` entry still runs. It is dropped from the returned
  mapping when empty.

  Args:
    test_files: The tree to shard, as from `all_test_files`. Defaults to walking
      the real one, and is a parameter so the checks can run against a tree that
      does not exist on disk.

  Returns:
    `(shards, catch_all, excluded, ignored)`. `shards` is a deep-enough copy of
    `SHARDS` to make the catch-all edit local. `catch_all` is the files that
    shard picked up, empty in the ordinary case. `excluded` and `ignored` are
    the files left out by design and the ones pytest never collects, returned
    so `check_consistency` can be handed the same partitions.
  """
  test_files = all_test_files() if test_files is None else test_files
  shards = {n: dict(s) for n, s in SHARDS.items()}

  claimed = set()
  for spec in shards.values():
    if spec['paths'] is not None:
      claimed |= collected_by(spec['paths'], test_files)

  # There are precisely two reasons a file belongs to no shard:
  # pytest is told to ignore it or we deliberately left it out.
  ignored = _matches(test_files, IGNORED_GLOBS)
  excluded = _matches(test_files, EXCLUDED_TESTS)
  catch_all = sorted(set(test_files) - claimed - excluded - ignored)

  if catch_all:
    shards[_CATCH_ALL_SHARD]['paths'] = tuple(catch_all)
  else:
    # An empty `paths` would reach the workflow as no arguments at all, and
    # pytest with no arguments collects the whole repository. Delete the shard
    # instead of emitting a job that quietly runs everything.
    del shards[_CATCH_ALL_SHARD]
  return shards, catch_all, excluded, ignored


def check_consistency(
    shards: ShardMap,
    test_files: Collection[str],
    excluded: Collection[str],
    ignored: Collection[str],
    class_reader: Callable[[str], list[str]] = declared_test_classes,
) -> list[str]:
  """Returns the problems that would cause tests to run wrongly.

  Args:
    shards: Shard name to spec, as `resolve_shards` returns it.
    test_files: The tree the shards were resolved against.
    excluded: Files left out by design, from `resolve_shards`. Subtracted before
      reporting a file as unclaimed, or every one would be an error.
    ignored: Files pytest never collects, from `resolve_shards`. Subtracted for
      the same reason.
    class_reader: Reads a file's class names, `declared_test_classes` in
      production. A parameter only so a test can supply them directly;
      everything else here works off `test_files`, so the checks stay runnable
      against a tree that does not exist on disk.

  Returns:
    One string per problem, empty if the table is sound.
  """
  errors = []
  # test file -> [(shard name, class name or None)]. Files rather than paths,
  # because whether a file runs twice is a question about the file: two shards
  # naming disjoint classes of it is the point of a split, and two shards
  # naming the whole of it is a bug, and both are one entry per path here.
  claims = collections.defaultdict(list)
  for name, spec in sorted(shards.items()):
    if not spec['paths']:
      errors.append(f'shard {name!r} has no paths')
      continue
    if flags := [t for t in spec['paths'] if t.startswith('-')]:
      errors.append(
          f'shard {name!r} passes flags in paths: {", ".join(flags)} -- paths'
          ' lists test files only, because a flag can narrow what pytest'
          ' collects in a way `collected_by` cannot see. Put selection flags'
          ' in PYTEST_SELECT_FLAGS.'
      )
    # Checked against `test_files`, not the filesystem: a name that exists on
    # disk but is not a file pytest would collect is just as dead, and reusing
    # the one source of truth keeps this testable without touching the tree.
    # This is also what rejects a directory target, which `collected_by` does
    # not model.
    targets = [split_node_id(p) for p in spec['paths'] if p not in flags]
    missing = sorted({f for f, _ in targets} - set(test_files))
    if missing:
      errors.append(
          f'shard {name!r} names paths that are not test files: '
          + ', '.join(missing)
      )

    files = collected_by(spec['paths'], test_files)
    if not files:
      errors.append(
          f'shard {name!r} collects no test files -- a target was probably'
          ' moved or renamed'
      )
    for file, selector in targets:
      if file in files:
        claims[file].append((name, selector))
    # `minutes` is sorted on and printed, so a string or a zero would quietly
    # misplace the shard rather than fail. Absent says there is no number yet.
    minutes = spec.get('minutes')
    if minutes is not None and (
        not isinstance(minutes, (int, float)) or minutes <= 0
    ):
      errors.append(
          f'shard {name!r} has minutes={minutes!r}: expected a positive'
          ' number, or the key omitted if unmeasured'
      )
    # Every shard runs on every runner, so `Spec` has no key for restricting
    # one and nothing here reads such a key. An entry carrying a `devices` key
    # would be silently ignored, and would read to anyone editing the table as
    # though it pinned the shard.
    if 'devices' in spec:
      errors.append(
          f'shard {name!r} has a `devices` key: every shard runs on every'
          ' runner, so this pins nothing. Drop it.'
      )

  uncovered = sorted(set(test_files) - set(claims) - excluded - ignored)
  if uncovered:
    errors.append('test files no shard runs: ' + ', '.join(uncovered))

  # A shard whose name matches no theme would sort into a nameless group at
  # the end of the job list, which is exactly the drift the themes exist to
  # prevent. Cheaper to reject the name than to notice the stray job later.
  if strays := sorted(n for n in shards if shard_theme(n) is None):
    errors.append(
        'shards whose name starts with no theme in THEMES: '
        + ', '.join(strays)
        + ' -- rename to `<theme>-<what it runs>`, or add a theme'
    )

  # Job names are built from (runner_short, shard_name), and a required status
  # check is matched on that string, so two runners sharing a short name would
  # produce two jobs that cannot be told apart.
  short_counts = collections.Counter(short for _, short in RUNNERS.values())
  if dupes := sorted(s for s, n in short_counts.items() if n > 1):
    errors.append('runner short names used twice: ' + ', '.join(dupes))

  # An exclusion for a file that no longer exists is dead config that reads as
  # a live decision.
  for glob in EXCLUDED_TESTS:
    if not _matches(test_files, (glob,)):
      errors.append(f'EXCLUDED_TESTS entry {glob!r} matches no test file')

  # What each file's claims are allowed to look like: either one shard runs
  # the whole file, or several run disjoint classes of it. Anything else runs
  # a test twice on scarce hardware, or -- only possible once node IDs are in
  # play -- stops running one entirely, with every path here still valid.
  for file, entries in sorted(claims.items()):
    selectors = [s for _, s in entries]
    where = ', '.join(sorted({n for n, _ in entries}))
    if selectors.count(None) > 1:
      errors.append(
          f'{file} is run by more than one shard, in {where}: the tests still'
          ' run, just twice, on scarce hardware'
      )
      continue
    if len(entries) > 1 and None in selectors:
      errors.append(
          f'{file} is claimed both whole and by node ID, in {where}: the'
          ' whole-file claim already runs every class in it, so each node ID'
          ' runs its class a second time'
      )
      continue
    if dupes := sorted({s for s in selectors if selectors.count(s) > 1}):
      errors.append(
          f'{file} is claimed more than once, in {where}: ' + ', '.join(dupes)
      )
      continue
    if selectors == [None]:
      continue

    # Split by node ID. pytest is handed classes, not the file, so a class
    # nobody names does not read as unclaimed anywhere above: the file is
    # covered, every path is still valid, and the class just stops running.
    # This is the only check that reads a test file's source, and the reason
    # `split_node_id` refuses to go deeper than a class.
    declared = set(class_reader(file))
    if orphans := sorted(declared - set(selectors)):
      errors.append(
          f'{file} is split by node ID across {where}, but no shard names '
          + ', '.join(orphans)
          + ' -- add each to a shard, or, if it holds no tests, give it a'
          ' leading underscore'
      )
    if unknown := sorted(set(selectors) - declared):
      errors.append(
          f'{file} is split by node ID across {where}, which name classes it'
          ' does not declare: '
          + ', '.join(unknown)
      )
  return errors


def shard_theme(name: str) -> tuple[int, str] | None:
  """Finds the theme a shard name declares.

  Args:
    name: A shard name.

  Returns:
    `(index into THEMES, theme)`, or None if the name declares no theme.
  """
  for index, theme in enumerate(THEMES):
    if name == theme or name.startswith(f'{theme}-'):
      return index, theme
  return None


def shard_order(item: tuple[str, Spec]) -> tuple[bool, bool, float, int, str]:
  """Sorts shards: catch-all first, then longest first, then by theme, then by name.

  The catch-all shard always runs first so newly added or unscheduled tests run
  immediately. Other shards run longest first because GitHub starts matrix jobs
  in matrix order as runners free up, so a long shard left until last sets the
  wall clock on its own.

  Args:
    item: A `(shard name, spec)` pair, as from `dict.items`.

  Returns:
    A sort key: catch-all shard first, then measured shards before unmeasured
    ones, then descending `minutes`, then theme in `THEMES` order with themeless
    shards last, then the name.
  """
  name, spec = item
  theme = shard_theme(name)
  minutes = spec.get('minutes')
  return (
      name != _CATCH_ALL_SHARD,
      minutes is None,  # unmeasured last, since absent is not zero
      -(minutes or 0),
      len(THEMES) if theme is None else theme[0],
      name,
  )


def build_matrix(shards: ShardMap) -> list[MatrixEntry]:
  """Builds `strategy.matrix.include`, one entry per (shard, runner) job.

  Every shard runs on every runner, so this is the full cross product.

  The only place a shard's `paths` becomes a command line: the workflow
  interpolates `test_paths` straight into a `run:` block, so `shlex.join`
  quotes it here rather than trusting every path to be shell-safe.

  Args:
    shards: Shard name to spec, as `resolve_shards` returns it.

  Returns:
    A job per shard per runner, in `shard_order` within each runner.
  """
  return [
      {
          'runner': runner,
          'runner_short': short,
          'device': device,
          'shard_name': name,
          'test_paths': shlex.join(spec['paths']),
      }
      for runner, (device, short) in RUNNERS.items()
      for name, spec in sorted(shards.items(), key=shard_order)
      # `check_consistency` rejects a shard with no paths, so this never emits
      # a job with no arguments -- which pytest would read as "collect the
      # whole repository".
      if spec['paths']
  ]


def main(argv: Sequence[str] | None = None) -> int:
  """Runs the `matrix`, `check` or `list` command.

  Args:
    argv: Command-line arguments, `sys.argv[1:]` when None.

  Returns:
    A process exit status, always 0; a bad table raises instead.

  Raises:
    SystemExit: If `check_consistency` finds a problem.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('command', choices=('matrix', 'check', 'list'))
  args = parser.parse_args(argv)

  test_files = all_test_files()
  shards, catch_all, excluded, ignored = resolve_shards(test_files)

  if errors := check_consistency(shards, test_files, excluded, ignored):
    raise SystemExit(
        'shard configuration is inconsistent:\n  ' + '\n  '.join(errors)
    )

  if args.command == 'check':
    # Subtracts both partitions `check_consistency` subtracts, or the count
    # would report files pytest never collects as covered.
    covered = len(test_files) - len(excluded) - len(ignored)
    print(
        f'{len(shards)} shards cover {covered} test files;'
        f' {len(excluded)} excluded by design,'
        f' {len(ignored)} never collected'
    )
    return 0

  if args.command == 'matrix':
    combos = build_matrix(shards)
    flags = shlex.join(PYTEST_SELECT_FLAGS)
    # Diagnostics on stderr, so they land in the job log rather than in
    # $GITHUB_OUTPUT, which stdout is redirected to.
    print(f'{len(shards)} shards, {len(combos)} jobs', file=sys.stderr)
    print(f'not run by design: {sorted(excluded)}', file=sys.stderr)
    print(f'pytest select flags: {flags}', file=sys.stderr)
    print(f'include={json.dumps(combos)}')
    print(f'pytest_flags={flags}')
    # The catch-all is empty in the ordinary case, and the workflow turns a
    # non-empty one into a `::warning::` annotation. On stdout as an output
    # rather than only logged to stderr, because a test file that no shard
    # claims should be visible from the PR without opening the job.
    print(f'catch_all={json.dumps(catch_all)}')
    return 0

  combos = build_matrix(shards)
  per_shard = collections.Counter(c['shard_name'] for c in combos)
  print(f'{len(shards)} shards, {len(combos)} jobs')
  # Longest first, shards without a number last. An absent `minutes` is an
  # unknown, so it sorts to the bottom rather than being read as a zero.
  order = sorted(shards.items(), key=shard_order)
  for name, spec in order:
    print(f'  {name:32s} x{per_shard[name]}')
    for path in spec['paths']:
      print(f'      {path}')

  # The files above are the ones that run. Naming the rest is what makes this
  # a complete account of the suite rather than a list of jobs: a reader can
  # add these up against `all_test_files()` and find nothing unaccounted for.
  targets = [p for s in shards.values() for p in s['paths']]
  files = {split_node_id(p)[0] for p in targets}
  print()
  print(
      f'  {len(test_files)} test files on disk = {len(files)} run'
      f' + {len(excluded)} excluded + {len(ignored)} never collected'
  )
  # More targets than files whenever a file is split by node ID: it is named
  # once per class, so the two numbers are not meant to agree.
  print(f'  {len(targets)} targets across {len(shards)} shards')
  for path in sorted(excluded):
    print(f'      excluded         {path}')
  for path in sorted(ignored):
    print(f'      never collected  {path}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
