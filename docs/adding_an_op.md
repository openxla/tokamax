## Adding an Op

This page walks through adding an operator to Tokamax, from the first files to
autotuning, benchmarking and CI.
[Contributing](https://openxla.org/tokamax/contributing) covers what reviewers
expect from a new op. The
[Creating a New Tokamax Op](https://openxla.org/tokamax/tokamax_infra) notebook
is a hands-on tutorial for wrapping a Pallas kernel as a `tokamax.Op`.

### Checklist

| Step | Files | Gives you |
| --- | --- | --- |
| [Choose a directory](#choose-a-directory) | `_src/ops/experimental/<op>/` | |
| [Write the op](#write-the-op) | `reference.py`, `base.py`, kernel, `pallas_mosaic_tpu.py`, `api.py` | A `tokamax.Op` with an XLA fallback |
| [Test it](#test-it) | `test_base.py`, `*_test.py` | Correctness against the reference |
| [Register it](#register-it) | `_src/ops/ops_list.py`, `tokamax/ops/experimental/__init__.py` | Op inventory, public API |
| [Define ArgSpecs](#define-argspecs) | `arg_specs.py` | Autotuning and benchmark shapes |
| [Hook up benchmarking](#hook-up-benchmarking) | `benchmarks/` (optional) | Benchmarks and A/B comparisons |
| [Add a CI shard](#add-a-ci-shard) | `.github/ci/shards.py` | Tests run on every change to the op |
| [Document it](#document-it) | `docs/supported_ops.md` | |

### Choose a directory

New ops start in `tokamax/_src/ops/experimental/<op>/`. Once the op is
complete, well tested and validated end to end, it can move to
`tokamax/_src/ops/<op>/`.

Each op gets one directory. A family of kernels that share code belongs in one
directory that exposes several entry points. For example, `mhc` exposes
`mhc_pre`, `mhc_post` and `mhc_fused_post_pre`. Group kernels by what they
compute rather than where they run in a model.

### Write the op

A typical op directory looks like this:

```
tokamax/_src/ops/experimental/my_op/
  __init__.py
  reference.py                 # Plain JAX implementation.
  base.py                      # MyOp(op.Op): the XLA implementation.
  pallas_mosaic_tpu_kernel.py  # The Pallas kernel.
  pallas_mosaic_tpu.py         # PallasMosaicTpuMyOp(base.MyOp).
  api.py                       # my_op(..., implementation=None).
  arg_specs.py                 # ARG_SPECS.
  test_base.py                 # Shared correctness tests.
  base_test.py
  pallas_mosaic_tpu_test.py
  api_test.py
```

*   `reference.py` is the plain JAX version of the computation. Tests use it as
    the ground truth.
*   `base.py` defines the op class, a subclass of `tokamax._src.ops.op.Op`, and
    its argument types. Its `_fwd` calls the reference, so the op runs on any
    device.
*   `pallas_mosaic_tpu.py` subclasses the base op to run the kernel. It sets
    `config_cls` to a `Config` dataclass of kernel parameters (tile sizes and
    so on), and implements:
    *   `_fwd`, which calls the kernel with the resolved config.
    *   `_get_heuristics_config`, the default config for given arguments.
    *   `_get_autotuning_configs`, the candidate configs the autotuner tries.
    *   `supported_on`, if the kernel only runs on some devices.
*   `api.py` holds the public function. It maps implementation names to op
    instances in `IMPLEMENTATIONS`, imports backend implementations inside
    `try`/`except ImportError`, and tries them in order when `implementation`
    is `None`.

Good examples are
[`experimental/mla_swa`](https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/experimental/mla_swa),
which wraps an existing kernel, and
[`proj_and_save_state`](https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/proj_and_save_state).

### Test it

Write the correctness tests once, in `test_base.py`, as a test class that takes
the implementation under test and compares it against `reference.py`. Then add
one small `*_test.py` per implementation that subclasses it, for example
`pallas_mosaic_tpu_test.py`. Kernel-specific tests, such as checks of tiling
edge cases, go in those files too.

Name test files `*_test.py`. CI skips `test_base.py` and `test_utils.py`, so
keep shared test classes and helpers in files with those names.

### Register it

Add an entry to `OPS` in
[`_src/ops/ops_list.py`](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/ops_list.py):

```python
Op(
    name="my_op",
    apis=("tokamax.ops.experimental.my_op",),
    impls=(
        Impl(
            name="xla",
            backend=XLA,
            module="_src/ops/experimental/my_op/base.py",
        ),
        Impl(
            name="mosaic_tpu",
            backend=MOSAIC_TPU,
            module="_src/ops/experimental/my_op/pallas_mosaic_tpu.py",
        ),
    ),
),
```

`_src/ops/ops_list_test.py` checks that:

*   Every directory under `_src/ops/` and `_src/ops/experimental/` that has
    Python files is covered by an entry.
*   Each entry has at least one implementation, every implementation module
    exists and lives in the op's directory, and the XLA implementation comes
    first when the op has a `base.py`.
*   If `apis` is not empty, `api.py` exists, the names in its
    `IMPLEMENTATIONS` are registered implementations (and every open-source
    implementation is in `IMPLEMENTATIONS`), and each API is exported. Export
    `tokamax.ops.experimental.<name>` APIs in
    `tokamax/ops/experimental/__init__.py` and `tokamax.ops.<name>` APIs in
    `tokamax/ops/__init__.py`.

`apis` can be empty for an op that is not public yet. If an implementation's
kernels live in a separate directory under `_src/ops/` that is not an op
itself, list it in the implementation's `kernel_dirs`. Each directory belongs
to exactly one op.

### Define ArgSpecs

`arg_specs.py` lists the inputs the op should be tuned and benchmarked on, as
an `ARG_SPECS` tuple of `tokamax._src.autotuning.arg_spec.ArgSpec`:

```python
ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    arg_spec.ArgSpec(
        args={"x": jax.ShapeDtypeStruct((4096, 7168), jnp.bfloat16), ...},
        project="my_model",
        name="t4096_bf16",
        tags=("primary", "ci_tests", "forward_only"),
    ),
    ...
)
```

Use the shapes and dtypes of the models the op serves. `project` and `name`
identify a spec in benchmark results. Tags select specs for different
workflows:

*   `primary`: important workloads, used for nightly benchmarking.
*   `ci_tests`: a subset of `primary` run in CI.
*   `forward_only`: no backward pass to tune.

For arguments whose values matter, such as indices that must be in bounds, use
an initializer from `tokamax._src.numerics` (for example
`RangedArrayInitializer`) instead of a `ShapeDtypeStruct`. See
[`deepseek_rope/arg_specs.py`](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/deepseek_rope/arg_specs.py).

[Autotuning](https://openxla.org/tokamax/autotuning) explains how tuned configs
are found and cached.

### Hook up benchmarking

Run a quick comparison against the reference with the tools in
[Benchmarking](https://openxla.org/tokamax/benchmarking). Contributing asks for
a sweep over all of the op's ArgSpecs against the plain JAX implementation, with
the results in the PR.

To add the op to the public nightly benchmarks, add a workload to
`benchmarks/benchmark_registry.pbtxt` and a script under `benchmarks/`. This is
optional.

### Add a CI shard

GitHub CI runs tests in shards defined in
[`.github/ci/shards.py`](https://github.com/openxla/tokamax/blob/main/.github/ci/shards.py).
Tests that no shard claims run in `catch-all`. Give the op its own shard:

```python
'experimental-my-op': Spec(
    paths=(
        'tokamax/_src/ops/experimental/my_op/api_test.py',
        'tokamax/_src/ops/experimental/my_op/base_test.py',
        'tokamax/_src/ops/experimental/my_op/pallas_mosaic_tpu_test.py',
    ),
),
```

*   List every test file individually. Directories are not allowed.
*   The shard name must start with a theme from `THEMES`, such as
    `experimental-` or `ops-`.
*   Leave out `minutes` until the shard has run in CI, then set it from the
    slowest runner's "Run shard" step time.

CI selects shards by following imports from the changed files, so a change to
the op runs its shard, plus the shards of any tests that import it. Run
`python3 .github/ci/shards.py check` to validate the shard definitions.

### Document it

Add a row for the op to
[Supported Ops](https://openxla.org/tokamax/supported_ops),
marked "(experimental)" while it lives under `experimental/`. Describe what it
computes, where it is used and how general it is. For a larger op, add a
`README.md` or a design doc under `docs/ops/` as well.

### Graduating from experimental

To move an op out of `experimental/`:

1.  Move the directory to `_src/ops/<op>/` and update `ops_list.py`, imports
    and `BUILD` files.
2.  Move its public API from `tokamax.ops.experimental` to `tokamax.ops`.
3.  Rename its shard to a non-experimental theme.
4.  Drop "(experimental)" from Supported Ops.
