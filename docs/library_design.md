## Library Design

Tokamax is designed to standardize and simplify typical developer journeys
around kernel authoring, optimization, maintenance, and production usage. It
does this through foundational classes and shared infrastructure that all
operators in the library inherit from and implement within a well-structured
framework.

### Operator Layers

#### Tokamax Base Class

[`tokamax.Op`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/op.py#L121)
is Tokamax’s foundational class, which serves as the base for every operator in
the library. When any derived operator is invoked, this layer automatically
handles the following functionality:

*   **Argument Binding**: Before any computation runs, the base class normalizes
    and validates the caller's arguments into a single canonical representation
    so that caching, heuristics, and kernel execution can all reason about the
    workload uniformly (through a
    [`BoundArguments`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/op.py#L438)
    object), regardless of how inputs were passed (see
    [`op.bind()`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/op.py#L317)).
*   **Device Validation**: To catch hardware mismatches early, the operator
    verifies device support against the target accelerator before running, while
    still allowing this check to be bypassed when cross-compiling from a CPU
    host (see
    [`op.supported_on()`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/op.py#L409),
    [`bypass_device_check=True`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/op.py#L169),
    and
    [`config.cross_compile(True)`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/config.py#L95)).
*   **Batching Capture**: Because the best kernel configuration
    ([`Config`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/op.py#L167))
    often depends on the full batched problem size rather than a single slice,
    the base class captures surrounding batch dimensions when an operator is
    vectorized
    ([`jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html)) so
    that heuristics and autotuning caches can optimize for the true workload
    shape (see
    [`op._capture_batched_args()`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/op.py#L384)).
*   **Config Resolution**: To provide strong performance out of the box without
    requiring manual tuning, the base class resolves kernel hyperparameters
    automatically whenever the caller does not supply them -- reusing previously
    autotuned settings in a disk-based autotuning cache if available, or falling
    back to built-in heuristics (see
    [`BoundArguments.default_config`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/op.py#L469)).
*   **StableHLO Metadata Serialization**: To make compiled programs
    introspectable for offline profiling and automated tuning, the base class
    embeds operator metadata into the lowered graph so downstream tools can
    reconstruct which operators ran and with what inputs (shapes, types, etc.)
    directly from a compiled module or performance profile (see
    [`_tokamax_metadata()`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/op.py#L108)).
*   **Forward Execution and Custom VJP**: Because specialized hardware kernels
    often cannot be differentiated efficiently by standard automatic
    differentiation, the base class coordinates forward and backward execution
    as a pair, allowing the forward pass to pass intermediate values directly to
    a custom backward kernel during gradient computation (see
    [`op._fwd()`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/op.py#L356))
    and calls to
    [`jax.custom_vjp`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/op.py#L314).

#### Op-Specific Base Class

Every operator family defines its own base class that inherits from the
foundational operator class and establishes the contract shared across all
backends (for example, see
[`RaggedDot`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/base.py#L201)):

*   **Reference Implementation**: Each operator base class provides a portable,
    pure-JAX reference implementation of the full mathematical specification,
    including gradients when applicable. This serves as the ground truth for
    numerical correctness tests and guarantees a working fallback on any
    hardware (see
    [`RaggedDot._fwd()`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/base.py#L277)).
*   **Canonical Binding**: Because users can express the same mathematical
    operation in multiple ways—such as passing different forms of masks,
    precision settings, or dimension specifications—the operator base class
    normalizes inputs and checks shape and type compatibility upfront so
    individual hardware backends only have to handle a single, predictable input
    format (see
    [`RaggedDot.bind()`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/base.py#L222)).

#### Op Implementation Classes

These classes provide the hardware-specific kernel implementations and can be
instantiated and invoked directly by users. Each implementation inherits from
its operator-specific base class and supplies the backend-specific behavior (for
example, see
[`PallasMosaicTpuRaggedDot`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L90)):

*   **Core Functionality**: This is where the actual hardware kernel execution
    is defined, typically using a kernel language such as Pallas. When an
    operation requires a specialized backward pass, the gradient kernel is
    usually modeled as its own first-class operator implementation so that it
    can be configured, tuned, and tested independently. Although the
    lowest-level kernel code is often factored into a separate helper file for
    modularity, this class serves as the primary interface for invoking that
    kernel (see
    [`PallasMosaicTpuRaggedDot._fwd()`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L119)).
*   **Configuration**: Because hardware kernels often expose performance
    knobs—such as tile sizes, pipeline stages, or memory buffering strategies,
    each backend defines a dedicated configuration structure that lets users or
    automated tools control how the kernel is partitioned, compiled and executed
    (see
    [`Config`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L58)).
*   **Heuristics**: Not every workload has been autotuned in advance, so each
    backend provides built-in heuristics that choose reasonable default tuning
    parameters based on the problem dimensions, data types, and accelerator
    specifications (see
    [`_get_heuristics_config()`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L308)).
*   **Autotuning Search Space**: For workloads where peak performance matters,
    each backend defines an autotuning search space of valid parameter
    combinations for a given problem shape, allowing the autotuner to benchmark
    candidates and discover the fastest configuration (see
    [`_get_autotuning_configs()`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L435)).
*   **Implementation-Specific Validation**: Hardware kernels frequently have
    stricter requirements than the general mathematical operation, such as
    dimension alignment rules or limited data type support. Each implementation
    validates these constraints and signals when a workload cannot be handled,
    allowing higher-level dispatchers to fall back gracefully to another backend
    (see
    [`PallasMosaicTpuRaggedDot._fwd()`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L136-L156)).

#### High-Level API

Most operators have multiple backends—such as a portable reference
implementation alongside one or more hardware-specific kernels—and expose a
single familiar, high-level entry point (e.g., a standard
[`jax.lax`](https://docs.jax.dev/en/latest/jax.lax.html) function). This layer
is designed to give users a simple, drop-in interface that automatically selects
the best supported backend for their hardware and input shapes, while still
letting them request a specific implementation when desired (see
[`tokamax.ragged_dot()`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/api.py#L77)).

#### ArgSpecs

Maintaining a shared catalog of important workload shapes is essential for
testing and optimizing kernels consistently across backends. Tokamax captures
these representative production and benchmark shapes as workload specifications
that are shared across all implementations of an operator. This gives continuous
integration tests, regression benchmarks, and autotuning pipelines a single
source of truth that can be filtered by tags for different workflows (see the
ArgSpecs for
[`ragged_dot`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/arg_specs.py)).

### Repository Structure

Internal implementation code lives under the private source tree (see
[`tokamax/_src`](https://github.com/openxla/tokamax/tree/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src)),
while public symbols for operators, autotuning, benchmarking, and configuration
are re-exported through top-level modules (see
[`tokamax/__init__.py`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/__init__.py)).

Within each operator directory (see
[`tokamax/_src/ops`](https://github.com/openxla/tokamax/tree/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops)),
files follow a consistent naming convention, illustrated here using ragged dot:

*   **High-Level API**: Defines the public functional entry point and backend
    dispatch logic (see
    [`api.py`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/api.py)).
*   **Op-Specific Base Class**: Defines the portable reference implementation
    (see
    [`base.py`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/base.py)).
*   **Backend Implementations**: Defines hardware-specific operator classes,
    configurations, heuristics, and autotuning spaces (see
    [`pallas_mosaic_tpu.py`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py)).
*   **Low-Level Kernels**: Separates standalone kernel implementations and
    shared low-level helpers into companion modules (see
    [`pallas_mosaic_tpu_kernel.py`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py)).
*   **Workload Specifications**: Catalogs canonical input shapes for testing,
    benchmarking, and autotuning (see
    [`arg_specs.py`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/arg_specs.py)).
*   **Test Suites**: Defines shared parameterized correctness checks alongside
    per-backend test suites (see
    [`test_base.py`](https://github.com/openxla/tokamax/blob/172338a8e2b969198a4282375b5db7e7add30a04/tokamax/_src/ops/ragged_dot/test_base.py)).

Ops within the `experimental` directory follow a similar structure, but may not
be exposed in the public API. These ops are intended for rapid iteration and
exploration, and may change frequently and significantly, with generally lower
expectations around ongoing maintenance and support.

### Which Layer Should Users Call

Choosing an entry point is fundamentally a tradeoff between convenience and
control. Most users should start by calling operators through the high-level
functional API, which provides a familiar style interface and automatically
selects a suitable backend implementation. When deterministic backend selection
is needed, users can pass an explicit implementation choice to the high-level
function while keeping the same call signature. For maximum control, users can
instantiate a specific operator implementation class directly to customize
kernel hyperparameters—such as tile sizes—and pass backend-specific arguments.

Calling low-level kernel functions directly without going through an operator
implementation class is not recommended. Doing so bypasses essential
infrastructure—including autotuning, input validation and canonicalization, and
metadata serialization—without offering any meaningful overhead savings, as the
operator wrappers are intentionally lightweight. In the absence of appropriate
autotuned hyperparameters, for instance, users will likely see significantly
suboptimal performance.

### Advanced Topics

#### Serialization

Kernels can be serialized to [StableHLO](https://openxla.org/stablehlo). Kernel
calls are JAX custom calls, which are by default banned in `jax.export`,
[requiring the use of](https://docs.jax.dev/en/latest/export/export.html#compatibility-guarantees-for-custom-calls)
`tokamax.DISABLE_JAX_EXPORT_CHECKS` to allow all Tokamax kernels to be exported:

```python
from jax import export

f_grad_exported = export.export(f_grad, disabled_checks=tokamax.DISABLE_JAX_EXPORT_CHECKS)(
    jax.ShapeDtypeStruct(x.shape, x.dtype),
    jax.ShapeDtypeStruct(scale.shape, scale.dtype),
)
```

Note that functions serialized with Tokamax kernels lose the device-independence
of standard StableHLO. Tokamax makes two serialization guarantees:

1.  A deserialized function serialized on a specific device will be guaranteed
    to run on the exact device it was serialized for.
2.  Tokamax gives the same
    [compatibility guarantees as JAX](https://docs.jax.dev/en/latest/export/export.html#compatibility-guarantees-for-custom-calls):
    6 month backward compatibility.

#### Cross-Compilation and CPU Export

Tokamax kernels can be exported or cross-compiled on a CPU host:

*   Explicit implementation selection (e.g. `implementation="mosaic_tpu_v2"`)
    automatically bypasses host device validation.
*   Alternatively, enable cross-compilation globally using
    `tokamax.config.cross_compile(True)`:

```python
with tokamax.config.cross_compile(True):
  f_exported = export.export(
      f, disabled_checks=tokamax.DISABLE_JAX_EXPORT_CHECKS
  )(
      jax.ShapeDtypeStruct(x.shape, x.dtype),
      ...,
  )
```
