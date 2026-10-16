## PyTorch TPU API

Tokamax's kernels are typically authored in JAX and Pallas. For developer
convenience, they are also bridged via
[Torch TPU](https://github.com/google-pytorch/torch_tpu) and exposed as native
PyTorch custom operators (see
[`tokamax.experimental.torch_tpu.ops`](https://github.com/openxla/tokamax/blob/main/tokamax/experimental/torch_tpu/ops/__init__.py)).
You do not need to write JAX code or manage framework conversions to use them:
every operator accepts standard PyTorch inputs (e.g., `torch.Tensor`),
integrates with `torch.autograd` for automatic differentiation, and compiles
without graph breaks under `torch.compile`.

### How PyTorch Tokamax Operators Work

Every PyTorch operator in Tokamax inherits from a shared foundational base class
that bridges from JAX to PyTorch's custom operator subsystem using
`torch.tpu.pallas.jax_op` (see
[`TorchOp`](https://github.com/openxla/tokamax/blob/main/tokamax/experimental/torch_tpu/ops/torch_op.py#L172)).
When you invoke a Tokamax PyTorch operator, the infrastructure automatically
handles the following, implemented either in the base class or through
op-specific overrides of key methods:

*   **Custom Op Registration**: The base class registers the operator under the
    `tokamax::` namespace as a PyTorch `custom_op`, allowing dispatch to the
    underlying TPU kernel in both eager execution and in compiled mode (see
    [`TorchOp._register_ops()`](https://github.com/openxla/tokamax/blob/main/tokamax/experimental/torch_tpu/ops/torch_op.py#L397)).
*   **Autograd Integration**: For operators that support training, forward and
    backward TPU kernels are wired into `torch.autograd`. Intermediate
    activations (residuals) produced by the forward kernel are saved in the
    autograd context and passed directly to a specialized backward kernel when
    `.backward()` is called (see
    [`TorchOp.setup_context()`](https://github.com/openxla/tokamax/blob/main/tokamax/experimental/torch_tpu/ops/torch_op.py#L518)
    and
    [`TorchOp.backward()`](https://github.com/openxla/tokamax/blob/main/tokamax/experimental/torch_tpu/ops/torch_op.py#L538)).
*   **Automatic Kernel Configuration**: Hardware kernels expose low-level
    performance knobs—such as tile sizes and memory buffering depths—grouped
    into a backend `Config` object. When you call an operator without supplying
    a configuration, the base class automatically selects tuning parameters for
    your input tensor shapes and dtypes using Tokamax's built-in heuristics (see
    [`TorchOp.op_impl_call_config_setup()`](https://github.com/openxla/tokamax/blob/main/tokamax/experimental/torch_tpu/ops/torch_op.py#L482)).
*   **Symbolic Shape Tracing (`FakeTensor` Support)**: During `torch.compile`
    graph capture, PyTorch traces operations with "meta" tensors to infer output
    shapes and dtypes without executing hardware kernels. Operators provide a
    lightweight meta implementation to enable this tracing—including kernels
    with dynamic dimensions—without re-exporting the underlying JAX function
    (see
    [`TorchOp.fake_impl`](https://github.com/openxla/tokamax/blob/main/tokamax/experimental/torch_tpu/ops/torch_op.py#L251)).
*   **Buffer Donation**: For kernels that update state in place (such as
    KV-cache updates), the framework can mark specific input tensor buffers as
    reusable for outputs, avoiding redundant TPU memory allocations (see
    [`TorchOp.donate_argnums`](https://github.com/openxla/tokamax/blob/main/tokamax/experimental/torch_tpu/ops/torch_op.py#L244)).

Each operator family provides two callable singleton objects that share the same
PyTorch call signature: a wrapper around the pure-XLA reference implementation
to serve as the ground truth for correctness and numerics, and a wrapper around
the high-performance Pallas kernel implementation. For a more in-depth example,
please see the [demo colabs](https://openxla.org/tokamax/tokamax_demos).

### Kernel Configurations and Autotuning Cache

The Tokamax library has integrated foundational infrastructure to simplify
common developer journeys; the PyTorch interface aims to leverage those as much
as possible, while adapting to idiomatic PyTorch ecosystem conventions. For
example, Tokamax’s JAX interface looks up autotuned configurations by default
from a cache file on disk. However, to preserve Torch compilation behavior,
PyTorch operators instead use heuristics to choose a configuration based on
input tensor shapes and dtypes, while optionally supporting explicit, sideband
user-driven autotuning cache lookups.

If you wish to inspect the chosen configuration ahead of time or load
pre-autotuned settings from Tokamax's on-disk autotuning cache, use
[`tokamax_torch_utils.get_configs()`](https://github.com/openxla/tokamax/blob/main/tokamax/experimental/torch_tpu/ops/torch_utils.py#L107),
in one of two ways, both of which support independent configs for the forward
and backward passes:

*   **Built-in Heuristics (`from_autotuning_cache=False` default)**: Computes
    default forward and backward configurations using the operator's analytical
    heuristics for the given tensor shapes and dtypes (see
    [`BoundArguments.heuristics_config`](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/op.py#L469)).
*   **Autotuning Cache Lookup (`from_autotuning_cache=True`)**: Checks Tokamax's
    repository of offline-autotuned configurations for a match on the target TPU
    generation and input shapes, falling back to heuristics if the workload has
    not been cached (see
    [`BoundArguments.default_config`](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/op.py#L469)).

### Repository Structure

Within each operator directory under
[`tokamax/experimental/torch_tpu/ops`](https://github.com/openxla/tokamax/tree/main/tokamax/experimental/torch_tpu/ops),
files follow a consistent layout, illustrated here using `ragged_dot`:

*   **Reference Wrapper**: Wraps the portable Tokamax base operator for
    numerical reference and fallback execution (see
    [`torch_base.py`](https://github.com/openxla/tokamax/blob/main/tokamax/experimental/torch_tpu/ops/ragged_dot/torch_base.py)).
*   **Pallas Mosaic TPU Wrapper**: Wraps the hardware-specific Pallas Mosaic TPU
    operator and its configuration serialization logic (see
    [`torch_pallas_mosaic_tpu.py`](https://github.com/openxla/tokamax/blob/main/tokamax/experimental/torch_tpu/ops/ragged_dot/torch_pallas_mosaic_tpu.py)).
*   **Test Suites**: Verifies numerical equivalence against PyTorch and JAX
    references, `torch.autograd` gradients, configuration lookup, and
    `@torch.compile(fullgraph=True)` execution (see
    [`torch_base_test.py`](https://github.com/openxla/tokamax/blob/main/tokamax/experimental/torch_tpu/ops/ragged_dot/torch_base_test.py)
    and
    [`torch_pallas_mosaic_tpu_test.py`](https://github.com/openxla/tokamax/blob/main/tokamax/experimental/torch_tpu/ops/ragged_dot/torch_pallas_mosaic_tpu_test.py)).
