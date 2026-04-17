## Benchmarking

### Benchmarking Basics

Tokamax provides integrated performance benchmarking infrastructure. You can
benchmark your op `f(x)` for a given set of inputs as follows:

```
# Conforming and initializing data to prepare for benchmarking
f_std, args = tokamax.standardize_function(f, kwargs={'x': x})

# Execute and measure timing
bench: tokamax.BenchmarkData = tokamax.benchmark(f_std, args)
```

`standardize_function` simplifies complicated functions, with non-array
arguments for example. It first creates a standard form with a single argument
`args`, which is a list of either abstract or concrete arrays `jax.Array |
jax.ShapeDtypeStruct`. It then randomly initializes all the abstract tensors,
and returns a standardized `f_std(args)` with only concrete array arguments.
This can be cleanly jitted without worrying about static arguments such as
strings.

Please see the respective docstrings for the full suite of options supported by
each of these functions; some key topics are discussed below.

### Advanced Benchmarking Topics

#### Run Iterations

`tokamax.benchmark` lets you pick the number of iterations; more iterations
typically results in reduced measurement noise e.g., `tokamax.benchmark(f_std,
args, iterations=num_iters).`However, if the number of iterations is too large
in a short period of time, thermal throttling may be triggered especially for
compute-heavy kernels, impacting execution time. Balancing these factors is
often an empirical exercise. A suggested approach is to run a small number of
iterations in each experiment with multiple spaced out experiments, at the cost
of increased wall clock time.

#### Benchmarking Method

JAX Python overhead is often much larger than the actual accelerator kernel
execution time. This means the usual approach of timing
`jax.block_until_ready(f(x))` won't be useful. `benchmark` lets you pick the
underlying timing methodology used for benchmarking e.g. `benchmark(f_std, args,
iterations=num_iters, method=method)`

For TPU kernels, we strongly recommend `method=xprof_hermetic`, which invokes
the [XProf](openxla.org/xprof) profiler and measures execution time on the
hardware. This method imposes almost no instrumentation overhead due to custom
full-stack support including the hardware and the compiler.

For GPU kernels, you may use `xprof_hermetic` as well; XProf in turn employs
NVIDIA’s [CUPTI](https://docs.nvidia.com/cupti) APIs. You may also directly
invoke a CUPTI timer with `method=cupti`. Either method does impose some
variable overhead, typically up to 5%.

#### Data Distribution

[Prior work](https://www.thonking.ai/p/strangely-matrix-multiplications) has
shown that performance can vary significantly based on data distributions, due
to complex hardware-level power and thermal interactions. To address this,
`standardize_function` initializes input arrays in a manner representative of
actual training jobs e.g., any real-valued `jax.ShapeDtypeStruct` will be
initialized randomly. You may wish to adapt this for your needs.

#### Benchmarking Mode

`standardize_function` lets you select `mode=forward` to choose forward only,
`forward_res` to choose forward and compute residuals, `vjp` to choose the
VJP-function only, and `forward_and_vjp` to compute a full forward and VJP pass
for benchmarking. Note that benchmarking VJP-only could result in OOMs, because
the forward pass is computed outside the returned standardized function, with
all intermediates baked into the HLO which remains resident in HBM memory.
