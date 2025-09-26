# Tokamax

[![CI](https://github.com/openxla/tokamax/actions/workflows/ci-build.yml/badge.svg)](https://github.com/openxla/tokamax/actions/workflows/ci-build.yml)
[![PyPI version](https://img.shields.io/pypi/v/tokamax)](https://pypi.org/project/tokamax/)
![Static Badge](https://img.shields.io/badge/Under_Development-red)

Tokamax is a library of custom accelerator kernels, supporting both NVIDIA GPUs
and Google [TPUs](https://cloud.google.com/tpu/docs/intro-to-tpu). Tokamax
provides state-of-the-art custom kernel implementations built on top of
[JAX](https://docs.jax.dev/en/latest/index.html) and
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html).

Tokamax also provides tooling for users to build and autotune their own custom
accelerator kernels.

## Status

Tokamax is still heavily under development. Incomplete features and API changes
are to be expected.

We currently support the following GPU kernels:

*   `tokamax.dot_product_attention`
    ([FlashAttention](https://arxiv.org/abs/2205.14135)).
*   `tokamax.gated_linear_unit`
    ([Gated linear units](https://arxiv.org/abs/2002.05202) (SwiGLU etc)).
*   `tokamax.layer_norm`
    ([Layer normalization](https://arxiv.org/abs/1607.06450) and
    [Root Mean Squared normalization](https://arxiv.org/abs/1910.07467)).

And the following for both GPU and TPU:

*   `tokamax.ragged_dot`
    ([Mixture of Experts](https://arxiv.org/abs/2211.15841)).

## Installation

The latest Tokamax [PyPI release](https://pypi.org/project/tokamax/):

```python
pip install -U tokamax
```

The latest bleeding edge version from Github, with no stability guarantees:

```python
pip install git+https://github.com/openxla/tokamax.git
```

## Using Tokamax

Consider a function containing Tokamax functions running on an H100 GPU:

```python
import jax
import jax.numpy as jnp
import tokamax

def loss(x, scale):
  x = tokamax.layer_norm(
      x, scale=scale, offset=None, implementation="triton"
  )
  x = tokamax.dot_product_attention(x, x, x, implementation="xla_chunked")
  x = tokamax.layer_norm(x, scale=scale, offset=None, implementation=None)
  x = tokamax.dot_product_attention(x, x, x, implementation="mosaic")
  return jnp.sum(x)

f_grad = jax.jit(jax.grad(loss))
```

With `implementation=None`, Tokamax is allowed to select the best implementation
for each kernel shape. It is even allowed to choose different implementations
for the forward pass and gradient. It will also always be supported, as it can
fall back to an XLA implementation `implementation='xla'`.

However, you may want to choose a specific implementation of the kernel, and
fail if it is unsupported. For instance, `implementation="mosaic"` will try to
use a [Pallas:Mosaic GPU](https://docs.jax.dev/en/latest/pallas/gpu/index.html)
kernel if possible, and throw an exception if this is unsupported for any
reason. For example, using FP64 inputs are unsupported, or older GPUs.

### Evaluate the Gradient

```python
channels, seq_len, batch_size, num_heads = (64, 2048, 32, 16)
scale = jax.random.normal(jax.random.key(0), (channels,), dtype=jnp.float32)
x = jax.random.normal(
    jax.random.key(1),
    (batch_size, seq_len, num_heads, channels),
    dtype=jnp.bfloat16,
)

out = f_grad(x, scale)
```

### Autotuning

To get the best performance, autotune all Tokamax kernels in `f_grad`:

```python
autotune_result: tokamax.AutotuningResult = tokamax.autotune(f, x, scale)
```

`autotune_result` can be used as a context-manager, using the autotuned configs
for all Tokamax kernels in `f_grad`:

```python
with autotune_result:
  out_autotuned = f_grad(x, scale)
```

To serialize and reuse the result of a potentially expensive
`tokamax.autotuning` call:

```python
autotune_result_json: str = autotune_result.dumps()
autotune_result = tokamax.AutotuningResult.loads(autotune_result_json)
```

Users can autotune their own kernels with `tokamax.autotune` by inheriting from
the `tokamax.Op` class and overriding the `tokamax.Op._get_autotuning_configs`
method to define the autotuning search-space.

Note that autotuning is fundamentally non-deterministic: measuring kernel
execution times is noisy. As different configs chosen during autotuning can lead
to different numerics, this is a potential source of numerical non-determinism.
Serializing and reusing fixed autotuning results is a way to ensure the same
numerics across sessions.

### Serialization

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

### Benchmarking

JAX Python overhead is often much larger than the actual accelerator kernel
execution time. This means the usual approach of timing
`jax.block_until_ready(f_grad(x, scale))` won't be useful. Tokamax has utilities
for only measuring accelerator execution time:

```python

f_std, args = tokamax.benchmarking.standardize_function(f, kwargs={'x': x, 'scale': scale})
run = tokamax.benchmarking.compile_benchmark(f_std, args)
bench: tokamax.benchmarking.BenchmarkData = run(args)
```

There are different measurement techniques: for example, on GPU, there is the
[CUPTI profiler](https://docs.nvidia.com/cupti) that can be specified via
`run(args, method='cupti')`. This instruments the kernel and adds some a small
overhead. Using `run(args, method='cuda_events')` uses CUDA synchronization
events to measure the device execution time. The default `run(args,
method=None)` allows Tokamax to choose the method, and works for both TPU and
GPU. Benchmark noise can be reduced by increasing the number of iterations
`run(args, iterations=10)`.

## Disclaimer

This is not an official Google product.
