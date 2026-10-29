# Tokamax

[![nightly](https://img.shields.io/github/actions/workflow/status/openxla/tokamax/ci-nightly.yml?label=nightly&logo=githubactions&logoColor=white)](https://github.com/openxla/tokamax/actions/workflows/ci-nightly.yml)
[![PyPI version](https://img.shields.io/pypi/v/tokamax?color=blue)](https://pypi.org/project/tokamax/)

Tokamax is a library of custom accelerator kernels. It provides highly optimized
implementations of key operators, including mixture-of-experts building blocks
like grouped matrix multiplication and token gathers/scatters, attention
variants like paged attention, splash attention, gated deltanet, Kimi delta
attention and multiheaded latent attention, and TopK computation. The operator
portfolio will evolve over time to support the latest OSS models running on the
latest hardware accelerators, including Google
[TPUs](https://cloud.google.com/tpu/docs/intro-to-tpu) and NVIDIA GPUs. The
library also provides foundational infrastructure to simplify kernel authoring,
maintenance, and deployment; for example, features like autotuning, low-overhead
benchmarking, and so on.

Tokamax kernels are typically built on top of
[JAX](https://docs.jax.dev/en/latest/index.html) and
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html), but also expose
native PyTorch APIs on TPUs for developer convenience.

## Documentation

Detailed documentation can be found at
[openxla.org/tokamax](https://openxla.org/tokamax/), including a list of
[supported ops](https://openxla.org/tokamax/supported_ops) on different hardware
accelerators, [benchmarking](https://openxla.org/tokamax/benchmarking)
infrastructure, and [autotuning](https://openxla.org/tokamax/autotuning).

## Getting Started

### Installation

The latest Tokamax [PyPI release](https://pypi.org/project/tokamax/):

```python
pip install -U tokamax
```

The latest bleeding edge version from Github, with no stability guarantees:

```python
pip install git+https://github.com/openxla/tokamax.git
```

### Usage Examples

Please see the Colab notebooks [here](https://openxla.org/tokamax/tokamax_demos)
for usage examples.

### Contributing

Please see documentation on
[how to contribute](https://openxla.org/tokamax/contributing) and our
[code style guide](https://openxla.org/tokamax/code_style).
