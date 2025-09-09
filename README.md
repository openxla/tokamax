# Tokamax

[![CI](https://github.com/openxla/tokamax/actions/workflows/ci-build.yml/badge.svg)](https://github.com/openxla/tokamax/actions/workflows/ci-build.yml)
[![PyPI version](https://img.shields.io/pypi/v/tokamax)](https://pypi.org/project/tokamax/)

Tokamax is a library of custom accelerator kernels, supporting both NVIDIA GPUs and Google [TPUs]( https://cloud.google.com/tpu/docs/intro-to-tpu). Tokamax provides state-of-the-art custom kernel implementations built on top of [JAX](https://docs.jax.dev/en/latest/index.html) and [Pallas](https://docs.jax.dev/en/latest/pallas/index.html).

## Status

Tokamax is still heavily under active development. Incomplete features and API changes are to be expected. 
We currently support the following kernels for GPUs:

* Attention
* Gated Linear Unit
* Normalization
* Ragged Dot

TPU support is incoming.

## Installation
The latest Tokamax [PyPI release](https://pypi.org/project/tokamax/):
```
pip install -U tokamax
```
The latest version from Github:
```
pip install git+https://github.com/openxla/tokamax.git
```

