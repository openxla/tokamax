# Tokamax

Tokamax is a library of custom accelerator kernels, supporting both NVIDIA GPUs and Google [TPUs]( https://cloud.google.com/tpu/docs/intro-to-tpu). Tokamax provides state-of-the-art custom kernel implementations built on top of [JAX](https://docs.jax.dev/en/latest/index.html) and [Pallas](https://docs.jax.dev/en/latest/pallas/index.html).

# Status

Tokamax is still heavily under active development. Incomplete features and API changes are to be expected. 
We currently support the following kernels for GPUs:

* Attention
* Gated Linear Unit
* Normalization
* Ragged Dot

TPU support is incoming.

## How to Install

```
git clone https://github.com/openxla/tokamax.git
```

