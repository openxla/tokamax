# Tokamax: High-performance Custom Kernels

<a href="https://github.com/openxla/tokamax" title="Go to repository">
  <img src="https://img.shields.io/github/stars/openxla/tokamax" width=100 align="center">
</a>

## What is Tokamax?

Tokamax is a custom kernel library, with two main pillars:

**Optimized Kernel Portfolio**: High-performance implementations of key
operators including attention variants and mixture-of-experts building blocks
crafted to support both training and inference of the latest open-source models.

**Foundational Infrastructure**: Robust, integrated infrastructure that
simplifies kernel usage, maintenance, deployment, and authoring to mitigate
operational complexity e.g., autotuning and benchmarking—shifting the
operational complexity of kernel engineering away from individual developers.

## Why Choose Tokamax?

*   **Comprehensive Operator Portfolio**: Find a [catalog of kernels](https://openxla.org/tokamax/supported_ops) required to
    train and serve popular model architectures, including building blocks to
    compose mixture-of-experts (grouped matrix multiplication, gather/permute,
    scatter/unpermute, etc.), different attention variants (paged attention,
    splash attention, gated deltanet, kimi delta attention, multi-headed latent
    attention, etc.).
*   **State-of-the-Art Performance**: Tokamax kernels benefit from ongoing
    performance optimization to squeeze out every last bit of performance for
    key operators, yielding end-to-end model speedup.
*   **Next-Generation Hardware Support**: Tokamax kernels are
    proactively optimized to support next-generation hardware as it emerges
    (e.g., TPU v7
    [Ironwood](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack?e=48754805),
    NVIDIA Blackwell, or the upcoming TPU v8 series).
*   **Natively Multi-Framework**: To support the full
    spectrum of production environments, Tokamax offers a native PyTorch
    interface for use with `torch_tpu` alongside its native JAX ecosystem. It is
    designed to integrate out-of-the-box into elite training and serving
    frameworks.
*   **Production Tooling**: The library features integrated
    foundational infrastructure, including an
    [autotuning framework](https://openxla.org/tokamax/autotuning) that defines
    and searches implementation hyperparameter spaces and caches the optimal
    configurations per-shape, a low-overhead
    [benchmarking framework](https://openxla.org/tokamax/benchmarking), and
    foundational base classes that simplify authoring workflows such as defining
    input shapes of interest or performance heuristics, generating PyTorch
    interfaces to the JAX kernels, or exposing multiple implementations of the
    operator under a shared API.
*   **Open-Source Ecosystem**: Backed by a highly collaborative
    community, the library benefits from active contributions from both Google’s
    core teams and external industry leaders.
