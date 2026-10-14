## Splash Attention

Splash Attention is an experimental TPU-optimized attention kernel within
Tokamax that enables efficient block-sparse attention computation with custom
masking patterns.

### Overview

Splash Attention provides a high-performance implementation of dot-product
attention on TPU that supports:

*   **Block-sparse attention masks** — efficiently skip computation for masked
    blocks, reducing both memory and compute cost.
*   **Ring attention** — distributed attention across multiple TPU devices for
    processing very long sequences.
*   **Custom mask patterns** — flexible mask specification via `SplashAttentionMask`
    and `SplashAttentionMaskInfo` for complex attention patterns.

### Status

Splash Attention is currently **experimental** and located under
`tokamax._src.ops.experimental.tpu.splash_attention`. The API may change without
notice.

### Components

| Module | Description |
|---|---|
| `splash_attention_kernel.py` | Core TPU splash attention kernel implementation |
| `ring_attention_kernel.py` | Ring attention variant for multi-device long-sequence processing |
| `splash_attention_mask.py` | Mask data structures for defining sparse attention patterns |
| `splash_attention_mask_info.py` | Mask metadata and analysis utilities |

### Usage

Splash Attention is designed for workloads where the attention pattern is
block-sparse (e.g., local attention, sliding window, or custom structured
sparsity). For dense attention on TPU, the standard
`tokamax.dot_product_attention` with `implementation="mosaic_tpu"` is
recommended.

```python
from tokamax._src.ops.experimental.tpu.splash_attention import (
    splash_attention_kernel,
    splash_attention_mask,
)
```

> **Note:** Splash Attention requires TPU hardware. It is not available on GPU.
> See the `microbenchmarks.pdf` in the source directory for performance
> characteristics.

### References

*   Internal source: `tokamax/_src/ops/experimental/tpu/splash_attention/`
*   Performance data: `microbenchmarks.pdf` (included in the source directory)