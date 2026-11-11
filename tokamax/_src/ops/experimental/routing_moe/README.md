# Routing MoE

## High Level Summary

- E2E MoE solution that can handle complex permutation of routing use cases.
- Leverages "Functional Core, Imperative Shell" design pattern.
  - Abstracts away complex memory transfer logic from core computational logic.
  - Maximizes extensibility of supporting complex data transfer strategy.
- Provides helper functions for any data movement happening outside of gmm.

## Status

- Supports in-kernel random local HBM gather/scatter and remote EP HBM scatter.
- Fuses gate/up, activation, and down projections when tiles fit in VMEM.
  - Falls back to a two-kernel split through compact HBM layout otherwise.
- Provides `output_reduce_unpack` for top-k reduction and layout unpacking.
  - Supports ragged reduction and optional per-lane FP8 dequantization.
- Supports quantized FP8 local and remote scatter with lane-strided scales.
- Supports subchannel weight quantization with dynamic activation quantization.

## Roadmap

- Support FP8 input using `io_relayout/input_pack.py`.
- Support K dimension tiling, including `k2` tiling for fused up/down kernels.
- Support compile-time heuristics for near-optimal OOTB routing strategies.
- Optimizations low-batch decode.

## Folder Structure

```text
routing_moe/
├── common/
│   ├── arg.py: PyTree wrappers pairing tensor data with static specs.
│   ├── dataclass_util.py: Invariant decorators for unresolved/resolved classes.
│   ├── layout_util.py: Utility for compact and large layout transformation.
│   ├── out_format.py: Target representation formats for kernel outputs.
│   ├── quant_util.py: Dynamic activation quantization helpers.
│   └── spec.py: Static specs of materialized kernel inputs and outputs.
├── gmm/
│   ├── compute.py: Stateless matrix multiplication logic.
│   ├── config.py: Compile-time static configuration object.
│   ├── kernel.py: GMM entry point expecting compact layout for activations.
│   ├── memory_ref.py: Memory module that handles buffer management and DMA.
│   ├── metadata.py: Per-step metadata for pipeline and memory modules.
│   ├── pipeline.py: Main orchestrator for stateful DMA and stateless compute.
│   ├── plan.py: Tunable execution parameters with no impact on outputs.
│   └── spec.py: Validated problem specification of the routing GMM kernel.
├── io_relayout/
│   ├── input_pack.py: Handles large -> compact layout transformation.
│   └── output_reduce.py: Handles top-k reduce and compact -> large relayout.
├── layer/
│   ├── config.py: Compile-time static configuration object for MoE layer.
│   ├── modular.py: Modular MoE entry point for pre-computed top-k routing.
│   ├── monolithic.py: Monolithic MoE entry point computing routing + MoE.
│   └── plan.py: Tunable MoE execution parameters with no impact on outputs.
└── test/
    ├── helper/
    │   ├── helper_shapes.py: Static model configs (dims, experts, top-k).
    │   ├── helper_cases.py: Serving-time cases (model config + token/shard).
    │   ├── helper_inputs.py: Shared input and routing tensor generators.
    │   ├── helper_reference.py: Pure-XLA reference implementations.
    │   ├── helper_util.py: Feature-support checks and benchmarking helpers.
    │   └── *_unit_test.py: Unit tests for shared test helpers.
    └── {commmon, gmm, io_relayout, layer}/
        ├── *_test_lib.py: Case generation and reference validation helpers.
        ├── *_unit_test.py: Synthetic edge-case and stress tests.
        └── *_prod_test.py: Production model shapes and benchmarks.
```
