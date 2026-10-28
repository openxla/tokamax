# Routing MoE

## High Level Summary

- E2E MoE solution that can handle complex permutation of routing use cases.
- Leverages "Functional Core, Imperative Shell" design pattern.
  - Abstracts away complex memory transfer logic from core computational logic.
  - Maximizes extensibility of supporting complex data transfer strategy.
- Provides helper functions for any data movement happening outside of gmm.

## Status

- Fully supports in-kernel random local HBM read/write.
- Provides basic compiler based helper functions.

## Roadmap

- Support FP8 remote HBM read/write to minimize penalty of pre-topk reduce collective.
- Support subchannel quantized weight and bias.
- Support calculating near-optimal routing strategy at compile time for good OOTB experience.

## Folder Structure

```text
routing_moe/
├── common/
│   ├── arg.py: Spec and wrapper of inputs for communicating between modules.
│   └── layout_util.py: Utility for compact and large layout transformation.
├── gmm/
│   ├── compute.py: Stateless matrix multiplication logic.
│   ├── config.py: Compile time static configuration object.
│   ├── kernel.py: gmm entry point that expects compact layout for activations.
│   ├── memory_ref.py: Memory module that handles buffer management and DMA.
│   ├── metadata.py: Dynamic per-step metadata consumed by pipeline and memory module.
│   ├── plan.py: Tunable parameters for kernel execution that has no impact on outputs.
│   └── pipeline.py: Main Orchestrator for stateful DMA and stateless compute.
├── io_relayout/
│   ├── input_pack.py: Handles large -> compact layout transformation.
│   └── output_reduce.py: Handles topk reduce and compact -> large transformation.
├── layer/
│   ├── config.py: Compile time static configuration object for MoE layer.
│   ├── modular.py: Modular MoE entry point accepting pre-computed top-k weights and indices.
│   ├── monolithic.py: Monolithic MoE entry point that computes routing and invokes modular MoE.
│   └── plan.py: Tunable parameters for MoE layer execution that have no impact on outputs.
└── test/
    ├── common/
    │   ├── test_shapes.py: Static model configs (dims, experts, top-k, dtype).
    │   ├── test_cases.py: Serving-time cases (model config + token/shard count).
    │   ├── test_inputs.py: Shared input and routing tensor generators.
    │   ├── test_reference.py: Pure-XLA reference implementations.
    │   └── test_util.py: Feature-support checks and benchmarking helpers.
    └── {gmm, io_relayout, layer}/
        ├── *_test_lib.py: Case generation and reference validation helpers.
        ├── *_unit_test.py: Synthetic edge-case and stress tests.
        └── *_prod_test.py: Production model shapes and benchmarks.
```
