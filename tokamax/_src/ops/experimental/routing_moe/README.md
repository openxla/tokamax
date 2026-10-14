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

- Replace compiler based helper functions with more efficient kernels.
- Support in-kernel random remote HBM read/write to hide collective behind compute.
- Support FP8 remote HBM read/write to minimize penalty of pre-topk reduce collective.
- Support up and down fusion to increase arithmetic intensity.
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
│   └── wrapper.py: MoE entry point that accepts activation and router output.
└── test/
    ├── gmm_test.py: Standalone test for grouped matrix multiplication.
    └── wrapper_test.py: A test for entire single MoE layer.
```
