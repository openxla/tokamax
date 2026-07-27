# KDA Chunk Backward — TPU/Pallas Kernel Design

Reference: Kimi Linear tech report — https://github.com/MoonshotAI/Kimi-Linear/blob/master/tech_report.pdf

Code:
- `tokamax/_src/ops/experimental/kda/base.py` — public KDA argument and recurrent-state contract
- `tokamax/_src/ops/experimental/kda/pallas_tpu.py` — Tokamax custom-VJP adapter
- `tokamax/_src/ops/experimental/kda/pallas_tpu_types.py` — typed `KdaResiduals` contract
- `tokamax/_src/ops/experimental/kda/pallas_tpu_fwd.py` — construction of forward residual matrices and saved states
- `tokamax/_src/ops/experimental/kda/pallas_tpu_bwd.py` — backward orchestrator and Pallas kernels
- `tokamax/_src/ops/experimental/kda/common.py` — shared gate cumsum, state recurrence, and mini-batch helper
- `tokamax/_src/ops/experimental/kda/cp_utils.py` — context-parallel gradient merge
- `tokamax/_src/ops/experimental/kda/utils.py` — alignment, unalignment, and L2-normalization backward helpers

---

## 1. Goal and Data Flow of the Backward Pass

KDA processes the sequence in chunks of `BT` time steps, with `BT=64` by default.

For each chunk, the forward output is the sum of two parts:

```math
\mathbf{o}
= \underbrace{s\,(\mathbf{q}\odot 2^{\mathbf{g}})\,\mathbf{h}}_{\text{inter: reads historical state}}
+ \underbrace{\mathrm{tril}(\mathbf{A}_{qk})\,\mathbf{v}_{\text{new}}}_{\text{intra: reads earlier tokens within the chunk}}
```

Intuitively, the output of a query comes from two paths:

- inter path: reads information from the historical state `h` that already exists when entering this chunk;
- intra path: reads information from the updated results of earlier tokens within this chunk.

The Tokamax entry is `PallasTpuKimiDeltaAttentionVjp._fwd(...)`, which passes the typed forward residuals and output cotangents to `chunk_kda_bwd_custom(...)`. The orchestrator receives the output gradient `do` and optional final-state gradient `dht`, and produces:

- `dq, dk, dv`: gradients with respect to query/key/value;
- `db`: gradient with respect to `beta`;
- `dg`: gradient with respect to the public gate input. The fused kernel's reverse cumsum first produces the gradient of the per-token activated gate before cumsum; when `use_gate_in_kernel=True`, `kda_gate_bwd(...)` maps that gradient further back to the raw public `g`;
- `dh0`: gradient with respect to the initial state, returned only when the caller provides `initial_state`;
- `dA, dbias`: gradients with respect to `A_log` and optional `dt_bias` when the gate activation is computed inside the kernel.

`dht` is an optional input representing the gradient of the final state. It is nonzero only when the final state continues to be used by a downstream loss; when an ordinary attention layer only consumes the output `o`, `dht` can be `None` and is treated as zero.

### 1.1 Forward Saved Values and Backward Recompute Boundary

The backward pass does not redo all forward computation from scratch. The current implementation divides the quantities needed by the backward pass into three categories.

Forward residuals saved:

- `Aqk`: the intra-chunk query-key attention matrix, shape `[H, B, T, BT]`;
- `Akk`: the WY inverse matrix, i.e. `A_kk^{-1}`, shape `[H, B, T, BT]`;
- `h`: the hidden state at the start of each chunk, shape `[H, B, NT, K, V]`; the forward stores it when `disable_recompute=True` and retains it in `KdaResiduals` when a custom backward is required.

Backward recompute:

- `w`: WY effective key / erase weight;
- `qg`: gated query;
- `kg`: gated key;
- `v_new`: WY-corrected value.

Optional recompute:

- When `use_gate_in_kernel=True`, both backward preparation paths currently recompute the post-cumsum gate from `g_org`, `A_log`, and optional `dt_bias` via `kda_gate_chunk_cumsum`, including the saved-`h` path where the forward residual currently also contains `g_cumsum`.
- When `disable_recompute=False`, it does not take the fast path that saves `h`, but instead reruns the forward state recurrence to recover `h` and `v_new`.

By naming convention, `disable_recompute=True` means "disable the full forward recurrence recompute, use the saved `h`." This is the current main path.

### 1.2 WY Representation and Gate Semantics

The state update of the intra-chunk delta-rule is inherently serial: each token controls its write strength with `beta`, and a later token depends on the state updated by the earlier token.

The WY representation encodes this serial update into a lower-triangular matrix `A_kk`, and uses the forward-saved inverse matrix `A = A_kk^{-1}` to rewrite the intra-chunk computation into parallel matmuls:

```math
\mathbf{u}=\mathbf{A}(\mathbf{v}\odot\boldsymbol\beta)
```

```math
\mathbf{w}=\mathbf{A}(\mathbf{k}\odot\boldsymbol\beta\odot 2^{\mathbf{g}})
```

```math
\mathbf{v}_{\text{new}}=\mathbf{u}-\mathbf{w}\,\mathbf{h}
```

`Akk` is the stored inverse `(I+L)^{-1}`. The strictly lower-triangular matrix `L` already depends on the row-wise `beta`, and `beta` is also multiplied explicitly into the value and gated-key right-hand sides used to construct `u` and `w`. Backward must therefore account for both paths by which `beta` affects the result.

In the equations below, `g` denotes the chunk-local cumulative gate in log2 space. This is an internal quantity, not the raw public gate argument. Therefore all decays are written as `2^(...)` and computed in the kernel via `exp2`:

- `qg = q * 2^g`: the query reads the historical state decayed from the chunk start to the current position;
- `kg = k * 2^(g_C - g)`: after the key is written it must decay to the end of the chunk before it can enter the next chunk's state;
- `2^g_C`: the decay of the entire state crossing one chunk, where `g_C` is the cumulative gate at the end of the chunk.

The corresponding inter-chunk recurrence is:

```math
\mathbf{h}^{[t+1]}
=2^{\mathbf{g}_C}\odot\mathbf{h}^{[t]}
+\mathrm{kg}^\top\mathbf{v}_{\text{new}}
```

### 1.3 API and Variable Reference

Entry point: `tokamax/_src/ops/experimental/kda/pallas_tpu.py::PallasTpuKimiDeltaAttentionVjp._fwd(...)`, followed by `tokamax/_src/ops/experimental/kda/pallas_tpu_bwd.py::chunk_kda_bwd_custom(...)`.

All main-path tensors use the head-first `[H, B, T, ...]` layout. The public KDA API and the Pallas backend already use this layout; there is no input transpose at the custom-VJP boundary. The backward consumes the aligned and optionally L2-normalized copies stored in `KdaResiduals`, rather than the replayed original tensors supplied by Tokamax's generic VJP contract.

Inputs and forward-saved values:

| Variable        | Shape                         | Meaning                                              |
| --------------- | ----------------------------- | ---------------------------------------------------- |
| `q, k`          | `[H, B, T, K]`                | query / key                                          |
| `v`             | `[H, B, T, V]`                | value                                                |
| `beta`          | `[H, B, T]`                   | delta-rule write strength per token                  |
| public `g`      | `[H, B, T, K]`                | per-token gate input before chunk-local cumsum       |
| `g_cumsum`      | `[H, B, T, K]` / `None`       | internal post-cumsum gate in log2 space; recomputed when omitted |
| `g_org`         | `[H, B, T, K]` / `None`       | retained raw gate when activation runs in the kernel |
| `A_log`         | `[H]` / `None`                 | gate activation parameter                            |
| `dt_bias`       | `[H*K]` / `None`               | optional gate activation bias                        |
| `Aqk`           | `[H, B, T, BT]`               | forward-saved intra-chunk attention matrix           |
| `Akk`           | `[H, B, T, BT]`               | forward-saved WY inverse matrix                      |
| `h`             | `[H, B, NT, K, V]`            | hidden state at each chunk start, saved on fast path |
| `do`            | `[H, B, T, V]`                | output gradient                                      |
| `initial_state` | `[B, N, H, K, V]` / `None`    | initial state under the public KDA contract          |
| `dht`           | `[B, N, H, K, V]` / `None`    | final-state cotangent under the public KDA contract  |

Backward intermediate quantities:

| Variable | Shape           | Meaning                                                                 |
| -------- | --------------- | ----------------------------------------------------------------------- |
| `u`      | `[H, B, T, V]`  | WY effective value, exists only transiently inside the recompute kernel |
| `w`      | `[H, B, T, K]`  | WY effective key / erase weight                                         |
| `qg, kg` | `[H, B, T, K]`  | gated query / key                                                       |
| `v_new`  | `[H, B, T, V]`  | value after WY and historical-state correction                          |
| `dAqk`   | `[H, B, T, BT]` | intra attention matrix gradient, from the dAv kernel                    |

Outputs:

| Variable     | Shape                      | Meaning                                                 |
| ------------ | -------------------------- | ------------------------------------------------------- |
| `dq, dk, dg` | `[H, B, T, K]`             | query / key / public gate-input gradients               |
| `dv`         | `[H, B, T, V]`             | value gradient                                          |
| `db`         | `[H, B, T]`                | `beta` gradient                                         |
| `dh0`        | `[B, N, H, K, V]` / `None` | initial-state gradient, matching the public input shape |
| `dA, dbias`  | `[H]` / `[H*K]` / `None`   | `A_log` / `dt_bias` gradients                           |

Static constraints:

- the delivered Pallas adapter requires `BT=chunk_size=64`;
- prepared `T` must be divisible by `BT`; variable-length inputs are aligned before residual construction;
- the main path uses the log2 gate, i.e. `use_exp2=True`;
- non-CP execution supports the backend's delivered `K<=256` contract, including non-128-aligned `K/V`; CP currently requires both `K` and `V` to be multiples of 128.

The backward orchestrator can normalize an internal four-dimensional final-state cotangent by inserting an `N=1` axis, but that is an implementation compatibility path. The public Tokamax KDA contract accepts and returns recurrent states in the five-dimensional `[B, N, H, K, V]` form. Fixed-length Pallas execution requires `N=1`.

---

## 2. Code-Based Call Chain and Kernel Structure

The current Tokamax call chain is:

```text
PallasTpuKimiDeltaAttentionVjp._fwd
└─ chunk_kda_bwd_custom
   ├─ unpack KdaResiduals
   ├─ align do with the retained cu_seqlens/aligned_cu_seqlens
   ├─ restore derived CP metadata retained by forward
   ├─ Stage 0: recover forward intermediate quantities
   │  ├─ if use_gate_in_kernel: kda_gate_chunk_cumsum(...)
   │  ├─ fast path disable_recompute=True:
   │  │  └─ fused_recompute_w_u_vnew_from_h_pallas(...)
   │  │     → w, qg, kg, v_new
   │  └─ fallback path disable_recompute=False:
   │     ├─ _recompute_w_u_fwd(...)
   │     │  → w, u, qg, kg
   │     └─ chunk_gated_delta_rule_fwd_h(...)
   │        → h, v_new
   ├─ Stage 1: chunk_kda_bwd_dAv_kernel(...)
   │  → dAqk, dv
   ├─ optional context parallel:
   │  ├─ chunk_gated_delta_rule_bwd_dhu_pre_process(...)
   │  │  → dS_ext, dM
   │  ├─ all_gather_into_tensor(...)
   │  ├─ _merge_dht(...)
   │  └─ construct the dht used by the fusion kernel
   ├─ Stage 2-5: _fused_dhu_wy_intra_cumsum_pallas_jit(...)
   │  → dq, dk, dv, db, dg, dh0
   ├─ optional gate backward:
   │  └─ kda_gate_bwd(...)
   │     → dg, dA, dbias
   ├─ optional l2norm_bwd(...) for dq/dk
   ├─ _unalign_output(...) for varlen dq/dk/dv/dg/db
   └─ restore dh0 shape and input dtypes
```

Mathematically, the backward pass can be decomposed into six stages:

1. recompute the forward intermediate quantities;
2. compute the gradient of the intra term `Aqk @ v_new`;
3. recur the state gradient `dh` backward along the chunk axis;
4. backpropagate the WY representation;
5. backpropagate the intra-chunk attention;
6. perform reverse cumsum on the gate gradient.

The saved-`h` main path maps these six stages onto three Pallas kernels:

```text
recompute fusion        → w, qg, kg, v_new
dAv                     → dAqk, dv
dhu/WY/intra/cumsum     → dq, dk, dv, db, dg, dh0
```

The low-memory path selected by `disable_recompute=False` additionally calls `_recompute_w_u_fwd` and `chunk_gated_delta_rule_fwd_h` to rerun the forward state recurrence. Context-parallel preparation, gate backward, L2-normalization backward, and varlen unalignment are optional operations around the core kernels.

### 2.1 Recompute Fusion Kernel

Entry point: `fused_recompute_w_u_vnew_from_h_pallas(...)`.

This kernel reads the saved `h` and the forward-saved `Akk`, and recomputes the following in parallel per chunk:

```text
u     = Akk @ (v * beta)
w     = Akk @ (k * beta * 2^g)
v_new = u - w @ h
qg    = q * 2^g
kg    = k * 2^(g_C - g)
```

`u` exists only inside the kernel and is not written back to HBM after `v_new` is computed. Since `h` is already saved, `v_new` no longer depends on the cross-chunk forward state recurrence, so each chunk can be scheduled independently.

The low-memory path is used when `h` is not retained. It first recovers `w, u, qg, kg` via `_recompute_w_u_fwd(...)`, then calls `chunk_gated_delta_rule_fwd_h(...)` to rerun the state recurrence and obtain `h, v_new`. This path reintroduces the inter-chunk serial dependency and trades compute for lower forward residual memory.

### 2.2 dAv Kernel

Entry point: `chunk_kda_bwd_dAv_kernel(...)`.

This kernel only handles the intra output:

```math
\mathrm{tril}(\mathbf{A}_{qk})\,\mathbf{v}_{\text{new}}
```

The corresponding backprop is:

```math
d\mathbf{A}_{qk}
=s\,\mathrm{tril}(d\mathbf{o}\,\mathbf{v}_{\text{new}}^\top)
```

```math
d\mathbf{v}_{\text{new}}
=\mathrm{tril}(\mathbf{A}_{qk})^\top d\mathbf{o}
```

In the code, `scale` is multiplied only once, when generating `dAqk`. The subsequent fusion kernel directly consumes the already-scaled `dAqk`, avoiding repeated scaling.

The stored `Aqk` already contains `scale`, so the value-gradient branch uses `Aqk^T @ do` without another factor. The tensor named `dAqk` is scaled before the fused intra backward because that kernel differentiates the underlying unscaled query-key relation directly. Although `chunk_kda_bwd_dAv_kernel` retains `q` and `k` in its public signature, the current launcher and kernel use only `v`, `Aqk`, `do`, `scale`, and tiling parameters.

### 2.3 dhu/WY/intra/cumsum Fusion Kernel

Entry point: `_fused_dhu_wy_intra_cumsum_pallas_jit(...)`.

This kernel executes in reverse chunk order and fuses four kinds of work:

1. `dhu` backward recurrence
   Maintains the cross-chunk state gradient `dh`, accumulating contributions from the output path, the delta-update path, and the chunk-decay path into the same VMEM scratch.

2. WY backward
   Backpropagates from `v_new = u - w @ h`, `u = Akk @ (v * beta)`, `w = Akk @ (...)` to `q, k, v, beta, g`, and obtains the gradient contribution to `Akk`.

3. intra backward
   Consumes the `dAqk` produced by the dAv kernel, continues backpropagating the gradient of the intra-chunk attention matrix to `q, k, beta, g`, and accumulates with the WY results.

4. reverse cumsum
   The forward gate is a chunk-local cumsum, so the backward pass needs a reverse cumsum. The current implementation keeps this inside the same kernel and no longer launches a separate cumsum kernel.

The chunk axis uses the `arbitrary` grid semantics to carry the backward recurrence state; the head and batch axes are parallel dimensions.

### 2.4 Optional Context Parallel

When context parallel is enabled, the backward pass inserts a cross-rank state-gradient merge between the dAv and fusion kernels.

The flow is:

1. `chunk_gated_delta_rule_bwd_dhu_pre_process(...)` scans only the first real local segment, because that is the segment that can receive state from an upstream rank, and produces:
   - `dS_ext`: this rank's external contribution to the input state gradient;
   - `dM`: this rank's backward transition-matrix chain product.
2. Pack `dS_ext` and `dM` along the last dimension and exchange them with a single `all_gather_into_tensor(...)`.
3. `_merge_dht(...)` merges downstream ranks' contributions in furthest-to-nearest order using the `post_num_ranks` and `is_last_rank` metadata retained by forward.
4. Construct a `[B,N,H,K,V]` `dht` and place each batch element's merged state gradient in its last real local segment slot before entering the fused backward kernel.

The CP backward direction goes from downstream rank to upstream rank, determined by aligned `segment_ids` and rank metadata restored into `CPContext`. `B>1` is handled independently per batch element. The CP contract does not accept an external `initial_state` or return an initial-state gradient.

### 2.5 Optional Gate Backward

When `use_gate_in_kernel=True`, `chunk_kda_bwd_custom(...)` calls `kda_gate_bwd(...)` after the core backward.

At this point the fusion kernel has already applied the chunk-local reverse cumsum, so its `dg` is the gradient with respect to the activated per-token gate before cumsum. `kda_gate_bwd(...)` then differentiates the activation and maps this gradient to the raw public `g`, `A_log`, and optional `dt_bias`, returning `dg`, `dA`, and `dbias` respectively.

---

## 3. TPU Adaptation

The goal of these implementation choices is to make the backward path align as closely as possible with the TPU's MXU/VPU/HBM execution model.

### 3.1 log2 Gate and `exp2`

The cumulative gate consumed by the intra and state-recurrence kernels is represented in log2 space, so decays are uniformly written as `2^g`. The public gate input is converted to this representation by the forward gate/cumsum stage; it should not be confused with the internal cumulative value.

This has two benefits:

- intra-chunk and inter-chunk decays can be combined via addition and subtraction in the exponent;
- the kernel uses the hardware `exp2`, avoiding the repeated change of base of `exp`.

### 3.2 fp32 Accumulation and bf16 Storage

Inputs are usually bf16, but matmuls use fp32 accumulate.

Inside the main fusion kernel, fp32 inputs select `jax.lax.Precision.HIGHEST`, while bf16 inputs use the default dot precision with fp32 accumulation. The reverse-cumsum dot explicitly uses `jax.lax.Precision.HIGHEST` for both input dtypes. These choices matter for `dg`, where cumulative gate contributions can nearly cancel.

### 3.3 VMEM Scratch Retains Cross-Stage Intermediates

The fusion kernel places the backward state `dh` in a `[MB, K, V]` VMEM scratch and updates it in reverse chunk order.

At the same time, `dv_new`, the WY intermediate gradients, the intra intermediate gradients, and the local results needed by the reverse cumsum all stay circulating within VMEM as much as possible. This avoids repeatedly writing back to and reading from HBM between the dhu, WY, intra, and cumsum logical stages.

### 3.4 Mini-Batch Controls DMA Granularity

A Pallas program processes `MB` head/chunk tiles at a time.

`MB` is selected from an estimated VMEM footprint, but the exact implementation differs by kernel. `chunk_kda_bwd_dAv_kernel` uses the shared `estimate_mini_batch(...)` helper. The saved-`h` recompute kernel, the main fusion kernel, and the CP pre-process currently use local VMEM-budget heuristics with kernel-specific caps and divisibility/alignment adjustments.

- too small leads to insufficient DMA granularity and low HBM bandwidth utilization;
- too large increases VMEM and register pressure and static unrolling overhead.

Therefore each launcher estimates its tile footprint and selects an `MB` subject to its grid, divisibility, and TPU minor-dimension alignment requirements. The saved-`h` recompute launcher can pad its flattened chunk count when no suitable divisor is available; the main fusion and CP launchers reduce `MB` until it divides the head count.

### 3.5 BlockSpec Data Movement and Explicit CP DMA

The three core saved-`h` kernels use Pallas grids and `BlockSpec` objects to describe HBM tiles. Their launchers do not contain an explicit hand-written async-copy loop.

The CP pre-process is different: `_chunk_gated_delta_rule_bwd_dhu_pre_process_kernel` is the active CP implementation and explicitly uses `pltpu.make_async_copy`, DMA semaphores, and double-buffered VMEM inputs. It is a single-program reverse scan over head groups and chunks, overlapping the next input transfer with the current matrix work and asynchronously writing each head group's `dS_ext/dM` summary.

### 3.6 Tiling and Padding Principles

TPU tiling aligns certain trailing dimensions to hardware tile boundaries. Although padded elements are semantically zero, they still consume real bandwidth at the HBM and DMA level.

The current implementation follows a practical principle: introduce a more complex packed layout only when the HBM traffic that changing the layout can eliminate exceeds the extra reshape/gather/copy cost.

Therefore:

- non-CP backward supports the delivered unaligned `K/V` cases without imposing the CP lane constraint, while the CP pre-process requires `K` and `V` to be multiples of 128;
- `[BT, BT]` small matrices such as `Aqk` and `Akk` accept the fixed padding brought by hardware alignment;
- scalar trailing dimensions such as `beta/db` use explicit singleton or two-dimensional layouts selected by each launcher.

In other words, padding itself is not necessarily a bad thing that must be eliminated. As long as the indexing and layout-conversion cost introduced by eliminating padding is higher, keeping the simple layout is actually faster.

### 3.7 Sub-block Normalization in the intra Backward

The intra backward involves decay terms like `2^(g_r - g_j)`. Since `g` is a cumulative quantity, directly subtracting and then exponentiating may cause numerical range problems.

The implementation splits the chunk into smaller sub-blocks and picks a reference gate within each sub-block for normalization. This way the exponent term is split into two parts relative to the reference point, keeping the intermediate values within a more controllable range.

The same sub-block structure also makes it convenient to organize the diagonal and off-diagonal blocks into a fixed number of batched matmuls, avoiding writing a quadratic loop over the sub-blocks.

### 3.8 Fixed Shapes Supporting varlen

TPU kernels require static shapes. Variable-length sequences do not trigger a separate ragged kernel, but still use fixed `[B, T]` tensors and a fixed `(H // MB, B, NT)` grid.

The adapter derives `cu_seqlens`, aligns every logical segment to `BT`, and retains both the original and aligned metadata in `KdaResiduals`. Backward first aligns `do`, then uses the retained aligned `segment_ids` and chunk mapping:

- chunks with segment id 0 are padding;
- the last chunk of each real segment seeds the backward recurrence with `dht`;
- the first chunk of each real segment outputs `dh0`.

For illustration only, consider `B=1, T=12, BT=4`, holding two sequences of lengths 8 and 4. The delivered adapter still requires `BT=64`; the smaller value only makes the metadata example compact.

```text
token segment_ids = [1,1,1,1, 1,1,1,1, 2,2,2,2]
chunk_seg_ids     = [   1,        1,        2   ]
                       chunk0     chunk1    chunk2
```

For fixed-length execution, `_fused_dhu_wy_intra_cumsum_pallas_jit` synthesizes an all-ones `[B,T]` segment-ID tensor, so every batch element is treated as one segment. After the core backward, variable-length token gradients are unaligned to the original `[B,T_original]` layout.

---

## 4. Fusion Performance Analysis

The measurements in this section are retained as historical optimization evidence. They predate the current Tokamax adapter, centralized alignment/residual contract, current CP metadata handoff, and some launcher-level naming changes. They must be rerun on the current commit before being used as an acceptance threshold or regression baseline.

Historical measurement configuration: TPU v6e, bf16, varlen `seq_lens=[1800,1500,2000,1200,800]`, `H=16, B=1, T=8192, K=V=128, BT=64`. After chunking, the leading dim is `16 × 133`.

### 4.1 Why Fusion Is the Main Optimization Direction

Most stages of the KDA backward have low arithmetic intensity and, when split apart individually, are easily HBM-bandwidth bound.

If each mathematical stage is turned into an independent kernel, a large number of intermediate tensors travel back and forth to HBM, for example:

- `w, qg, kg, v_new`;
- `dAqk, dv`;
- `dh, dv_new`;
- the WY and intra intermediate gradients;
- the input and output of the reverse cumsum.

The current implementation compresses these stages into three core Pallas kernels. The main benefit is not reducing the mathematical operations, but reducing the reads and writes at stage boundaries.

### 4.2 Historical Three-Kernel Main-Path Measurements

The core path consists of three parts:

Measurement: TPU **v6e**, bf16, varlen `seq_lens=[1800,1500,2000,1200,800]` (N=5 segments), H=16, B=1, T=8192, K=V=128, BT=64; after chunking the leading dim = 2128 = 16×133 (NT=133 = 128 base chunks + 5 segment paddings). HBM bandwidth taken as 1.6 TB/s.

| Kernel                            | Runtime (µs) | HBM (MB) | BW lower bound (µs) | BW utilization |
| --------------------------------- | ------------ | -------- | ------------------- | -------------- |
| `fused_recompute_w_u_vnew_from_h_pallas` | 285.0        | 418.9    | 261.83              | **91.9%**      |
| `chunk_kda_bwd_dAv_kernel`        | 148.3        | 209.2    | 130.74              | **88.2%**      |
| `_fused_dhu_wy_intra_cumsum_pallas_jit` | 900.1        | 951.8    | 594.9               | **66.1%**      |


Historically, the first two kernels showed that a simple compute structure and direct read/write pattern could utilize v6e HBM well for this workload.

In those measurements, the last fusion kernel was the main optimization target. Although it eliminates a large number of HBM round trips, it internally contains state recurrence, multiple intra-chunk matrix operations, gate gradient reductions, and reverse cumsum, so it mixes in compute, pipeline, and local layout overhead.

Measurement: TPU **v7x**, bf16, varlen `seq_lens=[1800,1500,2000,1200,800]` (N=5 segments), H=16, B=1, T=8192, K=V=128, BT=64; after chunking the leading dim = 2128 = 16×133 (NT=133 = 128 base chunks + 5 segment paddings). HBM bandwidth taken as 3.69 TB/s.

| Kernel                            | Runtime (µs) | HBM (MB) | BW lower bound (µs) | BW utilization |
| --------------------------------- | ------------ | -------- | ------------------- | -------------- |
| `fused_recompute_w_u_vnew_from_h_pallas` | 181.675      | 418.9    | 113.52              | **62.5%**      |
| `chunk_kda_bwd_dAv_kernel`        | 100.237      | 209.2    | 56.69               | **56.6%**      |
| `_fused_dhu_wy_intra_cumsum_pallas_jit` | 784.299      | 951.8    | 257.94              | **32.9%**      |

### 4.3 Performance Characteristics of Context Parallel

Measurement: TPU **v6e-4** (cp_size=4, 4-core SPMD parallelism), bf16, single segment `per_rank_T=8192` → global T=32768, H=16, K=V=128, BT=64; per-rank leading dim after chunking = 2128 = 16×133.

Single kernels (same padding basis):

| Kernel                                              | Runtime (µs) | HBM (MB) | BW lower bound (µs) | BW utilization |
| --------------------------------------------------- | ------------ | -------- | ------------------- | -------------- |
| `fused_recompute_w_u_vnew_from_h_pallas`            | 273.4        | 406.3    | 253.95              | **92.9%**      |
| `chunk_kda_bwd_dAv_kernel`                          | 146.8        | 202.9    | 126.81              | **86.4%**      |
| `_fused_dhu_wy_intra_cumsum_pallas_jit`             | 872.7        | 915.1    | 571.97              | **65.5%**      |
| **CP** `chunk_gated_delta_rule_bwd_dhu_pre_process` | 591.2        | 238.8    | 149.26              | **25.2%**      |

Measurement: TPU **v7x-4** (cp_size=4, 4-core SPMD parallelism), bf16, single segment `per_rank_T=8192` → global T=32768, H=16, K=V=128, BT=64; per-rank leading dim after chunking = 2128 = 16×133.

| Kernel                                              | Runtime (µs) | HBM (MB) | BW lower bound (µs) | BW utilization |
| --------------------------------------------------- | ------------ | -------- | ------------------- | -------------- |
| `fused_recompute_w_u_vnew_from_h_pallas`            | 181          | 406.3    | 110.11              | **60.8%**      |
| `chunk_kda_bwd_dAv_kernel`                          | 100          | 202.9    | 54.99               | **55.0%**      |
| `_fused_dhu_wy_intra_cumsum_pallas_jit`             | 785          | 915.1    | 247.99              | **31.6%**      |
| **CP** `chunk_gated_delta_rule_bwd_dhu_pre_process` | 511          | 238.8    | 64.72               | **12.7%**      |


The measured CP path used the same three core kernels as the non-CP path; its additional work came from the inter-rank state-gradient merge.

In the historical profile, the main CP cost was not the collective itself, but the local pre-process before communication:

- the pre-process needs to accumulate the input state gradient and the backward transition matrix along this rank's chunk sequence;
- this step has a scan/reduction nature and is hard to fully parallelize like an ordinary per-chunk matmul;
- `all_gather` itself is relatively cheap, because what is exchanged is the compressed state gradient and transition matrix, not the full-sequence token tensors.

Therefore, the optimization focus of the CP path should be on the pre-process:

- reduce its repeated reads of the full-chunk input;
- improve the parallelization of the chain product / scan;
- or move/fuse part of the state-merge logic forward into an existing backward stage.

### 4.4 Numerical Consistency

The current `compute_intra_backward(...)` implementation explicitly casts `dg_acc + dg_intra` to the input reference dtype and then back to fp32 before `compute_reverse_cumsum_dg(...)`. This dtype boundary is part of the delivered numerical behavior and should be preserved or deliberately revalidated when the fusion is refactored. Comparisons against an unfused or XLA reference should use tolerances appropriate to this truncation point.

### 4.5 Revalidation Matrix for the Current Implementation

Because the measurements above are historical, current-head validation should cover correctness and performance separately. At minimum, correctness revalidation should exercise:

- the saved-`h` path (`disable_recompute=True`) and the full-recompute path (`disable_recompute=False`);
- fixed-length and variable-length inputs, including multiple segments and `B>1`;
- a supplied `initial_state`, `output_final_state=True`, and a nonzero final-state cotangent;
- precomputed gates and `use_gate_in_kernel=True`, with and without `dt_bias`;
- `use_qk_l2norm_in_kernel=True`;
- non-CP unaligned `K/V` shapes allowed by the backend contract;
- CP execution at the supported 128-aligned `K/V` shapes and multiple CP sizes;
- bf16 inputs and the supported fp32 path.

Performance revalidation should report the exact commit, TPU generation and topology, software versions, shapes, varlen distribution, CP size, warmup policy, iteration count, and whether each number is a per-device or end-to-end latency. Kernel-level bandwidth estimates should be accompanied by an end-to-end backward measurement so that launch, collective, and adapter overheads are visible.

---

## 5. Future Optimization Directions

The current implementation is already centered around reducing HBM round trips through kernel fusion, but the profiling results leave two clear follow-up directions.

### 5.1 Improve v7 Performance

In the historical measurements, the core kernels ran faster on v7x but achieved lower estimated bandwidth utilization than on v6e. A current profile should first confirm that result; if it remains true, follow-up work should focus on making the fused kernels better match v7's execution model:

- revisit tile sizes, mini-batch selection, and VMEM pressure for v7;
- reduce local layout overhead and padding traffic where it is clearly visible in profiles;
- identify which parts of the main fusion kernel are compute- or pipeline-bound rather than bandwidth-bound.

The goal is not to change the mathematical decomposition, but to retune the existing fused path so it scales better with v7's higher bandwidth and compute capability.

### 5.2 Optimize Context Parallel

The CP path adds a local pre-process before the cross-rank exchange. The historical profile suggests that this local scan/reduction work was a larger bottleneck than the collective communication itself; a current profile should confirm the balance before optimization work begins.

Future CP optimization should therefore prioritize:

- reducing repeated reads in the pre-process;
- improving the parallelization of the transition-matrix chain product;
- exploring whether part of the CP state-gradient preparation can be fused into the existing backward kernels.

The communication pattern should stay compact: exchange state-level summaries rather than full token-level tensors.
