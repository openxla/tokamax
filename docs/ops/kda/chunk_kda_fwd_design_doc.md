# KDA Forward Delivery Design Document

This document describes the delivered forward design of Kimi Delta Attention (KDA). It progresses from scope and API contract through the canonical call chain, physical kernel design, numerical correctness, validation, performance, and design trade-offs, with mathematical derivation and historical performance retained as appendices. The mathematical definitions follow Section 3.1 and Appendix B of *Kimi Linear: An Expressive, Efficient Attention Architecture* (arXiv:2510.26692v2). Implementation details reflect the current code behavior.

## 1. Overview and Scope

### 1.1 Objective

KDA forward accepts query, key, value, a per-channel decay gate, and a delta-rule write coefficient. It produces an output for each token while maintaining a fixed-size associative memory state. For every batch element and attention head, the state has shape `K x V` and does not grow with sequence length.

The Tokamax TPU backend enters this implementation through:

```python
(o, final_state), residuals = backend._fwd(q, k, v, g, beta, ...)
```

The primary tensor inputs and outputs are:

| Symbol | Tokamax backend shape or type | Meaning |
|---|---:|---|
| `B` | - | Batch size |
| `T` | - | Input token length; rank-local length under Context Parallelism |
| `H` | - | Number of attention heads |
| `N` | - | Number of logical segments represented in the recurrent-state axis; one for fixed-length input |
| `K` | - | Query/key head dimension, denoted by `d_k` in the mathematical sections |
| `V` | - | Value head dimension, denoted by `d_v` in the mathematical sections |
| `q` | `[H, B, T, K]` | Query |
| `k` | `[H, B, T, K]` | Key |
| `v` | `[H, B, T, V]` | Value |
| `g` | `[H, B, T, K]` | Raw gate or an already activated natural-log decay |
| `beta` | `[H, B, T]` | Per-token delta-rule write coefficient |
| `initial_state` | `[B, N, H, K, V]`, or `None` | Optional initial `K x V` state for each sequence; fixed-length input requires `N=1`, and CP does not accept an external initial state |
| `o` | `[H, B, T, V]` | Token output |
| `final_state` | `[B, N, H, K, V]`, or `None` | Final `K x V` state for each sequence; `N=1` for fixed-length input |
| `residuals` | `KdaResiduals`, or `None` | Internal forward residual contract returned only when requested for the custom backward; it is not part of the public `(o, final_state)` result |

### 1.2 Problem Addressed

Standard causal attention explicitly computes pairwise token relationships, so its representation of history grows with sequence length. KDA compresses history into a state matrix. Queries read from that state, keys and values update it through the delta rule, and a per-channel gate controls the forgetting rate of each key channel.

A direct token-by-token recurrence has the following temporal dependency:

```text
S_0 -> S_1 -> S_2 -> ... -> S_T
```

This form maps directly to the mathematical definition but provides little parallelism along the time dimension for long sequences. The current implementation partitions the sequence into fixed-size chunks:

1. Within a chunk, sequential dependencies are rewritten as a unit lower-triangular linear system whose dominant work is matrix multiplication.
2. Across chunks, only the fixed-size state matrix is propagated recurrently.
3. Variable-length sequences are aligned to chunk boundaries so that one chunk never crosses two independent sequences.
4. Context Parallelism (CP) connects portions of the same logical sequence on different ranks through affine state summaries.

### 1.3 Semantic Differences from Standard Attention

| Aspect | Standard causal attention | KDA chunk forward |
|---|---|---|
| History representation | Token-token attention scores | `d_k x d_v` state matrix |
| Time complexity | Typically grows with `T^2` | State path grows linearly with `T` |
| Normalization | Softmax | No softmax; memory is controlled by decay gates and the delta rule |
| Causality | Specified by an attention mask | Enforced jointly by state recurrence and lower-triangular intra-chunk matrices |
| Cross-chunk dependency | Usually no recurrent state | The final state of one chunk initializes the next chunk |
| Variable-length isolation | Masking or packed attention | Segment alignment, state reset, and chunk-to-sequence mapping |

KDA output is not a softmax-weighted sum. It is the sum of a read from the state that existed before the current chunk and the contributions written earlier within the current chunk. Together, these terms reproduce the token-level KDA recurrence.

## 2. API Contract and Support Matrix

### 2.1 Forward Control Parameters

`KimiDeltaAttention.bind` supplies public defaults and validates the public contract before dispatch. `PallasTpuKimiDeltaAttention._fwd` therefore receives fully bound values. The table below records the public default together with the backend meaning of every semantically relevant auxiliary argument; the primary data tensors are not repeated.

| Group | Parameter | Public default | Backend contract and effect |
|---|---|---:|---|
| Gate | `A_log` | `None` | Auxiliary tensor shaped `[H]`; required when `use_gate_in_kernel=True` and used to scale raw-gate activation |
| Gate | `dt_bias` | `None` | Optional auxiliary tensor shaped `[H*K]`; added to the raw gate before activation |
| Gate | `use_gate_in_kernel` | `False` | `False` means `g` already contains `ln(alpha)`; `True` means the backend activates raw `g` using `A_log`, optional `dt_bias`, and `lower_bound` |
| Gate | `safe_gate` | `True` | Limits the exponent range within Stage 2 sub-blocks to prevent `exp2` overflow for large gate magnitudes |
| Gate | `lower_bound` | `None` | `None` selects softplus raw-gate activation; a non-`None` value selects the sigmoid variant and must satisfy `-5 <= lower_bound < 0` |
| Numerics | `scale` | `None` | The public contract canonicalizes `None` to `K^{-1/2}`; `_fwd` receives the resulting `float` query scale |
| Numerics | `use_qk_l2norm_in_kernel` | `False` | Requests q/k L2 normalization in `_preprocess_inputs` before Pallas execution |
| State/output | `output_final_state` | `False` | Requests final recurrent states; CP execution does not support `True` |
| Sequence | `segment_ids` | `None` | Optional 1-indexed varlen segment IDs shaped `[B,T]`; `0` denotes padding, and CP requires rank-local segment IDs |
| Sequence | `chunk_size` | `64` | Chunk size used by the Pallas path; the delivered implementation supports only `64` |
| Sequence | `N_max` | `None` | Static upper bound on varlen segment count; required when varlen input has no `initial_state`, and always required for CP |
| CP | `cp_context` | `None` | Optional CP mesh and axis metadata; active CP additionally forbids external/final state and requires `K` and `V` to be multiples of 128 |
| Residual policy | `disable_recompute` | `True` | Selects saved-state versus recompute behavior for the custom backward without changing the mathematical forward result |
| Residual policy | `return_residuals` | `False` | Internal bind/backend control that returns `KdaResiduals` for the custom backward; it is not exposed as an additional public KDA result |

`config` is injected by the Tokamax `Op` framework. The current Pallas TPU forward deletes it at entry and does not use it as part of the mathematical or kernel-selection contract.

### 2.2 Supported Functionality

| Capability | Delivered behavior | Call-chain location and applicability |
|---|---|---|
| Fixed-length forward | Supports batches, multiple heads, and optional initial/final state | `T` is partitioned with `chunk_size=64`, represented as one segment per batch element, and passed to `chunk_kda_fwd_h_o_varlen` |
| Variable-length forward | Supports `segment_ids` shaped `[B,T]` | The backend constructs `cu_seqlens`, aligns data, then calls `chunk_kda_fwd_h_o_varlen` |
| Segments not aligned to 64 | Supported | Each segment is padded independently to a multiple of 64, then output is restored to the original length |
| Variable length with `B>1` | Supported | Each batch element independently constructs boundaries, alignment, and chunk mapping |
| Context Parallel forward | Supports rank-local `segment_ids` and a minimal `CPContext` | Rank-chain metadata is derived automatically; delivered CP form uses no external initial state and does not request final state |
| CP with `B>1` | Supported | Each batch element independently merges upstream state summaries |
| Causal semantics | Supported | `Aqk` is lower triangular and `L` is strictly lower triangular |
| Segment boundaries | Supported | Chunks do not cross segments; Stages 3+4 reset state at boundaries |
| Padding semantics | Supported | Data is zero padded; pre-activated gate padding is zero, while raw-gate padding is replaced with `-1e4` before fused activation |
| Raw-gate activation | Supports softplus decay with `safe_gate=False`, or sigmoid decay with `-5 <= lower_bound < 0` | `_preprocess_inputs` prepares padding; `kda_fwd_intra_fused` requires `A_log`, with optional `dt_bias` |
| Pre-activated gate | Supported | `g` represents `ln(alpha)` and is non-positive under production semantics |
| Query/key L2 normalization | Supported | `_preprocess_inputs` normalizes the aligned q/k tensors before Pallas computation and retains `q_rstd/k_rstd` when residuals are requested |
| Initial state | Tokamax backend contract `[B,N,H,K,V]`; fixed length uses `N=1` | Loaded at the first chunk of each sequence |
| Final state | Tokamax backend contract `[B,N,H,K,V]`; supported for non-CP fixed-length and variable-length execution | Written at the final chunk of each sequence |
| bfloat16 | Supported | Fused Stage 1+2 with float32 critical accumulators |
| float32 | Supported | Separate gate cumsum and float32 lower-triangular forward substitution |
| Recompute control | Supported | `return_residuals` and `disable_recompute` jointly select the typed residual set and save-state behavior |
| Static segment bound | Supports `N_max` | Fixes compile-time shapes for `cu_seqlens` and chunk mapping; a tight value reduces empty-segment overhead |

## 3. Architecture and Canonical Call Chain

### 3.1 End-to-End Flow

```text
PallasTpuKimiDeltaAttention._fwd
  |
  |-- Validate dtype, chunk size, and fixed-length divisibility
  |-- _preprocess_inputs
  |     -> derive CP and segment metadata
  |     -> align variable-length tensors exactly once
  |     -> replace aligned raw-gate padding with -1e4
  |     -> prepare chunk indices and aligned segment IDs
  |     -> optionally apply q/k L2 normalization
  v
chunk_kda_fwd_custom
  |
  |-- Stage 1+2: kda_fwd_intra_fused
  |     Gate activation/cumsum + intra-chunk triangular solve
  |     -> g_cumsum, w, u, kg, Aqk, Akk, optional qg
  |
  |-- CP bridge, when CP is active
  |     _prepare_cp_initial_state
  |       -> chunk_gated_delta_rule_fwd_h_pre_process
  |       -> all_gather_into_tensor(S_ext_local, M_local)
  |       -> _merge_initial_state
  |       -> initial_state [B, N, H, K, V], float32
  |
  |-- Stage 3+4
  |     chunk_kda_fwd_h_o_varlen
  |       Variable length: aligned segment boundaries and chunk mapping
  |       Fixed length: one [0, T] segment per batch element
  v
Drop recomputable intermediates or construct KdaResiduals
  |
  |-- Restore variable-length output layout
  |-- Cast output to q.dtype
  |-- Restore the singleton N dimension for a fixed-length final state
  |
  v
o, optional final_state
```

### 3.2 Four Semantic Stages

| Stage | Objective | Primary results |
|---|---|---|
| Stage 1 | Activate the gate and compute the cumulative decay within each chunk | `g_cumsum` |
| Stage 2 | Eliminate intra-chunk delta-rule sequential dependencies | `w`, `u`, `kg`, `Aqk`, `Akk` |
| Stage 3 | Propagate the state along the chunk dimension | Chunk-start states, `v_new`, and an optional final state |
| Stage 4 | Combine the history read with current-chunk contributions | `o` |

These stages describe mathematical dependencies, not four mandatory kernel dispatches. The current bfloat16 production path fuses Stages 1 and 2. Both fixed-length and variable-length execution fuse Stages 3 and 4 in the same kernel. The CP bridge sits between Stages 2 and 3 and reconstructs the correct initial state for the current rank.

### 3.3 Why the Computation Is Staged

1. Every decay ratio in Stage 2 depends on the chunk-local cumulative gate produced by Stage 1.
2. Stage 3 can propagate state only after Stage 2 has compressed intra-chunk updates into the `W/U` representation.
3. Stage 4 depends both on the chunk-start state from Stage 3 and on the intra-chunk relation matrix from Stage 2.
4. CP can form a state summary only after local `W/U` values are available, and it must finish the cross-rank merge before Stage 3 begins.

Keeping these semantic stages makes correctness easier to explain. Fusing adjacent stages reduces round trips to High Bandwidth Memory (HBM).

### 3.4 Primary Functions and Responsibilities

| Function | Primary responsibility | Downstream function |
|---|---|---|
| `KimiDeltaAttention.bind` | Validate the public contract and canonicalize `scale=None` to `K^{-1/2}` | TPU backend hook |
| `PallasTpuKimiDeltaAttention._fwd` | Tokamax TPU backend dispatch and dtype/chunk validation | `_preprocess_inputs`, then `chunk_kda_fwd_custom` |
| `_preprocess_inputs` | One-time CP/varlen metadata construction, alignment, raw-gate padding, optional q/k L2 normalization, and backward metadata preparation | `chunk_kda_fwd_custom` |
| `chunk_kda_fwd_custom` | Unified forward orchestration, optional residual construction, output unalignment, and selection of Stage 1+2, CP, and Stage 3+4 | Stage functions below |
| `kda_fwd_intra_fused` | Stage 1+2 dtype routing and fused execution | Fused bfloat16 path or float32 forward-substitution path |
| `_prepare_cp_initial_state` | Encapsulate the complete CP bridge: build rank-local summaries, execute both all-gathers, merge upstream states per batch element, and materialize `[B,N,H,K,V]` float32 initial states | `chunk_kda_fwd_h_o_varlen` |
| `chunk_gated_delta_rule_fwd_h_pre_process` | Construct the rank-local affine state summary consumed by the CP bridge | `_prepare_cp_initial_state` |
| `_merge_initial_state` | Merge the applicable upstream rank summaries and recover the first local segment's initial state | `_prepare_cp_initial_state` |
| `chunk_kda_fwd_h_o_varlen` | Unified fused Stage 3+4 for fixed-length and variable-length input | `o`, optional state residual, and final state |
| `KdaResiduals` | Typed contract containing the prepared forward values consumed by the custom backward | `PallasTpuKimiDeltaAttentionVjp` |

### 3.5 Preprocessing Boundary and Single-Alignment Contract

When variable-length input is described by `segment_ids` with shape `[B,T]`, the entry path first produces a statically shaped `cu_seqlens`. Each segment is padded to a multiple of `chunk_size`, and a chunk-to-sequence mapping is generated.

Let `C=chunk_size`. For `N` segments with logical lengths `L_n`, define the aligned length of segment `n` as:

$$
\widehat L_n=C\left\lceil\frac{L_n}{C}\right\rceil.
$$

The total per-segment tail padding then satisfies:

$$
0\le\sum_{n=1}^{N}(\widehat L_n-L_n)<N(C-1).
$$

For `B>1`, each batch element is aligned independently by segment and then padded to a common static total length required by JAX compilation. This avoids padding every segment to the longest sequence in the batch.

`_preprocess_inputs` performs this alignment before `chunk_kda_fwd_custom`, regardless of whether residuals are requested. The lower forward therefore always consumes prepared tensors and never performs a second alignment. This guarantees that:

- Forward outputs and returned residual tensors use the same token mapping.
- `Aqk`, `Akk`, cumulative gates, and chunk indices remain consistent.
- The backward receives the same aligned and optionally L2-normalized `q/k` tensors that produced the forward output.
- Output is restored to the original token order and length only at the end of the call chain.

## 4. Physical Kernel Design

The matrix forms in this section describe one batch element, one head, and one chunk; the implementation batches the same operations over `B`, `H`, and the physical chunk grid. Unless stated otherwise:

$$
Q_c,K_c,G_c,W_c\in\mathbb R^{C\times K},
\qquad
V_c,U_c,O_c\in\mathbb R^{C\times V},
\qquad
S_c\in\mathbb R^{K\times V},
$$

$$
A_{qk,c},L_c,A_c\in\mathbb R^{C\times C},
\qquad
D_{\beta,c}=\operatorname{Diag}(\beta_c),
\qquad
\Gamma_c:=2^{G_c},
\qquad
\gamma_c^t:=\Gamma_c[t,:]=2^{G_c^t},
$$

$$
\widetilde Q_c=\mathtt{scale}\,Q_c.
$$

`Gamma_c` is the `C x K` matrix of cumulative decay vectors for every position in the chunk. `gamma_c^C` is its final `K`-element row. Consequently, `Diag(gamma_c^C)` is the `K x K` state-decay matrix used in Paper Eq. (8); it is related to `Gamma_c` but is not the same matrix.

The following table maps the mathematical symbols used by the physical design to the KDA paper and to the delivered runtime tensors.

| Symbol | Shape | Meaning and paper/runtime mapping |
|---|---:|---|
| `c`, `C` | scalar | Chunk index and chunk length. The delivered kernels fix `C=64`. |
| `t`, `s` | scalar | One-based target-row and source-row positions in `{1,...,C}` when matching the paper equations; runtime arrays use the corresponding zero-based indices. |
| `Q_c`, `K_c`, `V_c`, `beta_c` | `C x K`, `C x K`, `C x V`, `C` | Chunk-local query, key, value, and delta-rule coefficient from the paper; physical slices of runtime `q`, `k`, `v`, and `beta`. |
| `ell_c` | `C x K` | Natural-log decay `ln(alpha)` for the chunk. Runtime `g` contains this value when `use_gate_in_kernel=False`; otherwise the kernel derives it from the raw gate. |
| `G_c` | `C x K` | Implementation log2 cumulative decay, `G_c^t=log2(gamma_c^t)`; runtime `g_cumsum`. `G_c^C` is the final paper-position row, stored at runtime index `C-1`. This is the log-domain representation of the paper's cumulative decay, not an additional KDA state. |
| `Gamma_c`, `gamma_c^t` | `C x K`, `K` | Paper cumulative decay matrix and its row at token `t`: `Gamma_c=2^{G_c}`. `gamma_c^C` is the last-row decay that carries the state to the end of the chunk. |
| `Q_tilde_c` | `C x K` | Scaled query `scale Q_c`. It is the implementation form of the paper query when the attention scale is not one. |
| `D_beta,c` | `C x C` | Diagonal matrix formed from `beta_c`; used in the paper's lower-triangular transform in Eq. (6). |
| `K_c^+`, `K_c^-`, `Q_c^+` | `C x K` | Decay-weighted operands `Gamma_c odot K_c`, `K_c oslash Gamma_c`, and `Gamma_c odot Q_tilde_c`. The key operands express Paper Eq. (6)-(7); the query operand constructs the causal read matrix used in Paper Eq. (9). |
| `L_c` | `C x C` | Strictly lower-triangular dependency matrix in Paper Eq. (6). It contains only dependencies on earlier positions in the same chunk. |
| `A_c` | `C x C` | Unit lower-triangular inverse `(I_C+L_c)^{-1}`; stored as runtime `Akk`. The paper's Eq. (6) matrix is `M_c^paper=A_c D_beta,c`, so runtime `Akk` is the inverse factor, not the paper's complete `M_c^paper`. |
| `W_c`, `U_c` | `C x K`, `C x V` | Compact WY factors from Paper Eq. (7); runtime `w` and `u`. |
| `A_qk,c`, `bar K_c` | `C x C`, `C x K` | Causal intra-chunk read matrix and keys decayed to the chunk end, with `bar K_c^t=K_c^t odot (gamma_c^C oslash gamma_c^t)`; runtime `Aqk` and `kg`. They evaluate the output and state terms in Paper Eq. (8)-(9). |
| `S_c`, `V_c^new`, `O_c` | `K x V`, `C x V`, `C x V` | State at the start of chunk `c`, the shared delta-correction intermediate, and the chunk output in Paper Eq. (8)-(9). Runtime scratch stores `S_c`; runtime output `o` stores the concatenated `O_c`. |
| `Phi_c`, `Psi_c` | `K x K`, `K x V` | Affine transition and additive update obtained by expanding Paper Eq. (8): `S_{c+1}=Phi_c S_c+Psi_c`. These names are implementation derivations, not separate paper parameters. |
| `M_local`, `S_ext,local` | `K x K`, `K x V` | Rank-level composition of `Phi_c` and `Psi_c` used by CP. Runtime `M_local` is unrelated to the paper's `M_c^paper`; it represents `Phi_rank`, while `S_ext,local` represents `Psi_rank`. |

The remaining symbols describe the physical reformulation rather than new KDA semantics.

| Symbol | Meaning |
|---|---|
| `R_c` | The `2C x C` stacked relation-product result before its query-key and key-key row groups are sliced and causally masked. It is a temporary GEMM result, not a paper output. |
| `mu`, `G_ref` | Reference vectors in `R^K` used to factor exponent differences in Stage 1+2 relation construction and Stage 3+4 history reads. They change floating-point range, not the represented matrix product. |
| `L_d`, `F`, `P`, `J` | `C x C` Neumann factorization intermediates: block-diagonal part of `L`, cross-block remainder, `P=(I+L_d)^{-1}`, and `J=PF`. The kernel source names the last temporary `G`; this document uses `J` to avoid collision with cumulative-gate `G_c`. |
| `BC`, `BC_inv` | Relation-construction and triangular-inversion sub-block sizes; currently 16 and 8. |
| `N_T` | Number of physical chunks, equal to aligned token length divided by `C`. |
| `F_fwd` | Abstract complete-forward map, written `mathcal F_fwd` in equations. It is used only to state that residual policy does not alter the mathematical output. |
| `j`, `r`, `R` | CP rank index, number of upstream ranks in the current continuation chain, and total CP size. `S^(j)` is the state entering rank `j`. |
| `mathcal S_ext`, `mathcal M` | Stacks of the per-rank `S_ext,j` and `M_j` matrices produced by the two CP all-gathers. |

Matrix operators are used consistently as follows:

| Operator | Meaning |
|---|---|
| `I_d`, `0`, `1_(m x n)` | Identity of order `d`, a context-shaped zero matrix, and an `m x n` all-ones matrix. |
| `Diag(x)` | Matrix with vector `x` on its diagonal. |
| `Tril(X)`, `StrictTril(X)` | Lower-triangular part including or excluding the diagonal. |
| `odot`, `oslash`, `2^X` | Elementwise multiplication, division, and exponentiation with broadcasting where stated. |
| `X^T` | Matrix transpose. |
| `[A;B]`, `[A B]` | Vertical and horizontal matrix concatenation, respectively. |
| `X[a:b,:]` | Rows `a` through `b-1` of `X`. |

Appendix 9.1 provides the full derivation from the token recurrence through Paper Eq. (6)-(9); Section 4 uses the definitions above to describe how those equations are physically tiled, fused, stored, and communicated.

### 4.1 Stage 1+2: Gate Accumulation and Intra-Chunk Solve

#### 4.1.1 Objective, Inputs, and Outputs

`kda_fwd_intra_fused` dispatches Stages 1 and 2. The Tokamax backend contract and the Pallas kernels both use head-first `[H, B, T_a, ...]` tensors, where `T_a` is either the fixed input length or the aligned variable-length input length, and `N_T=T_a/C`.

For each chunk, the Stage 1+2 matrix map is:

$$
(Q_c,K_c,V_c,\ell_c,\beta_c)
\longmapsto
(G_c,L_c,A_{qk,c},A_c,W_c,U_c,\bar K_c),
\qquad A_c=(I+L_c)^{-1}.
$$

Inputs:

| Tensor | Shape | Meaning |
|---|---:|---|
| `q`, `k` | `[H, B, T_a, K]` | Query/key |
| `v` | `[H, B, T_a, V]` | Value |
| `g` | `[H, B, T_a, K]` | Raw gate or `ln(alpha)` |
| `beta` | `[H, B, T_a]` | Write coefficient |
| `A_log` | `[H]` or `None` | Raw-gate activation parameter |
| `dt_bias` | `[H*K]` or `None` | Raw-gate bias |

Outputs:

| Tensor | Shape | Mathematical meaning |
|---|---:|---|
| `g_cumsum` | `[H, B, T_a, K]`, float32 | `G=log2(Gamma)` |
| `w` | `[H, B, T_a, K]` | `W` |
| `u` | `[H, B, T_a, V]` | `U` |
| `kg` | `[H, B, T_a, K]` | `bar K` |
| `Aqk` | `[H, B, T_a, C]` | Flattened storage of `A_qk` |
| `Akk` | `[H, B, T_a, C]` | Flattened storage of `(I+L)^{-1}` |
| `qg` | `[H, B, T_a, K]` or `None` | Optionally materialized `Gamma odot Q` |

#### 4.1.2 Gate Activation and Accumulation

The bfloat16 production path performs chunk-local accumulation inside `_fused_gate_intra_kernel` and also performs gate activation there when `use_gate_in_kernel=True`. It multiplies the activated or pre-activated gate by a fixed lower-triangular matrix of ones to produce the chunk-local prefix sum:

$$
G_c=\operatorname{Tril}(\mathbf 1_{C\times C})
\frac{\ell_c}{\ln 2}.
$$

`G_c` is consumed immediately to construct `L` and `Aqk`, without first being written to and reread from HBM. Because Stages 3+4 and the residual-producing forward path require the cumulative gate, `g_cumsum` is eventually written as a stage output.

#### 4.1.3 Relation Matrices and Numerical Reference Points

The current production path fixes `C=64` and uses relation-construction sub-block size `BC=16` to construct `Aqk` and `L`. Given a reference vector `mu` for a sub-block, the exponent difference is factored as:

$$
2^{G_c^t-G_c^s}=2^{G_c^t-\mu}\,2^{\mu-G_c^s}.
$$

When `safe_gate=True`, the midpoint position of the sub-block is used as the reference, reducing the exponent range on either side. Outside the causal region, the implementation sets exponent differences to zero before evaluating `exp2`, then zeros the corresponding matrix-multiplication inputs. This prevents masked positions from first producing infinity and then forming a non-finite result when multiplied by zero.

The query-key and key-key left operands for a sub-block are concatenated. One batched matrix multiplication then produces both the corresponding rows of `Aqk` and `L`, reusing gated-key loads and matrix-multiplication scheduling.

In full-chunk matrix form, define:

$$
K_c^+=\Gamma_c\odot K_c,
\qquad
K_c^-=K_c\oslash\Gamma_c,
\qquad
Q_c^+=\Gamma_c\odot\widetilde Q_c.
$$

The two relation products can then be represented by one stacked matrix multiplication:

$$
R_c=
\begin{bmatrix}
Q_c^+\\
D_{\beta,c}K_c^+
\end{bmatrix}
(K_c^-)^\top,
\qquad
A_{qk,c}=\operatorname{Tril}(R_c[0:C,:]),
\qquad
L_c=\operatorname{StrictTril}(R_c[C:2C,:]).
$$

The `BC=16` implementation evaluates row blocks of this equation. Reference-point factorization changes the two GEMM operands but preserves their matrix product.

#### 4.1.4 Lower-Triangular Solve

The bfloat16 path uses inversion sub-block size `BC_inv=8` and splits `L` into a block-diagonal strictly lower-triangular part `L_d` and a cross-block part `F`:

$$
L=L_d+F.
$$

Each `8 x 8` diagonal block is strictly lower triangular, so `L_d^8=0` in exact arithmetic and:

$$
P=(I+L_d)^{-1}=\sum_{p=0}^{7}(-L_d)^p.
$$

Now define the transformed cross-block matrix `J`:

$$
J=PF,
\qquad
(I+L)^{-1}=(I+J)^{-1}P.
$$

`J` is also nilpotent in the block lower-triangular structure formed by the eight blocks, so `(I+J)^{-1}` is obtained from another finite Neumann expansion. The implementation organizes these finite series with doubling and replaces row-by-row forward substitution with small matrix multiplications. Runtime variable `G` inside `_fused_gate_intra_kernel` corresponds to this document's `J`; it does not denote `g_cumsum`.

In exact arithmetic, this expansion is an equivalent inverse of a strictly lower-triangular system; it is not a truncation of an infinite series. Differences between the bfloat16 path and row-wise solving come from input quantization and floating-point matrix-multiplication rounding. Float32 input takes the exact forward-substitution path for higher-precision execution and numerical comparison.

The same inverse is applied to the `D_beta V`, `D_beta K+`, and identity-matrix right-hand sides. This produces `U`, `W`, and `Akk` without deriving separate triangular systems.

The combined solve has the matrix form:

$$
\begin{bmatrix}
U_c & W_c & A_c
\end{bmatrix}
=
A_c
\begin{bmatrix}
D_{\beta,c}V_c & D_{\beta,c}K_c^+ & I_C
\end{bmatrix}.
$$

This is the combined mathematical representation, where:

$$
A_c:=(I_C+L_c)^{-1}.
$$

`A_c` is stored as runtime `Akk` and corresponds to the `A_inv`/`Akk_inv_out_ref` values in the implementation. It is an implementation-level inverse factor rather than the complete matrix `M_c` from Paper Eq. (6):

$$
M_c^{paper}=A_cD_{\beta,c}.
$$

Keeping these symbols distinct also avoids confusion with the CP transition matrix `M_local`. The bfloat16 implementation physically concatenates the `V/K` right-hand sides and forms `A_c` through a separate multiplication by `P`; the float32 forward-substitution path concatenates all three right-hand sides.

#### 4.1.5 TPU Benefits

- Fusing Stages 1 and 2 removes one write-then-read round trip of `g_cumsum` through HBM.
- On the bfloat16 path, the finite Neumann expansions and doubling reformulate unit lower-triangular inversion from row-wise forward substitution into batched matrix multiplications, moving the dominant solve work onto the MXU and increasing MXU utilization.
- Relation sub-block size `BC=16` and inversion sub-block size `BC_inv=8` bound peak Vector Memory (VMEM) use.
- One program point can process multiple heads as a mini-batch, amortizing Direct Memory Access (DMA) and dispatch overhead. The mini-batch is selected from the VMEM budget, capped at 16, and reduced as needed to divide `H`.
- Relation construction and solving use float32 accumulators before outputs are converted to the input compute dtype.

### 4.2 CP Bridge: State Merge Between Stages 2 and 3

#### 4.2.1 Objective, Inputs, and Outputs

Under CP, one logical sequence can continue from an upstream rank onto the current rank. Stages 1+2 can operate entirely on rank-local tokens, but Stage 3 requires the true initial state of the first local segment.

`chunk_kda_fwd_custom` delegates this entire branch to `_prepare_cp_initial_state`. The helper accepts:

| Input | Shape or value |
|---|---:|
| `kg`, `w` | `[H, B, T_local_aligned, K]` |
| `u` | `[H, B, T_local_aligned, V]` |
| `gk` (`g_cumsum` at the call site) | `[H, B, T_local_aligned, K]` |
| `cu_seqlens` | `[B,N_local+1]` |
| `chunk_indices` | `[B,N_T,2]` |
| `cp_context` | CP axis name and rank-chain metadata |
| `chunk_size` | Compile-time chunk size `C` |

It returns one tensor:

| Tensor | Shape | Meaning |
|---|---:|---|
| `initial_state` | `[B,N_local,H,K,V]`, float32 | Rank-local segment initial states; only segment slot 0 can contain an upstream state |

The batch dimension is mandatory for both metadata tensors, including when `B=1`. `_prepare_cp_initial_state` rejects unbatched `[N_local+1]` or `[N_T,2]` inputs rather than implicitly sharing one mapping across batch elements. The lower `_pre_process_pallas` launcher may slice the batched metadata to its internal single-batch form after this interface contract has been validated.

As its first step, `_prepare_cp_initial_state` calls `chunk_gated_delta_rule_fwd_h_pre_process`. This lower-level summary function processes only the final real local segment of each batch element and returns:

| Tensor | Shape | Meaning |
|---|---:|---|
| `S_ext_local` | `[H, B, K, V]`, float32 | `Psi_rank` assuming a zero input state |
| `M_local` | `[H, B, K, K]`, float32 | `Phi_rank` |

The affine matrix contract represented by this pair is:

$$
S_{out}^{rank}=M_{local}S_{in}^{rank}+S_{ext,local},
\qquad
M_{local}\in\mathbb R^{K\times K},
\quad
S_{ext,local},S_{in}^{rank},S_{out}^{rank}\in\mathbb R^{K\times V}.
$$

Here `S_in^rank` and `S_out^rank` are the `K x V` states entering and leaving the rank-local token range. The pair `(M_local,S_ext,local)` is the fixed-size representation of that complete local state transform.

#### 4.2.2 Local Summary Computation

Inside `_prepare_cp_initial_state`, `chunk_gated_delta_rule_fwd_h_pre_process` computes the rank-local affine summary. For each chunk in the final local segment, it computes:

$$
\Phi_c=\operatorname{Diag}(\gamma_c^C)-\bar K_c^\top W_c,
$$

$$
\Psi_c=\bar K_c^\top U_c.
$$

Two float32 VMEM scratch buffers accumulate:

$$
S_{ext}\leftarrow\Phi_cS_{ext}+\Psi_c,
$$

$$
M\leftarrow\Phi_cM,
$$

starting from `S_ext=0` and `M=I`. The rank-local summary is written at the final chunk of the segment.

Only the final local segment is compressed because segments that start and end on one rank do not require cross-device state transfer. Only the trailing segment can continue onto the next rank.

#### 4.2.3 Cross-Rank Merge

After local summarization, `_prepare_cp_initial_state` calls `all_gather_into_tensor` separately for `S_ext_local` and `M_local` along the CP mesh axis. Based on the number of preceding ranks that belong to the same logical sequence, it calls `_merge_initial_state` to apply upstream affine summaries in order and reconstruct the input state for the current rank.

When `B>1`, `_prepare_cp_initial_state` performs the merge independently for each batch element because cross-rank boundaries can differ. It concatenates the merged `[H,1,K,V]` results, transposes the result to `[B,H,K,V]`, allocates a zero-initialized `[B,N_local,H,K,V]` float32 tensor, and writes the recovered state to `initial_state[:,0]`. All remaining local segment states stay zero because only the first local segment can continue from an upstream rank. The helper returns this tensor directly to `chunk_kda_fwd_custom` for Stage 3+4.

Communication and merge arithmetic remain in float32 to avoid amplifying state error through low-precision affine products across many ranks.

If ranks `0,...,r-1` form the upstream continuation chain, `_merge_initial_state` evaluates the following matrix recurrence in rank order:

$$
S^{(j+1)}=M_jS^{(j)}+S_{ext,j},
\qquad j=0,\ldots,r-1,
$$

or equivalently:

$$
S^{(r)}=
M_{r-1}\cdots M_1M_0S^{(0)}
+\sum_{j=0}^{r-1}
\left(M_{r-1}\cdots M_{j+1}\right)S_{ext,j}.
$$

The product preceding `S_ext,r-1` is the identity. For the delivered CP contract, the beginning of a new logical sequence uses `S^(0)=0`. The recovered `S^(r)` is written only to `initial_state[:,0]`.

### 4.3 Stage 3+4: State Propagation and Output

#### 4.3.1 Objective, Inputs, and Outputs

Stages 3+4 consume the compressed results of Stages 1+2 and implement Paper Eq. (8)-(9). The primary inputs are:

| Tensor | Shape |
|---|---:|
| `w`, `kg` | `[H, B, T_a, K]` |
| `u` | `[H, B, T_a, V]` |
| `g_cumsum` | `[H, B, T_a, K]` |
| `q` | `[H, B, T_a, K]` |
| `Aqk` | `[H, B, T_a, C]` |
| `initial_state` | `[B,N,H,K,V]` at the fused kernel boundary; `N=1` for fixed-length input |

The primary outputs are:

| Tensor | Shape | Meaning |
|---|---:|---|
| `o` | `[H, B, T_a, V]` | Output in aligned internal layout |
| `h` | `[H, B, N_T, K, V]` or `None` | Optional saved state before each chunk |
| `final_state` | `[B,N,H,K,V]` or `None` at the fused kernel boundary | Final state of each sequence |

At the physical kernel boundary, each chunk applies the matrix map:

$$
(S_c,W_c,U_c,\bar K_c,G_c,Q_c,A_{qk,c})
\longmapsto
(O_c,S_{c+1}).
$$

#### 4.3.2 Per-Chunk Computation

For chunk-start state `S_c`, the computation is ordered as follows:

$$
V_c^{new}=U_c-W_cS_c,
$$

$$
O_c^{inter}=(\Gamma_c\odot\widetilde Q_c)S_c,
$$

$$
O_c^{intra}=A_{qk,c}V_c^{new},
$$

$$
O_c=O_c^{inter}+O_c^{intra},
$$

$$
S_{c+1}=\operatorname{Diag}(\gamma_c^C)S_c
+\bar K_c^\top V_c^{new}.
$$

`O_c^inter` is the read from the state that existed before chunk `c`; `O_c^intra` is the causal contribution from writes within chunk `c`. Their sum is the Paper Eq. (9) output, while the final equation is Paper Eq. (8).

This order lets the same `V_new` feed both the intra-chunk output and the state update. State is initialized from the corresponding initial state at the first chunk of a sequence and is optionally written as the final state after the final chunk.

#### 4.3.3 Unified Fused Path for Fixed and Variable Length

Both fixed-length and variable-length input call `chunk_kda_fwd_h_o_varlen`. The running state remains in a float32 VMEM scratch buffer. Head-group and batch grid dimensions are parallel, while the chunk dimension is declared `arbitrary` to preserve its ordered dependency.

Fixed-length input is represented as one segment `[0,T]` per batch element, using `cu_seqlens` shaped `[B,2]` and a direct chunk mapping. It therefore reuses the unified Stage 3+4 kernel without variable-length alignment or output unalignment.

Variable-length input uses the aligned `cu_seqlens` and `chunk_indices` produced by `_preprocess_inputs` to map every physical chunk to one logical segment. The mapping controls state initialization at a segment start and optional final-state writes at a segment end.

The launcher pads `K` and `V` to the TPU hardware major-block alignment before constructing BlockSpecs and trims outputs, final states, and saved states back to their logical dimensions. Heads are grouped into a VMEM-budgeted mini-batch capped at 16 and adjusted to divide `H`.

Each chunk performs four primary matrix multiplications:

1. `W @ S` to produce `V_new`.
2. Gated query `@ S` to produce the inter-chunk output.
3. `Aqk @ V_new` to produce the intra-chunk output.
4. `kg^T @ V_new` to update the state.

Equivalently, the first two products can be written as one block matrix map, followed by the output and state GEMMs:

$$
\begin{bmatrix}
V_c^{new}\\
O_c^{inter}
\end{bmatrix}
=
\begin{bmatrix}
U_c\\
0
\end{bmatrix}
+
\begin{bmatrix}
-W_c\\
\Gamma_c\odot\widetilde Q_c
\end{bmatrix}S_c,
$$

$$
O_c=O_c^{inter}+A_{qk,c}V_c^{new},
\qquad
S_{c+1}=\operatorname{Diag}(\gamma_c^C)S_c+\bar K_c^\top V_c^{new}.
$$

`V_new` is produced and consumed inside the current program point and is not written to HBM by the forward orchestrator. The running state remains in scratch across chunks.

The inter-chunk output uses the first cumulative-gate row in the chunk, `G_ref=G_c[0,:]`, as a reference point:

$$
(\widetilde Q_c\odot2^{G_c})S_c
=\left(\widetilde Q_c\odot2^{G_c-G_{ref}}\right)
\left(2^{G_{ref}}\odot S_c\right).
$$

Both exponent terms are bounded below by `-126`, treating extremely small decays as stable underflow to zero. State decay is also computed in float32.

#### 4.3.4 Recompute and Caching

Residual materialization depends on both `return_residuals` and `disable_recompute`. The forward defines:

```python
save_for_backward = return_residuals and disable_recompute
```

The current behavior is:

| `return_residuals` | `disable_recompute` | Forward materialization and returned residual behavior |
|---:|---:|---|
| `False` | `False` | No `KdaResiduals`; `qg` and `h` are not stored; recomputable intermediates are dropped after output production. |
| `False` | `True` | No `KdaResiduals`; Stage 1+2 does not request `qg`, but Stage 3+4 currently writes `h` because `store_h=disable_recompute`; `h` is then discarded before return. |
| `True` | `False` | Returns `KdaResiduals` without `h`. For raw-gate activation, `g_org` is saved and `g_cumsum` is omitted for backward recomputation; for a pre-activated gate, `g_cumsum` is retained. |
| `True` | `True` | Returns the save-state residual set including `h`, `g_cumsum`, `Aqk`, and `Akk`. Stage 1+2 also materializes `qg` internally, but `qg` is not a field of `KdaResiduals`. |

`KdaResiduals` retains the prepared, possibly aligned and L2-normalized `q/k`, along with `v`, `beta`, the selected gate representation, `Aqk/Akk`, optional `h`, initial-state data, and the alignment/CP metadata required by backward. `w`, `u`, `kg`, `qg`, and `V_new` are not part of the typed residual contract. `V_new` is never stored because the orchestrator fixes `store_v_new=False`.

CP and non-CP execution use the same residual policy. Neither flag changes `o`, `final_state`, the CP affine-summary equations, or the initial state recovered for the current rank.

In matrix terms, both policies evaluate the same forward map:

$$
\mathcal F_{fwd}(Q,K,V,G,D_\beta,S_0)=(O,S_{N_T}).
$$

They differ only in whether selected matrices from the evaluation of this map are retained in `KdaResiduals` or reconstructed by evaluating the corresponding Stage 1+2 and Stage 3 intermediates during backward.

### 4.4 VMEM and HBM Data Flow

The forward uses two fused physical phases. Stage 1+2 constructs cumulative gates, relation matrices, and solve outputs while their shared intermediates remain in VMEM. Stage 3+4 keeps `V_new` and the running state in VMEM/scratch while producing the output and next state. These phases remain separate because Stage 3+4 has an ordered chunk recurrence and CP may insert collective communication between Stages 2 and 3.

The resulting materialization boundaries are:

```text
Stage 1+2 HBM outputs
  g_cumsum, w, u, kg, Aqk, Akk, optional qg

Optional CP HBM/collective boundary
  _prepare_cp_initial_state
    -> local S_ext, M -> two all-gathers -> merge
    -> initial_state [B, N, H, K, V], float32

Stage 3+4 outputs
  o, optional final_state, optional h
```

The non-CP matrix flow across this boundary is:

$$
(Q_c,K_c,V_c,\ell_c,D_{\beta,c})
\xrightarrow{\text{Stage 1+2}}
\left[
\underbrace{G_c,W_c,U_c,\bar K_c,A_{qk,c}}_{\text{consumed by Stage 3+4}},
\underbrace{A_c}_{\text{side output}}
\right]_{\mathrm{HBM}},
$$

$$
(S_c,G_c,W_c,U_c,\bar K_c,A_{qk,c})
\xrightarrow{\text{Stage 3+4}}
(O_c,S_{c+1}).
$$

The transfer of `g_cumsum/w/u/kg/Aqk` is the primary remaining HBM boundary in forward.

`A_c` is materialized as `Akk` but is not consumed by the forward Stage 3+4 kernel. `qg` is another optional Stage 1+2 output selected by the current save-for-backward wiring, but it is neither consumed by Stage 3+4 nor retained in `KdaResiduals`. Stage 3+4 reconstructs the gated-query factor from `Q_c` and `G_c`. Under CP, only the initial state argument in the second equation changes to the matrix recovered by the affine merge.

### 4.5 Computation and Communication

Non-CP forward performs no cross-device communication. CP executes in the following order:

```text
Rank-local Stage 1+2
  -> _prepare_cp_initial_state
       -> rank-local (S_ext, M)
       -> two all-gathers of float32 summaries
       -> rank-local initial-state merge and [B, N, H, K, V] materialization
  -> rank-local Stage 3+4
```

Communication volume is independent of sequence length. Each batch/head/rank sends only one `K x V` summary and one `K x K` summary. The current implementation focuses on reducing communication volume and does not explicitly overlap communication with computation.

For CP size `R`, the two collectives materialize the stacked matrices:

$$
\mathcal S_{ext}=
\begin{bmatrix}
S_{ext,0}\\ \vdots\\ S_{ext,R-1}
\end{bmatrix}
\in\mathbb R^{R\times K\times V},
\qquad
\mathcal M=
\begin{bmatrix}
M_0\\ \vdots\\ M_{R-1}
\end{bmatrix}
\in\mathbb R^{R\times K\times K}.
$$

The receiving rank performs only the ordered affine matrix composition:

$$
S^{(j+1)}=M_jS^{(j)}+S_{ext,j}.
$$

Neither communicated matrix has a token-length dimension, which is why the collective payload is `O(KV+K^2)` per batch/head/rank rather than `O(T)`.

## 5. Numerical Correctness and Runtime Invariants

### 5.1 Precision and Numerical Stability

- Gate activation, cumulative gates, state, CP summaries, and critical matrix-multiplication accumulators use float32.
- Bfloat16 is the production dtype for fused Stage 1+2. Float32 uses separate cumulative-gate computation and an algebraically exact lower-triangular forward-substitution formulation, subject to floating-point rounding.
- All decays use `exp2` in log2 space, avoiding repeated conversion between natural and binary exponential representations.
- The `safe_gate` reference point reduces exponent ranges in Stage 2 sub-blocks.
- Anti-causal entries are cleared before `exp2`, preventing overflow at positions that will be masked.
- The unified Stage 3+4 path factors `2^G` around a reference point and applies a `-126` lower bound to very small exponents.
- CP summaries and multi-rank merging remain in float32.
- Final `o` is cast back to the query input dtype.

### 5.2 Mask and Causal Semantics

Three layers jointly implement the current forward masking semantics:

1. `Tril` on `Aqk` ensures that an output position reads only the current and earlier positions.
2. `StrictTril` on `L` ensures that a position's delta correction depends only on earlier writes.
3. Segment alignment and state reset isolate logical sequences, while padded positions preserve zero writes and identity decay; raw-gate mode uses `-1e4` padding so fused gate activation approaches that identity behavior.

Causality, segment boundaries, and padding are therefore fixed components of the KDA recurrence rather than three independently supplied arbitrary masks.

### 5.3 Required Runtime Invariants

- Production gate semantics require `alpha in (0,1]`, so pre-activated `ln(alpha)` is non-positive.
- With `use_gate_in_kernel=True`, `A_log` is required and must have shape `[H]`; an optional `dt_bias` must have shape `[H*K]`.
- Raw-gate softplus activation requires `lower_bound=None`. Under the public contract it is valid only with `safe_gate=False`, because `safe_gate=True` requires a finite lower bound.
- A non-`None` `lower_bound` selects the sigmoid gate variant and must satisfy `-5 <= lower_bound < 0`.
- The complete production path uses `chunk_size=64`.
- Fixed-length `T` must be divisible by the chunk size; the call chain automatically aligns each segment for variable-length input.
- State propagation and CP summary paths require `K<=256`.
- Positive `segment_ids` identify valid segments, while `0` denotes padding.
- `N_max` is a compile-time upper bound on segment count. It must be no smaller than the real segment count and should be as tight as practical.
- CP boundaries are derived from rank-local `segment_ids` and the CP mesh axis. Only the first local segment can inherit state from an upstream rank.
- Causal, segment-boundary, and padding semantics must remain consistent with both the lower-triangular system and state-reset logic; changing only one layer is incorrect.

## 6. Tests and Validation

### 6.1 Validation Strategy

Forward correctness is validated against implementations that do not use the Pallas TPU kernel decomposition:

- `tokamax/_src/ops/experimental/kda/pallas_tpu_test.py` compares the Pallas TPU result with the XLA implementation using identical inputs. It compares both `output` and `final_state` when a final state is requested. For CP, the Pallas path is sharded across the context mesh and is compared with the unsharded XLA result for the same global input.
- `tokamax/_src/ops/experimental/kda/api_test.py` compares the public KDA API with a direct token-recurrent reference implementation. This independently checks the chunked formulation against the recurrence in Section 9.1.2 rather than against another use of the same chunk equations.
- The parameterized Pallas TPU backward tests run the same case matrix and compare `dq`, `dk`, `dv`, `dg`, `dbeta`, and optional `dh0` with XLA. Although backward implementation details are outside the scope of this document, these tests validate that the forward residual and recomputation contracts provide the values required by the custom VJP.

### 6.2 Existing Forward Coverage

The Pallas TPU numerical test contains the following forward cases:

| Case | Dtype and shape | Forward behavior covered |
|---|---|---|
| `fixed_t8192` | bfloat16, `B=1`, `T=8192`, `H=16`, `K=V=128` | Fixed-length execution, in-kernel query/key L2 normalization, and final-state output; marked as a long test |
| `varlen` | bfloat16, `B=1`, `T=256`, `H=2`, `K=V=128`, segment lengths `(45,80,20)` | Per-segment alignment and padding, initial/final state, raw-gate activation with `lower_bound=-0.01`, in-kernel query/key L2 normalization, and `disable_recompute=False` |
| `fixed_unaligned_kv` | bfloat16, `B=1`, `T=64`, `H=1`, `K=129`, `V=127` | Non-major-block-aligned key/value dimensions on the non-CP path |
| `cp2` | float32, global `T=128`, `H=2`, `K=V=128`, `cp_size=2` | Two-rank CP execution with one rank-local chunk per device |

The API-level tests additionally cover:

- bfloat16 and float32 output and final-state comparison with the token-recurrent reference;
- variable-length input with `B=2`, padding, raw-gate activation, and in-kernel query/key L2 normalization;
- preservation of final state across padded tokens and zero output at padding positions;
- default omission of `final_state`;
- implementation registration and ordered fallback from Pallas TPU to XLA;
- rejection before Pallas kernel launch for `K>256`, empty kernel-grid dimensions, an invalid fixed-length state dimension, unaligned CP `K/V`, and unsupported CP state or metadata combinations;
- the requirement for `N_max` when variable-length input has no initial state, together with public shape and implementation-name validation.

### 6.3 Numerical Acceptance Criteria

The Pallas TPU comparison first requires matching dtypes and compatible shapes. It compares values with `atol=0.05` and `rtol=0.05`; if this comparison fails, the diagnostic helper also permits a one-ULP bound in the result dtype. It reports maximum absolute and relative differences and the location of the largest normalized mismatch. Both the Pallas result and its XLA reference are synchronized with `jax.block_until_ready` before comparison.

The API-level token-recurrence comparisons use `atol=0.01` and `rtol=0.01`. They also verify the public output contract: `output` has the value tensor shape and query dtype, while a requested `final_state` has the backend state shape and float32 dtype.

A forward case is accepted only when every requested result passes the numerical comparison. `None` is valid only when both implementations omit the same optional result.

### 6.4 Execution Requirements and Current Gaps

`pallas_tpu_test.py` requires a TPU default backend and skips a case when fewer than `cp_size` TPU devices are available. The `fixed_t8192` case is marked `long`. API and fallback tests can exercise the reference and dispatch contracts without a TPU, but they do not replace execution of the Pallas numerical cases.

The test sources establish the validation plan and existing coverage; they do not by themselves establish that an arbitrary current commit has passed. An upstream delivery result must record the tested source commit, JAX and libtpu versions, TPU generation and topology, exact test command, enabled test markers, and pass/fail result.

The current Pallas forward case matrix does not directly cover every capability in Section 2.2. In particular, it has no direct Pallas case for variable-length `B>1`, CP with `B>1`, bfloat16 CP, or an explicit extreme-gate overflow stress input. These are validation gaps rather than unsupported implementation behavior and should be added before treating the corresponding capabilities as fully covered by delivery tests.

## 7. Performance

### 7.1 Historical Performance Measurements and Revalidation

The measurements in this section predate the current adapter/residual refactor that introduced centralized `_preprocess_inputs`, the unified `chunk_kda_fwd_custom` entry, and the typed `KdaResiduals` contract. They are retained as historical optimization evidence, not as current-head delivery acceptance results. The current implementation must be re-benchmarked before these numbers are used for a performance gate or regression comparison.

#### 7.1.1 Environment

| Item | Value |
|------|-------|
| Chips | 2x2x1 (4-chip TPU v7) |
| JAX | 0.10.2 |
| libtpu | 0.0.42.1 |
| TPU device | 8 TpuDevice (4 chips x 2 cores, topology 2x2x1) |
| dtype | bf16 |
| Shape | B=1, T=8192, H=32, K=128, V=128, C=64, N_SEQS=25 (25 variable-length sequences packing T=8192) |
| Status | Historical pre-refactor baseline; current-head revalidation required |

The benchmark labels `1_intra_fused`, `2_fused_h_o`, and `3_e2e_fused` correspond to Stage 1+2, the fused Stage 3+4 output kernel, and the end-to-end fused pipeline respectively. `3_e2e_fused` is a pipeline measurement, not one Pallas kernel or one dispatch: the non-CP path still has distinct Stage 1+2 and Stage 3+4 calls. The historical CP measurement splits the sequence along a four-rank CP mesh.

#### 7.1.2 Non-CP Fused Pipeline (cp off)

| Benchmark target | Median (ms) | Min | P90 |
|--------|------------:|----:|----:|
| `1_intra_fused` | 1.548 | 1.529 | 1.563 |
| `2_fused_h_o` | 1.180 | 1.166 | 1.191 |
| `3_e2e_fused` | 3.944 | 3.922 | 3.972 |

`2_fused_h_o` is the fastest isolated target (1.180 ms), 0.76x of `1_intra_fused` (1.548 ms) and 0.30x of the `3_e2e_fused` pipeline measurement (3.944 ms). The Stage 3+4 kernel consumes the chunk-start state and chunk-local relation matrices and produces the token output while preserving the ordered inter-chunk state dependency. The end-to-end measurement additionally includes Stage 1+2, Stage 3+4, and pipeline-level data movement and orchestration; it must not be interpreted as a single four-stage dispatch.

#### 7.1.3 Context Parallel (cp on, cp_size=4)

This historical benchmark pair used `cp_size=4` with a reported global logical `T=8192` and rank-local length 2048. `baseline_chunk_kda` and `cp_e2e` below form the internally paired scaling comparison. They should not be mixed with the roofline workloads in Sections 7.1.6 and 7.1.7, which use different physical aligned lengths.

| Kernel | Median (ms) | Min | P90 |
|--------|------------:|----:|----:|
| `baseline_chunk_kda` (single-rank baseline) | 3.515 | 3.493 | 3.544 |
| `cp_e2e` (CP path, cp_size=4) | 2.288 | 2.246 | 2.352 |

**CP scaling:**

```text
baseline_chunk_kda     3.515 ms   (cp off, single rank)
cp_e2e                 2.288 ms   (cp on,  cp_size=4, 4 ranks)
speedup = 1.54x (ideal 4x), efficiency = 38.4%
```

#### 7.1.4 Where CP Cost Comes From

The non-CP and CP core computation is identical; the CP path adds `_prepare_cp_initial_state` between Stage 2 and Stage 3. This helper orchestrates the rank-local affine state summary, the cross-rank collection and merge, and initial-state materialization described in Section 4.2.

In this historical profile, the main incremental cost of CP was not the collective communication itself. Consistent with Section 4.5, `all_gather` only exchanges a compressed `K x V` and `K x K` summary per batch/head/rank, independent of sequence length. `_prepare_cp_initial_state` is the orchestration boundary; the larger local cost inside it was the affine-summary preprocessing before communication:

- `chunk_gated_delta_rule_fwd_h_pre_process` must accumulate the running state and state-transition matrices along this rank's chunk sequence to produce the affine summary;
- this step has a scan/reduction nature (it walks the chunk sequence in order and folds every chunk's `W/U` update into the summary), so it cannot be fully parallelized the way a per-chunk matmul can;
- as a result, although CP eliminates the global sequential dependency, this rank-local scan is a residual sequential section that caps the achievable speedup well below the 4x ideal.

#### 7.1.5 Optimization Focus

- **Historical non-CP result:** the end-to-end pipeline measurement (3.944 ms) dominated the reported isolated stage measurements. Candidate levers were reducing repeated full-chunk reads and improving the inter-chunk state chain.
- **Historical CP result:** the rank-local summary scan in `chunk_gated_delta_rule_fwd_h_pre_process` was more significant than the small all-gather. Candidate levers were reducing repeated input reads, improving the affine-summary chain, or fusing part of the merge preparation with an adjacent phase.

These priorities must be confirmed again on the current implementation because preprocessing ownership and residual materialization have changed.

#### 7.1.6 HBM Roofline (Non-CP, single rank)

Historical workload A: TPU **v7**, bf16, varlen 25 sequences, B=1, logical T=8192, physical padded T=9792, H=32, K=V=128, BT=64, NT=153. HBM bandwidth was taken as 3.69 TB/s.

| Kernel | Runtime (us) | HBM (MB) | BW lower bound (us) | BW utilization |
|--------|-------------:|---------:|--------------------:|---------------:|
| `pallas_kda_fwd_intra_fused` | 932.3 | 886.0 | 240.1 | 25.8% |
| `chunk_kda_fwd_h_o_varlen` | 709.7 | 733.9 | 198.9 | 28.0% |
| `copy_bitcast_fusion` | 178.2 | 229.5 | 62.2 | 34.9% |
| `convert_bitcast_fusion` | 77.9 | 229.5 | 62.2 | 79.9% |

HBM volume comes from the LLO dump (the custom-call profiler reports `bytes_accessed=0` for these kernels); for `chunk_kda_fwd_h_o_varlen` the trace `bytes_accessed=769572864` (~734 MB) matches the LLO-derived `733.9 MB`. BW lower bound = HBM volume / peak HBM bandwidth; BW utilization = lower bound / runtime.

#### 7.1.7 HBM Roofline (CP, per-device, cp_size=4)

Historical workload B: TPU **v7x-4** (cp_size=4, 4-device SPMD), bf16, varlen 25 segments already aligned to BT, B=1, physical global T=8192, global NT=128, H=32, K=V=128, BT=64; sharded along the CP axis, per-rank T=2048 and NT=32. HBM bandwidth was taken as 3.69 TB/s. Per-device values are shown below. Because workload A has physical T=9792 and workload B has physical T=8192, the two roofline tables are not an apples-to-apples CP scaling comparison.

| Kernel | Runtime (us) | HBM (MB) | BW lower bound (us) | BW utilization |
|--------|-------------:|---------:|--------------------:|---------------:|
| `pallas_kda_fwd_intra_fused` | 354.1 | 221.5 | 60.0 | 16.9% |
| `chunk_kda_fwd_h_o_varlen` | 263.2 | 183.5 | 49.7 | 18.9% |
| CP `all-gather` | 45.4 | 10.0 | 2.7 | 6.0% |

The two workloads execute the same core kernel families, but their different physical aligned lengths prevent a direct scaling conclusion. The numerical ratios between the historical table entries are 2.63x for Stage 1+2 and 2.70x for Stage 3+4, not 4x. The CP `all-gather` exchanges only the compressed `S_ext` and `M` summaries; its low measured bandwidth utilization is consistent with a small-message, latency/collective-overhead-bound operation. The ordered scan cost belongs to the separate local preprocessing kernel, not to the all-gather itself.

#### 7.1.8 Current-Head Revalidation Requirements

A replacement benchmark must record all of the following:

- Source commit, JAX/libtpu versions, TPU generation/topology, and benchmark command.
- Logical token length, aligned physical length, segment-length distribution, `N_max`, and per-rank shapes.
- `return_residuals`, `disable_recompute`, gate mode, L2-normalization mode, initial/final-state settings, and CP size.
- Warmup count, measured sample count, synchronization method, and whether compilation, preprocessing, alignment, and output unalignment are included.
- A paired single-rank and CP workload with identical logical inputs and identical physical alignment when reporting CP speedup.
- Kernel-level runtime/HBM measurements separately from the end-to-end pipeline latency.

## 8. Limitations and Trade-Offs

### 8.1 Trade-Offs in the Current Decomposition

A fully token-recurrent implementation has the most direct mathematical form but low temporal parallelism. Fully expanding the entire sequence would forfeit the linear-complexity advantage of fixed-size state. The current design takes an intermediate approach:

- Within a chunk, the compact WY representation and lower-triangular system convert recurrence into matrix computation.
- Across chunks, state recurrence is retained.
- Adjacent stages are fused where doing so does not break the CP boundary or the forward residual-output contract.
- Variable-length input is aligned to obtain regular tiles.
- CP uses affine summaries instead of transferring token-level state.

## 9. Appendices

### 9.1 Mathematical Formulation

#### 9.1.1 Symbols and Dimensions

The following derivation considers one batch element and one attention head, omitting batch and head indices.

| Symbol | Dimension | Meaning |
|---|---:|---|
| `L` | Scalar | Length of one sequence |
| `C` | Scalar | Chunk size; 64 on the current production path |
| `N_C` | Scalar | Number of chunks: `L/C`, or aligned length divided by `C` |
| `d_k` | Scalar | Query/key dimension, corresponding to backend `K` |
| `d_v` | Scalar | Value dimension, corresponding to backend `V` |
| `q_t, k_t` | `R^{d_k}` | Query/key column vectors at token `t` |
| `v_t, o_t` | `R^{d_v}` | Value/output column vectors at token `t` |
| `alpha_t` | `(0,1]^{d_k}` | Per-key-channel decay factors |
| `beta_t` | `R` | Delta-rule write coefficient |
| `S_t` | `R^{d_k x d_v}` | State after token `t` has been processed |
| `Q_c, K_c` | `R^{C x d_k}` | Stacked query/key matrices for chunk `c` |
| `V_c, O_c` | `R^{C x d_v}` | Stacked value/output matrices for chunk `c` |
| `S_c^0` | `R^{d_k x d_v}` | State at the start of chunk `c` |

`Diag(x)` denotes a matrix with vector `x` on its diagonal. `Tril(X)` and `StrictTril(X)` retain the lower-triangular and strictly lower-triangular parts, respectively. `odot` denotes elementwise multiplication with broadcasting, and `oslash` denotes elementwise division.

#### 9.1.2 Token-Level KDA Recurrence: Paper Eq. (1)

The KDA recurrence in the paper is:

$$
S_t = \left(I-\beta_t k_t k_t^\top\right)
      \operatorname{Diag}(\alpha_t)S_{t-1}
      + \beta_t k_t v_t^\top,
\qquad
o_t = S_t^\top q_t.
$$

To represent the output scale used by the implementation, let `s` be the attention scale and define:

$$
\widetilde q_t = s q_t.
$$

The implementation therefore computes `S_t^T \widetilde q_t`. This is identical to the paper when `s=1`.

The token-level state update can also be written in delta-correction form:

$$
\bar S_t = \operatorname{Diag}(\alpha_t)S_{t-1},
$$

$$
e_t = v_t - \bar S_t^\top k_t,
$$

$$
S_t = \bar S_t + \beta_t k_t e_t^\top.
$$

`e_t` is the reconstruction error of the current state's mapping from `k_t` to `v_t`. The state first decays channel by channel and is then corrected through a rank-1 update. The gate controls memory lifetime, while the delta rule reduces interference between old and new associations.

#### 9.1.3 Cumulative Decay Within a Chunk

For local position `r in [1,C]` in chunk `c`, define the cumulative decay used by the paper:

$$
\gamma_c^{i\rightarrow j}
= \prod_{m=i}^{j}\alpha_c^m,
\qquad
\gamma_c^r = \gamma_c^{1\rightarrow r}.
$$

The matrix formulation more often uses the ratio between two cumulative decays:

$$
\rho_c^{i\rightarrow r}
= \gamma_c^r \oslash \gamma_c^i
= \prod_{m=i+1}^{r}\alpha_c^m,
\qquad i \le r.
$$

This ratio is the per-channel decay experienced by a write at position `i` before position `r` is reached. Define:

$$
\Gamma_c =
\begin{bmatrix}
(\gamma_c^1)^\top\\
\vdots\\
(\gamma_c^C)^\top
\end{bmatrix}
\in \mathbb{R}^{C\times d_k}.
$$

The implementation does not multiply `alpha` values directly. Instead, it defines `ell_t = ln(alpha_t)` and computes the following value independently in each chunk:

$$
G_c^r = \log_2\gamma_c^r
= \frac{1}{\ln 2}\sum_{m=1}^{r}\ell_c^m.
$$

Consequently:

$$
\gamma_c^r = 2^{G_c^r},
\qquad
\rho_c^{i\rightarrow r}=2^{G_c^r-G_c^i}.
$$

When `use_gate_in_kernel=True`, the input is a raw gate. Let `x_t = g_t^{raw} + dt_bias` and `lambda_h = exp(A_log_h)`. The implementation supports two parameterizations of `ell_t`:

$$
\ell_t = -\lambda_h\,\operatorname{softplus}(x_t),
$$

or, when a negative `lower_bound = a` is provided:

$$
\ell_t = a\,\operatorname{sigmoid}(\lambda_h x_t).
$$

When `use_gate_in_kernel=False`, input `g` directly represents `ell=ln(alpha)`. Production semantics require this value to be non-positive so that `alpha` lies in `(0,1]`.

#### 9.1.4 Chunk Expansion and Compact WY Representation: Paper Eq. (2)-(5)

Define the token-level state transition matrix:

$$
T_c^r =
\left(I-\beta_c^r k_c^r(k_c^r)^\top\right)
\operatorname{Diag}(\alpha_c^r).
$$

Partially expanding the token-level recurrence within one chunk gives Paper Eq. (2):

$$
S_c^r =
\underbrace{\left(\prod_{m=1}^{r}T_c^m\right)}_{P_c^r} S_c^0
+
\underbrace{\sum_{i=1}^{r}
\left(\prod_{m=i+1}^{r}T_c^m\right)
\beta_c^i k_c^i(v_c^i)^\top}_{H_c^r}.
$$

Matrix products act from right to left in temporal order, so later transitions appear on the left. The compact WY representation compresses a sequence of rank-1 transformations into a small set of matrix factors. For KDA, it can be written as:

$$
P_c^r = \operatorname{Diag}(\gamma_c^r)
-\sum_{i=1}^{r}
\operatorname{Diag}(\rho_c^{i\rightarrow r})
k_c^i(w_c^i)^\top,
$$

$$
H_c^r = \sum_{i=1}^{r}
\operatorname{Diag}(\rho_c^{i\rightarrow r})
k_c^i(u_c^i)^\top.
$$

The auxiliary vectors follow Paper Eq. (4)-(5):

$$
w_c^r = \beta_c^r\left[
\operatorname{Diag}(\gamma_c^r)k_c^r
-\sum_{i=1}^{r-1}w_c^i
\left((k_c^i)^\top
\operatorname{Diag}(\rho_c^{i\rightarrow r})k_c^r\right)
\right],
$$

$$
u_c^r = \beta_c^r\left[
v_c^r
-\sum_{i=1}^{r-1}u_c^i
\left((k_c^i)^\top
\operatorname{Diag}(\rho_c^{i\rightarrow r})k_c^r\right)
\right].
$$

These equations still contain a recurrence over positions. The lower-triangular matrix transform in Paper Eq. (6) rewrites them as a linear system suitable for matrix units.

#### 9.1.5 Lower-Triangular Transform and Stage 2 Outputs: Paper Eq. (6)-(7)

For one chunk, define:

$$
K_c^+ = \Gamma_c\odot K_c,
\qquad
K_c^- = K_c\oslash\Gamma_c,
\qquad
D_{\beta,c}=\operatorname{Diag}(\beta_c).
$$

The strictly lower-triangular dependency matrix is:

$$
L_c = \operatorname{StrictTril}
\left(D_{\beta,c}K_c^+(K_c^-)^\top\right)
\in \mathbb{R}^{C\times C}.
$$

Paper Eq. (6) defines:

$$
M_c = (I+L_c)^{-1}D_{\beta,c}.
$$

Paper Eq. (7) gives:

$$
W_c = M_cK_c^+ \in \mathbb{R}^{C\times d_k},
\qquad
U_c = M_cV_c \in \mathbb{R}^{C\times d_v}.
$$

The implementation first computes and stores:

$$
A_c=(I+L_c)^{-1},
$$

then evaluates the equivalent expressions:

$$
W_c=A_c(D_{\beta,c}K_c^+),
\qquad
U_c=A_c(D_{\beta,c}V_c).
$$

The runtime tensor `Akk` denotes `A_c`; it is not the unsolved key-key relation matrix. Runtime tensors `w` and `u` denote `W_c` and `U_c`, respectively.

Stage 2 also constructs the scaled causal query-key matrix:

$$
A_{qk,c}=
\operatorname{Tril}\left[
(\Gamma_c\odot \widetilde Q_c)(K_c^-)^\top
\right]
\in \mathbb{R}^{C\times C},
$$

and the gated key required to propagate writes to the end of the chunk:

$$
\bar K_c^r = \rho_c^{r\rightarrow C}\odot k_c^r
= k_c^r\odot 2^{G_c^C-G_c^r}.
$$

Runtime tensor `Aqk` denotes `A_{qk,c}`, while `kg` denotes the stacked `bar K_c`. `Aqk` already includes the attention scale and the causal lower-triangular constraint.

#### 9.1.6 Chunk State Update and Output: Paper Eq. (8)-(9)

Given chunk-start state `S_c^0`, first compute:

$$
V_c^{new}=U_c-W_cS_c^0
\in \mathbb{R}^{C\times d_v}.
$$

Paper Eq. (8) gives the final state of the chunk:

$$
S_{c+1}^0
= \operatorname{Diag}(\gamma_c^C)S_c^0
+\bar K_c^\top V_c^{new}.
$$

Paper Eq. (9) gives all outputs in the chunk:

$$
O_c
= (\Gamma_c\odot\widetilde Q_c)S_c^0
+A_{qk,c}V_c^{new}.
$$

The first term reads history that existed before the current chunk and is the inter-chunk contribution. The second term contains only writes that have already occurred within the current chunk and is the intra-chunk contribution. The lower-triangular structure of `Aqk` prevents position `r` from reading a future position `i>r`.

#### 9.1.7 Variable-Length Sequences

Let the original interval of segment `n` in a batch be:

$$
[p_n,p_{n+1}),
\qquad
L_n=p_{n+1}-p_n.
$$

Each segment independently executes the recurrence above:

$$
S_{n,0}=S_{n,init},
$$

and state never propagates between different segments. The implementation extends the execution length of each segment to:

$$
\widehat L_n=C\left\lceil\frac{L_n}{C}\right\rceil.
$$

Query, key, value, and beta are zero at padded positions. For a pre-activated gate, zero padding directly represents the identity decay `alpha=1`. When gate activation is performed in the fused kernel, the aligned raw-gate padding is replaced with `-1e4`; after either the softplus or lower-bound sigmoid activation, this yields an effectively zero log decay rather than the nonzero decay that raw zero padding would produce. Padded positions therefore produce no output and preserve the carried state. After execution, outputs are mapped back to the original intervals `[p_n,p_{n+1})`.

#### 9.1.8 Affine State Summaries for Context Parallelism

Expanding Eq. (8) as an affine function of the chunk-start state gives:

$$
S_{c+1}^0=\Phi_c S_c^0+\Psi_c,
$$

where:

$$
\Phi_c=\operatorname{Diag}(\gamma_c^C)-\bar K_c^\top W_c
\in\mathbb{R}^{d_k\times d_k},
$$

$$
\Psi_c=\bar K_c^\top U_c
\in\mathbb{R}^{d_k\times d_v}.
$$

The composition of two consecutive summaries `a` and `b` remains affine:

$$
\Phi_{b\circ a}=\Phi_b\Phi_a,
\qquad
\Psi_{b\circ a}=\Phi_b\Psi_a+\Psi_b.
$$

Any number of consecutive chunks on one rank can therefore be compressed into fixed-size `(Phi_rank, Psi_rank)` tensors. Runtime CP summary `M` corresponds to `Phi_rank`, and `S_ext` corresponds to `Psi_rank`. This runtime `M` is distinct from the lower-triangular transform matrix `M_c` in Paper Eq. (6).
