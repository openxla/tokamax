# Code style

This guide covers coding conventions for Tokamax operators and kernels. It
starts with general conventions for file organization, configuration, error
handling, comments, naming, type annotations, and testing, followed by
conventions specific to Pallas and Mosaic TPU kernels.

--------------------------------------------------------------------------------

## 1. File organization

### The four-file layout

Standard operators are split across four files, each with a distinct
responsibility:

| File                           | Contains                   | Should not contain  |
| ------------------------------ | -------------------------- | ------------------- |
| `api.py`                       | Public function,           | Tuning parameters   |
:                                : `jaxtyping` annotations,   : or `config`         :
:                                : and backend fallback chain : arguments           :
| `base.py`                      | Base `Op` subclass,        | Backend-specific    |
:                                : `bind()` validation,       : code                :
:                                : plain-XLA `_fwd` reference :                     :
:                                : implementation, and        :                     :
:                                : optionally module-level    :                     :
:                                : `vjp`                      :                     :
| `<backend>.py` (e.g.,          | Backend `Op` subclass,     | Kernel bodies or    |
: `pallas_mosaic_tpu.py`)        : `Config`,                  : launch/pipeline     :
:                                : `_get_heuristics_config`,  : functions (e.g.,    :
:                                : `_get_autotuning_configs`, : `pl.pallas_call`,   :
:                                : and `supported_on`         : `pl.emit_pipeline`) :
| `<backend>_kernel.py` (e.g.,   | Kernel body and            | Heuristics or       |
: `pallas_mosaic_tpu_kernel.py`) : launch/pipeline functions  : policy decisions    :
:                                : (e.g., `pl.pallas_call`,   : about which config  :
:                                : `pl.emit_pipeline`,        : to select           :
:                                : `index_map`)               :                     :

Keep the code that chooses block sizes separate from the kernel code that uses
them. Block-size heuristics often involve lengthy VMEM calculations that obscure
the kernel logic if mixed into the same file. Aim to split files once they grow
past roughly 250 lines.

Always name the heuristics method `_get_heuristics_config` (not
`_default_config` or `_heuristic`).

### Splitting larger operators into additional modules

When an operator outgrows the four-file layout, split it further based on what
kind of data each module operates on (`jax.Array`s, `Ref`s, DMAs, or integer
tiling math). Keeping functions that take `Ref` objects in separate files from
functions that operate on plain arrays makes the code much easier to test: a
function that only takes `jax.Array`s can be called directly from XLA, from unit
tests, and from inside a kernel without setting up a `pallas_call`, grid, or
accelerator device.

A typical module layout for a large fused operator looks like this:

| Module          | Operates on                 | Benefit                      |
| --------------- | --------------------------- | ---------------------------- |
| `compute.py`    | `jax.Array`s only (no       | Directly unit-testable in    |
:                 : `Ref`s)                     : plain JAX against the        :
:                 :                             : `base.py` reference          :
| `load_store.py` | `Ref`s in, `jax.Array`s out | Isolates `ref[...]` reads    |
:                 :                             : and writes to a single       :
:                 :                             : module                       :
| `dma.py`        | `make_async_copy`,          | Isolates asynchronous copy   |
:                 : semaphores, memory spaces   : and synchronization logic    :
| `tiling.py`     | Plain integers (block and   | Testable on CPU without an   |
:                 : grid extents)               : accelerator                  :
| `config.py`     | `Config` dataclass          | Keeps configuration separate |
:                 :                             : from implementation          :
| `wrapper.py`    | Assembles the `pallas_call` | Entry point for the kernel   |
:                 : from the modules above      :                              :

This separation can be applied either across files or within a single file:

-   **Across files (splash attention):** `splash_attention_mask.py` implements
    mask construction using only `numpy` (no `jax` or `pallas` imports), making
    it testable with `pytest` on CPU. `splash_attention_mask_info.py` converts
    those host objects into JAX arrays, and the kernel file consumes the
    resulting refs.
-   **Within a single file (ragged dot):** In
    `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py`, helper functions
    that operate on plain arrays (`make_group_metadata`, `_get_store_mask`,
    `_quantize_as`, `_scale_out_by_scale`) are placed at the top of the file
    with `jax.Array` type annotations. Kernel functions that operate on refs
    (`gmm` and `tgmm`) are placed at the bottom.

Treat this section as less of a strict rule than the larger 4-file rule. If a
kernel needs to call out to one or more helper functions in different files,
that is fine. You can choose to create a `kernels` subdirectory that holds all
these files, and leave only the top-level four files in the main op directory.

--------------------------------------------------------------------------------

## 2. Config classes

```python
@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  tile_m: pydantic.PositiveInt = 128
  tile_n: pydantic.PositiveInt = 512
```

Use `@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)`
rather than the standard library's `dataclasses`. Configs are serialized to and
read from the autotuning cache, and `pydantic` handles serialization and
validation automatically.

Validate config fields in two places:

-   **Single-field constraints belong in the type annotation**, using types such
    as `pydantic.PositiveInt` or `Annotated[int, ...]`.
-   **Cross-field invariants belong in `__post_init__` and should raise
    `ValueError`** (for example, checking that `tile_k` divides `K`, which
    cannot be expressed on a single field).

--------------------------------------------------------------------------------

## 3. Raising `NotImplementedError` for unsupported features

When an operator is called using the high-level API wrapper, the backend
fallback chain can catch `NotImplementedError` to fall back to the next
available backend or the plain-XLA reference implementation. Other exception
types are not caught and will surface directly to the caller.

```python
# ❌ Raises ValueError, bypassing the fallback chain and crashing
if bias is not None:
  raise ValueError("Mosaic TPU backend does not support bias.")

# ✅ Raises NotImplementedError, allowing fallback to another backend
if bias is not None:
  raise NotImplementedError("Mosaic TPU backend does not support bias.")
```

Use `ValueError` when the caller passes arguments that are invalid for the
operator itself. Use `NotImplementedError` when the arguments are valid, but a
specific backend does not support them.

--------------------------------------------------------------------------------

## 4. Comments and docstrings

-   **Start every file with a one-line module docstring** stating the file's
    purpose. Any dimension glossary goes underneath the docstring as a regular
    comment block, not inside the docstring.
-   **Do not add docstrings to short nested closures** inside a kernel body
    (such as `def _():` blocks under `@pl.when`).
-   **Explain *why* a decision or tradeoff was made, rather than restating what
    the code does.** Kernel bodies generally need about three times the comment
    density of surrounding API code because hardware constraints, alignment
    rules, and precision tradeoffs are not obvious from the operations alone.

```python
# ✅ Explains the hardware or compiler constraint behind the code
# Every conditional is an optimization barrier, so the sixth (no-op) case is
# spelled out rather than folded into a guard.
# Zero-stride load: broadcasts across sublanes without a copy.
# bf16 inputs: DEFAULT precision is lossless here because operands already have ~7-bit mantissas.
# f32 accumulator: the D reduction has length 4096, where bf16 accumulation would drift.

# ❌ Restates the code
# Load x.
x = x_ref[...]
# Increment the accumulator.
acc_scratch[...] += y
```

--------------------------------------------------------------------------------

## 5. Naming dimensions

Place a `# Dimensions.` glossary comment at the top of each kernel or operator
file. Defining the dimension names once at the top makes shape annotations and
trailing shape comments easy to read without scrolling through the file.

```python
"""Fused RMSNorm + output projection kernel for Mosaic TPU."""

# Dimensions.
#   B      batch
#   T      tokens per sequence
#   D      model dim, the contraction extent
#   N      output features
#   td, tn tile extents of D and N seen by the kernel body
#   nd, nn grid extents: nd = D // td, nn = N // tn
```

Follow a consistent naming scheme to link full extents, tile extents, and grid
extents:

Form                   | Meaning                              | Example
---------------------- | ------------------------------------ | --------------
`UPPERCASE`            | Full logical extent                  | `D`, `N`, `T`
`t` + lowercase letter | **Tile** extent of that dimension    | `td`, `tn`
`n` + lowercase letter | **Grid** extent along that dimension | `nd = D // td`

Using a consistent letter for each dimension makes it immediately clear that
`tn` tiles `N` and that there are `nn` tiles across the grid.

```python
# ❌ Unrelated names for the same dimension
block_k = cfg.tile_k
num_k_blocks = contraction // block_k
K_DIM = x.shape[-1]
kk = pl.program_id(2)

# ✅ Consistent naming derived from D
td = cfg.tile_d          # tile extent of D
nd = D // td             # grid extent over D
d_i = pl.program_id(2)   # or lax.axis_index("d"): index of the D tile handled by this step
```

**Give padded dimensions their own name.** Always use a distinct name such as
`T_pad` rather than relying on context to distinguish padded and unpadded
lengths, as mixing them up is a common source of bugs.

--------------------------------------------------------------------------------

## 6. Naming variables

### When to use single-character names

Use single-character variable names only in four cases:

-   **Loop indices:** `i`, `j`.
-   **Math tensors whose letters appear in the function's shape annotations:**
    for example, `q`, `k`, `v` in a function annotated with `Float[Array, "*B T
    H D"]`.
-   **Uppercase static extents from the dimension glossary:** `D`, `N`, `T`.
-   **Gradient variables:** `d<x>` for the gradient with respect to `x` (such as
    `dq`, `dkv`, `dh`).

**Use descriptive names for computed values and grid indices.** Even though grid
program IDs (from `pl.program_id` or `lax.axis_index`) act as indices, give them
descriptive names such as `k_i`, `n_i`, `n_start`, `grid_id`, `block_id`, or
`core_id` so they do not collide with dimension names:

```python
# ❌ `k` collides with the contraction extent
k = pl.program_id(2)        # or lax.axis_index("k")
n = pl.program_id(1) * tn   # or lax.axis_index("n") * tn

# ✅ Distinct names for tile index and offset
k_i = pl.program_id(2)      # or lax.axis_index("k"): current K tile index
n_start = pl.program_id(1) * tn
```

### Unpacking shapes by name

Avoid indexing `.shape` with bare integers in the middle of an expression
(`x.shape[2]`), which forces the reader to count axes. Unpack the shape into
named dimensions at the top of the function instead:

```python
# ❌
head_bytes = x.shape[2] * x.shape[3] * 2

# ✅
_, _, H, D = x.shape
head_bytes = H * D * jnp.dtype(x.dtype).itemsize
```

--------------------------------------------------------------------------------

## 7. Type annotations

Type annotations follow different conventions at the public API boundary and
inside Pallas kernels.

**On the public wrapper and `_fwd`, annotate tensors with `jaxtyping` using the
dimension letters from the glossary.** This documents the expected shapes and
catches caller mistakes with clear error messages.

```python
@jaxtyping.jaxtyped
@override
def _fwd(
    self,
    x: Float[Array, "*B T D"],
    w: Float[Array, "D N"],
    scale: Float[Array, "#D N"],
) -> Float[Array, "*B T N"]:
```

When using the following `jaxtyping` conventions, document them in the file
where they appear:

-   `*B`: Zero or more leading batch dimensions.
-   **Lowercase letter (`Float[Array, "T t"]`):** An independent extent along
    the same semantic axis (for example, query sequence length `T` and key
    sequence length `t`).
-   **`#` prefix (`#D`):** Broadcastable along that axis elsewhere in the
    signature.

**Inside a kernel body, leave `Ref` parameters unannotated and document their
shapes in trailing comments (§11).** Annotating a `Ref` as `jax.Array` is
incorrect, and annotating it as `jax.Ref` adds no information beyond the `_ref`
suffix because `jax.Ref` does not capture the tile shape or memory space.

```python
# ❌ Incorrect: a Ref is not a jax.Array
def kernel(x_ref: jax.Array, out_ref: jax.Array): ...

# ❌ Redundant: repeats what the `_ref` suffix already indicates
def kernel(x_ref: jax.Ref, out_ref: jax.Ref): ...

# ✅ Leave Refs unannotated and use trailing shape comments
def kernel(
    # Inputs.
    x_ref,        # [tt, td] bf16
    # Outputs.
    out_ref,      # [tt, tn] bf16
): ...
```

If a kernel's parameter list grows past roughly eight refs, group related refs
into a dataclass instead of continuing to add positional parameters.

--------------------------------------------------------------------------------

## 8. Tests

-   **Test against the plain-XLA `_fwd` in `base.py`.** Validate kernels against
    the reference JAX implementation in the same repository rather than against
    golden files, which are harder to maintain and can lock in bugs from earlier
    implementations.
-   **Keep the `dtype`-to-tolerance mapping at the test call site**, rather than
    inside a helper function, so it is immediately clear what tolerance a
    failing test used.
-   **Check the reference implementation against a `float64` oracle before
    loosening tolerances.** When a test fails a numerical tolerance check,
    compare against `float64` first to determine whether the kernel or the
    reference implementation is inaccurate. Note that CPU interpret mode cannot
    reproduce hardware-specific precision behavior such as the TPU MXU, so a
    passing CPU test does not rule out precision issues on device.
-   **Pin `precision=jax.lax.Precision.HIGHEST` on every `f32` contraction in
    the `_fwd` reference.** On TPU, unset or `DEFAULT` precision lowers matrix
    multiplications to the MXU with `bf16` operands. Without
    `Precision.HIGHEST`, an `f32` reference actually computes in `bf16`, which
    can make a kernel that accumulates in `f32` on the VPU *more* accurate than
    the reference it is tested against.
-   **Skip unsupported hardware at runtime, not with a decorator.**
    `@unittest.skipIf` evaluates at import time before the device is
    initialized. Calling `self.skipTest(...)` inside the test body allows the
    test to inspect the active device (for example, via `pltpu.get_tpu_info()`)
    and give a clear reason for skipping.

--------------------------------------------------------------------------------

## 9. Explicit shapes and memory spaces in Pallas kernels

Standard Python and JAX functions rely on two things that Pallas kernels do not
provide:

1.  **Informative parameter types:** Every kernel parameter is an untyped `Ref`.
    A signature of eight refs is eight positional arguments that no type checker
    can distinguish if two are swapped.
2.  **Local shape visibility:** The shape seen inside a kernel body is
    determined outside the kernel by a `BlockSpec` and an `index_map` in the
    launcher (often in another function or file). Looking at `lhs_ref[...]`
    alone does not show whether it holds a 128-row tile in VMEM or the entire
    tensor in HBM.

Because of this, the kernel conventions below focus on keeping parameter roles,
shapes, dtypes, and memory spaces visible directly in the kernel code.

--------------------------------------------------------------------------------

## 10. Sectioning the kernel signature

Group the parameters of every kernel (even very short ones) under `# Inputs.`,
`# Outputs.`, and `# Scratch.` comments:

```python
def kernel(
    # Inputs.
    x_hbm_ref,        # [B, T, D] bf16   full array, pl.ANY
    indices_ref,      # [nt] i32         SMEM
    # Outputs.
    out_hbm_ref,      # [B, T, N] bf16
    # Scratch.
    acc_vmem_ref,     # [tt, tn] f32
    dma_sem,
):
```

A Pallas kernel signature is a flat positional list whose ordering must match
the `in_specs`, `out_specs`, and `scratch_shapes` passed to `pallas_call`.
Explicitly marking the input, output, and scratch sections makes the boundary
between them visible and helps prevent misordered arguments from writing into
the wrong buffer.

--------------------------------------------------------------------------------

## 11. Trailing shape comments on `Ref` parameters

**Every `Ref` parameter in a kernel signature must have a trailing `# [shape]
dtype` comment.** This serves in place of the type annotation omitted in §7.

```python
def kernel(
    # Inputs.
    x_ref,            # [tt, td] bf16
    w_ref,            # [td, tn] bf16
    scale_ref,        # [1, tn] f32
    # Outputs.
    out_ref,          # [tt, tn] bf16
    # Scratch.
    acc_scratch,      # [tt, tn] f32   persists across the D grid axis
    dma_sem,
):
```

Follow four rules for what goes inside the brackets:

-   **Use the letters from the dimension glossary, never hardcoded numbers.** `#
    [tt, td]` shows which dimensions the tile represents; `# [256, 512]` only
    records the tile size for one config and goes stale when the config changes.
    The only exception is a fixed hardware extent (such as `# [8, 128]` for a
    native TPU tile), in which case the comment should note why it is fixed.
-   **Indicate whether the shape is a single block or the full array.** A ref
    mapped through a `BlockSpec` has the block's shape, whereas a ref passed
    with `pl.ANY` has the full array's shape. Confusing the two can cause a DMA
    to copy the entire tensor instead of one tile.

```python
    x_hbm_ref,        # [B, T, D] f32     full array, pl.ANY
    x_vmem_ref,       # [1, tt, td] f32   one block, DMA'd from x_hbm_ref
```

-   **Include the dtype whenever it is non-obvious, and always on scratch
    buffers.** Using an `f32` accumulator with `bf16` inputs is a deliberate
    numerical choice that should be visible in the signature.
-   **Note any permutations, padding, or aliasing applied by `index_map`** that
    cannot be seen inside the kernel body:

```python
    w_ref,            # [td, tn]  transposed by index_map; caller passes [N, D]
    mask_ref,         # [tt, T_pad]  padded extent, not T
```

Do not put parameter shapes in an `Args:` docstring block, where they sit too
far from the parameters to stay in sync, and avoid `Args:` entries that merely
repeat the parameter names:

```python
# ❌ Restates the parameter names without adding shape information
"""...
Args:
  x_ref: The input reference.
  w_ref: The weight reference.
  out_ref: The output reference.
"""
```

--------------------------------------------------------------------------------

## 12. Naming `Ref` parameters by memory space

Use suffixes on reference parameters to indicate their role and memory space:

| Suffix                      | Use for                                        |
| --------------------------- | ---------------------------------------------- |
| `_ref`                      | A reference when all refs in the signature     |
:                             : share the same memory space                    :
| `_hbm_ref`                  | A reference explicitly in HBM                  |
| `_vmem_ref`                 | A reference explicitly in VMEM                 |
| `_smem_ref`                 | A reference explicitly in SMEM                 |
| `_scratch` / `_scratch_ref` | Scratch memory that persists across grid steps |
| `_slot_ref`                 | One slot of a multi-buffered scratch buffer    |
| `_sem`                      | A semaphore                                    |
| `_ptr`                      | **Do not use**                                 |

**Include the memory space in the parameter name whenever more than one memory
space appears in the same signature.** If all refs in a kernel live in VMEM,
`x_ref` is sufficient. If a kernel DMAs between HBM and VMEM, name them
`x_hbm_ref` and `x_vmem_ref` so the memory spaces are clear at each operation.

Never use `_ptr` for Pallas references. `Ref` objects have shapes and dtypes and
do not support pointer arithmetic, so calling them pointers is misleading.

--------------------------------------------------------------------------------

## 13. Deriving tile shapes from refs, not the config

Inside a kernel body, read tile dimensions from the `Ref` objects passed into
the kernel rather than from the config. Reading from the config introduces a
second source of truth that can drift if a `BlockSpec` changes the tile shape.

```python
# ❌ Depends on an external config and can disagree with the BlockSpec
tt, tn = cfg.tile_t, cfg.tile_n

# ✅ Reads the actual tile shape from the ref
tt, tn = out_ref.shape
```

Similarly, compute grid extents such as `nd = D // td` once in the launcher
function and close over them, rather than recomputing the division in multiple
places where rounding could differ.

--------------------------------------------------------------------------------

## 14. Kernel body structure

Treat this section as a reference rather than a strict requirement.
Specifically, not all kernels may require all eight steps, and that is OK, you
do not need to force this structure artificially. With than in mind, prefer the
following general order when writing the kernel body:

1.  **Grid indices:** All `pl.program_id` or `lax.axis_index` calls at the top.
2.  **Shape constants:** Tile dimensions read from the refs (§13).
3.  **Semaphore aliasing:** Named semaphore slices before any DMA code.
4.  **Initialization guard:** `@pl.when` block to zero or seed the accumulator
    on the first step.
5.  **Loads and DMA starts.**
6.  **Compute:** Operations accumulating into the scratch buffer.
7.  **Masked store:** `@pl.when` block guarded on the last iteration to write
    out the result.
8.  **Epilogue:** Activation functions or residual writes.

Keeping initialization (step 4) and final store (step 7) in fixed positions
makes stateful kernels much easier to read without tracing complex control flow.

```python
def kernel(x_ref, w_ref, out_ref, acc_scratch):   # shape comments omitted for brevity
  d_i = pl.program_id(2)  # or lax.axis_index("d")
  tt, tn = out_ref.shape

  @pl.when(d_i == 0)
  def _():
    acc_scratch[...] = jnp.zeros((tt, tn), jnp.float32)

  acc_scratch[...] += jax.lax.dot_general(
      x_ref[...], w_ref[...], DIMS, preferred_element_type=jnp.float32
  )

  @pl.when(d_i == nd - 1)
  def _():
    out_ref[...] = acc_scratch[...].astype(out_ref.dtype)
```

**Close over `nd` as a static Python `int` from the launcher**, rather than
calling `pl.num_programs(2)` (or `lax.axis_size("d")`). Both work, but using the
Python integer from the launcher matches the name in the dimension glossary and
makes it obvious that the bound is a compile-time constant.

--------------------------------------------------------------------------------

## 15. Querying TPU hardware constants

**Always query hardware constants from `pltpu.get_tpu_info()` rather than
hardcoding them.** Hardcoded constants written for one TPU generation become
silently wrong on newer chips.

```python
# ❌ Hardcoded constants from an older chip; wrong on tpu7x
NUM_SUBLANES = 8
VMEM_BUDGET = 8 * 1024 * 1024

# ✅ Query hardware constants at runtime
info = pltpu.get_tpu_info()
vmem_budget = info.vmem_capacity_bytes
sublanes = info.get_sublane_tiling(dtype)
```

The only place it is fine to write a hardware tile dimension literally is in a
trailing shape comment (such as `# [8, 128]`), where it serves as documentation
rather than computation.

--------------------------------------------------------------------------------

## Checklist

-   [ ] **File organization:** Four files (`api.py`, `base.py`, `<backend>.py`,
    `<backend>_kernel.py`, e.g., `pallas_mosaic_tpu.py` and
    `pallas_mosaic_tpu_kernel.py`). No `config` in `api.py`, and no kernel
    bodies or launch calls (e.g., `pallas_call`) outside `<backend>_kernel.py`.
-   [ ] **Helper modules:** When splitting larger ops further, draw module
    boundaries by what each file touches (arrays, refs, DMAs, or plain
    integers).
-   [ ] **Config classes:** Frozen, `kw_only`, `slots` `pydantic` dataclass;
    single-field constraints in annotations and cross-field checks in
    `__post_init__` raising `ValueError`.
-   [ ] **Error handling:** Unsupported features raise `NotImplementedError`,
    not `ValueError`.
-   [ ] **Comments and docstrings:** One-line module docstring, and comments
    that explain hardware or compiler tradeoffs rather than restating the code.
-   [ ] **Dimension glossary:** Defined at the top of the file using uppercase
    for full extents, `t`-prefixed tile extents, and `n`-prefixed grid extents.
    Padded extents get their own name.
-   [ ] **Variable names:** Single-character names used only for loop indices,
    annotated math tensors, uppercase extents, and `d<x>` gradients. Grid
    indices spelled out, and shapes unpacked by name rather than indexed with
    bare integers like `x.shape[2]`.
-   [ ] **Type annotations:** `jaxtyping` on the public wrapper and `_fwd` using
    the glossary letters; unannotated `Ref` parameters inside the kernel (never
    annotated as `jax.Array`).
-   [ ] **Tests:** Tested against the plain-XLA `_fwd` in `base.py`, tolerance
    table visible at the call site, checked against a `float64` oracle before
    loosening tolerances, `Precision.HIGHEST` pinned on `f32` contractions in
    `_fwd`, and unsupported devices skipped at runtime with
    `self.skipTest(...)`.
-   [ ] **Kernel signature sections:** Kernel parameters grouped under `#
    Inputs.`, `# Outputs.`, and `# Scratch.`
-   [ ] **Trailing shape comments:** Every `Ref` parameter has a `# [shape]
    dtype` comment using the glossary letters and indicating whether it is a
    single block or the full array (`pl.ANY`).
-   [ ] **Ref suffixes:** Memory space included in the ref name (`_hbm_ref`,
    `_vmem_ref`, `_smem_ref`) whenever multiple memory spaces share a signature.
    Never `_ptr`.
-   [ ] **Deriving shapes:** Tile extents derived from the refs inside the
    kernel rather than re-read from the config; grid extents computed once in
    the launcher.
-   [ ] **Kernel body order:** Grid indices → shape constants → semaphore
    aliases → init guard → loads/DMAs → compute → final store guard → epilogue.
-   [ ] **Hardware constants:** Read from `pltpu.get_tpu_info()`, never
    hardcoded.
