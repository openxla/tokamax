# Mosaic GPU: layout inference fails on a partial reduction with `keep_dims=True`

**TL;DR** — `jnp.mean(x, axis=-1, keepdims=True)` inside a Warpgroup-semantics
Mosaic GPU kernel makes layout inference fail with a generic
`Failed to infer a possible set of layouts`. The rank-reducing spelling of the
same computation works. Two acknowledged `TODO`s in the solver are the cause:
tiled layouts are simply not handled when `keep_dims=True`, in both the
evaluator and the candidate generator.

This is **not** a problem with `IsTransferableSmemRegisters` — see
[Ruled out](#ruled-out) below.

## Environment

- `jax` @ `ce30e77df8` ("Added cp.async support to the WG lowering").
  Also reproduces on released `jax`/`jaxlib` 0.11.0.
- Reproduces at **lowering time**; no GPU needed. With no GPU present,
  `_infer_arch()` (`jax/experimental/mosaic/gpu/core.py:787-798`) returns
  `(9, 0)`, so this is an sm_90 target.
- `lowering_semantics=LoweringSemantics.Warpgroup`.

## Repro

`repro.py` in this directory. Two kernels that compute the same thing:

```python
def with_keepdims(x_ref, y_ref):
  x = x_ref[...]
  y_ref[...] = x - jnp.mean(x, axis=-1, keepdims=True)        # FAILS

def without_keepdims(x_ref, y_ref):
  x = x_ref[...]
  mean = jnp.mean(x, axis=-1)                                 # OK
  y_ref[...] = x - jax.lax.broadcast_in_dim(mean, BLOCK, (0,))
```

```
FAIL  keepdims=True   ValueError: Failed to infer a possible set of layouts.
                      This should only happen if user-provided layout casts are unsatisfiable.
OK    keepdims=False
```

Independent of how the tile is obtained. Both of these fail identically:

| tile source | `keepdims=True` | `keepdims=False` + `broadcast_in_dim` |
|---|---|---|
| `pl.pallas_call` + tiled/swizzled `BlockSpec` (SMEM ref) | FAIL | OK |
| `plgpu.kernel` + `plgpu.load(gmem_ref, layout=<TiledLayout>)` | FAIL | OK |

Block size, reduced axis (0 or -1), and dtype (f32 / bf16 / f16) make no
difference.

## Root cause

Two matching `TODO`s, same author, both gating on `keep_dims`:

**1. The evaluator refuses to reduce a tiled layout**
(`jax/experimental/mosaic/gpu/constraints.py:303-313`):

```python
    case RegisterLayout(value=fa.TiledLayout() as layout):
      # TODO(allanrenucci): Add support for reducing tiled layouts when keep_dims=True.
      if expr.keep_dims:
        return default()          # <-- returns an unresolved symbolic expression
      num_untiled_dims = expr.rank - len(layout.base_tile_shape)
      ...
      return RegisterLayout(layout.reduce(reduced_tiling_axes))
```

`default()` hands back the `Reduce` expression unreduced, so the equation
`small = Reduce(large, axes, keep_dims=True)` never resolves to a constant.

Note the asymmetry directly below it: `WGStridedFragLayout` (`:314-320`) and
`WGSplatFragLayout` (`:321-323`) both handle `keep_dims` correctly, via
`utils.reduce_shape(layout.shape, expr.axes, expr.keep_dims)`. Only
`TiledLayout` is unimplemented — and `TiledLayout` is the only kind a partial
reduction accepts, because `_multi_dim_reduction_constraint_system`
(`layout_inference.py:1259-1282`) constrains the source with
`NotOfType(source_variable, fa.WGStridedFragLayout)`.

**2. The candidate generator yields nothing**
(`jax/experimental/mosaic/gpu/layout_inference.py:162-165`):

```python
  assert isinstance(small.value, fa.TiledLayout)
  # TODO(allanrenucci): Add support for reducing tiled layouts when keep_dims=True.
  if keep_dims:
    return
  candidates = [fa.WGMMA_LAYOUT, fa.WGMMA_TRANSPOSED_LAYOUT, fa.TCGEN05_LAYOUT, ...]
```

So `conjure_assignment` gets no tiled candidates from the reduce equation. Its
only remaining fallback for a register variable is
`_strided_layout_for_variable` (`layout_inference.py:513`) — strided, which the
reduction constraint rejects. The search then backtracks until it exhausts
`_DEFAULT_LAYOUT_INFERENCE_FUEL` (100k).

A fix needs both sites: (1) so a candidate can be *checked*, (2) so one is ever
*proposed*.

## Ruled out

`IsTransferableSmemRegisters` is **not** implicated, despite the failure also
appearing on the SMEM/`BlockSpec` path. Calling
`IsTransferableSmemRegisters._constant_holds()` directly over a sweep of
(SMEM transforms × tiled register layout × transfer kind) for a `(64, 128)` f32
block:

```
tried 192 triples -> 147 satisfiable
```

including `(WGMMA, tiling=None, swizzle=None, UNOPTIMIZED)` — i.e. precisely the
configuration that inference's own SMEM fallback (`SMEMTransforms(None, None)`,
`layout_inference.py:517-518`) conjures. The transfer constraint admits these
layouts; the reduce equation never proposes one to test.

## Possibly related

A rank-preserving `(n, 1)` register value appears to be unsupported against a
tiled layout even with **no reduction involved**. Loading a genuine `(BLOCK_M, 1)`
tile from a GMEM ref and broadcasting it over a `(BLOCK_M, N)` tiled value fails
the same way, as does reshaping that value to `(BLOCK_M,)` first. I did not trace
this one to a line; it may share a cause with the above (see the `kept_dims`
branch of `_broadcast_in_dim_constraint_system`, `layout_inference.py:1285-1341`)
or it may be separate.

## Impact

`keepdims=True` is the natural spelling for every normalization and softmax
kernel — it is what `jnp.mean`/`jnp.sum` documentation steers you to, and what
the equivalent Triton kernel uses. The failure mode is a generic
"Layout inference failed to find a solution. Consider adding layout annotations
to your program to guide the search", which actively misdirects: adding layout
annotations does not help, and the message gives no hint that `keepdims` is the
culprit. It also costs 100k fuel (~3 minutes of wall clock in our test suite)
before reporting.

Two cheap mitigations, independent of the real fix:

- Make `keep_dims=True` on a tiled layout raise `NotImplementedError` naming
  `keepdims`, instead of silently yielding no candidates and timing out.
- Have the reduce constraint short-circuit when it can prove no candidate exists,
  rather than burning the full fuel budget (there is already a
  `TODO(allanrenucci)` to this effect at `layout_inference.py:477-478`).

## Related gap: reduce candidates are all ≥64 rows tall

Separately from `keep_dims`, `extract_assignment_candidates_from_reduce_equation`
(`layout_inference.py:167-174`) offers only

```python
  candidates = [
      fa.WGMMA_LAYOUT, fa.WGMMA_TRANSPOSED_LAYOUT, fa.TCGEN05_LAYOUT,
      fa.TCGEN05_TRANSPOSED_LAYOUT, tcgen05.TMEM_NATIVE_LAYOUT,
  ]
```

all of which tile the slow axis by 64. So a partial reduction over a block
shorter than 64 rows is unsatisfiable. Measured, `pl.pallas_call` + tiled
`BlockSpec` + `jnp.mean(x, axis=-1)`, f32, `A=128`:

| `block_m` | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| result | fail | fail | fail | OK | OK |

This matters for memory-bound kernels. A normalization reduces over the whole
feature axis, so that axis cannot be blocked — a block is `block_m × A`. Forcing
`block_m ≥ 64` puts a floor of `64 × A × 4` bytes on SMEM, which caps the
reduced axis at 128 elements in f32 before exceeding the 232 KiB budget:

| `A` (reduced axis) | 128 | 256 | 512 | 1024 | 4096 |
|---|---|---|---|---|---|
| block SMEM | 32 KiB | 64 KiB | 128 KiB | 256 KiB | 1 MiB |
| result | OK | fail | SMEM | SMEM | SMEM |

Typical LLM hidden sizes (1024–8192) are all out of reach.

**This is a candidate-list gap, not a capability gap.** Every stock tiled layout
is 64 or 128 rows tall:

| layout | `base_tile_shape` |
|---|---|
| `WGMMA_LAYOUT`, `WGMMA_TRANSPOSED_LAYOUT` | `(64, 8)` |
| `TCGEN05_LAYOUT`, `TCGEN05_TRANSPOSED_LAYOUT` | `(128, 8)` |
| `TMEM_NATIVE_LAYOUT`, `tmem_native_layout(v)` | `(128, v)` |
| `fa_m64_collective_layout(c)` | `(64, c)` |

But a 4-row-tall tiled layout reduces correctly *and* is SMEM-transferable. With
one explicitly cast onto the load, the same `pl.pallas_call` + tiled `BlockSpec`
kernel lowers across the whole range that fails unannotated:

| `block_m` | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| unannotated | fail | fail | fail | fail | OK | OK |
| 4-row tiled layout cast | OK | OK | OK | OK | OK | OK |

and the SMEM floor drops with it — at `block_m=4`, reduced axes up to 4096
lower fine (vs. 128 at `block_m=64`). The layout used is

```python
TiledLayout(Tiling(((4, 32 * v), (4, v))), warp_dims=(-2,), lane_dims=(-3,), vector_dim=-1)
```

with `v` the largest power of two such that `cols % (32 * v) == 0` and
`v * bitwidth <= 128` — warps from the 4-row dim, lanes and vector from the
contiguous dim. (For fewer than 4 rows, the whole-warpgroup-per-row variant
`Tiling(((1, 128 * v), (32 * v,), (v,)))` with `warp_dims=(-3,)`,
`lane_dims=(-2,)` serves the same purpose.)

### Verified fix for the SMEM path

For a kernel whose tile arrives via a `BlockSpec`/SMEM ref, the ≥64-row
candidates come from `_register_layouts_for_optimized_transfer_to_smem`
(`layout_inference.py:220-252`), not from the reduce equation. Its non-Hopper
list already carries `WGMMA_LAYOUT` for exactly this reason (`:243-247`):

> Keep using WGMMA and WGMMA_TRANSPOSED layouts here, simply because they may
> apply to smaller shapes where TCGEN05 layouts do not apply. This can be useful
> for kernels not involving MMAs that still need optimized transfers to
> TiledLayouts […]

Appending a shape-parameterized short-tile family to that list — *after* the
existing entries, so nothing changes for shapes that already resolve — makes an
**unannotated** `emit_pipeline` normalization kernel lower at sm_80:

| `block_m` / `A` | 8/128 | 16/128 | 32/128 | 64/128 | 8/1024 | 8/2048 |
|---|---|---|---|---|---|---|
| stock | fail | fail | fail | OK | fail | fail |
| with short-tile candidate | **OK** | **OK** | **OK** | OK | **OK** | **OK** |

```python
 def _register_layouts_for_optimized_transfer_to_smem(shaped_type, smem_layout, arch):
   ...
   yield from candidate_layouts
+  # Short-tile layouts for non-MMA kernels: a partial reduction over a block of
+  # fewer than 64 rows has no candidate above.
+  if len(shaped_type.shape) == 2 and smem_layout.tiling is not None:
+    if (layout := short_tile_layout(shaped_type.shape[-1],
+                                    utils.bitwidth(shaped_type.element_type))):
+      yield layout
```

Note this does **not** help a kernel that loads its tile straight from GMEM. A
GMEM load imposes no transfer constraint, so there is no constraint to extract a
candidate from, and `conjure_assignment`'s only register fallback is
`_strided_layout_for_variable`. Covering that case needs a *tiled* fallback in
`conjure_assignment`, which is a larger change — any tiled layout is legal there,
but only some produce coalesced loads, so it wants a cost model.

### Suggested patch for the reduce equation

Add a shape-parameterized short-tile family alongside the
existing shape-parameterized entry, e.g.

```python
  if large_shape[-1] % 16 == 0:
    candidates.append(tcgen05.fa_m64_collective_layout(large_shape[-1]))
+ candidates.extend(short_tile_layouts(large_shape[-1], bitwidth))   # 4-row and 1-row families
```

This is safe by construction: the loop below already yields a candidate only if
`candidate.reduce(reduced_tiling_axes) == small.value`, so extra candidates can
only widen the solvable set, never change an existing solution.

Doing this would remove the need for hand-written layouts in this class of
kernel entirely — including on the GMEM path, where this same list is the only
source of tiled candidates (see [Root cause](#root-cause)).

## Workaround

Use the rank-reducing reduction plus an explicit broadcast:

```python
mean = jnp.mean(x, axis=-1)                                # not keepdims=True
x = x - jax.lax.broadcast_in_dim(mean, x.shape, (0,))
```
