# Mosaic GPU normalization: remaining work

State as of this branch: the **forward** kernel is ported to the pipelined
(`emit_pipeline` + `cp.async`) form and lowers for sm_80. It carries **no layout
annotations** — that depends on a local `jax` patch, see §5. Gradients currently
borrow the Triton VJP. Nothing has executed on hardware.

Ordered by dependency. Each item says how to check it.

---

## 0. Environment (read first)

The kernel needs jax HEAD *and* a jaxlib newer than any release. The project
venv (`.venv`, Python 3.13, jax 0.11.0) **cannot lower this kernel** — jax 0.11.0
has no pre-Hopper `emit_pipeline` path and fails with `arrive_expect_tx is only
supported on Hopper+ hardware`.

Use `.venv314` (not committed; `pyproject.toml` and `uv.lock` are untouched
deliberately, because the jaxlib wheel is a local macOS-arm64 artifact that would
break the Linux runner):

```
UV_PROJECT_ENVIRONMENT=.venv314 uv sync --python 3.14 --extra test
uv pip install --python .venv314/bin/python \
    -e ../jax -e ../flax ../jax/dist/jaxlib-*-cp314-*.whl
.venv314/bin/python -m pytest tokamax/_src/ops/normalization/mosaic_test.py -q
```

Gotchas that cost time:

- `uv pip` ignores `UV_PROJECT_ENVIRONMENT`; pass `--python .venv314/bin/python`.
- Installing `-e ../jax` alone silently pulls jaxlib 0.11.0 from PyPI and
  clobbers the local wheel. Install all three in **one** resolution.
- `flax` must come from `../flax`; the released one fails on jax HEAD with
  `jax.experimental.hijax has no attribute MutableHiType`.
- Mosaic reads the target arch from the default device and falls back to
  `(9, 0)` when there is no GPU, so a CPU-only box lowers for **Hopper** unless
  you override it. `mosaic_test._target_ampere()` patches
  `mgpu_core._infer_arch`; use it for anything arch-sensitive.
- jax's own tests skip under a self-built jaxlib (`setUp` compares
  `jax.version._version` to `jax.lib.__version__`). Set them equal to get signal.

---

## 1. Strided case (`num_b > 1`) — **done** (lowering only; §3 still applies)

`_fwd` no longer rejects a non-contiguous reduced axis. The blocker was never the
layout: a tiled layout reduces a strided axis fine, down the tile's slow
dimension, crossing warps through the reduction scratch. The `cp.async` tiled
copy is **2D only** (`launch_context.py:1451`, `Only 2D copies implemented`), but
that check runs on the rank *after* squeezing, so squeezing the degenerate
leading axis in the `BlockSpec` both makes the copy 2D and keeps the GMEM slice
contiguous. The same squeeze also sidesteps the `len(shaped_type.shape) == 2`
gate on the §5 patch.

What landed:

- `mosaic_tiling.Plan` carries `num_b`/`block_b` and derives everything
  case-dependent from `strided`: `tile_shape` (always 2D), `block_shape` (the
  `(None, A, block_b)` spec), `reduce_axis`, and the `out_index`/`stat_index`
  slices. `plan()` blocks whichever axis the case allows — `M` when contiguous,
  `B` when strided, where the tile can only hold one row of `M`.
- `steps` is now the *total* tile count and `block_index` a flat index that
  `tile_indices` splits into a block of `M` and a block of `B`, so the CTA
  hand-out and its clamped tail are unchanged.
- Stats are `(M, B)`-shaped rather than `(M,)`.
- `block_b` divides `num_b`, is a multiple of 32 (the lanes tile the tile's minor
  axis) and defaults to one cache line's worth; the swizzle comes from `block_b`
  rather than from `A`, and `A` needs `% 8` (the tiling) rather than `% 32`.
- `AmpereLoweringTest` gained `(8, 128, 256)` and `(4, 32, 96)` at `axis=1`; the
  whole file lowers for sm_80 (`(4, 6, 256)` from the old suite is gone: `A = 6`
  is not a multiple of 8).

Verified only at lowering time, on both the load and the store side. Worth
knowing about the store: the strided case stores through a non-contiguous GMEM
subview (`memref<128x32xf32, strided<[256, 1]>>`), which `store_untiled` →
`transfer_tiled` addresses off `get_strides_and_offset()`, so it is right by
construction rather than by accident.

New accepted limits, both of which raise `NotImplementedError` so dispatch can
pick another implementation:

- `B` needs a divisor that is a multiple of 32 at or below `block_n`. A shape
  like `B = 40` has none.
- SMEM bounds `A` harder than in the contiguous case, because `block_b` cannot go
  below 32 whereas `block_m` can go to 8: roughly `A ≤ 512` in f32.

## 2. Port the VJP to the pipelined form

`mosaic.py.__post_init__` currently wires `pallas_triton_vjp`. `mosaic_vjp.py` is
**stale** — it still calls the old `mosaic_tiling` API (`plan(..., block_b=...)`,
`Plan.load`, `bcast`, `drop_duplicate_rows`) and will fail if invoked. Either
port it or delete it; do not leave it as-is.

Design notes carried over:

- `dx` is stored by hand, exactly like `y` (output pipelining is Hopper-only).
- `dscale`/`doffset` are sums **over rows**, so they need per-CTA partials of
  shape `(num_ctas, A)` finished by an XLA reduction — cheap, it is grid-sized
  rather than `x`-sized.
- **The one place masking survives.** Tiles now divide `M` exactly, so there are
  no duplicate rows *within* a tile. But `Plan.block_index` clamps a CTA's step
  to `steps - 1`, so a CTA handed fewer than `steps_per_cta` steps re-reads its
  last tile. That is idempotent for `y`/`dx` and **wrong for a sum over rows** —
  it double-counts. Either make `num_ctas * steps_per_cta == steps` exactly, or
  zero the repeated contribution (the old `drop_duplicate_rows` did this; it is
  in git history).
- Restore `test_grad_gradients` as a *Mosaic* test at that point — right now it
  exercises the Triton VJP and would pass even if the Mosaic one were absent.

## 3. Verify on hardware — the real gate

Nothing in this branch has executed a single instruction; everything is
lowering-time. Needed on the A100 runner:

- jaxlib built from `../jax` for **linux x86_64**, including the §5 patch. Without
  the patch the kernel does not compile at all.
- `uv run pytest tokamax/_src/ops/normalization/mosaic_test.py -x -s` — the 37
  currently-skipped tests are the numerical ones.
- Priority checks: `104x128` (block falls to 8), `4096x2048` (block shrinks to
  fit SMEM), bf16, `test_batched_params_lower`'s vmap path, and — since §1 has
  only ever been lowered — the strided cases `(8, 128, 256)` and `(4, 32, 96)` at
  `axis=1`, where both the statistics indexing and the strided GMEM store are
  new.

**Untested behaviour worth attention:** `vmap`. The old kernel needed a custom
batching rule; this one relies on `plgpu.kernel`'s built-in rule prepending a
grid axis, while `index_map` and `rows()` read the grid by name
(`axis_index('cta')`), which should be unaffected. `test_batched_params_lower`
lowers, but no numerical vmap test has run.

## 4. Performance

The entire justification for this rewrite is `cp.async` latency hiding, which is
still unmeasured. Benchmark against (a) the previous GMEM→registers kernel — in
git history, it handled `A` up to ~8192 — and (b) `pallas_triton`.

Knobs, currently hardcoded in `mosaic_tiling`:

- `_STEPS_PER_CTA = 4` — tiles per CTA, i.e. how much there is to overlap.
- `num_stages = min(2, steps_per_cta)` — pipeline depth.
- `_SMEM_BUDGET = 227 * 1024`, taken from Mosaic's own error message. Confirm it
  against the real device.

If any of these matter, promote them to `Config` fields so autotuning can reach
them (the config is currently borrowed wholesale from Triton).

## 5. Upstream the jax patch

Uncommitted in `../jax`, and **this branch does not compile without it**:

- `fragmented_array.py`: new `short_tile_layout(cols, bitwidth)`.
- `layout_inference.py`: yields it from
  `_register_layouts_for_optimized_transfer_to_smem`, after the existing
  candidates so nothing that already resolves changes.
- `tests/mosaic/gpu_layout_inference_test.py`: two tests. Suite: 479 passed,
  95 skipped, no regressions.

Follow-ups for the same PR or the next one:

- Drop the `len(shaped_type.shape) == 2` gate so rank-3 tiles get a candidate
  (§1 works around it by squeezing).
- The two bugs written up in `jax-bug-mgpu-keepdims/`: `keep_dims=True` yields no
  tiled candidates (two `TODO(allanrenucci)` sites), and the reduce-equation
  candidate list has the same ≥64-row gap.

## 6. Decisions still open

- **Env migration.** `.venv314` is side-by-side. Moving the repo properly means
  bumping `requires-python`, and the runner needs its own jaxlib build. The
  full-suite comparison old-vs-new was still running when I stopped: the new env
  showed 1138 failed / 1540 passed / 4565 skipped, which is **uninterpretable
  without the old-env baseline** — most of this suite wants a GPU. Get the
  baseline before reading anything into it.
- **Accepted regression:** reduced axis ≤ 2048 in f32 (SMEM-bound; the reduced
  axis cannot be blocked). The old kernel reached ~8192. Shapes above the limit
  raise `NotImplementedError`, so dispatch can pick another implementation.
- `input_output_alias` and symbolic shapes remain unsupported, as before.

## 7. Housekeeping

- `.DS_Store` is staged in git.
- There is a copy of `jax-bug-mgpu-keepdims/` inside the `../jax` tree as well as
  this one.
- `jax-bug-mgpu-keepdims/repro.py` has never been executed (written under a
  standing instruction not to run scripts); the cases in it were all run inline.
