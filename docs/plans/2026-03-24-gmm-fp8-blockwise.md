# GMM FP8 Block-wise Quantization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add FP8 block-wise quantization support (float8_e4m3fn, block sizes 128/256/512) to GMM ragged dot on TPU v7x, covering forward and backward (gmm + dlhs + tgmm) with forward-save optimization.

**Architecture:** Reuse the existing subchannel quantization infrastructure. FP8 block-wise is semantically identical to int8 subchannel — each block has its own scale. Forward pass saves columnwise-quantized FP8 inputs as residuals for backward (aligned with Transformer Engine pattern). TPU v7x native FP8 dot is used directly without upcast.

**Tech Stack:** JAX Pallas, qwix (quantization library), TPU v7x Mosaic

---

### Task 1: `_quantize_as` FP8 support

Skip `jnp.round` for floating-point quantization types. FP8 is floating-point; `.astype()` handles rounding natively.

**Files:**
- Modify: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py:264-274`

**Step 1: Write the failing test**

Add to `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py`:

```python
from absl.testing import parameterized

class QuantizeAsTest(parameterized.TestCase):
  """Tests for _quantize_as with FP8."""

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  def test_quantize_as_fp8(self):
    from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_kernel as backend
    x = jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.bfloat16)
    result = backend._quantize_as(x, jnp.float8_e4m3fn, axis=1, scale=None)
    self.assertEqual(result.qvalue.dtype, jnp.float8_e4m3fn)
    self.assertEqual(result.scale.shape, (1, 1))
    # Dequantized value should be close to original
    restored = result.qvalue.astype(jnp.float32) * result.scale
    chex.assert_trees_all_close(restored, x.astype(jnp.float32), atol=0.5)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py::QuantizeAsTest::test_quantize_as_fp8 -v`

Expected: FAIL or behavior difference (round is unnecessary for FP8)

**Step 3: Write minimal implementation**

In `pallas_mosaic_tpu_kernel.py:264-274`, change:

```python
def _quantize_as(x, qdtype: jnp.dtype, axis: int, scale: float | None):
  info_fn = jnp.iinfo if jnp.dtype(qdtype).name.startswith("int") else jnp.finfo
  max_val = min(info_fn(qdtype).max, -info_fn(qdtype).min)
  if scale is None:
    scales = jnp.max(jnp.abs(x), axis=axis, keepdims=True) / jnp.array(
        max_val
    ).astype(jnp.bfloat16)
    inv_scales = jnp.broadcast_to(1.0 / scales, x.shape)
  else:  # compile-time (static) quantization scale
    scales, inv_scales = scale, 1.0 / scale
  qvalue = x * inv_scales
  if jnp.issubdtype(qdtype, jnp.integer):
    qvalue = jnp.round(qvalue)
  return QArray(qvalue.astype(qdtype), scales)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py::QuantizeAsTest -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py
git commit -m "feat(gmm): skip jnp.round in _quantize_as for FP8 dtypes"
```

---

### Task 2: gmm `dot()` FP8 native path

Add FP8 detection to the `dot()` function in `gmm()`. TPU v7x supports native FP8 dot, so FP8 operands pass directly to `jax.lax.dot_general` with `float32` accumulation.

**Files:**
- Modify: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py:406-421` (gmm dot)

**Step 1: Write the implementation**

In `pallas_mosaic_tpu_kernel.py`, modify the `dot()` function inside `gmm()` at line 406:

```python
  def dot(x, y, preferred_element_type, *, precision=precision):
    # TODO: Pallas-MTPU doesn't support `DotAlgorithmPreset`.
    is_fp8 = lambda d: jnp.issubdtype(d, jnp.floating) and jnp.dtype(d).itemsize == 1
    if is_fp8(x.dtype) or is_fp8(y.dtype):
      # TPU v7x native FP8 dot — pass through directly.
      precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
      preferred_element_type = jnp.float32
    elif precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
      x = x.astype(jnp.bfloat16)
      y = y.astype(jnp.bfloat16)
      precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
      preferred_element_type = jnp.float32

    rhs_contracting_dim = 1 if transpose_rhs else 0
    return jax.lax.dot_general(
        x,
        y,
        dimension_numbers=(((1,), (rhs_contracting_dim,)), ((), ())),
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
```

**Step 2: Commit**

```bash
git add tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py
git commit -m "feat(gmm): add FP8 native dot path in gmm kernel"
```

Note: This is tested end-to-end in Task 5. No isolated unit test needed since `dot()` is an inner function.

---

### Task 3: tgmm `dot()` FP8 native path

Same change for the `dot()` function inside `tgmm()`.

**Files:**
- Modify: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py:713-726` (tgmm dot)

**Step 1: Write the implementation**

In `pallas_mosaic_tpu_kernel.py`, modify the `dot()` function inside `tgmm()` at line 713:

```python
  def dot(x, y, preferred_element_type, *, precision=precision):
    # TODO: Pallas-MTPU doesn't support `DotAlgorithmPreset`.
    is_fp8 = lambda d: jnp.issubdtype(d, jnp.floating) and jnp.dtype(d).itemsize == 1
    if is_fp8(x.dtype) or is_fp8(y.dtype):
      # TPU v7x native FP8 dot — pass through directly.
      precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
      preferred_element_type = jnp.float32
    elif precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
      x = x.astype(jnp.bfloat16)
      y = y.astype(jnp.bfloat16)
      precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
      preferred_element_type = jnp.float32

    return jax.lax.dot(
        x,
        y,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
```

**Step 2: Commit**

```bash
git add tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py
git commit -m "feat(gmm): add FP8 native dot path in tgmm kernel"
```

---

### Task 4: tgmm subchannel iteration support

Remove `subchannel_iters != 1` NotImplementedError in tgmm. Implement subchannel iteration in `_do()` by slicing lhs/rhs along axis=0 (M dimension).

**Files:**
- Modify: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py:728-793`

**Step 1: Write the failing test**

Add to `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py`:

```python
class TgmmSubchannelTest(parameterized.TestCase):
  """Tests for tgmm with subchannel quantization."""

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  @parameterized.parameters(
      (jnp.int8, (128, 1), (128, 1)),
      (jnp.int8, (16, 1), (16, 1)),
  )
  def test_tgmm_subchannel(self, qdtype, lhs_tile_shape, rhs_tile_shape):
    """Test tgmm with subchannel quantization (previously unsupported)."""
    from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_kernel as backend
    num_groups, m, k, n = 4, 512, 256, 256
    rng = jax.random.PRNGKey(0)
    lhs = jax.random.normal(rng, (k, m), dtype=jnp.bfloat16)  # tgmm takes (k, m)
    rhs = jax.random.normal(jax.random.split(rng)[0], (m, n), dtype=jnp.bfloat16)
    group_sizes = jnp.array([m // num_groups] * num_groups, dtype=jnp.int32)

    # Quantize with subchannel scales
    lhs_q = qwix.quantize(lhs.mT, qdtype, tiled_axes={0: lhs_tile_shape[0], 1: lhs_tile_shape[1]}).mT
    rhs_q = qwix.quantize(rhs, qdtype, tiled_axes={0: rhs_tile_shape[0], 1: rhs_tile_shape[1]})

    # Reference: dequantize and compute
    lhs_deq = qwix.dequantize(lhs_q.mT).mT
    rhs_deq = qwix.dequantize(rhs_q)
    expected = backend.tgmm(
        lhs_deq, rhs_deq, group_sizes=group_sizes,
        precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
        out_dtype=jnp.float32,
    )

    # Actual: pass quantized arrays directly
    actual = backend.tgmm(
        lhs_q, rhs_q, group_sizes=group_sizes,
        precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
        out_dtype=jnp.float32,
    )
    chex.assert_trees_all_close(actual, expected, atol=0.5, rtol=0.1)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py::TgmmSubchannelTest -v`

Expected: FAIL with `NotImplementedError: subchannel_iters != 1 not supported yet in tgmm.`

**Step 3: Write minimal implementation**

Replace the tgmm kernel function at line 728-793. The key change is in `_do()` — add subchannel iteration loop:

```python
  def kernel(
      group_metadata,
      _,
      lhs_ref,
      rhs_ref,
      out_ref,
      acc_scratch,
      *,
      subchannel_iters: int,
  ):
    grid_id = pl.program_id(2)
    group_offsets, group_ids, _ = group_metadata
    group = group_ids[grid_id]
    prev_group = group_ids[jnp.where(grid_id > 0, grid_id - 1, 0)]
    is_prologue = (grid_id == 0) | (group != prev_group)
    is_end_of_grid = grid_id == (pl.num_programs(2) - 1)
    next_group = group_ids[jnp.where(is_end_of_grid, grid_id, grid_id + 1)]
    is_epilogue = is_end_of_grid | (group != next_group)
    group_size = group_offsets[group + 1] - group_offsets[group]
    nonzero_gs = group_size > 0

    def _zero_acc():
      acc_scratch[...] = jnp.zeros_like(acc_scratch)

    def _store_accum():
      acc = acc_scratch[...]
      if activation is not None:
        acc = activation(acc)
      out_ref[...] = acc.astype(out_dtype)

    def _do():
      for it in range(subchannel_iters):
        # load lhs and rhs sub-slices along M axis (axis=0)
        lhs = _maybe_get_subslice(lhs_ref, it, subchannel_iters, axis=0)
        rhs = _maybe_get_subslice(rhs_ref, it, subchannel_iters, axis=0)

        # optional dynamic quantization within the kernel
        if lhs_qdtype is not None and not isinstance(lhs, QArray):
          lhs = _quantize_as(lhs, lhs_qdtype, axis=1, scale=lhs_static_scale)
        if rhs_qdtype is not None and not isinstance(rhs, QArray):
          rhs = _quantize_as(rhs, rhs_qdtype, axis=1, scale=rhs_static_scale)

        # unpack quantized arrays for dot operation
        scales = []
        if isinstance(lhs, QArray):
          scales.append(lhs.scale.T)
          lhs = lhs.qvalue
        if isinstance(rhs, QArray):
          scales.append(rhs.scale)
          rhs = rhs.qvalue

        # mask: use sub-tile size for masking
        sc_tm = tm // subchannel_iters
        kwargs = dict(grid_id=grid_id, group_metadata=group_metadata, tm=sc_tm)
        lhs = jnp.where(_get_store_mask(**kwargs, tn=tk), lhs, 0)
        rhs = jnp.where(_get_store_mask(**kwargs, tn=tn), rhs, 0)

        is_int = lambda x: jnp.issubdtype(x.dtype, jnp.integer)
        acc_dtype = jnp.int32 if is_int(lhs) and is_int(rhs) else jnp.float32
        out = dot(lhs.T, rhs, acc_dtype)

        # apply scales to the output if the inputs were quantized
        for scale in scales:
          out = _scale_out_by_scale(out, scale)
        acc_scratch[...] += out.astype(acc_scratch.dtype)

    # ... rest of kernel (combine_scopes branching) stays the same ...
```

Important: `_get_store_mask` needs the `m_id` to be offset by `it * sc_tm` for each sub-iteration. Review and adjust the mask computation carefully. The `_maybe_get_subslice` handles the QArray scale slicing automatically.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py::TgmmSubchannelTest -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py
git commit -m "feat(gmm): implement tgmm subchannel iteration support"
```

---

### Task 5: Forward FP8 block-wise quantization (gmm)

Add FP8 qdtype support in the dispatch layer for forward gmm. This makes `PallasMosaicTpuRaggedDot(qdtype='float8_e4m3fn')` work for forward pass.

**Files:**
- Modify: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py:147-176` (DEFAULT branch in `_fwd`)
- Test: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py`

**Step 1: Write the failing test**

Add to `pallas_mosaic_tpu_test.py`:

```python
class FP8RaggedDotTest(parameterized.TestCase):
  """Tests for FP8 block-wise quantization in ragged dot."""

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  @parameterized.product(
      block_size=(128, 256),
      use_as_qarray=(True, False),
  )
  def test_fp8_forward(self, block_size, use_as_qarray):
    """Test FP8 block-wise quantized forward pass."""
    num_groups, m, k, n = 8, 512, 256, 512
    rng = jax.random.PRNGKey(42)
    a = jax.random.normal(rng, (m, k), dtype=jnp.bfloat16) * 0.1
    b = jax.random.normal(jax.random.split(rng)[0], (num_groups, k, n), dtype=jnp.bfloat16) * 0.1
    group_sizes = jnp.array([m // num_groups] * num_groups, dtype=jnp.int32)

    # Quantize inputs
    a_tile = {0: 1, 1: block_size}
    b_tile = {0: 1, 1: block_size, 2: 1}
    if use_as_qarray:
      a_q = quantization.AsQArray(a, jnp.float8_e4m3fn, tiled_axes=a_tile)
      b_q = quantization.AsQArray(b, jnp.float8_e4m3fn, tiled_axes=b_tile)
    else:
      a_q = qwix.quantize(a, jnp.float8_e4m3fn, tiled_axes=a_tile)
      b_q = qwix.quantize(b, jnp.float8_e4m3fn, tiled_axes=b_tile)

    # Reference: dequantize and compute with bf16
    ref_result = test_base.ref(a_q, b_q, group_sizes)

    # Actual: use FP8 ragged dot
    op = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot()
    actual = op(a_q, b_q, group_sizes=group_sizes, preferred_element_type=jnp.float32)

    count = sum(group_sizes)
    chex.assert_trees_all_close(actual[:count], ref_result[:count], atol=0.5, rtol=0.1)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py::FP8RaggedDotTest::test_fp8_forward -v`

Expected: FAIL (FP8 QArray not handled correctly in dispatch/kernel)

**Step 3: Verify existing `_fwd` DEFAULT branch handles FP8 QArray inputs**

The existing code path should already work since:
- `maybe_quantize` checks `isinstance(x, QArray)` — FP8 QArray passes through
- `quant_block_spec` is dtype-agnostic
- The kernel's subchannel logic handles QArray generically
- The modified `dot()` (Task 2) handles FP8 natively

If the test passes without additional changes, great. If not, debug and fix.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py::FP8RaggedDotTest -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py
git commit -m "test(gmm): add FP8 block-wise forward pass tests"
```

---

### Task 6: FP8 forward-save residuals

Add `FP8Residuals` dataclass. Modify `_fwd` to save columnwise-quantized FP8 inputs for backward use. Modify `Residuals` type from `NoneType` to support `FP8Residuals`.

**Files:**
- Modify: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py:46-48` (Residuals type)
- Modify: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py:162-176` (DEFAULT branch)
- Modify: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py:238-242` (residuals return)

**Step 1: Add FP8Residuals dataclass and helper**

At top of `pallas_mosaic_tpu.py` after imports, add:

```python
def _is_fp8(dtype) -> bool:
  dtype = jnp.dtype(dtype)
  return jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize == 1


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class FP8Residuals:
  """Pre-quantized FP8 inputs saved during forward for backward use."""
  lhs_for_drhs: QArray | None  # columnwise-quantized lhs, for drhs (tgmm)
  rhs_for_dlhs: QArray | None  # columnwise-quantized rhs, for dlhs
```

Update the `Residuals` type:

```python
Residuals = FP8Residuals | types.NoneType
```

**Step 2: Modify DEFAULT branch to save FP8 residuals**

In `_fwd`, after the forward gmm call in the DEFAULT branch (around line 162-176):

```python
if ragged_dot_dimension_numbers == DEFAULT_RAGGED_DOT_DIM_NUMS:  # gmm fwd
    lhs = maybe_quantize(lhs, (1, lhs.shape[1]))
    rhs = maybe_quantize(rhs, (1, rhs.shape[1], 1))

    # Pre-quantize columnwise for backward if FP8 block-wise
    fp8_residuals = None
    if return_residuals and isinstance(lhs, QArray) and _is_fp8(lhs.qvalue.dtype):
        lhs_raw = quantization.as_array(lhs)
        rhs_raw = quantization.as_array(rhs)
        k = lhs_raw.shape[1]
        block_size = k // max(1, lhs.scale.shape[1] // 1)  # infer block size from scale shape
        # For drhs (tgmm): lhs needs columnwise quantization (along axis 0)
        lhs_for_drhs = qwix.quantize(
            lhs_raw, lhs.qvalue.dtype,
            tiled_axes={0: lhs_raw.shape[0], 1: block_size},
        )
        # For dlhs: rhs needs columnwise quantization (along K for transposed access)
        rhs_for_dlhs = qwix.quantize(
            rhs_raw, rhs.qvalue.dtype if isinstance(rhs, QArray) else lhs.qvalue.dtype,
            tiled_axes={0: 1, 1: 1, 2: block_size},
        )
        fp8_residuals = FP8Residuals(lhs_for_drhs, rhs_for_dlhs)

    out = backend.gmm(...)
    # ... existing residuals/activation logic ...
    return out, fp8_residuals if fp8_residuals is not None else (residuals if return_residuals else None)
```

**Step 3: Commit**

```bash
git add tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py
git commit -m "feat(gmm): add FP8Residuals for forward-save optimization"
```

---

### Task 7: VJP uses FP8 residuals

Modify `vjp()` in `base.py` to check for `FP8Residuals` and substitute pre-quantized lhs/rhs. Also update `__post_init__` in `pallas_mosaic_tpu.py` to pass residuals through correctly.

**Files:**
- Modify: `tokamax/_src/ops/ragged_dot/base.py:304-369`
- Modify: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py:103-119`

**Step 1: Modify vjp() to use FP8 residuals**

In `base.py`, update `vjp()`:

```python
def vjp(
    residuals: Residuals,
    out: jax.Array,
    dout: jax.Array,
    lhs: jax.Array | AsQArray,
    rhs: jax.Array | AsQArray,
    *,
    group_sizes: jax.Array,
    ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
    precision: CanonicalPrecision,
    preferred_element_type: jnp.dtype | None,
    activation: ActivationFunction | None = None,
    dlhs_ragged_dot: Callable[..., jax.Array] = RaggedDot(),
    drhs_ragged_dot: Callable[..., jax.Array] = RaggedDot(),
) -> tuple[jax.Array, jax.Array]:
  """Ragged dot VJP."""
  del out, preferred_element_type  # Unused.

  if activation is not None:
    _, activation_grad_fn = jax.vjp(activation, residuals)
    (dout,) = activation_grad_fn(dout)

  # Use pre-quantized FP8 inputs from forward if available
  from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu  # local import to avoid circular
  rhs_for_dlhs = rhs
  lhs_for_drhs = lhs
  if isinstance(residuals, pallas_mosaic_tpu.FP8Residuals):
    if residuals.rhs_for_dlhs is not None:
      rhs_for_dlhs = residuals.rhs_for_dlhs
    if residuals.lhs_for_drhs is not None:
      lhs_for_drhs = residuals.lhs_for_drhs

  # ... existing dimension number computation unchanged ...

  dlhs = dlhs_ragged_dot(
      dout,
      rhs_for_dlhs,  # <-- changed from rhs
      group_sizes=group_sizes,
      ragged_dot_dimension_numbers=...,
      precision=precision,
      preferred_element_type=lhs.dtype,
  )

  drhs = drhs_ragged_dot(
      lhs_for_drhs,  # <-- changed from lhs
      dout,
      group_sizes=group_sizes,
      ragged_dot_dimension_numbers=...,
      precision=precision,
      preferred_element_type=rhs.dtype,
  )
  return dlhs, drhs
```

Note: The `residuals` parameter in `vjp()` currently receives the activation residuals (line 297-301 in `_fwd`). With `FP8Residuals`, we need to handle both: the activation residuals and the FP8 pre-quantized inputs. Consider extending `FP8Residuals` to include the activation residual, or restructuring the residuals flow. Read the exact `_fwd` return and `bwd` unpacking in `op.py:262-270` to ensure correct wiring.

**Step 2: Update `__post_init__` in `pallas_mosaic_tpu.py`**

The current vjp binding at line 103-119 already passes `dlhs_ragged_dot=fn` and `drhs_ragged_dot=fn` correctly. No changes needed here — the vjp function signature already receives residuals as first argument.

**Step 3: Commit**

```bash
git add tokamax/_src/ops/ragged_dot/base.py tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py
git commit -m "feat(gmm): wire FP8 residuals through VJP for forward-save optimization"
```

---

### Task 8: DLHS/DRHS backward FP8 fast path

Modify the DLHS and DRHS branches in `_fwd` to handle pre-quantized FP8 QArray inputs (from FP8 residuals) without dequant->requant.

**Files:**
- Modify: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py:177-232`

**Step 1: Modify DLHS branch**

At line 177, when `rhs` is already an FP8 QArray (from residuals), use it directly:

```python
elif ragged_dot_dimension_numbers == DLHS_RAGGED_DOT_DIM_NUMS:  # dlhs
    if isinstance(lhs, jax.Array) and isinstance(rhs, QArray):
        if _is_fp8(rhs.qvalue.dtype):
            # rhs is already columnwise FP8 QArray from forward residuals
            # Quantize dout (lhs) as FP8 for the matmul
            lhs = maybe_quantize(lhs, (1, lhs.shape[1]))
        elif rhs.scale.shape[1] == 1:
            # existing full-channel fast path ...
        else:
            rhs = maybe_quantize(qwix.dequantize(rhs), (1, 1, rhs.shape[2]))
    else:
        lhs = maybe_quantize(lhs, (1, lhs.shape[1]))
        rhs = maybe_quantize(rhs, (1, 1, rhs.shape[2]))
    out = backend.gmm(...)
```

**Step 2: Modify DRHS branch**

At line 204, when `lhs_trans` is already an FP8 QArray (from residuals), use it directly:

```python
elif ragged_dot_dimension_numbers == DRHS_RAGGED_DOT_DIM_NUMS:  # drhs
    lhs_trans = jax.tree.map(lambda x: x.mT, lhs)
    if isinstance(lhs_trans, QArray) and isinstance(rhs, jax.Array):
        if _is_fp8(lhs_trans.qvalue.dtype):
            # lhs_trans is already columnwise FP8 QArray from forward residuals
            # Quantize dout (rhs) as FP8 for the matmul
            rhs = maybe_quantize(rhs, (rhs.shape[0], 1))
        elif lhs_trans.scale.shape[0] == 1:
            # existing full-channel fast path ...
        else:
            lhs_trans = qwix.dequantize(lhs_trans)
            lhs_trans = maybe_quantize(lhs_trans, (1, lhs_trans.shape[1]))
    else:
        lhs_trans = maybe_quantize(lhs_trans, (1, lhs_trans.shape[1]))
        rhs = maybe_quantize(rhs, (rhs.shape[0], 1))
    out = backend.tgmm(...)
```

**Step 3: Commit**

```bash
git add tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py
git commit -m "feat(gmm): add FP8 fast path in DLHS/DRHS backward branches"
```

---

### Task 9: End-to-end backward FP8 test

Test the full forward + backward pass with FP8 block-wise quantization.

**Files:**
- Test: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py`

**Step 1: Write the test**

Add to `FP8RaggedDotTest` in `pallas_mosaic_tpu_test.py`:

```python
  @parameterized.product(
      block_size=(128, 256),
  )
  def test_fp8_vjp(self, block_size):
    """Test FP8 block-wise forward + backward with forward-save optimization."""
    num_groups, m, k, n = 4, 512, 256, 256
    rng = jax.random.PRNGKey(42)
    a = jax.random.normal(rng, (m, k), dtype=jnp.bfloat16) * 0.1
    b = jax.random.normal(jax.random.split(rng)[0], (num_groups, k, n), dtype=jnp.bfloat16) * 0.1
    group_sizes = jnp.array([m // num_groups] * num_groups, dtype=jnp.int32)

    # FP8 quantized op
    a_tile = {0: 1, 1: block_size}
    b_tile = {0: 1, 1: block_size, 2: 1}
    a_q = quantization.AsQArray(a, jnp.float8_e4m3fn, tiled_axes=a_tile)
    b_q = quantization.AsQArray(b, jnp.float8_e4m3fn, tiled_axes=b_tile)

    op = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot()
    f = lambda a, b: op(a, b, group_sizes=group_sizes, preferred_element_type=jnp.float32)
    f_ref = lambda a, b: test_base.ref(a, b, group_sizes)

    # Forward
    actual = f(a_q, b_q)
    expected = f_ref(a, b)
    count = sum(group_sizes)
    chex.assert_trees_all_close(actual[:count], expected[:count], atol=1.0, rtol=0.2)

    # Backward
    dout = jnp.ones_like(actual)
    actual_grads = jax.grad(lambda a, b: jnp.sum(f(a, b)), argnums=(0, 1))(a_q, b_q)
    expected_grads = jax.grad(lambda a, b: jnp.sum(f_ref(a, b)), argnums=(0, 1))(a, b)

    chex.assert_trees_all_close(actual_grads[0], expected_grads[0], atol=2.0, rtol=0.3)
    chex.assert_trees_all_close(actual_grads[1], expected_grads[1], atol=2.0, rtol=0.3)
```

**Step 2: Run test**

Run: `python -m pytest tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py::FP8RaggedDotTest::test_fp8_vjp -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py
git commit -m "test(gmm): add end-to-end FP8 block-wise backward tests"
```

---

### Task 10: Add FP8 to existing test_quantized parameterization

Extend the existing `test_quantized` parameterization in `test_base.py` to include FP8.

**Files:**
- Modify: `tokamax/_src/ops/ragged_dot/test_base.py:191-208`

**Step 1: Add FP8 test method**

Add a new test method after `test_quantized`:

```python
  @parameterized.product(
      a_tile_shape=((1, 128), (1, 256)),
      b_tile_shape=((1, 1, 128), (1, 1, 256)),
      use_as_qarray=(True, False),
      activation=(None, relu),
  )
  def test_quantized_fp8(
      self,
      a_tile_shape,
      b_tile_shape,
      use_as_qarray,
      activation,
  ):
    self._test_quantized(
        "float8_e4m3fn", "float8_e4m3fn",
        a_tile_shape, b_tile_shape, use_as_qarray, activation,
    )
```

**Step 2: Run tests**

Run: `python -m pytest tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py -k test_quantized_fp8 -v`

Expected: PASS (all parameterized variants)

**Step 3: Commit**

```bash
git add tokamax/_src/ops/ragged_dot/test_base.py
git commit -m "test(gmm): add FP8 to test_quantized parameterization"
```

---

### Task 11: Run full test suite and fix regressions

Ensure all existing tests still pass after the changes.

**Step 1: Run full test suite**

Run: `python -m pytest tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py -v`

Expected: All tests PASS (existing int8/int4 tests + new FP8 tests)

**Step 2: If any failures, debug and fix**

Common issues to check:
- `FP8Residuals` type registration — ensure `jax.tree_util.register_dataclass` is correct
- `Residuals` type change — ensure `None` still works for non-FP8 paths
- tgmm subchannel masking — verify `_get_store_mask` works correctly with sub-tile sizes

**Step 3: Final commit**

```bash
git add -u
git commit -m "fix(gmm): address test regressions from FP8 changes"
```
