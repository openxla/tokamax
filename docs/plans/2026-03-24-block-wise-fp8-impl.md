# Block-wise FP8 GMM Kernel Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable DeepSeek-V3 style block-wise FP8 (LHS 1x128 + RHS 128x128) in the TPU Pallas Mosaic GMM kernel.

**Architecture:** Relax the `quant_block_spec()` Limitation 1 guard in `mosaic_tpu.py` to allow block-level scales on non-reduction axes. The underlying `_get_scale_tile_info()` math already handles this case. Also update the test helper `_is_scale_tiling_supported` which mirrors the same constraint.

**Tech Stack:** JAX, Pallas Mosaic TPU, qwix (quantization library)

**Design doc:** `docs/plans/2026-03-24-block-wise-fp8-gmm-design.md`

---

### Task 1: Relax `quant_block_spec` Limitation 1

**Files:**
- Modify: `tokamax/_src/mosaic_tpu.py:112-121`

**Step 1: Read and understand the current constraint**

Read `tokamax/_src/mosaic_tpu.py` lines 100-162 to confirm the constraint location.

**Step 2: Relax the constraint**

Replace lines 112-121:

```python
  # Limitation 1: Currently, we only support full-axis scales or 1 scale per
  # each element for non-reduction axes, this is supported by the block-spec,
  # but not by the kernel implementation.
  for axis, eps in enumerate(eps_list):
    if axis != reduction_axis and eps not in (1, x_values.shape[axis]):
      raise NotImplementedError(
          "Non-reduction axes must have an either 1 scale per each element or a"
          " scalar full-axis scale, but"
          f" {jax.tree.map(jax.typeof, x)=} for {reduction_axis=}."
      )
```

With:

```python
  # Non-reduction axes: allow block-level scales as long as eps is compatible
  # with the tile size (checked by _get_scale_tile_info via its assert).
  # This enables DeepSeek-V3 style 128x128 block-wise FP8 on the weight's
  # N axis while keeping the reduction axis (K) subchannel support unchanged.
  for axis, eps in enumerate(eps_list):
    tile_size = tile_sizes[axis]
    if axis != reduction_axis and eps not in (1, x_values.shape[axis]):
      if tile_size is not None and not (eps % tile_size == 0 or tile_size % eps == 0):
        raise NotImplementedError(
            "Non-reduction axis block-level scales require eps and tile_size"
            " to be compatible (one must divide the other), but"
            f" {eps=}, {tile_size=}, {axis=},"
            f" {jax.tree.map(jax.typeof, x)=} for {reduction_axis=}."
        )
```

**Step 3: Commit**

```bash
git add tokamax/_src/mosaic_tpu.py
git commit -m "feat: relax quant_block_spec to allow block-level non-reduction scales"
```

---

### Task 2: Update test helper `_is_scale_tiling_supported`

**Files:**
- Modify: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py:33-45`

**Step 1: Read and understand the test helper**

Read `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py` lines 33-65.

The function `_is_scale_tiling_supported` mirrors Limitation 1 and will cause new test configurations to be skipped. Line 43 has the same constraint:
```python
if ax != axis and not (eps == 1 or eps == x.qvalue.shape[ax]):
    return False
```

**Step 2: Relax the test helper constraint**

Replace lines 33-45:

```python
def _is_scale_tiling_supported(x: qwix.QArray, axis: int) -> bool:
  min_addressable_sizes = (
      [1] * x.ndim
      + [common._adaptive_sublane_size(), pltpu.get_tpu_info().num_lanes]
  )[-x.ndim :]
  cdiv = lambda x, y: (x + y - 1) // y
  eps_list = [cdiv(x, y) for x, y in zip(x.qvalue.shape, x.scale.shape)]
  for ax, (mas, eps) in enumerate(zip(min_addressable_sizes, eps_list)):
    if eps != 1 and eps % mas != 0:
      return False
    if ax != axis and not (eps == 1 or eps == x.qvalue.shape[ax]):
      return False
  return True
```

With:

```python
def _is_scale_tiling_supported(x: qwix.QArray, axis: int) -> bool:
  min_addressable_sizes = (
      [1] * x.ndim
      + [common._adaptive_sublane_size(), pltpu.get_tpu_info().num_lanes]
  )[-x.ndim :]
  cdiv = lambda x, y: (x + y - 1) // y
  eps_list = [cdiv(x, y) for x, y in zip(x.qvalue.shape, x.scale.shape)]
  for ax, (mas, eps) in enumerate(zip(min_addressable_sizes, eps_list)):
    if eps != 1 and eps % mas != 0:
      return False
  return True
```

The only change: remove the non-reduction axis constraint (line 43-44). The `eps % mas` check on line 41 is sufficient — it ensures TPU addressability.

**Step 3: Commit**

```bash
git add tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py
git commit -m "test: relax _is_scale_tiling_supported for block-wise non-reduction scales"
```

---

### Task 3: Add block-wise FP8 test cases

**Files:**
- Modify: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py` (add new test method)

**Step 1: Write the test**

Add the following test method to the `PallasMosaicTpuRaggedDotTest` class, after the existing `_test_quantized` override (after line 119):

```python
  @parameterized.product(
      use_as_qarray=(True, False),
      task=(
          (8, 512, 128, 512),   # K=128: single K-tile
          (8, 512, 256, 512),   # K=256: two K-tiles
          (8, 512, 1024, 512),  # K=1024: multiple K-tiles
          (8, 512, 384, 512),   # K=384: K not divisible by 128
      ),
  )
  def test_blockwise_fp8(self, use_as_qarray, task):
    """DeepSeek-V3 style: LHS 1x128 activation + RHS 128x128 weight."""
    with test_base.override_chex_args(atol=0.4, rtol=0.1):
      super()._test_quantized(
          "float8_e4m3fn",
          "float8_e4m3fn",
          (1, 128),        # LHS: per-row, per-128-channel (1x128)
          (1, 128, 128),   # RHS: 128x128 block
          use_as_qarray,
          None,             # no activation
          task,
      )
```

This covers:
- Basic correctness with K=128, K=256, K=1024
- K not divisible by 128 (K=384) for boundary masking
- Both `use_as_qarray=True` (lazy quantization) and `False` (eager quantization)
- Multiple groups (num_groups=8 in all tasks)

**Step 2: Commit**

```bash
git add tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py
git commit -m "test: add block-wise FP8 test (DeepSeek-V3 style 1x128 + 128x128)"
```

---

### Task 4: Run tests and verify

**Step 1: Run the new test on TPU**

```bash
python tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py \
  -k test_blockwise_fp8 -v
```

Expected: All parameterized variants PASS.

**Step 2: Run the existing quantized tests to verify no regression**

```bash
python tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py \
  -k test_quantized -v
```

Expected: All existing tests continue to PASS (the relaxed constraint should not affect int8/int4 tests since their non-reduction axis eps values are either 1 or per-element).

**Step 3: If tests fail, debug and fix**

Check:
- Is the failure in the new test or an existing one?
- Does `_get_scale_tile_info` assert on line 80 for the new eps/tile_size combo?
- Does `_scale_out_by_scale` handle the inflated scale shape correctly?
- Are the tolerances (atol=0.4, rtol=0.1) too tight for FP8?

**Step 4: Final commit with all passing tests**

```bash
git add -A
git commit -m "feat: block-wise FP8 support for TPU GMM kernel (DeepSeek-V3 style)"
```
