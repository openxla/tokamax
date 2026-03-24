# GMM FP8 Block-wise Quantization Design

## Context

Add FP8 block-wise quantization support to GMM (ragged dot) on TPU v7x (gen7+),
covering forward and backward passes (gmm + dlhs + tgmm). TPU v7x has native
FP8 dot support.

**Scope:**
- FP8 variant: `float8_e4m3fn`
- Block sizes: configurable (128/256/512)
- Use case: training (forward + backward)
- Target: TPU v7x (gen7+, native FP8 dot)

## Design

### 1. `_quantize_as` (pallas_mosaic_tpu_kernel.py)

Skip `jnp.round` for FP8 types. FP8 is floating-point; `.astype(fp8_dtype)`
handles rounding natively.

```python
qvalue = x * inv_scales
if jnp.issubdtype(qdtype, jnp.integer):
    qvalue = jnp.round(qvalue)
return QArray(qvalue.astype(qdtype), scales)
```

### 2. `dot()` function (pallas_mosaic_tpu_kernel.py)

Add FP8 native path in both gmm and tgmm `dot()` functions. TPU v7x supports
FP8 dot natively, so FP8 operands pass through directly without upcast.

```python
is_fp8 = (jnp.issubdtype(x.dtype, jnp.floating) and x.dtype.itemsize == 1) or \
         (jnp.issubdtype(y.dtype, jnp.floating) and y.dtype.itemsize == 1)
if is_fp8:
    precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    preferred_element_type = jnp.float32
elif precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
    # existing BF16 path ...
```

### 3. Forward-save optimization (pallas_mosaic_tpu.py)

Aligned with Transformer Engine's approach: during forward, quantize lhs/rhs in
both rowwise (for forward GEMM) and columnwise (for backward GEMM) directions.
Save columnwise versions as residuals. This avoids dequant->requant in backward.

**New residuals type:**

```python
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class FP8Residuals:
    lhs_for_drhs: QArray | None  # lhs columnwise-quantized, for drhs (tgmm)
    rhs_for_dlhs: QArray | None  # rhs columnwise-quantized, for dlhs
```

**Forward (`DEFAULT_RAGGED_DOT_DIM_NUMS`):**
- Rowwise quantization of lhs/rhs for forward gmm (existing logic)
- Additionally: columnwise quantization of lhs (for drhs/tgmm) and rhs (for dlhs)
- Save as `FP8Residuals` in return value

**DLHS branch:** When rhs arrives as FP8 QArray from residuals (already
columnwise), use directly instead of dequant->requant.

**DRHS branch:** When lhs arrives as FP8 QArray from residuals (already
columnwise), use directly instead of dequant->requant.

### 4. VJP modification (base.py)

`vjp()` checks residuals type. If `FP8Residuals`, substitute pre-quantized
versions for lhs/rhs before calling dlhs_ragged_dot/drhs_ragged_dot.

```python
if isinstance(residuals, FP8Residuals):
    rhs_for_dlhs = residuals.rhs_for_dlhs or rhs
    lhs_for_drhs = residuals.lhs_for_drhs or lhs
```

### 5. tgmm subchannel support (pallas_mosaic_tpu_kernel.py)

Remove `subchannel_iters != 1` NotImplementedError. Implement subchannel
iteration in `_do()`:

- Subchannel iteration along M axis (axis=0) in tgmm
- Each sub-iteration: slice lhs/rhs -> optional quantize -> unpack QArray ->
  mask -> dot -> scale -> accumulate
- `_get_store_mask` adjusted for sub-tile size (`sc_tile = tm // subchannel_iters`)
- Group boundary / subchannel boundary intersection requires careful masking

### 6. quant_block_spec (mosaic_tpu.py)

No changes needed. All target block sizes (128/256/512) satisfy TPU v7x
constraints:
- Lane axis min addressable = 128: block sizes >= 128 pass
- Sublane axis min addressable = 16: block sizes >= 16 pass

### 7. Helper

```python
def _is_fp8(dtype) -> bool:
    dtype = jnp.dtype(dtype)
    return jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize == 1
```

## Files to modify

| File | Changes |
|------|---------|
| `pallas_mosaic_tpu_kernel.py` | `_quantize_as` FP8 skip round; `dot()` FP8 native path; tgmm subchannel |
| `pallas_mosaic_tpu.py` | `FP8Residuals` type; `_fwd` FP8 paths in all 3 branches; `Residuals` type |
| `base.py` | `vjp()` uses FP8 residuals to substitute lhs/rhs |
| `test_base.py` | FP8 test parameterization |
| `pallas_mosaic_tpu_test.py` | FP8 TPU-specific tests |

## Files unchanged

- `mosaic_tpu.py` (quant_block_spec): existing constraints sufficient
- `precision.py`: no FP8 presets needed

## Testing

1. Forward FP8 block-wise (precision vs bf16 baseline)
2. Backward dlhs with forward-saved FP8 QArray
3. Backward drhs/tgmm with subchannel FP8
4. End-to-end `jax.grad` numerical gradient check
5. Block sizes: 128, 256, 512
