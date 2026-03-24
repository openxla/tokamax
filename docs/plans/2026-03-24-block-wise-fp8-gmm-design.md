# Block-wise FP8 Support for TPU GMM Kernel

**Date**: 2026-03-24
**Target**: TPUv7x Pallas Mosaic TPU kernel
**Configuration**: DeepSeek-V3 style (LHS 1x128 + RHS 128x128)

## Problem

The TPU GMM kernel's `quant_block_spec()` in `mosaic_tpu.py` rejects block-level
scales on non-reduction axes (Limitation 1, lines 112-121). This prevents RHS
(weight) 128x128 block-wise FP8 quantization, where the N axis has `eps=128`.

LHS (activation) 1x128 per-row subchannel quantization is already fully supported
by the existing `subchannel_iters` mechanism.

## Configuration (matching TE Float8BlockScaling defaults)

| Tensor | Block Shape | Scale Shape (for M x K or K x N) | eps on non-reduction axis |
|--------|-------------|-----------------------------------|---------------------------|
| LHS (activation) | 1x128 | (M, K/128) | eps_M=1 (already supported) |
| RHS (weight) | 128x128 | (groups, K/128, N/128) | eps_N=128 (needs extension) |
| FP8 dtype | E4M3 (float8_e4m3fn) | | |
| Scale dtype | FP32 | | |

## Approach: Extend `quant_block_spec` (Approach A)

### Core Change

Relax Limitation 1 in `quant_block_spec()` to allow block-level scales on
non-reduction axes when `eps` is compatible with the tile size (i.e.,
`eps % tile_size == 0 or tile_size % eps == 0`).

The underlying `_get_scale_tile_info()` already handles the math correctly for
this case. The only barrier is the `NotImplementedError` guard.

### Why this works with tile = block = 128

With tm=128, tk=128, tn=128:
- `subchannel_iters = max(1, tk//lhs_eps_k, tk//rhs_eps_k) = max(1, 1, 1) = 1`
- Each tile corresponds to exactly one block -> one scalar scale per tile per operand
- `_get_scale_tile_info` computes `scales_per_tile=1`, inflates to TPU addressable size
- `_scale_out_by_scale` broadcasts the inflated scalar to output shape and multiplies

### Kernel execution path (per K-tile, loop runs once)

```
1. Pallas loads:
   lhs_ref: QArray(qvalue=[128,128] fp8, scale=[128,1] per-row)    # inflate to [sublane, lanes]
   rhs_ref: QArray(qvalue=[128,128] fp8, scale=[1,1] per-block)    # inflate to [sublane, lanes]

2. _maybe_get_subslice(subslice_count=1): returns entire tile

3. Unpack QArray: scales = [lhs.scale, rhs.scale], lhs/rhs = qvalues

4. dot(lhs_fp8, rhs_fp8) -> out [128,128] (float32, TPU MXU native f32 accumulation)

5. out *= lhs_scale  (broadcast [128,1] -> [128,128])
6. out *= rhs_scale  (broadcast scalar -> [128,128])

7. acc_scratch += out  (float32 accumulation across K-tiles)
```

### Precision comparison with GPU (DeepGEMM)

| Aspect | DeepGEMM (GPU) | Our TPU kernel |
|--------|----------------|----------------|
| Inner accumulation | ~14-bit (Tensor Core) | float32 (MXU) |
| Needs promotion? | Yes, every 128 K elements | No, MXU native f32 |
| Scale application | After dot, composite scale | After dot, sequential scale multiply |
| Cross K-tile accumulation | float32 (CUDA Core) | float32 (acc_scratch) |

### Backward pass (tgmm)

- With tk=128, `subchannel_iters=1`, so the existing tgmm rejection
  (`subchannel_iters != 1 not supported`) is NOT triggered
- tgmm's `quant_block_spec` needs the same Limitation 1 relaxation
- Transpose/requantization is handled by the caller (not kernel's responsibility)
  - TE uses dual quantization (rowwise + columnwise from high-precision input)
  - DeepSeek-V3 uses power-of-2 scales selectively for specific activations

## Files to modify

1. **`tokamax/_src/mosaic_tpu.py`** (~5 lines)
   - Relax Limitation 1 in `quant_block_spec()` (lines 112-121)
   - Allow `eps` values where `eps % tile_size == 0 or tile_size % eps == 0`

2. **`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py`** (0 lines)
   - No kernel changes needed

3. **Test file** (`pallas_mosaic_tpu_test.py`) (~20-30 lines)
   - Add test cases for 1x128 LHS + 128x128 RHS block-wise FP8

## Test plan

1. Basic correctness: K=128, K=256, K=1024
2. K not divisible by 128: K=384, boundary masking
3. Multiple groups: groups > 1
4. Compare with BF16 reference values
5. tgmm backward with subchannel_iters=1

## Out of scope

- LHS 128x128 block (TE explicitly uses 1x128 for activations, prohibits 2D x 2D)
- Arbitrary block sizes (constrained to tile = block = 128)
- Quantization strategy (power-of-2 scales, dual quantization — caller's responsibility)
