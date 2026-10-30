# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flash Attention with Pallas Mosaic GPU for SM80 (Ampere / A100).

Supports two execution pipelines tailored to head dimension and SM80 resources:
1. Cooperative Q-Parallel FlashAttention-2 mode (`v_split0 == 0`, default for
   `head_dim <= 256`):
   - `num_wgs == 2` (256 threads, `block_q >= 128`): Two warpgroups process
     independent Q subtiles in registers while cooperatively sharing multi-stage
     `K` and `V` ring buffers in SMEM (cutting GMEM->SMEM KV bandwidth in half),
     keeping `P` 100% in registers via zero-cost MMA_ACC->MMA_LHS relayout.
   - `num_wgs == 1` (128 threads, `block_q >= 64`): Single-warpgroup CTA with
     zero inter-warpgroup barriers for short sequences or high-CTA-occupancy
     regimes.
2. Split-Head mode (`v_split0 > 0`, used for `head_dim > 256`, e.g. `512`):
   - Splits output columns `[0:v_split0]` and `[v_split0:head_dim_out]` across
     two warpgroups via a multi-slot P/alpha SMEM ring buffer so accumulator
     register pressure per thread stays well within 255 registers.

All paths index native `(T, H, D)` tensors directly via `cp.async` without GMEM
transposes, pad `head_dim` and `head_dim_out` independently to multiples of 16,
stage 2D `bias`/`mask` tiles via `cp.async`, perform tile-level early exit and
boundary mask peeling for `query_seq_lengths` and `key_value_seq_lengths`, and
write outputs via swizzled SMEM (`SMEM_GMEM_COPY`) for coalesced 128-bit stores.
"""

import dataclasses
import math
from typing import Annotated, Any, Literal

import jax
from jax import lax
import jax.experimental.mosaic.gpu as mgpu
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src import mosaic_gpu as mgpu_lib
from tokamax._src import shape as shape_lib
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_mosaic_gpu_common as common

# pylint: disable=cell-var-from-loop

Residuals = base.Residuals

_MAX_SMEM: int = 162 * 1024
_FORBID_EXTRA = pydantic.ConfigDict(extra="forbid")


@pydantic.dataclasses.dataclass(
    frozen=True, kw_only=True, slots=True, config=_FORBID_EXTRA
)
class Config(common.ConfigBase):
  """Configuration parameters for Pallas-Mosaic-GPU kernels on SM80 GPUs.

  Attributes:
    block_q: CTA tile size along Q sequence length (multiple of 64).
    block_kv: CTA tile size along KV sequence length (multiple of 32).
    num_stages: Default number of SMEM pipeline stages for K and V.
    stages_k: Number of pipeline stages for K in SMEM (defaults to num_stages).
    stages_v: Number of pipeline stages for V in SMEM (defaults to num_stages).
    v_split0: Output channels assigned to WG0 in split-head mode (`0` selects
      Q-parallel FlashAttention-2 mode).
    q_smem: Whether Q remains resident in SMEM instead of registers.
    num_p_slots: Number of P/alpha ring buffer slots in split-head mode.
    num_wgs: Number of Q-parallel warpgroups per CTA when `v_split0 == 0`.
  """

  block_kv: Annotated[int, pydantic.Field(multiple_of=32, gt=0)] = 64
  num_stages: pydantic.PositiveInt = 2
  stages_k: pydantic.PositiveInt | None = None
  stages_v: pydantic.PositiveInt | None = None
  v_split0: Annotated[int, pydantic.Field(multiple_of=32, ge=0)] = 0
  q_smem: pydantic.StrictBool = False
  num_p_slots: pydantic.PositiveInt = 2
  num_wgs: Literal[1, 2] = 2


def _pad_head_dims(q, k, v):
  q = common.pad_head_dim_to_next_multiple_of_min_swizzle(q)
  k = common.pad_head_dim_to_next_multiple_of_min_swizzle(k)
  v = common.pad_head_dim_to_next_multiple_of_min_swizzle(v)
  return q, k, v


def _resolve_mode_params(
    head_dim_out: int,
    kv_seq_len: int,
    config: Config,
    *,
    has_bm_smem: bool = False,
) -> tuple[int, int, int, int, int]:
  """Returns `(v_split0, effective_wgs, wg_q, stages_k, stages_v)`."""
  block_q = config.block_q
  block_kv = config.block_kv
  num_kv_blocks = max(1, pl.cdiv(kv_seq_len, block_kv))
  stages_k = max(1, min(config.stages_k or config.num_stages, num_kv_blocks))
  stages_v = max(1, min(config.stages_v or config.num_stages, num_kv_blocks))

  v_split0 = config.v_split0
  if v_split0 >= head_dim_out:
    v_split0 = 0
  if v_split0 > 0:
    return v_split0, 2, block_q, stages_k, stages_v

  effective_wgs = min(config.num_wgs, block_q // 64)
  wg_q = block_q // effective_wgs
  if effective_wgs == 2 and not has_bm_smem:
    return 0, effective_wgs, wg_q, stages_k, stages_v
  stages = min(stages_k, stages_v)
  return 0, effective_wgs, wg_q, stages, stages


def _choose_k_swizzle(head_dim: int, block_kv: int, elem_bits: int) -> int:
  del block_kv
  swizzle = plgpu.find_swizzle(head_dim * elem_bits, "k")
  if head_dim > 192 and swizzle > 64:
    return 64
  return swizzle


def _choose_epi_cols(cols: int, out_dtype: jnp.dtype) -> int:
  out_bits = mgpu_lib.num_bits(out_dtype)
  if out_bits == 32 and cols > 64:
    for step in (64, 32, 16):
      if cols % step == 0:
        return step
  return cols


def _can_stage_2d_bias(bias, block_kv: int) -> bool:
  if bias is None or bias.shape[-2] == 1 or bias.shape[-1] == 1:
    return False
  elem_bits = mgpu_lib.num_bits(bias.dtype)
  return (block_kv * elem_bits) % 128 == 0


def _can_stage_2d_mask(mask, block_kv: int) -> bool:
  if mask is None or mask.shape[-2] == 1 or mask.shape[-1] == 1:
    return False
  return (block_kv * 8) >= 256 and (block_kv * 8) % 256 == 0


def _get_scratch_types(
    q,
    k,
    v,
    bias,
    mask,
    out_dtype: jnp.dtype,
    config: Config,
    *,
    stage_2d_bias: bool = True,
    stage_2d_mask: bool = True,
) -> dict[str, Any]:
  """Builds SMEM scratch allocations for either Q-parallel or split-head mode."""
  q, k, v = jax.eval_shape(_pad_head_dims, q, k, v)
  dtype = q.dtype
  elem_bits = mgpu_lib.num_bits(dtype)
  head_dim = q.shape[-1]
  head_dim_out = v.shape[-1]
  block_q = config.block_q
  block_kv = config.block_kv
  use_2d_bias = stage_2d_bias and _can_stage_2d_bias(bias, block_kv)
  use_2d_mask = stage_2d_mask and _can_stage_2d_mask(mask, block_kv)
  kv_seq_padded = pl.cdiv(k.shape[-3], block_kv) * block_kv
  v_split0, effective_wgs, wg_q, stages_k, stages_v = _resolve_mode_params(
      head_dim_out,
      kv_seq_padded,
      config,
      has_bm_smem=(use_2d_bias or use_2d_mask),
  )
  tiled_smem = mgpu_lib.tiled_swizzled_smem

  k_swizzle = _choose_k_swizzle(head_dim, block_kv, elem_bits)
  q_swizzle = (
      k_swizzle
      if config.q_smem
      else plgpu.find_swizzle(head_dim * elem_bits, "q")
  )

  if v_split0 > 0:
    v_split1 = head_dim_out - v_split0
    k_scratch = tiled_smem(
        (stages_k, block_kv, head_dim), dtype, "k", swizzle=k_swizzle
    )
    q_scratch = tiled_smem(
        (block_q, head_dim), dtype, "q", swizzle=q_swizzle
    )
    v0_scratch = tiled_smem((stages_v, block_kv, v_split0), dtype, "v0")
    v1_scratch = tiled_smem((stages_v, block_kv, v_split1), dtype, "v1")
    epi0_cols = _choose_epi_cols(v_split0, out_dtype)
    epi1_cols = _choose_epi_cols(v_split1, out_dtype)
    epi0_scratch = tiled_smem((block_q, epi0_cols), out_dtype, "epi0")
    epi1_scratch = tiled_smem((block_q, epi1_cols), out_dtype, "epi1")
    num_p_slots = config.num_p_slots
    p_scratch = tiled_smem((num_p_slots, block_q, block_kv), dtype, "p")
    al_scratch = plgpu.SMEM((num_p_slots + 1, block_q), jnp.float32)

    if config.q_smem:
      main_epi_union = plgpu.RefUnion(
          (q_scratch, k_scratch, v0_scratch, v1_scratch, p_scratch),
          (epi0_scratch, epi1_scratch),
      )
    else:
      main_epi_union = plgpu.RefUnion(
          (q_scratch,),
          (k_scratch, v0_scratch, v1_scratch, p_scratch),
          (epi0_scratch, epi1_scratch),
      )

    return dict(
        main_epi_union=main_epi_union,
        al_smem=al_scratch,
        p_ready_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=num_p_slots),
        p_done_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=num_p_slots),
    )

  # Q-Parallel FlashAttention-2 mode (`v_split0 == 0`).
  k_scratch = tiled_smem(
      (stages_k, block_kv, head_dim), dtype, "k", swizzle=k_swizzle
  )
  v_scratch = tiled_smem((stages_v, block_kv, head_dim_out), dtype, "v")
  q_scratch = tiled_smem(
      (effective_wgs, wg_q, head_dim), dtype, "q", swizzle=q_swizzle
  )
  epi_cols = _choose_epi_cols(head_dim_out, out_dtype)
  epi_scratch = tiled_smem(
      (effective_wgs, wg_q, epi_cols), out_dtype, "epi"
  )

  extras: dict[str, Any] = {}
  if use_2d_bias:
    extras["bias_smem"] = tiled_smem(
        (stages_k, effective_wgs, wg_q, block_kv), bias.dtype, "bias"
    )
  if use_2d_mask:
    extras["mask_smem"] = tiled_smem(
        (stages_k, effective_wgs, wg_q, block_kv), jnp.int8, "mask"
    )

  if config.q_smem:
    main_epi_union = plgpu.RefUnion(
        (q_scratch, k_scratch, v_scratch, extras),
        (epi_scratch,),
    )
  else:
    main_epi_union = plgpu.RefUnion(
        (q_scratch,),
        (k_scratch, v_scratch, extras),
        (epi_scratch,),
    )

  return dict(main_epi_union=main_epi_union)


def _estimate_shared_mem_usage_bytes(
    ba: op.BoundArguments, config: Config
) -> int:
  """Estimates dynamic SMEM usage in bytes for `config`."""
  q, k, v, bias, mask = common.eval_input_shapes(
      ba,
      fold_q_sequence_heads=config.fold_q_sequence_heads,
      split_k=config.split_k,
      extract_q_range_flag=True,
  )
  out_dtype = ba.args[0].dtype
  scratch_types = _get_scratch_types(q, k, v, bias, mask, out_dtype, config)
  return mgpu_lib.estimate_smem_bytes(scratch_types)


def get_heuristics_config(
    ba: op.BoundArguments, fold_q_sequence_heads: bool = False
) -> Config:
  """Returns a heuristic configuration for flash attention on SM80 GPUs."""
  q, k, v = ba.args[:3]
  bias = ba.kwargs.get("bias")
  batch_size = math.prod(q.shape[:-3]) * math.prod(ba.vmap_axis_sizes)
  q_seq_len, num_q_heads, orig_d = q.shape[-3:]
  kv_seq_len, num_kv_heads, orig_d_out = k.shape[-3], k.shape[-2], v.shape[-1]

  if fold_q_sequence_heads:
    q_seq_len *= num_q_heads // num_kv_heads
    num_q_heads = num_kv_heads

  head_dim = pl.cdiv(orig_d, 16) * 16
  head_dim_out = pl.cdiv(orig_d_out, 16) * 16
  max_d = max(head_dim, head_dim_out)
  has_2d_bias = (
      bias is not None
      and bias.ndim >= 2
      and bias.shape[-2] != 1
      and bias.shape[-1] != 1
  )

  # Split-head mode for very large head dims (`D > 256`, e.g. `512`).
  if head_dim_out > 256 or head_dim > 256:
    v_split0 = max(32, (head_dim_out // 2 // 32) * 32)
    cfg = Config(
        block_q=64,
        block_kv=32,
        num_stages=1,
        v_split0=v_split0,
        q_smem=True,
        num_p_slots=2,
        fold_q_sequence_heads=fold_q_sequence_heads,
    )
    if _estimate_shared_mem_usage_bytes(ba, cfg) <= _MAX_SMEM:
      return cfg
    return dataclasses.replace(cfg, q_smem=False)

  # Use 2-WG cooperative KV (`block_q=128`) whenever there is enough Q work to
  # saturate SMs; fall back to 1-WG (`block_q=64`) for short sequences or tiny
  # batch*heads grids.
  ctas_at_128 = batch_size * num_q_heads * pl.cdiv(q_seq_len, 128)
  prefer_128_q = q_seq_len >= 128 and (ctas_at_128 >= 54 or q_seq_len >= 512)
  block_q = 128 if prefer_128_q else 64
  num_wgs: Literal[1, 2] = 2 if block_q == 128 else 1

  if max_d > 192:  # e.g. head_dim = 256
    candidates = [
        Config(
            block_q=block_q,
            block_kv=64,
            num_stages=1,
            q_smem=True,
            num_wgs=num_wgs,
            fold_q_sequence_heads=fold_q_sequence_heads,
        ),
        Config(
            block_q=64,
            block_kv=64,
            num_stages=1,
            q_smem=True,
            num_wgs=1,
            fold_q_sequence_heads=fold_q_sequence_heads,
        ),
    ]
  elif max_d > 128:  # e.g. deepseek2 (192 / 128)
    candidates = [
        Config(
            block_q=block_q,
            block_kv=64,
            num_stages=s,
            q_smem=False,
            num_wgs=num_wgs,
            fold_q_sequence_heads=fold_q_sequence_heads,
        )
        for s in (3, 2, 1)
    ]
  elif max_d > 64:  # e.g. head_dim = 112, 128 (mixtral, FAv3 D=128)
    candidates = []
    if not has_2d_bias and block_q == 128 and kv_seq_len >= 2048:
      candidates.extend([
          Config(
              block_q=128,
              block_kv=128,
              num_stages=2,
              stages_k=3,
              stages_v=2,
              q_smem=False,
              num_wgs=2,
              fold_q_sequence_heads=fold_q_sequence_heads,
          ),
          Config(
              block_q=128,
              block_kv=128,
              num_stages=2,
              q_smem=False,
              num_wgs=2,
              fold_q_sequence_heads=fold_q_sequence_heads,
          ),
      ])
    candidates.extend([
        Config(
            block_q=block_q,
            block_kv=64,
            num_stages=s,
            q_smem=False,
            num_wgs=num_wgs,
            fold_q_sequence_heads=fold_q_sequence_heads,
        )
        for s in (3, 2, 1)
    ])
  else:  # head_dim <= 64 (FAv3 D=64, AlphaFold D=16, 32, 64)
    if (
        (max_d <= 32 or not has_2d_bias)
        and batch_size * num_q_heads * pl.cdiv(q_seq_len, 64) >= 108
    ):
      bq, nwgs = 64, 1
    else:
      bq, nwgs = block_q, num_wgs
    candidates = [
        Config(
            block_q=bq,
            block_kv=64,
            num_stages=s,
            q_smem=False,
            num_wgs=nwgs,
            fold_q_sequence_heads=fold_q_sequence_heads,
        )
        for s in (3, 2, 1)
    ]

  for cfg in candidates:
    if _estimate_shared_mem_usage_bytes(ba, cfg) <= _MAX_SMEM:
      return cfg

  return Config(
      block_q=64,
      block_kv=32,
      num_stages=1,
      v_split0=0,
      q_smem=False,
      num_wgs=1,
      fold_q_sequence_heads=fold_q_sequence_heads,
  )


def _async_copy_gmem_to_smem(src, dst):
  common.cp_async_gmem_to_smem(src, dst)


def _cta_barrier():
  mgpu_lib.bar_sync(8, 256)


def _relayout_p_to_lhs(p: jax.Array, mma_acc, mma_lhs) -> jax.Array:
  @plgpu.inline_mgpu(
      arg_types=(mma_acc,),
      return_type=plgpu.ShapeDtypeStruct(p.shape, p.dtype, layout=mma_lhs),
  )
  def relabel(ctx, x):
    del ctx
    with mgpu_lib.ir.Context():
      target = mma_lhs.to_mgpu()
    return mgpu.FragmentedArray(
        _registers=x.registers.reshape(target.registers_shape(x.shape)),
        _layout=target,
        _is_signed=x.is_signed,
    )

  return relabel(p)


def _compute_qk(
    q_reg: jax.Array | None,
    q_smem_wg,
    k_smem_slot,
    wg_q: int,
    block_kv: int,
    head_dim: int,
    step_d: int,
    mma_acc,
    mma_rhs,
    mma_lhs,
) -> jax.Array:
  """Computes `S = Q @ K.T` chunked along `head_dim` to bound RHS registers."""
  acc = plgpu.layout_cast(jnp.zeros((wg_q, block_kv), jnp.float32), mma_acc)
  for d0 in range(0, head_dim, step_d):
    d_slice = pl.ds(d0, step_d)
    q_chunk = (
        q_reg[:, d0 : d0 + step_d]
        if q_reg is not None
        else plgpu.load(q_smem_wg.at[:, d_slice], layout=mma_lhs)
    )
    k_chunk = plgpu.load(k_smem_slot.at[:, d_slice].T, layout=mma_rhs)
    acc = plgpu.mma(acc, q_chunk, k_chunk)
  return acc


def _compute_pv_step(
    p_lhs: jax.Array,
    acc: jax.Array,
    alpha: jax.Array,
    v_smem_slot,
    out_shape: tuple[int, int],
    step_kv: int,
    mma_acc,
    mma_rhs,
) -> jax.Array:
  """Rescales `acc` by `alpha` and accumulates `P @ V` chunked along `block_kv`."""
  acc = acc * plgpu.layout_cast(
      lax.broadcast_in_dim(alpha, out_shape, (0,)), mma_acc
  )
  acc = plgpu.layout_cast(acc, mma_acc)
  block_kv = p_lhs.shape[1]
  for k0 in range(0, block_kv, step_kv):
    v_chunk = plgpu.load(
        v_smem_slot.at[pl.ds(k0, step_kv), :], layout=mma_rhs
    )
    acc = plgpu.mma(acc, p_lhs[:, k0 : k0 + step_kv], v_chunk)
  return acc


def _online_softmax_step(
    s: jax.Array, m_prev: jax.Array, l_prev: jax.Array, mma_acc
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Base-2 online softmax immune to `(-inf) - (-inf)` NaNs on masked rows."""
  m_curr = jnp.maximum(m_prev, jnp.max(s, axis=-1))
  m_safe = jnp.where(m_curr == -jnp.inf, 0.0, m_curr)
  alpha = jnp.exp2(m_prev - m_safe)
  p = jnp.exp2(
      s
      - plgpu.layout_cast(lax.broadcast_in_dim(m_safe, s.shape, (0,)), mma_acc)
  )
  l_curr = l_prev * alpha + jnp.sum(p, axis=-1)
  return m_curr, alpha, p, l_curr


def _init_softmax_state(rows: int, mma_row) -> tuple[jax.Array, jax.Array]:
  return (
      plgpu.layout_cast(jnp.full((rows,), -jnp.inf, jnp.float32), mma_row),
      plgpu.layout_cast(jnp.zeros((rows,), jnp.float32), mma_row),
  )


def _store_via_swizzled_smem(
    gmem_ref,
    val: jax.Array,
    epi_smem,
    out_dtype: jnp.dtype,
    epi_swizzle: int,
):
  """Stages `MMA_ACC` register tile through swizzled SMEM for 128b GMEM stores."""
  rows, cols = val.shape
  epi_cols = epi_smem.shape[-1]
  copy_layout = plgpu.Layout.SMEM_GMEM_COPY(
      (rows, epi_cols), out_dtype, swizzle=epi_swizzle
  )
  for c0 in range(0, cols, epi_cols):
    plgpu.store(
        epi_smem,
        val[:, c0 : c0 + epi_cols].astype(out_dtype),
        optimized=False,
    )
    mgpu_lib.warpgroup_barrier()
    staged = plgpu.load(epi_smem, layout=copy_layout, optimized=False)
    plgpu.store(gmem_ref.at[:, pl.ds(c0, epi_cols)], staged, optimized=False)
    if c0 + epi_cols < cols:
      mgpu_lib.warpgroup_barrier()


def _write_output_epilogue(
    out_gtile,
    acc: jax.Array,
    l_final: jax.Array,
    epi_smem,
    out_dtype: jnp.dtype,
    epi_swizzle: int,
    mma_acc,
    normalize_output: bool = True,
):
  """Normalizes `acc` and writes to `out_gtile` via swizzled SMEM."""
  if normalize_output:
    inv_l = jnp.where(
        l_final == 0.0, 0.0, 1.0 / jnp.where(l_final == 0.0, 1.0, l_final)
    )
    acc = acc * plgpu.layout_cast(
        lax.broadcast_in_dim(inv_l, acc.shape, (0,)), mma_acc
    )
  _store_via_swizzled_smem(out_gtile, acc, epi_smem, out_dtype, epi_swizzle)


@jaxtyping.jaxtyped
def flash_attention_kernel(
    q: Float[Array, "T H D"],
    k: Float[Array, "t h D"],
    v: Float[Array, "t h d"],
    bias: Float[Array, "#H #T #t"] | None,
    mask: Bool[Array, "#H #T #t"] | None,
    k_start: Int[Array, "#H #T"] | None,
    k_end: Int[Array, "#H #T"] | None,
    q_start: Int[Array, "#H 1"] | None = None,
    q_end: Int[Array, "#H 1"] | None = None,
    *,
    is_causal: bool,
    logits_soft_cap: float | None,
    logits_scale: float,
    out_dtype: jnp.dtype,
    normalize_output: bool,
    return_residuals: bool,
    use_stable_softmax: bool,
    rescale_threshold: float,
    config: Config,
) -> tuple[Float[Array, "T H d"], Residuals | None]:
  """SM80 Pallas Mosaic GPU Flash Attention."""
  if not use_stable_softmax:
    raise NotImplementedError("Unstable softmax not supported on sm80.")
  if rescale_threshold != 1.0:
    raise NotImplementedError("rescale_threshold != 1.0 not supported on sm80.")

  orig_q_seq_len, num_q_heads, _ = q.shape
  _, num_kv_heads, orig_head_dim_out = v.shape

  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  q_heads_per_kv = num_q_heads // num_kv_heads

  dtype = q.dtype
  if jnp.dtype(dtype) not in map(jnp.dtype, [jnp.float16, jnp.bfloat16]):
    raise NotImplementedError(
        f"Only f16 and bf16 are supported, got dtype: {dtype}"
    )

  block_q, block_kv = config.block_q, config.block_kv

  q, k, v = _pad_head_dims(q, k, v)
  head_dim = q.shape[-1]
  head_dim_out = v.shape[-1]

  q = shape_lib.pad_to_next_multiple_of(q, block_q, -3)
  k = shape_lib.pad_to_next_multiple_of(k, block_kv, -3)
  v = shape_lib.pad_to_next_multiple_of(v, block_kv, -3)
  q_seq, kv_seq = q.shape[-3], k.shape[-3]

  if mask is not None:
    if mask.shape[-2] != 1:
      mask = shape_lib.pad_to_next_multiple_of(mask, block_q, -2)
    if mask.shape[-1] != 1:
      mask = shape_lib.pad_to_next_multiple_of(mask, block_kv, -1)
    mask = mask.astype(jnp.int8)

  if bias is not None:
    if bias.shape[-2] != 1:
      bias = shape_lib.pad_to_next_multiple_of(bias, block_q, -2)
    if bias.shape[-1] != 1:
      bias = shape_lib.pad_to_next_multiple_of(bias, block_kv, -1)

  as_2d = lambda x: None if x is None else jax.lax.broadcast_to_rank(x, 2)
  k_start, k_end, q_start, q_end = map(as_2d, (k_start, k_end, q_start, q_end))
  k_start_minmax = common.k_range_block_minmax(k_start, kv_seq, block_q)
  k_end_minmax = common.k_range_block_minmax(k_end, kv_seq, block_q)
  if k_start is not None and k_start.shape[-1] != 1:
    k_start = shape_lib.pad_to_next_multiple_of(
        k_start, block_q, -1, pad_value=kv_seq
    )
  if k_end is not None and k_end.shape[-1] != 1:
    k_end = shape_lib.pad_to_next_multiple_of(
        k_end, block_q, -1, pad_value=0
    )

  v_split0, effective_wgs, wg_q, stages_k, stages_v = _resolve_mode_params(
      head_dim_out, kv_seq, config
  )
  num_kv_blocks = kv_seq // block_kv
  num_q_tiles = q_seq // block_q

  # Stage 2D bias / mask in SMEM when it fits; otherwise fall back to GMEM.
  stage_2d_bias = True
  stage_2d_mask = True
  scratch_allocations = _get_scratch_types(
      q,
      k,
      v,
      bias,
      mask,
      out_dtype,
      config,
      stage_2d_bias=stage_2d_bias,
      stage_2d_mask=stage_2d_mask,
  )
  if mgpu_lib.estimate_smem_bytes(scratch_allocations) > _MAX_SMEM:
    stage_2d_mask = False
    scratch_allocations = _get_scratch_types(
        q,
        k,
        v,
        bias,
        mask,
        out_dtype,
        config,
        stage_2d_bias=stage_2d_bias,
        stage_2d_mask=stage_2d_mask,
    )
  if mgpu_lib.estimate_smem_bytes(scratch_allocations) > _MAX_SMEM:
    stage_2d_bias = False
    scratch_allocations = _get_scratch_types(
        q,
        k,
        v,
        bias,
        mask,
        out_dtype,
        config,
        stage_2d_bias=stage_2d_bias,
        stage_2d_mask=stage_2d_mask,
    )

  elem_bits = mgpu_lib.num_bits(dtype)
  k_swizzle = _choose_k_swizzle(head_dim, block_kv, elem_bits)
  step_d = 8 * k_swizzle // elem_bits

  mma_lhs = plgpu.Layout.MMA_LHS(dtype)
  mma_rhs = plgpu.Layout.MMA_RHS(dtype)
  mma_acc = plgpu.Layout.MMA_ACC(dtype)
  mma_row = mma_acc.reduce(1)
  finfo_min = float(jnp.finfo(jnp.float32).min)

  has_index_mask = (
      is_causal
      or q_start is not None
      or q_end is not None
      or k_start is not None
      or k_end is not None
  )

  def _pv_step_kv(cols: int) -> int:
    max_step = max(16, (64 * 64 // cols) // 16 * 16)
    for step in range(min(block_kv, max_step), 15, -16):
      if block_kv % step == 0:
        return step
    return 16

  def kernel(
      q_gmem,
      k_gmem,
      v_gmem,
      mask_gmem,
      bias_gmem,
      k_start_gmem,
      k_end_gmem,
      q_start_gmem,
      q_end_gmem,
      k_start_minmax_gmems,
      k_end_minmax_gmems,
      *output_gmems,
      main_epi_union,
      al_smem=None,
      p_ready_barrier=None,
      p_done_barrier=None,
  ):
    if not return_residuals:
      (out_gmem,) = output_gmems
      m_gmem = l_gmem = None
    else:
      out_gmem, m_gmem, l_gmem = output_gmems

    wg = lax.axis_index("wg") if (v_split0 > 0 or effective_wgs > 1) else 0
    hi = lax.axis_index("heads")
    qi_raw = lax.axis_index("q_tiles")
    qi = (num_q_tiles - 1 - qi_raw) if is_causal else qi_raw
    hi_kv = lax.div(hi, jnp.asarray(q_heads_per_kv, hi.dtype))

    cta_q_base = qi * block_q
    q_base = cta_q_base if v_split0 > 0 else (cta_q_base + wg * wg_q)
    qs = pl.ds(q_base, wg_q)
    out_gtile = out_gmem.at[qs, hi]

    # Scalar Q-range bounds `[q_start_val, q_end_val)` for early exit & causal.
    q_start_val = jnp.int32(0)
    if q_start_gmem is not None:
      q_hi = 0 if q_start_gmem.shape[0] == 1 else hi
      q_start_val = q_start_gmem[q_hi, 0]
    q_end_val = jnp.int32(orig_q_seq_len)
    if q_end_gmem is not None:
      q_hi = 0 if q_end_gmem.shape[0] == 1 else hi
      q_end_val = lax.min(q_end_val, q_end_gmem[q_hi, 0])

    eff_q_max = lax.min(cta_q_base + block_q, q_end_val)
    ub = (
        lax.min(num_kv_blocks, pl.cdiv(lax.max(eff_q_max, 0), block_kv))
        if is_causal
        else num_kv_blocks
    )
    lb = jnp.int32(0)
    k_start_max = jnp.int32(0)
    k_end_min = jnp.int32(kv_seq)

    if k_start_minmax_gmems is not None:
      k_start_min_gmem, k_start_max_gmem = k_start_minmax_gmems
      ks_min = common.load_bcast(k_start_min_gmem, (hi, qi), layout=None)
      k_start_max = common.load_bcast(k_start_max_gmem, (hi, qi), layout=None)
      lb = lax.max(lb, lax.div(lax.max(ks_min, 0), block_kv))
    if k_end_minmax_gmems is not None:
      k_end_min_gmem, k_end_max_gmem = k_end_minmax_gmems
      k_end_min = common.load_bcast(k_end_min_gmem, (hi, qi), layout=None)
      ke_max = common.load_bcast(k_end_max_gmem, (hi, qi), layout=None)
      ub = lax.min(ub, pl.cdiv(lax.max(ke_max, 0), block_kv))

    cta_q_valid = (cta_q_base + block_q > q_start_val) & (
        cta_q_base < q_end_val
    )
    cta_active = cta_q_valid & (ub > lb)

    q_cta_fully_in_range = True
    if q_start_gmem is not None:
      q_cta_fully_in_range = q_cta_fully_in_range & (cta_q_base >= q_start_val)
    if q_end_gmem is not None:
      q_cta_fully_in_range = q_cta_fully_in_range & (
          cta_q_base + block_q <= q_end_val
      )

    # Preload per-row `k_start` / `k_end` once outside the KV loop.
    def _preload_k_bound(ref):
      if ref is None:
        return None
      r_hi = 0 if ref.shape[0] == 1 else hi
      if ref.shape[-1] == 1:
        return ref[r_hi, 0]
      return plgpu.load(ref.at[r_hi, qs], layout=mma_row, optimized=False)

    def _score_postprocess(
        s: jax.Array,
        ki: jax.Array,
        k_start_bound,
        k_end_bound,
        *,
        apply_idx_mask: bool,
        bias_smem_slot=None,
        mask_smem_slot=None,
    ) -> jax.Array:
      log2e = math.log2(math.e)
      ks = pl.ds(ki * block_kv, block_kv)
      if bias_gmem is None and logits_soft_cap is None:
        s = s * (logits_scale * log2e)
      else:
        if bias_gmem is not None:
          if bias_smem_slot is not None:
            bias_tile = plgpu.load(
                bias_smem_slot, layout=mma_acc, optimized=False
            )
          else:
            bias_tile = common.load_bcast(
                bias_gmem, (hi, qs, ks), layout=mma_acc, optimized=False
            )
          s = s * logits_scale + bias_tile.astype(jnp.float32)
        else:
          s = s * logits_scale

        if logits_soft_cap is not None:
          s = jnp.tanh(s * (1.0 / logits_soft_cap)) * (logits_soft_cap * log2e)
        else:
          s = s * log2e

      if apply_idx_mask and has_index_mask:
        k_base = ki * block_kv
        q_pos = q_base + plgpu.broadcasted_iota(
            jnp.int32, s.shape, 0, layout=mma_acc
        )
        k_pos = k_base + plgpu.broadcasted_iota(
            jnp.int32, s.shape, 1, layout=mma_acc
        )
        keep = None
        if q_start_gmem is not None:
          keep = q_pos >= q_start_val
        if q_end_gmem is not None:
          cond = q_pos < q_end_val
          keep = cond if keep is None else (keep & cond)
        if is_causal:
          cond = q_pos >= k_pos
          keep = cond if keep is None else (keep & cond)
        if k_start_bound is not None:
          if k_start_bound.ndim == 0:
            bound = k_start_bound
          else:
            bound = plgpu.layout_cast(
                lax.broadcast_in_dim(k_start_bound, s.shape, (0,)),
                mma_acc,
            )
          cond = k_pos >= bound
          keep = cond if keep is None else (keep & cond)
        if k_end_bound is not None:
          if k_end_bound.ndim == 0:
            bound = k_end_bound
          else:
            bound = plgpu.layout_cast(
                lax.broadcast_in_dim(k_end_bound, s.shape, (0,)),
                mma_acc,
            )
          cond = k_pos < bound
          keep = cond if keep is None else (keep & cond)
        assert keep is not None
        s = jnp.where(keep, s, -jnp.inf)

      if mask_gmem is not None:
        if mask_smem_slot is not None:
          mask_tile = (
              plgpu.load(mask_smem_slot, layout=mma_acc, optimized=False) != 0
          )
        else:
          mask_tile = (
              common.load_bcast(
                  mask_gmem, (hi, qs, ks), layout=mma_acc, optimized=False
              )
              != 0
          )
        s = jnp.where(mask_tile, s, -jnp.inf)

      return s

    total_kv_steps = ub - lb
    if not has_index_mask:
      def _run_kv_loops(make_body, init_carry):
        return lax.fori_loop(0, total_kv_steps, make_body(False), init_carry)
    elif k_start_gmem is not None and k_start_gmem.shape[-1] != 1:
      def _run_kv_loops(make_body, init_carry):
        return lax.fori_loop(0, total_kv_steps, make_body(True), init_carry)
    else:
      ki_unmasked_lo = (
          lax.max(lb, pl.cdiv(lax.max(k_start_max, 0), block_kv))
          if k_start_gmem is not None
          else lb
      )
      ki_unmasked_hi = ub
      if is_causal:
        ki_unmasked_hi = lax.min(ki_unmasked_hi, lax.div(cta_q_base, block_kv))
      if k_end_gmem is not None:
        ki_unmasked_hi = lax.min(
            ki_unmasked_hi, lax.div(lax.max(k_end_min, 0), block_kv)
        )
      ki_unmasked_lo = lax.clamp(lb, ki_unmasked_lo, ub)
      ki_unmasked_hi = lax.clamp(ki_unmasked_lo, ki_unmasked_hi, ub)
      if q_start_gmem is not None or q_end_gmem is not None:
        ki_unmasked_hi = lax.select(
            q_cta_fully_in_range, ki_unmasked_hi, ki_unmasked_lo
        )
      start_unmasked = ki_unmasked_lo - lb
      end_unmasked = ki_unmasked_hi - lb

      def _run_kv_loops(make_body, init_carry):
        carry = init_carry
        if k_start_gmem is not None:
          carry = lax.fori_loop(0, start_unmasked, make_body(True), carry)
        carry = lax.fori_loop(
            start_unmasked, end_unmasked, make_body(False), carry
        )
        carry = lax.fori_loop(
            end_unmasked, total_kv_steps, make_body(True), carry
        )
        return carry

    # -------------------------------------------------------------------------
    # Mode 1: Q-Parallel FlashAttention-2 (`v_split0 == 0`)
    # -------------------------------------------------------------------------
    if v_split0 == 0:
      if config.q_smem:
        (q_smem, k_smem, v_smem, extras), (epi_smem,) = main_epi_union
      else:
        (q_smem,), (k_smem, v_smem, extras), (epi_smem,) = main_epi_union

      bias_smem = extras.get("bias_smem")
      mask_smem = extras.get("mask_smem")
      stages_k_smem = k_smem.shape[0]
      stages_v_smem = v_smem.shape[0]
      stages = stages_k_smem
      pv_step_kv = _pv_step_kv(head_dim_out)
      epi_cols = _choose_epi_cols(head_dim_out, out_dtype)
      epi_swizzle = plgpu.find_swizzle(
          epi_cols * mgpu_lib.num_bits(out_dtype), "epi"
      )

      def _sync_cta():
        if effective_wgs > 1:
          _cta_barrier()
        else:
          mgpu_lib.warpgroup_barrier()

      @pl.when(jnp.logical_not(cta_active))
      def _early_exit_zero():
        zero_acc = plgpu.layout_cast(
            jnp.zeros((wg_q, head_dim_out), jnp.float32), mma_acc
        )
        _store_via_swizzled_smem(
            out_gtile, zero_acc, epi_smem.at[wg], out_dtype, epi_swizzle
        )
        if return_residuals:
          assert m_gmem is not None and l_gmem is not None
          m_empty = plgpu.layout_cast(
              jnp.full((wg_q,), finfo_min, jnp.float32), mma_row
          )
          l_empty = plgpu.layout_cast(
              jnp.zeros((wg_q,), jnp.float32), mma_row
          )
          plgpu.store(m_gmem.at[hi, qs], m_empty, optimized=False)
          plgpu.store(l_gmem.at[hi, qs], l_empty, optimized=False)

      @pl.when(cta_active)
      def _active_q_parallel():
        k_start_bound = _preload_k_bound(k_start_gmem)
        k_end_bound = _preload_k_bound(k_end_gmem)

        # Stage Q into SMEM (and into registers if `not config.q_smem`).
        _async_copy_gmem_to_smem(q_gmem.at[qs, hi], q_smem.at[wg])
        plgpu.wait_gmem_to_smem(0)
        if config.q_smem:
          q_reg = None
          _sync_cta()
        else:
          q_reg = plgpu.load(q_smem.at[wg], layout=mma_lhs)
          _sync_cta()

        load_k_gmem = lambda ki: k_gmem.at[
            pl.ds(ki * block_kv, block_kv), hi_kv
        ]
        load_v_gmem = lambda ki: v_gmem.at[
            pl.ds(ki * block_kv, block_kv), hi_kv
        ]

        def _issue_bias_mask(ki, slot):
          ks = pl.ds(ki * block_kv, block_kv)
          if bias_smem is not None:
            b_hi = 0 if bias_gmem.shape[0] == 1 else hi
            _async_copy_gmem_to_smem(
                bias_gmem.at[b_hi, qs, ks], bias_smem.at[slot, wg]
            )
          if mask_smem is not None:
            m_hi = 0 if mask_gmem.shape[0] == 1 else hi
            _async_copy_gmem_to_smem(
                mask_gmem.at[m_hi, qs, ks], mask_smem.at[slot, wg]
            )

        num_bm_copies = int(bias_smem is not None) + int(mask_smem is not None)

        # Sub-path A: 2-WG staggered K/V ping-pong (supports stages_k != v).
        if effective_wgs == 2 and num_bm_copies == 0:
          @pl.when(wg == 0)
          def _prologue_k():
            for s_idx in range(stages_k_smem):
              init_ki = lb + s_idx
              lax.cond(
                  init_ki < ub,
                  lambda init_ki=init_ki, s_idx=s_idx: _async_copy_gmem_to_smem(
                      load_k_gmem(init_ki), k_smem.at[s_idx]
                  ),
                  common.commit_cp_async_group,
              )

          @pl.when(wg == 1)
          def _prologue_v():
            for s_idx in range(stages_v_smem - 1):
              init_ki = lb + s_idx
              lax.cond(
                  init_ki < ub,
                  lambda init_ki=init_ki, s_idx=s_idx: _async_copy_gmem_to_smem(
                      load_v_gmem(init_ki), v_smem.at[s_idx]
                  ),
                  common.commit_cp_async_group,
              )

          def _make_staggered_body(apply_idx: bool):
            def _staggered_body(step, carry):
              m_i, l_i, acc = carry
              ki = lb + step
              k_slot = lax.rem(step, stages_k_smem)
              v_slot = lax.rem(step, stages_v_smem)
              v_prefetch_slot = lax.rem(step + stages_v_smem - 1, stages_v_smem)

              @pl.when(wg == 0)
              def _wait_k():
                plgpu.wait_gmem_to_smem(stages_k_smem - 1)

              _cta_barrier()

              @pl.when(wg == 1)
              def _issue_next_v():
                next_v_ki = ki + stages_v_smem - 1
                lax.cond(
                    next_v_ki < ub,
                    lambda: _async_copy_gmem_to_smem(
                        load_v_gmem(next_v_ki), v_smem.at[v_prefetch_slot]
                    ),
                    common.commit_cp_async_group,
                )

              s = _compute_qk(
                  q_reg,
                  q_smem.at[wg] if config.q_smem else None,
                  k_smem.at[k_slot],
                  wg_q,
                  block_kv,
                  head_dim,
                  step_d,
                  mma_acc,
                  mma_rhs,
                  mma_lhs,
              )
              s = _score_postprocess(
                  s,
                  ki,
                  k_start_bound,
                  k_end_bound,
                  apply_idx_mask=apply_idx,
              )
              m_new, alpha, p, l_new = _online_softmax_step(
                  s, m_i, l_i, mma_acc
              )
              p_lhs = _relayout_p_to_lhs(p.astype(dtype), mma_acc, mma_lhs)

              @pl.when(wg == 1)
              def _wait_v():
                plgpu.wait_gmem_to_smem(stages_v_smem - 1)

              _cta_barrier()

              @pl.when(wg == 0)
              def _issue_next_k():
                next_k_ki = ki + stages_k_smem
                lax.cond(
                    next_k_ki < ub,
                    lambda: _async_copy_gmem_to_smem(
                        load_k_gmem(next_k_ki), k_smem.at[k_slot]
                    ),
                    common.commit_cp_async_group,
                )

              acc = _compute_pv_step(
                  p_lhs,
                  acc,
                  alpha,
                  v_smem.at[v_slot],
                  (wg_q, head_dim_out),
                  pv_step_kv,
                  mma_acc,
                  mma_rhs,
              )
              return m_new, l_new, acc

            return _staggered_body

          acc0 = plgpu.layout_cast(
              jnp.zeros((wg_q, head_dim_out), jnp.float32), mma_acc
          )
          m_final, l_final, acc = _run_kv_loops(
              _make_staggered_body,
              (*_init_softmax_state(wg_q, mma_row), acc0),
          )
          plgpu.wait_gmem_to_smem(0)
          _cta_barrier()

        # Sub-path B: 1-WG staggered K/V ping-pong (stages >= 2, no 2D bias).
        elif effective_wgs == 1 and num_bm_copies == 0 and stages >= 2:
          for s_idx in range(stages - 1):
            init_ki = lb + s_idx
            lax.cond(
                init_ki < ub,
                lambda init_ki=init_ki, s_idx=s_idx: _async_copy_gmem_to_smem(
                    load_k_gmem(init_ki), k_smem.at[s_idx]
                ),
                common.commit_cp_async_group,
            )
            lax.cond(
                init_ki < ub,
                lambda init_ki=init_ki, s_idx=s_idx: _async_copy_gmem_to_smem(
                    load_v_gmem(init_ki), v_smem.at[s_idx]
                ),
                common.commit_cp_async_group,
            )
          last_init_ki = lb + stages - 1
          lax.cond(
              last_init_ki < ub,
              lambda: _async_copy_gmem_to_smem(
                  load_k_gmem(last_init_ki), k_smem.at[stages - 1]
              ),
              common.commit_cp_async_group,
          )

          def _make_staggered_1wg_body(apply_idx: bool):
            def _staggered_1wg_body(step, carry):
              m_i, l_i, acc = carry
              ki = lb + step
              slot = lax.rem(step, stages)
              v_prefetch_slot = lax.rem(step + stages - 1, stages)

              plgpu.wait_gmem_to_smem(2 * (stages - 1))
              next_v_ki = ki + stages - 1
              lax.cond(
                  next_v_ki < ub,
                  lambda: _async_copy_gmem_to_smem(
                      load_v_gmem(next_v_ki), v_smem.at[v_prefetch_slot]
                  ),
                  common.commit_cp_async_group,
              )

              s = _compute_qk(
                  q_reg,
                  q_smem.at[0] if config.q_smem else None,
                  k_smem.at[slot],
                  wg_q,
                  block_kv,
                  head_dim,
                  step_d,
                  mma_acc,
                  mma_rhs,
                  mma_lhs,
              )
              s = _score_postprocess(
                  s,
                  ki,
                  k_start_bound,
                  k_end_bound,
                  apply_idx_mask=apply_idx,
              )
              m_new, alpha, p, l_new = _online_softmax_step(
                  s, m_i, l_i, mma_acc
              )
              p_lhs = _relayout_p_to_lhs(p.astype(dtype), mma_acc, mma_lhs)
              mgpu_lib.warpgroup_barrier()

              plgpu.wait_gmem_to_smem(2 * (stages - 1))
              next_k_ki = ki + stages
              lax.cond(
                  next_k_ki < ub,
                  lambda: _async_copy_gmem_to_smem(
                      load_k_gmem(next_k_ki), k_smem.at[slot]
                  ),
                  common.commit_cp_async_group,
              )

              acc = _compute_pv_step(
                  p_lhs,
                  acc,
                  alpha,
                  v_smem.at[slot],
                  (wg_q, head_dim_out),
                  pv_step_kv,
                  mma_acc,
                  mma_rhs,
              )
              mgpu_lib.warpgroup_barrier()
              return m_new, l_new, acc

            return _staggered_1wg_body

          acc0 = plgpu.layout_cast(
              jnp.zeros((wg_q, head_dim_out), jnp.float32), mma_acc
          )
          m_final, l_final, acc = _run_kv_loops(
              _make_staggered_1wg_body,
              (*_init_softmax_state(wg_q, mma_row), acc0),
          )
          plgpu.wait_gmem_to_smem(0)
          mgpu_lib.warpgroup_barrier()

        # Sub-path C: Unified multi-stage ring (2D bias/mask or stages == 1).
        else:
          copies_per_wg = (1 if effective_wgs == 2 else 2) + num_bm_copies

          def _issue_stage(fetch_ki, slot):
            def _do_copies():
              if effective_wgs == 1:
                _async_copy_gmem_to_smem(load_k_gmem(fetch_ki), k_smem.at[slot])
                _async_copy_gmem_to_smem(load_v_gmem(fetch_ki), v_smem.at[slot])
              else:
                @pl.when(wg == 0)
                def _cp_k():
                  _async_copy_gmem_to_smem(
                      load_k_gmem(fetch_ki), k_smem.at[slot]
                  )

                @pl.when(wg == 1)
                def _cp_v():
                  _async_copy_gmem_to_smem(
                      load_v_gmem(fetch_ki), v_smem.at[slot]
                  )

              _issue_bias_mask(fetch_ki, slot)

            def _do_empty():
              for _ in range(copies_per_wg):
                common.commit_cp_async_group()

            lax.cond(fetch_ki < ub, _do_copies, _do_empty)

          for s_idx in range(stages):
            _issue_stage(lb + s_idx, s_idx)

          def _make_ring_body(apply_idx: bool):
            def _ring_body(step, carry):
              m_i, l_i, acc = carry
              ki = lb + step
              slot = lax.rem(step, stages)

              plgpu.wait_gmem_to_smem((stages - 1) * copies_per_wg)
              _sync_cta()

              s = _compute_qk(
                  q_reg,
                  q_smem.at[wg] if config.q_smem else None,
                  k_smem.at[slot],
                  wg_q,
                  block_kv,
                  head_dim,
                  step_d,
                  mma_acc,
                  mma_rhs,
                  mma_lhs,
              )
              s = _score_postprocess(
                  s,
                  ki,
                  k_start_bound,
                  k_end_bound,
                  apply_idx_mask=apply_idx,
                  bias_smem_slot=(
                      None if bias_smem is None else bias_smem.at[slot, wg]
                  ),
                  mask_smem_slot=(
                      None if mask_smem is None else mask_smem.at[slot, wg]
                  ),
              )
              m_new, alpha, p, l_new = _online_softmax_step(
                  s, m_i, l_i, mma_acc
              )
              p_lhs = _relayout_p_to_lhs(p.astype(dtype), mma_acc, mma_lhs)
              acc = _compute_pv_step(
                  p_lhs,
                  acc,
                  alpha,
                  v_smem.at[slot],
                  (wg_q, head_dim_out),
                  pv_step_kv,
                  mma_acc,
                  mma_rhs,
              )
              _sync_cta()
              _issue_stage(ki + stages, slot)
              return m_new, l_new, acc

            return _ring_body

          acc0 = plgpu.layout_cast(
              jnp.zeros((wg_q, head_dim_out), jnp.float32), mma_acc
          )
          m_final, l_final, acc = _run_kv_loops(
              _make_ring_body,
              (*_init_softmax_state(wg_q, mma_row), acc0),
          )
          plgpu.wait_gmem_to_smem(0)
          _sync_cta()

        if return_residuals:
          assert m_gmem is not None and l_gmem is not None
          m_out = jnp.where(
              m_final == -jnp.inf,
              finfo_min,
              m_final * (1.0 / math.log2(math.e)),
          )
          plgpu.store(m_gmem.at[hi, qs], m_out, optimized=False)
          plgpu.store(l_gmem.at[hi, qs], l_final, optimized=False)

        _write_output_epilogue(
            out_gtile,
            acc,
            l_final,
            epi_smem.at[wg],
            out_dtype,
            epi_swizzle,
            mma_acc,
            normalize_output,
        )
      return

    # -------------------------------------------------------------------------
    # Mode 2: Split-Head Pipeline (`v_split0 > 0`, e.g. `head_dim == 512`)
    # -------------------------------------------------------------------------
    assert (
        al_smem is not None
        and p_ready_barrier is not None
        and p_done_barrier is not None
    )
    v_split1 = head_dim_out - v_split0
    num_p_slots = config.num_p_slots
    if config.q_smem:
      (q_smem, k_smem, v0_smem, v1_smem, p_smem), (epi0_smem, epi1_smem) = (
          main_epi_union
      )
    else:
      (
          (q_smem,),
          (k_smem, v0_smem, v1_smem, p_smem),
          (epi0_smem, epi1_smem),
      ) = main_epi_union

    epi0_cols = _choose_epi_cols(v_split0, out_dtype)
    epi1_cols = _choose_epi_cols(v_split1, out_dtype)
    epi0_swizzle = plgpu.find_swizzle(
        epi0_cols * mgpu_lib.num_bits(out_dtype), "epi0"
    )
    epi1_swizzle = plgpu.find_swizzle(
        epi1_cols * mgpu_lib.num_bits(out_dtype), "epi1"
    )

    @pl.when(jnp.logical_not(cta_active))
    def _split_early_exit():
      @pl.when(wg == 0)
      def _z0():
        z0 = plgpu.layout_cast(
            jnp.zeros((block_q, v_split0), jnp.float32), mma_acc
        )
        _store_via_swizzled_smem(
            out_gtile.at[:, pl.ds(0, v_split0)],
            z0,
            epi0_smem,
            out_dtype,
            epi0_swizzle,
        )
        if return_residuals:
          assert m_gmem is not None and l_gmem is not None
          m_empty = plgpu.layout_cast(
              jnp.full((block_q,), finfo_min, jnp.float32), mma_row
          )
          l_empty = plgpu.layout_cast(
              jnp.zeros((block_q,), jnp.float32), mma_row
          )
          plgpu.store(m_gmem.at[hi, qs], m_empty, optimized=False)
          plgpu.store(l_gmem.at[hi, qs], l_empty, optimized=False)

      @pl.when(wg == 1)
      def _z1():
        z1 = plgpu.layout_cast(
            jnp.zeros((block_q, v_split1), jnp.float32), mma_acc
        )
        _store_via_swizzled_smem(
            out_gtile.at[:, pl.ds(v_split0, v_split1)],
            z1,
            epi1_smem,
            out_dtype,
            epi1_swizzle,
        )

    @pl.when(cta_active)
    def _split_active():
      load_k_gmem = lambda ki: k_gmem.at[pl.ds(ki * block_kv, block_kv), hi_kv]
      load_v_gmem = lambda ki, lo, n: v_gmem.at[
          pl.ds(ki * block_kv, block_kv), hi_kv, pl.ds(lo, n)
      ]

      @pl.when(wg == 0)
      def _wg0():
        k_start_bound = _preload_k_bound(k_start_gmem)
        k_end_bound = _preload_k_bound(k_end_gmem)
        _async_copy_gmem_to_smem(q_gmem.at[qs, hi], q_smem)
        plgpu.wait_gmem_to_smem(0)
        if config.q_smem:
          q_reg = None
        else:
          q_reg = plgpu.load(q_smem, layout=mma_lhs)
          _cta_barrier()

        stages = min(stages_k, stages_v)
        for s_idx in range(stages):
          init_ki = lb + s_idx

          def _cp_kv0(init_ki=init_ki, s_idx=s_idx):
            _async_copy_gmem_to_smem(load_k_gmem(init_ki), k_smem.at[s_idx])
            _async_copy_gmem_to_smem(
                load_v_gmem(init_ki, 0, v_split0), v0_smem.at[s_idx]
            )

          def _empty_kv0():
            common.commit_cp_async_group()
            common.commit_cp_async_group()

          lax.cond(init_ki < ub, _cp_kv0, _empty_kv0)

        pv0_step_kv = _pv_step_kv(v_split0)

        def _make_split_wg0_body(apply_idx: bool):
          def loop_body(step, carry):
            m_i, l_i, acc = carry
            ki = lb + step
            slot = lax.rem(step, stages)
            p_slot = lax.rem(step, num_p_slots)

            @pl.when(step >= num_p_slots)
            def _wait_p_done():
              plgpu.barrier_wait(p_done_barrier.at[p_slot])

            plgpu.wait_gmem_to_smem((stages - 1) * 2)
            s = _compute_qk(
                q_reg,
                q_smem if config.q_smem else None,
                k_smem.at[slot],
                block_q,
                block_kv,
                head_dim,
                step_d,
                mma_acc,
                mma_rhs,
                mma_lhs,
            )
            s = _score_postprocess(
                s,
                ki,
                k_start_bound,
                k_end_bound,
                apply_idx_mask=apply_idx,
            )
            m_new, alpha, p, l_new = _online_softmax_step(s, m_i, l_i, mma_acc)
            p_cast = p.astype(dtype)
            p_smem[p_slot] = p_cast
            al_smem[p_slot] = alpha
            mgpu_lib.warpgroup_barrier()
            plgpu.barrier_arrive(p_ready_barrier.at[p_slot])

            acc = _compute_pv_step(
                _relayout_p_to_lhs(p_cast, mma_acc, mma_lhs),
                acc,
                alpha,
                v0_smem.at[slot],
                (block_q, v_split0),
                pv0_step_kv,
                mma_acc,
                mma_rhs,
            )
            mgpu_lib.warpgroup_barrier()

            next_ki = ki + stages
            lax.cond(
                next_ki < ub,
                lambda: (
                    _async_copy_gmem_to_smem(
                        load_k_gmem(next_ki), k_smem.at[slot]
                    ),
                    _async_copy_gmem_to_smem(
                        load_v_gmem(next_ki, 0, v_split0), v0_smem.at[slot]
                    ),
                ),
                lambda: (
                    common.commit_cp_async_group(),
                    common.commit_cp_async_group(),
                ),
            )
            return m_new, l_new, acc

          return loop_body

        acc0 = plgpu.layout_cast(
            jnp.zeros((block_q, v_split0), jnp.float32), mma_acc
        )
        m_final, l_final, acc = _run_kv_loops(
            _make_split_wg0_body,
            (*_init_softmax_state(block_q, mma_row), acc0),
        )
        plgpu.wait_gmem_to_smem(0)
        al_smem[num_p_slots] = l_final
        if return_residuals:
          assert m_gmem is not None and l_gmem is not None
          m_out = jnp.where(
              m_final == -jnp.inf,
              finfo_min,
              m_final * (1.0 / math.log2(math.e)),
          )
          plgpu.store(m_gmem.at[hi, qs], m_out, optimized=False)
          plgpu.store(l_gmem.at[hi, qs], l_final, optimized=False)
        _cta_barrier()
        _write_output_epilogue(
            out_gtile.at[:, pl.ds(0, v_split0)],
            acc,
            l_final,
            epi0_smem,
            out_dtype,
            epi0_swizzle,
            mma_acc,
            normalize_output,
        )

      @pl.when(wg == 1)
      def _wg1():
        if not config.q_smem:
          _cta_barrier()
        stages = min(stages_k, stages_v)
        for s_idx in range(stages):
          init_ki = lb + s_idx
          lax.cond(
              init_ki < ub,
              lambda init_ki=init_ki, s_idx=s_idx: _async_copy_gmem_to_smem(
                  load_v_gmem(init_ki, v_split0, v_split1), v1_smem.at[s_idx]
              ),
              common.commit_cp_async_group,
          )

        pv1_step_kv = _pv_step_kv(v_split1)

        def loop_body(step, acc):
          ki = lb + step
          slot = lax.rem(step, stages)
          p_slot = lax.rem(step, num_p_slots)
          plgpu.barrier_wait(p_ready_barrier.at[p_slot])
          alpha = plgpu.load(al_smem.at[p_slot], layout=mma_row)
          plgpu.wait_gmem_to_smem(stages - 1)
          acc = _compute_pv_step(
              plgpu.load(p_smem.at[p_slot], layout=mma_lhs),
              acc,
              alpha,
              v1_smem.at[slot],
              (block_q, v_split1),
              pv1_step_kv,
              mma_acc,
              mma_rhs,
          )
          mgpu_lib.warpgroup_barrier()
          next_ki = ki + stages
          lax.cond(
              next_ki < ub,
              lambda: _async_copy_gmem_to_smem(
                  load_v_gmem(next_ki, v_split0, v_split1), v1_smem.at[slot]
              ),
              common.commit_cp_async_group,
          )
          plgpu.barrier_arrive(p_done_barrier.at[p_slot])
          return acc

        acc0 = plgpu.layout_cast(
            jnp.zeros((block_q, v_split1), jnp.float32), mma_acc
        )
        acc = lax.fori_loop(0, ub - lb, loop_body, acc0)
        plgpu.wait_gmem_to_smem(0)
        _cta_barrier()
        _write_output_epilogue(
            out_gtile.at[:, pl.ds(v_split0, v_split1)],
            acc,
            plgpu.load(al_smem.at[num_p_slots], layout=mma_row),
            epi1_smem,
            out_dtype,
            epi1_swizzle,
            mma_acc,
            normalize_output,
        )

  out_type = (
      jax.ShapeDtypeStruct((q_seq, num_q_heads, head_dim_out), out_dtype),
  )
  if return_residuals:
    out_type += (jax.ShapeDtypeStruct((num_q_heads, q_seq), jnp.float32),) * 2

  raw_threads = 2 if v_split0 > 0 else effective_wgs
  num_threads = raw_threads if raw_threads > 1 else None
  thread_name = "wg" if raw_threads > 1 else None

  out, *residuals = plgpu.kernel(
      kernel,
      out_type=out_type,
      grid=(num_q_heads, num_q_tiles),
      grid_names=("heads", "q_tiles"),
      num_threads=num_threads,
      thread_name=thread_name,
      scratch_types=scratch_allocations,  # pyrefly: ignore[bad-argument-type]
      compiler_params=plgpu.CompilerParams(
          approx_math=True,
          unsafe_no_auto_barriers=True,
          lowering_semantics=plgpu.LoweringSemantics.Lane,
      ),
      kernel_name=(
          "flash_attention_sm80_split_head"
          if v_split0 > 0
          else f"flash_attention_sm80_{effective_wgs}wg"
      ),
  )(
      q,
      k,
      v,
      mask,
      bias,
      k_start,
      k_end,
      q_start,
      q_end,
      k_start_minmax,
      k_end_minmax,
  )

  out = out[:orig_q_seq_len, :, :orig_head_dim_out]
  residuals = tuple(res[..., :orig_q_seq_len] for res in residuals)
  return (out, residuals if residuals else None)


def get_autotuning_configs(ba: op.BoundArguments) -> set[Config]:
  """Returns a set of configs for autotuning flash attention on SM80 GPUs."""
  q, k, v = ba.args[:3]
  q_seq_len = q.shape[-3]
  kv_seq_len = k.shape[-3]
  head_dim = pl.cdiv(q.shape[-1], 16) * 16
  head_dim_out = pl.cdiv(v.shape[-1], 16) * 16
  max_d = max(head_dim, head_dim_out)

  configs: set[Config] = {
      get_heuristics_config(ba, fold_q_sequence_heads=False)
  }
  if max_d > 256:
    v_split0 = max(32, (head_dim_out // 2 // 32) * 32)
    for block_kv in (32, 64):
      for num_stages in (1, 2):
        for q_smem in (True, False):
          cfg = Config(
              block_q=64,
              block_kv=block_kv,
              num_stages=num_stages,
              v_split0=v_split0,
              q_smem=q_smem,
          )
          if _estimate_shared_mem_usage_bytes(ba, cfg) <= _MAX_SMEM:
            configs.add(cfg)
    return configs

  block_qs = [64, 128] if q_seq_len > 64 else [64]
  if max_d > 192:
    block_kvs = [32, 64] if kv_seq_len >= 64 else [32]
  elif kv_seq_len >= 128 and max_d <= 128:
    block_kvs = [64, 128]
  else:
    block_kvs = [64]

  for block_q in block_qs:
    wgs_opts: tuple[Literal[1, 2], ...] = (2, 1) if block_q >= 128 else (1,)
    for block_kv in block_kvs:
      for num_wgs in wgs_opts:
        if max_d > 128 and block_q // num_wgs > 64:
          continue
        q_smem_opts = (
            [True]
            if (max_d > 192 and block_kv > 32)
            else [False, True]
        )
        for num_stages in (1, 2, 3):
          if num_stages > pl.cdiv(kv_seq_len, block_kv):
            continue
          for q_smem in q_smem_opts:
            cfg = Config(
                block_q=block_q,
                block_kv=block_kv,
                num_stages=num_stages,
                num_wgs=num_wgs,
                q_smem=q_smem,
                v_split0=0,
            )
            if _estimate_shared_mem_usage_bytes(ba, cfg) <= _MAX_SMEM:
              configs.add(cfg)
        if block_q == 128 and block_kv == 128 and num_wgs == 2:
          asym_cfg = Config(
              block_q=128,
              block_kv=128,
              num_stages=2,
              stages_k=3,
              stages_v=2,
              num_wgs=2,
              q_smem=False,
              v_split0=0,
          )
          if _estimate_shared_mem_usage_bytes(ba, asym_cfg) <= _MAX_SMEM:
            configs.add(asym_cfg)
  return configs
