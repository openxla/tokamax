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
"""Split-head Flash Attention with Mosaic GPU for SM80 (Ampere).

This module implements split-head Flash Attention for SM80/SM86.

Architecture Overview:
- 2 Warp Groups (WGs) per CTA processing one Q block (block_q) at a time.
- Warpgroup Specialization:
  - WG0 (Producer + Split 0): Loads Q, stages K, computes Q@K.T, performs
    online softmax in the base-2 exp domain, publishes P and alpha to a
    multi-slot SMEM ring buffer, and optionally accumulates PV for output
    columns [0:v_split0] if v_split0 > 0.
  - WG1 (Split 1): Consumes P and alpha from the SMEM ring buffer and
    accumulates PV for tail output columns [v_split0:head_dim] under WG0's
    softmax execution.
- Inter-Warpgroup Synchronization:
  - Multi-slot SMEM ring buffer (P and alpha) with per-slot ready/done barriers.
  - Zero CTA-wide barriers inside inner loops. A single CTA barrier is used
    at the epilogue for final lse hand-off when normalizing output.
- Memory Optimizations:
  - Staging Q into registers through a temporary buffer unioned with K's SMEM
    ring, or resident SMEM tile when q_smem is enabled.
  - Asynchronous GMEM->SMEM copies with hardware-accelerated cp.async.
"""

import math
from typing import Annotated, Any

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

Residuals = base.Residuals

_CHUNK_SIZE: int = 64
_NUM_WARPGROUPS: int = 2
_MASK_VALUE: float = float(jnp.finfo(jnp.float32).min)
_FORBID_EXTRA = pydantic.ConfigDict(extra="forbid")


@pydantic.dataclasses.dataclass(
    frozen=True, kw_only=True, slots=True, config=_FORBID_EXTRA
)
class Config(common.ConfigBase):
  """Configuration parameters for the split-head Mosaic GPU kernel on SM80.

  Attributes:
    block_q: Block size along Q sequence length.
    block_kv: Block size along KV sequence length (multiple of 32).
    num_stages: Default number of pipeline stages for K and V in SMEM.
    stages_k: Number of pipeline stages for K in SMEM (defaults to num_stages).
    stages_v: Number of pipeline stages for V in SMEM (defaults to num_stages).
    v_split0: Number of head dimension channels assigned to WG0 (must be
      multiple of 32).
    q_smem: Whether Q uses a resident SMEM tile instead of union staging.
    num_p_slots: Number of slots in the P/alpha ring buffer between warpgroups.
  """

  block_kv: Annotated[int, pydantic.Field(multiple_of=32, gt=0)] = 32
  num_stages: pydantic.PositiveInt = 1
  stages_k: pydantic.PositiveInt | None = None
  stages_v: pydantic.PositiveInt | None = None
  v_split0: Annotated[int, pydantic.Field(multiple_of=32, ge=0)] = 0
  q_smem: pydantic.StrictBool = False
  num_p_slots: pydantic.PositiveInt = 3


def get_heuristics_config(
    ba: op.BoundArguments, fold_q_sequence_heads: bool = False
) -> Config:
  """Returns a heuristic configuration for flash attention on SM80 GPUs."""
  q, k, v = ba.args[:3]
  head_dim = max(q.shape[-1], k.shape[-1], v.shape[-1])
  head_dim = max(_CHUNK_SIZE, pl.cdiv(head_dim, _CHUNK_SIZE) * _CHUNK_SIZE)
  if head_dim == 64:
    return Config(
        block_q=64,
        block_kv=128,
        num_stages=2,
        v_split0=0,
        q_smem=False,
        num_p_slots=3,
        fold_q_sequence_heads=fold_q_sequence_heads,
    )
  if head_dim == 128:
    return Config(
        block_q=64,
        block_kv=64,
        num_stages=2,
        v_split0=0,
        q_smem=False,
        num_p_slots=3,
        fold_q_sequence_heads=fold_q_sequence_heads,
    )
  if head_dim == 256:
    return Config(
        block_q=64,
        block_kv=64,
        num_stages=1,
        v_split0=128,
        q_smem=False,
        num_p_slots=3,
        fold_q_sequence_heads=fold_q_sequence_heads,
    )
  if head_dim == 512:
    return Config(
        block_q=64,
        block_kv=32,
        num_stages=1,
        v_split0=128,
        q_smem=False,
        num_p_slots=3,
        fold_q_sequence_heads=fold_q_sequence_heads,
    )
  return Config(
      block_q=64,
      block_kv=32,
      num_stages=1,
      v_split0=0,
      q_smem=False,
      num_p_slots=3,
      fold_q_sequence_heads=fold_q_sequence_heads,
  )


def _get_q_staging_slots(
    head_dim: int, block_q: int, block_kv: int, stages_k: int
) -> int:
  return head_dim // _CHUNK_SIZE if block_q <= stages_k * block_kv else 1


def _create_tiled_smem(shape: tuple[int, ...], dtype: jnp.dtype) -> Any:
  bits = jnp.dtype(dtype).itemsize * 8
  swizzle = plgpu.find_swizzle(shape[-1] * bits)
  return plgpu.SMEM(
      shape,
      dtype,
      transforms=(
          plgpu.TilingTransform((8, 8 * swizzle // bits)),
          plgpu.SwizzleTransform(swizzle),
      ),
  )


def _async_copy_gmem_to_smem(src, dst):
  plgpu.copy_gmem_to_smem(
      src, dst, oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS
  )


def _cta_barrier():
  mgpu_lib.bar_sync(8, 256)


def _compute_qk(
    q_chunks: list[jax.Array] | None,
    q_smem,
    k_smem,
    block_q: int,
    block_kv: int,
    num_chunks: int,
    mma_acc,
    mma_rhs,
    mma_lhs,
) -> jax.Array:
  """Computes S = Q @ K.T chunk-by-chunk along the contraction axis."""
  acc = plgpu.layout_cast(jnp.zeros((block_q, block_kv), jnp.float32), mma_acc)
  for di in range(num_chunks):
    chunk_slice = pl.ds(di * _CHUNK_SIZE, _CHUNK_SIZE)
    q_chunk = (
        q_chunks[di]
        if q_chunks is not None
        else plgpu.load(q_smem.at[:, chunk_slice], layout=mma_lhs)
    )
    k_tile = plgpu.load(
        plgpu.transpose_ref(k_smem.at[:, chunk_slice], (1, 0)),
        layout=mma_rhs,
    )
    acc = plgpu.mma(acc, q_chunk, k_tile)
  return acc


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


def _compute_pv_step(
    p_lhs: jax.Array,
    acc: jax.Array,
    alpha: jax.Array,
    v_smem,
    out_shape: tuple[int, int],
    mma_acc,
    mma_rhs,
) -> jax.Array:
  acc = acc * plgpu.layout_cast(
      lax.broadcast_in_dim(alpha, out_shape, (0,)), mma_acc
  )
  return plgpu.mma(
      plgpu.layout_cast(acc, mma_acc),
      p_lhs,
      plgpu.load(v_smem, layout=mma_rhs),
  )


def _online_softmax_step(
    s: jax.Array, m_prev: jax.Array, l_prev: jax.Array, mma_acc
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  m_curr = jnp.maximum(m_prev, jnp.max(s, axis=-1))
  alpha = jnp.exp2(m_prev - m_curr)
  p = jnp.exp2(
      s
      - plgpu.layout_cast(lax.broadcast_in_dim(m_curr, s.shape, (0,)), mma_acc)
  )
  l_curr = l_prev * alpha + jnp.sum(p, axis=-1)
  return m_curr, alpha, p, l_curr


def _init_softmax_state(block_q: int, mma_row) -> tuple[jax.Array, jax.Array]:
  return (
      plgpu.layout_cast(jnp.full((block_q,), -jnp.inf, jnp.float32), mma_row),
      plgpu.layout_cast(jnp.zeros((block_q,), jnp.float32), mma_row),
  )


def _write_output_epilogue(
    out_gtile,
    acc: jax.Array,
    l_final: jax.Array,
    lo: int,
    n: int,
    out_dtype: jnp.dtype,
    mma_acc,
    normalize_output: bool = True,
):
  """Normalizes and writes the output tile to GMEM."""
  if normalize_output:
    inv_l = jnp.where(
        l_final == 0, 0.0, 1.0 / jnp.where(l_final == 0, 1.0, l_final)
    )
    acc = acc * plgpu.layout_cast(
        lax.broadcast_in_dim(inv_l, acc.shape, (0,)), mma_acc
    )
  plgpu.store(
      out_gtile.at[:, pl.ds(lo, n)], acc.astype(out_dtype), optimized=False
  )


def _load_q_to_registers(
    q_gtile,
    q_stage,
    num_chunks: int,
    num_staging_slots: int,
    mma_lhs,
) -> list[jax.Array]:
  if num_staging_slots > 1:
    for di in range(num_chunks):
      _async_copy_gmem_to_smem(
          q_gtile.at[:, pl.ds(di * _CHUNK_SIZE, _CHUNK_SIZE)], q_stage.at[di]
      )
    plgpu.wait_gmem_to_smem(0)
    chunks = [
        plgpu.load(q_stage.at[di], layout=mma_lhs) for di in range(num_chunks)
    ]
    mgpu_lib.warpgroup_barrier()
    return chunks

  chunks = []
  for di in range(num_chunks):
    _async_copy_gmem_to_smem(
        q_gtile.at[:, pl.ds(di * _CHUNK_SIZE, _CHUNK_SIZE)], q_stage.at[0]
    )
    plgpu.wait_gmem_to_smem(0)
    chunks.append(plgpu.load(q_stage.at[0], layout=mma_lhs))
    mgpu_lib.warpgroup_barrier()
  return chunks


def _compute_k_range_minmax(
    k_range_ref: jax.Array | None, block_q: int
) -> tuple[jax.Array | None, jax.Array | None]:
  """Extracts block-level min and max bounds for k_start / k_end."""
  if k_range_ref is None:
    return None, None
  if k_range_ref.shape[-1] == 1:
    return k_range_ref, k_range_ref
  reshaped = k_range_ref.reshape(*k_range_ref.shape[:-1], -1, block_q)
  return reshaped.min(-1), reshaped.max(-1)


@jaxtyping.jaxtyped
def flash_attention_kernel(
    q: Float[Array, "T H D"],
    k: Float[Array, "t h D"],
    v: Float[Array, "t h d"],
    bias: Float[Array, "#H #T #t"] | None,
    mask: Bool[Array, "#H #T #t"] | None,
    k_start: Int[Array, "#H #T"] | None,
    k_end: Int[Array, "#H #T"] | None,
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
  """SM80 Pallas Mosaic GPU Flash Attention with split-head pipeline."""
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

  head_dim = max(x.shape[-1] for x in (q, k, v))
  head_dim = pl.cdiv(head_dim, _CHUNK_SIZE) * _CHUNK_SIZE
  q, k, v = map(lambda x: shape_lib.pad_dim_to(x, head_dim, -1), (q, k, v))

  q = shape_lib.pad_to_next_multiple_of(q, block_q, -3)
  k = shape_lib.pad_to_next_multiple_of(k, block_kv, -3)
  v = shape_lib.pad_to_next_multiple_of(v, block_kv, -3)
  q_seq, kv_seq = q.shape[-3], k.shape[-3]

  # Original implementation assumes input layout is (H, T, D).
  q, k, v = map(lambda x: jnp.swapaxes(x, -3, -2), (q, k, v))

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
  k_start, k_end = map(as_2d, (k_start, k_end))

  if k_start is not None and k_start.shape[-1] != 1:
    k_start = shape_lib.pad_to_next_multiple_of(k_start, block_q, -1, kv_seq)
  if k_end is not None and k_end.shape[-1] != 1:
    k_end = shape_lib.pad_to_next_multiple_of(k_end, block_q, -1, 0)

  use_q_smem_resident = config.q_smem
  num_p_slots = config.num_p_slots
  k_start_min, _ = _compute_k_range_minmax(k_start, block_q)
  _, k_end_max = _compute_k_range_minmax(k_end, block_q)
  v_split0 = config.v_split0
  v_split1 = head_dim - v_split0

  num_kv_blocks = kv_seq // block_kv
  stages_k = min(config.stages_k or config.num_stages, num_kv_blocks)
  stages_v = min(config.stages_v or config.num_stages, num_kv_blocks)
  num_chunks = head_dim // _CHUNK_SIZE
  num_staging_slots = _get_q_staging_slots(
      head_dim, block_q, block_kv, stages_k
  )

  mma_lhs = plgpu.Layout.MMA_LHS(dtype)
  mma_rhs = plgpu.Layout.MMA_RHS(dtype)
  mma_acc = plgpu.Layout.MMA_ACC(dtype)
  mma_row = mma_acc.reduce(1)

  def kernel(
      q_gmem,
      k_gmem,
      v_gmem,
      mask_gmem,
      bias_gmem,
      k_start_gmem,
      k_end_gmem,
      k_start_min_gmem,
      k_end_max_gmem,
      *output_gmems,
      qk_smem,
      v0_smem,
      v1_smem,
      p_smem,
      al_smem,
      p_ready_barrier,
      p_done_barrier,
  ):
    if not return_residuals:
      (out_gmem,) = output_gmems
    else:
      out_gmem, m_gmem, l_gmem = output_gmems

    if use_q_smem_resident:
      q_tile, k_smem = qk_smem
    else:
      (q_stage,), (k_smem,) = qk_smem

    wg = lax.axis_index("wg")
    hi, qi = (lax.axis_index(n) for n in ("heads", "q_tiles"))
    hi_kv = hi // q_heads_per_kv
    q_base = qi * block_q
    qs = pl.ds(q_base, block_q)
    out_gtile = out_gmem.at[hi, qs]

    ub = (
        jnp.minimum(num_kv_blocks, pl.cdiv(q_base + block_q, block_kv))
        if is_causal
        else num_kv_blocks
    )
    lb = 0
    if k_start_min_gmem is not None:
      val = common.load_bcast(k_start_min_gmem, (hi, qi), layout=None)
      lb = lax.max(lb, lax.div(val, block_kv))
    if k_end_max_gmem is not None:
      val = common.load_bcast(k_end_max_gmem, (hi, qi), layout=None)
      ub = jnp.minimum(ub, pl.cdiv(val, block_kv))

    load_k_block = lambda ki: k_gmem.at[hi_kv, pl.ds(ki * block_kv, block_kv)]
    load_v_block = lambda ki, lo, n: v_gmem.at[
        hi_kv, pl.ds(ki * block_kv, block_kv), pl.ds(lo, n)
    ]

    def _load_range_row(ref, shape):
      row = common.load_bcast(ref, (hi, pl.ds(q_base, block_q)), layout=mma_row)
      return plgpu.layout_cast(lax.broadcast_in_dim(row, shape, (0,)), mma_acc)

    def _apply_mask_and_bias(s: jax.Array, ki: int) -> jax.Array:
      s = s * logits_scale
      if bias_gmem is not None:
        bias_tile = common.load_bcast(
            bias_gmem,
            (hi, qs, pl.ds(ki * block_kv, block_kv)),
            layout=mma_acc,
            optimized=False,
        )
        s = s + bias_tile.astype(jnp.float32)
      if logits_soft_cap is not None:
        s = jnp.tanh(s / logits_soft_cap) * logits_soft_cap
      s = s * math.log2(math.e)

      keep = None
      if is_causal:
        q_pos = q_base + plgpu.broadcasted_iota(
            jnp.int32, s.shape, 0, layout=mma_acc
        )
        k_pos = ki * block_kv + plgpu.broadcasted_iota(
            jnp.int32, s.shape, 1, layout=mma_acc
        )
        keep = q_pos >= k_pos

      if k_start_gmem is not None or k_end_gmem is not None:
        k_pos = ki * block_kv + plgpu.broadcasted_iota(
            jnp.int32, s.shape, 1, layout=mma_acc
        )
        if k_start_gmem is not None:
          cond = k_pos >= _load_range_row(k_start_gmem, s.shape)
          keep = cond if keep is None else keep & cond
        if k_end_gmem is not None:
          cond = k_pos < _load_range_row(k_end_gmem, s.shape)
          keep = cond if keep is None else keep & cond

      if mask_gmem is not None:
        mask_tile = (
            common.load_bcast(
                mask_gmem,
                (hi, qs, pl.ds(ki * block_kv, block_kv)),
                layout=mma_acc,
                optimized=False,
            )
            != 0
        )
        keep = mask_tile if keep is None else keep & mask_tile

      if keep is not None:
        s = jnp.where(keep, s, _MASK_VALUE)
      return s

    def _split_head_pipeline():
      @pl.when(wg == 0)
      def _compute_qk_softmax_wg():
        if use_q_smem_resident:
          _async_copy_gmem_to_smem(q_gmem.at[hi, qs], q_tile)
          q_chunks = None
        else:
          q_chunks = _load_q_to_registers(
              q_gmem.at[hi, qs],
              q_stage.at[0],
              num_chunks,
              num_staging_slots,
              mma_lhs,
          )

        for i in range(stages_k):

          @pl.when(lb + i < ub)
          def _preload_k(i=i):
            _async_copy_gmem_to_smem(
                load_k_block(lb + i),
                k_smem.at[0, lax.rem(lb + i, stages_k)],
            )

        if v_split0:
          for i in range(stages_v):

            @pl.when(lb + i < ub)
            def _preload_v0(i=i):
              _async_copy_gmem_to_smem(
                  load_v_block(lb + i, 0, v_split0),
                  v0_smem.at[0, lax.rem(lb + i, stages_v)],
              )

        plgpu.wait_gmem_to_smem(0)

        def loop_body(ki, carry):
          m_i, l_i, acc = carry
          k_slot = lax.rem(ki, stages_k)
          v_slot = lax.rem(ki, stages_v)
          p_slot = lax.rem(ki, num_p_slots)

          @pl.when(ki - lb >= num_p_slots)
          def _wait_p_done():
            plgpu.barrier_wait(p_done_barrier.at[p_slot])

          plgpu.wait_gmem_to_smem(
              min(2 * stages_k - 1, 2 * (stages_v - 1))
              if v_split0
              else stages_k - 1
          )

          s = _apply_mask_and_bias(
              _compute_qk(
                  q_chunks,
                  q_tile if use_q_smem_resident else None,
                  k_smem.at[0, k_slot],
                  block_q,
                  block_kv,
                  num_chunks,
                  mma_acc,
                  mma_rhs,
                  mma_lhs,
              ),
              ki,
          )
          mgpu_lib.warpgroup_barrier()

          @pl.when(ki + stages_k < ub)
          def _copy_next_k():
            _async_copy_gmem_to_smem(
                load_k_block(ki + stages_k), k_smem.at[0, k_slot]
            )

          m_new, alpha, p, l_i = _online_softmax_step(s, m_i, l_i, mma_acc)
          p = p.astype(dtype)
          p_smem[p_slot] = p
          al_smem[p_slot] = alpha
          mgpu_lib.warpgroup_barrier()
          plgpu.barrier_arrive(p_ready_barrier.at[p_slot])

          if v_split0:
            acc = _compute_pv_step(
                _relayout_p_to_lhs(p, mma_acc, mma_lhs),
                acc,
                alpha,
                v0_smem.at[0, v_slot],
                (block_q, v_split0),
                mma_acc,
                mma_rhs,
            )
            mgpu_lib.warpgroup_barrier()

            @pl.when(ki + stages_v < ub)
            def _copy_next_v0():
              _async_copy_gmem_to_smem(
                  load_v_block(ki + stages_v, 0, v_split0),
                  v0_smem.at[0, v_slot],
              )

          return m_new, l_i, acc

        acc0 = plgpu.layout_cast(
            jnp.zeros((block_q, max(v_split0, 32)), jnp.float32), mma_acc
        )
        m_final, l_final, acc = lax.fori_loop(
            lb, ub, loop_body, (*_init_softmax_state(block_q, mma_row), acc0)
        )
        al_smem[num_p_slots] = l_final
        if return_residuals:
          plgpu.store(
              m_gmem.at[hi, qs],
              m_final * (1.0 / math.log2(math.e)),
              optimized=False,
          )
          plgpu.store(l_gmem.at[hi, qs], l_final, optimized=False)
        _cta_barrier()
        if v_split0:
          _write_output_epilogue(
              out_gtile,
              acc,
              l_final,
              0,
              v_split0,
              out_dtype,
              mma_acc,
              normalize_output,
          )

      @pl.when(wg == 1)
      def _compute_pv_tail_wg():
        for i in range(stages_v):

          @pl.when(lb + i < ub)
          def _preload_v1(i=i):
            _async_copy_gmem_to_smem(
                load_v_block(lb + i, v_split0, v_split1),
                v1_smem.at[lax.rem(lb + i, stages_v)],
            )

        def loop_body(ki, acc):
          v_slot = lax.rem(ki, stages_v)
          p_slot = lax.rem(ki, num_p_slots)
          plgpu.barrier_wait(p_ready_barrier.at[p_slot])
          alpha = plgpu.load(al_smem.at[p_slot], layout=mma_row)
          plgpu.wait_gmem_to_smem(stages_v - 1)
          acc = _compute_pv_step(
              plgpu.load(p_smem.at[p_slot], layout=mma_lhs),
              acc,
              alpha,
              v1_smem.at[v_slot],
              (block_q, v_split1),
              mma_acc,
              mma_rhs,
          )
          mgpu_lib.warpgroup_barrier()

          @pl.when(ki + stages_v < ub)
          def _copy_next_v1():
            _async_copy_gmem_to_smem(
                load_v_block(ki + stages_v, v_split0, v_split1),
                v1_smem.at[v_slot],
            )

          plgpu.barrier_arrive(p_done_barrier.at[p_slot])
          return acc

        acc0 = plgpu.layout_cast(
            jnp.zeros((block_q, v_split1), jnp.float32), mma_acc
        )
        acc = lax.fori_loop(lb, ub, loop_body, acc0)
        _cta_barrier()
        _write_output_epilogue(
            out_gtile,
            acc,
            plgpu.load(al_smem.at[num_p_slots], layout=mma_row),
            v_split0,
            v_split1,
            out_dtype,
            mma_acc,
            normalize_output,
        )

    _split_head_pipeline()

  k_smem_type = _create_tiled_smem((1, stages_k, block_kv, head_dim), dtype)
  if use_q_smem_resident:
    qk_smem = (_create_tiled_smem((block_q, head_dim), dtype), k_smem_type)
  else:
    q_stage_shape = (1, num_staging_slots, block_q, _CHUNK_SIZE)
    q_stage_type = _create_tiled_smem(q_stage_shape, dtype)
    qk_smem = plgpu.RefUnion((q_stage_type,), (k_smem_type,))

  scratch_allocations = dict(
      qk_smem=qk_smem,
      v0_smem=(
          _create_tiled_smem((1, stages_v, block_kv, v_split0), dtype)
          if v_split0
          else None
      ),
      v1_smem=_create_tiled_smem((stages_v, block_kv, v_split1), dtype),
      p_smem=_create_tiled_smem((num_p_slots, block_q, block_kv), dtype),
      al_smem=plgpu.SMEM((num_p_slots + 1, block_q), jnp.float32),
      p_ready_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=num_p_slots),
      p_done_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=num_p_slots),
  )

  out_type = (jax.ShapeDtypeStruct((num_q_heads, q_seq, head_dim), out_dtype),)
  if return_residuals:
    out_type += (jax.ShapeDtypeStruct((num_q_heads, q_seq), jnp.float32),) * 2

  out, *residuals = plgpu.kernel(
      kernel,
      out_type=out_type,
      grid=(num_q_heads, q_seq // block_q),
      grid_names=("heads", "q_tiles"),
      num_threads=_NUM_WARPGROUPS,
      thread_name="wg",
      scratch_types=scratch_allocations,  # pyrefly: ignore[bad-argument-type]
      compiler_params=plgpu.CompilerParams(
          approx_math=True,
          unsafe_no_auto_barriers=True,
          lowering_semantics=plgpu.LoweringSemantics.Lane,
      ),
      kernel_name="flash_attention_sm80_split_head",
  )(q, k, v, mask, bias, k_start, k_end, k_start_min, k_end_max)

  out = jnp.swapaxes(out, -3, -2)[:orig_q_seq_len, :, :orig_head_dim_out]
  residuals = tuple(res[..., :orig_q_seq_len] for res in residuals)
  return (out, residuals if residuals else None)

def get_autotuning_configs(ba: op.BoundArguments) -> set[Config]:
  """Returns a set of configs for autotuning flash attention on SM80 GPUs."""
  return {get_heuristics_config(ba, fold_q_sequence_heads=False)}
