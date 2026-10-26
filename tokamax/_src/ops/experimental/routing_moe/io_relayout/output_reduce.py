# Copyright 2026 Google LLC
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

"""Output relayout and top-k reduction helper from compact to large layout."""

import dataclasses
import functools
import math

import jax
from jax import lax
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.common import layout_util


@dataclasses.dataclass(frozen=True)
class Config:
  """Static configuration for the output_reduce kernel."""

  num_tokens: int
  topk: int
  chunks: int
  nb: int
  tpb: int
  tpu_info: pltpu.TpuInfo
  out_dtype: jax.typing.DTypeLike = jnp.bfloat16
  acc_dtype: jax.typing.DTypeLike = jnp.bfloat16
  max_tpb: int = 128
  tile: int = 16
  slots: int = 3
  lanes: int = 128
  packing: int = 2

  @classmethod
  def create(
      cls,
      *,
      num_tokens: int,
      topk: int,
      chunks: int,
      out_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      acc_dtype: jax.typing.DTypeLike = jnp.bfloat16,
  ):
    if jnp.dtype(acc_dtype) != jnp.bfloat16:
      raise NotImplementedError(
          f"Only bfloat16 acc_dtype is supported, got {acc_dtype}."
      )
    if num_tokens <= cls.max_tpb:
      tpb = pl.cdiv(num_tokens, cls.tile) * cls.tile
    else:
      tpb = math.gcd(num_tokens, cls.max_tpb)
      if tpb % cls.tile != 0:
        tpb = cls.max_tpb
    nb = pl.cdiv(num_tokens, tpb)
    cfg = cls(
        num_tokens=num_tokens,
        topk=topk,
        chunks=chunks,
        nb=nb,
        tpb=tpb,
        tpu_info=pltpu.get_tpu_info(),
        out_dtype=out_dtype,
        acc_dtype=acc_dtype,
    )
    if cfg.smem_allocation >= cfg.smem_threshold:
      raise ValueError(
          "Total number of SMEM allocation must be less than"
          f" {cfg.smem_threshold} to avoid OOM. Got {cfg.smem_allocation}"
      )
    return cfg

  @property
  def hidden_size(self) -> int:
    return self.chunks * self.packing * self.lanes

  @property
  def pad_t(self) -> int:
    return self.nb * self.tpb - self.num_tokens

  @property
  def stride(self) -> int:
    return layout_util.to_bank_conflict_aware_stride(self.tpb // self.packing)

  @property
  def smem_threshold(self) -> int:
    return int(0.8 * self.tpu_info.smem_capacity_bytes)

  @property
  def smem_allocation(self) -> int:
    num_words = (
        2 * (self.nb + self.slots - 1) + self.nb * self.topk
    ) * self.tpb + self.slots
    return num_words * 4

  @property
  def kernel_name(self) -> str:
    return f"output_reduce-k_{self.topk}-h_{self.hidden_size}"


def _kernel(
    k_lo_smem: hijax.TransformedRef,
    span_smem: hijax.TransformedRef,
    w_smem: hijax.TransformedRef,
    rows_hbm: hijax.TransformedRef,
    out_hbm: hijax.TransformedRef,  # HBM in / out
    slots: hijax.TransformedRef,
    outs: hijax.TransformedRef,
    stage: hijax.TransformedRef,
    n_valid_smem: hijax.TransformedRef,
    sems: hijax.TransformedRef,  # scratch
    *,
    cfg: Config,
) -> None:
  """Three slots: batch b is reduced from slot b % 3 while batch b + 2 is fetched."""
  K, chunks, nb, tpb = cfg.topk, cfg.chunks, cfg.nb, cfg.tpb
  lanes, packing, stride = cfg.lanes, cfg.packing, cfg.stride

  slots = pltpu.annotate(slots, no_hazard_no_deps=True)
  outs = pltpu.annotate(outs, no_hazard_no_deps=True)
  stage = pltpu.annotate(stage, no_hazard_no_deps=True)
  sem_rows, sem_out = sems.at[0], sems.at[1]
  rows = rows_hbm.bitcast(jnp.uint32).reshape(
      rows_hbm.shape[0] // K, K, chunks, lanes
  )

  def fetch_tok(
      batch: jax.Array | int, s: jax.Array | int, t: int
  ) -> jax.Array:
    """Copy token t's active pick span [k_lo, k_lo + span) into slot s."""
    k_lo = k_lo_smem[batch * tpb + t]
    span = span_smem[batch * tpb + t]
    pltpu.make_async_copy(
        rows.at[batch * tpb + t, pl.ds(k_lo, span)],
        slots.at[s, t, pl.ds(k_lo, span)],
        sem_rows.at[s],
    ).start()
    return span

  def reduce_tok(
      slot: hijax.TransformedRef,
      batch_weights: hijax.TransformedRef,
      t: int,
  ) -> jax.Array:
    """Weighted sum across top-k picks for token t."""
    acc = None
    for k in range(K):
      weight = pltpu.bitcast(
          jnp.broadcast_to(batch_weights[t * K + k], (chunks, lanes)),
          jnp.bfloat16,
      )
      weighted_row = pltpu.bitcast(slot[t, k], jnp.bfloat16) * weight
      weighted_row = jnp.where(weight != 0, weighted_row, 0)
      acc = weighted_row if acc is None else acc + weighted_row
    assert acc is not None
    return pltpu.bitcast(acc, jnp.uint32)

  # Prologue: Prefetch first cfg.slots - 1 input batches.
  for b in range(cfg.slots - 1):
    n_valid = 0
    for t in range(tpb):
      n_valid += fetch_tok(b, b, t)
    n_valid_smem[b] = n_valid

  @pl.loop(0, nb)
  def _(batch: jax.Array) -> None:
    # Step 1: Compute slot indices for the current batch and prefetch batch.
    slots_u32 = jnp.uint32(cfg.slots)
    s = lax.rem(batch.astype(jnp.uint32), slots_u32)
    next_b = batch + cfg.slots - 1
    s_fetch = lax.rem(next_b.astype(jnp.uint32), slots_u32)

    # Step 2: Wait for input DMA of current slot and output DMA of reused slot.
    some = rows_hbm.at[pl.ds(0, n_valid_smem[s])]
    pltpu.make_async_copy(some, some, sem_rows.at[s]).wait()
    out_wait = jnp.where(batch >= cfg.slots, tpb, 0)
    pltpu.make_async_copy(
        outs.at[s, pl.ds(0, out_wait)],
        outs.at[s, pl.ds(0, out_wait)],
        sem_out.at[s],
    ).wait()

    # Step 3: Reduce top-k picks for token pairs while issuing prefetch DMAs,
    # and pack transposed token pairs into the staging buffer.
    slot = slots.at[s]
    out_slot = outs.at[s]
    batch_weights = w_smem.at[pl.ds(batch * (tpb * K), tpb * K)]
    n_valid = 0
    for t0 in range(0, tpb, packing):
      acc0_u32 = reduce_tok(slot, batch_weights, t0)
      n_valid += fetch_tok(next_b, s_fetch, t0)
      acc1_u32 = reduce_tok(slot, batch_weights, t0 + 1)
      n_valid += fetch_tok(next_b, s_fetch, t0 + 1)

      for j in range(packing):
        u0 = pltpu.unpack_elementwise(
            acc0_u32,
            index=j,
            packed_dtype=jnp.bfloat16,
            unpacked_dtype=jnp.float32,
        )
        u1 = pltpu.unpack_elementwise(
            acc1_u32,
            index=j,
            packed_dtype=jnp.bfloat16,
            unpacked_dtype=jnp.float32,
        )
        off = j * chunks * stride + t0 // packing
        stage[off : off + chunks * stride : stride] = pltpu.pack_elementwise(
            [u0, u1], packed_dtype=jnp.bfloat16
        )
    n_valid_smem[s_fetch] = n_valid

    # Step 4: Unpack transposed chunks from stage into the output VMEM slot.
    for c in range(chunks):
      for j in range(packing):
        batch_u32 = stage[pl.ds((j * chunks + c) * stride, tpb // packing)]
        out_slot[:, pl.ds((packing * c + j) * lanes, lanes)] = (
            pltpu.bitcast(batch_u32, jnp.bfloat16).astype(cfg.out_dtype)
        )

    # Step 5: Start async output DMA from VMEM to HBM.
    pltpu.make_async_copy(
        outs.at[s], out_hbm.at[pl.ds(batch * tpb, tpb)], sem_out.at[s]
    ).start()

  # Epilogue: Drain remaining output buffers.
  for b in range(max(0, nb - cfg.slots), nb):
    s = b % cfg.slots
    pltpu.make_async_copy(
        outs.at[s], out_hbm.at[pl.ds(b * tpb, tpb)], sem_out.at[s]
    ).wait()


@jax.jit(static_argnames=("num_local_experts", "out_dtype", "acc_dtype"))
def output_reduce_unpack(
    rows: arg.ActivationWrapper[jax.Array] | jax.Array,
    weights: jax.Array,
    topk_indices: jax.Array | None = None,
    group_offset: jax.Array | int | None = None,
    num_local_experts: int | None = None,
    shard_idx: jax.Array | None = None,
    *,
    out_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    acc_dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> jax.Array:
  """Reduces top-k rows and unpacks from compact 4D layout to 2D large layout.

  If `rows.spec.row_shard_axis` is set, `rows` hold all topk rows of this
  device's token shard only: the output covers those tokens, and every row is
  valid regardless of expert.
  """
  axis = None
  if isinstance(rows, arg.ActivationWrapper):
    axis = rows.spec.row_shard_axis
    row_size = rows.spec.row_size
    rows = rows.data
  else:
    row_size = rows.shape[0]

  if (
      rows.dtype != jnp.bfloat16
      or rows.ndim != 4
      or rows.shape[2:] != (2, Config.lanes)
  ):
    raise ValueError(
        "rows must be bf16[T * K, H / 256, 2, 128], got"
        f" {rows.dtype}{rows.shape}"
    )
  if weights.ndim != 2:
    raise ValueError(f"weights must be 2D [T, K], got {weights.shape}")

  num_tokens, topk = weights.shape
  if axis is not None:
    num_local_tokens = row_size // topk
    if num_local_tokens * topk != row_size:
      raise ValueError(
          f"Rows ({row_size}) must hold whole tokens of topk {topk}."
      )
    if num_local_tokens * jax.lax.axis_size(axis) != num_tokens:
      raise ValueError(
          f"{num_tokens} tokens must divide evenly across axis {axis!r}."
      )

    if shard_idx is None:
      shard_idx = jax.lax.axis_index(axis)
    elif not jnp.isscalar(shard_idx):
      shard_idx = shard_idx[0]
    start = shard_idx * num_local_tokens
    weights = jax.lax.dynamic_slice_in_dim(weights, start, num_local_tokens)
    num_tokens = num_local_tokens
  else:
    if rows.shape[0] != num_tokens * topk:
      raise ValueError(
          f"weights must be [T, K] with T * K == {rows.shape[0]}, got"
          f" {weights.shape}"
      )
    if topk_indices is not None:
      if topk_indices.shape != weights.shape:
        raise ValueError(
            f"topk_indices.shape {topk_indices.shape} must match weights.shape"
            f" {weights.shape}"
        )
      if group_offset is None or num_local_experts is None:
        raise ValueError(
            "group_offset and num_local_experts are required when"
            " topk_indices is provided"
        )
      local = jnp.logical_and(
          topk_indices >= group_offset,
          topk_indices < group_offset + num_local_experts,
      )
      weights = jnp.where(local, weights, 0)

  cfg = Config.create(
      num_tokens=num_tokens,
      topk=topk,
      chunks=rows.shape[1],
      out_dtype=out_dtype,
      acc_dtype=acc_dtype,
  )
  if cfg.pad_t:
    weights = jnp.pad(weights, ((0, cfg.pad_t), (0, 0)))
  w_padded = jnp.pad(weights, ((0, (cfg.slots - 1) * cfg.tpb), (0, 0)))

  # For each token, find the first and last non-zero pick [k_lo, k_hi] so
  # fetch_tok can issue a single contiguous DMA of length `span`.
  valid = w_padded != 0
  k_idx = jnp.arange(topk, dtype=jnp.int32)
  k_lo = jnp.min(jnp.where(valid, k_idx, topk), axis=1)
  k_hi = jnp.max(jnp.where(valid, k_idx, -1), axis=1)
  span = jnp.maximum(k_hi - k_lo + 1, 0)
  k_lo = jnp.where(span > 0, k_lo, 0)

  # Duplicate each bf16 weight into both 16-bit halves of a uint32 word so a
  # single 32-bit SMEM broadcast populates both packed bf16 sublanes.
  w16 = lax.bitcast_convert_type(
      weights.astype(jnp.bfloat16), jnp.uint16
  ).astype(jnp.uint32)
  # Pack [w16, w16] into bits [31:16] and [15:0] of each uint32 element.
  # Avoids per-pick SALU unpack/repack overhead from 16-bit SMEM loads.
  w_u32 = ((w16 << 16) | w16).reshape(-1)

  k_lo = pltpu.with_memory_space_constraint(k_lo, pltpu.SMEM)
  span = pltpu.with_memory_space_constraint(span, pltpu.SMEM)
  w_u32 = pltpu.with_memory_space_constraint(w_u32, pltpu.SMEM)
  out = pl.kernel(
      functools.partial(_kernel, cfg=cfg),
      out_type=jax.ShapeDtypeStruct(
          (cfg.nb * cfg.tpb, cfg.hidden_size), out_dtype
      ),
      mesh=pltpu.TensorCoreMesh(axis_name="core", num_cores=1),
      scratch_types=dict(
          slots=pltpu.VMEM(
              (cfg.slots, cfg.tpb, topk, cfg.chunks, cfg.lanes),
              jnp.uint32,
          ),
          outs=pltpu.VMEM((cfg.slots, cfg.tpb, cfg.hidden_size), out_dtype),
          stage=pltpu.VMEM(
              (cfg.packing * cfg.chunks * cfg.stride, cfg.lanes),
              jnp.uint32,
          ),
          n_valid_smem=pltpu.SMEM((cfg.slots,), jnp.int32),
          sems=pltpu.SemaphoreType.DMA((2, cfg.slots)),
      ),
      compiler_params=pltpu.CompilerParams(
          disable_bounds_checks=True,
          vmem_limit_bytes=cfg.tpu_info.vmem_capacity_bytes,
      ),
      name=cfg.kernel_name,
  )(k_lo, span, w_u32, rows)
  return out[:num_tokens] if cfg.pad_t else out
