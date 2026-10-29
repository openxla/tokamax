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
import json
import math
from typing import Self

import jax
from jax import lax
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.routing_moe.common import arg
from tokamax._src.ops.experimental.routing_moe.common import layout_util


@dataclasses.dataclass(frozen=True)
class TokenWeights:
  """A token's `[k_lo, span, w[0], ..., w[topk - 1]]` record in SMEM.

  `k_lo` / `span` bound the token's non-zero weights so its rows are fetched
  with one contiguous DMA; `w[k]` is the bf16 weight of pick k in both 16-bit
  halves of the word.
  """

  smem: hijax.TransformedRef
  pos: jax.Array | int

  @property
  def k_lo(self) -> jax.Array:
    return self.smem[self.pos]

  @property
  def span(self) -> jax.Array:
    return self.smem[self.pos + 1]

  def weight(self, k: int) -> jax.Array:
    return self.smem[self.pos + 2 + k]


@dataclasses.dataclass(frozen=True)
class Config:
  """Static configuration for the output_reduce kernel."""

  num_tokens: int
  topk: int
  chunks: int
  num_batches: int
  tokens_per_batch: int
  tpu_info: pltpu.TpuInfo
  in_dtype: jnp.dtype
  out_dtype: jnp.dtype
  acc_dtype: jnp.dtype = jnp.bfloat16.dtype
  max_tokens_per_batch: int = 128
  tile: int = 16
  # Input VMEM buffers (and matching output buffers): batch b is reduced from
  # buffer b % num_input_buffers while batch b + num_input_buffers - 1 is
  # fetched into its own buffer.
  num_input_buffers: int = 3

  @property
  def in_packing(self) -> int:
    return 4 // self.in_dtype.itemsize

  @property
  def acc_packing(self) -> int:
    return 4 // self.acc_dtype.itemsize

  @property
  def lanes(self) -> int:
    return self.tpu_info.num_lanes

  @classmethod
  def create(
      cls,
      *,
      rows: jax.Array,
      weights: jax.Array,
      out_dtype: jax.typing.DTypeLike,
      acc_dtype: jax.typing.DTypeLike,
      tpu_info: pltpu.TpuInfo,
  ) -> Self:
    """Derives batching parameters and validates SMEM capacity."""
    chunks = rows.shape[1]
    num_tokens, topk = weights.shape
    in_dtype = rows.dtype
    out_dtype = jnp.dtype(out_dtype)
    acc_dtype = jnp.dtype(acc_dtype)
    if acc_dtype != jnp.bfloat16:
      raise NotImplementedError(
          f"Only bfloat16 acc_dtype is supported, got {acc_dtype}."
      )
    if num_tokens <= cls.max_tokens_per_batch:
      tokens_per_batch = pl.cdiv(num_tokens, cls.tile) * cls.tile
    else:
      tokens_per_batch = math.gcd(num_tokens, cls.max_tokens_per_batch)
      if tokens_per_batch % cls.tile != 0:
        tokens_per_batch = cls.max_tokens_per_batch
    num_batches = pl.cdiv(num_tokens, tokens_per_batch)
    cfg = cls(
        num_tokens=num_tokens,
        topk=topk,
        chunks=chunks,
        num_batches=num_batches,
        tokens_per_batch=tokens_per_batch,
        tpu_info=tpu_info,
        out_dtype=out_dtype,
        in_dtype=in_dtype,
        acc_dtype=acc_dtype,
    )
    if cfg.smem_allocation >= cfg.smem_threshold:
      raise ValueError(
          "Total number of SMEM allocation must be less than"
          f" {cfg.smem_threshold} to avoid OOM. Got {cfg.smem_allocation}"
      )

    if (
        rows.dtype != jnp.bfloat16
        or rows.ndim != 4
        or rows.shape[2:] != (2, cfg.lanes)
    ):
      raise ValueError(
          "rows must be bf16[T * K, H / 256, 2, 128], got"
          f" {rows.dtype}{rows.shape}"
      )
    if weights.ndim != 2:
      raise ValueError(f"weights must be 2D [T, K], got {weights.shape}")

    num_tokens, topk = weights.shape
    if rows.shape[0] != num_tokens * topk:
      raise ValueError(
          f"weights must be [T, K] with T * K == {rows.shape[0]}, got"
          f" {weights.shape}"
      )
    return cfg

  @property
  def hidden_size(self) -> int:
    return self.chunks * self.in_packing * self.lanes

  @property
  def pad_tokens(self) -> int:
    return self.num_batches * self.tokens_per_batch - self.num_tokens

  @property
  def stride(self) -> int:
    return layout_util.to_bank_conflict_aware_stride(self.chunks)

  @property
  def num_topk_weights_buffers(self) -> int:
    # `num_input_buffers` batches [b .. b + num_input_buffers - 1] hold live
    # top-k weights in SMEM while 1 extra buffer prefetches the weights of
    # batch b + num_input_buffers from HBM.
    return self.num_input_buffers + 1

  @property
  def topk_weights_record_size(self) -> int:
    # Words per token: [k_lo, span, w[topk]], see `TokenWeights`.
    return 2 + self.topk

  @property
  def topk_weights_size(self) -> int:
    # Words per batch: `tokens_per_batch` records, padded to a lane multiple.
    words = self.tokens_per_batch * self.topk_weights_record_size
    return pl.cdiv(words, self.lanes) * self.lanes

  @property
  def topk_weights_pad(self) -> int:
    words = self.tokens_per_batch * self.topk_weights_record_size
    return self.topk_weights_size - words

  @property
  def smem_threshold(self) -> int:
    # Same budget as gmm/config.py: cap explicit scratch at 45% of SMEM so the
    # compiler keeps enough room for scalar spills, loop state and sflags.
    # The ring buffer makes `smem_allocation` independent of `num_batches`, so
    # this only trips for very large `topk`.
    return int(0.45 * self.tpu_info.smem_capacity_bytes)

  @property
  def smem_allocation(self) -> int:
    topk_weights_words = self.num_topk_weights_buffers * self.topk_weights_size
    num_rows_fetched_words = self.num_input_buffers
    return (topk_weights_words + num_rows_fetched_words) * 4

  @property
  def kernel_name(self) -> str:
    return (
        f"output_reduce-t_{self.num_tokens}_k_{self.topk}-h_{self.hidden_size}"
    )

  @property
  def kernel_metadata(self) -> dict[str, str]:
    """The JSON-serialized config metadata for XProf."""

    def to_json(obj: object) -> object:
      if isinstance(obj, pltpu.TpuInfo):
        return str(obj.chip_version)
      if dataclasses.is_dataclass(obj):
        return vars(obj)
      return str(obj)

    return {"xprof_metadata": json.dumps(self, default=to_json)}

  def scratch_types(self) -> dict[str, pl.MemoryRef]:
    """Scratch buffers of `_kernel`, keyed by its parameter names."""
    return dict(
        input_buffer=pltpu.VMEM(
            (
                self.num_input_buffers,
                self.tokens_per_batch,
                self.topk,
                self.chunks,
                self.in_packing,
                self.lanes,
            ),
            self.in_dtype,
        ),
        out_buffer=pltpu.VMEM(
            (self.num_input_buffers, self.tokens_per_batch, self.hidden_size),
            self.out_dtype,
        ),
        relayout_buffer=pltpu.VMEM(
            (self.tokens_per_batch, self.stride, self.acc_packing, self.lanes),
            self.acc_dtype,
        ),
        topk_weights_smem=pltpu.SMEM(
            (self.num_topk_weights_buffers * self.topk_weights_size,),
            jnp.int32,
        ),
        num_rows_fetched_smem=pltpu.SMEM((self.num_input_buffers,), jnp.int32),
        input_sems=pltpu.SemaphoreType.DMA((self.num_input_buffers,)),
        out_sems=pltpu.SemaphoreType.DMA((self.num_input_buffers,)),
        topk_weights_sems=pltpu.SemaphoreType.DMA((
            self.num_topk_weights_buffers,
        )),
    )


def _reduce_token(
    input_rows: hijax.TransformedRef,
    token_weights: TokenWeights,
    token: int,
    *,
    cfg: Config,
) -> jax.Array:
  """Weighted sum across a token's top-k picks, as packed bf16 words."""
  acc = None
  input_rows_u32 = input_rows.bitcast(jnp.uint32).reshape(
      cfg.tokens_per_batch, cfg.topk, -1, cfg.lanes
  )
  for k in range(cfg.topk):
    weight = jnp.broadcast_to(token_weights.weight(k), (cfg.chunks, cfg.lanes))
    weight = pltpu.bitcast(weight, cfg.in_dtype).astype(cfg.acc_dtype)
    input_row = pltpu.bitcast(input_rows_u32[token, k], cfg.in_dtype).astype(
        cfg.acc_dtype
    )
    weighted_row = input_row * weight
    weighted_row = jnp.where(weight != 0, weighted_row, 0)
    acc = weighted_row if acc is None else acc + weighted_row
  assert acc is not None
  return acc


def _kernel(
    topk_weights_hbm: hijax.TransformedRef,
    rows_hbm: hijax.TransformedRef,
    out_hbm: hijax.TransformedRef,  # HBM in / out
    input_buffer: hijax.TransformedRef,
    out_buffer: hijax.TransformedRef,
    relayout_buffer: hijax.TransformedRef,
    topk_weights_smem: hijax.TransformedRef,
    num_rows_fetched_smem: hijax.TransformedRef,
    input_sems: hijax.TransformedRef,  # scratch
    out_sems: hijax.TransformedRef,
    topk_weights_sems: hijax.TransformedRef,
    *,
    cfg: Config,
) -> None:
  """Reduces batch b from input_buffer[b % 3] while batch b + 2 is fetched."""
  input_buffer = pltpu.annotate(input_buffer, no_hazard_no_deps=True)
  out_buffer = pltpu.annotate(out_buffer, no_hazard_no_deps=True)
  rows = rows_hbm.reshape(-1, cfg.topk, cfg.chunks, cfg.in_packing, cfg.lanes)
  input_buffer_flat = input_buffer.reshape(
      cfg.num_input_buffers, -1, cfg.chunks, cfg.in_packing, cfg.lanes
  )

  def fetch_topk_weights(
      batch: jax.Array | int, weights_slot: jax.Array | int
  ) -> None:
    """Copy batch's top-k weights from HBM into SMEM ring slot weights_slot."""
    pltpu.make_async_copy(
        topk_weights_hbm.at[
            pl.ds(batch * cfg.topk_weights_size, cfg.topk_weights_size)
        ],
        topk_weights_smem.at[
            pl.ds(weights_slot * cfg.topk_weights_size, cfg.topk_weights_size)
        ],
        topk_weights_sems.at[weights_slot],
    ).start()

  def wait_topk_weights(weights_slot: jax.Array | int) -> None:
    """Wait for the top-k weights DMA into SMEM ring slot `weights_slot`."""
    copied = topk_weights_smem.at[
        pl.ds(weights_slot * cfg.topk_weights_size, cfg.topk_weights_size)
    ]
    pltpu.make_async_copy(
        copied, copied, topk_weights_sems.at[weights_slot]
    ).wait()

  def token_weights_at(
      weights_slot: jax.Array | int, token: int
  ) -> TokenWeights:
    """The record of `token` (index within its batch) in ring slot."""
    pos = weights_slot * cfg.topk_weights_size
    pos += token * cfg.topk_weights_record_size
    return TokenWeights(topk_weights_smem, pos)

  def fetch_token_rows(
      batch: jax.Array | int,
      weights_slot: jax.Array | int,
      input_slot: jax.Array | int,
      token: int,
  ) -> jax.Array:
    """Copy a token's active picks [k_lo, k_lo + span) into an input buffer."""
    token_weights = token_weights_at(weights_slot, token)
    k_lo, span = token_weights.k_lo, token_weights.span
    pltpu.make_async_copy(
        rows.at[batch * cfg.tokens_per_batch + token, pl.ds(k_lo, span)],
        input_buffer.at[input_slot, token, pl.ds(k_lo, span)],
        input_sems.at[input_slot],
    ).start()
    return span

  # Prologue: Prefetch top-k weights for the first `num_input_buffers`
  # batches and rows for the first `num_input_buffers - 1` batches.
  for b in range(cfg.num_input_buffers):
    fetch_topk_weights(b, b)
  for b in range(cfg.num_input_buffers - 1):
    wait_topk_weights(b)
    num_rows_fetched = 0
    for token in range(cfg.tokens_per_batch):
      num_rows_fetched += fetch_token_rows(b, b, b, token)
    num_rows_fetched_smem[b] = num_rows_fetched

  @pl.loop(0, cfg.num_batches)
  def _(batch: jax.Array) -> None:
    # Step 1: Compute the VMEM buffer and SMEM ring slots of the current batch,
    # the batch whose rows are fetched this iteration, and the batch whose
    # top-k weights are prefetched this iteration.
    num_input_buffers_u32 = jnp.uint32(cfg.num_input_buffers)
    num_weights_buffers_u32 = jnp.uint32(cfg.num_topk_weights_buffers)
    input_slot = lax.rem(batch.astype(jnp.uint32), num_input_buffers_u32)
    weights_slot = lax.rem(batch.astype(jnp.uint32), num_weights_buffers_u32)
    fetch_batch = batch + cfg.num_input_buffers - 1
    fetch_input_slot = lax.rem(
        fetch_batch.astype(jnp.uint32), num_input_buffers_u32
    )
    fetch_weights_slot = lax.rem(
        fetch_batch.astype(jnp.uint32), num_weights_buffers_u32
    )
    prefetch_batch = batch + cfg.num_input_buffers
    prefetch_weights_slot = lax.rem(
        prefetch_batch.astype(jnp.uint32), num_weights_buffers_u32
    )

    # Step 2: Wait for the current batch's row DMAs, the fetch batch's top-k
    # weights DMA, and the output DMA that last used this buffer; then start
    # the top-k weights prefetch for batch + num_input_buffers.
    wait_topk_weights(fetch_weights_slot)

    @pl.when(batch + 1 < cfg.num_batches)
    def _():
      fetch_topk_weights(prefetch_batch, prefetch_weights_slot)

    fetched = input_buffer_flat.at[
        input_slot, pl.ds(0, num_rows_fetched_smem[input_slot])
    ]
    pltpu.make_async_copy(fetched, fetched, input_sems.at[input_slot]).wait()
    out_wait_rows = jnp.where(
        batch >= cfg.num_input_buffers, cfg.tokens_per_batch, 0
    )
    pltpu.make_async_copy(
        out_buffer.at[input_slot, pl.ds(0, out_wait_rows)],
        out_buffer.at[input_slot, pl.ds(0, out_wait_rows)],
        out_sems.at[input_slot],
    ).wait()

    # Step 3: Reduce top-k picks for token pairs while issuing the fetch
    # batch's row DMAs, and pack transposed token pairs into relayout_buffer.
    input_rows = input_buffer.at[input_slot].reshape(
        cfg.tokens_per_batch, cfg.topk, cfg.chunks, cfg.in_packing, cfg.lanes
    )
    out_rows = out_buffer.at[input_slot]
    num_rows_fetched = 0
    relayout_buffer_u32 = relayout_buffer.bitcast(jnp.uint32).reshape(
        cfg.tokens_per_batch, -1, cfg.lanes
    )
    for token in range(cfg.tokens_per_batch):
      acc = _reduce_token(
          input_rows, token_weights_at(weights_slot, token), token, cfg=cfg
      )
      num_rows_fetched += fetch_token_rows(
          fetch_batch, fetch_weights_slot, fetch_input_slot, token
      )
      acc_u32 = pltpu.bitcast(acc, jnp.uint32)
      relayout_buffer_u32[token, : acc_u32.shape[0]] = acc_u32
    num_rows_fetched_smem[fetch_input_slot] = num_rows_fetched

    # Step 4: Unpack transposed chunks from relayout_buffer into the output
    # VMEM buffer, one 128-lane output column block at a time.
    relayout_vreg = layout_util.load_compact_to_large(
        relayout_buffer, cfg.hidden_size
    )
    out_rows[...] = relayout_vreg.astype(cfg.out_dtype)

    # Step 5: Start async output DMA from VMEM to HBM.
    pltpu.make_async_copy(
        out_buffer.at[input_slot],
        out_hbm.at[pl.ds(batch * cfg.tokens_per_batch, cfg.tokens_per_batch)],
        out_sems.at[input_slot],
    ).start()

  # Epilogue: Drain remaining output buffers.
  first_undrained = max(0, cfg.num_batches - cfg.num_input_buffers)
  for b in range(first_undrained, cfg.num_batches):
    out_slot = b % cfg.num_input_buffers
    pltpu.make_async_copy(
        out_buffer.at[out_slot],
        out_hbm.at[pl.ds(b * cfg.tokens_per_batch, cfg.tokens_per_batch)],
        out_sems.at[out_slot],
    ).wait()


@jax.jit(static_argnames=("num_local_experts", "out_dtype", "acc_dtype"))
def output_reduce_unpack(
    rows: arg.ActivationWrapper[jax.Array] | jax.Array,
    weights: jax.Array,
    topk_indices: jax.Array | None = None,
    group_offset: jax.Array | int | None = None,
    num_local_experts: int | None = None,
    *,
    out_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    acc_dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> jax.Array:
  """Reduces top-k rows and unpacks from compact 4D to large 2D layout."""
  if isinstance(rows, arg.ActivationWrapper):
    rows = rows.data

  cfg = Config.create(
      rows=rows,
      weights=weights,
      out_dtype=out_dtype,
      acc_dtype=acc_dtype,
      tpu_info=pltpu.get_tpu_info(),
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

  # Pad tokens to whole batches, plus the `num_input_buffers - 1` batches the
  # kernel's prologue fetches past the last one; zero weights copy nothing.
  prologue_tokens = (cfg.num_input_buffers - 1) * cfg.tokens_per_batch
  pad_tokens = cfg.pad_tokens + prologue_tokens
  w_padded = jnp.pad(weights, ((0, pad_tokens), (0, 0)))

  # For each token, find the first and last non-zero pick [k_lo, k_hi] so
  # fetch_token_rows can issue a single contiguous DMA of length `span`.
  valid = w_padded != 0
  k_idx = jnp.arange(cfg.topk, dtype=jnp.int32)
  k_lo = jnp.min(jnp.where(valid, k_idx, cfg.topk), axis=1)
  k_hi = jnp.max(jnp.where(valid, k_idx, -1), axis=1)
  span = jnp.maximum(k_hi - k_lo + 1, 0)
  k_lo = jnp.where(span > 0, k_lo, 0)

  # Duplicate each bf16 weight into both 16-bit halves of int32 so a
  # single 32-bit SMEM broadcast populates both packed bf16 sublanes.
  w16 = lax.bitcast_convert_type(
      w_padded.astype(jnp.bfloat16), jnp.uint16
  ).astype(jnp.int32)
  # Pack [w16, w16] into bits [31:16] and [15:0] of each int32 element.
  # Avoids per-pick SALU unpack/repack overhead from 16-bit SMEM loads.
  w_i32 = (w16 << 16) | w16
  # One `TokenWeights` record per token, [k_lo, span, w[0..topk)], grouped
  # into per-batch blocks streamed HBM -> SMEM (see Config.topk_weights_size).
  num_weights_batches = cfg.num_batches + cfg.num_input_buffers - 1
  topk_weights = jnp.concatenate(
      [k_lo[:, None], span[:, None], w_i32], axis=1
  ).reshape(num_weights_batches, -1)
  topk_weights = jnp.pad(topk_weights, ((0, 0), (0, cfg.topk_weights_pad)))
  topk_weights = pltpu.with_memory_space_constraint(
      topk_weights.reshape(-1), pltpu.HBM
  )
  out = pl.kernel(
      functools.partial(_kernel, cfg=cfg),
      out_type=jax.ShapeDtypeStruct(
          (cfg.num_batches * cfg.tokens_per_batch, cfg.hidden_size),
          cfg.out_dtype,
      ),
      mesh=pltpu.TensorCoreMesh(axis_name="core", num_cores=1),
      scratch_types=cfg.scratch_types(),
      compiler_params=pltpu.CompilerParams(
          disable_bounds_checks=True,
          vmem_limit_bytes=cfg.tpu_info.vmem_capacity_bytes,
      ),
      name=cfg.kernel_name,
      metadata=cfg.kernel_metadata,
  )(topk_weights, rows)
  return out[: cfg.num_tokens]
