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
"""Configuration dataclasses, dimensions, and validation for Batched MLA.

This module defines configuration structures and hardware layout rules for
Multi-Head Latent Attention (MLA) on TPU.
"""

import dataclasses
import enum
from typing import Any

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from tokamax._src.ops.experimental.mla.v3 import utils


@dataclasses.dataclass(frozen=True)
class BlockSizes:
  """Tuning block sizes and tiling parameters for the MLA kernel.

  Attributes:
    bq_sz: Query block size (number of sequence tokens per Q block).
    bq_c_sz: Chunked query/compute block size for split execution. Must divide
      `bq_sz`.
    bkv_sz: KV cache block size (number of context tokens processed per step).
    batch_size: Number of batch lanes packed into one kernel step. The lanes are
      chained *serially* within a step -- lane i's online-softmax state (m, l,
      acc) is rolled forward into lane i+1 -- so this is a work-per-step /
      DMA-overlap knob, not a parallelism knob.
    n_buffer: Pipelining buffer depth (e.g. 2 for double buffering).
  """

  bq_sz: int
  bq_c_sz: int
  bkv_sz: int
  batch_size: int
  n_buffer: int

  def __post_init__(self):
    # Attention walks `bq_sz // bq_c_sz` chunks of `bq_c_sz` query rows (see
    # `MlaConfigs.q_split`), so any remainder would never be computed.
    if self.bq_c_sz < 1:
      raise ValueError(f"Expected {self.bq_c_sz=} to be positive.")
    if self.bq_sz % self.bq_c_sz != 0:
      raise ValueError(
          f"Expected {self.bq_sz=} to be a multiple of {self.bq_c_sz=}:"
          " attention runs `bq_sz // bq_c_sz` chunks of `bq_c_sz` query rows,"
          " so the rest of each query block would be silently skipped."
      )


@dataclasses.dataclass(frozen=True)
class MlaModelConfigs:
  """Immutable architectural parameters of the MLA model.

  MLA decomposes attention representations into:
    - Non-positional latent query/key/value vectors of dimension `lkv_dim`
      (e.g., 512 in DeepSeek-V2/V3).
    - Decoupled rotary embedding (RoPE) query/key vectors of dimension `r_dim`
      (e.g., 64 in DeepSeek-V2/V3).
    - Single compressed KV head (H_kv = 1) shared across all `num_q_heads`
    (e.g., 128).

  Attributes:
    num_q_heads: Number of query attention heads (e.g. 128).
    lkv_dim: Dimension of latent non-positional KV vector d_nope (e.g. 512).
    r_dim: Dimension of decoupled RoPE positional key vector d_pe (e.g. 64).
    mask_value: Large negative value used for causal/padding masking.
    sm_scale: Softmax temperature scale factor (default: 1.0 / sqrt(d_q)).
    soft_cap: Optional logit soft-capping threshold (e.g., Gemma-2 style).
    sliding_window: Optional sliding window attention horizon in tokens.
  """

  num_q_heads: int
  lkv_dim: int
  r_dim: int
  mask_value: float
  sm_scale: float = 1.0
  soft_cap: float | None = None
  sliding_window: int | None = None


class KVLayout(enum.StrEnum):
  """Memory layout of the paged KV cache in HBM and VMEM.

  - SEQ_ALONG_LANE: Sequence tokens are packed along the 128 TPU physical lanes;
      latent dimension is on sublanes.

  This is the only layout v3 implements, and the enum exists to name it rather
  than to offer a choice.
  """

  SEQ_ALONG_LANE = enum.auto()

  @property
  def symbol(self) -> str:
    return "snh"


@dataclasses.dataclass(frozen=True)
class ServingConfigs:
  """Workload and serving batch configuration parameters.

  Attributes:
    num_seqs: Maximum number of active sequences in the current batch.
    page_size: Paged memory page size in tokens (must be multiple of 128 for
      SEQ_ALONG_LANE).
    total_q_tokens: Total flattened query tokens across all sequences (sum of
      q_lens).
    num_page_indices: Size of the flattened page index table (num_seqs *
      pages_per_seq).
    dtype_q: Data type for query tensors (e.g., jnp.bfloat16).
    dtype_kv: Data type for KV cache tensors (e.g., jnp.bfloat16).
    dtype_out: Data type for attention output tensors (e.g., jnp.bfloat16).
    scale_q: Optional scalar quantization multiplier for Q.
    scale_k: Optional scalar quantization multiplier for K.
    scale_v: Optional scalar quantization multiplier for V.
    kv_layout: Paged memory layout. Only SEQ_ALONG_LANE is implemented.
    smem_fraction_limit_for_schedule_generation: SMEM budget limit fraction.
    max_schedule_size_multiplier: Floor on the multiplier sizing the HBM
      schedule, in units of `MlaConfigs.max_steps_ub`. Raising it only
      over-allocates; `MlaConfigs.max_schedule_size_multiplier` already raises
      it on its own for any shape that provably needs more.
  """

  num_seqs: int
  page_size: int
  total_q_tokens: int
  num_page_indices: int
  dtype_q: Any
  dtype_kv: Any
  dtype_out: Any
  scale_q: float | None = None
  scale_k: float | None = None
  scale_v: float | None = None
  kv_layout: KVLayout = KVLayout.SEQ_ALONG_LANE
  smem_fraction_limit_for_schedule_generation: float = 0.33
  max_schedule_size_multiplier: int = 16

  @property
  def pages_per_seq(self) -> int:
    return self.num_page_indices // self.num_seqs

  @property
  def page_size_log2(self) -> int:
    return (self.page_size - 1).bit_length()

  @property
  def page_size_mask(self) -> int:
    return self.page_size - 1

  @property
  def packing_q(self) -> int:
    """Number of elements packed per 32-bit word (e.g. 2 for bfloat16)."""
    return utils.get_dtype_packing(self.dtype_q)

  @property
  def packing_kv(self) -> int:
    """Number of elements packed per 32-bit word (e.g. 2 for bfloat16)."""
    return utils.get_dtype_packing(self.dtype_kv)

class MlaCase(enum.StrEnum):
  """Execution mode for the MLA kernel.

  - DECODE: All sequences are in autoregressive decode (q_len = 1).
  - PREFILL: All sequences are in prompt prefill (q_len > 1, static).
  - MIXED: Batch contains a combination of prefill and decode sequences.
  """

  DECODE = enum.auto()
  PREFILL = enum.auto()
  MIXED = enum.auto()

  @property
  def symbol(self) -> str:
    match self:
      case MlaCase.DECODE:
        return "d"
      case MlaCase.PREFILL:
        return "p"
      case MlaCase.MIXED:
        return "m"

  def get_range(self, distribution: Any) -> tuple[Any, Any]:
    """Extracts sequence start and end indices for this execution mode."""
    match self:
      case MlaCase.DECODE:
        return 0, distribution[0]
      case MlaCase.PREFILL:
        return distribution[0], distribution[1]
      case MlaCase.MIXED:
        return distribution[1], distribution[2]


@dataclasses.dataclass(frozen=True, eq=True)
class MlaConfigs:
  """Master configuration combining block sizes, model, serving, and hardware constraints."""

  block: BlockSizes
  model: MlaModelConfigs
  serve: ServingConfigs
  mode: MlaCase
  vmem_limit_bytes: int = 16 * 1024 * 1024

  # Expose block sizes directly for convenient access
  @property
  def bq_sz(self) -> int:
    return self.block.bq_sz

  @property
  def bq_c_sz(self) -> int:
    return self.block.bq_c_sz

  @property
  def bkv_sz(self) -> int:
    return self.block.bkv_sz

  @property
  def batch_size(self) -> int:
    return self.block.batch_size

  @property
  def n_buffer(self) -> int:
    return self.block.n_buffer

  @property
  def kv_buffer_count(self) -> tuple[int, int]:
    """(in, out) buffer counts for the in/out KV staging buffer.

    The KV writeback (`KVBufferedRefSeqAlongLane.copy_out`) is async and is
    only waited one step later. A plain int `n` means `(n, 1)` to Pallas: it
    assumes a synchronous writeback and allocates `n` slots, so the copy-in
    issued at step t refills step t-1's slot before step t-1's writeback has
    been waited, and the writeback can read the new block's data. A second
    output buffer adds one slot, which delays reuse of a slot until after its
    writeback is waited.
    """
    return (self.n_buffer, 2)

  @property
  def kv_num_slots(self) -> int:
    """Physical KV staging slots; Pallas allocates `in + out - 1`."""
    in_count, out_count = self.kv_buffer_count
    return in_count + out_count - 1

  @property
  def q_split(self) -> int:
    """Number of query sub-chunks (bq_sz // bq_c_sz)."""
    return max(1, self.bq_sz // self.bq_c_sz)

  # Derived hardware alignment dimensions
  @property
  def kv_dim_align(self) -> int:
    """Granularity the KV sub-dimensions are padded to in VMEM: 128 lanes.
    """
    return pltpu.get_tpu_info().num_lanes

  @property
  def aligned_lkv_dim(self) -> int:
    """d_nope (512), padded to 128 lanes."""
    return utils.align_to(self.model.lkv_dim, self.kv_dim_align)

  @property
  def aligned_r_dim(self) -> int:
    """d_pe (64 -> 128), padded to 128 lanes. See `kv_dim_align`."""
    return utils.align_to(self.model.r_dim, self.kv_dim_align)

  @property
  def lkv_sublanes(self) -> int:
    """Sublanes the latent part occupies. Same in HBM and VMEM."""
    return self.aligned_lkv_dim // self.serve.packing_kv

  @property
  def aligned_kv_dim(self) -> int:
    """Combined KV dimension [C_kv, K_pe] aligned to 128-byte multiples."""
    return self.aligned_lkv_dim + self.aligned_r_dim

  @property
  def aligned_q_dim(self) -> int:
    """Combined Query dimension [Q_nope, Q_pe] aligned to 128-byte multiples."""
    return self.aligned_lkv_dim + self.aligned_r_dim

  @property
  def aligned_num_q_heads(self) -> int:
    """Number of Q heads aligned to word-packing boundary."""
    packing_q = self.serve.packing_q
    return utils.align_to(self.model.num_q_heads, packing_q)

  # Paged KV block calculations
  @property
  def bkv_p(self) -> int:
    """Number of physical pages spanning a single KV block."""
    return pl.cdiv(self.block.bkv_sz, self.serve.page_size)

  @property
  def one_new_token(self) -> bool:
    """Whether a lane-task can receive at most one new (unpaged) KV token.

    This is a statement about the *sequence*, not about the query block, and
    that distinction is a real correctness boundary. `bq_sz == 1` was used for
    it historically, which is wrong: a prefill sequence processed one query
    token at a time still has `q_len` new KV tokens to stitch in. On
    `[(256, 1024)]` at `num_queries_per_block=1` that took the decode stitch
    path, which writes exactly one lane and *zeroes* everything past the
    boundary -- so 255 of 256 new tokens were dropped. Query token `i` attends
    to `i` of those zeroed positions, so the error grew with `i` and crossed
    tolerance around `i = 20`, mismatching 1.372% of the output.

    Only DECODE guarantees one new token per sequence, so only DECODE may take
    the fast paths gated on this.
    """
    return self.mode == MlaCase.DECODE

  @property
  def bkv_p_wb(self) -> int:
    """Number of pages to write back to the paged KV cache per step.

    - In DECODE: at most 1 new token is written back, touching at most 1 page.
    - Otherwise: a KV block is page-aligned and spans `bkv_p` pages, so new
      tokens in the block touch at most `bkv_p` pages.
    """
    if self.one_new_token:
      return 1
    return self.bkv_p

  @property
  def p_same_dtype_as_v(self) -> bool:
    """Whether to cast softmax probabilities P to V's dtype before the PV matmul.

    On PREFILL/MIXED, the score tile is large enough that narrowing P so the MXU
    sees v.dtype x v.dtype rather than f32 x v.dtype is a major win (-16.1% on
    chunked_prefill_f8_kv8192). On DECODE (bq_sz = 1), the vector-unit cast pass
    costs more than the MXU savings (+5.9% to +9.4% slower), so it stays off on
    DECODE.
    """
    return self.mode != MlaCase.DECODE

  @property
  def max_steps_ub(self) -> int:
    """Calculates the maximum schedule steps that can fit in TPU SMEM."""
    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    fixed_bytes = (
        self.serve.num_seqs  # kv_lens
        + (self.serve.num_seqs + 1)  # cu_q_lens
        + (self.serve.num_seqs * self.serve.pages_per_seq)  # page_indices
        + 3  # distribution [decode, prefill, total]
        + self.block.batch_size  # lane_lengths
        # actual_steps: 1 word, but Mosaic pads 1D 32-bit SMEM refs to a
        # lane-row tile.
        + num_lanes
    ) * 4  # 4 bytes per int32

    smem_limit_bytes = (
        tpu_info.smem_capacity_bytes - 32 * 1024
    ) * self.serve.smem_fraction_limit_for_schedule_generation
    available_bytes = smem_limit_bytes - fixed_bytes

    # Per-step SMEM footprint, derived from the `MlaSchedule` leaves (int32
    # words, 4 bytes each). Per lane:
    #   s_idx, q_idx, k_idx, is_last_k, do_writeback, dma_q[2]  -> 7 words
    bytes_scalars_per_lane = 28
    #   dma_kv_cache[bkv_p, 2] -> 2 words per page. Budgeted as 3 words per
    #   page on purpose: the extra word per page absorbs the lane-tile padding
    #   Mosaic adds to every 1D SMEM leaf, which is not otherwise accounted for
    #   here. Keep this conservative rather than tightening it to 8 bytes.
    bytes_cache_dma_per_lane = 12 * self.bkv_p
    #   dma_kv_new_fetch[3] + dma_kv_new_wb[bkv_p_wb, 3] -> 3 words each
    bytes_new_dma_per_lane = 12 * (1 + self.bkv_p_wb)
    # Per step: total_wait_{kv_in, kv_out, q_in, o_out} -> 4 words
    bytes_global_waits = 16

    bytes_per_step = (
        bytes_scalars_per_lane
        + bytes_cache_dma_per_lane
        + bytes_new_dma_per_lane
    ) * self.block.batch_size + bytes_global_waits

    max_steps_ub = available_bytes // bytes_per_step
    return int(max(1, max_steps_ub // num_lanes) * num_lanes)

  # Scratch buffer shapes in VMEM
  @property
  def lm_scratch_shape(self) -> tuple[int, ...]:
    """Scratch shape for running row-max (m) and row-sum (l) vectors."""
    num_lanes = pltpu.get_tpu_info().num_lanes
    return (
        self.block.bq_sz * self.aligned_num_q_heads,
        num_lanes,
    )

  @property
  def acc_scratch_shape(self) -> tuple[int, ...]:
    """Scratch shape for accumulator matrix (width is strictly d_nope = 512)."""
    return (
        self.block.bq_sz * self.aligned_num_q_heads,
        self.aligned_lkv_dim,
    )

  @property
  def kv_token_align(self) -> int:
    """DMA token granularity: the 128-lane tile.

    Tokens are packed along the lane dim (SEQ_ALONG_LANE), so every KV DMA
    token count and lane offset must be a multiple of the lane tile. It used
    to be `page_size`, which was pinned to 128; pages may now be any
    power-of-two multiple of 128, and rounding every fetch up to a page
    wastes bandwidth and VMEM.
    """
    return pltpu.get_tpu_info().num_lanes

  @property
  def kv_vmem_lanes(self) -> int:
    """Lane extent of the KV staging buffer: `bkv_sz` plus stitch slack.

    Cache pages land at `i * page_size` with their token count rounded up to
    `kv_token_align`, so the cached region ends at
    `align_to(bkv_sz_cache, kv_token_align)`. The new tokens are fetched in one
    DMA starting there, widened to `kv_token_align` on both ends. Each [] is a
    128-lane tile (o = cache, n = new, - = pad):

      want  [onnn][nnnn]
      got   [o---][---n][nnnn][nn--]

    so the buffer needs two tiles past `bkv_sz`: one for rounding
    `bkv_sz_cache` up, one for the new tokens' offset within their tile.
    """
    return self.block.bkv_sz + 2 * self.kv_token_align

  @property
  def kv_vmem_shape(self) -> tuple[int, ...]:
    """VMEM allocation shape for KV buffer [batch_size, aligned_kv_dim, lanes]."""
    return (
        self.block.batch_size,
        self.aligned_kv_dim,
        self.kv_vmem_lanes,
    )

  @property
  def q_vmem_shape(self) -> tuple[int, ...]:
    """VMEM allocation shape for combined Query buffer [batch_size, bq_sz, num_q_heads, aligned_q_dim]."""
    return (
        self.block.batch_size,
        self.block.bq_sz,
        self.aligned_num_q_heads,
        self.aligned_q_dim,
    )

  @property
  def o_vmem_shape(self) -> tuple[int, ...]:
    """VMEM allocation shape for output buffer [batch_size, bq_sz, num_q_heads, lkv_dim]."""
    return (
        self.block.batch_size,
        self.block.bq_sz,
        self.aligned_num_q_heads,
        self.aligned_lkv_dim,
    )

  @property
  def max_steps_needed(self) -> int:
    """Upper bound on the schedule steps *any* input of this shape can need.

    The schedule loop increments a counter once per (sequence, q-block, k-block)
    task and packs `batch_size` consecutive tasks into one step, so bounding the
    task count bounds the step count. Both factors below hold for every ragged
    split of `total_q_tokens` across `num_seqs` sequences, which is what makes
    this computable at trace time from shapes alone.

    Nothing here can be tightened by looking at the actual `cu_q_lens` /
    `kv_lens`, because those are runtime values; the bound has to cover the
    worst split the shapes permit. It does assume `kv_len <= pages_per_seq *
    page_size` for every sequence, but a `kv_len` past the end of its page table
    is already an out-of-contract input that the kernel would read the wrong
    pages for.
    """
    # Sequence `s` contributes `cdiv(q_len_s, bq_sz)` q-blocks. Two bounds on
    # the sum are available and neither dominates: `cdiv(q, b) <= q` is tight
    # for decode (many one-token sequences), `cdiv(q, b) <= q // b + 1` is tight
    # for prefill (few long ones).
    q_blocks_ub = min(
        self.serve.total_q_tokens,
        self.serve.num_seqs + self.serve.total_q_tokens // self.block.bq_sz,
    )
    # A q-block visits at most every KV block the page table can address for
    # its sequence. Causal and sliding-window masking only remove blocks from
    # that range, so ignoring both is safe (and, for prefill, loose by ~2x).
    k_blocks_ub = pl.cdiv(
        self.serve.pages_per_seq * self.serve.page_size, self.block.bkv_sz
    )
    return pl.cdiv(q_blocks_ub * k_blocks_ub, self.block.batch_size)

  @property
  def max_schedule_size_multiplier(self) -> int:
    """Sizes the HBM schedule at `max_steps_ub *` this, in steps.

    Raised above the configured value whenever the shape provably needs more
    room, so the schedule can never outgrow its HBM buffer. Sizing is the fix
    rather than validation: the bound is a worst-case over ragged splits, so
    rejecting shapes that merely *might* overflow would reject shapes that in
    practice never do, whereas over-allocating costs only HBM - a few hundred
    bytes per step, in a buffer the kernel streams rather than resides in.
    """
    return max(
        self.serve.max_schedule_size_multiplier,
        pl.cdiv(self.max_steps_needed, self.max_steps_ub),
    )
