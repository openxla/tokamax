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
"""Configuration and static layout dataclasses for Batched RPA."""

import dataclasses
import enum
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa.kernel import utils


@dataclasses.dataclass(frozen=True)
class BlockSizes:
  """Tuning and pipeline-buffering parameters for the RPA kernel.

  Attributes:
    bq_sz: Query block size.
    bq_c_sz: Query chunk block size for prefill.
    bkv_sz: Key/Value block size.
    batch_size: Batch size.
    num_input_slots: VMEM multi-buffering slots for Q/KV inputs (3 for 2-step
      lookahead).
    num_output_slots: VMEM multi-buffering slots for O/LSE outputs (2).
  """

  bq_sz: int
  bq_c_sz: int
  bkv_sz: int
  batch_size: int
  num_input_slots: int = 3
  num_output_slots: int = 2

  def __post_init__(self):
    if self.num_input_slots != 3 or self.num_output_slots != 2:
      raise ValueError(
          "run_manual_triple_buffer requires num_input_slots=3 and"
          f" num_output_slots=2, got ({self.num_input_slots},"
          f" {self.num_output_slots})."
      )


@dataclasses.dataclass(frozen=True)
class ModelConfigs:
  """Model config that stays constant across requests.

  Attributes:
    num_q_heads: Number of query attention heads.
    num_kv_heads: Number of key/value attention heads.
    head_dim: Dimensionality of each attention head.
    mask_value: Attention mask value for masked positions.
    sm_scale: Softmax scaling factor.
    soft_cap: Soft-capping threshold for attention logits.
    sliding_window: Size of the sliding attention window.
  """

  num_q_heads: int
  num_kv_heads: int
  head_dim: int
  mask_value: float
  sm_scale: float = 1.0
  soft_cap: float | None = None
  sliding_window: int | None = None

  @property
  def num_q_heads_per_kv_head(self) -> int:
    return self.num_q_heads // self.num_kv_heads


class AttentionScope(enum.StrEnum):
  """Which KV positions to attend to.

  FULL:            attend all positions (default).
  CACHE_ONLY:      attend only cached tokens, skip new tokens.
  NEW_TOKENS_ONLY: attend only new tokens, skip cached tokens.
  """

  FULL = enum.auto()
  CACHE_ONLY = enum.auto()
  NEW_TOKENS_ONLY = enum.auto()


class KVLayout(enum.StrEnum):
  """Represents the different layouts for KV cache.

  - HEAD_ALONG_SUBLANE: Number of heads on sublane, head_dim on lane.
  - SEQ_ALONG_LANE: Sequence is packed along the lane, head_dim on sublane.
  """

  HEAD_ALONG_SUBLANE = enum.auto()
  SEQ_ALONG_LANE = enum.auto()

  @property
  def symbol(self) -> str:
    match self:
      case KVLayout.HEAD_ALONG_SUBLANE:
        return "snh"
      case KVLayout.SEQ_ALONG_LANE:
        return "nhs"


def num_lane_folds_in_head_dim(
    head_dim: int,
    dtype_kv: jnp.dtype,
    kv_layout: KVLayout,
    num_lanes: int,
) -> int:
  """Number of lane folds of a K/V head when `head_dim` is on the lane dim."""
  if (
      kv_layout != KVLayout.HEAD_ALONG_SUBLANE
      or jnp.dtype(dtype_kv) == jnp.uint8
  ):
    return 1
  if utils.get_dtype_packing(dtype_kv) < 4:
    return 1
  fold = utils.align_to(head_dim, num_lanes) // num_lanes
  while fold > 1 and head_dim % (fold * num_lanes) != 0:
    fold //= 2
  return fold


class RpaCase(enum.StrEnum):
  """Represents the different cases for Ragged Paged Attention.

  - DECODE: Sequences are in decode-only mode (q_len = 1).
  - PREFILL: Sequences are in prefill-only mode (q_len > 1, static). Note that
    Batched RPA folds PREFILL into MIXED, but the enum value is kept because
    distribution still has three entries and RPA v3 / MLA use it.
  - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
  """

  DECODE = enum.auto()
  PREFILL = enum.auto()
  MIXED = enum.auto()

  @property
  def symbol(self) -> str:
    match self:
      case RpaCase.DECODE:
        return "d"
      case RpaCase.PREFILL:
        return "p"
      case RpaCase.MIXED:
        return "m"

  def get_range(
      self, distribution: jax.Array
  ) -> tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]:
    assert distribution.shape == (3,)
    match self:
      case RpaCase.DECODE:
        return 0, distribution[0]
      case RpaCase.PREFILL:
        return distribution[0], distribution[1]
      case RpaCase.MIXED:
        return distribution[1], distribution[2]


@dataclasses.dataclass(frozen=True)
class ServingConfigs:
  """Serving config that can change depending on use cases."""

  num_seqs: int
  page_size: int
  total_q_tokens: int
  num_page_indices: int
  dtype_q: jnp.dtype
  dtype_kv: jnp.dtype
  dtype_out: jnp.dtype
  per_token_scale: bool | None = None
  per_token_scale_dtype: jnp.dtype | None = None
  scale_q: int | float | None = None
  scale_k: int | float | None = None
  scale_v: int | float | None = None
  kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE
  smem_fraction_limit_for_schedule_generation: float = 0.33
  max_schedule_size_multiplier: int = 16
  page_indices_window_size: int = 2048
  decode_query_size: int = 1
  cp_group_size: int | None = None
  attention_scope: AttentionScope = AttentionScope.FULL
  return_lse: bool = False
  skip_kv_update: bool = False
  quantize_p: bool = False

  @property
  def matmuls_are_8bit(self) -> bool:
    """True when both the qk and pv matmuls run at the 8-bit MXU rate.

    The MXU picks its data format from the LHS operand alone, evaluated after
    any in-kernel cast. qk's LHS is q, which is cast down when scale_q is set.
    pv's LHS is p, which is cast down only when quantize_p is set.
    """
    kv_is_8bit = jnp.dtype(self.dtype_kv).itemsize == 1
    qk_is_8bit = jnp.dtype(self.dtype_q).itemsize == 1 or (
        self.scale_q is not None and kv_is_8bit
    )
    pv_is_8bit = self.quantize_p and kv_is_8bit
    return qk_is_8bit and pv_is_8bit

  @property
  def max_decode_bkv_p_new(self) -> int:
    return 1 + pl.cdiv(self.decode_query_size - 1, self.page_size)

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
    return utils.get_dtype_packing(self.dtype_q)

  @property
  def packing_kv(self) -> int:
    return utils.get_dtype_packing(self.dtype_kv)

  @property
  def scale_channels(self) -> int:
    if self.per_token_scale:
      assert self.per_token_scale_dtype is not None
      scale_bits = jax.dtypes.itemsize_bits(self.per_token_scale_dtype)
      kv_bits = jax.dtypes.itemsize_bits(self.dtype_kv)
      return max(1, scale_bits // kv_bits)
    return 0


@dataclasses.dataclass(frozen=True)
class LayoutConfig:
  """Static tensor packing, alignment, and shape rules usable before BlockSizes."""

  num_q_heads: int
  num_kv_heads: int
  head_dim: int
  page_size: int
  dtype_q: jnp.dtype
  dtype_kv: jnp.dtype
  dtype_out: jnp.dtype
  kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE
  per_token_scale: bool = False
  per_token_scale_dtype: jnp.dtype | None = None
  _num_lanes: int | None = None
  _num_sublanes: int | None = None

  @classmethod
  def from_configs(
      cls,
      model: ModelConfigs,
      serve: ServingConfigs,
      *,
      num_lanes: int | None = None,
      num_sublanes: int | None = None,
  ) -> "LayoutConfig":
    return cls(
        num_q_heads=model.num_q_heads,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        page_size=serve.page_size,
        dtype_q=serve.dtype_q,
        dtype_kv=serve.dtype_kv,
        dtype_out=serve.dtype_out,
        kv_layout=serve.kv_layout,
        per_token_scale=bool(serve.per_token_scale),
        per_token_scale_dtype=serve.per_token_scale_dtype,
        _num_lanes=num_lanes,
        _num_sublanes=num_sublanes,
    )

  @property
  def is_seq_along_lane(self) -> bool:
    return self.kv_layout == KVLayout.SEQ_ALONG_LANE

  @property
  def num_lanes(self) -> int:
    if self._num_lanes is not None:
      return self._num_lanes
    return pltpu.get_tpu_info().num_lanes

  @property
  def num_sublanes(self) -> int:
    if self._num_sublanes is not None:
      return self._num_sublanes
    return pltpu.get_tpu_info().num_sublanes

  @property
  def softmax_dtype(self) -> jnp.dtype:
    return jnp.dtype(self.dtype_out)

  @property
  def packing_q(self) -> int:
    return utils.get_dtype_packing(self.dtype_q)

  @property
  def packing_kv(self) -> int:
    return utils.get_dtype_packing(self.dtype_kv)

  @property
  def num_q_heads_per_kv_head(self) -> int:
    return self.num_q_heads // self.num_kv_heads

  @property
  def aligned_num_q_heads_per_kv_head(self) -> int:
    return utils.align_to(self.num_q_heads_per_kv_head, self.packing_q)

  @property
  def aligned_num_q_heads(self) -> int:
    return self.aligned_num_q_heads_per_kv_head * self.num_kv_heads

  @property
  def aligned_q_head_dim(self) -> int:
    return utils.align_to(self.head_dim, self.num_lanes)

  @property
  def scale_channels(self) -> int:
    if self.per_token_scale:
      assert self.per_token_scale_dtype is not None
      scale_bits = jax.dtypes.itemsize_bits(self.per_token_scale_dtype)
      kv_bits = jax.dtypes.itemsize_bits(self.dtype_kv)
      return max(1, scale_bits // kv_bits)
    return 0

  @property
  def physical_kv_head_dim(self) -> int:
    if self.dtype_kv == jnp.uint8:
      return self.head_dim // 2
    return self.head_dim

  @property
  def num_lane_folds_in_head_dim(self) -> int:
    return num_lane_folds_in_head_dim(
        self.head_dim, self.dtype_kv, self.kv_layout, self.num_lanes
    )

  @property
  def compute_kv_head_dim(self) -> int:
    """Head dimension used for compute (without scale channels)."""
    if self.is_seq_along_lane:
      return utils.align_to(self.head_dim, self.num_sublanes * self.packing_kv)
    return utils.align_to(self.head_dim, self.num_lanes)

  @property
  def storage_kv_head_dim(self) -> int:
    """Head dimension stored in HBM/VMEM (includes scale channels & fp4 pack)."""
    if self.is_seq_along_lane:
      total_head_dim = self.physical_kv_head_dim + self.scale_channels
      return utils.align_to(total_head_dim, self.num_sublanes * self.packing_kv)
    return utils.align_to(
        self.head_dim // self.num_lane_folds_in_head_dim, self.num_lanes
    )

  @property
  def aligned_num_kv_heads_x2(self) -> int:
    if self.is_seq_along_lane:
      return self.num_kv_heads * 2
    return utils.align_to(
        self.num_kv_heads * 2 * self.num_lane_folds_in_head_dim,
        self.packing_kv,
    )

  @property
  def kv_hbm_stride(self) -> int:
    if self.is_seq_along_lane:
      return self.num_kv_heads * 2
    return self.aligned_num_kv_heads_x2 // self.packing_kv

  @property
  def bkv_stride(self) -> int:
    stride = pl.cdiv(
        self.num_kv_heads * 2 * self.num_lane_folds_in_head_dim,
        self.packing_kv,
    )
    if utils.has_bank_conflicts(stride):
      stride += 1
    return stride

  @property
  def vmem_kv_heads_x2(self) -> int:
    if self.is_seq_along_lane:
      return self.num_kv_heads * 2
    return self.bkv_stride * self.packing_kv

  @property
  def kv_token_align(self) -> int:
    """DMA token granularity: 128-lane tile for SEQ_ALONG_LANE, else 1."""
    if self.is_seq_along_lane:
      return self.num_lanes
    return 1

  @property
  def kv_bytes_per_token(self) -> int:
    num_elements = self.aligned_num_kv_heads_x2 * self.storage_kv_head_dim
    kv_dtype_bits = utils.get_dtype_bits(self.dtype_kv)
    return int(num_elements * kv_dtype_bits) >> 3

  @property
  def q_bytes_per_token(self) -> int:
    q_dtype_bits = utils.get_dtype_bits(self.dtype_q)
    return (
        int(self.aligned_num_q_heads * self.aligned_q_head_dim * q_dtype_bits)
        >> 3
    )

  @property
  def o_bytes_per_token(self) -> int:
    out_dtype_bits = utils.get_dtype_bits(self.dtype_out)
    return (
        int(self.aligned_num_q_heads * self.aligned_q_head_dim * out_dtype_bits)
        >> 3
    )

  @property
  def lse_bytes_per_token(self) -> int:
    lse_dtype_bits = utils.get_dtype_bits(self.softmax_dtype)
    return int(self.aligned_num_q_heads * self.num_lanes * lse_dtype_bits) >> 3

  @property
  def dma_wait_word_dtype(self) -> jnp.dtype:
    """Word dtype used when reshaping buffers into `[rows, num_lanes]` waits."""
    return jnp.dtype(jnp.uint32)

  @property
  def dma_wait_row_bytes(self) -> int:
    """Bytes per `dma_wait_word_dtype[num_lanes]` row awaited by `_wait_lanes`."""
    return self.num_lanes * self.dma_wait_word_dtype.itemsize

  def tokens_to_wait_rows(
      self, tokens: int | jax.Array, bytes_per_token: int
  ) -> int | jax.Array:
    """Converts a transferred token count into `dma_wait_row_bytes` wait rows."""
    return (tokens * bytes_per_token) // self.dma_wait_row_bytes

  def kv_vmem_lanes(self, bkv_sz: int) -> int:
    """Lane-dim size of the KV VMEM buffer; SEQ_ALONG_LANE only."""
    # o = kv from cache, n = kv from new, - = pad. Each [] is a 128-lane cell:
    #
    #   want  [onnn][nnnn]
    #   got   [o---][---n][nnnn][nn--]
    #
    # because we fetch at 128-lane boundary, we need 2 extra padding slots.
    return bkv_sz + 2 * self.kv_token_align

  def kv_cache_shape(self, total_num_pages: int) -> tuple[int, ...]:
    if self.is_seq_along_lane:
      return (
          total_num_pages,
          self.num_kv_heads * 2,
          self.storage_kv_head_dim // self.packing_kv,
          self.packing_kv,
          self.page_size,
      )
    return (
        total_num_pages,
        self.page_size,
        self.kv_hbm_stride,
        self.packing_kv,
        self.storage_kv_head_dim,
    )

  def q_hbm_shape(self, total_q_tokens: int) -> tuple[int, ...]:
    return (
        self.num_kv_heads,
        total_q_tokens,
        self.aligned_num_q_heads_per_kv_head // self.packing_q,
        self.packing_q,
        self.aligned_q_head_dim,
    )

  def lse_hbm_shape(self, total_q_tokens: int) -> tuple[int, ...]:
    return (
        self.num_kv_heads,
        total_q_tokens,
        self.aligned_num_q_heads_per_kv_head // self.packing_q,
        self.packing_q,
        self.num_lanes,
    )

  def q_vmem_shape(self, batch_size: int, bq_sz: int) -> tuple[int, ...]:
    return (
        batch_size,
        self.num_kv_heads,
        bq_sz,
        self.aligned_num_q_heads_per_kv_head // self.packing_q,
        self.packing_q,
        self.aligned_q_head_dim,
    )

  def kv_vmem_shape(self, batch_size: int, bkv_sz: int) -> tuple[int, ...]:
    if self.is_seq_along_lane:
      return (
          batch_size,
          self.num_kv_heads * 2,
          self.storage_kv_head_dim // self.packing_kv,
          self.packing_kv,
          self.kv_vmem_lanes(bkv_sz),
      )
    return (
        batch_size,
        bkv_sz,
        self.bkv_stride,
        self.packing_kv,
        self.storage_kv_head_dim,
    )

  def lse_vmem_shape(self, batch_size: int, bq_sz: int) -> tuple[int, ...]:
    return (
        batch_size,
        self.num_kv_heads,
        bq_sz,
        self.aligned_num_q_heads_per_kv_head // self.packing_q,
        self.packing_q,
        self.num_lanes,
    )

  def lm_scratch_shape(self, bq_sz: int) -> tuple[int, ...]:
    return (
        self.num_kv_heads,
        bq_sz * self.aligned_num_q_heads_per_kv_head,
        self.num_lanes,
    )

  def acc_scratch_shape(self, bq_sz: int) -> tuple[int, ...]:
    return (
        self.num_kv_heads,
        bq_sz * self.aligned_num_q_heads_per_kv_head,
        self.compute_kv_head_dim,
    )


@dataclasses.dataclass(frozen=True, eq=True)
class RpaConfigs:
  """Complete per-pass configuration for a DECODE or MIXED RPA kernel call."""

  block: BlockSizes
  model: ModelConfigs
  serve: ServingConfigs
  mode: RpaCase
  vmem_limit_bytes: int

  @property
  def layout(self) -> LayoutConfig:
    return LayoutConfig.from_configs(self.model, self.serve)

  @property
  def is_seq_along_lane(self) -> bool:
    return self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE

  # Expose block sizes for ease of use.

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
  def num_input_slots(self) -> int:
    return self.block.num_input_slots

  @property
  def num_output_slots(self) -> int:
    return self.block.num_output_slots

  def seq_range(
      self, distribution: jax.Array | jax.Ref
  ) -> tuple[int | jax.Array, int | jax.Array]:
    """Sequences this kernel call schedules.

    Batched RPA has no specialised prefill-only path, so MIXED also covers the
    prefill-only segment [distribution[0], distribution[1]).
    """
    if self.mode == RpaCase.DECODE:
      return 0, distribution[0]
    if self.mode == RpaCase.MIXED:
      return distribution[0], distribution[2]
    raise ValueError(f"Batched RPA does not dispatch {self.mode}; use MIXED.")

  @property
  def page_window_seqs(self) -> int:
    """Sequences per page_indices DMA window: power of two dividing num_seqs."""
    max_rows = max(
        1, self.serve.page_indices_window_size // self.serve.pages_per_seq
    )
    rows = 1 << (max_rows.bit_length() - 1)
    return math.gcd(rows, self.serve.num_seqs)

  @property
  def page_window_entries(self) -> int:
    """Flat page_indices entries covered by one window."""
    return self.page_window_seqs * self.serve.pages_per_seq

  @property
  def page_window_slot_len(self) -> int:
    """SMEM entries per window slot (lane-aligned, plus slack if unaligned)."""
    num_lanes = self.layout.num_lanes
    entries = self.page_window_entries
    return utils.align_to(entries, num_lanes) + (
        num_lanes if entries % num_lanes else 0
    )

  @property
  def bkv_p(self) -> int:
    return pl.cdiv(self.block.bkv_sz, self.serve.page_size)

  @property
  def bkv_p_cache(self) -> int:
    if self.mode == RpaCase.PREFILL:
      return 0
    return self.bkv_p

  @property
  def bkv_p_new(self) -> int:
    if self.mode == RpaCase.DECODE or self.block.bq_sz == 1:
      return min(self.serve.max_decode_bkv_p_new, self.bkv_p)
    return self.bkv_p

  @property
  def fuse_accum(self) -> bool:
    return self.mode == RpaCase.DECODE

  @property
  def max_schedule_size_multiplier(self) -> int:
    return self.serve.max_schedule_size_multiplier

  # Layout delegates (kept as convenience pass-throughs to self.layout).

  @property
  def num_lane_folds_in_head_dim(self) -> int:
    return self.layout.num_lane_folds_in_head_dim

  @property
  def bkv_stride(self) -> int:
    return self.layout.bkv_stride

  @property
  def aligned_q_head_dim(self) -> int:
    return self.layout.aligned_q_head_dim

  @property
  def compute_kv_head_dim(self) -> int:
    return self.layout.compute_kv_head_dim

  @property
  def aligned_kv_head_dim(self) -> int:
    return self.layout.storage_kv_head_dim

  @property
  def aligned_num_kv_heads_x2(self) -> int:
    return self.layout.aligned_num_kv_heads_x2

  @property
  def aligned_num_q_heads_per_kv_head(self) -> int:
    return self.layout.aligned_num_q_heads_per_kv_head

  @property
  def kv_hbm_stride(self) -> int:
    return self.layout.kv_hbm_stride

  @property
  def kv_bytes_per_token(self) -> int:
    return self.layout.kv_bytes_per_token

  @property
  def q_bytes_per_token(self) -> int:
    return self.layout.q_bytes_per_token

  @property
  def o_bytes_per_token(self) -> int:
    return self.layout.o_bytes_per_token

  @property
  def kv_token_align(self) -> int:
    return self.layout.kv_token_align

  @property
  def kv_vmem_lanes(self) -> int:
    return self.layout.kv_vmem_lanes(self.block.bkv_sz)

  @property
  def q_vmem_shape(self) -> tuple[int, ...]:
    return self.layout.q_vmem_shape(self.block.batch_size, self.block.bq_sz)

  @property
  def kv_vmem_shape(self) -> tuple[int, ...]:
    return self.layout.kv_vmem_shape(self.block.batch_size, self.block.bkv_sz)

  @property
  def lm_scratch_shape(self) -> tuple[int, ...]:
    return self.layout.lm_scratch_shape(self.block.bq_sz)

  @property
  def lse_vmem_shape(self) -> tuple[int, ...]:
    return self.layout.lse_vmem_shape(self.block.batch_size, self.block.bq_sz)

  @property
  def acc_scratch_shape(self) -> tuple[int, ...]:
    return self.layout.acc_scratch_shape(self.block.bq_sz)

  def validate_inputs(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      kv_cache: jax.Array,
      kv_lens: jax.Array,
      page_indices: jax.Array,
      cu_q_lens: jax.Array,
      distribution: jax.Array,
      attention_sink: jax.Array | None = None,
  ):
    """Validate inputs to the RPA kernel statically."""
    layout = self.layout

    if not q.ndim == k.ndim == v.ndim == 3:
      raise ValueError(
          f"Expected 3D array for {q.shape=}, {k.shape=}, {v.shape=}"
      )
    if k.shape != v.shape:
      raise ValueError(f"Expected {k.shape=} to be equal to {v.shape=}")
    if not (q.shape[0] == k.shape[0] == v.shape[0]):
      raise ValueError(
          "Expected number of sequences in Q, K, and V to be the same, but got"
          f" {q.shape[0]=}, {k.shape[0]=}, and {v.shape[0]=}"
      )
    expected_kv_head_dim = layout.physical_kv_head_dim

    if not (k.shape[2] == v.shape[2] == expected_kv_head_dim):
      raise ValueError(
          "Expected number of head dimensions in K and V to be"
          f" {expected_kv_head_dim} (matching Q shape {q.shape[2]} with"
          f" packing for {self.serve.dtype_kv}), but got {k.shape[2]=} and"
          f" {v.shape[2]=}"
      )

    if layout.is_seq_along_lane and self.serve.page_size % 128 != 0:
      raise ValueError(
          "Expected page_size to be a multiple of 128 for SEQ_ALONG_LANE"
          f" tile alignment, but got {self.serve.page_size=}"
      )
    expected_kv_cache_shape = layout.kv_cache_shape(kv_cache.shape[0])

    if kv_cache.shape != expected_kv_cache_shape:
      raise ValueError(
          f"Expected {kv_cache.shape=} to be equal to"
          f" {expected_kv_cache_shape=}"
      )

    # If we are using per-token quantization, we expect the kv-cache scale to be
    # in VMEM, and so it will not be in the configs.
    if self.serve.per_token_scale:
      if not layout.is_seq_along_lane:
        raise ValueError(
            "Expected kv_layout=KVLayout.SEQ_ALONG_LANE when per_token_scale is"
            f" True, but got {self.serve.kv_layout=}."
        )
      if self.serve.scale_k is not None or self.serve.scale_v is not None:
        raise ValueError(
            "Expected scale_k and scale_v to be None when per_token_scale is"
            " True."
        )

    if not (
        jnp.int32
        == kv_lens.dtype
        == page_indices.dtype
        == cu_q_lens.dtype
        == distribution.dtype
    ):
      raise ValueError(
          f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
          f" {cu_q_lens.dtype=}, {distribution.dtype=}"
      )

    if not (kv_lens.ndim == page_indices.ndim == cu_q_lens.ndim == 1):
      raise ValueError(
          f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
          f" {cu_q_lens.shape=}"
      )

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    if num_page_indices % max_num_seqs != 0:
      raise ValueError(
          f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
      )
    if cu_q_lens.shape != (max_num_seqs + 1,):
      raise ValueError(
          f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)."
      )
    if distribution.shape != (3,):
      raise ValueError(f"Expected {distribution.shape=} to be (3,).")
    if attention_sink is not None:
      if attention_sink.shape != (self.model.num_q_heads,):
        raise ValueError(
            f"Expected {attention_sink.shape=} to be"
            f" ({self.model.num_q_heads},)."
        )
      if attention_sink.dtype != jnp.float32:
        raise ValueError(
            f"Expected {attention_sink.dtype=} to be equal to {jnp.float32=}."
        )

    # Context Parallel Support
    if self.serve.cp_group_size is not None:
      if attention_sink is not None:
        raise ValueError("Context Parallel does not support attention sink")
      if self.serve.attention_scope == AttentionScope.FULL:
        raise ValueError(
            "Context Parallel does not support AttentionScope.FULL"
            " where cache is sharded but current tokens is sequential"
        )
      if self.model.sliding_window is not None:
        raise ValueError(
            "Context Parallel does not support sliding window right now"
        )

    bkv_sz = self.block.bkv_sz
    page_size = self.serve.page_size
    if bkv_sz % page_size != 0:
      raise NotImplementedError(
          f"BatchedRPA expects {bkv_sz=} to be aligned to {page_size=}."
      )
