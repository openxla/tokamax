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
"""Standalone Pallas Mosaic TPU ragged causal 1D convolution kernel."""

# Dimensions.
#   B         unpadded packed token count (batch_size)
#   B_pad     tile-aligned padded token count (cfgs.batch_size)
#   tb        token tile extent per Pallas step (cfgs.tile_size)
#   nb        active token tiles in the batch: nb = cdiv(num_tokens, tb)
#   D         channel dimension (cfgs.dim_size)
#   K         depthwise convolution kernel width (cfgs.kernel_size)
#   K_prev    state history length (cfgs.prev_kernel_size = K - 1)
#   M         maximum sequences in ragged metadata (max_seqs)
#   M_plus_1  cumulative token offset length (M + 1)
#   S         total state-cache slots in HBM (num_slots)

import abc
import dataclasses
import functools
from typing import Annotated, Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.ragged_causal_conv1d import strided_ldst

_DEFAULT_CPU_VMEM_CAPACITY_BYTES = 16 * 1024 * 1024
_DEFAULT_NUM_LANES = 128
_SUPPORTED_ACTIVATIONS = (None, "silu", "swish")

# Allow uppercase glossary dimension names (`B`, `B_pad`, `D`, `K`, `K_prev`,
# `M`, `M_plus_1`, `S`) per tokamax/docs/code_style.md §5–§6.
# pylint: disable=invalid-name


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ConvConfigs:
  """Static convolution configuration for the Pallas pipeline."""

  batch_size: pydantic.PositiveInt
  dim_size: pydantic.PositiveInt
  kernel_size: Annotated[int, pydantic.Field(ge=2)]
  tile_size: pydantic.PositiveInt = 64

  def __post_init__(self) -> None:
    if self.batch_size % self.tile_size != 0:
      raise ValueError(
          f"batch_size ({self.batch_size}) must be divisible by "
          f"tile_size ({self.tile_size})."
      )

  @property
  def prev_kernel_size(self) -> int:
    return self.kernel_size - 1


def _estimate_conv1d_vmem_bytes(
    *,
    tile_size: int,
    dim: int,
    kernel_size: int,
    x_itemsize: int,
    has_bias: bool,
    num_lanes: int = _DEFAULT_NUM_LANES,
) -> int:
  """Estimates peak VMEM footprint (scratches + VMEM weights + live tiles)."""
  aligned_dim = pl.cdiv(dim, num_lanes) * num_lanes
  K_prev = kernel_size - 1
  x_scratch = 2 * tile_size * aligned_dim * x_itemsize
  conv_state_scratch = 2 * tile_size * K_prev * aligned_dim * 4
  prev_x_scratch = K_prev * aligned_dim * 4
  rhs_vmem = (kernel_size + (1 if has_bias else 0)) * aligned_dim * 4
  live_vregs = (2 * tile_size + K_prev) * aligned_dim * 4
  return x_scratch + conv_state_scratch + prev_x_scratch + rhs_vmem + live_vregs


def _select_conv1d_tile_size_and_vmem_limit(
    *,
    batch_size: int,
    dim: int,
    kernel_size: int,
    requested_tile_size: int,
    sublane_tiling: int,
    x_itemsize: int,
    has_bias: bool,
    vmem_capacity_bytes: int,
    num_lanes: int = _DEFAULT_NUM_LANES,
) -> tuple[int, int]:
  """Selects a sublane-aligned `tile_size` and 90% `vmem_limit_bytes`."""
  if requested_tile_size <= 0:
    raise ValueError(f"Expected tile_size > 0, got {requested_tile_size}.")
  if requested_tile_size % sublane_tiling != 0:
    raise ValueError(
        f"Expected tile_size ({requested_tile_size}) to be divisible by "
        f"sublane_tiling ({sublane_tiling})."
    )
  vmem_limit_bytes = int(0.90 * vmem_capacity_bytes)
  vmem_budget_bytes = int(0.80 * vmem_capacity_bytes)
  padded_tokens = max(
      pl.cdiv(batch_size, sublane_tiling) * sublane_tiling,
      sublane_tiling,
  )
  tb = min(requested_tile_size, padded_tokens)
  tb = max(sublane_tiling, (tb // sublane_tiling) * sublane_tiling)
  while (
      tb > sublane_tiling
      and _estimate_conv1d_vmem_bytes(
          tile_size=tb,
          dim=dim,
          kernel_size=kernel_size,
          x_itemsize=x_itemsize,
          has_bias=has_bias,
          num_lanes=num_lanes,
      )
      > vmem_budget_bytes
  ):
    tb = max(
        sublane_tiling,
        ((tb // 2) // sublane_tiling) * sublane_tiling,
    )
  return tb, vmem_limit_bytes


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
  """SMEM-staged per-token and per-sequence ragged convolution metadata."""

  num_tiles: Any
  b_idx_to_s_idx: Any
  b_idx_to_sz_from_old: Any
  b_idx_should_write: Any
  s_idx_to_state_idx: Any
  s_idx_has_initial_state: Any

  def __len__(self) -> int:
    return len(dataclasses.fields(self))


@jaxtyping.jaxtyped
def preprocess_conv1d_metadata(
    cfgs: ConvConfigs,
    query_start_loc: Int[Array, "M_plus_1"],
    state_indices: Int[Array, "M"],
    has_initial_state: Bool[Array, "M"],
    num_seqs: int | Int[Array, ""],
    num_slots: int | None = None,
) -> MetadataRef:
  """Preprocesses ragged token-to-sequence and safe state-slot DMA metadata.

  Out-of-bounds `state_indices` (`< 0`, such as `PAD_SLOT_ID = -1`, or
  `>= num_slots` when `num_slots` is provided) and inactive sequences
  `seq_idx >= num_seqs` are clamped to slot `0` with `has_initial_state=False`
  and `b_idx_should_write=0` so they never issue out-of-bounds DMAs or
  overwrite active cache slot `0`. Dynamic `num_seqs` and `query_start_loc` are
  also clamped to `[0, M]` and `[0, B_pad]` for bounds safety.

  Args:
    cfgs: Static `ConvConfigs` describing padded batch, dim, kernel, and tile.
    query_start_loc: Cumulative token start offsets `(M + 1,)`.
    state_indices: Per-sequence state-cache slot indices `(M,)`.
    has_initial_state: Per-sequence boolean initial-state flags `(M,)`.
    num_seqs: Number of active sequences `0 <= num_seqs <= M`.
    num_slots: Optional upper bound on valid state-cache slots (`0 <= slot <
      num_slots`).

  Returns:
    Populated `MetadataRef` PyTree for SMEM staging.
  """
  if state_indices.ndim != 1:
    raise ValueError(
        f"Expected positive 1D state_indices, got shape {state_indices.shape}."
    )
  (M,) = state_indices.shape
  if M < 1:
    raise ValueError(
        f"Expected positive 1D state_indices, got shape {state_indices.shape}."
    )
  if query_start_loc.shape != (M + 1,):
    raise ValueError(
        f"Expected query_start_loc shape ({M + 1},), got"
        f" {query_start_loc.shape}."
    )
  if has_initial_state.shape != (M,):
    raise ValueError(
        f"Expected has_initial_state shape ({M},), got"
        f" {has_initial_state.shape}."
    )
  if num_slots is not None and num_slots <= 0:
    raise ValueError(f"Expected num_slots > 0, got {num_slots}.")

  B_pad = cfgs.batch_size
  tb = cfgs.tile_size
  K = cfgs.kernel_size
  K_prev = cfgs.prev_kernel_size

  # Clamp `num_seqs` and `query_start_loc` into valid HBM bounds so compiled
  # kernels with `disable_bounds_checks=True` cannot index or DMA out of bounds.
  safe_num_seqs = jnp.clip(jnp.asarray(num_seqs, dtype=jnp.int32), 0, M)
  clamped_qsl = jnp.clip(
      jnp.asarray(query_start_loc, dtype=jnp.int32), 0, B_pad
  )
  clamped_qsl = clamped_qsl.at[0].set(0)
  clamped_qsl = jnp.maximum.accumulate(clamped_qsl)
  num_tokens = clamped_qsl[safe_num_seqs]
  all_seqs = jnp.arange(M + 1, dtype=jnp.int32)
  clamped_qsl = jnp.where(all_seqs <= safe_num_seqs, clamped_qsl, num_tokens)

  # Clamp inactive sequences (`seqs >= safe_num_seqs`) and invalid slot indices
  # (`state_indices < 0` or `>= num_slots`) to `0`, suppressing both state reads
  # and state writes for them.
  query_lens = jnp.maximum(clamped_qsl[1:] - clamped_qsl[:-1], 0)
  seqs = jnp.arange(M, dtype=jnp.int32)
  state_indices_i32 = jnp.asarray(state_indices, dtype=jnp.int32)
  valid_state = state_indices_i32 >= 0
  if num_slots is not None:
    valid_state = valid_state & (state_indices_i32 < num_slots)
  valid_seq = (seqs < safe_num_seqs) & valid_state
  safe_state_indices = jnp.where(valid_seq, state_indices_i32, 0)
  safe_has_initial_state = (
      jnp.asarray(has_initial_state, dtype=bool) & valid_seq
  )

  b_idx_to_s_idx = jnp.repeat(seqs, query_lens, total_repeat_length=B_pad)
  b_idx_query_start_loc = clamped_qsl[b_idx_to_s_idx]
  all_b_idx = jnp.arange(B_pad, dtype=jnp.int32)
  valid_b = all_b_idx < num_tokens
  b_idx_query_len = 1 + all_b_idx - b_idx_query_start_loc

  b_idx_to_sz_from_new = jnp.minimum(b_idx_query_len, K)
  b_idx_to_sz_from_old = K - b_idx_to_sz_from_new
  b_idx_to_sz_from_old = jnp.minimum(b_idx_to_sz_from_old, K_prev)
  b_idx_to_sz_from_old = jnp.where(valid_b, b_idx_to_sz_from_old, 0)

  b_idx_should_write = (
      valid_b
      & valid_seq[b_idx_to_s_idx]
      & (all_b_idx == (clamped_qsl[b_idx_to_s_idx + 1] - 1))
  ).astype(jnp.int32)

  return MetadataRef(
      num_tiles=pl.cdiv(num_tokens, tb),
      b_idx_to_s_idx=b_idx_to_s_idx,
      b_idx_to_sz_from_old=b_idx_to_sz_from_old,
      b_idx_should_write=b_idx_should_write,
      s_idx_to_state_idx=safe_state_indices,
      s_idx_has_initial_state=safe_has_initial_state,
  )


def _validate_and_normalize_conv1d_args(
    x: jax.Array,
    conv_state: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None,
    query_start_loc: jax.Array | None,
    state_indices: jax.Array | None,
    has_initial_state: jax.Array | None,
    distribution: jax.Array | None,
    *,
    activation: str | None,
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
  """Validates `ragged_causal_conv1d` inputs and normalizes ragged metadata."""
  if x.ndim != 2:
    raise ValueError(
        f"Expected positive 2D x (batch_size, dim), got {x.shape}."
    )
  B, D = x.shape
  if B <= 0 or D <= 0:
    raise ValueError(
        f"Expected positive 2D x (batch_size, dim), got {x.shape}."
    )
  if activation not in _SUPPORTED_ACTIVATIONS:
    raise ValueError(
        f"Expected activation in {_SUPPORTED_ACTIVATIONS}, got {activation!r}."
    )
  if weight.ndim != 3:
    raise ValueError(
        f"Expected weight shape ({D}, 1, kernel_size), got {weight.shape}."
    )
  w_dim, w_in_ch, K = weight.shape
  if (w_dim, w_in_ch) != (D, 1):
    raise ValueError(
        f"Expected weight shape ({D}, 1, kernel_size), got {weight.shape}."
    )
  if K < 2:
    raise ValueError(f"Expected kernel_size >= 2, got {K}.")
  K_prev = K - 1

  valid_conv_state_shapes = ((K_prev, D), (K_prev, 1, D))
  if conv_state.ndim not in (3, 4):
    raise ValueError(
        "Expected conv_state shape "
        f"(num_slots, {K_prev}, {D}) or "
        f"(num_slots, {K_prev}, 1, {D}) with num_slots > 0, "
        f"got {conv_state.shape}."
    )
  S, *state_tail = conv_state.shape
  if S <= 0 or tuple(state_tail) not in valid_conv_state_shapes:
    raise ValueError(
        "Expected conv_state shape "
        f"(num_slots, {K_prev}, {D}) or "
        f"(num_slots, {K_prev}, 1, {D}) with num_slots > 0, "
        f"got {conv_state.shape}."
    )
  if bias is not None and bias.shape != (D,):
    raise ValueError(f"Expected bias shape ({D},), got {bias.shape}.")

  if query_start_loc is None:
    if state_indices is None or state_indices.shape == (1,):
      query_start_loc = jnp.array([0, B], dtype=jnp.int32)
    elif state_indices.ndim == 1 and state_indices.shape == (B,):
      query_start_loc = jnp.arange(B + 1, dtype=jnp.int32)
    else:
      raise ValueError(
          "When query_start_loc is None, state_indices must be None, length 1,"
          f" or length batch_size ({B}), got {state_indices.shape}."
      )

  if query_start_loc.ndim != 1:
    raise ValueError(
        "Expected 1D query_start_loc of length >= 2, got"
        f" {query_start_loc.shape}."
    )
  (M_plus_1,) = query_start_loc.shape
  if M_plus_1 < 2:
    raise ValueError(
        "Expected 1D query_start_loc of length >= 2, got"
        f" {query_start_loc.shape}."
    )
  M = M_plus_1 - 1

  if state_indices is None:
    if M > S:
      raise ValueError(
          f"Default state_indices requires num_seqs ({M}) <= num_slots ({S})."
      )
    state_indices = jnp.arange(M, dtype=jnp.int32)
  elif state_indices.shape != (M,):
    raise ValueError(
        f"Expected state_indices shape ({M},), got {state_indices.shape}."
    )

  if has_initial_state is None:
    has_initial_state = jnp.zeros((M,), dtype=bool)
  elif has_initial_state.shape != (M,):
    raise ValueError(
        f"Expected has_initial_state shape ({M},), got"
        f" {has_initial_state.shape}."
    )

  if distribution is None:
    distribution = jnp.array([0, M, M], dtype=jnp.int32)
  elif distribution.shape != (3,):
    raise ValueError(
        f"Expected distribution shape (3,), got {distribution.shape}."
    )

  return (
      B,
      D,
      K,
      K_prev,
      S,
      query_start_loc,
      state_indices,
      has_initial_state,
      distribution,
  )


def _update_conv_state_slot(
    slot_state: jax.Array,
    step_inputs: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, None]:
  """Updates `slot_state[slot]` when `should_write > 0`."""
  slot, should_write, new_window = step_inputs
  updated = jnp.where(should_write > 0, new_window, slot_state[slot])
  return slot_state.at[slot].set(updated), None


@jaxtyping.jaxtyped
def ragged_causal_conv1d_jax(
    x: Float[Array, "B D"],
    conv_state: Float[Array, "S K_prev D"] | Float[Array, "S K_prev 1 D"],
    weight: Float[Array, "D 1 K"],
    bias: Float[Array, "D"] | None = None,
    query_start_loc: Int[Array, "M_plus_1"] | None = None,
    state_indices: Int[Array, "M"] | None = None,
    has_initial_state: Bool[Array, "M"] | None = None,
    distribution: Int[Array, "3"] | None = None,
    *,
    activation: str | None = None,
) -> tuple[
    Float[Array, "B D"],
    Float[Array, "S K_prev D"] | Float[Array, "S K_prev 1 D"],
]:
  """Plain-XLA reference implementation of ragged depthwise causal 1D conv.

  Pins `precision=jax.lax.Precision.HIGHEST` on the depthwise contraction so
  TPU XLA does not round `float32` reference accumulations to `bfloat16`.
  """
  work_dtype = jnp.float64 if jnp.dtype(x.dtype) == jnp.float64 else jnp.float32
  (
      B,
      D,
      K,
      K_prev,
      S,
      query_start_loc_norm,
      state_indices_norm,
      has_initial_state_norm,
      distribution_norm,
  ) = _validate_and_normalize_conv1d_args(
      x,
      conv_state,
      weight,
      bias,
      query_start_loc,
      state_indices,
      has_initial_state,
      distribution,
      activation=activation,
  )
  cfgs = ConvConfigs(batch_size=B, dim_size=D, kernel_size=K, tile_size=1)
  meta = preprocess_conv1d_metadata(
      cfgs=cfgs,
      query_start_loc=query_start_loc_norm,
      state_indices=state_indices_norm,
      has_initial_state=has_initial_state_norm,
      num_seqs=distribution_norm[-1],
      num_slots=S,
  )

  orig_state_shape = conv_state.shape
  state_3d = (
      conv_state[:, :, 0, :] if conv_state.ndim == 4 else conv_state
  ).astype(work_dtype)
  x_work = x.astype(work_dtype)
  w_work = weight[:, 0, :].astype(work_dtype)
  b_work = bias.astype(work_dtype) if bias is not None else None

  padded_x = jnp.pad(x_work, ((K_prev, 0), (0, 0)))
  b_indices = jnp.arange(B, dtype=jnp.int32)
  s_idx = meta.b_idx_to_s_idx[b_indices]
  slot = meta.s_idx_to_state_idx[s_idx]
  has_init = meta.s_idx_has_initial_state[s_idx]
  sz_from_old = meta.b_idx_to_sz_from_old[b_indices]
  start_from_old = jnp.where(sz_from_old > 0, K_prev - sz_from_old, 0)

  taps = []
  for j in range(K):
    x_tap = padded_x[b_indices + j]
    if j < K_prev:
      old_idx = jnp.minimum(start_from_old + j, K_prev - 1)
      state_val = jnp.where(has_init[:, None], state_3d[slot, old_idx], 0.0)
      tap = jnp.where((j < sz_from_old)[:, None], state_val, x_tap)
    else:
      tap = x_tap
    taps.append(tap)
  windows = jnp.stack(taps, axis=1)  # [B, K, D]
  out_rows = jnp.einsum(
      "bkd,dk->bd",
      windows,
      w_work,
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=work_dtype,
  )
  if b_work is not None:
    out_rows = out_rows + b_work[None, :]
  if activation in ("silu", "swish"):
    out_rows = jax.nn.silu(out_rows)

  should_write = meta.b_idx_should_write[b_indices]
  final_state_3d, _ = jax.lax.scan(
      _update_conv_state_slot,
      state_3d,
      (slot, should_write, windows[:, 1:, :]),
  )
  final_state = final_state_3d.astype(conv_state.dtype).reshape(
      orig_state_shape
  )
  return out_rows.astype(x.dtype), final_state


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConvRhsRef:
  """VMEM-staged depthwise convolution weight and optional bias."""

  weight: Any
  bias: Any | None = None


@dataclasses.dataclass(frozen=True)
class BufferWrapper(abc.ABC):
  """Base double-buffered HBM <-> VMEM async DMA wrapper."""

  hbm_ref: Any
  vmem_ref: Any
  metadata_ref: MetadataRef

  @property
  def tile_size(self) -> int:
    _, tb, *_ = self.vmem_ref.shape
    return tb

  def get_slot_vmem(self, slot):
    return self.vmem_ref.at[slot]

  def is_lower_oob(self, row) -> jax.Array:
    return row < 0

  def is_upper_oob(self, row) -> jax.Array:
    nb = self.metadata_ref.num_tiles[...]
    last_row = nb * self.tile_size
    return row >= last_row

  @abc.abstractmethod
  def copy_in(self, b_start, slot, sem):
    """Starts async HBM -> VMEM DMA read for the tile starting at `b_start`."""

  @abc.abstractmethod
  def wait_in(self, b_start, slot, sem):
    """Waits for HBM -> VMEM DMA read completion for `b_start`."""

  @abc.abstractmethod
  def copy_out(self, b_start, slot, sem):
    """Starts async VMEM -> HBM DMA write for the tile starting at `b_start`."""

  @abc.abstractmethod
  def wait_out(self, b_start, slot, sem):
    """Waits for VMEM -> HBM DMA write completion for `b_start`."""


@dataclasses.dataclass(frozen=True)
class XBuffer(BufferWrapper):
  """Double-buffered HBM <-> VMEM DMA manager for activations `x`."""

  def copy_in(self, b_start, slot, sem):
    tb = self.tile_size
    is_no_op = self.is_upper_oob(b_start)
    dma_size = jnp.where(is_no_op, 0, tb)
    b_start = jnp.where(is_no_op, 0, b_start)

    pltpu.make_async_copy(
        self.hbm_ref.at[pl.ds(b_start, dma_size)],
        self.get_slot_vmem(slot).at[pl.ds(0, dma_size)],
        sem,
    ).start()

  def wait_in(self, b_start, slot, sem):
    del b_start, slot
    # Mosaic TPU DMA semaphores count transferred bytes rather than source
    # addresses, so waiting on slot 0 of the same tile shape waits for the
    # in-flight `tb`-row read to complete without recomputing HBM offsets.
    pltpu.make_async_copy(
        self.vmem_ref.at[0],
        self.vmem_ref.at[0],
        sem,
    ).wait()

  def copy_out(self, b_start, slot, sem):
    tb = self.tile_size
    pltpu.make_async_copy(
        self.get_slot_vmem(slot),
        self.hbm_ref.at[pl.ds(b_start, tb)],
        sem,
    ).start()

  def wait_out(self, b_start, slot, sem):
    del slot
    tb = self.tile_size
    is_no_op = self.is_lower_oob(b_start)
    dma_size = jnp.where(is_no_op, 0, tb)
    pltpu.make_async_copy(
        self.vmem_ref.at[0, pl.ds(0, dma_size)],
        self.vmem_ref.at[0, pl.ds(0, dma_size)],
        sem,
    ).wait()


@dataclasses.dataclass(frozen=True)
class ConvStateBuffer(BufferWrapper):
  """Double-buffered HBM <-> VMEM DMA manager for `conv_state` slots."""

  def copy_in(self, b_start, slot, sem):
    _, tb, K_prev, _, _ = self.vmem_ref.shape
    is_no_op = self.is_upper_oob(b_start)
    b_start = jnp.where(is_no_op, 0, b_start)

    for i in range(tb):
      b_idx = b_start + i
      s_idx = self.metadata_ref.b_idx_to_s_idx[b_idx]
      state_idx = self.metadata_ref.s_idx_to_state_idx[s_idx]
      has_init = self.metadata_ref.s_idx_has_initial_state[s_idx]
      sz_from_old = self.metadata_ref.b_idx_to_sz_from_old[b_idx]
      sz_from_old = jnp.where(is_no_op | (~has_init), 0, sz_from_old)
      # A row that needs no old state still issues a zero-length copy; clamp
      # `start_from_old` inside `[0, K_prev - 1]` so bounds checks never see
      # `start == K_prev`.
      start_from_old = jnp.where(sz_from_old > 0, K_prev - sz_from_old, 0)

      pltpu.make_async_copy(
          self.hbm_ref.at[state_idx, pl.ds(start_from_old, sz_from_old)],
          self.get_slot_vmem(slot).at[i, pl.ds(0, sz_from_old)],
          sem,
      ).start()

  def wait_in(self, b_start, slot, sem):
    del slot
    tb = self.tile_size
    all_sz_from_old = 0
    for i in range(tb):
      b_idx = b_start + i
      s_idx = self.metadata_ref.b_idx_to_s_idx[b_idx]
      has_init = self.metadata_ref.s_idx_has_initial_state[s_idx]
      sz_from_old = self.metadata_ref.b_idx_to_sz_from_old[b_idx]
      all_sz_from_old += jnp.where(has_init, sz_from_old, 0)

    pltpu.make_async_copy(
        self.vmem_ref.at[0, 0, pl.ds(0, all_sz_from_old)],
        self.vmem_ref.at[0, 0, pl.ds(0, all_sz_from_old)],
        sem,
    ).wait()

  def copy_out(self, b_start, slot, sem):
    tb = self.tile_size
    for i in range(tb):
      b_idx = b_start + i
      s_idx = self.metadata_ref.b_idx_to_s_idx[b_idx]
      state_idx = self.metadata_ref.s_idx_to_state_idx[s_idx]
      should_write = self.metadata_ref.b_idx_should_write[b_idx]

      pltpu.make_async_copy(
          self.get_slot_vmem(slot).at[pl.ds(i, should_write)],
          self.hbm_ref.at[pl.ds(state_idx, should_write)],
          sem,
      ).start()

  def wait_out(self, b_start, slot, sem):
    del slot
    tb = self.tile_size
    is_no_op = self.is_lower_oob(b_start)
    b_start = jnp.where(is_no_op, 0, b_start)

    all_writes = 0
    for i in range(tb):
      b_idx = b_start + i
      all_writes += self.metadata_ref.b_idx_should_write[b_idx]

    all_writes = jnp.where(is_no_op, 0, all_writes)
    pltpu.make_async_copy(
        self.vmem_ref.at[0, pl.ds(0, all_writes)],
        self.vmem_ref.at[0, pl.ds(0, all_writes)],
        sem,
    ).wait()


def _inner_kernel(
    b_i: jax.Array,
    *,
    # Inputs.
    metadata_smem_ref,  # MetadataRef of [1]/[B_pad]/[M] i32/bool in SMEM
    conv_rhs_vmem_ref,  # ConvRhsRef([K, 1, D], [D] | None) f32 in VMEM
    # Outputs / double-buffered HBM-VMEM DMA managers.
    x_buffer: XBuffer,
    conv_state_buffer: ConvStateBuffer,
    # Scratch.
    prev_x_scratch_ref,  # [K_prev, 1, D] f32, inter-tile history in VMEM
    dma_sem,  # [2] DMA semaphores (0: recv, 1: send)
    activation: str | None = None,
    interpret: bool = False,
):
  """Executes one double-buffered tile of DMA + depthwise causal conv1d."""
  # 1. Grid indices and slot selection.
  slot = b_i % 2
  other_slot = (slot + 1) % 2
  x_slot_ref = x_buffer.get_slot_vmem(slot)
  conv_state_slot_ref = conv_state_buffer.get_slot_vmem(slot)

  # 2. Shape constants derived from Refs (never re-read from config).
  tb, D = x_slot_ref.shape
  K, _, _ = conv_rhs_vmem_ref.weight.shape
  K_prev, _, _ = prev_x_scratch_ref.shape

  b_start = b_i * tb
  prev_b_start = b_start - tb
  next_b_start = b_start + tb

  # 3. Semaphore aliases.
  recv_sem = dma_sem.at[0]
  send_sem = dma_sem.at[1]

  # 4. Loads and async DMA prefetch (overlaps SALU DMA setup with VPU compute).
  x_buffer.wait_in(b_start, slot, recv_sem)
  conv_state_buffer.wait_in(b_start, slot, recv_sem)

  x_buffer.wait_out(prev_b_start, other_slot, send_sem)
  conv_state_buffer.wait_out(prev_b_start, other_slot, send_sem)

  x_buffer.copy_in(next_b_start, other_slot, recv_sem)
  conv_state_buffer.copy_in(next_b_start, other_slot, recv_sem)

  # 5. Compute: unpack `(tb, D)` into compact `(tb, 1, D)` VREGs so each token
  # row occupies sublane 0 of its own vector register, enabling sliding-window
  # indexing `x_compact[i + j]` across `K` taps without cross-sublane permutes.
  x_compact = strided_ldst.load_large_to_compact(
      x_slot_ref, jnp.float32, interpret=interpret
  )

  prev_x_scratch = prev_x_scratch_ref[...]
  x_compact = jnp.concat([prev_x_scratch, x_compact], axis=0)

  out_list = []
  for i in range(tb):
    b_idx = b_start + i

    s_idx = metadata_smem_ref.b_idx_to_s_idx[b_idx]
    sz_from_old = metadata_smem_ref.b_idx_to_sz_from_old[b_idx]
    has_initial_state = metadata_smem_ref.s_idx_has_initial_state[s_idx]

    out = jnp.zeros((1, D), jnp.float32)

    for j in range(K):
      in_idx = i + j
      lhs = x_compact[in_idx]

      if j < K_prev:
        conv_state = conv_state_slot_ref[i, j]
        conv_state = jnp.where(has_initial_state, conv_state, 0)
        lhs = jnp.where(j < sz_from_old, conv_state, lhs)

      if j > 0:
        # `conv_state` always caches the pre-activation input window in f32.
        conv_state_slot_ref[i, j - 1] = lhs

      rhs = conv_rhs_vmem_ref.weight[j]
      out += lhs * rhs

    if conv_rhs_vmem_ref.bias is not None:
      bias = conv_rhs_vmem_ref.bias[...].reshape(1, -1)
      out += bias
    out_list.append(out)

  out = jnp.stack(out_list, axis=0)
  # 6. Epilogue: fuse activation in float32 VREGs before sublane-packing to
  # `x_slot_ref.dtype` to avoid an extra bf16 -> f32 -> bf16 round-trip.
  if activation in ("silu", "swish"):
    out = jax.nn.silu(out)

  strided_ldst.store_compact_to_large(x_slot_ref, out, interpret=interpret)
  prev_x_scratch_ref[...] = x_compact[tb:]

  x_buffer.copy_out(b_start, slot, send_sem)
  conv_state_buffer.copy_out(b_start, slot, send_sem)


def _main_kernel(
    # Inputs.
    metadata_smem_ref,  # MetadataRef of [1]/[B_pad]/[M] i32/bool in SMEM
    x_hbm_ref,  # [B_pad, D] x.dtype, full array in HBM (pl.ANY)
    conv_state_hbm_ref,  # [S, K_prev, 1, D] f32, full array in HBM (pl.ANY)
    conv_rhs_vmem_ref,  # ConvRhsRef([K, 1, D], [D] | None) f32, one VMEM block
    # Outputs.
    x_out_hbm_ref,  # [B_pad, D] x.dtype, aliased to x_hbm_ref
    conv_state_out_hbm_ref,  # [S, K_prev, 1, D] f32, aliased to HBM state
    # Scratch.
    x_scratch_ref,  # [2, tb, D] x.dtype, double-buffered VMEM slots
    conv_state_scratch_ref,  # [2, tb, K_prev, 1, D] f32, VMEM slots
    prev_x_scratch_ref,  # [K_prev, 1, D] f32, inter-tile history in VMEM
    dma_sem,  # [2] DMA semaphores (0: recv, 1: send)
    *,
    activation: str | None = None,
    interpret: bool = False,
):
  """Top-level Pallas entry point driving double-buffered tile pipeline."""
  del x_out_hbm_ref, conv_state_out_hbm_ref

  # 1. Grid extent and shape constants derived from Refs.
  nb = metadata_smem_ref.num_tiles[...]
  _, tb, _ = x_scratch_ref.shape

  x_buffer = XBuffer(
      hbm_ref=x_hbm_ref,
      vmem_ref=x_scratch_ref,
      metadata_ref=metadata_smem_ref,
  )
  conv_state_buffer = ConvStateBuffer(
      hbm_ref=conv_state_hbm_ref,
      vmem_ref=conv_state_scratch_ref,
      metadata_ref=metadata_smem_ref,
  )

  # 2. Semaphore aliases.
  recv_sem = dma_sem.at[0]
  send_sem = dma_sem.at[1]

  # 3. Initialization guard & prologue DMA.
  @pl.when(nb > 0)
  def _():
    # VMEM scratch is uninitialized across launches; zero `prev_x_scratch_ref`
    # so inactive tail rows on tile 0 (`num_tokens < K_prev`, where
    # `sz_from_old == 0`) never read residual VMEM NaNs.
    prev_x_scratch_ref[...] = jnp.zeros_like(prev_x_scratch_ref)
    x_buffer.copy_in(0, 0, recv_sem)
    conv_state_buffer.copy_in(0, 0, recv_sem)

  # 4. Main double-buffered tile loop.
  @pl.loop(0, nb)
  def _loop_wrapper(b_i):
    _inner_kernel(
        b_i=b_i,
        metadata_smem_ref=metadata_smem_ref,
        conv_rhs_vmem_ref=conv_rhs_vmem_ref,
        x_buffer=x_buffer,
        conv_state_buffer=conv_state_buffer,
        prev_x_scratch_ref=prev_x_scratch_ref,
        dma_sem=dma_sem,
        activation=activation,
        interpret=interpret,
    )

  # 5. Final store guard (wait for last tile's async HBM write).
  @pl.when(nb > 0)
  def _():
    last_b_start = (nb - 1) * tb
    x_buffer.wait_out(last_b_start, 0, send_sem)
    conv_state_buffer.wait_out(last_b_start, 0, send_sem)


@functools.lru_cache(maxsize=64)
def _build_conv1d_pallas_call(
    cfgs: ConvConfigs,
    x_dtype: jnp.dtype,
    num_slots: int,
    has_bias: bool,
    activation: str | None,
    vmem_limit_bytes: int,
    interpret: bool,
):
  """Builds and caches the `pl.pallas_call` runner for static configs."""
  B_pad = cfgs.batch_size
  tb = cfgs.tile_size
  D = cfgs.dim_size
  K_prev = cfgs.prev_kernel_size
  S = num_slots

  smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
  vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  metadata_spec = MetadataRef(
      num_tiles=smem_spec,
      b_idx_to_s_idx=smem_spec,
      b_idx_to_sz_from_old=smem_spec,
      b_idx_should_write=smem_spec,
      s_idx_to_state_idx=smem_spec,
      s_idx_has_initial_state=smem_spec,
  )
  conv_rhs_spec = ConvRhsRef(
      weight=vmem_spec,
      bias=vmem_spec if has_bias else None,
  )
  out_shape = (
      jax.ShapeDtypeStruct((B_pad, D), x_dtype),
      jax.ShapeDtypeStruct((S, K_prev, 1, D), jnp.float32),
  )
  num_meta_fields = len(dataclasses.fields(MetadataRef))
  return pl.pallas_call(
      functools.partial(
          _main_kernel, activation=activation, interpret=interpret
      ),
      out_shape=out_shape,
      in_specs=(metadata_spec, hbm_spec, hbm_spec, conv_rhs_spec),
      out_specs=(hbm_spec, hbm_spec),
      scratch_shapes=(
          pltpu.VMEM((2, tb, D), x_dtype),
          pltpu.VMEM((2, tb, K_prev, 1, D), jnp.float32),
          pltpu.VMEM((K_prev, 1, D), jnp.float32),
          pltpu.SemaphoreType.DMA((2,)),
      ),
      input_output_aliases={num_meta_fields: 0, num_meta_fields + 1: 1},
      compiler_params=pltpu.CompilerParams(
          disable_bounds_checks=True,
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      interpret=pltpu.InterpretParams() if interpret else False,
      name="ragged_causal_conv1d_kernel",
  )


@jaxtyping.jaxtyped
def ragged_causal_conv1d(
    x: Float[Array, "B D"],
    conv_state: Float[Array, "S K_prev D"] | Float[Array, "S K_prev 1 D"],
    weight: Float[Array, "D 1 K"],
    bias: Float[Array, "D"] | None = None,
    query_start_loc: Int[Array, "M_plus_1"] | None = None,
    state_indices: Int[Array, "M"] | None = None,
    has_initial_state: Bool[Array, "M"] | None = None,
    distribution: Int[Array, "3"] | None = None,
    *,
    activation: str | None = None,
    tile_size: int = 64,
    interpret: bool | None = None,
) -> tuple[
    Float[Array, "B D"],
    Float[Array, "S K_prev D"] | Float[Array, "S K_prev 1 D"],
]:
  """Performs depthwise causal 1D convolution over ragged sequences on TPU.

  Args:
    x: Packed token activations `(B, D)` in `float32` or `bfloat16`.
    conv_state: Per-slot state cache in `float32` holding the preceding `(K -
      1)` pre-activation rows of each sequence, shaped either `(S, K - 1, D)` or
      compact 4D `(S, K - 1, 1, D)`.
    weight: Depthwise convolution weights `(D, 1, K)`.
    bias: Optional convolution bias `(D,)`; omitted when `None`.
    query_start_loc: Optional cumulative token start offsets `(M + 1,)`;
      defaults to `[0, B]` (or `0..B` when `state_indices` has length `B`) when
      `None`.
    state_indices: Optional slot indices into `conv_state` `(M,)`. Out-of-bounds
      indices (`< 0`, such as `PAD_SLOT_ID = -1`, or `>= S`) safely skip state
      reads and writes. Defaults to `0..M - 1` when `None`.
    has_initial_state: Optional boolean mask `(M,)` indicating whether each
      sequence reads prior `conv_state`; defaults to all-`False` when `None`.
    distribution: Optional `(decode_end, prefill_end, active_end)` int32 tuple
      of shape `(3,)`; defaults to `[0, M, M]` when `None`.
    activation: Optional fused activation applied in `float32` before storing to
      VMEM (`None`, `"silu"`, or `"swish"`). `conv_state` always stores the
      pre-activation input window.
    tile_size: Requested token rows per Pallas tile (clamped to fit VMEM).
    interpret: Whether to run Pallas in TPU host interpret mode
      (`pltpu.InterpretParams()`). When `None`, automatically enables interpret
      mode on non-TPU backends and runs the compiled Mosaic TPU kernel on TPU.

  Returns:
    Tuple `(output, new_conv_state)` where `output` has shape `(B, D)` in
    `x.dtype` and `new_conv_state` has the same shape and `float32` dtype as
    `conv_state`.
  """
  if tile_size <= 0:
    raise ValueError(f"Expected tile_size > 0, got {tile_size}.")
  (
      B,
      D,
      K,
      K_prev,
      S,
      query_start_loc_norm,
      state_indices_norm,
      has_initial_state_norm,
      distribution_norm,
  ) = _validate_and_normalize_conv1d_args(
      x,
      conv_state,
      weight,
      bias,
      query_start_loc,
      state_indices,
      has_initial_state,
      distribution,
      activation=activation,
  )

  # Raise `NotImplementedError` for floating-point dtypes that are valid for the
  # operator (and supported by `ragged_causal_conv1d_jax`) but unsupported by
  # the Mosaic TPU kernel (`tokamax/docs/code_style.md` §3).
  x_dtype = jnp.dtype(x.dtype)
  if x_dtype not in (jnp.float32, jnp.bfloat16):
    raise NotImplementedError(
        "Mosaic TPU ragged_causal_conv1d only supports x.dtype in"
        f" (float32, bfloat16), got {x.dtype}."
    )
  if jnp.dtype(conv_state.dtype) != jnp.float32:
    raise NotImplementedError(
        "Mosaic TPU ragged_causal_conv1d requires conv_state.dtype == float32,"
        f" got {conv_state.dtype}."
    )

  resolved_interpret = (
      (jax.default_backend() != "tpu") if interpret is None else interpret
  )
  packing = 4 // x_dtype.itemsize
  if resolved_interpret or jax.default_backend() != "tpu":
    sublane_tiling = 8 * packing
    vmem_capacity_bytes = _DEFAULT_CPU_VMEM_CAPACITY_BYTES
    num_lanes = _DEFAULT_NUM_LANES
  else:
    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    if D % num_lanes != 0:
      raise NotImplementedError(
          f"Mosaic TPU ragged_causal_conv1d requires dim ({D}) to be "
          f"divisible by {num_lanes}."
      )
    # `strided_ldst` bitcasts `(tb, D)` VMEM refs to `uint32` when `packing > 1`
    # (e.g. `bfloat16`), halving axis 0 to `tb // 2`. On TPU generations < 7
    # (such as v6e), `get_sublane_tiling(bfloat16)` returns `num_sublanes` (8),
    # so enforce `num_sublanes * packing` (16) as a floor.
    sublane_tiling = max(
        tpu_info.get_sublane_tiling(x_dtype),
        tpu_info.num_sublanes * packing,
    )
    vmem_capacity_bytes = tpu_info.vmem_capacity_bytes

  tb, vmem_limit_bytes = _select_conv1d_tile_size_and_vmem_limit(
      batch_size=B,
      dim=D,
      kernel_size=K,
      requested_tile_size=tile_size,
      sublane_tiling=sublane_tiling,
      x_itemsize=x_dtype.itemsize,
      has_bias=bias is not None,
      vmem_capacity_bytes=vmem_capacity_bytes,
      num_lanes=num_lanes,
  )
  if K_prev > tb:
    raise NotImplementedError(
        f"Mosaic TPU ragged_causal_conv1d requires kernel_size - 1 ({K_prev}) "
        f"<= tile_size ({tb})."
    )

  B_pad = max(pl.cdiv(B, tb) * tb, tb)
  if B_pad > B:
    x = jnp.pad(x, ((0, B_pad - B), (0, 0)))

  conv_state_shape = conv_state.shape
  if conv_state.ndim == 3:
    conv_state = conv_state.reshape(S, K_prev, 1, D)

  weight_t = weight.swapaxes(0, 2).astype(jnp.float32)
  bias_f32 = bias.astype(jnp.float32) if bias is not None else None

  cfgs = ConvConfigs(
      batch_size=B_pad,
      kernel_size=K,
      tile_size=tb,
      dim_size=D,
  )

  metadata = preprocess_conv1d_metadata(
      cfgs=cfgs,
      query_start_loc=query_start_loc_norm,
      state_indices=state_indices_norm,
      has_initial_state=has_initial_state_norm,
      num_seqs=distribution_norm[-1],
      num_slots=S,
  )
  conv_rhs = ConvRhsRef(weight=weight_t, bias=bias_f32)

  runner = _build_conv1d_pallas_call(
      cfgs=cfgs,
      x_dtype=x_dtype,
      num_slots=S,
      has_bias=bias_f32 is not None,
      activation=activation,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=resolved_interpret,
  )
  out, new_conv_state = runner(metadata, x, conv_state, conv_rhs)

  out = out[:B]
  if len(conv_state_shape) == 3:
    new_conv_state = new_conv_state.reshape(conv_state_shape)

  return out, new_conv_state


causal_conv1d = ragged_causal_conv1d

# pylint: enable=invalid-name
