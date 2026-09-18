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

import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config
from tokamax._src.ops.causal_conv1d_gated_delta_rule import memory_ref
from tokamax._src.ops.causal_conv1d_gated_delta_rule import typed_ldst


def load_strided_heads(
    vmem_ref: Any,
    num_heads: int,
    head_dim: int,
    lane_offset: int = 0,
) -> jax.Array:
  """Use strided LDST to split heads along the last dim and transpose.

  Args:
      vmem_ref: VMEM reference of shape [chunk_size, 1, cols].
      num_heads: Number of heads to split out.
      head_dim: Width of one head, in elements.
      lane_offset: TPU vector lanes the first head starts at, for a region that
        is part of a larger tensor (k and v within the fused qkv).

  Returns:
      [num_heads, chunk_size, head_dim]
  """
  num_lanes = pltpu.get_tpu_info().num_lanes
  lanes_per_col = vmem_ref.shape[-1] // num_lanes
  lanes_per_head = head_dim // num_lanes

  flat_ref = vmem_ref.reshape(-1, num_lanes)  # pyrefly: ignore[missing-attribute]
  head_list = []
  for head in range(num_heads):
    head_lanes = [
        flat_ref[lane_offset + head * lanes_per_head + lane :: lanes_per_col]
        for lane in range(lanes_per_head)
    ]
    head_list.append(jnp.concat(head_lanes, axis=-1))
  return jnp.stack(head_list, axis=0)


def load_compact_heads(
    vmem_ref: Any,
    num_heads: int,
    head_dim: int,
    dim_offset: int = 0,
) -> jax.Array:
  """Use contiguous slices to split heads along the last dim and stack.

  Args:
      vmem_ref: VMEM reference of shape [seq_tile_size, chunk_size, 1, cols].
      num_heads: Number of heads to split out.
      head_dim: Width of one head, in elements.
      dim_offset: Element the first head starts at, for a region that is part of
        a larger tensor (k and v within the fused qkv).

  Returns:
      [seq_tile_size, num_heads, chunk_size, 1, head_dim]
  """
  head_list = []
  for head in range(num_heads):
    start = dim_offset + head * head_dim
    head_list.append(vmem_ref[..., start : start + head_dim])
  return jnp.stack(head_list, axis=1)


def load_compact_to_large(vmem_ref: Any) -> jax.Array:
  """Use strided load to convert compact to large layout without transpose."""

  # NOTE: Only support 32-bits for now.
  assert vmem_ref.dtype.itemsize == 4
  assert vmem_ref.shape[-2] == 1
  col_size = vmem_ref.shape[-1]
  new_shape = vmem_ref.shape[:-2] + (col_size,)
  tpu_info = pltpu.get_tpu_info()
  num_lanes = tpu_info.num_lanes

  vreg_list = []
  vmem_ref = vmem_ref.reshape(-1, col_size)  # pyrefly: ignore[missing-attribute]
  for col_start in range(0, col_size, num_lanes):
    col_end = min(col_start + num_lanes, col_size)
    vreg = vmem_ref[..., col_start:col_end]
    vreg_list.append(vreg)
  return jnp.concat(vreg_list, axis=-1).reshape(new_shape)


def _region_rows_per_block(
    slot_ref: Any, region: config.StateRegion
) -> tuple[int, int]:
  """(typed rows per source block, typed lane count) of a slot tile."""
  out_lanes = slot_ref.shape[-1] // region.lane_split
  block_bytes = (
      math.prod(slot_ref.shape[1:]) * jnp.dtype(slot_ref.dtype).itemsize
  )
  rows_pb = block_bytes // (jnp.dtype(region.view_dtype).itemsize * out_lanes)
  return rows_pb, out_lanes


def load_state_region(
    slot_ref: Any, region: config.StateRegion, shape: tuple[int, ...]
) -> jax.Array:
  """Loads one slot's state from its raw source-layout tile.

  Args:
      slot_ref: One slot's VMEM tile of shape [region.nblocks, region.nrows,
        *source payload dims, source lanes] in the source dtype.
      region: The region's copy-plan (typed view parameters).
      shape: Logical state shape; its element count must equal
        ``region.rows_used`` times the typed view's lane count. The FP32 path
        narrows the ref to the logical minor dimension; BF16 keeps the full
        carrier width and reshapes the loaded array here.

  Returns:
      The logical state of ``shape`` in ``region.view_dtype``.
  """
  parts = [
      typed_ldst.load_typed(
          slot_ref.at[j],
          view_dtype=region.view_dtype,
          lane_split=region.lane_split,
      )
      for j in range(region.nblocks)
  ]
  arr = parts[0] if region.nblocks == 1 else jnp.concat(parts, axis=0)
  arr = arr[: region.rows_used]
  if region.rows_perm is not None:
    # Static row gather from the stored order to the logical order;
    # sublane-dim slices + concat only, no lane crossing.
    arr = jnp.concat([arr[p][None] for p in region.rows_perm], axis=0)
  return arr.reshape(shape)


def store_state_region(
    slot_ref: Any, region: config.StateRegion, values: jax.Array
) -> None:
  """Stores one slot's logical state into its raw source-layout tile.

  Typed rows past ``region.rows_used`` are zeroed so the tile's whole
  region has deterministic bytes when copied out.
  """
  rows_pb, out_lanes = _region_rows_per_block(slot_ref, region)
  arr = values.astype(region.view_dtype)
  arr = arr.reshape(-1, out_lanes)
  if region.rows_perm is not None:
    # Inverse of the load-side gather: logical row i is stored at
    # typed row rows_perm[i].
    inverse = [0] * len(region.rows_perm)
    for logical, stored in enumerate(region.rows_perm):
      inverse[stored] = logical
    arr = jnp.concat([arr[i][None] for i in inverse], axis=0)
  capacity = region.nblocks * rows_pb
  if region.rows_used < capacity:
    arr = jnp.pad(arr, ((0, capacity - region.rows_used), (0, 0)))
  for j in range(region.nblocks):
    typed_ldst.store_typed(
        slot_ref.at[j],
        arr[j * rows_pb : (j + 1) * rows_pb],
        lane_split=region.lane_split,
    )


def _load_conv_state(
    slot_ref: jax.Ref, cfg: config.GDNConfig, idx: int
) -> jax.Array:
  """One slot's conv state, decoded through the plan when there is one."""
  # NOTE: Conv1D mandates fp32 due to its usage of compact layout.
  if cfg.state_plan is None:
    return slot_ref[idx, 0].astype(jnp.float32)
  return load_state_region(
      slot_ref.at[idx, 0],
      cfg.state_plan.conv,
      (cfg.prev_kernel_size, 1, cfg.dim_size),
  ).astype(jnp.float32)


def _load_recurrent_state(
    slot_ref: jax.Ref, cfg: config.GDNConfig, idx: int
) -> jax.Array:
  """One slot's recurrent state, converted to FP32 for compute."""
  if cfg.state_plan is None:
    return slot_ref[idx, 0].astype(jnp.float32)
  return load_state_region(
      slot_ref.at[idx, 0],
      cfg.state_plan.recurrent,
      (cfg.num_v_heads, cfg.kq_head_dim, cfg.v_head_dim),
  ).astype(jnp.float32)


def load_and_select_states(
    metadata_ref: memory_ref.MetadataRef,
    p_id: jax.Array,
    conv_state_slot_ref: Any,
    recurrent_slot_ref: Any,
    carry_conv_scratch_ref: Any,
    carry_recurrent_scratch_ref: Any,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Load correct states from HBM or prior tile, and masks invalid states.

  Reference metadata to select the appropriate prior states. If `is_first_tile`
  is True, it selects states read from HBM. If it is False, it selects
  carry states from previous tile. If `has_initial_state` is False, states are
  zero initialized.

  When `cfg.state_plan` is set, the state refs hold raw source-layout
  tiles (see `memory_ref.ExternalStateBufferedRef`) and are decoded
  through the plan's typed region views instead of read directly.

  Args:
      metadata_ref: Metadata reference containing grid and sequence mappings.
      p_id: Current Pallas program ID.
      conv_state_slot_ref: Convolution state read from HBM of shape
        [seq_tile_size, window_size, prev_kernel_size, 1, dim_size].
      recurrent_slot_ref: Recurrent state read from HBM of shape [seq_tile_size,
        window_size, num_v_heads, kq_head_dim, v_head_dim].
      carry_conv_scratch_ref: Optional inter-tile convolution carry of shape
        [seq_tile_size, prev_kernel_size, 1, dim_size].
      carry_recurrent_scratch_ref: Optional inter-tile recurrent state carry of
        shape [seq_tile_size, num_v_heads, kq_head_dim, v_head_dim].
      cfg: GDN configuration object.

  Returns:
      real_sizes: Valid token count per sequence tile of shape [seq_tile_size].
      prev_conv_state: Selected convolution state of shape [seq_tile_size,
      prev_kernel_size, 1, dim_size] in float32.
      prev_recurrent_state: Selected recurrent state of shape [seq_tile_size,
      num_v_heads, kq_head_dim, v_head_dim].
  """

  real_sizes_list = []
  prev_conv_state_list = []
  prev_recurrent_state_list = []

  for idx in range(cfg.seq_tile_size):
    s_idx = metadata_ref.p_id_to_s_idx[p_id, idx]
    real_sizes = metadata_ref.p_id_to_r_size[p_id, idx]
    is_first_tile = metadata_ref.p_id_is_first_tile[p_id, idx]
    has_initial_state = metadata_ref.s_idx_has_initial_state[s_idx]

    # NOTE: The VMEM window holds one state per window position and the
    # initial state was DMA'd into position 0.
    # Both scratches are allocated together (see GDNConfig.scratch_shapes).
    # Only the pool decode is worth guarding; a dense slot read is cheap
    # enough that the guard costs more than it saves.
    if carry_conv_scratch_ref is None or cfg.state_plan is None:
      prev_conv_state = jnp.where(
          has_initial_state, _load_conv_state(conv_state_slot_ref, cfg, idx), 0
      )
      prev_recurrent_state = jnp.where(
          has_initial_state,
          _load_recurrent_state(recurrent_slot_ref, cfg, idx),
          0,
      )
      if (
          carry_conv_scratch_ref is not None
          and carry_recurrent_scratch_ref is not None
      ):
        prev_conv_state = jnp.where(
            is_first_tile, prev_conv_state, carry_conv_scratch_ref[idx]
        )
        prev_recurrent_state = jnp.where(
            is_first_tile,
            prev_recurrent_state,
            carry_recurrent_scratch_ref[idx],
        )
    else:
      # Later tiles resume from the carry, so decode both sources once
      # instead of on every tile. One guard, not one per state.
      # pl.when traces the body here, so `idx` is this iteration's.
      assert carry_conv_scratch_ref is not None
      assert carry_recurrent_scratch_ref is not None
      @pl.when(is_first_tile)
      def _():
        carry_conv_scratch_ref[idx] = jnp.where(
            has_initial_state,
            _load_conv_state(conv_state_slot_ref, cfg, idx),
            0,
        )
        carry_recurrent_scratch_ref[idx] = jnp.where(
            has_initial_state,
            _load_recurrent_state(recurrent_slot_ref, cfg, idx),
            0,
        )

      prev_conv_state = carry_conv_scratch_ref[idx]
      prev_recurrent_state = carry_recurrent_scratch_ref[idx]

    real_sizes_list.append(real_sizes)
    prev_conv_state_list.append(prev_conv_state)
    prev_recurrent_state_list.append(prev_recurrent_state)

  real_sizes = jnp.stack(real_sizes_list, axis=0)
  prev_conv_state = jnp.stack(prev_conv_state_list, axis=0)
  prev_recurrent_state = jnp.stack(prev_recurrent_state_list, axis=0)

  return real_sizes, prev_conv_state, prev_recurrent_state


def load_activation_as_compact(
    qkv_vreg: jax.Array,
    qkv_vmem_ref: Any,
    b_vmem_ref: Any,
    a_vmem_ref: Any,
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Load activations from VMEM as a compact layout."""

  qkv_vmem_ref[...] = qkv_vreg

  k_offset = cfgs.num_kq_heads * cfgs.kq_head_dim
  v_offset = 2 * k_offset

  q_compact = load_compact_heads(
      qkv_vmem_ref, cfgs.num_kq_heads, cfgs.kq_head_dim
  )
  k_compact = load_compact_heads(
      qkv_vmem_ref, cfgs.num_kq_heads, cfgs.kq_head_dim, k_offset
  )
  v_compact = load_compact_heads(
      qkv_vmem_ref, cfgs.num_v_heads, cfgs.v_head_dim, v_offset
  )
  b_compact = jnp.expand_dims(b_vmem_ref[...], axis=1)
  if cfgs.is_kda:
    # KDA's gate is per-channel, so it splits into heads like q/k/v do
    # instead of occupying a single lane per head.
    a_compact = load_compact_heads(
        a_vmem_ref, cfgs.num_v_heads, cfgs.kq_head_dim
    )
  else:
    a_compact = jnp.expand_dims(a_vmem_ref[...], axis=1)
  return q_compact, k_compact, v_compact, b_compact, a_compact


def load_activation_as_large(
    qkv_vreg: jax.Array,
    qkv_vmem_ref: Any,
    b_vmem_ref: Any,
    a_vmem_ref: Any,
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Load activations from VMEM as a large layout."""

  qkv_vmem_ref[...] = qkv_vreg

  num_lanes = pltpu.get_tpu_info().num_lanes
  kq_lanes = cfgs.num_kq_heads * (cfgs.kq_head_dim // num_lanes)

  q_large_list = []
  k_large_list = []
  v_large_list = []
  a_large_list = []
  for idx in range(cfgs.seq_tile_size):
    qkv_slot = qkv_vmem_ref.at[idx]
    q_large_list.append(
        load_strided_heads(qkv_slot, cfgs.num_kq_heads, cfgs.kq_head_dim)
    )
    k_large_list.append(
        load_strided_heads(
            qkv_slot, cfgs.num_kq_heads, cfgs.kq_head_dim, kq_lanes
        )
    )
    v_large_list.append(
        load_strided_heads(
            qkv_slot, cfgs.num_v_heads, cfgs.v_head_dim, 2 * kq_lanes
        )
    )
    if cfgs.is_kda:
      # Per-channel gate: split into heads and transpose like q/k/v.
      a_large_list.append(
          load_strided_heads(
              a_vmem_ref.at[idx], cfgs.num_v_heads, cfgs.kq_head_dim
          )
      )

  q_large = jnp.stack(q_large_list, axis=0)
  k_large = jnp.stack(k_large_list, axis=0)
  v_large = jnp.stack(v_large_list, axis=0)
  b_large = jnp.expand_dims(load_compact_to_large(b_vmem_ref), axis=1)
  if cfgs.is_kda:
    a_large = jnp.stack(a_large_list, axis=0)
  else:
    a_large = jnp.expand_dims(load_compact_to_large(a_vmem_ref), axis=1)

  return q_large, k_large, v_large, b_large, a_large
