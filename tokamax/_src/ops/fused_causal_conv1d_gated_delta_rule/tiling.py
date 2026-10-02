# Copyright 2026 Rabdos AI
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
"""Static eligibility, tile selection and memory planning for fused GDN."""

import math
from typing import Any

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import compute as fused_compute
from tokamax._src.ops.fused_causal_conv1d_gated_delta_rule import metadata as fused_metadata

MAX_DECODE_GROUP = 8

# Conservative allowances for arithmetic temporaries and scalar bookkeeping.
_ARITHMETIC_VMEM_BASE_RESERVE_BYTES = 20 * 1024 * 1024
_SMEM_HEADROOM_BYTES = 16 * 1024
# Off-device planning uses the existing v6e budget assumptions.
_FALLBACK_VMEM_LIMIT_BYTES = int(0.8 * 128 * 1024 * 1024)
_FALLBACK_SMEM_LIMIT_BYTES = 1024 * 1024


def state_ring_depth(n_kq: int, n_v: int) -> int:
  """Return the resident recurrent-state ring depth for a head layout.

  Args:
    n_kq: Number of query/key heads.
    n_v: Number of value heads; supported layouts group them evenly over Q/K
      heads.

  Returns:
    Three slots for narrow layouts, otherwise two.
  """
  return 3 if n_kq <= 4 and n_v <= 12 else 2


def padded_vmem_bytes(shape: tuple[int, ...], itemsize: int) -> int:
  """Estimate VMEM storage using fixed 16-by-128 planning padding.

  Args:
    shape: Logical array dimensions before VMEM padding.
    itemsize: Bytes per array element.

  Returns:
    Estimated bytes after padding the two innermost dimensions.
  """
  dims = list(shape)
  if dims:
    dims[-1] = (dims[-1] + 127) // 128 * 128
  if len(dims) > 1:
    dims[-2] = (dims[-2] + 15) // 16 * 16
  return math.prod(dims) * itemsize


def arithmetic_vmem_reserve(n_kq: int, n_v: int) -> int:
  """Reserve space for compiler arithmetic intermediates absent from scratch.

  Args:
    n_kq: Number of query/key heads.
    n_v: Number of value heads; supported layouts group them evenly over Q/K
      heads.

  Returns:
    Additional arithmetic working-memory allowance in bytes.
  """
  return _ARITHMETIC_VMEM_BASE_RESERVE_BYTES * max(
      1, (n_kq + 15) // 16, (n_v + 31) // 32
  )


def prefill_vmem_allowance(
    tile_rows: int, width: int, n_v: int, act_itemsize: int
) -> int:
  """Price scoped prefill QKV and the retained conservative gate windows.

  Args:
    tile_rows: Number of token rows in the candidate activation tile.
    width: Packed QKV channel count.
    n_v: Number of value heads; supported layouts group them evenly over Q/K
      heads.
    act_itemsize: Bytes per activation element.

  Returns:
    Estimated bytes for private prefill inputs and conservative gate storage.
  """
  return padded_vmem_bytes(
      (2, tile_rows + 8, width), act_itemsize
  ) + 2 * padded_vmem_bytes((2, 1, tile_rows, 1, n_v), act_itemsize)


def decode_gate_vmem_allowance(
    group_width: int, n_v: int, act_itemsize: int
) -> int:
  """Price both decode gates at the conservative token-by-token layout.

  Args:
    group_width: Number of decode members per group, between one and eight.
    n_v: Number of value heads; supported layouts group them evenly over Q/K
      heads.
    act_itemsize: Bytes per activation element.

  Returns:
    Conservative bytes reserved for both grouped decode gate buffers.
  """
  return 2 * padded_vmem_bytes((2, group_width, 1, n_v), act_itemsize)


def decode_group_width(requests: int, n_v: int, hint: int | None = None) -> int:
  """Choose <= requests decode members: eight for narrow layouts, four
  otherwise.

  Explicit hints still pass static_eligible's group-width/resource checks.

  Args:
    requests: Number of request entries, including inactive padding.
    n_v: Number of value heads; supported layouts group them evenly over Q/K
      heads.
    hint: Optional requested decode group width; unsupported values use the
      default.

  Returns:
    Decode group width bounded by requests; valid hints override the default.
  """
  width = 8 if n_v <= 4 else 4
  if hint is not None and 1 <= hint <= MAX_DECODE_GROUP:
    width = hint
  return min(width, requests)


def tile_rows_supported(tile_rows: int, n_kq: int, n_v: int) -> bool:
  """Require tile sizes divisible by 64 and containing whole recurrent
  subchunks.

  _resources_fit checks the memory budget separately.

  Args:
    tile_rows: Number of token rows in the candidate activation tile.
    n_kq: Number of query/key heads.
    n_v: Number of value heads; supported layouts group them evenly over Q/K
      heads.

  Returns:
    Whether the tile is positive, 64-row aligned, and contains whole subchunks.
  """
  if tile_rows <= 0 or tile_rows % 64:
    return False
  return tile_rows % fused_compute.sub_chunk_rows(tile_rows, n_kq, n_v) == 0


def prefill_tile_rows(width: int, mixed_tile_size: int | None) -> int:
  """Choose larger activation tiles for narrow layouts to amortize setup.

  Width 640 identifies one Q/K and three value heads in the supported
  envelope. static_eligible checks explicit hints.

  Args:
    width: Packed QKV channel count.
    mixed_tile_size: Optional requested prefill token rows per tile.

  Returns:
    Requested tile rows, or the packed-width default when no hint is supplied.
  """
  if mixed_tile_size is not None:
    return mixed_tile_size
  # The 1/3-head layout batches eight 128-row recurrent chunks per tile.
  if width == 640:
    return 1024
  if width <= 768:
    return 512
  if width > 1536:
    return 256 if width <= 3072 else 128
  return 256


# Gate vectors are staged before conversion, so their dtypes must support DMA.
_STAGEABLE = (jnp.bfloat16, jnp.float32, jnp.float16, jnp.int32)


def static_eligible(
    args: tuple[jax.Array | None, ...],
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    compute_precision: jnp.dtype,
    decode_tile_size: int | None,
    mixed_tile_size: int | None,
    zero_initialize_out: bool,
) -> bool:
  """Check one candidate's static layout and resource eligibility.

  args is fused_conv1d_gdn's first 13 positional arguments, in order. Options
  follow that entry point. Runtime schedule checks are separate, in
  build_smem_schedule. False rejects this candidate; select_tiles may try
  others before falling back to the upstream wrapper.

  Args:
    args: The public entry point's first 13 positional inputs, in signature
      order.
    n_kq: Number of query/key heads.
    n_v: Number of value heads; supported layouts group them evenly over Q/K
      heads.
    d_k: Key/query head dimension; the fused path requires 128.
    d_v: Value head dimension; the fused path requires 128.
    kernel_size: Positive static convolution window size.
    compute_precision: Requested accumulation dtype; fused execution requires
      FP32.
    decode_tile_size: Optional requested number of decode members processed per
      group.
    mixed_tile_size: Optional requested prefill token rows per tile.
    zero_initialize_out: Ignored by eligibility; fused execution writes every
      output row.

  Returns:
    Whether the candidate satisfies the static envelope and resource estimate.
  """
  del zero_initialize_out
  (
      qkv,
      b,
      a,
      conv_state,
      recurrent_state,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      query_start_loc,
      state_indices,
      distribution,
      seq_lens,
  ) = args

  # Grouped state projections require 128-wide Q/K and value heads.
  if (
      n_kq <= 0
      or n_v <= 0
      or n_v % n_kq != 0
      or d_k != 128
      or d_v != 128
      or kernel_size <= 0
  ):
    return False
  width_expected = 2 * n_kq * d_k + n_v * d_v
  tile_rows = prefill_tile_rows(width_expected, mixed_tile_size)
  if d_k % 128 or d_v % 128:
    return False
  if (
      jnp.dtype(compute_precision) != jnp.dtype(jnp.float32)
      or not (
          decode_tile_size is None or 1 <= decode_tile_size <= MAX_DECODE_GROUP
      )
      or not tile_rows_supported(tile_rows, n_kq, n_v)
  ):
    return False
  # Check ranks before indexing shapes.
  if qkv.ndim != 2 or qkv.shape[1] != width_expected or qkv.shape[0] <= 0:
    return False
  if state_indices.ndim != 1 or state_indices.size == 0 or conv_state.ndim != 3:
    return False

  tokens, width = qkv.shape
  requests = state_indices.size
  slots = conv_state.shape[0]
  metadata = (query_start_loc, state_indices, distribution, seq_lens)
  valid = (
      slots > 0
      and qkv.dtype == b.dtype == a.dtype
      and qkv.dtype in (jnp.bfloat16, jnp.float32)
      and conv_state.dtype in (jnp.bfloat16, jnp.float32)
      and recurrent_state.dtype in (jnp.bfloat16, jnp.float32)
      and all(x.dtype == jnp.int32 for x in metadata)
      and query_start_loc.shape == (requests + 1,)
      and seq_lens.shape == (requests,)
      and distribution.shape == (3,)
      and b.shape == a.shape == (tokens, n_v)
      and conv_weight.shape == (width, 1, kernel_size)
      and (conv_bias is None or conv_bias.shape == (width,))
      and a_log.shape == dt_bias.shape == (n_v,)
      and a_log.dtype in _STAGEABLE
      and dt_bias.dtype in _STAGEABLE
      and conv_weight.dtype in _STAGEABLE
      and (conv_bias is None or conv_bias.dtype in _STAGEABLE)
      and conv_state.shape == (slots, kernel_size - 1, width)
      and recurrent_state.shape == (slots, n_v, d_k, d_v)
  )
  if not valid:
    return False
  return _resources_fit(
      args,
      n_kq=n_kq,
      n_v=n_v,
      chunk_size=tile_rows,
      decode_tile_size=decode_tile_size,
  )


def _resources_fit(
    args: tuple[jax.Array | None, ...],
    *,
    n_kq: int,
    n_v: int,
    chunk_size: int,
    decode_tile_size: int | None = None,
) -> bool:
  """Check VMEM/SMEM planning budgets, including scoped buffers and headroom.

  These are estimates, not compiler measurements. Query device limits when
  available; off-TPU tracing uses v6e fallback budgets.

  Args:
    args: The public entry point's first 13 positional inputs, in signature
      order.
    n_kq: Number of query/key heads.
    n_v: Number of value heads; supported layouts group them evenly over Q/K
      heads.
    chunk_size: Number of token rows in one prefill tile.
    decode_tile_size: Requested group width, or None for the layout default.

  Returns:
    Whether SMEM and either full or reduced recurrent rings fit estimated
    budgets.
  """
  qkv, _, _, conv, recurrent, weight, bias, a_log, dt_bias, _, indices, _, _ = (
      args
  )
  tokens, width = qkv.shape
  requests, slots = indices.size, conv.shape[0]
  heads = 2 * n_kq + n_v
  group = decode_group_width(requests, n_v, decode_tile_size)
  ring_depth = state_ring_depth(n_kq, n_v)
  dim, output_width = 128, n_v * 128  # Head width, not pipeline row count.
  conv_bytes = jnp.dtype(conv.dtype).itemsize
  # State buffers keep the cache dtype; activation buffers keep the input dtype.
  state_itemsize = jnp.dtype(recurrent.dtype).itemsize
  act_bytes = jnp.dtype(qkv.dtype).itemsize
  kernel_size = weight.shape[-1]
  history_rows = max(1, kernel_size - 1)

  resident_vmem_allocations = [
      ((1, history_rows, 1, width), 4),  # Convolution carry.
      # Resident prefill state, one slot per request parity.
      ((2, n_v, dim, dim), state_itemsize),
      ((2, history_rows, width), conv_bytes),
      ((1, history_rows, 1, width), conv_bytes),
      ((2, history_rows, width), conv_bytes),  # Convolution stores.
      ((max(heads, ring_depth * n_v), max(chunk_size, dim), dim), 4),
      ((n_kq, dim, (n_v // n_kq) * dim), 4),
      ((2, chunk_size + 16, output_width), act_bytes),
      ((16, output_width), act_bytes),
      ((2, ((group + 14) // 8) * 8, width), act_bytes),  # Native decode QKV.
      ((2, group, history_rows, width), conv_bytes),
      # FP32 member zero borrows scratch; BF16 requires its own state buffers.
      (
          (
              ring_depth * (group if state_itemsize != 4 else group - 1),
              n_v,
              dim,
              dim,
          ),
          state_itemsize,
      ),
      ((kernel_size, heads, dim), 4),  # Dense head-major convolution weight.
      ((n_v,), jnp.dtype(a_log.dtype).itemsize),
      ((n_v,), jnp.dtype(dt_bias.dtype).itemsize),
  ]
  # BF16 caches need a separate FP32 carry between tiles.
  if state_itemsize != 4:
    resident_vmem_allocations.append(((1, n_v, dim, dim), 4))
  if bias is not None:
    # Only the dense head-major convolution bias is resident.
    resident_vmem_allocations.extend((((heads, dim), 4),))
  # Include head-scaled arithmetic space alongside explicit buffers.
  vmem_bytes = (
      sum(padded_vmem_bytes(*entry) for entry in resident_vmem_allocations)
      + prefill_vmem_allowance(chunk_size, width, n_v, act_bytes)
      + decode_gate_vmem_allowance(group, n_v, act_bytes)
      + arithmetic_vmem_reserve(n_kq, n_v)
  )
  capacity = fused_metadata.prefill_schedule_capacity(
      requests, tokens, chunk_size, n_kq, n_v
  )
  # SMEM bools occupy 32 bits. Include tile/request metadata and occupancy.
  smem_bytes = (
      4 * (2 * capacity + 10 * requests + slots + 6) + _SMEM_HEADROOM_BYTES
  )
  # Off-TPU tracing uses v6e budgets; these do not qualify other hardware.
  try:
    vmem_limit = config.get_vmem_limit_bytes()
    smem_limit = pltpu.get_tpu_info().smem_capacity_bytes
  except Exception:  # pylint: disable=broad-except
    vmem_limit = _FALLBACK_VMEM_LIMIT_BYTES
    smem_limit = _FALLBACK_SMEM_LIMIT_BYTES
  if vmem_limit is None:
    vmem_limit = _FALLBACK_VMEM_LIMIT_BYTES
  if smem_limit is None:
    smem_limit = _FALLBACK_SMEM_LIMIT_BYTES
  if smem_bytes > smem_limit:
    return False
  if vmem_bytes <= vmem_limit:
    return True

  # Retry with the reduced rings used by decode_state_ring_depths.
  state_bytes = padded_vmem_bytes((n_v, dim, dim), state_itemsize)
  buffered_members = range(group) if state_itemsize != 4 else range(1, group)
  full_state_bytes = ring_depth * len(buffered_members) * state_bytes
  reduced_state_bytes = (
      sum(ring_depth if member < 2 else 1 for member in buffered_members)
      * state_bytes
  )
  return vmem_bytes - full_state_bytes + reduced_state_bytes <= vmem_limit


_TILE_FALLBACKS = (256, 128, 64)


def select_mixed_tile(
    args: tuple[jax.Array | None, ...], hint: int | None, **options
) -> tuple[bool, int | None]:
  """Try the hint, default, then fallback prefill tiles against static
  eligibility.

  args/options follow static_eligible, without mixed_tile_size. Return
  (eligible, tile). tile=None selects the default. eligible=False means no
  prefill tile fits this decode group; select_tiles may retry smaller groups.

  Args:
    args: The public entry point's first 13 positional inputs, in signature
      order.
    hint: Preferred prefill tile size; None tries the default first.
    **options: Arguments for static_eligible, excluding mixed_tile_size.

  Returns:
    (eligible, tile); tile=None denotes the default or an unsuccessful search.
  """
  candidates: list[int | None] = [] if hint is None else [hint]
  candidates.append(None)
  candidates.extend(_TILE_FALLBACKS)
  width = 2 * options["n_kq"] * options["d_k"] + options["n_v"] * options["d_v"]

  seen: set[int] = set()
  for candidate in candidates:
    rows = prefill_tile_rows(width, candidate)
    if rows in seen:
      continue
    seen.add(rows)
    if static_eligible(args, mixed_tile_size=candidate, **options):
      return True, candidate
  return False, None


def select_tiles(
    args: tuple[jax.Array | None, ...], hint: int | None, **options
) -> tuple[bool, int | None, int | None]:
  """Select eligible decode/prefill tiles, shrinking automatic decode groups
  last.

  Return (eligible, decode_tile, prefill_tile). Preserve explicit decode
  hints and every configuration already admitted by select_mixed_tile.

  Args:
    args: The public entry point's first 13 positional inputs, in signature
      order.
    hint: Optional requested prefill tile size; None selects the default.
    **options: Static configuration overrides forwarded to the selected entry
      point.

  Returns:
    (eligible, decode_tile, prefill_tile), using smaller automatic groups only
    when needed.
  """
  decode_hint = options["decode_tile_size"]
  eligible, tile = select_mixed_tile(args, hint, **options)
  if eligible or decode_hint is not None:
    return eligible, decode_hint, tile
  if args[10].ndim != 1:
    return False, None, None

  group = decode_group_width(args[10].size, options["n_v"])
  for smaller in range(group - 1, 0, -1):
    eligible, tile = select_mixed_tile(
        args, hint, **dict(options, decode_tile_size=smaller)
    )
    if eligible:
      return True, smaller, tile
  return False, None, None


def _padded_storage_estimate(tree: Any) -> int:
  """Estimate VMEM bytes with padding on the two innermost dimensions.

  Array dimensions must be static. Skip leaves missing shape/dtype or using
  unsupported dtypes; this is a planning estimate, not compiler evidence.

  Args:
    tree: Pytree of array operands and memory descriptors to estimate.

  Returns:
    Estimated VMEM bytes across eligible leaves, excluding other memory spaces.
  """
  total = 0
  for leaf in jax.tree.leaves(tree):
    # Scratch pytrees also contain SMEM metadata and slot lists. Only VMEM
    # descriptors belong in this budget; ordinary array operands have no
    # memory_space attribute and are staged into VMEM by the caller.
    memory_space = getattr(leaf, "memory_space", None)
    if memory_space is not None and memory_space != pltpu.VMEM:
      continue
    shape = getattr(leaf, "shape", None)
    dtype = getattr(leaf, "dtype", None)
    if shape is None or dtype is None:
      continue
    try:
      itemsize = jnp.dtype(dtype).itemsize
    except (TypeError, ValueError):
      continue
    total += padded_vmem_bytes(tuple(int(d) for d in shape), itemsize)
  return total


def decode_state_ring_depths(
    group_width: int,
    resident: dict[str, Any],
    weights: Any,
    dense_conv: tuple[jax.Array, jax.Array | None],
    limit: int | None,
    member0_buffered: bool = False,
) -> tuple[int, ...]:
  """Choose per-member state-ring depths against an estimated VMEM budget.

  group_width is in [1, 8]. Member zero borrows resident FP32 scratch unless
  member0_buffered; BF16 caches require that flag. resident, weights and
  dense_conv describe existing allocations. limit=None retains full depth;
  otherwise reduce members after the first two to one slot when over budget.
  The reduced estimate is not checked here.

  Args:
    group_width: Number of decode members, between one and eight.
    resident: Existing scratch descriptors keyed by allocator buffer name.
    weights: Staged gate-parameter operands used in the memory estimate.
    dense_conv: Staged FP32 convolution weights and optional bias for memory
      estimation.
    limit: VMEM budget in bytes; None retains full recurrent rings.
    member0_buffered: Whether member zero has its own state storage instead of
      borrowing FP32 scratch.

  Returns:
    One recurrent-ring depth per member; reduced rings are not revalidated here.
  """
  # The head-major scratch can be borrowed only for a float32 cache.
  grouped_state_shape = resident["grouped_state_scratch_ref"].shape
  n_kq = grouped_state_shape[0]
  output_width = resident["output_carry_ref"].shape[-1]
  n_v = output_width // 128
  ring_depth = state_ring_depth(n_kq, n_v)
  state_ring_depths = (ring_depth,) * group_width
  act_itemsize = jnp.dtype(resident["output_scratch_ref"].dtype).itemsize
  # Include scoped prefill buffers and conservative gate allowances.
  tile_rows = (
      resident["output_scratch_ref"].shape[1] - fused_compute.OUTPUT_ALIGNMENT
  )
  scoped_prefill = prefill_vmem_allowance(
      tile_rows, resident["decode_group_qkv_ref"].shape[-1], n_v, act_itemsize
  )
  reserve = arithmetic_vmem_reserve(n_kq, n_v)
  # Apply the same conservative gate allowance to decode staging.
  gate_reserve = (
      decode_gate_vmem_allowance(group_width, n_v, act_itemsize)
      - _padded_storage_estimate(resident["decode_group_b_ref"])
      - _padded_storage_estimate(resident["decode_group_a_ref"])
  )
  estimated_base = (
      _padded_storage_estimate(resident)
      + _padded_storage_estimate(weights)
      + _padded_storage_estimate(dense_conv)
      + scoped_prefill
      + reserve
      + gate_reserve
  )
  # The grouped state scratch is always float32; recurrent rings use the cache dtype.
  prefill_state = resident["prefill_recurrent_scratch_ref"]
  state_bytes = padded_vmem_bytes(
      tuple(prefill_state.shape[1:]), jnp.dtype(prefill_state.dtype).itemsize
  )
  buffered = group_width if member0_buffered else group_width - 1
  extra_ring_bytes = buffered * ring_depth * state_bytes
  if limit is not None and estimated_base + extra_ring_bytes > limit:
    state_ring_depths = tuple(
        ring_depth if member < 2 else 1 for member in range(group_width)
    )
  return state_ring_depths
