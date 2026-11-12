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
"""Pallas Mosaic TPU kernels for Mamba-2 and Mamba-3 SSD scan."""

# Dimensions.
#   C       number of chunks
#   H       number of SSM heads
#   G       number of B/C projection groups (divides H)
#   th      head tile extent per Pallas grid step
#   tg      group tile extent per Pallas grid step (1 when G < H, else th)
#   nh      head grid extent: nh = H // th
#   L       packed chunk length (L = Q * R)
#   Q       tokens per chunk (Q = L // R)
#   R       MIMO channel rank (mimo_rank >= 1; R == 1 for SISO)
#   P       head value dimension (d_head)
#   N       SSM state dimension (d_state)
#   S       number of output state indices (num_out_states)
#   S_init  number of initial state sequences (num_seqs)

from typing import cast

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping

# Fallback TPU VMEM parameters used only when `interpret=True` on a CPU host.
_INTERPRET_DEFAULT_VMEM_BYTES = 16 * 1024 * 1024
_INTERPRET_DEFAULT_SUBLANES = 8
_INTERPRET_DEFAULT_LANES = 128
_F32_BYTES = 4


def _largest_divisor(n: int, max_div: int) -> int:
  """Returns the largest positive divisor of `n` that is at most `max_div`."""
  if n <= 0 or max_div <= 0:
    raise ValueError(f"Expected n > 0 and max_div > 0, got {n=}, {max_div=}.")
  for candidate in range(min(n, max_div), 1, -1):
    if n % candidate == 0:
      return candidate
  return 1


def _resolve_interpret(interpret: bool | None) -> bool:
  """Resolves `None` to host interpret mode on non-TPU backends."""
  return (jax.default_backend() != "tpu") if interpret is None else interpret


def _aligned_vmem_tile_bytes(
    lead_dim: int,
    rows: int,
    cols: int,
    elem_bytes: int,
    num_sublanes: int,
    num_lanes: int,
) -> int:
  """Byte footprint of a 3D VMEM block padded to `(num_sublanes, num_lanes)`."""
  aligned_rows = ((rows + num_sublanes - 1) // num_sublanes) * num_sublanes
  aligned_cols = ((cols + num_lanes - 1) // num_lanes) * num_lanes
  return lead_dim * aligned_rows * aligned_cols * elem_bytes


def _select_head_block_size(
    n_heads: int,
    n_groups: int,
    block_len: int,
    d_head: int,
    d_state: int,
    *,
    include_decay_tile: bool,
    interpret: bool,
) -> int:
  """Selects the largest divisor `th` that fits in TPU VMEM."""
  # When `n_groups < n_heads`, `th` divides `n_heads // n_groups` so each grid
  # step loads a single group slice of `B_c` or `C_c`. When `n_groups ==
  # n_heads`, `th` divides `n_heads`.
  if not interpret:
    tpu_info = pltpu.get_tpu_info()
    vmem_capacity_bytes = int(tpu_info.vmem_capacity_bytes)
    num_sublanes = int(tpu_info.num_sublanes)
    num_lanes = int(tpu_info.num_lanes)
  else:
    vmem_capacity_bytes = _INTERPRET_DEFAULT_VMEM_BYTES
    num_sublanes = _INTERPRET_DEFAULT_SUBLANES
    num_lanes = _INTERPRET_DEFAULT_LANES

  # Reserve half of VMEM for double-buffered input/output blocks, leaving the
  # remainder for compiler scratch and register spills.
  vmem_budget_bytes = vmem_capacity_bytes // 2

  bc_single_bytes = 2 * _aligned_vmem_tile_bytes(
      1, block_len, d_state, _F32_BYTES, num_sublanes, num_lanes
  )
  states_per_head_bytes = 2 * _aligned_vmem_tile_bytes(
      1, d_head, d_state, _F32_BYTES, num_sublanes, num_lanes
  )
  x_per_head_bytes = 2 * _aligned_vmem_tile_bytes(
      1, block_len, d_head, _F32_BYTES, num_sublanes, num_lanes
  )
  if include_decay_tile:
    decay_per_head_bytes = 2 * _aligned_vmem_tile_bytes(
        1, 1, block_len, _F32_BYTES, num_sublanes, num_lanes
    )
  else:
    decay_per_head_bytes = 0

  if n_groups < n_heads:
    target_heads = n_heads // n_groups
    fixed_bytes = bc_single_bytes
    per_head_bytes = (
        states_per_head_bytes + x_per_head_bytes + decay_per_head_bytes
    )
  else:
    target_heads = n_heads
    fixed_bytes = 0
    per_head_bytes = (
        bc_single_bytes
        + states_per_head_bytes
        + x_per_head_bytes
        + decay_per_head_bytes
    )

  max_head_block = max(1, (vmem_budget_bytes - fixed_bytes) // per_head_bytes)
  return _largest_divisor(target_heads, max_head_block)


# Keep standard State Space Duality (SSD) mathematical notation (`X_c`, `A_c`,
# `B_c`, `C_c`, `Y_diag`, `Y_off`, `C`, `H`, `G`, `L`, `Q`, `R`, `P`, `N`, `S`,
# `S_init`).
# pylint: disable=invalid-name


@jaxtyping.jaxtyped
def _segsum_jax(
    x: Float[Array, "... L"],
    *,
    strict_causal: bool = False,
    mimo_rank: int = 1,
) -> Float[Array, "... L L"]:
  """Computes lower-triangular segment sums of `x` along its last axis.

  Args:
    x: Log-decay tensor `(..., L)` where `L = Q * mimo_rank`.
    strict_causal: When `True` (Mamba-3), masks the diagonal token block to
      `-inf` (`exp(-inf) == 0`). When `False` (Mamba-2), sets the diagonal token
      block to `0.0` (`exp(0) == 1`).
    mimo_rank: MIMO channel rank `R >= 1` dividing `L`.

  Returns:
    Segment-sum matrix `(..., L, L)`.
  """
  if x.ndim < 1:
    raise ValueError(f"Expected at least 1D x, got shape {x.shape}.")
  *leading_dims, L = x.shape
  if mimo_rank < 1 or L % mimo_rank != 0:
    raise ValueError(
        f"Expected mimo_rank >= 1 dividing L={L}, got mimo_rank={mimo_rank}."
    )
  if mimo_rank > 1:
    Q = L // mimo_rank
    x_tok = jnp.sum(x.reshape(*leading_dims, Q, mimo_rank), axis=-1)
    segsum_tok = _segsum_jax(x_tok, strict_causal=strict_causal, mimo_rank=1)
    return jnp.repeat(
        jnp.repeat(segsum_tok, mimo_rank, axis=-2), mimo_rank, axis=-1
    )

  x_expanded = jnp.expand_dims(x, axis=-1)
  x_repeated = jnp.broadcast_to(x_expanded, (*leading_dims, L, L))

  mask_subdiag = jnp.tril(jnp.ones((L, L), dtype=bool), k=-1)
  x_masked = jnp.where(mask_subdiag, x_repeated, 0.0)
  x_segsum = jnp.cumsum(x_masked, axis=-2)

  mask_causal = (
      mask_subdiag
      if strict_causal
      else jnp.tril(jnp.ones((L, L), dtype=bool), k=0)
  )
  return jnp.where(mask_causal, x_segsum, -jnp.inf)


def _validate_and_prepare_ssd_inputs(
    X_c: jax.Array,
    A_c: jax.Array,
    B_c: jax.Array,
    C_c: jax.Array,
    last_chunk_indices: jax.Array,
    initial_states: jax.Array | None,
    reset_mask: jax.Array | None,
    *,
    mimo_rank: int,
    work_dtype: jax.typing.DTypeLike = jnp.float32,
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    jax.Array,
    jax.Array | None,
    jax.Array,
]:
  """Validates SSD scan inputs and normalizes `A_tok` and state masks."""
  if X_c.ndim != 4 or min(X_c.shape) <= 0:
    raise ValueError(
        "Expected positive 4D X_c (n_chunks, n_heads, block_len, d_head), "
        f"got {X_c.shape}."
    )
  C, H, L, P = X_c.shape
  if mimo_rank < 1 or L % mimo_rank != 0:
    raise ValueError(
        f"Expected mimo_rank >= 1 dividing block_len={L}, "
        f"got mimo_rank={mimo_rank}."
    )
  Q = L // mimo_rank

  if mimo_rank == 1:
    if A_c.shape != (C, H, L):
      raise ValueError(f"Expected A_c shape {(C, H, L)}, got {A_c.shape}.")
    A_tok = A_c.astype(work_dtype)
  else:
    valid_a_shapes = ((C, H, L), (C, H, Q))
    if A_c.shape not in valid_a_shapes:
      raise ValueError(
          f"Expected A_c shape in {valid_a_shapes}, got {A_c.shape}."
      )
    A_c_work = A_c.astype(work_dtype)
    _, _, a_len = A_c_work.shape
    if a_len == L:
      A_tok = jnp.sum(A_c_work.reshape(C, H, Q, mimo_rank), axis=-1)
    else:
      A_tok = A_c_work

  if B_c.ndim != 4:
    raise ValueError(
        "Expected positive 4D B_c (n_chunks, n_groups, block_len, d_state) "
        f"with leading/chunk dims ({C}, n_groups, {L}) and n_groups dividing "
        f"n_heads={H}, got {B_c.shape}."
    )
  b_chunks, G, b_len, N = B_c.shape
  if (b_chunks, b_len) != (C, L) or G <= 0 or N <= 0 or H % G != 0:
    raise ValueError(
        "Expected positive 4D B_c (n_chunks, n_groups, block_len, d_state) "
        f"with leading/chunk dims ({C}, n_groups, {L}) and n_groups dividing "
        f"n_heads={H}, got {B_c.shape}."
    )
  if C_c.shape != (C, G, L, N):
    raise ValueError(f"Expected C_c shape {(C, G, L, N)}, got {C_c.shape}.")
  if last_chunk_indices.ndim != 1:
    raise ValueError(
        "Expected non-empty 1D last_chunk_indices (num_out_states,), "
        f"got {last_chunk_indices.shape}."
    )
  (S,) = last_chunk_indices.shape
  if S <= 0:
    raise ValueError(
        "Expected non-empty 1D last_chunk_indices (num_out_states,), "
        f"got {last_chunk_indices.shape}."
    )

  if initial_states is None:
    initial_states_work = None
  else:
    if initial_states.ndim != 4:
      raise ValueError(
          f"Expected initial_states shape (num_seqs, {H}, {P}, {N}) with "
          f"num_seqs > 0, got {initial_states.shape}."
      )
    S_init, *init_tail = initial_states.shape
    if S_init <= 0 or tuple(init_tail) != (H, P, N):
      raise ValueError(
          f"Expected initial_states shape (num_seqs, {H}, {P}, {N}) with "
          f"num_seqs > 0, got {initial_states.shape}."
      )
    initial_states_work = initial_states.astype(work_dtype)

  if reset_mask is None:
    reset_mask_bool = jnp.zeros((C,), dtype=bool).at[0].set(True)
  elif reset_mask.shape != (C,):
    raise ValueError(
        f"Expected reset_mask shape {(C,)}, got {reset_mask.shape}."
    )
  else:
    reset_mask_bool = jnp.asarray(reset_mask, dtype=bool)

  return (
      C,
      H,
      G,
      L,
      Q,
      P,
      N,
      A_tok,
      initial_states_work,
      reset_mask_bool,
  )


def _ssd_inter_chunk_scan_zero_init(
    carry: jax.Array,
    args: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Inter-chunk state recurrence step when `initial_states is None`."""
  states_c_c, decay_chunk_c, reset_c = args
  incoming_state = jnp.where(reset_c, jnp.zeros_like(carry), carry)
  next_state = incoming_state * decay_chunk_c[:, None, None] + states_c_c
  return next_state, (incoming_state, next_state)


def _run_inter_chunk_scan(
    states_c: jax.Array,
    decay_chunk: jax.Array,
    initial_states_work: jax.Array | None,
    reset_mask_bool: jax.Array,
    *,
    H: int,
    P: int,
    N: int,
    work_dtype: jax.typing.DTypeLike,
) -> tuple[jax.Array, jax.Array]:
  """Runs sequential inter-chunk state recurrence over `C` chunks."""
  init_carry = jnp.zeros((H, P, N), dtype=work_dtype)
  if initial_states_work is None:
    _, (states_incoming, states_outgoing) = jax.lax.scan(
        _ssd_inter_chunk_scan_zero_init,
        init_carry,
        (states_c, decay_chunk, reset_mask_bool),
    )
    return states_incoming, states_outgoing

  S_init, _, _, _ = initial_states_work.shape

  def _scan_body_with_init(carry_and_seq, args):
    carry, seq_idx = carry_and_seq
    states_c_c, decay_chunk_c, reset_c = args
    safe_seq_idx = jnp.minimum(seq_idx, S_init - 1)
    incoming_state = jnp.where(
        reset_c, initial_states_work[safe_seq_idx], carry
    )
    next_seq_idx = jnp.where(reset_c, seq_idx + jnp.int32(1), seq_idx)
    next_state = incoming_state * decay_chunk_c[:, None, None] + states_c_c
    return (next_state, next_seq_idx), (incoming_state, next_state)

  _, (states_incoming, states_outgoing) = jax.lax.scan(
      _scan_body_with_init,
      (init_carry, jnp.int32(0)),
      (states_c, decay_chunk, reset_mask_bool),
  )
  return states_incoming, states_outgoing


@jaxtyping.jaxtyped
def ssd_scan_jax(
    X_c: Float[Array, "C H L P"],
    A_c: Float[Array, "C H L"] | Float[Array, "C H Q"],
    B_c: Float[Array, "C G L N"],
    C_c: Float[Array, "C G L N"],
    last_chunk_indices: Int[Array, "S"],
    initial_states: Float[Array, "S_init H P N"] | None = None,
    reset_mask: Bool[Array, "C"] | None = None,
    *,
    strict_causal: bool = False,
    mimo_rank: int = 1,
) -> tuple[Float[Array, "C H L P"], Float[Array, "S H P N"]]:
  """Plain-XLA reference implementation of the 4-step SSD scan.

  All contractions pin `precision=jax.lax.Precision.HIGHEST` so TPU XLA does
  not downcast `float32` reference accumulations to `bfloat16`.

  Args:
    X_c: Discretized input `(C, H, L, P)`.
    A_c: Log decay `(C, H, L)` or, when `mimo_rank > 1`, per-token `(C, H, Q)`.
    B_c: Input projection `(C, G, L, N)` where `G` divides `H`.
    C_c: Output projection `(C, G, L, N)` where `G` divides `H`.
    last_chunk_indices: Chunk indices `(S,)` in `[0, C)` (or `-1` sentinels).
    initial_states: Optional per-sequence initial states `(S_init, H, P, N)`.
    reset_mask: Optional boolean sequence-start mask `(C,)`.
    strict_causal: Exclude `s == l` diagonal from intra-chunk output (Mamba-3).
    mimo_rank: MIMO channel rank `R >= 1` dividing `L`.

  Returns:
    Tuple `(Y, final_states)` with `Y` of shape `(C, H, L, P)` in `X_c.dtype`
    and `final_states` of shape `(S, H, P, N)` in `float32` (or `float64` when
    inputs are `float64`).
  """
  work_dtype = (
      jnp.float64 if jnp.dtype(X_c.dtype) == jnp.float64 else jnp.float32
  )
  C, H, G, L, _, P, N, A_tok, initial_states_work, reset_mask_bool = (
      _validate_and_prepare_ssd_inputs(
          X_c,
          A_c,
          B_c,
          C_c,
          last_chunk_indices,
          initial_states,
          reset_mask,
          mimo_rank=mimo_rank,
          work_dtype=work_dtype,
      )
  )
  A_cumsum = jnp.cumsum(A_tok, axis=-1)

  # 1. Intra-chunk diagonal/sub-diagonal output (Y_diag).
  causal_mask_tok = jnp.exp(
      _segsum_jax(A_tok, strict_causal=strict_causal, mimo_rank=1)
  )
  if mimo_rank > 1:
    causal_mask = jnp.repeat(
        jnp.repeat(causal_mask_tok, mimo_rank, axis=-2), mimo_rank, axis=-1
    )
  else:
    causal_mask = causal_mask_tok

  if G == H:
    Y_diag = jnp.einsum(
        "chln,chsn,chls,chsp->chlp",
        C_c.astype(work_dtype),
        B_c.astype(work_dtype),
        causal_mask.astype(work_dtype),
        X_c.astype(work_dtype),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=work_dtype,
    )
  else:
    heads_per_group = H // G
    causal_grouped = causal_mask.astype(work_dtype).reshape(
        C, G, heads_per_group, L, L
    )
    X_grouped = X_c.astype(work_dtype).reshape(C, G, heads_per_group, L, P)
    Y_diag = jnp.einsum(
        "cgln,cgsn,cghls,cghsp->cghlp",
        C_c.astype(work_dtype),
        B_c.astype(work_dtype),
        causal_grouped,
        X_grouped,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=work_dtype,
    ).reshape(C, H, L, P)

  # 2. Intra-chunk state generation in work_dtype.
  decay_tok = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
  decay_expanded = (
      jnp.repeat(decay_tok, mimo_rank, axis=-1) if mimo_rank > 1 else decay_tok
  )
  if G == H:
    states_c = jnp.einsum(
        "chln,chlp,chl->chpn",
        B_c.astype(work_dtype),
        X_c.astype(work_dtype),
        decay_expanded.astype(work_dtype),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=work_dtype,
    )
  else:
    heads_per_group = H // G
    states_c = jnp.einsum(
        "cgln,cghlp,cghl->cghpn",
        B_c.astype(work_dtype),
        X_c.astype(work_dtype).reshape(C, G, heads_per_group, L, P),
        decay_expanded.astype(work_dtype).reshape(C, G, heads_per_group, L),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=work_dtype,
    ).reshape(C, H, P, N)

  # 3. Inter-chunk recurrence (JAX Scan).
  decay_chunk = jnp.exp(A_cumsum[..., -1])
  states_incoming, states_outgoing = _run_inter_chunk_scan(
      states_c,
      decay_chunk,
      initial_states_work,
      reset_mask_bool,
      H=H,
      P=P,
      N=N,
      work_dtype=work_dtype,
  )

  # 4. State-to-output conversion in work_dtype.
  if G == H:
    Y_off = jnp.einsum(
        "chln,chpn->chlp",
        C_c.astype(work_dtype),
        states_incoming,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=work_dtype,
    )
  else:
    heads_per_group = H // G
    Y_off = jnp.einsum(
        "cgln,cghpn->cghlp",
        C_c.astype(work_dtype),
        states_incoming.reshape(C, G, heads_per_group, P, N),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=work_dtype,
    ).reshape(C, H, L, P)

  state_decay_tok = jnp.exp(A_cumsum)
  state_decay_out = (
      jnp.repeat(state_decay_tok, mimo_rank, axis=-1)
      if mimo_rank > 1
      else state_decay_tok
  )[..., None]
  Y = (Y_diag + Y_off * state_decay_out).astype(X_c.dtype)

  safe_last_chunk_indices = jnp.maximum(last_chunk_indices, 0)
  final_states = states_outgoing[safe_last_chunk_indices]
  return Y, final_states


@jaxtyping.jaxtyped
def chunk_state_pallas(
    B_c: Float[Array, "C G L N"],
    decay_states: Float[Array, "C H 1 L"],
    X_c: Float[Array, "C H L P"],
    *,
    interpret: bool | None = None,
) -> Float[Array, "C H P N"]:
  """Pallas kernel for Step 2 of 4 of SSD scan (intra-chunk state generation).

  Computes in float32:
    states = jnp.einsum("cgln,chlp,chl->chpn", B_c, X_c, decay_states[:, :, 0])
  where each group `g` in `B_c` spans `H // G` consecutive heads.

  Args:
    B_c: Input state projection `(C, G, L, N)` where `G` divides `H`.
    decay_states: Intra-chunk decay `(C, H, 1, L)` with the size-1 axis in the
      sublane dimension so TPU VMEM pads `1` to `8` sublanes rather than `128`
      lanes.
    X_c: Discretized input `(C, H, L, P)`.
    interpret: Whether to run Pallas in host interpret mode. When `None`,
      automatically enables interpret mode on non-TPU backends and runs the
      compiled Mosaic TPU kernel on TPU.

  Returns:
    Intra-chunk states `(C, H, P, N)` in `float32`.
  """
  if X_c.ndim != 4 or min(X_c.shape) <= 0:
    raise ValueError(
        "Expected positive 4D X_c (n_chunks, n_heads, block_len, d_head), "
        f"got {X_c.shape}."
    )
  C, H, L, P = X_c.shape
  if B_c.ndim != 4:
    raise ValueError(
        "Expected positive 4D B_c (n_chunks, n_groups, block_len, d_state) "
        f"with leading/chunk dims ({C}, n_groups, {L}) and n_groups dividing "
        f"n_heads={H}, got {B_c.shape}."
    )
  b_chunks, G, b_len, N = B_c.shape
  if (b_chunks, b_len) != (C, L) or G <= 0 or N <= 0 or H % G != 0:
    raise ValueError(
        "Expected positive 4D B_c (n_chunks, n_groups, block_len, d_state) "
        f"with leading/chunk dims ({C}, n_groups, {L}) and n_groups dividing "
        f"n_heads={H}, got {B_c.shape}."
    )
  if decay_states.shape != (C, H, 1, L):
    raise ValueError(
        f"Expected decay_states shape {(C, H, 1, L)}, got {decay_states.shape}."
    )
  resolved_interpret = _resolve_interpret(interpret)
  th = _select_head_block_size(
      H,
      G,
      L,
      P,
      N,
      include_decay_tile=True,
      interpret=resolved_interpret,
  )
  tg = 1 if G < H else th
  nh = H // th
  blocks_per_group = (H // G) // th if G < H else 1

  def kernel(
      # Inputs.
      b_ref,  # [tg, L, N] B_c.dtype, one VMEM block
      decay_ref,  # [th, 1, L] decay_states.dtype, one VMEM block
      x_ref,  # [th, L, P] X_c.dtype, one VMEM block
      # Outputs.
      out_ref,  # [th, P, N] f32, one VMEM block
  ):
    b_val = b_ref[...].astype(jnp.float32)
    if tg != th:
      b_val = jnp.broadcast_to(b_val, (th, L, N))
    # Transpose `(th, 1, L)` -> `(th, L, 1)` in registers after loading from
    # VMEM so the size-1 axis is padded to 8 sublanes instead of 128 lanes.
    decay_val = jnp.swapaxes(decay_ref[...].astype(jnp.float32), -1, -2)
    x_val = x_ref[...].astype(jnp.float32)

    x_scaled = x_val * decay_val  # [th, L, P] f32
    # Keep Precision.HIGHEST so TPU Mosaic does not round the float32
    # `x_scaled` operand to bfloat16 before MXU contraction.
    out_ref[...] = jax.lax.dot_general(
        x_scaled,
        b_val,
        (((1,), (1,)), ((0,), (0,))),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

  grid = (C, nh)

  in_specs = [
      pl.BlockSpec(
          memory_space=pltpu.VMEM,
          index_map=lambda c_i, h_i: (c_i, h_i // blocks_per_group, 0, 0),
          block_shape=(None, tg, L, N),
      ),
      pl.BlockSpec(
          memory_space=pltpu.VMEM,
          index_map=lambda c_i, h_i: (c_i, h_i, 0, 0),
          block_shape=(None, th, 1, L),
      ),
      pl.BlockSpec(
          memory_space=pltpu.VMEM,
          index_map=lambda c_i, h_i: (c_i, h_i, 0, 0),
          block_shape=(None, th, L, P),
      ),
  ]

  out_spec = pl.BlockSpec(
      memory_space=pltpu.VMEM,
      index_map=lambda c_i, h_i: (c_i, h_i, 0, 0),
      block_shape=(None, th, P, N),
  )

  out_shape = (C, H, P, N)

  return cast(
      jax.Array,
      pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
          grid=grid,
          in_specs=in_specs,
          out_specs=out_spec,
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel", "parallel"),
              disable_bounds_checks=True,
          ),
          interpret=resolved_interpret,
          name="ssd_chunk_state_pallas",
      )(B_c, decay_states, X_c),
  )


@jaxtyping.jaxtyped
def chunk_scan_pallas(
    C_c: Float[Array, "C G L N"],
    states: Float[Array, "C H P N"],
    *,
    out_dtype: jax.typing.DTypeLike | None = None,
    interpret: bool | None = None,
) -> Float[Array, "C H L P"]:
  """Pallas kernel for Step 4 of 4 of SSD scan (state-to-output conversion).

  Computes:
    Y_off = jnp.einsum("cgln,chpn->chlp", C_c, states)
  where each group `g` in `C_c` spans `H // G` consecutive heads. The caller
  applies the per-position decay `exp(A_cumsum)` to the result.

  Args:
    C_c: Output state projection `(C, G, L, N)` where `G` divides `H`.
    states: Incoming chunk states `(C, H, P, N)`.
    out_dtype: Output dtype; defaults to `C_c.dtype` when `None`.
    interpret: Whether to run Pallas in host interpret mode. When `None`,
      automatically enables interpret mode on non-TPU backends and runs the
      compiled Mosaic TPU kernel on TPU.

  Returns:
    Off-diagonal scan output `(C, H, L, P)` in `out_dtype` (or `C_c.dtype` when
    `out_dtype` is `None`).
  """
  if C_c.ndim != 4 or min(C_c.shape) <= 0:
    raise ValueError(
        "Expected positive 4D C_c (n_chunks, n_groups, block_len, d_state), "
        f"got {C_c.shape}."
    )
  C, G, L, N = C_c.shape
  if states.ndim != 4 or min(states.shape) <= 0:
    raise ValueError(
        "Expected positive 4D states (n_chunks, n_heads, d_head, d_state) "
        f"with n_chunks={C}, d_state={N}, and n_heads divisible by "
        f"n_groups={G}, got {states.shape}."
    )
  s_chunks, H, P, s_state = states.shape
  if s_chunks != C or s_state != N or H % G != 0:
    raise ValueError(
        "Expected positive 4D states (n_chunks, n_heads, d_head, d_state) "
        f"with n_chunks={C}, d_state={N}, and n_heads divisible by "
        f"n_groups={G}, got {states.shape}."
    )
  resolved_out_dtype = jnp.dtype(C_c.dtype if out_dtype is None else out_dtype)
  resolved_interpret = _resolve_interpret(interpret)
  th = _select_head_block_size(
      H,
      G,
      L,
      P,
      N,
      include_decay_tile=False,
      interpret=resolved_interpret,
  )
  tg = 1 if G < H else th
  nh = H // th
  blocks_per_group = (H // G) // th if G < H else 1

  def kernel(
      # Inputs.
      c_ref,  # [tg, L, N] C_c.dtype, one VMEM block
      states_ref,  # [th, P, N] states.dtype, one VMEM block
      # Outputs.
      out_ref,  # [th, L, P] resolved_out_dtype, one VMEM block
  ):
    c_val = c_ref[...].astype(jnp.float32)
    if tg != th:
      c_val = jnp.broadcast_to(c_val, (th, L, N))
    states_val = states_ref[...].astype(jnp.float32)

    # Keep Precision.HIGHEST so TPU Mosaic preserves the float32 inter-chunk
    # carry `states_val` during MXU contraction instead of truncating to bf16.
    prod = jax.lax.dot_general(
        c_val,
        states_val,
        (((2,), (2,)), ((0,), (0,))),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    out_ref[...] = prod.astype(resolved_out_dtype)

  grid = (C, nh)

  in_specs = [
      pl.BlockSpec(
          memory_space=pltpu.VMEM,
          index_map=lambda c_i, h_i: (c_i, h_i // blocks_per_group, 0, 0),
          block_shape=(None, tg, L, N),
      ),
      pl.BlockSpec(
          memory_space=pltpu.VMEM,
          index_map=lambda c_i, h_i: (c_i, h_i, 0, 0),
          block_shape=(None, th, P, N),
      ),
  ]

  out_spec = pl.BlockSpec(
      memory_space=pltpu.VMEM,
      index_map=lambda c_i, h_i: (c_i, h_i, 0, 0),
      block_shape=(None, th, L, P),
  )

  out_shape = (C, H, L, P)

  return cast(
      jax.Array,
      pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(out_shape, resolved_out_dtype),
          grid=grid,
          in_specs=in_specs,
          out_specs=out_spec,
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel", "parallel"),
              disable_bounds_checks=True,
          ),
          interpret=resolved_interpret,
          name="ssd_chunk_scan_pallas",
      )(C_c, states),
  )


@jaxtyping.jaxtyped
def ssd_minimal_discrete_hybrid(
    X_c: Float[Array, "C H L P"],
    A_c: Float[Array, "C H L"] | Float[Array, "C H Q"],
    B_c: Float[Array, "C G L N"],
    C_c: Float[Array, "C G L N"],
    last_chunk_indices: Int[Array, "S"],
    initial_states: Float[Array, "S_init H P N"] | None = None,
    reset_mask: Bool[Array, "C"] | None = None,
    *,
    strict_causal: bool = False,
    mimo_rank: int = 1,
    interpret: bool | None = None,
) -> tuple[Float[Array, "C H L P"], Float[Array, "S H P N"]]:
  """Hybrid JAX/Pallas Selective State Space (SSD) scan for Mamba-2 and Mamba-3.

  Supports Mamba-2 SISO (`strict_causal=False, mimo_rank=1`), Mamba-3 SISO
  (`strict_causal=True, mimo_rank=1`), Mamba-3 MIMO (`mimo_rank > 1`, where
  `L = Q * mimo_rank` and `A_c` may be passed either rank-expanded as
  `(C, H, L)` or per-token as `(C, H, Q)`), and grouped `B_c`/`C_c` projections
  `(C, G, L, N)` where `G` divides `H`.

  Caller invariants for ragged/padded batches:
    - Sub-chunk tail padding: Padded tail positions in a sequence's final chunk
      must set `A_c = 0.0` (so `exp(A_c) == 1.0`, preserving the last valid
      token's state across padded positions) and `X_c = 0.0` (or `B_c = 0.0`).
    - Finite log-decay `A_c`: All entries of `A_c` must be finite (`<= 0.0`,
      never `-inf`), because `-inf - (-inf)` in `A_cumsum[..., -1:] - A_cumsum`
      produces `NaN`.
    - Checkpoint & final state gathering: `last_chunk_indices` of shape `(S,)`
      may index one final chunk per sequence or arbitrary intermediate/final
      chunk indices in `[0, C)`. Inactive padding slots may pass `-1`
      sentinels, which are clamped to chunk `0` via `jnp.maximum(..., 0)`
      without clamping upper-bound indices.

  Args:
    X_c: Discretized input `(C, H, L, P)`.
    A_c: Log decay `(C, H, L)` or, when `mimo_rank > 1`, per-token `(C, H, Q)`
      where `Q = L // mimo_rank`.
    B_c: Input projection `(C, G, L, N)` where `G` divides `H`.
    C_c: Output projection `(C, G, L, N)` where `G` divides `H`.
    last_chunk_indices: Chunk indices `(S,)` in `[0, C)` (or `-1` sentinels).
    initial_states: Optional `(S_init, H, P, N)` per-sequence initial states
      indexed in sequence order at chunks where `reset_mask` is `True`. When
      `None`, sequence starts reset directly to zero inside the scan without
      allocating a 4D zero buffer in HBM.
    reset_mask: Optional boolean sequence-start mask `(C,)`; defaults to `[True,
      False, ...]` (single contiguous sequence) when `None`.
    strict_causal: Exclude `s == l` diagonal from intra-chunk output (Mamba-3).
    mimo_rank: MIMO channel rank `R >= 1` dividing `L`.
    interpret: Whether to run Pallas kernels in host interpret mode. When
      `None`, automatically enables interpret mode on non-TPU backends and runs
      compiled Mosaic TPU kernels on TPU.

  Returns:
    Tuple `(Y, final_states)` where `Y` has shape `(C, H, L, P)` in `X_c.dtype`
    and `final_states` has shape `(S, H, P, N)` in `float32`.
  """
  C, H, G, L, _, P, N, A_tok, initial_states_f32, reset_mask_bool = (
      _validate_and_prepare_ssd_inputs(
          X_c,
          A_c,
          B_c,
          C_c,
          last_chunk_indices,
          initial_states,
          reset_mask,
          mimo_rank=mimo_rank,
          work_dtype=jnp.float32,
      )
  )
  resolved_interpret = _resolve_interpret(interpret)
  A_cumsum = jnp.cumsum(A_tok, axis=-1)

  # 1. Intra-chunk diagonal/sub-diagonal output (Y_diag). When `G < H`,
  # contract `B_c` and `C_c` along their native group axis rather than
  # materializing a `jnp.repeat` copy across `H` in HBM.
  causal_mask_tok = jnp.exp(
      _segsum_jax(A_tok, strict_causal=strict_causal, mimo_rank=1)
  )
  if mimo_rank > 1:
    causal_mask = jnp.repeat(
        jnp.repeat(causal_mask_tok, mimo_rank, axis=-2), mimo_rank, axis=-1
    )
  else:
    causal_mask = causal_mask_tok

  if G == H:
    Y_diag = jnp.einsum(
        "chln,chsn,chls,chsp->chlp",
        C_c,
        B_c,
        causal_mask,
        X_c,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
  else:
    heads_per_group = H // G
    causal_grouped = causal_mask.reshape(C, G, heads_per_group, L, L)
    X_grouped = X_c.reshape(C, G, heads_per_group, L, P)
    Y_diag = jnp.einsum(
        "cgln,cgsn,cghls,cghsp->cghlp",
        C_c,
        B_c,
        causal_grouped,
        X_grouped,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).reshape(C, H, L, P)

  # 2. Intra-chunk state generation in float32 (Pallas). Place the size-1 axis
  # in the sublane dimension `(C, H, 1, L)` so TPU VMEM pads `1` to `8`
  # sublanes instead of `128` lanes.
  decay_tok = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
  decay_expanded = (
      jnp.repeat(decay_tok, mimo_rank, axis=-1) if mimo_rank > 1 else decay_tok
  )
  decay_states = decay_expanded[:, :, None, :]
  states_c = chunk_state_pallas(
      B_c, decay_states, X_c, interpret=resolved_interpret
  )

  # 3. Inter-chunk recurrence in float32 (JAX Scan).
  decay_chunk = jnp.exp(A_cumsum[..., -1])
  states_incoming, states_outgoing = _run_inter_chunk_scan(
      states_c,
      decay_chunk,
      initial_states_f32,
      reset_mask_bool,
      H=H,
      P=P,
      N=N,
      work_dtype=jnp.float32,
  )

  # 4. State-to-output conversion (Pallas). Keep Y_off in float32 until summed
  # with Y_diag so opposite-sign terms do not suffer pre-cancellation bfloat16
  # rounding before the single output cast.
  Y_off = chunk_scan_pallas(
      C_c, states_incoming, out_dtype=jnp.float32, interpret=resolved_interpret
  )
  state_decay_tok = jnp.exp(A_cumsum)
  state_decay_out = (
      jnp.repeat(state_decay_tok, mimo_rank, axis=-1)
      if mimo_rank > 1
      else state_decay_tok
  )[..., None]
  Y = (Y_diag.astype(jnp.float32) + Y_off * state_decay_out).astype(X_c.dtype)

  # Clamp negative `-1` padding sentinels to `0` without clamping the upper
  # bound so out-of-range caller indices are not masked.
  safe_last_chunk_indices = jnp.maximum(last_chunk_indices, 0)
  final_states = states_outgoing[safe_last_chunk_indices]
  return Y, final_states


ssd_scan = ssd_minimal_discrete_hybrid

# pylint: enable=invalid-name
