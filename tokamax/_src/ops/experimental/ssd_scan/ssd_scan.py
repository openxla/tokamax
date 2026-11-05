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
#   C      number of chunks
#   H      number of SSM heads
#   th     head tile extent per Pallas grid step (divides H)
#   nh     head grid extent: nh = H // th
#   L      packed chunk length (L = Q * R)
#   Q      tokens per chunk (Q = L // R)
#   R      MIMO channel rank (mimo_rank >= 1; R == 1 for SISO)
#   P      head value dimension (d_head)
#   N      SSM state dimension (d_state)
#   S      number of output sequences (num_seqs)

from typing import cast

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping

# Maximum number of SSM heads blocked per Pallas grid step. At `th=8`, `L=64`,
# `P=64`, and `N=128` (or `L=256`), double-buffered VMEM blocks for `B_c`/`C_c`,
# `X_c`/`out`, `states`, and `chunk_state_pallas`'s 128-lane padded `(L, 1)`
# decay vector consume ~1.6 MiB (~5.5 MiB at `L=256` in float32), fitting in
# scoped VMEM.
_MAX_HEAD_BLOCK_SIZE = 8


def _largest_divisor(n: int, max_div: int) -> int:
  """Returns the largest positive divisor of `n` that is at most `max_div`."""
  if n <= 0 or max_div <= 0:
    raise ValueError(f"Expected n > 0 and max_div > 0, got {n=}, {max_div=}.")
  for candidate in range(min(n, max_div), 0, -1):
    if n % candidate == 0:
      return candidate
  return 1


def _resolve_interpret(interpret: bool | None) -> bool:
  """Resolves `None` to host interpret mode on non-TPU backends."""
  return (jax.default_backend() != "tpu") if interpret is None else interpret


# Keep standard State Space Duality (SSD) mathematical notation (`X_c`, `A_c`,
# `B_c`, `C_c`, `Y_diag`, `Y_off`, `C`, `H`, `L`, `Q`, `R`, `P`, `N`, `S`).
# pylint: disable=invalid-name


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
) -> tuple[int, int, int, int, int, int, jax.Array, jax.Array, jax.Array]:
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
        f"Expected positive 4D B_c with leading shape {(C, H, L)}, "
        f"got {B_c.shape}."
    )
  b_chunks, b_heads, b_len, N = B_c.shape
  if (b_chunks, b_heads, b_len) != (C, H, L) or N <= 0:
    raise ValueError(
        f"Expected positive 4D B_c with leading shape {(C, H, L)}, "
        f"got {B_c.shape}."
    )
  if C_c.shape != (C, H, L, N):
    raise ValueError(f"Expected C_c shape {(C, H, L, N)}, got {C_c.shape}.")
  if last_chunk_indices.ndim != 1:
    raise ValueError(
        "Expected 1D last_chunk_indices (num_seqs,), "
        f"got {last_chunk_indices.shape}."
    )

  if initial_states is None:
    initial_states_work = jnp.zeros((C, H, P, N), dtype=work_dtype)
  else:
    if initial_states.shape != (C, H, P, N):
      raise ValueError(
          f"Expected initial_states shape {(C, H, P, N)}, "
          f"got {initial_states.shape}."
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
      L,
      Q,
      P,
      N,
      A_tok,
      initial_states_work,
      reset_mask_bool,
  )


def _ssd_inter_chunk_scan_body(
    carry: jax.Array,
    args: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Inter-chunk state recurrence step for `jax.lax.scan`."""
  states_c_c, decay_chunk_c, init_state_c, reset_c = args
  incoming_state = jnp.where(reset_c, init_state_c, carry)
  next_state = incoming_state * decay_chunk_c[:, None, None] + states_c_c
  return next_state, (incoming_state, next_state)


@jaxtyping.jaxtyped
def ssd_scan_jax(
    X_c: Float[Array, "C H L P"],
    A_c: Float[Array, "C H L"] | Float[Array, "C H Q"],
    B_c: Float[Array, "C H L N"],
    C_c: Float[Array, "C H L N"],
    last_chunk_indices: Int[Array, "S"],
    initial_states: Float[Array, "C H P N"] | None = None,
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
    B_c: Input projection `(C, H, L, N)`.
    C_c: Output projection `(C, H, L, N)`.
    last_chunk_indices: Final chunk index per sequence `(S,)`.
    initial_states: Optional initial states `(C, H, P, N)`.
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
  C, H, _, _, P, N, A_tok, initial_states_work, reset_mask_bool = (
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
  Y_diag = jnp.einsum(
      "chln,chsn,chls,chsp->chlp",
      C_c.astype(work_dtype),
      B_c.astype(work_dtype),
      causal_mask.astype(work_dtype),
      X_c.astype(work_dtype),
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=work_dtype,
  )

  # 2. Intra-chunk state generation in work_dtype.
  decay_tok = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
  decay_expanded = (
      jnp.repeat(decay_tok, mimo_rank, axis=-1) if mimo_rank > 1 else decay_tok
  )
  states_c = jnp.einsum(
      "chln,chlp,chl->chpn",
      B_c.astype(work_dtype),
      X_c.astype(work_dtype),
      decay_expanded.astype(work_dtype),
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=work_dtype,
  )

  # 3. Inter-chunk recurrence (JAX Scan).
  decay_chunk = jnp.exp(A_cumsum[..., -1])
  init_carry = jnp.zeros((H, P, N), dtype=work_dtype)
  _, (states_incoming, states_outgoing) = jax.lax.scan(
      _ssd_inter_chunk_scan_body,
      init_carry,
      (states_c, decay_chunk, initial_states_work, reset_mask_bool),
  )

  # 4. State-to-output conversion in work_dtype.
  Y_off = jnp.einsum(
      "chln,chpn->chlp",
      C_c.astype(work_dtype),
      states_incoming,
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=work_dtype,
  )
  state_decay_tok = jnp.exp(A_cumsum)
  state_decay_out = (
      jnp.repeat(state_decay_tok, mimo_rank, axis=-1)
      if mimo_rank > 1
      else state_decay_tok
  )[..., None]
  Y = (Y_diag + Y_off * state_decay_out).astype(X_c.dtype)

  safe_last_chunk_indices = jnp.clip(last_chunk_indices, 0, C - 1)
  final_states = states_outgoing[safe_last_chunk_indices]
  return Y, final_states


@jaxtyping.jaxtyped
def chunk_state_pallas(
    B_c: Float[Array, "C H L N"],
    decay_states: Float[Array, "C H L 1"],
    X_c: Float[Array, "C H L P"],
    *,
    interpret: bool | None = None,
) -> Float[Array, "C H P N"]:
  """Pallas kernel for Step 2 of 4 of SSD scan (intra-chunk state generation).

  Computes in float32:
    states = jnp.einsum("chln,chlp,chl->chpn", B_c, X_c, decay_states[..., 0])

  Several heads are blocked per grid step (`block_shape=(1, th, ...)` with `th`
  dividing `H`) to amortize per-step Pallas launch overhead while keeping the
  chunk axis `parallel` for megacore.

  Args:
    B_c: Input state projection `(C, H, L, N)`.
    decay_states: Intra-chunk decay `(C, H, L, 1)`.
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
        f"Expected positive 4D B_c with leading shape {(C, H, L)}, "
        f"got {B_c.shape}."
    )
  b_chunks, b_heads, b_len, N = B_c.shape
  if (b_chunks, b_heads, b_len) != (C, H, L) or N <= 0:
    raise ValueError(
        f"Expected positive 4D B_c with leading shape {(C, H, L)}, "
        f"got {B_c.shape}."
    )
  if decay_states.shape != (C, H, L, 1):
    raise ValueError(
        f"Expected decay_states shape {(C, H, L, 1)}, got {decay_states.shape}."
    )
  th = _largest_divisor(H, _MAX_HEAD_BLOCK_SIZE)
  nh = H // th
  resolved_interpret = _resolve_interpret(interpret)

  def kernel(
      # Inputs.
      b_ref,  # [1, th, L, N] B_c.dtype, one VMEM block
      decay_ref,  # [1, th, L, 1] decay_states.dtype, one VMEM block
      x_ref,  # [1, th, L, P] X_c.dtype, one VMEM block
      # Outputs.
      out_ref,  # [1, th, P, N] f32, one VMEM block
  ):
    # Squeeze the singleton chunk block axis (`axis=0`) so Mosaic TPU lowers a
    # 3D `[th, L, P] x [th, L, N]` batch-matrix contraction on the MXU.
    b_val = jnp.squeeze(b_ref[...], axis=0).astype(jnp.float32)
    decay_val = jnp.squeeze(decay_ref[...], axis=0).astype(jnp.float32)
    x_val = jnp.squeeze(x_ref[...], axis=0).astype(jnp.float32)

    # `decay_ref` carries a trailing size-1 dimension `(th, L, 1)` to satisfy
    # Pallas TPU's 2D minimum tiled block rank and broadcast across `P`.
    x_scaled = x_val * decay_val  # [th, L, P] f32
    # Keep Precision.HIGHEST so TPU Mosaic does not round the float32
    # `x_scaled` operand to bfloat16 before MXU contraction.
    out = jax.lax.dot_general(
        x_scaled,
        b_val,
        (((1,), (1,)), ((0,), (0,))),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    out_ref[...] = jnp.expand_dims(out, axis=0)

  grid = (C, nh)

  in_specs = [
      pl.BlockSpec(
          memory_space=pltpu.VMEM,
          index_map=lambda c_i, h_i: (c_i, h_i, 0, 0),
          block_shape=(1, th, L, N),
      ),
      pl.BlockSpec(
          memory_space=pltpu.VMEM,
          index_map=lambda c_i, h_i: (c_i, h_i, 0, 0),
          block_shape=(1, th, L, 1),
      ),
      pl.BlockSpec(
          memory_space=pltpu.VMEM,
          index_map=lambda c_i, h_i: (c_i, h_i, 0, 0),
          block_shape=(1, th, L, P),
      ),
  ]

  out_spec = pl.BlockSpec(
      memory_space=pltpu.VMEM,
      index_map=lambda c_i, h_i: (c_i, h_i, 0, 0),
      block_shape=(1, th, P, N),
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
    C_c: Float[Array, "C H L N"],
    states: Float[Array, "C H P N"],
    *,
    out_dtype: jax.typing.DTypeLike | None = None,
    interpret: bool | None = None,
) -> Float[Array, "C H L P"]:
  """Pallas kernel for Step 4 of 4 of SSD scan (state-to-output conversion).

  Computes:
    Y_off = jnp.einsum("chln,chpn->chlp", C_c, states)

  The caller applies the per-position decay `exp(A_cumsum)` to the result.

  Args:
    C_c: Output state projection `(C, H, L, N)`.
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
        "Expected positive 4D C_c (n_chunks, n_heads, block_len, d_state), "
        f"got {C_c.shape}."
    )
  C, H, L, N = C_c.shape
  if states.ndim != 4:
    raise ValueError(
        f"Expected positive 4D states with shape ({C}, {H}, d_head, {N}), "
        f"got {states.shape}."
    )
  s_chunks, s_heads, P, s_state = states.shape
  if (s_chunks, s_heads) != (C, H) or P <= 0 or s_state != N:
    raise ValueError(
        f"Expected positive 4D states with shape ({C}, {H}, d_head, {N}), "
        f"got {states.shape}."
    )
  th = _largest_divisor(H, _MAX_HEAD_BLOCK_SIZE)
  nh = H // th
  resolved_out_dtype = jnp.dtype(C_c.dtype if out_dtype is None else out_dtype)
  resolved_interpret = _resolve_interpret(interpret)

  def kernel(
      # Inputs.
      c_ref,  # [1, th, L, N] C_c.dtype, one VMEM block
      states_ref,  # [1, th, P, N] states.dtype, one VMEM block
      # Outputs.
      out_ref,  # [1, th, L, P] resolved_out_dtype, one VMEM block
  ):
    # Squeeze the singleton chunk block axis (`axis=0`) so Mosaic TPU lowers a
    # 3D `[th, L, N] x [th, P, N]` batch-matrix contraction on the MXU.
    c_val = jnp.squeeze(c_ref[...], axis=0).astype(jnp.float32)
    states_val = jnp.squeeze(states_ref[...], axis=0).astype(jnp.float32)

    # Keep Precision.HIGHEST so TPU Mosaic preserves the float32 inter-chunk
    # carry `states_val` during MXU contraction instead of truncating to bf16.
    prod = jax.lax.dot_general(
        c_val,
        states_val,
        (((2,), (2,)), ((0,), (0,))),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    out_ref[...] = jnp.expand_dims(prod.astype(resolved_out_dtype), axis=0)

  grid = (C, nh)

  in_specs = [
      pl.BlockSpec(
          memory_space=pltpu.VMEM,
          index_map=lambda c_i, h_i: (c_i, h_i, 0, 0),
          block_shape=(1, th, L, N),
      ),
      pl.BlockSpec(
          memory_space=pltpu.VMEM,
          index_map=lambda c_i, h_i: (c_i, h_i, 0, 0),
          block_shape=(1, th, P, N),
      ),
  ]

  out_spec = pl.BlockSpec(
      memory_space=pltpu.VMEM,
      index_map=lambda c_i, h_i: (c_i, h_i, 0, 0),
      block_shape=(1, th, L, P),
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
    B_c: Float[Array, "C H L N"],
    C_c: Float[Array, "C H L N"],
    last_chunk_indices: Int[Array, "S"],
    initial_states: Float[Array, "C H P N"] | None = None,
    reset_mask: Bool[Array, "C"] | None = None,
    *,
    strict_causal: bool = False,
    mimo_rank: int = 1,
    interpret: bool | None = None,
) -> tuple[Float[Array, "C H L P"], Float[Array, "S H P N"]]:
  """Hybrid JAX/Pallas Selective State Space (SSD) scan for Mamba-2 and Mamba-3.

  Supports Mamba-2 SISO (`strict_causal=False, mimo_rank=1`), Mamba-3 SISO
  (`strict_causal=True, mimo_rank=1`), and Mamba-3 MIMO (`mimo_rank > 1`, where
  `L = Q * mimo_rank` and `A_c` may be passed either rank-expanded as
  `(C, H, L)` or per-token as `(C, H, Q)`).

  Args:
    X_c: Discretized input `(C, H, L, P)`.
    A_c: Log decay `(C, H, L)` or, when `mimo_rank > 1`, per-token `(C, H, Q)`
      where `Q = L // mimo_rank`.
    B_c: Input projection `(C, H, L, N)`.
    C_c: Output projection `(C, H, L, N)`.
    last_chunk_indices: Final chunk index per sequence `(S,)`.
    initial_states: Optional initial states `(C, H, P, N)` read at chunks where
      `reset_mask` is `True`; defaults to zeros when `None`.
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
  C, H, _, _, P, N, A_tok, initial_states_f32, reset_mask_bool = (
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
  Y_diag = jnp.einsum(
      "chln,chsn,chls,chsp->chlp",
      C_c,
      B_c,
      causal_mask,
      X_c,
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
  )

  # 2. Intra-chunk state generation in float32 (Pallas). Append a size-1
  # trailing axis `(..., L, 1)` to satisfy Pallas TPU's 2D minimum tiled
  # `block_shape` requirement and broadcast in registers.
  decay_tok = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
  decay_expanded = (
      jnp.repeat(decay_tok, mimo_rank, axis=-1) if mimo_rank > 1 else decay_tok
  )
  decay_states = decay_expanded[..., None]
  states_c = chunk_state_pallas(
      B_c, decay_states, X_c, interpret=resolved_interpret
  )

  # 3. Inter-chunk recurrence in float32 (JAX Scan).
  decay_chunk = jnp.exp(A_cumsum[..., -1])
  scan_args = (states_c, decay_chunk, initial_states_f32, reset_mask_bool)
  init_carry = jnp.zeros((H, P, N), dtype=jnp.float32)
  _, (states_incoming, states_outgoing) = jax.lax.scan(
      _ssd_inter_chunk_scan_body, init_carry, scan_args
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

  # Clip sentinel `-1` entries from padded sequence slots to `[0, C - 1]`
  # to avoid out-of-bounds gather indexing.
  safe_last_chunk_indices = jnp.clip(last_chunk_indices, 0, C - 1)
  final_states = states_outgoing[safe_last_chunk_indices]
  return Y, final_states


ssd_scan = ssd_minimal_discrete_hybrid

# pylint: enable=invalid-name
