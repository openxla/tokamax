# Copyright 2026 Ant Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Ported from primatrix/pallas-kernel dca6a09feec2659f2777a6db02f2ee55560f991e.

"""Pallas TPU cost estimates for KDA chunked-linear-attention kernels.

Why this module exists
----------------------
Every ``pl.pallas_call`` in the KDA kernels historically omitted the
``cost_estimate=`` argument. Without it the XLA:TPU latency-hiding
scheduler treats the custom kernel as opaque with unknown cost, so it
cannot overlap the kernel with collectives (all-gather / reduce-scatter)
or order it sensibly relative to other ops. Supplying even an approximate
FLOP count restores that ability.

Precision requirement
----------------------
The estimates only need to be correct to within ~2x: the scheduler uses
them for *relative* comparison, not for absolute timing. All formulas
below count the dominant matmuls of each stage; cheap elementwise ops are
folded into the ``transcendentals`` term (gate ``exp2``) where they matter.

Zero runtime overhead
----------------------
``pl.CostEstimate`` is a compile-time annotation attached to the HLO
custom-call op. The Python arithmetic here runs once at trace time; the
generated kernel machine code is byte-identical with or without it.

Shape glossary (head-first layout ``[H, B, T, ...]``):
  H  = num heads,  B = batch,       T = seq len
  BT = chunk size, NT = T // BT = num chunks
  K  = head dim,   V = value dim
"""

from __future__ import annotations

from jax.experimental import pallas as pl


def _mm(m: int, k: int, n: int) -> int:
  """FLOPs of an ``(m, k) @ (k, n)`` matmul (factor 2 = multiply + add)."""
  return 2 * m * k * n


def make_cost_estimate(
    flops, *, bytes_accessed: int = 0, transcendentals: int = 0
):
  """Wrap raw counts in a ``pl.CostEstimate``.

  Returns ``None`` when every count is zero. Pure data-movement kernels may
  legitimately use ``flops=0`` with a nonzero ``bytes_accessed`` estimate.
  """
  flops = int(flops)
  bytes_accessed = int(bytes_accessed)
  transcendentals = int(transcendentals)
  if flops < 0 or bytes_accessed < 0 or transcendentals < 0:
    raise ValueError("Cost-estimate counts must be non-negative")
  if flops == 0 and bytes_accessed == 0 and transcendentals == 0:
    return None
  return pl.CostEstimate(
      flops=flops,
      bytes_accessed=bytes_accessed,
      transcendentals=transcendentals,
  )


def resolve_cost_estimate(flops_override, computed):
  """Pick a user override over the auto-computed estimate.

  Mirrors MaxText main's splash-attention pattern (``attention_op.py``):
  a config int of ``-1`` (or any negative) means "use the kernel's own
  estimate", ``0`` disables the hint, and ``> 0`` overrides the FLOP count
  the XLA:TPU scheduler sees — letting operators hand-tune
  collective/compute overlap without editing kernel code.

  Args:
    flops_override: int from config. ``< 0`` -> use ``computed``;
      ``0`` -> no hint; ``> 0`` -> build a flat override estimate.
    computed: the auto-computed ``pl.CostEstimate`` (or ``None``).

  Returns:
    A ``pl.CostEstimate`` or ``None`` suitable for ``pallas_call``.
  """
  if flops_override == 0:
    return None
  if flops_override is not None and flops_override > 0:
    return make_cost_estimate(flops_override)
  return computed


def combine_cost_estimates(*parts):
  """Sum multiple ``pl.CostEstimate`` objects into one hint."""
  flops = sum(p.flops for p in parts if p is not None)
  bytes_accessed = sum(p.bytes_accessed for p in parts if p is not None)
  transcendentals = sum(p.transcendentals for p in parts if p is not None)
  return make_cost_estimate(
      flops,
      bytes_accessed=bytes_accessed,
      transcendentals=transcendentals,
  )


def split_cost_estimate_flops(total_flops, *parts):
  """Split a total override across kernel stages by their auto FLOP weights.

  ``cost_estimate_flops_fwd`` describes a complete KDA forward, while the
  staged implementation contains two Pallas calls.  Applying the total to
  both calls double counts it.  This helper preserves the public total-cost
  semantics while retaining one annotation per Pallas call.
  """
  if total_flops is None or total_flops < 0:
    return (-1,) * len(parts)
  if total_flops == 0:
    return (0,) * len(parts)
  weights = [p.flops if p is not None else 0 for p in parts]
  total_weight = sum(weights)
  if total_weight == 0:
    return (0,) * (len(parts) - 1) + (int(total_flops),)
  result = [
      int(round(total_flops * weight / total_weight)) for weight in weights
  ]
  result[-1] += int(total_flops) - sum(result)
  return tuple(result)


def _neumann_steps(block_size: int) -> int:
  if block_size < 4 or block_size & (block_size - 1):
    raise ValueError(
        f"block_size must be a power of two >= 4, got {block_size}"
    )
  return block_size.bit_length() - 2


def kda_intra_fwd_neumann_cost(
    H, B, NT, BT, K, V, *, block_size: int = 16, fused: bool = False
):
  """Cost for the block-Neumann S1+S2 implementation.

  The model follows the actual static block plan: diagonal doubling steps are
  derived from ``block_size`` and the off-diagonal solve from ``BT/block``.
  ``fused`` selects the merged-Horner implementation used by the BF16 fused
  and mega kernels.
  """
  block_size = min(block_size, BT)
  if BT % block_size:
    raise ValueError(f"BT={BT} must be divisible by block_size={block_size}")
  n = H * B * NT
  nc = BT // block_size
  rhs_width = BT + K + V
  # Aqk/L construction, diagonal Neumann doubling, and first P@[F|rhs].
  per_chunk = 2 * _mm(BT, K, BT)
  per_chunk += 2 * _neumann_steps(block_size) * _mm(BT, BT, BT)
  per_chunk += _mm(BT, BT, BT + rhs_width)
  if nc == 1:
    per_chunk += _mm(BT, BT, rhs_width)
  elif fused:
    horner_steps = max(0, (nc.bit_length() - 1) - 1)
    per_chunk += _mm(BT, BT, BT)  # H^2
    per_chunk += max(0, horner_steps - 1) * _mm(2 * BT, BT, BT)
    per_chunk += _mm(BT, BT, BT)  # final inv_I_G @ Hk
    per_chunk += _mm(BT, BT, rhs_width)
  else:
    per_chunk += max(0, nc - 2) * _mm(BT, BT, BT)
    per_chunk += _mm(BT, BT, rhs_width)
  trans = H * B * NT * BT * K  # gate exp2 over [., BT, K]
  return make_cost_estimate(n * per_chunk, transcendentals=trans)


def kda_intra_fwd_exact_cost(H, B, NT, BT, K, V, *, block_size: int = 16):
  """Cost for the FP32 block-forward-substitution S1+S2 implementation."""
  block_size = min(block_size, BT)
  if BT % block_size:
    raise ValueError(f"BT={BT} must be divisible by block_size={block_size}")
  n = H * B * NT
  # Aqk/L plus triangular solves for [v, k, I].  The solve is block based,
  # hence its dominant work scales with BT^2 times the RHS width.
  rhs_width = V + K + BT
  per_chunk = 2 * _mm(BT, K, BT) + _mm(BT, BT, rhs_width)
  trans = H * B * NT * BT * K
  return make_cost_estimate(n * per_chunk, transcendentals=trans)


def kda_intra_fwd_cost(
    H, B, NT, BT, K, V, *, use_neumann: bool = True, fused: bool = False
):
  """Compatibility dispatcher for the two S1+S2 algorithms."""
  if use_neumann:
    return kda_intra_fwd_neumann_cost(H, B, NT, BT, K, V, fused=fused)
  return kda_intra_fwd_exact_cost(H, B, NT, BT, K, V)


def kda_inter_fwd_cost(H, B, NT, BT, K, V):
  """Stage 3+4 forward: per-chunk state propagation + output.

  Dominant matmuls per (head, batch, chunk):
    v_new = u - w @ h              -> 2*BT*K*V
    o     = qg @ h                 -> 2*BT*K*V
    o    += Aqk @ v_new            -> 2*BT*BT*V
    h_new = decay*h + kg^T @ v_new -> 2*K*BT*V
  """
  n = H * B * NT
  per_chunk = _mm(BT, K, V) + _mm(BT, K, V) + _mm(BT, BT, V) + _mm(K, BT, V)
  trans = H * B * NT * BT * K  # exp2 gate scaling of q and h
  return make_cost_estimate(n * per_chunk, transcendentals=trans)


def kda_vnew_cost(H, B, NT, BT, K, V):
  """Recompute ``v_new = u - w @ h`` from a stored hidden state.

  Single state matmul per chunk: 2*BT*K*V.
  """
  n = H * B * NT
  return make_cost_estimate(n * _mm(BT, K, V))


def kda_recompute_cost(H, B, NT, BT, K, V):
  """Backward Stage 0: recompute ``w``, ``u``, ``qg``, and ``kg``.

  ``Akk`` is already available from forward, so this stage only applies it:
    u = Akk @ (beta * v) -> 2*BT*BT*V
    w = Akk @ gated(k)   -> 2*BT*BT*K

  A fused save-h variant additionally derives ``v_new``; callers add
  :func:`kda_vnew_cost` for that variant.
  """
  n = H * B * NT
  per_chunk = _mm(BT, BT, V) + _mm(BT, BT, K)
  trans = 2 * H * B * NT * BT * K  # qg and kg gate exp2
  return make_cost_estimate(n * per_chunk, transcendentals=trans)


def kda_compact_output_cost(B, H, T_aligned, T_out, V, *, dtype_bytes: int):
  """HBM traffic for compacting aligned token-major output to head-first.

  The kernel reads the aligned ``[B, T_aligned, H, V]`` workspace once and
  writes the packed ``[H, B, T_out, V]`` result once. Small sequence-offset
  metadata is intentionally omitted.
  """
  bytes_accessed = B * H * V * (T_aligned + T_out) * dtype_bytes
  return make_cost_estimate(0, bytes_accessed=bytes_accessed)


def kda_intra_bwd_cost(H, B, NT, BT, K, *, sub_chunk_size: int = 16):
  """Cost for subchunk intra backward, matching its NC×NC block loops."""
  sub_chunk_size = min(sub_chunk_size, BT)
  if BT % sub_chunk_size:
    raise ValueError(
        f"BT={BT} must be divisible by sub_chunk_size={sub_chunk_size}"
    )
  n = H * B * NT
  nc = BT // sub_chunk_size
  # Diagonal work plus row/column interactions for each distinct block pair.
  per_chunk = nc * 4 * _mm(sub_chunk_size, sub_chunk_size, K)
  per_chunk += 2 * nc * max(0, nc - 1) * _mm(sub_chunk_size, K, sub_chunk_size)
  trans = 2 * n * BT * K
  return make_cost_estimate(n * per_chunk, transcendentals=trans)


def kda_dav_cost(H, B, NT, BT, K, V):
  """Backward Stage 1: dAqk and dv from ``do`` and ``v_new``.

  Dominant matmuls per chunk:
    dAqk = do @ v_new^T -> 2*BT*V*BT
    dv  += Aqk^T @ do   -> 2*BT*BT*V
  """
  n = H * B * NT
  per_chunk = _mm(BT, V, BT) + _mm(BT, BT, V)
  return make_cost_estimate(n * per_chunk)


def kda_dh_cost(H, B, NT, BT, K, V):
  """Backward Stage 2: dh state recurrence + dv2 (mirror of inter fwd).

  Dominant matmuls per chunk:
    dh  += q^T @ do -> 2*K*BT*V
    dv2 += k  @ dh  -> 2*BT*K*V
    dh  -= w^T @ dv -> 2*K*BT*V
  """
  n = H * B * NT
  per_chunk = 2 * _mm(K, BT, V) + _mm(BT, K, V)
  return make_cost_estimate(n * per_chunk)


def kda_wy_bwd_cost(H, B, NT, BT, K, V):
  """Backward Stage 3: dq, dk, dv, db, dg, dAkk via WY representation.

  Dominant matmuls per chunk, excluding the Dhu recurrence counted by
  :func:`kda_dh_cost`:
    [do; dv] @ h^T, v_new @ dh^T     -> 6*BT*V*K
    dv @ v^T, A^T @ dv               -> 4*BT*BT*V
    dw @ kg^T, A^T @ dw              -> 4*BT*BT*K
    dA @ A^T, A^T @ dA               -> 4*BT*BT*BT
  """
  n = H * B * NT
  per_chunk = (
      _mm(2 * BT, V, K)
      + _mm(BT, V, K)
      + _mm(BT, V, BT)
      + _mm(BT, BT, V)
      + _mm(BT, K, BT)
      + _mm(BT, BT, K)
      + 2 * _mm(BT, BT, BT)
  )
  trans = 3 * H * B * NT * BT * K
  return make_cost_estimate(n * per_chunk, transcendentals=trans)


def kda_bwd_mega_cost(H, B, NT, BT, K, V, *, sub_chunk_size: int = 16):
  """Fused mega backward kernel: recompute + dAv + dh + WY + intra bwd.

  Sum the component estimates so the scheduler sees the full (large) cost
  of the single fused kernel.
  """
  parts = [
      kda_recompute_cost(H, B, NT, BT, K, V),
      kda_vnew_cost(H, B, NT, BT, K, V),
      kda_dav_cost(H, B, NT, BT, K, V),
      kda_dh_cost(H, B, NT, BT, K, V),
      kda_wy_bwd_cost(H, B, NT, BT, K, V),
      kda_intra_bwd_cost(H, B, NT, BT, K, sub_chunk_size=sub_chunk_size),
  ]
  flops = sum(p.flops for p in parts if p is not None)
  trans = sum(p.transcendentals for p in parts if p is not None)
  return make_cost_estimate(flops, transcendentals=trans)
