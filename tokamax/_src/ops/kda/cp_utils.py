# Copyright 2026 Ant Group. All Rights Reserved.
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
"""Context Parallel (CP) utilities for FLA-aligned KDA attention.

Provides the data class, communication primitive wrappers, metadata
derivation, and local-merge helper required to support sequence-dimension
parallelism over a JAX mesh axis (typically ``"context"``).

Aligned with ``flash-linear-attention/fla/ops/cp/`` (context.py, comm.py).
The torch ``ProcessGroup`` field is replaced by ``mesh + axis_name`` so the
context can be constructed in JAX without a torch.distributed dependency.

Algorithm: all-gather each rank's local state and transition matrix, then
merge the contributions in sequence order.

Each rank computes ``(S_ext, M)`` locally assuming ``S_in = 0`` (FWD) or
``dS_out = 0`` (BWD), all-gathers both tensors, and then rebuilds its true
boundary state ``S_in_r = sum_j (prod_{j' > j} M_{j'}) @ S_ext,j`` from
upstream ranks (forward) or downstream ranks (backward).

The forward and backward paths share the collectives and state-merge helpers
defined here.

**Entry form.** The framework (e.g. MaxText) shards ``segment_ids`` along
T and feeds the rank-local slice into ``chunk_kda``; the caller passes a
MINIMAL ``ContextParallelMetadata(mesh, axis_name)``. chunk_kda derives
``cu_local`` + chain metadata (``pre_num_ranks`` / ``is_first_rank`` / ...)
device-side via ``_derive_cp_metadata_from_segment_ids`` (one small
``all_gather`` over first/last segment ids) and fills them back into the
context. All ranks share one trace inside ``shard_map``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, TYPE_CHECKING, Any, TypeAlias

import jax
from jax import ad_checkpoint
import jax.numpy as jnp
import numpy as np
import pydantic

if TYPE_CHECKING:
  import jax.sharding


@dataclass(frozen=True, eq=False)
class ContextParallelMetadata:
  """Per-rank context-parallel metadata.

  Mirrors ``fla.ops.cp.context.FLACPContext`` field-for-field, with
  ``group: ProcessGroup`` replaced by ``mesh + axis_name``.

  All fields are rank-local: when a single segment spans multiple ranks,
  ``cu_seqlens`` reflects only the portion this rank holds, and
  ``pre_num_ranks`` / ``post_num_ranks`` describe the rank's position in
  the segment-spanning chain.

  In-place sequence CP (used by ``gated_delta_net`` and stateless
  ``kimi_delta_attention`` when ``T_local % chunk_size == 0``) requires only
  ``ContextParallelMetadata(mesh, axis_name)`` (plus ``max_num_segments`` on
  KDA when ``segment_ids`` is provided; ``segment_ids > max_num_segments`` are
  clipped to padding). The chain fields (``is_first_rank``, ``is_last_rank``,
  ``pre_num_ranks``, ``post_num_ranks``) are retained only for legacy varlen
  ``chunk_kda`` compatibility.

  The chain fields (``is_first_rank`` / ``pre_num_ranks`` etc.) accept
  EITHER Python static values OR traced ``jax.Array`` values — the
  traced path is what makes ``shard_map`` work (all ranks share one
  trace, so collective ops pair correctly).

  ``__hash__`` and ``__eq__`` are based on identity + static fields only
  (``mesh`` id, ``axis_name``, ``cp_size``), so dataclass instances stay
  hashable even when chain fields are traced jnp.array values. This
  matters because ``context_parallel_metadata`` is a ``nondiff_argnum`` of
  ``chunk_kda``'s ``jax.custom_vjp`` decoration.

  Attributes:
    mesh: The JAX mesh containing the CP axis.
    axis_name: Name of the CP axis on ``mesh`` (e.g. ``"context"``).
    is_first_rank: Filled by chunk_kda. Python ``bool`` or traced
      ``jnp.bool_`` scalar. True iff this rank's first segment is not
      continuation of an upstream rank.
    is_last_rank: Filled by chunk_kda. True iff this rank's last segment
      ends within this rank.
    pre_num_ranks: Filled by chunk_kda. How many ranks BEFORE this one
      share the first segment.
    post_num_ranks: Filled by chunk_kda. How many ranks AFTER this one
      share the last segment.
    conv1d_kernel_size: Reserved for conv1d CP. ``None`` for KDA.
    pre_num_conv_tokens: Reserved for conv1d CP. ``None`` for KDA.
    cp_residual_checkpoint_name: Optional JAX ``ad_checkpoint.checkpoint_name``
      tag applied to cross-rank halo and incoming-state activations.
  """

  mesh: "jax.sharding.Mesh"
  axis_name: str = "context"
  is_first_rank: bool | jax.Array | None = None
  is_last_rank: bool | jax.Array | None = None
  pre_num_ranks: int | jax.Array | None = None
  post_num_ranks: int | jax.Array | None = None
  conv1d_kernel_size: int | None = None
  pre_num_conv_tokens: int | None = None
  cp_residual_checkpoint_name: str | None = None

  # Identity-based hash + eq so traced jnp.array fields (which are not
  # hashable) don't break dataclass auto-generated __hash__. This is safe
  # for our use as a custom_vjp nondiff arg because each chunk_kda call
  # site constructs its own ContextParallelMetadata — identity hashing keys the
  # cache per call site, which is what we want.
  def __hash__(self) -> int:
    return hash((
        id(self.mesh),
        self.axis_name,
        self.cp_size,
        self.cp_residual_checkpoint_name,
    ))

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, ContextParallelMetadata):
      return NotImplemented
    return self is other

  @property
  def cp_size(self) -> int:
    """Number of ranks on the CP axis."""
    return int(self.mesh.shape[self.axis_name])

  @property
  def is_cp_enabled(self) -> bool:
    """True iff CP is actually parallelised (``cp_size > 1``)."""
    return self.cp_size > 1


def _exclude_context_parallel_metadata_from_json(
    _: ContextParallelMetadata | None,
) -> None:
  """Keeps the runtime mesh out of serialized Op metadata."""
  return None


ContextParallelMetadataArg: TypeAlias = Annotated[
    ContextParallelMetadata | None,
    pydantic.PlainSerializer(
        _exclude_context_parallel_metadata_from_json,
        return_type=type(None),
        when_used="json",
    ),
]


def all_gather_into_tensor(
  inp: jax.Array,
  axis_name: str,
) -> tuple[jax.Array, None]:
  """All-gather ``inp`` along ``axis_name``, stacking on a new leading axis.

  Must be invoked inside a ``shard_map`` (or other context that defines
  ``axis_name``). For GSPMD/auto-sharding callers, wrap the consumer in
  ``shard_map`` first.

  Mirrors the (tensor, handle) tuple from FLA ``comm.all_gather_into_tensor``
  so call-sites stay aligned; ``handle`` is always ``None`` (JAX dispatches
  collectives via XLA).

  Args:
    inp: Local shard, any shape / dtype.
    axis_name: Mesh axis to gather across.

  Returns:
    A tuple ``(out, None)`` where ``out`` has shape ``[cp_size, *inp.shape]``
    and ``out[j]`` holds rank ``j``'s input.
  """
  out = jax.lax.all_gather(inp, axis_name=axis_name, axis=0, tiled=False)
  return out, None


def _derive_cp_metadata_from_segment_ids(
  segment_ids_local: jax.Array,
  axis_name: str,
  max_num_segments: int,
  chunk_size: int = 1,
) -> tuple[jax.Array, dict]:
  """Derive ``cu_local`` and chain metadata from **rank-local** segment_ids.

  Production CP entry point: the framework (e.g. MaxText) shards
  ``segment_ids`` along T and feeds the rank-local slice into
  ``chunk_kda``. The kernel sees only ``segment_ids[t_start:t_end]`` per
  rank — there is no way for it to know ``cu_global`` or the chain
  position. We recover both:

  1. **cu_local** via the existing ``segment_ids_to_seqlens`` helper
     (transitions ``seg[i] != seg[i-1]`` mark segment boundaries; ID
     absolute values don't matter for boundary detection).
  2. **chain metadata** via one small ``all_gather`` (2 × cp_size int32):
     each rank advertises its first and last segment ID, then compares
     neighbours to detect whether its first segment is a continuation of
     the previous rank's last (→ ``is_first_rank=False``).

  **Assumption** (holds for MaxText-style frameworks): T-dim slicing is
  a plain ``segment_ids_global[t_start:t_end]`` — **IDs are NOT
  re-numbered per rank**. If a segment spans ranks, all owning ranks see
  the same ID for that segment. (The B dimension can re-number per
  batch element independently; that's fine because CP requires B=1.)

  Args:
    segment_ids_local: ``[T_local]`` or ``[1, T_local]`` int32. 1-indexed
      (``0 = padding``). Same id convention as ``segment_ids_to_seqlens``.
    axis_name: Mesh axis to gather across (the CP axis).
    max_num_segments: Static upper bound on per-rank segment count. Determines
      the padded shape of the returned ``cu_local``.
    chunk_size: Passed through to ``segment_ids_to_seqlens`` (default 1).

  Returns:
    Tuple ``(cu_local, chain_meta)``:
      - ``cu_local``: ``[max_num_segments + 1]`` int32 (padded by repeating the
        last value).
      - ``chain_meta``: dict with traced scalar entries
        ``{'pre_num_ranks', 'post_num_ranks', 'is_first_rank', 'is_last_rank'}``.

  Must be called inside a ``shard_map`` (or equivalent) where
  ``axis_name`` is defined.
  """
  # Normalise to 1D [T_local] — CP requires B=1.
  if segment_ids_local.ndim == 2:
    assert segment_ids_local.shape[0] == 1, (
      f"CP requires B=1 (packed varlen); got B={segment_ids_local.shape[0]}"
    )
    seg = segment_ids_local[0]
  else:
    assert segment_ids_local.ndim == 1, (
      f"segment_ids must be [T] or [1, T]; got ndim={segment_ids_local.ndim}"
    )
    seg = segment_ids_local
  T_local = seg.shape[0]

  # ── (1) cu_local: re-use the existing helper. It pads to
  # max_num_segments+1 by repeating the total valid length (matches the
  # zero-length-tail convention pre-process / chunk_kda_fwd already handle).
  from tokamax._src.ops.kda.utils import segment_ids_to_seqlens  # local import to avoid cycle
  cu_local = segment_ids_to_seqlens(
      seg, max_segs=max_num_segments, chunk_size=chunk_size
  )

  # ── (2) chain metadata via one small all_gather.
  # First/last *valid* (non-padding) segment id on this rank. argmax on a
  # boolean returns the FIRST True; reversing handles the last.
  valid = seg != 0  # [T_local]
  # If a rank is entirely padding (no real tokens), first_seg / last_seg
  # collapse to 0 — they won't match any neighbour's real id, so the
  # chain detection naturally classifies the rank as a singleton.
  first_idx = jnp.argmax(valid.astype(jnp.int32))  # first True position
  last_idx = jnp.int32(T_local - 1) - jnp.argmax(valid[::-1].astype(jnp.int32))
  first_seg = seg[first_idx].astype(jnp.int32)
  last_seg = seg[last_idx].astype(jnp.int32)

  first_all, _ = all_gather_into_tensor(first_seg, axis_name)  # [cp_size]
  last_all, _ = all_gather_into_tensor(last_seg, axis_name)
  rank = jax.lax.axis_index(axis_name)
  cp_size_static = first_all.shape[0]

  my_first = jax.lax.dynamic_index_in_dim(first_all, rank, axis=0, keepdims=False)
  my_last = jax.lax.dynamic_index_in_dim(last_all, rank, axis=0, keepdims=False)

  # is_first_rank: rank 0, OR previous rank's last seg id ≠ my first.
  prev_last = jax.lax.dynamic_index_in_dim(
    last_all, jnp.maximum(rank - 1, 0), axis=0, keepdims=False,
  )
  is_first_rank = (rank == 0) | (prev_last != my_first)

  # is_last_rank: rank cp_size-1, OR next rank's first seg id ≠ my last.
  next_first = jax.lax.dynamic_index_in_dim(
    first_all,
    jnp.minimum(rank + 1, jnp.int32(cp_size_static - 1)),
    axis=0, keepdims=False,
  )
  is_last_rank = (rank == cp_size_static - 1) | (next_first != my_last)

  # pre_num_ranks: largest k such that for all r' in [rank-k, rank-1],
  # last_all[r'] == my_first. Walk upstream from rank-1 down to 0,
  # short-circuit on first mismatch.
  def _pre_body(i, carry):
    keep_going, count = carry
    k = rank - 1 - jnp.asarray(i, jnp.int32)
    valid_k = k >= 0
    last_k = jax.lax.dynamic_index_in_dim(
      last_all, jnp.maximum(k, 0), axis=0, keepdims=False,
    )
    match = valid_k & (last_k == my_first)
    new_keep = keep_going & match
    new_count = count + jnp.where(new_keep, jnp.int32(1), jnp.int32(0))
    return (new_keep, new_count)

  _, pre_num_ranks = jax.lax.fori_loop(
    0, max(cp_size_static - 1, 1), _pre_body,
    (jnp.bool_(True), jnp.int32(0)),
  )

  # post_num_ranks: symmetric — walk downstream comparing first_all to my_last.
  def _post_body(i, carry):
    keep_going, count = carry
    k = rank + 1 + jnp.asarray(i, jnp.int32)
    valid_k = k < cp_size_static
    first_k = jax.lax.dynamic_index_in_dim(
      first_all,
      jnp.minimum(k, jnp.int32(cp_size_static - 1)),
      axis=0, keepdims=False,
    )
    match = valid_k & (first_k == my_last)
    new_keep = keep_going & match
    new_count = count + jnp.where(new_keep, jnp.int32(1), jnp.int32(0))
    return (new_keep, new_count)

  _, post_num_ranks = jax.lax.fori_loop(
    0, max(cp_size_static - 1, 1), _post_body,
    (jnp.bool_(True), jnp.int32(0)),
  )

  return cu_local, {
    "pre_num_ranks": pre_num_ranks,
    "post_num_ranks": post_num_ranks,
    "is_first_rank": is_first_rank,
    "is_last_rank": is_last_rank,
  }


def _merge_initial_state(
  S_ext_all: jax.Array,
  M_all: jax.Array,
  rank: jax.Array | int,
  pre_num_ranks: jax.Array | int,
  is_first_rank: jax.Array | bool,
) -> jax.Array:
  """Rebuild the current rank's ``S_in`` from all-gathered ``(S_ext, M)``.

  Applies the forward merge recurrence::

    S_in = 0
    for j in [rank - pre_num_ranks, rank):
        S_in = M_all[j] @ S_in + S_ext_all[j]

  Implementation uses a fixed-trip ``lax.fori_loop`` of ``cp_size - 1``
  iterations with a per-iter mask. This lets ``rank``, ``pre_num_ranks``,
  and ``is_first_rank`` be either Python static values OR traced jnp
  scalars, so all SPMD ranks share one trace inside ``shard_map``. The chain
  runs entirely in fp32 to avoid precision loss across ranks.

  Iteration order matches the Python reference: furthest upstream first
  (``j = rank - pre_num_ranks``), closest upstream last
  (``j = rank - 1``).

  Args:
    S_ext_all: ``[cp_size, H, B, K, V]`` fp32 — all-gathered ``S_ext``.
    M_all: ``[cp_size, H, B, K, K]`` fp32 — all-gathered transition matrix.
    rank: Current rank (Python int or traced int32 scalar).
    pre_num_ranks: Number of upstream ranks sharing this rank's first
      segment (Python int or traced int32 scalar).
    is_first_rank: True when no upstream contribution (Python bool or
      traced bool scalar).

  Returns:
    ``S_in`` with shape ``[B, H, K, V]`` fp32, ready to be inserted into
    ``initial_state[0]`` of the main forward kernel.
  """
  assert S_ext_all.dtype == jnp.float32, (
    f"S_ext_all must be fp32 (CP red line), got {S_ext_all.dtype}"
  )
  assert M_all.dtype == jnp.float32, (
    f"M_all must be fp32 (CP red line), got {M_all.dtype}"
  )
  assert S_ext_all.ndim == 5 and M_all.ndim == 5, (
    f"expected [cp_size, H, B, K, V] / [cp_size, H, B, K, K], "
    f"got S_ext_all.shape={S_ext_all.shape} M_all.shape={M_all.shape}"
  )

  cp_size, H, B, K, V = S_ext_all.shape
  assert M_all.shape == (cp_size, H, B, K, K), (
    f"M_all shape mismatch: expected {(cp_size, H, B, K, K)}, got {M_all.shape}"
  )

  S_in = jnp.zeros((H, B, K, V), dtype=jnp.float32)

  if cp_size == 1:
    # Static early-out: only one rank, no upstream to merge from.
    return S_in

  # Promote (possibly Python) values to jnp scalars so arithmetic / mask
  # combinators work uniformly across the static and traced cases.
  pre_num_arr = jnp.asarray(pre_num_ranks, dtype=jnp.int32)
  is_first_arr = jnp.asarray(is_first_rank, dtype=jnp.bool_)
  rank_arr = jnp.asarray(rank, dtype=jnp.int32)

  def body(i, S_in_carry):
    # offset goes from cp_size-1 down to 1; j = rank - offset goes from
    # (rank - cp_size + 1) up to (rank - 1), so the chain is applied
    # furthest-upstream first.
    offset = jnp.int32(cp_size - 1) - jnp.asarray(i, jnp.int32)
    j = rank_arr - offset
    active = (
      (offset <= pre_num_arr)
      & jnp.logical_not(is_first_arr)
      & (j >= 0)
      & (j < cp_size)
    )
    j_safe = jnp.clip(j, 0, cp_size - 1)
    M_j = jax.lax.dynamic_index_in_dim(M_all, j_safe, axis=0, keepdims=False)
    S_j = jax.lax.dynamic_index_in_dim(S_ext_all, j_safe, axis=0, keepdims=False)
    S_new = jnp.einsum("hbkj,hbjv->hbkv", M_j, S_in_carry) + S_j
    return jnp.where(active, S_new, S_in_carry)

  return jax.lax.fori_loop(0, cp_size - 1, body, S_in)


@jax.jit
def _merge_dht(
  dS_ext_all: jax.Array,
  dM_all: jax.Array,
  *,
  rank: jax.Array,
  post_num_ranks: jax.Array,
  is_last_rank: jax.Array,
) -> jax.Array:
  """Merge downstream final-state gradients in furthest-to-nearest order.

  Backward analogue of ``_merge_initial_state``. Accumulates in fp32.
  """
  cp_size = dS_ext_all.shape[0]
  dS_ext_all = dS_ext_all.astype(jnp.float32)
  dM_all = dM_all.astype(jnp.float32)
  zero = jnp.zeros_like(dS_ext_all[0])

  def body(i: int, carry: jax.Array) -> jax.Array:
    idx = jnp.clip(rank + post_num_ranks - i, 0, cp_size - 1)
    S_j = dS_ext_all[idx]
    M_j = dM_all[idx]
    merged = jnp.einsum("bhkj,bhjv->bhkv", M_j, carry) + S_j
    active = (i < post_num_ranks) & ~is_last_rank
    return jnp.where(active, merged, carry)

  return jax.lax.fori_loop(0, cp_size, body, zero)


def cp_shift_right(
    x: jax.Array, axis_name: str, cp_size: int
) -> jax.Array:
  """Sends `x` from rank `r - 1` to rank `r`, zeroing `rank == 0`."""
  if cp_size <= 1:
    return jnp.zeros_like(x)
  perm = [(i, i + 1) for i in range(cp_size - 1)]
  return jax.lax.ppermute(x, axis_name=axis_name, perm=perm)


def cp_shift_left(
    x: jax.Array, axis_name: str, cp_size: int
) -> jax.Array:
  """Sends `x` from rank `r + 1` to rank `r`, zeroing `rank == cp_size - 1`."""
  if cp_size <= 1:
    return jnp.zeros_like(x)
  perm = [(i + 1, i) for i in range(cp_size - 1)]
  return jax.lax.ppermute(x, axis_name=axis_name, perm=perm)


def cp_associative_scan_fwd(
    s_local: jax.Array,
    m_local: jax.Array,
    *,
    axis_name: str,
    cp_size: int,
) -> jax.Array:
  """Computes incoming state `S_in` (`[B, H, K, V]` f32) via O(log2 P) ppermutes."""
  s_acc = s_local.astype(jnp.float32)
  m_acc = m_local.astype(jnp.float32)
  if cp_size <= 1:
    return jnp.zeros_like(s_acc)

  _, _, K, V = s_acc.shape
  rank = jax.lax.axis_index(axis_name)
  # The trailing `cp_shift_right` discards rank `cp_size - 1`'s inclusive
  # prefix, so every rank's exclusive prefix spans at most `cp_size - 1` ranks
  # (0 doubling steps when `cp_size == 2`).
  stride = 1
  while stride < cp_size - 1:
    perm = [(i, i + stride) for i in range(cp_size - stride)]
    need_m = stride * 2 < cp_size - 1
    if need_m:
      packed = jnp.concatenate([s_acc, m_acc], axis=-1)
      packed_prev = jax.lax.ppermute(packed, axis_name=axis_name, perm=perm)
      s_prev = packed_prev[..., :V]
      m_prev = packed_prev[..., V : V + K]
    else:
      s_prev = jax.lax.ppermute(s_acc, axis_name=axis_name, perm=perm)
      m_prev = None
    s_comb = (
        jnp.matmul(
            m_acc,
            s_prev,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        + s_acc
    )
    active = rank >= stride
    s_acc = jnp.where(active, s_comb, s_acc)
    if need_m:
      assert m_prev is not None
      m_comb = jnp.matmul(
          m_acc,
          m_prev,
          precision=jax.lax.Precision.HIGHEST,
          preferred_element_type=jnp.float32,
      )
      m_acc = jnp.where(active, m_comb, m_acc)
    stride *= 2

  return cp_shift_right(s_acc, axis_name, cp_size)


def cp_associative_scan_bwd(
    ds_local: jax.Array,
    dm_local: jax.Array,
    *,
    axis_name: str,
    cp_size: int,
) -> jax.Array:
  """Computes outgoing state cotangent `dht` (`[B, H, K, V]` f32) via ppermutes."""
  ds_acc = ds_local.astype(jnp.float32)
  dm_acc = dm_local.astype(jnp.float32)
  if cp_size <= 1:
    return jnp.zeros_like(ds_acc)

  _, _, K, V = ds_acc.shape
  rank = jax.lax.axis_index(axis_name)
  # The trailing `cp_shift_left` discards rank 0's inclusive suffix, so every
  # rank's exclusive suffix spans at most `cp_size - 1` downstream ranks.
  stride = 1
  while stride < cp_size - 1:
    perm = [(i, i - stride) for i in range(stride, cp_size)]
    need_m = stride * 2 < cp_size - 1
    if need_m:
      packed = jnp.concatenate([ds_acc, dm_acc], axis=-1)
      packed_next = jax.lax.ppermute(packed, axis_name=axis_name, perm=perm)
      ds_next = packed_next[..., :V]
      dm_next = packed_next[..., V : V + K]
    else:
      ds_next = jax.lax.ppermute(ds_acc, axis_name=axis_name, perm=perm)
      dm_next = None
    ds_comb = (
        jnp.matmul(
            dm_acc,
            ds_next,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        + ds_acc
    )
    active = (rank + stride) < cp_size
    ds_acc = jnp.where(active, ds_comb, ds_acc)
    if need_m:
      assert dm_next is not None
      dm_comb = jnp.matmul(
          dm_acc,
          dm_next,
          precision=jax.lax.Precision.HIGHEST,
          preferred_element_type=jnp.float32,
      )
      dm_acc = jnp.where(active, dm_comb, dm_acc)
    stride *= 2

  return cp_shift_left(ds_acc, axis_name, cp_size)


def cp_canonicalize_packed_segment_ids(
    segment_ids_local: jax.Array,
    *,
    axis_name: str,
    cp_size: int,
    halo_len: int = 0,
    cp_checkpoint_name: str | None = None,
    precomputed_cp_halo: tuple[jax.Array, jax.Array | None] | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
    tuple[jax.Array, jax.Array | None],
]:
  """Canonicalizes rank-local `[B, T]` `segment_ids` across `axis_name`."""
  from tokamax._src.ops.kda.utils import canonicalize_packed_segment_ids  # pylint: disable=g-import-not-at-top

  seg = jnp.maximum(segment_ids_local.astype(jnp.int32), 0)
  _, t_local = seg.shape

  if precomputed_cp_halo is not None:
    prev_active_raw, halo_recv = precomputed_cp_halo
    cp_halo = precomputed_cp_halo
  else:
    def _prop_fn(lhs, rhs):
      l_has, l_val = lhs
      r_has, r_val = rhs
      return l_has | r_has, jnp.where(r_has, r_val, l_val)

    _, last_raw_local = jax.lax.associative_scan(
        _prop_fn, (seg > 0, seg), axis=1
    )
    has_acc = jnp.any(seg > 0, axis=1).astype(jnp.int32)
    val_acc = last_raw_local[:, -1]

    if cp_size > 1:
      rank = jax.lax.axis_index(axis_name)
      # `cp_shift_right` drops rank `cp_size - 1`, so only `cp_size - 1`
      # upstream ranks ever contribute to any exclusive prefix.
      stride = 1
      while stride < cp_size - 1:
        perm = [(i, i + stride) for i in range(cp_size - stride)]
        pair_prev = jax.lax.ppermute(
            jnp.stack([has_acc, val_acc], axis=-1),
            axis_name=axis_name,
            perm=perm,
        )
        prev_has = pair_prev[:, 0] > 0
        prev_val = pair_prev[:, 1]
        cur_has = has_acc > 0
        comb_has = (prev_has | cur_has).astype(jnp.int32)
        comb_val = jnp.where(cur_has, val_acc, prev_val)
        active = rank >= stride
        has_acc = jnp.where(active, comb_has, has_acc)
        val_acc = jnp.where(active, comb_val, val_acc)
        stride *= 2
      prev_active_raw = cp_shift_right(val_acc, axis_name, cp_size)
    else:
      prev_active_raw = jnp.zeros_like(val_acc)

    if cp_checkpoint_name is not None:
      prev_active_raw = ad_checkpoint.checkpoint_name(
          prev_active_raw, name=cp_checkpoint_name
      )

    if halo_len <= 0:
      halo_recv = None
    else:
      if t_local > halo_len:
        pre_seg = seg[:, : t_local - halo_len]
        pre_has = jnp.any(pre_seg > 0, axis=1)
        pre_end = last_raw_local[:, t_local - halo_len - 1]
        my_halo_init = jnp.where(pre_has, pre_end, prev_active_raw)
        my_halo_raw = seg[:, t_local - halo_len :]
      else:
        my_halo_init = prev_active_raw
        my_halo_raw = jnp.pad(seg, ((0, 0), (halo_len - t_local, 0)))

      halo_msg = jnp.concatenate([my_halo_init[:, None], my_halo_raw], axis=1)
      halo_recv = cp_shift_right(halo_msg, axis_name, cp_size)
      if cp_checkpoint_name is not None:
        halo_recv = ad_checkpoint.checkpoint_name(
            halo_recv, name=cp_checkpoint_name
        )
    cp_halo = (prev_active_raw, halo_recv)

  if halo_len <= 0 or halo_recv is None:
    seg_canon, act_doc, is_new_doc = canonicalize_packed_segment_ids(
        seg, prev_active_seg=prev_active_raw
    )
    return seg_canon, act_doc, is_new_doc, None, cp_halo

  halo_init_raw = halo_recv[:, 0]
  halo_seg_raw = halo_recv[:, 1:]
  h_recv = halo_seg_raw.shape[1]
  seg_canon_ext, act_doc_ext, is_new_doc_ext = canonicalize_packed_segment_ids(
      jnp.concatenate([halo_seg_raw, seg], axis=1),
      prev_active_seg=halo_init_raw,
  )
  return (
      seg_canon_ext[:, h_recv:],
      act_doc_ext[:, h_recv:],
      is_new_doc_ext[:, h_recv:],
      seg_canon_ext[:, h_recv - halo_len : h_recv],
      cp_halo,
  )
