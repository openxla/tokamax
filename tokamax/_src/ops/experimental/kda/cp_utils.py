# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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

Each rank computes ``(S_ext, M)`` locally assuming ``S_in = 0``, all-gathers
both tensors, and then rebuilds its true boundary state
``S_in_r = sum_j (prod_{j' > j} M_{j'}) @ S_ext,j`` from upstream ranks.

**Entry form.** The framework (e.g. MaxText) shards ``segment_ids`` along
T and feeds the rank-local slice into ``chunk_kda``; the caller passes a
MINIMAL ``CPContext(mesh, axis_name)``. chunk_kda derives
``cu_local`` + chain metadata (``pre_num_ranks`` / ``is_first_rank`` / ...)
device-side via ``_derive_cp_metadata_from_segment_ids`` (one small
``all_gather`` over first/last segment ids) and fills them back into the
context. All ranks share one trace inside ``shard_map``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, TYPE_CHECKING, Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import pydantic

if TYPE_CHECKING:
  import jax.sharding


@dataclass(frozen=True, eq=False)
class CPContext:
  """Per-rank context parallel state.

  Mirrors ``fla.ops.cp.context.FLACPContext`` field-for-field, with
  ``group: ProcessGroup`` replaced by ``mesh + axis_name``.

  All fields are rank-local: when a single segment spans multiple ranks,
  ``cu_seqlens`` reflects only the portion this rank holds, and
  ``pre_num_ranks`` / ``post_num_ranks`` describe the rank's position in
  the segment-spanning chain.

  All fields except ``mesh`` and ``axis_name`` are typically left as
  ``None`` by the caller: ``chunk_kda``'s CP entry path derives
  ``cu_seqlens`` + chain fields from rank-local ``segment_ids`` via one
  small ``all_gather`` (see ``_derive_cp_metadata_from_segment_ids``),
  then fills them back via ``dataclasses.replace`` before passing the
  completed context to the downstream Pallas kernel.

  The chain fields (``is_first_rank`` / ``pre_num_ranks`` etc.) accept
  EITHER Python static values OR traced ``jax.Array`` values — the
  traced path is what makes ``shard_map`` work (all ranks share one
  trace, so collective ops pair correctly).

  ``__hash__`` and ``__eq__`` are based on identity + static fields only
  (``mesh`` id, ``axis_name``, ``cp_size``), so dataclass instances stay
  hashable even when chain fields are traced jnp.array values. This
  keeps the context suitable for use as static operation metadata.

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
  """

  mesh: "jax.sharding.Mesh"
  axis_name: str = "context"
  is_first_rank: bool | jax.Array | None = None
  is_last_rank: bool | jax.Array | None = None
  pre_num_ranks: int | jax.Array | None = None
  post_num_ranks: int | jax.Array | None = None
  conv1d_kernel_size: int | None = None
  pre_num_conv_tokens: int | None = None

  # Identity-based hash + eq so traced jnp.array fields (which are not
  # hashable) don't break dataclass auto-generated __hash__. This is safe
  # because each chunk_kda call site constructs its own CPContext — identity
  # hashing keys the cache per call site, which is what we want.
  def __hash__(self) -> int:
    return hash((id(self.mesh), self.axis_name, self.cp_size))

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, CPContext):
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


def _exclude_cp_context_from_json(_: CPContext | None) -> None:
  """Keeps the runtime mesh out of serialized Op metadata."""
  return None


CPContextArg: TypeAlias = Annotated[
    CPContext | None,
    pydantic.PlainSerializer(
        _exclude_cp_context_from_json,
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
  n_max: int,
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
    n_max: Static upper bound on per-rank segment count. Determines the
      padded shape of the returned ``cu_local``.
    chunk_size: Passed through to ``segment_ids_to_seqlens`` (default 1).

  Returns:
    Tuple ``(cu_local, chain_meta)``:
      - ``cu_local``: ``[n_max + 1]`` int32 (padded by repeating last value).
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

  # ── (1) cu_local: re-use the existing helper. It pads to n_max+1 by
  # repeating the total valid length (matches the zero-length-tail
  # convention pre-process / chunk_kda_fwd already handle).
  from tokamax._src.ops.experimental.kda.utils import segment_ids_to_seqlens  # local import to avoid cycle
  cu_local = segment_ids_to_seqlens(seg, max_segs=n_max, chunk_size=chunk_size)

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
