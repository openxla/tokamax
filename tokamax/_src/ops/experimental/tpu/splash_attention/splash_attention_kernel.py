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

"""Implementation of Sparse Flash Attention, a.k.a. "Splash" attention."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import enum
import functools
import json
import math
from typing import Any, NamedTuple, Optional

import jax
from jax import ad_checkpoint
from jax import lax
from jax import tree_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask_info as mask_info_lib

P = jax.P
MaskInfo = mask_info_lib.MaskInfo
partial = functools.partial
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
# We predefine some useful dimension numbers for dot_general
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))  # standard matmul
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # RHS transposed

LOG2E = math.log2(math.e)
LOG2E_INV = 1 / LOG2E

# mypy: ignore-errors


class SegmentIds(NamedTuple):
  """SegmentIds for Q and KV sequences.

  SegmentIds are a mechanism to ensure that there is no cross-attention between
  segments (fraction of a sequence) that have been concatenated together into a
  sequence. Each array is a list of ids (integers). Only tokens with the same
  id are allowed to attend to each other.

  The static mask (e.g. causal) is "and-ed" with the segment id mask to form
  the actual attention mask. It is important that the latter does not have any
  all-zero rows (along dimension kv). Otherwise it would result in a invalid
  softmax (the denominator would be 0).
  This condition holds for causal self-attention because in this case segment
  ids form a block diagonal matrix so at least one element in each row is set.
  It is easy to break this condition with non-self-attention configurations.
  Attributes:
    q: segment ids along the Q sequence
    kv: segment ids along the KV sequence
  """

  q: jax.Array  # [q_seq_len]
  kv: jax.Array  # [kv_seq_len]


# Return type of SplashAttention function that implements the custom vjp rule.
SplashCustomReturnType = (
    jax.Array | tuple[jax.Array, tuple[jax.Array, jax.Array]]
)

SplashResidualsType = tuple[
    jax.Array,  # q
    jax.Array,  # k
    jax.Array,  # v
    Optional[SegmentIds],  # segment_ids
    jax.Array,  # out
    jax.Array,  # logsumexp
    Optional[MaskInfo],  # dq_mask_info
    Optional[MaskInfo],  # dkv_mask_info
]

MaskFunctionType = Callable[..., jax.Array]


def get_kernel_name(
    is_mqa: bool, save_residuals: bool, is_segmented: bool, phase: str
) -> str:
  """Returns a unique name for all SplashAttention kernel variants."""
  assert phase in ["dq", "dkv", "fwd"]
  # Saving residuals is supported only for the fwd phase.
  assert not save_residuals or phase == "fwd"
  residuals = "_residuals" if save_residuals else "_no_residuals"
  attention_type = "mqa" if is_mqa else "mha"
  segments = "_segmented" if is_segmented else ""
  return f"splash_{attention_type}_{phase}{segments}{residuals}"

# Reference attention implementations


def _attention_reference_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    segment_ids: SegmentIds | None,
    mask_value: float,
    save_residuals: bool,
    attn_logits_soft_cap: float | None,
) -> SplashCustomReturnType:
  logits = jnp.einsum("sd,td->st", q.astype(jnp.float32), k.astype(jnp.float32))

  if segment_ids is not None:
    mask = jnp.logical_and(
        mask, segment_ids.q[:, None] == segment_ids.kv[None, :]
    )

  if attn_logits_soft_cap is not None:
    logits = jnp.tanh(logits / attn_logits_soft_cap)
    logits = logits * attn_logits_soft_cap

  logits = jnp.where(mask, logits, mask_value)
  m = logits.max(axis=-1)
  s = jnp.exp(logits - m[..., None])
  l = s.sum(axis=-1)
  p = s / l[..., None]

  o = jnp.einsum("st,td->sd", p, v.astype(jnp.float32))

  if save_residuals:
    logsumexp = m + jnp.log(l)
    return o, (logsumexp, m)
  return o


def _attention_reference_custom_bwd(
    do,
    q,
    k,
    v,
    mask,
    segment_ids,
    o,
    logsumexp,
    mask_value: float = DEFAULT_MASK_VALUE,
    backward_impl: str = "vanilla",
    attn_logits_soft_cap: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, None, None]:
  uncapped_logits = jnp.einsum(
      "qc,kc->qk", q, k, preferred_element_type=jnp.float32
  )

  if attn_logits_soft_cap is not None:
    logits = jnp.tanh(uncapped_logits / attn_logits_soft_cap)
    logits = logits * attn_logits_soft_cap
  else:
    logits = uncapped_logits

  if segment_ids is not None:
    mask = jnp.logical_and(
        mask, segment_ids.q[:, None] == segment_ids.kv[None, :]
    )
  logits = jnp.where(mask, logits, mask_value)

  p = jnp.exp(logits - logsumexp[..., None])
  do = do.astype(jnp.float32)  # pytype: disable=attribute-error
  dv = jnp.einsum("pt,pd->td", p, do).astype(v.dtype)
  dp = jnp.einsum("pd,td->pt", do, v.astype(jnp.float32))

  # These two ways of computing ds are mathematically equivalent. The first
  # involves reducing over the head_dim dimension and the second involves
  # reducing over a sequence dimension. They tend to produce slightly different
  # numerics.
  if backward_impl == "flash":
    di = jnp.sum(o.astype(jnp.float32) * do, axis=-1)[..., None]
  else:
    di = jnp.einsum("st,st->s", dp, p)[:, None]
  ds = (dp - di) * p
  if attn_logits_soft_cap is not None:
    normalized = uncapped_logits / attn_logits_soft_cap
    d = jnp.tanh(normalized)
    g = ds * (1 - d)
    ds = g + g * d
  dk = jnp.einsum("sd,st->td", q.astype(jnp.float32), ds).astype(k.dtype)
  dq = jnp.einsum("st,td->sd", ds, k.astype(jnp.float32)).astype(q.dtype)
  return dq, dk, dv, None, None


@partial(
    jax.jit,
    static_argnames=[
        "mask_value",
        "save_residuals",
        "attn_logits_soft_cap",
        "is_mqa",
    ],
)
def attention_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    segment_ids: SegmentIds | None = None,
    *,
    is_mqa: bool,
    mask_value: float = DEFAULT_MASK_VALUE,
    save_residuals: bool = False,
    attn_logits_soft_cap: float | None = None,
):
  """A JIT-compiled reference implementation of attention, handles MQA and MHA."""
  attn_impl = partial(
      _attention_reference_impl,
      mask_value=mask_value,
      save_residuals=save_residuals,
      attn_logits_soft_cap=attn_logits_soft_cap,
  )

  if is_mqa:
    func = jax.vmap(attn_impl, in_axes=(0, None, None, None, None))
  else:
    # In grouped attention (1 < num_kv_heads && num_kv_heads < num_q_heads).
    # We interleave the KV heads across the Q heads.
    # For example: for 8 Q heads and 4 KV heads:
    # Q head [0, 1] see KV head 0
    # Q head [2, 3] see KV head 1
    # Q head [4, 5] see KV head 2
    # Q head [6, 7] see KV head 3

    kv_heads, q_heads = k.shape[0], q.shape[0]
    assert q_heads % kv_heads == 0

    if kv_heads < q_heads:
      # Repeat K and V heads to match the number of Q heads.
      q_heads_per_kv = q_heads // kv_heads
      k = jnp.repeat(k, repeats=q_heads_per_kv, axis=0)
      v = jnp.repeat(v, repeats=q_heads_per_kv, axis=0)

    func = jax.vmap(attn_impl, in_axes=(0, 0, 0, None, None))

  out = func(q, k, v, mask, segment_ids)
  return out


# Splash attention implementation

# We use an IntEnum to make it JSON serializable as regen metadata.
class QKVLayout(enum.IntEnum):
  HEAD_DIM_MINOR = enum.auto()  # [..., seq_len, head_dim]
  SEQ_MINOR = enum.auto()  # [..., head_dim, seq_len]


def from_head_minor(vals: tuple[Any, ...], layout: QKVLayout):
  if layout == QKVLayout.HEAD_DIM_MINOR:
    return vals
  return (*vals[:-2], vals[-1], vals[-2])


@dataclasses.dataclass(frozen=True, slots=True)
class SplashConfig:
  """Tile sizes parameterizing SplashAttention kernels.

  Those parameters have negligible effect on numerics, but affect performance
  greatly.

  Note that changing the layouts only influences the physical layout that the
  kernel will enforce. The logical interface to splash attention always takes
  the head dimension as the minormost one.
  """
  block_q: int
  block_kv: int
  block_kv_compute: int | None = None

  block_q_dkv: int | None = None
  block_kv_dkv: int | None = None
  block_kv_dkv_compute: int | None = None

  block_q_dq: int | None = None
  block_kv_dq: int | None = None

  use_fused_bwd_kernel: bool = False

  q_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  k_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  v_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR

  residual_checkpoint_name: str | None = None  # whether to checkpoint outputs
  attn_logits_soft_cap: float | None = None
  fuse_reciprocal: bool = True  # whether to compute o / lse inside the kernel
  use_base2_exp: bool = True
  max_logit_const: float | None = None
  interpret: bool = False

  def __post_init__(self):
    if self.block_kv_compute is None:
      object.__setattr__(self, "block_kv_compute", self.block_kv)
    if self.block_kv_dkv_compute is None:
      object.__setattr__(self, "block_kv_dkv_compute", self.block_kv_dkv)
    if self.use_fused_bwd_kernel:
      if self.block_q_dq is not None or self.block_kv_dq is not None:
        raise ValueError(
            "Block sizes for dq kernel are not needed with a fused kernel."
        )

  @property
  def has_backward_blocks(self) -> bool:
    backward_blocks = (
        self.block_q_dkv, self.block_kv_dkv, self.block_kv_dkv_compute,
    )
    if not self.use_fused_bwd_kernel:
      backward_blocks += (self.block_q_dq, self.block_kv_dq)
    return all(b is not None for b in backward_blocks)

  @classmethod
  def get_default(cls):
    # TODO: Select better parameters based on a heuristic.
    return SplashConfig(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        block_q_dq=128,
        block_kv_dq=128,
        fuse_reciprocal=True,
    )

to_i32 = lambda x: x.astype(jnp.int32)


def _unravel(f, grid_width, transposed_grid=False):
  """Creates a Pallas index_map for a 2D grid of blocks.

  Wraps a function `f` to map a 1D grid index to 2D `(i, j)` block indices.
  The mapping is dense if metadata is provided, otherwise it's a sparse
  lookup.

  Args:
    f: Function to call with computed indices: `f(h, i, j, refs)`.
    grid_width: The logical grid width for dense mapping.
    transposed_grid: If True, swaps `i` and `j` indices before calling `f`.
  """
  def index_map(h, grid_idx, rows_ref, cols_ref, *refs):
    if rows_ref is None:
      assert cols_ref is None
      i = grid_idx // grid_width
      j = grid_idx % grid_width
    else:
      i = to_i32(rows_ref[grid_idx])
      j = to_i32(cols_ref[grid_idx])

    if transposed_grid:
      i, j = j, i

    return f(h, i, j, refs)
  return index_map


def _apply_mask_and_soft_cap(
    qk: jax.Array,
    mask_value: float,
    mask_ref,
    q_sequence_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    *,
    attn_logits_soft_cap: float | None,
    k_slice: pl.Slice,
    k_offset: int | jax.Array,
    bq: int,
    k_in_lanes=True,
    mask_function=None,
    has_partial_mask: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  assert mask_ref is None or q_sequence_ref is None
  assert (q_sequence_ref is None) == (mask_function is None)

  masks = []
  if has_partial_mask:
    if mask_ref is not None:
      mask = mask_ref[:, k_slice] if k_in_lanes else mask_ref[k_slice, :]
      masks.append(mask)
    elif mask_function is not None:
      # Compute the mask using the given q_sequence indices.
      # KV indices are computed on the fly. This works because we only support Q
      # sequence sharding. If we wanted to compute Q indices too, then we would
      # need to keep into account the current shard along Q sequence.

      if k_in_lanes:
        assert q_sequence_ref.shape == (bq, NUM_LANES)

        k_sequence = k_offset + jax.lax.broadcasted_iota(
            jnp.int32, (bq, k_slice.size), 1
        )

        repeats, rem = divmod(k_slice.size, NUM_LANES)
        assert rem == 0
        q_sequence = pltpu.repeat(
            q_sequence_ref[...], repeats, axis=1
        )  # [bq, k_slice.size]
      else:
        assert q_sequence_ref.shape == (NUM_SUBLANES, bq)

        k_sequence = k_offset + jax.lax.broadcasted_iota(
            jnp.int32, (k_slice.size, bq), 0
        )
        q_sequence = q_sequence_ref[:1, :]  # [1, bq]
        q_sequence = jnp.broadcast_to(q_sequence, (k_slice.size, bq))

      assert q_sequence.shape == k_sequence.shape
      computed_mask = mask_function(q_sequence, k_sequence)  # pytype: disable=wrong-arg-count
      if computed_mask.dtype != jnp.dtype(jnp.bool_):
        raise ValueError(
            "Mask function must return a boolean-valued array, but got:"
            f" {computed_mask.dtype}"
        )
      masks.append(computed_mask)

  if q_segment_ids_ref is not None:
    if k_in_lanes:
      kv_ids = kv_segment_ids_ref[:1, k_slice]  # [1, k_slice]
      repeats, rem = divmod(kv_ids.shape[1], NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_kv must be a multiple of {NUM_LANES}")
      q_ids = pltpu.repeat(q_segment_ids_ref[:], repeats, axis=1)  # [bq, bkv]
    else:
      assert bq == q_segment_ids_ref.shape[-1]
      repeats, rem = divmod(bq, NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_q must be a multiple of {NUM_LANES}")
      kv_ids = pltpu.repeat(
          kv_segment_ids_ref[k_slice, :], repeats, axis=1
      )  # [k_slice, bq]
      q_ids = q_segment_ids_ref[:1, :]  # [1, bq]
    masks.append(q_ids == kv_ids)

  def cap_logits(logits):
    if attn_logits_soft_cap is not None:
      logits = jnp.tanh(qk / attn_logits_soft_cap)
      return logits * attn_logits_soft_cap
    else:
      return logits

  if masks:
    mask = functools.reduce(jnp.logical_and, masks)
    qk = cap_logits(qk)
    qk = jnp.where(mask, qk, mask_value)
  else:
    qk = cap_logits(qk)
  return qk


def flash_attention_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    mask_ref,
    q_sequence_ref,
    max_logit_value_ref,
    # Outputs
    o_ref,
    logsumexp_ref,
    max_logits_ref,
    # Scratch
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    *,
    mask_value: float,
    kv_steps: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    mask_function: MaskFunctionType | None,
    fuse_reciprocal: bool,  # config.fuse_reciprocal or not save_residuals
    config: SplashConfig,
):
  del mask_next_ref, active_rows_ref
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  head_dim_v_repeats, rem = divmod(head_dim_v, NUM_LANES)
  if rem != 0:
    raise NotImplementedError(
        f"{head_dim_v=} should be a multiple of {NUM_LANES}"
    )

  grid_idx = pl.program_id(1)
  h = pl.program_id(0)

  if block_mask_ref is not None:
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
    j = active_cols_ref[grid_idx].astype(jnp.int32)
  else:
    should_not_mask = False
    j = grid_idx % kv_steps
    should_initialize = j == 0
    should_write = j == kv_steps - 1

  max_logit_estimate = config.max_logit_const  # potentially None
  if max_logit_value_ref is not None:  # already ensures max_logit_const is None
    max_logit_estimate = max_logit_value_ref[0, h]

  @pl.when(should_initialize)
  def init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    if max_logit_estimate is None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
    else:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, max_logit_estimate)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

  def body(kv_compute_index, _, has_partial_mask=False):
    slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    assert m_prev.shape == (bq, NUM_LANES)
    assert l_prev.shape == (bq, NUM_LANES)

    q = q_ref[...] if config.q_layout == HEAD_DIM_MINOR else q_ref[...].T
    if config.use_base2_exp:
      q *= LOG2E

    qk_dims = (
        NT_DIM_NUMBERS if config.k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    )
    if config.k_layout == HEAD_DIM_MINOR:
      k = k_ref[slice_k, :]
    else:
      k = k_ref[:, slice_k]
    qk = lax.dot_general(q, k, qk_dims, preferred_element_type=float32)

    assert qk.shape == (bq, bkv_compute)
    apply_mask_and_soft_cap = functools.partial(
        _apply_mask_and_soft_cap,
        qk,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        k_offset=j * bkv + kv_compute_index * bkv_compute,
        bq=bq,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask
    )

    qk = apply_mask_and_soft_cap()

    if max_logit_estimate is None:
      m_curr = qk.max(axis=-1)[:, None]  # pytype: disable=attribute-error
      assert m_curr.shape == (bq, 1)
      m_next = jnp.maximum(m_prev, m_curr)
      assert m_next.shape == (bq, NUM_LANES)
    else:
      m_next = None

    bkv_repeats, rem = divmod(bkv_compute, NUM_LANES)
    if rem != 0:
      raise NotImplementedError(
          f"{bkv_compute=} should be a multiple of {NUM_LANES}"
      )

    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    if max_logit_estimate is None:
      s_curr = exp(qk - pltpu.repeat(m_next, bkv_repeats, axis=1))
    else:
      s_curr = exp(qk - max_logit_estimate)
    assert s_curr.shape == (bq, bkv_compute)

    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
    assert l_curr.shape == (bq, NUM_LANES)

    if max_logit_estimate is None:
      alpha = exp(m_prev - m_next)
      l_next = l_curr + alpha * l_prev
      m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next
    else:
      alpha = None
      l_scratch_ref[...] = l_curr + l_prev

    sv_dims = (
        NN_DIM_NUMBERS if config.v_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    )
    if config.v_layout == HEAD_DIM_MINOR:
      v = v_ref[slice_k, :]
    else:
      v = v_ref[:, slice_k]
    v = v.astype(float32)
    o_curr = lax.dot_general(s_curr, v, sv_dims)

    if max_logit_estimate is None:
      alpha_o = pltpu.repeat(alpha, head_dim_v_repeats, axis=1)
      o_scratch_ref[...] = alpha_o * o_scratch_ref[...] + o_curr
    else:
      o_scratch_ref[...] = o_scratch_ref[...] + o_curr

  assert bkv % bkv_compute == 0
  num_iters = (
      k_ref.shape[0 if config.k_layout == HEAD_DIM_MINOR else 1] // bkv_compute
  )

  @pl.when(should_not_mask)
  def _():
    lax.fori_loop(0, num_iters, body, None, unroll=True)

  @pl.when(~should_not_mask)
  def _():
    lax.fori_loop(
        0, num_iters, partial(body, has_partial_mask=True), None, unroll=True
    )
  @pl.when(should_write)
  def end():
    l = l_scratch_ref[...]
    if fuse_reciprocal:  # allows fusing reciprocal out of the kernel
      l_inv = pltpu.repeat(1.0 / l, head_dim_v_repeats, axis=1)
      o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)
    else:
      o_ref[...] = o_scratch_ref[...].astype(o_ref.dtype)
    if logsumexp_ref is not None:
      assert logsumexp_ref.shape == (bq, NUM_LANES)
      log = jnp.log2 if config.use_base2_exp else jnp.log
      logsumexp = m_scratch_ref[...] + log(l)
      logsumexp_ref[...] = logsumexp.astype(logsumexp_ref.dtype)
    if max_logits_ref is not None:
      assert max_logits_ref.shape == (bq, NUM_LANES)
      max_logits_ref[...] = m_scratch_ref[...].astype(max_logits_ref.dtype)

    m_scratch_ref[...] = jnp.zeros_like(m_scratch_ref)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)


def _div(dividend: int, divisor: int):
  if divisor == 1:
    return dividend

  return lax.div(dividend, divisor)


def _splash_attention_forward(
    fwd_mask_info: MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    save_residuals: bool,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    max_logit_value: jax.Array | None = None,
) -> SplashCustomReturnType:
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  bq, bkv = config.block_q, config.block_kv
  bkv_compute = config.block_kv_compute
  bounds_start, bounds_end = mask_info_lib.find_bounds(
      fwd_mask_info.active_rows
  )
  fuse_reciprocal = config.fuse_reciprocal or not save_residuals

  if is_mqa:
    expected_kv_rank = 2
    num_kv_heads = 1
  else:
    expected_kv_rank = 3
    num_kv_heads = k.shape[0]

  if len(k.shape) != expected_kv_rank:
    raise ValueError(
        f"Expected {expected_kv_rank}-dim 'key' tensor for MQA. Instead got a"
        f" {len(k.shape)}-dim one."
    )

  if k.shape[-1] != head_dim_qk:
    raise ValueError(
        f"Expected 'key' head dimension to be: {head_dim_qk}. Instead got:"
        f" {k.shape[-1]}."
    )

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
        f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  if k.shape[:-1] != v.shape[:-1]:
    raise ValueError(
        f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same "
        "leading dimensions."
    )

  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
  if bkv_compute % NUM_LANES:
    raise ValueError(f"{bkv_compute=} must be a multiple of {NUM_LANES}.")

  kv_seq_len = k.shape[-2]
  kv_steps = kv_seq_len // bkv
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  if segment_ids is not None:
    if segment_ids.q.shape != (q_seq_len,):
      raise ValueError(
          "Invalid shape for q segment_ids: "
          f"{segment_ids.q.shape}. Expected: {(q_seq_len,)}"
      )
    if segment_ids.kv.shape != (kv_seq_len,):
      raise ValueError(
          "Invalid shape for kv segment_ids: "
          f"{segment_ids.kv.shape}. Expected: {(kv_seq_len,)}"
      )
  if config.max_logit_const is not None and max_logit_value is not None:
    raise ValueError(f"Only one of {config.max_logit_const=} and"
                     f" {max_logit_value=} can be set.")
  if max_logit_value is not None:
    if max_logit_value.shape not in ((), (1,), (num_q_heads,)):
      raise ValueError(
          "max_logit_value should be a 0,1-dim jax.Array of shape (), (1,) or"
          f" ({num_q_heads=},) but got {jax.typeof(max_logit_value)}"
      )
    max_logit_value = jnp.broadcast_to(jnp.atleast_1d(max_logit_value),
                                       (num_q_heads,))

  q_layout = config.q_layout
  k_layout = config.k_layout
  v_layout = config.v_layout

  unravel = partial(_unravel, grid_width=kv_steps)

  def create_kv_index_map(layout):
    def index_map(h, i, j, *_):
      del i  # Unused.
      prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
      return from_head_minor((*prefix, j, 0), layout)
    return index_map

  q_index_map = unravel(
      lambda h, i, j, *_: from_head_minor((h, i, 0), q_layout)
  )
  out_index_map = unravel(lambda h, i, j, *_: (h, i, 0))
  k_index_map = unravel(create_kv_index_map(k_layout))
  v_index_map = unravel(create_kv_index_map(v_layout))

  def mask_index_map(h, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_):
    del h, rows_ref, cols_ref  # Unused.
    next_m = to_i32(mask_next_ref[grid_idx])
    return next_m, 0, 0

  q_segment_ids_index_map = unravel(lambda h, i, j, *_: (i, 0))
  kv_segment_ids_index_map = unravel(lambda h, i, j, *_: (0, j))

  # Convert the logical shape from head-minor to sequence-minor.
  in_specs = [
      pl.BlockSpec(
          from_head_minor((None, bq, head_dim_qk), q_layout), q_index_map
      ),
      pl.BlockSpec(
          from_head_minor(
              (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk), k_layout
          ),
          k_index_map,
      ),
      pl.BlockSpec(
          from_head_minor(
              (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v), v_layout
          ),
          v_index_map,
      ),
  ]
  if segment_ids is not None:
    in_specs += [
        pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map),
        pl.BlockSpec((NUM_SUBLANES, bkv), kv_segment_ids_index_map),
    ]
    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q, (q_seq_len, NUM_LANES), (0,)
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv, (NUM_SUBLANES, kv_seq_len), (1,)
    )
  else:
    in_specs += [None, None]
    q_segment_ids = kv_segment_ids = None

  if fwd_mask_info.partial_mask_blocks is not None:
    in_specs.append(pl.BlockSpec((None, bq, bkv), mask_index_map))
  else:
    in_specs.append(None)

  assert (
      fwd_mask_info.partial_mask_blocks is None
      or fwd_mask_info.q_sequence is None
  )

  if fwd_mask_info.q_sequence is not None:
    q_sequence = jax.lax.broadcast_in_dim(
        fwd_mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,)
    )
    in_specs.append(pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map))
  else:
    q_sequence = None
    in_specs.append(None)

  if max_logit_value is not None:
    # reshape to allow sublane selection for vmap-ping and shard_map-ping
    max_logit_value = jnp.broadcast_to(
        max_logit_value.astype(jnp.float32)[None, :],
        (NUM_SUBLANES, num_q_heads),
    )
    in_specs += [(pl.BlockSpec((NUM_SUBLANES, num_q_heads), lambda *_: (0, 0),
                               memory_space=pltpu.SMEM))]
  else:
    in_specs.append(None)

  out_shapes = [
      jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim_v), q.dtype),
  ]
  out_specs = [
      pl.BlockSpec((None, bq, head_dim_v), out_index_map),
  ]
  if save_residuals:
    out_shapes += [
        # logsumexp
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32),
        # max_logits
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32),
    ]

    logsumexp_and_max_logits_index_map = unravel(lambda h, i, j, *_: (h, i, 0))

    out_specs += [
        pl.BlockSpec((None, bq, NUM_LANES), logsumexp_and_max_logits_index_map),
        pl.BlockSpec((None, bq, NUM_LANES), logsumexp_and_max_logits_index_map),
    ]
  else:
    out_shapes += [None, None]
    out_specs += [None, None]

  kernel_name = get_kernel_name(
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      is_segmented=segment_ids is not None,
      phase="fwd",
  )
  metadata = {"xprof_metadata": json.dumps(dataclasses.asdict(config))}

  def _bytes(x: jax.Array | jax.ShapeDtypeStruct | None) -> int:
    if x is None:
      return 0

    if jnp.issubdtype(x.dtype, jnp.floating):
      info = jnp.finfo
    elif jnp.issubdtype(x.dtype, jnp.integer):
      info = jnp.iinfo
    else:
      raise ValueError(f"Unsupported dtype: {x.dtype}")
    return math.ceil(math.prod(x.shape) * info(x.dtype).bits / 8)

  def _fwd_cost_estimate(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      q_segment_ids: jax.Array | None,
      kv_segment_ids: jax.Array | None,
      partial_mask_blocks: jax.Array | None,
      out_shapes: list[jax.ShapeDtypeStruct],
      mask_sparsity: float,
  ) -> pl.CostEstimate | None:
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    kv_seq_len, head_dim_v = v.shape[-2:]

    matmul_flops = (
        2 * q_seq_len * kv_seq_len * head_dim_qk
        + 2 * kv_seq_len * kv_seq_len * head_dim_v
    )

    # This is an upper bound because `mask_sparsity` is actually the mean
    # sparsity of the non-fully masked **blocks**.
    total_flops = num_q_heads * matmul_flops * mask_sparsity

    # Count expensive exp() calls
    transcendentals = num_q_heads * q_seq_len * kv_seq_len

    inputs_ = [q, k, v, q_segment_ids, kv_segment_ids, partial_mask_blocks]
    input_bytes = sum(map(_bytes, inputs_))
    output_bytes = sum(map(_bytes, out_shapes))
    return pl.CostEstimate(
        flops=int(total_flops),
        transcendentals=int(transcendentals),
        bytes_accessed=int(input_bytes + output_bytes),
    )

  vmem_inputs = [
      q,
      k,
      v,
      q_segment_ids,
      kv_segment_ids,
      fwd_mask_info.partial_mask_blocks,
  ]
  cost_estimate = _fwd_cost_estimate(*vmem_inputs, out_shapes, fwd_mask_sparsity)

  if fwd_mask_info.num_active_blocks is not None:
    grid_size = fwd_mask_info.num_active_blocks[0]
  else:
    grid_size = kv_steps * (q_seq_len // bq)

  grid = (num_q_heads, grid_size)

  with jax.named_scope(kernel_name):
    all_out = pl.pallas_call(
        partial(
            flash_attention_kernel,
            mask_value=mask_value,
            kv_steps=kv_steps,
            bq=bq,
            bkv=bkv,
            bkv_compute=bkv_compute,
            head_dim_v=head_dim_v,
            # note: fuse_reciprocal can only be False if save_residuals is True
            # fuse_reciprocal = (config.fuse_reciprocal or not save_residuals)
            fuse_reciprocal=fuse_reciprocal,
            config=config,
            mask_function=mask_function,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=[
                pltpu.VMEM((bq, NUM_LANES), jnp.float32),  # m_scratch
                pltpu.VMEM((bq, NUM_LANES), jnp.float32),  # l_scratch
                pltpu.VMEM((bq, head_dim_v), jnp.float32),  # o_scratch
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
        ),
        out_shape=out_shapes,
        name=kernel_name,
        cost_estimate=cost_estimate,
        interpret=config.interpret,
        metadata=metadata,
    )(
        fwd_mask_info.active_rows,
        fwd_mask_info.active_cols,
        fwd_mask_info.mask_next,
        bounds_start,
        bounds_end,
        fwd_mask_info.block_mask,
        q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.swapaxes(-1, -2),
        k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.swapaxes(-1, -2),
        v if v_layout == QKVLayout.HEAD_DIM_MINOR else v.swapaxes(-1, -2),
        q_segment_ids,
        kv_segment_ids,
        fwd_mask_info.partial_mask_blocks,
        q_sequence,
        max_logit_value,
    )
  out, logsumexp, max_logits = all_out
  if not fuse_reciprocal:
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    l = exp(logsumexp[..., 0] - max_logits[..., 0])
    out = (out / l[..., None]).astype(out.dtype)

  if save_residuals:
    assert logsumexp is not None and max_logits is not None
    logsumexp, max_logits = logsumexp[..., 0], max_logits[..., 0]

  if config.residual_checkpoint_name is not None:
    out = ad_checkpoint.checkpoint_name(
        out, name=config.residual_checkpoint_name
    )
    if logsumexp is not None:
      logsumexp = ad_checkpoint.checkpoint_name(
          logsumexp, name=config.residual_checkpoint_name
      )
  if save_residuals:
    return out, (logsumexp, max_logits)
  return out


@partial(jax.custom_vjp, nondiff_argnames=(
    "save_residuals",
    "mask_value",
    "is_mqa",
    "config",
    "mask_function",
    "fwd_mask_sparsity",
))
def _splash_attention_custom(
    fwd_mask_info: MaskInfo,
    dq_mask_info: MaskInfo | None,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    max_logit_value: jax.Array | None = None,
) -> SplashCustomReturnType:
  # The forward function does not use the dq and dkv MaskInfos, it just forwards
  # them to the backward function as residuals. This is a way to communicate
  # arbitrary Arrays to the backward function. Since the three MaskInfos are
  # constants there is no overhead in passing them to the backward function as
  # residuals. When sharding computation MaskInfos are partitioned so both the
  # forward and the backward kernels need to work on the relevant slice. If we
  # recomputed the backward MaskInfos in the backward function from the numpy
  # mask then we would not work with the MaskInfo slice relevant to the current
  # device.
  del dq_mask_info, dkv_mask_info

  return _splash_attention_forward(  # pytype: disable=wrong-arg-types
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      save_residuals=save_residuals,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      max_logit_value=max_logit_value,
  )


def _splash_attention_fwd(
    fwd_mask_info: MaskInfo,
    dq_mask_info: MaskInfo | None,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    max_logit_value: jax.Array | None = None,
) -> tuple[tuple[jax.Array], SplashResidualsType]:
  if save_residuals:
    raise NotImplementedError("Higher-order AD not supported.")

  out, (logsumexp, max_logits) = _splash_attention_forward(  # pytype: disable=wrong-arg-types
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      save_residuals=True,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      max_logit_value=max_logit_value,
  )
  del max_logits
  return out, (
      q,
      k,
      v,
      segment_ids,
      out,
      logsumexp,
      dq_mask_info,
      dkv_mask_info,
  )


def _flash_attention_dq_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    # Outputs
    dq_scratch_ref,
    dq_ref,
    *,
    mask_value: float,
    kv_steps: int,
    bq: int,
    bkv: int,
    mask_function: MaskFunctionType | None,
    config: SplashConfig,
):
  del mask_next_ref, active_rows_ref
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  grid_idx = pl.program_id(1)
  if block_mask_ref is not None:
    kv_index = active_cols_ref[grid_idx].astype(jnp.int32)
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
  else:
    kv_index = grid_idx % kv_steps
    should_not_mask = False
    should_initialize = kv_index == 0
    should_write = kv_index == kv_steps - 1

  @pl.when(should_initialize)
  def init():
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

  def body(has_partial_mask: bool = False):
    q = q_ref[...] if config.q_layout == HEAD_DIM_MINOR else q_ref[...].T
    if config.use_base2_exp:
      q *= LOG2E
    # We keep k and v possibly transposed, since they are RHS of dots.
    k = k_ref[...]
    v = v_ref[...]
    logsumexp = jnp.expand_dims(logsumexp_ref[0], -1)
    do = do_ref[...]
    di = jnp.expand_dims(di_ref[0], -1)

    qk_dims = (
        NT_DIM_NUMBERS if config.k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    )
    qk_uncapped = lax.dot_general(q, k, qk_dims, preferred_element_type=float32)

    qk = _apply_mask_and_soft_cap(
        qk_uncapped,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=pl.ds(0, bkv),
        k_offset=kv_index * bkv,
        bq=bq,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    p = exp(qk - logsumexp)
    dp_dims = (
        NT_DIM_NUMBERS if config.v_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    )
    dp = lax.dot_general(
        do.astype(v.dtype), v, dp_dims, preferred_element_type=jnp.float32,
    )
    ds = (dp - di) * p
    if attn_logits_soft_cap is not None:
      normalized = qk_uncapped / attn_logits_soft_cap
      d = jnp.tanh(normalized)
      ds = ds * (1 - d * d)

    dq_dims = (
        NN_DIM_NUMBERS if config.k_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    )
    dq_scratch_ref[...] += lax.dot_general(
        ds.astype(k.dtype), k, dq_dims,
        preferred_element_type=jnp.float32,
    )

  @pl.when(should_not_mask)
  def _():
    body()

  @pl.when(~should_not_mask)
  def _():
    body(has_partial_mask=True)

  @pl.when(should_write)
  def end():
    dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)


def _splash_attention_bwd_dq(
    q,
    k,
    v,
    segment_ids,
    logsumexp,
    do,
    di,
    *,
    bq: int,
    bkv: int,
    is_mqa: bool,
    mask_info: MaskInfo,
    mask_value: float,
    mask_function: MaskFunctionType | None,
    config: SplashConfig,
):
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  kv_seq_len, head_dim_v = v.shape[-2:]
  num_kv_heads = 1 if is_mqa else k.shape[0]
  bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)

  if bq > q_seq_len:
    raise ValueError(f"{bq=} should not be greater than {q_seq_len=}")
  if bkv > kv_seq_len:
    raise ValueError(f"{bkv=} should not be greater than {kv_seq_len=}")

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
        f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  if k.shape[:-1] != v.shape[:-1]:
    raise ValueError(
        f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same "
        "leading dimensions."
    )

  if bkv % NUM_LANES:
    raise ValueError(f"{bkv=} must be a multiple of {NUM_LANES}.")

  # TODO: when adding block_compute, make sure that is a
  # multiple of NUM_LANES.

  kv_steps = kv_seq_len // bkv
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  unravel = partial(_unravel, grid_width=kv_steps)

  def create_kv_index_map(layout):
    def index_map(h, i, j, *_):
      del i  # Unused.
      prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
      return from_head_minor((*prefix, j, 0), layout)
    return index_map

  q_index_map = unravel(
      lambda h, i, j, *_: from_head_minor((h, i, 0), config.q_layout)
  )
  out_index_map = unravel(lambda h, i, j, *_: (h, i, 0))
  k_index_map = unravel(create_kv_index_map(config.k_layout))
  v_index_map = unravel(create_kv_index_map(config.v_layout))

  def mask_index_map(h, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_):
    del h, rows_ref, cols_ref  # Unused.
    next_m = to_i32(mask_next_ref[grid_idx])
    return next_m, 0, 0

  q_segment_ids_index_map = unravel(lambda h, i, j, *_: (i, 0))

  o_spec = pl.BlockSpec((None, bq, head_dim_v), out_index_map)

  q_spec = pl.BlockSpec(
      from_head_minor((None, bq, head_dim_qk), config.q_layout), q_index_map
  )

  k_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
          config.k_layout
      ),
      k_index_map,
  )
  v_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v),
          config.v_layout
      ),
      v_index_map,
  )

  mask_spec = pl.BlockSpec((None, bq, bkv), mask_index_map)

  if segment_ids is not None:
    kv_segment_ids_index_map = unravel(lambda h, i, j, *_: (0, j))
    q_segment_spec = pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map)
    kv_segment_spec = pl.BlockSpec(
        (NUM_SUBLANES, bkv), kv_segment_ids_index_map
    )
    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q, (q_seq_len, NUM_LANES), (0,)
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv, (NUM_SUBLANES, kv_seq_len), (1,)
    )
  else:
    q_segment_spec = kv_segment_spec = None
    q_segment_ids = kv_segment_ids = None

  do_spec = o_spec

  logsumexp_index_map = unravel(lambda h, i, j, *_: (h, 0, i))
  logsumexp = jnp.expand_dims(logsumexp, axis=-2)
  logsumexp_spec = pl.BlockSpec((None, 1, bq), logsumexp_index_map)
  assert logsumexp.ndim == len(logsumexp_spec.block_shape)

  di = jnp.expand_dims(di, axis=-2)
  di_spec = pl.BlockSpec((None, 1, bq), logsumexp_index_map)
  assert di.ndim == len(di_spec.block_shape)

  in_specs = [
      q_spec,
      k_spec,
      v_spec,
      q_segment_spec,
      kv_segment_spec,
      logsumexp_spec,
      do_spec,
      di_spec,
  ]
  if mask_info.partial_mask_blocks is not None:
    in_specs.append(mask_spec)
  else:
    in_specs.append(None)

  assert mask_info.partial_mask_blocks is None or mask_info.q_sequence is None

  if mask_info.q_sequence is not None:
    q_sequence = jax.lax.broadcast_in_dim(
        mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,)
    )
    in_specs.append(pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map))
  else:
    q_sequence = None
    in_specs.append(None)

  out_shapes = [
      jax.ShapeDtypeStruct((bq, head_dim_qk), jnp.float32),
      jax.ShapeDtypeStruct(q.shape, q.dtype),
  ]
  out_specs = [
      pl.BlockSpec((bq, head_dim_qk), lambda *_: (0, 0)),
      pl.BlockSpec((None, bq, head_dim_qk), out_index_map),
  ]
  if mask_info.num_active_blocks is not None:
    grid_size = mask_info.num_active_blocks[0]
  else:
    grid_size = kv_steps * (q_seq_len // bq)

  grid = (num_q_heads, grid_size)

  kernel = functools.partial(
      _flash_attention_dq_kernel,
      kv_steps=kv_steps,
      mask_value=mask_value,
      bq=bq,
      bkv=bkv,
      mask_function=mask_function,
      config=config,
  )

  kernel_name = get_kernel_name(
      is_mqa=is_mqa,
      save_residuals=False,
      is_segmented=segment_ids is not None,
      phase="dq",
  )
  metadata = {
      "xprof_metadata": json.dumps(
          dict(
              block_q_dq=bq,
              block_kv_dq=bkv,
              q_layout=config.q_layout,
              k_layout=config.k_layout,
              v_layout=config.v_layout,
          )
      )
  }
  q_ = q if config.q_layout == QKVLayout.HEAD_DIM_MINOR else q.swapaxes(-1, -2)
  k_ = k if config.k_layout == QKVLayout.HEAD_DIM_MINOR else k.swapaxes(-1, -2)
  v_ = v if config.v_layout == QKVLayout.HEAD_DIM_MINOR else v.swapaxes(-1, -2)
  with jax.named_scope(kernel_name):
    _, dq = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
        ),
        out_shape=out_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
        ),
        name=kernel_name,
        interpret=config.interpret,
        metadata=metadata,
    )(
        mask_info.active_rows,
        mask_info.active_cols,
        mask_info.mask_next,
        bounds_start,
        bounds_end,
        mask_info.block_mask,
        q_,
        k_,
        v_,
        q_segment_ids,
        kv_segment_ids,
        logsumexp,
        do,
        di,
        mask_info.partial_mask_blocks,
        q_sequence,
    )
  return dq


def _flash_attention_dkv_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    # aliases
    dq_alias,
    dk_alias,
    dv_alias,
    # Outputs
    dq_ref,
    dk_ref,
    dv_ref,
    # Scratch
    dq_scratch_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    mask_value: float,
    q_steps: int,
    bq: int,
    bkv_compute: int,
    bkv: int,
    mask_function: MaskFunctionType | None,
    config: SplashConfig,
):
  del mask_next_ref, active_cols_ref
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E
  grid_idx = pl.program_id(1)

  if block_mask_ref is not None:
    kv_index = active_rows_ref[grid_idx].astype(jnp.int32)
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
  else:
    q_index = grid_idx % q_steps
    kv_index = grid_idx // q_steps
    should_not_mask = False
    should_initialize = q_index == 0
    should_write = q_index == q_steps - 1

  # TODO: Update docstring explaining the accumulation logic

  # Consider this situation:
  # Q_heads:   0, 1, 2, 3, 4, 5, 6, 7
  # KV_heads:  0,    1,    2,    3
  # The gradient scratch buffers should be initialized for Q_heads 0, 2, 4, 6
  # (first Q_heads to 'see' a new KV_head).
  # The gradient output buffers should be written for Q_heads 1, 3, 5, 7 (last
  # Q_heads to 'see' the current KV_head).

  @pl.when(should_initialize)
  def init():
    dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
    dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)

  def body(i, _, has_partial_mask=False):

    slice_k = pl.ds(i * bkv_compute, bkv_compute)
    q = q_ref[...]  # We keep q potentially transposed, since it's always RHS
    if config.use_base2_exp:
      scaled_q = q * LOG2E
    else:
      scaled_q = q
    def _load_kv(ref, layout):
      if layout == HEAD_DIM_MINOR:
        return ref[slice_k, :]
      return ref[:, slice_k].T
    k = _load_kv(k_ref, config.k_layout)
    v = _load_kv(v_ref, config.v_layout)
    logsumexp = logsumexp_ref[:1, :]
    do = do_ref[...]
    di = di_ref[:1, :]

    qk_dims = (
        NT_DIM_NUMBERS if config.q_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    )
    qk_uncapped = lax.dot_general(
        k, scaled_q, qk_dims, preferred_element_type=jnp.float32
    )

    qk = _apply_mask_and_soft_cap(
        qk_uncapped,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        k_offset=kv_index * bkv + i * bkv_compute,
        bq=bq,
        k_in_lanes=False,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    p = exp(qk - logsumexp)
    dv = lax.dot(p.astype(do.dtype), do, preferred_element_type=jnp.float32)
    dv = dv.astype(dv_scratch_ref.dtype) + dv_scratch_ref[slice_k, :]
    dv_scratch_ref[slice_k, :] = dv

    dp = lax.dot_general(
        v, do, NT_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    ds = (dp - di) * p
    if attn_logits_soft_cap is not None:
      normalized = qk_uncapped / attn_logits_soft_cap
      d = jnp.tanh(normalized)
      ds = ds * (1 - d * d)
    dk_dims = (
        NN_DIM_NUMBERS if config.q_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    )
    dk = lax.dot_general(
        ds.astype(do.dtype), q, dk_dims, preferred_element_type=jnp.float32
    )
    dk = dk.astype(dk_scratch_ref.dtype) + dk_scratch_ref[slice_k, :]
    dk_scratch_ref[slice_k, :] = dk
    if dq_scratch_ref is not None or dq_ref is not None:
      dq = lax.dot_general(
          ds.T.astype(k.dtype), k, NN_DIM_NUMBERS,
          preferred_element_type=jnp.float32,
      )
      if dq_scratch_ref is not None:
        # Compute block size != memory block size
        dq_scratch_ref[...] += dq
      else:
        # Compute block size == memory block size
        assert dq_ref is not None
        dq_ref[...] = dq_alias[...] + dq.astype(dq_ref.dtype)

  if dq_scratch_ref is not None:
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

  num_iters = (
      k_ref.shape[0 if config.k_layout is HEAD_DIM_MINOR else 1] // bkv_compute
  )

  @pl.when(should_not_mask)
  def _():
    lax.fori_loop(0, num_iters, body, None, unroll=True)

  @pl.when(~should_not_mask)
  def _():
    lax.fori_loop(
        0, num_iters, partial(body, has_partial_mask=True), None, unroll=True
    )
  if dq_scratch_ref is not None:
    assert dq_ref is not None
    dq_ref[...] = dq_alias[...] + dq_scratch_ref[...].astype(dq_ref.dtype)

  @pl.when(should_write)
  def end():
    dk = dk_alias[...]
    dv = dv_alias[...]
    dk_ref[...] = dk + dk_scratch_ref[...].astype(dk_ref.dtype)
    dv_ref[...] = dv + dv_scratch_ref[...].astype(dv_ref.dtype)
    if dq_scratch_ref is not None:
      dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

    dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
    dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)


def _splash_attention_bwd_dkv(
    q,
    k,
    v,
    segment_ids,
    logsumexp,
    do,
    di,
    *,
    bq: int,
    bkv: int,
    bkv_compute: int,
    is_mqa: bool,
    mask_info: MaskInfo,
    mask_value: float,
    mask_function: MaskFunctionType | None,
    config: SplashConfig,
):
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  kv_seq_len, head_dim_v = v.shape[-2:]
  num_kv_heads = 1 if is_mqa else k.shape[0]

  bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)
  if bq > q_seq_len:
    raise ValueError(f"{bq=} should not be greater than {q_seq_len=}")
  if bkv > kv_seq_len:
    raise ValueError(f"{bkv=} should not be greater than {kv_seq_len=}")
  if bkv_compute > bkv:
    raise ValueError(f"{bkv_compute=} should not be greater than {bkv=}")
  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} should be a multiple of {bkv_compute=}")

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
        f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  if k.shape[:-1] != v.shape[:-1]:
    raise ValueError(
        f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same "
        "leading dimensions."
    )

  q_steps = q_seq_len // bq
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  unravel = partial(_unravel, grid_width=q_steps, transposed_grid=True)

  q_index_map = unravel(
      lambda h, i, j, *_: from_head_minor((h, i, 0), config.q_layout)
  )
  o_index_map = unravel(lambda h, i, j, *_: (h, i, 0))

  def create_kv_index_map(layout):
    def index_map(h, i, j, *_):
      del i  # Unused.
      prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
      return from_head_minor((*prefix, j, 0), layout)
    return index_map

  k_index_map = unravel(create_kv_index_map(config.k_layout))
  v_index_map = unravel(create_kv_index_map(config.v_layout))

  q_spec = pl.BlockSpec(
      from_head_minor((None, bq, head_dim_qk), config.q_layout), q_index_map
  )

  o_spec = pl.BlockSpec((None, bq, head_dim_v), o_index_map)
  k_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
          config.k_layout,
      ),
      k_index_map,
  )

  v_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v),
          config.v_layout,
      ),
      v_index_map,
  )

  def create_dkv_index_map(h, i, j, *_):
    del i  # Unused.
    prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
    return (*prefix, j, 0)

  dkv_index_map = unravel(create_dkv_index_map)

  dk_spec = pl.BlockSpec(
      (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
      dkv_index_map,
  )

  dv_spec = pl.BlockSpec(
      (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v),
      dkv_index_map,
  )

  def mask_index_map(h, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_):
    del h, rows_ref, cols_ref  # Unused.
    next_m = to_i32(mask_next_ref[grid_idx])
    return next_m, 0, 0
  mask_spec = pl.BlockSpec((None, bkv, bq), mask_index_map)

  q_segment_ids_index_map = unravel(lambda h, i, j, *_: (0, i))
  if segment_ids is not None:
    kv_segment_ids_index_map = unravel(lambda h, i, j, *_: (j, 0))

    q_segment_spec = pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map)
    kv_segment_spec = pl.BlockSpec((bkv, NUM_LANES), kv_segment_ids_index_map)
    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q, (NUM_SUBLANES, q_seq_len), (1,)
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv, (kv_seq_len, NUM_LANES), (0,)
    )
  else:
    q_segment_spec = kv_segment_spec = None
    q_segment_ids = kv_segment_ids = None

  do_spec = o_spec

  logsumexp_index_map = unravel(lambda h, i, j, *_: (h, 0, i))

  assert logsumexp.shape == di.shape == (num_q_heads, q_seq_len)
  # TODO: Remove the sublane expansion once Mosaic has all retilings
  logsumexp_shape = (num_q_heads, NUM_SUBLANES, q_seq_len)
  logsumexp = jnp.broadcast_to(jnp.expand_dims(logsumexp, -2), logsumexp_shape)
  logsumexp_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert logsumexp.ndim == len(logsumexp_spec.block_shape)

  # TODO: Remove the sublane expansion once Mosaic has all retilings
  di = jnp.broadcast_to(jnp.expand_dims(di, -2), logsumexp_shape)
  di_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert di.ndim == len(di_spec.block_shape)

  in_specs = [
      q_spec,
      k_spec,
      v_spec,
      q_segment_spec,
      kv_segment_spec,
      logsumexp_spec,
      do_spec,
      di_spec,
  ]
  if mask_info.partial_mask_blocks is not None:
    in_specs.append(mask_spec)
  else:
    in_specs.append(None)

  if mask_info.q_sequence is not None:
    in_specs.append(pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map))
    q_sequence = jax.lax.broadcast_in_dim(
        mask_info.q_sequence, (NUM_SUBLANES, q_seq_len), (1,)
    )
  else:
    q_sequence = None
    in_specs.append(None)

  if config.use_fused_bwd_kernel:
    dq_index_map = unravel(lambda h, i, j, *_: (j, h, i, 0))
    dq_spec = pl.BlockSpec((None, None, bq, head_dim_qk), dq_index_map)
    dq_shape = jax.ShapeDtypeStruct((kv_seq_len // bkv, *q.shape), jnp.float32)
    dq = jnp.zeros_like(dq_shape)
    if bkv == bkv_compute:
      dq_scratch = None
    else:
      dq_scratch = pltpu.VMEM((bq, head_dim_qk), jnp.float32)
  else:
    dq = dq_spec = dq_shape = dq_scratch = None

  in_specs += [dq_spec, dk_spec, dv_spec]
  out_shapes = [
      dq_shape,
      jax.ShapeDtypeStruct(k.shape, jnp.float32),
      jax.ShapeDtypeStruct(v.shape, jnp.float32),
  ]
  out_specs = [dq_spec, dk_spec, dv_spec]

  if mask_info.num_active_blocks is not None:
    grid_size = mask_info.num_active_blocks[0]
  else:
    grid_size = (kv_seq_len // bkv) * q_steps

  dk = jnp.zeros(k.shape, dtype=jnp.float32)
  dv = jnp.zeros(v.shape, dtype=jnp.float32)

  kernel = functools.partial(
      _flash_attention_dkv_kernel,
      mask_value=mask_value,
      q_steps=q_steps,
      bq=bq,
      bkv_compute=bkv_compute,
      config=config,
      bkv=bkv,
      mask_function=mask_function,
  )

  kernel_name = get_kernel_name(
      is_mqa=is_mqa,
      save_residuals=False,
      is_segmented=segment_ids is not None,
      phase="dkv",
  )
  metadata = {
      "xprof_metadata": json.dumps(
          dict(
              block_q_dkv=bq,
              block_kv_dkv=bkv,
              block_kv_dkv_compute=bkv_compute,
              q_layout=config.q_layout,
              k_layout=config.k_layout,
              v_layout=config.v_layout,
          ),
      )
  }
  args = [
      # scalar prefetch
      mask_info.active_rows,
      mask_info.active_cols,
      mask_info.mask_next,
      bounds_start,
      bounds_end,
      mask_info.block_mask,
      # inputs
      q if config.q_layout == QKVLayout.HEAD_DIM_MINOR else q.swapaxes(-1, -2),
      k if config.k_layout == QKVLayout.HEAD_DIM_MINOR else k.swapaxes(-1, -2),
      v if config.v_layout == QKVLayout.HEAD_DIM_MINOR else v.swapaxes(-1, -2),
      q_segment_ids,
      kv_segment_ids,
      logsumexp,
      do,
      di,
      mask_info.partial_mask_blocks,
      q_sequence,
  ]
  num_args = sum(1 for x in args if x is not None)
  if config.use_fused_bwd_kernel:
    input_output_aliases = {num_args + i: i for i in (0, 1, 2)}
  else:
    input_output_aliases = {num_args + i: i for i in (0, 1)}

  scratch_shapes = [
      dq_scratch,
      pltpu.VMEM((bkv, head_dim_qk), jnp.float32),
      pltpu.VMEM((bkv, head_dim_v), jnp.float32),
  ]

  grid = (num_q_heads, grid_size)
  with jax.named_scope(kernel_name):
    dq_unreduced, dk, dv = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        input_output_aliases=input_output_aliases,
        # We set all dimensions to arbitrary because:
        # 1) for heads, we are reducing over heads
        # 2) for kv_seq_len, the splash attention prefetch schedule assumes no
        #     megacore
        # 3) for q_seq_len, we are reducing over it to compute dkv
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
        ),
        name=kernel_name,
        interpret=config.interpret,
        metadata=metadata,
    )(*args, dq, dk, dv)
  if config.use_fused_bwd_kernel:
    assert dq_unreduced is not None
    dq = dq_unreduced.sum(axis=0)
    dq = dq.astype(q.dtype)
  else:
    assert dq_unreduced is None
    dq = None

  dk = dk.astype(k.dtype)
  dv = dv.astype(v.dtype)
  return dq, dk, dv


def _splash_attention_bwd(
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    res: SplashResidualsType,
    do: jax.Array,
) -> tuple[
    MaskInfo | None,  # fwd_mask_info
    MaskInfo | None,  # dq_mask_info
    MaskInfo | None,  # dvk_mask_info
    jax.Array,  # q
    jax.Array,  # k
    jax.Array,  # v
    SegmentIds | None,  # segment_ids
    jax.Array | None,   # max_logit_estimate
]:
  del save_residuals, fwd_mask_sparsity
  if not config.has_backward_blocks:
    raise ValueError("Need to specify backward blocks.")
  bq_dq, bkv_dq = config.block_q_dq, config.block_kv_dq
  bq_dkv, bkv_dkv_memory, bkv_dkv_compute = (
      config.block_q_dkv,
      config.block_kv_dkv,
      config.block_kv_dkv_compute,
  )
  use_fused_bwd_kernel = config.use_fused_bwd_kernel
  (
      q,
      k,
      v,
      segment_ids,
      o,
      logsumexp,
      dq_mask_info,
      dkv_mask_info,
  ) = res

  # di: [num_heads, q_seq_len]
  di = jnp.einsum("hsd,hsd->hs", o.astype(jnp.float32), do.astype(jnp.float32))  # pytype: disable=attribute-error
  dq, dk, dv = _splash_attention_bwd_dkv(
      q,
      k,
      v,
      segment_ids,
      logsumexp,
      do,
      di,
      bq=bq_dkv,
      bkv=bkv_dkv_memory,
      bkv_compute=bkv_dkv_compute,
      is_mqa=is_mqa,
      mask_info=dkv_mask_info,
      mask_value=mask_value,
      mask_function=mask_function,
      config=config,
  )
  if not use_fused_bwd_kernel:
    assert dq is None
    dq = _splash_attention_bwd_dq(
        q,
        k,
        v,
        segment_ids,
        logsumexp,
        do,
        di,
        bq=bq_dq,
        bkv=bkv_dq,
        is_mqa=is_mqa,
        mask_info=dq_mask_info,
        mask_value=mask_value,
        mask_function=mask_function,
        config=config,
    )
  # Match the signature of the fwd function.
  assert dq is not None
  return (
      None,  # fwd_mask_info
      None,  # dq_mask_info
      None,  # dvk_mak_info
      dq,  # q
      dk,  # k
      dv,  # v
      None,  # segment_ids
      None,  # max_logit_estimate
  )


_splash_attention_custom.defvjp(_splash_attention_fwd, _splash_attention_bwd)


@partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "config",
        "save_residuals",
        "mask_value",
        "mask_function",
        "fwd_mask_sparsity",
    ],
)
def _splash_attention(
    fwd_mask_info: MaskInfo,
    dq_mask_info: MaskInfo | None,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None = None,
    *,
    is_mqa: bool,
    config: SplashConfig | None,
    save_residuals: bool,
    mask_value: float,
    max_logit_value: jax.Array | None = None,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
) -> SplashCustomReturnType:
  return _splash_attention_custom(
      fwd_mask_info,
      dq_mask_info,
      dkv_mask_info,
      q,
      k,
      v,
      segment_ids,
      mask_value=mask_value,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      config=config,
      max_logit_value=max_logit_value,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
  )


@jax.tree_util.register_pytree_node_class
class SplashAttentionKernel:

  def __init__(
      self,
      fwd_mask_info: MaskInfo,
      dq_mask_info: MaskInfo | None,
      dkv_mask_info: MaskInfo | None,
      **kwargs,
  ):
    self.kwargs = kwargs
    self.fwd_mask_info = fwd_mask_info
    self.dq_mask_info = dq_mask_info
    self.dkv_mask_info = dkv_mask_info

  def __call__(self, *args, **kwargs) -> SplashCustomReturnType:
    return _splash_attention(
        self.fwd_mask_info,
        self.dq_mask_info,
        self.dkv_mask_info,
        *args,
        **kwargs,
        **self.kwargs,
    )

  def manual_sharding_spec(self, sharding: jax.sharding.NamedSharding):
    """Returns a value that can be used as a shard_map partition spec for the kernel."""
    if self.fwd_mask_info.block_mask is not None:
      block_mask_shape = self.fwd_mask_info.block_mask.shape
      try:
        shard_shape = sharding.shard_shape(block_mask_shape)
      except ValueError as exc:
        raise ValueError(
            "The sharding must divide the mask blocks evenly between devices"
        ) from exc
    # Only q sequence sharding is supported.
    spec = sharding.spec
    assert len(spec) == 1
    mask_info_specs = MaskInfo(  # pytype: disable=wrong-arg-types
        mask_next=spec if self.fwd_mask_info.mask_next is not None else None,
        active_rows=spec if self.fwd_mask_info.active_rows is not None else None,
        active_cols=spec if self.fwd_mask_info.active_cols is not None else None,
        num_active_blocks=spec if self.fwd_mask_info.num_active_blocks is not None else None,
        block_mask=spec if self.fwd_mask_info.block_mask is not None else None,
        partial_mask_blocks=jax.sharding.PartitionSpec()  # replicated
        if self.fwd_mask_info.partial_mask_blocks is not None
        else None,
        q_sequence=spec if self.fwd_mask_info.q_sequence is not None else None,
    )
    return SplashAttentionKernel(
        mask_info_specs,
        mask_info_specs if self.dq_mask_info is not None else None,
        mask_info_specs if self.dkv_mask_info is not None else None,
        **self.kwargs,
    )

  def tree_flatten(self):
    return (
        (self.fwd_mask_info, self.dq_mask_info, self.dkv_mask_info),
        self.kwargs,
    )

  @classmethod
  def tree_unflatten(cls, kwargs, values):
    fwd_mask_info, dq_mask_info, dkv_mask_info = values
    # NamedTuples are not preserved during pytree serialization.
    dq_mask_info = (
        MaskInfo(*dq_mask_info)
        if dq_mask_info is not None
        else None
    )
    dkv_mask_info = (
        MaskInfo(*dkv_mask_info)
        if dkv_mask_info is not None
        else None
    )
    return SplashAttentionKernel(
        MaskInfo(*fwd_mask_info),
        dq_mask_info,
        dkv_mask_info,
        **kwargs,
    )


def _make_splash_attention(
    mask: np.ndarray | jax.Array | mask_lib.Mask,
    *,
    config: SplashConfig | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jnp.DTypeLike = np.int8,
    q_seq_shards: int,
):
  if len(mask.shape) != 2:
    raise ValueError(f'Unexpected mask shape: {mask.shape}')

  if isinstance(mask, np.ndarray):
    mask = mask_lib.NumpyMask(mask)

  if config is None:
    config = SplashConfig.get_default()

  process_fn = partial(
      mask_info_lib.process_mask,
      downcast_smem_data=downcast_smem_data,
      partial_mask_blocks_dtype=partial_mask_blocks_dtype,
      q_seq_shards=q_seq_shards,
  )

  fwd_mask_info, mask_function_fwd = process_fn(
      mask,
      (config.block_q, config.block_kv),
  )
  fwd_mask_sparsity = float(np.mean(fwd_mask_info.block_mask != 0))

  fwd_mask_info = tree_util.tree_map(jnp.array, fwd_mask_info)

  dq_mask_info = None
  dkv_mask_info = None
  if config.has_backward_blocks:
    if config.use_fused_bwd_kernel:
      dq_mask_info = None
    else:
      bq_dq, bkv_dq = config.block_q_dq, config.block_kv_dq
      dq_mask_info, mask_function_dq = process_fn(mask, (bq_dq, bkv_dq))
      assert (mask_function_fwd is None) == (mask_function_dq is None)
      dq_mask_info = tree_util.tree_map(jnp.array, dq_mask_info)
    bq_dkv, bkv_dkv = config.block_q_dkv, config.block_kv_dkv
    dkv_mask_info, mask_function_dkv = process_fn(
        mask,
        (bq_dkv, bkv_dkv),
        is_dkv=True,
    )
    assert (mask_function_fwd is None) == (mask_function_dkv is None)

    dkv_mask_info = tree_util.tree_map(jnp.array, dkv_mask_info)

  return SplashAttentionKernel(
      fwd_mask_info,
      dq_mask_info,
      dkv_mask_info,
      config=config,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      mask_function=mask_function_fwd,
      fwd_mask_sparsity=fwd_mask_sparsity,
  )


def _process_mask_shard(
    mask: jax.Array,
    *,
    config: SplashConfig,
    downcast_smem_data: bool,
    partial_mask_blocks_dtype: jnp.DTypeLike,
) -> tuple[MaskInfo, MaskInfo | None, MaskInfo | None]:
  process_mask_fn = functools.partial(
      mask_info_lib._process_dynamic_mask,
      downcast_smem_data=downcast_smem_data,
      partial_mask_blocks_dtype=partial_mask_blocks_dtype,
  )

  fwd_mask_info = process_mask_fn(
      mask, (config.block_q, config.block_kv), is_dkv=False
  )

  dq_mask_info = None
  dkv_mask_info = None
  if config.has_backward_blocks:
    if not config.use_fused_bwd_kernel:
      bq_dq, bkv_dq = config.block_q_dq, config.block_kv_dq
      dq_mask_info = process_mask_fn(mask, (bq_dq, bkv_dq), is_dkv=False)
    bq_dkv, bkv_dkv = config.block_q_dkv, config.block_kv_dkv
    dkv_mask_info = process_mask_fn(mask, (bq_dkv, bkv_dkv), is_dkv=True)

  return (fwd_mask_info, dq_mask_info, dkv_mask_info)


def _make_dynamic_splash_attention(
    mask: jax.Array,
    *,
    mesh: jax.sharding.Mesh | None = None,
    mask_spec: jax.sharding.PartitionSpec | None = None,
    config: SplashConfig | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jnp.DTypeLike = np.int8,
):
  if (mesh is not None) != (mask_spec is not None):
    raise ValueError(
        "Either both or neither of mesh and mask_spec must be specified."
    )

  if mask_spec is not None and len(mask_spec) != 1:
    raise ValueError("Only shard over the query sequence dimension.")

  if len(mask.shape) != 2:
    raise ValueError(f'Unexpected mask shape: {mask.shape}')

  if config is None:
    config = SplashConfig.get_default()

  processing_fn = partial(
      _process_mask_shard,
      config=config,
      downcast_smem_data=downcast_smem_data,
      partial_mask_blocks_dtype=partial_mask_blocks_dtype,
  )

  kwargs = dict(
      config=config,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      attn_logits_soft_cap=config.attn_logits_soft_cap,
      residual_checkpoint_name=config.residual_checkpoint_name,
      mask_function=None,
      fwd_mask_sparsity=1.0,
      interpret=config.interpret,
  )

  # If the input mask is replicated we don't need to call shard_map.
  if mask_spec is None:
    fwd_mask_info, dq_mask_info, dkv_mask_info = processing_fn(mask)
    kernel = SplashAttentionKernel(
        fwd_mask_info, dq_mask_info, dkv_mask_info, **kwargs
    )
    return kernel

  mask_info_specs = MaskInfo(  # pytype: disable=wrong-arg-types
      mask_next=mask_spec,
      active_rows=mask_spec,
      active_cols=mask_spec,
      num_active_blocks=mask_spec,
      block_mask=mask_spec,
      partial_mask_blocks=mask_spec,
      q_sequence=None,
  )
  has_dkv_mask_info = config.has_backward_blocks
  has_dq_mask_info = (
      config.has_backward_blocks and not config.use_fused_bwd_kernel
  )

  in_specs = mask_spec
  out_specs = (
      mask_info_specs,
      mask_info_specs if has_dq_mask_info else None,
      mask_info_specs if has_dkv_mask_info else None,
  )

  fwd_mask_info, dq_mask_info, dkv_mask_info = jax.shard_map(
      processing_fn,
      mesh=mesh,
      in_specs=in_specs,
      out_specs=out_specs,
      check_vma=False,
  )(mask)

  kernel = SplashAttentionKernel(
      fwd_mask_info, dq_mask_info, dkv_mask_info, **kwargs
  )
  kernel_spec = SplashAttentionKernel(*out_specs, **kwargs)

  return (kernel, kernel_spec)

make_splash_mha = partial(_make_splash_attention, is_mqa=False)
make_splash_mqa = partial(_make_splash_attention, is_mqa=True)

make_splash_mha_single_device = partial(
    make_splash_mha, is_mqa=False, q_seq_shards=1
)

make_splash_mqa_single_device = partial(
    make_splash_mha, is_mqa=True, q_seq_shards=1
)

make_dynamic_splash_mqa = partial(_make_dynamic_splash_attention, is_mqa=True)
make_dynamic_splash_mha = partial(_make_dynamic_splash_attention, is_mqa=False)
