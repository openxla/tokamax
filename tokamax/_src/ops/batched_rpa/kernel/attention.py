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
"""Stateless FlashAttention math, QK masking, and online-softmax updates."""

from collections.abc import Sequence

import jax
from jax import lax
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.batched_rpa.kernel import configs
from tokamax._src.ops.batched_rpa.kernel import utils


def init_softmax_values(
    sink: jax.Array | None,
    *,
    cfgs: configs.RpaConfigs,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Returns initial `(m, l, acc)` arrays prior to schedule execution."""
  dtype = cfgs.layout.softmax_dtype
  if sink is None:
    m = jnp.full(cfgs.lm_scratch_shape, -jnp.inf, dtype=dtype)
    l = jnp.zeros(cfgs.lm_scratch_shape, dtype=dtype)
  else:
    m = jnp.concat([sink] * cfgs.bq_sz, axis=1)
    l = jnp.ones(cfgs.lm_scratch_shape, dtype=dtype)
  acc = jnp.zeros(cfgs.acc_scratch_shape, dtype=dtype)
  return m, l, acc


def build_qk_mask(
    shape: tuple[int, int, int, int],
    *,
    bq_start: int,
    causal_offset: Sequence[jax.Array],
    k_min: Sequence[jax.Array] | None = None,
    k_max: Sequence[jax.Array] | None = None,
    cfgs: configs.RpaConfigs,
) -> list[jax.Array]:
  """Builds causal, sliding-window, and `[k_min, k_max)` masks per batch lane."""
  b, _, tq, s = shape

  # Keep iotas as rank-3 tensors with singleton dims to avoid allocating a
  # (k_heads, tq, s) integer tensor for q_iota - kv_iota.
  kv_iota = lax.broadcasted_iota(jnp.int32, (1, 1, s), 2)
  q_iota = lax.broadcasted_iota(jnp.int32, (1, tq, 1), 1)
  q_iota //= cfgs.aligned_num_q_heads_per_kv_head

  masks = []
  for b_idx in range(b):
    offset = causal_offset[b_idx] - bq_start
    kv_bound = kv_iota + offset
    mask_b = q_iota >= kv_bound
    if (sliding_window := cfgs.model.sliding_window) is not None:
      mask_b = jnp.logical_and(mask_b, q_iota < kv_bound + sliding_window)
    if k_min is not None:
      mask_b = jnp.logical_and(mask_b, kv_iota >= k_min[b_idx])
    if k_max is not None:
      mask_b = jnp.logical_and(mask_b, kv_iota < k_max[b_idx])
    masks.append(mask_b)

  return masks


def qk_softmax_step(
    q: jax.Array,  # [B, H_kv, TQ, D]
    k: jax.Array,  # [B, H_kv, D, S] (SEQ) or [B, H_kv, S, D] (HEAD)
    m_prev: jax.Array,  # [H_kv, TQ, 128]
    l_prev: jax.Array,  # [H_kv, TQ, 128]
    is_last_k: Sequence[jax.Array | bool],
    *,
    custom_mask: Sequence[jax.Array],
    cfgs: configs.RpaConfigs,
    k_scale: jax.Array | float | None = None,
    attention_sink: jax.Array | None = None,
) -> tuple[
    jax.Array,
    list[jax.Array],
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
  """Computes `QK^T`, applies masking/scaling, and advances online `(m, l)`."""
  b, k_heads, tq, h_size = q.shape
  softmax_dtype = cfgs.layout.softmax_dtype

  if cfgs.serve.scale_q is not None:
    q = q / cfgs.serve.scale_q
    if jnp.issubdtype(k.dtype, jnp.floating):
      dtype_info = jnp.finfo(k.dtype)
      q = jnp.clip(q, min=float(dtype_info.min), max=float(dtype_info.max))
    q = q.astype(k.dtype)

  if cfgs.is_seq_along_lane:
    s = k.shape[-1]
    qk = lax.dot(
        q.reshape(-1, tq, h_size),
        k.reshape(-1, h_size, s),
        dimension_numbers=(([2], [1]), ([0], [0])),
        preferred_element_type=jnp.float32,
    )
  else:
    s = k.shape[-2]
    qk = lax.dot(
        q.reshape(-1, tq, h_size),
        k.reshape(-1, s, h_size),
        dimension_numbers=(([2], [2]), ([0], [0])),
        preferred_element_type=jnp.float32,
    )
  qk = qk.reshape(b, k_heads, tq, s)

  qk *= cfgs.model.sm_scale
  if cfgs.serve.scale_q is not None:
    qk *= cfgs.serve.scale_q
  if k_scale is not None:
    qk *= k_scale

  qk = qk.astype(softmax_dtype)
  if cfgs.model.soft_cap is not None:
    qk = cfgs.model.soft_cap * jnp.tanh(qk / cfgs.model.soft_cap)

  qk_masked = [
      jnp.where(custom_mask[b_idx], qk[b_idx], cfgs.model.mask_value)
      for b_idx in range(cfgs.block.batch_size)
  ]
  qk = jnp.stack(qk_masked, axis=0)

  m_curr = jnp.max(qk, axis=-1, keepdims=True)
  alpha_list = []
  m_next_list = []
  m_reset = -jnp.inf if attention_sink is None else attention_sink

  for b_idx in range(cfgs.block.batch_size):
    m_curr_b = m_curr[b_idx]
    m_next_b = jnp.maximum(m_prev, m_curr_b)
    alpha_b = jnp.exp(m_prev - m_next_b)
    alpha_list.append(alpha_b)
    m_next_list.append(m_next_b)
    m_prev = jnp.where(is_last_k[b_idx], m_reset, m_next_b)

  m_next = jnp.stack(m_next_list, axis=0)
  p = jnp.exp(qk - utils.broadcast_minor(m_next, qk.shape))
  p_rowsum = jnp.sum(p, axis=-1, keepdims=True, dtype=softmax_dtype)

  l_next_list = []
  l_reset = 0.0 if attention_sink is None else 1.0
  for b_idx in range(cfgs.block.batch_size):
    l_next_b = alpha_list[b_idx] * l_prev + p_rowsum[b_idx]
    l_next_list.append(l_next_b)
    if cfgs.fuse_accum or attention_sink is not None:
      l_prev = jnp.where(is_last_k[b_idx], l_reset, l_next_b)
    else:
      l_prev = l_next_b

  l_next = jnp.stack(l_next_list, axis=0)
  return p, alpha_list, m_next, l_next, m_prev, l_prev


def pv_step(
    p: jax.Array,  # [B, H_kv, TQ, S]
    v: jax.Array,  # [B, H_kv, D, S] (SEQ) or [B, H_kv, S, D] (HEAD)
    alpha_list: list[jax.Array],
    o_prev: jax.Array,  # [H_kv, TQ, D]
    is_last_k: Sequence[jax.Array | bool],
    *,
    cfgs: configs.RpaConfigs,
    v_scale: jax.Array | float | None = None,
    attention_sink: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Computes `PV` and rescales/accumulates into `o_prev` across batch lanes."""
  b, k_heads, tq, s = p.shape
  softmax_dtype = cfgs.layout.softmax_dtype

  if cfgs.serve.per_token_scale:
    assert isinstance(v_scale, jax.Array)
    v_sc = v_scale[:, :, jnp.newaxis, :]
    p = p.astype(v_sc.dtype) * v_sc

  if cfgs.serve.quantize_p:
    p = p.astype(v.dtype)

  if cfgs.is_seq_along_lane:
    h_size = v.shape[-2]
    pv = lax.dot(
        p.reshape(-1, tq, s),
        v.reshape(-1, h_size, s),
        dimension_numbers=(([2], [2]), ([0], [0])),
        preferred_element_type=jnp.float32,
    )
  else:
    h_size = v.shape[-1]
    pv = lax.dot(
        p.reshape(-1, tq, s),
        v.reshape(-1, s, h_size),
        dimension_numbers=(([2], [1]), ([0], [0])),
        preferred_element_type=jnp.float32,
    )
  pv = pv.astype(softmax_dtype).reshape(b, k_heads, tq, h_size)

  if not cfgs.serve.per_token_scale and cfgs.serve.scale_v is not None:
    pv *= cfgs.serve.scale_v

  o_next_list = []
  for b_idx in range(cfgs.block.batch_size):
    alpha_b = utils.broadcast_minor(alpha_list[b_idx], o_prev.shape)
    o_next_b = alpha_b * o_prev + pv[b_idx]
    o_next_list.append(o_next_b)
    if cfgs.fuse_accum or attention_sink is not None:
      o_prev = jnp.where(is_last_k[b_idx], 0.0, o_next_b)
    else:
      o_prev = o_next_b
  o_next = jnp.stack(o_next_list, axis=0)
  return o_next, o_prev


def normalize_out(
    batch_acc: jax.Array,
    batch_l: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
) -> jax.Array:
  """Divides accumulated `PV` sum `batch_acc` by softmax denominator `batch_l`."""
  batch_l_bcast = utils.broadcast_minor(batch_l, batch_acc.shape)
  if (
      cfgs.serve.dtype_out == jnp.float32
      or cfgs.serve.dtype_out == batch_l_bcast.dtype == jnp.bfloat16
  ):
    result = lax.div(batch_acc, batch_l_bcast)
  else:
    result = batch_acc * pltpu.reciprocal(batch_l_bcast, approx=True)
  return result.astype(cfgs.serve.dtype_out)


def compute_lse(batch_m: jax.Array, batch_l: jax.Array) -> jax.Array:
  """Computes log-sum-exp `m + log(max(l, 1e-9))` for partial-attention merge."""
  return batch_m + jnp.log(jnp.maximum(batch_l, 1e-9))
