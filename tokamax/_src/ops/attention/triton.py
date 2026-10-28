# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Triton FlashAttention implementation."""

import dataclasses
import functools
import math
from typing import Annotated, Any, ClassVar, override

import jax
from jax.extend import backend
import jax.numpy as jnp
import jax_triton as jt
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import batching
from tokamax._src import gpu_utils
from tokamax._src import jaxtyping
from tokamax._src import precision as precision_lib
from tokamax._src import pydantic as pydantic_lib
from tokamax._src import quantization
from tokamax._src import triton_utils
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import triton_vjp as vjp
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

Mask = base.Mask
QArray = base.QArray
Residuals = base.Residuals
PagingInfo = base.PagingInfo


@triton.jit
def _load_2d(ptr, offs_r, bound_r, stride_r, start_c, bound_c, block_c):
  block_r: tl.constexpr = offs_r.shape[0]
  offs_c = start_c + tl.arange(0, block_c)
  mask = triton_utils.mask_2d(
      offs_r,
      bound_r,
      offs_c,
      bound_c,
      bound_r % block_r != 0,
      bound_c % block_c != 0,
  )
  return tl.load(
      ptr + offs_r[:, None].to(tl.int64) * stride_r + offs_c[None, :],
      mask=mask,
      other=None if mask is None else 0.0,
  )


@triton.jit
def _advance_scale_ptr(ptr, pid_b, batch_shape, batch_strides, pid_h, shape):
  """Advances `ptr` to `scales[pid_b, 0, pid_h, 0]`.

  `pid_b` is a flat index into the batch dims, and `shape` is the per-batch
  `(T, H, D)` scales shape. If `H` is broadcast, `pid_h` is ignored.
  """
  ptr += triton_utils.batch_offset(pid_b, batch_shape, batch_strides)
  return ptr if shape[1] == 1 else ptr + pid_h * shape[2]


@triton.jit
def _load_scales(ptr, offs_r, start_d, head_dim, block_d, shape):
  """Loads the scales of tile `[offs_r, start_d : start_d + block_d]`.

  `shape` is the per-batch `(T, H, D)` scales shape. `T` and `H` are either 1
  or full, and `D` is 1 (a scale per row), `head_dim` (a scale per column) or
  the number of subchannels.
  """
  rows: tl.constexpr = shape[0]
  num_scales: tl.constexpr = shape[2]
  row_stride: tl.constexpr = shape[1] * num_scales
  zero = tl.zeros((1, 1), dtype=tl.int64)
  r = zero if rows == 1 else offs_r[:, None].to(tl.int64) * row_stride
  check_r: tl.constexpr = rows > 1 and rows % offs_r.shape[0] != 0
  if num_scales == 1:
    mask = offs_r[:, None] < rows if check_r else None
    scale = tl.load(ptr + r, mask=mask, other=None if mask is None else 0.0)
  else:
    subchannel_size: tl.constexpr = head_dim // num_scales
    # Load a scale per subchannel and repeat it, since software pipelining
    # would otherwise stage a full tile of repeated scales in shared memory.
    block_s: tl.constexpr = block_d // subchannel_size
    offs_s = start_d // subchannel_size + tl.arange(0, block_s)
    check_s: tl.constexpr = num_scales % block_s != 0
    mask = triton_utils.mask_2d(
        offs_r, rows, offs_s, num_scales, check_r, check_s
    )
    scale = tl.load(
        ptr + r + offs_s[None, :],
        mask=mask,
        other=None if mask is None else 0.0,
    )
    if subchannel_size > 1:
      scale_rows: tl.constexpr = scale.shape[0]
      scale = tl.broadcast_to(
          scale[:, :, None], (scale_rows, block_s, subchannel_size)
      ).reshape(scale_rows, block_d)
  return scale


@triton.jit
def _load_qarray(
    ptr,
    scale_ptr,
    offs_r,
    bound_r,
    num_heads,
    start_d,
    head_dim,
    block_d,
    scale_shape,
    is_int4,
):
  """Loads a 2D tile of values and, if `scale_ptr` is not `None`, scales."""
  pack: tl.constexpr = tl.constexpr(2 if is_int4 else 1)
  x = _load_2d(
      ptr,
      offs_r,
      bound_r,
      num_heads * head_dim // pack,
      start_d // pack,
      head_dim // pack,
      block_d // pack,
  )
  if is_int4:
    # Low nibbles hold even elements (see `_pack_int4`), and the shifts
    # sign-extend each nibble.
    x = tl.join((x << 4) >> 4, x >> 4).reshape(offs_r.shape[0], block_d)
  if scale_ptr is None:
    return x, None
  else:
    return x, _load_scales(
        scale_ptr, offs_r, start_d, head_dim, block_d, scale_shape
    )


@triton.jit
def _dequantize(x, scale):
  return x if scale is None else x.to(scale.dtype) * scale


@triton.jit
def _fwd_kernel(
    q_start_ptr,
    q_end_ptr,
    q_ptr,
    q_scale_ptr,
    k_ptr,
    k_scale_ptr,
    v_ptr,
    v_scale_ptr,
    bias_ptr,
    mask_ptr,
    dropout_mask_ptr,
    k_start_ptr,
    k_end_ptr,
    out_ptr,
    l_ptr,
    m_ptr,
    batch_shape: tl.constexpr,
    batch_strides: tl.constexpr,
    seq_len_q: tl.constexpr,
    num_heads_q: tl.constexpr,
    head_dim: tl.constexpr,
    seq_len_k: tl.constexpr,
    num_heads_k: tl.constexpr,
    head_dim_out: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    block_d: tl.constexpr,
    block_d_out: tl.constexpr,
    q_scale_shape: tl.constexpr,
    k_scale_shape: tl.constexpr,
    v_scale_shape: tl.constexpr,
    q_is_int4: tl.constexpr,
    k_is_int4: tl.constexpr,
    v_is_int4: tl.constexpr,
    quantize_qk_dot: tl.constexpr,
    bias_shape: tl.constexpr,
    mask_shape: tl.constexpr,
    dropout_shape: tl.constexpr,
    k_start_shape: tl.constexpr,
    k_end_shape: tl.constexpr,
    sm_scale: tl.constexpr,
    is_causal: tl.constexpr,
    dropout_rate: tl.constexpr,
    logits_dtype: tl.constexpr,
    logits_soft_cap: tl.constexpr,
    use_base2: tl.constexpr,
    use_stable_softmax: tl.constexpr,
    qk_prec: tl.constexpr,
    qk_dtype: tl.constexpr,
    wv_prec: tl.constexpr,
    wv_dtype: tl.constexpr,
    normalize_output: tl.constexpr,
    pack_mask: tl.constexpr,
):
  """Triton MHA forward kernel."""
  grid_m: tl.constexpr = tl.cdiv(seq_len_q, block_m)
  pid_1d = tl.program_id(0)
  pid_m = (pid_1d % grid_m).to(tl.int64)
  pid_h = (pid_1d // grid_m % num_heads_q).to(tl.int64)
  pid_b = (
      tl.full((), 0, tl.int64)
      if batch_shape == ()
      else (pid_1d // (grid_m * num_heads_q)).to(tl.int64)
  )
  pid_h_kv = pid_h // (num_heads_q // num_heads_k)

  start_m = pid_m.to(tl.int32) * block_m
  span_m = start_m + tl.arange(0, block_m)
  check_m: tl.constexpr = seq_len_q % block_m != 0
  check_d_out: tl.constexpr = head_dim_out % block_d_out != 0
  mask_m_1d = span_m < seq_len_q if check_m else None

  out_stride_m: tl.constexpr = num_heads_q * head_dim_out
  out_ptr += pid_b * seq_len_q * out_stride_m + pid_h * head_dim_out
  if l_ptr is not None:
    l_ptr += pid_b * num_heads_q * seq_len_q + pid_h * seq_len_q
  if m_ptr is not None:
    m_ptr += pid_b * num_heads_q * seq_len_q + pid_h * seq_len_q

  (
      stride_q_start_b,
      stride_q_end_b,
      stride_qb,
      stride_q_scale_b,
      stride_kb,
      stride_k_scale_b,
      stride_vb,
      stride_v_scale_b,
      stride_bias_b,
      stride_mask_b,
      stride_dropout_b,
      stride_k_start_b,
      stride_k_end_b,
  ) = batch_strides

  skip_cond = tl.full((), 0, tl.int1)
  q_start = 0
  q_end = 0
  if q_start_ptr is not None:
    q_start_ptr += triton_utils.batch_offset(
        pid_b, batch_shape, stride_q_start_b
    )
    q_start = tl.load(q_start_ptr)
    skip_cond |= q_start >= start_m + block_m
  if q_end_ptr is not None:
    q_end_ptr += triton_utils.batch_offset(pid_b, batch_shape, stride_q_end_b)
    q_end = tl.load(q_end_ptr)
    skip_cond |= q_end <= start_m

  stride_qh: tl.constexpr = head_dim // 2 if q_is_int4 else head_dim
  stride_kh: tl.constexpr = head_dim // 2 if k_is_int4 else head_dim
  stride_vh: tl.constexpr = head_dim_out // 2 if v_is_int4 else head_dim_out
  q_ptr += (
      triton_utils.batch_offset(pid_b, batch_shape, stride_qb)
      + pid_h * stride_qh
  )
  k_ptr += (
      triton_utils.batch_offset(pid_b, batch_shape, stride_kb)
      + pid_h_kv * stride_kh
  )
  v_ptr += (
      triton_utils.batch_offset(pid_b, batch_shape, stride_vb)
      + pid_h_kv * stride_vh
  )
  if q_scale_ptr is not None:
    q_scale_ptr = _advance_scale_ptr(
        q_scale_ptr, pid_b, batch_shape, stride_q_scale_b, pid_h, q_scale_shape
    )
  if k_scale_ptr is not None:
    k_scale_ptr = _advance_scale_ptr(
        k_scale_ptr,
        pid_b,
        batch_shape,
        stride_k_scale_b,
        pid_h_kv,
        k_scale_shape,
    )
  if v_scale_ptr is not None:
    v_scale_ptr = _advance_scale_ptr(
        v_scale_ptr,
        pid_b,
        batch_shape,
        stride_v_scale_b,
        pid_h_kv,
        v_scale_shape,
    )

  if bias_ptr is not None:
    bias_ptr = triton_utils.advance_bcast_ptr(
        bias_ptr,
        pid_b,
        batch_shape,
        stride_bias_b,
        pid_h,
        num_heads_q,
        bias_shape,
    )
  if mask_ptr is not None:
    mask_ptr = triton_utils.advance_bcast_ptr(
        mask_ptr,
        pid_b,
        batch_shape,
        stride_mask_b,
        pid_h,
        num_heads_q,
        mask_shape,
    )
  if dropout_mask_ptr is not None:
    dropout_mask_ptr = triton_utils.advance_bcast_ptr(
        dropout_mask_ptr,
        pid_b,
        batch_shape,
        stride_dropout_b,
        pid_h,
        num_heads_q,
        dropout_shape,
    )
  if k_start_ptr is not None:
    k_start_ptr = triton_utils.advance_bcast_ptr(
        k_start_ptr,
        pid_b,
        batch_shape,
        stride_k_start_b,
        pid_h,
        num_heads_q,
        k_start_shape,
    )
  if k_end_ptr is not None:
    k_end_ptr = triton_utils.advance_bcast_ptr(
        k_end_ptr,
        pid_b,
        batch_shape,
        stride_k_end_b,
        pid_h,
        num_heads_q,
        k_end_shape,
    )

  # See base class for explanation of why this is not `-inf`.
  mask_value = float(jnp.finfo(jnp.float32).min)

  split_d: tl.constexpr = tl.cdiv(head_dim, block_d)
  split_d_out: tl.constexpr = tl.cdiv(head_dim_out, block_d_out)
  m_i = tl.full(
      (block_m,),
      mask_value if use_stable_softmax else 0.0,
      dtype=tl.float32,
  )
  l_i = tl.zeros((block_m,), dtype=tl.float32)
  accs = ()
  for _ in tl.static_range(split_d_out):
    accs += (tl.zeros((block_m, block_d_out), dtype=tl.float32),)

  if not skip_cond:
    qs = ()
    q_scales = ()
    for i_d in tl.static_range(split_d):
      q, q_scale = _load_qarray(
          q_ptr,
          q_scale_ptr,
          span_m,
          seq_len_q,
          num_heads_q,
          i_d * block_d,
          head_dim,
          block_d,
          q_scale_shape,
          q_is_int4,
      )
      if quantize_qk_dot:
        qs += (q,)
        q_scales += (q_scale * sm_scale,)
      else:
        qs += ((_dequantize(q, q_scale) * sm_scale).to(qk_dtype),)

    num_k_blocks = tl.cdiv(seq_len_k, block_k)
    # Rows past ``seq_len_q`` get fill values that are neutral for the min/max
    # reductions below, so they don't add k blocks to the loop.
    k_start = (
        None
        if k_start_ptr is None
        else triton_utils.load_1d_bcast(
            k_start_ptr, span_m, k_start_shape, other=seq_len_k
        )
    )
    k_end = (
        None
        if k_end_ptr is None
        else triton_utils.load_1d_bcast(k_end_ptr, span_m, k_end_shape, other=0)
    )

    lo, mid, hi = 0, num_k_blocks, num_k_blocks
    if k_start is not None:
      lo = tl.maximum(tl.min(k_start, axis=0) // block_k, 0)
    if is_causal:
      mid = tl.minimum(start_m // block_k, num_k_blocks)
      hi = tl.minimum(tl.cdiv(start_m + block_m, block_k), num_k_blocks)
    elif k_end is not None:
      mid = tl.minimum(
          tl.maximum(tl.min(k_end, axis=0) // block_k, 0), num_k_blocks
      )
      # Clamped first, since `tl.cdiv` overflows for `k_end` near the int32 max.
      k_end_max = tl.minimum(tl.max(k_end, axis=0), seq_len_k)
      hi = tl.maximum(tl.cdiv(k_end_max, block_k), 0)
    mid = tl.maximum(lo, mid)

    use_mask_k: tl.constexpr = seq_len_k % block_k != 0
    exp_fn: tl.constexpr = tl.math.exp2 if use_base2 else tl.math.exp

    for phase in tl.static_range(
        2 if is_causal or k_end_ptr is not None else 1
    ):
      for i_k in range(lo if phase == 0 else mid, mid if phase == 0 else hi):
        start_k = i_k * block_k
        span_k = start_k + tl.arange(0, block_k)
        s = tl.zeros((block_m, block_k), dtype=tl.float32)
        for i_d in tl.static_range(split_d):
          k, k_scale = _load_qarray(
              k_ptr,
              k_scale_ptr,
              span_k,
              seq_len_k,
              num_heads_k,
              i_d * block_d,
              head_dim,
              block_d,
              k_scale_shape,
              k_is_int4,
          )
          if quantize_qk_dot:
            qk_scale = q_scales[i_d] * tl.trans(k_scale)
            s += tl.dot(qs[i_d], tl.trans(k)).to(tl.float32) * qk_scale
          else:
            k = _dequantize(k, k_scale)
            s += tl.dot(
                qs[i_d], tl.trans(k.to(qk_dtype)), input_precision=qk_prec
            )
        s = s.to(logits_dtype)

        if bias_ptr is not None:
          s += triton_utils.load_2d_bcast(
              bias_ptr, span_m, span_k, bias_shape
          ).to(logits_dtype)

        if logits_soft_cap is not None:
          s = logits_soft_cap * libdevice.tanh(
              (s / logits_soft_cap).to(tl.float32)
          )

        if use_base2:
          s *= math.log2(math.e)

        if mask_ptr is not None:
          mask_val = triton_utils.load_2d_bcast(
              mask_ptr,
              span_m,
              span_k // 8 if pack_mask else span_k,
              mask_shape,
              0,
          )
          if pack_mask:
            mask_val &= (1 << span_k % 8)[None, :]
          s = tl.where(mask_val != 0, s, mask_value)

        if q_start_ptr is not None:
          s = tl.where(q_start <= span_m[:, None], s, mask_value)
        if q_end_ptr is not None:
          s = tl.where(q_end > span_m[:, None], s, mask_value)
        if k_start is not None:
          s = tl.where(k_start[:, None] <= span_k[None, :], s, mask_value)
        if phase == 1 and k_end is not None:
          s = tl.where(k_end[:, None] > span_k[None, :], s, mask_value)
        if phase == 1 and is_causal:
          s = tl.where(span_m[:, None] >= span_k[None, :], s, mask_value)

        if use_mask_k:
          s = tl.where((span_k < seq_len_k)[None, :], s, float("-inf"))

        if use_stable_softmax:
          m_ij = tl.maximum(m_i, tl.max(s, axis=1))
          p = exp_fn(s - m_ij[:, None])
          alpha = exp_fn(m_i - m_ij)
          m_i = m_ij
          rescaled_accs = ()
          for i_o in tl.static_range(split_d_out):
            rescaled_accs += (accs[i_o] * alpha[:, None],)
          accs = rescaled_accs
          l_i *= alpha
        else:
          p = exp_fn(s.to(tl.float32))
        l_i += tl.sum(p, axis=1)

        if dropout_mask_ptr is not None:
          dropout_mask = triton_utils.load_2d_bcast(
              dropout_mask_ptr, span_m, span_k, dropout_shape, 0
          )
          p *= (dropout_mask != 0).to(p.dtype) / (1.0 - dropout_rate)

        p = p.to(wv_dtype)
        new_accs = ()
        for i_o in tl.static_range(split_d_out):
          v, v_scale = _load_qarray(
              v_ptr,
              v_scale_ptr,
              span_k,
              seq_len_k,
              num_heads_k,
              i_o * block_d_out,
              head_dim_out,
              block_d_out,
              v_scale_shape,
              v_is_int4,
          )
          v = _dequantize(v, v_scale)
          acc = accs[i_o] + tl.dot(p, v.to(wv_dtype), input_precision=wv_prec)
          new_accs += (acc,)
        accs = new_accs

  # It is possible that every value in a row was masked to f32 min, and that
  # `l_i` is `0` for that row. Add epsilon value to avoid NaNs from `0 / 0`.
  l_i += float(jnp.finfo(jnp.float32).tiny)
  if l_ptr is not None:
    tl.store(l_ptr + span_m, l_i, mask=mask_m_1d)
  if m_ptr is not None:
    if use_base2:
      # `m` consumers expect natural-log units, with `mask_value` marking fully
      # masked rows.
      m_i = tl.where(m_i == mask_value, m_i, m_i * math.log(2.0))
    tl.store(m_ptr + span_m, m_i, mask=mask_m_1d)

  for i_o in tl.static_range(split_d_out):
    acc = accs[i_o]
    if normalize_output:
      acc /= l_i[:, None]
    offs_d_out = i_o * block_d_out + tl.arange(0, block_d_out)
    out_mask = triton_utils.mask_2d(
        span_m, seq_len_q, offs_d_out, head_dim_out, check_m, check_d_out
    )
    tl.store(
        out_ptr
        + span_m[:, None].to(tl.int64) * out_stride_m
        + offs_d_out[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=out_mask,
    )


@jaxtyping.jaxtyped
def _fwd(
    q: Float[Array | QArray, "T H D"],
    k: Float[Array | QArray, "t h D"],
    v: Float[Array | QArray, "t h d"],
    bias: Float[Array, "#H #T #t"] | None,
    mask: Bool[Array, "#H #T #t"] | Bool[Array, "#h #T #t"] | None,
    dropout_mask: Bool[Array, "#H #T #t"] | None,
    q_start: Int[Array, "1 1"] | None,
    q_end: Int[Array, "1 1"] | None,
    k_start: Int[Array, "#H #T"] | Int[Array, "#h #T"] | None,
    k_end: Int[Array, "#H #T"] | Int[Array, "#h #T"] | None,
    *,
    dropout_rate: float,
    logits_dtype: jnp.dtype,
    logits_scale: float,
    logits_soft_cap: float | None,
    is_causal: bool,
    config: "Config",
    use_base2: bool,
    use_stable_softmax: bool,
    normalize_output: bool,
    return_residuals: bool,
    q_k_dot_precision: jax.lax.DotAlgorithmPreset,
    weights_v_dot_precision: jax.lax.DotAlgorithmPreset,
) -> tuple[Float[Array, "T H d"], Residuals | None]:
  """Forward pass of Triton FlashAttention."""
  q, k, v = map(quantization.as_array_or_qarray_without_zero_point, (q, k, v))
  q_values, q_scale = _split_qarray(q)
  k_values, k_scale = _split_qarray(k)
  v_values, v_scale = _split_qarray(v)

  seq_len_q, num_heads_q, head_dim = q.shape
  seq_len_k, num_heads_k, head_dim_out = v.shape

  if config.pack_mask and mask is not None and mask.shape[-1] != 1:
    mask = jnp.packbits(mask, axis=-1, bitorder="little")
    pack_mask = True
  else:
    pack_mask = False

  block_m = config.block_q
  block_k = config.block_k
  block_d = max(
      16, triton.next_power_of_2(min(config.block_d or head_dim, head_dim))
  )
  block_d_out = max(
      16,
      triton.next_power_of_2(
          min(config.block_d_out or head_dim_out, head_dim_out)
      ),
  )
  quantize_qk_dot = (
      q_scale is not None
      and k_scale is not None
      and q_scale.shape[-1] == k_scale.shape[-1] == 1
      and q_values.dtype == k_values.dtype == jnp.int8
      and block_d >= 32  # 8-bit `tl.dot` needs K >= 32.
  )

  qk_dtype, qk_prec = triton_utils.get_dot_dtype_and_precision(
      q_k_dot_precision, q.dtype, k.dtype
  )
  wv_dtype, wv_prec = triton_utils.get_dot_dtype_and_precision(
      weights_v_dot_precision, v.dtype, v.dtype
  )
  logits_tl_dtype = triton_utils.jnp_to_tl_dtype(logits_dtype)
  name = (
      "triton_flash_attention_fwd_res"
      if return_residuals
      else "triton_flash_attention"
  )

  @triton_utils.vmap_kernel
  def fwd_call(batch_shape, batch_strides, *args):
    out_dtype = q.dtype if normalize_output else jnp.float32
    out_shape = jax.ShapeDtypeStruct(
        (*batch_shape, seq_len_q, num_heads_q, head_dim_out), out_dtype
    )
    residual_shape = jax.ShapeDtypeStruct(
        (*batch_shape, num_heads_q, seq_len_q), jnp.float32
    )
    out_type = (
        out_shape,
        residual_shape if return_residuals else None,
        residual_shape if return_residuals and use_stable_softmax else None,
    )
    none_outs: dict[str, Any] = {
        k: None
        for k, v in zip(("l_ptr", "m_ptr"), out_type[1:], strict=True)
        if v is None
    }
    return jt.triton_call(
        *args,
        kernel=_fwd_kernel,
        out_type=out_type,
        grid=(
            math.prod(batch_shape)
            * num_heads_q
            * triton.cdiv(seq_len_q, block_m),
        ),
        name=name,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        batch_shape=batch_shape,
        batch_strides=batch_strides,
        seq_len_q=seq_len_q,
        num_heads_q=num_heads_q,
        head_dim=head_dim,
        seq_len_k=seq_len_k,
        num_heads_k=num_heads_k,
        head_dim_out=head_dim_out,
        block_m=block_m,
        block_k=block_k,
        block_d=block_d,
        block_d_out=block_d_out,
        q_scale_shape=(1, 1, 1) if q_scale is None else q_scale.shape[-3:],
        k_scale_shape=(1, 1, 1) if k_scale is None else k_scale.shape[-3:],
        v_scale_shape=(1, 1, 1) if v_scale is None else v_scale.shape[-3:],
        q_is_int4=q_values.dtype == jnp.int4,
        k_is_int4=k_values.dtype == jnp.int4,
        v_is_int4=v_values.dtype == jnp.int4,
        quantize_qk_dot=quantize_qk_dot,
        bias_shape=(1, 1, 1) if bias is None else bias.shape[-3:],
        mask_shape=(1, 1, 1) if mask is None else mask.shape[-3:],
        dropout_shape=(
            (1, 1, 1) if dropout_mask is None else dropout_mask.shape[-3:]
        ),
        k_start_shape=(1, 1) if k_start is None else k_start.shape[-2:],
        k_end_shape=(1, 1) if k_end is None else k_end.shape[-2:],
        sm_scale=float(logits_scale),
        is_causal=is_causal,
        dropout_rate=float(dropout_rate),
        logits_dtype=logits_tl_dtype,
        logits_soft_cap=(
            None if logits_soft_cap is None else float(logits_soft_cap)
        ),
        use_base2=use_base2,
        use_stable_softmax=use_stable_softmax,
        qk_prec=qk_prec,
        qk_dtype=qk_dtype,
        wv_prec=wv_prec,
        wv_dtype=wv_dtype,
        normalize_output=normalize_output,
        pack_mask=pack_mask,
        **none_outs,
    )

  out, l, m = fwd_call(
      q_start,
      q_end,
      _pack_int4(q_values),
      q_scale,
      _pack_int4(k_values),
      k_scale,
      _pack_int4(v_values),
      v_scale,
      bias,
      mask,
      dropout_mask,
      k_start,
      k_end,
  )
  if not return_residuals:
    return out, None
  return out, (jnp.zeros_like(l) if m is None else m, l)


def _split_qarray(x):
  """Returns the values and scales (or `None`) of a possibly quantized array."""
  if not isinstance(x, QArray):
    return x, None
  if x.qvalue.dtype not in (jnp.int8, jnp.int4):
    raise NotImplementedError(f"Unsupported quantized dtype: {x.qvalue.dtype}")
  if any(s not in (1, d) for s, d in zip(x.scale.shape[:-1], x.shape[:-1])):
    raise NotImplementedError(
        "Subchannel quantization is only supported along the head dim."
    )
  subchannel_size = x.shape[-1] // x.scale.shape[-1]
  if x.scale.shape[-1] > 1 and (
      triton.next_power_of_2(subchannel_size) != subchannel_size
  ):
    raise NotImplementedError("Subchannel size must be a power of two.")
  if x.qvalue.dtype == jnp.int4 and x.shape[-1] % 2 != 0:
    raise NotImplementedError("int4 quantization requires an even head dim.")
  return x.qvalue, x.scale


def _pack_int4(x):
  # Triton has no int4 type, so pairs of int4s are passed as int8s.
  if x.dtype != jnp.int4:
    return x
  pairs = x.reshape(*x.shape[:-1], -1, 2)
  return jax.lax.bitcast_convert_type(pairs, jnp.int8)


def _broadcast_to_rank(x, rank):
  return None if x is None else jax.lax.broadcast_to_rank(x, rank)


def _is_per_batch_scalar(x):
  """Returns whether `x` holds a single value per batch element."""
  return x is not None and x.size > 0 and math.prod(x.shape[-2:]) == 1


def _decompose_mask(mask, q, k, q_indices, k_indices):
  """Decomposes `mask` into a mask array, `is_causal` and {k,q}_{start,end}."""
  if mask is None:
    return None, False, None, None, None, None

  num_heads_q, num_heads_k = q.shape[-2], k.shape[-2]
  is_causal = False
  q_start = None
  q_end = None
  k_start = None
  k_end = None

  if k_indices is None:
    mask, is_causal, k_start, k_end = mask.take("is_causal", "k_start", "k_end")
    if q_indices is None:
      if _is_per_batch_scalar(mask.q_start):
        mask, q_start = mask.take("q_start")
        q_start = _broadcast_to_rank(q_start, 2)
      if _is_per_batch_scalar(mask.q_end):
        mask, q_end = mask.take("q_end")
        q_end = _broadcast_to_rank(q_end, 2)

    # Fold `is_causal` into `k_end`. If `q_indices` is not `None`, then this is
    # necessary for correctness. Otherwise, it is a performance optimization.
    if is_causal and (q_indices is not None or k_end is not None):
      if q_indices is None:
        q_indices = jnp.arange(q.shape[-3])
      k_end_ = q_indices + 1
      if k_end is not None and any(
          x.ndim >= 2 and x.shape[-2] == num_heads_q for x in (k_end, k_end_)
      ):
        k_end = base.repeat_kv_heads(k_end, -2, num_heads_q, num_heads_k)
        k_end_ = base.repeat_kv_heads(k_end_, -2, num_heads_q, num_heads_k)
      k_end = k_end_ if k_end is None else jnp.minimum(k_end, k_end_)
      is_causal = False

    k_start = _broadcast_to_rank(k_start, 2)
    k_end = _broadcast_to_rank(k_end, 2)

  return (
      base.mask_as_array(mask, q, k, q_indices, k_indices),
      is_causal,
      q_start,
      q_end,
      k_start,
      k_end,
  )


def _can_have_block_d(*args):
  for arg in args:
    if isinstance(arg, QArray) and any(
        s not in (1, v) for v, s in zip(arg.qvalue.shape, arg.scale.shape)
    ):
      return False  # TODO: Make block_d work with subchannel quant
    # TODO: Make block_d work with non-pow2 head_dims.
    if triton.next_power_of_2(arg.shape[-1]) != arg.shape[-1]:
      return False
  return True


_PowerOfTwoGe16 = Annotated[pydantic_lib.PowerOfTwo, pydantic.Field(ge=16)]


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_q: pydantic_lib.PowerOfTwo
  block_k: _PowerOfTwoGe16
  num_stages: pydantic.PositiveInt
  num_warps: pydantic_lib.PowerOfTwo
  block_d: _PowerOfTwoGe16 | None = None
  block_d_out: _PowerOfTwoGe16 | None = None
  split_k: pydantic.PositiveInt = 1
  pack_mask: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class TritonFlashAttention(base.DotProductAttention[Config, None]):
  """Triton FlashAttention implementation."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  use_base2: bool = False
  use_stable_softmax: bool | type[base.AUTO] = base.AUTO

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(self, "vjp", vjp.TritonFlashAttentionVjp())

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array | QArray, "*B T H D"],
      k: Float[Array | QArray, "*B t h D"],
      v: Float[Array | QArray, "*B t h d"],
      *,
      precision: tuple[base.CanonicalPrecision, base.CanonicalPrecision],
      logits_dtype: jnp.dtype,
      logits_scale: float,
      bias: Float[Array, "*#B #H #T #t"] | None,
      logits_soft_cap: float | None,
      mask: Mask,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
      dropout_rate: float,
      paging_info: PagingInfo | None,
      q_indices: Int[Array, "*#B #H T"] | None,
      k_indices: Int[Array, "*#B #h t"] | None,
      normalize_output: bool,
      return_residuals: bool,
      config: Config,
  ) -> tuple[Float[Array, "*B T H d"], Residuals | None]:
    if paging_info is not None:
      raise NotImplementedError("Paged attention not supported.")

    block_d, block_d_out = config.block_d, config.block_d_out
    if not _can_have_block_d(q, k, v) and not (block_d == block_d_out == None):
      raise ValueError("block_d[_out] unsupported for subchannel quantization.")

    mask, is_causal, q_start, q_end, k_start, k_end = _decompose_mask(
        mask, q, k, q_indices, k_indices
    )

    q_k_dot_precision, weights_v_dot_precision = precision
    q_k_dot_precision = precision_lib.to_dot_algorithm_preset(
        q.dtype, k.dtype, q_k_dot_precision
    )
    weights_v_dot_precision = precision_lib.to_dot_algorithm_preset(
        v.dtype, v.dtype, weights_v_dot_precision
    )

    use_stable_softmax = self.use_stable_softmax
    if use_stable_softmax is base.AUTO:
      wv_dtype, _ = triton_utils.get_dot_dtype_and_precision(
          weights_v_dot_precision, v.dtype, v.dtype
      )
      # Unshifted `exp(s)` can overflow float16 weights even with a soft cap.
      use_stable_softmax = (
          base.needs_stable_softmax(logits_dtype, logits_soft_cap)
          or wv_dtype == tl.float16
      )

    f = functools.partial(
        _fwd,
        is_causal=is_causal,
        config=config,
        use_base2=self.use_base2,
        use_stable_softmax=use_stable_softmax,  # pyrefly: ignore[bad-argument-type]
        return_residuals=return_residuals,
        dropout_rate=dropout_rate,
        logits_dtype=logits_dtype,
        logits_scale=logits_scale,
        logits_soft_cap=logits_soft_cap,
        q_k_dot_precision=q_k_dot_precision,
        weights_v_dot_precision=weights_v_dot_precision,
        normalize_output=normalize_output,
    )

    bias = _broadcast_to_rank(bias, q.ndim)
    mask = _broadcast_to_rank(mask, q.ndim)
    dropout_mask = _broadcast_to_rank(dropout_mask, q.ndim)
    q_start = _broadcast_to_rank(q_start, q.ndim - 1)
    q_end = _broadcast_to_rank(q_end, q.ndim - 1)
    k_start = _broadcast_to_rank(k_start, q.ndim - 1)
    k_end = _broadcast_to_rank(k_end, q.ndim - 1)

    if (split_k := config.split_k) > 1:
      # TODO: Handle `is_causal`.
      if is_causal:
        raise ValueError("`is_causal` not supported with `split_k` > 1.")
      seq_len_k = k.shape[-3]
      if seq_len_k % split_k != 0:
        raise ValueError(
            f"`seq_len_k` ({seq_len_k}) must be divisible by `split_k`"
            f" ({split_k})."
        )

      # `k_{start,end}` hold global key indices, so give each part a copy
      # relative to its chunk, concatenated along the split axis.
      offsets = jnp.arange(split_k) * (seq_len_k // split_k)

      def per_part(x):
        if x is None:
          return None
        return (x[..., None, :] - offsets[:, None]).reshape(*x.shape[:-1], -1)

      k_start, k_end = per_part(k_start), per_part(k_end)

      # Perform `split_k` using `vmap`.
      seq_k_axes = (None, -3, -3, -1, -1, -1, -1, -1, -1, -1)
      f = functools.partial(f, normalize_output=False, return_residuals=True)
      f = batching.vmap_split(f, in_axes=seq_k_axes, num_parts=split_k)
      combine_partial_results = functools.partial(
          base.combine_partial_results, normalize_output=normalize_output
      )
      f = lambda *args, f=f: combine_partial_results(*f(*args))  # pyrefly: ignore[bad-argument-type]

    out, residuals = base.vmap_batch_dims(f)(
        q, k, v, bias, mask, dropout_mask, q_start, q_end, k_start, k_end  # pyrefly: ignore[bad-argument-type]
    )
    return out.astype(q.dtype), (residuals if return_residuals else None)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    q, k, v = ba.args[:3]
    *_, seq_len_q, _, head_dim = q.shape
    seq_len_k = k.shape[-3]

    min_block, max_block = 16, 64
    block_q = min(max(triton.next_power_of_2(seq_len_q), min_block), max_block)
    block_k = min(max(triton.next_power_of_2(seq_len_k), min_block), max_block)
    block_d = None
    num_warps, num_stages = 4, 2

    devices = op.infer_devices(ba) or {backend.get_default_device()}
    is_sm80 = any(map(gpu_utils.is_sm80, devices))

    if is_sm80:
      head_dim_bytes = jnp.dtype(q.dtype).itemsize * head_dim
      if head_dim_bytes >= 512:
        num_stages = 1

      if _can_have_block_d(q, k, v):
        block_qk = block_q * block_k
        block_d = max(
            16, triton.next_power_of_2(min(head_dim, 64**3 // block_qk))
        )

    return Config(
        block_q=block_q,
        block_k=block_k,
        block_d=block_d,
        block_d_out=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    q, k, _ = ba.args
    clamp = lambda lo, x, hi: max(lo, min(x, hi))
    configs = set()
    for block_q in [16, 32, 64, 128]:
      for block_k in [16, 32, 64, 128]:
        for num_warps in [1, 2, 4, 8]:
          for num_stages in [1, 2, 3, 4]:
            configs.add(
                Config(
                    block_q=clamp(
                        16, block_q, triton.next_power_of_2(q.shape[-3])
                    ),
                    block_k=clamp(
                        16, block_k, triton.next_power_of_2(k.shape[-3])
                    ),
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)
