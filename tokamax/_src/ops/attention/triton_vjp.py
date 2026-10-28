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
"""Triton FlashAttention VJP implementation."""

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
from tokamax._src import gpu_utils
from tokamax._src import jaxtyping
from tokamax._src import precision as precision_lib
from tokamax._src import pydantic as pydantic_lib
from tokamax._src import triton_utils
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
import triton
import triton.language as tl

Mask = base.Mask
Residuals = base.Residuals
PagingInfo = base.PagingInfo

_F32_TINY = tl.constexpr(float(jnp.finfo(jnp.float32).tiny))
# Tiles on the causal diagonal are processed in slices this many times smaller.
# This skips fully masked elements only when `block_m1 > block_n1` (for dK/dV)
# or `block_n2 > block_m2` (for dQ).
_MASK_BLOCK_SLICE_FACTOR = tl.constexpr(2)


@triton.jit
def _dot(a, b, input_precision, input_dtype):
  """Casts both operands to `input_dtype` before `tl.dot`."""
  return tl.dot(
      a.to(input_dtype), b.to(input_dtype), input_precision=input_precision
  )


@triton.jit
def _bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    mask_ptr,
    q_start_ptr,
    q_end_ptr,
    k_start_ptr,
    k_end_ptr,
    m_ptr,
    l_ptr,
    delta_ptr,
    dout_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    ds_ptr,
    batch_shape: tl.constexpr,
    batch_strides: tl.constexpr,
    seq_len_q: tl.constexpr,
    num_heads_q: tl.constexpr,
    head_dim: tl.constexpr,
    seq_len_k: tl.constexpr,
    num_heads_k: tl.constexpr,
    head_dim_out: tl.constexpr,
    head_dim_pow2: tl.constexpr,
    head_dim_out_pow2: tl.constexpr,
    block_m1: tl.constexpr,
    block_n1: tl.constexpr,
    block_m2: tl.constexpr,
    block_n2: tl.constexpr,
    bias_shape: tl.constexpr,
    mask_shape: tl.constexpr,
    k_start_shape: tl.constexpr,
    k_end_shape: tl.constexpr,
    sm_scale: tl.constexpr,
    is_causal: tl.constexpr,
    logits_dtype: tl.constexpr,
    qk_prec: tl.constexpr,
    qk_dtype: tl.constexpr,
    wv_prec: tl.constexpr,
    wv_dtype: tl.constexpr,
):
  """Triton MHA backward kernel."""
  grid_dkdv: tl.constexpr = tl.cdiv(seq_len_k, block_n1)
  grid_dq: tl.constexpr = tl.cdiv(seq_len_q, block_m2)
  grid_0: tl.constexpr = tl.constexpr(
      grid_dkdv if grid_dkdv >= grid_dq else grid_dq
  )
  pid_1d = tl.program_id(0)
  pid = (pid_1d % grid_0).to(tl.int32)
  pid_h = (pid_1d // grid_0 % num_heads_q).to(tl.int64)
  pid_b = (
      tl.full((), 0, tl.int64)
      if batch_shape == ()
      else (pid_1d // (grid_0 * num_heads_q)).to(tl.int64)
  )
  pid_h_kv = pid_h // (num_heads_q // num_heads_k)

  (
      stride_qb,
      stride_kb,
      stride_vb,
      stride_bias_b,
      stride_mask_b,
      stride_q_start_b,
      stride_q_end_b,
      stride_k_start_b,
      stride_k_end_b,
      stride_m_b,
      stride_l_b,
      stride_delta_b,
      stride_dout_b,
  ) = batch_strides

  q_ptr += triton_utils.batch_offset(pid_b, batch_shape, stride_qb) + (
      pid_h * head_dim
  )
  k_ptr += triton_utils.batch_offset(pid_b, batch_shape, stride_kb) + (
      pid_h_kv * head_dim
  )
  v_ptr += triton_utils.batch_offset(pid_b, batch_shape, stride_vb) + (
      pid_h_kv * head_dim_out
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
  q_start = 0
  if q_start_ptr is not None:
    q_start_ptr += triton_utils.batch_offset(
        pid_b, batch_shape, stride_q_start_b
    )
    q_start = tl.load(q_start_ptr)
  q_end = 0
  if q_end_ptr is not None:
    q_end_ptr += triton_utils.batch_offset(pid_b, batch_shape, stride_q_end_b)
    q_end = tl.load(q_end_ptr)
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
  m_ptr += triton_utils.batch_offset(pid_b, batch_shape, stride_m_b) + (
      pid_h * seq_len_q
  )
  l_ptr += triton_utils.batch_offset(pid_b, batch_shape, stride_l_b) + (
      pid_h * seq_len_q
  )
  delta_ptr += triton_utils.batch_offset(pid_b, batch_shape, stride_delta_b) + (
      pid_h * seq_len_q
  )
  dout_ptr += triton_utils.batch_offset(pid_b, batch_shape, stride_dout_b) + (
      pid_h * head_dim_out
  )

  dq_ptr += pid_b * (seq_len_q * num_heads_q * head_dim) + pid_h * head_dim
  dk_ptr += pid_b * (seq_len_k * num_heads_q * head_dim) + pid_h * head_dim
  dv_ptr += (
      pid_b * (seq_len_k * num_heads_q * head_dim_out) + pid_h * head_dim_out
  )
  if ds_ptr is not None:
    ds_ptr += pid_b * (num_heads_q * seq_len_q * seq_len_k) + (
        pid_h * seq_len_q * seq_len_k
    )

  offs_d = tl.arange(0, head_dim_pow2)
  offs_d_out = tl.arange(0, head_dim_out_pow2)
  check_d: tl.constexpr = head_dim != head_dim_pow2
  check_d_out: tl.constexpr = head_dim_out != head_dim_out_pow2
  q_stride_m: tl.constexpr = num_heads_q * head_dim
  do_stride_m: tl.constexpr = num_heads_q * head_dim_out
  k_stride_n: tl.constexpr = num_heads_k * head_dim
  v_stride_n: tl.constexpr = num_heads_k * head_dim_out
  num_phases: tl.constexpr = tl.constexpr(2 if is_causal else 1)
  # Slicing must not make tiles narrower than 16, the `tl.dot` minimum.
  min_sliced_block: tl.constexpr = 16 * _MASK_BLOCK_SLICE_FACTOR
  slice_m1: tl.constexpr = tl.constexpr(
      1 if block_m1 < min_sliced_block else _MASK_BLOCK_SLICE_FACTOR
  )
  slice_n2: tl.constexpr = tl.constexpr(
      1 if block_n2 < min_sliced_block else _MASK_BLOCK_SLICE_FACTOR
  )
  qk_compute_dtype = tl.float64 if qk_dtype == tl.float64 else tl.float32

  start_n = pid * block_n1
  if start_n < seq_len_k:
    offs_n1 = start_n + tl.arange(0, block_n1)
    check_n1: tl.constexpr = seq_len_k % block_n1 != 0
    k_mask1 = triton_utils.mask_2d(
        offs_n1, seq_len_k, offs_d, head_dim, check_n1, check_d
    )
    k1 = tl.load(
        k_ptr + offs_n1[:, None].to(tl.int64) * k_stride_n + offs_d[None, :],
        mask=k_mask1,
        other=None if k_mask1 is None else 0.0,
    )
    k1 = (k1.to(qk_compute_dtype) * sm_scale).to(qk_dtype)

    v_mask1 = triton_utils.mask_2d(
        offs_n1, seq_len_k, offs_d_out, head_dim_out, check_n1, check_d_out
    )
    v1 = tl.load(
        v_ptr
        + offs_n1[:, None].to(tl.int64) * v_stride_n
        + offs_d_out[None, :],
        mask=v_mask1,
        other=None if v_mask1 is None else 0.0,
    )

    dv = tl.zeros((block_n1, head_dim_out_pow2), dtype=tl.float32)
    dk = tl.zeros((block_n1, head_dim_pow2), dtype=tl.float32)

    block_m1_mask: tl.constexpr = block_m1 // slice_m1
    hi_mask_max1 = tl.cdiv(seq_len_q, block_m1_mask)
    lo_mask1 = tl.minimum(start_n // block_m1_mask, hi_mask_max1)
    lo_full1 = tl.cdiv(start_n + block_n1, block_m1)
    hi_mask1 = tl.minimum(lo_full1 * slice_m1, hi_mask_max1)

     # Static since each phase needs its own compile-time tile size and mask.
    for phase in tl.static_range(num_phases):
      lo1 = (
          lo_mask1
          if is_causal and phase == 0
          else (lo_full1 if is_causal else 0)
      )
      hi1 = (
          hi_mask1 if is_causal and phase == 0 else tl.cdiv(seq_len_q, block_m1)
      )

      for i in range(lo1, hi1):
        # A `tl.constexpr` can't be reassigned, and only this loop (not the
        # unrolled phase loop) gets a fresh scope for each phase.
        masked1: tl.constexpr = tl.constexpr(is_causal and phase == 0)
        cur_block_m1: tl.constexpr = tl.constexpr(
            block_m1_mask if masked1 else block_m1
        )
        check_m1: tl.constexpr = seq_len_q % cur_block_m1 != 0
        offs_m1 = i * cur_block_m1 + tl.arange(0, cur_block_m1)
        mask_m1_1d = (offs_m1 < seq_len_q) if check_m1 else None

        q_mask1 = triton_utils.mask_2d(
            offs_m1, seq_len_q, offs_d, head_dim, check_m1, check_d
        )
        q1 = tl.load(
            q_ptr
            + offs_m1[:, None].to(tl.int64) * q_stride_m
            + offs_d[None, :],
            mask=q_mask1,
            other=None if q_mask1 is None else 0.0,
        )
        m1 = tl.load(
            m_ptr + offs_m1,
            mask=mask_m1_1d,
            other=None if mask_m1_1d is None else float("inf"),
        )
        l1 = tl.load(
            l_ptr + offs_m1,
            mask=mask_m1_1d,
            other=None if mask_m1_1d is None else _F32_TINY,
        )
        sT = _dot(k1, tl.trans(q1), qk_prec, qk_dtype).to(logits_dtype)

        if bias_ptr is not None:
          b1 = triton_utils.load_2d_bcast(
              bias_ptr, offs_m1, offs_n1, bias_shape
          ).to(logits_dtype)
          sT += tl.trans(b1)
        sT = sT.to(tl.float32)

        if mask_ptr is not None:
          mask_val1 = triton_utils.load_2d_bcast(
              mask_ptr, offs_m1, offs_n1, mask_shape, 0
          )
          sT = tl.where(tl.trans(mask_val1) != 0, sT, float("-inf"))

        if q_start_ptr is not None:
          sT = tl.where(offs_m1[None, :] >= q_start, sT, float("-inf"))
        if q_end_ptr is not None:
          sT = tl.where(offs_m1[None, :] < q_end, sT, float("-inf"))
        if k_start_ptr is not None:
          k_start1 = triton_utils.load_1d_bcast(
              k_start_ptr, offs_m1, k_start_shape, other=seq_len_k
          )
          sT = tl.where(
              offs_n1[:, None] >= k_start1[None, :], sT, float("-inf")
          )
        if k_end_ptr is not None:
          k_end1 = triton_utils.load_1d_bcast(
              k_end_ptr, offs_m1, k_end_shape, other=0
          )
          sT = tl.where(offs_n1[:, None] < k_end1[None, :], sT, float("-inf"))

        if masked1:
          sT = tl.where(offs_m1[None, :] >= offs_n1[:, None], sT, float("-inf"))

        if check_n1:
          sT = tl.where((offs_n1 < seq_len_k)[:, None], sT, float("-inf"))

        pT = tl.math.exp(sT - m1[None, :]) / l1[None, :]
        # Rows that saw no keys in the forward pass are left with `l == tiny`.
        pT = tl.where((l1 == _F32_TINY)[None, :], 0.0, pT)
        do_mask1 = triton_utils.mask_2d(
            offs_m1, seq_len_q, offs_d_out, head_dim_out, check_m1, check_d_out
        )
        do1 = tl.load(
            dout_ptr
            + offs_m1[:, None].to(tl.int64) * do_stride_m
            + offs_d_out[None, :],
            mask=do_mask1,
            other=None if do_mask1 is None else 0.0,
        )
        dv += _dot(pT, do1, wv_prec, wv_dtype).to(tl.float32)
        delta1 = tl.load(
            delta_ptr + offs_m1,
            mask=mask_m1_1d,
            other=None if mask_m1_1d is None else 0.0,
        )
        dpT = (
            _dot(v1, tl.trans(do1), wv_prec, wv_dtype).to(tl.float32)
            - delta1[None, :]
        )
        dsT = pT * dpT

        if ds_ptr is not None:
          ds_mask = triton_utils.mask_2d(
              offs_m1, seq_len_q, offs_n1, seq_len_k, check_m1, check_n1
          )
          tl.store(
              ds_ptr
              + offs_m1[:, None].to(tl.int64) * seq_len_k
              + offs_n1[None, :],
              tl.trans(dsT).to(ds_ptr.dtype.element_ty),
              mask=ds_mask,
          )

        dk += _dot(dsT, q1, qk_prec, qk_dtype).to(tl.float32)

    if is_causal and ds_ptr is not None:
      check_m_zero: tl.constexpr = seq_len_q % block_m1_mask != 0
      z = tl.zeros((block_m1_mask, block_n1), dtype=ds_ptr.dtype.element_ty)
      for i in range(0, lo_mask1):
        offs_m_zero = i * block_m1_mask + tl.arange(0, block_m1_mask)
        mask_zero = triton_utils.mask_2d(
            offs_m_zero, seq_len_q, offs_n1, seq_len_k, check_m_zero, check_n1
        )
        tl.store(
            ds_ptr
            + offs_m_zero[:, None].to(tl.int64) * seq_len_k
            + offs_n1[None, :],
            z,
            mask=mask_zero,
        )

    tl.store(
        dv_ptr
        + offs_n1[:, None].to(tl.int64) * do_stride_m
        + offs_d_out[None, :],
        dv.to(dv_ptr.dtype.element_ty),
        mask=v_mask1,
    )
    tl.store(
        dk_ptr + offs_n1[:, None].to(tl.int64) * q_stride_m + offs_d[None, :],
        (dk * sm_scale).to(dk_ptr.dtype.element_ty),
        mask=k_mask1,
    )

  start_m = pid * block_m2
  if start_m < seq_len_q:
    offs_m2 = start_m + tl.arange(0, block_m2)
    check_m2: tl.constexpr = seq_len_q % block_m2 != 0
    mask_m2_1d = (offs_m2 < seq_len_q) if check_m2 else None
    q_mask2 = triton_utils.mask_2d(
        offs_m2, seq_len_q, offs_d, head_dim, check_m2, check_d
    )
    q2 = tl.load(
        q_ptr + offs_m2[:, None].to(tl.int64) * q_stride_m + offs_d[None, :],
        mask=q_mask2,
        other=None if q_mask2 is None else 0.0,
    )
    q2 = (q2.to(qk_compute_dtype) * sm_scale).to(qk_dtype)

    do_mask2 = triton_utils.mask_2d(
        offs_m2, seq_len_q, offs_d_out, head_dim_out, check_m2, check_d_out
    )
    do2 = tl.load(
        dout_ptr
        + offs_m2[:, None].to(tl.int64) * do_stride_m
        + offs_d_out[None, :],
        mask=do_mask2,
        other=None if do_mask2 is None else 0.0,
    )
    m2 = tl.load(
        m_ptr + offs_m2,
        mask=mask_m2_1d,
        other=None if mask_m2_1d is None else float("inf"),
    )[:, None]
    l2 = tl.load(
        l_ptr + offs_m2,
        mask=mask_m2_1d,
        other=None if mask_m2_1d is None else _F32_TINY,
    )[:, None]
    delta2 = tl.load(
        delta_ptr + offs_m2,
        mask=mask_m2_1d,
        other=None if mask_m2_1d is None else 0.0,
    )[:, None]
    k_start2 = 0
    if k_start_ptr is not None:
      k_start2 = triton_utils.load_1d_bcast(
          k_start_ptr, offs_m2, k_start_shape, other=seq_len_k
      )[:, None]
    k_end2 = seq_len_k
    if k_end_ptr is not None:
      k_end2 = triton_utils.load_1d_bcast(
          k_end_ptr, offs_m2, k_end_shape, other=0
      )[:, None]

    dq = tl.zeros((block_m2, head_dim_pow2), dtype=tl.float32)

    block_n2_mask: tl.constexpr = block_n2 // slice_n2
    hi_full2 = tl.cdiv(seq_len_k, block_n2)
    lo_mask2 = (start_m // block_n2) * slice_n2
    hi_mask2 = tl.minimum(
        tl.cdiv(start_m + block_m2, block_n2_mask),
        tl.cdiv(seq_len_k, block_n2_mask),
    )

    # Static since each phase needs its own compile-time tile size and mask.
    for phase in tl.static_range(num_phases):
      lo2 = lo_mask2 if is_causal and phase == 0 else 0
      hi2 = (
          hi_mask2
          if is_causal and phase == 0
          else (
              tl.minimum(start_m // block_n2, hi_full2)
              if is_causal
              else hi_full2
          )
      )

      for i in range(lo2, hi2):
        masked2: tl.constexpr = tl.constexpr(is_causal and phase == 0)
        cur_block_n2: tl.constexpr = tl.constexpr(
            block_n2_mask if masked2 else block_n2
        )
        check_n2: tl.constexpr = seq_len_k % cur_block_n2 != 0
        offs_n2 = i * cur_block_n2 + tl.arange(0, cur_block_n2)

        k_mask2 = triton_utils.mask_2d(
            offs_n2, seq_len_k, offs_d, head_dim, check_n2, check_d
        )
        k2 = tl.load(
            k_ptr
            + offs_n2[:, None].to(tl.int64) * k_stride_n
            + offs_d[None, :],
            mask=k_mask2,
            other=None if k_mask2 is None else 0.0,
        )
        v_mask2 = triton_utils.mask_2d(
            offs_n2, seq_len_k, offs_d_out, head_dim_out, check_n2, check_d_out
        )
        v2 = tl.load(
            v_ptr
            + offs_n2[:, None].to(tl.int64) * v_stride_n
            + offs_d_out[None, :],
            mask=v_mask2,
            other=None if v_mask2 is None else 0.0,
        )
        s = _dot(q2, tl.trans(k2), qk_prec, qk_dtype).to(logits_dtype)

        if bias_ptr is not None:
          s += triton_utils.load_2d_bcast(
              bias_ptr, offs_m2, offs_n2, bias_shape
          ).to(logits_dtype)
        s = s.to(tl.float32)

        if mask_ptr is not None:
          mask_val2 = triton_utils.load_2d_bcast(
              mask_ptr, offs_m2, offs_n2, mask_shape, 0
          )
          s = tl.where(mask_val2 != 0, s, float("-inf"))

        if q_start_ptr is not None:
          s = tl.where(offs_m2[:, None] >= q_start, s, float("-inf"))
        if q_end_ptr is not None:
          s = tl.where(offs_m2[:, None] < q_end, s, float("-inf"))
        if k_start_ptr is not None:
          s = tl.where(offs_n2[None, :] >= k_start2, s, float("-inf"))
        if k_end_ptr is not None:
          s = tl.where(offs_n2[None, :] < k_end2, s, float("-inf"))

        if masked2:
          s = tl.where(offs_m2[:, None] >= offs_n2[None, :], s, float("-inf"))

        if check_n2:
          s = tl.where((offs_n2 < seq_len_k)[None, :], s, float("-inf"))

        p = tl.math.exp(s - m2) / l2
        p = tl.where(l2 == _F32_TINY, 0.0, p)

        dp = _dot(do2, tl.trans(v2), wv_prec, wv_dtype).to(tl.float32) - delta2
        ds = p * dp
        dq += _dot(ds, k2, qk_prec, qk_dtype).to(tl.float32)

    tl.store(
        dq_ptr + offs_m2[:, None].to(tl.int64) * q_stride_m + offs_d[None, :],
        (dq * sm_scale).to(dq_ptr.dtype.element_ty),
        mask=q_mask2,
    )


_PowerOfTwoGe16 = Annotated[pydantic_lib.PowerOfTwo, pydantic.Field(ge=16)]


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_m1: _PowerOfTwoGe16
  block_n1: _PowerOfTwoGe16
  block_m2: _PowerOfTwoGe16
  block_n2: _PowerOfTwoGe16
  num_stages: pydantic.PositiveInt
  num_warps: pydantic_lib.PowerOfTwo


@jaxtyping.jaxtyped
def _bwd(
    q: Float[Array, "T H D"],
    k: Float[Array, "t h D"],
    v: Float[Array, "t h d"],
    bias: Float[Array, "#H #T #t"] | None,
    mask: Bool[Array, "#H #T #t"] | Bool[Array, "#h #T #t"] | None,
    dropout_mask: Bool[Array, "#H #T #t"] | None,
    q_start: Int[Array, "1 1"] | None,
    q_end: Int[Array, "1 1"] | None,
    k_start: Int[Array, "#H #T"] | Int[Array, "#h #T"] | None,
    k_end: Int[Array, "#H #T"] | Int[Array, "#h #T"] | None,
    residuals: Residuals,
    out: Float[Array, "T H d"],
    dout: Float[Array, "T H d"],
    *,
    q_k_dot_precision: jax.lax.DotAlgorithmPreset,
    logits_dtype: jnp.dtype,
    logits_scale: float,
    logits_soft_cap: float | None,
    is_causal: bool,
    dropout_rate: float,
    weights_v_dot_precision: jax.lax.DotAlgorithmPreset,
    ds_dtype: jax.typing.DTypeLike | None,
    config: Config,
) -> tuple[
    Float[Array, "T H D"],  # dq
    Float[Array, "t h D"],  # dk
    Float[Array, "t h d"],  # dv
    Float[Array, "H T t"] | None,  # ds
]:
  """Compute FlashAttention VJP."""
  del dropout_rate  # Unused.

  if dropout_mask is not None:
    raise NotImplementedError("`dropout_mask` unsupported.")
  if logits_soft_cap is not None:
    raise NotImplementedError("`logits_soft_cap` unsupported.")

  seq_len_q, num_heads_q, head_dim = q.shape
  seq_len_k, num_heads_k, head_dim_out = v.shape

  m, l = residuals
  delta = jnp.einsum(
      "qhd,qhd->hq", out, dout, preferred_element_type=jnp.float32
  )

  head_dim_pow2 = max(16, triton.next_power_of_2(head_dim))
  head_dim_out_pow2 = max(16, triton.next_power_of_2(head_dim_out))

  qk_dtype, qk_prec = triton_utils.get_dot_dtype_and_precision(
      q_k_dot_precision, q.dtype, k.dtype
  )
  wv_dtype, wv_prec = triton_utils.get_dot_dtype_and_precision(
      weights_v_dot_precision, v.dtype, v.dtype
  )
  logits_tl_dtype = triton_utils.jnp_to_tl_dtype(logits_dtype)

  grid_dkdv = triton.cdiv(seq_len_k, config.block_n1)
  grid_dq = triton.cdiv(seq_len_q, config.block_m2)

  @triton_utils.vmap_kernel
  def bwd_call(
      batch_shape,
      batch_strides,
      q,
      k,
      v,
      bias,
      mask,
      q_start,
      q_end,
      k_start,
      k_end,
      m,
      l,
      delta,
      dout,
  ):
    dq_shape = jax.ShapeDtypeStruct(
        (*batch_shape, seq_len_q, num_heads_q, head_dim), q.dtype
    )
    dk_shape = jax.ShapeDtypeStruct(
        (*batch_shape, seq_len_k, num_heads_q, head_dim), k.dtype
    )
    dv_shape = jax.ShapeDtypeStruct(
        (*batch_shape, seq_len_k, num_heads_q, head_dim_out), v.dtype
    )
    if bias is None:
      ds_shape_dtype = None
      none_outs: dict[str, Any] = {"ds_ptr": None}
    else:
      ds_shape_dtype = jax.ShapeDtypeStruct(
          (*batch_shape, num_heads_q, seq_len_q, seq_len_k), ds_dtype
      )
      none_outs = {}
    out_type = (dq_shape, dk_shape, dv_shape, ds_shape_dtype)
    return jt.triton_call(
        q,
        k,
        v,
        bias,
        mask,
        q_start,
        q_end,
        k_start,
        k_end,
        m,
        l,
        delta,
        dout,
        kernel=_bwd_kernel,
        out_type=out_type,
        grid=(math.prod(batch_shape) * num_heads_q * max(grid_dkdv, grid_dq),),
        name="triton_flash_attention_vjp",
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
        head_dim_pow2=head_dim_pow2,
        head_dim_out_pow2=head_dim_out_pow2,
        block_m1=config.block_m1,
        block_n1=config.block_n1,
        block_m2=config.block_m2,
        block_n2=config.block_n2,
        bias_shape=(1, 1, 1) if bias is None else bias.shape[-3:],
        mask_shape=(1, 1, 1) if mask is None else mask.shape[-3:],
        k_start_shape=(1, 1) if k_start is None else k_start.shape[-2:],
        k_end_shape=(1, 1) if k_end is None else k_end.shape[-2:],
        sm_scale=float(logits_scale),
        is_causal=is_causal,
        logits_dtype=logits_tl_dtype,
        qk_prec=qk_prec,
        qk_dtype=qk_dtype,
        wv_prec=wv_prec,
        wv_dtype=wv_dtype,
        **none_outs,
    )

  dq, dk, dv, ds = bwd_call(
      q, k, v, bias, mask, q_start, q_end, k_start, k_end, m, l, delta, dout
  )

  if num_heads_k < num_heads_q:
    sum_shape = dk.shape[:-2] + (num_heads_k, num_heads_q // num_heads_k, -1)
    dk = jnp.sum(dk.reshape(sum_shape), axis=-2)
    dv = jnp.sum(dv.reshape(sum_shape), axis=-2)

  return dq, dk, dv, ds


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
    elif is_causal:
      # The kernel's causal path assumes that query positions are row indices.
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


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class TritonFlashAttentionVjp(base.DotProductAttentionVjp[Config, None]):
  """Triton FlashAttention VJP implementation."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  dbias_intermediate_dtype: jax.typing.DTypeLike | None = None

  @override
  def _fwd(
      self,
      residuals: Residuals,
      out: Float[Array, "*B T H d"],
      dout: Float[Array, "*B T H d"],
      q: Float[Array, "*B T H D"],
      k: Float[Array, "*B t h D"],
      v: Float[Array, "*B t h d"],
      *,
      precision: tuple[base.CanonicalPrecision, base.CanonicalPrecision],
      logits_dtype: jnp.dtype,
      logits_scale: float,
      bias: Float[Array, "*#B #H #T #t"] | None,
      logits_soft_cap: float | None,
      mask: Mask | None,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
      dropout_rate: float,
      paging_info: PagingInfo | None,
      q_indices: Int[Array, "*#B #H T"] | None,
      k_indices: Int[Array, "*#B #h t"] | None,
      normalize_output: bool,
      return_residuals: bool,
      config: Config,
  ) -> tuple[base.DotProductAttentionGrads, None]:
    if not normalize_output:
      raise NotImplementedError("`normalize_output=False` not supported.")

    if return_residuals:
      raise NotImplementedError("`return_residuals` not supported.")

    if paging_info is not None:
      raise NotImplementedError("Paged attention not supported.")

    mask_, is_causal, q_start, q_end, k_start, k_end = _decompose_mask(
        mask, q, k, q_indices, k_indices
    )

    orig_bias_shape = None if bias is None else bias.shape
    bias = _broadcast_to_rank(bias, q.ndim)
    mask_ = _broadcast_to_rank(mask_, q.ndim)
    dropout_mask = _broadcast_to_rank(dropout_mask, q.ndim)
    q_start = _broadcast_to_rank(q_start, q.ndim - 1)
    q_end = _broadcast_to_rank(q_end, q.ndim - 1)
    k_start = _broadcast_to_rank(k_start, q.ndim - 1)
    k_end = _broadcast_to_rank(k_end, q.ndim - 1)

    if bias is None:
      ds_dtype = None
    elif self.dbias_intermediate_dtype is None:
      ds_dtype = bias.dtype
    elif bias.shape == (*q.shape[:-3], q.shape[-2], q.shape[-3], k.shape[-3]):
      ds_dtype = bias.dtype
    else:
      ds_dtype = self.dbias_intermediate_dtype

    q_k_dot_precision, weights_v_dot_precision = precision
    q_k_dot_precision = precision_lib.to_dot_algorithm_preset(
        q.dtype, k.dtype, q_k_dot_precision
    )
    weights_v_dot_precision = precision_lib.to_dot_algorithm_preset(
        v.dtype, v.dtype, weights_v_dot_precision
    )
    f = functools.partial(
        _bwd,
        is_causal=is_causal,
        config=config,
        dropout_rate=dropout_rate,
        logits_dtype=logits_dtype,
        logits_scale=logits_scale,
        logits_soft_cap=logits_soft_cap,
        q_k_dot_precision=q_k_dot_precision,
        weights_v_dot_precision=weights_v_dot_precision,
        ds_dtype=ds_dtype,
    )
    f = base.vmap_batch_dims(f)

    dq, dk, dv, ds = f(
        q,
        k,
        v,
        bias,
        mask_,
        dropout_mask,
        q_start,
        q_end,
        k_start,
        k_end,
        residuals,
        out,
        dout,
    )
    if bias is None:
      dbias = None
    else:
      assert ds is not None
      broadcast_bias_axes = [i for i, d in enumerate(bias.shape) if d == 1]
      dbias = jnp.sum(ds, axis=broadcast_bias_axes)
      dbias = dbias.astype(bias.dtype).reshape(orig_bias_shape)
    return base.DotProductAttentionGrads(q=dq, k=dk, v=dv, bias=dbias), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    # TODO: Implement heuristics.
    q = ba.arguments["q"]
    k = ba.arguments["k"]
    v = ba.arguments["v"]
    q_k_dot_precision, weights_v_dot_precision = ba.arguments["precision"]
    q_k_dot_dtype = precision_lib.precision_input_dtype(
        precision_lib.to_dot_algorithm_preset(
            q.dtype, k.dtype, q_k_dot_precision
        )
    )
    weights_v_dot_dtype = precision_lib.precision_input_dtype(
        precision_lib.to_dot_algorithm_preset(
            v.dtype, v.dtype, weights_v_dot_precision
        )
    )
    head_dim = max(q.shape[-1], v.shape[-1])
    max_itemsize = max(
        jnp.dtype(d).itemsize
        for d in (q.dtype, k.dtype, v.dtype, q_k_dot_dtype, weights_v_dot_dtype)
    )
    devices = op.infer_devices(ba) or {backend.get_default_device()}
    is_sm80 = any(map(gpu_utils.is_sm80, devices))
    head_dim_bytes = max_itemsize * triton.next_power_of_2(head_dim)
    num_stages = 1 if head_dim_bytes >= (512 if is_sm80 else 1024) else 2
    block_n1 = 32 if head_dim_bytes >= 1024 else 64
    return Config(
        block_m1=32,
        block_n1=block_n1,
        block_m2=block_n1,
        block_n2=32,
        num_warps=4,
        num_stages=num_stages,
    )

  # TODO: Implement an autotuning search space.
  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    configs = set()
    for block_m in [32, 64, 128, 256]:
      for block_n in [32, 64, 128, 256]:
        if block_n < block_m:
          continue
        for num_warps in [2, 4]:
          for num_stages in [1, 2, 3, 4]:
            configs.add(
                Config(
                    block_m1=block_m,
                    block_n1=block_n,
                    block_m2=block_n,
                    block_n2=block_m,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)
