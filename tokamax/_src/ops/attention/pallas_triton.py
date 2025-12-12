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
"""Pallas-Triton FlashAttention implementation."""

import dataclasses
import functools
from typing import ClassVar

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import pydantic
from tokamax._src import batching
from tokamax._src import gpu_utils
from tokamax._src import jaxtyping
from tokamax._src import pydantic as pydantic_lib
from tokamax._src import quantization
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_triton_vjp as vjp
from tokamax._src.pallas import block
from typing_extensions import override

Mask = base.Mask
QArray = base.QArray
Residuals = base.Residuals
PagingInfo = base.PagingInfo


def _fwd_kernel(q_start_ref, q_end_ref, q_ref, *args, **kwargs):
  """Pallas MHA forward kernel conditioned on q_start and q_end."""

  def fwd_kernel(q_start, q_end):
    _fwd_kernel_impl(q_start, q_end, q_ref, *args, **kwargs)

  if (q_start_ref, q_end_ref) == (None, None):
    fwd_kernel(None, None)
  else:

    def _zero_store():
      out_ref, l_ref, m_ref = args[-3:]  # pylint: disable=unbalanced-tuple-unpacking
      if l_ref is not None:
        l_ref.store(jnp.zeros(l_ref.shape, l_ref.dtype))
      if m_ref is not None:
        m_ref.store(jnp.zeros(m_ref.shape, m_ref.dtype))
      out_ref.store(jnp.zeros(out_ref.shape, out_ref.dtype))

    q_start, q_end, skip_cond = None, None, False
    if q_start_ref is not None:
      q_start = q_start_ref.load()
      skip_cond |= (jnp.min(q_start) // q_ref.shape[0]) > pl.program_id(1)
    if q_end_ref is not None:
      q_end = q_end_ref.load()
      skip_cond |= pl.cdiv(jnp.max(q_end), q_ref.shape[0]) <= pl.program_id(1)
    jax.lax.cond(skip_cond, _zero_store, lambda: fwd_kernel(q_start, q_end))


def _rescale(
    values, scales_ref, slice_k, slice_d, keep_quantized=False, const_scale=1.0
):
  """Load the scales correctly and apply them to the values if requested."""
  if scales_ref is None:
    return values * const_scale, None
  assert values is not None

  subchannel_quant = scales_ref.shape[1] not in (1, values.shape[1])

  if slice_k is None or scales_ref.shape[0] == 1:
    slice_k = slice(None)
  if slice_d is None or scales_ref.shape[1] == 1 or subchannel_quant:
    slice_d = slice(None)

  scales = scales_ref.at[slice_k, slice_d].load() * const_scale
  if subchannel_quant:
    scales = jnp.repeat(scales, values.shape[1] // scales.shape[1], axis=1)
  if keep_quantized:
    return values, scales
  return values.astype(scales.dtype) * scales, None


# Based on Algorithm 1 of https://arxiv.org/abs/2205.14135.
# Inspired by the official Triton tutorial implementation
# https://triton-lang.org/master/getting-started/tutorials/06-fused-attention.html
def _fwd_kernel_impl(
    q_start,
    q_end,
    q_ref,
    k_ref,
    v_ref,
    bias_ref,
    mask_ref,
    dropout_mask_ref,
    k_start_ref,
    k_end_ref,
    out_ref,
    l_ref,
    m_ref,
    block_k: int,
    block_d: int | None,
    block_d_out: int | None,
    sm_scale: float,
    is_causal: bool,
    dropout_rate: float,
    logits_dtype: jnp.dtype,
    logits_soft_cap: float | None,
    use_base2: bool,
    use_stable_softmax: bool,
    q_k_dot_precision: jax.lax.DotAlgorithmPreset,
    weights_v_dot_precision: jax.lax.DotAlgorithmPreset,
    normalize_output: bool,
    pack_mask: bool,
):
  """Pallas MHA forward kernel implementation."""

  def get_values_and_scales(x):
    if isinstance(x, QArray):
      assert x.zero_point is None
      return x.qvalue, x.scale
    return x, None

  q_ref, q_scales_ref = get_values_and_scales(q_ref)
  k_ref, k_scales_ref = get_values_and_scales(k_ref)
  v_ref, v_scales_ref = get_values_and_scales(v_ref)

  block_m = q_ref.shape[0]

  block_d = q_ref.shape[-1] if block_d is None else block_d
  split_d = q_ref.shape[-1] // block_d

  block_d_out = out_ref.shape[-1] if block_d_out is None else block_d_out
  split_d_out = out_ref.shape[-1] // block_d_out

  # TODO: For split_d > 1, try to use a for loop around the whole
  # kernel rather than having a list of tiles for acc and q. M and L will be
  # computed only in the first iteration.

  # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
  if use_stable_softmax:
    m_i = jnp.full([block_m], float("-inf"), dtype=jnp.float32)
  else:
    m_i = 0.0
  l_i = jnp.zeros([block_m], dtype=jnp.float32)
  accs = [jnp.zeros([block_m, block_d_out], dtype=jnp.float32)] * split_d_out

  start_m = pl.program_id(1) * block_m
  span_m = start_m + jnp.arange(block_m)

  quantize_qk_dot = (
      q_scales_ref is not None
      and k_scales_ref is not None
      and q_scales_ref.shape[1] == 1
      and k_scales_ref.shape[1] == 1
  )

  q = [None] * split_d
  q_scales = [None] * split_d

  for i in range(split_d):
    slice_d = block.ds(i, block_d)
    q[i] = q_ref.at[:, slice_d].load()
    q[i], q_scales[i] = _rescale(
        q[i], q_scales_ref, None, slice_d, quantize_qk_dot, sm_scale
    )

  k_start = None if k_start_ref is None else k_start_ref.load()
  k_end = None if k_end_ref is None else k_end_ref.load()

  def body(i, carry, *, is_causal=False, use_k_end=False):
    accs, m_i, l_i = carry

    start_k = i * block_k
    slice_k = pl.ds(start_k, block_k)
    span_k = start_k + jnp.arange(block_k)
    use_mask_k = k_ref.shape[0] % block_k != 0
    s = jnp.zeros([block_m, block_k], dtype=jnp.float32)

    for i in range(split_d):
      assert q[i] is not None
      slice_d = block.ds(i, block_d)
      k = k_ref.at[slice_k, slice_d].load()
      k, k_scales = _rescale(k, k_scales_ref, slice_k, slice_d, quantize_qk_dot)

      if quantize_qk_dot:
        s += pl.dot(q[i], k.T).astype(jnp.float32) * (q_scales[i] * k_scales.T)  # pytype: disable=attribute-error
      else:
        s += pl.dot(q[i].astype(k.dtype), k.T, precision=q_k_dot_precision)
    s = s.astype(logits_dtype)

    if bias_ref is not None:
      bias_slice_k = slice_k if (bias_ref.shape[-1] > 1) else slice(None)
      s += plgpu.load(bias_ref.at[:, bias_slice_k])

    if logits_soft_cap is not None:
      s = logits_soft_cap * jnp.tanh(s / logits_soft_cap)

    if use_base2:
      # NOTE: This rescaling must happen after the bias add and soft-cap, but
      # before the attention masking (as the multiplication will cause `-inf`s).
      # TODO: Scale `q` instead if no bias or tanh clipping (we could
      # scale `logits_soft_cap` by the same factor).
      s *= np.log2(np.e)

    # See base class for explanation of why this is not `-inf`.
    mask_value = float(jnp.finfo(jnp.float32).min)

    if mask_ref is not None:
      if pack_mask:
        mask_slice_k = pl.ds(start_k // 8, block_k // 8)
        mask = plgpu.load(mask_ref.at[:, mask_slice_k])
        mask = jnp.repeat(mask, 8, axis=-1)
        mask = mask & (1 << (jnp.arange(block_k) % 8))
      else:
        mask_slice_k = slice_k if (mask_ref.shape[-1] > 1) else slice(None)
        mask = plgpu.load(mask_ref.at[:, mask_slice_k])
      s = jnp.where(mask, s, mask_value)

    if q_start is not None:
      s = jnp.where(q_start[None, :] <= span_m[:, None], s, mask_value)
    if q_end is not None:
      s = jnp.where(q_end[None, :] > span_m[:, None], s, mask_value)
    if k_start is not None:
      s = jnp.where(k_start[:, None] <= span_k[None, :], s, mask_value)
    if use_k_end:
      s = jnp.where(k_end[:, None] > span_k[None, :], s, mask_value)
    if is_causal:
      s = jnp.where(span_m[:, None] >= span_k[None, :], s, mask_value)

    if use_mask_k:
      s = jnp.where((span_k < k_ref.shape[0])[None, :], s, float("-inf"))

    exp = jnp.exp2 if use_base2 else jnp.exp

    if use_stable_softmax:
      m_ij = jnp.maximum(m_i, jnp.max(s, axis=1))
      p = exp(s - m_ij[:, None])
      alpha = exp(m_i - m_ij)
      m_i = m_ij
      accs = [acc * alpha[:, None] for acc in accs]
      l_i *= alpha
    else:
      p = exp(s)
    l_i += jnp.sum(p, axis=1)

    if dropout_mask_ref is not None:
      dropout_mask_slice_k = (
          slice_k if (dropout_mask_ref.shape[-1] > 1) else slice(None)
      )
      dropout_mask = plgpu.load(dropout_mask_ref.at[:, dropout_mask_slice_k])
      p *= dropout_mask.astype(p.dtype) / (1 - dropout_rate)

    for i in range(split_d_out):
      slice_d = block.ds(i, block_d_out)
      v = v_ref.at[slice_k, slice_d].load(bounds_check=(True, False))
      v, _ = _rescale(v, v_scales_ref, slice_k, slice_d)
      accs[i] += pl.dot(p.astype(v.dtype), v, precision=weights_v_dot_precision)

    return accs, m_i, l_i

  num_k_blocks = pl.cdiv(k_ref.shape[0], block_k)
  lo, hi = 0, num_k_blocks

  if k_start is not None:
    lo = jnp.maximum(jnp.floor_divide(jnp.min(k_start), block_k), lo)

  if is_causal:
    # Run all but last iteration without performing causal masking.
    hi = jnp.minimum(jnp.floor_divide(start_m, block_k), num_k_blocks)
    accs, m_i, l_i = jax.lax.fori_loop(lo, hi, body, (accs, m_i, l_i))
    lo, hi = hi, jnp.minimum(pl.cdiv(start_m + block_m, block_k), num_k_blocks)
    body = functools.partial(body, is_causal=True)
  elif k_end is not None:
    hi = jnp.clip(jnp.min(k_end) // block_k, 0, num_k_blocks)
    accs, m_i, l_i = jax.lax.fori_loop(lo, hi, body, (accs, m_i, l_i))
    lo, hi = hi, jnp.clip(pl.cdiv(jnp.max(k_end), block_k), 0, num_k_blocks)
    body = functools.partial(body, use_k_end=True)

  accs, m_i, l_i = jax.lax.fori_loop(lo, hi, body, (accs, m_i, l_i))

  if normalize_output:
    # It is possible that every value in a row was masked to f32 min, and that
    # `l_i` is `0` for that row. Add epsilon value to avoid NaNs from `0 / 0`.
    l_i += float(jnp.finfo(jnp.float32).tiny)
    accs = [acc / l_i[:, None] for acc in accs]

  if l_ref is not None:
    l_ref.store(l_i)
  if m_ref is not None:
    m_ref.store(m_i)
  for i in range(split_d_out):
    out_ref.at[:, block.ds(i, block_d_out)].store(accs[i].astype(out_ref.dtype))


@jaxtyping.jaxtyped
def _fwd(
    q: Float[Array | QArray, "T H D"],
    k: Float[Array | QArray, "t h D"],
    v: Float[Array | QArray, "t h d"],
    bias: Float[Array, "#H #T #t"] | None,
    mask: Bool[Array, "#H #T #t"] | None,
    dropout_mask: Bool[Array, "#H #T #t"] | None,
    q_start: Int[Array, "#H #t"] | None,
    q_end: Int[Array, "#H #t"] | None,
    k_start: Int[Array, "#H #T"] | None,
    k_end: Int[Array, "#H #T"] | None,
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
  """Forward pass of Pallas FlashAttention."""
  q, k, v = map(quantization.as_array_or_qarray_without_zero_point, (q, k, v))

  seq_len_q, num_heads_q, head_dim = q.shape
  seq_len_k, num_heads_k, head_dim_out = v.shape

  pack_mask = config.pack_mask and mask is not None and mask.shape[-1] != 1
  if pack_mask:
    mask = jnp.packbits(mask, axis=-1, bitorder="little")

  kernel = functools.partial(
      _fwd_kernel,
      block_k=config.block_k,
      block_d=config.block_d,
      block_d_out=config.block_d_out,
      sm_scale=logits_scale,
      is_causal=is_causal,
      dropout_rate=dropout_rate,
      logits_dtype=logits_dtype,
      logits_soft_cap=logits_soft_cap,
      use_base2=use_base2,
      use_stable_softmax=use_stable_softmax,
      q_k_dot_precision=q_k_dot_precision,
      weights_v_dot_precision=weights_v_dot_precision,
      normalize_output=normalize_output,
      pack_mask=pack_mask,
  )

  block_q = config.block_q
  # TODO: Could these just be multiple of block_d[_out]?
  head_dim_pow2 = pl.next_power_of_2(head_dim)
  head_dim_out_pow2 = pl.next_power_of_2(head_dim_out)
  q_index_map = lambda i, j: (j, i, 0)
  q_block_shape = (block_q, None, head_dim_pow2)
  k_index_map = lambda i, j: (0, i // (num_heads_q // num_heads_k), 0)
  k_block_shape = (seq_len_k, None, head_dim_pow2)
  v_index_map = k_index_map
  v_block_shape = (seq_len_k, None, head_dim_out_pow2)

  def spec(x, index_map, block_shape):
    if x is None:
      return None

    def wrapped_index_map(*args):
      return [x if (d > 1) else 0 for x, d in zip(index_map(*args), x.shape)]

    block_shape = [
        bd if (bd is None or (d > 1)) else 1
        for bd, d in zip(block_shape, x.shape)
    ]
    return pl.BlockSpec(tuple(block_shape), wrapped_index_map)

  def input_spec(x, index_map, block_shape):
    if isinstance(x, QArray):
      assert x.zero_point is None
      scales_block_shape = [
          None if b is None else min(s, b)
          for s, b in zip(x.scale.shape, block_shape)
      ]
      return QArray(  # pytype: disable=wrong-arg-types
          pl.BlockSpec(block_shape, index_map),
          spec(x.scale, index_map, scales_block_shape),
          qtype=x.qvalue.dtype,
      )
    return pl.BlockSpec(block_shape, index_map)

  def bias_mask_spec(x):
    seq_k = x.shape[-1] if x is not None else None
    return spec(x, lambda i, j: (i, j, 0), (None, block_q, seq_k))

  def range_k_spec(x):
    return spec(x, lambda i, j: (i, j), (None, block_q))

  def range_q_spec(x):
    return spec(x, lambda i, j: (i, j), (None, seq_len_k))

  in_specs = (
      range_q_spec(q_start),
      range_q_spec(q_end),
      input_spec(q, q_index_map, q_block_shape),
      input_spec(k, k_index_map, k_block_shape),
      input_spec(v, v_index_map, v_block_shape),
      bias_mask_spec(bias),
      bias_mask_spec(mask),
      bias_mask_spec(dropout_mask),
      range_k_spec(k_start),
      range_k_spec(k_end),
  )

  out_spec = pl.BlockSpec((block_q, None, head_dim_out_pow2), q_index_map)
  out_shape = jax.ShapeDtypeStruct(q.shape[:-1] + (head_dim_out,), q.dtype)

  residual_spec = pl.BlockSpec((None, block_q), lambda i, j: (i, j))
  residual_shape = jax.ShapeDtypeStruct((num_heads_q, seq_len_q), jnp.float32)

  if return_residuals:
    l_spec = residual_spec
    m_spec = residual_spec if use_stable_softmax else None
    l_shape = residual_shape
    m_shape = residual_shape if use_stable_softmax else None
    name_suffix = "_fwd_res"
  else:
    l_spec = m_spec = None
    l_shape = m_shape = None
    name_suffix = ""

  out_specs = [out_spec, l_spec, m_spec]
  out_shapes = [out_shape, l_shape, m_shape]

  out, l, m = block.pallas_call(
      kernel,
      in_specs=in_specs,
      out_specs=out_specs,
      out_shape=out_shapes,
      name="pallas_flash_attention" + name_suffix,
      grid=(num_heads_q, pl.cdiv(seq_len_q, config.block_q)),
      compiler_params=plgpu.CompilerParams(
          num_stages=config.num_stages, num_warps=config.num_warps
      ),
  )(q_start, q_end, q, k, v, bias, mask, dropout_mask, k_start, k_end)

  if not return_residuals:
    return out, None

  return out, (jnp.zeros_like(l) if m is None else m, l)


def _broadcast_to_rank(x, rank):
  return None if x is None else jax.lax.broadcast_to_rank(x, rank)


def _decompose_mask(mask, q, k, q_indices, k_indices):
  """Decomposes `mask` into a mask array, `is_causal` and {k,q}_{start,end}."""
  if mask is None:
    return None, False, None, None, None, None

  is_causal = False
  q_start = None
  q_end = None
  k_start = None
  k_end = None

  if k_indices is None:
    mask, is_causal, k_start, k_end = mask.take("is_causal", "k_start", "k_end")
    if q_indices is None:
      if mask.q_start is not None and mask.q_start.size == 1:
        mask, q_start = mask.take("q_start")
        q_start = _broadcast_to_rank(q_start, 2)
      if mask.q_end is not None and mask.q_end.size == 1:
        mask, q_end = mask.take("q_end")
        q_end = _broadcast_to_rank(q_end, 2)

    # Fold `is_causal` into `k_end`. If `q_indices` is not `None`, then this is
    # necessary for correctness. Otherwise, it is a performance optimization.
    if is_causal and (q_indices is not None or k_end is not None):
      if q_indices is None:
        q_indices = jnp.arange(q.shape[-3])
      k_end_ = q_indices + 1
      k_end = k_end_ if k_end is None else jnp.minimum(k_end, k_end_)
      is_causal = False

    k_start = _broadcast_to_rank(k_start, 2)
    k_end = _broadcast_to_rank(k_end, 2)

  q_len_or_indices = q.shape[-3] if q_indices is None else q_indices
  k_len_or_indices = k.shape[-3] if k_indices is None else k_indices
  return (
      mask.as_array(q_len_or_indices, k_len_or_indices),
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
    if pl.next_power_of_2(arg.shape[-1]) != arg.shape[-1]:
      return False  # TODO: Make block_d work with non-pow2 head_dims.
  return True


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_q: pydantic.PositiveInt
  block_k: pydantic.PositiveInt
  num_stages: pydantic.PositiveInt
  num_warps: pydantic_lib.PowerOfTwo
  block_d: pydantic.PositiveInt | None = None
  block_d_out: pydantic.PositiveInt | None = None
  split_k: pydantic.PositiveInt = 1
  pack_mask: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTritonFlashAttention(base.DotProductAttention[Config, None]):
  """Pallas-Triton FlashAttention implementation."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  use_base2: bool = False
  use_stable_softmax: bool | type[base.AUTO] = base.AUTO

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(self, "vjp", vjp.PallasTritonFlashAttentionVjp())

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array | QArray, "*B T H D"],
      k: Float[Array | QArray, "*B t h D"],
      v: Float[Array | QArray, "*B t h d"],
      *,
      precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset],
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

    use_stable_softmax = self.use_stable_softmax
    if use_stable_softmax is base.AUTO:
      use_stable_softmax = base.needs_stable_softmax(
          logits_dtype, logits_soft_cap
      )

    mask, is_causal, q_start, q_end, k_start, k_end = _decompose_mask(
        mask, q, k, q_indices, k_indices
    )

    q_k_dot_precision, weights_v_dot_precision = precision
    f = functools.partial(
        _fwd,
        is_causal=is_causal,
        config=config,
        use_base2=self.use_base2,
        use_stable_softmax=use_stable_softmax,
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

      # Perform `split_k` using `vmap`.
      seq_k_axes = (None, -3, -3, -1, -1, -1, -1, -1, None, None)
      f = functools.partial(f, normalize_output=False, return_residuals=True)
      f = batching.vmap_split(f, in_axes=seq_k_axes, num_parts=split_k)
      combine_partial_results = functools.partial(
          base.combine_partial_results, normalize_output=normalize_output
      )
      f = lambda *args, f=f: combine_partial_results(*f(*args))

    out, residuals = base.vmap_batch_dims(f)(
        q, k, v, bias, mask, dropout_mask, q_start, q_end, k_start, k_end
    )
    return out.astype(q.dtype), (residuals if return_residuals else None)

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    q, k, v = ba.args[:3]
    *_, seq_len_q, _, head_dim = q.shape
    seq_len_k = k.shape[-3]

    min_block, max_block = 16, 64
    block_q = min(max(pl.next_power_of_2(seq_len_q), min_block), max_block)
    block_k = min(max(pl.next_power_of_2(seq_len_k), min_block), max_block)
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
        block_d = pl.next_power_of_2(min(head_dim, 64**3 // block_qk))

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
                    block_q=clamp(16, block_q, pl.next_power_of_2(q.shape[-3])),
                    block_k=clamp(16, block_k, pl.next_power_of_2(k.shape[-3])),
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)
