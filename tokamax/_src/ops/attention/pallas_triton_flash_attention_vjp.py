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
"""Pallas-Triton FlashAttention VJP implementation."""

import dataclasses
import functools
from typing import ClassVar
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import batching
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.pallas import block
import pydantic
from tokamax._src import pydantic as pydantic_lib

Mask = base.Mask
Residuals = base.Residuals
PagingInfo = base.PagingInfo


def _bwd_dkdv(
    dk,
    dv,
    q_ref,
    k,
    v,
    bias_ref,
    mask_ref,
    do_ref,
    m_ref,
    l_ref,
    delta_ref,
    ds_ref,
    lo,
    hi,
    *,
    block_m1: int,
    is_causal: bool = False,
    logits_dtype: jnp.dtype,
    q_k_dot_precision: jax.lax.DotAlgorithmPreset,
    weights_v_dot_precision: jax.lax.DotAlgorithmPreset,
):
  """Computes dk and dv."""
  start_n = pl.program_id(0) * k.shape[0]
  span_n = pl.ds(start_n, k.shape[0])

  bias = None
  bias_span_n = span_n
  if bias_ref is not None:
    if bias_ref.shape[1] == 1:
      bias_span_n = pl.ds(0, 1)
    if bias_ref.shape[0] == 1:
      bias = bias_ref.at[pl.ds(0, 1), bias_span_n].load()

  mask = None
  mask_span_n = span_n
  if mask_ref is not None:
    if mask_ref.shape[1] == 1:
      mask_span_n = pl.ds(0, 1)
    if mask_ref.shape[0] == 1:
      mask = mask_ref.at[pl.ds(0, 1), mask_span_n].load()

  def body(i, carry):
    # pylint: disable=invalid-name
    dk, dv = carry
    curr_m = i * block_m1
    span_m = pl.ds(curr_m, block_m1)
    q = q_ref.at[span_m].load()
    # Load m before computing qk to reduce pipeline stall.
    m = m_ref.at[span_m].load()
    l = l_ref.at[span_m].load()
    sT = pl.dot(k, q.T, precision=q_k_dot_precision).astype(logits_dtype)

    if bias_ref is not None:
      nonlocal bias
      if bias is None:
        bias = bias_ref.at[span_m, bias_span_n].load()
      sT += bias.T

    # See forward base class for explanation of why this is not `-inf`.
    mask_value = float(jnp.finfo(jnp.float32).min)

    if mask_ref is not None:
      nonlocal mask
      if mask is None:
        mask = mask_ref.at[span_m, mask_span_n].load()
      sT = jnp.where(mask.T, sT, mask_value)

    if is_causal:
      offs_m = curr_m + jnp.arange(0, block_m1)
      offs_n = start_n + jnp.arange(0, k.shape[0])
      sT = jnp.where(offs_m[None, :] >= offs_n[:, None], sT, mask_value)

    pT = jnp.exp(sT - m) / l
    do = do_ref.at[span_m].load()
    dv += pl.dot(pT.astype(do.dtype), do, precision=weights_v_dot_precision)
    delta = delta_ref.at[span_m].load()
    dpT = pl.dot(v, do.T, precision=weights_v_dot_precision) - delta
    dsT = pT * dpT

    # If we have an attention mask, it is possible that the entire row is
    # masked out. In that case, the forwards pass will calculate `p`'s values
    # as `1 / seq_len_k`. The corresponding `ds` values must be zeroed.
    if mask_ref is not None:
      dsT = jnp.where(m == mask_value, 0.0, dsT)

    # TODO: Would this be better in `_bwd_dq`? Benchmark it.
    if ds_ref is not None:
      ds_ref.at[span_m, span_n].store(dsT.T.astype(ds_ref.dtype))

    dk += pl.dot(dsT.astype(q.dtype), q, precision=q_k_dot_precision)
    return dk, dv

  return jax.lax.fori_loop(lo, hi, body, (dk, dv))


def _bwd_dq(
    dq,
    q,
    k_ref,
    v_ref,
    bias_ref,
    mask_ref,
    do,
    m,
    l,
    delta,
    lo,
    hi,
    *,
    block_n2: int,
    is_causal: bool = False,
    logits_dtype: jnp.dtype,
    q_k_dot_precision: jax.lax.DotAlgorithmPreset,
    weights_v_dot_precision: jax.lax.DotAlgorithmPreset,
):
  """Computes dq."""
  start_m = pl.program_id(0) * q.shape[0]
  span_m = pl.ds(start_m, q.shape[0])

  bias = None
  bias_span_m = span_m
  if bias_ref is not None:
    if bias_ref.shape[0] == 1:
      bias_span_m = pl.ds(0, 1)
    if bias_ref.shape[1] == 1:
      bias = bias_ref.at[bias_span_m, pl.ds(0, 1)].load()

  mask = None
  mask_span_m = span_m
  if mask_ref is not None:
    if mask_ref.shape[0] == 1:
      mask_span_m = pl.ds(0, 1)
    if mask_ref.shape[1] == 1:
      mask = mask_ref.at[mask_span_m, pl.ds(0, 1)].load()

  def body(i, dq):
    curr_n = i * block_n2
    span_n = pl.ds(curr_n, block_n2)
    k = k_ref.at[span_n].load()
    v = v_ref.at[span_n].load()
    s = pl.dot(q, k.T, precision=q_k_dot_precision).astype(logits_dtype)

    if bias_ref is not None:
      nonlocal bias
      if bias is None:
        bias = bias_ref.at[bias_span_m, span_n].load()
      s += bias

    p = jnp.exp(s - m) / l

    if mask_ref is not None:
      nonlocal mask
      if mask is None:
        mask = mask_ref.at[mask_span_m, span_n].load()
      p = jnp.where(mask, p, 0.0)

    if is_causal:
      offs_m = start_m + jnp.arange(0, q.shape[0])
      offs_n = curr_n + jnp.arange(0, block_n2)
      p = jnp.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)

    dp = pl.dot(do, v.T, precision=weights_v_dot_precision) - delta
    ds = p * dp
    return dq + pl.dot(ds.astype(k.dtype), k, precision=q_k_dot_precision)

  return jax.lax.fori_loop(lo, hi, body, dq)


def _zero_ds(ds_ref, lo, hi, *, block_m: int, block_n: int):
  span_n = block.ds(pl.program_id(0), block_n)

  def body(i, _):
    span_m = block.ds(i, block_m)
    ds_ref.at[span_m, span_n].store(jnp.zeros((block_m, block_n), ds_ref.dtype))

  return jax.lax.fori_loop(lo, hi, body, None)


def _bwd_kernel(
    q_ref,
    k_ref,
    v_ref,
    bias_ref,
    mask_ref,
    m_ref,
    l_ref,
    delta_ref,
    dout_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    ds_ref,
    *,
    block_m1: int,
    block_n2: int,
    mask_block_slice_factor: int = 2,
    sm_scale: float,
    is_causal: bool,
    logits_dtype: jnp.dtype,
    q_k_dot_precision: jax.lax.DotAlgorithmPreset,
    weights_v_dot_precision: jax.lax.DotAlgorithmPreset,
):
  """Pallas MHA backward kernel implementation."""
  seq_len_q = q_ref.shape[0]
  seq_len_k = k_ref.shape[0]
  block_m2 = dq_ref.shape[0]
  block_n1 = dk_ref.shape[0]

  pid = pl.program_id(0)
  start_n = pid * block_n1

  @pl.when(start_n < seq_len_k)
  def calc_dkdv():
    span_n = pl.ds(start_n, block_n1)
    k = k_ref.at[span_n].load() * sm_scale
    v = v_ref.at[span_n].load()

    bwd_dkdv = functools.partial(
        _bwd_dkdv,
        q_ref=q_ref,
        k=k,
        v=v,
        bias_ref=bias_ref,
        mask_ref=mask_ref,
        do_ref=dout_ref,
        m_ref=m_ref,
        l_ref=l_ref,
        delta_ref=delta_ref,
        ds_ref=ds_ref,
        block_m1=block_m1,
        logits_dtype=logits_dtype,
        q_k_dot_precision=q_k_dot_precision,
        weights_v_dot_precision=weights_v_dot_precision,
    )

    dv = jnp.zeros(dv_ref.shape)
    dk = jnp.zeros(dk_ref.shape)
    lo = 0

    if is_causal:
      block_m1_mask = block_m1 // mask_block_slice_factor
      lo_mask = start_n // block_m1_mask
      hi_mask = lo_mask + (block_n1 // block_m1_mask)
      dk, dv = bwd_dkdv(
          dk, dv, lo=lo_mask, hi=hi_mask, block_m1=block_m1_mask, is_causal=True
      )
      lo = hi_mask // mask_block_slice_factor

      if ds_ref is not None:
        _zero_ds(ds_ref, 0, lo_mask, block_m=block_m1_mask, block_n=block_n1)

    dk, dv = bwd_dkdv(dk, dv, lo=lo, hi=pl.cdiv(seq_len_q, block_m1))
    dv_ref.store(dv.astype(dv_ref.dtype))
    dk_ref.store((dk * sm_scale).astype(dk_ref.dtype))

  start_m = pid * block_m2

  @pl.when(start_m < seq_len_q)
  def calc_dq():
    span_m = pl.ds(start_m, block_m2)
    q = q_ref.at[span_m].load() * sm_scale
    do = dout_ref.at[span_m].load()
    m = m_ref.at[span_m].load()[:, None]
    l = l_ref.at[span_m].load()[:, None]
    # delta is pre-divided by ds_scale.
    delta = delta_ref.at[span_m].load()[:, None]

    bwd_dq = functools.partial(
        _bwd_dq,
        q=q,
        k_ref=k_ref,
        v_ref=v_ref,
        bias_ref=bias_ref,
        mask_ref=mask_ref,
        do=do,
        m=m,
        l=l,
        delta=delta,
        block_n2=block_n2,
        logits_dtype=logits_dtype,
        q_k_dot_precision=q_k_dot_precision,
        weights_v_dot_precision=weights_v_dot_precision,
    )

    dq = jnp.zeros(dq_ref.shape)
    hi = pl.cdiv(seq_len_k, block_n2)

    if is_causal:
      block_n2_mask = block_n2 // mask_block_slice_factor
      lo_mask = start_m // block_n2_mask
      hi_mask = lo_mask + (block_m2 // block_n2_mask)
      hi_mask = jnp.minimum(hi_mask, hi * mask_block_slice_factor)
      dq = bwd_dq(
          dq, lo=lo_mask, hi=hi_mask, block_n2=block_n2_mask, is_causal=True
      )
      hi = jnp.minimum(lo_mask // mask_block_slice_factor, hi)

    dq = bwd_dq(dq, lo=0, hi=hi)
    dq_ref.store((dq * sm_scale).astype(dq_ref.dtype))


@dataclasses.dataclass(frozen=True)
class Config:
  block_m1: pydantic.PositiveInt
  block_n1: pydantic.PositiveInt
  block_m2: pydantic.PositiveInt
  block_n2: pydantic.PositiveInt
  num_stages: pydantic.PositiveInt
  num_warps: pydantic_lib.PowerOfTwo


@jaxtyping.jaxtyped
def _bwd(
    q: Float[Array, "T H D"],
    k: Float[Array, "t h D"],
    v: Float[Array, "t h d"],
    bias: Float[Array, "#H #T #t"] | None,
    mask: Bool[Array, "#H #T #t"] | None,
    dropout_mask: Bool[Array, "#H #T #t"] | None,
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
  # TODO: check whether this contributes significantly, and use
  # kernel if it does.
  delta = jnp.sum((out * dout).astype(jnp.float32), axis=-1).swapaxes(-2, -1)

  kernel = functools.partial(
      _bwd_kernel,
      block_m1=config.block_m1,
      block_n2=config.block_n2,
      sm_scale=logits_scale,
      is_causal=is_causal,
      logits_dtype=logits_dtype,
      q_k_dot_precision=q_k_dot_precision,
      weights_v_dot_precision=weights_v_dot_precision,
  )

  head_dim_pow2 = pl.next_power_of_2(head_dim)
  head_dim_out_pow2 = pl.next_power_of_2(head_dim_out)
  q_index_map = lambda i, j: (0, j, 0)
  k_index_map = lambda i, j: (0, j // (num_heads_q // num_heads_k), 0)
  v_index_map = k_index_map
  stat_spec = pl.BlockSpec((None, seq_len_q), lambda i, j: (j, 0))

  if bias is None:
    bias_spec = None
  else:
    bias_index_map = lambda i, j: (0 if bias.shape[0] == 1 else j, 0, 0)
    bias_spec = pl.BlockSpec((None, *bias.shape[1:]), bias_index_map)

  if mask is None:
    mask_spec = None
  else:
    mask_index_map = lambda i, j: (0 if mask.shape[0] == 1 else j, 0, 0)
    mask_spec = pl.BlockSpec((None, *mask.shape[1:]), mask_index_map)

  in_specs = (
      pl.BlockSpec((seq_len_q, None, head_dim_pow2), q_index_map),
      pl.BlockSpec((seq_len_k, None, head_dim_pow2), k_index_map),
      pl.BlockSpec((seq_len_k, None, head_dim_out_pow2), v_index_map),
      bias_spec,
      mask_spec,
      stat_spec,  # m
      stat_spec,  # l
      stat_spec,  # delta
      pl.BlockSpec((seq_len_q, None, head_dim_out_pow2), q_index_map),  # dout
  )

  dq_block_shape = (config.block_m2, None, head_dim_pow2)
  dk_block_shape = (config.block_n1, None, head_dim_pow2)
  dv_block_shape = (config.block_n1, None, head_dim_out_pow2)
  # TODO: Do partial reduction in kernel if bias bcast in a sequence dim.
  ds_block_shape = (None, seq_len_q, seq_len_k)
  ds_index_map = lambda i, j: (j, 0, 0)
  out_specs = (
      pl.BlockSpec(dq_block_shape, lambda i, j: (i, j, 0)),
      pl.BlockSpec(dk_block_shape, lambda i, j: (i, j, 0)),
      pl.BlockSpec(dv_block_shape, lambda i, j: (i, j, 0)),
      None if bias is None else pl.BlockSpec(ds_block_shape, ds_index_map),
  )
  dk_shape = (seq_len_k, num_heads_q, head_dim)
  dv_shape = (seq_len_k, num_heads_q, head_dim_out)
  ds_shape = (num_heads_q, seq_len_q, seq_len_k)
  out_shapes = (
      q,
      jax.ShapeDtypeStruct(dk_shape, k.dtype),
      jax.ShapeDtypeStruct(dv_shape, v.dtype),
      None if bias is None else jax.ShapeDtypeStruct(ds_shape, bias.dtype),
  )

  grid_dkdv = pl.cdiv(seq_len_k, config.block_n1)
  grid_dq = pl.cdiv(seq_len_q, config.block_m2)

  dq, dk, dv, ds = block.pallas_call(
      kernel,
      in_specs=in_specs,
      out_specs=out_specs,
      out_shape=out_shapes,
      name="pallas_flash_attention_vjp",
      grid=(max(grid_dkdv, grid_dq), num_heads_q),
      compiler_params=plgpu.CompilerParams(
          num_stages=config.num_stages, num_warps=config.num_warps
      ),
  )(q, k, v, bias, mask, m, l, delta, dout)

  if num_heads_k < num_heads_q:
    sum_shape = dk.shape[:-2] + (num_heads_k, num_heads_q // num_heads_k, -1)
    dk = jnp.sum(dk.reshape(sum_shape), axis=-2)
    dv = jnp.sum(dv.reshape(sum_shape), axis=-2)

  return dq, dk, dv, ds


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTritonFlashAttentionVjp(base.DotProductAttentionVjp[Config, None]):
  """Pallas-Triton FlashAttention VJP implementation."""
  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  def _fwd(
      self,
      residuals: Residuals,
      out: Float[Array, "*B T H d"],
      dout: Float[Array, "*B T H d"],
      q: Float[Array, "*B T H D"],
      k: Float[Array, "*B t h D"],
      v: Float[Array, "*B t h d"],
      *,
      precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset],
      logits_dtype: jnp.dtype,
      logits_scale: float,
      bias: Float[Array, "*#B #H #T #t"] | None,
      logits_soft_cap: float | None,
      mask: Mask | None,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
      dropout_rate: float,
      paging_info: PagingInfo | None,
      q_indices: Int[Array, "*#B #H T"] | None,
      k_indices: Int[Array, "*#B #H t"] | None,
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

    is_causal = False
    if mask is not None:
      if q_indices is None and k_indices is None:
        mask, is_causal = mask.take("is_causal")

      q_len_or_indices = q.shape[-3] if q_indices is None else q_indices
      k_len_or_indices = k.shape[-3] if k_indices is None else k_indices
      mask = mask.as_array(q_len_or_indices, k_len_or_indices)

    q_k_dot_precision, weights_v_dot_precision = precision
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
    )

    def broadcast_to_rank(x, rank):
      return None if x is None else jax.lax.broadcast_to_rank(x, rank)

    orig_bias_shape = bias.shape if bias is not None else None
    bias = broadcast_to_rank(bias, q.ndim)
    mask = broadcast_to_rank(mask, q.ndim)
    dropout_mask = broadcast_to_rank(dropout_mask, q.ndim)
    args = (q, k, v, bias, mask, dropout_mask, residuals, out, dout)

    for _ in q.shape[:-3]:
      f = batching.vmap_maybe_bcast(f, (0,) * len(args))

    dq, dk, dv, ds = f(*args)
    if bias is None:
      dbias = None
    else:
      broadcast_bias_axes = [i for i, d in enumerate(bias.shape) if d == 1]
      dbias = jnp.sum(ds, axis=broadcast_bias_axes, keepdims=True)
      dbias = dbias.reshape(orig_bias_shape)
    return base.DotProductAttentionGrads(q=dq, k=dk, v=dv, bias=dbias), None

  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    # TODO: Heuristics.
    return Config(
        block_m1=32,
        block_n1=64,
        block_m2=64,
        # block_n1=128,
        # block_m2=128,
        block_n2=32,
        num_warps=4,
        num_stages=2,  # 5,
    )
