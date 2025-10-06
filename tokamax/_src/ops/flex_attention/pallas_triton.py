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
"""Pallas-Triton FlexAttention implementation."""

import dataclasses
import functools
from typing import ClassVar
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import pydantic
from tokamax._src import jaxtyping
from tokamax._src import pydantic as pydantic_lib
from tokamax._src import quantization
from tokamax._src.ops import op
from tokamax._src.ops.flex_attention import base
from tokamax._src.pallas import block

ScoreMod = base.ScoreMod
MaskMod = base.MaskMod
QuantizedArray = quantization.QuantizedArray
Residuals = base.Residuals


def _rescale(values, scales_ref, slice_k, slice_d, keep_quantized=False):
  """Load the scales correctly and apply them to the values if requested."""
  if scales_ref is None:
    return values, None
  assert values is not None

  subchannel_quant = scales_ref.shape[1] not in (1, values.shape[1])

  if slice_k is None or scales_ref.shape[0] == 1:
    slice_k = slice(None)
  if slice_d is None or scales_ref.shape[1] == 1 or subchannel_quant:
    slice_d = slice(None)

  scales = scales_ref.at[slice_k, slice_d].load()
  if subchannel_quant:
    scales = jnp.repeat(scales, values.shape[1] // scales.shape[1], axis=1)
  if keep_quantized:
    return values, scales
  return values.astype(scales.dtype) * scales, None


def _fwd_kernel(
    q_ref,
    k_ref,
    v_ref,
    score_mod_value_refs,
    mask_mod_value_refs,
    dropout_mask_ref,
    out_ref,
    l_ref,
    m_ref,
    block_k: int,
    block_d: int | None,
    block_d_out: int | None,
    score_mod_fn,
    score_mod_value_specs,
    mask_mod_fn,
    mask_mod_value_specs,
    dropout_rate: float,
    use_base2: bool,
    use_stable_softmax: bool,
    q_k_dot_precision: jax.lax.DotAlgorithmPreset,
    weights_v_dot_precision: jax.lax.DotAlgorithmPreset,
    normalize_output: bool,
):
  """Pallas-Triton FlexAttention forward kernel implementation."""

  def get_values_and_scales(x):
    if isinstance(x, QuantizedArray):
      # TODO: Allow scales dimensions to be non-broadcastable.
      return x.values, x.scales
    return x, None

  q_ref, q_scales_ref = get_values_and_scales(q_ref)
  k_ref, k_scales_ref = get_values_and_scales(k_ref)
  v_ref, v_scales_ref = get_values_and_scales(v_ref)

  block_m = q_ref.shape[0]
  block_d = q_ref.shape[-1] if block_d is None else block_d
  split_d = q_ref.shape[-1] // block_d

  block_d_out = out_ref.shape[-1] if block_d_out is None else block_d_out
  split_d_out = out_ref.shape[-1] // block_d_out

  # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
  if use_stable_softmax:
    m_i = jnp.full([block_m], float("-inf"), dtype=jnp.float32)
  else:
    m_i = 0.0
  l_i = jnp.zeros([block_m], dtype=jnp.float32)
  accs = [jnp.zeros([block_m, block_d_out], dtype=jnp.float32)] * split_d_out

  quantize_qk_dot = (
      q_scales_ref is not None
      and k_scales_ref is not None
      and q_scales_ref.shape[1] == 1
      and k_scales_ref.shape[1] == 1
  )

  q = [None] * split_d
  q_scales = [None] * split_d

  for i in range(split_d):
    slice_d = pl.dslice(i * block_d, block_d)
    q[i] = q_ref.at[:, slice_d].load()
    q[i], q_scales[i] = _rescale(
        q[i], q_scales_ref, None, slice_d, quantize_qk_dot
    )

  def body(i, carry):
    accs, m_i, l_i = carry

    start_k = i * block_k
    slice_k = pl.dslice(start_k, block_k)
    span_k = start_k + jnp.arange(block_k)
    use_mask_k = k_ref.shape[0] % block_k != 0
    s = jnp.zeros([block_m, block_k], dtype=jnp.float32)

    for j in range(split_d):
      assert q[j] is not None
      slice_d = pl.dslice(j * block_d, block_d)
      k = k_ref.at[slice_k, slice_d].load()
      k, k_scales = _rescale(k, k_scales_ref, slice_k, slice_d, quantize_qk_dot)

      if quantize_qk_dot:
        s += pl.dot(q[j], k.T).astype(jnp.float32) * (q_scales[j] * k_scales.T)  # pytype: disable=attribute-error
      else:
        s += pl.dot(q[j].astype(k.dtype), k.T, precision=q_k_dot_precision)

    pids = tuple(pl.program_id(i) for i in range(len(out_ref.full_shape) - 1))
    mod_fn_pids = (*pids[2:], pids[0], pids[1], i)

    def load_mod_fn_value(ref, spec):
      idx = spec.index_map(*mod_fn_pids)
      idx = tuple(
          i if b is None else pl.ds(i * b, b)
          for i, b in zip(idx, spec.block_shape, strict=True)
      )
      return pl.load(ref, idx)

    # TODO: Lift loop invariant mod function values out of loop.
    score_mod_values = jax.tree.map(
        load_mod_fn_value, score_mod_value_refs, score_mod_value_specs
    )
    s = score_mod_fn(mod_fn_pids, (), score_mod_values, s)

    if use_base2:
      # NOTE: This rescaling must happen after the `score_mod`, but
      # before the attention masking (as the multiplication will cause `-inf`s).
      # TODO: Scale `q` instead if no `score_mod`.
      s *= np.log2(np.e)

    mask_mod_values = jax.tree.map(
        load_mod_fn_value, mask_mod_value_refs, mask_mod_value_specs
    )
    mask = mask_mod_fn(mod_fn_pids, (), mask_mod_values)
    # See base class for explanation of why the mask value is not `-inf`.
    s = jnp.where(mask, s, float(jnp.finfo(jnp.float32).min))

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
      dropout_mask = pl.load(dropout_mask_ref, (slice(None), dropout_mask_slice_k))
      p *= dropout_mask.astype(p.dtype) / (1 - dropout_rate)

    for j in range(split_d_out):
      slice_d = pl.dslice(j * block_d_out, block_d_out)
      v = v_ref.at[slice_k, slice_d].load(bounds_check=(True, False))
      v, _ = _rescale(v, v_scales_ref, slice_k, slice_d)
      accs[j] += pl.dot(p.astype(v.dtype), v, precision=weights_v_dot_precision)

    return accs, m_i, l_i

  num_k_blocks = pl.cdiv(k_ref.shape[0], block_k)
  accs, m_i, l_i = jax.lax.fori_loop(0, num_k_blocks, body, (accs, m_i, l_i))

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
    slice_d = pl.dslice(i * block_d_out, block_d_out)
    out_ref.at[:, slice_d].store(accs[i].astype(out_ref.dtype))


def _tile_score_mod(score_mod, scores, block_q, block_k):
  """Returns tiled version of `score_mod` with input values and specs."""
  if score_mod is None:
    return lambda pids, prefetch, vals, s: s, (), ()

  fn, values, prefetch = fuser.get_fusion_values(score_mod, scores)
  if prefetch:
    raise ValueError("prefetch not supported.")

  block_shape = (None,) * (scores.ndim - 2) + (block_q, block_k)
  spec = pl.BlockSpec(block_shape, lambda *pids: pids)
  # A virtual grid that includes k-sequence dimension.
  num_blocks_q = pl.cdiv(scores.shape[-2], block_q)
  num_blocks_k = pl.cdiv(scores.shape[-1], block_k)
  grid = (*scores.shape[:-2], num_blocks_q, num_blocks_k)
  fn_spec_puller = fuser.pull_block_spec(fn, spec, grid=grid)
  fn, (value_specs, _), _ = fn_spec_puller(values, scores)
  return fn, values, value_specs


def _tile_mask_mod(mask_mod, scores, block_q, block_k):
  """Returns tiled version of `mask_mod` with input values and specs."""
  if mask_mod is None:
    return lambda pids, prefetch, vals: True, (), ()

  # Re-use logic in `_tile_score_mod`. Create a dummy `scores` argument then
  # remove it again below.
  def score_mod(scores):
    return jnp.broadcast_to(mask_mod(scores.shape), scores.shape)

  fn, values, value_specs = _tile_score_mod(score_mod, scores, block_q, block_k)
  fn_ = lambda pids, prefetch, values: fn(pids, prefetch, values, scores)
  return fn_, values, value_specs


@jaxtyping.jaxtyped
def _fwd(
    q: Float[Array | QuantizedArray, "*B T H D"],
    k: Float[Array | QuantizedArray, "*B t h D"],
    v: Float[Array | QuantizedArray, "*B t h d"],
    *,
    score_mod: ScoreMod | None,
    mask_mod: MaskMod | None,
    dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
    dropout_rate: float,
    config: "Config",
    use_base2: bool,
    use_stable_softmax: bool,
    normalize_output: bool,
    return_residuals: bool,
    q_k_dot_precision: jax.lax.DotAlgorithmPreset,
    weights_v_dot_precision: jax.lax.DotAlgorithmPreset,
) -> tuple[Float[Array, "*B T H d"], Residuals | None]:
  """Forward pass of Pallas-Trtion FlexAttention."""

  *batch, seq_len_q, num_heads_q, head_dim = q.shape
  *_, seq_len_k, num_heads_k, head_dim_out = v.shape

  block_q = config.block_q
  block_k = config.block_k
  # TODO: Could these just be multiple of block_d[_out]?
  head_dim_pow2 = pl.next_power_of_2(head_dim)
  head_dim_out_pow2 = pl.next_power_of_2(head_dim_out)
  q_index_map = lambda i, j: (j, i, 0)
  q_block_shape = (block_q, None, head_dim_pow2)
  k_index_map = lambda i, j: (0, i // (num_heads_q // num_heads_k), 0)
  k_block_shape = (seq_len_k, None, head_dim_pow2)
  v_index_map = k_index_map
  v_block_shape = (seq_len_k, None, head_dim_out_pow2)

  def spec(index_map, block_shape):
    def batched_index_map(*pids):
      i = len(pids) - len(batch)
      return pids[i:] + index_map(*pids[:i])

    block_shape = (None,) * len(batch) + tuple(block_shape)
    return pl.BlockSpec(block_shape, batched_index_map)

  def bcast_spec(x, index_map, block_shape):
    spec_ = spec(index_map, block_shape)

    def bcast_index_map(*pids):
      idx = spec_.index_map(*pids)
      return [x if (d > 1) else 0 for x, d in zip(idx, x.shape, strict=True)]

    block_shape = tuple(
        bd if (bd is None or (d > 1)) else 1
        for bd, d in zip(spec_.block_shape, x.shape, strict=True)
    )
    return pl.BlockSpec(block_shape, bcast_index_map)

  def input_spec(x, index_map, block_shape):
    spec_ = spec(index_map, block_shape)
    if not isinstance(x, QuantizedArray):
      return spec_

    scales_block_shape = [
        None if b is None else min(s, b)
        for s, b in zip(x.scales.shape[-3:], block_shape, strict=True)
    ]
    scales_spec = bcast_spec(x.scales, index_map, scales_block_shape)
    return QuantizedArray(spec_, scales_spec)  # pytype: disable=wrong-arg-types

  scores_shape = (*batch, num_heads_q, seq_len_q, seq_len_k)
  scores = jax.ShapeDtypeStruct(scores_shape, jnp.float32)
  score_mod_fn, score_mod_values, score_mod_value_specs = _tile_score_mod(
      score_mod, scores, block_q, block_k
  )
  mask_mod_fn, mask_mod_values, mask_mod_value_specs = _tile_mask_mod(
      mask_mod, scores, block_q, block_k
  )
  if dropout_mask is None:
    dropout_mask_spec = None
  else:
    dropout_mask = jax.lax.broadcast_to_rank(dropout_mask, q.ndim)
    dropout_mask_spec = bcast_spec(
        dropout_mask, lambda i, j: (i, j, 0), (None, block_q, seq_len_k)
    )
  in_specs = (
      input_spec(q, q_index_map, q_block_shape),
      input_spec(k, k_index_map, k_block_shape),
      input_spec(v, v_index_map, v_block_shape),
      jax.tree.map(lambda _: pl.no_block_spec, score_mod_values),
      jax.tree.map(lambda _: pl.no_block_spec, mask_mod_values),
      dropout_mask_spec,
  )

  out_shape = jax.ShapeDtypeStruct((*q.shape[:-1], head_dim_out), q.dtype)
  out_spec = spec(q_index_map, (block_q, None, head_dim_out_pow2))

  residual_shape = jax.ShapeDtypeStruct(scores_shape[:-1], jnp.float32)
  residual_spec = spec(lambda i, j: (i, j), (None, block_q))

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

  kernel = functools.partial(
      _fwd_kernel,
      block_k=block_k,
      block_d=config.block_d,
      block_d_out=config.block_d_out,
      score_mod_fn=score_mod_fn,
      score_mod_value_specs=score_mod_value_specs,
      mask_mod_fn=mask_mod_fn,
      mask_mod_value_specs=mask_mod_value_specs,
      dropout_rate=dropout_rate,
      use_base2=use_base2,
      use_stable_softmax=use_stable_softmax,
      q_k_dot_precision=q_k_dot_precision,
      weights_v_dot_precision=weights_v_dot_precision,
      normalize_output=normalize_output,
  )

  out, l, m = block.pallas_call(
      kernel,
      in_specs=in_specs,
      out_specs=out_specs,
      out_shape=out_shapes,
      name="pallas_triton_flex_attention" + name_suffix,
      grid=(num_heads_q, pl.cdiv(seq_len_q, block_q), *batch),
      compiler_params=plgpu.CompilerParams(
          num_stages=config.num_stages, num_warps=config.num_warps
      ),
  )(q, k, v, score_mod_values, mask_mod_values, dropout_mask)

  if not return_residuals:
    return out, None

  return out, (jnp.zeros_like(l) if m is None else m, l)


def _can_have_block_d(*args):
  for arg in args:
    if isinstance(arg, QuantizedArray) and any(
        s not in (1, v) for v, s in zip(arg.values.shape, arg.scales.shape)
    ):
      return False  # TODO: Make block_d work with subchannel quant.
    if pl.next_power_of_2(arg.shape[-1]) != arg.shape[-1]:
      return False  # TODO: Make block_d work with non-pow2 head_dims.
  return True


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  block_q: pydantic.PositiveInt
  block_k: pydantic.PositiveInt
  num_stages: pydantic.PositiveInt
  num_warps: pydantic_lib.PowerOfTwo
  block_d: pydantic.PositiveInt | None = None
  block_d_out: pydantic.PositiveInt | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasTritonFlexAttention(base.FlexAttention[Config, None]):
  """Pallas-Triton FlexAttention implementation."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  use_base2: bool = False
  use_stable_softmax: bool = True

  @jaxtyping.jaxtyped
  def _fwd(
      self,
      q: Float[Array | QuantizedArray, "*B T H D"],
      k: Float[Array | QuantizedArray, "*B t h D"],
      v: Float[Array | QuantizedArray, "*B t h d"],
      *,
      precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset],
      score_mod: ScoreMod | None,
      mask_mod: MaskMod | None,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
      dropout_rate: float,
      normalize_output: bool,
      return_residuals: bool,
      config: Config,
  ) -> tuple[Float[Array, "*B T H d"], Residuals | None]:
    block_d, block_d_out = config.block_d, config.block_d_out
    if not _can_have_block_d(q, k, v) and not (block_d == block_d_out == None):
      raise ValueError("block_d[_out] unsupported for subchannel quantization.")

    q_k_dot_precision, weights_v_dot_precision = precision
    return _fwd(
        q,
        k,
        v,
        score_mod=score_mod,
        mask_mod=mask_mod,
        config=config,
        use_base2=self.use_base2,
        use_stable_softmax=self.use_stable_softmax,
        return_residuals=return_residuals,
        dropout_mask=dropout_mask,
        dropout_rate=dropout_rate,
        q_k_dot_precision=q_k_dot_precision,
        weights_v_dot_precision=weights_v_dot_precision,
        normalize_output=normalize_output,
    )

  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    q, k, v = ba.args
    *_, seq_len_q, _, head_dim = q.shape
    seq_len_k = k.shape[-3]

    min_block, max_block = 16, 64
    block_q = min(max(pl.next_power_of_2(seq_len_q), min_block), max_block)
    block_k = min(max(pl.next_power_of_2(seq_len_k), min_block), max_block)
    num_warps, num_stages = 4, 2

    head_dim_bytes = jnp.dtype(q.dtype).itemsize * head_dim
    if head_dim_bytes >= 512:
      num_stages = 1

    if _can_have_block_d(q, k, v):
      block_d = pl.next_power_of_2(min(head_dim, 64**3 // (block_q * block_k)))
    else:
      block_d = None

    return Config(
        block_q=block_q,
        block_k=block_k,
        block_d=block_d,
        block_d_out=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )

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
