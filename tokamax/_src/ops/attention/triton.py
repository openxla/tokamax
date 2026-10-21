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
from typing import ClassVar, override

import jax
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import batching
from tokamax._src import gpu_utils
from tokamax._src import jaxtyping
from tokamax._src import precision as precision_lib
from tokamax._src import pydantic as pydantic_lib
from tokamax._src import quantization
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import triton_vjp as vjp
import triton

Mask = base.Mask
QArray = base.QArray
Residuals = base.Residuals
PagingInfo = base.PagingInfo


@jaxtyping.jaxtyped
def _fwd(  # pylint: disable=unused-argument,unused-variable
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
  """Forward pass of Triton FlashAttention."""
  q, k, v = map(quantization.as_array_or_qarray_without_zero_point, (q, k, v))

  seq_len_q, num_heads_q, head_dim = q.shape
  seq_len_k, num_heads_k, head_dim_out = v.shape

  if config.pack_mask and mask is not None and mask.shape[-1] != 1:
    mask = jnp.packbits(mask, axis=-1, bitorder="little")
    pack_mask = True
  else:
    pack_mask = False

  raise NotImplementedError("TODO")


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
    # TODO: Make block_d work with non-pow2 head_dims.
    if triton.next_power_of_2(arg.shape[-1]) != arg.shape[-1]:
      return False
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

    use_stable_softmax = self.use_stable_softmax
    if use_stable_softmax is base.AUTO:
      use_stable_softmax = base.needs_stable_softmax(
          logits_dtype, logits_soft_cap
      )

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

      # Perform `split_k` using `vmap`.
      seq_k_axes = (None, -3, -3, -1, -1, -1, -1, -1, None, None)
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
        block_d = triton.next_power_of_2(min(head_dim, 64**3 // block_qk))

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
