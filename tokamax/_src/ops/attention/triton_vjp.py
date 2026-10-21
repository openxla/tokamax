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
from typing import ClassVar, override

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import gpu_utils
from tokamax._src import jaxtyping
from tokamax._src import precision as precision_lib
from tokamax._src import pydantic as pydantic_lib
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
import triton

Mask = base.Mask
Residuals = base.Residuals
PagingInfo = base.PagingInfo


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_m1: pydantic.PositiveInt
  block_n1: pydantic.PositiveInt
  block_m2: pydantic.PositiveInt
  block_n2: pydantic.PositiveInt
  num_stages: pydantic.PositiveInt
  num_warps: pydantic_lib.PowerOfTwo


@jaxtyping.jaxtyped
def _bwd(  # pylint: disable=unused-argument,unused-variable
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

  grid_dkdv = triton.cdiv(seq_len_k, config.block_n1)
  grid_dq = triton.cdiv(seq_len_q, config.block_m2)

  raise NotImplementedError("TODO")


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

    is_causal = False
    if mask is None:
      mask_ = None
    else:
      if q_indices is None and k_indices is None:
        mask, is_causal = mask.take("is_causal")

      q_len_or_indices = q.shape[-3] if q_indices is None else q_indices
      k_len_or_indices = k.shape[-3] if k_indices is None else k_indices
      mask_ = mask.as_array(q_len_or_indices, k_len_or_indices)

    def broadcast_to_rank(x, rank):
      return None if x is None else jax.lax.broadcast_to_rank(x, rank)

    orig_bias_shape = None if bias is None else bias.shape
    bias = broadcast_to_rank(bias, q.ndim)
    mask_ = broadcast_to_rank(mask_, q.ndim)
    dropout_mask = broadcast_to_rank(dropout_mask, q.ndim)

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

    dq, dk, dv, ds = f(q, k, v, bias, mask_, dropout_mask, residuals, out, dout)
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

  # TODO: Implement an autotuning search space.
  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    configs = set()
    for block_m in [16, 32, 64, 128, 256]:
      for block_n in [16, 32, 64, 128, 256]:
        for num_warps in [2, 4]:
          for num_stages in [1, 2, 3, 4]:
            configs.add(
                Config(
                    block_m1=block_m,
                    block_m2=block_m,
                    block_n1=block_n,
                    block_n2=block_n,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_triton_support(device)
