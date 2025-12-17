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
"""Flash Attention Pallas-Mosaic-GPU VJP implementation."""

# pylint: disable=invalid-name

import dataclasses
import functools
from typing import ClassVar

import jax
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import gpu_utils
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_mosaic_gpu_vjp_common as vjp_common
from tokamax._src.ops.attention import pallas_mosaic_gpu_vjp_kernel_sm90 as sm90
from typing_extensions import override

Config = vjp_common.Config
Mask = base.Mask
PagingInfo = base.PagingInfo
Residuals = base.Residuals


def _broadcast_to_rank(x, rank):
  return None if x is None else jax.lax.broadcast_to_rank(x, rank)


def _decompose_mask(mask, q, k, q_indices, k_indices):
  """Decomposes `mask` into a mask array, `is_causal`, `k_start` and `k_end`."""
  if mask is None:
    return None, False, None, None

  is_causal = False
  k_start = None
  k_end = None

  if k_indices is None:
    mask, is_causal, k_start, k_end = mask.take("is_causal", "k_start", "k_end")

    # Fold `is_causal` into `k_end`. If `q_indices` is not `None`, then this is
    # necessary for correctness. Otherwise, it is a performance optimization.
    if is_causal and (q_indices is not None or k_end is not None):
      if q_indices is None:
        q_indices = jnp.arange(q.shape[-3])
      k_end_ = q_indices + 1
      k_end = k_end_ if k_end is None else jnp.minimum(k_end, k_end_)
      is_causal = False

    if k_start is not None:
      k_start = jax.lax.broadcast_to_rank(k_start, 2)
    if k_end is not None:
      k_end = jax.lax.broadcast_to_rank(k_end, 2)

  q_len_or_indices = q.shape[-3] if q_indices is None else q_indices
  k_len_or_indices = k.shape[-3] if k_indices is None else k_indices
  mask = mask.as_array(q_len_or_indices, k_len_or_indices)
  return mask, is_causal, k_start, k_end


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasMosaicGpuFlashAttentionVjp(
    base.DotProductAttentionVjp[Config, None]
):
  """Pallas-Triton FlashAttention VJP implementation."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  use_base2: bool = True
  dbias_intermediate_dtype: jax.typing.DTypeLike | None = None

  @jaxtyping.jaxtyped
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
      precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset],
      logits_dtype: jnp.dtype,
      logits_scale: float,
      bias: Float[Array, "*#B #H #T #t"] | None,
      logits_soft_cap: float | None,
      mask: base.Mask,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
      dropout_rate: float,
      paging_info: PagingInfo | None,
      q_indices: Int[Array, "*#B #H T"] | None,
      k_indices: Int[Array, "*#B #h t"] | None,
      normalize_output: bool,
      return_residuals: bool,
      config: Config,
  ) -> tuple[base.DotProductAttentionGrads, None]:
    del dropout_rate

    if not gpu_utils.has_mosaic_gpu_support():
      raise NotImplementedError("Mosaic GPU not supported on this platform.")

    compute_capability = float(backend.get_default_device().compute_capability)

    if compute_capability < 9.0 or compute_capability >= 10.0:
      raise NotImplementedError(
          "Mosaic GPU backend only supported for sm90 GPUs for now."
      )

    if paging_info is not None:
      raise NotImplementedError("Paged attention not supported.")

    if not normalize_output:
      raise NotImplementedError("`normalize_output=False` not supported.")

    if logits_dtype != jnp.float32:
      raise NotImplementedError("`logits_dtype` must be float32.")

    if dropout_mask is not None:
      raise NotImplementedError("dropout is not supported.")

    if return_residuals:
      raise NotImplementedError("`return_residuals` not supported.")

    mask, is_causal, k_start, k_end = _decompose_mask(
        mask, q, k, q_indices, k_indices
    )

    def cast(x, precision):
      msg = lambda dt: f"Only {dt} supported for {precision=}, got {x.dtype=}"
      if precision == jax.lax.DotAlgorithmPreset.DEFAULT:
        if x.dtype not in (jnp.float16, jnp.bfloat16):
          raise NotImplementedError(msg("f16 and bf16"))
        return x
      if x.dtype not in precision.supported_lhs_types:
        raise NotImplementedError(msg(precision.supported_lhs_types))
      if precision == jax.lax.DotAlgorithmPreset.BF16_BF16_F32:
        return x.astype(jnp.bfloat16)
      if precision == jax.lax.DotAlgorithmPreset.F16_F16_F32:
        return x.astype(jnp.float16)
      raise NotImplementedError(f"Unsupported {precision=}")

    q_k_dot_precision, weights_v_dot_precision = precision
    # TODO: Avoid silently downcasting types.
    q = cast(q, q_k_dot_precision)
    k = cast(k, q_k_dot_precision)
    v = cast(v, weights_v_dot_precision)
    dout = cast(dout, weights_v_dot_precision)

    orig_bias_shape = None if bias is None else bias.shape
    bias = _broadcast_to_rank(bias, q.ndim)
    mask = _broadcast_to_rank(mask, q.ndim)
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

    f = functools.partial(
        sm90.flash_attention_vjp_kernel,
        logits_scale=logits_scale,
        logits_soft_cap=logits_soft_cap,
        is_causal=is_causal,
        use_base2=self.use_base2,
        ds_dtype=ds_dtype,
        config=config,
    )

    dq, dk, dv, ds = base.vmap_batch_dims(f)(
        q, k, v, residuals, out, dout, bias, mask, k_start, k_end
    )
    if bias is None:
      dbias = None
    else:
      broadcast_bias_axes = [i for i, d in enumerate(bias.shape) if d == 1]
      dbias = jnp.sum(ds, axis=broadcast_bias_axes)
      dbias = dbias.astype(bias.dtype).reshape(orig_bias_shape)
    return base.DotProductAttentionGrads(q=dq, k=dk, v=dv, bias=dbias), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config(
        block_q_dkv=64,
        block_kv_dkv=64,
        block_q_dq=64,
        block_kv_dq=64,
        num_stages=2,
    )

  # TODO: Implement an autotuning search space.
  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    return set()

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_mosaic_gpu_support(device)
