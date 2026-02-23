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
from typing import Any, ClassVar

import jax
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import gpu_utils
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_mosaic_gpu_common as common
from tokamax._src.ops.attention import pallas_mosaic_gpu_vjp_common as vjp_common
from tokamax._src.ops.attention import pallas_mosaic_gpu_vjp_kernel_sm100 as sm100
from tokamax._src.ops.attention import pallas_mosaic_gpu_vjp_kernel_sm90 as sm90
from typing_extensions import override

Config = vjp_common.Config
Mask = base.Mask
PagingInfo = base.PagingInfo
Residuals = base.Residuals


def _broadcast_to_rank(x, rank):
  return None if x is None else jax.lax.broadcast_to_rank(x, rank)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasMosaicGpuFlashAttentionVjp(
    base.DotProductAttentionVjp[Config, None]
):
  """Pallas-Triton FlashAttention VJP implementation."""

  config_cls: ClassVar[Any] = Config | sm100.Config
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
      config: Config | sm100.Config,
  ) -> tuple[base.DotProductAttentionGrads, None]:
    del dropout_rate

    if not gpu_utils.has_mosaic_gpu_support():
      raise NotImplementedError("Mosaic GPU not supported on this platform.")

    compute_capability = float(backend.get_default_device().compute_capability)

    if compute_capability < 9.0 or compute_capability > 10.0:
      raise NotImplementedError(
          "Mosaic GPU backend only supported for sm90 and sm100 GPUs."
      )

    if compute_capability >= 10.0 and not isinstance(config, sm100.Config):
      raise NotImplementedError("SM100 config required for sm100 GPUs.")

    if compute_capability < 10.0 and not isinstance(config, sm90.Config):
      raise NotImplementedError("SM90 config required for sm90 GPUs.")

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

    mask, is_causal, k_start, k_end = common.decompose_mask(
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

    if isinstance(config, sm100.Config):
      if compute_capability < 10.0:
        raise ValueError("SM100 config provided but compute capability < 10.0")
      # TMA requires 16-byte aligned strides. If the last dimension is
      # 1 (e.g.  broadcasting), the stride of the second-to-last
      # dimension is small (1 element), violating the requirement. We
      # broadcast explicitly to fix this.  Note: mask is cast to int8
      # in the kernel, which materializes the array and fixes the
      # strides. bias is not cast, so we must materialize it here.
      # TODO: Fuse the broadcast.
      kv_seq_len = k.shape[-3]
      if mask is not None and mask.shape[-1] == 1 and kv_seq_len > 1:
        mask = jnp.broadcast_to(mask, mask.shape[:-1] + (kv_seq_len,))
      if bias is not None and bias.shape[-1] == 1 and kv_seq_len > 1:
        bias = jnp.broadcast_to(bias, bias.shape[:-1] + (kv_seq_len,)).astype(
            bias.dtype
        )

    if bias is None:
      ds_dtype = None
    elif self.dbias_intermediate_dtype is None:
      ds_dtype = bias.dtype
    elif bias.shape == (*q.shape[:-3], q.shape[-2], q.shape[-3], k.shape[-3]):
      ds_dtype = bias.dtype
    else:
      ds_dtype = self.dbias_intermediate_dtype

    kwargs = dict(
        is_causal=is_causal,
        logits_scale=logits_scale,
        logits_soft_cap=logits_soft_cap,
        use_base2=self.use_base2,
        ds_dtype=ds_dtype,
        config=config,
    )

    if isinstance(config, sm100.Config):
      kernel_fn = sm100.flash_attention_vjp_kernel
    else:
      kernel_fn = sm90.flash_attention_vjp_kernel

    f = functools.partial(kernel_fn, **kwargs)
    if isinstance(config, sm90.Config):
      f = base.vmap_batch_dims(f)

    dq, dk, dv, ds = f(q, k, v, residuals, out, dout, bias, mask, k_start, k_end)

    dbias = None
    if isinstance(config, sm90.Config):
      if bias is not None:
        broadcast_bias_axes = [i for i, d in enumerate(bias.shape) if d == 1]
        dbias = jnp.sum(ds, axis=broadcast_bias_axes).astype(bias.dtype)
        dbias = dbias.reshape(orig_bias_shape)
    else:
      dbias = ds

    return base.DotProductAttentionGrads(q=dq, k=dk, v=dv, bias=dbias), None

  @override
  def _get_heuristics_config(
      self, ba: op.BoundArguments
  ) -> sm90.Config | sm100.Config:
    compute_capability = float(backend.get_default_device().compute_capability)
    if compute_capability >= 10.0:
      return sm100.get_heuristics_config(ba)
    return sm90.get_heuristics_config(ba)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[sm90.Config | sm100.Config]:
    compute_capability = float(backend.get_default_device().compute_capability)
    if compute_capability >= 10.0:
      return sm100.get_autotuning_configs(ba)
    return sm90.get_autotuning_configs(ba)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_mosaic_gpu_support(device)
