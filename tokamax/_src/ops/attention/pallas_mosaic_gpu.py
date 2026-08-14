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
"""Flash Attention Pallas-Mosaic-GPU implementation."""

import dataclasses
import functools
from typing import Any, ClassVar, override

import immutabledict
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import batching
from tokamax._src import gpu_utils
from tokamax._src import jaxtyping
from tokamax._src import shape as shape_lib
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_mosaic_gpu_common as common
from tokamax._src.ops.attention import pallas_mosaic_gpu_kernel_sm100 as sm100
from tokamax._src.ops.attention import pallas_mosaic_gpu_kernel_sm90 as sm90
from tokamax._src.ops.attention import pallas_mosaic_gpu_vjp as vjp


# TODO: Make attention Config a pydantic discriminated union.
ConfigSM90 = sm90.Config
ConfigSM100 = sm100.Config
type Config = ConfigSM90 | ConfigSM100
type Key = immutabledict.immutabledict[str, Any]
Mask = base.Mask
PagingInfo = base.PagingInfo
QArray = base.QArray
Residuals = base.Residuals


def _get_kernel_module():
  if not gpu_utils.has_mosaic_gpu_support():
    raise NotImplementedError("Mosaic GPU not supported on this platform.")
  if not (gpu_utils.is_sm90() or gpu_utils.is_sm100()):
    raise NotImplementedError("Only supported for sm90 and sm100 GPUs.")
  return sm100 if gpu_utils.is_sm100() else sm90


@dataclasses.dataclass(frozen=True)
class PallasMosaicGpuFlashAttention(base.DotProductAttention[Config, Key]):
  """Flash attention with Mosaic GPU.

  Attributes:
    use_base2: If `True`, use base-2 exponential and logarithms.
    use_stable_softmax: If `True`, use stable softmax, where the maximum value
      will be subtracted from the logits before applying the softmax function,
      improving numerical stability. If `AUTO` (the default), then stable
      softmax will be `True` unless `logits_soft_cap` is not `None`, in which
      case it will depend upon the value of `logits_soft_cap`.
    rescale_threshold: The threshold for rescaling the accumulator when using
      stable softmax. We have a rescaling factor, `alpha` where `alpha =
      exp(prev_max - new_max)`. If `alpha < rescale_threshold`, we rescale the
      accumulator. By default, it is `1`, which means rescaling happens every
      time the maximum seen value changes. However, it can be set to a lower
      value to perform rescaling less often, at the potential cost of numerical
      stability. It is ignored when not using stable softmax.
  """

  config_cls: ClassVar[type[Config]] = Config.__value__
  supports_symbolic_shapes: ClassVar[bool] = False
  use_stable_softmax: bool | type[base.AUTO] = base.AUTO
  rescale_threshold: float = 1.0

  def __post_init__(self):
    if self.vjp is None:
      vjp_ = vjp.PallasMosaicGpuFlashAttentionVjp()
      object.__setattr__(self, "vjp", vjp_)

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
      mask: base.Mask,
      dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
      dropout_rate: float,
      paging_info: PagingInfo | None,
      q_indices: Int[Array, "*#B #H T"] | None,
      k_indices: Int[Array, "*#B #h t"] | None,
      normalize_output: bool,
      return_residuals: bool,
      config: Config,
  ) -> tuple[Float[Array, "*B T H d"], Residuals | None]:
    """Performs attention, optionally returning softmax residuals."""
    if not gpu_utils.has_mosaic_gpu_support():
      raise NotImplementedError("Mosaic GPU not supported on this platform.")

    if not (gpu_utils.is_sm90() or gpu_utils.is_sm100()):
      raise NotImplementedError(
          "Mosaic GPU backend only supported for sm90+ GPUs for now."
      )

    supported_dtypes = (jnp.float32, jnp.float16, jnp.bfloat16)
    if any(dt not in supported_dtypes for dt in [x.dtype for x in (q, k, v)]):
      raise NotImplementedError("Only f32, f16 and bf16 inputs are supported.")
    if logits_dtype != jnp.float32:
      raise NotImplementedError("`logits_dtype` must be float32.")
    if dropout_mask is not None:
      raise NotImplementedError("dropout is not supported.")
    if paging_info is not None:
      raise NotImplementedError("Paged attention not supported.")

    out_dtype = q.dtype
    orig_seq_len_q = q.shape[-3]
    orig_seq_len_k = k.shape[-3]
    q, k, v, bias, mask, is_causal, k_start, k_end = common.prepare_inputs(
        q,
        k,
        v,
        bias=bias,
        mask=mask,
        q_indices=q_indices,
        k_indices=k_indices,
        precision=precision,
        fold_q_sequence_heads=config.fold_q_sequence_heads,
    )

    if orig_seq_len_k % (config.split_k * config.block_kv) != 0:
      if k_end is None:
        k_end = jnp.full([1] * (q.ndim - 1), orig_seq_len_k, jnp.int32)
      else:
        k_end = jnp.minimum(k_end, orig_seq_len_k)

    use_stable_softmax = self.use_stable_softmax
    split_k = config.split_k

    if isinstance(config, ConfigSM100):
      kernel_module = sm100
      if use_stable_softmax is base.AUTO:
        # TODO: Support sm100 with unstable softmax.
        use_stable_softmax = True
    elif isinstance(config, ConfigSM90):
      kernel_module = sm90
      if use_stable_softmax is base.AUTO:
        use_stable_softmax = base.needs_stable_softmax(
            logits_dtype, logits_soft_cap
        )
    else:
      raise TypeError(f"Unsupported config type: {type(config)}")

    f = functools.partial(
        kernel_module.flash_attention_kernel,
        is_causal=is_causal,
        logits_soft_cap=logits_soft_cap,
        logits_scale=logits_scale,
        out_dtype=out_dtype,
        normalize_output=normalize_output,
        return_residuals=return_residuals,
        use_stable_softmax=use_stable_softmax,
        rescale_threshold=self.rescale_threshold,
        config=config,
    )

    def pad_seq_k(x, axis):
      if x is None or axis is None or x.shape[axis] == 1:
        return x
      block = split_k * config.block_kv
      return shape_lib.pad_to_next_multiple_of(x, block, axis)

    seq_k_axes = (None, -3, -3, -1, -1, None, None)
    args = (q, k, v, bias, mask, k_start, k_end)
    args = tuple(pad_seq_k(x, ax) for x, ax in zip(args, seq_k_axes))

    if split_k > 1:
      if is_causal or k_start is not None or k_end is not None:
        raise ValueError(
            # TODO: Support causality and k_start/k_end with split_k > 1.
            "split_k > 1 only supported without causality and k_start/k_end."
        )
      f = functools.partial(f, normalize_output=False, return_residuals=True)
      f = batching.vmap_split(f, in_axes=seq_k_axes, num_parts=split_k)
      combine_partial_results = functools.partial(
          base.combine_partial_results, normalize_output=normalize_output
      )
      f = lambda *args, f=f: combine_partial_results(*f(*args))

    out, residuals = base.vmap_batch_dims(f)(*args)
    if config.fold_q_sequence_heads:
      return base.unfold_q_sequence_heads(out, residuals, orig_seq_len_q)
    return out, residuals

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return self._get_heuristics_config_impl(ba, fold_q_sequence_heads=False)

  def _get_heuristics_config_impl(
      self, ba: op.BoundArguments, *, fold_q_sequence_heads: bool
  ) -> Config:
    return _get_kernel_module().get_heuristics_config(
        ba, fold_q_sequence_heads=fold_q_sequence_heads
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    return _get_kernel_module().get_autotuning_configs(ba)

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_mosaic_gpu_support(device)
