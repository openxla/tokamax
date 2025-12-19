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
from typing import Any, ClassVar, TypeAlias

import immutabledict
import jax
import jax.experimental.pallas as pl
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import batching
from tokamax._src import gpu_utils
from tokamax._src import jaxtyping
from tokamax._src import quantization
from tokamax._src import shape as shape_lib
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import pallas_mosaic_gpu_common as common
from tokamax._src.ops.attention import pallas_mosaic_gpu_kernel_sm90 as sm90
from tokamax._src.ops.attention import pallas_mosaic_gpu_vjp as vjp
from typing_extensions import override

Config = common.Config
Key: TypeAlias = immutabledict.immutabledict[str, Any]
Mask = base.Mask
PagingInfo = base.PagingInfo
QArray = base.QArray
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


@dataclasses.dataclass(frozen=True)
class PallasMosaicGpuFlashAttention(base.DotProductAttention[Config, Key]):
  """Flash attention with Mosaic GPU."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  use_base2: bool = True
  use_stable_softmax: bool | type[base.AUTO] = base.AUTO

  def __post_init__(self):
    if self.vjp is None:
      vjp_ = vjp.PallasMosaicGpuFlashAttentionVjp(use_base2=self.use_base2)
      object.__setattr__(self, "vjp", vjp_)

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

    compute_capability = float(backend.get_default_device().compute_capability)

    if compute_capability < 9.0 or compute_capability >= 10.0:
      raise NotImplementedError(
          "Mosaic GPU backend only supported for sm90 GPUs for now."
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

    # TODO: Support in-kernel dequantization.
    q, k, v = map(quantization.as_array, (q, k, v))
    out_dtype = q.dtype

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

    orig_seq_len_q = q.shape[-3]
    if config.fold_q_sequence_heads:
      q, bias, mask, dropout_mask, q_indices = base.fold_q_sequence_heads(
          q, bias, mask, dropout_mask, q_indices, k.shape[-3], k.shape[-2]
      )

    mask, is_causal, k_start, k_end = _decompose_mask(
        mask, q, k, q_indices, k_indices
    )

    use_stable_softmax = self.use_stable_softmax
    if use_stable_softmax is base.AUTO:
      use_stable_softmax = base.needs_stable_softmax(
          logits_dtype, logits_soft_cap
      )

    f = functools.partial(
        sm90.flash_attention_kernel,
        is_causal=is_causal,
        logits_soft_cap=logits_soft_cap,
        logits_scale=logits_scale,
        out_dtype=out_dtype,
        normalize_output=normalize_output,
        return_residuals=return_residuals,
        use_base2=self.use_base2,
        use_stable_softmax=use_stable_softmax,
        config=config,
    )
    bias = _broadcast_to_rank(bias, q.ndim)
    mask = _broadcast_to_rank(mask, q.ndim)
    k_start = _broadcast_to_rank(k_start, q.ndim - 1)
    k_end = _broadcast_to_rank(k_end, q.ndim - 1)

    args = (q, k, v, bias, mask, k_start, k_end)

    if (split_k := config.split_k) > 1:
      if is_causal or k_start is not None or k_end is not None:
        raise ValueError(
            # TODO: Support causality and k_start/k_end with split_k > 1.
            "split_k > 1 only supported without causality and k_start/k_end."
        )

      def pad_seq_k(x, axis):
        if x is None or axis is None or x.shape[axis] == 1:
          return x
        block = split_k * config.block_kv
        return shape_lib.pad_to_next_multiple_of(x, block, axis)

      seq_k_axes = (None, -3, -3, -1, -1, None, None)
      args = tuple(pad_seq_k(x, ax) for x, ax in zip(args, seq_k_axes))
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
  def _get_heuristics_config(self, ba: op.BoundArguments):
    q, k, v = ba.args
    head_dim = k.shape[-1]
    head_dim_out = v.shape[-1]

    mask = ba.kwargs["mask"]
    q_indices = ba.kwargs["q_indices"]
    k_indices = ba.kwargs["k_indices"]
    mask, *_ = jax.eval_shape(_decompose_mask, mask, q, k, q_indices, k_indices)
    # 32-bit floats are downcast to 16-bit before the kernel call.
    dtype_bits = jnp.finfo(jnp.bfloat16).bits

    def shared_mem_usage_bytes(block_q, block_kv, num_stages):
      bytes_per_stage = (
          block_kv * head_dim * dtype_bits // 8
          + block_kv * head_dim_out * dtype_bits // 8
      )
      if (bias := ba.kwargs["bias"]) is not None:
        bytes_per_stage += (
            2 * block_q * block_kv * jnp.finfo(bias.dtype).bits // 8
        )
      # FIXME: This is an overestimate for broadcast masks.
      if mask is not None:
        bytes_per_stage += 2 * block_q * block_kv
      return (
          2 * block_q * head_dim * dtype_bits // 8
          + num_stages * bytes_per_stage
          + 1000  # Add some extra for barriers.
      )

    if shared_mem_usage_bytes(64, 128, 2) < 227 * 1024:
      return Config(block_q=64, block_kv=128, num_stages=2)

    # This is a pretty good option that works for most cases.
    return Config(block_q=64, block_kv=64, num_stages=2)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    q, k, _ = ba.args
    block_qs = set([
        min(x, pl.next_power_of_2(q.shape[-3] // 2))
        for x in [64, 128, 256]
        if q.shape[-3] % (x * 2) == 0  # 2 * block_q must divide seq_len_q.
    ])
    block_kvs = set([
        min(x, pl.next_power_of_2(k.shape[-3]))
        for x in [64, 128, 256]
        if k.shape[-3] % x == 0  # block_kv must divide seq_len_kv.
    ])
    configs = set()
    for block_q in block_qs:
      for block_kv in block_kvs:
        for num_stages in [2, 3, 4]:
          configs.add(
              Config(block_q=block_q, block_kv=block_kv, num_stages=num_stages)
          )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return gpu_utils.has_mosaic_gpu_support(device)
