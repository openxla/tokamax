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
"""Flash attention with Mosaic TPU."""

import dataclasses
import functools
from typing import Any, ClassVar, Final, TypeAlias
import immutabledict
import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib
from typing_extensions import override


QArray = base.QArray
Residuals = base.Residuals
PagingInfo = base.PagingInfo
Key: TypeAlias = immutabledict.immutabledict[str, Any]

_NUM_LANES: Final[int] = 128


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_q: pydantic.conint(multiple_of=_NUM_LANES, gt=0)
  block_kv: pydantic.conint(multiple_of=_NUM_LANES, gt=0)
  block_kv_compute: pydantic.conint(multiple_of=_NUM_LANES, gt=0)

  def __post_init__(self):
    if self.block_kv % self.block_kv_compute:
      raise ValueError(
          f"{self.block_kv=} must be a multiple of {self.block_kv_compute=}."
      )


@dataclasses.dataclass(frozen=True)
class PallasMosaicTpuFlashAttention(base.DotProductAttention[Config, Key]):
  """Flash attention with Mosaic TPU."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  use_base2: bool = True

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

    supported_dtypes = (jnp.bfloat16, jnp.float32)
    if any(x.dtype not in supported_dtypes for x in (q, k, v)):
      raise NotImplementedError(
          "Only bfloat16 and float32 inputs are supported."
      )

    # TODO: support a learnable bias.
    if bias is not None:
      raise NotImplementedError("Bias is not supported.")
    if logits_dtype != jnp.float32:
      raise NotImplementedError("`logits_dtype` must be float32.")
    if dropout_mask is not None:
      raise NotImplementedError("dropout is not supported.")
    if paging_info is not None:
      raise NotImplementedError("Paged attention not supported.")
    if dropout_rate != 0.0:
      raise NotImplementedError("Dropout is not supported.")

    # TODO: support sequence subsets.
    if mask.q_start is not None:
      raise NotImplementedError("mask.q_start is not supported.")
    if mask.q_end is not None:
      raise NotImplementedError("mask.q_end is not supported.")
    if mask.k_start is not None:
      raise NotImplementedError("mask.k_start is not supported.")
    if mask.k_end is not None:
      raise NotImplementedError("mask.k_end is not supported.")

    q *= logits_scale

    orig_q_shape = q.shape
    as_4d = lambda x: jax.lax.collapse(jax.lax.broadcast_to_rank(x, 4), 0, -3)
    q, k, v = map(as_4d, (q, k, v))

    _, q_seq_len, num_q_heads, _ = q.shape
    _, kv_seq_len, num_kv_heads, head_dim_out = v.shape

    # TODO: The SplashAttention kernel expects the sequence
    # dimension to be after the num_heads dimension. This requires transposing
    # the inputs, which introduces overhead.
    axis_swapper = lambda x: jnp.swapaxes(x, 1, 2)
    q, k, v = map(axis_swapper, (q, k, v))

    is_mqa = num_q_heads != num_kv_heads
    if num_q_heads % num_kv_heads:
      raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")

    if is_mqa and num_kv_heads != 1:
      raise NotImplementedError("Grouped query attention is not implemented.")
    if is_mqa:
      k = jnp.squeeze(k, axis=1)
      v = jnp.squeeze(v, axis=1)

    config_splash = splash.SplashConfig(
        use_base2_exp=self.use_base2,
        attn_logits_soft_cap=logits_soft_cap,
        q_layout=splash.QKVLayout.HEAD_DIM_MINOR,
        k_layout=splash.QKVLayout.HEAD_DIM_MINOR,
        v_layout=splash.QKVLayout.HEAD_DIM_MINOR,
        **dataclasses.asdict(config),
    )

    # TODO: support multiple shards.
    shard_count = 1

    mask_shape = (q_seq_len, kv_seq_len)
    if mask.is_causal:
      splash_mask = mask_lib.CausalMask(
          shape=(q_seq_len, kv_seq_len), shard_count=shard_count
      )
    elif mask.bool_mask is not None:
      if mask.is_causal:
        raise NotImplementedError(
            "Causal attention with a boolean mask is not supported."
        )
      splash_mask = as_4d(mask.bool_mask)
      mask_batch_size, num_mask_heads, _, _ = splash_mask.shape
      # TODO: Support boolean masks differing across heads.
      if num_mask_heads != 1:
        raise NotImplementedError(
            "Only num_mask_heads=1 is supported with a boolean mask."
        )
      if mask_batch_size != 1:
        raise NotImplementedError("Only unbatched boolean masks are supported.")
      splash_mask = jnp.squeeze(splash_mask, axis=(0, 1))  # (seq_q, seq_kv)
    else:
      splash_mask = mask_lib.FullMask(mask_shape)

    is_dynamic_mask = isinstance(splash_mask, jax.Array)
    make_splash_attention = functools.partial(
        splash._make_splash_attention, q_seq_shards=shard_count  # pylint: disable=protected-access
    )
    splash_maker = (
        splash._make_dynamic_splash_attention  # pylint: disable=protected-access
        if is_dynamic_mask
        else make_splash_attention
    )
    splash_fn = splash_maker(
        mask=splash_mask,
        config=config_splash,
        is_mqa=is_mqa,
        save_residuals=return_residuals,
        mask_value=float(jnp.finfo(jnp.float32).min),
        downcast_smem_data=False,
    )

    out = jax.vmap(splash_fn)(q, k, v)
    out = axis_swapper(out)
    out = out.reshape(*orig_q_shape[:-1], out.shape[-1])[..., :head_dim_out]
    return out, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments):
    # TODO: Select better parameters based on a heuristic.
    return Config(block_q=128, block_kv=128, block_kv_compute=128)

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    # TODO: Add support for autotuning.
    return set()

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
