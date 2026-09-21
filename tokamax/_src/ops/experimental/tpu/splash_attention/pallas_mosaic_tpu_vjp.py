# Copyright 2026 Google LLC. All Rights Reserved.
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
"""Pallas/Mosaic kernel wrapper for Splash Attention VJP."""

import dataclasses
import itertools
from typing import Annotated, ClassVar, Final, override
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.splash_attention import base
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib

NUM_LANES: Final[int] = 128
LOG2E = splash_attention_kernel.LOG2E


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Configuration for Splash Attention Pallas TPU VJP kernel."""

  block_q_dkv: Annotated[int, pydantic.Field(multiple_of=NUM_LANES, gt=0)]
  block_kv_dkv: Annotated[int, pydantic.Field(multiple_of=NUM_LANES, gt=0)]
  block_kv_dkv_compute: Annotated[
      int, pydantic.Field(multiple_of=NUM_LANES, gt=0)
  ]
  use_base2_exp: bool = True

  def __post_init__(self):
    if self.block_kv_dkv % self.block_kv_dkv_compute:
      block_kv_dkv = self.block_kv_dkv
      block_kv_dkv_compute = self.block_kv_dkv_compute
      raise ValueError(
          f"{block_kv_dkv=} must be a multiple of {block_kv_dkv_compute=}."
      )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasMosaicTpuSplashAttentionVjp(base.SplashAttentionVjp[Config]):
  """Pallas-Mosaic SplashAttention VJP implementation."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      residuals: base.Residuals,
      out: Float[Array, "num_q_heads q_seq_len head_dim_v"],
      dout: Float[Array, "num_q_heads q_seq_len head_dim_v"],
      q: Float[Array, "num_q_heads q_seq_len head_dim_qk"],
      k: Float[Array, "..."],
      v: Float[Array, "..."],
      *,
      mask: base.Mask | mask_lib.Mask,
      segment_ids: base.SegmentIds | None = None,
      sinks: Float[Array, "..."] | None = None,
      is_mqa: bool = False,
      mask_value: float = base.DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      dropout_rate: float = 0.0,
      return_residuals: bool = False,
      config: Config,
  ) -> tuple[base.SplashAttentionGrads, None]:
    if dropout_rate != 0.0:
      raise NotImplementedError(
          "Dropout is not supported in Pallas/Mosaic TPU VJP."
      )

    if return_residuals:
      raise NotImplementedError("`return_residuals` not supported.")

    _, lse = residuals
    if config.use_base2_exp:
      lse = lse * LOG2E

    seq_len_q = q.shape[1]
    seq_len_kv = k.shape[0] if is_mqa and k.ndim == 2 else k.shape[1]
    mask_shape = (seq_len_q, seq_len_kv)

    if isinstance(mask, mask_lib.Mask):
      splash_mask = mask
    elif isinstance(mask, base.Mask):
      if mask.bool_mask is not None and mask.is_causal:
        splash_mask = mask.as_array(seq_len_q, seq_len_kv)
      elif mask.bool_mask is not None:
        splash_mask = mask.bool_mask
      elif mask.is_causal:
        splash_mask = mask_lib.CausalMask(shape=mask_shape, shard_count=1)
      else:
        splash_mask = mask_lib.FullMask(mask_shape)
    else:
      splash_mask = mask

    if is_mqa and k.ndim == 3:
      k_in = k[0]
      v_in = v[0]
    else:
      k_in = k
      v_in = v

    splash_config = dataclasses.replace(
        splash_attention_kernel.SplashConfig.get_default(),
        attn_logits_soft_cap=attn_logits_soft_cap,
        block_q=config.block_q_dkv,
        block_kv=config.block_kv_dkv,
        block_kv_compute=config.block_kv_dkv_compute,
        **dataclasses.asdict(config),
    )

    if isinstance(splash_mask, (jax.Array, np.ndarray)):
      splash_maker = (
          splash_attention_kernel.make_dynamic_splash_mqa
          if is_mqa
          else splash_attention_kernel.make_dynamic_splash_mha
      )
      attn_fn = splash_maker(
          jnp.asarray(splash_mask),
          config=splash_config,
          mask_value=mask_value,
      )
    else:
      splash_maker = (
          splash_attention_kernel.make_splash_mqa_single_device
          if is_mqa
          else splash_attention_kernel.make_splash_mha_single_device
      )
      attn_fn = splash_maker(
          splash_mask,
          config=splash_config,
          mask_value=mask_value,
      )

    res = (
        q,
        k_in,
        v_in,
        segment_ids,
        sinks,
        out,
        lse,
        attn_fn.dkv_mask_info,
        None,
    )

    splash_fn_kwargs = attn_fn.kwargs
    _, _, dq, dk, dv, _, dsinks, _, _ = (
        splash_attention_kernel._splash_attention_bwd(  # pylint: disable=protected-access
            save_residuals=False,
            mask_value=mask_value,
            is_mqa=is_mqa,
            config=splash_fn_kwargs["config"],
            mask_function=splash_fn_kwargs["mask_function"],
            fwd_mask_sparsity=splash_fn_kwargs["fwd_mask_sparsity"],
            dkv_mask_sparsity=splash_fn_kwargs["dkv_mask_sparsity"],
            res=res,
            grads=dout,
        )
    )

    if is_mqa and k.ndim == 3:
      dk = dk.reshape(k.shape)
      dv = dv.reshape(v.shape)

    grads = base.SplashAttentionGrads(q=dq, k=dk, v=dv, sinks=dsinks)
    return grads, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba

    return Config(
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        use_base2_exp=True,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    q = ba.arguments["q"]
    k = ba.arguments["k"]
    is_mqa = ba.arguments.get("is_mqa", False)
    q_seq_len = q.shape[1]
    kv_seq_len = k.shape[0] if is_mqa and k.ndim == 2 else k.shape[1]

    tiles = [128, 256, 512, 1024, 2048]
    configs = set()
    for bq, bkv, bkv_c in itertools.product(
        tiles,
        tiles,
        tiles,
    ):
      if bq > q_seq_len or q_seq_len % bq != 0:
        continue
      if bkv > kv_seq_len or kv_seq_len % bkv != 0:
        continue
      if bkv % bkv_c != 0:
        continue

      if q_seq_len >= 1024 and bq < 1024:
        continue
      if kv_seq_len >= 1024 and bkv < 1024:
        continue
      # TODO: Make these conditions more configurable
      if bkv_c > 1024:
        continue

      if bq >= 4096 or bkv >= 4096:
        continue

      configs.add(
          Config(
              block_q_dkv=bq,
              block_kv_dkv=bkv,
              block_kv_dkv_compute=bkv_c,
          )
      )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
