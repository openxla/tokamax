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
from tokamax._src.ops.splash_attention import base
from tokamax._src.ops.splash_attention import splash_attention_kernel
from tokamax._src.ops.splash_attention import splash_attention_mask as mask_lib

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
  dropout_block_q: (
      Annotated[int, pydantic.Field(multiple_of=NUM_LANES, gt=0)] | None
  ) = None
  dropout_block_kv: (
      Annotated[int, pydantic.Field(multiple_of=NUM_LANES, gt=0)] | None
  ) = None
  use_base2_exp: bool = True
  dq_reduction_steps: int | None = None
  qk_diag_skip: bool = False
  qk_diag_grid: int = 2
  diag_band_skip: bool = False

  def __post_init__(self):
    if self.block_kv_dkv % self.block_kv_dkv_compute:
      block_kv_dkv = self.block_kv_dkv
      block_kv_dkv_compute = self.block_kv_dkv_compute
      raise ValueError(
          f"{block_kv_dkv=} must be a multiple of {block_kv_dkv_compute=}."
      )
    if (
        self.dropout_block_q is not None
        and self.block_q_dkv % self.dropout_block_q
    ):
      raise ValueError(
          f"{self.block_q_dkv=} must be a multiple of {self.dropout_block_q=}."
      )
    if (
        self.dropout_block_kv is not None
        and self.block_kv_dkv_compute % self.dropout_block_kv
    ):
      raise ValueError(
          f"{self.block_kv_dkv_compute=} must be a multiple of"
          f" {self.dropout_block_kv=}."
      )
    if self.dq_reduction_steps not in (None, 1, 3):
      raise ValueError(
          f"dq_reduction_steps has value: {self.dq_reduction_steps}, only 1, 3"
          " or None are supported."
      )
    if (self.qk_diag_skip or self.diag_band_skip) and not (
        self.block_q_dkv == self.block_kv_dkv == self.block_kv_dkv_compute
    ):
      raise ValueError(
          "qk_diag_skip and diag_band_skip require square backward blocks "
          f"({self.block_q_dkv=}, {self.block_kv_dkv=},"
          f" {self.block_kv_dkv_compute=})."
      )
    if self.diag_band_skip and not self.qk_diag_skip:
      raise ValueError("diag_band_skip requires qk_diag_skip=True.")
    if self.qk_diag_grid < 2 or (self.qk_diag_grid & (self.qk_diag_grid - 1)):
      raise ValueError(f"{self.qk_diag_grid=} must be a power of 2 >= 2.")
    if self.qk_diag_skip and self.block_kv_dkv_compute % self.qk_diag_grid:
      raise ValueError(
          f"{self.block_kv_dkv_compute=} must be a multiple of"
          f" {self.qk_diag_grid=}."
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
      out: Float[Array, "*B H T d"],
      dout: Float[Array, "*B H T d"],
      q: Float[Array, "*B H T D"],
      k: Float[Array, "*B h t D"] | Float[Array, "*B t D"],
      v: Float[Array, "*B h t d"] | Float[Array, "*B t d"],
      *,
      mask: base.Mask | mask_lib.Mask,
      segment_ids: base.SegmentIds | None = None,
      sinks: Float[Array, "*#B H"] | None = None,
      is_mqa: bool = False,
      mask_value: float = base.DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      dropout_rate: float = 0.0,
      prng_key: jax.Array | None = None,
      return_residuals: bool = False,
      config: Config,
  ) -> tuple[base.SplashAttentionGrads, None]:
    if return_residuals:
      raise NotImplementedError("`return_residuals` not supported.")

    _, lse = residuals
    if config.use_base2_exp:
      lse = lse * LOG2E

    q_seq_len = q.shape[-2]
    kv_seq_len = k.shape[-2]
    mask_shape = (q_seq_len, kv_seq_len)

    if isinstance(mask, mask_lib.Mask):
      splash_mask = mask
    elif isinstance(mask, base.Mask):
      if mask.bool_mask is not None and mask.is_causal:
        splash_mask = mask.as_array(q_seq_len, kv_seq_len)
      elif mask.bool_mask is not None:
        splash_mask = mask.bool_mask
      elif mask.is_causal:
        splash_mask = mask_lib.CausalMask(
            shape=mask_shape, offset=mask.offset, shard_count=1
        )
      else:
        splash_mask = mask_lib.FullMask(mask_shape)
    else:
      splash_mask = mask

    if is_mqa and k.ndim == q.ndim:
      k_in = jnp.squeeze(k, axis=-3)
      v_in = jnp.squeeze(v, axis=-3)
    else:
      k_in = k
      v_in = v

    splash_config = dataclasses.replace(
        splash_attention_kernel.SplashConfig.get_default(),
        attn_logits_soft_cap=attn_logits_soft_cap,
        dropout_rate=dropout_rate,
        block_q=config.block_q_dkv,
        block_kv=config.block_kv_dkv,
        block_kv_compute=config.block_kv_dkv_compute,
        **dataclasses.asdict(config),
    )

    orig_sinks_shape = None if sinks is None else sinks.shape
    broadcast_to_rank = (
        lambda x, rank: None
        if x is None
        else jax.lax.broadcast_to_rank(x, rank)
    )

    if segment_ids is not None:
      segment_ids = base.SegmentIds(
          q=jax.lax.broadcast_to_rank(segment_ids.q, q.ndim - 2)
          if isinstance(segment_ids.q, (jax.Array, np.ndarray))
          else segment_ids.q,
          kv=jax.lax.broadcast_to_rank(segment_ids.kv, q.ndim - 2)
          if isinstance(segment_ids.kv, (jax.Array, np.ndarray))
          else segment_ids.kv,
      )
    sinks_bcast = broadcast_to_rank(sinks, q.ndim - 2)

    if isinstance(splash_mask, (jax.Array, np.ndarray)):
      splash_mask = broadcast_to_rank(jnp.asarray(splash_mask), q.ndim - 1)

      def single_bwd_dynamic(dout, q, k, v, out, lse, mask, segment_ids, sinks):
        splash_maker = (
            splash_attention_kernel.make_dynamic_splash_mqa
            if is_mqa
            else splash_attention_kernel.make_dynamic_splash_mha
        )
        attn_fn = splash_maker(
            mask,
            config=splash_config,
            mask_value=mask_value,
        )
        res = (
            q,
            k,
            v,
            segment_ids,
            sinks,
            out,
            lse,
            attn_fn.dkv_mask_info,
            prng_key,
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
        return dq, dk, dv, dsinks

      dq, dk, dv, dsinks = base.vmap_batch_dims(single_bwd_dynamic)(
          dout, q, k_in, v_in, out, lse, splash_mask, segment_ids, sinks_bcast
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

      def single_bwd_static(dout, q, k, v, out, lse, segment_ids, sinks):
        res = (
            q,
            k,
            v,
            segment_ids,
            sinks,
            out,
            lse,
            attn_fn.dkv_mask_info,
            prng_key,
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
        return dq, dk, dv, dsinks

      dq, dk, dv, dsinks = base.vmap_batch_dims(single_bwd_static)(
          dout, q, k_in, v_in, out, lse, segment_ids, sinks_bcast
      )

    if is_mqa and k.ndim == q.ndim:
      dk = jnp.expand_dims(dk, axis=-3)
      dv = jnp.expand_dims(dv, axis=-3)

    if sinks is not None and dsinks is not None:
      assert sinks_bcast is not None
      broadcast_sinks_axes = [
          i
          for i, d in enumerate(sinks_bcast.shape)
          if i < len(sinks_bcast.shape) - 1 and d == 1
      ]
      if broadcast_sinks_axes:
        dsinks = jnp.sum(dsinks, axis=broadcast_sinks_axes)
      dsinks = dsinks.astype(sinks.dtype).reshape(orig_sinks_shape)

    grads = base.SplashAttentionGrads(q=dq, k=dk, v=dv, sinks=dsinks)
    return grads, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    q = ba.arguments["q"]
    k = ba.arguments["k"]
    is_mqa = ba.arguments.get("is_mqa", False)
    dropout_rate = ba.arguments.get("dropout_rate", 0.0)
    q_seq_len = q.shape[-2]
    kv_seq_len = k.shape[-2]

    # Use larger block sizes for longer sequences if possible
    block_q_dkv = 128
    if q_seq_len >= 1024 and q_seq_len % 1024 == 0:
      block_q_dkv = 1024

    block_kv_dkv = 128
    block_kv_dkv_compute = 128
    if kv_seq_len >= 1024 and kv_seq_len % 1024 == 0:
      block_kv_dkv = 1024
      block_kv_dkv_compute = 512

    dropout_block_q = block_q_dkv if dropout_rate > 0.0 else None
    dropout_block_kv = block_kv_dkv_compute if dropout_rate > 0.0 else None

    kv_steps = kv_seq_len // block_kv_dkv

    # Similar heuristic is used in the splash attention kernel
    dq_reduction_steps = None
    if kv_steps > 3:
      dq_reduction_steps = 3

    return Config(
        block_q_dkv=block_q_dkv,
        block_kv_dkv=block_kv_dkv,
        block_kv_dkv_compute=block_kv_dkv_compute,
        dropout_block_q=dropout_block_q,
        dropout_block_kv=dropout_block_kv,
        use_base2_exp=True,
        dq_reduction_steps=dq_reduction_steps,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    q = ba.arguments["q"]
    k = ba.arguments["k"]
    is_mqa = ba.arguments.get("is_mqa", False)
    mask = ba.arguments.get("mask")
    dropout_rate = ba.arguments.get("dropout_rate", 0.0)
    q_seq_len = q.shape[-2]
    kv_seq_len = k.shape[-2]

    is_causal = False
    if isinstance(mask, mask_lib.CausalMask) and mask.offset == 0:
      is_causal = True
    elif (
        isinstance(mask, base.Mask)
        and mask.is_causal
        and mask.offset == 0
        and mask.bool_mask is None
    ):
      is_causal = True

    tiles = [128, 256, 512, 1024, 2048]
    # TODO: Unify forward and VJP autotuning so that they use the
    # same dropout block shapes.
    dropout_tiles = tiles if dropout_rate > 0.0 else [None]
    configs = set()
    for bq, bkv, bkv_c, dbq, dbkv in itertools.product(
        tiles,
        tiles,
        tiles,
        dropout_tiles,
        dropout_tiles,
    ):
      if bq > q_seq_len or q_seq_len % bq != 0:
        continue
      if bkv > kv_seq_len or kv_seq_len % bkv != 0:
        continue
      if bkv % bkv_c != 0:
        continue
      if dbq is not None and bq % dbq != 0:
        continue
      if dbkv is not None and bkv_c % dbkv != 0:
        continue

      if q_seq_len >= 1024 and bq < 1024:
        continue
      if kv_seq_len >= 1024 and bkv < 1024:
        continue
      if q_seq_len >= 1024 and dbq is not None and dbq < 1024:
        continue
      if kv_seq_len >= 1024 and dbkv is not None and dbkv < min(1024, bkv_c):
        continue
      q_steps = q_seq_len // bq
      kv_steps = kv_seq_len // bkv
      can_use_diag_skip_2048 = (
          is_causal and bq == bkv == bkv_c and dropout_rate == 0.0
      )
      can_use_vmem_dq = can_use_diag_skip_2048 and (
          q_steps <= 2 or kv_steps == 1
      )
      # TODO: Make these conditions more configurable
      if bkv_c > 1024 and not can_use_diag_skip_2048:
        continue

      if bq >= 4096 or bkv >= 4096:
        continue

      dq_reduction_steps = None
      if kv_steps > 3:
        dq_reduction_steps = 3

      if bkv_c <= 1024:
        configs.add(
            Config(
                block_q_dkv=bq,
                block_kv_dkv=bkv,
                block_kv_dkv_compute=bkv_c,
                dropout_block_q=dbq,
                dropout_block_kv=dbkv,
                dq_reduction_steps=dq_reduction_steps,
            )
        )
      if is_causal and bq == bkv == bkv_c:
        if bkv_c >= 1024:
          qk_diag_grid_sizes = [2, 4]
        else:
          qk_diag_grid_sizes = [2]
        for qk_grid_size in qk_diag_grid_sizes:
          if bkv_c <= 1024 or can_use_diag_skip_2048:
            configs.add(
                Config(
                    block_q_dkv=bq,
                    block_kv_dkv=bkv,
                    block_kv_dkv_compute=bkv_c,
                    dropout_block_q=dbq,
                    dropout_block_kv=dbkv,
                    dq_reduction_steps=dq_reduction_steps,
                    qk_diag_skip=True,
                    qk_diag_grid=qk_grid_size,
                )
            )
          if can_use_vmem_dq and (bkv_c // qk_grid_size) % NUM_LANES == 0:
            configs.add(
                Config(
                    block_q_dkv=bq,
                    block_kv_dkv=bkv,
                    block_kv_dkv_compute=bkv_c,
                    dropout_block_q=dbq,
                    dropout_block_kv=dbkv,
                    dq_reduction_steps=1,
                    qk_diag_skip=True,
                    qk_diag_grid=qk_grid_size,
                    diag_band_skip=True,
                )
            )
    if not configs:
      configs.add(self._get_heuristics_config(ba))
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
