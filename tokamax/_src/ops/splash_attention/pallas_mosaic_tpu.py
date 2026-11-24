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
"""Pallas/Mosaic kernel wrapper for Splash Attention on TPU."""

import dataclasses
import itertools
from typing import Annotated, ClassVar, Final, override
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import pydantic
from tokamax._src import jaxtyping
from tokamax._src import mosaic_tpu
from tokamax._src.ops import op
from tokamax._src.ops.splash_attention import base
from tokamax._src.ops.splash_attention import pallas_mosaic_tpu_vjp
from tokamax._src.ops.splash_attention import reference
from tokamax._src.ops.splash_attention import splash_attention_kernel
from tokamax._src.ops.splash_attention import splash_attention_mask as mask_lib

NUM_LANES: Final[int] = 128
QKVLayout = splash_attention_kernel.QKVLayout


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning and execution configuration for Splash Attention Pallas TPU kernel."""

  block_q: Annotated[int, pydantic.Field(multiple_of=NUM_LANES, gt=0)]
  block_kv: Annotated[int, pydantic.Field(multiple_of=NUM_LANES, gt=0)]
  block_kv_compute: Annotated[int, pydantic.Field(multiple_of=NUM_LANES, gt=0)]
  dropout_block_q: (
      Annotated[int, pydantic.Field(multiple_of=NUM_LANES, gt=0)] | None
  ) = None
  dropout_block_kv: (
      Annotated[int, pydantic.Field(multiple_of=NUM_LANES, gt=0)] | None
  ) = None
  q_layout: QKVLayout
  k_layout: QKVLayout
  v_layout: QKVLayout
  use_experimental_scheduler: bool
  use_base2_exp: bool = True
  max_logit_const: float | None = None
  fuse_reciprocal: bool = True
  residual_checkpoint_name: str | None = None
  fold_q_seq_heads: bool = False
  qk_diag_skip: bool = False
  qk_diag_grid: int = 2
  sv_diag_skip: bool = False
  diag_band_skip: bool = False

  def __post_init__(self):
    if self.fold_q_seq_heads and (
        self.qk_diag_skip or self.sv_diag_skip or self.diag_band_skip
    ):
      raise ValueError(
          "Diagonal skip options are incompatible with `fold_q_seq_heads=True`."
      )
    if self.block_kv % self.block_kv_compute:
      raise ValueError(
          f"{self.block_kv=} must be a multiple of {self.block_kv_compute=}."
      )
    if self.dropout_block_q is not None and self.block_q % self.dropout_block_q:
      raise ValueError(
          f"{self.block_q=} must be a multiple of {self.dropout_block_q=}."
      )
    if (
        self.dropout_block_kv is not None
        and self.block_kv_compute % self.dropout_block_kv
    ):
      raise ValueError(
          f"{self.block_kv_compute=} must be a multiple of"
          f" {self.dropout_block_kv=}."
      )
    if self.qk_diag_skip and self.block_q != self.block_kv:
      raise ValueError(
          "qk_diag_skip requires block_q == block_kv "
          f"({self.block_q=}, {self.block_kv=})."
      )
    if self.diag_band_skip and not self.qk_diag_skip:
      raise ValueError("diag_band_skip requires qk_diag_skip=True.")
    if (self.sv_diag_skip or self.diag_band_skip) and not (
        self.block_q == self.block_kv == self.block_kv_compute
    ):
      raise ValueError(
          "sv_diag_skip and diag_band_skip require square forward blocks "
          f"({self.block_q=}, {self.block_kv=}, {self.block_kv_compute=})."
      )
    if self.qk_diag_grid < 2 or (self.qk_diag_grid & (self.qk_diag_grid - 1)):
      raise ValueError(f"{self.qk_diag_grid=} must be a power of 2 >= 2.")
    if (
        self.qk_diag_skip or self.sv_diag_skip
    ) and self.block_kv_compute % self.qk_diag_grid:
      raise ValueError(
          f"{self.block_kv_compute=} must be a multiple of"
          f" {self.qk_diag_grid=}."
      )
    if (
        self.diag_band_skip
        and (self.block_kv_compute // self.qk_diag_grid) % NUM_LANES != 0
    ):
      raise ValueError(
          f"{self.block_kv_compute=} // {self.qk_diag_grid=} must be a "
          f"multiple of {NUM_LANES=} when diag_band_skip=True."
      )


@dataclasses.dataclass(frozen=True)
class PallasMosaicTpuSplashAttention(base.SplashAttention[Config]):
  """Tokamax operator wrapper for Splash Attention Pallas TPU kernel."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  supports_analytic_local_mask: ClassVar[bool] = True

  def _default_vjp(
      self,
  ) -> pallas_mosaic_tpu_vjp.PallasMosaicTpuSplashAttentionVjp:
    if self.config is None:
      return pallas_mosaic_tpu_vjp.PallasMosaicTpuSplashAttentionVjp()
    return pallas_mosaic_tpu_vjp.PallasMosaicTpuSplashAttentionVjp(
        q_layout=self.config.q_layout,
        k_layout=self.config.k_layout,
        v_layout=self.config.v_layout,
        use_base2_exp=self.config.use_base2_exp,
        fold_q_seq_heads=self.config.fold_q_seq_heads,
    )

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(self, "vjp", self._default_vjp())

  @override
  def replace(self, **kwargs):
    if (
        "config" in kwargs
        and "vjp" not in kwargs
        and self.vjp == self._default_vjp()
    ):
      return dataclasses.replace(self, **kwargs, vjp=None)
    return base.SplashAttention.replace(self, **kwargs)

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array, "*B H T D"],
      k: Float[Array, "*B h t D"] | Float[Array, "*B t D"],
      v: Float[Array, "*B h t d"] | Float[Array, "*B t d"],
      *,
      mask: base.Mask | mask_lib.Mask,
      segment_ids: reference.SegmentIds | None = None,
      sinks: Float[Array, "*#B H"] | None = None,
      is_mqa: bool = False,
      mask_value: float = reference.DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      max_logit_value: (
          float | Float[Array, ""] | Float[Array, "*#B #H"] | None
      ) = None,
      dropout_rate: float = 0.0,
      prng_key: jax.Array | None = None,
      return_residuals: bool = False,
      config: Config,
  ) -> tuple[Float[Array, "*B H T d"], base.Residuals | None]:
    if config.max_logit_const is not None and max_logit_value is not None:
      raise ValueError(
          "Cannot set both `config.max_logit_const` and `max_logit_value`."
      )
    config_dict = dataclasses.asdict(config)
    fold_q_seq_heads = config_dict.pop("fold_q_seq_heads")
    num_kv_heads = 1 if is_mqa else k.shape[-3]
    if fold_q_seq_heads:
      pallas_mosaic_tpu_vjp.check_fold_q_seq_heads(
          q.shape[-3],
          num_kv_heads,
          is_mqa,
          sinks=sinks,
          dropout_rate=dropout_rate,
      )
    splash_config = dataclasses.replace(
        splash_attention_kernel.SplashConfig.get_default(),
        attn_logits_soft_cap=attn_logits_soft_cap,
        dropout_rate=dropout_rate,
        block_q_dkv=None,
        block_kv_dkv=None,
        block_kv_dkv_compute=None,
        **config_dict,
    )

    q_seq_shards, q_shard_idx = (
        mask.get_q_seq_shard_info() if isinstance(mask, base.Mask) else (1, 0)
    )
    q_seq_len = q.shape[-2]
    kv_seq_len = k.shape[-2]
    splash_mask = pallas_mosaic_tpu_vjp.build_splash_mask(
        mask,
        q_seq_len,
        kv_seq_len,
        q_seq_shards,
        fold_q_seq_heads=fold_q_seq_heads,
    )

    if is_mqa and k.ndim == q.ndim:
      k_in = jnp.squeeze(k, axis=-3)
      v_in = jnp.squeeze(v, axis=-3)
    else:
      k_in = k
      v_in = v

    broadcast_to_rank = (
        lambda x, rank: None
        if x is None
        else jax.lax.broadcast_to_rank(x, rank)
    )

    if segment_ids is not None:
      seg_q = segment_ids.q
      if isinstance(seg_q, (jax.Array, np.ndarray)):
        seg_q = jax.lax.broadcast_to_rank(seg_q, q.ndim - 2)
        if fold_q_seq_heads:
          seg_q = jnp.repeat(seg_q, 2, axis=-1)
      seg_kv = segment_ids.kv
      if isinstance(seg_kv, (jax.Array, np.ndarray)):
        seg_kv = jax.lax.broadcast_to_rank(seg_kv, q.ndim - 2)
      segment_ids = reference.SegmentIds(q=seg_q, kv=seg_kv)
    sinks = broadcast_to_rank(sinks, q.ndim - 2)

    q_in = pallas_mosaic_tpu_vjp.fold_q_heads(q) if fold_q_seq_heads else q

    max_logit_val = None
    if max_logit_value is not None:
      max_logit_arr = jnp.asarray(max_logit_value, dtype=jnp.float32)
      if (
          fold_q_seq_heads
          and max_logit_arr.ndim > 0
          and max_logit_arr.shape[-1] != 1
      ):
        raise NotImplementedError(
            "`fold_q_seq_heads` requires `max_logit_value` to be scalar or"
            " broadcastable along heads."
        )
      max_logit_val = jnp.broadcast_to(max_logit_arr, q_in.shape[:-2])

    if isinstance(splash_mask, (jax.Array, np.ndarray)):
      splash_mask = broadcast_to_rank(jnp.asarray(splash_mask), q_in.ndim - 1)

      def single_fwd(q, k, v, mask, segment_ids, sinks, max_logit_val):
        splash_maker = (
            splash_attention_kernel.make_dynamic_splash_mqa
            if is_mqa
            else splash_attention_kernel.make_dynamic_splash_mha
        )
        attn_fn = splash_maker(
            mask,
            config=splash_config,
            mask_value=mask_value,
            save_residuals=return_residuals,
        )
        return attn_fn(
            q,
            k,
            v,
            segment_ids=segment_ids,
            sinks=sinks,
            max_logit_value=max_logit_val,
            prng_key=prng_key,
        )

      splash_output = base.vmap_batch_dims(single_fwd)(
          q_in, k_in, v_in, splash_mask, segment_ids, sinks, max_logit_val
      )
    else:
      splash_maker = (
          splash_attention_kernel.make_splash_mqa
          if is_mqa
          else splash_attention_kernel.make_splash_mha
      )
      attn_fn = splash_maker(
          splash_mask,
          config=splash_config,
          mask_value=mask_value,
          save_residuals=return_residuals,
          q_seq_shards=q_seq_shards,
      )

      if q_seq_shards > 1:
        fwd_mask_info = pallas_mosaic_tpu_vjp.slice_mask_info(
            attn_fn.fwd_mask_info, q_shard_idx, q_seq_shards
        )
        dkv_mask_info = None
        if attn_fn.dkv_mask_info:
          dkv_mask_info = pallas_mosaic_tpu_vjp.slice_mask_info(
              attn_fn.dkv_mask_info, q_shard_idx, q_seq_shards
          )

        attn_fn = splash_attention_kernel.SplashAttentionKernel(
            fwd_mask_info,
            dkv_mask_info,
            **attn_fn.kwargs,
        )

      def single_fwd_static(q, k, v, segment_ids, sinks, max_logit_val):
        return attn_fn(
            q,
            k,
            v,
            segment_ids=segment_ids,
            sinks=sinks,
            max_logit_value=max_logit_val,
            prng_key=prng_key,
        )

      splash_output = base.vmap_batch_dims(single_fwd_static)(
          q_in, k_in, v_in, segment_ids, sinks, max_logit_val
      )

    if return_residuals:
      out, stats = splash_output
      max_logits = stats["max_logits"]
      logsumexp = stats["logsumexp"]
      if fold_q_seq_heads:
        max_logits = pallas_mosaic_tpu_vjp.unfold_q_residuals(max_logits)
        logsumexp = pallas_mosaic_tpu_vjp.unfold_q_residuals(logsumexp)
      residuals = (max_logits, logsumexp)
    else:
      out = splash_output
      residuals = None
    if fold_q_seq_heads:
      out = pallas_mosaic_tpu_vjp.unfold_q_heads(out)
    return out, residuals

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    q = ba.arguments["q"]
    k = ba.arguments["k"]
    is_mqa = ba.arguments.get("is_mqa", False)
    dropout_rate = ba.arguments.get("dropout_rate", 0.0)
    q_seq_len = q.shape[-2]
    kv_seq_len = k.shape[-2]

    block_q = 128
    if q_seq_len >= 1024 and q_seq_len % 1024 == 0:
      block_q = 1024

    block_kv = 128
    block_kv_compute = 128
    if kv_seq_len >= 1024 and kv_seq_len % 1024 == 0:
      block_kv = 1024
      block_kv_compute = 512

    dropout_block_q = block_q if dropout_rate > 0.0 else None
    dropout_block_kv = block_kv_compute if dropout_rate > 0.0 else None

    # TODO: Implement a more sophisticated heuristic.
    return Config(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        dropout_block_q=dropout_block_q,
        dropout_block_kv=dropout_block_kv,
        q_layout=QKVLayout.HEAD_DIM_MINOR,
        k_layout=QKVLayout.HEAD_DIM_MINOR,
        v_layout=QKVLayout.HEAD_DIM_MINOR,
        use_experimental_scheduler=True,
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
        and mask.window_size is None
    ):
      is_causal = True

    tiles = [128, 256, 512, 1024, 2048, 4096]
    # TODO: Unify forward and VJP autotuning so that they use the
    # same dropout block shapes.
    dropout_tiles = tiles if dropout_rate > 0.0 else [None]
    layouts = [QKVLayout.HEAD_DIM_MINOR, QKVLayout.SEQ_MINOR]
    schedulers = [True, False]
    configs = set()
    for (
        bq,
        bkv,
        bkv_c,
        dbq,
        dbkv,
        ql,
        kl,
        vl,
        sched,
    ) in itertools.product(
        tiles,
        tiles,
        tiles,
        dropout_tiles,
        dropout_tiles,
        layouts,
        layouts,
        layouts,
        schedulers,
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

      # TODO: Make these conditions more configurable
      if bkv_c > 1024:
        continue

      if q_seq_len >= 1024 and bq < 1024:
        continue
      if kv_seq_len >= 1024 and bkv < 1024:
        continue
      if q_seq_len >= 1024 and dbq is not None and dbq < 1024:
        continue
      if kv_seq_len >= 1024 and dbkv is not None and dbkv < min(1024, bkv_c):
        continue

      if bq >= 4096 or bkv >= 4096:
        continue

      configs.add(
          Config(
              block_q=bq,
              block_kv=bkv,
              block_kv_compute=bkv_c,
              dropout_block_q=dbq,
              dropout_block_kv=dbkv,
              q_layout=ql,
              k_layout=kl,
              v_layout=vl,
              use_experimental_scheduler=sched,
          )
      )
      if is_causal and bq == bkv:
        if bkv_c >= 1024:
          qk_diag_grid_sizes = [2, 4]
        else:
          qk_diag_grid_sizes = [2]
        for qk_grid_size in qk_diag_grid_sizes:
          configs.add(
              Config(
                  block_q=bq,
                  block_kv=bkv,
                  block_kv_compute=bkv_c,
                  dropout_block_q=dbq,
                  dropout_block_kv=dbkv,
                  q_layout=ql,
                  k_layout=kl,
                  v_layout=vl,
                  use_experimental_scheduler=sched,
                  qk_diag_skip=True,
                  sv_diag_skip=(bq == bkv == bkv_c),
                  qk_diag_grid=qk_grid_size,
              )
          )
          if (
              bq == bkv == bkv_c
              and dropout_rate == 0.0
              and (bq // qk_grid_size) % NUM_LANES == 0
          ):
            configs.add(
                Config(
                    block_q=bq,
                    block_kv=bkv,
                    block_kv_compute=bkv_c,
                    dropout_block_q=dbq,
                    dropout_block_kv=dbkv,
                    q_layout=ql,
                    k_layout=kl,
                    v_layout=vl,
                    use_experimental_scheduler=sched,
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
    return mosaic_tpu.has_mosaic_tpu_support(device)
