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

from collections.abc import Callable, Mapping
import dataclasses
import itertools
from typing import Annotated, Any, ClassVar, Final, override
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import pydantic
from tokamax._src import jaxtyping
from tokamax._src import mosaic_tpu
from tokamax._src.ops import op
from tokamax._src.ops.splash_attention import base
from tokamax._src.ops.splash_attention import splash_attention_kernel
from tokamax._src.ops.splash_attention import splash_attention_mask as mask_lib
from tokamax._src.ops.splash_attention import splash_attention_mask_info as mask_info_lib

NUM_LANES: Final[int] = 128
LOG2E = splash_attention_kernel.LOG2E
QKVLayout = splash_attention_kernel.QKVLayout
_MISSING = object()


class _FilteredAutotuningCache[_ConfigT](
    dict[Any, op.AutotuningData[_ConfigT]]
):
  """Dict view of an autotuning cache filtering entries by a config predicate."""

  def __init__(
      self,
      raw_cache: dict[Any, op.AutotuningData[_ConfigT]],
      predicate: Callable[[_ConfigT], bool],
  ):
    filtered_items = []
    for k, data in raw_cache.items():
      matching = op.AutotuningData(
          {cfg: ev for cfg, ev in data.items() if predicate(cfg)}
      )
      if matching:
        filtered_items.append((k, matching))
    super().__init__(filtered_items)
    self._raw = raw_cache
    self._predicate = predicate

  def _update_raw_entry(
      self, key: Any, matching: Mapping[_ConfigT, Any]
  ) -> None:
    """Merges `matching` entries into `self._raw` without clobbering others."""
    if key in self._raw:
      preserved = {
          cfg: ev
          for cfg, ev in self._raw[key].items()
          if not self._predicate(cfg)
      }
    else:
      preserved = {}
    merged = {**preserved, **matching}
    if merged:
      self._raw[key] = op.AutotuningData(merged)
    elif key in self._raw:
      del self._raw[key]

  @override
  def __setitem__(self, key: Any, value: op.AutotuningData[_ConfigT]) -> None:
    matching_dict = {
        cfg: ev for cfg, ev in value.items() if self._predicate(cfg)
    }
    self._update_raw_entry(key, matching_dict)
    if matching_dict:
      super().__setitem__(key, op.AutotuningData(matching_dict))
    elif key in self:
      super().__delitem__(key)

  @override
  def __delitem__(self, key: Any) -> None:
    super().__delitem__(key)
    self._update_raw_entry(key, {})

  @override
  def pop(self, key: Any, default: Any = _MISSING) -> Any:
    if key in self:
      val = super().pop(key)
      self._update_raw_entry(key, {})
      return val
    if default is _MISSING:
      raise KeyError(key)
    return default

  @override
  def setdefault(
      self, key: Any, default: op.AutotuningData[_ConfigT] | None = None
  ) -> Any:
    if key in self:
      return self[key]
    if default is None:
      return None
    self[key] = default
    if key in self:
      return self[key]
    return default

  @override
  def update(self, other: Any = (), /, **kwargs: Any) -> None:
    items = other.items() if isinstance(other, Mapping) else other
    for k, v in items:
      self[k] = v
    for k, v in kwargs.items():
      self[k] = v

  @override
  def clear(self) -> None:
    for k in list(self):
      del self[k]

  def __ior__(self, other: Any):
    self.update(other)
    return self


def filter_autotuning_cache[_ConfigT](
    raw_cache: dict[Any, op.AutotuningData[_ConfigT]],
    predicate: Callable[[_ConfigT], bool],
) -> dict[Any, op.AutotuningData[_ConfigT]]:
  """Returns a view of `raw_cache` keeping only configs matching `predicate`."""
  return _FilteredAutotuningCache(raw_cache, predicate)


def slice_mask_info(
    mask_info: mask_info_lib.MaskInfo,
    q_shard_idx: int | jax.Array,
    q_seq_shards: int,
) -> mask_info_lib.MaskInfo:
  """Slices `MaskInfo` along the Q sequence shard dimension."""
  if q_seq_shards == 1:
    return mask_info

  def _slice(arr: jax.Array | np.ndarray | None):
    if arr is None:
      return None
    shard_len = int(arr.shape[0]) // q_seq_shards
    return jax.lax.dynamic_slice_in_dim(
        arr, q_shard_idx * shard_len, shard_len, axis=0
    )

  return mask_info_lib.MaskInfo(
      mask_next=_slice(mask_info.mask_next),
      active_rows=_slice(mask_info.active_rows),
      active_cols=_slice(mask_info.active_cols),
      num_active_blocks=_slice(mask_info.num_active_blocks),
      block_mask=_slice(mask_info.block_mask),
      partial_mask_blocks=mask_info.partial_mask_blocks,
      q_sequence=_slice(mask_info.q_sequence),
  )


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
  q_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  k_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  v_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
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
    if (
        self.diag_band_skip
        and (self.block_kv_dkv_compute // self.qk_diag_grid) % NUM_LANES != 0
    ):
      raise ValueError(
          f"{self.block_kv_dkv_compute=} // {self.qk_diag_grid=} must be a "
          f"multiple of {NUM_LANES=} when diag_band_skip=True."
      )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasMosaicTpuSplashAttentionVjp(base.SplashAttentionVjp[Config]):
  """Pallas-Mosaic SplashAttention VJP implementation."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False
  q_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  k_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  v_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  use_base2_exp: bool = True

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
      max_logit_value: (
          float | Float[Array, ""] | Float[Array, "*#B #H"] | None
      ) = None,
      dropout_rate: float = 0.0,
      prng_key: jax.Array | None = None,
      return_residuals: bool = False,
      config: Config,
  ) -> tuple[base.SplashAttentionGrads, None]:
    del max_logit_value
    if return_residuals:
      raise NotImplementedError("`return_residuals` not supported.")

    _, lse = residuals
    if config.use_base2_exp:
      lse = lse * LOG2E

    q_seq_shards, q_shard_idx = (
        mask.get_q_seq_shard_info() if isinstance(mask, base.Mask) else (1, 0)
    )
    q_seq_len = q.shape[-2]
    kv_seq_len = k.shape[-2]
    mask_shape = (q_seq_len * q_seq_shards, kv_seq_len)

    if isinstance(mask, mask_lib.Mask):
      splash_mask = mask
    elif isinstance(mask, base.Mask):
      if mask.bool_mask is not None and mask.is_causal:
        splash_mask = mask.as_array(q_seq_len, kv_seq_len)
      elif mask.bool_mask is not None:
        splash_mask = mask.bool_mask
      elif mask.is_causal:
        splash_mask = mask_lib.CausalMask(
            shape=mask_shape, offset=mask.offset, shard_count=q_seq_shards
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
          splash_attention_kernel.make_splash_mqa
          if is_mqa
          else splash_attention_kernel.make_splash_mha
      )
      attn_fn = splash_maker(
          splash_mask,
          config=splash_config,
          mask_value=mask_value,
          q_seq_shards=q_seq_shards,
      )
      dkv_mask_info = (
          slice_mask_info(attn_fn.dkv_mask_info, q_shard_idx, q_seq_shards)
          if attn_fn.dkv_mask_info is not None
          else None
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
            dkv_mask_info,
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
        q_layout=self.q_layout,
        k_layout=self.k_layout,
        v_layout=self.v_layout,
        use_base2_exp=self.use_base2_exp,
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
      can_use_diag_band_skip = (
          is_causal and bq == bkv == bkv_c and dropout_rate == 0.0
      )
      try_vmem_dq_band_skip = can_use_diag_band_skip and (
          q_steps <= 2 or kv_steps == 1 or bkv_c >= 2048
      )
      # TODO: Make these conditions more configurable
      if bkv_c > 1024 and not can_use_diag_band_skip:
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
                q_layout=self.q_layout,
                k_layout=self.k_layout,
                v_layout=self.v_layout,
                use_base2_exp=self.use_base2_exp,
                dq_reduction_steps=dq_reduction_steps,
            )
        )
      if is_causal and bq == bkv == bkv_c:
        if bkv_c >= 1024:
          qk_diag_grid_sizes = [2, 4]
        else:
          qk_diag_grid_sizes = [2]
        for qk_grid_size in qk_diag_grid_sizes:
          if bkv_c <= 1024 or can_use_diag_band_skip:
            configs.add(
                Config(
                    block_q_dkv=bq,
                    block_kv_dkv=bkv,
                    block_kv_dkv_compute=bkv_c,
                    dropout_block_q=dbq,
                    dropout_block_kv=dbkv,
                    q_layout=self.q_layout,
                    k_layout=self.k_layout,
                    v_layout=self.v_layout,
                    use_base2_exp=self.use_base2_exp,
                    dq_reduction_steps=dq_reduction_steps,
                    qk_diag_skip=True,
                    qk_diag_grid=qk_grid_size,
                )
            )
          if (
              can_use_diag_band_skip
              and (bkv_c // qk_grid_size) % NUM_LANES == 0
          ):
            configs.add(
                Config(
                    block_q_dkv=bq,
                    block_kv_dkv=bkv,
                    block_kv_dkv_compute=bkv_c,
                    dropout_block_q=dbq,
                    dropout_block_kv=dbkv,
                    q_layout=self.q_layout,
                    k_layout=self.k_layout,
                    v_layout=self.v_layout,
                    use_base2_exp=self.use_base2_exp,
                    dq_reduction_steps=dq_reduction_steps,
                    qk_diag_skip=True,
                    qk_diag_grid=qk_grid_size,
                    diag_band_skip=True,
                )
            )
          if (
              try_vmem_dq_band_skip
              and (bkv_c // qk_grid_size) % NUM_LANES == 0
          ):
            configs.add(
                Config(
                    block_q_dkv=bq,
                    block_kv_dkv=bkv,
                    block_kv_dkv_compute=bkv_c,
                    dropout_block_q=dbq,
                    dropout_block_kv=dbkv,
                    q_layout=self.q_layout,
                    k_layout=self.k_layout,
                    v_layout=self.v_layout,
                    use_base2_exp=self.use_base2_exp,
                    dq_reduction_steps=1,
                    qk_diag_skip=True,
                    qk_diag_grid=qk_grid_size,
                    diag_band_skip=True,
                )
            )
    if not configs:
      configs.add(self._get_heuristics_config(ba))
    return configs

  def _matches_config(self, cfg: Config) -> bool:
    return (
        cfg.q_layout == self.q_layout
        and cfg.k_layout == self.k_layout
        and cfg.v_layout == self.v_layout
        and cfg.use_base2_exp == self.use_base2_exp
    )

  @override
  def get_autotuning_cache(
      self, device_kind: op.DeviceKind | None = None
  ) -> dict[Any, op.AutotuningData[Config]]:
    return filter_autotuning_cache(
        base.SplashAttentionVjp.get_autotuning_cache(self, device_kind),
        self._matches_config,
    )

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return mosaic_tpu.has_mosaic_tpu_support(device)
