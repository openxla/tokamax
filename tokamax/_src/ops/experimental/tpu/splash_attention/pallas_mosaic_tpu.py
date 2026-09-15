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
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.tpu.splash_attention import base
from tokamax._src.ops.experimental.tpu.splash_attention import reference
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib

NUM_LANES: Final[int] = 128
QKVLayout = splash_attention_kernel.QKVLayout


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  """Autotuning and execution configuration for Splash Attention Pallas TPU kernel."""

  block_q: Annotated[int, pydantic.Field(multiple_of=NUM_LANES, gt=0)]
  block_kv: Annotated[int, pydantic.Field(multiple_of=NUM_LANES, gt=0)]
  block_kv_compute: Annotated[int, pydantic.Field(multiple_of=NUM_LANES, gt=0)]
  dropout_block_q: int | None = None
  dropout_block_kv: int | None = None
  q_layout: QKVLayout
  k_layout: QKVLayout
  v_layout: QKVLayout
  use_experimental_scheduler: bool
  use_base2_exp: bool = True
  qk_diag_skip: bool = False
  qk_diag_grid: int = 2
  sv_diag_skip: bool = False

  def __post_init__(self):
    if self.block_kv % self.block_kv_compute:
      raise ValueError(
          f"{self.block_kv=} must be a multiple of {self.block_kv_compute=}."
      )
    if (self.qk_diag_skip or self.sv_diag_skip) and not (
        self.block_q == self.block_kv == self.block_kv_compute
    ):
      raise ValueError(
          "qk_diag_skip and sv_diag_skip require square forward blocks "
          f"({self.block_q=}, {self.block_kv=}, {self.block_kv_compute=})."
      )
    if self.qk_diag_grid < 2 or (self.qk_diag_grid & (self.qk_diag_grid - 1)):
      raise ValueError(f"{self.qk_diag_grid=} must be a power of 2 >= 2.")


@dataclasses.dataclass(frozen=True)
class PallasTpuSplashAttention(base.SplashAttention[Config]):
  """Tokamax operator wrapper for Splash Attention Pallas TPU kernel."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array, "num_q_heads q_seq_len head_dim_qk"],
      k: Float[Array, "..."],
      v: Float[Array, "..."],
      mask: base.Mask | mask_lib.Mask,
      segment_ids: reference.SegmentIds | None = None,
      sinks: Float[Array, "..."] | None = None,
      *,
      is_mqa: bool = False,
      mask_value: float = reference.DEFAULT_MASK_VALUE,
      attn_logits_soft_cap: float | None = None,
      dropout_rate: float = 0.0,
      return_residuals: bool = False,
      config: Config,
  ) -> tuple[jax.Array, None]:
    if dropout_rate != 0.0:
      raise NotImplementedError(
          "Dropout is not supported in PallasTpuSplashAttention."
      )

    splash_config = dataclasses.replace(
        splash_attention_kernel.SplashConfig.get_default(),
        attn_logits_soft_cap=attn_logits_soft_cap,
        dropout_rate=dropout_rate,
        **dataclasses.asdict(config),
    )

    q_seq_len = q.shape[1]
    kv_seq_len = k.shape[0] if is_mqa and k.ndim == 2 else k.shape[1]
    mask_shape = (q_seq_len, kv_seq_len)

    if isinstance(mask, mask_lib.Mask):
      splash_mask = mask
    elif isinstance(mask, base.Mask):
      if mask.bool_mask is not None and mask.is_causal:
        splash_mask = mask.as_array(q_seq_len, kv_seq_len)
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

    out = attn_fn(
        q,
        k_in,
        v_in,
        segment_ids=segment_ids,
        sinks=sinks,
    )
    return out, None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    del ba
    return Config(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
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
    q_seq_len = q.shape[1]
    kv_seq_len = k.shape[0] if is_mqa and k.ndim == 2 else k.shape[1]

    is_causal = False
    if isinstance(mask, mask_lib.CausalMask):
      is_causal = True
    elif (
        isinstance(mask, base.Mask)
        and mask.is_causal
        and mask.bool_mask is None
    ):
      is_causal = True

    tiles = [128, 256, 512, 1024, 2048, 4096]
    layouts = [QKVLayout.HEAD_DIM_MINOR, QKVLayout.SEQ_MINOR]
    schedulers = [True, False]
    configs = set()
    for bq, bkv, bkv_c, ql, kl, vl, sched in itertools.product(
        tiles,
        tiles,
        tiles,
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

      # TODO: Make these conditions more configurable
      if bkv_c > 1024:
        continue

      if q_seq_len >= 1024 and bq < 1024:
        continue
      if kv_seq_len >= 1024 and bkv < 1024:
        continue

      if bq >= 4096 or bkv >= 4096:
        continue

      configs.add(
          Config(
              block_q=bq,
              block_kv=bkv,
              block_kv_compute=bkv_c,
              q_layout=ql,
              k_layout=kl,
              v_layout=vl,
              use_experimental_scheduler=sched,
          )
      )
      if is_causal and bq == bkv == bkv_c:
        if bq >= 1024:
          qk_diag_grid_sizes = [2, 4]
        else:
          qk_diag_grid_sizes = [2]
        for qk_grid_size in qk_diag_grid_sizes:
          configs.add(
              Config(
                  block_q=bq,
                  block_kv=bkv,
                  block_kv_compute=bkv_c,
                  q_layout=ql,
                  k_layout=kl,
                  v_layout=vl,
                  use_experimental_scheduler=sched,
                  qk_diag_skip=True,
                  sv_diag_skip=True,
                  qk_diag_grid=qk_grid_size,
              )
          )
    return configs

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
