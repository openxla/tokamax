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
"""Flash attention with Mosaic TPU VJP."""

import dataclasses
import functools
import itertools
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
from tokamax._src.ops.attention import pallas_mosaic_tpu_common as common
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from typing_extensions import override

QArray = base.QArray
Residuals = base.Residuals
PagingInfo = base.PagingInfo
Key: TypeAlias = immutabledict.immutabledict[str, Any]


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_q_dkv: pydantic.conint(multiple_of=common.NUM_LANES, gt=0)
  block_kv_dkv: pydantic.conint(multiple_of=common.NUM_LANES, gt=0)
  block_kv_dkv_compute: pydantic.conint(multiple_of=common.NUM_LANES, gt=0)
  use_base2_exp: bool = True

  def __post_init__(self):
    if self.block_kv_dkv % self.block_kv_dkv_compute:
      block_kv_dkv = self.block_kv_dkv
      block_kv_dkv_compute = self.block_kv_dkv_compute
      raise ValueError(
          f"{block_kv_dkv=} must be a multiple of {block_kv_dkv_compute=}."
      )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PallasMosaicTpuFlashAttentionVjp(
    base.DotProductAttentionVjp[Config, None]
):
  """Pallas-Mosaic FlashAttention VJP implementation."""

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

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
    if not normalize_output:
      raise NotImplementedError("`normalize_output=False` not supported.")
    if return_residuals:
      raise NotImplementedError("`return_residuals` not supported.")

    common.check_inputs_support(
        q,
        k,
        v,
        bias=bias,
        logits_dtype=logits_dtype,
        dropout_mask=dropout_mask,
        dropout_rate=dropout_rate,
        paging_info=paging_info,
        mask=mask,
    )

    q *= logits_scale

    as_4d = lambda x: jax.lax.collapse(jax.lax.broadcast_to_rank(x, 4), 0, -3)
    q_4d, k_4d, v_4d, dout_4d, out_4d = map(as_4d, (q, k, v, dout, out))

    _, q_seq_len, num_q_heads, _ = q_4d.shape
    _, kv_seq_len, num_kv_heads, _ = v_4d.shape
    # TODO: The SplashAttention kernel expects the sequence
    # dimension to be after the num_heads dimension.
    q_swap, k_swap, v_swap, dout_swap, out_swap = map(
        lambda x: jnp.swapaxes(x, 1, 2), (q_4d, k_4d, v_4d, dout_4d, out_4d)
    )

    is_mqa = num_kv_heads == 1
    if num_q_heads % num_kv_heads:
      raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
    splash_config = dataclasses.replace(
        splash.SplashConfig.get_default(),
        attn_logits_soft_cap=logits_soft_cap,
        **dataclasses.asdict(config),
    )
    splash_fn = common.build_splash_kernel(
        mask=mask,
        splash_config=splash_config,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        is_mqa=is_mqa,
        save_residuals=True,
    )
    if is_mqa:
      k_splash, v_splash = jnp.squeeze(k_swap, axis=1), jnp.squeeze(
          v_swap, axis=1
      )
    else:
      k_splash, v_splash = k_swap, v_swap

    # Always in natural base.
    lse = residuals[1]
    # BWD kernel expects in corresponding base.
    if config.use_base2_exp:
      lse = lse * splash.LOG2E

    splash_fn_kwargs = splash_fn.kwargs
    bwd_fn = functools.partial(
        splash._splash_attention_bwd,  # pylint: disable=protected-access
        True,  # save_residuals
        splash_fn_kwargs["mask_value"],
        is_mqa,
        splash_fn_kwargs["config"],
        splash_fn_kwargs["mask_function"],
        splash_fn_kwargs["fwd_mask_sparsity"],
        splash_fn_kwargs["dkv_mask_sparsity"],
    )

    res = (
        q_swap,
        k_splash,
        v_splash,
        None,
        None,
        out_swap,
        lse,
        splash_fn.dkv_mask_info,
    )
    res_in_axes = (0, 0, 0, None, None, 0, (0 if lse.ndim == 3 else None), None)
    # vmap over batch dimension
    _, _, dq, dk, dv, _, _, _ = jax.vmap(bwd_fn, in_axes=(res_in_axes, 0))(
        res, dout_swap
    )

    dq = jnp.swapaxes(dq, 1, 2) * logits_scale
    if is_mqa:
      dk = jnp.swapaxes(jnp.expand_dims(dk, 1), 1, 2)
      dv = jnp.swapaxes(jnp.expand_dims(dv, 1), 1, 2)
    else:
      dk = jnp.swapaxes(dk, 1, 2)
      dv = jnp.swapaxes(dv, 1, 2)

    # Reshape back to the original batch dimensions (*B)
    dq = dq.reshape(q.shape)
    dk = dk.reshape(k.shape)
    dv = dv.reshape(v.shape)
    return base.DotProductAttentionGrads(q=dq, k=dk, v=dv), None

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    return Config(
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        use_base2_exp=True,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    q, k = ba.arguments['q'], ba.arguments['k']
    q_seq_len, kv_seq_len = q.shape[-3], k.shape[-3]
    # TODO: Add 256, 8192 once autotuning bugs are fixed.
    tiles = [512, 1024, 2048, 4096]
    configs = set()
    for bq, bkv, bkv_c in itertools.product(
        tiles,
        tiles,
        tiles,
    ):
      # TODO: For sparse masks smaller block sizes could give
      # better performance.
      if q_seq_len >= 1024 and bq < 1024:
        continue
      if kv_seq_len >= 1024 and bkv < 1024:
        continue
      if bkv_c > 1024:
        continue
      if bkv % bkv_c == 0 and bq <= q_seq_len and bkv <= kv_seq_len:
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
