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
import itertools
from typing import Any, ClassVar, TypeAlias
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
from tokamax._src.ops.attention import pallas_mosaic_tpu_vjp
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from typing_extensions import override


QArray = base.QArray
Residuals = base.Residuals
PagingInfo = base.PagingInfo
Key: TypeAlias = immutabledict.immutabledict[str, Any]


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_q: pydantic.conint(multiple_of=common.NUM_LANES, gt=0)
  block_kv: pydantic.conint(multiple_of=common.NUM_LANES, gt=0)
  block_kv_compute: pydantic.conint(multiple_of=common.NUM_LANES, gt=0)
  q_layout: splash.QKVLayout
  k_layout: splash.QKVLayout
  v_layout: splash.QKVLayout
  use_experimental_scheduler: bool
  use_base2_exp: bool = True

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

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(
          self,
          "vjp",
          pallas_mosaic_tpu_vjp.PallasMosaicTpuFlashAttentionVjp(),
      )

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

    orig_q_shape = q.shape
    as_4d = lambda x: jax.lax.collapse(jax.lax.broadcast_to_rank(x, 4), 0, -3)
    q, k, v = map(as_4d, (q, k, v))

    _, q_seq_len, num_q_heads, _ = q.shape
    _, kv_seq_len, num_kv_heads, head_dim_out = v.shape

    # TODO: The SplashAttention kernel expects the sequence
    # dimension to be after the num_heads dimension. This requires transposing
    # the inputs, which introduces overhead.
    q, k, v = map(lambda x: jnp.swapaxes(x, 1, 2), (q, k, v))

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
        save_residuals=return_residuals,
    )
    if is_mqa:
      k = jnp.squeeze(k, axis=1)
      v = jnp.squeeze(v, axis=1)

    splash_output = jax.vmap(splash_fn)(q, k, v)

    if return_residuals:
      out, stats = splash_output
      residuals = (stats["max_logits"], stats["logsumexp"])
      # Restore the original batch dimensions (*B) and head/sequence structure
      # (H, T).
      # Input q is [*B, T, H, D]. Target residuals are [*B, H, T].
      # Input q is [T, H, D]. Target residuals are [H, T].
      res_target_shape = (
          *orig_q_shape[:-3],
          orig_q_shape[-2],
          orig_q_shape[-3],
      )
      residuals = jax.tree.map(lambda x: x.reshape(res_target_shape), residuals)
    else:
      out = splash_output
      residuals = None
    out = jnp.swapaxes(out, 1, 2)
    # Reshape back to original shape.
    out = out.reshape(*orig_q_shape[:-1], out.shape[-1])[..., :head_dim_out]
    return out, residuals

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments):
    # TODO: Select better parameters based on a heuristic.
    return Config(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        q_layout=splash.QKVLayout.HEAD_DIM_MINOR,
        k_layout=splash.QKVLayout.HEAD_DIM_MINOR,
        v_layout=splash.QKVLayout.HEAD_DIM_MINOR,
        use_experimental_scheduler=True,
        use_base2_exp=True,
    )

  @override
  def _get_autotuning_configs(self, ba: op.BoundArguments) -> set[Config]:
    q, k = ba.arguments['q'], ba.arguments['k']
    q_seq_len, kv_seq_len = q.shape[-3], k.shape[-3]
    # TODO: Add 8192 once autotuning bugs are fixed.
    tiles = [128, 256, 512, 1024, 2048, 4096]
    layouts = [splash.QKVLayout.HEAD_DIM_MINOR, splash.QKVLayout.SEQ_MINOR]
    schedulers = [True, False]
    config = set()
    for bq, bkv, bkv_c, ql, kl, vl, sched in itertools.product(
        tiles,
        tiles,
        tiles,
        layouts,
        layouts,
        layouts,
        schedulers,
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
        config.add(
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
    return config

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5
