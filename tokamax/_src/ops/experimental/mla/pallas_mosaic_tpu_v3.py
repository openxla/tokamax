# Copyright 2026 Google LLC
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
"""Pallas/Mosaic TPU Multi-Head Latent Attention, using the v3 kernel."""

from collections.abc import Mapping
from typing import Annotated, ClassVar, override

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops import op
from tokamax._src.ops.experimental.mla import base
from tokamax._src.ops.experimental.mla.v3 import bref_override
from tokamax._src.ops.experimental.mla.v3 import configs as v3_configs
from tokamax._src.ops.experimental.mla.v3 import mla_wrapper
from tokamax._src.ops.experimental.mla.v3 import schedule as v3_schedule

# Heuristic KV block sizes, in tokens. With the 1024-token pages used in serving
# these are 2 pages per decode block and 1 per prefill/mixed block.
_DECODE_KV_TOKENS_PER_BLOCK = 2048
_PREFILL_KV_TOKENS_PER_BLOCK = 1024
# At or below this many tokens per device the decode pass packs 2 lanes per
# step rather than 4, as there are too few sequences to fill 4.
_SMALL_DECODE_BUCKET_TOKENS = 4

Schedules = Mapping[v3_configs.MlaCase, v3_schedule.MlaSchedule]


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Tunable parameters of the MLA v3 kernel.

  The decode pass (`bq_sz == 1`) and the prefill/mixed passes are tuned
  separately; the prefill and mixed passes share their block sizes.

  Attributes:
    decode_num_kv_pages_per_block: KV pages per decode block.
    decode_batch_size: Decode lanes packed into one kernel step.
    decode_n_buffer: Pipelining depth of the decode Q and KV buffers.
    prefill_num_kv_pages_per_block: KV pages per prefill/mixed block.
    prefill_num_queries_per_block: Query tokens per prefill/mixed block.
    prefill_q_split: Sub-chunks each prefill/mixed query block is split into;
      must divide `prefill_num_queries_per_block`.
    prefill_n_buffer: Pipelining depth of the prefill/mixed Q and KV buffers.
    vmem_limit_bytes: Scoped VMEM limit, or `None` for all of VMEM.
  """

  decode_num_kv_pages_per_block: Annotated[int, pydantic.Field(gt=0)] = 2
  decode_batch_size: Annotated[int, pydantic.Field(gt=0)] = 4
  decode_n_buffer: Annotated[int, pydantic.Field(ge=2)] = 2
  prefill_num_kv_pages_per_block: Annotated[int, pydantic.Field(gt=0)] = 1
  prefill_num_queries_per_block: Annotated[int, pydantic.Field(gt=0)] = 16
  prefill_q_split: Annotated[int, pydantic.Field(gt=0)] = 1
  prefill_n_buffer: Annotated[int, pydantic.Field(ge=2)] = 2
  vmem_limit_bytes: Annotated[int, pydantic.Field(gt=0)] | None = None

  def __post_init__(self):
    if self.prefill_num_queries_per_block % self.prefill_q_split:
      raise ValueError(
          f"{self.prefill_q_split=} must divide"
          f" {self.prefill_num_queries_per_block=}."
      )

  def block_sizes(
      self, page_size: int
  ) -> tuple[v3_configs.BlockSizes, v3_configs.BlockSizes]:
    """Returns the (decode, prefill/mixed) kernel block sizes."""
    decode = v3_configs.BlockSizes(
        bq_sz=1,
        bq_c_sz=1,
        bkv_sz=self.decode_num_kv_pages_per_block * page_size,
        batch_size=self.decode_batch_size,
        n_buffer=self.decode_n_buffer,
    )
    prefill = v3_configs.BlockSizes(
        bq_sz=self.prefill_num_queries_per_block,
        bq_c_sz=self.prefill_num_queries_per_block // self.prefill_q_split,
        bkv_sz=self.prefill_num_kv_pages_per_block * page_size,
        batch_size=1,
        n_buffer=self.prefill_n_buffer,
    )
    return decode, prefill


class PallasTpuMultiHeadLatentAttentionV3(
    base.MultiHeadLatentAttention[Config]
):
  """MLA v3 kernel: KV cache with tokens along lanes, head-major queries.

  Only supports `layout=MlaLayout.HEAD_MAJOR_SEQ_ALONG_LANE`, i.e. `ql_nope` and
  the output are `[heads, tokens, lkv_dim]` and `cache_kv` is
  `[pages, kv_dim, page_size]`. Logits are always computed in float32, so
  `s_dtype` is ignored.

  The kernel runs one pass per `distribution` band, each with a DMA schedule
  built by a separate kernel. A model can build the schedules once per step and
  pass them to every layer:

  ```python
  schedules = op.build_schedules(*args, **kwargs)
  for layer in layers:
    out, cache = op(*args, **kwargs, schedules=schedules)
  ```
  """

  config_cls: ClassVar[type[Config]] = Config
  supports_symbolic_shapes: ClassVar[bool] = False

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      ql_nope: Float[Array, "num_q_heads max_num_tokens actual_lkv_dim"],
      q_pe: Float[Array, "max_num_tokens num_q_heads actual_r_dim"],
      new_kv_c: Float[Array, "max_num_tokens actual_lkv_dim"],
      new_k_pe: Float[Array, "max_num_tokens actual_r_dim"],
      cache_kv: Float[Array, "total_num_pages aligned_kv_dim page_size"],
      kv_lens: Int[Array, "max_num_seqs"],
      page_indices: Int[Array, "num_page_indices"],
      cu_q_lens: Int[Array, "max_num_seqs_plus_1"],
      distribution: Int[Array, "3"],
      *,
      sm_scale: float = 1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      mask_value: float | None = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      s_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      layout: base.MlaLayout = base.MlaLayout.TOKEN_MAJOR,
      schedules: base.OpaqueMetadata = None,
      debug_mode: bool = False,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array], None]:
    del s_dtype, return_residuals  # Unused.
    assert config is not None, "Config must be specified."
    # Token-major inputs normally fail the jaxtyping shape check on entry, as
    # `ql_nope` and `q_pe` then disagree on `num_q_heads`/`max_num_tokens`; this
    # only fires with jaxtyping disabled.
    if layout != base.MlaLayout.HEAD_MAJOR_SEQ_ALONG_LANE:
      raise NotImplementedError(
          f"{type(self).__name__} only supports"
          f" {base.MlaLayout.HEAD_MAJOR_SEQ_ALONG_LANE}, got {layout=}."
      )
    decode_blocks, prefill_blocks = config.block_sizes(cache_kv.shape[-1])
    out = mla_wrapper.mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        decode_block_sizes=decode_blocks,
        prefill_block_sizes=prefill_blocks,
        vmem_limit_bytes=config.vmem_limit_bytes,
        schedules=schedules,
        debug_mode=debug_mode,
    )
    return out, None

  def build_schedules(self, *args, **kwargs) -> Schedules:
    """Builds the per-pass schedules for the given arguments.

    Takes the same arguments as the op. The schedules depend on the shapes,
    the config and the *values* of `cu_q_lens`, `kv_lens`, `page_indices` and
    `distribution` -- not on the activations or the cache contents. Build them
    once per step, pass them as `schedules=` to every layer of that step, and
    rebuild them whenever those values change.

    The config is resolved exactly as a call with the same arguments would
    resolve it, so the schedules match it. Only the configs (shapes, dtypes,
    block sizes) are checked when schedules are passed in: schedules built from
    different lengths or page indices with the same shapes are not detected and
    produce wrong results.

    Args:
      *args: Positional arguments, as for the op.
      **kwargs: Keyword arguments, as for the op.

    Returns:
      The schedules, keyed by pass.
    """
    ba = self.bind(*args, **kwargs)
    a = ba.arguments
    if a["layout"] != base.MlaLayout.HEAD_MAJOR_SEQ_ALONG_LANE:
      raise NotImplementedError(
          f"{type(self).__name__} only supports"
          f" {base.MlaLayout.HEAD_MAJOR_SEQ_ALONG_LANE}."
      )
    config = ba.default_config
    decode_blocks, prefill_blocks = config.block_sizes(a["cache_kv"].shape[-1])
    cfgs_by_mode = mla_wrapper.mode_configs(
        a["ql_nope"],
        a["q_pe"],
        a["cache_kv"],
        a["kv_lens"],
        a["page_indices"],
        sm_scale=a["sm_scale"],
        sliding_window=a["sliding_window"],
        soft_cap=a["soft_cap"],
        mask_value=a["mask_value"],
        q_scale=a["q_scale"],
        k_scale=a["k_scale"],
        v_scale=a["v_scale"],
        decode_block_sizes=decode_blocks,
        prefill_block_sizes=prefill_blocks,
        vmem_limit_bytes=config.vmem_limit_bytes,
    )
    return mla_wrapper.build_schedules(
        a["cu_q_lens"],
        a["kv_lens"],
        a["page_indices"],
        a["distribution"],
        cfgs_by_mode,
    )

  @override
  def _get_heuristics_config(self, ba: op.BoundArguments) -> Config:
    """The configuration the end-to-end DeepSeek serving runs used."""
    page_size = ba.arguments["cache_kv"].shape[-1]
    num_tokens = ba.arguments["ql_nope"].shape[1]
    small_bucket = num_tokens <= _SMALL_DECODE_BUCKET_TOKENS
    return Config(
        decode_num_kv_pages_per_block=max(
            1, _DECODE_KV_TOKENS_PER_BLOCK // page_size
        ),
        decode_batch_size=2 if small_bucket else 4,
        prefill_num_kv_pages_per_block=max(
            1, _PREFILL_KV_TOKENS_PER_BLOCK // page_size
        ),
    )

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return (
        device.platform == "tpu"
        and pltpu.get_tpu_info().generation >= 7
        and bref_override.SUPPORTS_IN_OUT_BUFFER_COUNT
    )
